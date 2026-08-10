from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Callable

from .execution_plan import STAGES, StepExecutionPlan
from .whitebox_model import WhiteboxMoEModel


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def _actual_components(row: dict[str, Any]) -> dict[str, float]:
    components = {stage: 0.0 for stage in STAGES}
    for layer in row["layer_records"]:
        for stage in STAGES:
            components[stage] += float(layer["timing_us"][stage])
    components["fixed"] = float(row["fixed_us"])
    return components


def _compatibility(components: dict[str, float]) -> dict[str, float]:
    return {
        "attention": components["attention"],
        "moe": sum(components[key] for key in ("router", "dispatch", "expert_gemm", "combine")),
        "dense": components["dense"],
        "transition_fixed": components["transition"] + components["fixed"],
    }


def _summarize(rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"evaluated_steps": 0, "skipped_steps": skipped, "status": "not_predictable"}
    errors = [row["absolute_percentage_error"] for row in rows]
    component_names = sorted(rows[0]["component_ape"])
    component_error = {
        name: {
            "mape": sum(row["component_ape"][name] for row in rows) / len(rows),
            "p95_ape": _percentile([row["component_ape"][name] for row in rows], 0.95),
        }
        for name in component_names
    }
    distance_bins = {}
    for name, lower, upper in (
        ("exact_to_0.1", 0.0, 0.1),
        ("0.1_to_0.5", 0.1, 0.5),
        ("0.5_to_1.0", 0.5, 1.0),
        ("above_1.0", 1.0, float("inf")),
    ):
        selected = [
            row for row in rows if lower <= row["signature_distance_max"] < upper
        ]
        distance_bins[name] = {
            "steps": len(selected),
            "mape": (
                sum(row["absolute_percentage_error"] for row in selected) / len(selected)
                if selected
                else None
            ),
        }
    return {
        "status": "ok",
        "evaluated_steps": len(rows),
        "skipped_steps": skipped,
        "step_error": {
            "mape": sum(errors) / len(errors),
            "p50_ape": median(errors),
            "p95_ape": _percentile(errors, 0.95),
            "max_ape": max(errors),
        },
        "component_error": component_error,
        "bottleneck_accuracy": sum(row["bottleneck_correct"] for row in rows) / len(rows),
        "signature_distance": {
            "mean": sum(row["signature_distance_max"] for row in rows) / len(rows),
            "max": max(row["signature_distance_max"] for row in rows),
            "error_by_distance_bin": distance_bins,
        },
        "rows": rows,
    }


def _evaluate_folds(
    raw_records: list[dict[str, Any]],
    folds: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    allow_synthetic: bool,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fold_name, train_rows, test_rows in folds:
        if not train_rows or not test_rows:
            skipped.append({"fold": fold_name, "reason": "empty train or test partition"})
            continue
        model = WhiteboxMoEModel().fit_raw(train_rows, allow_synthetic=allow_synthetic)
        for row in test_rows:
            plan = StepExecutionPlan.from_raw_step(row)
            try:
                profile = model.predict(plan)
            except KeyError as error:
                skipped.append({"fold": fold_name, "phase": plan.phase, "reason": str(error)})
                continue
            actual_step = float(row["timing_us"]["step"])
            actual_components = _actual_components(row)
            actual_groups = _compatibility(actual_components)
            predicted_groups = _compatibility(profile.components_us)
            component_ape = {
                name: abs(profile.components_us[name] - actual) / actual if actual else 0.0
                for name, actual in actual_components.items()
            }
            absolute_error = abs(profile.step_us - actual_step)
            predictions.append(
                {
                    "fold": fold_name,
                    "workload_index": row.get("workload_index"),
                    "phase": plan.phase,
                    "actual_step_us": actual_step,
                    "predicted_step_us": profile.step_us,
                    "absolute_error_us": absolute_error,
                    "absolute_percentage_error": absolute_error / actual_step if actual_step else 0.0,
                    "component_ape": component_ape,
                    "actual_bottleneck": max(actual_groups, key=actual_groups.get),
                    "predicted_bottleneck": max(predicted_groups, key=predicted_groups.get),
                    "bottleneck_correct": max(actual_groups, key=actual_groups.get)
                    == max(predicted_groups, key=predicted_groups.get),
                    "signature_distance_max": profile.signature_distance["max"],
                    "extrapolated_signature_count": len(profile.extrapolated_signatures),
                }
            )
    return _summarize(predictions, skipped)


def _group_folds(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any], prefix: str
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return [
        (f"{prefix}={value}", [row for other, values in groups.items() if other != value for row in values], test)
        for value, test in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def _requested(row: dict[str, Any], name: str) -> int:
    value = row.get("requested_workload", {}).get(name)
    if not isinstance(value, int):
        raise ValueError(f"strict evaluation requires requested_workload.{name}")
    return value


def evaluate_leave_one_workload_out(
    raw_records: list[dict[str, Any]], *, allow_synthetic: bool = False
) -> dict[str, Any]:
    if len({row.get("workload_index") for row in raw_records}) < 2:
        raise ValueError("holdout evaluation requires at least two workload groups")
    folds = _group_folds(raw_records, lambda row: row.get("workload_index"), "workload")
    summary = _evaluate_folds(raw_records, folds, allow_synthetic=allow_synthetic)
    if summary["evaluated_steps"] == 0:
        raise ValueError("no holdout row was predictable; collect anchors for each phase")
    rows = summary["rows"]
    summary.update(
        {
            "semantics": "leave-one-requested-workload-out; true step CUDA Event is evaluation-only",
            "adaptive_anchor_suggestions": sorted(
                rows, key=lambda row: row["absolute_percentage_error"], reverse=True
            )[:10],
        }
    )
    return summary


def evaluate_strict_generalization(
    raw_records: list[dict[str, Any]], *, allow_synthetic: bool = False
) -> dict[str, Any]:
    unseen_batch = _evaluate_folds(
        raw_records,
        _group_folds(raw_records, lambda row: _requested(row, "batch_size"), "batch"),
        allow_synthetic=allow_synthetic,
    )
    def length_band(row: dict[str, Any]) -> str:
        length = _requested(row, "prompt_tokens")
        return "small_1_64" if length <= 64 else "medium_65_256" if length <= 256 else "large_257_plus"
    unseen_length = _evaluate_folds(
        raw_records,
        _group_folds(raw_records, length_band, "length_band"),
        allow_synthetic=allow_synthetic,
    )
    train_region = [row for row in raw_records if _requested(row, "prompt_tokens") <= 256]
    test_region = [row for row in raw_records if _requested(row, "prompt_tokens") > 256]
    region = _evaluate_folds(
        raw_records,
        [("train_prompt_le_256_test_gt_256", train_region, test_region)],
        allow_synthetic=allow_synthetic,
    )
    mixed = [row for row in raw_records if row.get("phase") == "mixed"]
    non_mixed = [row for row in raw_records if row.get("phase") != "mixed"]
    phase = _evaluate_folds(
        raw_records,
        [("holdout_mixed", non_mixed, mixed)],
        allow_synthetic=allow_synthetic,
    )
    return {
        "semantics": "strict group holdouts; held-out labels are evaluation-only",
        "unseen_batch": unseen_batch,
        "unseen_length_interval": unseen_length,
        "region_holdout": region,
        "phase_holdout": phase,
        "profiling_surface": {
            "raw_steps": len(raw_records),
            "requested_workloads": len({row.get("workload_index") for row in raw_records}),
            "batch_values": sorted({_requested(row, "batch_size") for row in raw_records}),
            "prompt_lengths": sorted({_requested(row, "prompt_tokens") for row in raw_records}),
        },
    }

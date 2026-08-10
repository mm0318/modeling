from __future__ import annotations

import io
import json
import copy
import math
import statistics
import time
from collections import defaultdict
from typing import Any, Iterable
from pathlib import Path

import joblib
import numpy as np

from .fixed_config_oracle import (
    FixedConfigForwardOracle,
    _config_identity,
    _stable_json,
    features_from_raw,
)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _signature(row: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    return features_from_raw(row)


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    actual_rank = _rank(actual)
    predicted_rank = _rank(predicted)
    if np.std(actual_rank) == 0 or np.std(predicted_rank) == 0:
        return None
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_rows"}
    apes = [row["ape"] for row in rows]
    return {
        "status": "ok",
        "count": len(rows),
        "mape": statistics.mean(apes),
        "p95_ape": _p95(apes),
        "max_ape": max(apes),
    }


def _unique_anchor_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_signature(row)].append(row)
    representatives = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        samples = grouped[key]
        median_target = statistics.median(float(row["timing_us"]["step"]) for row in samples)
        representative = copy.deepcopy(min(
            samples, key=lambda row: abs(float(row["timing_us"]["step"]) - median_target)
        ))
        representative["timing_us"]["step"] = median_target
        representatives.append(representative)
    return representatives, grouped


def select_phase_aware_farthest(
    candidates: list[dict[str, Any]], budget: int
) -> list[tuple[str, tuple[float, ...]]]:
    """Select shape/routing coverage without reading any latency label."""
    if budget < 1 or budget > len(candidates):
        raise ValueError("budget must be within the unique candidate count")
    phase_rows: dict[str, list[tuple[tuple[str, tuple[float, ...]], np.ndarray]]] = defaultdict(list)
    for row in candidates:
        signature = _signature(row)
        phase_rows[signature[0]].append((signature, np.asarray(signature[1], dtype=np.float64)))
    normalized: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
    for phase, items in phase_rows.items():
        matrix = np.asarray([item[1] for item in items])
        lower, upper = matrix.min(axis=0), matrix.max(axis=0)
        scale = np.where(upper > lower, upper - lower, 1.0)
        for signature, vector in items:
            normalized[signature] = (vector - lower) / scale

    selected: list[tuple[str, tuple[float, ...]]] = []
    # Seed every phase at the point nearest its centroid when the budget permits.
    phases = sorted(phase_rows, key=lambda phase: (-len(phase_rows[phase]), phase))
    for phase in phases[:budget]:
        items = phase_rows[phase]
        centroid = np.mean([normalized[item[0]] for item in items], axis=0)
        selected.append(min(items, key=lambda item: np.linalg.norm(normalized[item[0]] - centroid))[0])
    selected_set = set(selected)
    while len(selected) < budget:
        best = None
        best_distance = -1.0
        for row in candidates:
            signature = _signature(row)
            if signature in selected_set:
                continue
            same_phase = [item for item in selected if item[0] == signature[0]]
            if not same_phase:
                distance = float("inf")
            else:
                distance = min(
                    float(np.linalg.norm(normalized[signature] - normalized[item])
                          / math.sqrt(len(signature[1])))
                    for item in same_phase
                )
            if distance > best_distance:
                best, best_distance = signature, distance
        assert best is not None
        selected.append(best)
        selected_set.add(best)
    return selected


def _normalized_feature_map(
    candidates: list[dict[str, Any]],
) -> tuple[
    dict[str, list[tuple[str, tuple[float, ...]]]],
    dict[tuple[str, tuple[float, ...]], np.ndarray],
]:
    by_phase: dict[str, list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    for row in candidates:
        signature = _signature(row)
        by_phase[signature[0]].append(signature)
    normalized: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
    for signatures in by_phase.values():
        matrix = np.asarray([signature[1] for signature in signatures], dtype=np.float64)
        lower, upper = matrix.min(axis=0), matrix.max(axis=0)
        scale = np.where(upper > lower, upper - lower, 1.0)
        for signature, vector in zip(signatures, matrix):
            normalized[signature] = (vector - lower) / scale
    return by_phase, normalized


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / math.sqrt(len(left)))


def _observed_residuals(
    signatures: list[tuple[str, tuple[float, ...]]],
    normalized: dict[tuple[str, tuple[float, ...]], np.ndarray],
    targets: dict[tuple[str, tuple[float, ...]], float],
) -> tuple[dict[tuple[str, tuple[float, ...]], float], float]:
    if len(signatures) < 2:
        return {signature: 0.0 for signature in signatures}, 0.0
    residuals = {}
    for signature in signatures:
        neighbors = sorted(
            (other for other in signatures if other != signature),
            key=lambda other: _rms_distance(normalized[signature], normalized[other]),
        )[:4]
        distances = np.asarray([
            _rms_distance(normalized[signature], normalized[other]) for other in neighbors
        ])
        weights = 1.0 / np.square(distances + 1e-6)
        prediction = float(np.dot(weights, [targets[other] for other in neighbors]) / weights.sum())
        residuals[signature] = min(1.0, abs(prediction - targets[signature]) / targets[signature])
    return residuals, float(_p95(list(residuals.values())))


def select_observed_error_aware(
    candidates: list[dict[str, Any]], budget: int, *, beta: float = 0.35,
    gamma: float = 0.50, exploration_floor: float = 0.50,
    phase_cap: float = 0.85, warm_start: int = 16,
) -> tuple[list[tuple[str, tuple[float, ...]]], list[dict[str, Any]]]:
    """Nested active selection using labels only after an anchor is selected."""
    if budget < 1 or budget > len(candidates):
        raise ValueError("budget must be within the unique candidate count")
    by_phase, normalized = _normalized_feature_map(candidates)
    phase_caps = {
        phase: min(len(signatures), math.ceil(phase_cap * len(signatures)))
        for phase, signatures in by_phase.items()
    }
    if budget > sum(phase_caps.values()):
        raise ValueError("budget exceeds observed-error phase caps")
    targets = {_signature(row): float(row["timing_us"]["step"]) for row in candidates}
    selected = select_phase_aware_farthest(candidates, min(warm_start, budget))
    trace: list[dict[str, Any]] = [{
        "selection_index": index, "phase": signature[0],
        "reason": "phase_aware_farthest_warm_start", "score": None,
        "distance": None, "local_disagreement": None,
        "local_observed_residual": None, "phase_observed_loo_error": None,
    } for index, signature in enumerate(selected)]

    selected_set = set(selected)
    while len(selected) < budget:
        residual_by_phase = {}
        phase_error = {}
        for phase in by_phase:
            phase_selected = [signature for signature in selected if signature[0] == phase]
            residual_by_phase[phase], phase_error[phase] = _observed_residuals(
                phase_selected, normalized, targets
            )
        best_signature = None
        best_details = None
        phase_selected_counts = {
            phase: sum(signature[0] == phase for signature in selected)
            for phase in by_phase
        }
        next_size = len(selected) + 1
        under_floor = {
            phase for phase, signatures in by_phase.items()
            if phase_selected_counts[phase] < math.floor(
                exploration_floor * next_size * len(signatures) / len(candidates)
            )
            and phase_selected_counts[phase] < phase_caps[phase]
        }
        for signature in sorted(
            (
                item for item in normalized
                if item not in selected_set
                and phase_selected_counts[item[0]] < phase_caps[item[0]]
                and (not under_floor or item[0] in under_floor)
            ),
            key=lambda item: (item[0], item[1]),
        ):
            phase_selected = [item for item in selected if item[0] == signature[0]]
            neighbors = sorted(
                phase_selected,
                key=lambda item: _rms_distance(normalized[signature], normalized[item]),
            )[:4]
            distances = np.asarray([
                _rms_distance(normalized[signature], normalized[item]) for item in neighbors
            ])
            weights = 1.0 / np.square(distances + 1e-6)
            neighbor_targets = np.asarray([targets[item] for item in neighbors])
            local_mean = float(np.dot(weights, neighbor_targets) / weights.sum())
            local_variance = float(
                np.dot(weights, np.square(neighbor_targets - local_mean)) / weights.sum()
            )
            disagreement = min(1.0, math.sqrt(local_variance) / max(local_mean, 1e-6))
            local_residual = min(1.0, float(np.dot(
                weights, [residual_by_phase[signature[0]][item] for item in neighbors]
            ) / weights.sum()))
            distance = float(distances[0])
            observed_error = phase_error[signature[0]]
            score = (
                distance + beta * disagreement + beta * local_residual
                + gamma * observed_error
            )
            details = {
                "selection_index": len(selected), "phase": signature[0],
                "reason": "observed_error_acquisition", "score": score,
                "distance": distance, "local_disagreement": disagreement,
                "local_observed_residual": local_residual,
                "phase_observed_loo_error": observed_error,
                "exploration_floor_active": bool(under_floor),
                "eligible_phases": sorted(under_floor) if under_floor else sorted(by_phase),
            }
            if best_details is None or score > best_details["score"]:
                best_signature, best_details = signature, details
        assert best_signature is not None and best_details is not None
        selected.append(best_signature)
        selected_set.add(best_signature)
        trace.append(best_details)
    return selected, trace


def _evaluate_selected_sequence(
    selection_representatives: list[dict[str, Any]],
    evaluation_representatives: list[dict[str, Any]],
    grouped: dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]],
    selected_sequence: list[tuple[str, tuple[float, ...]]],
    requested_budgets: list[int], *, neighbors: int, method: str,
    mark_selected_as_profiled: bool,
) -> list[dict[str, Any]]:
    results = []
    for budget in requested_budgets:
        selected_signatures = selected_sequence[:budget]
        selected_set = set(selected_signatures)
        train_rows = [row for signature in selected_signatures for row in grouped[signature]]
        started = time.perf_counter_ns()
        model = FixedConfigForwardOracle(neighbors=neighbors, method=method).fit(train_rows)
        training_wall_ms = (time.perf_counter_ns() - started) / 1e6
        evaluated = []
        for row in evaluation_representatives:
            actual = float(row["timing_us"]["step"])
            try:
                prediction = model.predict_raw(row, allow_unsupported=True)
            except ValueError:
                continue
            candidate = prediction.candidate_step_us
            evaluated.append({
                "signature": _signature(row), "phase": row["phase"],
                "actual_us": actual, "predicted_us": candidate,
                "ape": abs(candidate - actual) / actual * 100.0,
                "support_status": prediction.support_status,
                "prediction_wall_us": prediction.prediction_wall_us,
                "profiled_anchor": mark_selected_as_profiled and _signature(row) in selected_set,
            })
        supported = [row for row in evaluated if row["support_status"] == "supported"]
        unprofiled = [row for row in evaluated if not row["profiled_anchor"]]
        supported_unprofiled = [
            row for row in unprofiled if row["support_status"] == "supported"
        ]
        top_k = max(1, math.ceil(0.1 * len(evaluated)))
        actual_top = set(sorted(range(len(evaluated)), key=lambda i: evaluated[i]["actual_us"], reverse=True)[:top_k])
        predicted_top = set(sorted(range(len(evaluated)), key=lambda i: evaluated[i]["predicted_us"], reverse=True)[:top_k])
        artifact_buffer = io.BytesIO()
        joblib.dump(model, artifact_buffer)
        selected_medians = [statistics.median(
            float(row["timing_us"]["step"]) for row in grouped[signature]
        ) for signature in selected_signatures]
        results.append({
            "budget": budget, "selected_raw_record_count": len(train_rows),
            "selected_phase_counts": {
                phase: sum(signature[0] == phase for signature in selected_signatures)
                for phase in sorted({signature[0] for signature in selected_signatures})
            },
            "surface_metrics": _metrics(evaluated),
            "unprofiled_surface_metrics": _metrics(unprofiled),
            "unprofiled_by_phase": {
                phase: _metrics([row for row in unprofiled if row["phase"] == phase])
                for phase in ("prefill", "decode", "mixed")
            },
            "supported_fraction": len(supported) / len(evaluation_representatives),
            "supported_metrics": _metrics(supported),
            "unprofiled_supported_fraction": (
                len(supported_unprofiled) / len(unprofiled) if unprofiled else None
            ),
            "unprofiled_supported_metrics": _metrics(supported_unprofiled),
            "workload_surface_spearman": _spearman(
                [row["actual_us"] for row in evaluated],
                [row["predicted_us"] for row in evaluated],
            ),
            "slowest_workload_top_k": top_k,
            "slowest_workload_top_k_recall": len(actual_top & predicted_top) / top_k,
            "measured_forward_time_ms": sum(selected_medians) / 1000.0,
            "profiling_time_scope": "selected median forward CUDA time only; excludes startup/load/warmup",
            "training_wall_ms": training_wall_ms,
            "median_query_wall_us": statistics.median(row["prediction_wall_us"] for row in evaluated),
            "median_unprofiled_query_wall_us": (
                statistics.median(row["prediction_wall_us"] for row in unprofiled)
                if unprofiled else None
            ),
            "artifact_size_bytes": artifact_buffer.tell(),
            "selected_signatures": [
                {"phase": signature[0], "features": list(signature[1])}
                for signature in selected_signatures
            ],
        })
    return results


def evaluate_profiling_budgets(
    rows: list[dict[str, Any]], *, budgets: Iterable[int], neighbors: int = 4,
    method: str = "routing_rf", strategy: str = "phase_aware_farthest",
    protocol: str = "pool_surface_recovery", holdout_modulus: int = 5,
    holdout_fold: int = 0,
) -> dict[str, Any]:
    identities = {_stable_json(_config_identity(row)) for row in rows}
    if len(identities) != 1:
        raise ValueError("budget evaluation requires exactly one deployment identity")
    if protocol not in {"pool_surface_recovery", "fixed_workload_holdout"}:
        raise ValueError("unknown budget evaluation protocol")
    workload_ids = sorted({int(row["workload_index"]) for row in rows})
    if protocol == "fixed_workload_holdout":
        if holdout_modulus < 2 or not 0 <= holdout_fold < holdout_modulus:
            raise ValueError("holdout fold must be in [0, holdout_modulus)")
        test_workloads = {
            workload for index, workload in enumerate(workload_ids)
            if index % holdout_modulus == holdout_fold
        }
        selection_rows = [row for row in rows if int(row["workload_index"]) not in test_workloads]
        evaluation_rows = [row for row in rows if int(row["workload_index"]) in test_workloads]
    else:
        test_workloads = set()
        selection_rows = rows
        evaluation_rows = rows
    representatives, grouped = _unique_anchor_rows(selection_rows)
    evaluation_representatives, _ = _unique_anchor_rows(evaluation_rows)
    maximum = len(representatives)
    requested_budgets = list(dict.fromkeys(int(value) for value in budgets))
    if any(value < 1 or value > maximum for value in requested_budgets):
        raise ValueError(f"budgets must be in [1, {maximum}]")
    if strategy not in {"phase_aware_farthest", "observed_error_aware", "both"}:
        raise ValueError("unknown profiling selection strategy")
    strategies = (
        ("phase_aware_farthest", "observed_error_aware")
        if strategy == "both" else (strategy,)
    )
    curves = {}
    maximum_budget = max(requested_budgets)
    for current in strategies:
        if current == "phase_aware_farthest":
            sequence = select_phase_aware_farthest(representatives, maximum_budget)
            trace: list[dict[str, Any]] = []
            selection = {
                "strategy": current, "uses_latency_labels": False,
                "uses_unprofiled_latency_labels": False,
                "protocol": "retrospective candidate-pool replay",
            }
        else:
            sequence, trace = select_observed_error_aware(representatives, maximum_budget)
            selection = {
                "strategy": current, "uses_latency_labels": "profiled_anchors_only",
                "uses_unprofiled_latency_labels": False,
                "protocol": "retrospective candidate-pool active replay",
                "acquisition_config": {
                    "geometry_weight": 1.0, "local_disagreement_weight": 0.35,
                    "local_observed_residual_weight": 0.35,
                    "phase_observed_loo_error_weight": 0.50,
                    "initial_seeds": "phase_aware_farthest_warm_start",
                    "warm_start_anchors": 16,
                    "exploration_floor": 0.50,
                    "phase_cap": 0.85,
                },
            }
        curves[current] = {
            "selection": selection,
            "selection_trace": trace,
            "results": _evaluate_selected_sequence(
                representatives, evaluation_representatives, grouped, sequence,
                requested_budgets, neighbors=neighbors, method=method,
                mark_selected_as_profiled=protocol == "pool_surface_recovery",
            ),
        }
    common = {
        "schema_version": "fixed-config-budget-curve/v2",
        "deployment_identity": json.loads(next(iter(identities))),
        "unique_anchor_count": maximum, "budgets": requested_budgets,
        "method": method, "neighbors": neighbors,
        "evaluation_protocol": protocol,
        "train_workload_ids": [item for item in workload_ids if item not in test_workloads],
        "test_workload_ids": sorted(test_workloads),
        "test_unique_signature_count": len(evaluation_representatives),
        "test_features_visible_to_acquisition": protocol != "fixed_workload_holdout",
        "configuration_ranking": {
            "status": "not_applicable",
            "reason": "only one deployment configuration is present",
        },
    }
    if strategy == "both":
        baseline = {row["budget"]: row for row in curves["phase_aware_farthest"]["results"]}
        adaptive = {row["budget"]: row for row in curves["observed_error_aware"]["results"]}
        common["curves"] = curves
        common["comparison"] = [{
            "budget": budget,
            "delta_unprofiled_mape": (
                adaptive[budget]["unprofiled_surface_metrics"].get("mape", 0.0)
                - baseline[budget]["unprofiled_surface_metrics"].get("mape", 0.0)
            ),
            "delta_unprofiled_p95_ape": (
                adaptive[budget]["unprofiled_surface_metrics"].get("p95_ape", 0.0)
                - baseline[budget]["unprofiled_surface_metrics"].get("p95_ape", 0.0)
            ),
            "delta_top_k_recall": (
                adaptive[budget]["slowest_workload_top_k_recall"]
                - baseline[budget]["slowest_workload_top_k_recall"]
            ),
            "delta_surface_mape": (
                adaptive[budget]["surface_metrics"].get("mape", 0.0)
                - baseline[budget]["surface_metrics"].get("mape", 0.0)
            ),
            "delta_surface_p95_ape": (
                adaptive[budget]["surface_metrics"].get("p95_ape", 0.0)
                - baseline[budget]["surface_metrics"].get("p95_ape", 0.0)
            ),
            "delta_measured_forward_time_ms": (
                adaptive[budget]["measured_forward_time_ms"]
                - baseline[budget]["measured_forward_time_ms"]
            ),
        } for budget in requested_budgets]
        return common
    only = curves[strategy]
    common.update(only)
    return common


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_budget_holdout_folds(
    paths: Iterable[str | Path], *, target_mape: float = 3.0,
    target_p95_ape: float = 15.0, target_top_k_recall: float = 0.95,
    target_supported_fraction: float = 0.80,
) -> dict[str, Any]:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(reports) < 2:
        raise ValueError("at least two holdout folds are required")
    if any(report.get("evaluation_protocol") != "fixed_workload_holdout" for report in reports):
        raise ValueError("fold aggregation requires fixed_workload_holdout reports")
    if any(set(report.get("curves", {})) != {
        "phase_aware_farthest", "observed_error_aware"
    } for report in reports):
        raise ValueError("each fold must contain both selection strategies")
    budgets = reports[0]["budgets"]
    identity = _stable_json(reports[0]["deployment_identity"])
    if any(report["budgets"] != budgets for report in reports):
        raise ValueError("holdout folds use different budgets")
    if any(_stable_json(report["deployment_identity"]) != identity for report in reports):
        raise ValueError("holdout folds use different deployment identities")
    aggregated: dict[str, list[dict[str, Any]]] = {}
    for strategy in ("phase_aware_farthest", "observed_error_aware"):
        by_fold = [
            {row["budget"]: row for row in report["curves"][strategy]["results"]}
            for report in reports
        ]
        aggregated[strategy] = []
        for budget in budgets:
            fold_rows = [fold[budget] for fold in by_fold]
            phase_metrics = {}
            for phase in ("prefill", "decode", "mixed"):
                phase_rows = [row["unprofiled_by_phase"][phase] for row in fold_rows]
                phase_metrics[phase] = {
                    metric: _distribution([float(row[metric]) for row in phase_rows])
                    for metric in ("mape", "p95_ape")
                    if all(row.get("status") == "ok" and metric in row for row in phase_rows)
                }
            aggregated[strategy].append({
                "budget": budget,
                "test_mape": _distribution([
                    row["surface_metrics"]["mape"] for row in fold_rows
                ]),
                "test_p95_ape": _distribution([
                    row["surface_metrics"]["p95_ape"] for row in fold_rows
                ]),
                "test_supported_fraction": _distribution([
                    row["supported_fraction"] for row in fold_rows
                ]),
                "test_top_k_recall": _distribution([
                    row["slowest_workload_top_k_recall"] for row in fold_rows
                ]),
                "measured_forward_time_ms": _distribution([
                    row["measured_forward_time_ms"] for row in fold_rows
                ]),
                "selected_phase_counts": {
                    phase: _distribution([
                        float(row["selected_phase_counts"].get(phase, 0)) for row in fold_rows
                    ]) for phase in ("prefill", "decode", "mixed")
                },
                "test_by_phase": phase_metrics,
            })
    baseline = {row["budget"]: row for row in aggregated["phase_aware_farthest"]}
    adaptive = {row["budget"]: row for row in aggregated["observed_error_aware"]}
    paired = []
    recommended = None
    for budget in budgets:
        base_rows = [
            {row["budget"]: row for row in report["curves"]["phase_aware_farthest"]["results"]}[budget]
            for report in reports
        ]
        adaptive_rows = [
            {row["budget"]: row for row in report["curves"]["observed_error_aware"]["results"]}[budget]
            for report in reports
        ]
        delta = {
            "budget": budget,
            "adaptive_minus_baseline_mape": _distribution([
                adaptive_row["surface_metrics"]["mape"] - base_row["surface_metrics"]["mape"]
                for base_row, adaptive_row in zip(base_rows, adaptive_rows)
            ]),
            "adaptive_minus_baseline_p95_ape": _distribution([
                adaptive_row["surface_metrics"]["p95_ape"] - base_row["surface_metrics"]["p95_ape"]
                for base_row, adaptive_row in zip(base_rows, adaptive_rows)
            ]),
            "adaptive_minus_baseline_top_k_recall": _distribution([
                adaptive_row["slowest_workload_top_k_recall"]
                - base_row["slowest_workload_top_k_recall"]
                for base_row, adaptive_row in zip(base_rows, adaptive_rows)
            ]),
        }
        paired.append(delta)
        candidate = adaptive[budget]
        stable_vs_baseline = (
            candidate["test_mape"]["mean"] <= baseline[budget]["test_mape"]["mean"]
            and candidate["test_p95_ape"]["mean"] <= baseline[budget]["test_p95_ape"]["mean"]
            and candidate["test_top_k_recall"]["mean"] >= baseline[budget]["test_top_k_recall"]["mean"]
        )
        meets_targets = (
            candidate["test_mape"]["mean"] <= target_mape
            and candidate["test_p95_ape"]["mean"] <= target_p95_ape
            and candidate["test_top_k_recall"]["mean"] >= target_top_k_recall
            and candidate["test_supported_fraction"]["mean"] >= target_supported_fraction
        )
        if recommended is None and stable_vs_baseline and meets_targets:
            recommended = budget
    return {
        "schema_version": "fixed-config-budget-holdout-aggregate/v1",
        "fold_count": len(reports), "budgets": budgets,
        "deployment_identity": json.loads(identity),
        "selection_protocol": "fixed_workload_holdout_paired_folds",
        "targets": {
            "mape": target_mape, "p95_ape": target_p95_ape,
            "top_k_recall": target_top_k_recall,
            "supported_fraction": target_supported_fraction,
        },
        "strategies": aggregated, "paired_comparison": paired,
        "recommendation": {
            "status": "recommended" if recommended is not None else "no_stable_improvement",
            "strategy": "observed_error_aware" if recommended is not None else None,
            "minimum_anchor_budget": recommended,
            "selection_rule": "minimum budget meeting targets without mean regression vs baseline",
        },
        "hyperparameter_policy": "frozen_after_five_fold_evaluation",
    }

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path

from moe_perf_model.collective_model import NcclResponseTable
from moe_perf_model.execution_plan import StepExecutionPlan
from moe_perf_model.fixed_config_oracle import load_multirank_records
from moe_perf_model.fixed_config_oracle import features_from_raw
from moe_perf_model.budget_evaluation import select_phase_aware_farthest
from moe_perf_model.parallel_forward import ParallelForwardGraphCompiler, ParallelLayout
from moe_perf_model.rank_local_compute import (
    RankLocalComputeModel,
    load_rank_local_compute_artifact,
)
from moe_perf_model.structural_calibration import (
    PhaseAffineCalibrator,
    StructuralCalibrationSample,
)
from moe_perf_model.structural_forward import ForwardStructuralPredictor
from moe_perf_model.whitebox_model import load_whitebox_artifact


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    apes = [abs(left - right) / left * 100.0 for left, right in zip(actual, predicted)]
    ordered = sorted(apes)
    return {
        "count": len(apes),
        "mape": statistics.mean(apes),
        "p95_ape": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max_ape": max(apes),
    }


def _request_shape(row):
    """Return the realized request shape used to disambiguate scheduler steps."""
    return tuple(
        (
            request["phase"],
            int(request["scheduled_tokens"]),
            int(request["context_length"]),
        )
        for request in row["requests"]
    )


def _key(row):
    # A requested workload can realize more than one shape at the same logical
    # step index across repeats.  Omitting the request shape silently aliases
    # predictions and invalidates both aggregate metrics and config alignment.
    return (
        int(row["workload_index"]),
        int(row["repeat"]),
        int(row["step_index"]),
        row["phase"],
        _request_shape(row),
    )


def _representative_repeats(rows):
    groups = {}
    for row in rows:
        request_shape = _request_shape(row)
        key = (row["workload_index"], row["step_index"], row["phase"], request_shape)
        groups.setdefault(key, []).append(row)
    result = []
    for samples in groups.values():
        target = statistics.median(max(row["rank_step_us"]) for row in samples)
        representative = copy.deepcopy(min(
            samples, key=lambda row: abs(max(row["rank_step_us"]) - target)
        ))
        representative["repeat"] = 0
        representative["rank_step_us"] = [target] * representative["rank_count"]
        representative["timing_us"]["step"] = target
        result.append(representative)
    return sorted(result, key=_key)


def _max_rank_local_component_us(row):
    values = []
    for rank_record in row.get("rank_records", []):
        for layer in rank_record.get("layer_records", []):
            compute = layer.get("rank_local_compute_us", {})
            for stage in ("attention", "router", "expert_gemm"):
                value = compute.get(stage, {}).get("duration_us")
                if value is not None:
                    values.append(float(value))
    return max(values, default=0.0)


def _quality_filter(rows, max_component_us):
    if max_component_us is None:
        return rows, []
    rejected = [
        row for row in rows
        if _max_rank_local_component_us(row) > max_component_us
    ]
    accepted = [
        row for row in rows
        if _max_rank_local_component_us(row) <= max_component_us
    ]
    return accepted, [{
        "key": _key(row),
        "max_rank_local_component_us": _max_rank_local_component_us(row),
    } for row in rejected]


def _predict_config(args, rows, ep_size, rank_artifact):
    return _predict_with_rank_model(
        args, rows, ep_size, load_rank_local_compute_artifact(rank_artifact),
        label=f"EP{ep_size}",
    )


def _predict_with_rank_model(args, rows, ep_size, rank_model, *, label):
    compiler = ParallelForwardGraphCompiler(
        layout=ParallelLayout.vllm(2, 2, ep_size, 8),
        collective_backend=NcclResponseTable.load(args.nccl_response),
        collective_backend_name="nccl",
        hidden_size=2048,
        all2all_backend="allgather_reducescatter",
        rank_local_compute_backend=rank_model,
    )
    predictor = ForwardStructuralPredictor(
        compute_model=load_whitebox_artifact(args.whitebox), graph_compiler=compiler
    )
    output = []
    seen_keys = set()
    for index, row in enumerate(rows):
        prediction = predictor.predict(
            StepExecutionPlan.from_raw_step(row), include_trace=False
        )
        key = _key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate realized-step evaluation key: {key}")
        seen_keys.add(key)
        output.append({
            "key": key, "phase": row["phase"],
            "workload_index": int(row["workload_index"]),
            "actual_us": max(row["rank_step_us"]),
            "structural_us": prediction.step_us,
            "primary_bottleneck": prediction.primary_bottleneck,
            "bottleneck_shares": prediction.bottleneck_shares,
            "support_reasons": list(prediction.support_reasons),
        })
        print(f"{label}: {index + 1}/{len(rows)}", flush=True)
    return output


def _cross_validate(rows, folds):
    predictions = {}
    fold_reports = []
    for fold in range(folds):
        train = [row for row in rows if row["workload_index"] % folds != fold]
        test = [row for row in rows if row["workload_index"] % folds == fold]
        calibrator = PhaseAffineCalibrator().fit([
            StructuralCalibrationSample(
                row["phase"], row["structural_us"], row["actual_us"],
                row["workload_index"],
            ) for row in train
        ])
        actual = []
        predicted = []
        for row in test:
            value = calibrator.predict(row["phase"], row["structural_us"])
            predictions[tuple(row["key"])] = value
            actual.append(row["actual_us"])
            predicted.append(value)
        fold_reports.append({
            "fold": fold,
            "train_workloads": sorted({row["workload_index"] for row in train}),
            "test_workloads": sorted({row["workload_index"] for row in test}),
            "calibration": calibrator.to_dict(),
            "metrics": _metrics(actual, predicted),
        })
    actual = [row["actual_us"] for row in rows]
    predicted_values = [predictions[tuple(row["key"])] for row in rows]
    return predictions, {"folds": fold_reports, "aggregate": _metrics(actual, predicted_values)}


def _strict_cross_validate(args, raw_rows, representative_rows, ep_size, folds, budget):
    all_test_predictions = []
    calibrated_predictions = {}
    fold_reports = []
    for fold in range(folds):
        train_raw = [row for row in raw_rows if row["workload_index"] % folds != fold]
        train_representatives = [
            row for row in representative_rows if row["workload_index"] % folds != fold
        ]
        test = [row for row in representative_rows if row["workload_index"] % folds == fold]
        rank_model = RankLocalComputeModel().fit(train_raw)
        unique_candidates = {}
        for row in train_representatives:
            unique_candidates.setdefault(features_from_raw(row), row)
        candidate_rows = list(unique_candidates.values())
        anchor_budget = min(budget, len(candidate_rows))
        signatures = select_phase_aware_farthest(candidate_rows, anchor_budget)
        signature_set = set(signatures)
        anchor_by_signature = {}
        for row in train_representatives:
            signature = features_from_raw(row)
            if signature in signature_set and signature not in anchor_by_signature:
                anchor_by_signature[signature] = row
        anchors = [anchor_by_signature[signature] for signature in signatures]
        anchor_predictions = _predict_with_rank_model(
            args, anchors, ep_size, rank_model, label=f"EP{ep_size}/fold{fold}/anchor"
        )
        calibrator = PhaseAffineCalibrator().fit([
            StructuralCalibrationSample(
                row["phase"], row["structural_us"], row["actual_us"],
                row["workload_index"],
            ) for row in anchor_predictions
        ])
        test_predictions = _predict_with_rank_model(
            args, test, ep_size, rank_model, label=f"EP{ep_size}/fold{fold}/test"
        )
        actual = []
        calibrated = []
        for row in test_predictions:
            key = tuple(row["key"])
            value = calibrator.predict(row["phase"], row["structural_us"])
            calibrated_predictions[key] = value
            actual.append(row["actual_us"])
            calibrated.append(value)
        all_test_predictions.extend(test_predictions)
        fold_reports.append({
            "fold": fold,
            "train_workloads": sorted({row["workload_index"] for row in train_raw}),
            "test_workloads": sorted({row["workload_index"] for row in test}),
            "rank_local_database": rank_model.database.stats(),
            "anchor_budget": len(anchor_predictions),
            "anchor_signatures_selected_without_latency": True,
            "calibration": calibrator.to_dict(),
            "metrics": _metrics(actual, calibrated),
        })
    all_test_predictions.sort(key=lambda row: tuple(row["key"]))
    actual = [row["actual_us"] for row in all_test_predictions]
    calibrated = [calibrated_predictions[tuple(row["key"])] for row in all_test_predictions]
    return all_test_predictions, calibrated_predictions, {
        "folds": fold_reports,
        "aggregate": _metrics(actual, calibrated),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitebox", type=Path, required=True)
    parser.add_argument("--nccl-response", type=Path, required=True)
    parser.add_argument("--ep1-steps", type=Path, required=True)
    parser.add_argument("--ep2-steps", type=Path, required=True)
    parser.add_argument("--ep1-rank-compute", type=Path, required=True)
    parser.add_argument("--ep2-rank-compute", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--representative-repeats", action="store_true")
    parser.add_argument("--strict-component-holdout", action="store_true")
    parser.add_argument("--calibration-budget", type=int, default=32)
    parser.add_argument("--max-rank-local-component-us", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_inputs = {
        1: load_multirank_records(args.ep1_steps),
        2: load_multirank_records(args.ep2_steps),
    }
    quality_rejections = {}
    for ep_size in (1, 2):
        raw_inputs[ep_size], quality_rejections[ep_size] = _quality_filter(
            raw_inputs[ep_size], args.max_rank_local_component_us
        )
    inputs = raw_inputs
    if args.representative_repeats:
        inputs = {key: _representative_repeats(value) for key, value in inputs.items()}
    if args.strict_component_holdout:
        predicted = {}
        calibrated = {}
        calibration_reports = {}
        for ep_size in (1, 2):
            predicted[ep_size], calibrated[ep_size], calibration_reports[ep_size] = (
                _strict_cross_validate(
                    args, raw_inputs[ep_size], inputs[ep_size], ep_size,
                    args.folds, args.calibration_budget,
                )
            )
    else:
        predicted = {
            1: _predict_config(args, inputs[1], 1, args.ep1_rank_compute),
            2: _predict_config(args, inputs[2], 2, args.ep2_rank_compute),
        }
        calibrated = {}
        calibration_reports = {}
        for ep_size in (1, 2):
            calibrated[ep_size], calibration_reports[ep_size] = _cross_validate(
                predicted[ep_size], args.folds
            )
    raw_metrics = {
        str(ep): _metrics(
            [row["actual_us"] for row in predicted[ep]],
            [row["structural_us"] for row in predicted[ep]],
        ) for ep in (1, 2)
    }
    actual_maps = {
        ep: {tuple(row["key"]): row["actual_us"] for row in predicted[ep]}
        for ep in (1, 2)
    }
    common = sorted(set(actual_maps[1]) & set(actual_maps[2]))
    ranking = {
        "aligned_steps": len(common),
        "raw_top1_accuracy": statistics.mean(
            (actual_maps[1][key] <= actual_maps[2][key])
            == (next(row["structural_us"] for row in predicted[1] if tuple(row["key"]) == key)
                <= next(row["structural_us"] for row in predicted[2] if tuple(row["key"]) == key))
            for key in common
        ) if common else None,
        "calibrated_top1_accuracy": statistics.mean(
            (actual_maps[1][key] <= actual_maps[2][key])
            == (calibrated[1][key] <= calibrated[2][key])
            for key in common
        ) if common else None,
    }
    report = {
        "schema_version": "structural-forward-pilot-evaluation/v1",
        "protocol": (
            "strict_requested_workload_holdout_rank_db_rebuilt_phase_affine"
            if args.strict_component_holdout
            else "requested_workload_modulo_holdout_phase_affine"
        ),
        "important_boundary": (
            "rank-local component DB and phase calibration both exclude test workloads"
            if args.strict_component_holdout else
            "rank-local component DB is fitted on the full surface; only phase affine calibration is workload-held-out"
        ),
        "repeat_policy": (
            "median_target_representative_realized_shape"
            if args.representative_repeats else "all_repeats"
        ),
        "measurement_quality_gate": {
            "rule": "reject if any rank-local attention/router/expert_gemm CUDA Event exceeds fixed threshold",
            "threshold_us": args.max_rank_local_component_us,
            "uses_forward_label_or_prediction_error": False,
            "rejected": {str(key): value for key, value in quality_rejections.items()},
        },
        "raw_structural_metrics": raw_metrics,
        "phase_affine_holdout": {str(key): value for key, value in calibration_reports.items()},
        "configuration_ranking": ranking,
        "predictions": {str(key): value for key, value in predicted.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "raw": raw_metrics,
        "calibrated": {str(ep): calibration_reports[ep]["aggregate"] for ep in (1, 2)},
        "ranking": ranking,
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import math
import statistics
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .cross_config_calibration import (
    CrossConfigRatioCalibrator,
    train_cross_config_calibrator,
)
from .fixed_config_oracle import (
    FixedConfigForwardOracle,
    _config_identity,
    features_from_raw,
)


SERVICE_METRICS: dict[str, str] = {
    "mean_ttft_us": "lower",
    "mean_tpot_us": "lower",
    "p95_e2e_us": "lower",
    "output_token_throughput_per_s": "higher",
}


def _percentile(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _canonical_workload(row: dict[str, Any]) -> dict[str, Any]:
    workload = row["requested_workload"]
    batch = int(workload["batch_size"])
    lengths = workload.get("prompt_token_lengths")
    if lengths is None:
        lengths = [int(workload["prompt_tokens"])] * batch
    return {
        "batch_size": batch,
        "prompt_token_lengths": tuple(int(value) for value in lengths),
        "output_tokens": int(workload["output_tokens"]),
    }


def group_trajectories(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["workload_index"]), int(row["repeat"]))].append(row)
    for key, trajectory in grouped.items():
        trajectory.sort(key=lambda row: int(row["step_index"]))
        indices = [int(row["step_index"]) for row in trajectory]
        if indices != list(range(len(indices))):
            raise ValueError(f"trajectory {key} step indices must start at zero and be contiguous")
        workload = _canonical_workload(trajectory[0])
        if any(_canonical_workload(row) != workload for row in trajectory[1:]):
            raise ValueError(f"trajectory {key} changes requested workload")
    return dict(grouped)


def _slo_dict(
    *, ttft_slo_us: float | None, tpot_slo_us: float | None,
    e2e_slo_us: float | None,
) -> dict[str, Any]:
    values = {
        "ttft_us": ttft_slo_us, "tpot_us": tpot_slo_us, "e2e_us": e2e_slo_us,
    }
    for name, value in values.items():
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} SLO must be positive and finite")
    return {
        "status": "configured" if any(value is not None for value in values.values())
        else "not_configured",
        **values,
    }


def replay_multirank_trajectory(
    rows: list[dict[str, Any]],
    duration_provider: Callable[[dict[str, Any]], dict[str, Any]],
    *, ttft_slo_us: float | None = None, tpot_slo_us: float | None = None,
    e2e_slo_us: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("service replay requires a non-empty trajectory")
    rows = sorted(rows, key=lambda row: int(row["step_index"]))
    workload = _canonical_workload(rows[0])
    if any(_canonical_workload(row) != workload for row in rows):
        raise ValueError("trajectory changes requested workload")
    slo = _slo_dict(
        ttft_slo_us=ttft_slo_us, tpot_slo_us=tpot_slo_us,
        e2e_slo_us=e2e_slo_us,
    )
    current_us = 0.0
    request_tokens: dict[str, list[float]] = defaultdict(list)
    step_rows = []
    supported_steps = 0
    for expected_index, row in enumerate(rows):
        if int(row["step_index"]) != expected_index:
            raise ValueError("trajectory step indices must start at zero and be contiguous")
        prediction = duration_provider(row)
        duration_us = float(prediction["duration_us"])
        if not math.isfinite(duration_us) or duration_us <= 0:
            raise ValueError("step duration must be positive and finite")
        start_us, end_us = current_us, current_us + duration_us
        current_us = end_us
        emitted_ids = []
        for request in row["requests"]:
            request_id = str(request["request_id"])
            phase = str(request["phase"])
            emits = phase == "decode" or (
                phase == "prefill"
                and int(request["context_length"]) >= int(request["input_length"])
            )
            if emits:
                if len(request_tokens[request_id]) >= workload["output_tokens"]:
                    raise ValueError(f"request {request_id} emits after requested completion")
                request_tokens[request_id].append(end_us)
                emitted_ids.append(request_id)
        supported = prediction.get("support_status", "supported") == "supported"
        supported_steps += int(supported)
        step_rows.append({
            "step_index": expected_index, "phase": row["phase"],
            "start_us": start_us, "end_us": end_us, "duration_us": duration_us,
            "emitted_request_ids": emitted_ids,
            "support_status": prediction.get("support_status", "supported"),
            "support_reasons": prediction.get("support_reasons", []),
        })
    if len(request_tokens) != workload["batch_size"]:
        raise ValueError("trajectory does not contain the requested batch size")

    request_rows = []
    for request_id, times in sorted(request_tokens.items()):
        if len(times) != workload["output_tokens"]:
            raise ValueError(
                f"request {request_id} emitted {len(times)} tokens, expected "
                f"{workload['output_tokens']}"
            )
        ttft = times[0]
        tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) >= 2 else None
        e2e = times[-1]
        criteria = []
        if ttft_slo_us is not None:
            criteria.append(ttft <= ttft_slo_us)
        if tpot_slo_us is not None:
            criteria.append(tpot is not None and tpot <= tpot_slo_us)
        if e2e_slo_us is not None:
            criteria.append(e2e <= e2e_slo_us)
        request_rows.append({
            "request_id": request_id, "arrival_us": 0.0,
            "output_tokens": len(times), "ttft_us": ttft,
            "tpot_us": tpot, "e2e_us": e2e,
            "slo_attained": all(criteria) if criteria else None,
        })
    ttfts = [row["ttft_us"] for row in request_rows]
    tpots = [row["tpot_us"] for row in request_rows if row["tpot_us"] is not None]
    e2es = [row["e2e_us"] for row in request_rows]
    total_tokens = workload["batch_size"] * workload["output_tokens"]
    summary = {
        "step_count": len(rows), "request_count": len(request_rows),
        "elapsed_us": current_us,
        "mean_ttft_us": statistics.mean(ttfts), "p95_ttft_us": _percentile(ttfts, 0.95),
        "mean_tpot_us": statistics.mean(tpots) if tpots else None,
        "p95_tpot_us": _percentile(tpots, 0.95),
        "mean_e2e_us": statistics.mean(e2es), "p95_e2e_us": _percentile(e2es, 0.95),
        "output_token_throughput_per_s": total_tokens * 1e6 / current_us,
        "request_throughput_per_s": workload["batch_size"] * 1e6 / current_us,
        "supported_step_fraction": supported_steps / len(rows),
        "slo_status": slo["status"],
        "slo_attainment": (
            statistics.mean(row["slo_attained"] for row in request_rows)
            if slo["status"] == "configured" else None
        ),
    }
    return {
        "requested_workload": {
            **workload, "prompt_token_lengths": list(workload["prompt_token_lengths"])
        },
        "summary": summary, "requests": request_rows, "steps": step_rows,
        "semantics": {
            "execution": "real_scheduler_trajectory_forward_time_replay",
            "arrival_model": "static_batch_all_requests_arrive_at_zero",
            "token_visibility": "step_end",
            "excluded_time": [
                "scheduler_cpu", "output_processing", "dynamic_arrival_queueing"
            ],
        },
    }


def counterfactual_source_prediction(
    source_oracle: FixedConfigForwardOracle, target_row: dict[str, Any]
) -> Any:
    projected = copy.deepcopy(target_row)
    projected["profile_header"]["runtime"]["expert_parallel_size"] = int(
        source_oracle.identity["expert_parallel_size"]
    )
    phase, features = features_from_raw(projected)
    return source_oracle.predict_features(phase, features, allow_unsupported=True)


def _actual_duration(row: dict[str, Any]) -> dict[str, Any]:
    return {"duration_us": float(row["timing_us"]["step"]), "support_status": "supported"}


def _fixed_duration(
    oracle: FixedConfigForwardOracle, row: dict[str, Any]
) -> dict[str, Any]:
    prediction = oracle.predict_raw(row, allow_unsupported=True)
    return {
        "duration_us": prediction.candidate_step_us,
        "support_status": prediction.support_status,
        "support_reasons": [prediction.support_reason],
    }


def _calibrated_duration(
    source_oracle: FixedConfigForwardOracle,
    calibrator: CrossConfigRatioCalibrator,
    row: dict[str, Any],
) -> dict[str, Any]:
    source = counterfactual_source_prediction(source_oracle, row)
    target = calibrator.predict(source.candidate_step_us, row, allow_unsupported=True)
    supported = source.support_status == "supported" and target.support_status == "supported"
    return {
        "duration_us": target.candidate_target_us,
        "support_status": "supported" if supported else "not_predictable",
        "support_reasons": [
            f"source:{source.support_reason}", f"calibration:{target.support_reason}",
        ],
    }


def _ape(actual: float, predicted: float) -> float:
    return abs(predicted - actual) / actual * 100.0


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "no_values"}
    return {
        "status": "ok", "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "std": statistics.pstdev(values),
        "min": min(values), "max": max(values),
        "p95": _percentile(values, 0.95),
    }


def _winner(source_value: float, target_value: float, direction: str) -> str:
    if math.isclose(source_value, target_value, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    if direction == "lower":
        return "source" if source_value < target_value else "target"
    return "source" if source_value > target_value else "target"


def _ranking_row(
    metric: str, direction: str,
    source_actual: dict[str, Any], target_actual: dict[str, Any],
    source_predicted: dict[str, Any], target_predicted: dict[str, Any],
) -> dict[str, Any] | None:
    actual_values = (source_actual.get(metric), target_actual.get(metric))
    predicted_values = (source_predicted.get(metric), target_predicted.get(metric))
    if any(value is None for value in (*actual_values, *predicted_values)):
        return None
    actual_winner = _winner(float(actual_values[0]), float(actual_values[1]), direction)
    predicted_winner = _winner(float(predicted_values[0]), float(predicted_values[1]), direction)
    if predicted_winner == "tie":
        gap = 0.0 if actual_winner == "tie" else None
    else:
        selected = float(actual_values[0 if predicted_winner == "source" else 1])
        best = min(actual_values) if direction == "lower" else max(actual_values)
        gap = abs(float(best) - selected) / max(abs(float(best)), 1e-12) * 100.0
    return {
        "actual_winner": actual_winner, "predicted_winner": predicted_winner,
        "correct": actual_winner == predicted_winner,
        "actual_optimality_gap_percent": gap,
    }


def evaluate_service_configuration_ranking(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
    source_oracle: FixedConfigForwardOracle, target_oracle: FixedConfigForwardOracle,
    *, target_calibration_budget: int = 32, folds: int = 5,
    ttft_slo_ms: float | None = None, tpot_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    if source_oracle.identity != _config_identity(source_rows[0]):
        raise ValueError("source oracle identity mismatch")
    if target_oracle.identity != _config_identity(target_rows[0]):
        raise ValueError("target oracle identity mismatch")
    source_groups = group_trajectories(source_rows)
    target_groups = group_trajectories(target_rows)
    common_keys = sorted(set(source_groups) & set(target_groups))
    compatible = [
        key for key in common_keys
        if _canonical_workload(source_groups[key][0]) == _canonical_workload(target_groups[key][0])
    ]
    workload_ids = sorted({key[0] for key in compatible})
    slo_us = {
        "ttft_slo_us": ttft_slo_ms * 1000.0 if ttft_slo_ms is not None else None,
        "tpot_slo_us": tpot_slo_ms * 1000.0 if tpot_slo_ms is not None else None,
        "e2e_slo_us": e2e_slo_ms * 1000.0 if e2e_slo_ms is not None else None,
    }
    slo = _slo_dict(**slo_us)
    ranking_metrics = dict(SERVICE_METRICS)
    if slo["status"] == "configured":
        ranking_metrics["slo_attainment"] = "higher"

    fold_reports = []
    all_trajectory_rows = []
    for fold in range(folds):
        test_workloads = [value for index, value in enumerate(workload_ids) if index % folds == fold]
        test_set = set(test_workloads)
        calibrator, calibration_report = train_cross_config_calibrator(
            source_rows,
            [row for row in target_rows if int(row["workload_index"]) not in test_set],
            source_oracle, budget=target_calibration_budget,
            method="local_ratio_idw",
        )
        trajectories = []
        for key in compatible:
            if key[0] not in test_set:
                continue
            source_trajectory, target_trajectory = source_groups[key], target_groups[key]
            actual_source = replay_multirank_trajectory(
                source_trajectory, _actual_duration, **slo_us
            )
            actual_target = replay_multirank_trajectory(
                target_trajectory, _actual_duration, **slo_us
            )
            predicted_source = replay_multirank_trajectory(
                source_trajectory, lambda row: _fixed_duration(source_oracle, row), **slo_us
            )
            predicted_target_full = replay_multirank_trajectory(
                target_trajectory, lambda row: _fixed_duration(target_oracle, row), **slo_us
            )
            predicted_target_calibrated = replay_multirank_trajectory(
                target_trajectory,
                lambda row: _calibrated_duration(source_oracle, calibrator, row),
                **slo_us,
            )
            ranking = {}
            for mode, target_prediction in (
                ("independent_full_oracles", predicted_target_full),
                ("source_plus_32_anchor_calibration", predicted_target_calibrated),
            ):
                ranking[mode] = {
                    metric: _ranking_row(
                        metric, direction,
                        actual_source["summary"], actual_target["summary"],
                        predicted_source["summary"], target_prediction["summary"],
                    )
                    for metric, direction in ranking_metrics.items()
                }
            trajectory_row = {
                "workload_index": key[0], "repeat": key[1],
                "requested_workload": actual_source["requested_workload"],
                "source_step_count": len(source_trajectory),
                "target_step_count": len(target_trajectory),
                "step_sequences_identical": _step_sequences_identical(
                    source_trajectory, target_trajectory
                ),
                "actual": {
                    "source": actual_source["summary"], "target": actual_target["summary"]
                },
                "predicted": {
                    "source_full": predicted_source["summary"],
                    "target_full": predicted_target_full["summary"],
                    "target_calibrated": predicted_target_calibrated["summary"],
                },
                "ranking": ranking,
            }
            trajectories.append(trajectory_row)
            all_trajectory_rows.append(trajectory_row)
        fold_reports.append({
            "fold": fold,
            "train_workload_ids": sorted(set(workload_ids) - test_set),
            "test_workload_ids": test_workloads,
            "calibration": calibration_report,
            "trajectories": trajectories,
        })

    service_error = {}
    for config, actual_key, predicted_keys in (
        ("source", "source", ("source_full",)),
        ("target", "target", ("target_full", "target_calibrated")),
    ):
        service_error[config] = {}
        for prediction_key in predicted_keys:
            service_error[config][prediction_key] = {
                metric: _distribution([
                    _ape(
                        float(row["actual"][actual_key][metric]),
                        float(row["predicted"][prediction_key][metric]),
                    )
                    for row in all_trajectory_rows
                    if row["actual"][actual_key].get(metric) is not None
                    and row["predicted"][prediction_key].get(metric) is not None
                ])
                for metric in SERVICE_METRICS
            }
            if slo["status"] == "configured":
                service_error[config][prediction_key]["slo_attainment_absolute_error"] = (
                    _distribution([
                        abs(
                            float(row["actual"][actual_key]["slo_attainment"])
                            - float(row["predicted"][prediction_key]["slo_attainment"])
                        )
                        for row in all_trajectory_rows
                    ])
                )
            service_error[config][prediction_key]["supported_step_fraction"] = (
                _distribution([
                    float(row["predicted"][prediction_key]["supported_step_fraction"])
                    for row in all_trajectory_rows
                ])
            )

    ranking_aggregate = {}
    for mode in ("independent_full_oracles", "source_plus_32_anchor_calibration"):
        ranking_aggregate[mode] = {}
        for metric in ranking_metrics:
            values = [
                row["ranking"][mode][metric] for row in all_trajectory_rows
                if row["ranking"][mode][metric] is not None
            ]
            gaps = [
                value["actual_optimality_gap_percent"] for value in values
                if value["actual_optimality_gap_percent"] is not None
            ]
            ranking_aggregate[mode][metric] = {
                "count": len(values),
                "top1_configuration_accuracy": (
                    statistics.mean(value["correct"] for value in values) if values else None
                ),
                "optimality_gap_percent": _distribution(gaps),
                "actual_winner_counts": {
                    winner: sum(value["actual_winner"] == winner for value in values)
                    for winner in ("source", "target", "tie")
                },
                "predicted_winner_counts": {
                    winner: sum(value["predicted_winner"] == winner for value in values)
                    for winner in ("source", "target", "tie")
                },
            }
    actual_config_comparison = {}
    for metric, direction in SERVICE_METRICS.items():
        ratios = [
            float(row["actual"]["target"][metric])
            / float(row["actual"]["source"][metric])
            for row in all_trajectory_rows
            if row["actual"]["source"].get(metric) is not None
            and row["actual"]["target"].get(metric) is not None
        ]
        actual_config_comparison[metric] = {
            "target_over_source_ratio": _distribution(ratios),
            "target_better_fraction": statistics.mean(
                ratio < 1.0 if direction == "lower" else ratio > 1.0
                for ratio in ratios
            ) if ratios else None,
        }
    shape_sequence_analysis = {}
    for subset_name, expected_identical in (
        ("identical_step_sequence", True), ("different_step_sequence", False)
    ):
        subset = [
            row for row in all_trajectory_rows
            if bool(row["step_sequences_identical"]) == expected_identical
        ]
        shape_sequence_analysis[subset_name] = {
            "trajectory_count": len(subset),
            "target_calibrated_service_mape": {
                metric: _distribution([
                    _ape(
                        float(row["actual"]["target"][metric]),
                        float(row["predicted"]["target_calibrated"][metric]),
                    )
                    for row in subset
                    if row["actual"]["target"].get(metric) is not None
                    and row["predicted"]["target_calibrated"].get(metric) is not None
                ])
                for metric in SERVICE_METRICS
            },
            "target_calibrated_supported_step_fraction": _distribution([
                float(row["predicted"]["target_calibrated"]["supported_step_fraction"])
                for row in subset
            ]),
        }
    return {
        "schema_version": "service-config-ranking/v1",
        "protocol": "requested_workload_5fold_real_scheduler_trajectory_forward_time_replay",
        "source_identity": source_oracle.identity, "target_identity": target_oracle.identity,
        "trajectory_pairing": {
            "common_workload_repeat_count": len(common_keys),
            "compatible_requested_workload_repeat_count": len(compatible),
            "incompatible_requested_workload_repeat_count": len(common_keys) - len(compatible),
            "requires_step_alignment": False,
            "different_step_sequence_count": sum(
                not _step_sequences_identical(source_groups[key], target_groups[key])
                for key in compatible
            ),
        },
        "target_calibration_budget": target_calibration_budget,
        "fold_count": folds, "slo": slo,
        "service_error": service_error,
        "configuration_ranking": ranking_aggregate,
        "actual_config_comparison": actual_config_comparison,
        "shape_sequence_analysis": shape_sequence_analysis,
        "folds": fold_reports,
        "provenance": {
            "real_scheduler_trajectory": True,
            "step_actual_label": "cuda_event_multirank_slowest_rank",
            "target_test_workloads_visible_to_calibration": False,
            "target_mismatched_step_policy": (
                "retain target trajectory and query counterfactual source oracle features"
            ),
            "counterfactual_source_signal": True,
            "candidate_estimates_used_for_offline_evaluation": True,
            "excluded_time": [
                "scheduler_cpu", "output_processing", "dynamic_arrival_queueing"
            ],
        },
    }


def _step_sequences_identical(
    source: list[dict[str, Any]], target: list[dict[str, Any]]
) -> bool:
    if len(source) != len(target):
        return False
    for left, right in zip(source, target):
        left_shape = tuple(
            (request["phase"], int(request["scheduled_tokens"]), int(request["context_length"]))
            for request in left["requests"]
        )
        right_shape = tuple(
            (request["phase"], int(request["scheduled_tokens"]), int(request["context_length"]))
            for request in right["requests"]
        )
        if left["phase"] != right["phase"] or left_shape != right_shape:
            return False
    return True

from __future__ import annotations

import copy

from moe_perf_model.candidate_pruning import (
    build_candidate_table_from_service_report,
    prune_candidate_table,
)
from moe_perf_model.fixed_config_oracle import _config_identity

from test_fixed_config_oracle import _row


def _prediction(*, latency=(60, 70, 80), throughput=(110, 120, 130), coverage=1.0,
                memory=100):
    return {
        "metrics": {
            "ttft_us": {"lower": latency[0], "point": latency[1], "upper": latency[2]},
            "tpot_us": {"lower": latency[0], "point": latency[1], "upper": latency[2]},
            "e2e_us": {"lower": latency[0], "point": latency[1], "upper": latency[2]},
            "throughput_per_s": {
                "lower": throughput[0], "point": throughput[1], "upper": throughput[2]
            },
        },
        "coverage": coverage,
        "prediction_status": "supported" if coverage == 1 else "partially_supported",
        "memory_bytes_per_gpu": memory,
    }


def _candidate(candidate_id, prediction, *, launch="measured_success", cost=2.0):
    return {
        "candidate_id": candidate_id,
        "parallel": {"world_size": 2, "tensor_parallel_size": 2,
                     "expert_parallel_size": 1},
        "launch_status": launch, "cost_units": cost,
        "predictions": {"w0": prediction},
    }


def _table(candidates, *, arrival=None, slo=None):
    return {
        "schema_version": "config-candidate-table/v1",
        "hardware_pool": {"available_gpus": 2, "gpu_memory_bytes_per_device": 200},
        "workloads": [{
            "workload_id": "w0", "arrival_rate_rps": arrival,
            "request_shape": {"output_tokens": 1},
            "slo": slo or {"ttft_us": None, "tpot_us": None, "e2e_us": None},
        }],
        "candidates": candidates,
    }


def _decision(report, candidate_id):
    return next(row for row in report["decisions"] if row["candidate_id"] == candidate_id)


def test_low_coverage_is_queued_instead_of_pruned():
    report = prune_candidate_table(_table([
        _candidate("c", _prediction(coverage=0.5))
    ]))
    assert _decision(report, "c")["status"] == "needs_profile"
    assert report["profile_queue"][0]["candidate_id"] == "c"


def test_memory_and_parallel_infeasibility_are_hard_pruned():
    memory = _candidate("memory", _prediction(memory=191))
    parallel = _candidate("parallel", _prediction())
    parallel["parallel"]["world_size"] = 3
    report = prune_candidate_table(_table([memory, parallel]))
    assert _decision(report, "memory")["status"] == "pruned"
    assert "memory_capacity_exceeded" in _decision(report, "memory")["reasons"]
    assert "parallel_layout_incompatible" in _decision(report, "parallel")["reasons"]


def test_slo_policy_distinguishes_risk_from_certain_violation():
    table = _table([_candidate("c", _prediction(latency=(70, 80, 90)))], slo={
        "ttft_us": 85, "tpot_us": None, "e2e_us": None,
    })
    risk = prune_candidate_table(table)
    decision = _decision(risk, "c")
    assert decision["status"] == "pruned"
    assert not decision["rules"]["slo"]["metrics"]["ttft_us"]["certain_violation"]
    certain = prune_candidate_table(table, {
        "slo_policy": "certain_violation_lower_bound"
    })
    assert _decision(certain, "c")["status"] == "retained"


def test_insufficient_capacity_requests_replica_evaluation():
    report = prune_candidate_table(_table([
        _candidate("c", _prediction(throughput=(60, 70, 80)))
    ], arrival=100))
    decision = _decision(report, "c")
    assert decision["status"] == "needs_replica_evaluation"
    assert decision["rules"]["capacity"]["required_replicas"] == 2


def test_conservative_interval_pareto_prunes_only_separated_candidate():
    fast = _candidate("fast", _prediction(
        latency=(50, 60, 70), throughput=(120, 130, 140)
    ))
    slow = _candidate("slow", _prediction(
        latency=(80, 90, 100), throughput=(90, 95, 110)
    ))
    report = prune_candidate_table(_table([fast, slow]))
    assert _decision(report, "fast")["status"] == "retained"
    assert _decision(report, "slow")["status"] == "pruned"
    assert _decision(report, "slow")["rules"]["pareto"]["dominators"] == ["fast"]


def test_overlapping_intervals_are_not_point_pruned():
    left = _candidate("left", _prediction(
        latency=(50, 60, 80), throughput=(100, 120, 130)
    ))
    right = _candidate("right", _prediction(
        latency=(70, 75, 90), throughput=(90, 100, 115)
    ))
    report = prune_candidate_table(_table([left, right]))
    assert {_decision(report, value)["status"] for value in ("left", "right")} == {"retained"}


def test_candidate_builder_uses_predictions_not_actual_service_labels():
    source = _row(8, 100, 0)
    source.update({"repeat": 0, "step_index": 0, "peak_memory_bytes": 100})
    target = copy.deepcopy(source)
    target["profile_header"]["runtime"]["expert_parallel_size"] = 1
    target["peak_memory_bytes"] = 90
    summary = {
        "mean_ttft_us": 70.0, "mean_tpot_us": 60.0,
        "p95_e2e_us": 200.0, "output_token_throughput_per_s": 20.0,
        "supported_step_fraction": 1.0,
    }
    trajectory = {
        "workload_index": 0,
        "requested_workload": {
            "batch_size": 1, "prompt_token_lengths": [8], "output_tokens": 2,
        },
        "actual": {"source": {"mean_ttft_us": 99999}, "target": {}},
        "predicted": {"source_full": summary, "target_calibrated": summary},
    }
    errors = {
        metric: {"p95": 10.0} for metric in (
            "mean_ttft_us", "mean_tpot_us", "p95_e2e_us",
            "output_token_throughput_per_s",
        )
    }
    report = {
        "schema_version": "service-config-ranking/v1",
        "source_identity": _config_identity(source),
        "target_identity": _config_identity(target),
        "slo": {"ttft_us": None, "tpot_us": None, "e2e_us": None},
        "folds": [{"trajectories": [trajectory]}],
        "service_error": {
            "source": {"source_full": errors},
            "target": {"target_calibrated": errors},
        },
    }
    table = build_candidate_table_from_service_report(
        report, [source], [target], gpu_memory_bytes=200, available_gpus=2
    )
    point = table["candidates"][0]["predictions"]["w0"]["metrics"]["ttft_us"]["point"]
    assert point == 70.0
    assert table["provenance"]["uses_actual_service_labels_as_pruning_features"] is False

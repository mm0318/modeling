from __future__ import annotations

import copy

from moe_perf_model.fixed_config_oracle import FixedConfigForwardOracle
from moe_perf_model.service_replay import (
    evaluate_service_configuration_ranking,
    replay_multirank_trajectory,
)

from test_fixed_config_oracle import _row


def _trajectory(workload: int, prompt: int, scale: float, *, ep: int):
    request_id = f"r-{ep}-{workload}"
    rows = []
    for step_index, (phase, scheduled, context) in enumerate((
        ("prefill", prompt, prompt), ("decode", 1, prompt + 1),
    )):
        row = _row(scheduled, scale * (100 + prompt + step_index * 10), workload)
        row.update({"repeat": 0, "step_index": step_index, "phase": phase})
        row["profile_header"]["runtime"]["expert_parallel_size"] = ep
        row["requests"] = [{
            "request_id": request_id, "phase": phase,
            "scheduled_tokens": scheduled, "context_length": context,
            "prior_context_length": 0 if phase == "prefill" else prompt,
            "input_length": prompt,
        }]
        row["requested_workload"] = {
            "batch_size": 1, "prompt_tokens": prompt,
            "prompt_token_lengths": None, "output_tokens": 2, "repeats": 1,
        }
        rows.append(row)
    return rows


def test_service_replay_computes_step_end_ttft_tpot_and_slo():
    rows = _trajectory(0, 8, 1.0, ep=2)
    result = replay_multirank_trajectory(
        rows,
        lambda row: {"duration_us": row["timing_us"]["step"]},
        ttft_slo_us=200_000, tpot_slo_us=200_000,
    )
    first, second = [row["timing_us"]["step"] for row in rows]
    assert result["summary"]["mean_ttft_us"] == first
    assert result["summary"]["mean_tpot_us"] == second
    assert result["summary"]["slo_attainment"] == 1.0


def test_service_ranking_uses_independent_trajectories_and_holdout_calibration():
    source, target = [], []
    for workload, prompt in enumerate((4, 8, 16, 32, 64, 128)):
        source.extend(_trajectory(workload, prompt, 1.0, ep=2))
        target.extend(_trajectory(workload, prompt, 0.9, ep=1))
    # Make one target scheduler trajectory differ without removing the workload.
    target[1] = copy.deepcopy(target[1])
    target[1]["requests"][0]["context_length"] += 1
    source_oracle = FixedConfigForwardOracle().fit(source)
    target_oracle = FixedConfigForwardOracle().fit(target)
    report = evaluate_service_configuration_ranking(
        source, target, source_oracle, target_oracle,
        target_calibration_budget=2, folds=2,
        ttft_slo_ms=1.0, tpot_slo_ms=1.0, e2e_slo_ms=2.0,
    )
    assert report["trajectory_pairing"]["requires_step_alignment"] is False
    assert report["trajectory_pairing"]["different_step_sequence_count"] == 1
    assert report["provenance"]["target_test_workloads_visible_to_calibration"] is False
    ranking = report["configuration_ranking"]["source_plus_32_anchor_calibration"]
    assert ranking["output_token_throughput_per_s"]["count"] == 6

from __future__ import annotations

import pytest

from moe_perf_model.replay import load_replay_steps, replay_steps


def test_scheduler_replay_outputs_request_metrics(demo_bundle):
    _, paths, _, model = demo_bundle
    result = replay_steps(model, load_replay_steps(paths["replay_trace"]))
    summary = result["summary"]
    assert summary["step_count"] == 3
    assert summary["request_count"] == 2
    assert summary["completed_requests"] == 2
    assert summary["mean_ttft_us"] > 0
    assert summary["mean_tpot_us"] > 0
    assert summary["output_token_throughput_per_s"] > 0
    assert summary["max_predicted_memory_bytes"] > 0
    assert all(row["finished"] for row in result["requests"])


from __future__ import annotations

import copy

from moe_perf_model.configuration_comparison import compare_fixed_config_records

from test_fixed_config_oracle import _row


def _indexed(tokens: int, step: float, workload: int):
    row = _row(tokens, step, workload)
    row.update({"repeat": 0, "step_index": 0})
    return row


def test_configuration_comparison_only_uses_exact_realized_shape_pairs():
    source = [_indexed(8, 100, 0), _indexed(16, 200, 1)]
    target = copy.deepcopy(source)
    for row in target:
        row["profile_header"]["runtime"]["expert_parallel_size"] = 1
    target[0]["timing_us"]["step"] = 90
    target[1]["requests"][0]["context_length"] = 17
    report = compare_fixed_config_records(source, target)
    assert report["key_aligned_step_count"] == 2
    assert report["shape_aligned_step_count"] == 1
    assert report["overall"]["median_target_over_source"] == 0.9
    assert report["interpretation"].endswith("not a cross-config predictor")

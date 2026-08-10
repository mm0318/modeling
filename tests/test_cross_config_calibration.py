from __future__ import annotations

import copy

import pytest

from moe_perf_model.cross_config_calibration import (
    CrossConfigRatioCalibrator,
    build_aligned_pairs,
    load_cross_config_artifact,
    save_cross_config_artifact,
    train_cross_config_calibrator,
)
from moe_perf_model.fixed_config_oracle import FixedConfigForwardOracle

from test_fixed_config_oracle import _row


def _indexed(tokens: int, step: float, workload: int, *, ep: int):
    row = _row(tokens, step, workload)
    row.update({"repeat": 0, "step_index": 0})
    row["profile_header"]["runtime"]["expert_parallel_size"] = ep
    return row


def _pair(tokens: int, source_us: float, target_us: float, workload: int):
    source = _indexed(tokens, source_us, workload, ep=2)
    target = _indexed(tokens, target_us, workload, ep=1)
    return {
        "source_row": source, "target_row": target,
        "source_us": source_us, "workload_index": workload, "phase": "prefill",
    }


def test_ratio_calibrator_exact_anchor_and_identity_rejection():
    pairs = [_pair(8, 100, 90, 0), _pair(32, 400, 320, 1)]
    model = CrossConfigRatioCalibrator().fit(pairs)
    prediction = model.predict(100, pairs[0]["target_row"])
    assert prediction.target_us == pytest.approx(90)
    assert prediction.ratio == pytest.approx(0.9)
    wrong = copy.deepcopy(pairs[0]["target_row"])
    wrong["profile_header"]["hardware"]["hardware_id"] = "other"
    with pytest.raises(ValueError, match="target calibration identity"):
        model.predict(100, wrong)


def test_alignment_excludes_request_shape_mismatch():
    source = [_indexed(8, 100, 0, ep=2), _indexed(16, 200, 1, ep=2)]
    target = [_indexed(8, 90, 0, ep=1), _indexed(16, 180, 1, ep=1)]
    target[1]["requests"][0]["context_length"] = 17
    oracle = FixedConfigForwardOracle().fit(source)
    pairs, report = build_aligned_pairs(source, target, oracle)
    assert len(pairs) == 1
    assert report["shape_mismatch_excluded_count"] == 1


def test_training_budget_is_nested_label_free_and_artifact_roundtrip(tmp_path):
    source = [_indexed(tokens, tokens * 10, index, ep=2)
              for index, tokens in enumerate((4, 8, 16, 32, 64, 128))]
    target = [_indexed(tokens, tokens * 9, index, ep=1)
              for index, tokens in enumerate((4, 8, 16, 32, 64, 128))]
    oracle = FixedConfigForwardOracle().fit(source)
    small, _ = train_cross_config_calibrator(source, target, oracle, budget=2)
    large, _ = train_cross_config_calibrator(source, target, oracle, budget=4)
    small_signatures = {
        (row["phase"], tuple(row["features"]))
        for row in small.provenance["selected_target_signatures"]
    }
    large_signatures = {
        (row["phase"], tuple(row["features"]))
        for row in large.provenance["selected_target_signatures"]
    }
    assert small_signatures < large_signatures
    assert not small.provenance["uses_unprofiled_target_latency_labels"]
    path = tmp_path / "calibration.joblib"
    save_cross_config_artifact(small, path)
    restored = load_cross_config_artifact(path)
    assert restored.predict(40, target[0], allow_unsupported=True).candidate_target_us > 0

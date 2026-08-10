from __future__ import annotations

import copy

import pytest

from moe_perf_model.fixed_config_oracle import FixedConfigForwardOracle, features_from_raw


def _row(tokens: int, step: float, workload: int = 0):
    histogram = [tokens, tokens, 0, 0]
    return {
        "schema_version": "multirank-forward/v1", "label_source": "cuda_event_multirank",
        "rank_count": 2, "workload_index": workload, "phase": "prefill",
        "profile_header": {
            "model": {"model_id": "m"}, "hardware": {"hardware_id": "h"},
            "runtime": {"tensor_parallel_size": 2, "expert_parallel_size": 2,
                        "vllm_version": "0.18.1", "precision": "bfloat16",
                        "cuda_graph": False, "all2all_backend": "x",
                        "expert_placement_strategy": "linear"},
        },
        "kernel_identity": {"kernel": "k"},
        "requests": [{"phase": "prefill", "scheduled_tokens": tokens,
                      "context_length": tokens}],
        "layer_records": [{"tokens_per_expert": histogram}],
        "timing_us": {"step": step},
    }


def test_fixed_config_exact_signature_uses_duplicate_median():
    rows = [_row(8, 100), _row(8, 300), _row(32, 500)]
    model = FixedConfigForwardOracle().fit(rows)
    prediction = model.predict_raw(rows[0])
    assert prediction.step_us == 200
    assert prediction.nearest_distance == 0


def test_fixed_config_rejects_identity_mismatch_and_marks_extrapolation():
    rows = [_row(8, 100), _row(32, 500)]
    model = FixedConfigForwardOracle().fit(rows)
    outside = _row(64, 700)
    prediction = model.predict_raw(outside)
    assert prediction.extrapolated
    assert prediction.support_status == "not_predictable"
    assert prediction.needs_profile
    assert prediction.step_us is None
    assert model.predict_raw(outside, allow_unsupported=True).step_us is not None
    wrong = copy.deepcopy(rows[0])
    wrong["profile_header"]["hardware"]["hardware_id"] = "other"
    with pytest.raises(ValueError, match="configuration"):
        model.predict_raw(wrong)


def test_feature_vector_includes_ep_load_imbalance():
    balanced = _row(8, 100)
    imbalanced = _row(8, 100)
    imbalanced["layer_records"][0]["tokens_per_expert"] = [16, 0, 0, 0]
    assert features_from_raw(balanced)[1] != features_from_raw(imbalanced)[1]


def test_routing_rf_keeps_exact_signature_median():
    rows = [_row(8, 100), _row(8, 300), _row(16, 250), _row(32, 500)]
    model = FixedConfigForwardOracle(method="routing_rf").fit(rows)
    assert model.predict_raw(rows[0]).step_us == 200

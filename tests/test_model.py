from __future__ import annotations

import pytest

from moe_perf_model.demo import _config
from moe_perf_model.model import ComponentModel
from moe_perf_model.schema import InferenceConfig, SchemaError


def test_synthetic_records_require_explicit_opt_in(demo_bundle):
    _, _, records, _ = demo_bundle
    with pytest.raises(SchemaError, match="allow_synthetic"):
        ComponentModel().fit(records)


def test_component_prediction_is_complete(demo_bundle):
    _, paths, _, model = demo_bundle
    from moe_perf_model.schema import load_config

    profile = model.predict(load_config(paths["predict_config"]))
    component_sum = (
        profile.attention_us + profile.expert_us + profile.dense_us + profile.other_us
    )
    assert profile.step_us == pytest.approx(component_sum)
    assert sum(profile.component_shares.values()) == pytest.approx(1.0)
    assert profile.primary_bottleneck in profile.component_shares
    assert profile.memory_bytes > 61_000_000_000
    assert profile.artifact_provenance["label_source"] == "synthetic_fixture"
    assert profile.artifact_provenance["calibration_scope"] == "synthetic-fixture"


def test_routing_only_changes_expert_for_same_shape(demo_bundle):
    _, _, _, model = demo_bundle
    balanced = InferenceConfig.from_dict(_config("decode", 8, 4096, 0.3, True))
    skewed = InferenceConfig.from_dict(_config("decode", 8, 4096, 1.2, True))
    left = model.predict(balanced)
    right = model.predict(skewed)
    assert left.attention_us == pytest.approx(right.attention_us)
    assert left.dense_us == pytest.approx(right.dense_us)
    assert left.expert_us != pytest.approx(right.expert_us)


def test_out_of_range_query_is_auditable(demo_bundle):
    _, _, _, model = demo_bundle
    config = InferenceConfig.from_dict(_config("decode", 32, 16384, 1.0, True))
    profile = model.predict(config)
    assert {"attention", "expert", "dense"} <= set(profile.extrapolated_components)


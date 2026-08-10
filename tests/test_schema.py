from __future__ import annotations

import copy
import json

import pytest

from moe_perf_model.schema import InferenceConfig, SchemaError


def test_valid_config_has_stable_signature(demo_bundle):
    _, paths, _, _ = demo_bundle
    raw = json.loads(open(paths["predict_config"], encoding="utf-8").read())
    left = InferenceConfig.from_dict(raw)
    right = InferenceConfig.from_dict(copy.deepcopy(raw))
    assert left.signature() == right.signature()
    assert len(left.routing.tokens_per_expert) == 128
    assert sum(left.routing.tokens_per_expert) == (
        left.workload.scheduled_tokens * left.model.top_k * left.model.num_moe_layers
    )


def test_routing_conservation_is_required(demo_bundle):
    _, paths, _, _ = demo_bundle
    raw = json.loads(open(paths["predict_config"], encoding="utf-8").read())
    raw["routing"]["tokens_per_expert"][0] += 1
    with pytest.raises(SchemaError, match="sum must equal"):
        InferenceConfig.from_dict(raw)


def test_unsupported_parallelism_fails(demo_bundle):
    _, paths, _, _ = demo_bundle
    raw = json.loads(open(paths["predict_config"], encoding="utf-8").read())
    raw["runtime"]["expert_parallel_size"] = 2
    with pytest.raises(SchemaError, match="communication model"):
        InferenceConfig.from_dict(raw)

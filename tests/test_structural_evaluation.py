from scripts.evaluate_structural_pilot import _key, _quality_filter


def _row(*, scheduled_tokens=1, component_us=10.0):
    return {
        "workload_index": 3,
        "repeat": 0,
        "step_index": 1,
        "phase": "decode",
        "requests": [{
            "phase": "decode",
            "scheduled_tokens": scheduled_tokens,
            "context_length": 128,
        }],
        "rank_records": [{
            "layer_records": [{
                "rank_local_compute_us": {
                    "attention": {"duration_us": component_us},
                    "router": {"duration_us": 2.0},
                    "expert_gemm": {"duration_us": 3.0},
                }
            }]
        }],
    }


def test_realized_step_key_includes_request_shape():
    assert _key(_row(scheduled_tokens=1)) != _key(_row(scheduled_tokens=2))


def test_quality_filter_uses_rank_local_component_threshold():
    accepted, rejected = _quality_filter(
        [_row(component_us=9.0), _row(component_us=11.0)], 10.0
    )
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0]["max_rank_local_component_us"] == 11.0

from types import SimpleNamespace

import pytest

from moe_perf_model.collective_model import SyntheticCollectiveTimingModel
from moe_perf_model.execution_plan import StepExecutionPlan
from moe_perf_model.parallel_forward import ParallelForwardGraphCompiler, ParallelLayout
from moe_perf_model.structural_forward import ForwardStructuralPredictor
from moe_perf_model.rank_local_compute import RankLocalComputeModel


def _plan() -> StepExecutionPlan:
    return StepExecutionPlan.from_dict({
        "phase": "prefill",
        "realized_batch_size": 2,
        "scheduled_tokens": 8,
        "n_q": 4,
        "total_kv_tokens": 16,
        "kv_mean": 8.0,
        "kv_max": 8,
        "dtype": "bf16",
        "attention_backend": "flash",
        "moe_backend": "fused_moe",
        "moe_kernel_config": "fixture",
        "dense_backend": "torch",
        "num_experts": 4,
        "top_k": 2,
        "model_weight_bytes": 1_000,
        "kv_bytes_per_token": 10,
        "activation_bytes_per_scheduled_token": 20,
        "layers": [{
            "layer_id": 0,
            "tokens_per_expert": [8, 4, 2, 2],
            "execution_counts": {
                "attention": 1, "router": 1, "dispatch": 1,
                "expert_gemm": 1, "combine": 1, "dense": 1,
                "transition": 1,
            },
        }],
    })


def _profile():
    times = {
        "attention": 10.0, "router": 2.0, "dispatch": 3.0,
        "expert_gemm": 20.0, "combine": 3.0, "dense": 8.0,
        "transition": 1.0,
    }
    return SimpleNamespace(
        artifact_provenance={"label_source": "cuda_event"},
        layer_profiles=({"layer_id": 0, "components_us": times},),
        components_us={"fixed": 2.0},
    )


def _rank_local_row():
    plan = _plan()
    base_layer = {
        "layer_id": 0,
        "tokens_per_expert": list(plan.layers[0].tokens_per_expert),
        "execution_counts": plan.layers[0].execution_counts,
    }
    identity = {
        "dtype": "bf16", "attention_backend": "flash",
        "moe_backend": "fused_moe", "moe_kernel_config": "fixture",
        "dense_backend": "torch",
    }
    header = {
        "model": {"model_id": "fixture", "num_experts": 4, "top_k": 2},
        "hardware": {
            "hardware_id": "fixture-hardware", "model_weight_bytes": 1000,
            "kv_bytes_per_token": 10, "activation_bytes_per_scheduled_token": 20,
        },
        "runtime": {
            "runtime": "vllm", "vllm_version": "0.18.1", "precision": "bf16",
            "tensor_parallel_size": 2, "expert_parallel_size": 2,
            "all2all_backend": "allgather_reducescatter",
            "expert_placement_strategy": "linear",
        },
    }
    rank_records = []
    for rank, expert_us in ((0, 12.0), (1, 8.0)):
        layer = dict(base_layer)
        layer["rank_local_compute_us"] = {
            "attention": {"duration_us": 7.0, "valid": True},
            "router": {"duration_us": 2.0, "valid": True},
            "expert_gemm": {"duration_us": expert_us, "valid": True},
        }
        rank_records.append({
            "rank": rank, "rank_local_compute_schema": "rank-local-compute/v1",
            "layer_records": [layer],
        })
    return {
        "profile_header": header, "kernel_identity": identity, "rank_count": 2,
        "rank_records": rank_records,
        "phase": "prefill",
        "requests": [
            {"request_id": "a", "phase": "prefill", "input_length": 4,
             "prior_context_length": 4, "scheduled_tokens": 4, "context_length": 8},
            {"request_id": "b", "phase": "prefill", "input_length": 4,
             "prior_context_length": 4, "scheduled_tokens": 4, "context_length": 8},
        ],
        "layer_records": [base_layer],
    }


def _compiler(layout, ep_backend="allgather_reducescatter"):
    return ParallelForwardGraphCompiler(
        layout=layout,
        collective_backend=SyntheticCollectiveTimingModel(startup_us=1, bandwidth_gbps=100),
        hidden_size=16,
        all2all_backend=ep_backend,
    )


def test_vllm_tp2_ep2_aliases_process_groups_and_coordinates():
    layout = ParallelLayout.vllm(2, 2, 2, 8)
    assert layout.tp_groups == ((0, 1),)
    assert layout.ep_groups == ((0, 1),)
    assert layout.expert_coordinate(0) == 0
    assert layout.expert_coordinate(1) == 1


def test_vllm_tp2_ep1_uses_singleton_ep_groups():
    layout = ParallelLayout.vllm(2, 2, 1, 8)
    assert layout.ep_groups == ((0,), (1,))
    assert layout.expert_coordinate(1) == 0


def test_uncaptured_intermediate_ep_layout_is_rejected():
    with pytest.raises(ValueError, match="explicit captured"):
        ParallelLayout.vllm(4, 2, 2, 8)


def test_routing_placement_conserves_tokens_and_exposes_imbalance():
    compiler = _compiler(ParallelLayout.vllm(2, 2, 2, 8), "alltoall")
    placement = compiler.routing_placement(_plan().layers[0].tokens_per_expert)
    assert placement.expert_coordinate_loads == (12, 4)
    assert sum(sum(row) for row in placement.source_destination_tokens) == 16
    assert placement.max_mean_imbalance == pytest.approx(1.5)


def test_ep_backend_changes_collective_type_and_volume_semantics():
    layout = ParallelLayout.vllm(2, 2, 2, 8)
    gathered = _compiler(layout).compile(_profile(), _plan())
    categories = {event.category for event in gathered.events}
    assert {"all_gather", "reduce_scatter"} <= categories
    assert not any(event.category == "all_to_all" for event in gathered.events)
    alltoall = _compiler(layout, "alltoall").compile(_profile(), _plan())
    ep_event = next(event for event in alltoall.events if event.category == "all_to_all")
    assert ep_event.metadata["volume_semantics"] == (
        "max_source_routed_activation_bytes_per_rank"
    )
    assert ep_event.input_bytes == 8 * 16 * 2


def test_structural_prediction_marks_uncalibrated_assumptions():
    plan = _plan()
    profile = _profile()

    class FakeComputeModel:
        def predict(self, requested_plan):
            assert requested_plan is plan
            return SimpleNamespace(
                **profile.__dict__,
                extrapolated_signatures=(),
            )

    predictor = ForwardStructuralPredictor(
        compute_model=FakeComputeModel(),
        graph_compiler=_compiler(ParallelLayout.vllm(2, 2, 2, 8)),
    )
    prediction = predictor.predict(plan)
    assert prediction.schema_version == "structural-forward/v1"
    assert prediction.step_us > 0
    assert not prediction.formal_hardware_prediction
    assert "parallel_compute_scaling_not_cuda_event_calibrated" in prediction.support_reasons
    assert "collective_timing_is_synthetic" in prediction.support_reasons
    assert "serialized_no_overlap_dependency_policy" in prediction.support_reasons
    assert set(prediction.rank_finish_us) == {"0", "1"}


def test_compiler_rejects_layer_mismatch():
    bad_profile = _profile()
    bad_profile.layer_profiles = ({"layer_id": 7, "components_us": bad_profile.layer_profiles[0]["components_us"]},)
    with pytest.raises(ValueError, match="layer ids/order"):
        _compiler(ParallelLayout.vllm(2, 2, 2, 8)).compile(bad_profile, _plan())


def test_rank_local_compute_artifact_replaces_major_scaled_nodes():
    model = RankLocalComputeModel().fit([_rank_local_row()])
    compiler = ParallelForwardGraphCompiler(
        layout=ParallelLayout.vllm(2, 2, 2, 8),
        collective_backend=SyntheticCollectiveTimingModel(startup_us=1, bandwidth_gbps=100),
        hidden_size=16,
        all2all_backend="allgather_reducescatter",
        rank_local_compute_backend=model,
    )
    graph = compiler.compile(_profile(), _plan())
    attention = next(event for event in graph.events if event.event_id == "l0.attention.r0")
    expert0 = next(event for event in graph.events if event.event_id == "l0.expert.r0")
    expert1 = next(event for event in graph.events if event.event_id == "l0.expert.r1")
    assert attention.duration_us == 7.0
    assert expert0.duration_us == 12.0
    assert expert1.duration_us == 8.0
    assert graph.provenance["rank_local_compute_source"]["label_source"] == (
        "cuda_event_rank_local_compute_exclusive"
    )

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_perf_model.collective_model import (
    SyntheticCollectiveTimingModel,
    TopologyAwareCollectiveTimingModel,
)
from moe_perf_model.event_engine import DiscreteEventEngine
from moe_perf_model.execution_plan import StepExecutionPlan
from moe_perf_model.parallel_forward import ParallelForwardGraphCompiler, ParallelLayout
from moe_perf_model.vllm_forward_hijack import (
    ProfileReplayRoutingProvider,
    SchedulerStepAdapter,
)
from moe_perf_model.whitebox_model import load_whitebox_artifact


ROOT = Path(__file__).parents[1]
RAW = ROOT / "results/h20_sparse_anchor_v3_diverse/raw_steps.jsonl"
ARTIFACT = ROOT / "results/h20_sparse_anchor_v3_diverse/whitebox_event_v3.joblib"
requires_raw_profile = pytest.mark.skipif(
    not RAW.exists(), reason="requires external H20 raw profiling artifact"
)
requires_whitebox_artifact = pytest.mark.skipif(
    not (RAW.exists() and ARTIFACT.exists()),
    reason="requires external H20 raw profile and whitebox artifact",
)


def _backend():
    return TopologyAwareCollectiveTimingModel(
        SyntheticCollectiveTimingModel(startup_us=3, bandwidth_gbps=300),
        SyntheticCollectiveTimingModel(startup_us=15, bandwidth_gbps=25),
    )


def test_scheduler_adapter_tracks_prefill_then_decode():
    adapter = SchedulerStepAdapter()
    new = SimpleNamespace(req_id="r1", prompt_token_ids=list(range(8)), num_computed_tokens=0)
    cached_empty = SimpleNamespace(req_ids=[], num_computed_tokens=[])
    first = SimpleNamespace(
        num_scheduled_tokens={"r1": 4}, scheduled_new_reqs=[new],
        scheduled_cached_reqs=cached_empty, finished_req_ids=set(),
    )
    context = adapter.adapt(first)
    assert context.phase == "prefill"
    assert context.mid_prefill_ids == ("r1",)
    cached = SimpleNamespace(req_ids=["r1"], num_computed_tokens=[8])
    second = SimpleNamespace(
        num_scheduled_tokens={"r1": 1}, scheduled_new_reqs=[],
        scheduled_cached_reqs=cached, finished_req_ids=set(),
    )
    context = adapter.adapt(second)
    assert context.phase == "decode"
    assert context.total_kv_tokens == 9


@requires_raw_profile
def test_routing_replay_preserves_exact_token_count():
    provider = ProfileReplayRoutingProvider(RAW)
    context = SimpleNamespace(
        phase="prefill", realized_batch_size=2, scheduled_tokens=17, n_q=9,
        total_kv_tokens=25, kv_mean=12.5, kv_max=16,
    )
    plan, provenance = provider.make_plan(context)
    assert provenance["routing_source"] == "profile_replay"
    assert not provenance["formal_online_routing"]
    assert all(sum(layer.tokens_per_expert) == 17 * plan.top_k for layer in plan.layers)


@requires_whitebox_artifact
def test_tp1_ep1_parallel_graph_degenerates_to_whitebox_step():
    model = load_whitebox_artifact(ARTIFACT)
    raw = json.loads(RAW.read_text().splitlines()[0])
    plan = StepExecutionPlan.from_raw_step(raw)
    profile = model.predict(plan)
    compiler = ParallelForwardGraphCompiler(
        layout=ParallelLayout.rectangular(1, 1, 8), collective_backend=_backend()
    )
    report = DiscreteEventEngine().simulate(compiler.compile(profile, plan))
    assert report.step_us == pytest.approx(profile.step_us)


@requires_whitebox_artifact
def test_multinode_graph_contains_ep_and_network_resources():
    model = load_whitebox_artifact(ARTIFACT)
    raw = json.loads(RAW.read_text().splitlines()[0])
    plan = StepExecutionPlan.from_raw_step(raw)
    profile = model.predict(plan)
    compiler = ParallelForwardGraphCompiler(
        layout=ParallelLayout.rectangular(2, 4, 4), collective_backend=_backend()
    )
    graph = compiler.compile(profile, plan)
    report = DiscreteEventEngine().simulate(graph)
    assert "network:fabric" in graph.resources
    assert any(event.category == "all_to_all" for event in graph.events)
    assert report.communication_work_us > 0
    assert report.rank_synchronization_wait_us
    assert not graph.provenance["formal_hardware_prediction"]

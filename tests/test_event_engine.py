from __future__ import annotations

import pytest

from moe_perf_model.collective_model import CollectiveQuery, SyntheticCollectiveTimingModel
from moe_perf_model.event_engine import (
    DiscreteEventEngine,
    ExecutionEvent,
    ExecutionGraph,
    ResourceSpec,
    build_single_rank_graph,
    build_synthetic_collective_graph,
)
from moe_perf_model.runtime_frontend import RuntimeEventRecorder


def test_single_rank_dag_degenerates_to_serial_sum():
    layers = [
        {
            "layer_id": 0,
            "components_us": {
                "attention": 10.0,
                "router": 2.0,
                "dispatch": 3.0,
                "expert_gemm": 20.0,
                "combine": 4.0,
                "dense": 5.0,
                "transition": 6.0,
            },
        },
        {
            "layer_id": 1,
            "components_us": {
                "attention": 11.0,
                "router": 2.0,
                "dispatch": 3.0,
                "expert_gemm": 21.0,
                "combine": 4.0,
                "dense": 5.0,
                "transition": 6.0,
            },
        },
    ]
    report = DiscreteEventEngine().simulate(build_single_rank_graph(layers, fixed_us=7.0))
    assert report.step_us == pytest.approx(109.0)
    assert report.total_work_us == pytest.approx(report.step_us)
    assert sum(report.critical_path_shares.values()) == pytest.approx(1.0)
    assert len(report.critical_path) == 15


def test_collective_can_be_fully_hidden_by_compute():
    report = DiscreteEventEngine().simulate(build_synthetic_collective_graph("overlap"))
    assert report.step_us == pytest.approx(60.0)
    assert report.communication_work_us == 30.0
    assert report.exposed_communication_us == pytest.approx(0.0)
    assert report.overlapped_communication_us == pytest.approx(30.0)


def test_collective_waits_for_slowest_rank():
    report = DiscreteEventEngine().simulate(build_synthetic_collective_graph("slow_rank"))
    collective = next(event for event in report.events if event.event_id == "collective/all_to_all/0")
    assert collective.start_us == pytest.approx(20.0)
    assert report.rank_synchronization_wait_us["collective/all_to_all/0:rank0"] == 10.0
    assert report.rank_synchronization_wait_us["collective/all_to_all/0:rank1"] == 0.0
    assert report.exposed_communication_us == pytest.approx(30.0)


def test_link_contention_serializes_collectives():
    report = DiscreteEventEngine().simulate(build_synthetic_collective_graph("contention"))
    collectives = [event for event in report.events if event.category in {"all_to_all", "all_reduce"}]
    assert collectives[0].end_us <= collectives[1].start_us
    assert report.resource_utilization["link:nvlink:0-1"] > 0.0


def test_graph_rejects_cycles_and_missing_resources():
    graph = ExecutionGraph()
    graph.add_event(ExecutionEvent("a", 0, "s", "attention", 1.0, ("b",)))
    graph.add_event(ExecutionEvent("b", 0, "s", "dense", 1.0, ("a",)))
    with pytest.raises(ValueError, match="cycle"):
        graph.validate()
    bad = ExecutionGraph()
    bad.add_event(
        ExecutionEvent("a", 0, "s", "attention", 1.0, resource_demands={"gpu": 1.0})
    )
    with pytest.raises(ValueError, match="missing resources"):
        bad.validate()


def test_synthetic_collective_model_cannot_claim_formal_timing():
    estimate = SyntheticCollectiveTimingModel().predict(
        CollectiveQuery("all_reduce", 1 << 20, 8, "nvlink")
    )
    assert estimate.duration_us > 0
    assert estimate.provenance["label_source"] == "synthetic_fixture"
    assert estimate.provenance["formal_hardware_prediction"] is False


def test_runtime_recorder_preserves_stream_order_and_collective_rendezvous():
    recorder = RuntimeEventRecorder((0, 1), provenance={"frontend": "fixture"})
    recorder.add_resource(ResourceSpec("link", kind="nvlink"))
    recorder.record_operator(
        event_id="r0/attention", rank=0, stream="compute", category="attention", duration_us=10
    )
    recorder.record_operator(
        event_id="r1/attention", rank=1, stream="compute", category="attention", duration_us=20
    )
    query = CollectiveQuery("all_reduce", 1024, 2, "nvlink")
    estimate = SyntheticCollectiveTimingModel(startup_us=5, bandwidth_gbps=100).predict(query)
    recorder.record_collective(
        event_id="tp/all_reduce",
        participants={0: "compute", 1: "compute"},
        query=query,
        estimate=estimate,
        resource_demands={"link": 1.0},
    )
    recorder.record_operator(
        event_id="r0/dense", rank=0, stream="compute", category="dense", duration_us=3
    )
    graph = recorder.finalize()
    dense = next(event for event in graph.events if event.event_id == "r0/dense")
    assert dense.dependencies == ("tp/all_reduce",)
    assert graph.provenance["formal_hardware_prediction"] is False
    report = DiscreteEventEngine().simulate(graph)
    assert report.rank_synchronization_wait_us["tp/all_reduce:rank0"] == 10.0

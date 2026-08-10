from __future__ import annotations

from typing import Any, Iterable

from .collective_model import CollectiveEstimate, CollectiveQuery
from .event_engine import ExecutionEvent, ExecutionGraph, ResourceSpec


class RuntimeEventRecorder:
    """Target interface for future vLLM operator/collective interception.

    It records control-flow order and metadata only. It neither launches kernels nor
    creates placeholder tensors, so using it does not claim that vLLM interception is complete.
    """

    def __init__(self, ranks: Iterable[int], *, provenance: dict[str, Any] | None = None):
        self.graph = ExecutionGraph(provenance=provenance)
        self.stream_tails: dict[tuple[int, str], str] = {}
        for rank in sorted(set(ranks)):
            self.graph.add_resource(
                ResourceSpec(f"gpu:{rank}:compute", kind="gpu_compute")
            )

    def add_resource(self, resource: ResourceSpec) -> None:
        self.graph.add_resource(resource)

    def record_operator(
        self,
        *,
        event_id: str,
        rank: int,
        stream: str,
        category: str,
        duration_us: float,
        resource_demands: dict[str, float] | None = None,
        data_dependencies: Iterable[str] = (),
        input_bytes: int = 0,
        output_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        dependencies = list(data_dependencies)
        tail = self.stream_tails.get((rank, stream))
        if tail is not None and tail not in dependencies:
            dependencies.append(tail)
        demands = (
            dict(resource_demands)
            if resource_demands is not None
            else {f"gpu:{rank}:compute": 1.0}
        )
        self.graph.add_event(
            ExecutionEvent(
                event_id=event_id,
                rank=rank,
                stream=stream,
                category=category,
                duration_us=duration_us,
                dependencies=tuple(dependencies),
                resource_demands=demands,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                metadata=dict(metadata or {}),
            )
        )
        self.stream_tails[(rank, stream)] = event_id
        return event_id

    def record_collective(
        self,
        *,
        event_id: str,
        participants: dict[int, str],
        query: CollectiveQuery,
        estimate: CollectiveEstimate,
        resource_demands: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        label_source = estimate.provenance.get("label_source")
        if label_source not in {"synthetic_fixture", "cuda_event", "nccl_benchmark"}:
            raise ValueError("collective timing estimate requires explicit provenance")
        participant_dependencies = {}
        for rank, stream in participants.items():
            tail = self.stream_tails.get((rank, stream))
            if tail is None:
                raise ValueError(f"rank {rank} stream {stream} has no collective predecessor")
            participant_dependencies[str(rank)] = tail
        self.graph.add_event(
            ExecutionEvent(
                event_id=event_id,
                rank=None,
                stream="collective",
                category=query.collective_type,
                duration_us=estimate.duration_us,
                dependencies=tuple(participant_dependencies.values()),
                resource_demands=dict(resource_demands),
                input_bytes=query.message_bytes,
                output_bytes=query.message_bytes,
                metadata={
                    "collective_type": query.collective_type,
                    "rank_count": query.rank_count,
                    "topology": query.topology,
                    "participants": sorted(participants),
                    "participant_dependencies": participant_dependencies,
                    "timing_provenance": estimate.provenance,
                    **dict(metadata or {}),
                },
            )
        )
        for rank, stream in participants.items():
            self.stream_tails[(rank, stream)] = event_id
        if label_source == "synthetic_fixture":
            self.graph.provenance["formal_hardware_prediction"] = False
        return event_id

    def record_stream_wait(
        self,
        *,
        event_id: str,
        rank: int,
        destination_stream: str,
        source_event: str,
    ) -> str:
        dependencies = [source_event]
        tail = self.stream_tails.get((rank, destination_stream))
        if tail is not None and tail not in dependencies:
            dependencies.append(tail)
        self.graph.add_event(
            ExecutionEvent(
                event_id=event_id,
                rank=rank,
                stream=destination_stream,
                category="synchronization",
                duration_us=0.0,
                dependencies=tuple(dependencies),
                metadata={"waits_for": source_event},
            )
        )
        self.stream_tails[(rank, destination_stream)] = event_id
        return event_id

    def finalize(self) -> ExecutionGraph:
        self.graph.validate()
        return self.graph

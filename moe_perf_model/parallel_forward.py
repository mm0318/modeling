from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .collective_model import CollectiveQuery, CollectiveTimingBackend
from .event_engine import ExecutionGraph, ResourceSpec
from .runtime_frontend import RuntimeEventRecorder


@dataclass(frozen=True)
class ParallelLayout:
    """Explicit vLLM process groups; TP and EP groups may alias the same ranks."""

    tp_groups: tuple[tuple[int, ...], ...]
    ep_groups: tuple[tuple[int, ...], ...]
    rank_to_node: dict[int, int]
    gpus_per_node: int

    def __post_init__(self) -> None:
        if self.gpus_per_node < 1 or not self.tp_groups or not self.ep_groups:
            raise ValueError("parallel layout requires non-empty groups and positive gpus_per_node")
        ranks = set(self.rank_to_node)
        if not ranks or any(rank < 0 or node < 0 for rank, node in self.rank_to_node.items()):
            raise ValueError("rank_to_node requires non-negative ranks and nodes")
        for name, groups in (("tp", self.tp_groups), ("ep", self.ep_groups)):
            if any(not group or len(set(group)) != len(group) for group in groups):
                raise ValueError(f"{name} groups must be non-empty and contain unique ranks")
            flattened = [rank for group in groups for rank in group]
            if set(flattened) != ranks or len(flattened) != len(ranks):
                raise ValueError(f"each rank must occur exactly once across {name} groups")
            if len({len(group) for group in groups}) != 1:
                raise ValueError(f"{name} groups must have uniform size")

    @classmethod
    def rectangular(cls, tp_size: int, ep_size: int, gpus_per_node: int) -> "ParallelLayout":
        """Build the legacy orthogonal TP-by-EP fixture used by synthetic tests."""
        if min(tp_size, ep_size, gpus_per_node) < 1:
            raise ValueError("parallel sizes and gpus_per_node must be positive")
        world = tp_size * ep_size
        tp_groups = tuple(
            tuple(ep * tp_size + tp for tp in range(tp_size)) for ep in range(ep_size)
        )
        ep_groups = tuple(
            tuple(ep * tp_size + tp for ep in range(ep_size)) for tp in range(tp_size)
        )
        return cls(
            tp_groups=tp_groups,
            ep_groups=ep_groups,
            rank_to_node={rank: rank // gpus_per_node for rank in range(world)},
            gpus_per_node=gpus_per_node,
        )

    @classmethod
    def vllm(
        cls, world_size: int, tp_size: int, ep_size: int, gpus_per_node: int
    ) -> "ParallelLayout":
        """Build the layouts currently validated by the vLLM forward collector.

        EP=1 produces singleton groups. EP=world_size produces one global EP
        group, which intentionally aliases a global TP group for TP=world_size.
        Orthogonal TP/EP layouts remain available through ``rectangular`` and are
        rejected here until their concrete vLLM rank mapping is captured.
        """
        if min(world_size, tp_size, ep_size, gpus_per_node) < 1:
            raise ValueError("parallel sizes and gpus_per_node must be positive")
        if world_size % tp_size or world_size % ep_size:
            raise ValueError("world_size must be divisible by TP and EP sizes")
        ranks = tuple(range(world_size))
        tp_groups = tuple(
            tuple(range(begin, begin + tp_size)) for begin in range(0, world_size, tp_size)
        )
        if ep_size == 1:
            ep_groups = tuple((rank,) for rank in ranks)
        elif ep_size == world_size:
            ep_groups = (ranks,)
        else:
            raise ValueError(
                "intermediate EP layouts require an explicit captured process-group mapping"
            )
        return cls(
            tp_groups=tp_groups,
            ep_groups=ep_groups,
            rank_to_node={rank: rank // gpus_per_node for rank in ranks},
            gpus_per_node=gpus_per_node,
        )

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(sorted(self.rank_to_node))

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def tp_size(self) -> int:
        return len(self.tp_groups[0])

    @property
    def ep_size(self) -> int:
        return len(self.ep_groups[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def expert_coordinate(self, rank: int) -> int:
        for group in self.ep_groups:
            if rank in group:
                return group.index(rank)
        raise ValueError(f"rank {rank} does not belong to an EP group")


@dataclass(frozen=True)
class ParallelScalingPolicy:
    tp_compute_efficiency: float = 0.90
    ep_compute_efficiency: float = 0.95
    label_source: str = "synthetic_fixture"

    def __post_init__(self) -> None:
        if not 0 < self.tp_compute_efficiency <= 1 or not 0 < self.ep_compute_efficiency <= 1:
            raise ValueError("parallel efficiency must be in (0, 1]")
        if self.label_source not in {"synthetic_fixture", "cuda_event"}:
            raise ValueError("parallel scaling needs explicit provenance")


@dataclass(frozen=True)
class RoutingPlacement:
    expert_coordinate_loads: tuple[int, ...]
    source_destination_tokens: tuple[tuple[int, ...], ...]
    max_mean_imbalance: float
    semantics: str


class ParallelForwardGraphCompiler:
    """Compile a realized MoE shape into a serialized multi-rank forward DAG."""

    VALID_EP_BACKENDS = {"allgather_reducescatter", "alltoall"}

    def __init__(
        self,
        *,
        layout: ParallelLayout,
        collective_backend: CollectiveTimingBackend,
        scaling: ParallelScalingPolicy | None = None,
        hidden_size: int = 2048,
        dtype_bytes: int = 2,
        all2all_backend: str = "alltoall",
        collective_backend_name: str = "synthetic",
        routing_semantics: str = "global_histogram_deterministic_source_split",
        rank_local_compute_backend: Any | None = None,
    ):
        if hidden_size < 1 or dtype_bytes < 1:
            raise ValueError("hidden_size and dtype_bytes must be positive")
        if all2all_backend not in self.VALID_EP_BACKENDS:
            raise ValueError(f"unsupported EP communication backend: {all2all_backend}")
        if collective_backend_name not in {"synthetic", "nccl"}:
            raise ValueError("collective backend name must be synthetic or nccl")
        self.layout = layout
        self.collective_backend = collective_backend
        self.scaling = scaling or ParallelScalingPolicy()
        self.hidden_size = hidden_size
        self.dtype_bytes = dtype_bytes
        self.all2all_backend = all2all_backend
        self.collective_backend_name = collective_backend_name
        self.routing_semantics = routing_semantics
        self.rank_local_compute_backend = rank_local_compute_backend

    def routing_placement(self, tokens_per_expert: tuple[int, ...]) -> RoutingPlacement:
        loads = [0] * self.layout.ep_size
        destinations: list[int] = []
        for expert, tokens in enumerate(tokens_per_expert):
            destination = min(
                self.layout.ep_size - 1,
                expert * self.layout.ep_size // len(tokens_per_expert),
            )
            loads[destination] += tokens
            destinations.append(destination)

        # The current plan contains a global histogram, not source-rank routing.
        # Split each expert count deterministically and preserve exact totals.
        matrix = [[0] * self.layout.ep_size for _ in range(self.layout.ep_size)]
        for expert, tokens in enumerate(tokens_per_expert):
            quotient, remainder = divmod(tokens, self.layout.ep_size)
            destination = destinations[expert]
            for source in range(self.layout.ep_size):
                matrix[source][destination] += quotient + int(source < remainder)
        mean = sum(loads) / len(loads) if loads else 0.0
        return RoutingPlacement(
            expert_coordinate_loads=tuple(loads),
            source_destination_tokens=tuple(tuple(row) for row in matrix),
            max_mean_imbalance=max(loads) / mean if mean else 0.0,
            semantics=self.routing_semantics,
        )

    def _collective_resources(self, ranks: tuple[int, ...]) -> tuple[str, dict[str, float]]:
        nodes = sorted({self.layout.rank_to_node[rank] for rank in ranks})
        if len(nodes) == 1:
            return "intra_node", {f"node:{nodes[0]}:local_fabric": 1.0}
        demands = {"network:fabric": 1.0}
        demands.update({f"node:{node}:nic": 1.0 for node in nodes})
        return "inter_node", demands

    def _collective(
        self,
        recorder: RuntimeEventRecorder,
        *,
        event_id: str,
        ranks: tuple[int, ...],
        collective_type: str,
        message_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(ranks) < 2:
            return
        for rank in ranks:
            recorder.record_stream_wait(
                event_id=f"{event_id}.ready.r{rank}", rank=rank,
                destination_stream="comm", source_event=recorder.stream_tails[(rank, "compute")],
            )
        topology, demands = self._collective_resources(ranks)
        query = CollectiveQuery(
            collective_type, message_bytes, len(ranks), topology,
            backend=self.collective_backend_name,
        )
        estimate = self.collective_backend.predict(query)
        recorder.record_collective(
            event_id=event_id,
            participants={rank: "comm" for rank in ranks},
            query=query,
            estimate=estimate,
            resource_demands=demands,
            metadata=metadata,
        )
        for rank in ranks:
            recorder.record_stream_wait(
                event_id=f"{event_id}.done.r{rank}", rank=rank,
                destination_stream="compute", source_event=event_id,
            )

    def _ep_collective(
        self,
        recorder: RuntimeEventRecorder,
        *,
        layer_id: int,
        group_index: int,
        group: tuple[int, ...],
        direction: str,
        placement: RoutingPlacement,
        scheduled_tokens: int,
    ) -> None:
        if len(group) < 2:
            return
        if self.all2all_backend == "allgather_reducescatter":
            collective_type = "all_gather" if direction == "dispatch" else "reduce_scatter"
            message_bytes = (
                scheduled_tokens * self.hidden_size * self.dtype_bytes // self.layout.tp_size
            )
            volume_semantics = "local_hidden_shard_bytes_per_rank"
        else:
            collective_type = "all_to_all"
            max_source_tokens = max(sum(row) for row in placement.source_destination_tokens)
            message_bytes = max_source_tokens * self.hidden_size * self.dtype_bytes
            volume_semantics = "max_source_routed_activation_bytes_per_rank"
        self._collective(
            recorder,
            event_id=f"l{layer_id}.ep_{direction}.g{group_index}",
            ranks=group,
            collective_type=collective_type,
            message_bytes=message_bytes,
            metadata={
                "ep_backend": self.all2all_backend,
                "volume_semantics": volume_semantics,
                "routing_semantics": placement.semantics,
                "source_destination_tokens": placement.source_destination_tokens,
                "expert_coordinate_loads": placement.expert_coordinate_loads,
                "max_mean_imbalance": placement.max_mean_imbalance,
            },
        )

    def compile(self, profile: Any, plan: Any) -> ExecutionGraph:
        if len(profile.layer_profiles) != len(plan.layers):
            raise ValueError("profile and execution plan must contain the same layers")
        profile_ids = tuple(int(layer["layer_id"]) for layer in profile.layer_profiles)
        plan_ids = tuple(layer.layer_id for layer in plan.layers)
        if profile_ids != plan_ids:
            raise ValueError("profile and execution plan layer ids/order must match")
        formal = (
            profile.artifact_provenance.get("label_source") == "cuda_event"
            and self.scaling.label_source == "cuda_event"
        )
        recorder = RuntimeEventRecorder(
            self.layout.ranks,
            provenance={
                "schema_version": "parallel-forward-dag/v2",
                "graph_type": "parallel_forward",
                "compute_timing_source": profile.artifact_provenance.get("label_source"),
                "parallel_scaling_source": self.scaling.label_source,
                "formal_hardware_prediction": formal,
                "layout_assumption": "explicit_process_groups",
                "dependency_policy": "serialized_collectives_no_overlap",
                "ep_communication_backend": self.all2all_backend,
                "routing_semantics": self.routing_semantics,
                "rank_local_compute_source": (
                    self.rank_local_compute_backend.provenance
                    if self.rank_local_compute_backend is not None else None
                ),
            },
        )
        nodes = sorted(set(self.layout.rank_to_node.values()))
        for node in nodes:
            recorder.add_resource(ResourceSpec(f"node:{node}:local_fabric", kind="local_interconnect"))
            recorder.add_resource(ResourceSpec(f"node:{node}:nic", kind="network_interface"))
        if len(nodes) > 1:
            recorder.add_resource(ResourceSpec("network:fabric", kind="inter_node_network"))

        tp_factor = 1.0 if self.layout.tp_size == 1 else (
            self.layout.tp_size * self.scaling.tp_compute_efficiency
        )
        activation_bytes = plan.scheduled_tokens * self.hidden_size * self.dtype_bytes
        ranks = self.layout.ranks
        layer_placements: dict[str, Any] = {}
        for layer, layer_plan in zip(profile.layer_profiles, plan.layers):
            layer_id = int(layer["layer_id"])
            times = layer["components_us"]
            placement = self.routing_placement(layer_plan.tokens_per_expert)
            layer_placements[str(layer_id)] = asdict(placement)
            rank_local = {
                rank: self.rank_local_compute_backend.predict_layer(
                    plan, layer_plan, rank, self.layout
                )
                for rank in ranks
            } if self.rank_local_compute_backend is not None else {}
            for rank in ranks:
                local_prediction = rank_local.get(rank)
                recorder.record_operator(
                    event_id=f"l{layer_id}.attention.r{rank}", rank=rank, stream="compute",
                    category="attention", duration_us=(
                        local_prediction.attention_us
                        if local_prediction is not None else times["attention"] / tp_factor
                    ),
                    metadata={
                        "scaling_source": (
                            "rank_local_compute_response"
                            if local_prediction is not None else self.scaling.label_source
                        ),
                        "rank_local_estimate": (
                            local_prediction.estimates["attention"]
                            if local_prediction is not None else None
                        ),
                    },
                )
            for group_index, group in enumerate(self.layout.tp_groups):
                self._collective(
                    recorder, event_id=f"l{layer_id}.tp_attention.g{group_index}",
                    ranks=group, collective_type="all_reduce", message_bytes=activation_bytes,
                    metadata={"volume_semantics": "full_activation_bytes_per_rank"},
                )
            for rank in ranks:
                local_prediction = rank_local.get(rank)
                recorder.record_operator(
                    event_id=f"l{layer_id}.router.r{rank}", rank=rank, stream="compute",
                    category="router", duration_us=(
                        local_prediction.router_us
                        if local_prediction is not None else times["router"] / tp_factor
                    ),
                    metadata={
                        "scaling_source": (
                            "rank_local_compute_response"
                            if local_prediction is not None else self.scaling.label_source
                        ),
                        "rank_local_estimate": (
                            local_prediction.estimates["router"]
                            if local_prediction is not None else None
                        ),
                    },
                )
                recorder.record_operator(
                    event_id=f"l{layer_id}.dispatch.r{rank}", rank=rank, stream="compute",
                    category="dispatch", duration_us=times["dispatch"],
                )
            for group_index, group in enumerate(self.layout.ep_groups):
                self._ep_collective(
                    recorder, layer_id=layer_id, group_index=group_index, group=group,
                    direction="dispatch", placement=placement,
                    scheduled_tokens=plan.scheduled_tokens,
                )

            total_load = max(1, sum(placement.expert_coordinate_loads))
            for rank in ranks:
                coordinate = self.layout.expert_coordinate(rank)
                load = placement.expert_coordinate_loads[coordinate]
                local_prediction = rank_local.get(rank)
                if local_prediction is not None:
                    expert_us = local_prediction.expert_gemm_us
                elif self.layout.ep_size == 1:
                    expert_us = times["expert_gemm"] / tp_factor
                else:
                    expert_us = (
                        times["expert_gemm"] * load / total_load
                        / self.scaling.ep_compute_efficiency
                    )
                recorder.record_operator(
                    event_id=f"l{layer_id}.expert.r{rank}", rank=rank, stream="compute",
                    category="expert_gemm", duration_us=expert_us,
                    metadata={
                        "assigned_routed_tokens": load,
                        "max_mean_imbalance": placement.max_mean_imbalance,
                        "scaling_source": (
                            "rank_local_compute_response"
                            if local_prediction is not None else self.scaling.label_source
                        ),
                        "rank_local_estimate": (
                            local_prediction.estimates["expert_gemm"]
                            if local_prediction is not None else None
                        ),
                    },
                )
            for group_index, group in enumerate(self.layout.ep_groups):
                self._ep_collective(
                    recorder, layer_id=layer_id, group_index=group_index, group=group,
                    direction="combine", placement=placement,
                    scheduled_tokens=plan.scheduled_tokens,
                )
            for rank in ranks:
                for stage in ("combine", "dense", "transition"):
                    divisor = tp_factor if stage in {"dense", "transition"} else 1.0
                    recorder.record_operator(
                        event_id=f"l{layer_id}.{stage}.r{rank}", rank=rank, stream="compute",
                        category=stage, duration_us=times[stage] / divisor,
                    )
            for group_index, group in enumerate(self.layout.tp_groups):
                self._collective(
                    recorder, event_id=f"l{layer_id}.tp_output.g{group_index}",
                    ranks=group, collective_type="all_reduce", message_bytes=activation_bytes,
                    metadata={"volume_semantics": "full_activation_bytes_per_rank"},
                )
        for rank in ranks:
            recorder.record_operator(
                event_id=f"fixed.r{rank}", rank=rank, stream="compute", category="fixed",
                duration_us=profile.components_us["fixed"] / tp_factor,
            )
        graph = recorder.finalize()
        graph.provenance["parallel_layout"] = self.layout.to_dict()
        graph.provenance["layer_routing_placements"] = layer_placements
        graph.provenance["rank_local_compute_extrapolated"] = any(
            (event.metadata.get("rank_local_estimate") or {}).get("extrapolated", False)
            for event in graph.events
        )
        return graph

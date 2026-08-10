from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable


COMMUNICATION_CATEGORIES = {
    "all_to_all",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "communication",
}


@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    capacity: float = 1.0
    kind: str = "generic"

    def __post_init__(self) -> None:
        if not self.resource_id or self.capacity <= 0:
            raise ValueError("resource requires a non-empty id and positive capacity")


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    rank: int | None
    stream: str
    category: str
    duration_us: float
    dependencies: tuple[str, ...] = ()
    resource_demands: dict[str, float] = field(default_factory=dict)
    input_bytes: int = 0
    output_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.stream or not self.category:
            raise ValueError("event id, stream and category must be non-empty")
        if self.duration_us < 0:
            raise ValueError("event duration must be non-negative")
        if self.input_bytes < 0 or self.output_bytes < 0:
            raise ValueError("event byte counts must be non-negative")
        if any(value <= 0 for value in self.resource_demands.values()):
            raise ValueError("event resource demands must be positive")


class ExecutionGraph:
    def __init__(self, *, provenance: dict[str, Any] | None = None):
        self.resources: dict[str, ResourceSpec] = {}
        self.events: list[ExecutionEvent] = []
        self._event_ids: set[str] = set()
        self.provenance = dict(provenance or {})

    def add_resource(self, resource: ResourceSpec) -> None:
        if resource.resource_id in self.resources:
            raise ValueError(f"duplicate resource: {resource.resource_id}")
        self.resources[resource.resource_id] = resource

    def add_event(self, event: ExecutionEvent) -> None:
        if event.event_id in self._event_ids:
            raise ValueError(f"duplicate event: {event.event_id}")
        self.events.append(event)
        self._event_ids.add(event.event_id)

    def validate(self) -> None:
        ids = {event.event_id for event in self.events}
        if not self.events:
            raise ValueError("execution graph requires at least one event")
        for event in self.events:
            missing = set(event.dependencies) - ids
            if missing:
                raise ValueError(f"event {event.event_id} has missing dependencies: {sorted(missing)}")
            missing_resources = set(event.resource_demands) - set(self.resources)
            if missing_resources:
                raise ValueError(
                    f"event {event.event_id} uses missing resources: {sorted(missing_resources)}"
                )
            for resource_id, demand in event.resource_demands.items():
                if demand > self.resources[resource_id].capacity:
                    raise ValueError(f"event {event.event_id} exceeds {resource_id} capacity")
        indegree = {event.event_id: len(event.dependencies) for event in self.events}
        successors: dict[str, list[str]] = {event_id: [] for event_id in ids}
        for event in self.events:
            for dependency in event.dependencies:
                successors[dependency].append(event.event_id)
        queue = [event.event_id for event in self.events if indegree[event.event_id] == 0]
        visited = 0
        while queue:
            current = queue.pop(0)
            visited += 1
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if visited != len(self.events):
            raise ValueError("execution graph contains a dependency cycle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "resources": [asdict(value) for value in self.resources.values()],
            "events": [asdict(value) for value in self.events],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutionGraph":
        graph = cls(provenance=raw.get("provenance", {}))
        for resource in raw.get("resources", []):
            graph.add_resource(ResourceSpec(**resource))
        for event in raw.get("events", []):
            value = dict(event)
            value["dependencies"] = tuple(value.get("dependencies", ()))
            graph.add_event(ExecutionEvent(**value))
        graph.validate()
        return graph


@dataclass(frozen=True)
class ScheduledEvent:
    event_id: str
    rank: int | None
    stream: str
    category: str
    start_us: float
    end_us: float
    duration_us: float
    causal_predecessor: str | None
    resource_demands: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionReport:
    step_us: float
    total_work_us: float
    events: tuple[ScheduledEvent, ...]
    critical_path: tuple[str, ...]
    critical_path_category_us: dict[str, float]
    critical_path_shares: dict[str, float]
    communication_work_us: float
    exposed_communication_us: float
    overlapped_communication_us: float
    resource_utilization: dict[str, float]
    rank_synchronization_wait_us: dict[str, float]
    bottleneck_scores_us: dict[str, float]
    primary_bottleneck: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["events"] = [asdict(event) for event in self.events]
        result["critical_path"] = list(self.critical_path)
        return result


def _interval_fits(
    start: float,
    duration: float,
    demand: float,
    capacity: float,
    reservations: list[tuple[float, float, float, str]],
) -> bool:
    if duration == 0:
        return True
    end = start + duration
    points = {start, end}
    overlapping = []
    for left, right, amount, event_id in reservations:
        if right > start and left < end:
            overlapping.append((left, right, amount, event_id))
            points.add(max(start, left))
            points.add(min(end, right))
    ordered = sorted(points)
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        probe = (left + right) / 2.0
        used = sum(amount for begin, finish, amount, _ in overlapping if begin <= probe < finish)
        if used + demand > capacity + 1e-12:
            return False
    return True


class DiscreteEventEngine:
    """Deterministic non-preemptive earliest-ready multi-resource scheduler."""

    def simulate(self, graph: ExecutionGraph) -> ExecutionReport:
        return self._simulate(graph, communication_counterfactual=True)

    def _simulate(
        self, graph: ExecutionGraph, *, communication_counterfactual: bool
    ) -> ExecutionReport:
        graph.validate()
        events = {event.event_id: event for event in graph.events}
        order = {event.event_id: index for index, event in enumerate(graph.events)}
        reservations: dict[str, list[tuple[float, float, float, str]]] = {
            resource_id: [] for resource_id in graph.resources
        }
        latest_reservation: dict[str, tuple[float, str] | None] = {
            resource_id: None for resource_id in graph.resources
        }
        scheduled: dict[str, ScheduledEvent] = {}
        indegree = {
            event.event_id: len(event.dependencies) for event in graph.events
        }
        successors: dict[str, list[str]] = {
            event.event_id: [] for event in graph.events
        }
        for event in graph.events:
            for dependency in event.dependencies:
                successors[dependency].append(event.event_id)
        ready_ids = {
            event_id for event_id, degree in indegree.items() if degree == 0
        }
        remaining = len(events)

        while remaining:
            if not ready_ids:
                raise ValueError("execution graph cannot make progress")
            ready = [events[event_id] for event_id in ready_ids]
            candidates = []
            for event in ready:
                dependency_ready = max(
                    (scheduled[value].end_us for value in event.dependencies), default=0.0
                )
                boundaries = {dependency_ready}
                for resource_id, demand in event.resource_demands.items():
                    capacity = graph.resources[resource_id].capacity
                    if demand >= capacity - 1e-12:
                        latest = latest_reservation[resource_id]
                        if latest is not None and latest[0] >= dependency_ready:
                            boundaries.add(latest[0])
                    else:
                        boundaries.update(
                            end
                            for _, end, _, _ in reservations[resource_id]
                            if end >= dependency_ready
                        )
                start = None
                for candidate in sorted(boundaries):
                    if all(
                        (
                            candidate >= (latest_reservation[resource_id] or (0.0, ""))[0]
                            - 1e-12
                            if demand >= graph.resources[resource_id].capacity - 1e-12
                            else _interval_fits(
                                candidate,
                                event.duration_us,
                                demand,
                                graph.resources[resource_id].capacity,
                                reservations[resource_id],
                            )
                        )
                        for resource_id, demand in event.resource_demands.items()
                    ):
                        start = candidate
                        break
                if start is None:
                    raise RuntimeError(f"no resource slot found for {event.event_id}")
                candidates.append((start, order[event.event_id], event))
            start, _, event = min(candidates, key=lambda value: (value[0], value[1]))
            predecessor_candidates = [
                (scheduled[value].end_us, value) for value in event.dependencies
            ]
            for resource_id, demand in event.resource_demands.items():
                if demand >= graph.resources[resource_id].capacity - 1e-12:
                    latest = latest_reservation[resource_id]
                    if latest is not None and latest[0] <= start + 1e-12:
                        predecessor_candidates.append(latest)
                else:
                    predecessor_candidates.extend(
                        (end, event_id)
                        for _, end, _, event_id in reservations[resource_id]
                        if end <= start + 1e-12
                    )
            causal = None
            if predecessor_candidates:
                latest_end, latest_id = max(predecessor_candidates, key=lambda value: value[0])
                if abs(latest_end - start) <= 1e-9:
                    causal = latest_id
            result = ScheduledEvent(
                event_id=event.event_id,
                rank=event.rank,
                stream=event.stream,
                category=event.category,
                start_us=start,
                end_us=start + event.duration_us,
                duration_us=event.duration_us,
                causal_predecessor=causal,
                resource_demands=dict(event.resource_demands),
                metadata=dict(event.metadata),
            )
            scheduled[event.event_id] = result
            ready_ids.remove(event.event_id)
            remaining -= 1
            for successor in successors[event.event_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready_ids.add(successor)
            for resource_id, demand in event.resource_demands.items():
                reservations[resource_id].append((result.start_us, result.end_us, demand, event.event_id))
                latest = latest_reservation[resource_id]
                if latest is None or result.end_us >= latest[0]:
                    latest_reservation[resource_id] = (result.end_us, event.event_id)

        makespan = max(event.end_us for event in scheduled.values())
        final = max(scheduled.values(), key=lambda value: (value.end_us, order[value.event_id]))
        path = []
        current: ScheduledEvent | None = final
        while current is not None:
            path.append(current.event_id)
            current = (
                scheduled[current.causal_predecessor]
                if current.causal_predecessor is not None
                else None
            )
        path.reverse()
        category_us: dict[str, float] = {}
        for event_id in path:
            event = scheduled[event_id]
            category_us[event.category] = category_us.get(event.category, 0.0) + event.duration_us
        shares = {
            category: duration / makespan if makespan else 0.0
            for category, duration in category_us.items()
        }
        utilization = {
            resource_id: (
                sum((end - start) * demand for start, end, demand, _ in values)
                / (makespan * graph.resources[resource_id].capacity)
                if makespan
                else 0.0
            )
            for resource_id, values in reservations.items()
        }
        synchronization: dict[str, float] = {}
        for event in graph.events:
            participants = event.metadata.get("participant_dependencies")
            if not isinstance(participants, dict) or event.event_id not in scheduled:
                continue
            start = scheduled[event.event_id].start_us
            for rank, dependency in participants.items():
                ready_time = scheduled[str(dependency)].end_us
                synchronization[f"{event.event_id}:rank{rank}"] = max(0.0, start - ready_time)
        communication_work = sum(
            event.duration_us
            for event in graph.events
            if event.category in COMMUNICATION_CATEGORIES
        )
        exposed = 0.0
        if communication_counterfactual and communication_work:
            counterfactual = ExecutionGraph(provenance=graph.provenance)
            for resource in graph.resources.values():
                counterfactual.add_resource(resource)
            for event in graph.events:
                counterfactual.add_event(
                    replace(event, duration_us=0.0)
                    if event.category in COMMUNICATION_CATEGORIES
                    else event
                )
            no_communication = self._simulate(
                counterfactual, communication_counterfactual=False
            ).step_us
            exposed = max(0.0, makespan - no_communication)
        grouped = self._bottleneck_scores(category_us, synchronization)
        return ExecutionReport(
            step_us=makespan,
            total_work_us=sum(event.duration_us for event in graph.events),
            events=tuple(scheduled[event.event_id] for event in graph.events),
            critical_path=tuple(path),
            critical_path_category_us=category_us,
            critical_path_shares=shares,
            communication_work_us=communication_work,
            exposed_communication_us=exposed,
            overlapped_communication_us=max(0.0, communication_work - exposed),
            resource_utilization=utilization,
            rank_synchronization_wait_us=synchronization,
            bottleneck_scores_us=grouped,
            primary_bottleneck=max(grouped, key=grouped.get),
            provenance=dict(graph.provenance),
        )

    @staticmethod
    def _bottleneck_scores(
        category_us: dict[str, float], synchronization: dict[str, float]
    ) -> dict[str, float]:
        expert_categories = {"router", "dispatch", "expert_gemm", "combine"}
        scores = {
            "attention": category_us.get("attention", 0.0),
            "expert": sum(category_us.get(name, 0.0) for name in expert_categories),
            "communication": sum(
                category_us.get(name, 0.0) for name in COMMUNICATION_CATEGORIES
            ),
            "dense_other": sum(
                value
                for name, value in category_us.items()
                if name not in expert_categories
                and name != "attention"
                and name not in COMMUNICATION_CATEGORIES
            ),
            "synchronization": max(synchronization.values(), default=0.0),
        }
        return scores


def build_single_rank_graph(
    layer_profiles: Iterable[dict[str, Any]],
    *,
    fixed_us: float,
    provenance: dict[str, Any] | None = None,
) -> ExecutionGraph:
    graph = ExecutionGraph(provenance=provenance)
    graph.add_resource(ResourceSpec("gpu:0:compute", kind="gpu_compute"))
    predecessor: str | None = None
    for layer in layer_profiles:
        layer_id = int(layer["layer_id"])
        for category in (
            "attention",
            "router",
            "dispatch",
            "expert_gemm",
            "combine",
            "dense",
            "transition",
        ):
            event_id = f"rank0/layer{layer_id}/{category}"
            graph.add_event(
                ExecutionEvent(
                    event_id=event_id,
                    rank=0,
                    stream="compute",
                    category=category,
                    duration_us=float(layer["components_us"][category]),
                    dependencies=(predecessor,) if predecessor else (),
                    resource_demands=(
                        {"gpu:0:compute": 1.0} if category != "transition" else {}
                    ),
                )
            )
            predecessor = event_id
    graph.add_event(
        ExecutionEvent(
            event_id="rank0/fixed",
            rank=0,
            stream="compute",
            category="fixed",
            duration_us=float(fixed_us),
            dependencies=(predecessor,) if predecessor else (),
            resource_demands={"gpu:0:compute": 1.0},
        )
    )
    graph.validate()
    return graph


def build_synthetic_collective_graph(scenario: str) -> ExecutionGraph:
    """Build small TP/EP fixtures for engine verification, never formal timing."""
    if scenario not in {"serialized", "overlap", "slow_rank", "contention"}:
        raise ValueError("unknown synthetic event scenario")
    graph = ExecutionGraph(
        provenance={
            "label_source": "synthetic_fixture",
            "formal_hardware_prediction": False,
            "scenario": scenario,
        }
    )
    for rank in (0, 1):
        graph.add_resource(ResourceSpec(f"gpu:{rank}:compute", kind="gpu_compute"))
    graph.add_resource(ResourceSpec("link:nvlink:0-1", kind="nvlink"))
    prep_durations = (10.0, 20.0) if scenario == "slow_rank" else (10.0, 10.0)
    for rank, duration in enumerate(prep_durations):
        graph.add_event(
            ExecutionEvent(
                event_id=f"rank{rank}/prepare",
                rank=rank,
                stream="compute",
                category="dispatch",
                duration_us=duration,
                resource_demands={f"gpu:{rank}:compute": 1.0},
            )
        )
    collective_dependencies = ("rank0/prepare", "rank1/prepare")
    graph.add_event(
        ExecutionEvent(
            event_id="collective/all_to_all/0",
            rank=None,
            stream="comm",
            category="all_to_all",
            duration_us=30.0,
            dependencies=collective_dependencies,
            resource_demands={"link:nvlink:0-1": 1.0},
            input_bytes=1 << 20,
            output_bytes=1 << 20,
            metadata={
                "collective_type": "all_to_all",
                "participants": [0, 1],
                "participant_dependencies": {
                    "0": "rank0/prepare",
                    "1": "rank1/prepare",
                },
                "timing_source": "synthetic_fixture",
            },
        )
    )
    if scenario == "contention":
        graph.add_event(
            ExecutionEvent(
                event_id="collective/all_reduce/1",
                rank=None,
                stream="comm-2",
                category="all_reduce",
                duration_us=20.0,
                dependencies=collective_dependencies,
                resource_demands={"link:nvlink:0-1": 1.0},
                metadata={"timing_source": "synthetic_fixture"},
            )
        )
    for rank in (0, 1):
        dependencies = [f"rank{rank}/prepare"]
        if scenario in {"serialized", "slow_rank", "contention"}:
            dependencies.append("collective/all_to_all/0")
            if scenario == "contention":
                dependencies.append("collective/all_reduce/1")
        graph.add_event(
            ExecutionEvent(
                event_id=f"rank{rank}/expert",
                rank=rank,
                stream="compute",
                category="expert_gemm",
                duration_us=50.0 if scenario == "overlap" else 5.0,
                dependencies=tuple(dependencies),
                resource_demands={f"gpu:{rank}:compute": 1.0},
            )
        )
    graph.add_event(
        ExecutionEvent(
            event_id="step/join",
            rank=None,
            stream="control",
            category="fixed",
            duration_us=0.0,
            dependencies=(
                "rank0/expert",
                "rank1/expert",
                "collective/all_to_all/0",
            ),
        )
    )
    graph.validate()
    return graph

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .event_engine import COMMUNICATION_CATEGORIES, DiscreteEventEngine, ExecutionGraph
from .execution_plan import StepExecutionPlan
from .parallel_forward import ParallelForwardGraphCompiler
from .whitebox_model import WhiteboxMoEModel


@dataclass(frozen=True)
class StructuralForwardPrediction:
    schema_version: str
    step_us: float
    memory_bytes_per_rank: int
    critical_path_category_us: dict[str, float]
    bottleneck_shares: dict[str, float]
    primary_bottleneck: str
    exposed_communication_us: float
    synchronization_wait_us: float
    rank_finish_us: dict[str, float]
    supported: bool
    formal_hardware_prediction: bool
    support_reasons: tuple[str, ...]
    provenance: dict[str, Any]
    execution_report: dict[str, Any]
    execution_graph: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["support_reasons"] = list(self.support_reasons)
        return value


class ForwardStructuralPredictor:
    """Forward-only predictor: realized shape -> multi-rank DAG -> critical path."""

    def __init__(
        self,
        *,
        compute_model: WhiteboxMoEModel,
        graph_compiler: ParallelForwardGraphCompiler,
        engine: DiscreteEventEngine | None = None,
    ):
        self.compute_model = compute_model
        self.graph_compiler = graph_compiler
        self.engine = engine or DiscreteEventEngine()

    @staticmethod
    def _rank_finish(graph: ExecutionGraph, report: Any) -> dict[str, float]:
        participants_by_event = {
            event.event_id: event.metadata.get("participants", []) for event in graph.events
        }
        finish = {str(rank): 0.0 for rank in sorted(graph.provenance["parallel_layout"]["rank_to_node"])}
        for event in report.events:
            if event.rank is not None:
                finish[str(event.rank)] = max(finish[str(event.rank)], event.end_us)
            for rank in participants_by_event.get(event.event_id, []):
                finish[str(rank)] = max(finish[str(rank)], event.end_us)
        return finish

    @staticmethod
    def _bottleneck_shares(report: Any) -> dict[str, float]:
        scores = dict(report.bottleneck_scores_us)
        total = sum(scores.values())
        return {name: value / total if total else 0.0 for name, value in scores.items()}

    def predict(
        self, plan: StepExecutionPlan, *, include_trace: bool = True
    ) -> StructuralForwardPrediction:
        profile = self.compute_model.predict(plan)
        graph = self.graph_compiler.compile(profile, plan)
        report = self.engine.simulate(graph)

        reasons: list[str] = []
        if self.graph_compiler.rank_local_compute_backend is None:
            if self.graph_compiler.scaling.label_source != "cuda_event":
                reasons.append("parallel_compute_scaling_not_cuda_event_calibrated")
        else:
            reasons.append("partial_rank_local_compute_coverage")
            if graph.provenance.get("rank_local_compute_extrapolated"):
                reasons.append("rank_local_compute_signature_extrapolation")
        if profile.extrapolated_signatures:
            reasons.append("operator_signature_extrapolation")
        if self.graph_compiler.all2all_backend == "alltoall" and (
            self.graph_compiler.routing_semantics
            == "global_histogram_deterministic_source_split"
        ):
            reasons.append("source_rank_routing_matrix_is_heuristic")
        collective_events = [
            event for event in graph.events if event.category in COMMUNICATION_CATEGORIES
        ]
        if any(
            event.metadata.get("timing_provenance", {}).get("label_source")
            == "synthetic_fixture"
            for event in collective_events
        ):
            reasons.append("collective_timing_is_synthetic")
        if any(
            event.metadata.get("timing_provenance", {}).get("extrapolated", False)
            for event in collective_events
        ):
            reasons.append("collective_message_size_extrapolation")
        if graph.provenance.get("dependency_policy") != "runtime_captured_overlap":
            reasons.append("serialized_no_overlap_dependency_policy")

        formal = not reasons and bool(graph.provenance.get("formal_hardware_prediction"))
        # Weight/KV sharding is an explicit first-order memory model. Activations
        # are retained per rank because their lifetime is not yet DAG-allocated.
        memory = (
            (plan.model_weight_bytes + self.graph_compiler.layout.world_size - 1)
            // self.graph_compiler.layout.world_size
            + (
                plan.total_kv_tokens * plan.kv_bytes_per_token
                + self.graph_compiler.layout.tp_size - 1
            )
            // self.graph_compiler.layout.tp_size
            + plan.scheduled_tokens * plan.activation_bytes_per_scheduled_token
        )
        provenance = {
            **graph.provenance,
            "predictor_type": "forward_only_structural",
            "compute_oracle_artifact": profile.artifact_provenance,
            "memory_model": "weights/world_plus_kv/tp_plus_activation_per_rank",
            "scope": "realized_forward_shape_only_no_scheduler_or_queueing",
        }
        return StructuralForwardPrediction(
            schema_version="structural-forward/v1",
            step_us=report.step_us,
            memory_bytes_per_rank=memory,
            critical_path_category_us=dict(report.critical_path_category_us),
            bottleneck_shares=self._bottleneck_shares(report),
            primary_bottleneck=report.primary_bottleneck,
            exposed_communication_us=report.exposed_communication_us,
            synchronization_wait_us=max(
                report.rank_synchronization_wait_us.values(), default=0.0
            ),
            rank_finish_us=self._rank_finish(graph, report),
            supported=not any(
                reason in {
                    "operator_signature_extrapolation",
                    "collective_message_size_extrapolation",
                }
                for reason in reasons
            ),
            formal_hardware_prediction=formal,
            support_reasons=tuple(reasons),
            provenance=provenance,
            execution_report=report.to_dict() if include_trace else {},
            execution_graph=graph.to_dict() if include_trace else {},
        )

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution_plan import LayerExecution, StepExecutionPlan
from .operator_db import OperatorEstimate, OperatorResponseDatabase, OperatorSample
from .parallel_forward import ParallelLayout


ARTIFACT_TYPE = "rank-local-compute-v1"


def save_rank_local_compute_artifact(model: "RankLocalComputeModel", path: str | Path) -> None:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_rank_local_compute_artifact(path: str | Path) -> "RankLocalComputeModel":
    import joblib

    model = joblib.load(Path(path))
    if not isinstance(model, RankLocalComputeModel):
        raise TypeError("artifact is not a rank-local-compute-v1 model")
    return model


def _local_histogram(
    histogram: tuple[int, ...], ep_size: int, ep_coordinate: int
) -> tuple[int, ...]:
    begin = len(histogram) * ep_coordinate // ep_size
    end = len(histogram) * (ep_coordinate + 1) // ep_size
    return histogram[begin:end]


def _routing_shape(histogram: tuple[int, ...]) -> tuple[float, ...]:
    ordered = sorted(histogram)
    p95_index = min(len(ordered) - 1, max(0, (95 * len(ordered) + 99) // 100 - 1))
    return (
        float(sum(histogram)),
        float(sum(value > 0 for value in histogram)),
        float(max(histogram, default=0)),
        float(ordered[p95_index] if ordered else 0),
    )


@dataclass(frozen=True)
class RankLocalLayerPrediction:
    attention_us: float
    router_us: float
    expert_gemm_us: float
    extrapolated: bool
    nearest_distance: float
    estimates: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankLocalComputeModel:
    artifact_type = ARTIFACT_TYPE

    def __init__(self, *, neighbors: int = 4):
        self.database = OperatorResponseDatabase(neighbors=neighbors)
        self.identity: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {}

    @staticmethod
    def _identity(row: dict[str, Any]) -> dict[str, Any]:
        runtime = row["profile_header"]["runtime"]
        kernel = row["kernel_identity"]
        return {
            "model_id": row["profile_header"]["model"]["model_id"],
            "hardware_id": row["profile_header"]["hardware"]["hardware_id"],
            "vllm_version": runtime["vllm_version"],
            "precision": runtime["precision"],
            "tensor_parallel_size": int(runtime["tensor_parallel_size"]),
            "expert_parallel_size": int(runtime["expert_parallel_size"]),
            "all2all_backend": runtime["all2all_backend"],
            "expert_placement_strategy": runtime["expert_placement_strategy"],
            "attention_backend": kernel["attention_backend"],
            "moe_backend": kernel["moe_backend"],
            "moe_kernel_config": kernel["moe_kernel_config"],
            "num_experts": int(row["profile_header"]["model"]["num_experts"]),
        }

    def fit(self, rows: list[dict[str, Any]]) -> "RankLocalComputeModel":
        if not rows:
            raise ValueError("rank-local compute model requires multirank rows")
        identities = [self._identity(row) for row in rows]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("rank-local compute records cannot mix configuration identities")
        self.identity = identities[0]
        tp_size = self.identity["tensor_parallel_size"]
        ep_size = self.identity["expert_parallel_size"]
        samples: list[OperatorSample] = []
        for row in rows:
            plan = StepExecutionPlan.from_raw_step(row)
            rank_records = row.get("rank_records", [])
            if len(rank_records) != row.get("rank_count"):
                raise ValueError("rank-local compute row misses rank records")
            for rank_record in rank_records:
                if rank_record.get("rank_local_compute_schema") != "rank-local-compute/v1":
                    raise ValueError("record does not contain rank-local-compute/v1 anchors")
                rank = int(rank_record["rank"])
                layer_map = {
                    int(layer["layer_id"]): layer for layer in rank_record["layer_records"]
                }
                for layer in plan.layers:
                    raw_layer = layer_map[layer.layer_id]
                    compute = raw_layer["rank_local_compute_us"]
                    if not all(compute[stage]["valid"] for stage in compute):
                        raise ValueError("rank-local compute record contains invalid anchors")
                    attention_shape = (
                        float(plan.n_q), float(plan.total_kv_tokens),
                        float(plan.realized_batch_size), plan.kv_mean, float(plan.kv_max),
                    )
                    local = _local_histogram(
                        layer.tokens_per_expert, ep_size, rank % ep_size
                    )
                    samples.extend([
                        OperatorSample(
                            "rank_attention", plan.phase, attention_shape, plan.dtype,
                            self.identity["attention_backend"], f"tp={tp_size}",
                            float(compute["attention"]["duration_us"]),
                        ),
                        OperatorSample(
                            "rank_router", plan.phase, (float(plan.scheduled_tokens),),
                            plan.dtype, self.identity["moe_backend"], f"tp={tp_size}",
                            float(compute["router"]["duration_us"]),
                        ),
                        OperatorSample(
                            "rank_expert_gemm", plan.phase, _routing_shape(local),
                            plan.dtype, self.identity["moe_backend"],
                            f"tp={tp_size}:ep={ep_size}:local_experts={len(local)}:"
                            f"{self.identity['moe_kernel_config']}",
                            float(compute["expert_gemm"]["duration_us"]),
                        ),
                    ])
        self.database.fit(samples)
        self.provenance = {
            "artifact_type": ARTIFACT_TYPE,
            "artifact_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "label_source": "cuda_event_rank_local_compute_exclusive",
            "training_step_count": len(rows),
            "training_rank_step_count": sum(len(row["rank_records"]) for row in rows),
            "component_coverage": ["attention", "router", "expert_gemm"],
            "attention_boundary": "module_start_to_first_collective_start",
            "database": self.database.stats(),
            "identity": self.identity,
        }
        return self

    def predict_layer(
        self,
        plan: StepExecutionPlan,
        layer: LayerExecution,
        rank: int,
        layout: ParallelLayout,
    ) -> RankLocalLayerPrediction:
        if (layout.tp_size, layout.ep_size) != (
            self.identity["tensor_parallel_size"],
            self.identity["expert_parallel_size"],
        ):
            raise ValueError("rank-local compute artifact does not match TP/EP layout")
        tp_size = layout.tp_size
        ep_size = layout.ep_size
        attention = self.database.query(
            "rank_attention", plan.phase,
            (float(plan.n_q), float(plan.total_kv_tokens), float(plan.realized_batch_size),
             plan.kv_mean, float(plan.kv_max)),
            plan.dtype, self.identity["attention_backend"], f"tp={tp_size}",
        )
        router = self.database.query(
            "rank_router", plan.phase, (float(plan.scheduled_tokens),), plan.dtype,
            self.identity["moe_backend"], f"tp={tp_size}",
        )
        local = _local_histogram(
            layer.tokens_per_expert, ep_size, layout.expert_coordinate(rank)
        )
        expert = self.database.query(
            "rank_expert_gemm", plan.phase, _routing_shape(local), plan.dtype,
            self.identity["moe_backend"],
            f"tp={tp_size}:ep={ep_size}:local_experts={len(local)}:"
            f"{self.identity['moe_kernel_config']}",
        )
        estimates: dict[str, OperatorEstimate] = {
            "attention": attention, "router": router, "expert_gemm": expert,
        }
        return RankLocalLayerPrediction(
            attention_us=attention.latency_us,
            router_us=router.latency_us,
            expert_gemm_us=expert.latency_us,
            extrapolated=any(value.extrapolated for value in estimates.values()),
            nearest_distance=max(value.nearest_distance for value in estimates.values()),
            estimates={key: asdict(value) for key, value in estimates.items()},
        )

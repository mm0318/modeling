from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .execution_plan import STAGES, StepExecutionPlan
from .event_engine import DiscreteEventEngine, build_single_rank_graph
from .operator_db import OperatorEstimate, OperatorResponseDatabase, OperatorSample


ARTIFACT_TYPE = "whitebox-moe-v3"


def load_fine_raw_records(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            StepExecutionPlan.from_raw_step(row)
            rows.append(row)
    if not rows:
        raise ValueError("fine raw file contains no records")
    return rows


def save_whitebox_artifact(model: "WhiteboxMoEModel", path: str | Path) -> None:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_whitebox_artifact(path: str | Path) -> "WhiteboxMoEModel":
    import joblib

    model = joblib.load(Path(path))
    if not isinstance(model, WhiteboxMoEModel):
        raise TypeError("artifact is not a whitebox-moe-v3 model")
    return model


def _stage_query(
    plan: StepExecutionPlan, stage: str, routing: dict[str, float] | None = None
) -> tuple[tuple[float, ...], str, str]:
    if stage == "attention":
        return (
            (
                float(plan.n_q),
                float(plan.total_kv_tokens),
                float(plan.realized_batch_size),
                plan.kv_mean,
                float(plan.kv_max),
            ),
            plan.attention_backend,
            "attention",
        )
    if stage in {"router", "dense", "transition"}:
        backend = plan.moe_backend if stage == "router" else plan.dense_backend
        kernel = plan.moe_kernel_config if stage == "router" else stage
        return ((float(plan.scheduled_tokens),), backend, kernel)
    assert routing is not None
    if stage in {"dispatch", "combine"}:
        shape = (routing["routed_tokens"], routing["active_experts"])
    elif stage == "expert_gemm":
        shape = (
            routing["routed_tokens"],
            routing["active_experts"],
            routing["max_expert_load"],
            routing["p95_expert_load"],
        )
    else:
        raise ValueError(f"unsupported white-box stage: {stage}")
    return shape, plan.moe_backend, plan.moe_kernel_config


def samples_from_raw(raw_records: Iterable[dict[str, Any]]) -> list[OperatorSample]:
    samples: list[OperatorSample] = []
    for raw in raw_records:
        if raw.get("label_source") == "cuda_event" and raw.get("fixed_source") != (
            "direct_non_layer_dense_cuda_events"
        ):
            raise ValueError(
                "formal fine record Fixed must come from direct non-layer CUDA Events"
            )
        plan = StepExecutionPlan.from_raw_step(raw)
        fine_by_id = {int(row["layer_id"]): row for row in raw["layer_records"]}
        for layer in plan.layers:
            timing = fine_by_id[layer.layer_id].get("timing_us", {})
            routing = layer.routing_shape()
            for stage in STAGES:
                if stage not in timing:
                    raise ValueError(f"fine record layer {layer.layer_id} misses {stage} timing")
                shape, backend, kernel = _stage_query(plan, stage, routing)
                samples.append(
                    OperatorSample(
                        stage,
                        plan.phase,
                        shape,
                        plan.dtype,
                        backend,
                        kernel,
                        float(timing[stage]),
                    )
                )
        fixed_us = raw.get("fixed_us")
        if fixed_us is None:
            raise ValueError("fine record misses fixed_us")
        samples.append(
            OperatorSample(
                "fixed",
                plan.phase,
                (
                    float(plan.realized_batch_size),
                    float(plan.scheduled_tokens),
                    float(plan.total_kv_tokens),
                ),
                plan.dtype,
                plan.dense_backend,
                "fixed",
                float(fixed_us),
            )
        )
    return samples


@dataclass(frozen=True)
class WhiteboxProfile:
    step_us: float
    total_work_us: float
    memory_bytes: int
    components_us: dict[str, float]
    component_shares: dict[str, float]
    work_shares: dict[str, float]
    compatibility_shares: dict[str, float]
    primary_bottleneck: str
    layer_profiles: tuple[dict[str, Any], ...]
    extrapolated_signatures: tuple[str, ...]
    signature_distance: dict[str, float]
    composition_mode: str
    overlap_audit_status: str
    artifact_provenance: dict[str, Any]
    execution_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_profiles"] = list(self.layer_profiles)
        result["extrapolated_signatures"] = list(self.extrapolated_signatures)
        return result


class WhiteboxMoEModel:
    artifact_type = ARTIFACT_TYPE

    def __init__(self, *, composition_mode: str = "event_dag", neighbors: int = 4):
        if composition_mode not in {"event_dag", "serial"}:
            raise ValueError("white-box composition must be event_dag")
        self.composition_mode = composition_mode
        self.database = OperatorResponseDatabase(neighbors=neighbors)
        self.provenance: dict[str, Any] = {}

    def fit_raw(
        self,
        raw_records: list[dict[str, Any]],
        *,
        allow_synthetic: bool = False,
        audit_report: dict[str, Any] | None = None,
    ) -> "WhiteboxMoEModel":
        if not raw_records:
            raise ValueError("white-box model requires fine-grained raw records")
        label_sources = {row.get("label_source") for row in raw_records}
        if len(label_sources) != 1:
            raise ValueError("white-box records cannot mix label sources")
        label_source = next(iter(label_sources))
        if label_source != "cuda_event" and not (
            label_source == "synthetic_fixture" and allow_synthetic
        ):
            raise ValueError(
                "formal white-box training requires cuda_event; synthetic fixture requires explicit opt-in"
            )
        self.database.fit(samples_from_raw(raw_records))
        phases = sorted({StepExecutionPlan.from_raw_step(row).phase for row in raw_records})
        prompt_patterns = sorted(
            {str(row.get("prompt_pattern", "unspecified")) for row in raw_records}
        )
        audit_status = "serial_assumption_unverified"
        if audit_report is not None:
            if label_source != "cuda_event":
                raise ValueError("Nsight audit can only certify formal CUDA Event records")
            if not audit_report.get("serial_validated"):
                raise ValueError("Nsight audit did not validate serial composition")
            audit_status = "nsight_serial_validated"
        self.provenance = {
            "artifact_type": ARTIFACT_TYPE,
            "artifact_version": "3.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_scope": (
                "same-runtime-fine-grained"
                if label_source == "cuda_event"
                else "synthetic-fixture"
            ),
            "label_source": label_source,
            "composition_mode": self.composition_mode,
            "overlap_audit_status": audit_status,
            "overlap_audit": audit_report,
            "training_step_count": len(raw_records),
            "phases": phases,
            "prompt_patterns": prompt_patterns,
            "operator_database": self.database.stats(),
            "fixed_semantics": "direct_non_layer_dense_cuda_events",
        }
        return self

    def _query(
        self,
        plan: StepExecutionPlan,
        stage: str,
        routing: dict[str, float] | None = None,
    ) -> OperatorEstimate:
        shape, backend, kernel = _stage_query(plan, stage, routing)
        return self.database.query(stage, plan.phase, shape, plan.dtype, backend, kernel)

    def predict(self, plan: StepExecutionPlan) -> WhiteboxProfile:
        totals = {stage: 0.0 for stage in STAGES}
        totals["fixed"] = 0.0
        extrapolated: list[str] = []
        estimates: list[OperatorEstimate] = []
        cache: dict[tuple[Any, ...], OperatorEstimate] = {}

        def query(stage: str, routing: dict[str, float] | None = None) -> OperatorEstimate:
            shape, backend, kernel = _stage_query(plan, stage, routing)
            key = (stage, plan.phase, shape, plan.dtype, backend, kernel)
            if key not in cache:
                cache[key] = self.database.query(*key)
            estimate = cache[key]
            estimates.append(estimate)
            if estimate.extrapolated:
                signature = repr(key)
                if signature not in extrapolated:
                    extrapolated.append(signature)
            return estimate

        layer_profiles = []
        for layer in plan.layers:
            routing = layer.routing_shape()
            layer_times = {}
            for stage in STAGES:
                estimate = query(stage, routing)
                value = estimate.latency_us * layer.execution_counts[stage]
                layer_times[stage] = value
                totals[stage] += value
            layer_profiles.append(
                {
                    "layer_id": layer.layer_id,
                    "routing": routing,
                    "components_us": layer_times,
                    "step_us": sum(layer_times.values()),
                }
            )
        fixed_shape = (
            float(plan.realized_batch_size),
            float(plan.scheduled_tokens),
            float(plan.total_kv_tokens),
        )
        fixed_key = (
            "fixed",
            plan.phase,
            fixed_shape,
            plan.dtype,
            plan.dense_backend,
            "fixed",
        )
        fixed_estimate = self.database.query(*fixed_key)
        estimates.append(fixed_estimate)
        totals["fixed"] = fixed_estimate.latency_us * plan.fixed_execution_count
        if fixed_estimate.extrapolated:
            extrapolated.append(repr(fixed_key))
        graph = build_single_rank_graph(
            layer_profiles,
            fixed_us=totals["fixed"],
            provenance={
                "timing_source": self.provenance.get("label_source"),
                "composition": "event_dag",
                "formal_hardware_prediction": self.provenance.get("label_source")
                == "cuda_event",
            },
        )
        execution = DiscreteEventEngine().simulate(graph)
        step_us = execution.step_us
        total_work_us = execution.total_work_us
        shares = {
            key: execution.critical_path_category_us.get(key, 0.0) / step_us
            if step_us
            else 0.0
            for key in totals
        }
        work_shares = {
            key: value / total_work_us if total_work_us else 0.0
            for key, value in totals.items()
        }
        compatibility = {
            "attention": shares["attention"],
            "moe": sum(shares[key] for key in ("router", "dispatch", "expert_gemm", "combine")),
            "dense": shares["dense"],
            "transition_fixed": shares["transition"] + shares["fixed"],
        }
        memory = (
            plan.model_weight_bytes
            + plan.total_kv_tokens * plan.kv_bytes_per_token
            + plan.scheduled_tokens * plan.activation_bytes_per_scheduled_token
        )
        return WhiteboxProfile(
            step_us=step_us,
            total_work_us=total_work_us,
            memory_bytes=memory,
            components_us=totals,
            component_shares=shares,
            work_shares=work_shares,
            compatibility_shares=compatibility,
            primary_bottleneck=execution.primary_bottleneck,
            layer_profiles=tuple(layer_profiles),
            extrapolated_signatures=tuple(extrapolated),
            signature_distance={
                "mean": sum(item.nearest_distance for item in estimates) / len(estimates),
                "max": max(item.nearest_distance for item in estimates),
                "queries": float(len(estimates)),
            },
            composition_mode=self.composition_mode,
            overlap_audit_status=self.provenance["overlap_audit_status"],
            artifact_provenance=self.provenance,
            execution_report=execution.to_dict(),
        )

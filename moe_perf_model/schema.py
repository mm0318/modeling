from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SchemaError(ValueError):
    """Raised when an input violates the modeling contract."""


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SchemaError(f"{name} must be a JSON object")
    return raw


def _positive_int(raw: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{key} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        raise SchemaError(f"{key} must be >= {lower}")
    return value


def _positive_float(raw: dict[str, Any], key: str, *, allow_zero: bool = False) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{key} must be numeric")
    value = float(value)
    lower_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not lower_ok:
        op = ">=" if allow_zero else ">"
        raise SchemaError(f"{key} must be finite and {op} 0")
    return value


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_moe_layers: int

    @classmethod
    def from_dict(cls, raw: Any) -> "ModelSpec":
        raw = _mapping(raw, "model")
        result = cls(
            model_id=_string(raw, "model_id"),
            num_layers=_positive_int(raw, "num_layers"),
            hidden_size=_positive_int(raw, "hidden_size"),
            num_experts=_positive_int(raw, "num_experts"),
            top_k=_positive_int(raw, "top_k"),
            expert_intermediate_size=_positive_int(raw, "expert_intermediate_size"),
            num_attention_heads=_positive_int(raw, "num_attention_heads"),
            num_kv_heads=_positive_int(raw, "num_kv_heads"),
            num_moe_layers=(
                _positive_int(raw, "num_moe_layers")
                if "num_moe_layers" in raw
                else _positive_int(raw, "num_layers")
            ),
        )
        if result.top_k > result.num_experts:
            raise SchemaError("top_k cannot exceed num_experts")
        if result.num_attention_heads % result.num_kv_heads:
            raise SchemaError("num_attention_heads must be divisible by num_kv_heads")
        if result.num_moe_layers > result.num_layers:
            raise SchemaError("num_moe_layers cannot exceed num_layers")
        return result


@dataclass(frozen=True)
class HardwareSpec:
    hardware_id: str
    model_weight_bytes: int
    kv_bytes_per_token: int
    activation_bytes_per_scheduled_token: int

    @classmethod
    def from_dict(cls, raw: Any) -> "HardwareSpec":
        raw = _mapping(raw, "hardware")
        return cls(
            hardware_id=_string(raw, "hardware_id"),
            model_weight_bytes=_positive_int(raw, "model_weight_bytes"),
            kv_bytes_per_token=_positive_int(raw, "kv_bytes_per_token"),
            activation_bytes_per_scheduled_token=_positive_int(
                raw, "activation_bytes_per_scheduled_token", allow_zero=True
            ),
        )


@dataclass(frozen=True)
class WorkloadSpec:
    phase: str
    batch_size: int
    input_length: int
    context_length: int
    query_length: int
    scheduled_tokens: int
    total_kv_tokens: int

    @classmethod
    def from_dict(cls, raw: Any) -> "WorkloadSpec":
        raw = _mapping(raw, "workload")
        phase = _string(raw, "phase").lower()
        if phase not in {"prefill", "decode"}:
            raise SchemaError("phase must be prefill or decode")
        result = cls(
            phase=phase,
            batch_size=_positive_int(raw, "batch_size"),
            input_length=_positive_int(raw, "input_length", allow_zero=True),
            context_length=_positive_int(raw, "context_length"),
            query_length=_positive_int(raw, "query_length"),
            scheduled_tokens=_positive_int(raw, "scheduled_tokens"),
            total_kv_tokens=_positive_int(raw, "total_kv_tokens"),
        )
        if result.phase == "decode" and result.query_length > result.batch_size:
            raise SchemaError("decode query_length cannot exceed batch_size")
        if result.context_length < result.query_length:
            raise SchemaError("context_length cannot be smaller than query_length")
        return result


@dataclass(frozen=True)
class RoutingSpec:
    routed_tokens: int
    active_experts: int
    tokens_per_expert: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Any, model: ModelSpec, scheduled_tokens: int) -> "RoutingSpec":
        raw = _mapping(raw, "routing")
        values = raw.get("tokens_per_expert")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise SchemaError("tokens_per_expert must be a list of non-negative integers")
        result = cls(
            routed_tokens=_positive_int(raw, "routed_tokens"),
            active_experts=_positive_int(raw, "active_experts"),
            tokens_per_expert=tuple(values),
        )
        if len(values) != model.num_experts:
            raise SchemaError(
                f"tokens_per_expert must contain {model.num_experts} values"
            )
        if sum(values) != result.routed_tokens:
            raise SchemaError("tokens_per_expert sum must equal routed_tokens")
        if sum(value > 0 for value in values) != result.active_experts:
            raise SchemaError("non-zero expert count must equal active_experts")
        expected = scheduled_tokens * model.top_k * model.num_moe_layers
        if result.routed_tokens != expected:
            raise SchemaError(
                "routed_tokens must equal scheduled_tokens * top_k * "
                f"num_moe_layers ({expected})"
            )
        return result

    def statistics(self) -> dict[str, float]:
        values = [float(value) for value in self.tokens_per_expert]
        ordered = sorted(values)
        mean = sum(values) / len(values)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        cumulative = sum((index + 1) * value for index, value in enumerate(ordered))
        total = sum(values)
        gini = (
            (2 * cumulative) / (len(values) * total) - (len(values) + 1) / len(values)
            if total
            else 0.0
        )
        return {
            "mean": mean,
            "p95": ordered[p95_index],
            "max": max(values),
            "cv": math.sqrt(variance) / mean if mean else 0.0,
            "gini": max(0.0, min(1.0, gini)),
            "zero_ratio": sum(value == 0 for value in values) / len(values),
        }


@dataclass(frozen=True)
class RuntimeSpec:
    runtime: str
    vllm_version: str
    precision: str
    cuda_graph: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    tensor_parallel_size: int
    expert_parallel_size: int

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeSpec":
        raw = _mapping(raw, "runtime")
        cuda_graph = raw.get("cuda_graph")
        if not isinstance(cuda_graph, bool):
            raise SchemaError("cuda_graph must be boolean")
        result = cls(
            runtime=_string(raw, "runtime").lower(),
            vllm_version=_string(raw, "vllm_version"),
            precision=_string(raw, "precision").lower(),
            cuda_graph=cuda_graph,
            max_num_batched_tokens=_positive_int(raw, "max_num_batched_tokens"),
            max_num_seqs=_positive_int(raw, "max_num_seqs"),
            tensor_parallel_size=_positive_int(raw, "tensor_parallel_size"),
            expert_parallel_size=_positive_int(raw, "expert_parallel_size"),
        )
        if result.runtime != "vllm":
            raise SchemaError("current component model supports runtime=vllm only")
        if result.tensor_parallel_size != 1 or result.expert_parallel_size != 1:
            raise SchemaError("TP/EP > 1 requires a communication model and is not supported yet")
        return result


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    routing: RoutingSpec
    runtime: RuntimeSpec

    @classmethod
    def from_dict(cls, raw: Any) -> "InferenceConfig":
        raw = _mapping(raw, "config")
        model = ModelSpec.from_dict(raw.get("model"))
        workload = WorkloadSpec.from_dict(raw.get("workload"))
        result = cls(
            model=model,
            hardware=HardwareSpec.from_dict(raw.get("hardware")),
            workload=workload,
            routing=RoutingSpec.from_dict(
                raw.get("routing"), model, workload.scheduled_tokens
            ),
            runtime=RuntimeSpec.from_dict(raw.get("runtime")),
        )
        if workload.batch_size > result.runtime.max_num_seqs:
            raise SchemaError("batch_size exceeds max_num_seqs")
        if workload.scheduled_tokens > result.runtime.max_num_batched_tokens:
            raise SchemaError("scheduled_tokens exceeds max_num_batched_tokens")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TrainingLabels:
    label_source: str
    runtime: str
    vllm_version: str
    precision: str
    attention_us: float
    expert_us: float
    dense_us: float
    other_us: float
    step_us: float
    peak_memory_bytes: int

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingLabels":
        raw = _mapping(raw, "labels")
        result = cls(
            label_source=_string(raw, "label_source").lower(),
            runtime=_string(raw, "runtime").lower(),
            vllm_version=_string(raw, "vllm_version"),
            precision=_string(raw, "precision").lower(),
            attention_us=_positive_float(raw, "attention_us", allow_zero=True),
            expert_us=_positive_float(raw, "expert_us", allow_zero=True),
            dense_us=_positive_float(raw, "dense_us", allow_zero=True),
            other_us=_positive_float(raw, "other_us", allow_zero=True),
            step_us=_positive_float(raw, "step_us"),
            peak_memory_bytes=_positive_int(raw, "peak_memory_bytes"),
        )
        if result.label_source not in {"cuda_event", "synthetic_fixture"}:
            raise SchemaError(
                "label_source must be cuda_event or synthetic_fixture"
            )
        component_sum = (
            result.attention_us + result.expert_us + result.dense_us + result.other_us
        )
        if component_sum > result.step_us * (1 + 1e-6):
            raise SchemaError("component CUDA times cannot exceed step_us")
        return result


@dataclass(frozen=True)
class TrainingRecord:
    config: InferenceConfig
    labels: TrainingLabels

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingRecord":
        raw = _mapping(raw, "training record")
        result = cls(
            config=InferenceConfig.from_dict(raw.get("config")),
            labels=TrainingLabels.from_dict(raw.get("labels")),
        )
        runtime = result.config.runtime
        labels = result.labels
        if (labels.runtime, labels.vllm_version, labels.precision) != (
            runtime.runtime,
            runtime.vllm_version,
            runtime.precision,
        ):
            raise SchemaError("label runtime/version/precision must match config")
        return result


@dataclass(frozen=True)
class PerformanceProfile:
    config_id: str
    step_us: float
    attention_us: float
    expert_us: float
    dense_us: float
    other_us: float
    residual_us: float
    memory_bytes: int
    component_shares: dict[str, float]
    primary_bottleneck: str
    extrapolated_components: tuple[str, ...]
    artifact_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["extrapolated_components"] = list(self.extrapolated_components)
        return result


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: str | Path) -> InferenceConfig:
    return InferenceConfig.from_dict(load_json(path))


def load_training_records(path: str | Path) -> list[TrainingRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(TrainingRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, SchemaError) as error:
                raise SchemaError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise SchemaError("training file contains no records")
    return records

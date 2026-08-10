from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .schema import SchemaError


VALID_PHASES = {"prefill", "decode", "mixed"}
STAGES = (
    "attention",
    "router",
    "dispatch",
    "expert_gemm",
    "combine",
    "dense",
    "transition",
)


@dataclass(frozen=True)
class LayerExecution:
    layer_id: int
    tokens_per_expert: tuple[int, ...]
    execution_counts: dict[str, int]

    @classmethod
    def from_dict(cls, raw: Any, num_experts: int) -> "LayerExecution":
        if not isinstance(raw, dict):
            raise SchemaError("layer execution must be an object")
        layer_id = raw.get("layer_id")
        values = raw.get("tokens_per_expert")
        counts = raw.get("execution_counts", {stage: 1 for stage in STAGES})
        if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
            raise SchemaError("layer_id must be a non-negative integer")
        if not isinstance(values, list) or len(values) != num_experts or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise SchemaError(f"tokens_per_expert must contain {num_experts} integers")
        if not isinstance(counts, dict) or set(counts) != set(STAGES):
            raise SchemaError("execution_counts must contain every white-box stage")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
            raise SchemaError("execution counts must be non-negative integers")
        return cls(layer_id, tuple(values), dict(counts))

    def routing_shape(self) -> dict[str, float]:
        values = sorted(float(value) for value in self.tokens_per_expert)
        mean = sum(values) / len(values)
        p95 = values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
        return {
            "routed_tokens": float(sum(values)),
            "active_experts": float(sum(value > 0 for value in values)),
            "max_expert_load": max(values),
            "p95_expert_load": p95,
            "max_mean": max(values) / mean if mean else 0.0,
        }


@dataclass(frozen=True)
class StepExecutionPlan:
    phase: str
    realized_batch_size: int
    scheduled_tokens: int
    n_q: int
    total_kv_tokens: int
    kv_mean: float
    kv_max: int
    dtype: str
    attention_backend: str
    moe_backend: str
    moe_kernel_config: str
    dense_backend: str
    num_experts: int
    top_k: int
    model_weight_bytes: int
    kv_bytes_per_token: int
    activation_bytes_per_scheduled_token: int
    layers: tuple[LayerExecution, ...]
    fixed_execution_count: int = 1

    @classmethod
    def from_dict(cls, raw: Any) -> "StepExecutionPlan":
        if not isinstance(raw, dict):
            raise SchemaError("execution plan must be an object")
        phase = str(raw.get("phase", "")).lower()
        if phase not in VALID_PHASES:
            raise SchemaError("plan phase must be prefill, decode, or mixed")
        integers = {}
        for key in (
            "realized_batch_size", "scheduled_tokens", "n_q", "total_kv_tokens",
            "kv_max", "num_experts", "top_k", "model_weight_bytes",
            "kv_bytes_per_token", "activation_bytes_per_scheduled_token",
        ):
            value = raw.get(key)
            minimum = 0 if key == "activation_bytes_per_scheduled_token" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SchemaError(f"{key} must be an integer >= {minimum}")
            integers[key] = value
        kv_mean = raw.get("kv_mean")
        if isinstance(kv_mean, bool) or not isinstance(kv_mean, (int, float)) or not math.isfinite(float(kv_mean)) or kv_mean <= 0:
            raise SchemaError("kv_mean must be positive and finite")
        strings = {}
        for key in ("dtype", "attention_backend", "moe_backend", "moe_kernel_config", "dense_backend"):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"{key} must be a non-empty string")
            strings[key] = value.strip()
        layer_rows = raw.get("layers")
        if not isinstance(layer_rows, list) or not layer_rows:
            raise SchemaError("layers must be a non-empty list")
        layers = tuple(LayerExecution.from_dict(row, integers["num_experts"]) for row in layer_rows)
        if len({layer.layer_id for layer in layers}) != len(layers):
            raise SchemaError("layer_id values must be unique")
        expected_routed = integers["scheduled_tokens"] * integers["top_k"]
        for layer in layers:
            if sum(layer.tokens_per_expert) != expected_routed:
                raise SchemaError(
                    f"layer {layer.layer_id} routed tokens must equal scheduled_tokens * top_k"
                )
        fixed_count = raw.get("fixed_execution_count", 1)
        if isinstance(fixed_count, bool) or not isinstance(fixed_count, int) or fixed_count < 0:
            raise SchemaError("fixed_execution_count must be non-negative")
        return cls(
            phase=phase,
            layers=layers,
            kv_mean=float(kv_mean),
            fixed_execution_count=fixed_count,
            **integers,
            **strings,
        )

    @classmethod
    def from_raw_step(cls, raw: dict[str, Any]) -> "StepExecutionPlan":
        requests = raw.get("requests", [])
        layer_records = raw.get("layer_records")
        identity = raw.get("kernel_identity", {})
        header = raw.get("profile_header", {})
        if not requests or not isinstance(layer_records, list) or not layer_records:
            raise SchemaError("fine raw step requires requests and layer_records")
        phases = {row["phase"] for row in requests}
        phase = next(iter(phases)) if len(phases) == 1 else "mixed"
        contexts = [int(row["context_length"]) for row in requests]
        scheduled = [int(row["scheduled_tokens"]) for row in requests]
        model = header.get("model", {})
        runtime = header.get("runtime", {})
        hardware = header.get("hardware", {})
        return cls.from_dict(
            {
                "phase": phase,
                "realized_batch_size": len(requests),
                "scheduled_tokens": sum(scheduled),
                "n_q": max(scheduled),
                "total_kv_tokens": sum(contexts),
                "kv_mean": sum(contexts) / len(contexts),
                "kv_max": max(contexts),
                "dtype": runtime.get("precision", identity.get("dtype", "unknown")),
                "attention_backend": identity.get("attention_backend", "unknown"),
                "moe_backend": identity.get("moe_backend", "unknown"),
                "moe_kernel_config": identity.get("moe_kernel_config", "unknown"),
                "dense_backend": identity.get("dense_backend", "torch"),
                "num_experts": model.get("num_experts", len(layer_records[0]["tokens_per_expert"])),
                "top_k": model.get("top_k", 0),
                "model_weight_bytes": hardware.get("model_weight_bytes", 0),
                "kv_bytes_per_token": hardware.get("kv_bytes_per_token", 0),
                "activation_bytes_per_scheduled_token": hardware.get(
                    "activation_bytes_per_scheduled_token", 0
                ),
                "layers": [
                    {
                        "layer_id": row["layer_id"],
                        "tokens_per_expert": row["tokens_per_expert"],
                        "execution_counts": row.get("execution_counts", {stage: 1 for stage in STAGES}),
                    }
                    for row in layer_records
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

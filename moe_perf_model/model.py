from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .response import IDWResponse, PiecewiseLinearResponse
from .schema import InferenceConfig, PerformanceProfile, SchemaError, TrainingRecord


ARTIFACT_VERSION = "1.0"


def _shape_features(config: InferenceConfig, component: str) -> list[float]:
    workload = config.workload
    if component == "attention":
        return [
            math.log1p(workload.batch_size),
            math.log1p(workload.query_length),
            math.log1p(workload.context_length),
            math.log1p(workload.total_kv_tokens),
        ]
    if component == "expert":
        return [
            math.log1p(config.routing.routed_tokens),
            math.log1p(config.routing.active_experts),
        ]
    raise ValueError(f"unsupported component shape: {component}")


def _routing_features(config: InferenceConfig) -> list[float]:
    stats = config.routing.statistics()
    mean = max(stats["mean"], 1e-9)
    return [
        stats["max"] / mean,
        stats["p95"] / mean,
        stats["cv"],
        stats["gini"],
        stats["zero_ratio"],
    ]


def _small_features(config: InferenceConfig) -> list[float]:
    workload = config.workload
    return [
        1.0 if workload.phase == "prefill" else 0.0,
        math.log1p(workload.batch_size),
        math.log1p(workload.query_length),
        math.log1p(workload.context_length),
        math.log1p(workload.scheduled_tokens),
        math.log1p(workload.total_kv_tokens),
        1.0 if config.runtime.cuda_graph else 0.0,
    ]


def _basic_features(config: InferenceConfig) -> list[float]:
    return _small_features(config)[:-1] + [
        math.log1p(config.runtime.max_num_batched_tokens),
        math.log1p(config.runtime.max_num_seqs),
        1.0 if config.runtime.cuda_graph else 0.0,
    ]


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


class ComponentModel:
    artifact_type = "component"

    def __init__(self):
        self.identity: dict[str, Any] | None = None
        self.attention: dict[str, IDWResponse] = {}
        self.dense: dict[str, PiecewiseLinearResponse] = {}
        self.expert: dict[str, IDWResponse] = {}
        self.routing_correction: dict[str, Any] = {}
        self.other_model: Any | None = None
        self.residual_model: Any | None = None
        self.provenance: dict[str, Any] = {}

    @staticmethod
    def _identity(config: InferenceConfig) -> dict[str, Any]:
        return {
            "model_id": config.model.model_id,
            "model_structure": {
                "num_layers": config.model.num_layers,
                "hidden_size": config.model.hidden_size,
                "num_experts": config.model.num_experts,
                "top_k": config.model.top_k,
                "expert_intermediate_size": config.model.expert_intermediate_size,
                "num_attention_heads": config.model.num_attention_heads,
                "num_kv_heads": config.model.num_kv_heads,
            },
            "hardware_id": config.hardware.hardware_id,
            "runtime": config.runtime.runtime,
            "vllm_version": config.runtime.vllm_version,
            "precision": config.runtime.precision,
            "tensor_parallel_size": config.runtime.tensor_parallel_size,
            "expert_parallel_size": config.runtime.expert_parallel_size,
        }

    def _validate_identity(self, config: InferenceConfig) -> None:
        if self.identity is None:
            raise RuntimeError("component model has not been fitted")
        candidate = self._identity(config)
        if candidate != self.identity:
            mismatches = [
                key for key in self.identity if candidate.get(key) != self.identity.get(key)
            ]
            raise SchemaError(
                "config is incompatible with artifact: " + ", ".join(mismatches)
            )

    def fit(
        self, records: list[TrainingRecord], *, allow_synthetic: bool = False
    ) -> "ComponentModel":
        if len(records) < 6:
            raise ValueError("component model requires at least six training records")
        label_sources = {record.labels.label_source for record in records}
        if len(label_sources) != 1:
            raise SchemaError("training records cannot mix label sources")
        label_source = next(iter(label_sources))
        if label_source != "cuda_event" and not (
            label_source == "synthetic_fixture" and allow_synthetic
        ):
            raise SchemaError(
                "formal training requires cuda_event labels; synthetic fixtures require "
                "allow_synthetic=True"
            )
        identities = [self._identity(record.config) for record in records]
        first_identity = identities[0]
        if any(identity != first_identity for identity in identities[1:]):
            raise SchemaError("all training records must share model/hardware/runtime identity")
        phase_counts = Counter(record.config.workload.phase for record in records)
        if any(phase_counts.get(phase, 0) < 3 for phase in ("prefill", "decode")):
            raise ValueError("component model requires at least three records per phase")
        self.identity = first_identity

        for phase in ("prefill", "decode"):
            subset = [record for record in records if record.config.workload.phase == phase]
            self.attention[phase] = IDWResponse(k=4).fit(
                [_shape_features(record.config, "attention") for record in subset],
                [record.labels.attention_us for record in subset],
            )
            self.dense[phase] = PiecewiseLinearResponse().fit(
                [math.log1p(record.config.workload.scheduled_tokens) for record in subset],
                [record.labels.dense_us for record in subset],
            )
            self.expert[phase] = IDWResponse(k=4).fit(
                [_shape_features(record.config, "expert") for record in subset],
                [record.labels.expert_us for record in subset],
            )
            base_expert = np.asarray([
                self.expert[phase].predict(_shape_features(record.config, "expert")).value
                for record in subset
            ])
            routing_target = np.asarray(
                [record.labels.expert_us for record in subset], dtype=float
            ) - base_expert
            correction = _ridge()
            correction.fit(
                np.asarray([_routing_features(record.config) for record in subset]),
                routing_target,
            )
            self.routing_correction[phase] = correction

        features = np.asarray([_small_features(record.config) for record in records])
        self.other_model = _ridge()
        self.other_model.fit(
            features, np.asarray([record.labels.other_us for record in records], dtype=float)
        )

        component_predictions = []
        for record in records:
            attention, expert, dense, other, _ = self._predict_components(record.config)
            component_predictions.append(attention + expert + dense + other)
        residual_target = np.asarray(
            [record.labels.step_us for record in records], dtype=float
        ) - np.asarray(component_predictions)
        self.residual_model = _ridge()
        self.residual_model.fit(features, residual_target)

        ranges = {}
        for phase in ("prefill", "decode"):
            subset = [record.config for record in records if record.config.workload.phase == phase]
            ranges[phase] = {
                "batch_size": [
                    min(config.workload.batch_size for config in subset),
                    max(config.workload.batch_size for config in subset),
                ],
                "query_length": [
                    min(config.workload.query_length for config in subset),
                    max(config.workload.query_length for config in subset),
                ],
                "context_length": [
                    min(config.workload.context_length for config in subset),
                    max(config.workload.context_length for config in subset),
                ],
                "scheduled_tokens": [
                    min(config.workload.scheduled_tokens for config in subset),
                    max(config.workload.scheduled_tokens for config in subset),
                ],
                "routed_tokens": [
                    min(config.routing.routed_tokens for config in subset),
                    max(config.routing.routed_tokens for config in subset),
                ],
                "active_experts": [
                    min(config.routing.active_experts for config in subset),
                    max(config.routing.active_experts for config in subset),
                ],
            }
        self.provenance = {
            "artifact_version": ARTIFACT_VERSION,
            "artifact_type": self.artifact_type,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_record_count": len(records),
            "phase_counts": dict(phase_counts),
            "training_ranges": ranges,
            "label_source": label_source,
            "calibration_scope": (
                "same-runtime" if label_source == "cuda_event" else "synthetic-fixture"
            ),
        }
        return self

    def _predict_components(
        self, config: InferenceConfig
    ) -> tuple[float, float, float, float, list[str]]:
        phase = config.workload.phase
        attention = self.attention[phase].predict(_shape_features(config, "attention"))
        expert_base = self.expert[phase].predict(_shape_features(config, "expert"))
        routing_delta = float(
            self.routing_correction[phase].predict([_routing_features(config)])[0]
        )
        expert = max(0.0, expert_base.value + routing_delta)
        dense = self.dense[phase].predict(math.log1p(config.workload.scheduled_tokens))
        other = max(0.0, float(self.other_model.predict([_small_features(config)])[0]))
        extrapolated = []
        if attention.extrapolated:
            extrapolated.append("attention")
        if expert_base.extrapolated:
            extrapolated.append("expert")
        if dense.extrapolated:
            extrapolated.append("dense")
        return attention.value, expert, dense.value, other, extrapolated

    def predict(self, config: InferenceConfig) -> PerformanceProfile:
        self._validate_identity(config)
        if self.other_model is None or self.residual_model is None:
            raise RuntimeError("component model has not been fitted")
        attention, expert, dense, other_base, extrapolated = self._predict_components(config)
        residual_raw = float(self.residual_model.predict([_small_features(config)])[0])
        other = max(0.0, other_base + residual_raw)
        residual_effective = other - other_base
        components = {
            "attention": attention,
            "expert": expert,
            "dense": dense,
            "other": other,
        }
        step = sum(components.values())
        if step <= 0:
            raise RuntimeError("component model produced a non-positive step latency")
        shares = {key: value / step for key, value in components.items()}
        primary = max(shares, key=shares.get)
        hardware = config.hardware
        memory = (
            hardware.model_weight_bytes
            + hardware.kv_bytes_per_token * config.workload.total_kv_tokens
            + hardware.activation_bytes_per_scheduled_token
            * config.workload.scheduled_tokens
        )
        return PerformanceProfile(
            config_id=config.signature(),
            step_us=step,
            attention_us=attention,
            expert_us=expert,
            dense_us=dense,
            other_us=other,
            residual_us=residual_effective,
            memory_bytes=int(memory),
            component_shares=shares,
            primary_bottleneck=primary,
            extrapolated_components=tuple(extrapolated),
            artifact_provenance=dict(self.provenance),
        )


class BasicRFModel:
    artifact_type = "basic-rf"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.identity: dict[str, Any] | None = None
        self.regressor = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=1, random_state=seed, n_jobs=-1
        )
        self.provenance: dict[str, Any] = {}

    def fit(
        self, records: list[TrainingRecord], *, allow_synthetic: bool = False
    ) -> "BasicRFModel":
        if len(records) < 4:
            raise ValueError("Basic RF requires at least four records")
        label_sources = {record.labels.label_source for record in records}
        if len(label_sources) != 1:
            raise SchemaError("training records cannot mix label sources")
        label_source = next(iter(label_sources))
        if label_source != "cuda_event" and not (
            label_source == "synthetic_fixture" and allow_synthetic
        ):
            raise SchemaError(
                "formal training requires cuda_event labels; synthetic fixtures require "
                "allow_synthetic=True"
            )
        identities = [ComponentModel._identity(record.config) for record in records]
        if any(identity != identities[0] for identity in identities[1:]):
            raise SchemaError("all training records must share model/hardware/runtime identity")
        self.identity = identities[0]
        self.regressor.fit(
            np.asarray([_basic_features(record.config) for record in records]),
            np.asarray([record.labels.step_us for record in records]),
        )
        self.provenance = {
            "artifact_version": ARTIFACT_VERSION,
            "artifact_type": self.artifact_type,
            "training_record_count": len(records),
            "label_source": label_source,
            "calibration_scope": (
                "same-runtime" if label_source == "cuda_event" else "synthetic-fixture"
            ),
        }
        return self

    def predict_step_us(self, config: InferenceConfig) -> float:
        if self.identity != ComponentModel._identity(config):
            raise SchemaError("config is incompatible with Basic RF artifact")
        return max(0.0, float(self.regressor.predict([_basic_features(config)])[0]))


def save_artifact(model: ComponentModel | BasicRFModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_artifact(path: str | Path) -> ComponentModel | BasicRFModel:
    model = joblib.load(Path(path))
    if not isinstance(model, (ComponentModel, BasicRFModel)):
        raise TypeError("artifact is not a supported performance model")
    return model

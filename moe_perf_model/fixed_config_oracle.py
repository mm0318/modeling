from __future__ import annotations

import json
import math
import copy
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .execution_plan import StepExecutionPlan


ARTIFACT_TYPE = "fixed-config-forward-v1"
FEATURE_NAMES = (
    "log_batch", "log_scheduled", "log_n_q", "log_total_kv", "log_kv_mean",
    "log_kv_max", "prefill_request_fraction", "prefill_token_fraction",
    "ep_rank_max_mean_avg", "ep_rank_max_mean_p95", "ep_rank_max_mean_max",
    "expert_max_mean_avg", "active_expert_fraction_avg",
)


def load_multirank_records(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "multirank-forward/v1":
                raise ValueError(f"{path}:{line_number}: not a multi-rank forward record")
            if row.get("label_source") != "cuda_event_multirank":
                raise ValueError(f"{path}:{line_number}: invalid label source")
            rows.append(row)
    if not rows:
        raise ValueError("multi-rank record file is empty")
    return rows


def merge_multirank_record_files(
    paths: Iterable[str | Path], output: str | Path
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    identity: str | None = None
    next_workload = 0
    sources = []
    for source_value in paths:
        source = Path(source_value)
        rows = load_multirank_records(source)
        source_identities = {_stable_json(_config_identity(row)) for row in rows}
        if len(source_identities) != 1:
            raise ValueError(f"{source} contains multiple deployment identities")
        source_identity = next(iter(source_identities))
        if identity is None:
            identity = source_identity
        elif source_identity != identity:
            raise ValueError("cannot merge different fixed configuration identities")
        workload_map = {
            old: next_workload + index
            for index, old in enumerate(sorted({int(row["workload_index"]) for row in rows}))
        }
        for row in rows:
            copied = copy.deepcopy(row)
            old_workload = int(row["workload_index"])
            copied["source_file"] = str(source)
            copied["source_workload_index"] = old_workload
            copied["workload_index"] = workload_map[old_workload]
            merged.append(copied)
        next_workload += len(workload_map)
        sources.append({
            "path": str(source), "record_count": len(rows),
            "workload_count": len(workload_map),
        })
    if not merged:
        raise ValueError("at least one multi-rank record file is required")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    return {
        "output": str(destination), "record_count": len(merged),
        "workload_count": next_workload, "sources": sources,
        "config_identity": json.loads(identity),
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _routing_features(layers: Iterable[Any], ep_size: int) -> tuple[float, ...]:
    rank_ratios = []
    expert_ratios = []
    active_fractions = []
    for layer in layers:
        values = list(layer.tokens_per_expert if hasattr(layer, "tokens_per_expert")
                      else layer["tokens_per_expert"])
        if len(values) % ep_size:
            raise ValueError("experts must divide evenly across EP ranks")
        width = len(values) // ep_size
        rank_loads = [sum(values[index * width:(index + 1) * width]) for index in range(ep_size)]
        rank_mean = sum(rank_loads) / ep_size
        expert_mean = sum(values) / len(values)
        rank_ratios.append(max(rank_loads) / rank_mean if rank_mean else 0.0)
        expert_ratios.append(max(values) / expert_mean if expert_mean else 0.0)
        active_fractions.append(sum(value > 0 for value in values) / len(values))
    return (
        statistics.mean(rank_ratios), _p95(rank_ratios), max(rank_ratios),
        statistics.mean(expert_ratios), statistics.mean(active_fractions),
    )


def features_from_raw(raw: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    requests = raw["requests"]
    scheduled = [int(row["scheduled_tokens"]) for row in requests]
    contexts = [int(row["context_length"]) for row in requests]
    prefill = [row for row in requests if row["phase"] == "prefill"]
    prefill_tokens = sum(int(row["scheduled_tokens"]) for row in prefill)
    total_scheduled = sum(scheduled)
    ep_size = int(raw["profile_header"]["runtime"]["expert_parallel_size"])
    routing = _routing_features(raw["layer_records"], ep_size)
    return raw["phase"], (
        math.log1p(len(requests)), math.log1p(total_scheduled), math.log1p(max(scheduled)),
        math.log1p(sum(contexts)), math.log1p(sum(contexts) / len(contexts)),
        math.log1p(max(contexts)), len(prefill) / len(requests),
        prefill_tokens / total_scheduled if total_scheduled else 0.0, *routing,
    )


def features_from_plan(
    plan: StepExecutionPlan, *, prefill_requests: int, decode_requests: int,
    prefill_scheduled_tokens: int, expert_parallel_size: int = 2,
) -> tuple[str, tuple[float, ...]]:
    total_requests = prefill_requests + decode_requests
    if total_requests != plan.realized_batch_size:
        raise ValueError("request composition does not match plan batch")
    routing = _routing_features(plan.layers, expert_parallel_size)
    return plan.phase, (
        math.log1p(plan.realized_batch_size), math.log1p(plan.scheduled_tokens),
        math.log1p(plan.n_q), math.log1p(plan.total_kv_tokens), math.log1p(plan.kv_mean),
        math.log1p(plan.kv_max), prefill_requests / total_requests,
        prefill_scheduled_tokens / plan.scheduled_tokens, *routing,
    )


def _config_identity(raw: dict[str, Any]) -> dict[str, Any]:
    header = raw["profile_header"]
    runtime = header["runtime"]
    return {
        "model_id": header["model"]["model_id"],
        "hardware_id": header["hardware"]["hardware_id"],
        "rank_count": raw["rank_count"],
        "tensor_parallel_size": runtime["tensor_parallel_size"],
        "expert_parallel_size": runtime["expert_parallel_size"],
        "vllm_version": runtime["vllm_version"],
        "precision": runtime["precision"],
        "cuda_graph": runtime["cuda_graph"],
        "all2all_backend": runtime.get("all2all_backend"),
        "expert_placement_strategy": runtime.get("expert_placement_strategy"),
        "kernel_identity": raw["kernel_identity"],
    }


@dataclass(frozen=True)
class FixedConfigPrediction:
    step_us: float | None
    candidate_step_us: float
    support_status: str
    needs_profile: bool
    support_reason: str
    nearest_distance: float
    coverage_threshold: float
    extrapolated: bool
    neighbors: tuple[dict[str, float], ...]
    prediction_wall_us: float
    config_identity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FixedConfigForwardOracle:
    artifact_type = ARTIFACT_TYPE

    def __init__(self, *, neighbors: int = 4, method: str = "idw"):
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        if method not in {"idw", "routing_rf"}:
            raise ValueError("fixed-config method must be idw or routing_rf")
        self.neighbor_count = neighbors
        self.method = method
        self.identity: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {}
        self.tables: dict[str, dict[str, Any]] = {}

    def fit(self, rows: list[dict[str, Any]], *, communication_provenance: dict[str, Any] | None = None) -> "FixedConfigForwardOracle":
        identities = {_stable_json(_config_identity(row)) for row in rows}
        if len(identities) != 1:
            raise ValueError("fixed-config oracle cannot mix configuration identities")
        self.identity = json.loads(next(iter(identities)))
        grouped: dict[tuple[str, tuple[float, ...]], list[float]] = {}
        workloads: dict[tuple[str, tuple[float, ...]], set[int]] = {}
        for row in rows:
            phase, features = features_from_raw(row)
            key = (phase, features)
            grouped.setdefault(key, []).append(float(row["timing_us"]["step"]))
            workloads.setdefault(key, set()).add(int(row["workload_index"]))
        for phase in sorted({key[0] for key in grouped}):
            keys = [key for key in grouped if key[0] == phase]
            matrix = np.asarray([key[1] for key in keys], dtype=np.float64)
            targets = np.asarray([statistics.median(grouped[key]) for key in keys], dtype=np.float64)
            lower = matrix.min(axis=0)
            upper = matrix.max(axis=0)
            scale = np.where(upper > lower, upper - lower, 1.0)
            table = {
                "features": matrix, "targets": targets, "lower": lower, "upper": upper,
                "scale": scale, "sample_counts": np.asarray([len(grouped[key]) for key in keys]),
            }
            if len(matrix) == 1:
                nearest_other = [0.0]
            else:
                normalized = (matrix[:, None, :] - matrix[None, :, :]) / scale
                pairwise = np.linalg.norm(normalized, axis=2) / math.sqrt(matrix.shape[1])
                np.fill_diagonal(pairwise, np.inf)
                nearest_other = np.min(pairwise, axis=1).tolist()
            table["coverage_threshold"] = float(_p95(nearest_other))
            if self.method == "routing_rf":
                from sklearn.ensemble import RandomForestRegressor
                raw_phase = [row for row in rows if row["phase"] == phase]
                raw_matrix = np.asarray([features_from_raw(row)[1] for row in raw_phase])
                raw_targets = np.asarray([float(row["timing_us"]["step"]) for row in raw_phase])
                table["regressor"] = RandomForestRegressor(
                    n_estimators=96, min_samples_leaf=2, max_features=0.8,
                    n_jobs=1, random_state=0,
                ).fit(raw_matrix, raw_targets)
            self.tables[phase] = table
        self.provenance = {
            "artifact_type": ARTIFACT_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "label_source": "cuda_event_multirank",
            "formal_hardware_prediction": True,
            "config_identity": self.identity,
            "raw_step_count": len(rows),
            "unique_anchor_count": len(grouped),
            "phase_anchor_counts": {phase: len(table["targets"]) for phase, table in self.tables.items()},
            "feature_names": FEATURE_NAMES,
            "duplicate_reduction": "median_exact_feature_signature",
            "prediction_method": self.method,
            "neighbor_index": "phase_strict_range_normalized_distance",
            "trusted_range": {
                "rule": "inside_phase_bounding_box_and_nearest_distance_le_anchor_nn_p95",
                "phase_thresholds": {
                    phase: table["coverage_threshold"] for phase, table in self.tables.items()
                },
                "unsupported_default": "not_predictable",
            },
            "communication_provenance": communication_provenance,
        }
        return self

    def predict_features(
        self, phase: str, features: tuple[float, ...], *, allow_unsupported: bool = False
    ) -> FixedConfigPrediction:
        started = time.perf_counter_ns()
        if phase not in self.tables:
            raise ValueError(f"artifact has no {phase} response family")
        table = self.tables[phase]
        query = np.asarray(features, dtype=np.float64)
        matrix = table["features"]
        distances = np.linalg.norm((matrix - query) / table["scale"], axis=1) / math.sqrt(len(features))
        order = np.argsort(distances)[:min(self.neighbor_count, len(distances))]
        if distances[order[0]] <= 1e-12:
            prediction = float(table["targets"][order[0]])
        elif self.method == "routing_rf":
            prediction = float(table["regressor"].predict(query.reshape(1, -1))[0])
        else:
            weights = 1.0 / np.square(distances[order] + 1e-6)
            prediction = float(np.dot(weights, table["targets"][order]) / weights.sum())
        extrapolated = bool(np.any(query < table["lower"] - 1e-12)
                            or np.any(query > table["upper"] + 1e-12))
        nearest_distance = float(distances[order[0]])
        coverage_threshold = float(table.get("coverage_threshold", 0.0))
        if extrapolated:
            support_reason = "outside_phase_feature_bounds"
        elif nearest_distance > coverage_threshold + 1e-12:
            support_reason = "nearest_anchor_beyond_phase_coverage_threshold"
        else:
            support_reason = "inside_trusted_range"
        supported = support_reason == "inside_trusted_range"
        elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        return FixedConfigPrediction(
            step_us=prediction if supported or allow_unsupported else None,
            candidate_step_us=prediction,
            support_status="supported" if supported else "not_predictable",
            needs_profile=not supported,
            support_reason=support_reason,
            nearest_distance=nearest_distance,
            coverage_threshold=coverage_threshold,
            extrapolated=extrapolated,
            neighbors=tuple({"distance": float(distances[index]),
                             "step_us": float(table["targets"][index])} for index in order),
            prediction_wall_us=elapsed_us,
            config_identity=self.identity,
        )

    def predict_raw(
        self, raw: dict[str, Any], *, allow_unsupported: bool = False
    ) -> FixedConfigPrediction:
        if _config_identity(raw) != self.identity:
            raise ValueError("query configuration does not match fixed-config artifact")
        phase, features = features_from_raw(raw)
        return self.predict_features(phase, features, allow_unsupported=allow_unsupported)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def save_fixed_config_artifact(model: FixedConfigForwardOracle, path: str | Path) -> None:
    import joblib
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_fixed_config_artifact(path: str | Path) -> FixedConfigForwardOracle:
    import joblib
    model = joblib.load(Path(path))
    if not isinstance(model, FixedConfigForwardOracle):
        raise TypeError("artifact is not a fixed-config forward oracle")
    return model


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_test_rows"}
    apes = sorted(row["ape"] for row in rows)
    return {
        "status": "ok", "count": len(rows),
        "mape": statistics.mean(apes), "p95_ape": _p95(apes), "max_ape": max(apes),
        "median_prediction_wall_us": statistics.median(row["prediction_wall_us"] for row in rows),
        "extrapolated_fraction": statistics.mean(float(row["extrapolated"]) for row in rows),
        "supported_fraction": statistics.mean(
            float(row.get("support_status") == "supported") for row in rows
        ),
    }


def evaluate_fixed_config(
    rows: list[dict[str, Any]], *, neighbors: int = 4, method: str = "idw"
) -> dict[str, Any]:
    predictions = []
    unsupported = []
    for workload in sorted({int(row["workload_index"]) for row in rows}):
        train = [row for row in rows if int(row["workload_index"]) != workload]
        test = [row for row in rows if int(row["workload_index"]) == workload]
        model = FixedConfigForwardOracle(neighbors=neighbors, method=method).fit(train)
        for row in test:
            try:
                prediction = model.predict_raw(row, allow_unsupported=True)
            except ValueError as error:
                unsupported.append({"workload_index": workload, "phase": row["phase"], "reason": str(error)})
                continue
            actual = float(row["timing_us"]["step"])
            predictions.append({
                "workload_index": workload, "phase": row["phase"], "actual_us": actual,
                "predicted_us": prediction.candidate_step_us,
                "ape": abs(prediction.candidate_step_us - actual) / actual * 100.0,
                "nearest_distance": prediction.nearest_distance,
                "coverage_threshold": prediction.coverage_threshold,
                "extrapolated": prediction.extrapolated,
                "support_status": prediction.support_status,
                "support_reason": prediction.support_reason,
                "prediction_wall_us": prediction.prediction_wall_us,
            })
    phase_metrics = {
        phase: _metrics([row for row in predictions if row["phase"] == phase])
        for phase in ("prefill", "decode", "mixed")
    }
    return {
        "protocol": "leave-one-requested-workload-out",
        "neighbors": neighbors, "method": method,
        "overall": _metrics(predictions),
        "supported_only": _metrics([
            row for row in predictions if row["support_status"] == "supported"
        ]),
        "by_phase": phase_metrics,
        "unsupported": unsupported,
        "predictions": predictions,
    }

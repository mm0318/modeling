from __future__ import annotations

import copy
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .budget_evaluation import select_phase_aware_farthest
from .fixed_config_oracle import (
    FEATURE_NAMES,
    FixedConfigForwardOracle,
    _config_identity,
    _p95,
    _stable_json,
    features_from_raw,
)


ARTIFACT_TYPE = "cross-config-calibration/v1"
# EP-rank load features (8, 9, 10) change semantics when EP size changes.
TRANSFER_FEATURE_INDICES = tuple(range(8)) + (11, 12)
TRANSFER_FEATURE_NAMES = tuple(FEATURE_NAMES[index] for index in TRANSFER_FEATURE_INDICES)


def _pair_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["workload_index"]), int(row["repeat"]),
        int(row["step_index"]), str(row["phase"]),
    )


def _request_shape(row: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(request["phase"]), int(request["scheduled_tokens"]),
         int(request["context_length"]))
        for request in row["requests"]
    )


def _target_signature(row: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    return features_from_raw(row)


def _transfer_features(row: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    phase, features = features_from_raw(row)
    return phase, tuple(features[index] for index in TRANSFER_FEATURE_INDICES)


def build_aligned_pairs(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_oracle: FixedConfigForwardOracle,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not source_rows or not target_rows:
        raise ValueError("source and target records must be non-empty")
    source_identity = _config_identity(source_rows[0])
    target_identity = _config_identity(target_rows[0])
    if source_identity == target_identity:
        raise ValueError("cross-config calibration requires different identities")
    if source_oracle.identity != source_identity:
        raise ValueError("source records do not match the source oracle identity")
    if any(_config_identity(row) != source_identity for row in source_rows):
        raise ValueError("source records contain multiple configuration identities")
    if any(_config_identity(row) != target_identity for row in target_rows):
        raise ValueError("target records contain multiple configuration identities")

    source = {_pair_key(row): row for row in source_rows}
    target = {_pair_key(row): row for row in target_rows}
    common = sorted(set(source) & set(target))
    aligned = [key for key in common if _request_shape(source[key]) == _request_shape(target[key])]
    pairs = []
    for key in aligned:
        source_prediction = source_oracle.predict_raw(source[key], allow_unsupported=True)
        pairs.append({
            "key": key,
            "workload_index": key[0],
            "phase": key[3],
            "source_row": source[key],
            "target_row": target[key],
            "source_us": source_prediction.candidate_step_us,
            "source_support_status": source_prediction.support_status,
        })
    return pairs, {
        "source_step_count": len(source_rows),
        "target_step_count": len(target_rows),
        "key_aligned_step_count": len(common),
        "shape_aligned_step_count": len(aligned),
        "shape_mismatch_excluded_count": len(common) - len(aligned),
        "matching_rule": (
            "workload_index/repeat/step/phase plus exact request "
            "phase/scheduled/context"
        ),
        "source_oracle_unsupported_signal_count": sum(
            pair["source_support_status"] != "supported" for pair in pairs
        ),
    }


def _representative_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[_target_signature(pair["target_row"])].append(pair)
    representatives = []
    for signature in sorted(grouped, key=lambda item: (item[0], item[1])):
        samples = grouped[signature]
        representative = copy.deepcopy(samples[0])
        target_us = statistics.median(
            float(pair["target_row"]["timing_us"]["step"]) for pair in samples
        )
        source_us = statistics.median(float(pair["source_us"]) for pair in samples)
        representative["target_row"]["timing_us"]["step"] = target_us
        representative["source_us"] = source_us
        representative["repeat_count"] = len(samples)
        representatives.append(representative)
    return representatives


@dataclass(frozen=True)
class CrossConfigPrediction:
    target_us: float | None
    candidate_target_us: float
    source_us: float
    ratio: float
    support_status: str
    needs_profile: bool
    support_reason: str
    nearest_distance: float
    coverage_threshold: float
    extrapolated: bool
    prediction_wall_us: float
    source_identity: dict[str, Any]
    target_identity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossConfigRatioCalibrator:
    artifact_type = ARTIFACT_TYPE

    def __init__(self, *, method: str = "local_ratio_idw", neighbors: int = 4):
        if method not in {"phase_scale", "local_ratio_idw"}:
            raise ValueError("method must be phase_scale or local_ratio_idw")
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.method = method
        self.neighbor_count = neighbors
        self.source_identity: dict[str, Any] = {}
        self.target_identity: dict[str, Any] = {}
        self.tables: dict[str, dict[str, Any]] = {}
        self.provenance: dict[str, Any] = {}

    def fit(self, paired_anchors: list[dict[str, Any]]) -> "CrossConfigRatioCalibrator":
        if not paired_anchors:
            raise ValueError("at least one paired target anchor is required")
        source_identities = {
            _stable_json(_config_identity(pair["source_row"])) for pair in paired_anchors
        }
        target_identities = {
            _stable_json(_config_identity(pair["target_row"])) for pair in paired_anchors
        }
        if len(source_identities) != 1 or len(target_identities) != 1:
            raise ValueError("calibration anchors must each have one source and target identity")
        self.source_identity = json.loads(next(iter(source_identities)))
        self.target_identity = json.loads(next(iter(target_identities)))
        if self.source_identity == self.target_identity:
            raise ValueError("source and target identities must differ")

        grouped: dict[tuple[str, tuple[float, ...]], list[float]] = defaultdict(list)
        for pair in paired_anchors:
            phase, features = _transfer_features(pair["target_row"])
            source_us = float(pair["source_us"])
            if source_us <= 0:
                raise ValueError("source latency must be positive")
            grouped[(phase, features)].append(
                float(pair["target_row"]["timing_us"]["step"]) / source_us
            )
        for phase in sorted({key[0] for key in grouped}):
            keys = [key for key in grouped if key[0] == phase]
            matrix = np.asarray([key[1] for key in keys], dtype=np.float64)
            ratios = np.asarray([statistics.median(grouped[key]) for key in keys])
            lower, upper = matrix.min(axis=0), matrix.max(axis=0)
            scale = np.where(upper > lower, upper - lower, 1.0)
            if len(matrix) == 1:
                nearest_other = [0.0]
            else:
                pairwise = np.linalg.norm(
                    (matrix[:, None, :] - matrix[None, :, :]) / scale, axis=2
                ) / math.sqrt(matrix.shape[1])
                np.fill_diagonal(pairwise, np.inf)
                nearest_other = np.min(pairwise, axis=1).tolist()
            self.tables[phase] = {
                "features": matrix,
                "ratios": ratios,
                "lower": lower,
                "upper": upper,
                "scale": scale,
                "phase_median_ratio": float(statistics.median(ratios.tolist())),
                "coverage_threshold": float(_p95(nearest_other)),
            }
        self.provenance = {
            "artifact_type": ARTIFACT_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": self.method,
            "neighbors": self.neighbor_count,
            "source_identity": self.source_identity,
            "target_identity": self.target_identity,
            "paired_target_measurement_count": len(paired_anchors),
            "unique_target_anchor_count": len(grouped),
            "phase_anchor_counts": {
                phase: len(table["ratios"]) for phase, table in self.tables.items()
            },
            "transfer_feature_names": TRANSFER_FEATURE_NAMES,
            "excluded_feature_names": tuple(
                FEATURE_NAMES[index] for index in (8, 9, 10)
            ),
            "source_signal_contract": (
                "caller supplies latency from the source fixed-config oracle for an "
                "exactly shape-aligned source scheduler/routing step"
            ),
            "target_label_model": "phase-strict target/source ratio",
            "trusted_range": (
                "inside target calibration phase bounds and nearest distance <= anchor NN p95"
            ),
        }
        return self

    def predict(
        self, source_us: float, target_raw: dict[str, Any], *,
        allow_unsupported: bool = False,
    ) -> CrossConfigPrediction:
        started = time.perf_counter_ns()
        if _config_identity(target_raw) != self.target_identity:
            raise ValueError("query does not match target calibration identity")
        if source_us <= 0:
            raise ValueError("source latency must be positive")
        phase, features = _transfer_features(target_raw)
        if phase not in self.tables:
            raise ValueError(f"calibrator has no {phase} ratio family")
        table = self.tables[phase]
        query = np.asarray(features, dtype=np.float64)
        distances = np.linalg.norm((table["features"] - query) / table["scale"], axis=1)
        distances = distances / math.sqrt(len(features))
        order = np.argsort(distances)[:min(self.neighbor_count, len(distances))]
        phase_ratio = float(table["phase_median_ratio"])
        if self.method == "phase_scale" or len(order) == 0:
            ratio = phase_ratio
        elif distances[order[0]] <= 1e-12:
            ratio = float(table["ratios"][order[0]])
        else:
            weights = 1.0 / np.square(distances[order] + 1e-6)
            local_ratio = float(np.dot(weights, table["ratios"][order]) / weights.sum())
            threshold = float(table["coverage_threshold"])
            reliability = 0.0 if threshold <= 0 else 1.0 / (
                1.0 + (float(distances[order[0]]) / threshold) ** 2
            )
            ratio = reliability * local_ratio + (1.0 - reliability) * phase_ratio
        extrapolated = bool(
            np.any(query < table["lower"] - 1e-12)
            or np.any(query > table["upper"] + 1e-12)
        )
        nearest = float(distances[order[0]])
        threshold = float(table["coverage_threshold"])
        if extrapolated:
            reason = "outside_target_calibration_bounds"
        elif nearest > threshold + 1e-12:
            reason = "nearest_calibration_anchor_beyond_threshold"
        else:
            reason = "inside_target_calibration_domain"
        supported = reason == "inside_target_calibration_domain"
        candidate = float(source_us) * ratio
        elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        return CrossConfigPrediction(
            target_us=candidate if supported or allow_unsupported else None,
            candidate_target_us=candidate,
            source_us=float(source_us), ratio=ratio,
            support_status="supported" if supported else "not_predictable",
            needs_profile=not supported, support_reason=reason,
            nearest_distance=nearest, coverage_threshold=threshold,
            extrapolated=extrapolated, prediction_wall_us=elapsed_us,
            source_identity=self.source_identity, target_identity=self.target_identity,
        )


def save_cross_config_artifact(model: CrossConfigRatioCalibrator, path: str | Path) -> None:
    import joblib
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_cross_config_artifact(path: str | Path) -> CrossConfigRatioCalibrator:
    import joblib
    model = joblib.load(Path(path))
    if not isinstance(model, CrossConfigRatioCalibrator):
        raise TypeError("artifact is not a cross-config calibration model")
    return model


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_rows"}
    apes = [float(row["ape"]) for row in rows]
    supported = [row for row in rows if row.get("support_status") == "supported"]
    result = {
        "status": "ok", "count": len(rows), "mape": statistics.mean(apes),
        "p95_ape": _p95(apes), "max_ape": max(apes),
        "median_prediction_wall_us": statistics.median(
            float(row["prediction_wall_us"]) for row in rows
        ),
        "supported_fraction": statistics.mean(
            row.get("support_status") == "supported" for row in rows
        ),
    }
    if supported:
        supported_apes = [float(row["ape"]) for row in supported]
        result["supported_only"] = {
            "count": len(supported), "mape": statistics.mean(supported_apes),
            "p95_ape": _p95(supported_apes), "max_ape": max(supported_apes),
        }
    else:
        result["supported_only"] = {"status": "no_supported_rows"}
    return result


def _evaluate_model(
    name: str, test_pairs: list[dict[str, Any]], predictor: Any,
) -> dict[str, Any]:
    predictions = []
    for pair in test_pairs:
        started = time.perf_counter_ns()
        result = predictor(pair)
        wall_us = (time.perf_counter_ns() - started) / 1000.0
        actual = float(pair["target_row"]["timing_us"]["step"])
        predicted = float(result["prediction"])
        predictions.append({
            "workload_index": int(pair["workload_index"]), "phase": pair["phase"],
            "actual_us": actual, "predicted_us": predicted,
            "ape": abs(predicted - actual) / actual * 100.0,
            "support_status": result.get("support_status", "supported"),
            "prediction_wall_us": result.get("prediction_wall_us", wall_us),
        })
    return {
        "model": name,
        "overall": _metrics(predictions),
        "by_phase": {
            phase: _metrics([row for row in predictions if row["phase"] == phase])
            for phase in ("prefill", "decode", "mixed")
        },
    }


def train_cross_config_calibrator(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
    source_oracle: FixedConfigForwardOracle, *, budget: int,
    method: str = "local_ratio_idw", neighbors: int = 4,
) -> tuple[CrossConfigRatioCalibrator, dict[str, Any]]:
    pairs, diagnostics = build_aligned_pairs(source_rows, target_rows, source_oracle)
    representatives = _representative_pairs(pairs)
    if budget > len(representatives):
        raise ValueError("budget exceeds shape-aligned unique target anchors")
    sequence = select_phase_aware_farthest(
        [pair["target_row"] for pair in representatives], budget
    )
    selected = set(sequence)
    anchors = [
        pair for pair in pairs
        if _target_signature(pair["target_row"]) in selected
    ]
    model = CrossConfigRatioCalibrator(method=method, neighbors=neighbors).fit(anchors)
    model.provenance.update({
        "selection_strategy": "phase_aware_farthest",
        "uses_unprofiled_target_latency_labels": False,
        "selected_target_signatures": [
            {"phase": phase, "features": list(features)} for phase, features in sequence
        ],
        "pairing_diagnostics": diagnostics,
    })
    return model, {
        "budget": budget, "unique_paired_candidate_count": len(representatives),
        "selected_phase_counts": {
            phase: sum(signature[0] == phase for signature in sequence)
            for phase in ("prefill", "decode", "mixed")
        },
        "pairing_diagnostics": diagnostics,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "no_values"}
    return {
        "status": "ok",
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values), "max": max(values),
    }


def evaluate_cross_config_calibration(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
    source_oracle: FixedConfigForwardOracle, *, budgets: list[int], folds: int = 5,
    neighbors: int = 4, target_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    budgets = sorted(set(int(value) for value in budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must be positive")
    raw_pairs, pairing = build_aligned_pairs(source_rows, target_rows, source_oracle)
    workload_ids = sorted({int(pair["workload_index"]) for pair in raw_pairs})
    fold_reports = []
    for fold in range(folds):
        test_workloads = [value for index, value in enumerate(workload_ids) if index % folds == fold]
        test_set = set(test_workloads)
        train_raw_pairs = [
            pair for pair in raw_pairs if int(pair["workload_index"]) not in test_set
        ]
        train_pairs = _representative_pairs(train_raw_pairs)
        test_pairs = _representative_pairs([
            pair for pair in raw_pairs if int(pair["workload_index"]) in test_set
        ])
        max_budget = max(budgets)
        if max_budget > len(train_pairs):
            raise ValueError(f"budget {max_budget} exceeds fold {fold} train candidates")
        sequence = select_phase_aware_farthest(
            [pair["target_row"] for pair in train_pairs], max_budget
        )
        budget_reports = []
        for budget in budgets:
            selected_sequence = sequence[:budget]
            selected = set(selected_sequence)
            anchors = [
                pair for pair in train_raw_pairs
                if _target_signature(pair["target_row"]) in selected
            ]
            phase_scale = CrossConfigRatioCalibrator(
                method="phase_scale", neighbors=neighbors
            ).fit(anchors)
            local_ratio = CrossConfigRatioCalibrator(
                method="local_ratio_idw", neighbors=neighbors
            ).fit(anchors)
            target_only_rows = [
                row for row in target_rows
                if int(row["workload_index"]) not in test_set
                and _target_signature(row) in selected
            ]
            target_only = FixedConfigForwardOracle(
                method="routing_rf", neighbors=neighbors
            ).fit(target_only_rows)

            models = []
            models.append(_evaluate_model(
                "zero_shot_source", test_pairs,
                lambda pair: {"prediction": pair["source_us"]},
            ))
            for name, calibrator in (
                ("phase_scale", phase_scale), ("local_ratio_idw", local_ratio)
            ):
                models.append(_evaluate_model(
                    name, test_pairs,
                    lambda pair, model=calibrator: _calibration_result(model, pair),
                ))
            models.append(_evaluate_model(
                "target_only_same_budget", test_pairs,
                lambda pair: _target_only_result(target_only, pair),
            ))
            budget_reports.append({
                "budget": budget,
                "selected_phase_counts": {
                    phase: sum(signature[0] == phase for signature in selected_sequence)
                    for phase in ("prefill", "decode", "mixed")
                },
                "models": models,
            })
        fold_reports.append({
            "fold": fold, "train_workload_ids": sorted(set(workload_ids) - test_set),
            "test_workload_ids": test_workloads,
            "train_unique_paired_candidates": len(train_pairs),
            "test_unique_paired_signatures": len(test_pairs),
            "budgets": budget_reports,
        })

    aggregate = []
    for budget in budgets:
        model_names = [row["model"] for row in fold_reports[0]["budgets"][budgets.index(budget)]["models"]]
        models = []
        for name in model_names:
            entries = []
            for fold_report in fold_reports:
                budget_row = next(row for row in fold_report["budgets"] if row["budget"] == budget)
                entries.append(next(row for row in budget_row["models"] if row["model"] == name))
            models.append({
                "model": name,
                "mape": _distribution([row["overall"]["mape"] for row in entries]),
                "p95_ape": _distribution([row["overall"]["p95_ape"] for row in entries]),
                "supported_fraction": _distribution([
                    row["overall"]["supported_fraction"] for row in entries
                ]),
                "phase_mape": {
                    phase: _distribution([
                        row["by_phase"][phase]["mape"] for row in entries
                        if row["by_phase"][phase].get("status") == "ok"
                    ])
                    for phase in ("prefill", "decode", "mixed")
                },
                "phase_p95_ape": {
                    phase: _distribution([
                        row["by_phase"][phase]["p95_ape"] for row in entries
                        if row["by_phase"][phase].get("status") == "ok"
                    ])
                    for phase in ("prefill", "decode", "mixed")
                },
            })
        aggregate.append({"budget": budget, "models": models})
    qualifying = []
    for budget_row in aggregate:
        local = next(
            row for row in budget_row["models"] if row["model"] == "local_ratio_idw"
        )
        if (local["mape"]["mean"] <= 3.0
                and local["p95_ape"]["mean"] <= 15.0
                and local["supported_fraction"]["mean"] >= 0.80):
            qualifying.append(budget_row["budget"])
    return {
        "schema_version": "cross-config-calibration-evaluation/v1",
        "protocol": "fixed_requested_workload_5fold_shape_aligned_transfer",
        "source_identity": source_oracle.identity,
        "target_identity": _config_identity(target_rows[0]),
        "budgets": budgets, "fold_count": folds, "pairing": pairing,
        "selection": {
            "strategy": "phase_aware_farthest_nested_prefix",
            "uses_unprofiled_target_latency_labels": False,
            "target_test_features_visible_to_acquisition": False,
            "target_test_labels_visible": False,
            "source_test_signal_available": True,
        },
        "source_signal_boundary": (
            "evaluation queries the prebuilt source oracle for the shape-aligned source "
            "step, including held-out target workloads"
        ),
        "transfer_feature_names": TRANSFER_FEATURE_NAMES,
        "folds": fold_reports, "aggregate": aggregate,
        "recommendation": {
            "status": "recommended" if qualifying else "no_budget_meets_targets",
            "method": "local_ratio_idw",
            "minimum_target_anchor_budget": min(qualifying) if qualifying else None,
            "targets": {"mape": 3.0, "p95_ape": 15.0, "supported_fraction": 0.80},
        },
        "target_only_64_anchor_reference": target_reference or {"status": "not_provided"},
        "reference_comparability": (
            "external 64-anchor aggregate is contextual only; it may use a broader, "
            "non-paired target test surface and is not a paired delta"
        ),
    }


def _calibration_result(
    model: CrossConfigRatioCalibrator, pair: dict[str, Any]
) -> dict[str, Any]:
    prediction = model.predict(
        float(pair["source_us"]), pair["target_row"], allow_unsupported=True
    )
    return {
        "prediction": prediction.candidate_target_us,
        "support_status": prediction.support_status,
        "prediction_wall_us": prediction.prediction_wall_us,
    }


def _target_only_result(
    model: FixedConfigForwardOracle, pair: dict[str, Any]
) -> dict[str, Any]:
    prediction = model.predict_raw(pair["target_row"], allow_unsupported=True)
    return {
        "prediction": prediction.candidate_step_us,
        "support_status": prediction.support_status,
        "prediction_wall_us": prediction.prediction_wall_us,
    }

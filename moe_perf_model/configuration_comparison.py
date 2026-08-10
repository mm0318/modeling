from __future__ import annotations

import statistics
from typing import Any

from .fixed_config_oracle import _config_identity, _p95, load_multirank_records


def _key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["workload_index"]), int(row["repeat"]),
        int(row["step_index"]), str(row["phase"]),
    )


def _request_shape(row: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(request["phase"]), int(request["scheduled_tokens"]), int(request["context_length"]))
        for request in row["requests"]
    )


def _ratio_metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "no_shape_aligned_steps"}
    return {
        "status": "ok", "count": len(values),
        "median_target_over_source": statistics.median(values),
        "mean_target_over_source": statistics.mean(values),
        "p95_target_over_source": _p95(values),
        "target_faster_fraction": statistics.mean(value < 1.0 for value in values),
    }


def compare_fixed_config_records(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    source_identity = _config_identity(source_rows[0])
    target_identity = _config_identity(target_rows[0])
    if source_identity == target_identity:
        raise ValueError("configuration comparison requires different identities")
    source = {_key(row): row for row in source_rows}
    target = {_key(row): row for row in target_rows}
    common = sorted(set(source) & set(target))
    aligned = [
        key for key in common
        if _request_shape(source[key]) == _request_shape(target[key])
    ]
    ratios = {
        key: float(target[key]["timing_us"]["step"]) / float(source[key]["timing_us"]["step"])
        for key in aligned
    }
    return {
        "schema_version": "fixed-config-paired-comparison/v1",
        "source_identity": source_identity, "target_identity": target_identity,
        "source_step_count": len(source_rows), "target_step_count": len(target_rows),
        "key_aligned_step_count": len(common),
        "shape_aligned_step_count": len(aligned),
        "shape_mismatch_excluded_count": len(common) - len(aligned),
        "matching_rule": "workload_index/repeat/step/phase plus exact request phase/scheduled/context",
        "interpretation": "paired measured diagnostic; not a cross-config predictor",
        "overall": _ratio_metrics(list(ratios.values())),
        "by_phase": {
            phase: _ratio_metrics([value for key, value in ratios.items() if key[3] == phase])
            for phase in ("prefill", "decode", "mixed")
        },
    }


def compare_fixed_config_files(source: str, target: str) -> dict[str, Any]:
    return compare_fixed_config_records(
        load_multirank_records(source), load_multirank_records(target)
    )

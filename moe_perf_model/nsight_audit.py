from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _intersection_duration(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    a, b = _merge(left), _merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _load_raw_composition(raw_steps: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(raw_steps).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("raw step audit input is empty")
    overrun = [float(row.get("composition_overrun_us", 0.0)) for row in rows]
    uncovered = [float(row.get("uncovered_us", 0.0)) for row in rows]
    step = [float(row["timing_us"]["step"]) for row in rows]
    fixed_sources = sorted({str(row.get("fixed_source")) for row in rows})
    prompt_patterns = sorted({str(row.get("prompt_pattern", "unspecified")) for row in rows})
    return {
        "steps": len(rows),
        "phases": sorted({str(row["phase"]) for row in rows}),
        "fixed_sources": fixed_sources,
        "prompt_patterns": prompt_patterns,
        "max_composition_overrun_us": max(overrun),
        "max_uncovered_us": max(uncovered),
        "max_absolute_composition_delta_ratio": max(
            max(overrun[index], uncovered[index]) / step[index] for index in range(len(rows))
        ),
    }


def audit_nsight_sqlite(
    sqlite_path: str | Path,
    raw_steps: str | Path,
    *,
    overlap_ratio_threshold: float = 0.01,
    composition_delta_threshold: float = 0.001,
) -> dict[str, Any]:
    """Audit WB/STEP NVTX ranges and CUDA kernel stream overlap.

    NVTX and CUDA activity timestamps in an exported Nsight Systems SQLite file
    share the same nanosecond clock. Kernel intervals are clipped to each WB step.
    """
    connection = sqlite3.connect(str(sqlite_path))
    try:
        tables = {
            row[0]
            for row in connection.execute("select name from sqlite_master where type='table'")
        }
        required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if not required.issubset(tables):
            raise ValueError(f"Nsight SQLite misses tables: {sorted(required - tables)}")
        steps = list(
            connection.execute(
                "select start, end, text from NVTX_EVENTS "
                "where text like 'WB/STEP/%' and end is not null order by start"
            )
        )
        kernels = list(
            connection.execute(
                "select start, end, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
            )
        )
    finally:
        connection.close()
    if not steps or not kernels:
        raise ValueError("Nsight report contains no WB step ranges or CUDA kernels")

    phase_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_streams: set[int] = set()
    for step_start, step_end, text in steps:
        phase = str(text).rsplit("/", 1)[-1]
        per_stream: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for start, end, stream_id in kernels:
            if end <= step_start:
                continue
            if start >= step_end:
                break
            clipped = (max(start, step_start), min(end, step_end))
            if clipped[1] > clipped[0]:
                per_stream[int(stream_id)].append(clipped)
        if not per_stream:
            continue
        all_streams.update(per_stream)
        busy_ns = sum(_duration(intervals) for intervals in per_stream.values())
        overlap_ns = 0
        stream_ids = sorted(per_stream)
        for index, left in enumerate(stream_ids):
            for right in stream_ids[index + 1 :]:
                overlap_ns += _intersection_duration(per_stream[left], per_stream[right])
        phase_rows[phase].append(
            {
                "stream_ids": stream_ids,
                "kernel_count": sum(len(value) for value in per_stream.values()),
                "gpu_busy_ns": busy_ns,
                "cross_stream_overlap_ns": overlap_ns,
                "cross_stream_overlap_ratio": overlap_ns / busy_ns if busy_ns else 0.0,
            }
        )

    phase_summary = {}
    max_overlap_ratio = 0.0
    for phase, rows in sorted(phase_rows.items()):
        phase_max = max(row["cross_stream_overlap_ratio"] for row in rows)
        max_overlap_ratio = max(max_overlap_ratio, phase_max)
        phase_summary[phase] = {
            "captured_steps": len(rows),
            "stream_ids": sorted({stream for row in rows for stream in row["stream_ids"]}),
            "kernel_count": sum(row["kernel_count"] for row in rows),
            "max_cross_stream_overlap_ratio": phase_max,
        }

    composition = _load_raw_composition(raw_steps)
    fixed_valid = composition["fixed_sources"] == ["direct_non_layer_dense_cuda_events"]
    serial_validated = (
        max_overlap_ratio <= overlap_ratio_threshold
        and composition["max_absolute_composition_delta_ratio"] <= composition_delta_threshold
        and fixed_valid
    )
    return {
        "audit_type": "nsight-whitebox-overlap-v1",
        "sqlite_path": str(sqlite_path),
        "raw_steps_path": str(raw_steps),
        "kernel_stream_ids": sorted(all_streams),
        "phase_summary": phase_summary,
        "max_cross_stream_overlap_ratio": max_overlap_ratio,
        "composition": composition,
        "thresholds": {
            "cross_stream_overlap_ratio": overlap_ratio_threshold,
            "composition_delta_ratio": composition_delta_threshold,
        },
        "fixed_semantics_valid": fixed_valid,
        "serial_validated": serial_validated,
        "validated_execution_topology": (
            "single_stream_serial" if serial_validated else "overlap_requires_resource_dag"
        ),
        "recommended_composition_mode": "event_dag",
    }

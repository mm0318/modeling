from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TRACE_NAME = re.compile(
    r"rank(?P<rank>\d+)_w(?P<workload>\d+)_r(?P<repeat>\d+)_s(?P<step>\d+)_(?P<phase>\w+)\.json$"
)
COMMUNICATION_KERNEL_MARKERS = (
    "ncclDevKernel",
    "vllm::cross_device_reduce",
    "two_shot_all_reduce_kernel",
)


def _union_us(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _intersection_us(
    left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]
) -> float:
    a = sorted(left)
    b = sorted(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _component(name: str, model_family: str) -> str | None:
    if not name.startswith("WB/"):
        return None
    stage = name.split("/", 2)[1]
    if stage == "attention":
        return "attention"
    if model_family == "dense" and stage == "mlp":
        return "expert"
    if model_family == "moe" and stage in {
        "router", "dispatch", "expert_gemm", "combine"
    }:
        return "expert"
    return "other"


def analyze_trace(path: Path, model_family: str) -> dict[str, Any]:
    match = TRACE_NAME.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized trace filename: {path.name}")
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    cpu_by_external: dict[int, float] = {}
    ranges: list[tuple[float, float, str]] = []
    payload_ranges: list[tuple[float, float]] = []
    synchronize_ends: list[float] = []
    for event in events:
        args = event.get("args", {})
        external = args.get("External id")
        if event.get("cat") == "cpu_op" and external is not None:
            cpu_by_external.setdefault(int(external), float(event["ts"]))
        if event.get("cat") == "cuda_runtime" and event.get("name") == "cudaDeviceSynchronize":
            synchronize_ends.append(
                float(event["ts"]) + float(event.get("dur", 0.0))
            )
        if event.get("cat") == "user_annotation":
            if event.get("name") == "WB/profile_payload":
                payload_start = float(event["ts"])
                payload_ranges.append(
                    (payload_start, payload_start + float(event.get("dur", 0.0)))
                )
            component = _component(event.get("name", ""), model_family)
            if component is not None:
                start = float(event["ts"])
                ranges.append((start, start + float(event.get("dur", 0.0)), component))

    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    work_us: dict[str, float] = defaultdict(float)
    all_intervals: list[tuple[float, float]] = []
    kernel_count = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        kernel_count += 1
        cpu_ts = cpu_by_external.get(int(event.get("args", {}).get("External id", -1)))
        if payload_ranges and not (
            cpu_ts is not None
            and any(left <= cpu_ts <= right for left, right in payload_ranges)
        ):
            continue
        start = float(event["ts"])
        if not payload_ranges and synchronize_ends and start <= min(synchronize_ends):
            continue
        end = start + float(event.get("dur", 0.0))
        name = event.get("name", "")
        if any(marker in name for marker in COMMUNICATION_KERNEL_MARKERS):
            component = "communication"
        else:
            candidates = [
                (right - left, category)
                for left, right, category in ranges
                if cpu_ts is not None and left <= cpu_ts <= right
            ]
            component = min(candidates)[1] if candidates else "other"
        intervals[component].append((start, end))
        work_us[component] += end - start
        all_intervals.append((start, end))

    if not all_intervals:
        raise ValueError(f"trace contains no CUDA kernels: {path}")
    start = min(row[0] for row in all_intervals)
    end = max(row[1] for row in all_intervals)
    span_us = end - start
    communication_union = _union_us(intervals["communication"])
    compute_intervals = [
        interval
        for component, rows in intervals.items()
        if component != "communication"
        for interval in rows
    ]
    communication_overlap = _intersection_us(
        intervals["communication"], compute_intervals
    )
    labels = {key: work_us.get(key, 0.0) for key in ("attention", "expert", "communication")}
    dominant = max(labels, key=labels.get)
    return {
        **{key: int(value) if key != "phase" else value for key, value in match.groupdict().items()},
        "model_family": model_family,
        "path": str(path),
        "kernel_count": kernel_count,
        "gpu_span_us": span_us,
        "gpu_busy_union_us": _union_us(all_intervals),
        "work_us": dict(work_us),
        "communication_kernel_us": [
            end - start for start, end in intervals["communication"]
        ],
        "work_share": {key: value / sum(work_us.values()) for key, value in work_us.items()},
        "communication_union_us": communication_union,
        "communication_overlap_us": communication_overlap,
        "exposed_communication_us": max(0.0, communication_union - communication_overlap),
        "dominant": dominant,
    }


def analyze_directory(trace_dir: Path, model_family: str) -> list[dict[str, Any]]:
    return [
        analyze_trace(path, model_family)
        for path in sorted(trace_dir.glob("rank*_w*_r*_s*_*.json"))
    ]


def _critical_rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["repeat"], row["step"], row["phase"])].append(row)
    return [max(group, key=lambda row: row["gpu_span_us"]) for group in groups.values()]


def _rank_min_service_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["repeat"], row["step"], row["phase"])].append(row)
    result = []
    for key, ranks in groups.items():
        sequences = [row["communication_kernel_us"] for row in ranks]
        count = min(map(len, sequences), default=0)
        result.append(
            {
                "workload": key[0],
                "repeat": key[1],
                "step": key[2],
                "phase": key[3],
                "communication_service_lower_bound_us": sum(
                    min(sequence[index] for sequence in sequences)
                    for index in range(count)
                ),
                "communication_kernel_count": count,
            }
        )
    return result


def summarize(
    rows: list[dict[str, Any]],
    workloads: list[dict[str, Any]],
    real_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _critical_rank_rows(rows):
        groups[(row["workload"], row["step"], row["phase"])].append(row)
    service_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _rank_min_service_rows(rows):
        service_groups[(row["workload"], row["step"], row["phase"])].append(row)
    real_groups: dict[tuple[int, int, int, str], list[float]] = defaultdict(list)
    for row in real_steps:
        requested = row["requested_workload"]
        real_groups[
            (
                requested["batch_size"],
                requested["prompt_tokens"],
                row["step_index"],
                row["phase"],
            )
        ].append(float(row["timing_us"]["step"]))
    result = []
    for (workload, step, phase), group in sorted(groups.items()):
        components = ("attention", "expert", "communication", "other")
        work = {
            key: median(row["work_us"].get(key, 0.0) for row in group)
            for key in components
        }
        denominator = sum(work.values()) or 1.0
        dominant_values = {key: work[key] for key in components[:3]}
        service = median(
            row["communication_service_lower_bound_us"]
            for row in service_groups[(workload, step, phase)]
        )
        requested = workloads[workload]
        real_step = median(
            real_groups[
                (requested["batch_size"], requested["prompt_tokens"], step, phase)
            ]
        )
        result.append(
            {
                "workload_index": workload,
                "requested_workload": workloads[workload],
                "step_index": step,
                "phase": phase,
                "observations": len(group),
                "gpu_span_us": median(row["gpu_span_us"] for row in group),
                "real_step_us": real_step,
                "gpu_busy_union_us": median(row["gpu_busy_union_us"] for row in group),
                "work_us": work,
                "work_share": {key: work[key] / denominator for key in components},
                "exposed_communication_us": median(
                    row["exposed_communication_us"] for row in group
                ),
                "communication_service_lower_bound_us": service,
                "communication_service_lower_bound_share": service / real_step,
                "communication_overlap_us": median(
                    row["communication_overlap_us"] for row in group
                ),
                "dominant": max(dominant_values, key=dominant_values.get),
            }
        )
    return result


def _markdown(models: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Multi-GPU Kernel-Trace Bottleneck Audit",
        "",
        "Communication includes NCCL and vLLM custom all-reduce kernels. Rank-min communication sums the shorter duration for each matched collective across ranks, reducing profiler-induced wait skew. It is a service-time lower bound, not additive wall-time or exposed critical-path attribution.",
        "",
    ]
    for model, rows in models.items():
        lines.extend(
            [
                f"## {model}",
                "",
                "| Workload | Phase | Real step ms | Rank-min comm ms | Comm lower-bound share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            workload = row["requested_workload"]
            label = f"B{workload['batch_size']}/P{workload['prompt_tokens']}"
            lines.append(
                f"| {label} | {row['phase']} | {row['real_step_us'] / 1000:.2f} | "
                f"{row['communication_service_lower_bound_us'] / 1000:.2f} | "
                f"{row['communication_service_lower_bound_share']:.1%} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--moe-dir", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--moe-config", type=Path, required=True)
    parser.add_argument("--dense-steps", type=Path, required=True)
    parser.add_argument("--moe-steps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configs = {
        "Dense Qwen3-8B TP2": json.loads(args.dense_config.read_text(encoding="utf-8")),
        "MoE Qwen3-30B-A3B TP2/EP2": json.loads(args.moe_config.read_text(encoding="utf-8")),
    }
    directories = {
        "Dense Qwen3-8B TP2": (args.dense_dir, "dense"),
        "MoE Qwen3-30B-A3B TP2/EP2": (args.moe_dir, "moe"),
    }
    step_paths = {
        "Dense Qwen3-8B TP2": args.dense_steps,
        "MoE Qwen3-30B-A3B TP2/EP2": args.moe_steps,
    }
    raw = {}
    summaries = {}
    for name, (directory, family) in directories.items():
        raw[name] = analyze_directory(directory, family)
        real_steps = [
            json.loads(line)
            for line in step_paths[name].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summaries[name] = summarize(raw[name], configs[name]["workloads"], real_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"raw_traces": raw, "summaries": summaries}
    (args.output_dir / "kernel_trace_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "KERNEL_TRACE_AUDIT.md").write_text(
        _markdown(summaries), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

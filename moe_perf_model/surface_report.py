from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .whitebox_model import load_fine_raw_records


def _percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def summarize_surface(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for raw in raw_records:
        components = {
            key: 0.0
            for key in ("attention", "router", "dispatch", "expert_gemm", "combine", "dense", "transition")
        }
        imbalance = []
        active = []
        for layer in raw["layer_records"]:
            for key in components:
                components[key] += float(layer["timing_us"][key])
            histogram = [int(value) for value in layer["tokens_per_expert"]]
            routed = sum(histogram)
            average = routed / len(histogram) if histogram else 0.0
            active.append(sum(value > 0 for value in histogram))
            imbalance.append(
                {
                    "max_to_mean": max(histogram) / average if average else 0.0,
                    "p95_to_mean": _percentile(histogram, 0.95) / average if average else 0.0,
                }
            )
        components["fixed"] = float(raw["fixed_us"])
        total = sum(components.values())
        grouped = {
            "attention": components["attention"],
            "expert": sum(components[key] for key in ("router", "dispatch", "expert_gemm", "combine")),
            "dense": components["dense"],
            "other": components["transition"] + components["fixed"],
        }
        requested = raw["requested_workload"]
        rows.append(
            {
                "workload_index": raw["workload_index"],
                "phase": raw["phase"],
                "requested_batch": requested["batch_size"],
                "requested_prompt_tokens": requested["prompt_tokens"],
                "realized_batch": len(raw["requests"]),
                "scheduled_tokens": sum(int(item["scheduled_tokens"]) for item in raw["requests"]),
                "max_context_length": max(int(item["context_length"]) for item in raw["requests"]),
                "step_ms": float(raw["timing_us"]["step"]) / 1000.0,
                "shares": {key: value / total for key, value in grouped.items()},
                "primary_bottleneck": max(grouped, key=grouped.get),
                "routing": {
                    "mean_active_experts": mean(active),
                    "mean_max_to_mean": mean(item["max_to_mean"] for item in imbalance),
                    "mean_p95_to_mean": mean(item["p95_to_mean"] for item in imbalance),
                },
            }
        )
    return {
        "semantics": {
            "shares": "direct CUDA Event components divided by explicit serial composition",
            "expert": "router + dispatch + expert_gemm + combine",
            "other": "transition + fixed direct non-layer CUDA Events",
            "imbalance": "per-layer max/p95 expert load divided by mean load over all experts, then averaged over layers",
        },
        "rows": rows,
    }


def render_surface_markdown(report: dict[str, Any]) -> str:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in report["rows"]:
        grouped[int(row["requested_batch"])].append(row)
    lines = ["# H20 MoE Bottleneck Surface", ""]
    for batch, rows in sorted(grouped.items()):
        lines.extend(
            [
                f"## Requested Batch {batch}",
                "",
                "| Phase | Prompt/KV anchor | Realized batch | Scheduled | Step ms | Attention | Expert | Dense | Other | Active experts | Max/mean | p95/mean | Bottleneck |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(rows, key=lambda item: (item["requested_prompt_tokens"], item["phase"])):
            shares = row["shares"]
            routing = row["routing"]
            lines.append(
                f"| {row['phase']} | {row['requested_prompt_tokens']} | {row['realized_batch']} | {row['scheduled_tokens']} | "
                f"{row['step_ms']:.2f} | {shares['attention']:.1%} | {shares['expert']:.1%} | "
                f"{shares['dense']:.1%} | {shares['other']:.1%} | {routing['mean_active_experts']:.1f} | "
                f"{routing['mean_max_to_mean']:.2f} | {routing['mean_p95_to_mean']:.2f} | "
                f"{row['primary_bottleneck']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_surface_report(raw_steps: str | Path, json_path: Path, markdown_path: Path) -> dict[str, Any]:
    import json

    report = summarize_surface(load_fine_raw_records(raw_steps))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_surface_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path), "rows": len(report["rows"])}

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPONENTS = ("attention_compute", "ffn_compute", "communication", "other")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def _load(path: Path, family: str) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        critical_rank = int(row["critical_rank"])
        rank = row["rank_records"][critical_rank]
        requested = row["requested_workload"]
        communication = rank["communication"]
        result.append(
            {
                "family": family,
                "workload_index": int(row["workload_index"]),
                "repeat": int(row["repeat"]),
                "step_index": int(row["step_index"]),
                "phase": str(row["phase"]),
                "batch": int(requested["batch_size"]),
                "prompt": int(requested.get("prompt_tokens") or max(requested["prompt_token_lengths"])),
                "step_us": float(rank["timing_us"]["step"]),
                "components_us": {
                    key: float(rank["exclusive_timing_us"][key]) for key in COMPONENTS
                },
                "communication_by_group_us": {
                    key: float(value)
                    for key, value in communication.get("by_group_us", {}).items()
                },
                "communication_work_us": float(communication["work_us"]),
                "communication_union_us": float(communication["union_us"]),
                "communication_overlap_us": float(communication["overlap_us"]),
                "critical_rank": critical_rank,
                "rank_skew_us": float(row["rank_skew_us"]),
            }
        )
    if not result:
        raise ValueError(f"{path} contains no multi-rank records")
    return result


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectories: dict[tuple[int, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectories[(row["batch"], row["prompt"], row["phase"], row["repeat"])].append(row)

    per_repeat = []
    for (batch, prompt, phase, repeat), values in sorted(trajectories.items()):
        step_us = sum(row["step_us"] for row in values)
        components = {
            component: sum(row["components_us"][component] for row in values)
            for component in COMPONENTS
        }
        groups = {
            group: sum(row["communication_by_group_us"].get(group, 0.0) for row in values)
            for group in {key for row in values for key in row["communication_by_group_us"]}
        }
        per_repeat.append(
            {
                "batch": batch, "prompt": prompt, "phase": phase, "repeat": repeat,
                "step_us": step_us,
                "components_us": components,
                "shares": {key: value / step_us for key, value in components.items()},
                "communication_by_group_us": groups,
                "communication_group_shares": {
                    key: value / step_us for key, value in groups.items()
                },
                "step_count": len(values),
                "max_rank_skew_us": max(row["rank_skew_us"] for row in values),
            }
        )

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_repeat:
        grouped[(row["batch"], row["prompt"], row["phase"])].append(row)
    points = []
    for (batch, prompt, phase), values in sorted(grouped.items()):
        shares = {
            component: statistics.median(row["shares"][component] for row in values)
            for component in COMPONENTS
        }
        groups = sorted({key for row in values for key in row["communication_group_shares"]})
        communication_groups = {
            group: statistics.median(
                row["communication_group_shares"].get(group, 0.0) for row in values
            )
            for group in groups
        }
        comm_samples = [row["shares"]["communication"] for row in values]
        points.append(
            {
                "batch": batch, "prompt": prompt, "phase": phase,
                "repeat_count": len(values),
                "median_step_us": statistics.median(row["step_us"] for row in values),
                "median_step_count": statistics.median(row["step_count"] for row in values),
                "shares": shares,
                "communication_group_shares": communication_groups,
                "communication_share_p95": _percentile(comm_samples, 0.95),
                "communication_share_max": max(comm_samples),
                "max_rank_skew_us": max(row["max_rank_skew_us"] for row in values),
                "bottleneck": max(COMPONENTS, key=lambda key: shares[key]),
                "statistically_supported": len(values) >= 3,
            }
        )
    return points


def _transitions(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        if not point["statistically_supported"]:
            continue
        grouped[(point["batch"], point["phase"])].append(point)
    result = []
    for (batch, phase), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: row["prompt"])
        for left, right in zip(ordered, ordered[1:]):
            if left["bottleneck"] != right["bottleneck"]:
                result.append({
                    "batch": batch, "phase": phase,
                    "lower_prompt_tokens": left["prompt"],
                    "upper_prompt_tokens": right["prompt"],
                    "from": left["bottleneck"], "to": right["bottleneck"],
                })
    return result


def compare_multigpu_bottlenecks(dense_path: Path, moe_path: Path) -> dict[str, Any]:
    dense = _aggregate(_load(dense_path, "dense"))
    moe = _aggregate(_load(moe_path, "moe"))
    dense_index = {(r["batch"], r["prompt"], r["phase"]): r for r in dense}
    moe_index = {(r["batch"], r["prompt"], r["phase"]): r for r in moe}
    pairs = []
    for key in sorted(dense_index.keys() & moe_index.keys()):
        pairs.append({
            "batch": key[0], "prompt": key[1], "phase": key[2],
            "dense": dense_index[key], "moe": moe_index[key],
        })
    return {
        "schema_version": "multigpu-dense-moe-bottleneck/v1",
        "label_source": "cuda_event_multirank_critical_rank",
        "aggregation": "sum_steps_within_repeat_phase_then_median_across_repeats",
        "dense": {"points": dense, "transitions": _transitions(dense)},
        "moe": {"points": moe, "transitions": _transitions(moe)},
        "paired_points": pairs,
    }


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def write_multigpu_report(report: dict[str, Any], output: Path) -> None:
    names = {
        "attention_compute": "Attention", "ffn_compute": "FFN",
        "communication": "Communication", "other": "Other",
    }
    lines = [
        "# Multi-GPU Dense–MoE Bottleneck Transfer", "",
        "## Measurement Contract", "",
        "2×H20，Dense=TP2，MoE=TP2/EP2。组件来自最慢 rank 的真实 CUDA Event；",
        "每个 repeat 内先累加同 phase steps，再跨 repeats 取中位数；少于 3 个 phase observations 的点",
        "不参与 transfer detection。", "",
    ]
    for batch in sorted({row["batch"] for row in report["paired_points"]}):
        lines.extend([f"## Batch {batch}", ""])
        for phase in ("prefill", "decode", "mixed"):
            rows = [r for r in report["paired_points"] if r["batch"] == batch and r["phase"] == phase]
            if not rows:
                continue
            lines.extend([
                f"### {phase.title()}", "",
                "| Prompt | Model | Attention compute | FFN compute | Communication | TP comm | EP comm | Other | Bottleneck | Step ms | Comm p95 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
            ])
            for row in rows:
                for family in ("dense", "moe"):
                    value = row[family]; shares = value["shares"]
                    lines.append(
                        f"| {row['prompt']} | {family.upper()} | {_pct(shares['attention_compute'])} | "
                        f"{_pct(shares['ffn_compute'])} | {_pct(shares['communication'])} | "
                        f"{_pct(value['communication_group_shares'].get('tp', 0.0))} | "
                        f"{_pct(value['communication_group_shares'].get('ep', 0.0))} | "
                        f"{_pct(shares['other'])} | {names[value['bottleneck']]} | "
                        f"{value['median_step_us']/1000.0:.2f} | {_pct(value['communication_share_p95'])} |"
                    )
            lines.append("")
    lines.extend(["## Transfers", ""])
    for family in ("dense", "moe"):
        lines.extend([f"### {family.upper()}", ""])
        transitions = report[family]["transitions"]
        if not transitions:
            lines.append("当前 grid 内未观察到主瓶颈变化。")
        for row in transitions:
            lines.append(
                f"- batch={row['batch']} {row['phase']}: {names[row['from']]} → "
                f"{names[row['to']]}, prompt {row['lower_prompt_tokens']}–{row['upper_prompt_tokens']}。"
            )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

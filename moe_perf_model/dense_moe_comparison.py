from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPONENTS = ("attention", "ffn", "other")


def _load(path: Path, family: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        timing = raw["timing_us"]
        step = float(timing["step"])
        attention = float(timing["attention"])
        ffn = float(timing["mlp"] if family == "dense" else timing["expert"])
        requested = raw["requested_workload"]
        requests = raw["requests"]
        phase = str(raw.get("phase") or (
            next(iter({row["phase"] for row in requests}))
            if len({row["phase"] for row in requests}) == 1 else "mixed"
        ))
        components = {
            "attention": attention,
            "ffn": ffn,
            "other": max(0.0, step - attention - ffn),
        }
        rows.append(
            {
                "family": family,
                "workload_index": int(raw["workload_index"]),
                "repeat": int(raw["repeat"]),
                "step_index": int(raw["step_index"]),
                "batch": int(requested["batch_size"]),
                "prompt": int(requested.get("prompt_tokens") or max(requested["prompt_token_lengths"])),
                "phase": phase,
                "realized_batch": len(requests),
                "scheduled": sum(int(row["scheduled_tokens"]) for row in requests),
                "total_kv": sum(int(row["context_length"]) for row in requests),
                "max_context": max(int(row["context_length"]) for row in requests),
                "step_us": step,
                "components_us": components,
            }
        )
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["batch"], row["prompt"], row["phase"])].append(row)
    result = []
    for (batch, prompt, phase), values in sorted(grouped.items()):
        step = statistics.median(row["step_us"] for row in values)
        components = {
            component: statistics.median(
                row["components_us"][component] for row in values
            )
            for component in COMPONENTS
        }
        shares = {component: components[component] / step for component in COMPONENTS}
        bottleneck = max(COMPONENTS, key=lambda component: shares[component])
        result.append(
            {
                "batch": batch,
                "prompt": prompt,
                "phase": phase,
                "samples": len(values),
                "step_us": step,
                "components_us": components,
                "shares": shares,
                "bottleneck": bottleneck,
                "median_realized_batch": statistics.median(
                    row["realized_batch"] for row in values
                ),
                "median_scheduled_tokens": statistics.median(
                    row["scheduled"] for row in values
                ),
                "median_total_kv_tokens": statistics.median(
                    row["total_kv"] for row in values
                ),
            }
        )
    return result


def _transitions(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[(point["batch"], point["phase"])].append(point)
    result = []
    for (batch, phase), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: row["prompt"])
        for left, right in zip(ordered, ordered[1:]):
            if left["bottleneck"] != right["bottleneck"]:
                result.append(
                    {
                        "batch": batch,
                        "phase": phase,
                        "lower_prompt_tokens": left["prompt"],
                        "upper_prompt_tokens": right["prompt"],
                        "from": left["bottleneck"],
                        "to": right["bottleneck"],
                    }
                )
    return result


def _alignment(dense: list[dict[str, Any]], moe: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["workload_index"], row["repeat"], row["step_index"], row["phase"],
            row["realized_batch"], row["scheduled"], row["total_kv"],
        )

    dense_keys = {key(row) for row in dense}
    moe_keys = {key(row) for row in moe}
    common = dense_keys & moe_keys
    return {
        "dense_steps": len(dense_keys),
        "moe_steps": len(moe_keys),
        "exact_aligned_steps": len(common),
        "dense_coverage": len(common) / len(dense_keys),
        "moe_coverage": len(common) / len(moe_keys),
    }


def compare_dense_moe(dense_path: Path, moe_path: Path) -> dict[str, Any]:
    dense_raw = _load(dense_path, "dense")
    moe_raw = _load(moe_path, "moe")
    dense = _aggregate(dense_raw)
    moe = _aggregate(moe_raw)
    dense_index = {(r["batch"], r["prompt"], r["phase"]): r for r in dense}
    moe_index = {(r["batch"], r["prompt"], r["phase"]): r for r in moe}
    common = sorted(dense_index.keys() & moe_index.keys())
    paired = []
    for key in common:
        left, right = dense_index[key], moe_index[key]
        paired.append(
            {
                "batch": key[0],
                "prompt": key[1],
                "phase": key[2],
                "dense": left,
                "moe": right,
                "same_bottleneck": left["bottleneck"] == right["bottleneck"],
                "share_delta_dense_minus_moe": {
                    component: left["shares"][component] - right["shares"][component]
                    for component in COMPONENTS
                },
            }
        )
    return {
        "schema_version": "dense-moe-bottleneck-comparison/v1",
        "component_definition": {
            "attention": "complete attention module",
            "ffn_dense": "complete Qwen3 dense MLP",
            "ffn_moe": "complete Qwen3 MoE block including routing and experts",
            "other": "step latency not enclosed by Attention or FFN module events",
        },
        "alignment": _alignment(dense_raw, moe_raw),
        "dense": {"points": dense, "transitions": _transitions(dense)},
        "moe": {"points": moe, "transitions": _transitions(moe)},
        "paired_points": paired,
        "paired_point_count": len(paired),
        "bottleneck_agreement": (
            sum(row["same_bottleneck"] for row in paired) / len(paired) if paired else 0.0
        ),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_dense_moe_report(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Qwen3 Dense–MoE Bottleneck Transfer",
        "",
        "## Conclusion",
        "",
        "Qwen3-8B Dense 也存在负载相关瓶颈变化，但已测范围内与 Qwen3-30B-A3B MoE 的方向不同：",
        "Dense Prefill 从小负载 Attention-bound 转为 FFN-bound；MoE 则在长 Prefill 从 FFN-bound 转为",
        "Attention-bound。Decode 中 Dense 保持 Attention-bound，MoE 保持 FFN-bound。",
        "",
        "## Measurement Contract",
        "",
        "组件统一为 Attention / FFN / Other；Dense FFN 是完整 MLP，MoE FFN 是完整 MoE block。",
        "所有数字来自同 runtime CUDA Event 实测，不是模拟结果。",
        "两款模型的层数、hidden/FFN size 和 active parameters 不同，因此这是具体模型的系统对照，",
        "不是仅改变 sparsity 的严格架构消融；差异不能全部归因于 Dense/MoE。",
        "",
        "## Alignment",
        "",
        f"- Dense steps: {report['alignment']['dense_steps']}",
        f"- MoE steps: {report['alignment']['moe_steps']}",
        f"- Exact-aligned steps: {report['alignment']['exact_aligned_steps']}",
        f"- Dense exact coverage: {_pct(report['alignment']['dense_coverage'])}",
        f"- MoE exact coverage: {_pct(report['alignment']['moe_coverage'])}",
        "",
    ]
    pairs = report["paired_points"]
    for batch in sorted({row["batch"] for row in pairs}):
        lines.extend([f"## Batch {batch}", ""])
        for phase in ("prefill", "decode", "mixed"):
            rows = [r for r in pairs if r["batch"] == batch and r["phase"] == phase]
            if not rows:
                continue
            lines.extend([
                f"### {phase.title()}", "",
                "| Prompt/KV anchor | Model | Attention | FFN | Other | Bottleneck | Step ms |",
                "|---:|---|---:|---:|---:|---|---:|",
            ])
            for row in rows:
                for family in ("dense", "moe"):
                    value = row[family]
                    anchor = (
                        str(row["prompt"]) if phase == "prefill"
                        else f"{row['prompt']} (KV≈{value['median_total_kv_tokens']:.0f})"
                    )
                    lines.append(
                        f"| {anchor} | {family.upper()} | {_pct(value['shares']['attention'])} | "
                        f"{_pct(value['shares']['ffn'])} | {_pct(value['shares']['other'])} | "
                        f"{value['bottleneck']} | {value['step_us'] / 1000.0:.2f} |"
                    )
            lines.append("")
    lines.extend(["## Observed Transfers", ""])
    for family in ("dense", "moe"):
        transitions = report[family]["transitions"]
        lines.append(f"### {family.upper()}")
        lines.append("")
        if not transitions:
            lines.append("当前 workload grid 内未观察到主瓶颈类别变化。")
        else:
            for row in transitions:
                lines.append(
                    f"- batch={row['batch']} {row['phase']}: {row['from']} → {row['to']}, "
                    f"prompt anchor {row['lower_prompt_tokens']}–{row['upper_prompt_tokens']}。"
                )
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

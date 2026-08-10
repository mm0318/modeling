from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .execution_plan import STAGES, StepExecutionPlan


def _header() -> dict[str, Any]:
    return {
        "model": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "num_layers": 48,
            "hidden_size": 2048,
            "num_experts": 128,
            "top_k": 8,
            "expert_intermediate_size": 768,
            "num_attention_heads": 32,
            "num_kv_heads": 4,
            "num_moe_layers": 48,
        },
        "hardware": {
            "hardware_id": "NVIDIA-H20",
            "model_weight_bytes": 61064794624,
            "kv_bytes_per_token": 98304,
            "activation_bytes_per_scheduled_token": 4096,
        },
        "runtime": {
            "runtime": "vllm",
            "vllm_version": "0.18.1",
            "precision": "bfloat16",
            "cuda_graph": False,
            "max_num_batched_tokens": 65536,
            "max_num_seqs": 32,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
        },
    }


def _routing(scheduled_tokens: int, active_experts: int, skew: float) -> dict[str, Any]:
    routed = scheduled_tokens * 8 * 48
    if active_experts > routed:
        active_experts = routed
    weights = [(index + 1) ** skew for index in range(active_experts)]
    remaining = routed - active_experts
    counts = [1 + int(remaining * weight / sum(weights)) for weight in weights]
    difference = routed - sum(counts)
    for index in range(abs(difference)):
        selected = index % active_experts
        counts[selected] += 1 if difference > 0 else -1
    counts.extend([0] * (128 - active_experts))
    return {
        "routed_tokens": routed,
        "active_experts": active_experts,
        "tokens_per_expert": counts,
    }


def _config(
    phase: str,
    batch: int,
    length: int,
    skew: float = 0.5,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    raw = deepcopy(_header())
    raw["runtime"]["cuda_graph"] = cuda_graph
    if phase == "prefill":
        scheduled = batch * length
        workload = {
            "phase": phase,
            "batch_size": batch,
            "input_length": length,
            "context_length": length,
            "query_length": length,
            "scheduled_tokens": scheduled,
            "total_kv_tokens": batch * length,
        }
        active = min(128, max(16, int(math.sqrt(scheduled) * 4)))
    else:
        scheduled = batch
        workload = {
            "phase": phase,
            "batch_size": batch,
            "input_length": max(0, length - 1),
            "context_length": length,
            "query_length": 1,
            "scheduled_tokens": scheduled,
            "total_kv_tokens": batch * length,
        }
        active = min(128, max(8, batch * 4))
    raw["workload"] = workload
    raw["routing"] = _routing(scheduled, active, skew)
    return raw


def _stats(values: list[int]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance) / mean if mean else 0.0


def _labels(config: dict[str, Any]) -> dict[str, Any]:
    workload = config["workload"]
    routing = config["routing"]
    phase = workload["phase"]
    batch = workload["batch_size"]
    scheduled = workload["scheduled_tokens"]
    context = workload["context_length"]
    _, cv = _stats(routing["tokens_per_expert"])
    if phase == "prefill":
        attention = 80.0 + 0.018 * scheduled + 0.000002 * batch * context * context
    else:
        attention = 45.0 + 1.7 * batch + 0.0007 * workload["total_kv_tokens"]
    expert = (
        110.0
        + 0.055 * routing["routed_tokens"]
        + 2.1 * routing["active_experts"]
        + 18.0 * cv
    )
    dense = 35.0 + 0.014 * scheduled
    other = 28.0 + 2.0 * math.log1p(batch) + (8.0 if config["runtime"]["cuda_graph"] else 16.0)
    transition_residual = 12.0 + 1.5 * math.log1p(scheduled)
    step = attention + expert + dense + other + transition_residual
    memory = (
        config["hardware"]["model_weight_bytes"]
        + config["hardware"]["kv_bytes_per_token"] * workload["total_kv_tokens"]
        + config["hardware"]["activation_bytes_per_scheduled_token"] * scheduled
    )
    return {
        "label_source": "synthetic_fixture",
        "runtime": "vllm",
        "vllm_version": "0.18.1",
        "precision": "bfloat16",
        "attention_us": attention,
        "expert_us": expert,
        "dense_us": dense,
        "other_us": other,
        "step_us": step,
        "peak_memory_bytes": memory,
    }


def write_demo(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "prefill": [(1, 128), (4, 256), (8, 512), (4, 1024), (8, 2048), (16, 256)],
        "decode": [(1, 1024), (4, 1024), (8, 4096), (16, 4096), (4, 8192), (16, 8192)],
    }
    records = []
    for phase, phase_shapes in shapes.items():
        for batch, length in phase_shapes:
            for skew in (0.3, 1.2):
                config = _config(phase, batch, length, skew, cuda_graph=batch in {8, 16})
                records.append({"config": config, "labels": _labels(config)})
    training_path = output_dir / "training_records.jsonl"
    with training_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    predict_config = _config("prefill", 8, 1024, 0.8, cuda_graph=True)
    predict_path = output_dir / "predict_config.json"
    predict_path.write_text(json.dumps(predict_config, indent=2), encoding="utf-8")

    replay_configs = [
        _config("prefill", 2, 256, 0.5),
        _config("decode", 2, 256, 0.6),
        _config("decode", 2, 257, 0.7),
    ]
    replay_rows = [
        {
            "step_id": 0,
            "config": replay_configs[0],
            "request_events": [
                {"request_id": "r0", "arrival_us": 0.0, "emitted_tokens": 0, "finished": False},
                {"request_id": "r1", "arrival_us": 0.0, "emitted_tokens": 0, "finished": False},
            ],
        },
        {
            "step_id": 1,
            "config": replay_configs[1],
            "request_events": [
                {"request_id": "r0", "arrival_us": 0.0, "emitted_tokens": 1, "finished": False},
                {"request_id": "r1", "arrival_us": 0.0, "emitted_tokens": 1, "finished": False},
            ],
        },
        {
            "step_id": 2,
            "config": replay_configs[2],
            "request_events": [
                {"request_id": "r0", "arrival_us": 0.0, "emitted_tokens": 1, "finished": True},
                {"request_id": "r1", "arrival_us": 0.0, "emitted_tokens": 1, "finished": True},
            ],
        },
    ]
    replay_path = output_dir / "replay_trace.jsonl"
    with replay_path.open("w", encoding="utf-8") as handle:
        for row in replay_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    note_path = output_dir / "SYNTHETIC_ONLY.txt"
    note_path.write_text(
        "These records are deterministic synthetic fixtures for CPU smoke tests.\n"
        "They are not measured H20/vLLM performance data.\n",
        encoding="utf-8",
    )
    return {
        "training_records": str(training_path),
        "predict_config": str(predict_path),
        "replay_trace": str(replay_path),
        "warning": str(note_path),
    }


def _fine_histogram(scheduled: int, layer_id: int, skew: float) -> list[int]:
    routed = scheduled * 8
    weights = [((index + layer_id) % 128 + 1) ** skew for index in range(128)]
    counts = [int(routed * weight / sum(weights)) for weight in weights]
    for index in range(routed - sum(counts)):
        counts[(index + layer_id) % 128] += 1
    return counts


def _fine_record(workload_index: int, phase: str, batch: int, length: int) -> dict[str, Any]:
    scheduled = batch * length if phase == "prefill" else batch
    queries = [length] * batch if phase == "prefill" else [1] * batch
    prior = 0 if phase == "prefill" else length - 1
    requests = [
        {
            "request_id": f"fixture-{workload_index}-{phase}-{index}",
            "phase": phase,
            "input_length": length if phase == "prefill" else length - 1,
            "prior_context_length": prior,
            "scheduled_tokens": query,
            "context_length": prior + query,
        }
        for index, query in enumerate(queries)
    ]
    layer_records = []
    explicit = 0.0
    aggregate = [0] * 128
    for layer_id in range(48):
        histogram = _fine_histogram(scheduled, layer_id, 0.5 + 0.1 * workload_index)
        active = sum(value > 0 for value in histogram)
        maximum = max(histogram)
        timing = {
            "attention": 120.0 + 2.0 * batch + 0.025 * scheduled + 0.001 * batch * length,
            "router": 18.0 + 0.009 * scheduled,
            "dispatch": 8.0 + 0.003 * sum(histogram) + 0.02 * active,
            "expert_gemm": 90.0 + 0.018 * sum(histogram) + 0.35 * maximum,
            "combine": 9.0 + 0.0025 * sum(histogram) + 0.015 * active,
            "dense": 32.0 + 0.011 * scheduled,
            "transition": 5.0 + 0.001 * scheduled,
        }
        explicit += sum(timing.values())
        aggregate = [left + right for left, right in zip(aggregate, histogram)]
        layer_records.append(
            {
                "layer_id": layer_id,
                "tokens_per_expert": histogram,
                "timing_us": timing,
                "execution_counts": {stage: 1 for stage in STAGES},
            }
        )
    fixed_us = 70.0 + 1.5 * batch
    return {
        "collector_version": "0.1.0",
        "fine_grained_version": "2.0",
        "label_source": "synthetic_fixture",
        "workload_index": workload_index,
        "repeat": 0,
        "step_index": 0 if phase == "prefill" else 1,
        "requests": requests,
        "profile_header": _header(),
        "kernel_identity": {
            "dtype": "bfloat16",
            "attention_backend": "FLASH_ATTN_V3-fixture",
            "moe_backend": "TRITON-fixture",
            "moe_kernel_config": "TritonExperts:E128:N768:fixture",
            "dense_backend": "vllm-eager-fixture",
        },
        "tokens_per_expert": aggregate,
        "layer_tokens_per_expert": [row["tokens_per_expert"] for row in layer_records],
        "layer_records": layer_records,
        "non_layer_dense_us": 40.0,
        "fixed_us": fixed_us,
        "timing_us": {"step": explicit + fixed_us},
        "peak_memory_bytes": 0,
    }


def write_whitebox_demo(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = [(1, 16), (1, 32), (2, 32), (2, 64), (4, 64), (4, 128)]
    rows = []
    for workload_index, (batch, length) in enumerate(shapes):
        rows.append(_fine_record(workload_index, "prefill", batch, length))
        rows.append(_fine_record(workload_index, "decode", batch, length))
    raw_path = output_dir / "fine_raw_steps.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    plan = StepExecutionPlan.from_raw_step(rows[6])
    plan_path = output_dir / "whitebox_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    warning_path = output_dir / "SYNTHETIC_WHITEBOX_ONLY.txt"
    warning_path.write_text(
        "Formula-generated fine-grained fixture. Not an H20 measurement.\n",
        encoding="utf-8",
    )
    return {
        "fine_raw_steps": str(raw_path),
        "plan": str(plan_path),
        "warning": str(warning_path),
    }

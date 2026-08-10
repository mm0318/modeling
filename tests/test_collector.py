from __future__ import annotations

import json

import pytest

from moe_perf_model.collector import (
    CollectionWorkload,
    COLLECTOR_VERSION,
    CollectionManifest,
    aggregate_rank_step,
    aggregate_dense_rank_step,
    make_diverse_prompt,
    raw_step_to_training_record,
)
from moe_perf_model.schema import SchemaError, TrainingRecord


def _header():
    return {
        "model": {
            "model_id": "Qwen/Qwen3-30B-A3B",
            "num_layers": 2,
            "num_moe_layers": 2,
            "hidden_size": 16,
            "num_experts": 4,
            "top_k": 2,
            "expert_intermediate_size": 8,
            "num_attention_heads": 4,
            "num_kv_heads": 2,
        },
        "hardware": {
            "hardware_id": "NVIDIA-H20",
            "model_weight_bytes": 1,
            "kv_bytes_per_token": 1,
            "activation_bytes_per_scheduled_token": 1,
        },
        "runtime": {
            "runtime": "vllm",
            "vllm_version": "0.18.1",
            "precision": "bfloat16",
            "cuda_graph": False,
            "max_num_batched_tokens": 32,
            "max_num_seqs": 2,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
        },
    }


def test_raw_step_conversion_uses_scheduler_batch_and_cross_layer_routing():
    raw = {
        "collector_version": COLLECTOR_VERSION,
        "requests": [
            {"phase": "decode", "input_length": 8, "scheduled_tokens": 1, "context_length": 9},
            {"phase": "decode", "input_length": 12, "scheduled_tokens": 1, "context_length": 13},
        ],
        "tokens_per_expert": [2, 2, 2, 2],
        "timing_us": {"step": 10, "attention": 3, "expert": 4, "dense": 1},
        "peak_memory_bytes": 100,
    }
    converted = raw_step_to_training_record(raw, _header())
    record = TrainingRecord.from_dict(converted)
    assert record.config.workload.batch_size == 2
    assert record.config.workload.query_length == 1
    assert record.config.workload.total_kv_tokens == 22
    assert record.config.routing.routed_tokens == 8
    assert record.labels.other_us == 2


def test_manifest_rejects_cuda_graph_and_parallel_collection():
    raw = {
        "profile_header": _header(),
        "model_path": "/model",
        "seed": 0,
        "warmup_repeats": 0,
        "workloads": [{"batch_size": 1, "prompt_tokens": 8, "output_tokens": 1, "repeats": 1}],
    }
    assert CollectionManifest.from_dict(json.loads(json.dumps(raw))).workloads
    raw["profile_header"]["runtime"]["cuda_graph"] = True
    with pytest.raises(SchemaError, match="cuda_graph"):
        CollectionManifest.from_dict(raw)


def test_manifest_accepts_tp2_ep2_but_rejects_incompatible_ep():
    raw = {
        "profile_header": _header(), "model_path": "/model", "seed": 0,
        "warmup_repeats": 0,
        "workloads": [{"batch_size": 1, "prompt_tokens": 8, "output_tokens": 1, "repeats": 1}],
    }
    raw["profile_header"]["runtime"].update(
        {"tensor_parallel_size": 2, "expert_parallel_size": 2}
    )
    assert CollectionManifest.from_dict(raw).profile_header["runtime"]["expert_parallel_size"] == 2
    raw["profile_header"]["runtime"]["expert_parallel_size"] = 3
    with pytest.raises(SchemaError, match="EP=1 or EP=TP"):
        CollectionManifest.from_dict(raw)


def _rank_record(rank: int, step_us: float):
    stages = {stage: 1.0 + rank for stage in (
        "attention", "router", "dispatch", "expert_gemm", "combine", "dense", "transition"
    )}
    return {
        "rank": rank, "collector_version": COLLECTOR_VERSION,
        "fine_grained_version": "2.0", "label_source": "cuda_event", "phase": "prefill",
        "requests": [{"phase": "prefill", "input_length": 1, "prior_context_length": 0,
                      "scheduled_tokens": 1, "context_length": 1, "request_id": "r"}],
        "timing_us": {"step": step_us, "attention": 2, "expert": 3, "dense": 1},
        "fixed_us": 1.0, "non_layer_dense_us": 1.0, "peak_memory_bytes": 10 + rank,
        "layer_records": [{"layer_id": 0, "tokens_per_expert": [1, 1],
                           "timing_us": stages, "execution_counts": {key: 1 for key in stages}}],
        "layer_tokens_per_expert": [[1, 1]], "tokens_per_expert": [1, 1],
        "kernel_identity": {"backend": "x"}, "module_counts": {}, "observed_moe_layers": 1,
        "serial_composed_us": 8, "composition_overrun_us": 0, "uncovered_us": 0,
    }


def test_multirank_aggregate_uses_slowest_rank_and_preserves_raw_records():
    row = aggregate_rank_step([_rank_record(0, 10), _rank_record(1, 12)])
    assert row["timing_us"]["step"] == 12
    assert row["rank_skew_us"] == 2
    assert row["layer_records"][0]["timing_us"]["expert_gemm"] == 2
    assert len(row["rank_records"]) == 2
    assert row["label_source"] == "cuda_event_multirank"
    assert row["critical_rank"] == 1


def test_dense_multirank_aggregate_uses_one_complete_critical_rank_record():
    rows = []
    for rank, step in ((0, 10.0), (1, 12.0)):
        rows.append({
            "rank": rank, "phase": "prefill",
            "requests": [{"request_id": "r", "phase": "prefill"}],
            "timing_us": {"step": step, "attention": 4.0, "mlp": 5.0},
            "exclusive_timing_us": {
                "attention_compute": 2.0 + rank, "ffn_compute": 3.0,
                "communication": 1.0, "other": 4.0,
            },
            "communication": {"union_us": 1.0},
            "kernel_identity": {"backend": "x"},
            "module_counts": {"attention": 1, "mlp": 1},
            "peak_memory_bytes": 100 + rank,
        })
    row = aggregate_dense_rank_step(rows)
    assert row["critical_rank"] == 1
    assert row["exclusive_timing_us"]["attention_compute"] == 3.0
    assert row["rank_step_us"] == [10.0, 12.0]
    assert row["peak_memory_bytes"] == 101


def test_diverse_prompts_are_reproducible_and_not_constant():
    first = make_diverse_prompt(32, seed=0, workload_index=1, request_index=0, vocab_size=32000)
    assert first == make_diverse_prompt(32, seed=0, workload_index=1, request_index=0, vocab_size=32000)
    assert len(set(first)) > 16
    assert first != make_diverse_prompt(32, seed=0, workload_index=1, request_index=1, vocab_size=32000)


def test_workload_accepts_heterogeneous_prompt_lengths():
    workload = CollectionWorkload.from_dict({
        "batch_size": 3,
        "prompt_token_lengths": [8, 64, 512],
        "output_tokens": 4,
        "repeats": 2,
    })
    assert workload.prompt_tokens is None
    assert workload.prompt_lengths == (8, 64, 512)
    with pytest.raises(SchemaError, match="one entry per request"):
        CollectionWorkload.from_dict({
            "batch_size": 2, "prompt_token_lengths": [8],
            "output_tokens": 1, "repeats": 1,
        })

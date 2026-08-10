#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description="Run real vLLM scheduling with a virtual MoE forward")
    parser.add_argument("--model", default="/home/marongbai/profiling/Qwen3-30B-A3B")
    parser.add_argument("--fixed-config-artifact", default=str(root / "results/h20_tp2_ep2/combined_v3/fixed_config_routing_rf.joblib"))
    parser.add_argument("--routing-steps", default=str(root / "results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl"))
    parser.add_argument("--emulator-profile-pack", default="/home/marongbai/profiling/llm-emu/results/H20-adaptive-h20-qwen3-8b-full/serving-full.json")
    parser.add_argument("--trace", default=str(root / "results/vllm_forward_hijack/trace.jsonl"))
    parser.add_argument("--summary", default=str(root / "results/vllm_forward_hijack/summary.json"))
    parser.add_argument("--virtual-tp", type=int, default=2)
    parser.add_argument("--virtual-ep", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--prompt-lengths", default="8,8")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = Path(args.trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    if trace.exists():
        trace.unlink()

    # These must be set before importing vllm_emulator or vllm.
    os.environ["VLLM_EMULATOR_ENABLE_ORACLE"] = "1"
    os.environ["VLLM_EMULATOR_EXECUTOR_HOOK"] = "1"
    os.environ["VLLM_EMULATOR_MOCK_CUDA"] = "1"
    os.environ["VLLM_EMULATOR_PROFILE_PACK"] = str(Path(args.emulator_profile_pack).resolve())
    os.environ["VLLM_EMULATOR_MODE"] = "accelerated" if not args.realtime else "realtime"
    # Keep EngineCore in this process so the injected hook object is authoritative.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from moe_perf_model.vllm_forward_hijack import (
        ProfileReplayRoutingProvider,
        VllmForwardHijackHook,
    )
    from moe_perf_model.fixed_config_oracle import load_fixed_config_artifact

    fixed_oracle = load_fixed_config_artifact(args.fixed_config_artifact)
    if (args.virtual_tp, args.virtual_ep) != (
        fixed_oracle.identity["tensor_parallel_size"],
        fixed_oracle.identity["expert_parallel_size"],
    ):
        raise ValueError("virtual TP/EP must match the fixed-config artifact")
    routing = ProfileReplayRoutingProvider(args.routing_steps)
    hook = VllmForwardHijackHook(
        model=None, routing_provider=routing, compiler=None,
        trace_path=trace, realtime=args.realtime,
        fixed_config_oracle=fixed_oracle, record_auxiliary_graph=False,
    )

    import vllm_emulator.hooks.executor_hook as executor_hook
    executor_hook.get_executor_hook = lambda: hook

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(1, len(args.prompt_lengths.split(","))),
        gpu_memory_utilization=0.50,
        trust_remote_code=True,
    )
    lengths = [int(value) for value in args.prompt_lengths.split(",") if value.strip()]
    prompts = [
        {"prompt_token_ids": [100 + (index % 1000) for index in range(length)]}
        for length in lengths
    ]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    summary = {
        "schema_version": "vllm-forward-hijack-summary/v1",
        "runtime_control_flow": "vllm-0.18.1-uniproc-real-scheduler",
        "forward_execution": "fixed_config_forward_oracle",
        "virtual_parallelism": {"tp": args.virtual_tp, "ep": args.virtual_ep,
                                "world_size": fixed_oracle.identity["rank_count"],
                                "gpus_per_node": args.gpus_per_node},
        "requests": len(outputs),
        "steps": len(rows),
        "phases": sorted({row["scheduler_context"]["phase"] for row in rows}),
        "predicted_step_us": [row["predicted_step_us"] for row in rows],
        "formal": all(row["formal"] for row in rows) if rows else False,
        "formal_note": "scheduler/control flow and TP2/EP2 timing artifact are measured; routing is profile replay",
        "trace": str(trace.resolve()),
    }
    destination = Path(args.summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

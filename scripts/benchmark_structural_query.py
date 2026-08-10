#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from moe_perf_model.collective_model import NcclResponseTable
from moe_perf_model.execution_plan import StepExecutionPlan
from moe_perf_model.fixed_config_oracle import load_multirank_records
from moe_perf_model.parallel_forward import ParallelForwardGraphCompiler, ParallelLayout
from moe_perf_model.rank_local_compute import load_rank_local_compute_artifact
from moe_perf_model.structural_forward import ForwardStructuralPredictor
from moe_perf_model.whitebox_model import load_whitebox_artifact


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitebox", type=Path, required=True)
    parser.add_argument("--nccl-response", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--rank-compute", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--plan-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    rows = load_multirank_records(args.steps)
    plan = StepExecutionPlan.from_raw_step(rows[args.plan_index])
    compiler = ParallelForwardGraphCompiler(
        layout=ParallelLayout.vllm(2, 2, args.ep_size, 8),
        collective_backend=NcclResponseTable.load(args.nccl_response),
        collective_backend_name="nccl",
        hidden_size=2048,
        all2all_backend="allgather_reducescatter",
        rank_local_compute_backend=load_rank_local_compute_artifact(args.rank_compute),
    )
    predictor = ForwardStructuralPredictor(
        compute_model=load_whitebox_artifact(args.whitebox),
        graph_compiler=compiler,
    )
    begin = time.perf_counter_ns()
    first = predictor.predict(plan, include_trace=False)
    cold_ms = (time.perf_counter_ns() - begin) / 1e6
    samples_ms = []
    for _ in range(args.iterations):
        begin = time.perf_counter_ns()
        predictor.predict(plan, include_trace=False)
        samples_ms.append((time.perf_counter_ns() - begin) / 1e6)
    report = {
        "schema_version": "structural-query-benchmark/v1",
        "ep_size": args.ep_size,
        "plan_index": args.plan_index,
        "iterations": args.iterations,
        "predicted_step_us": first.step_us,
        "cold_query_ms": cold_ms,
        "warm_query_ms": {
            "mean": statistics.mean(samples_ms),
            "median": statistics.median(samples_ms),
            "p95": _p95(samples_ms),
            "min": min(samples_ms),
            "max": max(samples_ms),
        },
        "includes_artifact_load": False,
        "cache_policy": "exact_signature_index_plus_query_cache",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

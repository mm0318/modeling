#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile NCCL collectives with CUDA Events")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864,268435456")
    parser.add_argument("--collectives", default="all_reduce,all_gather,reduce_scatter,all_to_all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--target-bytes-per-repetition", type=int, default=268435456)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def _gpu_uuids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    mapping = {}
    for row in rows:
        index, uuid = [value.strip() for value in row.split(",", 1)]
        mapping[index] = uuid
    indices = [value.strip() for value in visible.split(",") if value.strip()]
    return [mapping[index] for index in indices] if indices else list(mapping.values())


def _driver_version() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0].strip()


def _operation(name: str, elements: int, world_size: int, device: torch.device) -> Callable[[], None]:
    source = torch.ones(elements, dtype=torch.float16, device=device)
    if name == "all_reduce":
        return lambda: dist.all_reduce(source)
    if name == "all_gather":
        destination = torch.empty(elements * world_size, dtype=torch.float16, device=device)
        return lambda: dist.all_gather_into_tensor(destination, source)
    if name == "reduce_scatter":
        source = torch.ones(elements * world_size, dtype=torch.float16, device=device)
        destination = torch.empty(elements, dtype=torch.float16, device=device)
        return lambda: dist.reduce_scatter_tensor(destination, source)
    if name == "all_to_all":
        destination = torch.empty_like(source)
        return lambda: dist.all_to_all_single(destination, source)
    raise ValueError(f"unsupported collective: {name}")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    collectives = [value.strip() for value in args.collectives.split(",") if value.strip()]
    samples = []
    try:
        for name in collectives:
            for requested_bytes in sizes:
                message_bytes = max(2, requested_bytes + requested_bytes % 2)
                elements = message_bytes // 2
                operation = _operation(name, elements, world_size, device)
                for _ in range(args.warmup):
                    operation()
                torch.cuda.synchronize()
                iterations = max(
                    args.min_iterations,
                    min(args.max_iterations, args.target_bytes_per_repetition // message_bytes),
                )
                rank_repetitions: list[list[float]] = [[] for _ in range(world_size)]
                for _ in range(args.repetitions):
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        operation()
                    end.record()
                    end.synchronize()
                    local_us = start.elapsed_time(end) * 1000.0 / iterations
                    local_tensor = torch.tensor([local_us], dtype=torch.float64, device=device)
                    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered, local_tensor)
                    for peer, value in enumerate(gathered):
                        rank_repetitions[peer].append(float(value.item()))
                if rank == 0:
                    synchronized = [max(values[index] for values in rank_repetitions)
                                    for index in range(args.repetitions)]
                    samples.append({
                        "collective_type": name,
                        "message_bytes_per_rank": message_bytes,
                        "rank_count": world_size,
                        "iterations_per_repetition": iterations,
                        "rank_repetitions_us": rank_repetitions,
                        "synchronized_repetitions_us": synchronized,
                        "median_us": statistics.median(synchronized),
                        "p95_us": _percentile(synchronized, 0.95),
                        "min_us": min(synchronized),
                        "max_us": max(synchronized),
                    })
        if rank == 0:
            properties = torch.cuda.get_device_properties(0)
            result = {
                "schema_version": "nccl-response/v1",
                "profile_header": {
                    "label_source": "nccl_benchmark",
                    "formal_hardware_prediction": True,
                    "hostname": socket.gethostname(),
                    "gpu_name": properties.name,
                    "gpu_uuids": _gpu_uuids(),
                    "rank_count": world_size,
                    "topology": "single_node_nvlink",
                    "backend": "nccl",
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "driver_version": _driver_version(),
                    "dtype": "float16",
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                },
                "samples": samples,
            }
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import random
import copy
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import InferenceConfig, SchemaError
from .execution_plan import STAGES


COLLECTOR_VERSION = "0.1.0"
PROMPT_PATTERN = "deterministic_diverse_uniform_v1"


def _integer(raw: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(f"collector {key} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CollectionWorkload:
    batch_size: int
    prompt_tokens: int | None
    prompt_token_lengths: tuple[int, ...] | None
    output_tokens: int
    repeats: int

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        if self.prompt_token_lengths is not None:
            return self.prompt_token_lengths
        assert self.prompt_tokens is not None
        return (self.prompt_tokens,) * self.batch_size

    @classmethod
    def from_dict(cls, raw: Any) -> "CollectionWorkload":
        if not isinstance(raw, dict):
            raise SchemaError("collector workload must be an object")
        batch_size = _integer(raw, "batch_size")
        lengths_raw = raw.get("prompt_token_lengths")
        prompt_tokens = raw.get("prompt_tokens")
        if lengths_raw is None:
            prompt_tokens = _integer(raw, "prompt_tokens")
            prompt_lengths = None
        else:
            if not isinstance(lengths_raw, list) or len(lengths_raw) != batch_size:
                raise SchemaError("prompt_token_lengths must contain one entry per request")
            prompt_lengths = tuple(
                _integer({"value": value}, "value") for value in lengths_raw
            )
            if prompt_tokens is not None:
                raise SchemaError("set prompt_tokens or prompt_token_lengths, not both")
            prompt_tokens = None
        return cls(
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            prompt_token_lengths=prompt_lengths,
            output_tokens=_integer(raw, "output_tokens"),
            repeats=_integer(raw, "repeats"),
        )


@dataclass(frozen=True)
class CollectionManifest:
    profile_header: dict[str, Any]
    model_path: str
    gpu_memory_utilization: float
    seed: int
    warmup_repeats: int
    workloads: tuple[CollectionWorkload, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "CollectionManifest":
        if not isinstance(raw, dict):
            raise SchemaError("collection manifest must be an object")
        header = raw.get("profile_header")
        if not isinstance(header, dict) or not all(
            isinstance(header.get(key), dict) for key in ("model", "hardware", "runtime")
        ):
            raise SchemaError("profile_header must contain model/hardware/runtime objects")
        runtime = header["runtime"]
        if runtime.get("vllm_version") != "0.18.1":
            raise SchemaError("collector currently requires vLLM 0.18.1")
        if runtime.get("cuda_graph") is not False:
            raise SchemaError("collector currently requires cuda_graph=false/enforce_eager")
        tp_size = _integer(runtime, "tensor_parallel_size")
        ep_size = _integer(runtime, "expert_parallel_size")
        if ep_size not in {1, tp_size}:
            raise SchemaError("single-DP collector requires EP=1 or EP=TP")
        model_path = raw.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise SchemaError("model_path must be a non-empty string")
        utilization = raw.get("gpu_memory_utilization", 0.9)
        if isinstance(utilization, bool) or not isinstance(utilization, (int, float)):
            raise SchemaError("gpu_memory_utilization must be numeric")
        utilization = float(utilization)
        if not 0 < utilization <= 1:
            raise SchemaError("gpu_memory_utilization must be in (0, 1]")
        workloads = raw.get("workloads")
        if not isinstance(workloads, list) or not workloads:
            raise SchemaError("workloads must be a non-empty list")
        return cls(
            profile_header=header,
            model_path=model_path.strip(),
            gpu_memory_utilization=utilization,
            seed=_integer(raw, "seed", minimum=0),
            warmup_repeats=_integer(raw, "warmup_repeats", minimum=0),
            workloads=tuple(CollectionWorkload.from_dict(item) for item in workloads),
        )


def load_collection_manifest(path: str | Path) -> CollectionManifest:
    return CollectionManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_diverse_prompt(
    length: int, *, seed: int, workload_index: int, request_index: int, vocab_size: int
) -> list[int]:
    if vocab_size <= 2048:
        raise ValueError("vocabulary is too small for the collector's safe token range")
    stream_seed = (seed + 1) * 1_000_003 + workload_index * 10_007 + request_index * 101
    generator = random.Random(stream_seed)
    return [generator.randrange(1000, vocab_size - 1000) for _ in range(length)]


def convert_raw_collection(
    raw_path: Path, manifest: CollectionManifest, output_dir: Path
) -> dict[str, str | int]:
    records = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise SchemaError(f"{raw_path}:{line_number}: {error}") from error
        workload_index = raw.get("workload_index")
        if (
            "requested_workload" not in raw
            and isinstance(workload_index, int)
            and 0 <= workload_index < len(manifest.workloads)
        ):
            raw["requested_workload"] = asdict(manifest.workloads[workload_index])
        records.append(raw)
    if not records:
        raise SchemaError("raw collection contains no records")
    return write_collection_outputs(records, manifest.profile_header, output_dir)


def raw_step_to_training_record(
    raw: dict[str, Any], profile_header: dict[str, Any]
) -> dict[str, Any]:
    """Convert one auditable worker record into the strict training schema."""
    if raw.get("collector_version") != COLLECTOR_VERSION:
        raise SchemaError("raw record collector_version is unsupported")
    request_rows = raw.get("requests")
    if not isinstance(request_rows, list) or not request_rows:
        raise SchemaError("raw step must contain requests")
    histogram = raw.get("tokens_per_expert")
    model = profile_header["model"]
    if not isinstance(histogram, list) or len(histogram) != model["num_experts"]:
        raise SchemaError("raw step has invalid expert histogram")
    scheduled = sum(int(row["scheduled_tokens"]) for row in request_rows)
    phases = {str(row["phase"]) for row in request_rows}
    phase = "decode" if phases == {"decode"} else "prefill"
    contexts = [int(row["context_length"]) for row in request_rows]
    prompt_lengths = [int(row["input_length"]) for row in request_rows]
    queries = [int(row["scheduled_tokens"]) for row in request_rows]
    config = {
        "model": profile_header["model"],
        "hardware": profile_header["hardware"],
        "runtime": profile_header["runtime"],
        "workload": {
            "phase": phase,
            "batch_size": len(request_rows),
            "input_length": max(prompt_lengths),
            "context_length": max(contexts),
            "query_length": max(queries),
            "scheduled_tokens": scheduled,
            "total_kv_tokens": sum(contexts),
        },
        "routing": {
            "routed_tokens": sum(histogram),
            "active_experts": sum(value > 0 for value in histogram),
            "tokens_per_expert": histogram,
        },
    }
    timing = raw["timing_us"]
    component_sum = sum(float(timing[key]) for key in ("attention", "expert", "dense"))
    step_us = max(float(timing["step"]), component_sum)
    labels = {
        "label_source": "cuda_event",
        "runtime": profile_header["runtime"]["runtime"],
        "vllm_version": profile_header["runtime"]["vllm_version"],
        "precision": profile_header["runtime"]["precision"],
        "attention_us": float(timing["attention"]),
        "expert_us": float(timing["expert"]),
        "dense_us": float(timing["dense"]),
        "other_us": max(0.0, step_us - component_sum),
        "step_us": step_us,
        "peak_memory_bytes": int(raw["peak_memory_bytes"]),
    }
    record = {"config": config, "labels": labels}
    InferenceConfig.from_dict(config)
    return record


def write_collection_outputs(
    raw_records: list[dict[str, Any]], profile_header: dict[str, Any], output_dir: Path
) -> dict[str, str | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_steps.jsonl"
    training_path = output_dir / "training_records.jsonl"
    summary_path = output_dir / "collection_summary.json"
    plans_path = output_dir / "execution_plans.jsonl"
    summary_rows = []
    fine_plan_rows = []
    with raw_path.open("w", encoding="utf-8") as raw_handle, training_path.open(
        "w", encoding="utf-8"
    ) as training_handle:
        for raw in raw_records:
            raw_handle.write(json.dumps(raw, sort_keys=True) + "\n")
            training_handle.write(
                json.dumps(raw_step_to_training_record(raw, profile_header), sort_keys=True)
                + "\n"
            )
            if raw.get("fine_grained_version") == "2.0":
                from .execution_plan import StepExecutionPlan

                fine_plan_rows.append(StepExecutionPlan.from_raw_step(raw).to_dict())
            histogram = [float(value) for value in raw["tokens_per_expert"]]
            mean = sum(histogram) / len(histogram)
            ordered = sorted(histogram)
            p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
            timing = raw["timing_us"]
            denominator = max(float(timing["step"]), 1e-9)
            attention = float(timing["attention"])
            expert = float(timing["expert"])
            dense = float(timing["dense"])
            summary_rows.append(
                {
                    "workload_index": raw.get("workload_index"),
                    "repeat": raw.get("repeat"),
                    "step_index": raw.get("step_index"),
                    "requested_workload": raw.get("requested_workload"),
                    "realized_batch_size": len(raw["requests"]),
                    "realized_phases": sorted({row["phase"] for row in raw["requests"]}),
                    "realized_scheduled_tokens": sum(row["scheduled_tokens"] for row in raw["requests"]),
                    "realized_max_context_length": max(row["context_length"] for row in raw["requests"]),
                    "step_us": denominator,
                    "component_shares": {
                        "attention": attention / denominator,
                        "expert": expert / denominator,
                        "dense": dense / denominator,
                        "other": max(0.0, denominator - attention - expert - dense) / denominator,
                    },
                    "routing": {
                        "active_experts": sum(value > 0 for value in histogram),
                        "max_mean": max(histogram) / mean if mean else 0.0,
                        "p95_mean": p95 / mean if mean else 0.0,
                    },
                }
            )
            if raw.get("fine_grained_version") == "2.0":
                fine_totals = {stage: 0.0 for stage in (
                    "attention", "router", "dispatch", "expert_gemm",
                    "combine", "dense", "transition",
                )}
                for layer in raw["layer_records"]:
                    for stage in fine_totals:
                        fine_totals[stage] += float(layer["timing_us"][stage])
                fine_totals["dense"] += float(raw.get("non_layer_dense_us", 0.0))
                fine_totals["fixed"] = float(raw["fixed_us"])
                summary_rows[-1]["fine_components_us"] = fine_totals
                summary_rows[-1]["fine_component_shares"] = {
                    stage: latency / denominator for stage, latency in fine_totals.items()
                }
    summary_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if fine_plan_rows:
        with plans_path.open("w", encoding="utf-8") as handle:
            for plan in fine_plan_rows:
                handle.write(json.dumps(plan, sort_keys=True) + "\n")
    return {
        "raw_steps": str(raw_path),
        "training_records": str(training_path),
        "collection_summary": str(summary_path),
        **({"execution_plans": str(plans_path)} if fine_plan_rows else {}),
        "count": len(raw_records),
    }


def aggregate_rank_step(rank_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not rank_records:
        raise SchemaError("multi-rank step requires rank records")
    ordered = sorted(rank_records, key=lambda row: int(row["rank"]))
    expected_ranks = list(range(len(ordered)))
    if [int(row["rank"]) for row in ordered] != expected_ranks:
        raise SchemaError("multi-rank records must contain contiguous ranks")
    reference = ordered[0]
    for row in ordered[1:]:
        if row["requests"] != reference["requests"] or row["phase"] != reference["phase"]:
            raise SchemaError("scheduler metadata differs across ranks")
        if row["layer_tokens_per_expert"] != reference["layer_tokens_per_expert"]:
            raise SchemaError("global routing histogram differs across ranks")
        if row["kernel_identity"] != reference["kernel_identity"]:
            raise SchemaError("kernel identity differs across ranks")
    aggregate = copy.deepcopy(reference)
    aggregate["schema_version"] = "multirank-forward/v1"
    aggregate["label_source"] = "cuda_event_multirank"
    aggregate["rank_count"] = len(ordered)
    aggregate["rank_records"] = ordered
    aggregate["rank_step_us"] = [float(row["timing_us"]["step"]) for row in ordered]
    aggregate["critical_rank"] = max(
        range(len(ordered)), key=lambda index: aggregate["rank_step_us"][index]
    )
    aggregate["rank_skew_us"] = max(aggregate["rank_step_us"]) - min(aggregate["rank_step_us"])
    aggregate["routing_consistency"] = "identical_global_histogram"
    for key in ("step", "attention", "expert", "dense"):
        aggregate["timing_us"][key] = max(float(row["timing_us"][key]) for row in ordered)
    aggregate["peak_memory_bytes"] = max(int(row["peak_memory_bytes"]) for row in ordered)
    aggregate["fixed_us"] = max(float(row["fixed_us"]) for row in ordered)
    aggregate["non_layer_dense_us"] = max(float(row["non_layer_dense_us"]) for row in ordered)
    layer_maps = [{int(layer["layer_id"]): layer for layer in row["layer_records"]} for row in ordered]
    aggregate_layers = []
    for layer_id in sorted(layer_maps[0]):
        if any(layer_id not in values for values in layer_maps):
            raise SchemaError("layer records differ across ranks")
        layer = copy.deepcopy(layer_maps[0][layer_id])
        for stage in STAGES:
            layer["timing_us"][stage] = max(
                float(values[layer_id]["timing_us"][stage]) for values in layer_maps
            )
        aggregate_layers.append(layer)
    aggregate["layer_records"] = aggregate_layers
    aggregate["serial_composed_us"] = aggregate["fixed_us"] + sum(
        sum(layer["timing_us"].values()) for layer in aggregate_layers
    )
    step_us = aggregate["timing_us"]["step"]
    aggregate["composition_overrun_us"] = max(0.0, aggregate["serial_composed_us"] - step_us)
    aggregate["uncovered_us"] = max(0.0, step_us - aggregate["serial_composed_us"])
    return aggregate


def aggregate_dense_rank_step(rank_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not rank_records:
        raise SchemaError("multi-rank dense step requires rank records")
    ordered = sorted(rank_records, key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in ordered] != list(range(len(ordered))):
        raise SchemaError("multi-rank dense records must contain contiguous ranks")
    reference = ordered[0]
    for row in ordered[1:]:
        if row["requests"] != reference["requests"] or row["phase"] != reference["phase"]:
            raise SchemaError("dense scheduler metadata differs across ranks")
        if row["kernel_identity"] != reference["kernel_identity"]:
            raise SchemaError("dense kernel identity differs across ranks")
        if row["module_counts"] != reference["module_counts"]:
            raise SchemaError("dense module discovery differs across ranks")
    rank_step_us = [float(row["timing_us"]["step"]) for row in ordered]
    critical_rank = max(range(len(ordered)), key=lambda index: rank_step_us[index])
    aggregate = copy.deepcopy(ordered[critical_rank])
    aggregate.update(
        {
            "schema_version": "dense-multirank-forward/v1",
            "label_source": "cuda_event_multirank",
            "rank_count": len(ordered),
            "rank_records": ordered,
            "rank_step_us": rank_step_us,
            "rank_skew_us": max(rank_step_us) - min(rank_step_us),
            "critical_rank": critical_rank,
        }
    )
    aggregate["peak_memory_bytes"] = max(
        int(row["peak_memory_bytes"]) for row in ordered
    )
    return aggregate


def write_multirank_collection_outputs(
    records: list[dict[str, Any]], profile_header: dict[str, Any], output_dir: Path
) -> dict[str, str | int]:
    if not records:
        raise SchemaError("multi-rank collection contains no records")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "multirank_steps.jsonl"
    summary_path = output_dir / "multirank_summary.json"
    summaries = []
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            requests = row["requests"]
            summaries.append({
                "workload_index": row["workload_index"],
                "repeat": row["repeat"],
                "step_index": row["step_index"],
                "phase": row["phase"],
                "realized_batch_size": len(requests),
                "scheduled_tokens": sum(item["scheduled_tokens"] for item in requests),
                "total_kv_tokens": sum(item["context_length"] for item in requests),
                "step_us": row["timing_us"]["step"],
                "rank_step_us": row["rank_step_us"],
                "rank_skew_us": row["rank_skew_us"],
                "critical_rank": row.get("critical_rank"),
                **(
                    {
                        "exclusive_timing_us": row["rank_records"][
                            int(row["critical_rank"])
                        ]["exclusive_timing_us"],
                        "communication": row["rank_records"][
                            int(row["critical_rank"])
                        ]["communication"],
                    }
                    if row.get("critical_rank") is not None
                    and row["rank_records"][int(row["critical_rank"])].get(
                        "exclusive_timing_us"
                    )
                    else {}
                ),
            })
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    return {"multirank_steps": str(raw_path), "multirank_summary": str(summary_path),
            "count": len(records)}


def write_dense_collection_outputs(
    records: list[dict[str, Any]], profile_header: dict[str, Any], output_dir: Path
) -> dict[str, str | int]:
    """Write auditable Dense steps without manufacturing MoE routing fields."""
    if not records:
        raise SchemaError("dense collection contains no records")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_steps.jsonl"
    summary_path = output_dir / "collection_summary.json"
    summaries = []
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            timing = row["timing_us"]
            step_us = max(float(timing["step"]), 1e-9)
            attention_us = float(timing["attention"])
            ffn_us = float(timing["mlp"])
            summaries.append(
                {
                    "model_family": "dense",
                    "workload_index": row["workload_index"],
                    "repeat": row["repeat"],
                    "step_index": row["step_index"],
                    "requested_workload": row["requested_workload"],
                    "phase": row["phase"],
                    "realized_batch_size": len(row["requests"]),
                    "realized_scheduled_tokens": sum(
                        request["scheduled_tokens"] for request in row["requests"]
                    ),
                    "realized_total_kv_tokens": sum(
                        request["context_length"] for request in row["requests"]
                    ),
                    "realized_max_context_length": max(
                        request["context_length"] for request in row["requests"]
                    ),
                    "step_us": step_us,
                    "components_us": {
                        "attention": attention_us,
                        "ffn": ffn_us,
                        "other": max(0.0, step_us - attention_us - ffn_us),
                    },
                    "component_shares": {
                        "attention": attention_us / step_us,
                        "ffn": ffn_us / step_us,
                        "other": max(0.0, step_us - attention_us - ffn_us) / step_us,
                    },
                }
            )
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "raw_steps": str(raw_path),
        "collection_summary": str(summary_path),
        "count": len(records),
    }


def run_collection(manifest: CollectionManifest, output_dir: Path) -> dict[str, Any]:
    """Run deterministic offline requests through an unmodified vLLM runtime."""
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != "0.18.1":
        raise RuntimeError(f"installed vLLM is {vllm.__version__}, expected 0.18.1")
    runtime = manifest.profile_header["runtime"]
    model_config = json.loads((Path(manifest.model_path) / "config.json").read_text(encoding="utf-8"))
    is_moe = int(model_config.get("num_experts", model_config.get("num_local_experts", 0))) > 0
    vocab_size = int(model_config["vocab_size"])
    maximum_length = max(
        max(workload.prompt_lengths) + workload.output_tokens
        for workload in manifest.workloads
    )
    llm = LLM(
        model=manifest.model_path,
        skip_tokenizer_init=True,
        dtype=runtime["precision"],
        tensor_parallel_size=runtime["tensor_parallel_size"],
        enable_expert_parallel=runtime["expert_parallel_size"] > 1,
        all2all_backend=runtime.get("all2all_backend", "allgather_reducescatter"),
        expert_placement_strategy=runtime.get("expert_placement_strategy", "linear"),
        disable_custom_all_reduce=runtime.get("disable_custom_all_reduce", False),
        enforce_eager=True,
        seed=manifest.seed,
        gpu_memory_utilization=manifest.gpu_memory_utilization,
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        max_num_seqs=runtime["max_num_seqs"],
        max_model_len=maximum_length,
        disable_log_stats=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        worker_extension_cls=(
            "moe_perf_model.worker_collector.CollectorWorkerExtension"
            if is_moe
            else "moe_perf_model.dense_worker_collector.DenseCollectorWorkerExtension"
        ),
    )

    install_results = llm.collective_rpc("moe_collector_install")
    profiling_repeats = int(runtime.get("profiling_repeats", 0))
    if profiling_repeats < 0:
        raise SchemaError("runtime profiling_repeats must be >= 0")
    raw_records: list[dict[str, Any]] = []
    try:
        for workload_index, workload in enumerate(manifest.workloads):
            prompts = [
                make_diverse_prompt(
                    prompt_length,
                    seed=manifest.seed,
                    workload_index=workload_index,
                    request_index=request_index,
                    vocab_size=vocab_size,
                )
                for request_index, prompt_length in enumerate(workload.prompt_lengths)
            ]
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=workload.output_tokens,
                min_tokens=workload.output_tokens,
                ignore_eos=True,
            )
            for warmup_index in range(manifest.warmup_repeats):
                llm.collective_rpc(
                    "moe_collector_set_context",
                    args=({"workload_index": workload_index, "repeat": f"warmup{warmup_index}"},),
                )
                llm.collective_rpc("moe_collector_clear_records")
                llm.generate(prompts, sampling, use_tqdm=False)
            llm.collective_rpc("moe_collector_clear_records")
            effective_repeats = max(workload.repeats, profiling_repeats)
            for repeat in range(effective_repeats):
                llm.collective_rpc(
                    "moe_collector_set_context",
                    args=({"workload_index": workload_index, "repeat": repeat},),
                )
                llm.collective_rpc("moe_collector_clear_records")
                llm.generate(prompts, sampling, use_tqdm=False)
                per_rank = llm.collective_rpc("moe_collector_get_records")
                if len(per_rank) != runtime["tensor_parallel_size"]:
                    raise RuntimeError("collector worker count does not match TP size")
                step_counts = {len(records) for records in per_rank}
                if len(step_counts) != 1:
                    raise RuntimeError("worker step counts differ across ranks")
                if len(per_rank) == 1:
                    step_rows = per_rank[0]
                elif is_moe:
                    step_rows = [
                        aggregate_rank_step(list(values)) for values in zip(*per_rank)
                    ]
                else:
                    step_rows = [
                        aggregate_dense_rank_step(list(values))
                        for values in zip(*per_rank)
                    ]
                for step_index, raw in enumerate(step_rows):
                    raw["workload_index"] = workload_index
                    raw["repeat"] = repeat
                    raw["step_index"] = step_index
                    raw["requested_workload"] = {
                        **asdict(workload), "repeats": effective_repeats
                    }
                    raw["prompt_pattern"] = PROMPT_PATTERN
                    raw["profile_header"] = manifest.profile_header
                    raw_records.append(raw)
    finally:
        llm.collective_rpc("moe_collector_uninstall")
    if not is_moe and runtime["tensor_parallel_size"] == 1:
        outputs = write_dense_collection_outputs(
            raw_records, manifest.profile_header, output_dir
        )
    elif is_moe and runtime["tensor_parallel_size"] == 1:
        outputs = write_collection_outputs(raw_records, manifest.profile_header, output_dir)
    else:
        outputs = write_multirank_collection_outputs(
            raw_records, manifest.profile_header, output_dir
        )
    return {
        **outputs,
        "collector_version": COLLECTOR_VERSION,
        "worker_install": install_results,
        "runtime": {
            "vllm_version": vllm.__version__,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "prompt_pattern": PROMPT_PATTERN,
            "model_family": "moe" if is_moe else "dense",
        },
    }

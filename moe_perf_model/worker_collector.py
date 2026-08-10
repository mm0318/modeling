from __future__ import annotations

import re
import types
import os
from typing import Any

from .collector import COLLECTOR_VERSION
from .communication_hooks import (
    install_communication_hooks,
    summarize_communication_events,
    uninstall_communication_hooks,
)
from .execution_plan import STAGES
from .torch_trace import execute_with_optional_torch_trace


def _request_metadata(runner: Any, scheduler_output: Any) -> list[dict[str, Any]]:
    new = {item.req_id: item for item in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    cached_index = {req_id: index for index, req_id in enumerate(cached.req_ids)}
    rows = []
    for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        if req_id in new:
            item = new[req_id]
            prior = int(item.num_computed_tokens)
            prompt_length = len(item.prompt_token_ids or [])
        else:
            index = cached_index.get(req_id)
            prior = int(cached.num_computed_tokens[index]) if index is not None else 0
            state = runner.requests.get(req_id)
            prompt_ids = getattr(state, "prompt_token_ids", None) if state is not None else None
            prompt_length = len(prompt_ids or [])
        rows.append(
            {
                "request_id": req_id,
                "phase": "prefill" if prior < prompt_length else "decode",
                "input_length": prompt_length,
                "prior_context_length": prior,
                "scheduled_tokens": int(scheduled),
                "context_length": prior + int(scheduled),
            }
        )
    return rows


def _layer_id(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def install_worker_collector(worker: Any) -> dict[str, Any]:
    """Install fine-grained hooks on one worker instance without source mutation."""
    import torch

    if hasattr(worker, "_moe_component_collector"):
        raise RuntimeError("MoE component collector is already installed")
    runner = worker.model_runner
    model = runner.model
    nvtx_enabled = os.environ.get("MOE_WHITEBOX_NVTX_AUDIT", "").lower() in {
        "1", "true", "yes"
    }
    torch_trace_enabled = bool(
        os.environ.get("MOE_WHITEBOX_TORCH_COMPONENT_RANGES")
    )
    state: dict[str, Any] = {
        "rank": int(getattr(worker, "rank", getattr(runner, "rank", 0))),
        "records": [],
        "handles": [],
        "patched": [],
        "current": None,
        "context": {},
        "stacks": {},
        "routing_ids": {},
        "original_execute_model": runner.execute_model,
    }
    communication_install = install_communication_hooks(state, include_ep=True)
    from vllm.distributed.parallel_state import get_tp_group

    tp_size = int(get_tp_group().world_size)

    def append_event(stage: str, layer_id: int, start: Any, end: Any) -> None:
        current = state["current"]
        if current is not None:
            current["fine_events"].append(
                {"stage": stage, "layer_id": layer_id, "start": start, "end": end}
            )

    def timer_hooks(category: str, *, stage: str | None = None, layer_id: int = -1):
        def before(module: Any, inputs: Any) -> None:
            if state["current"] is None:
                return
            event = torch.cuda.Event(enable_timing=True)
            if nvtx_enabled and stage is not None:
                torch.cuda.nvtx.range_push(f"WB/{stage}/L{layer_id}")
            trace_range = None
            if torch_trace_enabled and stage is not None:
                trace_range = torch.profiler.record_function(
                    f"WB/{stage}/L{layer_id}"
                )
                trace_range.__enter__()
            event.record()
            state["stacks"].setdefault((id(module), category), []).append(
                (event, trace_range)
            )

        def after(module: Any, inputs: Any, output: Any) -> None:
            current = state["current"]
            stack = state["stacks"].get((id(module), category), [])
            if current is None or not stack:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if nvtx_enabled and stage is not None:
                torch.cuda.nvtx.range_pop()
            start, trace_range = stack.pop()
            if trace_range is not None:
                trace_range.__exit__(None, None, None)
            current["events"][category].append((start, end))
            if stage is not None:
                append_event(stage, layer_id, start, end)

        return before, after

    def patch_method(obj: Any, attribute: str, stage: str, layer_id: int, capture_routing: bool = False) -> None:
        original = getattr(obj, attribute)

        def wrapped(self: Any, *args: Any, **kwargs: Any):
            if state["current"] is None:
                return original(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if nvtx_enabled:
                torch.cuda.nvtx.range_push(f"WB/{stage}/L{layer_id}")
            trace_range = None
            if torch_trace_enabled:
                trace_range = torch.profiler.record_function(f"WB/{stage}/L{layer_id}")
                trace_range.__enter__()
            start.record()
            try:
                result = original(*args, **kwargs)
                end.record()
            finally:
                if trace_range is not None:
                    trace_range.__exit__(None, None, None)
                if nvtx_enabled:
                    torch.cuda.nvtx.range_pop()
            append_event(stage, layer_id, start, end)
            if capture_routing:
                topk_ids = result[1]
                state["routing_ids"].setdefault(layer_id, []).append(topk_ids.detach())
            return result

        setattr(obj, attribute, types.MethodType(wrapped, obj))
        state["patched"].append((obj, attribute, original))

    module_counts = {
        "attention": 0,
        "expert": 0,
        "dense": 0,
        "router_gate": 0,
        "modular_moe": 0,
    }
    kernel_identities = set()
    attention_backends = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        layer_id = _layer_id(name)
        if class_name == "Qwen3MoeAttention":
            before, after = timer_hooks("attention", stage="attention", layer_id=layer_id)
            state["handles"].extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
            module_counts["attention"] += 1
            attention_impl = getattr(getattr(module, "attn", None), "impl", None)
            attention_backends.add(
                attention_impl.__class__.__name__ if attention_impl is not None else "FLASH_ATTN"
            )
        elif class_name == "Qwen3MoeSparseMoeBlock":
            before, after = timer_hooks("expert")
            state["handles"].extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
            module_counts["expert"] += 1
            experts = module.experts
            kernel = getattr(experts.quant_method, "kernel", None)
            impl = getattr(kernel, "impl", None)
            if impl is None or not all(hasattr(impl, method) for method in ("_prepare", "_fused_experts", "_finalize")):
                raise RuntimeError(
                    "fine-grained collector requires modular FusedMoE; monolithic fallback is forbidden"
                )
            patch_method(experts.router, "select_experts", "router", layer_id, capture_routing=True)
            patch_method(impl, "_prepare", "dispatch", layer_id)
            patch_method(impl, "_fused_experts", "expert_gemm", layer_id)
            patch_method(impl, "_finalize", "combine", layer_id)
            backend = getattr(experts.quant_method, "unquantized_backend", None)
            backend_name = getattr(backend, "value", str(backend))
            kernel_identities.add(
                (
                    str(backend_name),
                    impl.fused_experts.__class__.__name__,
                    impl.prepare_finalize.__class__.__name__,
                )
            )
            module_counts["modular_moe"] += 1
        elif class_name in {"VocabParallelEmbedding", "ParallelLMHead"} or (
            class_name == "RMSNorm" and ".self_attn." not in name and ".mlp." not in name
        ):
            before, after = timer_hooks("dense", stage="dense", layer_id=layer_id)
            state["handles"].extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
            module_counts["dense"] += 1
        if name.endswith(".mlp.gate"):
            before, after = timer_hooks("router", stage="router", layer_id=layer_id)
            state["handles"].extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
            module_counts["router_gate"] += 1

    expected_layers = int(getattr(model, "num_moe_layers", 0))
    if not expected_layers or module_counts["modular_moe"] != expected_layers:
        raise RuntimeError("discovered modular MoE layer count does not match model")
    if len(kernel_identities) != 1 or len(attention_backends) != 1:
        raise RuntimeError("heterogeneous kernel identities require a per-layer identity schema")
    moe_backend, expert_kernel, prepare_kernel = next(iter(kernel_identities))
    kernel_identity = {
        "dtype": str(runner.dtype).replace("torch.", ""),
        "attention_backend": next(iter(attention_backends)),
        "moe_backend": moe_backend,
        "moe_kernel_config": f"{expert_kernel}:{prepare_kernel}:runtime-selected",
        "dense_backend": "vllm-eager",
    }

    original_execute = state["original_execute_model"]

    def wrapped_execute_model(self: Any, scheduler_output: Any, intermediate_tensors: Any = None):
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return original_execute(scheduler_output, intermediate_tensors)
        requests = _request_metadata(self, scheduler_output)
        phases = {row["phase"] for row in requests}
        step_phase = next(iter(phases)) if len(phases) == 1 else "mixed"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        state["current"] = {
            "events": {"attention": [], "expert": [], "dense": [], "router": []},
            "fine_events": [],
            "communication_events": [],
        }
        state["routing_ids"] = {}
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(f"WB/STEP/{step_phase}")
            context = state["context"]
            workload = context.get("workload_index", "unknown")
            repeat = context.get("repeat", "unknown")
            torch.cuda.nvtx.range_push(f"WB/CONTEXT/W{workload}/R{repeat}")
        step_nvtx_open = nvtx_enabled
        context_nvtx_open = nvtx_enabled
        step_start.record()
        try:
            result = execute_with_optional_torch_trace(
                torch,
                state,
                step_phase,
                lambda: original_execute(scheduler_output, intermediate_tensors),
            )
            step_end.record()
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()
                context_nvtx_open = False
                torch.cuda.nvtx.range_pop()
                step_nvtx_open = False
            num_experts = int(self.model_config.hf_text_config.num_experts)
            layer_histograms_gpu = {}
            for layer_id, tensors in state["routing_ids"].items():
                indices = torch.cat([tensor.reshape(-1) for tensor in tensors])
                layer_histograms_gpu[layer_id] = torch.bincount(indices, minlength=num_experts)
            torch.cuda.synchronize()

            def elapsed(start: Any, end: Any) -> float:
                return float(start.elapsed_time(end) * 1000.0)

            timing = {
                category: sum(elapsed(start, end) for start, end in state["current"]["events"][category])
                for category in ("attention", "expert", "dense")
            }
            timing["step"] = elapsed(step_start, step_end)
            component_intervals = {
                category: [
                    (elapsed(step_start, start), elapsed(step_start, end))
                    for start, end in state["current"]["events"][category]
                ]
                for category in ("attention", "expert", "dense")
            }
            communication = summarize_communication_events(
                state["current"]["communication_events"],
                component_intervals,
                lambda event: elapsed(step_start, event),
                elapsed,
                {"expert": "ep"},
            )
            nested = communication["by_component_us"]
            exclusive_timing = {
                "attention_compute": max(
                    0.0, timing["attention"] - nested.get("attention", 0.0)
                ),
                "ffn_compute": max(
                    0.0, timing["expert"] - nested.get("expert", 0.0)
                ),
                "communication": communication["union_us"],
            }
            exclusive_timing["other"] = max(
                0.0,
                timing["step"]
                - exclusive_timing["attention_compute"]
                - exclusive_timing["ffn_compute"]
                - exclusive_timing["communication"],
            )
            fine_events = state["current"]["fine_events"]
            by_layer: dict[int, list[dict[str, Any]]] = {}
            non_layer_dense_us = 0.0
            for event in fine_events:
                event["start_us"] = elapsed(step_start, event["start"])
                event["end_us"] = elapsed(step_start, event["end"])
                event["latency_us"] = elapsed(event["start"], event["end"])
                if event["layer_id"] < 0:
                    if event["stage"] == "dense":
                        non_layer_dense_us += event["latency_us"]
                else:
                    by_layer.setdefault(event["layer_id"], []).append(event)

            ordered_events = sorted(fine_events, key=lambda event: event["start_us"])
            layer_ids = sorted(by_layer)
            transition_by_layer = {layer_id: 0.0 for layer_id in layer_ids}

            def nearest_layer(event_index: int) -> int:
                for distance in range(len(ordered_events)):
                    for candidate in (event_index - distance, event_index + distance):
                        if 0 <= candidate < len(ordered_events):
                            candidate_layer = ordered_events[candidate]["layer_id"]
                            if candidate_layer >= 0:
                                return candidate_layer
                return layer_ids[0]

            if ordered_events:
                transition_by_layer[nearest_layer(0)] += max(
                    0.0, ordered_events[0]["start_us"]
                )
                for index, (left, right) in enumerate(
                    zip(ordered_events, ordered_events[1:])
                ):
                    gap = max(0.0, right["start_us"] - left["end_us"])
                    transition_by_layer[nearest_layer(index + 1)] += gap
                transition_by_layer[nearest_layer(len(ordered_events) - 1)] += max(
                    0.0, timing["step"] - ordered_events[-1]["end_us"]
                )

            layer_records = []
            aggregate = [0] * num_experts
            for layer_id in sorted(by_layer):
                events = sorted(by_layer[layer_id], key=lambda event: event["start_us"])
                stage_times = {stage: 0.0 for stage in STAGES}
                for event in events:
                    stage_times[event["stage"]] += event["latency_us"]
                stage_times["transition"] = transition_by_layer[layer_id]
                histogram = layer_histograms_gpu[layer_id].cpu().tolist()
                aggregate = [left + right for left, right in zip(aggregate, histogram)]
                attention_events = [
                    event for event in events if event["stage"] == "attention"
                ]
                attention_compute_us = 0.0
                attention_collective_count = 0
                attention_valid = True
                for attention_event in attention_events:
                    candidates = []
                    for communication_event in state["current"]["communication_events"]:
                        start_us = elapsed(step_start, communication_event["start"])
                        if (
                            attention_event["start_us"] - 1e-6
                            <= start_us
                            <= attention_event["end_us"] + 1e-6
                        ):
                            candidates.append((start_us, communication_event))
                    if candidates:
                        _, first_collective = min(candidates, key=lambda item: item[0])
                        attention_compute_us += elapsed(
                            attention_event["start"], first_collective["start"]
                        )
                        attention_collective_count += len(candidates)
                    elif tp_size == 1:
                        attention_compute_us += attention_event["latency_us"]
                    else:
                        attention_valid = False
                rank_local_compute = {
                    "attention": {
                        "duration_us": attention_compute_us if attention_valid else None,
                        "valid": attention_valid,
                        "boundary": (
                            "module_start_to_first_collective_start"
                            if tp_size > 1
                            else "full_attention_module_no_tp_collective"
                        ),
                        "collective_event_count": attention_collective_count,
                        "invalid_reason": (
                            None if attention_valid else "missing_tp_collective_in_attention"
                        ),
                    },
                    "router": {
                        "duration_us": stage_times["router"],
                        "valid": True,
                        "boundary": "gate_plus_select_experts_cuda_events",
                    },
                    "expert_gemm": {
                        "duration_us": stage_times["expert_gemm"],
                        "valid": True,
                        "boundary": "modular_fused_experts_cuda_event",
                    },
                }
                layer_records.append(
                    {
                        "layer_id": layer_id,
                        "tokens_per_expert": histogram,
                        "timing_us": stage_times,
                        "rank_local_compute_us": rank_local_compute,
                        "execution_counts": {stage: 1 for stage in STAGES},
                    }
                )
            if len(layer_records) != expected_layers:
                raise RuntimeError("not every MoE layer produced a fine-grained record")
            fixed_us = non_layer_dense_us
            serial_composed_us = fixed_us + sum(
                sum(row["timing_us"].values()) for row in layer_records
            )
            state["records"].append(
                {
                    "rank": state["rank"],
                    "collector_version": COLLECTOR_VERSION,
                    "fine_grained_version": "2.0",
                    "label_source": "cuda_event",
                    "phase": step_phase,
                    "requests": requests,
                    "timing_us": timing,
                    "exclusive_timing_us": exclusive_timing,
                    "communication": communication,
                    "non_layer_dense_us": non_layer_dense_us,
                    "fixed_us": fixed_us,
                    "fixed_source": "direct_non_layer_dense_cuda_events",
                    "serial_composed_us": serial_composed_us,
                    "composition_overrun_us": max(0.0, serial_composed_us - timing["step"]),
                    "uncovered_us": max(0.0, timing["step"] - serial_composed_us),
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "tokens_per_expert": aggregate,
                    "layer_tokens_per_expert": [row["tokens_per_expert"] for row in layer_records],
                    "layer_records": layer_records,
                    "observed_moe_layers": len(layer_records),
                    "module_counts": module_counts,
                    "kernel_identity": kernel_identity,
                    "rank_local_compute_schema": "rank-local-compute/v1",
                    "rank_local_compute_attention_valid": all(
                        layer["rank_local_compute_us"]["attention"]["valid"]
                        for layer in layer_records
                    ),
                    "nvtx_audit_enabled": nvtx_enabled,
                }
            )
            return result
        finally:
            if context_nvtx_open:
                torch.cuda.nvtx.range_pop()
            if step_nvtx_open:
                torch.cuda.nvtx.range_pop()
            state["current"] = None
            state["routing_ids"] = {}

    runner.execute_model = types.MethodType(wrapped_execute_model, runner)
    worker._moe_component_collector = state
    return {
        "installed": True,
        "fine_grained_version": "2.0",
        "module_counts": module_counts,
        "kernel_identity": kernel_identity,
        "communication_hooks": communication_install,
    }


def get_worker_records(worker: Any) -> list[dict[str, Any]]:
    state = getattr(worker, "_moe_component_collector", None)
    if state is None:
        raise RuntimeError("MoE component collector is not installed")
    return list(state["records"])


def clear_worker_records(worker: Any) -> dict[str, int]:
    state = getattr(worker, "_moe_component_collector", None)
    if state is None:
        raise RuntimeError("MoE component collector is not installed")
    count = len(state["records"])
    state["records"].clear()
    return {"cleared": count}


def uninstall_worker_collector(worker: Any) -> dict[str, bool]:
    state = getattr(worker, "_moe_component_collector", None)
    if state is None:
        return {"uninstalled": False}
    worker.model_runner.execute_model = state["original_execute_model"]
    for obj, attribute, original in reversed(state["patched"]):
        setattr(obj, attribute, original)
    for handle in state["handles"]:
        handle.remove()
    uninstall_communication_hooks(state)
    del worker._moe_component_collector
    return {"uninstalled": True}


class CollectorWorkerExtension:
    def moe_collector_install(self) -> dict[str, Any]:
        return install_worker_collector(self)

    def moe_collector_get_records(self) -> list[dict[str, Any]]:
        return get_worker_records(self)

    def moe_collector_clear_records(self) -> dict[str, int]:
        return clear_worker_records(self)

    def moe_collector_set_context(self, context: dict[str, Any]) -> dict[str, Any]:
        state = getattr(self, "_moe_component_collector", None)
        if state is None:
            raise RuntimeError("MoE component collector is not installed")
        state["context"] = dict(context)
        return dict(context)

    def moe_collector_uninstall(self) -> dict[str, bool]:
        return uninstall_worker_collector(self)

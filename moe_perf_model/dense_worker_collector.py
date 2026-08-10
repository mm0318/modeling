from __future__ import annotations

import os
import re
import types
from typing import Any

from .collector import COLLECTOR_VERSION
from .communication_hooks import (
    install_communication_hooks,
    summarize_communication_events,
    uninstall_communication_hooks,
)
from .worker_collector import _request_metadata
from .torch_trace import execute_with_optional_torch_trace


def _layer_id(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def install_worker_collector(worker: Any) -> dict[str, Any]:
    """Install Qwen3 dense Attention/MLP CUDA-event hooks."""
    import torch

    if hasattr(worker, "_dense_component_collector"):
        raise RuntimeError("Dense component collector is already installed")
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
        "current": None,
        "context": {},
        "stacks": {},
        "original_execute_model": runner.execute_model,
    }
    communication_install = install_communication_hooks(state, include_ep=False)

    def timer_hooks(category: str, layer_id: int):
        def before(module: Any, inputs: Any) -> None:
            if state["current"] is None:
                return
            event = torch.cuda.Event(enable_timing=True)
            if nvtx_enabled:
                torch.cuda.nvtx.range_push(f"DENSE/{category.upper()}/L{layer_id}")
            trace_range = None
            if torch_trace_enabled:
                trace_range = torch.profiler.record_function(
                    f"WB/{category}/L{layer_id}"
                )
                trace_range.__enter__()
            event.record()
            state["stacks"].setdefault((id(module), category), []).append(
                (event, trace_range)
            )

        def after(module: Any, inputs: Any, output: Any) -> None:
            stack = state["stacks"].get((id(module), category), [])
            if state["current"] is None or not stack:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()
            start, trace_range = stack.pop()
            if trace_range is not None:
                trace_range.__exit__(None, None, None)
            state["current"]["events"][category].append((layer_id, start, end))

        return before, after

    counts = {"attention": 0, "mlp": 0}
    attention_backends = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        layer_id = _layer_id(name)
        category = None
        if class_name == "Qwen3Attention":
            category = "attention"
            attention_impl = getattr(getattr(module, "attn", None), "impl", None)
            attention_backends.add(
                attention_impl.__class__.__name__ if attention_impl is not None else "FLASH_ATTN"
            )
        elif class_name in {"Qwen2MLP", "Qwen3MLP"} and name.endswith(".mlp"):
            category = "mlp"
        if category is not None:
            before, after = timer_hooks(category, layer_id)
            state["handles"].extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
            counts[category] += 1

    expected_layers = int(getattr(runner.model_config.hf_text_config, "num_hidden_layers", 0))
    if expected_layers <= 0 or counts != {"attention": expected_layers, "mlp": expected_layers}:
        raise RuntimeError(
            f"dense module discovery mismatch: expected {expected_layers}, observed {counts}"
        )
    if len(attention_backends) != 1:
        raise RuntimeError("heterogeneous attention backends require a per-layer schema")

    kernel_identity = {
        "dtype": str(runner.dtype).replace("torch.", ""),
        "attention_backend": next(iter(attention_backends)),
        "ffn_backend": "vllm-dense-mlp",
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
            "events": {"attention": [], "mlp": []},
            "communication_events": [],
        }
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
            torch.cuda.synchronize()

            def elapsed(start: Any, end: Any) -> float:
                return float(start.elapsed_time(end) * 1000.0)

            layer_map: dict[int, dict[str, float]] = {
                layer_id: {"attention": 0.0, "mlp": 0.0}
                for layer_id in range(expected_layers)
            }
            timing = {"attention": 0.0, "mlp": 0.0}
            for category in timing:
                for layer_id, start, end in state["current"]["events"][category]:
                    latency = elapsed(start, end)
                    timing[category] += latency
                    layer_map[layer_id][category] += latency
            timing["step"] = elapsed(step_start, step_end)
            component_intervals = {
                category: [
                    (elapsed(step_start, start), elapsed(step_start, end))
                    for _, start, end in state["current"]["events"][category]
                ]
                for category in ("attention", "mlp")
            }
            communication = summarize_communication_events(
                state["current"]["communication_events"],
                component_intervals,
                lambda event: elapsed(step_start, event),
                elapsed,
            )
            nested = communication["by_component_us"]
            exclusive_timing = {
                "attention_compute": max(
                    0.0, timing["attention"] - nested.get("attention", 0.0)
                ),
                "ffn_compute": max(0.0, timing["mlp"] - nested.get("mlp", 0.0)),
                "communication": communication["union_us"],
            }
            exclusive_timing["other"] = max(
                0.0,
                timing["step"]
                - exclusive_timing["attention_compute"]
                - exclusive_timing["ffn_compute"]
                - exclusive_timing["communication"],
            )
            if any(
                layer_map[layer_id][category] <= 0
                for layer_id in layer_map
                for category in ("attention", "mlp")
            ):
                raise RuntimeError("not every dense layer produced Attention and MLP timings")
            state["records"].append(
                {
                    "rank": state["rank"],
                    "model_family": "dense",
                    "collector_version": COLLECTOR_VERSION,
                    "dense_component_version": "1.0",
                    "label_source": "cuda_event",
                    "phase": step_phase,
                    "requests": requests,
                    "timing_us": timing,
                    "exclusive_timing_us": exclusive_timing,
                    "communication": communication,
                    "layer_records": [
                        {"layer_id": layer_id, "timing_us": layer_map[layer_id]}
                        for layer_id in sorted(layer_map)
                    ],
                    "observed_dense_layers": len(layer_map),
                    "module_counts": counts,
                    "kernel_identity": kernel_identity,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
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

    runner.execute_model = types.MethodType(wrapped_execute_model, runner)
    worker._dense_component_collector = state
    return {
        "installed": True,
        "model_family": "dense",
        "dense_component_version": "1.0",
        "module_counts": counts,
        "kernel_identity": kernel_identity,
        "communication_hooks": communication_install,
    }


def _state(worker: Any) -> dict[str, Any]:
    value = getattr(worker, "_dense_component_collector", None)
    if value is None:
        raise RuntimeError("Dense component collector is not installed")
    return value


class DenseCollectorWorkerExtension:
    def moe_collector_install(self) -> dict[str, Any]:
        return install_worker_collector(self)

    def moe_collector_get_records(self) -> list[dict[str, Any]]:
        return list(_state(self)["records"])

    def moe_collector_clear_records(self) -> dict[str, int]:
        state = _state(self)
        count = len(state["records"])
        state["records"].clear()
        return {"cleared": count}

    def moe_collector_set_context(self, context: dict[str, Any]) -> dict[str, Any]:
        _state(self)["context"] = dict(context)
        return dict(context)

    def moe_collector_uninstall(self) -> dict[str, bool]:
        state = getattr(self, "_dense_component_collector", None)
        if state is None:
            return {"uninstalled": False}
        self.model_runner.execute_model = state["original_execute_model"]
        uninstall_communication_hooks(state)
        for handle in state["handles"]:
            handle.remove()
        del self._dense_component_collector
        return {"uninstalled": True}

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def execute_with_optional_torch_trace(
    torch: Any,
    state: dict[str, Any],
    phase: str,
    execute: Callable[[], Any],
) -> Any:
    """Capture real CUDA kernels after distributed initialization, when requested."""
    trace_dir = os.environ.get("MOE_WHITEBOX_TORCH_TRACE_DIR")
    context = state.get("context", {})
    repeat = context.get("repeat")
    if not trace_dir or not isinstance(repeat, int):
        return execute()

    context_key = (context.get("workload_index"), repeat)
    if state.get("trace_context_key") != context_key:
        state["trace_context_key"] = context_key
        state["trace_step_index"] = 0
    step_index = int(state.get("trace_step_index", 0))
    state["trace_step_index"] = step_index + 1
    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{state['rank']}_w{context.get('workload_index', 'unknown')}_"
        f"r{repeat}_s{step_index}_{phase}.json"
    )
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        result = execute()
    profiler.export_chrome_trace(str(output_dir / filename))
    return result

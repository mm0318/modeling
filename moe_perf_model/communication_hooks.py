from __future__ import annotations

import os
import types
from typing import Any, Iterable


COLLECTIVE_METHODS = {
    "all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "all_gatherv": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reduce_scatterv": "reduce_scatter",
}


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0


def install_communication_hooks(
    state: dict[str, Any], *, include_ep: bool
) -> dict[str, Any]:
    """Wrap the real vLLM TP/EP group calls during active collector steps."""
    import torch
    from vllm.distributed.parallel_state import get_ep_group, get_tp_group

    tp_group = get_tp_group()
    groups: list[tuple[str, Any]] = [("tp", tp_group)]
    if include_ep:
        ep_group = get_ep_group()
        groups = (
            [("tp_ep_alias", tp_group)]
            if id(ep_group) == id(tp_group)
            else [("tp", tp_group), ("ep", ep_group)]
        )
    installed = []
    seen: set[tuple[int, str]] = set()
    state.setdefault("communication_patched", [])

    for group_kind, group in groups:
        for method_name, collective_type in COLLECTIVE_METHODS.items():
            identity = (id(group), method_name)
            if identity in seen or not hasattr(group, method_name):
                continue
            seen.add(identity)
            original = getattr(group, method_name)

            def wrapped(
                self: Any,
                *args: Any,
                _original: Any = original,
                _group_kind: str = group_kind,
                _collective_type: str = collective_type,
                _method_name: str = method_name,
                **kwargs: Any,
            ):
                current = state.get("current")
                if current is None:
                    return _original(*args, **kwargs)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                trace_range = None
                if os.environ.get("MOE_WHITEBOX_TORCH_COMPONENT_RANGES"):
                    trace_range = torch.profiler.record_function(
                        f"WB/communication/{_group_kind}/{_collective_type}"
                    )
                    trace_range.__enter__()
                start.record()
                try:
                    result = _original(*args, **kwargs)
                    end.record()
                finally:
                    if trace_range is not None:
                        trace_range.__exit__(None, None, None)
                current.setdefault("communication_events", []).append(
                    {
                        "group": _group_kind,
                        "collective": _collective_type,
                        "method": _method_name,
                        "message_bytes": _tensor_bytes(args[0]) if args else 0,
                        "start": start,
                        "end": end,
                    }
                )
                return result

            setattr(group, method_name, types.MethodType(wrapped, group))
            state["communication_patched"].append((group, method_name, original))
            installed.append(f"{group_kind}:{method_name}")
    return {"installed_methods": installed, "include_ep": include_ep}


def uninstall_communication_hooks(state: dict[str, Any]) -> None:
    for group, method_name, original in reversed(state.get("communication_patched", [])):
        setattr(group, method_name, original)
    state["communication_patched"] = []


def summarize_communication_events(
    events: Iterable[dict[str, Any]],
    component_intervals: dict[str, list[tuple[float, float]]],
    elapsed_from_step: Any,
    elapsed: Any,
    group_override_by_owner: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make communication work and module-exclusive timing auditable."""
    rows = []
    by_group: dict[str, float] = {}
    by_collective: dict[str, float] = {}
    by_component: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        start_us = float(elapsed_from_step(event["start"]))
        end_us = float(elapsed_from_step(event["end"]))
        latency_us = float(elapsed(event["start"], event["end"]))
        owner = "other"
        midpoint = (start_us + end_us) / 2.0
        for component, candidates in component_intervals.items():
            if any(left <= midpoint <= right for left, right in candidates):
                owner = component
                break
        effective_group = event["group"]
        if group_override_by_owner and owner in group_override_by_owner:
            effective_group = group_override_by_owner[owner]
        if effective_group == "tp_ep_alias":
            effective_group = "ep" if owner in {"expert", "ffn"} else "tp"
        row = {
            "group": effective_group,
            "raw_group": event["group"],
            "collective": event["collective"],
            "method": event["method"],
            "message_bytes": int(event["message_bytes"]),
            "start_us": start_us,
            "end_us": end_us,
            "latency_us": latency_us,
            "owner": owner,
        }
        rows.append(row)
        intervals.append((start_us, end_us))
        by_group[row["group"]] = by_group.get(row["group"], 0.0) + latency_us
        by_collective[row["collective"]] = (
            by_collective.get(row["collective"], 0.0) + latency_us
        )
        by_component[owner] = by_component.get(owner, 0.0) + latency_us

    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    union_us = sum(end - start for start, end in merged)
    work_us = sum(row["latency_us"] for row in rows)
    return {
        "events": rows,
        "event_count": len(rows),
        "work_us": work_us,
        "union_us": union_us,
        "overlap_us": max(0.0, work_us - union_us),
        "by_group_us": by_group,
        "by_collective_us": by_collective,
        "by_component_us": by_component,
    }

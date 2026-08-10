from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .model import ComponentModel
from .schema import InferenceConfig, SchemaError
from .execution_plan import StepExecutionPlan
from .whitebox_model import WhiteboxMoEModel


@dataclass(frozen=True)
class RequestEvent:
    request_id: str
    arrival_us: float
    emitted_tokens: int
    finished: bool

    @classmethod
    def from_dict(cls, raw: Any) -> "RequestEvent":
        if not isinstance(raw, dict):
            raise SchemaError("request event must be a JSON object")
        request_id = raw.get("request_id")
        arrival = raw.get("arrival_us")
        emitted = raw.get("emitted_tokens")
        finished = raw.get("finished")
        if not isinstance(request_id, str) or not request_id:
            raise SchemaError("request_id must be a non-empty string")
        if isinstance(arrival, bool) or not isinstance(arrival, (int, float)):
            raise SchemaError("arrival_us must be numeric")
        if not math.isfinite(float(arrival)) or float(arrival) < 0:
            raise SchemaError("arrival_us must be finite and non-negative")
        if isinstance(emitted, bool) or not isinstance(emitted, int) or emitted < 0:
            raise SchemaError("emitted_tokens must be a non-negative integer")
        if not isinstance(finished, bool):
            raise SchemaError("finished must be boolean")
        return cls(request_id, float(arrival), emitted, finished)


@dataclass(frozen=True)
class ReplayStep:
    step_id: int
    config: InferenceConfig
    request_events: tuple[RequestEvent, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "ReplayStep":
        if not isinstance(raw, dict):
            raise SchemaError("replay step must be a JSON object")
        step_id = raw.get("step_id")
        events = raw.get("request_events")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise SchemaError("step_id must be a non-negative integer")
        if not isinstance(events, list) or not events:
            raise SchemaError("request_events must be a non-empty list")
        return cls(
            step_id=step_id,
            config=InferenceConfig.from_dict(raw.get("config")),
            request_events=tuple(RequestEvent.from_dict(event) for event in events),
        )


@dataclass(frozen=True)
class WhiteboxReplayStep:
    step_id: int
    plan: StepExecutionPlan
    request_events: tuple[RequestEvent, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "WhiteboxReplayStep":
        if not isinstance(raw, dict):
            raise SchemaError("white-box replay step must be an object")
        step_id = raw.get("step_id")
        events = raw.get("request_events")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise SchemaError("step_id must be a non-negative integer")
        if not isinstance(events, list) or not events:
            raise SchemaError("request_events must be a non-empty list")
        return cls(
            step_id,
            StepExecutionPlan.from_dict(raw.get("plan")),
            tuple(RequestEvent.from_dict(event) for event in events),
        )


@dataclass
class _RequestState:
    arrival_us: float
    token_times_us: list[float] = field(default_factory=list)
    finished_us: float | None = None


def _percentile(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def load_replay_steps(path: str | Path) -> list[ReplayStep]:
    steps = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                steps.append(ReplayStep.from_dict(json.loads(line)))
            except (json.JSONDecodeError, SchemaError) as error:
                raise SchemaError(f"{path}:{line_number}: {error}") from error
    if not steps:
        raise SchemaError("replay trace contains no steps")
    if any(right.step_id <= left.step_id for left, right in zip(steps, steps[1:])):
        raise SchemaError("replay step_id values must be strictly increasing")
    return steps


def load_whitebox_replay_steps(path: str | Path) -> list[WhiteboxReplayStep]:
    steps = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                steps.append(WhiteboxReplayStep.from_dict(json.loads(line)))
            except (json.JSONDecodeError, SchemaError) as error:
                raise SchemaError(f"{path}:{line_number}: {error}") from error
    if not steps:
        raise SchemaError("white-box replay trace contains no steps")
    if any(right.step_id <= left.step_id for left, right in zip(steps, steps[1:])):
        raise SchemaError("replay step_id values must be strictly increasing")
    return steps


def _replay_with_predictor(
    predict: Any, steps: Iterable[Any], workload_attribute: str
) -> dict[str, Any]:
    steps = list(steps)
    if not steps:
        raise ValueError("scheduler replay requires at least one step")

    current_us = 0.0
    requests: dict[str, _RequestState] = {}
    step_rows = []
    for step in steps:
        earliest_new_arrival = min(event.arrival_us for event in step.request_events)
        current_us = max(current_us, earliest_new_arrival)
        start_us = current_us
        profile = predict(getattr(step, workload_attribute))
        end_us = start_us + profile.step_us
        for event in step.request_events:
            state = requests.get(event.request_id)
            if state is None:
                state = _RequestState(arrival_us=event.arrival_us)
                requests[event.request_id] = state
            elif abs(state.arrival_us - event.arrival_us) > 1e-6:
                raise SchemaError(
                    f"request {event.request_id} has inconsistent arrival_us"
                )
            if state.finished_us is not None:
                raise SchemaError(f"request {event.request_id} appears after completion")
            state.token_times_us.extend([end_us] * event.emitted_tokens)
            if event.finished:
                state.finished_us = end_us
        step_rows.append(
            {
                "step_id": step.step_id,
                "start_us": start_us,
                "end_us": end_us,
                "profile": profile.to_dict(),
            }
        )
        current_us = end_us

    request_rows = []
    ttfts = []
    tpots = []
    e2es = []
    completed = 0
    total_output_tokens = 0
    for request_id, state in sorted(requests.items()):
        token_count = len(state.token_times_us)
        total_output_tokens += token_count
        ttft = (
            state.token_times_us[0] - state.arrival_us if state.token_times_us else None
        )
        tpot = (
            (state.token_times_us[-1] - state.token_times_us[0]) / (token_count - 1)
            if token_count >= 2
            else None
        )
        e2e = (
            state.finished_us - state.arrival_us if state.finished_us is not None else None
        )
        if ttft is not None:
            ttfts.append(ttft)
        if tpot is not None:
            tpots.append(tpot)
        if e2e is not None:
            e2es.append(e2e)
            completed += 1
        request_rows.append(
            {
                "request_id": request_id,
                "arrival_us": state.arrival_us,
                "output_tokens": token_count,
                "finished": state.finished_us is not None,
                "ttft_us": ttft,
                "tpot_us": tpot,
                "e2e_us": e2e,
            }
        )

    first_arrival = min(state.arrival_us for state in requests.values())
    elapsed_us = max(current_us - first_arrival, 1e-9)
    return {
        "summary": {
            "step_count": len(step_rows),
            "request_count": len(request_rows),
            "completed_requests": completed,
            "elapsed_us": elapsed_us,
            "mean_ttft_us": float(np.mean(ttfts)) if ttfts else None,
            "p50_ttft_us": _percentile(ttfts, 0.50),
            "p90_ttft_us": _percentile(ttfts, 0.90),
            "mean_tpot_us": float(np.mean(tpots)) if tpots else None,
            "p50_tpot_us": _percentile(tpots, 0.50),
            "p90_tpot_us": _percentile(tpots, 0.90),
            "mean_e2e_us": float(np.mean(e2es)) if e2es else None,
            "output_token_throughput_per_s": total_output_tokens * 1e6 / elapsed_us,
            "request_throughput_per_s": completed * 1e6 / elapsed_us,
            "max_predicted_memory_bytes": max(
                row["profile"]["memory_bytes"] for row in step_rows
            ),
        },
        "requests": request_rows,
        "steps": step_rows,
        "semantics": {
            "execution": "offline serial scheduler-step replay",
            "token_visibility": "all tokens emitted by a step become visible at step end",
            "tpot": "(last_token_time - first_token_time) / (output_tokens - 1)",
        },
    }


def replay_steps(model: ComponentModel, steps: Iterable[ReplayStep]) -> dict[str, Any]:
    if not isinstance(model, ComponentModel):
        raise TypeError("scheduler replay requires a component artifact")
    return _replay_with_predictor(model.predict, steps, "config")


def replay_whitebox_steps(
    model: WhiteboxMoEModel, steps: Iterable[WhiteboxReplayStep]
) -> dict[str, Any]:
    if not isinstance(model, WhiteboxMoEModel):
        raise TypeError("white-box replay requires a whitebox-moe-v3 artifact")
    result = _replay_with_predictor(model.predict, steps, "plan")
    result["semantics"]["latency_model"] = "white-box timing oracle + execution DAG"
    return result

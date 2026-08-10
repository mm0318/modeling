from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .event_engine import DiscreteEventEngine
from .execution_plan import StepExecutionPlan
from .parallel_forward import ParallelForwardGraphCompiler
from .whitebox_model import WhiteboxMoEModel
from .fixed_config_oracle import FixedConfigForwardOracle, features_from_plan


@dataclass(frozen=True)
class RequestStep:
    request_id: str
    phase: str
    prompt_length: int
    prior_context_length: int
    scheduled_tokens: int
    context_length: int


@dataclass(frozen=True)
class SchedulerStepContext:
    phase: str
    requests: tuple[RequestStep, ...]
    realized_batch_size: int
    scheduled_tokens: int
    n_q: int
    total_kv_tokens: int
    kv_mean: float
    kv_max: int
    mid_prefill_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SchedulerStepAdapter:
    """Stateful adapter from vLLM SchedulerOutput to model workload fields."""

    def __init__(self) -> None:
        self._prompt_lengths: dict[str, int] = {}
        self._computed: dict[str, int] = {}

    def adapt(self, output: Any) -> SchedulerStepContext:
        scheduled = dict(output.num_scheduled_tokens)
        new_by_id = {row.req_id: row for row in output.scheduled_new_reqs}
        for request_id, row in new_by_id.items():
            prompt_length = len(row.prompt_token_ids or ())
            self._prompt_lengths[request_id] = prompt_length
            self._computed[request_id] = int(getattr(row, "num_computed_tokens", 0) or 0)

        cached = output.scheduled_cached_reqs
        cached_ids = list(getattr(cached, "req_ids", ()))
        cached_computed = list(getattr(cached, "num_computed_tokens", ()))
        for request_id, computed in zip(cached_ids, cached_computed):
            self._computed[request_id] = int(computed)

        rows: list[RequestStep] = []
        mid_prefill: list[str] = []
        for request_id, count in scheduled.items():
            prior = self._computed.get(request_id, 0)
            prompt_length = self._prompt_lengths.get(request_id, prior)
            phase = "prefill" if prior < prompt_length else "decode"
            context = prior + int(count)
            if phase == "prefill" and context < prompt_length:
                mid_prefill.append(request_id)
            rows.append(RequestStep(
                request_id=request_id,
                phase=phase,
                prompt_length=prompt_length,
                prior_context_length=prior,
                scheduled_tokens=int(count),
                context_length=max(1, context),
            ))
            self._computed[request_id] = context

        finished = set(getattr(output, "finished_req_ids", ()))
        for request_id in finished:
            self._prompt_lengths.pop(request_id, None)
            self._computed.pop(request_id, None)
        if not rows:
            raise ValueError("scheduler step has no scheduled requests")
        phases = {row.phase for row in rows}
        phase = next(iter(phases)) if len(phases) == 1 else "mixed"
        contexts = [row.context_length for row in rows]
        counts = [row.scheduled_tokens for row in rows]
        return SchedulerStepContext(
            phase=phase,
            requests=tuple(rows),
            realized_batch_size=len(rows),
            scheduled_tokens=sum(counts),
            n_q=max(counts),
            total_kv_tokens=sum(contexts),
            kv_mean=sum(contexts) / len(contexts),
            kv_max=max(contexts),
            mid_prefill_ids=tuple(mid_prefill),
        )


def _rescale_histogram(values: tuple[int, ...], target: int) -> list[int]:
    total = sum(values)
    if target < 0 or total <= 0:
        raise ValueError("routing histogram cannot be rescaled")
    exact = [value * target / total for value in values]
    result = [math.floor(value) for value in exact]
    remainder = target - sum(result)
    order = sorted(range(len(values)), key=lambda index: exact[index] - result[index], reverse=True)
    for index in order[:remainder]:
        result[index] += 1
    return result


class ProfileReplayRoutingProvider:
    """Replay and scale nearest measured per-layer routing histograms."""

    def __init__(self, raw_steps_path: str | Path):
        self.source_path = str(Path(raw_steps_path).resolve())
        self.rows: list[tuple[dict[str, Any], StepExecutionPlan]] = []
        with Path(raw_steps_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw = json.loads(line)
                    self.rows.append((raw, StepExecutionPlan.from_raw_step(raw)))
        if not self.rows:
            raise ValueError("routing replay source is empty")

    def make_plan(self, context: SchedulerStepContext) -> tuple[StepExecutionPlan, dict[str, Any]]:
        candidates = [(raw, plan) for raw, plan in self.rows if plan.phase == context.phase]
        if not candidates:
            raise ValueError(f"routing replay has no {context.phase} anchor")
        raw, anchor = min(
            candidates,
            key=lambda item: abs(math.log2(max(1, item[1].realized_batch_size))
                                 - math.log2(max(1, context.realized_batch_size)))
            + abs(math.log2(max(1, item[1].scheduled_tokens))
                                 - math.log2(max(1, context.scheduled_tokens)))
            + abs(math.log2(max(1, item[1].total_kv_tokens))
                  - math.log2(max(1, context.total_kv_tokens)))
            + abs(math.log2(max(1, item[1].n_q))
                  - math.log2(max(1, context.n_q))),
        )
        target = context.scheduled_tokens * anchor.top_k
        plan_raw = anchor.to_dict()
        plan_raw.update({
            "phase": context.phase,
            "realized_batch_size": context.realized_batch_size,
            "scheduled_tokens": context.scheduled_tokens,
            "n_q": context.n_q,
            "total_kv_tokens": context.total_kv_tokens,
            "kv_mean": context.kv_mean,
            "kv_max": context.kv_max,
            "layers": [
                {
                    "layer_id": layer.layer_id,
                    "tokens_per_expert": _rescale_histogram(layer.tokens_per_expert, target),
                    "execution_counts": layer.execution_counts,
                }
                for layer in anchor.layers
            ],
        })
        plan = StepExecutionPlan.from_dict(plan_raw)
        return plan, {
            "routing_source": "profile_replay",
            "formal_online_routing": False,
            "source_path": self.source_path,
            "anchor_step_index": raw.get("step_index"),
            "anchor_workload_index": raw.get("workload_index"),
            "anchor_phase": anchor.phase,
            "anchor_scheduled_tokens": anchor.scheduled_tokens,
            "anchor_total_kv_tokens": anchor.total_kv_tokens,
            "target_routed_tokens_per_layer": target,
        }


class JsonlTraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class VllmForwardHijackHook:
    """Executor hook preserving vLLM control flow while virtualizing forward."""

    def __init__(
        self,
        *,
        model: WhiteboxMoEModel | None,
        routing_provider: ProfileReplayRoutingProvider,
        compiler: ParallelForwardGraphCompiler | None,
        trace_path: str | Path,
        realtime: bool = False,
        filler_token_id: int = 100,
        fixed_config_oracle: FixedConfigForwardOracle | None = None,
        record_auxiliary_graph: bool = False,
    ):
        self.model = model
        self.routing_provider = routing_provider
        self.compiler = compiler
        self.trace_writer = JsonlTraceWriter(trace_path)
        self.realtime = realtime
        self.filler_token_id = filler_token_id
        self.fixed_config_oracle = fixed_config_oracle
        self.record_auxiliary_graph = record_auxiliary_graph
        if self.fixed_config_oracle is None and self.model is None:
            raise ValueError("forward hijack requires a fixed-config or white-box oracle")
        if self.record_auxiliary_graph and self.compiler is None:
            raise ValueError("auxiliary execution graph requires a compiler")
        self.adapter = SchedulerStepAdapter()
        self.engine = DiscreteEventEngine()
        self._sample_future: Future | None = None
        self._pending_output: Any = None
        self._gpu_free_time = 0.0
        self._step_index = 0

    @property
    def is_enabled(self) -> bool:
        return True

    def should_use_oracle(self, scheduler_output: Any) -> bool:
        return scheduler_output.total_num_scheduled_tokens > 0

    def has_pending_future(self) -> bool:
        return self._sample_future is not None

    def get_sample_future(self) -> Future:
        future = self._sample_future
        self._sample_future = None
        if future is None:
            raise RuntimeError("no pending sample future")
        return future

    def has_pending_output(self) -> bool:
        return self._pending_output is not None

    def get_pending_output(self) -> Any:
        output = self._pending_output
        self._pending_output = None
        return output

    def create_delayed_future(self, scheduler_output: Any, non_block: bool = False) -> Any:
        context = self.adapter.adapt(scheduler_output)
        plan, routing_provenance = self.routing_provider.make_plan(context)
        fixed_prediction = None
        report = None
        profile = None
        if self.fixed_config_oracle is not None:
            prefill_requests = sum(row.phase == "prefill" for row in context.requests)
            prefill_tokens = sum(
                row.scheduled_tokens for row in context.requests if row.phase == "prefill"
            )
            phase, features = features_from_plan(
                plan,
                prefill_requests=prefill_requests,
                decode_requests=context.realized_batch_size - prefill_requests,
                prefill_scheduled_tokens=prefill_tokens,
                expert_parallel_size=self.fixed_config_oracle.identity["expert_parallel_size"],
            )
            fixed_prediction = self.fixed_config_oracle.predict_features(phase, features)
            if fixed_prediction.step_us is None:
                raise ValueError(
                    "fixed-config oracle returned not_predictable: "
                    f"{fixed_prediction.support_reason}; nearest_distance="
                    f"{fixed_prediction.nearest_distance:.6f}, threshold="
                    f"{fixed_prediction.coverage_threshold:.6f}. Profile this region first."
                )
        if self.fixed_config_oracle is None or self.record_auxiliary_graph:
            if self.model is None:
                raise ValueError("auxiliary execution graph requires white-box model")
            profile = self.model.predict(plan)
            graph = self.compiler.compile(profile, plan)
            report = self.engine.simulate(graph)
        predicted_step_us = (
            fixed_prediction.step_us if fixed_prediction is not None else report.step_us
        )
        fake_output = self._create_fake_output(scheduler_output, context)
        timing_provenance = (
            self.fixed_config_oracle.provenance
            if self.fixed_config_oracle is not None else profile.artifact_provenance
        )
        self.trace_writer.write({
            "schema_version": "vllm-forward-hijack/v1",
            "step_index": self._step_index,
            "scheduler_context": context.to_dict(),
            "routing_provenance": routing_provenance,
            "timing_provenance": timing_provenance,
            "parallel_layout": (
                self.compiler.layout.to_dict() if self.compiler is not None else {
                    "layout_source": "fixed_config_identity",
                    "tensor_parallel_size": self.fixed_config_oracle.identity["tensor_parallel_size"],
                    "expert_parallel_size": self.fixed_config_oracle.identity["expert_parallel_size"],
                    "world_size": self.fixed_config_oracle.identity["rank_count"],
                    "rank_group_semantics": "same_ranks_participate_in_tp_and_ep",
                }
            ),
            "execution_plan": plan.to_dict(),
            "predicted_step_us": predicted_step_us,
            "latency_prediction": fixed_prediction.to_dict() if fixed_prediction else None,
            "execution_report": report.to_dict() if report else None,
            "fake_output": {"filler_token_id": self.filler_token_id,
                            "mid_prefill_ids": list(context.mid_prefill_ids)},
            "formal": bool(timing_provenance.get("formal_hardware_prediction", False))
                      and routing_provenance["formal_online_routing"],
        })
        self._step_index += 1
        latency_s = predicted_step_us / 1e6
        if not non_block:
            if self.realtime and latency_s >= 0.001:
                time.sleep(latency_s)
            return fake_output
        execution_future: Future = Future()
        execution_future.set_result(None)
        sample_future: Future = Future()
        if self.realtime and latency_s >= 0.001:
            now = time.perf_counter()
            end = max(now, self._gpu_free_time) + latency_s
            self._gpu_free_time = end
            timer = threading.Timer(end - now, lambda: sample_future.set_result(fake_output))
            timer.daemon = True
            timer.start()
        else:
            sample_future.set_result(fake_output)
        self._sample_future = sample_future
        return execution_future

    def _create_fake_output(self, scheduler_output: Any, context: SchedulerStepContext) -> Any:
        import numpy as np
        from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        mid_prefill = set(context.mid_prefill_ids)
        sampled = [[] if request_id in mid_prefill else [self.filler_token_id]
                   for request_id in req_ids]
        token_values = [tokens[0] for tokens in sampled if tokens]
        logprobs = None
        if token_values:
            logprobs = LogprobsLists(
                logprob_token_ids=np.array([[token] for token in token_values], dtype=np.int32),
                logprobs=np.full((len(token_values), 1), -0.1, dtype=np.float32),
                sampled_token_ranks=np.zeros(len(token_values), dtype=np.int32),
            )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=logprobs,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )

    def shutdown(self) -> None:
        return None

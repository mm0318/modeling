from __future__ import annotations

import pytest

from moe_perf_model.evaluation import evaluate_leave_one_workload_out, evaluate_strict_generalization
from moe_perf_model.execution_plan import StepExecutionPlan
from moe_perf_model.operator_db import OperatorResponseDatabase, OperatorSample
from moe_perf_model.whitebox_model import WhiteboxMoEModel
from moe_perf_model.replay import RequestEvent, WhiteboxReplayStep, replay_whitebox_steps


def _histogram(scheduled: int, layer: int) -> list[int]:
    total = scheduled * 2
    values = [0, 0, 0, 0]
    for index in range(total):
        values[(index + layer) % 4] += 1
    return values


def _fine_raw(workload: int, phase: str, scheduled: int, batch: int) -> dict:
    per_request = [scheduled // batch] * batch
    for index in range(scheduled % batch):
        per_request[index] += 1
    requests = []
    for index, query in enumerate(per_request):
        prior = 0 if phase == "prefill" else 16 + workload
        prompt = query if phase == "prefill" else 16
        requests.append(
            {
                "request_id": f"{workload}-{phase}-{index}",
                "phase": phase,
                "input_length": prompt,
                "prior_context_length": prior,
                "scheduled_tokens": query,
                "context_length": prior + query,
            }
        )
    layers = []
    explicit = 0.0
    for layer_id in range(2):
        timing = {
            "attention": 8.0 + scheduled + 0.5 * layer_id,
            "router": 2.0 + 0.2 * scheduled,
            "dispatch": 1.0 + 0.1 * scheduled,
            "expert_gemm": 10.0 + 0.8 * scheduled + layer_id,
            "combine": 1.5 + 0.1 * scheduled,
            "dense": 3.0 + 0.3 * scheduled,
            "transition": 0.5 + 0.05 * scheduled,
        }
        explicit += sum(timing.values())
        layers.append(
            {
                "layer_id": layer_id,
                "tokens_per_expert": _histogram(scheduled, layer_id),
                "timing_us": timing,
                "execution_counts": {
                    "attention": 1,
                    "router": 1,
                    "dispatch": 1,
                    "expert_gemm": 1,
                    "combine": 1,
                    "dense": 1,
                    "transition": 1,
                },
            }
        )
    fixed = 4.0 + batch
    return {
        "collector_version": "0.1.0",
        "fine_grained_version": "2.0",
        "label_source": "synthetic_fixture",
        "fixed_source": "synthetic_fixture_formula",
        "workload_index": workload,
        "requests": requests,
        "profile_header": {
            "model": {"num_experts": 4, "top_k": 2},
            "hardware": {
                "model_weight_bytes": 1000,
                "kv_bytes_per_token": 10,
                "activation_bytes_per_scheduled_token": 2,
            },
            "runtime": {"precision": "bfloat16"},
        },
        "kernel_identity": {
            "dtype": "bfloat16",
            "attention_backend": "flash-attn-test",
            "moe_backend": "triton-test",
            "moe_kernel_config": "E4:N8:test",
            "dense_backend": "eager-test",
        },
        "layer_records": layers,
        "fixed_us": fixed,
        "timing_us": {"step": explicit + fixed},
    }


def test_operator_database_deduplicates_and_is_categorically_strict():
    samples = [
        OperatorSample("router", "decode", (1.0,), "bf16", "triton", "k", value)
        for value in (10.0, 12.0, 100.0)
    ]
    database = OperatorResponseDatabase().fit(samples)
    assert database.query("router", "decode", (1.0,), "bf16", "triton", "k").latency_us == 12.0
    assert database.stats()["unique_signatures"] == 1
    with pytest.raises(KeyError):
        database.query("router", "decode", (1.0,), "bf16", "other", "k")


def test_plan_preserves_per_layer_routing_and_mixed_phase():
    raw = _fine_raw(0, "prefill", 4, 1)
    raw["requests"].append(
        {
            "request_id": "decode",
            "phase": "decode",
            "input_length": 16,
            "prior_context_length": 16,
            "scheduled_tokens": 1,
            "context_length": 17,
        }
    )
    for layer in raw["layer_records"]:
        layer["tokens_per_expert"] = _histogram(5, layer["layer_id"])
    plan = StepExecutionPlan.from_raw_step(raw)
    assert plan.phase == "mixed"
    assert plan.realized_batch_size == 2
    assert len(plan.layers) == 2
    assert sum(plan.layers[0].tokens_per_expert) == 10


def test_whitebox_explicit_composition_has_no_learned_residual():
    rows = [
        _fine_raw(0, "prefill", 2, 1),
        _fine_raw(1, "prefill", 4, 1),
        _fine_raw(2, "decode", 1, 1),
        _fine_raw(3, "decode", 2, 2),
    ]
    model = WhiteboxMoEModel().fit_raw(rows, allow_synthetic=True)
    plan = StepExecutionPlan.from_raw_step(rows[1])
    profile = model.predict(plan)
    assert profile.step_us == pytest.approx(rows[1]["timing_us"]["step"])
    assert sum(profile.component_shares.values()) == pytest.approx(1.0)
    assert "residual" not in profile.components_us
    assert profile.compatibility_shares["moe"] > 0
    assert profile.memory_bytes == 1000 + plan.total_kv_tokens * 10 + plan.scheduled_tokens * 2
    assert model.provenance["overlap_audit_status"] == "serial_assumption_unverified"


def test_whitebox_synthetic_fixture_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit opt-in"):
        WhiteboxMoEModel().fit_raw([_fine_raw(0, "prefill", 2, 1)])


def test_audit_report_promotes_formal_artifact_only():
    row = _fine_raw(0, "prefill", 2, 1)
    row["label_source"] = "cuda_event"
    row["fixed_source"] = "direct_non_layer_dense_cuda_events"
    model = WhiteboxMoEModel().fit_raw([row], audit_report={"serial_validated": True})
    assert model.provenance["overlap_audit_status"] == "nsight_serial_validated"
    with pytest.raises(ValueError, match="did not validate"):
        WhiteboxMoEModel().fit_raw([row], audit_report={"serial_validated": False})


def test_holdout_evaluation_reports_error_and_anchor_suggestions():
    rows = []
    for workload, scheduled in enumerate((2, 4, 6)):
        rows.append(_fine_raw(workload, "prefill", scheduled, 1))
        rows.append(_fine_raw(workload, "decode", 1, 1))
    report = evaluate_leave_one_workload_out(rows, allow_synthetic=True)
    assert report["evaluated_steps"] == 6
    assert report["step_error"]["mape"] >= 0
    assert report["adaptive_anchor_suggestions"]
    assert "component_error" in report
    assert "bottleneck_accuracy" in report


def test_strict_evaluation_reports_protocols():
    rows = []
    for workload, scheduled in enumerate((2, 4, 8)):
        for phase in ("prefill", "decode"):
            row = _fine_raw(workload, phase, scheduled if phase == "prefill" else 1, 1)
            row["requested_workload"] = {
                "batch_size": 1 + workload % 2,
                "prompt_tokens": (16, 128, 512)[workload],
            }
            rows.append(row)
    report = evaluate_strict_generalization(rows, allow_synthetic=True)
    assert report["profiling_surface"]["prompt_lengths"] == [16, 128, 512]
    assert "unseen_batch" in report
    assert "region_holdout" in report


def test_whitebox_scheduler_replay_uses_explicit_profile():
    row = _fine_raw(0, "prefill", 2, 1)
    model = WhiteboxMoEModel().fit_raw([row], allow_synthetic=True)
    step = WhiteboxReplayStep(
        step_id=0,
        plan=StepExecutionPlan.from_raw_step(row),
        request_events=(RequestEvent("r0", 0.0, 1, True),),
    )
    result = replay_whitebox_steps(model, [step])
    assert result["summary"]["completed_requests"] == 1
    assert result["semantics"]["latency_model"] == "white-box timing oracle + execution DAG"

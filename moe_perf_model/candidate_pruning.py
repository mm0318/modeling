from __future__ import annotations

import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .fixed_config_oracle import _config_identity


TABLE_SCHEMA = "config-candidate-table/v1"
REPORT_SCHEMA = "candidate-pruning-report/v1"
SERVICE_TO_CANDIDATE_METRIC = {
    "mean_ttft_us": "ttft_us",
    "mean_tpot_us": "tpot_us",
    "p95_e2e_us": "e2e_us",
    "output_token_throughput_per_s": "throughput_per_s",
}
LATENCY_METRICS = ("ttft_us", "tpot_us", "e2e_us")


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty value list")
    return float(statistics.median(values))


def _interval(value: float, p95_ape_percent: float) -> dict[str, float]:
    if value < 0 or p95_ape_percent < 0:
        raise ValueError("prediction and error bound must be non-negative")
    relative = p95_ape_percent / 100.0
    return {
        "point": value,
        "lower": max(0.0, value * (1.0 - relative)),
        "upper": value * (1.0 + relative),
        "relative_error_p95": relative,
    }


def _load_workload_policy(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not isinstance(value.get("workloads", {}), dict):
        raise ValueError("workload policy must contain an optional workloads object")
    return value["workloads"]


def build_candidate_table_from_service_report(
    service_report: dict[str, Any],
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
    *, gpu_memory_bytes: int, available_gpus: int,
    source_cost_units: float | None = None, target_cost_units: float | None = None,
    workload_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if service_report.get("schema_version") != "service-config-ranking/v1":
        raise ValueError("input is not a service configuration ranking report")
    if gpu_memory_bytes <= 0 or available_gpus <= 0:
        raise ValueError("GPU capacity and count must be positive")
    if _config_identity(source_rows[0]) != service_report["source_identity"]:
        raise ValueError("source rows do not match service report identity")
    if _config_identity(target_rows[0]) != service_report["target_identity"]:
        raise ValueError("target rows do not match service report identity")
    policies = _load_workload_policy(workload_policy)
    trajectories = [
        trajectory for fold in service_report["folds"]
        for trajectory in fold["trajectories"]
    ]
    by_workload: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_workload[int(trajectory["workload_index"])].append(trajectory)

    memory_by_role: dict[str, dict[int, int]] = {"source": {}, "target": {}}
    for role, rows in (("source", source_rows), ("target", target_rows)):
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if row.get("peak_memory_bytes") is not None:
                grouped[int(row["workload_index"])].append(int(row["peak_memory_bytes"]))
        memory_by_role[role] = {
            workload: max(values) for workload, values in grouped.items()
        }

    slo = service_report["slo"]
    workloads = []
    for workload_index, rows in sorted(by_workload.items()):
        workload_id = f"w{workload_index}"
        override = policies.get(workload_id, {})
        if not isinstance(override, dict):
            raise ValueError(f"workload policy {workload_id} must be an object")
        workload_slo = override.get("slo", {
            "ttft_us": slo.get("ttft_us"),
            "tpot_us": slo.get("tpot_us"),
            "e2e_us": slo.get("e2e_us"),
        })
        workloads.append({
            "workload_id": workload_id, "workload_index": workload_index,
            "request_shape": rows[0]["requested_workload"],
            "arrival_rate_rps": override.get("arrival_rate_rps"),
            "slo": workload_slo,
            "weight": float(override.get("weight", 1.0)),
        })

    candidates = []
    specifications = (
        ("tp2_ep2", "source", service_report["source_identity"],
         "source_full", source_cost_units),
        ("tp2_ep1", "target", service_report["target_identity"],
         "target_calibrated", target_cost_units),
    )
    for candidate_id, role, identity, prediction_key, cost in specifications:
        error_key = "source_full" if role == "source" else "target_calibrated"
        error_report = service_report["service_error"][role][error_key]
        predictions = {}
        for workload_index, trajectory_rows in sorted(by_workload.items()):
            workload_id = f"w{workload_index}"
            metrics = {}
            for service_metric, candidate_metric in SERVICE_TO_CANDIDATE_METRIC.items():
                point = _median([
                    float(row["predicted"][prediction_key][service_metric])
                    for row in trajectory_rows
                ])
                p95_error = float(error_report[service_metric]["p95"])
                metrics[candidate_metric] = _interval(point, p95_error)
            coverage = _median([
                float(row["predicted"][prediction_key]["supported_step_fraction"])
                for row in trajectory_rows
            ])
            predictions[workload_id] = {
                "metrics": metrics, "coverage": coverage,
                "prediction_status": "supported" if coverage >= 1.0 - 1e-12
                else "partially_supported",
                "memory_bytes_per_gpu": memory_by_role[role].get(workload_index),
                "repeat_count": len(trajectory_rows),
                "error_bound_source": (
                    f"service_report.service_error.{role}.{error_key}.metric.p95"
                ),
            }
        rank_count = int(identity["rank_count"])
        candidates.append({
            "candidate_id": candidate_id, "role": role,
            "config_identity": identity,
            "parallel": {
                "world_size": rank_count,
                "tensor_parallel_size": int(identity["tensor_parallel_size"]),
                "expert_parallel_size": int(identity["expert_parallel_size"]),
                "node_count": 1, "gpus_per_node": rank_count,
            },
            "launch_status": "measured_success",
            "cost_units": float(cost) if cost is not None else None,
            "memory_bytes_per_gpu_worst_case": max(
                (value for value in memory_by_role[role].values()), default=None
            ),
            "predictions": predictions,
        })
    return {
        "schema_version": TABLE_SCHEMA,
        "hardware_pool": {
            "available_gpus": available_gpus,
            "gpu_memory_bytes_per_device": gpu_memory_bytes,
            "hardware_id": service_report["source_identity"]["hardware_id"],
        },
        "workloads": workloads, "candidates": candidates,
        "provenance": {
            "source": "service-config-ranking/v1",
            "uses_actual_service_labels_as_pruning_features": False,
            "repeat_reduction": "median",
            "uncertainty": "global_service_metric_p95_ape_multiplicative_interval",
            "memory_source": "measured_multirank_peak_memory_per_workload",
            "arrival_rate_default": "not_evaluated",
            "cost_default": "not_evaluated",
        },
    }


DEFAULT_POLICY = {
    "min_coverage": 0.80,
    "max_memory_utilization": 0.95,
    "slo_policy": "risk_averse_upper_bound",
    "capacity_action": "needs_replica_evaluation",
    "max_replicas": None,
    "pareto_mode": "conservative_interval",
}


def load_candidate_table(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_table(value)
    return value


def validate_candidate_table(table: dict[str, Any]) -> None:
    if table.get("schema_version") != TABLE_SCHEMA:
        raise ValueError("candidate table has an unsupported schema")
    workload_ids = [row.get("workload_id") for row in table.get("workloads", [])]
    candidate_ids = [row.get("candidate_id") for row in table.get("candidates", [])]
    if not workload_ids or len(workload_ids) != len(set(workload_ids)):
        raise ValueError("workload IDs must be non-empty and unique")
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be non-empty and unique")
    pool = table.get("hardware_pool", {})
    if int(pool.get("available_gpus", 0)) <= 0 or int(
        pool.get("gpu_memory_bytes_per_device", 0)
    ) <= 0:
        raise ValueError("hardware pool must define positive GPU count and memory")
    for candidate in table["candidates"]:
        if set(candidate.get("predictions", {})) != set(workload_ids):
            raise ValueError("every candidate must predict every workload")
        for prediction in candidate["predictions"].values():
            coverage = prediction.get("coverage")
            if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
                raise ValueError("coverage must be between zero and one")
            for metric in (*LATENCY_METRICS, "throughput_per_s"):
                interval = prediction.get("metrics", {}).get(metric)
                if not isinstance(interval, dict):
                    raise ValueError(f"missing interval for {metric}")
                lower, point, upper = (
                    float(interval[name]) for name in ("lower", "point", "upper")
                )
                if lower < 0 or not lower <= point <= upper:
                    raise ValueError(f"invalid interval for {metric}")


def _merged_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    result = {**DEFAULT_POLICY, **(policy or {})}
    if not 0 <= float(result["min_coverage"]) <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    if not 0 < float(result["max_memory_utilization"]) <= 1:
        raise ValueError("max_memory_utilization must be in (0, 1]")
    if result["slo_policy"] not in {
        "risk_averse_upper_bound", "certain_violation_lower_bound"
    }:
        raise ValueError("unsupported SLO policy")
    if result["capacity_action"] not in {"needs_replica_evaluation", "prune"}:
        raise ValueError("unsupported capacity action")
    if result["max_replicas"] is not None and int(result["max_replicas"]) < 1:
        raise ValueError("max_replicas must be positive when configured")
    if result["pareto_mode"] != "conservative_interval":
        raise ValueError("only conservative_interval Pareto is supported")
    return result


def _initial_decision(
    candidate: dict[str, Any], workload: dict[str, Any], table: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    candidate_id, workload_id = candidate["candidate_id"], workload["workload_id"]
    prediction = candidate["predictions"][workload_id]
    reasons = []
    rules: dict[str, Any] = {}
    status = "retained"

    launch = candidate.get("launch_status", "unknown")
    if launch == "failed":
        status, reasons = "pruned", ["configuration_launch_failed"]
    elif launch not in {"measured_success", "validated_success"}:
        status, reasons = "needs_validation", ["configuration_launch_status_unknown"]
    parallel = candidate.get("parallel", {})
    world = int(parallel.get("world_size", 0))
    tp, ep = int(parallel.get("tensor_parallel_size", 0)), int(
        parallel.get("expert_parallel_size", 0)
    )
    available = int(table["hardware_pool"]["available_gpus"])
    parallel_ok = world > 0 and tp > 0 and ep > 0 and world <= available
    parallel_ok = parallel_ok and world % tp == 0 and world % ep == 0
    rules["parallel_feasibility"] = {
        "passed": parallel_ok, "world_size": world, "tp": tp, "ep": ep,
        "available_gpus": available,
    }
    if not parallel_ok:
        status, reasons = "pruned", reasons + ["parallel_layout_incompatible"]

    memory = prediction.get("memory_bytes_per_gpu")
    memory_limit = int(table["hardware_pool"]["gpu_memory_bytes_per_device"]) * float(
        policy["max_memory_utilization"]
    )
    rules["memory_feasibility"] = {
        "memory_bytes_per_gpu": memory, "allowed_bytes_per_gpu": memory_limit,
        "passed": None if memory is None else float(memory) <= memory_limit,
    }
    if memory is None and status == "retained":
        status, reasons = "needs_validation", reasons + ["memory_measurement_missing"]
    elif memory is not None and float(memory) > memory_limit:
        status, reasons = "pruned", reasons + ["memory_capacity_exceeded"]

    confidence_ok = (
        prediction.get("prediction_status") in {"supported", "partially_supported"}
        and float(prediction["coverage"]) >= float(policy["min_coverage"])
    )
    rules["confidence"] = {
        "passed": confidence_ok, "coverage": prediction["coverage"],
        "minimum": policy["min_coverage"],
        "prediction_status": prediction.get("prediction_status"),
    }
    if status == "retained" and not confidence_ok:
        status, reasons = "needs_profile", reasons + ["prediction_confidence_insufficient"]

    slo_results = {}
    if status == "retained":
        for metric in LATENCY_METRICS:
            threshold = workload.get("slo", {}).get(metric)
            if threshold is None:
                slo_results[metric] = {"status": "not_evaluated"}
                continue
            interval = prediction["metrics"][metric]
            risk_violation = float(interval["upper"]) > float(threshold)
            certain_violation = float(interval["lower"]) > float(threshold)
            slo_results[metric] = {
                "status": "violation" if risk_violation else "passed",
                "threshold": threshold, "prediction": interval,
                "risk_violation": risk_violation,
                "certain_violation": certain_violation,
            }
        violations = [row for row in slo_results.values() if row.get("status") == "violation"]
        if policy["slo_policy"] == "certain_violation_lower_bound":
            violations = [row for row in slo_results.values() if row.get("certain_violation")]
        if violations:
            status, reasons = "pruned", reasons + ["slo_policy_violation"]
    else:
        slo_results = {metric: {"status": "skipped"} for metric in LATENCY_METRICS}
    rules["slo"] = {"policy": policy["slo_policy"], "metrics": slo_results}

    arrival = workload.get("arrival_rate_rps")
    if status == "retained" and arrival is not None:
        lower = float(prediction["metrics"]["throughput_per_s"]["lower"])
        output_tokens = int(workload.get("request_shape", {}).get("output_tokens", 1))
        demand_tokens_per_s = float(arrival) * output_tokens
        required = math.ceil(demand_tokens_per_s / lower) if lower > 0 else None
        enough = lower >= demand_tokens_per_s
        rules["capacity"] = {
            "status": "passed" if enough else "insufficient_single_replica",
            "arrival_rate_rps": arrival, "throughput_lower_bound": lower,
            "output_tokens_per_request": output_tokens,
            "demand_output_tokens_per_s": demand_tokens_per_s,
            "required_replicas": required, "max_replicas": policy["max_replicas"],
        }
        if not enough:
            exceeds_max = (
                policy["max_replicas"] is not None
                and (required is None or required > int(policy["max_replicas"]))
            )
            if policy["capacity_action"] == "prune" or exceeds_max:
                status, reasons = "pruned", reasons + ["single_replica_capacity_insufficient"]
                if exceeds_max:
                    reasons.append("required_replicas_exceed_policy_limit")
            else:
                status, reasons = "needs_replica_evaluation", reasons + [
                    "single_replica_capacity_insufficient"
                ]
    else:
        rules["capacity"] = {
            "status": "not_evaluated" if arrival is None else "skipped",
            "arrival_rate_rps": arrival,
        }
    return {
        "candidate_id": candidate_id, "workload_id": workload_id,
        "status": status, "reasons": reasons, "rules": rules,
        "prediction": prediction,
    }


def _dominates(
    left: dict[str, Any], right: dict[str, Any], *, use_cost: bool,
    left_cost: float | None, right_cost: float | None,
) -> bool:
    left_metrics, right_metrics = left["prediction"]["metrics"], right["prediction"]["metrics"]
    comparisons = [
        float(left_metrics[metric]["upper"]) <= float(right_metrics[metric]["lower"])
        for metric in LATENCY_METRICS
    ]
    comparisons.append(
        float(left_metrics["throughput_per_s"]["lower"])
        >= float(right_metrics["throughput_per_s"]["upper"])
    )
    if use_cost:
        comparisons.append(float(left_cost) <= float(right_cost))
    if not all(comparisons):
        return False
    strict = any(
        float(left_metrics[metric]["upper"]) < float(right_metrics[metric]["lower"])
        for metric in LATENCY_METRICS
    ) or (
        float(left_metrics["throughput_per_s"]["lower"])
        > float(right_metrics["throughput_per_s"]["upper"])
    )
    if use_cost:
        strict = strict or float(left_cost) < float(right_cost)
    return strict


def prune_candidate_table(
    table: dict[str, Any], policy: dict[str, Any] | None = None
) -> dict[str, Any]:
    validate_candidate_table(table)
    policy = _merged_policy(policy)
    candidates = {row["candidate_id"]: row for row in table["candidates"]}
    decisions = []
    for workload in table["workloads"]:
        rows = [
            _initial_decision(candidate, workload, table, policy)
            for candidate in table["candidates"]
        ]
        eligible = [row for row in rows if row["status"] == "retained"]
        use_cost = bool(eligible) and all(
            candidates[row["candidate_id"]].get("cost_units") is not None for row in eligible
        )
        for right in eligible:
            dominators = [
                left["candidate_id"] for left in eligible
                if left is not right and _dominates(
                    left, right, use_cost=use_cost,
                    left_cost=candidates[left["candidate_id"]].get("cost_units"),
                    right_cost=candidates[right["candidate_id"]].get("cost_units"),
                )
            ]
            right["rules"]["pareto"] = {
                "mode": policy["pareto_mode"], "cost_dimension_used": use_cost,
                "dominators": dominators,
            }
            if dominators:
                right["status"] = "pruned"
                right["reasons"].append("conservatively_pareto_dominated")
        for row in rows:
            row["rules"].setdefault("pareto", {
                "mode": policy["pareto_mode"], "status": "skipped",
                "cost_dimension_used": use_cost,
            })
        decisions.extend(rows)

    by_status = Counter(row["status"] for row in decisions)
    workload_sets = {}
    for workload in table["workloads"]:
        workload_id = workload["workload_id"]
        subset = [row for row in decisions if row["workload_id"] == workload_id]
        workload_sets[workload_id] = {
            status: [row["candidate_id"] for row in subset if row["status"] == status]
            for status in (
                "retained", "pruned", "needs_profile",
                "needs_replica_evaluation", "needs_validation",
            )
        }
    all_workloads = {row["workload_id"] for row in table["workloads"]}
    globally_retained = [
        candidate_id for candidate_id in candidates
        if any(
            row["candidate_id"] == candidate_id and row["status"] == "retained"
            for row in decisions
        )
    ]
    globally_pruned = [
        candidate_id for candidate_id in candidates
        if {
            row["workload_id"] for row in decisions
            if row["candidate_id"] == candidate_id and row["status"] == "pruned"
        } == all_workloads
    ]
    return {
        "schema_version": REPORT_SCHEMA, "policy": policy,
        "input_schema": table["schema_version"],
        "summary": {
            "candidate_count": len(candidates),
            "workload_count": len(table["workloads"]),
            "candidate_workload_decision_count": len(decisions),
            "status_counts": dict(sorted(by_status.items())),
            "globally_retained_candidates": globally_retained,
            "globally_pruned_candidates": globally_pruned,
        },
        "workload_candidate_sets": workload_sets,
        "profile_queue": [
            {"candidate_id": row["candidate_id"], "workload_id": row["workload_id"],
             "reasons": row["reasons"]}
            for row in decisions if row["status"] == "needs_profile"
        ],
        "replica_evaluation_queue": [
            {"candidate_id": row["candidate_id"], "workload_id": row["workload_id"],
             "capacity": row["rules"]["capacity"]}
            for row in decisions if row["status"] == "needs_replica_evaluation"
        ],
        "validation_queue": [
            {"candidate_id": row["candidate_id"], "workload_id": row["workload_id"],
             "reasons": row["reasons"]}
            for row in decisions if row["status"] == "needs_validation"
        ],
        "decisions": decisions,
        "provenance": {
            "uses_actual_service_labels": False,
            "pareto_scope": "workload_conditioned",
            "unknown_is_not_pruned": True,
        },
    }

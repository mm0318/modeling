from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .demo import write_demo, write_whitebox_demo
from .collector import convert_raw_collection, load_collection_manifest, run_collection
from .model import BasicRFModel, ComponentModel, load_artifact, save_artifact
from .replay import (
    load_replay_steps,
    load_whitebox_replay_steps,
    replay_steps,
    replay_whitebox_steps,
)
from .schema import load_config, load_training_records
from .execution_plan import StepExecutionPlan
from .evaluation import evaluate_leave_one_workload_out, evaluate_strict_generalization
from .nsight_audit import audit_nsight_sqlite
from .surface_report import write_surface_report
from .event_engine import (
    DiscreteEventEngine,
    ExecutionGraph,
    build_synthetic_collective_graph,
)
from .whitebox_model import (
    WhiteboxMoEModel,
    load_fine_raw_records,
    load_whitebox_artifact,
    save_whitebox_artifact,
)
from .collective_model import NcclResponseTable
from .fixed_config_oracle import (
    FixedConfigForwardOracle,
    evaluate_fixed_config,
    load_multirank_records,
    merge_multirank_record_files,
    save_fixed_config_artifact,
)
from .budget_evaluation import aggregate_budget_holdout_folds, evaluate_profiling_budgets
from .configuration_comparison import compare_fixed_config_files
from .cross_config_calibration import (
    evaluate_cross_config_calibration,
    load_cross_config_artifact,
    save_cross_config_artifact,
    train_cross_config_calibrator,
)
from .fixed_config_oracle import load_fixed_config_artifact
from .service_replay import evaluate_service_configuration_ranking
from .candidate_pruning import (
    build_candidate_table_from_service_report,
    load_candidate_table,
    prune_candidate_table,
)
from .dense_moe_comparison import compare_dense_moe, write_dense_moe_report
from .multigpu_bottleneck import (
    compare_multigpu_bottlenecks,
    write_multigpu_report,
)
from .parallel_forward import (
    ParallelForwardGraphCompiler,
    ParallelLayout,
    ParallelScalingPolicy,
)
from .structural_forward import ForwardStructuralPredictor
from .rank_local_compute import (
    RankLocalComputeModel,
    load_rank_local_compute_artifact,
    save_rank_local_compute_artifact,
)


def _write_json(value: Any, path: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_perf_model",
        description="Same-runtime component performance modeling for MoE inference",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("records", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", choices=("component", "basic-rf"), default="component")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--allow-synthetic-fixture", action="store_true")

    predict = subparsers.add_parser("predict")
    predict.add_argument("artifact", type=Path)
    predict.add_argument("config", type=Path)
    predict.add_argument("--output", type=Path)

    replay = subparsers.add_parser("replay")
    replay.add_argument("artifact", type=Path)
    replay.add_argument("trace", type=Path)
    replay.add_argument("--output", type=Path, required=True)

    demo = subparsers.add_parser("make-demo")
    demo.add_argument("--output-dir", type=Path, default=Path("examples"))

    whitebox_demo = subparsers.add_parser("make-whitebox-demo")
    whitebox_demo.add_argument("--output-dir", type=Path, default=Path("examples/whitebox"))

    collect = subparsers.add_parser("collect")
    collect.add_argument("manifest", type=Path)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--dry-run", action="store_true")

    convert = subparsers.add_parser("convert-collection")
    convert.add_argument("manifest", type=Path)
    convert.add_argument("raw_steps", type=Path)
    convert.add_argument("--output-dir", type=Path, required=True)

    train_whitebox = subparsers.add_parser("train-whitebox")
    train_whitebox.add_argument("raw_steps", type=Path)
    train_whitebox.add_argument("--output", type=Path, required=True)
    train_whitebox.add_argument("--allow-synthetic-fixture", action="store_true")
    train_whitebox.add_argument("--audit-report", type=Path)

    train_rank_local = subparsers.add_parser("train-rank-local-compute")
    train_rank_local.add_argument("multirank_steps", type=Path)
    train_rank_local.add_argument("--neighbors", type=int, default=4)
    train_rank_local.add_argument("--output", type=Path, required=True)

    audit_nsight = subparsers.add_parser("audit-nsight")
    audit_nsight.add_argument("sqlite", type=Path)
    audit_nsight.add_argument("raw_steps", type=Path)
    audit_nsight.add_argument("--output", type=Path, required=True)

    predict_whitebox = subparsers.add_parser("predict-whitebox")
    predict_whitebox.add_argument("artifact", type=Path)
    predict_whitebox.add_argument("plan", type=Path)
    predict_whitebox.add_argument("--output", type=Path)
    predict_whitebox.add_argument("--plan-index", type=int, default=0)

    predict_structural = subparsers.add_parser("predict-structural-forward")
    predict_structural.add_argument("artifact", type=Path)
    predict_structural.add_argument("plan", type=Path)
    predict_structural.add_argument("nccl_response", type=Path)
    predict_structural.add_argument("--world-size", type=int, required=True)
    predict_structural.add_argument("--tp-size", type=int, required=True)
    predict_structural.add_argument("--ep-size", type=int, required=True)
    predict_structural.add_argument("--gpus-per-node", type=int, default=8)
    predict_structural.add_argument("--hidden-size", type=int, default=2048)
    predict_structural.add_argument("--dtype-bytes", type=int, default=2)
    predict_structural.add_argument(
        "--ep-communication-backend",
        choices=("alltoall", "allgather_reducescatter"),
        default="allgather_reducescatter",
    )
    predict_structural.add_argument("--tp-compute-efficiency", type=float, default=0.90)
    predict_structural.add_argument("--ep-compute-efficiency", type=float, default=0.95)
    predict_structural.add_argument("--rank-local-compute", type=Path)
    predict_structural.add_argument("--plan-index", type=int, default=0)
    predict_structural.add_argument("--output", type=Path, required=True)

    evaluate_whitebox = subparsers.add_parser("evaluate-whitebox")
    evaluate_whitebox.add_argument("raw_steps", type=Path)
    evaluate_whitebox.add_argument("--output", type=Path, required=True)
    evaluate_whitebox.add_argument("--allow-synthetic-fixture", action="store_true")
    evaluate_whitebox.add_argument("--strict-output", type=Path)

    replay_whitebox = subparsers.add_parser("replay-whitebox")
    replay_whitebox.add_argument("artifact", type=Path)
    replay_whitebox.add_argument("trace", type=Path)
    replay_whitebox.add_argument("--output", type=Path, required=True)

    inspect_whitebox = subparsers.add_parser("inspect-whitebox")
    inspect_whitebox.add_argument("artifact", type=Path)
    inspect_whitebox.add_argument("--output", type=Path)

    surface = subparsers.add_parser("report-surface")
    surface.add_argument("raw_steps", type=Path)
    surface.add_argument("--output-json", type=Path, required=True)
    surface.add_argument("--output-markdown", type=Path, required=True)

    dense_moe = subparsers.add_parser("compare-dense-moe")
    dense_moe.add_argument("dense_raw_steps", type=Path)
    dense_moe.add_argument("moe_raw_steps", type=Path)
    dense_moe.add_argument("--output-json", type=Path, required=True)
    dense_moe.add_argument("--output-markdown", type=Path, required=True)

    multigpu_bottleneck = subparsers.add_parser("compare-multigpu-bottlenecks")
    multigpu_bottleneck.add_argument("dense_multirank_steps", type=Path)
    multigpu_bottleneck.add_argument("moe_multirank_steps", type=Path)
    multigpu_bottleneck.add_argument("--output-json", type=Path, required=True)
    multigpu_bottleneck.add_argument("--output-markdown", type=Path, required=True)

    simulate_graph = subparsers.add_parser("simulate-event-graph")
    simulate_graph.add_argument("graph", type=Path)
    simulate_graph.add_argument("--output", type=Path, required=True)

    event_demo = subparsers.add_parser("make-event-demo")
    event_demo.add_argument(
        "--scenario",
        choices=("serialized", "overlap", "slow_rank", "contention"),
        required=True,
    )
    event_demo.add_argument("--output-graph", type=Path, required=True)
    event_demo.add_argument("--output-report", type=Path, required=True)

    train_fixed = subparsers.add_parser("train-fixed-config")
    train_fixed.add_argument("multirank_steps", type=Path)
    train_fixed.add_argument("--output", type=Path, required=True)
    train_fixed.add_argument("--nccl-response", type=Path)
    train_fixed.add_argument("--neighbors", type=int, default=4)
    train_fixed.add_argument("--method", choices=("idw", "routing_rf"), default="idw")

    evaluate_fixed = subparsers.add_parser("evaluate-fixed-config")
    evaluate_fixed.add_argument("multirank_steps", type=Path)
    evaluate_fixed.add_argument("--output", type=Path, required=True)
    evaluate_fixed.add_argument("--neighbors", type=int, default=4)
    evaluate_fixed.add_argument("--method", choices=("idw", "routing_rf"), default="idw")

    budget_curve = subparsers.add_parser("evaluate-budget-curve")
    budget_curve.add_argument("multirank_steps", type=Path)
    budget_curve.add_argument("--output", type=Path, required=True)
    budget_curve.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 96, 128, 187])
    budget_curve.add_argument("--neighbors", type=int, default=4)
    budget_curve.add_argument("--method", choices=("idw", "routing_rf"), default="routing_rf")
    budget_curve.add_argument(
        "--strategy",
        choices=("phase_aware_farthest", "observed_error_aware", "both"),
        default="phase_aware_farthest",
    )
    budget_curve.add_argument(
        "--protocol", choices=("pool_surface_recovery", "fixed_workload_holdout"),
        default="pool_surface_recovery",
    )
    budget_curve.add_argument("--holdout-modulus", type=int, default=5)
    budget_curve.add_argument("--holdout-fold", type=int, default=0)

    merge_multirank = subparsers.add_parser("merge-multirank")
    merge_multirank.add_argument("inputs", type=Path, nargs="+")
    merge_multirank.add_argument("--output", type=Path, required=True)

    aggregate_folds = subparsers.add_parser("aggregate-budget-folds")
    aggregate_folds.add_argument("inputs", type=Path, nargs="+")
    aggregate_folds.add_argument("--output", type=Path, required=True)
    aggregate_folds.add_argument("--target-mape", type=float, default=3.0)
    aggregate_folds.add_argument("--target-p95-ape", type=float, default=15.0)
    aggregate_folds.add_argument("--target-top-k-recall", type=float, default=0.95)
    aggregate_folds.add_argument("--target-supported-fraction", type=float, default=0.80)

    compare_configs = subparsers.add_parser("compare-fixed-configs")
    compare_configs.add_argument("source", type=Path)
    compare_configs.add_argument("target", type=Path)
    compare_configs.add_argument("--output", type=Path, required=True)

    train_cross = subparsers.add_parser("train-cross-config-calibration")
    train_cross.add_argument("source", type=Path)
    train_cross.add_argument("target", type=Path)
    train_cross.add_argument("source_artifact", type=Path)
    train_cross.add_argument("--budget", type=int, required=True)
    train_cross.add_argument(
        "--method", choices=("phase_scale", "local_ratio_idw"),
        default="local_ratio_idw",
    )
    train_cross.add_argument("--neighbors", type=int, default=4)
    train_cross.add_argument("--output", type=Path, required=True)

    evaluate_cross = subparsers.add_parser("evaluate-cross-config-calibration")
    evaluate_cross.add_argument("source", type=Path)
    evaluate_cross.add_argument("target", type=Path)
    evaluate_cross.add_argument("source_artifact", type=Path)
    evaluate_cross.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    evaluate_cross.add_argument("--folds", type=int, default=5)
    evaluate_cross.add_argument("--neighbors", type=int, default=4)
    evaluate_cross.add_argument("--target-reference", type=Path)
    evaluate_cross.add_argument("--output", type=Path, required=True)

    predict_cross = subparsers.add_parser("predict-cross-config-calibration")
    predict_cross.add_argument("calibration_artifact", type=Path)
    predict_cross.add_argument("source_artifact", type=Path)
    predict_cross.add_argument("source_row", type=Path)
    predict_cross.add_argument("target_row", type=Path)
    predict_cross.add_argument("--allow-unsupported", action="store_true")
    predict_cross.add_argument("--output", type=Path)

    service_ranking = subparsers.add_parser("evaluate-service-ranking")
    service_ranking.add_argument("source", type=Path)
    service_ranking.add_argument("target", type=Path)
    service_ranking.add_argument("source_artifact", type=Path)
    service_ranking.add_argument("target_artifact", type=Path)
    service_ranking.add_argument("--target-calibration-budget", type=int, default=32)
    service_ranking.add_argument("--folds", type=int, default=5)
    service_ranking.add_argument("--ttft-slo-ms", type=float)
    service_ranking.add_argument("--tpot-slo-ms", type=float)
    service_ranking.add_argument("--e2e-slo-ms", type=float)
    service_ranking.add_argument("--output", type=Path, required=True)

    candidate_table = subparsers.add_parser("build-candidate-table")
    candidate_table.add_argument("service_report", type=Path)
    candidate_table.add_argument("source", type=Path)
    candidate_table.add_argument("target", type=Path)
    candidate_table.add_argument("--gpu-memory-bytes", type=int, required=True)
    candidate_table.add_argument("--available-gpus", type=int, required=True)
    candidate_table.add_argument("--source-cost-units", type=float)
    candidate_table.add_argument("--target-cost-units", type=float)
    candidate_table.add_argument("--workload-policy", type=Path)
    candidate_table.add_argument("--output", type=Path, required=True)

    prune = subparsers.add_parser("prune-candidates")
    prune.add_argument("candidate_table", type=Path)
    prune.add_argument("--policy", type=Path)
    prune.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        _write_json({"valid": True, "config_id": config.signature()}, None)
    elif args.command == "train":
        records = load_training_records(args.records)
        if args.model == "component":
            model = ComponentModel().fit(
                records, allow_synthetic=args.allow_synthetic_fixture
            )
        else:
            model = BasicRFModel(seed=args.seed).fit(
                records, allow_synthetic=args.allow_synthetic_fixture
            )
        save_artifact(model, args.output)
        _write_json(
            {
                "artifact": str(args.output),
                "artifact_type": model.artifact_type,
                "provenance": model.provenance,
            },
            None,
        )
    elif args.command == "predict":
        model = load_artifact(args.artifact)
        if not isinstance(model, ComponentModel):
            raise TypeError(
                "predict requires a component artifact; Basic RF has no component attribution"
            )
        _write_json(model.predict(load_config(args.config)).to_dict(), args.output)
    elif args.command == "replay":
        model = load_artifact(args.artifact)
        if not isinstance(model, ComponentModel):
            raise TypeError("replay requires a component artifact")
        _write_json(replay_steps(model, load_replay_steps(args.trace)), args.output)
    elif args.command == "make-demo":
        _write_json(write_demo(args.output_dir), None)
    elif args.command == "make-whitebox-demo":
        _write_json(write_whitebox_demo(args.output_dir), None)
    elif args.command == "collect":
        manifest = load_collection_manifest(args.manifest)
        if args.dry_run:
            _write_json(
                {
                    "valid": True,
                    "model_path": manifest.model_path,
                    "workloads": len(manifest.workloads),
                    "planned_generations": sum(
                        max(
                            item.repeats,
                            int(manifest.profile_header["runtime"].get(
                                "profiling_repeats", 0
                            )),
                        )
                        for item in manifest.workloads
                    ),
                },
                None,
            )
        else:
            _write_json(run_collection(manifest, args.output_dir), None)
    elif args.command == "convert-collection":
        manifest = load_collection_manifest(args.manifest)
        _write_json(convert_raw_collection(args.raw_steps, manifest, args.output_dir), None)
    elif args.command == "train-whitebox":
        audit_report = None
        if args.audit_report is not None:
            audit_report = json.loads(args.audit_report.read_text(encoding="utf-8"))
        model = WhiteboxMoEModel().fit_raw(
            load_fine_raw_records(args.raw_steps),
            allow_synthetic=args.allow_synthetic_fixture,
            audit_report=audit_report,
        )
        save_whitebox_artifact(model, args.output)
        _write_json(
            {"artifact": str(args.output), "artifact_type": model.artifact_type, "provenance": model.provenance},
            None,
        )
    elif args.command == "train-rank-local-compute":
        model = RankLocalComputeModel(neighbors=args.neighbors).fit(
            load_multirank_records(args.multirank_steps)
        )
        save_rank_local_compute_artifact(model, args.output)
        _write_json(
            {"artifact": str(args.output), "artifact_type": model.artifact_type,
             "provenance": model.provenance},
            None,
        )
    elif args.command == "audit-nsight":
        _write_json(audit_nsight_sqlite(args.sqlite, args.raw_steps), args.output)
    elif args.command == "predict-whitebox":
        plan_text = args.plan.read_text(encoding="utf-8")
        try:
            raw_plan = json.loads(plan_text)
        except json.JSONDecodeError:
            plan_rows = [json.loads(line) for line in plan_text.splitlines() if line.strip()]
            if args.plan_index < 0 or args.plan_index >= len(plan_rows):
                raise IndexError("plan-index is outside the JSONL execution plan file")
            raw_plan = plan_rows[args.plan_index]
        profile = load_whitebox_artifact(args.artifact).predict(StepExecutionPlan.from_dict(raw_plan))
        _write_json(profile.to_dict(), args.output)
    elif args.command == "predict-structural-forward":
        plan_text = args.plan.read_text(encoding="utf-8")
        try:
            parsed_plan = json.loads(plan_text)
            plan_rows = parsed_plan if isinstance(parsed_plan, list) else [parsed_plan]
        except json.JSONDecodeError:
            plan_rows = [
                json.loads(line) for line in plan_text.splitlines() if line.strip()
            ]
        if args.plan_index < 0 or args.plan_index >= len(plan_rows):
            raise IndexError("plan-index is outside the JSON/JSONL plan file")
        raw_plan = plan_rows[args.plan_index]
        plan = (
            StepExecutionPlan.from_raw_step(raw_plan)
            if "layer_records" in raw_plan
            else StepExecutionPlan.from_dict(raw_plan)
        )
        layout = ParallelLayout.vllm(
            args.world_size, args.tp_size, args.ep_size, args.gpus_per_node
        )
        compiler = ParallelForwardGraphCompiler(
            layout=layout,
            collective_backend=NcclResponseTable.load(args.nccl_response),
            collective_backend_name="nccl",
            hidden_size=args.hidden_size,
            dtype_bytes=args.dtype_bytes,
            all2all_backend=args.ep_communication_backend,
            scaling=ParallelScalingPolicy(
                tp_compute_efficiency=args.tp_compute_efficiency,
                ep_compute_efficiency=args.ep_compute_efficiency,
                label_source="synthetic_fixture",
            ),
            rank_local_compute_backend=(
                load_rank_local_compute_artifact(args.rank_local_compute)
                if args.rank_local_compute is not None else None
            ),
        )
        prediction = ForwardStructuralPredictor(
            compute_model=load_whitebox_artifact(args.artifact),
            graph_compiler=compiler,
        ).predict(plan)
        _write_json(prediction.to_dict(), args.output)
    elif args.command == "evaluate-whitebox":
        raw_records = load_fine_raw_records(args.raw_steps)
        report = evaluate_leave_one_workload_out(
            raw_records,
            allow_synthetic=args.allow_synthetic_fixture,
        )
        _write_json(report, args.output)
        if args.strict_output is not None:
            _write_json(
                evaluate_strict_generalization(
                    raw_records, allow_synthetic=args.allow_synthetic_fixture
                ),
                args.strict_output,
            )
    elif args.command == "replay-whitebox":
        model = load_whitebox_artifact(args.artifact)
        _write_json(
            replay_whitebox_steps(model, load_whitebox_replay_steps(args.trace)),
            args.output,
        )
    elif args.command == "inspect-whitebox":
        model = load_whitebox_artifact(args.artifact)
        _write_json(
            {"provenance": model.provenance, "operator_database": model.database.to_dict()},
            args.output,
        )
    elif args.command == "report-surface":
        _write_json(
            write_surface_report(args.raw_steps, args.output_json, args.output_markdown),
            None,
        )
    elif args.command == "compare-dense-moe":
        report = compare_dense_moe(args.dense_raw_steps, args.moe_raw_steps)
        _write_json(report, args.output_json)
        write_dense_moe_report(report, args.output_markdown)
    elif args.command == "compare-multigpu-bottlenecks":
        report = compare_multigpu_bottlenecks(
            args.dense_multirank_steps, args.moe_multirank_steps
        )
        _write_json(report, args.output_json)
        write_multigpu_report(report, args.output_markdown)
    elif args.command == "simulate-event-graph":
        graph = ExecutionGraph.from_dict(
            json.loads(args.graph.read_text(encoding="utf-8"))
        )
        _write_json(DiscreteEventEngine().simulate(graph).to_dict(), args.output)
    elif args.command == "make-event-demo":
        graph = build_synthetic_collective_graph(args.scenario)
        _write_json(graph.to_dict(), args.output_graph)
        _write_json(
            DiscreteEventEngine().simulate(graph).to_dict(), args.output_report
        )
    elif args.command == "train-fixed-config":
        communication = None
        if args.nccl_response is not None:
            communication = NcclResponseTable.load(args.nccl_response).header
        model = FixedConfigForwardOracle(neighbors=args.neighbors, method=args.method).fit(
            load_multirank_records(args.multirank_steps),
            communication_provenance=communication,
        )
        save_fixed_config_artifact(model, args.output)
        _write_json({"artifact": str(args.output), "artifact_type": model.artifact_type,
                     "provenance": model.provenance}, None)
    elif args.command == "evaluate-fixed-config":
        _write_json(
            evaluate_fixed_config(
                load_multirank_records(args.multirank_steps), neighbors=args.neighbors,
                method=args.method,
            ),
            args.output,
        )
    elif args.command == "evaluate-budget-curve":
        _write_json(
            evaluate_profiling_budgets(
                load_multirank_records(args.multirank_steps), budgets=args.budgets,
                neighbors=args.neighbors, method=args.method, strategy=args.strategy,
                protocol=args.protocol, holdout_modulus=args.holdout_modulus,
                holdout_fold=args.holdout_fold,
            ),
            args.output,
        )
    elif args.command == "merge-multirank":
        _write_json(merge_multirank_record_files(args.inputs, args.output), None)
    elif args.command == "aggregate-budget-folds":
        _write_json(
            aggregate_budget_holdout_folds(
                args.inputs, target_mape=args.target_mape,
                target_p95_ape=args.target_p95_ape,
                target_top_k_recall=args.target_top_k_recall,
                target_supported_fraction=args.target_supported_fraction,
            ),
            args.output,
        )
    elif args.command == "compare-fixed-configs":
        _write_json(compare_fixed_config_files(args.source, args.target), args.output)
    elif args.command == "train-cross-config-calibration":
        model, report = train_cross_config_calibrator(
            load_multirank_records(args.source), load_multirank_records(args.target),
            load_fixed_config_artifact(args.source_artifact), budget=args.budget,
            method=args.method, neighbors=args.neighbors,
        )
        save_cross_config_artifact(model, args.output)
        _write_json({
            "artifact": str(args.output), "artifact_type": model.artifact_type,
            "provenance": model.provenance, "training_report": report,
        }, None)
    elif args.command == "evaluate-cross-config-calibration":
        reference = None
        if args.target_reference is not None:
            reference = json.loads(args.target_reference.read_text(encoding="utf-8"))
        _write_json(evaluate_cross_config_calibration(
            load_multirank_records(args.source), load_multirank_records(args.target),
            load_fixed_config_artifact(args.source_artifact), budgets=args.budgets,
            folds=args.folds, neighbors=args.neighbors, target_reference=reference,
        ), args.output)
    elif args.command == "predict-cross-config-calibration":
        calibration = load_cross_config_artifact(args.calibration_artifact)
        source_oracle = load_fixed_config_artifact(args.source_artifact)
        source_row = json.loads(args.source_row.read_text(encoding="utf-8"))
        target_row = json.loads(args.target_row.read_text(encoding="utf-8"))
        source_prediction = source_oracle.predict_raw(source_row, allow_unsupported=True)
        _write_json(calibration.predict(
            source_prediction.candidate_step_us, target_row,
            allow_unsupported=args.allow_unsupported,
        ).to_dict(), args.output)
    elif args.command == "evaluate-service-ranking":
        _write_json(evaluate_service_configuration_ranking(
            load_multirank_records(args.source), load_multirank_records(args.target),
            load_fixed_config_artifact(args.source_artifact),
            load_fixed_config_artifact(args.target_artifact),
            target_calibration_budget=args.target_calibration_budget,
            folds=args.folds, ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms, e2e_slo_ms=args.e2e_slo_ms,
        ), args.output)
    elif args.command == "build-candidate-table":
        workload_policy = None
        if args.workload_policy is not None:
            workload_policy = json.loads(args.workload_policy.read_text(encoding="utf-8"))
        _write_json(build_candidate_table_from_service_report(
            json.loads(args.service_report.read_text(encoding="utf-8")),
            load_multirank_records(args.source), load_multirank_records(args.target),
            gpu_memory_bytes=args.gpu_memory_bytes,
            available_gpus=args.available_gpus,
            source_cost_units=args.source_cost_units,
            target_cost_units=args.target_cost_units,
            workload_policy=workload_policy,
        ), args.output)
    elif args.command == "prune-candidates":
        policy = None
        if args.policy is not None:
            policy = json.loads(args.policy.read_text(encoding="utf-8"))
        _write_json(prune_candidate_table(
            load_candidate_table(args.candidate_table), policy
        ), args.output)
    return 0

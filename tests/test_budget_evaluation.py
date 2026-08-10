from __future__ import annotations

import copy
import json

from moe_perf_model.budget_evaluation import (
    evaluate_profiling_budgets,
    aggregate_budget_holdout_folds,
    select_phase_aware_farthest,
    select_observed_error_aware,
)
from moe_perf_model.fixed_config_oracle import features_from_raw

from test_fixed_config_oracle import _row


def test_budget_curve_reports_coverage_and_does_not_claim_config_ranking():
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate((4, 8, 16, 32, 64))]
    report = evaluate_profiling_budgets(rows, budgets=[2, 4], method="idw")
    assert report["unique_anchor_count"] == 5
    assert report["selection"]["uses_latency_labels"] is False
    assert report["configuration_ranking"]["status"] == "not_applicable"
    assert [row["budget"] for row in report["results"]] == [2, 4]
    assert all(0 <= row["supported_fraction"] <= 1 for row in report["results"])


def test_observed_error_selection_does_not_read_unselected_latency():
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]
    selected, _ = select_observed_error_aware(rows, 3)
    unselected = next(row for row in rows if features_from_raw(row) not in set(selected))
    changed = copy.deepcopy(rows)
    changed[rows.index(unselected)]["timing_us"]["step"] *= 1000
    selected_after, _ = select_observed_error_aware(changed, 3)
    assert selected_after == selected


def test_both_strategies_report_explicit_comparison():
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]
    report = evaluate_profiling_budgets(
        rows, budgets=[4], method="idw", strategy="both"
    )
    assert set(report["curves"]) == {"phase_aware_farthest", "observed_error_aware"}
    assert report["curves"]["observed_error_aware"]["selection"][
        "uses_unprofiled_latency_labels"
    ] is False
    assert report["comparison"][0]["budget"] == 4


def test_fixed_workload_holdout_is_removed_from_acquisition_pool():
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]
    report = evaluate_profiling_budgets(
        rows, budgets=[4], method="idw", strategy="both",
        protocol="fixed_workload_holdout", holdout_modulus=4, holdout_fold=0,
    )
    assert report["test_workload_ids"] == [0, 4]
    assert report["test_features_visible_to_acquisition"] is False
    assert report["unique_anchor_count"] == 6
    assert report["test_unique_signature_count"] == 2
    assert report["curves"]["observed_error_aware"]["results"][0][
        "unprofiled_surface_metrics"
    ]["count"] == 2


def test_observed_error_uses_identical_sixteen_anchor_warm_start():
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate(range(2, 42, 2))]
    baseline = select_phase_aware_farthest(rows, 16)
    adaptive, trace = select_observed_error_aware(rows, 17)
    assert adaptive[:16] == baseline
    assert all(row["reason"] == "phase_aware_farthest_warm_start" for row in trace[:16])


def test_holdout_fold_aggregation_returns_auditable_recommendation(tmp_path):
    rows = [_row(tokens, float(tokens * 10), workload=index)
            for index, tokens in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]
    paths = []
    for fold in (0, 1):
        report = evaluate_profiling_budgets(
            rows, budgets=[4], method="idw", strategy="both",
            protocol="fixed_workload_holdout", holdout_modulus=4,
            holdout_fold=fold,
        )
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    aggregate = aggregate_budget_holdout_folds(
        paths, target_mape=1e9, target_p95_ape=1e9,
        target_top_k_recall=0, target_supported_fraction=0,
    )
    assert aggregate["fold_count"] == 2
    assert aggregate["recommendation"]["minimum_anchor_budget"] == 4
    assert aggregate["hyperparameter_policy"] == "frozen_after_five_fold_evaluation"

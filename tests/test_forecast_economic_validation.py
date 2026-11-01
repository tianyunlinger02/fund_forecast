"""Small reusable checks for the offline economic study, not investment evidence."""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund.forecast import MODEL_ID, _distribution
from rmbfund.forecast_candidates import SHRINK_ID, TREND_ID
from rmbfund.forecast_comparison import evaluate_candidate_rows
from rmbfund.forecast_research import replay_actions
from rmbfund.forecast_economic_validation import (
    EMPIRICAL_ID, common_replay_origin, compare_continuous_actions,
    compare_single_decision_payoffs, continuous_effects, empirical_history,
    fold_coverage, prepare_economic_history,
)


def _series(count=100):
    start = date(2024, 12, 1)
    return [{"date": (start + timedelta(days=i)).isoformat(),
             "nav": str(1 + .001 * i + .02 * np.sin(i / 7))} for i in range(count)]


def _bundle(series, mean=.01, method="synthetic_ridge", first_available=5):
    rows = []
    for origin, point in enumerate(series):
        for lag in (3, 4):
            pred = _distribution(np.log1p(np.asarray([mean])), horizon=20,
                                 entry_lag=lag, method=method)
            baseline = _distribution(np.log1p(np.asarray([.01, -.01, .02])), horizon=20,
                                     entry_lag=lag, method="empirical_historical")
            if origin < first_available:
                pred.update(status="unavailable", reason="synthetic_warmup")
            rows.append({"origin_index": origin, "origin_date": point["date"],
                         "horizon": 20, "entry_lag": lag,
                         "knowledge_cutoff_index": origin, "knowledge_cutoff_date": point["date"],
                         "assumed_known_index": origin + 1,
                         "assumed_known_date": series[origin + 1]["date"] if origin + 1 < len(series) else None,
                         "prediction": pred, "historical_baseline": baseline})
    return {"historical_forecasts": rows}


def _bundles(series):
    old = _bundle(series)
    return {key: prepare_economic_history(series, bundle) for key, bundle in (
        (MODEL_ID, old), (SHRINK_ID, _bundle(series, mean=.006)),
        (EMPIRICAL_ID, empirical_history(old)), (TREND_ID, _bundle(series, mean=.015)))}


def _case(lag=3):
    return {"id": "synthetic", "buy_fee": .0015, "redemption_fee": .005,
            "entry_lag": lag, "redemption_lag": lag, "confirmation_lag": 1,
            "cash_lag": 3, "nav_availability_lag": 1}


def test_fold_scoring_purges_cross_cutoff_targets_and_passive_reference_is_not_a_regressor():
    series = _series()
    bundles = _bundles(series)
    period = {"development": ("2024-12-01", "2024-12-31")}
    result = evaluate_candidate_rows(bundles, heads=((20, 3),), periods=period,
                                     mature_by_period_end=True,
                                     fitted_candidate_ids=(MODEL_ID, SHRINK_ID))
    score = result["heads"][0]["periods"]["development"]
    assert score["all_fitted_common"]["count"] > 0
    assert score["last_target_date"] <= "2024-12-31"
    assert score["actual_origin_end"] == "2024-12-08"
    coverage = fold_coverage(bundles, period)[SHRINK_ID]["H20L3"]["development"]
    assert coverage["expected_origins"] == 31
    assert coverage["not_mature_at_fold_end"] == 23
    assert coverage["mature_valid_outcomes"] == 8


def test_continuous_matrix_starts_together_and_never_drops_an_immature_tail_head():
    series = _series()
    bundles = _bundles(series)
    origin = common_replay_origin(series, bundles, ((20, 3), (20, 4)))
    assert origin == 5
    result = compare_continuous_actions(series, bundles, [_case()])
    assert result["status"] == "available"
    assert len(result["scenarios"]) == 7
    assert {r["replay"]["start_date"] for r in result["scenarios"]} == {series[6]["date"]}
    assert {r["replay"]["end_date"] for r in result["scenarios"]} == {series[-1]["date"]}
    flat_wait = next(r for r in result["scenarios"]
                     if r["candidate_id"] == SHRINK_ID and r["policy"] == "net_payoff")
    assert flat_wait["strategy_summary"]["trade_count"] == 0
    assert flat_wait["strategy_summary"]["return"] == 0
    assert len(flat_wait["replay"]["strategy"]["action_trace"]) == len(series) - 1 - origin
    assert all(r["attribution"]["identity_residual"] == pytest.approx(0, abs=1e-14)
               for r in result["scenarios"])
    assert len(continuous_effects(result["scenarios"])[0]["model_effect_at_same_policy"]) == 2
    old_legacy = next(r for r in result["scenarios"]
                      if r["candidate_id"] == MODEL_ID and r["policy"] == "legacy")
    explicit = [{"origin_index": i, "action": "buy"} for i in range(origin, len(series) - 1)]
    control = replay_actions(series, explicit, entry_fee_rate=.0015, redemption_fee_rate=.005,
                             entry_lag=3, redemption_lag=3, start_index=origin + 1)
    assert old_legacy["replay"]["strategy"]["daily"] == control["strategy"]["daily"]
    assert old_legacy["replay"]["strategy"]["skipped_actions"] == control["strategy"]["skipped_actions"]
    assert all(r["action"] == "buy" for r in old_legacy["replay"]["strategy"]["action_trace"])
    broken = deepcopy(bundles)
    broken[SHRINK_ID]["historical_forecasts"] = [r for r in broken[SHRINK_ID]["historical_forecasts"]
                                                if (r["origin_index"], r["entry_lag"]) != (len(series) - 2, 3)]
    failed = compare_continuous_actions(series, broken, [_case()])
    unavailable = [r for r in failed["scenarios"] if r["status"] == "unavailable"]
    assert len(unavailable) == 2
    assert all(r["unavailable_origin_indices"] == [len(series) - 2] for r in unavailable)


def test_single_decision_net_values_keep_equal_capital_and_block_intervals_are_not_path_returns():
    series = _series(150)
    bundles = _bundles(series)
    result = compare_single_decision_payoffs(bundles, [_case()],
                                             {"all": (series[0]["date"], series[-1]["date"])},
                                             replicates=20)
    case = result["cases"][0]
    row = case["rows"][0]
    new = row["values"][f"{SHRINK_ID}:net_payoff"]
    assert new["flat"]["action"] == "wait"
    assert new["flat"]["initial_capital"] == 1
    assert new["flat"]["actual_terminal_value"] == 1
    assert new["held"]["initial_capital"] == 2
    assert new["held"]["actual_terminal_value"] == pytest.approx((1 + row["actual_change"]) * .995 + 1)
    bootstrap = case["periods"]["all"]
    assert [b["block_length"] for b in bootstrap["blocks"]] == [23, 46]
    assert bootstrap["skill_certified"] is False
    assert "NOT a compounded path" in result["basis"]


def test_outcomes_and_new_fees_cannot_leak_through_the_stateful_callback():
    series = _series()
    bundles = _bundles(series)
    changed = deepcopy(bundles)
    for bundle in changed.values():
        for row in bundle["historical_forecasts"]:
            row["actual_change"] = 1000000
            row["future_nav"] = 0
    normal = compare_continuous_actions(series, bundles, [_case()])
    other = compare_continuous_actions(series, changed, [_case()])
    assert normal == other

"""Reusable numerical/maturity tests, not evidence of investment skill."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund import forecast as core
from rmbfund.forecast_candidates import (
    RISK_ID,
    RISK_METHOD,
    issue_ridge_distribution_forecasts,
    issue_risk_forecasts,
)
from rmbfund.forecast_comparison import (
    _empirical_bundle,
    compare_action_scenarios,
    evaluate_candidate_rows,
)
from rmbfund.forecast_research import (
    _metrics,
    evaluate_forecasts,
    paired_loss_bootstrap,
    replay_actions,
)


def _series(count=600):
    rng = np.random.default_rng(2941)
    x = np.arange(count)
    changes = 0.0004 + 0.003 * np.sin(x / 13) + rng.normal(size=count) * np.where(x < 340, 0.004, 0.018)
    navs = np.exp(np.cumsum(changes))
    start = date(2023, 12, 1)
    return [{"date": (start + timedelta(days=i)).isoformat(), "nav": str(nav)}
            for i, nav in enumerate(navs)]


def _identity(row):
    return row["origin_index"], row["horizon"], row["entry_lag"]


def test_same_log_location_exact_scale_stored_at_origin_and_one_equal_weight_distribution():
    series, heads = _series(), ((20, 3), (20, 0), (1, 0), (5, 0))
    risk = issue_risk_forecasts(series, heads=heads)
    old = issue_ridge_distribution_forecasts(series, candidate_id=core.MODEL_ID, heads=heads)
    logs = np.log([float(row["nav"]) for row in series])
    old_map = {_identity(row): row for row in old["historical_forecasts"]}
    for head in risk["forecasts"]:
        horizon, lag = head["horizon"], head["entry_lag"]
        history = [row for row in risk["historical_forecasts"]
                   if (row["horizon"], row["entry_lag"]) == (horizon, lag)]
        z = []
        for row in history:
            i, prediction = row["origin_index"], row["prediction"]
            mu, scale = prediction.get("ridge_log_location"), prediction.get("risk_scale")
            if mu is None or scale is None:
                continue
            assert scale == pytest.approx(np.std(np.diff(logs)[i - 20:i]) * np.sqrt(horizon), abs=1e-15)
            if old_map[_identity(row)]["prediction"].get("ridge_log_location") is not None:
                assert mu == old_map[_identity(row)]["prediction"]["ridge_log_location"]
            if i + horizon + lag < len(logs):
                z.append((logs[i + horizon + lag] - logs[i + lag] - mu) / scale)
        expected_samples = np.expm1(head["ridge_log_location"] + head["risk_scale"] * np.asarray(z))
        np.testing.assert_array_equal(head["samples"], expected_samples)
        assert head["method"] == RISK_METHOD and head["fallback_used"] is False
        assert head["sample_count"] == head["residual_sample_count"] == len(z)
        assert head["expected_change"] == float(np.sum(expected_samples / len(z)))
        assert head["up_probability"] == np.mean(expected_samples > 0)
        assert head["down_probability"] == np.mean(expected_samples < 0)
        assert head["flat_probability"] == np.mean(expected_samples == 0)
        np.testing.assert_array_equal([head["lower"], head["median_change"], head["upper"]],
                                      np.quantile(expected_samples, [0.1, 0.5, 0.9]))
        original = next(item for item in old["forecasts"]
                        if (item["horizon"], item["entry_lag"]) == (horizon, lag))
        assert head["ridge_log_location"] == original["ridge_log_location"]
        assert head["expected_change"] != original["expected_change"]
    assert risk["candidate_hash"] != old["candidate_hash"] == core.MODEL_HASH


def test_original_kernel_outputs_stay_numerically_identical_and_future_cannot_rewrite_risk():
    rows, origin = _series(), 520
    old = issue_ridge_distribution_forecasts(rows, candidate_id=core.MODEL_ID)
    legacy = core.fit_forecast(rows, product_id="synthetic:test", generated_at="2026-10-29T00:00:00Z",
                               entry_lags=(3,), include_evaluation=True)
    assert old["evaluation_rows"] == legacy["evaluation_rows"]
    assert old["issued_forecasts"] == legacy["issued_forecasts"]
    risk = issue_risk_forecasts(rows)
    prefix = issue_risk_forecasts(rows[:origin + 1])
    changed = copy.deepcopy(rows)
    for row in changed[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.3)
    changed[origin + 10]["invalid_return"] = True
    future = issue_risk_forecasts(changed)
    for bundle in (risk, future):
        for row in bundle["historical_forecasts"]:
            if row["origin_index"] != origin:
                continue
            head = next(item for item in prefix["forecasts"]
                        if (item["horizon"], item["entry_lag"]) == (row["horizon"], row["entry_lag"]))
            for field in ("expected_change", "lower", "upper", "up_probability", "ridge_log_location", "risk_scale", "residual_sample_count"):
                assert row["prediction"][field] == head[field]
    assert [row for row in risk["issued_forecasts"] if row["origin_index"] <= origin] == [
        row for row in future["issued_forecasts"] if row["origin_index"] <= origin]
    # Residual admissions follow the original exact H+L endpoint, including the
    # newest label at equality, never the last 20-return scale's later revision.
    for row in risk["historical_forecasts"]:
        if row["prediction"].get("status") == "available":
            assert row["prediction"]["residual_target_end_index"] <= row["origin_index"]
            assert row["prediction"]["training_target_end_index"] <= row["origin_index"]


def test_zero_scale_is_unavailable_not_epsilon_empirical_fallback_or_dropped_scenario():
    rows = _series()
    for row in rows[-22:]:
        row["nav"] = rows[-23]["nav"]
    risk = issue_risk_forecasts(rows)
    assert not risk["forecasts"]
    assert all(row["reason"] == "invalid_or_nonpositive_risk_scale"
               and row["risk_scale"] is None and row["fallback_used"] is False
               for row in risk["unavailable_forecasts"])
    old = issue_ridge_distribution_forecasts(rows, candidate_id=core.MODEL_ID)
    metrics = evaluate_candidate_rows({core.MODEL_ID: old, RISK_ID: risk}, heads=((20, 3),))
    assert metrics["coverage"][RISK_ID]["fallback_count"] == 0
    assert metrics["coverage"][RISK_ID]["unavailable_count"] >= 1
    # An unavailable issued decision must fail the whole year, including when
    # the target is immature and hence absent from evaluation_rows.
    issued = copy.deepcopy(risk["issued_forecasts"])
    issued[-2]["prediction"].update(status="unavailable", reason="test_missing_scale")
    for scenario in compare_action_scenarios(rows, {RISK_ID: {"issued_forecasts": issued}}, entry_lags=(3,))["scenarios"]:
        if scenario["year"] == 2025:
            assert scenario["status"] == "unavailable"
            assert scenario["unavailable_decision_count"] > 0


def test_interval_score_wis_reliability_and_block_bootstrap_are_paired_and_chronological():
    rows = []
    for i in range(60):
        prediction = {"up_probability": 0.6, "down_probability": 0.4, "flat_probability": 0.0,
                      "expected_change": 0.0, "lower": -0.1, "median_change": 0.0, "upper": 0.1}
        rows.append({"origin_index": i, "origin_date": f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}",
                     "entry_index": i + 3, "target_index": i + 23, "horizon": 20, "entry_lag": 3,
                     "actual_change": 0.2 if i % 2 else -0.2,
                     "prediction": prediction, "historical_baseline": dict(prediction)})
    metrics = _metrics(rows, "prediction")
    assert metrics["interval_score_80"] == pytest.approx(1.2)
    assert metrics["wis_80_median"] == pytest.approx((0.1 + 0.12) / 1.5)
    assert metrics["up_calibration"][3]["count"] == 60
    boot = paired_loss_bootstrap(rows, replicates=100)
    assert [block["block_length"] for block in boot["blocks"]] == [23, 46]
    assert [block["nonoverlap_block_count_upper"] for block in boot["blocks"]] == [2, 1]
    assert boot["skill_certified"] is False
    for block in boot["blocks"]:
        assert block["few_blocks"] is True
        assert all(value == {"mean_delta": 0.0, "lower_95": 0.0, "upper_95": 0.0}
                   for value in block["intervals"].values())
    varying = copy.deepcopy(rows)
    for i, row in enumerate(varying):
        row["prediction"]["expected_change"] = i / 1000
    actual = np.asarray([row["actual_change"] for row in varying])
    delta = np.abs(np.arange(60) / 1000 - actual) - np.abs(actual)
    draws = np.random.default_rng(20261029).choice(np.arange(38), size=(100, 3))
    manual = np.asarray([
        (delta[a:a + 23].sum() + delta[b:b + 23].sum() + delta[c:c + 14].sum()) / 60
        for a, b, c in draws
    ])
    interval = paired_loss_bootstrap(varying, replicates=100)["blocks"][0]["intervals"]["absolute_error"]
    assert interval["mean_delta"] == pytest.approx(np.mean(delta))
    assert [interval["lower_95"], interval["upper_95"]] == pytest.approx(np.quantile(manual, [0.025, 0.975]))
    with pytest.raises(core.ValidationError, match="time order"):
        paired_loss_bootstrap(list(reversed(rows)))
    gap = rows[:20] + rows[40:]
    assert all(block["status"] == "unavailable"
               for block in paired_loss_bootstrap(gap, replicates=100)["blocks"])
    assert evaluate_forecasts([])["skill_certified"] is False


def test_ledger_turnover_fees_and_empirical_strategy_use_identical_head_and_no_in_transit_reuse():
    rows = _series(50)
    actions = [{"origin_index": i, "action": action} for i, action in
               ((0, "buy"), (5, "redeem"), (9, "buy"), (13, "buy"), (20, "redeem"), (27, "buy"))]
    replay = replay_actions(rows, actions, entry_fee_rate=0.0015, redemption_fee_rate=0.005,
                            entry_lag=3, redemption_lag=3)
    path = replay["strategy"]
    assert any(row["cash_in_transit"] > 0 for row in path["daily"])
    assert any(item["reason"] == "not_spendable_cash" for item in path["skipped_actions"])
    assert all(trade["gross_notional"] == pytest.approx(trade["units"] * trade["execution_nav"])
               for trade in path["trades"])
    original = issue_ridge_distribution_forecasts(_series(), candidate_id=core.MODEL_ID)
    empirical = _empirical_bundle(original)
    history = {_identity(row): row for row in original["historical_forecasts"]}
    assert len(empirical["issued_forecasts"]) == len(original["issued_forecasts"])
    for row in empirical["issued_forecasts"]:
        assert row["prediction"] == history[_identity(row)]["historical_baseline"]
        assert row["prediction"]["method"] == "empirical_historical"

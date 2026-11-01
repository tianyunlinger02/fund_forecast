"""Small synthetic fixed-shrinkage checks; not evidence of investment skill."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund import forecast as core
from rmbfund.forecast_candidates import (
    SHRINK_HEADS,
    SHRINK_ID,
    SHRINK_METHOD,
    TREND_ID,
    candidate_definition,
    candidate_hash,
    issue_candidate_forecasts,
    issue_ridge_distribution_forecasts,
    issue_shrinkage_forecasts,
)


def _series(count=420, *, start=date(2024, 6, 1)):
    rng = np.random.default_rng(61030)
    increments = 0.0003 + 0.003 * np.sin(np.arange(count) / 11) + rng.normal(0, 0.008, count)
    return [
        {"date": (start + timedelta(days=i)).isoformat(), "nav": str(nav)}
        for i, nav in enumerate(np.exp(np.cumsum(increments)))
    ]


def _identity(row):
    return row["origin_index"], row["horizon"], row["entry_lag"]


def _valid_label(rows, origin, horizon, lag):
    endpoint = origin + horizon + lag
    return endpoint < len(rows) and not any(
        row.get("invalid_return", False) or row.get("label_valid") is False
        for row in rows[origin + lag + 1:endpoint + 1]
    )


def test_fixed_half_log_contribution_hand_calculation_and_research_identity():
    location = core._shrink50_log_location(0.10, 0.02)
    assert location == pytest.approx(0.06)
    assert 0.08 - location == pytest.approx(0.02)  # Own raw matured error, not old -0.02.
    definition = candidate_definition(SHRINK_ID)
    assert SHRINK_ID == "nav_ridge_shrink50_empirical_v1"
    assert SHRINK_HEADS == ((20, 3), (20, 4))
    assert definition["shrink_weight"] == 0.5
    assert definition["location_definition"]["ridge_alpha"] == 10.0
    assert definition["minimum_training_samples"] == 120
    assert definition["minimum_residual_samples"] == 40
    assert definition["runtime_issuance"] is definition["promotion"] is False
    assert definition["hyperparameter_search"] is False
    assert definition["location_model_hash"] == core.MODEL_HASH
    assert candidate_hash(SHRINK_ID) not in (core.MODEL_HASH, core.RUNTIME_MODEL_HASH)
    assert SHRINK_ID not in core._canonical(core.RUNTIME_MODEL_DEFINITION)


def test_same_matured_feature_qualified_training_mean_and_own_raw_residual_distribution():
    rows = _series(480)
    rows[300]["invalid_return"] = True
    rows[355]["label_valid"] = False
    result = issue_shrinkage_forecasts(rows)
    old = issue_ridge_distribution_forecasts(rows, candidate_id=core.MODEL_ID, heads=SHRINK_HEADS)
    logs = np.log([float(row["nav"]) for row in rows])
    for head in result["forecasts"]:
        horizon, lag = head["horizon"], head["entry_lag"]
        span = horizon + lag
        history = [row for row in result["historical_forecasts"]
                   if (row["horizon"], row["entry_lag"]) == (horizon, lag)]
        originals = [row for row in old["forecasts"]
                     if (row["horizon"], row["entry_lag"]) == (horizon, lag)]
        assert head["old_ridge_log_location"] == originals[0]["ridge_log_location"]
        assert head["conditional_log_contribution"] == (
            head["old_ridge_log_location"] - head["training_log_mean"]
        )
        assert head["candidate_log_location"] == (
            head["training_log_mean"] + 0.5 * head["conditional_log_contribution"]
        )
        qualified = [
            i for i in range(60, len(rows) - span)
            if _valid_label(rows, i, horizon, lag) and not any(
                row.get("invalid_return", False) or row.get("label_valid") is False
                for row in rows[i - 59:i + 1]
            )
        ]
        targets = [logs[i + span] - logs[i + lag] for i in qualified]
        assert head["training_sample_count"] == len(targets)
        assert head["training_log_mean"] == pytest.approx(np.mean(targets), abs=1e-15)
        errors = [
            logs[row["origin_index"] + span] - logs[row["origin_index"] + lag]
            - row["prediction"]["candidate_log_location"]
            for row in history
            if row["prediction"]["candidate_log_location"] is not None
            and _valid_label(rows, row["origin_index"], horizon, lag)
        ]
        samples = np.expm1(head["candidate_log_location"] + np.asarray(errors))
        np.testing.assert_array_equal(head["samples"], samples)
        assert head["sample_count"] == head["residual_sample_count"] == len(errors)
        assert head["residual_log_mean"] == float(np.mean(errors))
        assert head["expected_change"] == float(np.sum(samples / len(samples)))
        assert head["up_probability"] == np.mean(samples > 0)
        assert head["down_probability"] == np.mean(samples < 0)
        assert head["flat_probability"] == np.mean(samples == 0)
        np.testing.assert_array_equal(
            [head["lower"], head["median_change"], head["upper"]],
            np.quantile(samples, [0.1, 0.5, 0.9]),
        )
        # The first 40 errors include forecasts issued before the distribution
        # itself was available; admission occurs at full H+L, including equality.
        first_location = next(row["origin_index"] for row in history
                              if row["prediction"]["candidate_log_location"] is not None)
        assert history[first_location + span - 1]["prediction"]["residual_sample_count"] == 0
        assert history[first_location + span]["prediction"]["residual_sample_count"] == 1
        assert history[first_location + span + 38]["prediction"]["status"] == "unavailable"
        first_available = history[first_location + span + 39]["prediction"]
        assert first_available["status"] == "available"
        assert first_available["residual_sample_count"] == 40
        assert first_available["method"] == SHRINK_METHOD


@pytest.mark.parametrize("candidate_id", [SHRINK_ID, TREND_ID])
def test_full_history_prefix_and_future_nav_event_invariance(candidate_id):
    def issue(rows):
        return issue_candidate_forecasts(
            rows, candidate_id=candidate_id, heads=SHRINK_HEADS, include_history=True,
        )

    rows, origin = _series(), 350
    original = issue(rows)
    prefix = issue(rows[:origin + 1])
    changed = copy.deepcopy(rows)
    for row in changed[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.4)
    changed[origin + 5]["invalid_return"] = True
    changed[origin + 7]["label_valid"] = False
    future = issue(changed)
    history = original["historical_forecasts"]
    assert len(history) == len(rows) * len(SHRINK_HEADS)
    assert [_identity(row) for row in history] == sorted(_identity(row) for row in history)
    assert len({_identity(row) for row in history}) == len(history)
    past = [row for row in history if row["origin_index"] <= origin]
    assert past == [row for row in future["historical_forecasts"] if row["origin_index"] <= origin]
    for full, cut in zip(past, prefix["historical_forecasts"], strict=True):
        assert full["prediction"] == cut["prediction"]
        assert full.get("historical_baseline") == cut.get("historical_baseline")
    assert [row for row in original["issued_forecasts"] if row["origin_index"] <= origin] == [
        row for row in future["issued_forecasts"] if row["origin_index"] <= origin
    ]
    for row in history:
        assert not {"actual_change", "actual_log_return", "target_date", "issued_at"} & row.keys()
        assert row["prediction"]["method"] != "empirical_historical"
        assert row["prediction"]["status"] in ("available", "unavailable")
        for kind in ("training", "residual"):
            endpoint = row["prediction"].get(f"{kind}_target_end_index")
            assert endpoint is None or endpoint <= row["origin_index"]
        if "historical_baseline" in row:
            assert row["historical_baseline"]["method"] == "empirical_historical"
    assert history[-1]["origin_index"] == len(rows) - 1
    assert len(future["historical_forecasts"]) == len(history)  # Do not filter invalid/immature tails.


def test_legacy_outputs_unchanged_and_trend_history_only_adds_explicit_chronology():
    rows = _series()
    args = dict(product_id="synthetic:shrink", generated_at="2026-10-30T00:00:00Z",
                entry_lags=(3, 4), include_evaluation=True)
    v1 = core.fit_forecast(rows, **args)
    v2 = core.fit_runtime_forecast(rows, **args)
    heads = ((1, 0), (5, 0), (20, 0), *SHRINK_HEADS)
    old = issue_ridge_distribution_forecasts(rows, candidate_id=core.MODEL_ID, heads=heads)
    assert old["evaluation_rows"] == v1["evaluation_rows"]
    assert old["issued_forecasts"] == v1["issued_forecasts"]
    legacy_trend = issue_candidate_forecasts(rows, candidate_id=TREND_ID, heads=SHRINK_HEADS)
    full_trend = issue_candidate_forecasts(
        rows, candidate_id=TREND_ID, heads=SHRINK_HEADS, include_history=True,
    )
    assert {key: value for key, value in full_trend.items()
            if key != "historical_forecasts"} == legacy_trend
    assert "historical_forecasts" not in legacy_trend
    assert any(row["origin_date"] < "2025-01-01" and row["prediction"]["status"] == "available"
               for row in full_trend["historical_forecasts"])
    issue_shrinkage_forecasts(rows)
    assert core.fit_forecast(rows, **args) == v1
    assert core.fit_runtime_forecast(rows, **args) == v2
    assert all(row["origin_date"] >= "2024-12-31" for row in v1["issued_forecasts"])
    with pytest.raises(core.ValidationError, match="include_history"):
        issue_candidate_forecasts(rows, candidate_id=TREND_ID, include_history=1)


def test_unavailable_never_borrows_baseline_and_zero_volatility_does_not_disable_shrink():
    short = issue_shrinkage_forecasts(_series(180, start=date(2025, 1, 1)))
    assert not short["forecasts"] and len(short["unavailable_forecasts"]) == 2
    for row in short["historical_forecasts"]:
        head = row["prediction"]
        assert head["status"] == "unavailable" and head["method"] == SHRINK_METHOD
        assert head["fallback_used"] is False and "expected_change" not in head
    assert short["evaluation_rows"] and short["issued_forecasts"]
    assert all(row["prediction"]["status"] == "unavailable"
               for row in short["evaluation_rows"] + short["issued_forecasts"])
    flat = _series(320)
    for row in flat:
        row["nav"] = "1"
    available = issue_shrinkage_forecasts(flat)
    assert len(available["forecasts"]) == 2
    assert all(head["expected_change"] == 0 and head["candidate_log_location"] == 0
               and head["residual_log_mean"] == 0 and "risk_scale" not in head
               for head in available["forecasts"])
    flat[-1]["invalid_return"] = True
    invalid = issue_shrinkage_forecasts(flat)
    assert not invalid["forecasts"]
    assert all(head["reason"] == "feature_window_crosses_invalid_event_or_shorter_than_60"
               and head["fallback_used"] is False
               and head["historical_baseline"]["status"] == "available"
               for head in invalid["unavailable_forecasts"])

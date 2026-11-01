"""Synthetic actual issues, not backtests or evidence of investment skill."""

from __future__ import annotations


from copy import deepcopy


import pytest


from rmbfund.domain import ValidationError, timestamp


from rmbfund.forecast_models import INVESTMENT_MODEL_ID, NAV_MODEL_ID
from rmbfund.forecast_outcomes import RAW_NAV_TARGET, summarize_outcomes


ISSUED = "2026-01-01T12:00:00+08:00"


SCORED = "2026-01-05T12:00:00+08:00"


AS_OF = "2026-02-10T12:00:00+08:00"


def _head(*, horizon=1, lag=2, probability=0.8, expected=0.08, lower=-0.05, upper=0.15):
    return {
        "status": "available",
        "horizon": horizon,
        "entry_lag": lag,
        "up_probability": probability,
        "down_probability": 1 - probability,
        "flat_probability": 0.0,
        "expected_change": expected,
        "median_change": expected,
        "lower": lower,
        "upper": upper,
        "sample_count": 20,
        "method": "ridge_prequential_empirical",
        "interval_coverage": 0.8,
    }


def _issue(
    record_id="one",
    *,
    generated_at=ISSUED,
    recorded=True,
    baseline=True,
    horizon=1,
    lag=2,
    primary_lag=2,
    role="primary",
    actual=0.1,
    entry="2026-01-02",
    target="2026-01-03",
    probability=0.8,
):
    head = _head(horizon=horizon, lag=lag, probability=probability)
    if baseline:
        head["historical_baseline"] = {
            **_head(
                horizon=horizon, lag=lag, probability=0.6, expected=0.02, lower=-0.03, upper=0.05
            ),
            "method": "empirical_historical",
            "target_definition": {"id": RAW_NAV_TARGET},
        }
    row = {
        "record_id": record_id,
        "product_id": "synthetic:fund:C",
        "generated_at": generated_at,
        "model_id": "nav_ridge_empirical_v2",
        "model_hash": "hash-" + record_id,
        "input_hash": "input-" + record_id,
        "target_definition": {"id": RAW_NAV_TARGET},
        "anchor": {"date": "2025-12-31", "nav": "1"},
        "baseline_entry_lag": primary_lag,
        "horizon": horizon,
        "entry_lag": lag,
        "role": role,
        "roles": [role],
        "head": head,
        "outcome": None,
        "as_of": AS_OF,
    }
    if recorded:
        eligible = entry > timestamp(generated_at).date().isoformat()
        scored = max(SCORED, target + "T23:00:00+08:00")
        row["outcome"] = {
            "id": "outcome-" + record_id,
            "forecast_id": record_id,
            "product_id": row["product_id"],
            "horizon": horizon,
            "entry_lag": lag,
            "issued_at": generated_at,
            "scored_at": scored,
            "kind": "forward" if eligible else "nowcast",
            "eligible_forward": eligible,
            "entry": {"date": entry, "nav": "1", "provenance": {"available_at": scored}},
            "target": {
                "date": target,
                "nav": str(1 + actual),
                "provenance": {"available_at": scored},
            },
            "actual_change": actual,
            "up_probability": probability,
            "brier_score": (probability - int(actual > 0)) ** 2,
            "interval_covered": head["lower"] <= actual <= head["upper"],
            "input_hash": row["input_hash"],
        }
    return row


def _dates(*days):
    return [
        {"date": day, "provenance": {"available_at": day + "T23:00:00+08:00"}}
        for day in (days or ("2025-12-31", "2026-01-01", "2026-01-02", "2026-01-03"))
    ]


def _summary(rows, **kwargs):
    return summarize_outcomes(rows, horizon=1, **kwargs)


def test_zero_results_are_unknown_not_zero_percent_success():
    result = _summary([])
    assert all(number == 0 for number in result["counts"].values())
    assert result["forward_metrics"]["count"] == result["paired"]["count"] == 0
    assert all(value is None for key, value in result["forward_metrics"].items() if key != "count")
    assert all(value is None for value in result["paired"]["delta"].values())
    assert result["groups"] == result["windows"] == result["issues"] == []


def test_forward_metrics_and_same_issue_paired_delta_are_on_identical_windows():
    issue = _issue()
    before = deepcopy(issue)
    result = _summary([issue])
    assert issue == before
    assert result["counts"]["issued"] == result["counts"]["recorded_forward"] == 1
    assert result["counts"]["forward_windows"] == result["counts"]["paired"] == 1
    assert result["forward_metrics"] == pytest.approx(
        {
            "count": 1,
            "brier_score": 0.04,
            "mae": 0.02,
            "interval_coverage": 1,
            "interval_width": 0.2,
            "direction_accuracy": 1,
        }
    )
    assert result["paired"]["baseline"] == pytest.approx(
        {
            "count": 1,
            "brier_score": 0.16,
            "mae": 0.08,
            "interval_coverage": 0,
            "interval_width": 0.08,
            "direction_accuracy": 1,
        }
    )
    assert result["paired"]["delta"] == pytest.approx(
        {
            "brier_score": -0.12,
            "mae": -0.06,
            "interval_coverage": 1,
            "interval_width": 0.12,
            "direction_accuracy": 0,
        }
    )
    assert result["windows"][0]["window_source"] == "outcome"


def test_first_issue_missing_outcome_blocks_a_later_better_scored_refresh():
    first = _issue("first", recorded=False)
    later = _issue("later", generated_at="2026-01-01T13:00:00+08:00", probability=0.99)
    result = _summary([later, first], series=_dates())
    assert result["counts"]["issued"] == 2
    assert result["counts"]["recorded_forward"] == result["counts"]["unrecorded"] == 1
    assert result["counts"]["unique_windows"] == result["counts"]["selected_unrecorded"] == 1
    assert result["counts"]["forward_windows"] == 0
    assert result["forward_metrics"]["count"] == result["paired"]["count"] == 0
    assert result["windows"][0]["record_id"] == "first"
    assert [row["selected"] for row in result["issues"]] == [True, False]


def test_model_families_keep_their_own_earliest_issue_and_never_pool_scores():
    first = _issue("nav-original", baseline=False)
    first["model_id"] = "nav_ridge_empirical_v1"
    refresh = _issue("nav-refresh", generated_at="2026-01-01T13:00:00+08:00", probability=0.99)
    investment = _issue("investment", generated_at="2026-01-01T14:00:00+08:00", probability=0.4)
    investment["model_id"] = INVESTMENT_MODEL_ID
    rows = [refresh, investment, first]
    result = _summary(rows)
    assert result["counts"]["issued"] == 3
    assert result["counts"]["forward_windows"] == result["forward_metrics"]["count"] == 2
    assert result["counts"]["nonoverlapping_windows"] == 2
    assert result["counts"]["paired"] == result["paired"]["count"] == 1
    assert result["metrics_scope"] == "models_only" and result["metrics_available"] is False
    assert result["paired"]["comparison_state"] == "not_pooled"
    assert all(value is None for key, value in result["forward_metrics"].items() if key != "count")
    assert all(value is None for value in result["paired"]["delta"].values())
    by_model = {group["model_id"]: group for group in result["by_model"]}
    assert by_model[NAV_MODEL_ID]["windows"][0]["record_id"] == "nav-original"
    assert by_model[NAV_MODEL_ID]["paired"]["count"] == 0  # Never borrow the later baseline.
    assert by_model[INVESTMENT_MODEL_ID]["windows"][0]["record_id"] == "investment"
    assert by_model[INVESTMENT_MODEL_ID]["forward_metrics"]["brier_score"] == pytest.approx(0.36)
    for selected in ("nav_ridge_empirical_v1", NAV_MODEL_ID):
        nav = _summary(rows, model_id=selected)
        assert nav["counts"]["issued"] == 2 and nav["counts"]["forward_windows"] == 1
        assert nav["forward_metrics"]["brier_score"] == pytest.approx(0.04)
        assert nav["paired"]["count"] == 0
    first["outcome"] = None
    pending = _summary(rows)
    groups = {group["model_id"]: group for group in pending["by_model"]}
    assert groups[NAV_MODEL_ID]["counts"]["selected_unrecorded"] == 1
    assert groups[NAV_MODEL_ID]["forward_metrics"]["count"] == 0
    assert groups[INVESTMENT_MODEL_ID]["forward_metrics"]["count"] == 1


def test_same_day_economic_entry_is_nowcast_despite_later_actual_availability():
    row = _issue(entry="2026-01-01", target="2026-01-02")
    result = _summary([row])
    assert result["counts"]["recorded_nowcast"] == result["counts"]["selected_nowcast"] == 1
    assert result["forward_metrics"]["count"] == result["paired"]["count"] == 0
    row["outcome"].update(kind="forward", eligible_forward=True)
    with pytest.raises(ValidationError, match="前向分类"):
        _summary([row])

"""Stable calculation contracts for the pure timing page/reconstructed replay."""

from copy import deepcopy
from datetime import date, timedelta
import json

import pytest

from rmbfund.domain import ValidationError
from rmbfund.forecast_decision import build_decision
from rmbfund.timing_scenario import (
    build_calendar_scenario,
    build_scenario,
    build_sale_date_comparison,
    build_timing_decision,
    next_lower_fee_tier,
    resolve_fees,
    resolve_plan_fees,
    summarize_calendar_history,
    summarize_history,
)


SUBJECT = {"id": "open_fund:example:C", "kind": "open_fund", "share_class": "C"}


def head(change=0.04, *, horizon=5, lag=2):
    return {"horizon": horizon, "entry_lag": lag, "status": "available",
            "expected_change": change, "lower": -0.1, "upper": 0.2,
            "up_probability": 0.6, "down_probability": 0.2, "flat_probability": 0.2,
            "samples": [-0.1, -0.01, 0.0, 0.01, 0.1]}


def prediction(change=0.04):
    return {"anchor": {"date": "2026-01-01", "nav": "1"},
            "entry_assumptions": {"baseline_entry_lag": 2},
            "execution_forecasts": [head(change)]}


def navs(values):
    return [{"date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(), "nav": value}
            for i, value in enumerate(values)]


def action_rows(value):
    return {row["action"]: row for row in value["rows"]}


@pytest.mark.parametrize("change,buy_fee,sell_fee", [
    (0.1, 0.01, 0.005), (-0.1, 0.01, 0.005), (0.0, 0.0, 0.0),
    (0.01, 0.01, 0.01), (-0.01, 0.01, 0.01),
])
def test_live_values_match_decision_and_equal_capital(change, buy_fee, sell_fee):
    value = prediction(change)
    result = build_scenario(value, subject=SUBJECT, horizon=5, base_amount=10000,
                            subscription_fee=buy_fee, redemption_fee=sell_fee)
    rows = action_rows(result)
    expected = {
        "buy": 10000 * (1 + change) / (1 + buy_fee), "wait": 10000,
        "hold": 10000 * (1 + change) + 10000,
        "add": 10000 * (1 + change) + 10000 * (1 + change) / (1 + buy_fee),
        "reduce50": 10000 * ((1 + change) / 2 + (1 - sell_fee) / 2) + 10000,
        "exit": 10000 * (1 - sell_fee) + 10000,
    }
    for action, terminal in expected.items():
        initial = 10000 if action in {"buy", "wait"} else 20000
        assert rows[action]["initial_value"] == initial
        assert rows[action]["expected_value"] == pytest.approx(terminal)
        assert rows[action]["expected_change"] == pytest.approx(terminal / initial - 1)
    decision = build_decision(value, horizon=5, subscription_fee=buy_fee,
                              redemption_fee=sell_fee)
    assert result["decision"] == decision
    assert rows["buy"]["advantage"] == pytest.approx(10000 * decision["buy_advantage_vs_cash"])
    assert rows["add"]["advantage"] == rows["buy"]["advantage"]
    assert rows["hold"]["advantage"] == pytest.approx(10000 * decision["hold_advantage_vs_exit"])
    assert rows["exit"]["advantage"] == -rows["hold"]["advantage"]
    assert rows["reduce50"]["advantage"] == -rows["hold"]["advantage"] / 2
    assert rows["buy"]["better_probability"] == decision["primary"]["buy_profit_probability"]
    assert rows["reduce50"]["better_probability"] == decision["primary"]["sell_better_probability"]
    assert rows["exit"]["lower_advantage"] <= rows["exit"]["upper_advantage"]
    json.dumps(result, allow_nan=False)


def test_past_purchase_is_not_compounded_into_future_or_market_probability():
    series = navs([1, 1.1, 1.2])
    value = prediction()
    before = deepcopy(value)
    first = build_scenario(value, subject=SUBJECT, series=series, purchase_date=series[0]["date"],
                           horizon=5, subscription_fee=0.0015, redemption_fee=0.005)
    second = build_scenario(value, subject=SUBJECT, series=series, purchase_date=series[1]["date"],
                            horizon=5, subscription_fee=0.0015, redemption_fee=0.005)
    assert first["purchase"] == {"date": "2026-01-01", "nav": 1.0,
        "latest_date": "2026-01-03", "latest_nav": 1.2, "change": pytest.approx(0.2),
        "profit": pytest.approx(2000), "holding_days": 2, "status": "available"}
    assert first["decision"] == second["decision"]
    assert first["market"] == second["market"]
    assert first["rows"] == second["rows"]
    assert first["purchase"]["profit"] != second["purchase"]["profit"]
    assert value == before
    missing = build_scenario(value, subject=SUBJECT, series=series,
                             purchase_date="2025-12-31", horizon=5)
    assert missing["purchase"]["status"] == "missing_nav"
    assert missing["purchase"]["nav"] is None


def test_exact_primary_head_and_unknown_fees_have_no_substitute():
    value = prediction()
    value["forecasts"] = [head(0.99, lag=0)]
    value["execution_forecasts"] += [head(-0.3, lag=1), head(-0.4, horizon=20)]
    result = build_scenario(value, subject=SUBJECT, horizon=5)
    assert result["decision"]["primary_head"] == value["execution_forecasts"][0]
    rows = action_rows(result)
    assert rows["buy"]["expected_value"] is None
    assert rows["exit"]["expected_value"] is None
    assert rows["hold"]["expected_value"] == pytest.approx(20400)
    assert rows["hold"]["advantage"] is None
    value["execution_forecasts"].pop(0)
    missing = build_scenario(value, subject=SUBJECT, horizon=5)
    assert missing["status"] == "missing_primary_head"
    assert missing["rows"] == []
    del value["entry_assumptions"]
    assert build_scenario(value, subject=SUBJECT, horizon=5)["status"] == "missing_primary_lag"


@pytest.mark.parametrize("base,fee,series", [
    (0, 0, []), (-1, 0, []), (float("nan"), 0, []), (float("inf"), 0, []),
    (100, -0.01, []), (100, 1.01, []), (100, float("nan"), []),
    (100, True, []), (100, 0, [{"date": "2026-01-02", "nav": 1},
                              {"date": "2026-01-01", "nav": 2}]),
    (100, 0, [{"date": "2026-01-01", "nav": 0}]),
    (100, 0, [{"date": "20260101", "nav": 1}]),
])
def test_invalid_amount_fee_and_actual_nav_dates(base, fee, series):
    with pytest.raises(ValidationError):
        build_scenario(prediction(), subject=SUBJECT, horizon=5, base_amount=base,
                       subscription_fee=fee, series=series)


def test_fee_resolution_public_zero_inclusive_bands_and_explicit_simulation():
    record = {"subscription_rate": 0, "holding_day_basis": "unknown",
              "redemption_bands": [{"min_days": 0, "max_days": 6, "rate": 0.015},
                                   {"min_days": 7, "max_days": 29, "rate": 0.005},
                                   {"min_days": 30, "max_days": None, "rate": 0}],
              "evidence_refs": ["fees:public"], "observed_at": "2026-02-01T08:00:00+08:00",
              "available_at": "2026-02-01T08:00:00+08:00", "channel_verified": False,
              "management_rate": 0.99}
    for day, rate in [("2026-01-07", 0.015), ("2026-01-08", 0.005), ("2026-01-31", 0)]:
        result = resolve_fees(record, purchase_date="2026-01-01", valuation_date=day)
        assert result["subscription_fee"] == 0
        assert result["redemption_fee"] == rate
        assert result["subscription"]["source"] == "public_standard"
        assert result["holding_day_basis"] == "calendar_days_simulation"
        assert "模拟" in result["redemption"]["label"]
        assert result["evidence_refs"] == ["fees:public"]
        assert result["channel_verified"] is False
    fallback = resolve_fees(None, purchase_date=None, valuation_date="2026-02-01")
    assert (fallback["subscription_fee"], fallback["redemption_fee"]) == (0.0015, 0.005)
    assert fallback["subscription"]["source"] == fallback["redemption"]["source"] == "illustrative"
    no_nav = resolve_fees(None, purchase_date=None, valuation_date=None)
    assert no_nav["valuation_date"] is None and no_nav["holding_days"] is None
    assert no_nav["redemption"]["source"] == "illustrative"
    flat = resolve_fees({"subscription_rate": 0, "redemption_bands": [
        {"min_days": 0, "max_days": None, "rate": 0}]},
        purchase_date=None, valuation_date=None)
    assert flat["subscription_fee"] == flat["redemption_fee"] == 0
    assert flat["redemption"]["source"] == "public_standard"
    override = resolve_fees(record, purchase_date="2026-01-01", valuation_date="2026-01-08",
                            subscription_override=0.02, redemption_override=0)
    assert override["subscription_fee"] == 0.02 and override["redemption_fee"] == 0
    assert override["redemption"]["source"] == "override"


def history(series, changes, *, horizon=1, lag=1):
    records = []
    for origin, change in enumerate(changes):
        record = {"horizon": horizon, "entry_lag": lag, "origin_index": origin,
                  "origin_date": series[origin]["date"], "knowledge_cutoff_index": origin,
                  "knowledge_cutoff_date": series[origin]["date"], "assumed_known_index": origin + 1,
                  "assumed_known_date": series[origin + 1]["date"] if origin + 1 < len(series) else None,
                  "prediction": head(change, horizon=horizon, lag=lag)}
        records.append(record)
    return records


def test_history_uses_predictions_not_fabricated_outcomes_and_retains_tail_and_invalid():
    series = navs([1, 1, 0.9, 1.1, 0.8, 1.2])
    records = history(series, [0.1, -0.1, -0.1, 0.1, 0.1, 0.1])
    records[2]["historical_baseline"] = head(0.1, horizon=1, lag=1)
    series[4]["invalid_return"] = True
    kwargs = {"horizon": 1, "entry_lag": 1, "subscription_fee": 0,
              "redemption_fee": 0, "base_amount": 10000}
    first = summarize_history(series, records, **kwargs)
    poisoned = deepcopy(records)
    for row in poisoned:
        row["actual_change"] = 100
        row["outcome"] = {"actual_change": 100, "action": "buy"}
        row["evaluation_rows"] = [{"action": "buy", "actual_change": 100}]
    assert first == summarize_history(series, poisoned, **kwargs)
    assert [row["outcome_status"] for row in first["rows"]] == [
        "mature", "mature", "invalid", "mature", "immature", "immature"]
    assert first["counts"]["wrong_buys"] == 1
    assert first["counts"]["missed_profits"] == 1
    assert first["rows"][0]["actions"]["flat"]["outcome_label"] == "wrong_buy"
    assert first["rows"][1]["actions"]["held"]["outcome_label"] == "missed_profit"
    for row in first["rows"][:2]:
        origin = row["origin_index"]
        actual = float(series[origin + 2]["nav"]) / float(series[origin + 1]["nav"]) - 1
        assert row["actual_change"] == pytest.approx(actual)
        by_action = {item["action"]: item for item in row["comparisons"]}
        assert by_action["hold"]["actual_value"] == pytest.approx(10000 * (1 + actual) + 10000)
        assert by_action["reduce50"]["realized_advantage"] == pytest.approx(-10000 * actual / 2)
    json.dumps(first, allow_nan=False)


def test_history_baseline_overlap_unavailable_and_fixed_fee_assumptions():
    series = navs([1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    records = history(series, [0.1] * 9, horizon=5, lag=1)
    records[0]["prediction"] = {"horizon": 5, "entry_lag": 1, "status": "insufficient_history"}
    records[1]["historical_baseline"] = head(-0.1, horizon=5, lag=1)
    result = summarize_history(series, records, horizon=5, entry_lag=1,
                               subscription_fee=0.01, redemption_fee=0.005)
    assert result["kind"] == "reconstructed"
    assert result["counts"]["total"] == 9
    assert result["counts"]["unavailable"] == 1
    assert result["rows"][0]["status"] == "unavailable"
    assert result["rows"][0]["outcome_status"] == "mature"
    assert result["counts"]["mature"] == 2
    assert result["counts"]["nonoverlapping_windows"] == 1
    assert result["counts"]["overlapping_windows"] == 1
    assert result["paired"]["count"] == 1
    assert result["paired"]["model"]["flat"]["mean_realized_advantage"] > 0
    assert result["paired"]["historical_baseline"]["flat"]["mean_realized_advantage"] == 0
    assert result["fees"]["basis"] == "fixed_across_replay"
    assert "cumulative_return" not in result["metrics"]


@pytest.mark.parametrize("actual", [-0.1, 0.1])
def test_paired_payoff_uses_common_initial_capital_not_opposite_action_margins(actual):
    series = navs([1, 1, 1 + actual])
    records = history(series, [0.1])
    records[0]["historical_baseline"] = head(-0.1, horizon=1, lag=1)
    result = summarize_history(series, records, horizon=1, entry_lag=1,
                               subscription_fee=0, redemption_fee=0)
    paired = result["paired"]
    assert paired["model"]["held"]["mean_realized_payoff"] == pytest.approx(10000 * actual)
    assert paired["historical_baseline"]["held"]["mean_realized_payoff"] == 0
    assert paired["delta"]["held"]["mean_realized_payoff"] == pytest.approx(10000 * actual)
    exit_row = next(row for row in result["rows"][0]["comparisons"] if row["action"] == "exit")
    assert exit_row["outcome_label"] == ("avoided_loss" if actual < 0 else "missed_profit")


def test_anchor_entry_before_knowledge_is_nowcast_not_executable_replay():
    series = navs([1, 1.1, 1.2])
    result = summarize_history(series, history(series, [0.1] * 3, lag=0),
                               horizon=1, entry_lag=0, subscription_fee=0, redemption_fee=0)
    assert result["counts"]["nowcast"] == 2
    assert result["counts"]["mature"] == 0
    assert result["counts"]["immature"] == 1
    assert result["metrics"]["held"]["count"] == 0
    assert result["rows"][0]["status"] == "nowcast"
    assert result["rows"][0]["actual_change"] == pytest.approx(0.1)


def test_replay_chronology_is_not_shifted_to_another_nav_date():
    series = navs([1, 1.1, 1.2])
    records = history(series, [0.1])
    records[0]["assumed_known_date"] = series[2]["date"]
    with pytest.raises(ValidationError):
        summarize_history(series, records, horizon=1, entry_lag=1,
                          subscription_fee=0, redemption_fee=0)
    with pytest.raises(ValidationError):
        summarize_history(series, [], horizon=1, entry_lag=None,
                          subscription_fee=0, redemption_fee=0)


def calendar_plan(*, horizon=9, lag=2, **dates):
    return {"kind": "calendar_trade_plan", "decision_date": "2026-01-07",
            "application_date": "2026-01-07", "sale_date": "2026-02-06",
            "deadline": "2026-02-06", "new_confirmation_date": "2026-01-09",
            "horizon": horizon, "entry_lag": lag, "status": "ready",
            "new_shares_sellable": True, **dates}


def calendar_prediction(change=0.04, *, horizon=9, lag=2, samples=None):
    value = prediction(change)
    value.update(model_id="saved_calendar_model", model_hash="saved_model_hash",
                 input_hash="saved_input_hash", request_hash="saved_request_hash")
    value["execution_forecasts"] = [head(change, horizon=horizon, lag=lag)]
    if samples is not None:
        value["execution_forecasts"][0]["samples"] = samples
    return value


def calendar_fees(*, buy=0, current=0, old=0, new=0):
    return {"subscription_fee": buy, "current_redemption_fee": current,
            "future_old_redemption_fee": old, "future_new_redemption_fee": new}


def public_tiers():
    return {"subscription_rate": 0, "holding_day_basis": "confirmation_calendar_days",
            "redemption_bands": [{"min_days": 0, "max_days": 6, "rate": 0.015},
                                 {"min_days": 7, "max_days": 29, "rate": 0.005},
                                 {"min_days": 30, "max_days": None, "rate": 0}],
            "evidence_refs": ["public:fee_table"], "channel_verified": False}


@pytest.mark.parametrize("age,current_rate,future_age,future_rate", [
    (6, 0.015, 7, 0.005), (7, 0.005, 29, 0.005), (29, 0.005, 30, 0), (30, 0, 31, 0),
])
def test_plan_fees_preserve_inclusive_tiers_zero_and_independent_new_age(
    age, current_rate, future_age, future_rate,
):
    origin = date(2026, 1, 1)
    application = (origin + timedelta(days=age)).isoformat()
    sale = (origin + timedelta(days=future_age)).isoformat()
    plan = calendar_plan(application_date=application, sale_date=sale,
                         new_confirmation_date=application)
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date=origin.isoformat())
    assert fees["subscription_fee"] == 0
    assert fees["current_redemption_fee"] == current_rate
    assert fees["future_old_redemption_fee"] == future_rate
    assert fees["contexts"]["current"]["holding_days"] == age
    assert fees["contexts"]["future_old"]["holding_days"] == future_age
    new_age = future_age - age
    expected_new = 0.015 if new_age <= 6 else 0.005 if new_age <= 29 else 0
    assert fees["future_new_redemption_fee"] == expected_new
    assert fees["contexts"]["future_new"]["holding_days"] == new_age
    for context in fees["contexts"].values():
        assert context["confirmation_date_basis"] == "hypothetical_not_actual_record"
        assert context["evidence_refs"] == ["public:fee_table"]
        assert context["channel_verified"] is False
        assert "模拟" in context["holding_day_label"]


def test_plan_fee_overrides_do_not_spread_to_future_lots_and_next_lower_is_optional():
    plan = calendar_plan(sale_date="2026-01-31", new_confirmation_date="2026-01-30")
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2026-01-01",
                             subscription_override=0.02, current_redemption_override=0)
    assert fees["subscription_fee"] == 0.02
    assert fees["current_redemption_fee"] == 0
    assert fees["future_old_redemption_fee"] == 0
    assert fees["future_new_redemption_fee"] == 0.015
    assert fees["contexts"]["current"]["redemption"]["source"] == "override"
    assert fees["contexts"]["future_new"]["redemption"]["source"] == "public_standard"
    assert fees["next_lower_fee"]["status"] == "unavailable"
    override = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2026-01-01",
                                 future_old_redemption_override=0.03,
                                 future_new_redemption_override=0)
    assert override["current_redemption_fee"] == 0.015
    assert override["future_old_redemption_fee"] == 0.03
    assert override["future_new_redemption_fee"] == 0
    next_fee = next_lower_fee_tier(public_tiers(), confirmation_date="2026-01-01",
                                  application_date="2026-01-07", deadline_date="2026-01-07")
    assert next_fee["date"] == "2026-01-08"
    assert next_fee["days_to_wait"] == 1
    assert next_fee["saving_per_base"] == pytest.approx(100)
    assert next_fee["within_deadline"] is False
    fallback = resolve_plan_fees(None, plan=plan, old_confirmation_date=None)
    assert fallback["subscription"]["source"] == "illustrative"
    assert fallback["contexts"]["future_old"]["redemption"]["source"] == "illustrative"


@pytest.mark.parametrize("change", [0.1, -0.1, 0])
def test_calendar_cash_formulas_equal_capital_exact_nonstandard_head_and_legacy_unchanged(change):
    plan, fees = calendar_plan(), calendar_fees(buy=0.01, current=0.02, old=0.005, new=0.015)
    forecast = calendar_prediction(change)
    before = deepcopy(forecast)
    result = build_calendar_scenario(forecast, subject=SUBJECT, plan=plan, fee_basis=fees)
    rows = action_rows(result)
    k, b = 1 + change, 10000
    values = {
        "wait": b, "buy": b * k * 0.985 / 1.01, "hold": b * k * 0.995 + b,
        "add": b * k * 0.995 + b * k * 0.985 / 1.01,
        "reduce50": 0.5 * b * 0.98 + 0.5 * b * k * 0.995 + b, "exit": b * 0.98 + b,
    }
    for action, value in values.items():
        row = rows[action]
        assert row["status"] == "ready"
        assert row["expected_value"] == pytest.approx(value)
        assert row["initial_value"] == (b if action in {"buy", "wait"} else 2 * b)
        assert row["advantage"] == pytest.approx(value - values[row["comparison_action"]])
    assert result["kind"] == "calendar_cash"
    assert result["horizon"] == 9
    assert result["entry_lag"] == 2
    assert result["fees"] == fees
    assert result["end_value_basis"] == "cash_and_redemption_receivable"
    assert result["decision"]["primary_head"] == forecast["execution_forecasts"][0]
    assert result["decision"]["model_identity"]["model_hash"] == "saved_model_hash"
    assert result["decision"]["model_identity"]["horizon"] == 9
    assert forecast == before
    # Schema5's original mark-to-NAV policy still does not liquidate future lots.
    legacy = build_scenario(prediction(change), subject=SUBJECT, horizon=5,
                            subscription_fee=0.01, redemption_fee=0.02)
    assert action_rows(legacy)["buy"]["expected_value"] == pytest.approx(b * k / 1.01)
    assert action_rows(legacy)["hold"]["expected_value"] == pytest.approx(b * k + b)
    with pytest.raises(ValidationError, match="1、5或20"):
        build_scenario(forecast, subject=SUBJECT, horizon=9)
    json.dumps(result, allow_nan=False)


def test_calendar_probabilities_transform_same_samples_and_strictly_exclude_cash_ties():
    plan = calendar_plan()
    forecast = calendar_prediction(0.25, samples=[-0.25, 0, 0.25, 0.5])
    expensive = build_calendar_scenario(forecast, subject=SUBJECT, plan=plan,
                                        fee_basis=calendar_fees(old=0.2, new=0.2))
    rows = action_rows(expensive)
    # r=.25 is an exact cash tie after a 20% future redemption fee, not a win.
    assert rows["buy"]["advantage"] == 0
    assert rows["hold"]["advantage"] == 0
    assert rows["buy"]["better_probability"] == 0.25
    assert rows["add"]["better_probability"] == 0.25
    assert rows["hold"]["better_probability"] == 0.25
    assert rows["reduce50"]["better_probability"] == 0.5
    assert rows["exit"]["better_probability"] == 0.5
    assert expensive["decision"]["branches"]["flat"]["action"] == "wait"
    assert expensive["decision"]["branches"]["held"]["action"] == "hold"
    cheap = build_calendar_scenario(forecast, subject=SUBJECT, plan=plan,
                                    fee_basis=calendar_fees())
    assert action_rows(cheap)["buy"]["better_probability"] == 0.5
    assert expensive["market"] == cheap["market"]
    assert expensive["decision"]["primary_head"] == cheap["decision"]["primary_head"]
    no_samples = deepcopy(forecast)
    no_samples["execution_forecasts"][0].pop("samples")
    summary = build_calendar_scenario(no_samples, subject=SUBJECT, plan=plan,
                                      fee_basis=calendar_fees())
    assert action_rows(summary)["buy"]["better_probability"] is None
    assert summary["market"]["up_probability"] == 0.6


def test_calendar_same_h_l_reasons_and_missing_exact_head_never_use_h20_or_other_lag():
    forecast = calendar_prediction()
    forecast["execution_forecasts"][0]["drivers"] = [
        {"feature": "same_head", "horizon": 9, "entry_lag": 2, "role": "execution",
         "value": 0.1, "contribution_log_return": -0.01},
        {"feature": "wrong_horizon", "horizon": 20, "entry_lag": 2, "role": "execution",
         "value": 0.9, "contribution_log_return": 0.9},
    ]
    forecast["execution_forecasts"].extend([head(0.9, horizon=20), head(0.8, horizon=9, lag=1)])
    result = build_calendar_scenario(forecast, subject=SUBJECT, plan=calendar_plan(),
                                     fee_basis=calendar_fees())
    explanation = result["decision"]["execution"]
    assert [row["feature"] for row in explanation["drivers"]] == ["same_head"]
    assert explanation["identity"]["entry_lag"] == 2
    forecast["execution_forecasts"].pop(0)
    forecast["unavailable_forecasts"] = [{"horizon": 9, "entry_lag": 2,
                                          "status": "unavailable", "reason": "saved_factor_not_qualified"}]
    missing = build_calendar_scenario(forecast, subject=SUBJECT, plan=calendar_plan(),
                                      fee_basis=calendar_fees())
    assert missing["decision"]["primary_head"] is None
    assert missing["status"] == "unavailable_primary_head"
    assert "saved_factor_not_qualified" in missing["decision"]["branches"]["held"]["reasons"]
    assert len(missing["rows"]) == 6
    assert action_rows(missing)["wait"]["expected_value"] == 10000
    assert action_rows(missing)["exit"]["expected_value"] == 20000
    assert action_rows(missing)["hold"]["expected_value"] is None


def test_new_lot_after_sale_only_disables_buy_add_and_keeps_old_lot_comparison():
    plan = calendar_plan(new_confirmation_date="2026-02-09", new_shares_sellable=False)
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2025-12-01")
    assert fees["future_new_redemption_fee"] is None
    assert fees["contexts"]["future_new"]["holding_days"] == -3
    result = build_calendar_scenario(calendar_prediction(), subject=SUBJECT, plan=plan, fee_basis=fees)
    rows = action_rows(result)
    assert result["status"] == "ready"
    for action in ("buy", "add"):
        assert rows[action]["expected_value"] is None
        assert "新份额" in "".join(rows[action]["reasons"])
    for action in ("wait", "hold", "reduce50", "exit"):
        assert rows[action]["status"] == "ready"
        assert rows[action]["expected_value"] is not None
    assert result["decision"]["branches"]["flat"]["action"] == "unavailable"
    assert result["decision"]["branches"]["held"]["action"] == "hold"


@pytest.mark.parametrize("horizon,lag", [(0, 2), (None, 2), (-1, 2), (9, None), (9, -1)])
def test_calendar_no_h_or_l_keeps_known_cash_and_real_unavailable_reasons(horizon, lag):
    result = build_calendar_scenario(calendar_prediction(), subject=SUBJECT,
                                     plan=calendar_plan(horizon=horizon, lag=lag),
                                     fee_basis=calendar_fees(current=0.01))
    rows = action_rows(result)
    assert len(rows) == 6
    assert rows["wait"]["expected_value"] == 10000
    assert rows["exit"]["expected_value"] == 19900
    assert rows["buy"]["expected_value"] is None
    assert rows["buy"]["reasons"]
    assert rows["hold"]["reasons"]


def test_future_old_confirmation_invalidates_current_exit_not_new_buy():
    plan = calendar_plan()
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2026-01-08")
    assert fees["current_redemption_fee"] is None
    result = build_calendar_scenario(calendar_prediction(), subject=SUBJECT, plan=plan, fee_basis=fees)
    rows = action_rows(result)
    assert rows["buy"]["status"] == "ready"
    assert rows["hold"]["status"] == "ready"
    assert rows["exit"]["status"] == "unavailable"
    assert rows["reduce50"]["status"] == "unavailable"
    assert rows["hold"]["advantage"] is None
    assert "旧份额" in "".join(rows["exit"]["reasons"])


def test_calendar_history_reuses_full_same_h_l_stream_and_fixed_fees_for_model_and_baseline():
    series = navs([1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    records = history(series, [0.2] * 9, horizon=3, lag=1)
    records[0]["prediction"] = {"horizon": 3, "entry_lag": 1, "status": "insufficient_history"}
    for record in records:
        record["prediction"].pop("samples", None)
    records[1]["historical_baseline"] = head(-0.1, horizon=3, lag=1)
    records[1]["historical_baseline"].pop("samples")
    records[1]["saved_origin_metadata"] = {"do_not_drop": True}
    series[6]["invalid_return"] = True
    plan = calendar_plan(horizon=3, lag=1)
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2026-01-01")
    before = deepcopy(fees)
    result = summarize_calendar_history(series, records, plan=plan, fee_basis=fees)
    assert result["kind"] == "calendar_cash_fixed_fee_nav_history"
    assert result["fee_context_basis"] == "fixed_across_origins_and_baseline"
    assert result["fees"] == before == fees
    assert len(result["rows"]) == len(records)
    assert result["counts"]["unavailable"] == 1
    assert result["counts"]["mature"] == 1
    assert result["counts"]["invalid"] == 3
    assert result["counts"]["immature"] == 4
    row = result["rows"][1]
    assert row["saved_origin_metadata"] == {"do_not_drop": True}
    assert all(item["better_probability"] is None for item in row["comparisons"]
               if item["action"] != "wait")
    assert all(item["better_probability"] is None for item in row["baseline_comparisons"]
               if item["action"] != "wait")
    assert row["decision"]["primary_horizon"] == row["baseline_decision"]["primary_horizon"] == 3
    actual = series[5]["nav"] / series[2]["nav"] - 1
    model_payoff = 10000 * (1 + actual) * (1 - fees["future_old_redemption_fee"]) - 10000
    exit_payoff = -10000 * fees["current_redemption_fee"]
    assert result["paired"]["delta"]["held"]["mean_realized_payoff"] == pytest.approx(
        model_payoff - exit_payoff)
    assert row["actions"]["held"]["initial_value"] == 20000
    assert row["baseline_actions"]["held"]["initial_value"] == 20000
    json.dumps(result, allow_nan=False)


def test_calendar_replay_nowcasts_not_counted_and_fee_changes_do_not_change_nav_targets():
    series = navs([1, 1.1, 1.2, 1.3])
    records = history(series, [0.1] * 4, horizon=1, lag=0)
    plan = calendar_plan(horizon=1, lag=0)
    first = summarize_calendar_history(series, records, plan=plan, fee_basis=calendar_fees())
    second = summarize_calendar_history(series, records, plan=plan,
                                         fee_basis=calendar_fees(old=0.2, new=0.2))
    assert first["counts"]["nowcast"] == 3
    assert first["counts"]["mature"] == 0
    assert first["metrics"]["held"]["count"] == 0
    assert [row["actual_change"] for row in first["rows"]] == [row["actual_change"] for row in second["rows"]]
    assert build_timing_decision(calendar_prediction(), plan=calendar_plan(),
                                  fee_basis=calendar_fees())["status"] == "ready"


def test_calendar_past_purchase_does_not_supply_new_lot_fee_age_or_change_market():
    series, plan = navs([1, 1.1, 1.2]), calendar_plan()
    fees = resolve_plan_fees(public_tiers(), plan=plan, old_confirmation_date="2025-12-01")
    first = build_calendar_scenario(calendar_prediction(), subject=SUBJECT, series=series,
                                    purchase_date=series[0]["date"], plan=plan, fee_basis=fees)
    second = build_calendar_scenario(calendar_prediction(), subject=SUBJECT, series=series,
                                     purchase_date=series[1]["date"], plan=plan, fee_basis=fees)
    assert first["purchase"]["profit"] != second["purchase"]["profit"]
    assert first["rows"] == second["rows"]
    assert first["decision"] == second["decision"]
    assert first["market"] == second["market"]
    assert first["fees"]["contexts"]["future_new"]["holding_days"] == 28
    assert first["fees"]["contexts"]["future_old"]["holding_days"] == 67


def test_calendar_sale_before_application_retains_known_exit_without_raising():
    plan = calendar_plan(sale_date="2026-01-06")
    result = build_calendar_scenario(calendar_prediction(), subject=SUBJECT, plan=plan,
                                     fee_basis=calendar_fees(current=0.015))
    rows = action_rows(result)
    assert len(rows) == 6
    assert rows["wait"]["expected_value"] == 10000
    assert rows["exit"]["expected_value"] == 19850
    assert rows["hold"]["expected_value"] is None
    assert "早于本次申请日" in "".join(rows["hold"]["reasons"])


def test_calendar_history_does_not_trust_supplied_actions_or_mismatched_baseline():
    series = navs([1, 1.1, 1.2, 1.3, 1.4, 1.5])
    records = history(series, [0.1] * 6, horizon=2, lag=1)
    records[0]["historical_baseline"] = head(-0.1, horizon=20, lag=1)
    records[1]["prediction"]["expected_change"] = float("nan")
    poisoned = deepcopy(records)
    for row in poisoned:
        row.update(actual_change=99, flat_action="sell", held_action="buy",
                   baseline_actions={"flat": {"realized_payoff": 999},
                                     "held": {"realized_payoff": 999}})
    kwargs = {"plan": calendar_plan(horizon=2, lag=1), "fee_basis": calendar_fees()}
    clean = summarize_calendar_history(series, records, **kwargs)
    result = summarize_calendar_history(series, poisoned, **kwargs)
    assert result == clean
    assert result["counts"]["invalid_predictions"] == 1
    assert result["counts"]["baseline_pairs"] == 0
    assert result["rows"][1]["prediction_status"] == "invalid"
    assert result["rows"][0]["actions"]["flat"]["action"] == "buy"


def sale_plan(days=7):
    from rmbfund.trade_calendar import build_trade_plan

    return build_trade_plan(decision_at="2026-01-09T12:00:00+08:00",
                            anchor_date="2026-01-08", calendar_days=days)


def weekend_tiers():
    # The first lower band begins Saturday; a further reduction is Sunday.
    return {"holding_day_basis": "confirmation_calendar_days", "redemption_bands": [
        {"min_days": 0, "max_days": 6, "rate": 0.015},
        {"min_days": 7, "max_days": 7, "rate": 0.005},
        {"min_days": 8, "max_days": None, "rate": 0},
    ]}


def sale_rows(value):
    return {row["candidate_id"]: row for row in value["rows"]}


def sale_prediction(horizon, change=0.01, samples=None):
    value = calendar_prediction(change, horizon=horizon, lag=1, samples=samples)
    value["input_hash"] = f"same-input-H{horizon}"
    return value


def test_sale_candidates_forward_weekend_projection_recompute_rate_and_exact_h_l():
    plan, record = sale_plan(), weekend_tiers()
    forecasts = [sale_prediction(1, -0.01), sale_prediction(5, 0.05)]
    before = deepcopy((plan, record, forecasts))
    result = build_sale_date_comparison(forecasts, plan=plan, fee_record=record,
                                        old_confirmation_date="2026-01-03")
    rows = sale_rows(result)
    assert [(row["sale_date"], row["horizon"], row["entry_lag"]) for row in result["rows"]] == [
        ("2026-01-09", 0, 1), ("2026-01-12", 1, 1), ("2026-01-16", 5, 1)]
    assert result["next_lower_fee"]["date"] == "2026-01-10"
    assert result["next_lower_fee"]["next_rate"] == 0.005
    assert result["next_lower_fee"]["projected_sale_date"] == "2026-01-12"
    assert result["next_lower_fee"]["projected_rate"] == 0
    assert rows["next_lower_fee"]["redemption_fee"] == 0
    assert rows["now"]["expected_value"] == 19850
    assert rows["next_lower_fee"]["expected_value"] == 19900
    assert rows["next_lower_fee"]["advantage"] == 50
    assert rows["next_lower_fee"]["fee_saving"] == 150
    assert rows["next_lower_fee"]["break_even_change"] == -0.015
    assert rows["plan"]["advantage"] == 650
    assert rows["next_lower_fee"]["model_identity"]["horizon"] == 1
    assert all(row["initial_value"] == 20000 and row["reserve_cash"] == 10000 for row in result["rows"])
    assert (plan, record, forecasts) == before
    json.dumps(result, allow_nan=False)


def test_sale_comparison_decimal_strict_ties_and_independent_different_head_samples():
    short = sale_prediction(1, -0.2, [
        -0.2, "-0.19999999999999999999999999", "-0.20000000000000000000000001",
    ])
    long = sale_prediction(5, -0.3, [-0.5, -0.4, -0.3, -0.1])
    result = build_sale_date_comparison([short, long], plan=sale_plan(), fee_record=weekend_tiers(),
                                        old_confirmation_date="2026-01-03", current_redemption_override=0.2)
    rows = sale_rows(result)
    assert rows["next_lower_fee"]["advantage"] == 0
    assert rows["next_lower_fee"]["better_probability"] == 1 / 3
    assert rows["plan"]["better_probability"] == 0.25
    assert rows["now"]["better_probability"] == 0
    assert rows["next_lower_fee"]["comparison_candidate_id"] == rows["plan"]["comparison_candidate_id"] == "now"
    assert not {"best_date_probability", "joint_samples", "candidate_probabilities"} & result.keys()
    short["execution_forecasts"][0].pop("samples")
    missing = build_sale_date_comparison([short, long], plan=sale_plan(), fee_record=weekend_tiers(),
                                         old_confirmation_date="2026-01-03", current_redemption_override=0.2)
    assert sale_rows(missing)["next_lower_fee"]["better_probability"] is None
    assert sale_rows(missing)["plan"]["better_probability"] == 0.25


def test_sale_missing_exact_head_preserves_fee_boundary_never_borrows_other_h_or_l():
    forecast = sale_prediction(20, 0.99)
    forecast["execution_forecasts"].append(head(0.99, horizon=1, lag=2))
    forecast["unavailable_forecasts"] = [
        {"horizon": 1, "entry_lag": 1, "reason": "not-qualified", "status": "unavailable"}]
    result = build_sale_date_comparison(forecast, plan=sale_plan(), fee_record=weekend_tiers(),
                                        old_confirmation_date="2026-01-03")
    row = sale_rows(result)["next_lower_fee"]
    assert row["primary_head"] is None
    assert row["expected_value"] is row["advantage"] is row["better_probability"] is None
    assert row["fee_saving"] == 150 and row["break_even_change"] == -0.015
    assert row["status"] == "fee_only"
    assert "not-qualified" in row["reasons"]
    unknown = build_sale_date_comparison(plan=sale_plan(), old_confirmation_date="2026-01-03")
    assert all(row["fee_status"] == "unknown" and row["fee_known"] is False for row in unknown["rows"])
    assert all(row["fee_saving"] is row["break_even_change"] is None for row in unknown["rows"])
    assert all("示例" in row["fee_label"] for row in unknown["rows"])


def test_sale_manual_current_only_a_future_only_s_without_invented_manual_tiers():
    options = {"plan": sale_plan(), "fee_record": weekend_tiers(), "old_confirmation_date": "2026-01-03",
               "current_redemption_override": 0.02, "future_old_redemption_override": 0.007}
    result = build_sale_date_comparison([sale_prediction(1), sale_prediction(5)], **options)
    rows = sale_rows(result)
    assert [row["redemption_fee"] for row in result["rows"]] == [0.02, 0, 0.007]
    assert rows["now"]["fee_status"] == rows["plan"]["fee_status"] == "override"
    assert rows["next_lower_fee"]["fee_status"] == "public_standard"
    assert result["next_lower_fee"]["current_rate"] == 0.015  # Public schedule, not the A override.
    zero_current = build_sale_date_comparison(**{**options, "current_redemption_override": 0})
    assert sale_rows(zero_current)["next_lower_fee"]["fee_saving"] == 0
    assert zero_current["next_lower_fee"]["date"] == "2026-01-10"
    only_manual = build_sale_date_comparison(plan=sale_plan(), current_redemption_override=0.02,
                                             future_old_redemption_override=0.01)
    assert only_manual["next_lower_fee"]["status"] == "unavailable"
    assert [row["candidate_id"] for row in only_manual["rows"]] == ["now", "plan"]


def test_sale_deduplicates_d_equal_s_and_now_equal_s_without_leaking_overrides():
    result = build_sale_date_comparison(plan=sale_plan(3), fee_record=weekend_tiers(),
                                        old_confirmation_date="2026-01-03", future_old_redemption_override=0.03)
    assert len(result["rows"]) == 2
    future = result["rows"][1]
    assert future["candidate_ids"] == ["next_lower_fee", "plan"]
    assert future["redemption_fee"] == 0.03
    assert future["break_even_change"] == pytest.approx(0.985 / 0.97 - 1)
    same = build_sale_date_comparison(plan=sale_plan(2), fee_record=weekend_tiers(),
                                      old_confirmation_date="2026-01-03", current_redemption_override=0.01,
                                      future_old_redemption_override=0.03)
    assert len(same["rows"]) == 1
    assert same["rows"][0]["candidate_ids"] == ["now", "plan"]
    assert same["rows"][0]["redemption_fee"] == 0.01
    assert same["notes"]
    assert same["next_lower_fee"]["within_sale_window"] is False


def test_sale_tier_outside_window_no_lower_zero_fee_and_fee_100_boundary():
    result = build_sale_date_comparison(plan=sale_plan(), fee_record=public_tiers(),
                                        old_confirmation_date="2025-12-22")
    assert result["next_lower_fee"]["date"] == "2026-01-21"
    assert result["next_lower_fee"]["within_sale_window"] is False
    assert [row["candidate_id"] for row in result["rows"]] == ["now", "plan"]
    zero = build_sale_date_comparison(plan=sale_plan(), fee_record=public_tiers(),
                                      old_confirmation_date="2025-11-01")
    assert zero["next_lower_fee"]["status"] == "no_lower_tier"
    assert sale_rows(zero)["plan"]["fee_saving"] == sale_rows(zero)["plan"]["break_even_change"] == 0
    full = build_sale_date_comparison([sale_prediction(5)], plan=sale_plan(),
                                      current_redemption_override=1, future_old_redemption_override=1)
    row = sale_rows(full)["plan"]
    assert row["break_even_change"] is None and row["break_even_status"] == "no_finite_boundary"
    assert row["advantage"] == row["better_probability"] == 0
    assert row["expected_value"] == 10000


def test_sale_rejects_duplicate_exact_predictions_and_malformed_boolean_horizon():
    forecast = sale_prediction(1)
    options = {"plan": sale_plan(), "fee_record": weekend_tiers(), "old_confirmation_date": "2026-01-03"}
    with pytest.raises(ValidationError, match="重复"):
        build_sale_date_comparison([forecast, deepcopy(forecast)], **options)
    forecast["execution_forecasts"][0]["horizon"] = True
    result = build_sale_date_comparison(forecast, **options)
    row = sale_rows(result)["next_lower_fee"]
    assert row["primary_head"] is row["better_probability"] is None
    assert row["fee_saving"] == 150

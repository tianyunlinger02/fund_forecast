"""Reusable synthetic checks for fee decisions and the continuous cash ledger."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from rmbfund.domain import ValidationError
from rmbfund.forecast_research import (
    ledger_attribution,
    paired_delta_bootstrap,
    paired_loss_bootstrap,
    replay_actions,
)
from rmbfund.timing_scenario import fee_adjusted_actions


def _series(navs, start="2025-01-01"):
    first = date.fromisoformat(start)
    return [{"date": (first + timedelta(days=i)).isoformat(), "nav": nav}
            for i, nav in enumerate(navs)]


def _fees(**changes):
    return {"subscription_fee": 0.0015, "current_redemption_fee": 0.005,
            "future_old_redemption_fee": 0.005, "future_new_redemption_fee": 0.005,
            **changes}


def test_net_payoff_rule_uses_strict_shared_decimal_arithmetic_and_constant_fee_cancellation():
    # The old rule buys at 0.6%; new-buy liquidation costs also matter.
    assert 0.006 > _fees()["subscription_fee"]
    result = fee_adjusted_actions(0.006, **_fees())
    assert result["flat"] == "wait" and result["held"] == "hold"
    assert result["buy_advantage_vs_cash"] < 0
    for fee in (0.0, 0.005, 0.015, 1.0):
        result = fee_adjusted_actions(0, **_fees(
            current_redemption_fee=fee, future_old_redemption_fee=fee))
        assert result["held"] == "hold"
        assert result["hold_advantage_vs_redeem"] == 0

    flat_fees = _fees(subscription_fee=0, future_new_redemption_fee=0.2)
    assert fee_adjusted_actions(0.25, **flat_fees)["flat"] == "wait"
    assert fee_adjusted_actions(
        Decimal("0.250000000000000000000000000001"), **flat_fees)["flat"] == "buy"
    held_fees = _fees(current_redemption_fee=0.1, future_old_redemption_fee=0.2)
    assert fee_adjusted_actions(0.125, **held_fees)["held"] == "hold"
    assert fee_adjusted_actions(
        Decimal("0.124999999999999999999999999999"), **held_fees)["held"] == "redeem"


def test_fee_rule_missing_inputs_are_not_zero_fees_and_invalid_values_are_rejected():
    result = fee_adjusted_actions(0.1, **_fees(subscription_fee=None))
    assert result["flat"] == "unavailable" and result["held"] == "hold"
    assert result["buy_advantage_vs_cash"] is None
    result = fee_adjusted_actions(-0.1, **_fees(current_redemption_fee=None))
    assert result["flat"] == "wait" and result["held"] == "unavailable"
    result = fee_adjusted_actions(None, **_fees())
    assert result["flat"] == result["held"] == "unavailable"
    assert fee_adjusted_actions(-1, **_fees())["held"] == "redeem"
    for invalid in (True, float("nan"), float("inf"), -1.01,
                    Decimal("-1.000000000000000000001"), "not-a-number"):
        with pytest.raises(ValidationError):
            fee_adjusted_actions(invalid, **_fees())
    for invalid in (True, float("nan"), float("inf"), -0.001, 1.001,
                    Decimal("1.000000000000000000001")):
        with pytest.raises(ValidationError):
            fee_adjusted_actions(0, **_fees(future_new_redemption_fee=invalid))


def test_callback_execution_confirmation_settlement_trace_and_same_signal_zero_fee_replay():
    series = _series([1.0] * 13)
    signals = ("buy", "buy", "redeem", "redeem", "cash", "redeem",
               "buy", "buy", "buy", "hold", "cash")
    items = [{"origin_index": i, "prediction": {"signal": signal, "expected_change": 0.01}}
             for i, signal in enumerate(signals)]
    seen = []

    def policy(item, state):
        assert set(state) == {"can_buy", "can_redeem", "has_units", "order_pending",
                              "unconfirmed_units", "cash_in_transit"}
        assert all(type(value) is bool for value in state.values())
        seen.append((item, state))
        return item["prediction"]["signal"]

    options = {"entry_fee_rate": 0.0015, "redemption_fee_rate": 0.005,
               "entry_lag": 3, "redemption_lag": 3, "confirmation_lag": 1, "cash_lag": 3}
    replay = replay_actions(series, items, **options, decision_policy=policy)
    path = replay["strategy"]
    trace = path["action_trace"]
    assert len(seen) == len(items)  # The passive reference never invokes policy.
    assert [row["action"] for row in trace] == list(signals)
    buy, sell, reentry = path["trades"]
    assert (buy["origin_index"], buy["decision_index"], buy["execution_index"],
            buy["confirmation_index"]) == (0, 1, 3, 4)
    assert (sell["origin_index"], sell["decision_index"], sell["execution_index"],
            sell["settlement_index"]) == (3, 4, 6, 9)
    assert buy["confirmation_date"] == series[4]["date"]
    assert sell["settlement_date"] == series[9]["date"]
    assert buy["confirmation_status"] == "confirmed"
    assert sell["settlement_status"] == "settled"
    assert trace[1]["reason"] == "order_pending"
    assert trace[2]["reason"] == "no_confirmed_units"
    assert trace[2]["state"]["has_units"] is True
    assert trace[2]["state"]["unconfirmed_units"] is True
    assert trace[3]["state"]["can_redeem"] is True
    assert trace[4]["status"] == trace[9]["status"] == trace[10]["status"] == "noop"
    assert path["daily"][5]["units"] > 0  # cash is not a sale.
    for origin in (6, 7):
        assert trace[origin]["state"]["cash_in_transit"] is True
        assert trace[origin]["state"]["can_buy"] is False
        assert trace[origin]["reason"] == "not_spendable_cash"
    assert trace[8]["state"]["can_buy"] is True
    assert reentry["decision_index"] == 9 and reentry["execution_index"] == 11
    assert all(path["daily"][index]["cash_available"] == 0 for index in range(1, 9))
    assert path["daily"][8]["cash_in_transit"] == pytest.approx(0.995 / 1.0015)

    zero = replay_actions(series, trace, **{**options, "entry_fee_rate": 0,
                                          "redemption_fee_rate": 0})
    assert len(seen) == len(items)
    assert [(row["action"], row["origin_index"], row["execution_index"])
            for row in zero["strategy"]["trades"]] == [
                (row["action"], row["origin_index"], row["execution_index"])
                for row in path["trades"]]
    assert zero["strategy"]["terminal_liquidated_wealth"] == 1.0
    assert "action_trace" not in replay["buy_and_hold"]


def test_callback_sees_known_prediction_and_boolean_state_not_current_execution_nav():
    series = _series([1.0] * 10)
    items = [{"origin_index": i, "prediction": {"expected_change": 0.01}} for i in range(8)]
    untouched = deepcopy(items)
    calls = []

    def policy(item, state):
        assert item == untouched[item["origin_index"]]
        calls.append((deepcopy(item), dict(state)))
        item["prediction"]["expected_change"] = 999  # Never mutate input or saved trace.
        if item["origin_index"] == 0:
            return "buy"
        return "redeem" if state["can_redeem"] else "hold"

    options = {"entry_fee_rate": 0.0015, "redemption_fee_rate": 0.005,
               "entry_lag": 3, "redemption_lag": 3, "decision_policy": policy}
    replay_actions(series, items, **options)
    original_calls = calls[:]
    calls.clear()
    changed = deepcopy(series)
    for row in changed[3:]:
        row["nav"] = 20.0
    replay_actions(changed, items, **options)
    assert calls == original_calls  # Due execution changes units, not policy information.
    assert items == untouched
    with pytest.raises(ValidationError, match="decision_policy action"):
        replay_actions(series, items[:1], **{**options, "decision_policy": lambda item, state: "exit"})


def test_constant_nav_external_subscription_fee_receivable_and_terminal_fee_are_counted_once():
    series = _series([1.0] * 10)
    options = {"entry_fee_rate": 0.0015, "redemption_fee_rate": 0.005,
               "entry_lag": 3, "redemption_lag": 3}
    actions = [{"origin_index": 0, "action": "buy"}, {"origin_index": 3, "action": "redeem"}]
    full = replay_actions(series, actions, **options)["strategy"]
    transit = replay_actions(series, actions, **options, end_index=7)["strategy"]
    held = replay_actions(series, actions[:1], **options)["strategy"]
    expected = (1 - 0.005) / (1 + 0.0015)
    for path in (full, transit, held):
        assert path["terminal_liquidated_wealth"] == pytest.approx(expected)
        attribution = ledger_attribution(path, series)
        assert attribution["market_pnl"] == 0
        assert attribution["identity_residual"] == pytest.approx(0, abs=1e-15)
    buy = full["trades"][0]
    assert buy["fee"] == pytest.approx(0.0015 / 1.0015)
    assert buy["units"] == pytest.approx(1 / 1.0015)
    assert full["terminal_liquidation_fee"] == transit["terminal_liquidation_fee"] == 0
    assert transit["daily"][-1]["cash_available"] == 0
    assert transit["daily"][-1]["cash_in_transit"] == pytest.approx(expected)
    assert held["terminal_liquidation_fee"] == pytest.approx(0.005 / 1.0015)


def test_explicit_default_replay_keeps_old_exact_structure_values_and_cash_noop():
    series = _series([10.0] * 3)
    actions = [{"origin_index": 0, "action": "buy"},
               {"origin_index": 0, "action": "hold"},
               {"origin_index": 1, "action": "cash"},
               {"origin_index": 1, "action": "redeem"}]
    replay = replay_actions(series, actions, entry_fee_rate=0, redemption_fee_rate=0)
    expected_daily = []
    for index, (cash, reserved, units) in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                    (0.0, 0.0, 0.1))):
        expected_daily.append({"index": index, "date": series[index]["date"],
                               "cash_available": cash, "cash_reserved": reserved,
                               "cash_in_transit": 0.0, "units": units, "confirmed_units": 0.0,
                               "market_value": units * 10, "marked_wealth": 1.0, "fees_paid": 0.0})
    expected = {
        "terminal_liquidated_wealth": 1.0, "return": 0.0, "terminal_liquidation_fee": 0.0,
        "fees_paid_before_terminal": 0.0,
        "trades": [{"action": "buy", "origin_index": 0, "execution_index": 2,
                    "execution_date": series[2]["date"], "fee": 0.0, "gross_notional": 1.0,
                    "units": 0.1, "execution_nav": 10.0}],
        "skipped_actions": [{"origin_index": 1, "action": "redeem",
                             "reason": "execution_after_replay_end"}],
        "daily": expected_daily,
    }
    assert replay["strategy"] == expected
    assert replay["buy_and_hold"] == {**expected, "skipped_actions": []}
    assert replay == replay_actions(series, actions, entry_fee_rate=0, redemption_fee_rate=0,
                                    decision_policy=None)
    selected = replay_actions(series, actions, entry_fee_rate=0, redemption_fee_rate=0,
                              decision_policy=lambda item, state: item["action"])["strategy"]
    old_trade_keys = set(expected["trades"][0])
    assert {**{key: value for key, value in selected.items() if key != "action_trace"},
            "trades": [{key: row[key] for key in old_trade_keys} for row in selected["trades"]]} == expected
    assert selected["action_trace"][-1]["execution_index"] is None
    assert selected["action_trace"][-1]["reason"] == "execution_after_replay_end"

    same_index = replay_actions(
        series, [{"origin_index": 0, "action": kind} for kind in ("buy", "redeem", "cash")],
        entry_fee_rate=0, redemption_fee_rate=0, entry_lag=1, redemption_lag=1,
        confirmation_lag=0, cash_lag=0, decision_policy=lambda item, state: item["action"],
    )["strategy"]
    assert [row["execution_index"] for row in same_index["trades"]] == [1, 1]
    assert same_index["action_trace"][1]["state"]["can_redeem"] is True
    assert same_index["action_trace"][2]["state"]["can_buy"] is True
    assert same_index["daily"][1]["cash_available"] == 1.0
    assert same_index["daily"][1]["units"] == 0


@pytest.mark.parametrize("sell", [False, True])
def test_continuous_year_contributions_include_unconfirmed_exposure_not_reserved_cash(sell):
    series = _series([7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], start="2024-12-29")
    actions = [{"origin_index": 0, "action": "buy"}]
    if sell:
        actions.append({"origin_index": 4, "action": "redeem"})
    path = replay_actions(series, actions, entry_fee_rate=0.0015, redemption_fee_rate=0.005,
                          confirmation_lag=3)["strategy"]
    result = ledger_attribution(path, series)
    prior, following = result["years"]["2024"], result["years"]["2025"]
    assert prior["market_pnl"] == 0
    assert prior["terminal_liquidation_fee"] == 0
    assert following["opening_wealth"] == prior["closing_marked_wealth"]
    assert following["opening_economic_units"] == prior["closing_economic_units"] > 0
    assert path["daily"][3]["confirmed_units"] == path["daily"][4]["confirmed_units"] == 0
    assert result["daily"][3]["market_pnl"] > 0
    assert result["daily"][4]["market_pnl"] > 0
    assert result["daily"][2]["market_pnl"] == 0  # Entry cash had not been priced.
    assert result["market_pnl"] == pytest.approx((4 if sell else 5) / 10 / 1.0015)
    assert result["wealth_change"] == pytest.approx(
        result["market_pnl"] - result["executed_fees"] - result["terminal_liquidation_fee"])
    assert result["identity_residual"] == pytest.approx(0, abs=1e-15)
    assert sum(row["net_contribution"] for row in result["years"].values()) == pytest.approx(
        path["terminal_liquidated_wealth"] - 1)
    assert sum(row["terminal_liquidation_fee"] for row in result["half_years"].values()) == (
        path["terminal_liquidation_fee"])
    assert (path["daily"][-1]["cash_in_transit"] > 0) == sell


def test_generic_paired_delta_bootstrap_reuses_loss_blocks_and_preserves_gaps():
    head = {"up_probability": 0.6, "down_probability": 0.4, "flat_probability": 0.0,
            "expected_change": 0.01, "lower": -0.1, "upper": 0.1}
    rows = [{"origin_index": i, "origin_date": _series([1.0] * 24)[i]["date"],
             "entry_index": i + 1, "target_index": i + 3, "horizon": 2, "entry_lag": 1,
             "actual_change": 0.05, "prediction": head, "historical_baseline": dict(head)}
            for i in range(24)]
    legacy = paired_loss_bootstrap(rows, replicates=50)
    generic = paired_delta_bootstrap(rows, {name: [0.0] * len(rows)
                                           for name in legacy["mean_deltas"]},
                                     replicates=50, seed=20261029)
    assert legacy == {**generic, "delta_direction": legacy["delta_direction"],
                      "interpretation": legacy["interpretation"]}
    gapped = rows[:10] + rows[12:]
    values = np.arange(len(gapped), dtype=float) / 100
    result = paired_delta_bootstrap(gapped, {"net_payoff_delta": values}, replicates=50)
    assert result["seed"] == 20261030
    assert result["contiguous_run_lengths"] == [10, 12]
    assert [block["block_length"] for block in result["blocks"]] == [3, 6]
    starts = np.asarray([*range(8), *range(10, 20)])
    full, remainder = divmod(len(gapped), 3)
    draws = np.random.default_rng(20261030).choice(starts, size=(50, full + bool(remainder)))
    manual = np.asarray([(sum(values[start:start + 3].sum() for start in draw[:full])
                          + values[draw[-1]:draw[-1] + remainder].sum()) / len(gapped)
                         for draw in draws])
    interval = result["blocks"][0]["intervals"]["net_payoff_delta"]
    assert interval["mean_delta"] == pytest.approx(values.mean())
    assert [interval["lower_95"], interval["upper_95"]] == pytest.approx(np.quantile(manual, [0.025, 0.975]))
    assert paired_delta_bootstrap([], {}) == {"count": 0, "blocks": [], "skill_certified": False}
    for invalid in ([0], [float("nan")] * len(rows)):
        with pytest.raises(ValidationError, match="finite aligned"):
            paired_delta_bootstrap(rows, {"invalid": invalid})
    short_run = paired_delta_bootstrap(rows[:2] + rows[12:], {"payoff": [0.0] * 14})
    assert all(block["status"] == "unavailable" for block in short_run["blocks"])

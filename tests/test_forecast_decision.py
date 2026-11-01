"""Action comparisons use forecasted value and costs, not holding-age gates."""

import pytest


from rmbfund.forecast_decision import build_decision


def prediction(*changes):
    return {
        "anchor": {"date": "2026-10-20", "nav": "2.107"},
        "entry_assumptions": {"baseline_entry_lag": 2},
        "execution_forecasts": [
            {
                "horizon": 20,
                "entry_lag": 2 + i,
                "expected_change": change,
                "up_probability": 0.6,
                "lower": -0.08,
                "upper": 0.09,
            }
            for i, change in enumerate(changes or (0.02,))
        ],
    }


def test_external_subscription_fee_formula_and_break_even():
    value = prediction(0.02)
    result = build_decision(value, subscription_fee="0.01")
    assert result["branches"]["flat"]["action"] == "buy"
    assert result["scenarios"][0]["buy_net_change"] == pytest.approx(1.02 / 1.01 - 1)
    assert result["break_even"]["subscription_fee"] == 0.02
    assert build_decision(value, subscription_fee="0.02")["branches"]["flat"]["action"] == "wait"
    assert build_decision(value, subscription_fee="0.03")["branches"]["flat"]["action"] == "wait"


def test_held_sunk_entry_cost_does_not_get_charged_again():
    first = build_decision(prediction(-0.02), subscription_fee="0", redemption_fee="0.005")
    second = build_decision(prediction(-0.02), subscription_fee="0.8", redemption_fee="0.005")
    assert first["branches"]["held"] == second["branches"]["held"]
    assert first["branches"]["held"]["action"] == "sell"
    assert first["scenarios"][0]["hold_advantage"] == pytest.approx(-0.015)
    assert (
        build_decision(prediction(-0.02), redemption_fee="0.03")["branches"]["held"]["action"]
        == "hold"
    )


def test_unknown_fees_are_not_zero_and_do_not_erase_market_forecast():
    positive = build_decision(prediction(0.02))
    assert positive["branches"]["flat"]["action"] == "conditional_buy"
    assert positive["branches"]["flat"]["expected_change"] is None
    assert positive["branches"]["held"]["action"] == "hold"
    negative = build_decision(prediction(-0.02))
    assert negative["branches"]["flat"]["action"] == "wait"
    assert negative["branches"]["held"]["action"] == "conditional_sell"
    assert negative["break_even"]["redemption_fee"] == 0.02
    zero_known = build_decision(prediction(-0.02), redemption_fee="0")
    assert zero_known["branches"]["held"]["action"] == "sell"


def test_delays_can_reverse_action_and_are_reported_not_vetoed():
    result = build_decision(
        prediction(0.02, -0.01, -0.03), subscription_fee="0", redemption_fee="0"
    )
    assert [item["flat_action"] for item in result["scenarios"]] == ["buy", "wait", "wait"]
    assert [item["held_action"] for item in result["scenarios"]] == ["hold", "sell", "sell"]
    assert "改变动作比较" in " ".join(result["branches"]["held"]["reasons"])


@pytest.mark.parametrize(
    "horizon,change,probability", [(1, -0.03, 0.2), (5, 0.05, 0.8), (20, 0.02, 0.6)]
)
def test_each_horizon_uses_its_own_execution_head_and_samples(horizon, change, probability):
    value = prediction(0.02)
    value["execution_forecasts"][0]["samples"] = [-0.1, 0.02, 0.04]
    value["execution_forecasts"] += [
        {
            "horizon": 1,
            "entry_lag": 2,
            "expected_change": -0.03,
            "up_probability": 0.2,
            "samples": [-0.1, -0.04, -0.02, -0.01, 0.02],
        },
        {
            "horizon": 5,
            "entry_lag": 2,
            "expected_change": 0.05,
            "up_probability": 0.8,
            "samples": [-0.1, 0.02, 0.04, 0.06, 0.2],
        },
    ]
    result = build_decision(value, horizon=horizon, subscription_fee="0.01", redemption_fee="0.01")
    primary = result["primary"]
    assert result["primary_horizon"] == horizon
    assert result["primary_head"]["horizon"] == horizon
    assert primary["up_probability"] == probability
    assert primary["market_expected_change"] == change
    assert primary["buy_advantage_vs_cash"] == pytest.approx((1 + change) / 1.01 - 1)
    assert primary["hold_advantage_vs_exit"] == pytest.approx(change + 0.01)
    assert all(row["horizon"] == horizon for row in result["scenarios"])
    expected_samples = result["primary_head"]["samples"]
    assert primary["buy_profit_probability"] == sum(x > 0.01 for x in expected_samples) / len(
        expected_samples
    )
    assert f"之后{horizon}笔净值" in " ".join(result["branches"]["held"]["reasons"])
    assert result["branches"]["held"]["hold_advantage_vs_exit"] == pytest.approx(change + 0.01)


def test_shorter_missing_execution_horizon_cannot_borrow_h20_or_anchor():
    value = prediction(0.02)
    value["forecasts"] = [{"horizon": 5, "entry_lag": 0, "expected_change": 0.9}]
    result = build_decision(value, horizon=5, redemption_fee="0")
    assert result["status"] == "missing_primary_head"
    assert result["scenarios"] == []
    assert result["branches"]["held"]["action"] == "unavailable"

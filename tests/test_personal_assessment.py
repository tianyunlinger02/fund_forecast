"""Local-only scenarios use exact forecast heads, original probabilities and Decimal."""

from __future__ import annotations


from copy import deepcopy
from decimal import Decimal
from math import log1p

import pytest


from rmbfund.personal_assessment import build_assessment, explain_forecast_head, format_factor_value


from rmbfund.portfolio import value_holding


SUBJECT = {
    "id": "synthetic-private-test",
    "symbol": "SYNTH",
    "name": "合成基金C",
    "kind": "open_fund",
    "share_class": "C",
    "exchange": "",
    "is_demo": False,
}


ISSUED = "2026-10-22T15:45:00+08:00"


def prediction():
    def head(horizon, lag, change):
        return {
            "horizon": horizon,
            "entry_lag": lag,
            "expected_change": change,
            "lower": -0.2,
            "upper": 0.3,
            "up_probability": 0.6,
            "down_probability": 0.2,
            "flat_probability": 0.2,
            "interval_coverage": 0.8,
            "samples": [-0.2, 0, 0.03, 0.05, 0.3],
            "sample_count": 5,
            "method": "ridge_prequential_empirical",
        }

    return {
        "record_id": "original-public-record",
        "product_id": SUBJECT["id"],
        "model_id": "original-model",
        "model_hash": "original-hash",
        "schema_version": 2,
        "input_hash": "original-input",
        "basis_hash": "original-basis",
        "generated_at": ISSUED,
        "entry_assumptions": {"baseline_entry_lag": 2, "entry_lags": [2, 3]},
        "anchor": {"date": "2026-10-20", "nav": "2.00", "currency": "CNY"},
        "forecasts": [head(1, 0, -0.01), head(5, 0, 0.03), head(20, 0, 0.1)],
        "execution_forecasts": [
            head(1, 2, -0.04),
            head(5, 2, 0.06),
            head(20, 2, 0.02),
            head(1, 3, 0.04),
            head(5, 3, -0.04),
            head(20, 3, -0.03),
        ],
        "drivers": [
            {
                "horizon": horizon,
                "entry_lag": lag,
                "feature": f"log_return_{horizon}",
                "value": 0.01,
                "contribution_log_return": -0.02 if lag == 2 else 0.03,
            }
            for horizon in (1, 5, 20)
            for lag in (0, 2, 3)
        ],
    }


def holding(shares="10.00", cost="18.00", currency="CNY"):
    return {
        "id": "private-holding-id",
        "product_id": SUBJECT["id"],
        "subject": {key: value for key, value in SUBJECT.items() if key != "id"},
        "shares": shares,
        "cost": cost,
        "currency": currency,
    }


def assess(value=None, **kwargs):
    return build_assessment(prediction() if value is None else value, subject=SUBJECT, **kwargs)


def test_anchor_values_use_own_nav_not_today_or_execution_change():
    fund = holding()
    today = value_holding(fund, {"date": "2026-10-24", "value": "3.00", "currency": "CNY"})
    result = assess(holding=fund, current_value=today, subscription_fee="0.01")
    money = result["amounts"]["anchor"]
    assert money["basis_date"] == "2026-10-20"
    assert Decimal(money["basis_nav"]) == 2
    assert Decimal(money["basis_value"]) == 20
    assert {key: Decimal(value) for key, value in money["expected"].items()} == {
        "value": Decimal("22"),
        "change": Decimal("2"),
        "profit": Decimal("4"),
    }
    assert {key: Decimal(value) for key, value in money["lower"].items()} == {
        "value": Decimal("16"),
        "change": Decimal("-4"),
        "profit": Decimal("-2"),
    }
    assert {key: Decimal(value) for key, value in money["upper"].items()} == {
        "value": Decimal("26"),
        "change": Decimal("6"),
        "profit": Decimal("8"),
    }
    assert result["amounts"]["current"] == today
    assert Decimal(today["market_value"]) == 30
    assert result["amounts"]["execution_holding"]["status"] == "unknown_entry_nav"
    assert "value" not in result["amounts"]["execution_holding"]
    assert "未扣除未来赎回费" in " ".join(money["reasons"])


def test_new_buy_uses_total_budget_external_fee_and_exact_execution_scenarios():
    result = assess(holding=holding(), budget="101.00", subscription_fee="0.01")
    money = result["amounts"]["new_buy"]
    assert money["status"] == "ready"
    assert money["basis_value"] == "101.00"
    assert money["basis_nav"] is money["basis_date"] is None
    assert {key: Decimal(value) for key, value in money["expected"].items()} == {
        "value": Decimal("102"),
        "change": Decimal("1"),
        "profit": Decimal("1"),
    }
    assert Decimal(money["lower"]["value"]) == 80
    assert Decimal(money["upper"]["value"]) == 130
    assert result["decision"]["branches"]["flat"]["buy_profit_probability"] == 0.6
    assert Decimal(result["amounts"]["anchor"]["expected"]["value"]) == 22


def test_unknown_fee_no_budget_and_explicit_zero_have_distinct_money_states():
    absent = assess(holding=holding(), subscription_fee="0")
    assert absent["amounts"]["new_buy"]["status"] == "missing_budget"
    assert absent["inputs"]["budget"] is None
    unknown = assess(holding=holding(), budget="100")
    assert unknown["decision"]["branches"]["flat"]["action"] == "conditional_buy"
    assert unknown["decision"]["branches"]["flat"]["buy_profit_probability"] is None
    assert unknown["amounts"]["new_buy"]["status"] == "unknown_subscription_fee"
    assert unknown["amounts"]["new_buy"]["expected"] is None
    blank = assess(holding=holding(), subscription_fee="", redemption_fee=" ")
    assert blank["inputs"]["subscription_fee"] is blank["inputs"]["redemption_fee"] is None
    explicit_zero = assess(holding=holding(), budget="0.00", subscription_fee="0")
    assert explicit_zero["amounts"]["new_buy"]["status"] == "ready"
    assert all(
        Decimal(value) == 0 for value in explicit_zero["amounts"]["new_buy"]["expected"].values()
    )


@pytest.mark.parametrize("confirmation,ready", [("2026-10-21", False), ("2026-10-30", False),
                                                 ("2026-10-26", True)])
def test_calendar_personal_amount_requires_feasible_buy_action(confirmation, ready):
    from rmbfund.timing_scenario import build_calendar_scenario, resolve_plan_fees
    from rmbfund.trade_calendar import adjust_new_confirmation_date, build_trade_plan

    value = prediction()
    plan = build_trade_plan(decision_at="2026-10-22T09:00:00+08:00", anchor_date="2026-10-20",
                            calendar_days=7)
    plan = adjust_new_confirmation_date(plan, confirmation)
    fees = resolve_plan_fees(None, plan=plan, old_confirmation_date="2026-10-01",
                             subscription_override=0, current_redemption_override=0,
                             future_old_redemption_override=0, future_new_redemption_override=0)
    scenario = build_calendar_scenario(value, subject=SUBJECT, plan=plan, fee_basis=fees)
    result = build_assessment(value, subject=SUBJECT, horizon=5,
                              budget={"amount": "100", "currency": "CNY"},
                              subscription_fee=0, redemption_fee=0, trade_scenario=scenario)
    buy = next(row for row in scenario["rows"] if row["action"] == "buy")
    money = result["amounts"]["new_buy"]
    assert money["end_value_basis"] == "cash_and_redemption_receivable"
    if ready:
        assert buy["status"] == money["status"] == "ready"
        assert Decimal(money["expected"]["value"]) == 106
    else:
        assert buy["status"] == "unavailable" and money["status"] == "unavailable_trade_plan"
        assert money["expected"] is money["lower"] is money["upper"] is None
        assert set(buy["reasons"]) <= set(money["reasons"])
    assert result["trade_scenario"] == scenario


def test_investment_conditions_are_saved_separately_without_changing_market_forecast():
    value = prediction()
    original = deepcopy(value)
    analysis = {
        "profile": {"product_id": SUBJECT["id"], "as_of": ISSUED},
        "scenario": {"covered_contribution": "-0.08", "whole_fund_return": None},
        "prediction_relationship": "separate_from_nav_forecast",
        "issuer_fundamentals": {
            "analysis": {"product_id": SUBJECT["id"], "issuers": [
                {"issuer_id": "synthetic-company", "ttm_eps": "4", "price": "120"},
            ]},
            "scenario": {"product_id": SUBJECT["id"], "terminal_multiple": "25",
                         "required_eps": "4.8", "required_growth": "0.2"},
        },
    }
    result = assess(value, investment_analysis=analysis)
    without = assess(value)
    assert result["decision"] == without["decision"]
    assert result["execution"] == without["execution"]
    assert result["forecast"] == without["forecast"]
    assert value == original
    assert result["investment_analysis"] == analysis
    analysis["scenario"]["covered_contribution"] = "-0.99"
    assert result["investment_analysis"]["scenario"]["covered_contribution"] == "-0.08"
    analysis["issuer_fundamentals"]["analysis"]["issuers"][0]["ttm_eps"] = "99"
    analysis["issuer_fundamentals"]["scenario"]["required_growth"] = "-0.9"
    saved = result["investment_analysis"]["issuer_fundamentals"]
    assert saved["analysis"]["issuers"][0]["ttm_eps"] == "4"
    assert saved["scenario"]["required_growth"] == "0.2"


def test_saved_factor_units_are_not_all_treated_as_fund_return_percentages():
    cases = [
        ("log_return", log1p(0.1), "+10.00%"),
        ("log_change", log1p(-0.02), "-2.00%"),
        ("percent", 4, "4.00%"),
        ("percentage_points", 0.1, "+0.10pp（百分点）"),
        ("log_return_difference", 0.03, "+0.0300（相对强弱的对数差，正数更强；不是基金收益）"),
    ]
    for unit, value, expected in cases:
        assert format_factor_value({"feature": "not_a_nav_whitelist_name", "unit": unit}, value) == expected


def test_actual_factor_reasons_and_compact_inputs_survive_save_without_mutating_prediction():
    from rmbfund.investment_forecast import FEATURE_DEFINITIONS, MODEL_ID

    value = prediction()
    value["model_id"] = MODEL_ID
    value["model_definition"] = {"model_id": MODEL_ID, "features": deepcopy(list(FEATURE_DEFINITIONS))}
    value["features"] = [dict(row, value=4 if row["unit"] == "percent" else 0.1)
                         for row in FEATURE_DEFINITIONS]
    value["profile_mapping"] = {
        "product_id": SUBJECT["id"], "as_of": ISSUED,
        "portfolio": {"record_id": "saved-portfolio", "report_date": "2026-06-30", "payload": {
            "denominator": "nav", "assets": {"equity": "0.8"},
            "sectors": [{"name": "信息技术", "weight": "0.4"}],
        }},
        "mandate": {"record_id": "saved-mandate", "report_date": "2026-06-30", "payload": {
            "base_currency": "CNY", "benchmark": {"region": "emerging_markets"},
        }},
        "basis": "current_ex_post_mapping_not_historical_portfolio_weights",
    }
    value["factor_provenance"] = {"series_manifest": [{"series_id": "saved-market", "source": "synthetic"}]}
    value["factor_input_hash"] = "original-factor-hash"
    value["profile_mapping_hash"] = "original-mapping-hash"
    value["factor_inputs"] = {"schema_version": 1, "series": {"market": {"rows": [
        {"date": "2026-10-16", "value": 100, "available_at": ISSUED} for _ in range(100)
    ]}}}
    value["model_state"] = [{"horizon": h, "entry_lag": lag, "intercept": 0.01}
                            for h in (1, 5, 20) for lag in (0, 2, 3)]
    for head in [*value["forecasts"], *value["execution_forecasts"]]:
        head["drivers"] = []
        for index, definition in enumerate(FEATURE_DEFINITIONS):
            name = definition["name"]
            effect = 0.2 if name.startswith("eem_") else 0.01
            # The small opposing input must survive three stronger positives.
            if definition["unit"] == "percentage_points":
                effect = -0.0001
            head["drivers"].append({**definition, "feature": name,
                "horizon": head["horizon"], "entry_lag": head["entry_lag"],
                "value": value["features"][index]["value"], "contribution_log_return": effect,
                "observation_ends": {"synthetic-source": "2026-10-16"},
            })
    # A global driver from a different situation cannot replace the actual head.
    value["drivers"] = [{"feature": "WRONG_HEAD", "horizon": 5, "entry_lag": 3,
                         "value": 999, "contribution_log_return": 999}]
    original = deepcopy(value)
    result = assess(value)
    conditional = assess(value, investment_analysis={
        "profile": {"product_id": SUBJECT["id"], "as_of": "2026-10-24T12:00:00+08:00"},
        "scenario": {"nav_return_contribution": "-0.5"},
    })
    assert conditional["decision"] == result["decision"]
    assert conditional["execution"] == result["execution"]
    assert conditional["forecast"] == result["forecast"]
    assert value == original
    execution = result["execution"]
    original_head = next(row for row in value["execution_forecasts"]
                         if (row["horizon"], row["entry_lag"]) == (20, 2))
    assert execution["drivers"] == original_head["drivers"]
    assert len(execution["drivers"]) == 14
    assert len(execution["factor_reasons"]) == 3
    assert {row["direction"] for row in execution["factor_reasons"]} == {"支持上涨", "反对上涨"}
    reasons = " ".join(execution["reasons"])
    assert "新兴市场" in reasons and "+0.10pp" in reasons and "2026-06-30" in reasons
    assert "WRONG_HEAD" not in reasons
    assert execution["head"]["up_probability"] == original_head["up_probability"]
    assert execution["head"]["expected_change"] == original_head["expected_change"]
    saved = result["forecast"]
    assert saved["model_name"] == "投资因素预测（试验）"
    for key in ("model_definition", "features", "profile_mapping", "factor_provenance",
                "factor_input_hash", "profile_mapping_hash"):
        assert saved[key] == original[key]
    assert {(row["horizon"], row["entry_lag"]) for row in saved["model_state"]} == {(20, 0), (20, 2)}
    assert "factor_inputs" not in saved
    assert saved["factor_input_snapshot"]["factor_input_hash"] == "original-factor-hash"
    detail = explain_forecast_head(value, original_head, horizon=20, entry_lag=2, role="execution", all_factors=True)
    assert len(detail["factor_reasons"]) == 14
    assert any(row["formatted_value"] == "4.00%" for row in detail["factor_reasons"])
    saved_reasons = list(execution["reasons"])
    original_head["drivers"][0]["value"] = 999
    value["profile_mapping"]["portfolio"]["report_date"] = "2099-01-01"
    assert saved["profile_mapping"]["portfolio"]["report_date"] == "2026-06-30"
    assert result["execution"]["reasons"] == saved_reasons
    assert result["execution"]["drivers"][0]["value"] != 999


def test_head_local_empty_drivers_and_mismatched_identity_never_borrow_global_reasons():
    value = prediction()
    head = value["execution_forecasts"][2]
    head["drivers"] = []
    explanation = explain_forecast_head(value, head, horizon=20, entry_lag=2, role="execution")
    assert explanation["drivers"] == explanation["factor_reasons"] == []
    assert "没有保存" in " ".join(explanation["reasons"])
    assert explain_forecast_head(value, head, horizon=5, entry_lag=2, role="execution")["head"] is None
    head["drivers"] = [{"horizon": 20, "entry_lag": 2, "role": "anchor", "feature": "wrong-role",
                        "value": 1, "contribution_log_return": 1}]
    assert not explain_forecast_head(value, head, horizon=20, entry_lag=2, role="execution")["drivers"]


def test_model_catalog_and_source_gate_follow_actual_selected_method():
    from rmbfund.forecast_models import INVESTMENT_MODEL_ID, NAV_MODEL_ID
    from rmbfund.ui.forecast import _catalog, model_source_text

    catalog = _catalog([
        {"id": "nav_ridge_empirical_v1"}, {"id": NAV_MODEL_ID, "available": True},
        {"id": INVESTMENT_MODEL_ID, "available": False, "unavailable_reason": "missing-mapping"},
    ])
    assert [row["id"] for row in catalog] == [NAV_MODEL_ID, INVESTMENT_MODEL_ID]
    assert catalog[1]["available"] is False and catalog[1]["unavailable_reason"] == "missing-mapping"
    nav = model_source_text({"selected_model_id": INVESTMENT_MODEL_ID,
                             "prediction": {"model_id": NAV_MODEL_ID}})
    assert "只参考本基金已保存净值" in nav and "额外特征" in nav
    investment = model_source_text({"selected_model_id": INVESTMENT_MODEL_ID, "prediction": {
        "model_id": INVESTMENT_MODEL_ID, "features": [
            {"name": role, "economic_role": role} for role in ("market", "industry", "fx", "discount_rate")
        ],
    }})
    assert "实际拟合" in investment and "中间价" in investment and "十年期" in investment
    assert "盈利、估值和经理能力没有进入拟合" in investment
    assert "不能覆盖上方分布" in investment

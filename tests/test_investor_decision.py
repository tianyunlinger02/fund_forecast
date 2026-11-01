"""Core fund-buyer invariants using clearly synthetic local records."""
from copy import deepcopy

from rmbfund.personal_assessment import build_assessment
from rmbfund.portfolio import Portfolio
from rmbfund.position_sizing import conditional_size

SUBJECT = {"id": "open_fund:SYNTH:C", "symbol": "SYNTH", "name": "合成买卖测试基金", "kind": "open_fund", "share_class": "C", "exchange": "", "is_demo": False}
HOLDING = {"product_id": SUBJECT["id"], "shares": "100", "cost": "250", "currency": "CNY"}


def thesis(posture):
    return {"status": "available", "record_id": "synthetic-thesis-" + posture, "needed_review": [], "payload": {
        "product": SUBJECT, "posture": posture, "horizon": {"start": "2026-10-26", "end": "2027-10-26"},
        "facts": [{"id": "synthetic-fact"}], "citations": [{"id": "synthetic-citation"}],
        "assessment": {"summary": "合成论证，不是真实投资建议", "supporting": [{"reason": "合成支持依据", "fact_ids": ["synthetic-fact"]}],
                       "counterevidence": [{"reason": "合成反对依据", "fact_ids": ["synthetic-fact"]}],
                       "priced_in": {"status": "unknown", "reason": "未虚构便宜的价格"}, "missing": ["未知但不必阻断整个论证"],
                       "falsifiers": ["原盈利逻辑失效"], "change_conditions": ["复核实际披露"]}}}


def prediction():
    head = {"horizon": 20, "entry_lag": 2, "expected_change": 0.02, "up_probability": 0.6, "lower": -0.1, "upper": 0.15}
    return {"product_id": SUBJECT["id"], "record_id": "synthetic-forecast", "model_id": "nav_ridge_empirical_v2",
            "model_hash": "synthetic", "entry_assumptions": {"baseline_entry_lag": 2}, "execution_forecasts": [head]}


def test_legacy_posture_is_not_presented_as_three_independent_actions():
    original = prediction()
    untouched = deepcopy(original)
    results = {posture: build_assessment(original, subject=SUBJECT, holding=HOLDING,
               research_thesis=thesis(posture), subscription_fee="0", redemption_fee="0")
               for posture in ("accumulate", "maintain", "trim", "exit")}
    for row in results.values():
        decision = row["investor_decision"]
        assert decision["action_basis"] == "legacy_opinion"
        assert decision["branches"]["held"]["action"] == "review"
        assert decision["branches"]["held"]["reasons"] == []
        assert decision["branches"]["flat"]["action"] == "wait"
        assert decision["branches"]["addition"]["action"] == "wait"
        for key in ("decision", "execution", "forecast"):
            assert row[key] == results["maintain"][key]
    assert original == untouched
    missing = build_assessment(original, subject=SUBJECT, holding=HOLDING)
    assert missing["investor_decision"]["branches"]["held"]["action"] == "review"
    assert missing["investor_decision"]["branches"]["flat"]["action"] == "wait"


def test_absent_forecast_and_unknown_shares_save_exact_snapshot(tmp_path):
    research = thesis("maintain")
    saved = build_assessment(None, subject=SUBJECT, research_thesis=research,
                             current_value={"market_value": None, "currency": "CNY"})
    assert saved["schema_version"] == 4
    assert saved["forecast"] == saved["decision"] == saved["execution"] == {}
    assert saved["forecast_status"] == "absent" and saved["holding_status"] == "unknown"
    assert saved["amounts"]["anchor"]["status"] == "missing_forecast"
    assert set(saved["investor_decision"]["branches"]) == {"held", "flat", "addition"}
    portfolio = Portfolio(tmp_path)
    receipt = portfolio.save_assessment(saved)
    prior = deepcopy(portfolio.assessment_detail(receipt["id"])["assessment"])
    research["payload"]["posture"] = "exit"
    portfolio.save_fund(SUBJECT, shares="200", cost="999")
    assert portfolio.assessment_detail(receipt["id"])["assessment"] == prior == saved
    mismatch = build_assessment(prediction(), subject=SUBJECT, forecast_status="model_mismatch")
    assert mismatch["forecast"] == {} and mismatch["forecast_status"] == "model_mismatch"


def test_explicit_existing_loss_changes_personal_capacity_without_changing_research():
    sizing = conditional_size(scenario_return="-0.2", subscription_fee="0.01", existing_scenario_loss="2000",
        loss_budget="1500", spendable_cash="5000", existing_market_value="10000", redemption_fee="0.005",
        scenario_scope="whole_fund", budget_scope="fund", scenario_label="合成整基金压力假设")
    assert sizing["gross_cash_cap"] == "0"
    result = build_assessment(None, subject=SUBJECT, holding=HOLDING,
                              research_thesis=thesis("accumulate"), sizing_plan=sizing)
    branches = result["investor_decision"]["branches"]
    assert branches["held"]["action"] == "reduce"
    assert branches["flat"]["action"] == branches["addition"]["action"] == "wait"
    assert result["investor_decision"]["posture"] == "accumulate"

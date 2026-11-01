"""Authored choices and exact cash comparisons remain separate, with no IO fitting."""
from copy import deepcopy
import json
from importlib.resources import files

import pytest

from rmbfund.decision_brief import (
    build_decision_brief, build_legacy_decision_brief, load_decision_rationale,
    normalize_decision_rationale,
)
from rmbfund.domain import ValidationError
from rmbfund.personal_assessment import build_assessment
from rmbfund.timing_scenario import build_calendar_scenario, resolve_plan_fees
from rmbfund.trade_calendar import build_trade_plan
from test_ui import PRODUCT, generated_thesis, prediction


def scenario(value=None, *, new_confirmation=None):
    plan = build_trade_plan(decision_at="2026-10-29T09:00:00+08:00", anchor_date="2026-10-21", calendar_days=30)
    if new_confirmation:
        from rmbfund.trade_calendar import adjust_new_confirmation_date
        plan = adjust_new_confirmation_date(plan, new_confirmation)
    fees = resolve_plan_fees(None, plan=plan, old_confirmation_date="2026-10-01",
                             subscription_override=0, current_redemption_override=0,
                             future_old_redemption_override=0, future_new_redemption_override=0)
    value = deepcopy(value or prediction())
    horizon, lag = plan["horizon"], plan["entry_lag"]
    value["execution_forecasts"] = [{**value["execution_forecasts"][0], "horizon": horizon, "entry_lag": lag,
                                     "expected_change": 0.04}]
    return value, build_calendar_scenario(value, subject=PRODUCT, plan=plan, fee_basis=fees)


def authored(thesis):
    payload = thesis["payload"]
    facts = [row["id"] for row in payload["facts"]]
    reason = {"reason": "合成作者按经营与短期检验证据作明确取舍。", "fact_ids": facts}
    return {"schema_version": 1, "id": "synthetic-rationale", "product_id": PRODUCT["id"],
            "research_record_id": thesis["record_id"], "generated_at": "2026-10-28T12:00:00+08:00",
            "origin": {"provider": "synthetic"}, "fact_refs": facts,
            "scope": {"window_semantics": "review_not_forced_exit", "model_families": ["nav_ridge_empirical"]},
            "cases": {key: {**reason, "action": case["action"],
                             "why_not": [{**reason, "action": "wait" if key == "new_buy" else "hold"}],
                             "short_window_resolution": {status: deepcopy(reason) for status in ("aligned", "opposed", "unavailable")},
                             "checks": [{"condition": "下次经营现金是否转弱", "effect": "重新比较现金与基金", "fact_ids": facts}]}
                      for key, case in payload["action_cases"].items()}}


def assessment(thesis=None, *, rationale=None, value=None, trade=None):
    thesis = thesis or generated_thesis()
    if trade is None:
        value, trade = scenario(value)
    return build_assessment(value, subject=PRODUCT, horizon=trade["horizon"], trade_scenario=trade,
                            research_thesis=thesis, include_decision_brief=True, decision_rationale=rationale)


@pytest.mark.parametrize("held,addition,expected", [
    ("hold", "add", "add"), ("hold", "wait", "hold"),
    ("reduce", "add", "reduce"), ("redeem", "add", "redeem"), ("review", "add", None),
])
def test_independent_cases_determine_choice_not_greatest_amount(held, addition, expected):
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["held"]["action"] = held
    thesis["payload"]["action_cases"]["addition"]["action"] = addition
    result = assessment(thesis, rationale=authored(thesis))["decision_brief"]
    assert result["cases"]["held"]["preferred_action"] == expected
    assert result["cases"]["held"]["reduction_ratio"] is None
    assert result["cases"]["held"]["research_action"] == expected


def test_unverified_model_disagreement_is_diagnostic_not_a_veto_of_economic_wait():
    thesis = generated_thesis()  # Authored wait, rising synthetic market.
    missing = assessment(thesis)["decision_brief"]["cases"]["flat"]
    assert missing["conflict"]["status"] == "opposed"
    assert missing["conflict"]["role"] == "diagnostic_only"
    assert missing["research_action"] == missing["preferred_action"] == "wait"
    assert missing["decision_status"] == "economic_research"
    matched = assessment(thesis, rationale=authored(thesis))["decision_brief"]
    assert matched["rationale_status"] == "not_used_by_economic_policy"
    assert matched["cases"]["flat"]["preferred_action"] == "wait"
    assert matched["cases"]["flat"]["conflict"]["resolution"] is None


def test_rationale_references_and_authorship_cannot_be_borrowed():
    thesis = generated_thesis()
    value = authored(thesis)
    assert normalize_decision_rationale(value, research_thesis=thesis, product_id=PRODUCT["id"]) == value
    for changed in ({"research_record_id": "different"}, {"fact_refs": ["nonexistent"]},
                    {"generated_at": "2026-10-20T12:00:00+08:00"}):
        with pytest.raises(ValidationError):
            normalize_decision_rationale({**value, **changed}, research_thesis=thesis, product_id=PRODUCT["id"])
    invalid = {**value, "research_record_id": "different"}
    result = assessment(thesis, rationale=invalid)["decision_brief"]
    assert result["rationale"] is None and result["rationale_status"] == "not_used_by_economic_policy"
    assert result["cases"]["flat"]["reasons"]
    assert result["cases"]["flat"]["preferred_action"] == "wait"


def test_fee_edit_recomputes_same_capital_delta_but_not_authored_reason_or_head():
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    thesis["payload"]["action_cases"]["addition"]["action"] = "add"
    value, trade = scenario()
    original = deepcopy((value, trade, thesis))
    before = assessment(thesis, rationale=authored(thesis), value=value, trade=trade)["decision_brief"]
    altered = deepcopy(trade)
    for row in altered["rows"]:
        if row["action"] in {"buy", "add"}:
            row["expected_value"] -= 123
    after = assessment(thesis, rationale=authored(thesis), value=value, trade=altered)["decision_brief"]
    assert after["cases"]["held"]["reasons"] == before["cases"]["held"]["reasons"]
    for result in (before, after):
        case = result["cases"]["held"]
        hold = next(row for row in case["why_not"] if row["action"] == "hold")
        assert hold["expected_difference_vs_preferred"] == pytest.approx(hold["expected_value"] - case["expected_value"])
    assert after["cases"]["held"]["why_not"][0]["expected_difference_vs_preferred"] == pytest.approx(
        before["cases"]["held"]["why_not"][0]["expected_difference_vs_preferred"] + 123)
    assert (value, trade, thesis) == original


def test_late_new_confirmation_is_not_real_subscription_block():
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    value, trade = scenario(new_confirmation="2026-12-01")
    result = assessment(thesis, rationale=authored(thesis), value=value, trade=trade)["decision_brief"]
    assert result["cases"]["flat"]["preferred_action"] == "buy"
    assert result["cases"]["flat"]["comparison_status"] != "ready"
    assert result["cases"]["flat"]["execution_status"]["status"] != "blocked"
    assert result["cases"]["held"]["comparison_status"] == "partial"


def test_scheduled_review_does_not_erase_independent_cases_but_elapsed_horizon_is_visible():
    thesis = generated_thesis()
    thesis["needed_review"] = ["review_date_elapsed"]
    current = assessment(thesis, rationale=authored(thesis))
    assert current["investor_decision"]["action_basis"] == "independent_cases"
    assert current["decision_brief"]["cases"]["held"]["preferred_action"] == "hold"
    assert current["decision_brief"]["cases"]["held"]["decision_status"] == "provisional"
    thesis["needed_review"] = ["research_horizon_elapsed"]
    later = assessment(thesis, rationale=authored(thesis))["decision_brief"]
    assert later["cases"]["held"]["research_action"] == "hold"
    assert later["cases"]["held"]["preferred_action"] is None


def test_actual_subscription_block_stays_separate_from_economic_preference():
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    value, trade = scenario()
    result = build_assessment(value, subject=PRODUCT, horizon=trade["horizon"], trade_scenario=trade,
        research_thesis=thesis, include_decision_brief=True, decision_rationale=authored(thesis),
        execution_conditions={"subscription": "closed", "redemption": "open"})["decision_brief"]
    assert result["cases"]["flat"]["preferred_action"] == "wait"
    assert result["cases"]["flat"]["economic_action"] == "buy"
    assert result["cases"]["flat"]["execution_status"]["status"] == "blocked"
    assert result["cases"]["flat"]["decision_status"] == "execution_blocked"


@pytest.mark.parametrize("constraint", ["zero_capacity", "near_term_use"])
@pytest.mark.parametrize("subscription", ["open", "closed"])
def test_explicit_money_constraints_keep_economic_choices_and_do_not_reuse_buy_copy(constraint, subscription):
    from rmbfund.investor_decision import build_investor_decision

    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    thesis["payload"]["action_cases"]["addition"]["action"] = "add"
    options = ({"sizing_plan": {"whole_fund": {"gross_cash_cap": 0}}} if constraint == "zero_capacity" else
               {"cash_use_fit": {"status": "not_suitable_for_new_money", "near_term_required": "yes",
                                 "reason": "合成明确近期用钱，应保留现金。"}})
    investor = build_investor_decision(product_id=PRODUCT["id"], research_thesis=thesis,
        execution_conditions={"subscription": subscription, "redemption": "open"},
        preserve_research_cases=True, **options)
    _, trade = scenario()
    result = build_decision_brief(product_id=PRODUCT["id"], research_thesis=thesis,
        investor_decision=investor, trade_scenario=trade, rationale=authored(thesis))
    flat, held = result["cases"]["flat"], result["cases"]["held"]
    assert flat["preferred_action"] == "wait" and held["preferred_action"] == "hold"
    assert flat["research_action"] == "buy" and held["research_action"] == "add"
    assert flat["decision_status"] == held["decision_status"] == "constrained"
    assert all("合成作者按经营" not in reason for case in (flat, held) for reason in case["reasons"])
    assert flat["reasons"][0] in investor["branches"]["flat"]["constraint_reasons"]
    assert held["reasons"][0] in investor["branches"]["addition"]["constraint_reasons"]


def test_packaged_rationale_is_version_specific_and_does_not_require_artifact_files():
    resource = json.loads(files("rmbfund").joinpath("resources/decision_rationales.json").read_text())
    for value in resource:
        result = load_decision_rationale(value["product_id"], value["research_record_id"])
        assert result == value
        assert all(set(row["short_window_resolution"]) == {"aligned", "opposed", "unavailable"}
                   for row in value["cases"].values())
    assert load_decision_rationale(PRODUCT["id"], "synthetic-other") is None
    assert load_decision_rationale(None, None) is None


def test_no_research_does_not_invent_sale_or_active_hold():
    value, trade = scenario()
    result = build_decision_brief(product_id=PRODUCT["id"], research_thesis=None,
        investor_decision={"action_basis": "unavailable"}, trade_scenario=trade)
    assert all(case["preferred_action"] is None for case in result["cases"].values())
    assert all(case["label"] == "暂不操作，先核对买卖依据" for case in result["cases"].values())


def packaged_operation(*, buy=10400, hold=20400, add=20800, subscription="open", redemption="open",
                       builder=build_legacy_decision_brief):
    rationale = json.loads(files("rmbfund").joinpath("resources/decision_rationales.json").read_text())[0]
    identifier = rationale["product_id"]
    thesis = {"record_id": rationale["research_record_id"], "payload": {
        "product": {"id": identifier}, "origin": {"generated_at": "2026-10-28T12:00:00+08:00"},
        "facts": [{"id": key} for key in rationale["fact_refs"]],
        "action_cases": {key: {"action": case["action"]} for key, case in rationale["cases"].items()}}}
    investor = {"action_basis": "independent_cases", "branches": {
        key: {"action": action, "research_action": action, "reasons": ["原研究支持产业参与。"]}
        for key, action in (("flat", "buy"), ("held", "hold"), ("addition", "add"))},
        "execution_conditions": {"subscription": subscription, "redemption": redemption}}
    amounts = {"buy": buy, "wait": 10000, "hold": hold, "add": add,
               "exit": 20000, "reduce50": 999999}
    trade = {"plan": {"decision_at": "2026-10-31T12:00:00+08:00"}, "rows": [
        {"action": action, "status": "ready" if value is not None else "unavailable",
         "expected_value": value, "initial_value": 10000 if action in {"buy", "wait"} else 20000}
        for action, value in amounts.items()]}
    return builder(product_id=identifier, research_thesis=thesis,
        investor_decision=investor, trade_scenario=trade, rationale=rationale,
        model_id="nav_ridge_empirical_v2")


@pytest.mark.parametrize("buy,add", [(9800, 20200), (10000, 20400)])
def test_authored_operation_resolves_unprofitable_or_equal_new_money_as_cash(buy, add):
    result = packaged_operation(buy=buy, add=add)
    assert result["schema_version"] == 2
    assert result["policy_version"] == "fund_buyer_advice_v2"
    flat, held = result["cases"]["flat"], result["cases"]["held"]
    assert flat["preferred_action"] == "wait" and held["preferred_action"] == "hold"
    assert flat["research_action"] == "buy" and held["research_action"] == "add"
    assert flat["comparison_action"] == "buy" and held["comparison_action"] == "add"
    assert flat["decision_status"] == held["decision_status"] == "authored_operation"
    assert "不如留现金" in flat["advice_reason"]
    assert "不追加" in held["advice_reason"]
    assert flat["expected_value"] == 10000 and held["expected_value"] == 20400
    assert flat["next_step"] and held["next_step"]
    assert next(row for row in flat["why_not"] if row["action"] == "buy")["expected_difference_vs_preferred"] == buy - 10000
    assert next(row for row in held["why_not"] if row["action"] == "add")["expected_difference_vs_preferred"] == add - 20400


def test_held_exit_resolution_is_not_overridden_by_addition_advice_or_largest_amount():
    result = packaged_operation(buy=9800, hold=19950, add=19750)["cases"]["held"]
    assert result["preferred_action"] == "redeem"
    assert result["research_action"] == "add"
    assert result["expected_value"] == 20000
    assert result["comparison_action"] == "hold"
    assert result["reduction_ratio"] is None
    assert "现在退出" in result["advice_reason"]


def test_favorable_authored_new_money_recommends_buy_and_add_not_the_largest_row():
    result = packaged_operation()
    assert result["cases"]["flat"]["preferred_action"] == "buy"
    assert result["cases"]["held"]["preferred_action"] == "add"
    assert result["cases"]["held"]["expected_value"] == 20800


def test_missing_exact_new_money_comparison_defers_new_money_without_claiming_a_real_block():
    result = packaged_operation(buy=None, add=None)
    flat, held = result["cases"]["flat"], result["cases"]["held"]
    assert flat["preferred_action"] == "wait" and held["preferred_action"] == "hold"
    assert flat["execution_status"]["status"] == "not_required"
    assert "暂停" not in flat["advice_reason"]
    assert "还没算出" in flat["advice_reason"]


@pytest.mark.parametrize("operation", ["subscription", "redemption"])
def test_actual_execution_block_resolves_to_an_actionable_non_trade_without_changing_research(operation):
    result = packaged_operation(subscription="closed" if operation == "subscription" else "open",
        redemption="closed" if operation == "redemption" else "open",
        hold=19950 if operation == "redemption" else 20400,
        add=19750 if operation == "redemption" else 20800)
    held = result["cases"]["held"]
    assert held["preferred_action"] == "hold" and held["research_action"] == "add"
    assert held["economic_action"] == ("add" if operation == "subscription" else "redeem")
    assert held["execution_status"]["status"] == "blocked"
    assert held["decision_status"] == "execution_blocked"

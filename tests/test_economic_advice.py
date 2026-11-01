"""Current economic policy cannot trade on an unvalidated short-window mean."""
from copy import deepcopy

import pytest

from rmbfund.decision_brief import POLICY_VERSION, build_decision_brief
from rmbfund.research_contract import normalize_generated_research
from test_decision_brief import assessment, packaged_operation, scenario
from test_ui import generated_thesis


@pytest.mark.parametrize("hold,add", [(19981.12, 19981.11), (19999.99, 19999.98), (20000, 20000), (21000, 21500)])
def test_tiny_or_equal_or_positive_model_values_do_not_select_an_action(hold, add):
    result = packaged_operation(buy=9999.99, hold=hold, add=add, builder=build_decision_brief)
    assert result["schema_version"] == 3 and result["policy_version"] == POLICY_VERSION
    assert result["cases"]["flat"]["preferred_action"] == "buy"
    assert result["cases"]["held"]["preferred_action"] == "add"
    assert result["cases"]["held"]["forecast_evidence"]["timing_advantage_established"] is False
    assert result["rationale"] is None


def test_economic_redeem_survives_bullish_prediction_and_not_maximum_row():
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["held"].update(
        action="redeem", reason="合成需求与回款已经转弱，当前价格仍要求快速增长。",
        summary="经营兑现不足以支持当前价格，退出。", next_step="经营和价格要求重新匹配后再考虑。")
    result = assessment(thesis)["decision_brief"]["cases"]["held"]
    assert result["preferred_action"] == "redeem"
    assert result["advice_reason"] == "经营兑现不足以支持当前价格，退出。"
    assert result["next_step"] == "经营和价格要求重新匹配后再考虑。"
    assert result["reduction_ratio"] is None


def test_weak_model_cannot_turn_economic_wait_into_a_buy():
    thesis = generated_thesis()
    result = assessment(thesis)["decision_brief"]["cases"]["flat"]
    assert result["preferred_action"] == "wait"
    assert result["conflict"]["role"] == "diagnostic_only"


def test_missing_exact_prediction_does_not_erase_economic_case_or_fake_block():
    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    value, trade = scenario(new_confirmation="2026-12-01")
    before = deepcopy((value, trade, thesis))
    result = assessment(thesis, value=value, trade=trade)["decision_brief"]
    assert result["cases"]["flat"]["preferred_action"] == "buy"
    assert result["cases"]["flat"]["execution_status"]["status"] != "blocked"
    assert result["data_date"] == trade["plan"]["anchor_date"]
    assert (value, trade, thesis) == before


def test_authored_short_summary_and_alternative_share_validated_facts():
    thesis = generated_thesis()
    payload = thesis["payload"]
    ids = payload["action_cases"]["new_buy"]["fact_ids"]
    payload["action_cases"]["new_buy"].update(
        summary="当前价格暂未体现足够参与价值，先留现金。",
        next_step="价格要求降低，或经营兑现增强后再考虑。",
        comparison={"action": "buy", "reason": "相较立刻买入，现金避免支付尚无充分依据的风险成本。", "fact_ids": ids},
    )
    normalized = normalize_generated_research(payload)
    assert normalized["action_cases"]["new_buy"]["summary"] == payload["action_cases"]["new_buy"]["summary"]
    result = assessment(thesis)["decision_brief"]["cases"]["flat"]
    assert result["advice_reason"] == payload["action_cases"]["new_buy"]["summary"]
    assert result["comparison_action"] == "buy"
    assert next(row for row in result["why_not"] if row["action"] == "buy")["reason"]


@pytest.mark.parametrize("group,action,operation", [("flat", "buy", "subscription"), ("held", "redeem", "redemption")])
def test_expired_economic_case_is_not_restored_by_a_suspension(group, action, operation):
    from rmbfund.personal_assessment import build_assessment
    from test_ui import PRODUCT

    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["new_buy" if group == "flat" else "held"]["action"] = action
    thesis["needed_review"] = ["research_horizon_elapsed"]
    value, trade = scenario()
    result = build_assessment(value, subject=PRODUCT, horizon=trade["horizon"], trade_scenario=trade,
        research_thesis=thesis, include_decision_brief=True,
        execution_conditions={"subscription": "closed" if operation == "subscription" else "open",
                              "redemption": "closed" if operation == "redemption" else "open"})["decision_brief"]["cases"][group]
    assert result["preferred_action"] is None and result["economic_action"] is None
    assert result["decision_status"] == "needs_review"
    assert "已过适用期限" in result["advice_reason"]
    assert "新的经营与价格依据" in result["next_step"]


def test_actual_redemption_suspension_preserves_economic_sale_not_a_filled_order():
    from rmbfund.personal_assessment import build_assessment
    from test_ui import PRODUCT

    thesis = generated_thesis()
    thesis["payload"]["action_cases"]["held"]["action"] = "redeem"
    value, trade = scenario()
    result = build_assessment(value, subject=PRODUCT, horizon=trade["horizon"], trade_scenario=trade,
        research_thesis=thesis, include_decision_brief=True,
        execution_conditions={"subscription": "open", "redemption": "closed"})["decision_brief"]["cases"]["held"]
    assert result["preferred_action"] == "hold" and result["economic_action"] == "redeem"
    assert result["execution_status"]["status"] == "blocked"

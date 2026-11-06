"""Schema5/6/7 history replays stored JSON, not today's plans, fees, or builders."""

from copy import deepcopy
import json

import pytest
from streamlit.testing.v1 import AppTest

from rmbfund.portfolio import Portfolio


# Deliberately literal saved values: the compatibility decision and personal
# amounts differ from the standard scenario, so accidental schema4 dispatch is
# visible without asking a current builder to reproduce an older snapshot.
SAVED_JSON = r'''{
  "schema_version": 5,
  "policy_version": "standard_trade_scenarios_v1",
  "mode": "standard_trade",
  "subject": {"id": "synthetic", "symbol": "SYNTHETIC", "kind": "open_fund", "name": "合成基金", "share_class": "C"},
  "holding_status": "held",
  "horizon": 20,
  "entry_lag": 4,
  "forecast": {"record_id": "retained-id-is-not-issuance", "generated_at": "2026-09-25T10:15:00+08:00", "model_id": "saved_synthetic_model", "model_name": "原保存方法", "model_hash": "saved-model-version", "anchor": {"date": "2026-09-24", "nav": "1.25", "currency": "CNY"}},
  "calculation": {"kind": "local_hypothetical", "persisted": false, "actual_issue": false, "historical_availability_basis": "one_observation_lag_reconstruction", "cache_key": "original-local-key"},
  "decision": {"branches": {"held": {"action": "hold", "label": "兼容旧字段不是标准交易", "reasons": ["兼容个人费用理由"]}}},
  "investor_decision": {"action_basis": "independent_cases", "research_summary": "原研究只讨论整只基金", "branches": {"flat": {"action": "wait", "label": "原研究新买待观察", "research_reason": "原新买研究理由", "research_action": "wait"}, "held": {"action": "hold", "label": "原研究保留", "research_reason": "原持有研究理由", "research_action": "hold"}, "addition": {"action": "wait", "label": "原研究暂不追加", "research_reason": "原追加研究理由", "research_action": "wait"}}},
  "inputs": {"holding": {"shares": "999", "cost": "999", "currency": "CNY"}, "subscription_fee": "0.99", "redemption_fee": "0.88"},
  "amounts": {"current": {"market_value": "999999", "profit": "999999", "currency": "CNY", "nav_date": "2026-09-24"}},
  "trade_scenario": {
    "kind": "standard_trade", "status": "ready", "horizon": 5, "entry_lag": 2, "base_amount": 10000,
    "purchase": {"date": "2026-09-01", "nav": 1.0, "latest_date": "2026-09-24", "latest_nav": 1.25, "change": 0.25, "profit": 2500, "holding_days": 23, "status": "available"},
    "fees": {"subscription_fee": 0.0015, "redemption_fee": 0.005, "subscription_fee_known": true, "redemption_fee_known": true, "basis": {"subscription": {"rate": 0.0015, "source": "illustrative", "label": "原示例申购费，不是支付宝实际费"}, "redemption": {"rate": 0.005, "source": "public_standard", "label": "原公开标准赎回费，渠道未核实", "band": {"min_days": 7, "max_days": 29, "rate": 0.005}}, "valuation_date": "2026-09-24", "holding_days": 23, "holding_day_label": "原自然日模拟口径", "observed_at": "2026-09-25T09:00:00+08:00", "channel_verified": false}},
    "decision": {"status": "ready", "primary_horizon": 5, "primary_entry_lag": 2, "entry_lag": 2, "primary_head": {"horizon": 5, "entry_lag": 2, "up_probability": 0.73, "expected_change": 0.0317, "lower": -0.025, "upper": 0.091, "down_probability": 0.27, "flat_probability": 0, "reasons": ["保存的主预测理由"]}, "primary": {"market_expected_change": 0.0317, "buy_advantage_vs_cash": 0.03015476784822758, "hold_advantage_vs_exit": 0.0367, "buy_profit_probability": 0.7, "hold_better_probability": 0.8, "sell_better_probability": 0.2}, "branches": {"flat": {"action": "buy", "label": "原标准新买判断", "reasons": ["原标准新买理由"]}, "held": {"action": "hold", "label": "原标准持有判断", "reasons": ["原标准持有理由"]}}, "scenarios": [{"horizon": 5, "entry_lag": 2, "is_primary": true}], "delay_sensitive": false, "assumptions": ["原保存口径"]},
    "rows": [
      {"action": "buy", "label": "原单笔买入", "initial_value": 10000, "expected_value": 10301.55, "lower_value": 9735.40, "upper_value": 10893.66, "expected_change": 0.030155, "comparison_label": "保持现金", "advantage": 301.55, "better_probability": 0.7},
      {"action": "wait", "label": "原保留现金", "initial_value": 10000, "expected_value": 10000, "lower_value": 10000, "upper_value": 10000, "expected_change": 0, "comparison_label": "保持现金", "advantage": 0, "better_probability": 0},
      {"action": "hold", "label": "原继续持有", "initial_value": 20000, "expected_value": 20317, "lower_value": 19750, "upper_value": 20910, "expected_change": 0.01585, "comparison_label": "全部退出", "advantage": 367, "better_probability": 0.8},
      {"action": "add", "label": "原追加同额", "initial_value": 20000, "expected_value": 20618.55, "lower_value": 19485.4, "upper_value": 21803.66, "expected_change": 0.0309275, "comparison_label": "继续持有", "advantage": 301.55, "better_probability": 0.7},
      {"action": "reduce50", "label": "原减仓一半", "initial_value": 20000, "expected_value": 20133.5, "lower_value": 19850, "upper_value": 20430, "expected_change": 0.006675, "comparison_label": "继续持有", "advantage": -183.5, "better_probability": 0.2},
      {"action": "exit", "label": "原全部退出", "initial_value": 20000, "expected_value": 19950, "lower_value": 19950, "upper_value": 19950, "expected_change": -0.0025, "comparison_label": "继续持有", "advantage": -367, "better_probability": 0.2}
    ]
  },
  "reconstructed_history": {"kind": "reconstructed", "horizon": 5, "entry_lag": 2, "counts": {"mature": 7, "immature": 2, "invalid": 1, "unavailable": 3, "nonoverlapping_windows": 2}, "metrics": {"flat": {"count": 7, "mean_realized_advantage": 54.321}, "held": {"count": 7, "mean_realized_advantage": -12.345}}, "rows": [{"origin_date": "2026-06-01", "assumed_known_date": "2026-06-02", "status": "mature", "expected_change": 0.021, "actual_change": -0.047, "flat_action": "buy", "held_action": "hold"}]},
  "research_thesis": {"record_id": "research-original", "payload": {"schema_version": 1, "research_date": "2026-09-20", "author": "合成作者", "reviewer": "合成复核", "assessment": {"summary": "保存的研究原文", "priced_in": {"reason": "保存的价格解释"}}, "facts": [{"id": "fact-original", "claim": "保存的研究事实"}], "citations": []}},
  "online_context": {"checked_at": "2026-09-25T09:00:00+08:00", "messages_by_product": {"synthetic": [{"published_at": "2026-09-22", "title": "保存的在线消息", "content_status": "online_excerpt", "excerpt": "保存的原始在线摘录"}]}},
  "investment_analysis": {"source_gate": "保存的独立条件分析", "profile": {"as_of": "2026-09-21T12:00:00+08:00"}},
  "personal_conversion": {"schema_version": 4, "horizon": 5, "holding_status": "held", "inputs": {"holding": {"shares": "12.30", "cost": "10.00", "currency": "CNY"}, "subscription_fee": "0.012", "redemption_fee": "0.023", "budget": {"amount": "123.45", "currency": "CNY"}}, "amounts": {"current": {"market_value": "15.375", "profit": "5.375", "nav_date": "2026-09-24", "currency": "CNY"}, "new_buy": {"basis_value": "123.45", "currency": "CNY", "expected": {"value": "125.67", "change": "2.22", "profit": "2.22"}}}}
}'''


# Independent literal: never derive a schema6 snapshot from a current builder or
# silently upgrade the schema5 mark-to-NAV fixture above.
CALENDAR_SAVED_JSON = r'''{
  "schema_version": 6,
  "policy_version": "calendar_trade_cash_v1",
  "mode": "calendar_trade",
  "subject": {"id": "synthetic", "symbol": "SYNTHETIC", "kind": "open_fund", "name": "合成基金", "share_class": "C"},
  "holding_status": "held", "horizon": 8, "entry_lag": 2,
  "forecast": {"record_id": "saved-id-not-issuance", "generated_at": "2026-09-25T10:15:00+08:00", "model_id": "saved_synthetic_model", "model_name": "原自然日保存方法", "model_hash": "calendar-model-original", "anchor": {"date": "2026-09-23", "nav": "1.25", "currency": "CNY"}, "target_unit": "nav_observations", "contract": {"id": "calendar-local-original", "horizons": [8], "entry_lags": [2]}},
  "calculation": {"kind": "local_hypothetical", "generated_at": "2026-09-25T10:16:00+08:00", "actual_issue": false, "persisted": false, "target_unit": "nav_observations", "contract": {"id": "calendar-local-original", "horizons": [8], "entry_lags": [2]}, "cache_key": "calendar-original-key"},
  "decision": {"branches": {"held": {"action": "sell", "label": "兼容字段不得代替自然日交易", "reasons": ["兼容字段不回放"]}}},
  "investor_decision": {"action_basis": "independent_cases", "branches": {"flat": {"action": "wait", "label": "原自然日基金研究待观察", "research_reason": "原研究不能折算短期概率"}, "held": {"action": "hold", "label": "原自然日基金研究持有", "research_reason": "原研究理由不随日历重写"}, "addition": {"action": "wait", "label": "原自然日基金研究不追加", "research_reason": "原追加研究理由"}}},
  "inputs": {"redemption_fee": "0.99", "subscription_fee": "0.88"},
  "amounts": {"current": {"market_value": "999999", "profit": "999999", "currency": "CNY"}},
  "trade_scenario": {
    "kind": "calendar_cash", "status": "ready", "horizon": 8, "entry_lag": 2, "base_amount": 10000,
    "end_value_basis": "cash_and_redemption_receivable",
    "plan": {"decision_at": "2026-09-25T10:15:00+08:00", "decision_date": "2026-09-25", "calendar_days": 12, "deadline": "2026-10-07", "application_date": "2026-09-25", "sale_date": "2026-10-07", "entry_lag": 2, "horizon": 8, "new_confirmation_date": "2026-09-29", "calendar_basis": "weekday_simulation", "assumptions": ["原周一至周五模拟日程，不承诺真实申赎开放", "原每业务日一笔净值只是预测日程假设"], "reasons": ["原计划截止日前卖出，不承诺资金到账"]},
    "purchase": {"date": "2026-09-01", "nav": 1.0, "latest_date": "2026-09-23", "latest_nav": 1.25, "change": 0.25, "profit": 2500, "holding_days": 22, "status": "available"},
    "fees": {
      "subscription_fee": 0.0015, "current_redemption_fee": 0.015, "future_old_redemption_fee": 0.005, "future_new_redemption_fee": 0.0125,
      "old_confirmation_date": "2026-09-21", "new_confirmation_date": "2026-09-29",
      "assumptions": ["原假设当前费表未来不变，不是未来渠道承诺"],
      "contexts": {
        "current": {"subscription_fee": 0.0015, "redemption_fee": 0.015, "subscription": {"rate": 0.0015, "source": "illustrative", "label": "原自然日示例申购费，非支付宝实际费"}, "redemption": {"rate": 0.015, "source": "public_standard", "label": "原旧份额当前公开分段费率", "band": {"min_days": 0, "max_days": 6, "rate": 0.015}}, "purchase_date": "2026-09-21", "valuation_date": "2026-09-25", "holding_days": 4, "holding_day_label": "原旧份额确认日起点模拟口径", "observed_at": "2026-09-24T09:00:00+08:00", "channel_verified": false},
        "future_old": {"subscription_fee": 0.0015, "redemption_fee": 0.005, "redemption": {"rate": 0.005, "source": "public_standard", "label": "原旧份额计划卖出公开分段费率", "band": {"min_days": 7, "max_days": 29, "rate": 0.005}}, "purchase_date": "2026-09-21", "valuation_date": "2026-10-07", "holding_days": 16, "holding_day_label": "原旧份额未来费龄模拟口径", "assumptions": ["原公开费表保持不变"]},
        "future_new": {"subscription_fee": 0.0015, "redemption_fee": 0.0125, "redemption": {"rate": 0.0125, "source": "override", "label": "原手动新份额计划卖出费，不覆盖旧份额", "band": null}, "purchase_date": "2026-09-29", "valuation_date": "2026-10-07", "holding_days": 8, "holding_day_label": "原新份额独立确认日起点模拟口径", "reasons": ["原新份额不继承历史买入净值日"]}
      }
    },
    "decision": {"status": "ready", "primary_horizon": 8, "primary_entry_lag": 2, "entry_lag": 2, "primary_head": {"horizon": 8, "entry_lag": 2, "target_unit": "nav_observations", "up_probability": 0.73, "expected_change": 0.0317, "lower": -0.025, "upper": 0.091, "reasons": ["原八笔准确预测头"]}, "primary": {"market_expected_change": 0.0317, "buy_advantage_vs_cash": 0.0175, "hold_advantage_vs_exit": 0.0415, "buy_profit_probability": 0.64, "hold_better_probability": 0.76, "sell_better_probability": 0.24}, "branches": {"flat": {"action": "buy", "label": "原自然日新买判断", "reasons": ["原自然日新买净额理由"]}, "held": {"action": "hold", "label": "原自然日持有判断", "reasons": ["原自然日旧份额净额理由"]}, "addition": {"action": "wait", "label": "原自然日追加独立判断", "reasons": ["原自然日新份额费用理由"]}}, "delay_sensitive": false},
    "rows": [
      {"action": "buy", "label": "原自然日新买后卖出", "initial_value": 10000, "expected_value": 10175.03, "lower_value": 9613.70, "upper_value": 10759.74, "expected_change": 0.017503, "comparison_label": "保持现金", "advantage": 175.03, "better_probability": 0.64},
      {"action": "wait", "label": "原自然日保留现金", "initial_value": 10000, "expected_value": 10000, "lower_value": 10000, "upper_value": 10000, "expected_change": 0, "comparison_label": "保持现金", "advantage": 0, "better_probability": 0},
      {"action": "hold", "label": "原自然日旧份额到期卖出", "initial_value": 20000, "expected_value": 20265.42, "lower_value": 19701.25, "upper_value": 20855.45, "expected_change": 0.013271, "comparison_label": "当前全部退出", "advantage": 415.42, "better_probability": 0.76},
      {"action": "add", "label": "原自然日两批份额到期卖出", "initial_value": 20000, "expected_value": 20440.45, "lower_value": 19314.95, "upper_value": 21615.19, "expected_change": 0.0220225, "comparison_label": "旧份额到期卖出", "advantage": 175.03, "better_probability": 0.64},
      {"action": "reduce50", "label": "原自然日现在卖半再卖剩余", "initial_value": 20000, "expected_value": 20057.71, "lower_value": 19775.63, "upper_value": 20352.73, "expected_change": 0.0028855, "comparison_label": "旧份额到期卖出", "advantage": -207.71, "better_probability": 0.24},
      {"action": "exit", "label": "原自然日当前全部退出", "initial_value": 20000, "expected_value": 19850, "lower_value": 19850, "upper_value": 19850, "expected_change": -0.0075, "comparison_label": "旧份额到期卖出", "advantage": -415.42, "better_probability": 0.24}
    ]
  },
  "reconstructed_history": {"kind": "calendar_cash_fixed_fee_nav_history", "end_value_basis": "cash_and_redemption_receivable", "horizon": 8, "entry_lag": 2, "counts": {"mature": 7, "immature": 2, "nowcast": 1, "invalid": 1, "unavailable": 3, "nonoverlapping_windows": 2}, "metrics": {"flat": {"count": 7, "mean_realized_advantage": 43.21}, "held": {"count": 7, "mean_realized_advantage": -12.34}}, "rows": [{"origin_date": "2026-06-03", "assumed_known_date": "2026-06-04", "status": "mature", "prediction": {"expected_change": 0.021}, "actual_change": -0.047, "flat_action": "buy", "held_action": "hold"}]},
  "trade_research_connection": {"research_date": "2026-09-20", "horizon": {"start": "2026-09-20", "end": "2027-09-20"}, "lines": ["原保存的自然日计划与研究对照，不重生成"]},
  "research_thesis": {"record_id": "calendar-research-original", "payload": {"schema_version": 1, "research_date": "2026-09-20", "assessment": {"summary": "原自然日研究原文"}, "facts": [{"id": "calendar-fact", "claim": "原自然日研究事实"}], "citations": []}},
  "online_context": {"checked_at": "2026-09-25T09:00:00+08:00", "messages_by_product": {"synthetic": [{"published_at": "2026-09-22", "title": "原自然日在线消息", "content_status": "online_excerpt", "excerpt": "原自然日在线摘录"}]}},
  "investment_analysis": {"source_gate": "原自然日独立条件分析"},
  "personal_conversion": {"schema_version": 4, "inputs": {"holding": {"shares": "12.30", "cost": "10.00", "currency": "CNY"}, "budget": {"amount": "123.45", "currency": "CNY"}}, "amounts": {"current": {"market_value": "15.375", "profit": "5.375", "currency": "CNY", "nav_date": "2026-09-23"}, "new_buy": {"basis_value": "123.45", "currency": "CNY", "end_value_basis": "cash_and_redemption_receivable", "expected": {"value": "125.67", "change": "2.22", "profit": "2.22"}}}, "trade_scenario": {"kind": "calendar_cash", "horizon": 8, "entry_lag": 2, "base_amount": 123.45, "end_value_basis": "cash_and_redemption_receivable", "plan": {"calendar_days": 12, "deadline": "2026-10-07", "horizon": 8, "entry_lag": 2}, "decision": {"primary_head": {"horizon": 8, "entry_lag": 2, "up_probability": 0.73, "expected_change": 0.0317, "lower": -0.025, "upper": 0.091}, "branches": {"flat": {"action": "buy", "label": "原可选个人净额换算"}}}, "rows": [{"action": "buy", "label": "原个人新买计划卖出", "initial_value": 123.45, "expected_value": 125.67, "lower_value": 120.01, "upper_value": 130.02, "expected_change": 0.01798, "comparison_label": "留现金", "advantage": 2.22, "better_probability": 0.64}]}}
}'''


def saved_json():
    return json.loads(SAVED_JSON)


def calendar_saved_json():
    return json.loads(CALENDAR_SAVED_JSON)


# Schema7 retains the same literal calendar inputs, with a separately authored
# brief whose choices intentionally differ from the saved short-window branches.
# No current builder or packaged rationale is used to create this fixture.
BRIEF_SAVED_JSON = r'''{
  "schema_version": 1, "product_id": "synthetic", "research_record_id": "calendar-research-original",
  "research_date": "2026-09-20",
  "rationale": {"id": "saved-rationale-original", "product_id": "synthetic", "research_record_id": "calendar-research-original", "origin": {"provider": "interactive_assistant", "model": "saved-author"}, "generated_at": "2026-09-25T08:30:00+08:00", "scope": {"capital_task": "原保存的标准单笔资金任务"}},
  "cases": {
    "flat": {"preferred_action": "wait", "research_action": "wait", "label": "原首选先留现金", "reasons": ["原首选：价格尚未补偿新增资金风险", "原第二条经济依据"], "why_not": [{"action": "buy", "label": "原备选新买", "reason": "原备选取舍：保留现金放弃上涨，但不新增价格风险", "fact_ids": ["calendar-fact"], "expected_difference_vs_preferred": 175.03}], "counterevidence": [{"reason": "原最强反方：盈利可能超出价格预期", "fact_ids": ["calendar-fact"]}], "checks": [{"condition": "原观察盈利是否兑现", "effect": "兑现且价格未透支才重看新买", "fact_ids": ["calendar-fact"], "status": "watch"}], "conflict": {"status": "opposed", "resolution": "原作者裁决：短窗净额为正仍不足以补足价格依据", "fact_ids": ["calendar-fact"]}, "execution_status": {"status": "not_required"}, "comparison_status": "ready", "decision_status": "ready"},
    "held": {"preferred_action": "hold", "research_action": "hold", "label": "原首选保留但不追加", "reasons": ["原持有机会成本已与退出比较"], "why_not": [{"action": "add", "label": "原备选追加", "reason": "原追加取舍：新增一笔缺少独立价格余量", "fact_ids": ["calendar-fact"], "expected_difference_vs_preferred": 175.03}], "counterevidence": [], "checks": [], "conflict": {"status": "aligned", "resolution": "原已有份额取舍与净额比较同向", "fact_ids": ["calendar-fact"]}, "execution_status": {"status": "not_required"}, "comparison_status": "ready", "decision_status": "ready"}
  },
  "plan_semantics": "原N是考虑操作与复核窗口；S只用于统一金额情景，不是必须用款或清仓日期。"
}'''


def buyer_saved_json():
    payload = calendar_saved_json()
    payload.update(schema_version=7, mode="buyer_decision", policy_version="fund_buyer_brief_v1",
                   decision_brief=json.loads(BRIEF_SAVED_JSON),
                   sale_comparison={"rows": [{"label": "原保存费档日", "sale_date": "2026-10-01",
                       "redemption_fee": 0.005, "fee_saving": 100, "break_even_change": -0.01005,
                       "reasons": ["原卖出时点费用边界，不是必须等待"]}]})
    return payload


def tree_text(at):
    values = []

    def visit(node):
        if getattr(node, "type", "") in {
            "markdown", "caption", "info", "warning", "error", "subheader", "metric", "json",
        }:
            values.append(str(node.value))
        if getattr(node, "type", "") == "dataframe":
            values.append(str(node.value.to_dict("records")))
        for child in getattr(node, "children", {}).values():
            visit(child)

    visit(at.main)
    return "\n".join(values).replace("\\", "")


def history_entry(root, subject):
    from rmbfund.portfolio import Portfolio
    from rmbfund.ui.history import render_history

    repository = Portfolio(root)
    render_history(None, read=None, subject=subject,
                   assessments=repository.assessment_history,
                   assessment_detail=repository.assessment_detail)


def store_snapshots(root, monkeypatch, *payloads):
    repository = Portfolio(root)
    times = iter(f"2026-10-{index:02d}T10:00:00+08:00" for index in range(1, len(payloads) + 1))
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: next(times))
    envelopes = [repository.save_assessment(payload) for payload in payloads]
    for envelope, payload in zip(envelopes, payloads):
        assert repository.assessment_detail(envelope["id"])["assessment"] == payload
    return repository, envelopes


def forbid_recalculation(monkeypatch):
    import httpx

    from rmbfund import decision_brief, domain, forecast, forecast_decision, personal_assessment, service, timing_scenario, trade_calendar
    from rmbfund import portfolio
    from rmbfund.service import Workspace
    from rmbfund.ui import trade

    def forbidden(*args, **kwargs):
        pytest.fail("saved history must not compute, fetch, issue, score, or write")

    for module, names in (
        (decision_brief, ("build_decision_brief", "load_decision_rationale")),
        (forecast, ("fit_forecast", "fit_runtime_forecast")),
        (forecast_decision, ("build_decision",)),
        (personal_assessment, ("build_assessment",)),
        (timing_scenario, ("build_scenario", "resolve_fees", "summarize_history")),
        (Workspace, ("current_online", "update_forecast", "_score_forecasts", "calculate_timing")),
        (Portfolio, ("save_assessment", "save_fund", "save_research_preferences")),
        (httpx.Client, ("send",)),
        (httpx.AsyncClient, ("send",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for module, names in (
        (trade_calendar, ("build_trade_plan",)),
        (timing_scenario, ("resolve_plan_fees", "build_calendar_scenario", "build_timing_decision", "summarize_calendar_history")),
        (trade, ("build_scenario", "resolve_fees", "summarize_history", "build_trade_plan", "resolve_plan_fees", "build_calendar_scenario", "build_timing_decision", "summarize_calendar_history", "_connection")),
        (Workspace, ("timing_inputs",)),
        (domain, ("now_iso",)),
        (portfolio, ("now_iso",)),
        (forecast, ("now_iso",)),
        (personal_assessment, ("now_iso",)),
        (service, ("now_iso",)),
        (trade, ("now_iso",)),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def test_schema5_real_json_round_trip_reuses_saved_renderers_and_original_provenance(tmp_path, monkeypatch):
    from rmbfund.ui import buyer, trade

    payload = saved_json()
    subject = deepcopy(payload["subject"])
    repository, (envelope,) = store_snapshots(tmp_path, monkeypatch, payload)
    original = deepcopy(payload)
    rendered_scenarios, rendered_histories, rendered_buyers = [], [], []
    for name, target, calls in (("render_scenario", trade, rendered_scenarios),
                                ("render_reconstructed", trade, rendered_histories),
                                ("render_saved_buyer", buyer, rendered_buyers)):
        render = getattr(target, name)

        def replay(value, *, _render=render, _calls=calls):
            _calls.append(deepcopy(value))
            return _render(value)

        monkeypatch.setattr(target, name, replay)
    forbid_recalculation(monkeypatch)
    # Today's mutable caller inputs are not the repository's saved JSON.
    payload["trade_scenario"]["purchase"]["date"] = "2026-10-28"
    payload["trade_scenario"]["fees"]["redemption_fee"] = 0.99
    payload["online_context"] = {"messages": [{"title": "今天的消息不得回填"}]}
    at = AppTest.from_function(history_entry, args=(str(tmp_path), subject), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "原标准新买判断" in shown and "原标准持有判断" in shown
    assert "兼容旧字段不是标准交易" not in shown and "999,999" not in shown
    assert "2026-09-01" in shown and "2026-09-24" in shown
    assert "原观察期限 5笔净值" in shown and "原执行延迟 2笔净值" in shown
    assert "原标准费用假设：申购费 0.15% · 赎回费 0.50%" in shown
    assert "原示例申购费，不是支付宝实际费" in shown and "原自然日模拟口径" in shown
    assert "本地假设测算，未实际发行" in shown and "本地测算，未实际发行" in shown
    assert "原计算时间 2026-09-25 10:15" in shown
    assert "当年真实发行" in shown and "当时保存的在线依据" in shown
    assert "保存的研究事实" in shown and "保存的原始在线摘录" in shown
    assert "今天的消息不得回填" not in shown
    assert "保存的独立条件分析" in shown and "125.67 CNY" in shown
    assert "2026-06-01" in shown and "-4.7%" in shown
    action_table = next(table.value for table in at.dataframe if "做法" in table.value.columns)
    assert action_table["预期期末账面值"].tolist() == [
        "10,301.55", "10,000.00", "20,317.00", "20,618.55", "20,133.50", "19,950.00",
    ]
    at.run()
    assert not at.exception
    assert rendered_scenarios == [original["trade_scenario"]] * 2
    assert rendered_histories == [original["reconstructed_history"]] * 2
    assert rendered_buyers == [original] * 2
    assert repository.assessment_detail(envelope["id"])["assessment"] == original


def test_two_schema5_snapshots_compare_only_saved_scenario_and_research_json(tmp_path, monkeypatch):
    old, new = saved_json(), saved_json()
    new["trade_scenario"]["purchase"]["date"] = "2026-09-08"
    new["trade_scenario"]["horizon"] = 1
    new["trade_scenario"]["entry_lag"] = 3
    new["trade_scenario"]["fees"]["redemption_fee"] = 0.0125
    new["trade_scenario"]["decision"]["branches"]["held"] = {
        "action": "sell", "label": "新保存标准退出", "reasons": ["新保存交易理由"],
    }
    new["research_thesis"] = {"record_id": "research-new", "payload": {
        "research_date": "2026-09-27", "revision": {"supersedes": "research-original"},
        "change_evaluation": {"summary": "保存的研究明确接续说明"},
    }}
    new["inputs"]["redemption_fee"] = "0.77"  # Not the standard scenario's fee.
    repository, (old_record, new_record) = store_snapshots(tmp_path, monkeypatch, old, new)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), old["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "2026-09-01 → 2026-09-08" in shown
    assert "观察期限：5笔净值 → 1笔净值" in shown
    assert "假定执行延迟：2笔净值 → 3笔净值" in shown
    assert "标准假设赎回费：0.50% → 1.25%" in shown
    assert "88.00% → 77.00%" not in shown
    assert "原标准持有判断 → 新保存标准退出" in shown and "新保存交易理由" in shown
    assert "research-original → research-new" in shown and "保存的研究明确接续说明" in shown
    at.selectbox(key="history_assessment_selected").set_value(old_record["id"]).run()
    assert not at.exception and "没有更早的保存判断" in tree_text(at)
    assert "新保存交易理由" not in tree_text(at)
    assert repository.assessment_detail(new_record["id"])["assessment"] == new


@pytest.mark.parametrize("schema", [3, 4])
def test_old_to_schema5_never_turns_missing_standard_scenario_into_defaults(tmp_path, monkeypatch, schema):
    old = {
        "schema_version": schema, "policy_version": f"saved-policy-{schema}",
        "subject": saved_json()["subject"], "holding_status": "held", "horizon": 20,
        "forecast": {}, "decision": {"branches": {"held": {
            "action": "hold", "label": "旧版短期保存标签", "reasons": ["旧版短期原理由"],
        }}},
        "investor_decision": {"branches": {"held": {
            "action": "hold", "label": "旧版研究保存标签", "reasons": ["旧版研究原理由"],
        }}},
        "inputs": {"redemption_fee": "0", "subscription_fee": "0"},
        "research_thesis": {"record_id": "legacy-research", "payload": {"research_date": "2026-08-01"}},
    }
    new = saved_json()
    _, (old_record, _) = store_snapshots(tmp_path, monkeypatch, old, new)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), old["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert f"保存口径：schema{schema} → schema5" in shown
    assert "旧版个人判断没有保存标准交易情景" in shown
    assert "0.00% → 0.50%" not in shown and "观察期限：20笔净值 → 5笔净值" not in shown
    assert "legacy-research → research-original" in shown
    at.selectbox(key="history_assessment_selected").set_value(old_record["id"]).run()
    assert not at.exception and not at.error
    old_shown = tree_text(at)
    assert "旧版短期原理由" in old_shown and "当时保存的判断" in old_shown
    if schema == 4:
        assert "旧版研究保存标签" in old_shown and "旧版研究原理由" in old_shown
    assert "当时保存的标准交易假设与判断" not in old_shown


def test_schema5_missing_objects_and_actual_issue_provenance_are_not_guessed(tmp_path, monkeypatch):
    payload = saved_json()
    payload.update(trade_scenario=None, reconstructed_history=None, personal_conversion=None)
    payload["calculation"] = {"kind": "issued_forecast", "actual_issue": True}
    store_snapshots(tmp_path, monkeypatch, payload)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), payload["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "未保存标准交易情景" in shown
    assert "原快照明确记录采用实际发行预测" in shown
    assert "当时未保存历史回算结果" in shown and "没有保存标准交易情景" in shown
    assert "兼容旧字段不是标准交易" not in shown
    assert not any("做法" in table.value.columns for table in at.dataframe)


def test_schema6_literal_json_replays_exact_saved_plan_fees_and_rendered_objects(tmp_path, monkeypatch):
    from rmbfund.ui import buyer, trade

    payload = calendar_saved_json()
    original = deepcopy(payload)
    subject = deepcopy(payload["subject"])
    repository, (envelope,) = store_snapshots(tmp_path, monkeypatch, payload)
    calls = {name: [] for name in ("render_scenario", "render_reconstructed", "render_connection", "render_saved_buyer")}
    for name in calls:
        target = buyer if name == "render_saved_buyer" else trade
        render = getattr(target, name)

        def replay(value, *, _render=render, _calls=calls[name]):
            _calls.append(deepcopy(value))
            return _render(value)

        monkeypatch.setattr(target, name, replay)
    forbid_recalculation(monkeypatch)
    payload["trade_scenario"]["plan"].update(calendar_days=90, deadline="2099-12-31")
    payload["trade_scenario"]["fees"]["future_new_redemption_fee"] = 0.99
    payload["online_context"] = {"messages": [{"title": "今天消息不得改写自然日历史"}]}
    at = AppTest.from_function(history_entry, args=(str(tmp_path), subject), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    list_table = next(table.value for table in at.dataframe if "当时判断" in table.value.columns)
    label = list_table.iloc[0]["当时判断"]
    assert "未来12天" in label and "截止2026-10-07" in label
    assert all(text in label for text in ("原自然日新买判断", "原自然日持有判断", "原自然日追加独立判断"))
    assert "原自然日窗口 12天" in shown and "原日程预计 8笔净值" in shown
    assert "原本次申请日 2026-09-25" in shown and "原计划卖出申请日 2026-10-07" in shown
    assert "原新份额确认日 2026-09-29" in shown
    assert "旧份额当前赎回费 1.50%" in shown and "旧份额计划卖出赎回费 0.50%" in shown
    assert "新份额计划卖出赎回费 1.25%" in shown
    assert "原旧份额模拟确认日起点 2026-09-21" in shown
    assert "原自然日示例申购费，非支付宝实际费" in shown
    assert "原手动新份额计划卖出费，不覆盖旧份额" in shown
    assert "原新份额独立确认日起点模拟口径" in shown
    assert "原假设当前费表未来不变" in shown
    assert "含赎回应收款" in shown and "不是严格自然日择时策略" in shown
    assert "本地假设测算，未实际发行" in shown and "本地测算，未实际发行" in shown
    assert "原计算时间 2026-09-25 10:16" in shown
    assert "原保存的自然日计划与研究对照，不重生成" in shown
    assert "原自然日研究事实" in shown and "原自然日在线摘录" in shown
    assert "原自然日独立条件分析" in shown and "原可选个人净额换算" in shown
    assert "今天消息不得改写自然日历史" not in shown and "2099-12-31" not in shown
    assert "兼容字段不得代替自然日交易" not in list_table.to_string()
    assert "999,999" not in shown
    action_table = next(table.value for table in at.dataframe if "做法" in table.value.columns)
    value_column = next(column for column in action_table.columns if "预期" in column and "净额" in column)
    assert action_table[value_column].tolist() == [
        "10,175.03", "10,000.00", "20,265.42", "20,440.45", "20,057.71", "19,850.00",
    ]
    full_saved = [json.loads(item.value) for item in at.json]
    assert original in full_saved
    at.run()
    assert not at.exception and not at.error
    assert calls["render_scenario"] == [original["trade_scenario"], original["personal_conversion"]["trade_scenario"]] * 2
    assert calls["render_reconstructed"] == [original["reconstructed_history"]] * 2
    assert calls["render_connection"] == [original["trade_research_connection"]] * 2
    assert calls["render_saved_buyer"] == [original] * 2
    assert repository.assessment_detail(envelope["id"])["assessment"] == original


def test_two_schema6_snapshots_compare_saved_calendar_mapping_batches_and_contract(tmp_path, monkeypatch):
    old, new = calendar_saved_json(), calendar_saved_json()
    scenario = new["trade_scenario"]
    scenario["plan"].update(calendar_days=18, deadline="2026-10-13", application_date="2026-09-28",
                          sale_date="2026-10-13", new_confirmation_date="2026-09-30", horizon=11,
                          entry_lag=3, calendar_basis="saved_verified_calendar")
    scenario.update(horizon=11, entry_lag=3, end_value_basis="saved_different_basis")
    scenario["fees"].update(current_redemption_fee=0.01, future_old_redemption_fee=0,
                            future_new_redemption_fee=0.005, old_confirmation_date="2026-09-20",
                            new_confirmation_date="2026-09-30")
    scenario["fees"]["contexts"]["future_new"]["holding_days"] = 13
    scenario["decision"]["primary_head"].update(horizon=11, entry_lag=3, up_probability=0.61)
    scenario["decision"]["branches"]["addition"] = {
        "action": "add", "label": "新保存自然日追加", "reasons": ["新保存的追加净额理由"],
    }
    new["calculation"]["contract"]["id"] = "calendar-local-revised"
    new["research_thesis"] = {"record_id": "calendar-research-new", "payload": {
        "research_date": "2026-09-27", "revision": {"supersedes": "calendar-research-original"},
        "change_evaluation": {"summary": "原保存的自然日研究接续说明"},
    }}
    repository, (old_record, new_record) = store_snapshots(tmp_path, monkeypatch, old, new)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), old["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "自然日窗口：12天 → 18天" in shown
    assert "计划截止日：2026-10-07 → 2026-10-13" in shown
    assert "本次假定申请日：2026-09-25 → 2026-09-28" in shown
    assert "日程映射的净值延迟：2笔净值 → 3笔净值" in shown
    assert "日程映射的预测期限：8笔净值 → 11笔净值" in shown
    assert "旧份额模拟确认日起点：2026-09-21 → 2026-09-20" in shown
    assert "新份额模拟确认日起点：2026-09-29 → 2026-09-30" in shown
    assert "旧份额当前赎回费：1.50% → 1.00%" in shown
    assert "旧份额计划卖出赎回费：0.50% → 0.00%" in shown
    assert "新份额计划卖出赎回费：1.25% → 0.50%" in shown
    assert "分批费用上下文有变化" in shown and "金额比较口径：" in shown
    assert "准确预测头有变化" in shown and "本地计算合同有变化" in shown
    assert "原自然日追加独立判断 → 新保存自然日追加" in shown
    assert "新保存的追加净额理由" in shown and "原保存的自然日研究接续说明" in shown
    at.selectbox(key="history_assessment_selected").set_value(old_record["id"]).run()
    assert not at.exception and "没有更早的保存判断" in tree_text(at)
    assert "新保存的追加净额理由" not in tree_text(at)
    assert repository.assessment_detail(new_record["id"])["assessment"] == new


@pytest.mark.parametrize("field", ["fee_context", "plan_basis", "head", "contract", "value_basis"])
def test_schema6_metadata_changes_are_visible_even_when_rates_and_labels_do_not_change(monkeypatch, field):
    def entry(current, prior):
        from rmbfund.ui.history import _snapshot_changes

        _snapshot_changes(current, prior)

    old, new = calendar_saved_json(), calendar_saved_json()
    if field == "fee_context":
        new["trade_scenario"]["fees"]["contexts"]["future_old"]["redemption"]["source"] = "override"
        expected = "分批费用上下文有变化"
    elif field == "plan_basis":
        new["trade_scenario"]["plan"]["calendar_basis"] = "another_saved_basis"
        expected = "日程依据或保存假设有变化"
    elif field == "head":
        new["trade_scenario"]["decision"]["primary_head"]["expected_change"] = 0.019
        expected = "准确预测头有变化"
    elif field == "contract":
        new["calculation"]["contract"]["id"] = "different-saved-contract"
        expected = "本地计算合同有变化"
    else:
        new["trade_scenario"]["end_value_basis"] = "different-saved-basis"
        expected = "金额比较口径："
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(entry, args=(
        {"saved_at": "2026-10-02T10:00:00+08:00", "assessment": new},
        {"saved_at": "2026-10-01T10:00:00+08:00", "assessment": old},
    )).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert expected in shown and "没有上述可见变化" not in shown


@pytest.mark.parametrize("schema", [3, 4, 5])
def test_old_to_schema6_marks_changed_basis_without_inventing_days_or_future_fees(tmp_path, monkeypatch, schema):
    if schema == 5:
        old = saved_json()
    else:
        old = {"schema_version": schema, "policy_version": f"legacy-{schema}",
               "subject": saved_json()["subject"], "holding_status": "held", "horizon": 20,
               "forecast": {}, "decision": {"branches": {"held": {
                   "action": "hold", "label": "原旧版个人判断", "reasons": ["原旧版个人理由"],
               }}}, "inputs": {"subscription_fee": "0", "redemption_fee": "0"}}
    new = calendar_saved_json()
    repository, (old_record, _) = store_snapshots(tmp_path, monkeypatch, old, new)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), old["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "比较口径变化：" in shown and f"schema{schema}" in shown and "schema6" in shown
    assert "旧版未保存的自然日天数、截止日、确认日及未来分批赎回费均为未记录" in shown
    assert "自然日窗口：30天" not in shown
    assert "旧份额计划卖出赎回费：0.00%" not in shown
    assert "新份额计划卖出赎回费：0.00%" not in shown
    assert "观察期限：20笔净值 → 8笔净值" not in shown
    at.selectbox(key="history_assessment_selected").set_value(old_record["id"]).run()
    assert not at.exception and not at.error
    legacy_shown = tree_text(at)
    if schema == 5:
        assert "原观察期限 5笔净值" in legacy_shown
        assert "不假设届时一定赎回" in legacy_shown
        assert any("预期期末账面值" in table.value.columns for table in at.dataframe)
        assert "原自然日窗口" not in legacy_shown
    else:
        assert "原旧版个人理由" in legacy_shown
        assert "当时保存的自然日计划与卖出净额判断" not in legacy_shown
    assert repository.assessment_detail(old_record["id"])["assessment"] == old


def test_schema6_missing_saved_objects_never_falls_through_to_legacy_or_current_defaults(tmp_path, monkeypatch):
    payload = calendar_saved_json()
    payload.update(trade_scenario=None, reconstructed_history=None, personal_conversion=None,
                   trade_research_connection=None)
    payload["calculation"] = {"kind": "issued_forecast", "actual_issue": True}
    store_snapshots(tmp_path, monkeypatch, payload)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), payload["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "未保存自然日交易情景" in shown and "原快照明确记录采用实际发行预测" in shown
    assert "当时未保存历史回算结果" in shown and "不从兼容旧版字段或今天日期重建" in shown
    assert "原自然日窗口 30天" not in shown and "原分批费用假设" not in shown
    assert "当时的短期费用比较" not in shown
    assert not any("做法" in table.value.columns for table in at.dataframe)


def test_schema7_replays_saved_brief_before_exact_original_scenario_and_evidence(tmp_path, monkeypatch):
    from rmbfund.ui import buyer, trade

    payload = buyer_saved_json()
    original = deepcopy(payload)
    repository, (envelope,) = store_snapshots(tmp_path, monkeypatch, payload)
    calls = []
    brief_renderer = buyer.render_decision_brief

    def render_brief(value, *, replay=False):
        calls.append(("brief", deepcopy(value), replay))
        return brief_renderer(value, replay=replay)

    monkeypatch.setattr(buyer, "render_decision_brief", render_brief)

    def no_duplicate_investor(*args, **kwargs):
        pytest.fail("schema7 replays the brief, not another old investor action card")

    monkeypatch.setattr(buyer, "render_investor", no_duplicate_investor)
    for name in ("render_sale_comparison", "render_scenario", "render_reconstructed", "render_connection"):
        renderer = getattr(trade, name)

        def render(value, *, _name=name, _renderer=renderer):
            calls.append((_name, deepcopy(value)))
            return _renderer(value)

        monkeypatch.setattr(trade, name, render)
    forbid_recalculation(monkeypatch)
    payload["decision_brief"]["cases"]["flat"]["label"] = "今天的新首选不得重写"
    payload["decision_brief"]["rationale"]["generated_at"] = "2099-12-31T10:00:00+08:00"
    payload["trade_scenario"]["plan"]["calendar_days"] = 90
    at = AppTest.from_function(history_entry, args=(str(tmp_path), original["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    table = next(table.value for table in at.dataframe if "当时判断" in table.value.columns)
    assert "基金买卖首选 · 未来12天" in table.iloc[0]["当时判断"]
    assert "原首选先留现金" in table.iloc[0]["当时判断"]
    assert "原自然日新买判断" not in table.iloc[0]["当时判断"]
    assert "本地测算，未实际发行" in table.iloc[0]["原计算或发行时间"]
    assert shown.index("当时的首选与取舍") < shown.index("当时保存的自然日计划与卖出净额判断")
    assert "原首选：价格尚未补偿新增资金风险" in shown
    assert "原备选取舍" in shown and "175.03" in shown
    assert "原最强反方" in shown and "原观察盈利是否兑现" in shown
    assert "原作者裁决" in shown and "2026-09-25T08:30:00+08:00" in shown
    assert "原自然日研究事实" in shown and "原自然日在线摘录" in shown
    assert "原保存费档日" in shown and "原卖出时点费用边界" in shown
    assert "今天的新首选不得重写" not in shown and "2099-12-31" not in shown
    assert calls == [
        ("brief", original["decision_brief"], True),
        ("render_sale_comparison", original["sale_comparison"]),
        ("render_scenario", original["trade_scenario"]),
        ("render_reconstructed", original["reconstructed_history"]),
        ("render_connection", original["trade_research_connection"]),
        ("render_scenario", original["personal_conversion"]["trade_scenario"]),
    ]
    at.run()
    assert not at.exception and not at.error
    assert len(calls) == 12
    assert repository.assessment_detail(envelope["id"])["assessment"] == original


@pytest.mark.parametrize("schema", [3, 4, 5, 6])
def test_schema7_vs_old_compares_only_common_saved_fields_without_backfilling(tmp_path, monkeypatch, schema):
    from rmbfund.ui import buyer

    old = calendar_saved_json() if schema == 6 else saved_json() if schema == 5 else {
        "schema_version": schema, "policy_version": f"legacy-{schema}",
        "subject": saved_json()["subject"], "forecast": {},
        "holding_status": "held", "decision": {"branches": {"held": {
            "action": "hold", "label": "旧版个人持有原意见", "reasons": ["旧版个人原理由"],
        }}}, "inputs": {"subscription_fee": "0", "redemption_fee": "0"},
    }
    new = buyer_saved_json()
    _, (old_record, _) = store_snapshots(tmp_path, monkeypatch, old, new)
    brief_calls = []
    original_render = buyer.render_decision_brief

    def render(value, *, replay=False):
        brief_calls.append(deepcopy(value))
        return original_render(value, replay=replay)

    monkeypatch.setattr(buyer, "render_decision_brief", render)
    forbid_recalculation(monkeypatch)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), old["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert f"schema{schema}" in shown and "schema7" in shown
    assert "至少一份快照未保存当前首选摘要" in shown
    assert "自然日窗口：30天" not in shown
    assert "旧份额计划卖出赎回费：0.00%" not in shown
    assert "新份额计划卖出赎回费：0.00%" not in shown
    if schema < 6:
        assert "旧版未保存的自然日天数、截止日、确认日及未来分批赎回费均为未记录" in shown
    at.selectbox(key="history_assessment_selected").set_value(old_record["id"]).run()
    assert not at.exception and not at.error
    assert len(brief_calls) == 1
    assert "当时的首选与取舍" not in tree_text(at)
    if schema == 6:
        assert "原自然日窗口 12天" in tree_text(at)
    elif schema == 5:
        assert "原观察期限 5笔净值" in tree_text(at)
    else:
        assert "旧版个人原理由" in tree_text(at)


def test_schema7_missing_brief_and_calendar_objects_never_rebuilds_or_uses_legacy_choices(tmp_path, monkeypatch):
    from rmbfund.ui import buyer

    payload = buyer_saved_json()
    payload.update(decision_brief=None, trade_scenario=None, reconstructed_history=None, personal_conversion=None)
    store_snapshots(tmp_path, monkeypatch, payload)
    forbid_recalculation(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("a missing saved brief must not be rendered from another object")

    monkeypatch.setattr(buyer, "render_decision_brief", forbidden)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), payload["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "未保存基金买卖首选摘要" in shown
    assert "不从旧交易分支、作者说明或今天资料重建" in shown
    assert "没有保存自然日交易情景" in shown and "当时未保存历史回算结果" in shown
    assert "原首选先留现金" not in shown
    assert "原自然日窗口 30天" not in shown


def test_pure_changes_separate_simulation_inputs_actual_head_and_authored_brief():
    from rmbfund.assessment_changes import extract_assessment_changes

    old, new = buyer_saved_json(), buyer_saved_json()
    assert extract_assessment_changes(new, old) == []
    new["trade_scenario"]["plan"]["calendar_days"] = 18
    new["trade_scenario"]["fees"].update(future_new_redemption_fee=0.005, new_confirmation_date="2026-09-30")
    new["trade_scenario"]["decision"]["primary_head"]["expected_change"] = -0.01
    new["inputs"]["budget"] = {"amount": "200", "currency": "CNY"}
    new["decision_brief"]["rationale"]["id"] = "saved-rationale-revised"
    new["decision_brief"]["cases"]["flat"].update(preferred_action="buy", label="新保存首选新买",
        reasons=["新作者理由，不是用模型均值替代研究"])
    originals = deepcopy((new, old))
    changes = extract_assessment_changes(new, old)
    fields = {item["field"]: item for item in changes}
    assert fields["trade_scenario.plan.calendar_days"]["group"] == "user_inputs"
    assert fields["trade_scenario.fees.future_new_redemption_fee"]["group"] == "user_inputs"
    assert fields["trade_scenario.fees.new_confirmation_date"]["group"] == "user_inputs"
    assert fields["inputs.budget"]["group"] == "user_inputs"
    assert fields["trade_scenario.decision.primary_head"]["group"] == "market_data"
    assert fields["decision_brief.rationale"]["group"] == "research"
    assert fields["decision_brief.cases.flat.preferred_action"]["before"] == "wait"
    assert fields["decision_brief.cases.flat.preferred_action"]["after"] == "buy"
    assert "原首选先留现金 → 新保存首选新买" in fields["decision_brief.cases.flat.preferred_action"]["text"]
    assert (new, old) == originals
    fields["decision_brief.rationale"]["after"]["id"] = "caller-mutated-delta"
    assert (new, old) == originals


def test_changed_window_head_is_not_labelled_same_window_market_change():
    from rmbfund.assessment_changes import extract_assessment_changes

    old, new = buyer_saved_json(), buyer_saved_json()
    new["trade_scenario"]["plan"].update(calendar_days=18, horizon=11)
    new["trade_scenario"]["decision"]["primary_head"].update(horizon=11, expected_change=-0.02)
    changes = extract_assessment_changes(new, old)
    head = next(item for item in changes if item["field"] == "trade_scenario.decision.primary_head")
    assert head["group"] == "user_inputs"
    assert "不同窗口的分布不当作同一窗口的行情改判" in head["text"]


@pytest.mark.parametrize("supersedes,summary,prior_id,explained", [
    ("calendar-research-original", "原经营理由减弱，原价格理由未变", "calendar-research-original", True),
    ("unrelated-record", "看似有解释但不是接续这次原判断", "calendar-research-original", False),
    ("calendar-research-original", None, "calendar-research-original", False),
    (None, "首次研究不是对缺失研究的接续", None, False),
])
def test_pure_research_explanation_requires_prior_record_and_explicit_successor(supersedes, summary, prior_id, explained):
    from rmbfund.assessment_changes import extract_assessment_changes

    old, new = buyer_saved_json(), buyer_saved_json()
    old["research_thesis"]["record_id"] = prior_id
    new["research_thesis"] = {"record_id": "new-research", "payload": {
        "research_date": "2026-09-27", "revision": {"supersedes": supersedes},
        "change_evaluation": {"summary": summary, "fact_ids": ["calendar-fact"]},
    }}
    changes = extract_assessment_changes(new, old)
    change = next(item for item in changes if item["field"] == "research_thesis.payload.change_evaluation")
    assert change["explained"] is explained
    assert ("本次研究对上一版的解释" in change["text"]) is explained
    if not explained:
        assert "不把版本不同自行解释成原理由已失效" in change["text"]


def test_online_fetch_title_and_research_source_clock_never_become_economic_changes():
    from rmbfund.assessment_changes import extract_assessment_changes

    old, new = buyer_saved_json(), buyer_saved_json()
    new["online_context"]["checked_at"] = "2026-10-29T10:00:00+08:00"
    new["online_context"]["messages_by_product"]["synthetic"][0]["title"] = "新标题不能当盈利改善"
    new["research_thesis"]["source"] = {"checked_at": "2026-10-29T10:00:00+08:00"}
    new["forecast"]["generated_at"] = "2026-10-29T10:00:00+08:00"
    changes = extract_assessment_changes(new, old)
    assert {item["group"] for item in changes} == {"provenance", "online_context"}
    assert not any(item.get("explained") for item in changes)
    assert any("新标题或尚未解释的正文" in item["text"] for item in changes)


def test_pure_changes_reuses_cited_economic_review_without_automatically_changing_action():
    from rmbfund.assessment_changes import extract_assessment_changes

    old, new = buyer_saved_json(), buyer_saved_json()
    original_record = {"id": "original-manager-record", "product_id": "synthetic", "family": "management",
        "source_url": "https://example.com/managers", "report_date": "2026-09-20", "available_at": "2026-09-21T10:00:00+08:00",
        "payload": {"managers": [{"name": "合成经理甲"}]}}
    incoming_record = {**deepcopy(original_record), "id": "incoming-manager-record", "report_date": "2026-09-26",
        "available_at": "2026-09-27T10:00:00+08:00", "payload": {"managers": [{"name": "合成经理乙"}]}}
    for snapshot in (old, new):
        snapshot["research_thesis"]["payload"].update(product={"id": "synthetic"}, citations=[{
            "id": "c-manager", "kind": "online_evidence", "record_ids": [original_record["id"]],
            "record_basis": [deepcopy(original_record)],
        }])
    old["online_context"]["records_by_product"] = {"synthetic": [original_record]}
    new["online_context"]["records_by_product"] = {"synthetic": [incoming_record]}
    original = deepcopy((new, old))
    changes = extract_assessment_changes(new, old)
    signal = next(item for item in changes if item["field"] == "online_context.review_signals")
    assert signal["group"] == "market_data"
    assert signal["before"] == "original-manager-record" and signal["after"] == "incoming-manager-record"
    assert "c-manager" in signal["text"] and "待解释的资料变化，不是自动改判" in signal["text"]
    assert not any(item.get("explained") for item in changes)
    assert not any(item["field"].endswith("preferred_action") for item in changes)
    assert (new, old) == original
    assert extract_assessment_changes(new, new) == []


def test_history_forwards_optional_notes_callbacks_only_for_schema7(monkeypatch):
    from rmbfund.ui import assessment_notes

    rows = [{"id": "buyer-saved", "saved_at": "2026-10-02T10:00:00+08:00", "assessment": buyer_saved_json()},
            {"id": "calendar-saved", "saved_at": "2026-10-01T10:00:00+08:00", "assessment": calendar_saved_json()}]
    calls = []

    def callback(*args, **kwargs):
        pytest.fail("history must forward optional operations, not invoke them itself")

    callbacks = {"action_notes": callback, "save_action_note": callback, "replay_assessment": callback}

    def render(saved, *, read=None, save=None, replay=None):
        calls.append((deepcopy(saved), read, save, replay))

    monkeypatch.setattr(assessment_notes, "render_assessment_notes", render)
    forbid_recalculation(monkeypatch)

    def entry(rows, callbacks):
        from rmbfund.ui.history import render_history

        render_history(None, read=None, subject=rows[0]["assessment"]["subject"],
            assessments=lambda *args, **kwargs: rows,
            assessment_detail=lambda record_id: next(row for row in rows if row["id"] == record_id),
            **callbacks)

    at = AppTest.from_function(entry, args=(rows, callbacks), default_timeout=20).run()
    assert not at.exception and not at.error
    assert calls == [(rows[0], callback, callback, callback)]
    at.selectbox(key="history_assessment_selected").set_value("calendar-saved").run()
    assert not at.exception and not at.error
    assert len(calls) == 1


def projected_buyer_saved_json():
    """Literal presentation intentionally disagrees with its raw compatibility fields."""
    payload = buyer_saved_json()
    payload.update(policy_version="fund_buyer_economic_advice_v3", selected_branch="held",
        decision_brief={"schema_version": 3, "cases": {
            "flat": {"label": "未展示的另一支"}, "held": {"label": "原始brief不再重投影"}}},
        buyer_view={"schema_version": 1, "kind": "buyer_view", "branch": "held", "status": "ready",
            "preferred_action": "hold", "label": "保存展示：继续持有，不追加", "advice_reason": "保存展示的经济原因。",
            "next_step": "保存展示的下次复核条件。", "execution_status": {"status": "not_required"},
            "data_date": "2026-09-23", "missing_fields": []},
        forecast_chart={"schema_version": 1, "kind": "buyer_forecast_chart", "status": "ready",
            "product": {"name": "合成基金", "symbol": "SYNTHETIC", "share_class": "C"},
            "data_date": "2026-09-23", "identity": {"model_id": "不得展示的内部模型"},
            "history": {"points": [{"date": "2026-09-01", "value": "1.0", "currency": "CNY"},
                                    {"date": "2026-09-23", "value": "1.25", "currency": "CNY"}],
                        "currency": "CNY", "unit": "unit_nav", "date_basis": "nav_date"},
            "forecast": {"application_date": "2026-09-25", "sale_date": "2026-10-07", "deadline": "2026-10-07",
                "horizon": 8, "entry_lag": 2, "unit": "simple_return", "date_basis": "conditional_plan_sale_date",
                "point": {"date": "2026-10-07", "expected_change": 0.0317, "lower": -0.025, "upper": 0.091}}})
    return payload


@pytest.mark.parametrize("missing", [False, True])
def test_brief3_history_uses_saved_presentation_only_and_never_legacy_fallback(tmp_path, monkeypatch, missing):
    from rmbfund import buyer_view
    from rmbfund.ui import buyer, buyer_view as view_ui, charts, trade

    payload = projected_buyer_saved_json()
    if missing:
        payload.pop("buyer_view")
        payload.pop("forecast_chart")
    original = deepcopy(payload)
    repository, (envelope,) = store_snapshots(tmp_path, monkeypatch, payload)
    rendered = []
    for name in ("render_buyer_view", "render_forecast_chart"):
        renderer = getattr(view_ui, name)

        def render(value, *, _name=name, _renderer=renderer, **kwargs):
            rendered.append((_name, deepcopy(value)))
            return _renderer(value, **kwargs)

        monkeypatch.setattr(view_ui, name, render)
    forbid_recalculation(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("brief3 replay must not project again or enter a legacy renderer")

    for module, names in ((buyer_view, ("build_buyer_view", "build_forecast_chart")),
                          (charts, ("build_forecast_chart",)),
                          (buyer, ("render_decision_brief", "render_saved_buyer")),
                          (trade, ("render_scenario", "render_sale_comparison"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    at = AppTest.from_function(history_entry, args=(str(tmp_path), payload["subject"]), default_timeout=20).run()
    assert not at.exception and not at.error
    shown = tree_text(at)
    assert "当时选择：已经持有" in shown
    assert "原始brief不再重投影" not in shown and "未展示的另一支" not in shown
    assert "原自然日新买判断" not in shown and "不得展示的内部模型" not in shown
    assert not at.json and not at.number_input and not at.text_input
    if missing:
        assert not rendered and "没有保存当时展示的建议" in shown and "没有保存图表" in shown
    else:
        assert rendered == [("render_buyer_view", original["buyer_view"]),
                            ("render_forecast_chart", original["forecast_chart"])]
        assert "保存展示的经济原因" in shown and "保存展示的下次复核条件" in shown
        assert len(at.get("plotly_chart")) == 1
    assert repository.assessment_detail(envelope["id"])["assessment"] == original


def test_schema2_forecast_chart_history_replays_saved_points_ranges_and_table(tmp_path, monkeypatch):
    from rmbfund import buyer_view
    from rmbfund.ui import buyer, buyer_view as view_ui, charts, trade

    payload = projected_buyer_saved_json()
    chart = payload["forecast_chart"]
    chart["schema_version"] = 2
    chart["forecast"].update(
        requested_horizons=[1, 2, 3],
        points_status="partial",
        points_unavailable_horizons=[2],
        points_missing_fields={"H=2": ["condition_date", "exact_head"]},
        points=[
            {"horizon": 1, "label": "H=1", "entry_lag": 2, "date": "2026-09-26",
             "status": "available", "expected_change": 0.01, "lower": -0.01, "upper": 0.03},
            {"horizon": 2, "label": "H=2", "entry_lag": 2, "date": None,
             "status": "unavailable", "expected_change": None, "lower": None, "upper": None,
             "reason": "保存时没有对应条件日期。"},
            {"horizon": 3, "label": "H=3", "entry_lag": 2, "date": "2026-09-29",
             "status": "partial", "expected_change": 0.03, "lower": None, "upper": None,
             "reason": "保存时只有预期涨跌。"},
        ],
    )
    original = deepcopy(payload)
    repository, (envelope,) = store_snapshots(tmp_path, monkeypatch, payload)
    rendered = []
    figures = []
    renderer = view_ui.render_forecast_chart
    figure_renderer = view_ui.forecast_figure

    def capture_figure(value, *, _renderer=figure_renderer, **kwargs):
        figure = _renderer(value, **kwargs)
        figures.append(figure)
        return figure

    def replay(value, *, _renderer=renderer, **kwargs):
        rendered.append(deepcopy(value))
        return _renderer(value, **kwargs)

    monkeypatch.setattr(view_ui, "forecast_figure", capture_figure)
    monkeypatch.setattr(view_ui, "render_forecast_chart", replay)
    forbid_recalculation(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("schema2 history must replay the saved chart without projecting it")

    for module, names in (
        (buyer_view, ("build_buyer_view", "build_forecast_chart")),
        (charts, ("build_forecast_chart",)),
        (buyer, ("render_decision_brief", "render_saved_buyer")),
        (trade, ("render_scenario", "render_sale_comparison")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)

    at = AppTest.from_function(
        history_entry, args=(str(tmp_path), payload["subject"]), default_timeout=20
    ).run()
    assert not at.exception and not at.error
    assert rendered == [original["forecast_chart"]]
    assert len(at.get("plotly_chart")) == 1
    htmls = [node.proto.body for node in at.get("html")]
    table_html = " ".join(htmls)
    assert all(value in table_html for value in (
        "H=1", "H=2", "H=3", "暂不可用", "+1.00%", "中间80%范围下限",
    ))
    assert len(figures) == 1
    figure = figures[0]
    marker = next(trace for trace in figure.data if trace.name == "各目标点预期涨跌")
    interval = next(trace for trace in figure.data if trace.name == "各目标点中间80%范围")
    assert list(marker.x) == ["H=1", "H=3"]
    assert list(interval.x) == ["H=1", "H=1"]
    assert list(figure.layout.xaxis2.categoryarray) == ["H=1", "H=2", "H=3"]
    assert repository.assessment_detail(envelope["id"])["assessment"] == original

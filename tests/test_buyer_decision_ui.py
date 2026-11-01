"""Current preferred actions and the explicit few-date cache through the app."""
from copy import deepcopy
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from rmbfund.domain import digest
from rmbfund.service import Workspace
from test_standard_trade_ui import TimingWorkspace, open_public, save_snapshot
from test_ui import PRODUCT, choose_branch, click, displayed, open_decision, open_tools, text


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    for target in ("test_ui.now_iso", "rmbfund.ui.trade.now_iso", "rmbfund.ui.app.now_iso"):
        monkeypatch.setattr(target, lambda: "2026-10-29T09:00:00+08:00")


class SaleWorkspace(TimingWorkspace):
    sale_timing_inputs = Workspace.sale_timing_inputs
    rebuild_sale_timing = staticmethod(Workspace.rebuild_sale_timing)
    calculate_sale_timing = Workspace.calculate_sale_timing

    def __init__(self, root):
        super().__init__(root)
        self.sale_fits = []

    def timing_inputs(self, *args, **kwargs):
        result = super().timing_inputs(*args, **kwargs)
        if result.get("trade_plan"):
            plan = result["trade_plan"]
            result["local_request_identity"] = {"product_id": PRODUCT["id"], "input_version": self.input_version,
                "entry_lags": [plan["entry_lag"], plan["entry_lag"] + 1, plan["entry_lag"] + 2]}
            result["cache_key"] = digest({**result["local_request_identity"], "horizon": plan["horizon"]})
            result["calculation"] = {"kind": "local_hypothetical", "actual_issue": False}
        return result

    def _calculate_prepared_timing(self, inputs):
        result = deepcopy(inputs)
        self.sale_fits.append((inputs["trade_plan"]["horizon"], inputs["trade_plan"]["entry_lag"]))
        current = deepcopy(self.prediction)
        h, lag = self.sale_fits[-1]
        for field in ("forecasts", "execution_forecasts"):
            template = next(row for row in current[field] if row["horizon"] == 20)
            current[field] = [{**template, "horizon": h, "entry_lag": selected}
                             for selected in ([0] if field == "forecasts" else [lag, lag + 1, lag + 2])]
        current["entry_assumptions"] = result["entry_assumptions"]
        current.pop("record_id", None)
        result.update(prediction=current, status="ready")
        return result


def test_explicit_sale_comparison_reuses_same_head_then_fee_edit_only_arithmetic(tmp_path):
    workspace = SaleWorkspace(tmp_path)
    at = open_public(workspace)
    at.date_input(key="trade_old_confirmation_date").set_value(date(2026, 10, 28)).run()
    click(at, "计算本周期判断")
    assert not at.exception and len(workspace.calculations) == 1 and workspace.sale_fits == []
    online = len(workspace.online_calls)
    assert not any(button.key == "trade_compare_sales" for button in at.button)
    open_tools(at)
    click(at, "比较这几个卖出时点")
    assert not at.exception and len(workspace.sale_fits) == 1  # S reused; only D is missing.
    assert "抵消这点节费" in text(at)
    open_decision(at)
    saved = save_snapshot(at, tmp_path)
    rows = saved["sale_comparison"]["rows"]
    assert len(rows) == 3 and all(row["expected_value"] is not None for row in rows)
    assert saved["decision_brief"] and saved["simulation_reference"]["branches"]
    at.text_input(key="trade_redemption_fee").input("2").run()
    at.text_input(key="trade_future_old_redemption_fee").input("0").run()
    assert len(workspace.sale_fits) == len(workspace.calculations) == 1
    open_tools(at)
    click(at, "比较这几个卖出时点")
    open_decision(at)
    assert len(workspace.sale_fits) == len(workspace.calculations) == 1
    assert len(workspace.online_calls) == online and not workspace.research_calls
    assert "和上次保存相比" in text(at)


def test_brief3_unverified_mean_does_not_replace_research_and_history_replays_display(tmp_path, monkeypatch):
    from rmbfund import buyer_view, decision_brief
    from rmbfund.portfolio import Portfolio
    from rmbfund.ui import charts

    workspace = SaleWorkspace(tmp_path)
    thesis = workspace.detail["research_thesis"]
    thesis["payload"]["action_cases"]["new_buy"]["action"] = "buy"
    thesis["payload"]["action_cases"]["addition"]["action"] = "add"
    at = open_public(workspace)
    choose_branch(at, "held")
    at.slider(key="trade_calendar_days").set_value(8).run()
    click(at, "计算本周期判断")
    original_advice = deepcopy(displayed(at)["buyer_view"])
    at.text_input(key="trade_future_new_redemption_fee").input("20").run()
    saved = save_snapshot(at, tmp_path)
    brief, case = saved["decision_brief"], saved["decision_brief"]["cases"]["held"]
    assert brief["schema_version"] == 3 and saved["policy_version"] == "fund_buyer_economic_advice_v3"
    assert case["preferred_action"] == "add" and saved["buyer_view"] == original_advice
    for field in ("preferred_action", "label", "advice_reason", "next_step", "execution_status"):
        assert saved["buyer_view"][field] == case[field]
    assert saved["selected_branch"] == "held"
    assert case["advice_reason"] in text(at) and case["next_step"] in text(at)
    assert not at.json and not any(table.type == "dataframe" for table in at)
    original = deepcopy(saved)
    calls = len(workspace.online_calls), len(workspace.calculations)
    thesis["payload"]["action_cases"]["held"]["action"] = "redeem"
    workspace.prediction["execution_forecasts"][0]["expected_change"] = -0.9

    def forbidden(*args, **kwargs):
        raise AssertionError("历史必须回放保存的展示，不重新build或加载今天作者说明")

    for module, names in ((decision_brief, ("build_decision_brief", "load_decision_rationale")),
                          (buyer_view, ("build_buyer_view", "build_forecast_chart")),
                          (charts, ("build_forecast_chart",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    at.session_state["fund_navigation"] = "历史记录"
    at.run()
    assert not at.exception and "当时保存的操作建议" in text(at)
    assert case["label"] in text(at) and case["advice_reason"] in text(at)
    assert Portfolio(tmp_path).assessment_history(PRODUCT)[0]["assessment"] == original
    assert calls == (len(workspace.online_calls), len(workspace.calculations))
    assert not workspace.research_calls


def direct_brief():
    case = {"preferred_action": "hold", "research_action": "add", "label": "继续持有，不追加",
            "comparison_action": "add", "advice_reason": "追加一笔的费用后结果不如先留现金。",
            "next_step": "等新增一笔的费用下降后，再考虑追加。",
            "reasons": ["原研究支持追加，完整经营理由仍然保留。", "原研究的补充风险依据。"],
            "fact_ids": ["f1"], "conflict": {"status": "opposed", "resolution": "双方都对的原分歧说明仅留详情。"},
            "why_not": [{"action": "exit", "label": "现在退出", "expected_difference_vs_preferred": 999,
                         "reason": "退出的完整备选理由。"},
                        {"action": "add", "label": "再追加一笔", "expected_difference_vs_preferred": -18,
                         "reason": "追加的完整备选理由。"}],
            "counterevidence": [{"reason": "最强反方的经营现金流依据。"}],
            "checks": [{"condition": "经营现金流转弱", "effect": "重新检查保留理由"}],
            "expected_value": 20200, "lower_value": 18000, "upper_value": 22400,
            "comparison_status": "ready", "execution_status": {"status": "open"}}
    flat = {**deepcopy(case), "preferred_action": "wait", "research_action": "buy", "label": "先留现金",
            "comparison_action": "buy", "advice_reason": "当前价格还没有充分补偿费用和下跌风险。",
            "next_step": "等价格与盈利预期重新匹配后，再考虑新买。",
            "why_not": [{"action": "buy", "label": "新买一笔", "expected_difference_vs_preferred": 36,
                         "reason": "新买的完整备选理由。"}]}
    return {"schema_version": 2, "policy_version": "fund_buyer_advice_v2", "research_date": "2026-10-28",
            "research_summary": "原研究总判断及其经营范围。", "cases": {"flat": flat, "held": case},
            "rationale": {"generated_at": "2026-10-28T12:34:56+08:00",
                          "scope": {"capital_task": "资金任务长文", "model_families": ["synthetic"]}},
            "plan_semantics": "所选日期仅用于同口径算账，并非强制卖出。",
            "amount_basis": "原金额范围和计算口径。", "origin": {"provider": "synthetic"}}


def render_brief_entry(brief, replay=False):
    from rmbfund.ui.buyer import render_decision_brief

    render_decision_brief(brief, replay=replay)


def main_card_text(node):
    """Read visible advice without counting the contents of its folded details."""
    if node.type == "expander":
        return ""
    if hasattr(node, "children"):
        return "\n".join(main_card_text(child) for child in node.children.values())
    return str(getattr(node, "value", "")).replace("\\", "")


@pytest.mark.parametrize("replay", [False, True])
def test_schema2_shows_one_action_reason_amount_and_next_step_per_case(replay):
    brief = direct_brief()
    original = deepcopy(brief)
    at = AppTest.from_function(render_brief_entry, args=(brief, replay), default_timeout=20).run()
    assert not at.exception and brief == original
    assert at.subheader[0].value == ("当时的操作建议" if replay else "操作建议")
    shown = main_card_text(at.main)
    assert "还没买：先留现金" in shown and "已经持有：继续持有，不追加" in shown
    for case in brief["cases"].values():
        assert case["advice_reason"] in shown and case["next_step"] in shown
    assert shown.count("按所选计划和同样本金测算") == 2
    assert "比新买一笔少 ¥36.00" in shown and "比再追加一笔多 ¥18.00" in shown
    assert "¥999.00" not in shown and "¥20,200.00" not in shown
    assert all(fragment not in shown for fragment in (
        "原研究", "双方都对", "资金任务长文", "最强反方", "80%金额范围", "2026-10-28T12:34:56",
        "卖一半仅为金额示例", "为什么不选其他做法",
    ))
    assert len(at.expander) == 1 and at.expander[0].proto.expanded is False
    details = text(at.expander[0])
    assert all(fragment in details for fragment in (
        "原研究支持追加", "双方都对", "资金任务长文", "最强反方", "80%金额范围", "¥999.00",
        "退出的完整备选理由", "经营现金流转弱", "原金额范围和计算口径",
    ))
    assert at.expander[0].json


def test_schema2_real_pause_is_explicit_but_missing_prediction_is_not_a_sale_ban():
    brief = direct_brief()
    flat, held = brief["cases"]["flat"], brief["cases"]["held"]
    flat["execution_status"] = {"status": "blocked", "operation": "subscription"}
    held.update(preferred_action=None, label="暂不操作，先补齐研究依据", comparison_action=None,
                advice_reason="暂时没有足够依据决定持有还是卖出。", next_step="先补齐研究依据再作判断。",
                comparison_status="unavailable", expected_value=None, lower_value=None, upper_value=None)
    held["why_not"] = [{"action": "exit", "label": "现在退出", "expected_difference_vs_preferred": None}]
    at = AppTest.from_function(render_brief_entry, args=(brief,), default_timeout=20).run()
    assert not at.exception
    shown = main_card_text(at.main)
    assert "当前暂停申购，暂不提交买入。" in shown
    assert "已经持有：暂不操作，先补齐研究依据" in shown
    assert "先补齐研究依据再作判断" in shown
    assert len(at.warning) == 1 and "暂停赎回" not in shown and "禁止卖出" not in shown
    assert "已经持有：继续持有" not in shown and "已经持有：现在退出" not in shown


def test_schema1_history_uses_original_renderer_and_ignores_new_advice_fields():
    brief = direct_brief()
    brief["schema_version"] = 1
    brief["cases"]["held"]["label"] = "旧快照原持有标签"
    at = AppTest.from_function(render_brief_entry, args=(brief, True), default_timeout=20).run()
    assert not at.exception and at.subheader[0].value == "当时的首选与取舍"
    shown = text(at)
    assert "旧快照原持有标签" in shown and "原研究支持追加" in shown
    assert "80%金额范围" in shown and "减仓比例没有被优化" in shown
    assert brief["cases"]["held"]["advice_reason"] not in shown
    assert "当时的操作建议" not in shown


def test_brief3_dispatches_one_selected_context_instead_of_schema1_renderer():
    brief = direct_brief()
    brief["schema_version"] = 3
    at = AppTest.from_function(render_brief_entry, args=(brief,), default_timeout=20).run()
    assert not at.exception
    shown = text(at)
    assert brief["cases"]["flat"]["advice_reason"] in shown
    assert brief["cases"]["held"]["advice_reason"] not in shown
    assert "当时的首选与取舍" not in shown and not at.json
    assert "¥36.00" not in shown and "最强反方" not in shown

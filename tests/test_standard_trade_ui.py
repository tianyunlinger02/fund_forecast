"""Calendar trade decisions need no personal portfolio or implicit calculation."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from rmbfund.portfolio import Portfolio
from test_ui import (FakeWorkspace, PRODUCT, choose_branch, click, displayed, generated_thesis,
                     open_decision, open_tools, run, text)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    current = ["2026-10-29T09:00:00+08:00"]
    for target in ("test_ui.now_iso", "rmbfund.ui.trade.now_iso", "rmbfund.ui.app.now_iso"):
        monkeypatch.setattr(target, lambda: current[0])
    return current


class TimingWorkspace(FakeWorkspace):
    def __init__(self, root):
        super().__init__(root)
        self.input_version = "synthetic-input-v1"
        self.series = [{"date": (date(2026, 9, 7) + timedelta(days=i)).isoformat(),
                        "nav": str(2 + i / 1000), "currency": "CNY"} for i in range(45)]
        self.prediction["anchor"].update(nav=self.series[-1]["nav"])
        self.detail["research_thesis"] = generated_thesis()
        self.detail["history"] = [{**row, "value": row["nav"]} for row in self.series]
        self.detail["background_records"] = [{"family": "fees", "id": "synthetic-fee",
            "source_url": "https://example.test/fee", "report_date": "2026-10-20",
            "observed_at": "2026-10-21T08:00:00+08:00", "available_at": "2026-10-21T08:00:00+08:00",
            "payload": {"subscription_rate": "0", "redemption_bands": [
                {"min_days": 0, "max_days": 6, "rate": "0.015"},
                {"min_days": 7, "max_days": None, "rate": "0"}],
                "holding_day_basis": "unknown", "channel_verified": False}}]


def open_public(workspace):
    at = run(workspace)
    at.text_input(key="fund_public_code").input("018147").run()
    return click(at, "公开预览")


def save_snapshot(at, root):
    click(at, "保存这次判断")
    assert not at.exception
    return Portfolio(root).assessment_history(PRODUCT)[0]["assessment"]


def test_code_decision_survives_unreadable_private_holdings_and_needs_no_money(tmp_path, monkeypatch):
    workspace = TimingWorkspace(tmp_path)
    monkeypatch.setattr(Portfolio, "list_funds", lambda self: (_ for _ in ()).throw(ValueError("synthetic unavailable")))
    at = open_public(workspace)
    assert not at.exception
    assert "新买需要更有吸引力的价格" in text(at)
    assert displayed(at)["selected_branch"] == "flat"
    assert not at.select_slider
    choose_branch(at, "held")
    assert "假设买入净值日" in [item.label for item in at.select_slider]
    choose_branch(at, "flat")
    assert at.slider(key="trade_calendar_days").value == 30
    assert "观察至 2026-11-28" in text(at)
    assert not any(item.key == "trade_personal_enabled" for item in at.toggle)
    assert not any(item.key == "forecast_budget" for item in at.text_input)
    assert displayed(at)["buyer_view"]["branch"] == "flat"
    assert not workspace.calculations and not workspace.updates and not workspace.research_calls
    click(at, "计算本周期判断")
    saved = save_snapshot(at, tmp_path)
    assert saved["schema_version"] == 7 and saved["mode"] == "buyer_decision"
    assert saved["policy_version"] == "fund_buyer_economic_advice_v3"
    assert saved["decision_brief"]["cases"]
    assert saved["inputs"]["holding"]["shares"] is None
    assert saved["trade_research_connection"]["lines"]
    scenario = saved["trade_scenario"]
    assert scenario["kind"] == "calendar_cash"
    assert {row["action"] for row in scenario["rows"]} == {"buy", "wait", "hold", "add", "reduce50", "exit"}
    assert scenario["fees"]["subscription_fee"] == 0
    assert scenario["fees"]["contexts"]["current"]["subscription"]["source"] == "public_standard"
    assert set(scenario["fees"]["contexts"]) == {"current", "future_old", "future_new"}
    assert scenario["end_value_basis"] == "cash_and_redemption_receivable"
    assert not workspace.updates and not workspace.research_calls


def test_explicit_local_compute_is_cached_for_same_head_date_fee_and_history_edits(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    at.slider(key="trade_calendar_days").set_value(8).run()  # Friday sale; H=6.
    assert not workspace.calculations
    click(at, "计算本周期判断")
    assert workspace.calculations == [(PRODUCT["id"], "nav_ridge_empirical_v2", False, 8)]
    point = deepcopy(displayed(at)["forecast_chart"]["forecast"]["point"])
    online = len(workspace.online_calls)
    at.slider(key="trade_calendar_days").set_value(9).run()  # Saturday cutoff, same Friday sale/H.
    assert displayed(at)["forecast_chart"]["forecast"]["point"] == point
    assert "观察至 2026-11-07" in text(at)
    assert "至 2026-11-06 的净值变化" in text(at)
    choose_branch(at, "held")
    at.select_slider(key="trade_purchase_date").set_value(workspace.series[0]["date"]).run()
    at.text_input(key="trade_redemption_fee").input("0").run()
    assert len(workspace.calculations) == 1 and len(workspace.online_calls) == online
    assert displayed(at)["calculation"]["kind"] == "local_hypothetical"
    open_tools(at)
    click(at, "运行历史检验")
    open_decision(at)
    assert workspace.calculations[-1] == (PRODUCT["id"], "nav_ridge_empirical_v2", True, 9)
    at.slider(key="trade_calendar_days").set_value(10).run()  # Sunday, still the same H/L.
    open_tools(at)
    click(at, "运行历史检验")
    assert "当时判断可以买" in text(at) and "后续未走完" in text(at)
    open_decision(at)
    assert len(workspace.calculations) == 2
    assert displayed(at)["reconstructed_history"]
    assert not workspace.updates and not workspace.research_calls
    assert len(workspace.online_calls) == online
    saved = save_snapshot(at, tmp_path)
    assert saved["calculation"]["actual_issue"] is False
    assert saved["forecast"].get("record_id") is None
    assert saved["trade_scenario"]["horizon"] == 6
    assert saved["trade_scenario"]["plan"]["calendar_days"] == 10
    assert saved["trade_scenario"]["plan"]["deadline"] == "2026-11-08"
    assert saved["reconstructed_history"]["horizon"] == 6
    assert saved["reconstructed_history"]["kind"] == "calendar_cash_fixed_fee_nav_history"
    original = deepcopy(saved)
    workspace.prediction["execution_forecasts"][0]["expected_change"] = -0.9
    at.session_state["fund_navigation"] = "历史记录"
    at.run()
    assert not at.exception and len(workspace.calculations) == 2
    assert Portfolio(tmp_path).assessment_history(PRODUCT)[0]["assessment"] == original


def test_new_day_head_waits_for_explicit_compute_instead_of_borrowing_cached_horizon(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    at.slider(key="trade_calendar_days").set_value(8).run()
    click(at, "计算本周期判断")
    online = len(workspace.online_calls)
    at.slider(key="trade_calendar_days").set_value(11).run()  # Monday sale now requires H=7.
    assert not at.exception and len(workspace.calculations) == 1
    assert not any(metric.label == "上涨概率" for metric in at.metric)
    assert displayed(at)["forecast_chart"]["forecast"]["point"] is None
    assert "新买需要更有吸引力的价格" in text(at)
    click(at, "计算本周期判断")
    assert workspace.calculations[-1] == (PRODUCT["id"], "nav_ridge_empirical_v2", False, 11)
    assert len(workspace.calculations) == 2 and len(workspace.online_calls) == online
    saved = save_snapshot(at, tmp_path)
    scenario = saved["trade_scenario"]
    assert scenario["horizon"] == 7
    assert scenario["decision"]["primary_head"]["horizon"] == 7
    assert scenario["entry_lag"] == scenario["decision"]["primary_head"]["entry_lag"]
    assert not workspace.updates and not workspace.research_calls


def test_bad_optional_amount_in_tools_does_not_change_current_advice_or_forecast(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    click(at, "计算本周期判断")
    original = deepcopy(displayed(at))
    open_tools(at)
    at.text_input(key="forecast_budget").input("not-money").run()
    assert not at.exception and "这里只影响个人换算" in text(at)
    open_decision(at)
    saved = save_snapshot(at, tmp_path)
    assert saved["forecast_chart"] == original["forecast_chart"]
    assert saved["buyer_view"] == original["buyer_view"]
    assert saved["personal_conversion"] is None and saved["inputs"]["budget"] is None
    assert len(workspace.calculations) == 1


def test_fee_age_and_deadline_use_actual_calculation_day_not_stale_nav_date(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    choose_branch(at, "held")
    at.select_slider(key="trade_purchase_date").set_value(workspace.series[-1]["date"]).run()
    scenario = save_snapshot(at, tmp_path)["trade_scenario"]
    assert scenario["purchase"]["latest_date"] == "2026-10-21"
    assert scenario["plan"]["decision_date"] == "2026-10-29"
    assert scenario["plan"]["deadline"] == "2026-11-28"
    assert scenario["plan"]["sale_date"] == "2026-11-27"
    current = scenario["fees"]["contexts"]["current"]
    assert current["valuation_date"] == "2026-10-29" and current["holding_days"] == 8
    assert scenario["fees"]["current_redemption_fee"] == 0
    assert scenario["fees"]["old_confirmation_date"] == "2026-10-21"
    assert scenario["fees"]["new_confirmation_date"] == "2026-11-02"
    assert not workspace.calculations


def test_future_batch_fee_changes_net_cash_and_fee_probability_not_market_head(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    at.slider(key="trade_calendar_days").set_value(8).run()
    click(at, "计算本周期判断")
    before = save_snapshot(at, tmp_path)["trade_scenario"]
    point = deepcopy(displayed(at)["forecast_chart"]["forecast"]["point"])
    online = len(workspace.online_calls)
    at.text_input(key="trade_future_new_redemption_fee").input("15").run()
    after = save_snapshot(at, tmp_path)["trade_scenario"]
    assert displayed(at)["forecast_chart"]["forecast"]["point"] == point
    assert after["market"] == before["market"]
    assert after["decision"]["primary_head"] == before["decision"]["primary_head"]
    before_rows = {row["action"]: row for row in before["rows"]}
    after_rows = {row["action"]: row for row in after["rows"]}
    assert after_rows["buy"]["expected_value"] < before_rows["buy"]["expected_value"]
    assert after_rows["add"]["expected_value"] < before_rows["add"]["expected_value"]
    assert after_rows["buy"]["better_probability"] < before_rows["buy"]["better_probability"]
    assert after_rows["hold"] == before_rows["hold"]
    assert after["fees"]["future_new_redemption_fee"] == 0.15
    assert after["fees"]["future_old_redemption_fee"] == before["fees"]["future_old_redemption_fee"]
    assert after["fees"]["current_redemption_fee"] == before["fees"]["current_redemption_fee"]
    assert after["fees"]["contexts"]["future_new"]["redemption"]["source"] == "override"
    assert len(workspace.calculations) == 1 and len(workspace.online_calls) == online
    assert not workspace.updates and not workspace.research_calls


def test_new_confirmation_after_sale_only_removes_actions_using_new_shares(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    at.slider(key="trade_calendar_days").set_value(8).run()
    click(at, "计算本周期判断")
    point = deepcopy(displayed(at)["forecast_chart"]["forecast"]["point"])
    online = len(workspace.online_calls)
    at.date_input(key="trade_new_confirmation_date").set_value(date(2026, 11, 9)).run()
    scenario = save_snapshot(at, tmp_path)["trade_scenario"]
    assert displayed(at)["forecast_chart"]["forecast"]["point"] == point
    assert scenario["plan"]["sale_date"] == "2026-11-06"
    assert scenario["plan"]["new_shares_sellable"] is False
    rows = {row["action"]: row for row in scenario["rows"]}
    for action in ("buy", "add"):
        assert rows[action]["status"] == "unavailable" and rows[action]["expected_value"] is None
    for action in ("hold", "reduce50", "exit"):
        assert rows[action]["status"] == "ready" and rows[action]["expected_value"] is not None
    assert scenario["decision"]["branches"]["flat"]["action"] == "unavailable"
    assert scenario["decision"]["branches"]["held"]["action"] != "unavailable"
    assert scenario["fees"]["contexts"]["future_new"]["status"] == "confirmation_after_application"
    assert len(workspace.calculations) == 1 and len(workspace.online_calls) == online
    assert not workspace.updates and not workspace.research_calls


def test_adjusting_confirmation_into_window_removes_obsolete_plan_reason(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    at.slider(key="trade_calendar_days").set_value(1).run()
    click(at, "计算本周期判断")
    before = save_snapshot(at, tmp_path)["trade_scenario"]
    assert before["plan"]["sale_date"] == "2026-10-30"
    assert not before["plan"]["new_shares_sellable"]
    obsolete = "新份额在计划卖出日前尚未假定确认"
    assert any(obsolete in reason for reason in before["plan"]["reasons"])
    at.date_input(key="trade_new_confirmation_date").set_value(date(2026, 10, 29)).run()
    after = save_snapshot(at, tmp_path)["trade_scenario"]
    assert after["plan"]["new_shares_sellable"]
    assert not any(obsolete in reason for reason in after["plan"]["reasons"])
    assert after["plan"]["confirmation_basis"] == "manual_date_scenario"
    assert before["plan"]["confirmation_label"] not in after["plan"]["assumptions"]
    assert all(row["status"] == "ready" for row in after["rows"] if row["action"] in {"buy", "add"})
    assert len(workspace.calculations) == 1


def test_new_input_invalidates_ram_cache_without_implicitly_fitting(tmp_path):
    workspace = TimingWorkspace(tmp_path)
    at = open_public(workspace)
    click(at, "计算本周期判断")
    workspace.input_version = "synthetic-input-v2"
    workspace.series[-1]["nav"] = "3"
    workspace.prediction["anchor"]["nav"] = "3"
    at.run()
    assert len(workspace.calculations) == 1
    assert displayed(at)["forecast_chart"]["forecast"]["point"] is None
    assert not any(metric.label == "上涨概率" for metric in at.metric)
    click(at, "计算本周期判断")
    assert not at.exception and len(workspace.calculations) == 2


def test_empty_nav_still_displays_research_and_can_save(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    workspace.detail["history"] = []
    workspace.detail["latest"] = None
    workspace.detail["research_thesis"] = generated_thesis()
    workspace.prediction = None
    at = open_public(workspace)
    assert not at.exception and "新买需要更有吸引力的价格" in text(at)
    choose_branch(at, "held")
    assert "还没有可选的实际净值日" in text(at)
    saved = save_snapshot(at, tmp_path)
    scenario = saved["trade_scenario"]
    assert saved["schema_version"] == 7 and scenario["purchase"]["date"] is None
    assert scenario["decision"]["primary_head"] is None
    assert scenario["market"]["up_probability"] is None
    assert all(row["expected_value"] is None for row in scenario["rows"]
               if row["action"] in {"buy", "hold", "add", "reduce50"})
    assert not workspace.calculations and not workspace.updates and not workspace.research_calls


def test_schema7_save_callback_receives_exact_displayed_snapshot_before_date_or_input_refresh(tmp_path, monkeypatch, fixed_clock):
    from rmbfund.ui import decision as decision_ui

    workspace = TimingWorkspace(tmp_path)
    built, received = [], []
    build, save = decision_ui.build_assessment, Portfolio.save_assessment

    def track_build(*args, **kwargs):
        value = build(*args, **kwargs)
        built.append(value)
        return value

    def track_save(repository, assessment, **kwargs):
        assert assessment is built[-1]
        received.append(deepcopy(assessment))
        return save(repository, assessment, **kwargs)

    monkeypatch.setattr(decision_ui, "build_assessment", track_build)
    monkeypatch.setattr(Portfolio, "save_assessment", track_save)
    at = open_public(workspace)
    click(at, "计算本周期判断")
    displayed = deepcopy(built[-1])
    assert displayed["schema_version"] == 7 and displayed["mode"] == "buyer_decision"
    assert displayed["trade_scenario"]["decision"]["primary_head"]
    assert not Portfolio(tmp_path).assessment_history(PRODUCT)
    fixed_clock[0] = "2026-10-30T16:00:00+08:00"
    workspace.input_version = "next-not-yet-displayed-input"
    workspace.prediction["execution_forecasts"][6]["expected_change"] = -0.9
    stored = save_snapshot(at, tmp_path)
    assert received == [displayed] and stored == displayed
    assert stored["trade_scenario"]["plan"]["decision_date"] == "2026-10-29"
    assert stored["trade_scenario"]["plan"]["deadline"] == "2026-11-28"
    assert len(workspace.calculations) == 1 and not workspace.updates and not workspace.research_calls

"""Pure final-brief projections, using synthetic assessments only."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from rmbfund.buyer_view import build_buyer_view


def assessment():
    return {
        "schema_version": 7,
        "anchor": {"basis": {"date": "2026-10-26", "nav": "5.4", "currency": "CNY"}},
        "forecast": {"generated_at": "2099-01-01T12:00:00+08:00", "model_id": "SECRET_MODEL"},
        "decision": {"branches": {"flat": {"action": "buy"}, "held": {"action": "sell"}}},
        "trade_scenario": {"rows": [{"action": "buy", "expected_value": 9999999}]},
        "decision_brief": {
            "schema_version": 3, "data_date": "2026-10-27",
            "cases": {
                "flat": {"preferred_action": "wait", "label": "先保留现金",
                         "advice_reason": "当前价格还没有补足新增资金的风险。",
                         "next_step": "盈利或价格变化后再核对。",
                         "execution_status": {"status": "not_required", "reasons": ["internal"]},
                         "decision_basis": {"model_id": "SECRET_MODEL"},
                         "forecast_evidence": {"expected_value": 9999999}},
                "held": {"preferred_action": "hold", "label": "保留已有份额，先不追加",
                         "advice_reason": "保留理由成立，新增资金缺少价格余量。",
                         "next_step": "复核盈利兑现情况。",
                         "execution_status": {"status": "unknown", "operation": "subscription"}},
            },
        },
    }


@pytest.mark.parametrize("branch,action", [("flat", "wait"), ("held", "hold")])
def test_projects_final_case_not_quantitative_choice_and_cannot_rewrite_action(branch, action):
    source = assessment()
    before = deepcopy(source)
    view = build_buyer_view(source, branch=branch)
    assert view["status"] == "ready"
    assert view["preferred_action"] == action
    for field in ("preferred_action", "label", "advice_reason", "next_step", "execution_status"):
        assert view[field] == source["decision_brief"]["cases"][branch][field]
    assert view["data_date"] == "2026-10-27"
    assert view["branch"] == branch
    assert source == before
    assert "decision_basis" not in view and "forecast_evidence" not in view
    assert "SECRET_MODEL" not in json.dumps(view)
    view["execution_status"]["status"] = "mutated"
    assert source == before


def test_unresolved_preferred_action_is_authoritative_not_default_wait_or_hold():
    source = assessment()
    source["decision_brief"]["cases"]["flat"]["preferred_action"] = None
    view = build_buyer_view(source)
    assert view["preferred_action"] is None
    assert view["status"] == "ready"
    source["decision_brief"]["cases"]["flat"].pop("preferred_action")
    view = build_buyer_view(source)
    assert view["preferred_action"] is None and "preferred_action" in view["missing_fields"]
    assert view["status"] == "partial"


def test_missing_brief_or_branch_never_falls_back_to_other_action_or_reason():
    source = assessment()
    del source["decision_brief"]["cases"]["held"]
    view = build_buyer_view(source, branch="held")
    assert view["status"] == "missing"
    assert all(view[key] is None for key in ("preferred_action", "label", "advice_reason", "next_step"))
    del source["decision_brief"]
    assert build_buyer_view(source)["preferred_action"] is None
    assert build_buyer_view({})["data_date"] is None
    with pytest.raises(ValueError, match="branch"):
        build_buyer_view(source, branch="addition")


def test_data_date_is_projection_of_real_nav_date_never_generation_or_research_time():
    source = assessment()
    source["decision_brief"]["cases"]["flat"]["data_date"] = "2026-10-28"
    assert build_buyer_view(source)["data_date"] == "2026-10-28"
    del source["decision_brief"]["cases"]["flat"]["data_date"]
    del source["decision_brief"]["data_date"]
    assert build_buyer_view(source)["data_date"] == "2026-10-26"
    source.pop("anchor")
    source["forecast"]["anchor"] = {"date": "2026-10-25"}
    assert build_buyer_view(source)["data_date"] == "2026-10-25"
    source["forecast"].pop("anchor")
    source["decision_brief"]["research_date"] = "2026-10-29"
    view = build_buyer_view(source)
    assert view["data_date"] is None
    assert "data_date" in view["missing_fields"]


def test_no_prediction_still_projects_actual_purchase_series_date_not_a_generated_date():
    source = assessment()
    source.pop("anchor")
    source["forecast"] = {}
    source["decision_brief"].pop("data_date")
    source["trade_scenario"]["purchase"] = {"latest_date": "2026-10-23"}
    assert build_buyer_view(source)["data_date"] == "2026-10-23"
    source["trade_scenario"]["plan"] = {"anchor_date": "2026-10-24"}
    assert build_buyer_view(source)["data_date"] == "2026-10-24"
    source["trade_scenario"]["plan"]["anchor_date"] = "2099-01-01T12:00:00+08:00"
    assert build_buyer_view(source)["data_date"] == "2026-10-23"


def test_pure_builder_does_not_open_files_or_use_clock(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("projection must not perform IO")
    source = assessment()
    monkeypatch.setattr("builtins.open", forbidden)
    view = build_buyer_view(source)
    assert view["preferred_action"] == "wait"
    json.dumps(view, ensure_ascii=False, allow_nan=False)

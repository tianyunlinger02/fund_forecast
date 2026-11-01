"""Synthetic current-public-data examples, without requests or database opens."""
from copy import deepcopy
import json

import pytest

from rmbfund.advice_evidence import normalize_record
from rmbfund.current_online import (
    derive_review_signals,
    project_current_detail,
    snapshot_online_context,
)
from rmbfund.domain import ValidationError
from rmbfund.investment_profile import build_investment_profile
from rmbfund.investor_decision import build_investor_decision
from rmbfund.personal_assessment import build_assessment
from rmbfund.portfolio import value_holding

PID = "open_fund:SYNTH:C"
SUBJECT = {"id": PID, "symbol": "SYNTH", "name": "合成在线基金", "kind": "open_fund",
           "share_class": "C", "exchange": "", "is_demo": False}
OLD = "2026-10-22T18:00:00+08:00"
NOW = "2026-10-26T18:00:00+08:00"


def nav(day="2026-10-23", value="2", observed=NOW, **changes):
    return {"date": day, "value": value, "currency": "CNY", "observed_at": observed,
            "available_at": observed, "published_at": None, "share_class": "C",
            "source": "synthetic-public-nav", "source_url": "https://example.com/nav",
            "nav_type": "final_unit_nav_candidate", "basis": "unit_nav", **changes}


def context(**changes):
    return {"schema_version": 1, "kind": "current_online_context", "status": "ready",
            "started_at": NOW, "evaluated_at": NOW, "checked_at": NOW, "requests_made": 2,
            "public_subjects": [deepcopy(SUBJECT)], "source_checks": [
                {"product_id": PID, "job_id": "nav-test", "family": "nav", "status": "ok",
                 "parse_status": "parsed", "checked_at": NOW, "observed_at": NOW,
                 "source_url": "https://example.com/nav", "source_date": "2026-10-23"}],
            "nav_by_product": {PID: [nav("2026-10-22", "1.8"), nav()]},
            "records_by_product": {}, "messages_by_product": {}, "warnings": [], **changes}


def detail(**changes):
    old_nav = nav("2026-10-22", "1.7", OLD)
    return {"product": deepcopy(SUBJECT), "status": "ready", "latest": old_nav,
            "history": [deepcopy(old_nav)], "previous_change": None,
            "last_checked_at": OLD, "last_successful_check_at": OLD,
            "latest_attempt_status": "ok", "background_records": [], "reasons": [], **changes}


def record(family="management", report_date="2026-10-22", observed=OLD, payload=None):
    return normalize_record({
        "product_id": PID, "family": family, "source": "synthetic-publisher",
        "upstream": "synthetic-publisher", "source_url": "https://example.com/" + family,
        "report_date": report_date, "observed_at": observed, "available_at": observed,
        "payload": payload or {"managers": [{"name": "合成经理甲", "start_date": "2025-01-01"}]},
    })


def thesis(posture="accumulate", cited=None):
    return {"status": "available", "record_id": "synthetic-thesis", "needed_review": [],
            "source": {"available_at": OLD}, "payload": {
                "product": deepcopy(SUBJECT), "research_date": "2026-10-22", "posture": posture,
                "horizon": {"start": "2026-10-22", "end": "2027-10-22"},
                "facts": [{"id": "f1", "claim": "合成事实"}],
                "citations": [{"id": "c1", "kind": "archived_evidence",
                               "record_id": cited["id"] if cited else "synthetic-basis"}],
                "assessment": {"summary": "合成研究，不是真实建议", "supporting": [
                    {"reason": "合成的长期买入依据", "fact_ids": ["f1"]}],
                    "counterevidence": [], "falsifiers": ["原经营逻辑失效"],
                    "change_conditions": ["原管理团队发生变化"], "missing": []}}}


def action_thesis(*, held="hold"):
    from rmbfund.research_contract import normalize_generated_research, research_output_example

    value = research_output_example(product=SUBJECT, as_of=OLD)
    value["facts"][0]["claim"] = "合成独立动作事实"
    value["citations"][0].update(evidence_id="synthetic-doc", locator="section:1", quote="合成独立动作事实")
    value["action_cases"]["new_buy"]["action"] = "buy"
    value["action_cases"]["held"]["action"] = held
    value["action_cases"]["addition"]["action"] = "add"
    doc = {"id": "synthetic-doc", "product_id": PID, "source_url": "https://example.com/actions",
           "title": "合成材料", "observed_at": OLD, "available_at": OLD,
           "published_at": "2026-10-22", "publication_precision": "date", "content_status": "full_text",
           "sections": [{"locator": "section:1", "text": "合成独立动作事实"}]}
    return {"status": "available", "needed_review": [],
            "payload": normalize_generated_research(value, evidence=[doc], as_of=OLD)}


def message(content_status="title_only", event_status="factual"):
    return {"id": "synthetic-news", "title": "合成公司发布投资者活动安排",
            "source_url": "https://example.com/new-item", "index_url": "https://example.com/news",
            "published_at": "2026-10-26", "publication_precision": "date",
            "excerpt": "合成摘录：公司拟公布活动日程，不代表盈利已经发生。",
            "content_status": content_status, "event_status": event_status,
            "issuer_ids": ["issuer:synthetic"], "channels": ["earnings"],
            "relevance_reason": "对应已披露持有的合成发行人，不代表今天仍持有",
            "interpretation_status": "unreviewed"}


def test_current_ram_nav_drives_valuation_without_mutating_inputs():
    local, online = detail(), context()
    original = deepcopy((local, online))
    result = project_current_detail(local, online, PID)
    assert result["latest"]["value"] == "2"
    assert result["latest"]["date"] == "2026-10-23"
    assert result["latest"]["observed_at"] == NOW
    assert [row["value"] for row in result["history"]] == ["1.8", "2"]
    assert result["previous_change"] == pytest.approx(2 / 1.8 - 1)
    assert result["online_source_context"]["nav_origin"] == "current_online"
    holding = {"shares": "100", "cost": "150", "currency": "CNY"}
    assert value_holding(holding, result["latest"])["market_value"] == "200"
    assert (local, online) == original


@pytest.mark.parametrize("changes", [
    {"product_id": "open_fund:OTHER:C"}, {"share_class": "A"}, {"currency": "USD"},
    {"nav_type": "intraday_estimate"}, {"available_at": "2026-10-27T18:00:00+08:00"},
])
def test_incompatible_online_nav_is_not_silently_mixed(changes):
    local = detail()
    online = context(nav_by_product={PID: [nav(**changes)]})
    result = project_current_detail(local, online, PID)
    assert result["latest"] == local["latest"]
    assert result["history"] == local["history"]
    assert result["online_source_context"]["nav_origin"] == "local_reference"
    assert result["online_source_context"]["rejected_nav_reasons"]


def test_current_public_identity_works_without_registered_local_product():
    result = project_current_detail({"product": None, "history": [], "latest": None}, context(), PID)
    assert result["product"] == SUBJECT
    assert result["status"] == "ready" and result["latest"]["value"] == "2"
    assert result["current_nav_history"] == context()["nav_by_product"][PID]


def test_older_online_date_does_not_make_a_newer_local_nav_current():
    local = detail(latest=nav("2026-10-26", "2.1", NOW), history=[nav("2026-10-26", "2.1", NOW)])
    result = project_current_detail(local, context(), PID)
    assert result["latest"] == local["latest"]
    assert result["online_source_context"]["nav_origin"] == "local_reference"
    assert any("未续鲜" in reason for reason in result["reasons"])


def test_disagreeing_same_date_current_navs_are_not_silently_selected():
    online = context(nav_by_product={PID: [nav(value="2"), nav(value="3", source="other-source")]})
    result = project_current_detail(detail(), online, PID)
    assert result["latest"]["value"] == "1.7"
    assert result["current_nav_history"] == []
    assert any("不同数值" in reason for reason in result["online_source_context"]["rejected_nav_reasons"])


def test_conflicting_public_identity_cannot_replace_current_details():
    online = context(public_subjects=[{**SUBJECT, "share_class": "A"}])
    result = project_current_detail(detail(), online, PID)
    assert result["latest"]["observed_at"] == OLD
    assert result["online_source_context"]["rejected_nav_reasons"]
    with pytest.raises(ValidationError, match="身份"):
        project_current_detail(detail(), context(), "open_fund:OTHER:C")


def test_failure_retains_old_dates_and_independent_family_checks():
    online = context(status="partial", nav_by_product={PID: []}, source_checks=[
        {"product_id": PID, "family": "nav", "status": "error", "checked_at": NOW,
         "reason": "synthetic failure"},
        {"product_id": PID, "family": "policy_news", "status": "ok", "checked_at": OLD},
    ])
    result = project_current_detail(detail(), online, PID)
    assert result["latest"]["observed_at"] == result["last_successful_check_at"] == OLD
    assert result["last_checked_at"] == NOW and result["latest_attempt_status"] == "error"
    assert result["online_source_context"]["nav_origin"] == "local_reference"
    assert result["online_context"]["source_checks"][1]["checked_at"] == OLD
    assert any("仅按原净值日" in reason for reason in result["reasons"])


def test_latest_online_check_does_not_redate_historical_disclosure():
    old = record()
    profile = build_investment_profile([old], as_of=OLD)
    authored = thesis(cited=old)
    local = detail(background_records=[old], investment_profile=profile, research_thesis=authored)
    result = project_current_detail(local, context(messages_by_product={PID: [message()]}), PID)
    assert result["investment_profile"] == profile
    assert result["research_thesis"] == authored
    assert result["online_context"]["review_signals_by_product"][PID] == []


def test_new_normalized_disclosure_updates_profile_and_specific_review_not_original_thesis():
    old = record()
    new = record(report_date="2026-10-26", observed=NOW,
                 payload={"managers": [{"name": "合成经理乙", "start_date": "2026-10-25"}]})
    authored = thesis(cited=old)
    original = deepcopy(authored)
    result = project_current_detail(detail(evidence_records=[old], research_thesis=authored),
                                    context(records_by_product={PID: [new]}), PID)
    manager = result["investment_profile"]["management"]
    assert manager["report_date"] == "2026-10-26"
    assert manager["payload"]["managers"][0]["name"] == "合成经理乙"
    signals = result["online_context"]["review_signals_by_product"][PID]
    assert len(signals) == 1 and signals[0]["kind"] == "cited_economic_basis_changed"
    assert signals[0]["citation_id"] == "c1" and signals[0]["current_record_id"] == new["id"]
    decision = build_investor_decision(product_id=PID, research_thesis=authored,
                                       online_context=result["online_context"])
    assert decision["status"] == "review_required"
    assert decision["branches"]["held"]["action"] == "review"
    assert decision["branches"]["flat"]["action"] == "wait"
    assert "c1" in decision["needed_review"][0]
    assert authored == original == result["research_thesis"]
    assert derive_review_signals(authored, [old, {**old, "available_at": NOW}], [message()]) == []


def test_saved_online_economic_basis_detects_change_without_source_record_in_store():
    old = record()
    new = record(report_date="2026-10-26", observed=NOW,
                 payload={"managers": [{"name": "合成经理乙", "start_date": "2026-10-25"}]})
    authored = thesis(cited=old)
    authored["payload"]["schema_version"] = 2
    authored["payload"]["change_evaluation"] = {"summary": "原理由。", "fact_ids": []}
    authored["payload"]["citations"] = [{"id": "c1", "kind": "online_evidence",
        "record_ids": [old["id"]], "record_basis": [old]}]
    before = deepcopy(authored)
    signals = derive_review_signals(authored, [new])
    assert len(signals) == 1 and signals[0]["previous_record_id"] == old["id"]
    assert signals[0]["current_record_id"] == new["id"]
    result = project_current_detail(detail(research_thesis=authored),
                                    context(records_by_product={PID: [new]}), PID)
    assert result["online_context"]["review_signals_by_product"][PID] == signals
    assert authored == before
    assert authored["payload"]["citations"][0]["record_basis"][0]["available_at"] == OLD


def test_adopted_body_id_survives_compact_context_and_drives_message_explanation():
    from rmbfund.investor_decision import _online_basis

    entry = {**message("online_html_sections"), "evidence_id": "actual-body-id",
             "channels": ["fund_disclosure", "policy"]}
    payload = {"schema_version": 2, "facts": [], "change_evaluation": {"fact_ids": []},
               "citations": [{"id": "c1", "kind": "online_evidence", "evidence_id": "actual-body-id"}]}
    compact = snapshot_online_context(context(messages_by_product={PID: [entry]}))
    evidence, reasons, _ = _online_basis(compact, PID, payload)
    assert evidence[0]["basis_relationship"] == "adopted_in_generated_research"
    assert "本次AI研究已引用" in reasons[0] and "基金披露、政策" in reasons[0]
    assert "fund_disclosure" not in reasons[0] and "尚未判断它支持" not in reasons[0]


def test_operations_record_reaches_existing_execution_branch_and_saved_conditions():
    operations = record("operations", "2026-10-26", NOW, payload={
        "subscription_state": "closed", "redemption_state": "closed", "channel_verified": False,
    })
    online = context(records_by_product={PID: [operations]})
    saved = build_assessment(None, subject=SUBJECT, research_thesis=action_thesis(held="redeem"),
                             online_context=online, execution_conditions={
                                 "subscription": "open", "redemption": "open", "report_date": "2026-10-22"})
    decision = saved["investor_decision"]
    assert decision["branches"]["flat"]["action"] == "blocked"
    assert decision["branches"]["addition"]["action"] == "blocked"
    assert decision["branches"]["held"]["action"] == "redeem"
    assert any("暂停赎回" in reason for reason in decision["execution_notes"])
    assert saved["execution_conditions"]["report_date"] == "2026-10-26"
    assert saved["execution_conditions"]["channel_verified"] is False
    assert decision["online_evidence"][0]["basis_relationship"] == "execution_condition"
    assert "暂停" in decision["online_reasons"][0]


def test_future_and_older_execution_disclosures_do_not_replace_available_conditions():
    old_closed = record("operations", "2026-10-22", OLD, payload={
        "subscription_state": "closed", "redemption_state": "closed", "channel_verified": False,
    })
    newer_open = {"subscription": "open", "redemption": "open", "report_date": "2026-10-26"}
    decision = build_investor_decision(product_id=PID, research_thesis=action_thesis(),
                                       execution_conditions=newer_open,
                                       online_context=context(records_by_product={PID: [old_closed]}))
    assert decision["branches"]["flat"]["action"] == "buy"
    future_closed = record("operations", "2026-10-27", "2026-10-27T18:00:00+08:00", payload={
        "subscription_state": "closed", "redemption_state": "closed", "channel_verified": False,
    })
    decision = build_investor_decision(product_id=PID, research_thesis=action_thesis(),
                                       online_context=context(records_by_product={PID: [future_closed]}))
    assert decision["branches"]["flat"]["action"] == "buy"
    assert decision["online_evidence"] == []


def test_neutral_titles_and_uninterpreted_plans_are_not_investment_reasons():
    authored = action_thesis()
    baseline = build_investor_decision(product_id=PID, research_thesis=authored)
    for content, status in (("title_only", "factual"), ("online_excerpt", "plan")):
        online = context(messages_by_product={PID: [message(content, status)]})
        decision = build_investor_decision(product_id=PID, research_thesis=authored, online_context=online)
        assert decision["branches"] == baseline["branches"]
        assert decision["needed_review"] == []
        assert decision["online_evidence"][0]["excerpt"] == message()["excerpt"]
        assert decision["online_evidence"][0]["basis_relationship"] == "uninterpreted"
        assert all(decision["online_reasons"][0] not in branch["reasons"]
                   for branch in decision["branches"].values())
        assert "合成发行人" in decision["online_reasons"][0]
        assert ("只取得标题" if content == "title_only" else "这是计划") in decision["online_reasons"][0]
    missing = build_investor_decision(product_id=PID, research_thesis=authored,
                                      online_context=context(status="unavailable", nav_by_product={}))
    assert missing["branches"]["held"]["action"] == "hold"


def test_compact_snapshot_excludes_raw_documents_and_full_series_is_detached_and_idempotent():
    macro = record("macro", "2026-10-26", NOW, payload={
        "kind": "fx", "series_id": "synthetic-fx", "currency": "USD", "unit": "CNY_per_currency",
        "observations": [{"date": "2026-10-23", "value": "7"}, {"date": "2026-10-26", "value": "7.1"}],
    })
    online = context(raw_html="<html>not saved</html>", nav_by_product={PID: [nav(), nav("2026-10-26", "2.1")]},
                     records_by_product={PID: [macro]}, messages_by_product={PID: [{**message(), "raw_pdf": "not saved"}]})
    online["nav_by_product"][PID][0]["raw_response"] = "not saved"
    compact = snapshot_online_context(online)
    assert "raw_html" not in compact
    assert len(compact["nav_by_product"][PID]) == 1
    assert compact["nav_observation_counts"][PID] == 2
    assert "not saved" not in json.dumps(compact)
    payload = compact["records_by_product"][PID][0]["payload"]
    assert "observations" not in payload and payload["observations_count"] == 2
    assert payload["latest_observations"]["date"] == "2026-10-26"
    assert snapshot_online_context(compact) == compact
    online["messages_by_product"][PID][0]["title"] = "later title"
    assert compact["messages_by_product"][PID][0]["title"] != "later title"


def test_schema4_online_snapshot_does_not_change_prediction_and_absent_keeps_old_shape():
    head = {"horizon": 20, "entry_lag": 2, "expected_change": 0.02,
            "up_probability": 0.6, "lower": -0.1, "upper": 0.15}
    prediction = {"product_id": PID, "record_id": "synthetic-forecast", "model_id": "nav_ridge_empirical_v2",
                  "entry_assumptions": {"baseline_entry_lag": 2}, "execution_forecasts": [head]}
    authored, online = thesis(), context(messages_by_product={PID: [message()]})
    before = deepcopy((prediction, authored, online))
    baseline = build_assessment(prediction, subject=SUBJECT, research_thesis=authored)
    saved = build_assessment(prediction, subject=SUBJECT, research_thesis=authored, online_context=online)
    assert saved["schema_version"] == 4 and "online_context" not in baseline
    assert (prediction, authored, online) == before
    for key in ("forecast", "decision", "execution", "anchor"):
        assert saved[key] == baseline[key]
    frozen = deepcopy(saved)
    online["messages_by_product"][PID][0]["excerpt"] = "subsequent source change"
    online["checked_at"] = "2026-10-27T18:00:00+08:00"
    authored["payload"]["posture"] = "exit"
    assert saved == frozen
    assert "raw_html" not in json.dumps(saved)

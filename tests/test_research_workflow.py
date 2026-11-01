"""Small integration checks for explicit research and remembered local conditions."""
from copy import deepcopy

from rmbfund.buyer_research import current_public_identity
from rmbfund.personal_assessment import build_assessment, cash_use_suitability
from rmbfund.portfolio import Portfolio
from rmbfund.service import Workspace

TARGET = "open_fund:900001:C"


def test_saved_local_conditions_do_not_add_a_holding_or_change_snapshots(tmp_path):
    portfolio = Portfolio(tmp_path / "synthetic", "public")
    subject = current_public_identity(TARGET)
    assert portfolio.research_preferences(subject) is None
    assert not portfolio.path.exists()
    assessment = build_assessment(None, subject=subject)
    original = portfolio.save_assessment(assessment)
    conditions = {"horizon": 5, "subscription_fee": "0.15", "redemption_fee": "",
                  "budget": "100", "sizing": {"near_term_required": "yes", "plan_horizon": "下周"}}
    receipt = portfolio.save_research_preferences(subject, conditions)
    conditions["budget"] = "999"
    reopened = Portfolio(portfolio.root, "public")
    assert reopened.research_preferences(subject)["conditions"]["budget"] == "100"
    assert receipt["saved_at"]
    assert not reopened.list_funds()
    assert reopened.assessment_detail(original["id"])["assessment"] == assessment
    other_class = {**subject, "id": "open_fund:900001:A", "share_class": "A"}
    assert reopened.research_preferences(other_class) is None


def test_explicit_near_term_need_is_separate_from_investment_research():
    profile = {"mandate": {"payload": {"name": "合成QDII基金"}}}
    fit = cash_use_suitability({"near_term_required": "yes", "near_term_use_time": "下周"}, profile=profile)
    assert fit["status"] == "not_suitable_for_new_money" and fit["qdii"]
    assert "没有确认到账日期" in fit["reason"]
    snapshot = build_assessment(None, subject=current_public_identity(TARGET),
        sizing_plan={"user_conditions": {"near_term_required": "yes"}},
        investment_analysis={"profile": profile})
    assert snapshot["cash_use_fit"]["status"] == "not_suitable_for_new_money"
    assert snapshot["forecast"] == {} and snapshot["decision"] == {}


def test_unconfigured_research_does_not_acquire_register_or_initialize_evidence(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "synthetic", "public")
    try:
        calls = []
        monkeypatch.setattr(workspace, "current_online", lambda *args, **kwargs: calls.append(args))
        monkeypatch.setattr("rmbfund.research_settings.is_configured", lambda *args, **kwargs: False)
        result = workspace.update_research(current_public_identity(TARGET), settings={})
        assert result["status"] == "unconfigured" and not calls
        assert workspace.store.get_product(TARGET) is None
        assert not workspace.store.has_evidence_schema()
    finally:
        workspace.store.close()


def test_generated_research_is_reloaded_without_renewing_source_check(tmp_path, monkeypatch):
    """Runner/writer are injected; actual contract/API tests live in their modules."""
    at = "2026-10-27T10:00:00+08:00"
    later = "2026-10-27T10:02:00+08:00"
    workspace = Workspace(tmp_path / "synthetic", "public")
    try:
        subject = current_public_identity(TARGET)
        context = {"schema_version": 1, "kind": "current_online_context", "checked_at": at,
                   "evaluated_at": at, "public_subjects": [subject], "source_checks": [],
                   "nav_by_product": {}, "records_by_product": {}, "messages_by_product": {}}
        calls, saves = [], []
        monkeypatch.setattr("rmbfund.service.now_iso", lambda: later)
        monkeypatch.setattr("rmbfund.research_settings.is_configured", lambda *args, **kwargs: True)

        def run(**kwargs):
            calls.append(kwargs)
            assert "evidence_records" in kwargs["detail"]
            assert "evidence_links" in kwargs["detail"]
            return {"status": "generated", "research": {"synthetic": True}}

        monkeypatch.setattr("rmbfund.research_runner.run_research", run)
        monkeypatch.setattr("rmbfund.providers.advice_evidence.append_generated_research",
                            lambda *args, **kwargs: saves.append(args) or {"records_added": 1})
        before = deepcopy(context)
        result = workspace.update_research(subject, settings={}, api_key="synthetic-key", context=context)
        assert result["status"] == "generated" and len(calls) == len(saves) == 1
        assert result["context"]["checked_at"] == at
        assert result["evaluation"]["evaluated_at"] == later
        assert context == before
        assert workspace.store.has_evidence_schema()
    finally:
        workspace.store.close()


def test_research_follow_up_reads_only_new_publisher_attachments(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "synthetic", "public")
    subject = current_public_identity(TARGET)
    at = "2026-10-27T10:00:00+08:00"
    source_url = "https://example.com/fund-report.pdf"
    context = {
        "checked_at": at, "evaluated_at": at, "public_subjects": [subject],
        "source_checks": [{"source_url": "https://example.com/already.pdf", "status": "ok"},
                          {"source_url": "https://example.com/bulletin", "status": "ok"},
                          {"source_url": source_url, "status": "budget_exhausted", "request_made": False}],
        "stopped_urls": ["https://stopped.example/no.pdf"], "nav_by_product": {},
        "records_by_product": {}, "messages_by_product": {TARGET: [{
            "id": "message-1", "source_url": "https://example.com/bulletin", "title": "合成披露",
            "published_at": "2026-10-26", "content_status": "unparsed_document",
            "attachment_urls": [source_url, "https://example.com/already.pdf", "https://stopped.example/no.pdf"],
        }]},
    }
    calls = []

    def current(subjects, **kwargs):
        calls.append(kwargs)
        assert [row["source_url"] for row in kwargs["source_plan"]] == [source_url]
        assert kwargs["request_budget"] == 2
        return {"evidence_by_product": {TARGET: [{"id": "doc-1", "source_url": source_url}]},
                "records_by_product": {}, "messages_by_product": {}, "requests_made": 1,
                "source_checks": [{"job_id": "research-attachment-0", "source_url": source_url,
                                   "status": "ok"}]}

    def run(**kwargs):
        callback = kwargs["acquire_more"]
        first = callback({"evidence_ids": ["message-1"], "questions": [], "round": 1})
        assert first["evidence_by_product"][TARGET][0]["id"] == "doc-1"
        second = callback({"evidence_ids": ["message-1"], "questions": [], "round": 2})
        assert second["status"] == "unavailable"
        return {"status": "partial", "research": None}

    try:
        monkeypatch.setattr("rmbfund.research_settings.is_configured", lambda *a, **k: True)
        monkeypatch.setattr(workspace, "current_online", current)
        monkeypatch.setattr("rmbfund.research_runner.run_research", run)
        before = deepcopy(context)
        result = workspace.update_research(subject, settings={}, api_key="synthetic-key", context=context)
        assert result["status"] == "partial" and "receipt" not in result
        assert len(calls) == 1 and context == before
        assert not workspace.store.has_evidence_schema()
    finally:
        workspace.store.close()


def test_partial_parse_does_not_erase_fuller_same_document_disclosure():
    from rmbfund.investment_profile import prefer_disclosure

    original = {"id": "a", "family": "portfolio", "report_date": "2026-06-30",
                "source_url": "https://example.com/h1.pdf", "available_at": "2026-10-26T10:00:00+08:00",
                "payload": {"denominator": "nav", "assets": {"equity": "0.816"},
                            "sectors": [], "countries": [], "currencies": [],
                            "holdings": [{"issuer_id": "issuer:nvidia", "weight": "0.0938"},
                                         {"issuer_id": "issuer:tsmc", "weight": "0.0968"}]}}
    partial = deepcopy(original)
    partial.update(id="b", available_at="2026-10-27T10:00:00+08:00")
    partial["payload"]["holdings"] = partial["payload"]["holdings"][:1]
    assert not prefer_disclosure(partial, original)
    assert prefer_disclosure(original, partial)
    partial["report_date"] = "2026-09-30"
    assert prefer_disclosure(partial, original)

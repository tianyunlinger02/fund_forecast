"""Repeated v2 bind/append/action checks using synthetic pytest-local evidence only."""
from copy import deepcopy
import json

import pytest

from rmbfund.advice_evidence import (
    normalize_record, normalize_research_thesis, research_thesis_source_uri,
)
from rmbfund.domain import Product, ValidationError
from rmbfund.investor_decision import build_investor_decision
from rmbfund.providers.advice_evidence import (
    append_generated_research, import_local_research_note, import_reviewed_document,
)
from rmbfund.research_contract import normalize_generated_research, research_output_example
from rmbfund.research_thesis import (
    _same_source_basis, build_research_thesis, economic_content_compatible,
    research_explained_record_ids, with_research_basis,
)
from rmbfund.storage import Store

PRODUCT = Product("synthetic:research:C", "SYNTH", "合成研究测试基金", "open_fund",
                  share_class="C", exchange="OTC")
IDENTITY = {key: getattr(PRODUCT, key) for key in
            ("id", "symbol", "name", "kind", "share_class", "exchange")}
GENERATED = "2026-10-27T12:00:00+08:00"
START = "2026-10-27T12:01:00+08:00"
END = "2026-10-27T12:02:00+08:00"
LATER = "2026-10-28T12:00:00+08:00"
EARLY = "2026-10-26T12:00:00+08:00"


def clock(start=START, end=END):
    return iter((start, end)).__next__


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "synthetic-research.sqlite3", mode="public")
    value.add_product(PRODUCT)
    value.initialize_evidence_schema()
    yield value
    value.close()


def note():
    value = research_output_example(product=IDENTITY, as_of=GENERATED)
    value["citations"][0].update(evidence_id="doc-1", locator="page:3;section:results",
                                  quote="Revenue rose to 12.", claim="Revenue rose to 12.")
    value["facts"][0]["claim"] = "合成原文报告收入为12。"
    return value


def document():
    return {
        "id": "doc-1", "product_id": PRODUCT.id, "source_url": "https://example.com/results",
        "title": "Synthetic report", "published_at": "2026-10-27", "publication_precision": "date",
        "observed_at": "2026-10-27T10:00:00+08:00", "available_at": "2026-10-27T10:00:00+08:00",
        "content_status": "full_text", "sections": [
            {"locator": "page:3;section:results", "text": "Reported result. Revenue\nrose to 12. Other discussion."},
            {"locator": "page:4", "text": "UNADOPTED_DOCUMENT_TEXT must not be saved."},
        ],
    }


def bound(value=None):
    return normalize_generated_research(value or note(), evidence=[document()], as_of=GENERATED)


def projected(value):
    return {"status": "available", "record_id": "synthetic-projection", "payload": value,
            "needed_review": []}


def test_online_binding_uses_real_locator_url_clocks_and_retains_only_adopted_excerpt():
    value, doc = note(), document()
    value["citations"][0]["source_url"] = "https://example.com/model-invented-url"
    before = deepcopy((value, doc))
    result = normalize_generated_research(value, evidence={"documents": [doc]}, as_of=GENERATED)
    citation = result["citations"][0]
    assert citation["source_url"] == doc["source_url"]
    assert citation["observed_at"] == doc["observed_at"]
    assert citation["quote"] == "Revenue rose to 12."
    assert result["origin"]["human_reviewed"] is False
    assert normalize_research_thesis(result) == result
    assert "UNADOPTED_DOCUMENT_TEXT" not in json.dumps(result)
    assert "sections" not in citation and "excerpt" not in citation
    record = normalize_record({
        "product_id": PRODUCT.id, "family": "scale", "source": "synthetic-source",
        "upstream": "synthetic-source", "source_url": doc["source_url"], "report_date": "2026-10-27",
        "observed_at": "2026-10-27T09:00:00+08:00", "available_at": "2026-10-27T09:00:00+08:00",
        "payload": {"total_assets_cny": "100", "combined_share_classes": True},
    })
    mapped = normalize_generated_research(note(), evidence={
        "documents": [doc], "records": [record], "record_links": [{
            "record_id": record["id"], "evidence_id": "doc-1", "locator": "page:3;section:results",
            "quote": "Revenue rose to 12.",
        }],
    }, as_of=GENERATED)
    assert mapped["citations"][0]["record_ids"] == [record["id"]]
    assert mapped["citations"][0]["record_basis"] == [record]
    assert normalize_generated_research(mapped) == mapped
    unmapped = normalize_generated_research(note(), evidence={
        "documents": [doc], "records": [record], "record_links": [{
            "record_id": record["id"], "evidence_id": "doc-1", "locator": "page:3;section:results",
            "quote": "A different sentence.",
        }],
    }, as_of=GENERATED)
    assert unmapped["citations"][0]["record_ids"] == []
    retained = deepcopy(doc)
    retained.update(content_status="retained_excerpt", sections=[
        {"locator": "page:3;section:results", "text": "Revenue rose to 12."},
        {"locator": "page:3;section:results", "text": "Costs also rose."},
    ])
    both = note()
    both["citations"].append({**both["citations"][0], "id": "c2", "quote": "Costs also rose."})
    fragments = normalize_generated_research(both, evidence=[retained], as_of=GENERATED)
    assert len(fragments["citations"]) == 2
    joined = note()
    joined["citations"][0]["quote"] = "Revenue rose to 12. Costs also rose."
    with pytest.raises(ValidationError, match="实际正文"):
        normalize_generated_research(joined, evidence=[retained], as_of=GENERATED)
    retained["content_status"] = "full_text"
    with pytest.raises(ValidationError, match="不唯一"):
        normalize_generated_research(both, evidence=[retained], as_of=GENERATED)
    assert (value, doc) == before
    for field, incorrect in (("quote", "Revenue rose to 99."), ("locator", "page:99"),
                             ("evidence_id", "missing"), ("kind", "reviewed_excerpt")):
        invalid = note()
        invalid["citations"][0][field] = incorrect
        with pytest.raises(ValidationError):
            normalize_generated_research(invalid, evidence=[doc], as_of=GENERATED)
    with pytest.raises(ValidationError, match="RAM"):
        normalize_generated_research(note())
    with pytest.raises(ValidationError):
        normalize_generated_research(result, evidence=[])
    doc["content_status"] = "title_only"
    with pytest.raises(ValidationError, match="标题"):
        normalize_generated_research(note(), evidence=[doc])


def _v1_note():
    value = note()
    value = {key: value[key] for key in ("product", "note_id", "research_date", "horizon", "posture",
                                        "facts", "citations", "assessment", "revision")}
    value.update(author="Synthetic author", reviewer="Synthetic reviewer", research_date="2026-10-26")
    value["horizon"]["start"] = "2026-10-26"
    value["citations"] = [{
        "id": "c1", "kind": "reviewed_excerpt", "publisher_url": "https://example.com/previous",
        "title": "Synthetic earlier note", "publication_precision": "date", "published_at": "2026-10-26",
        "checked_at": EARLY, "locator": "section:results", "claim": "Earlier result.", "excerpt": "Earlier result.",
    }]
    return value


@pytest.mark.parametrize("provider,namespace,source", [
    ("runner-supplied", "api", "api_generated_research"),
    ("interactive_assistant", "assistant", "assistant_generated_research"),
])
def test_generated_append_reuses_existing_tables_actual_clocks_and_explicit_v1_successor(store, provider, namespace, source):
    prior = import_local_research_note(store, PRODUCT, _v1_note(), clock=clock(EARLY, EARLY))["records"][0]
    value = note()
    value["origin"]["provider"] = provider
    value["revision"] = {"id": "generated-v2", "supersedes": prior["id"]}
    normalized = bound(value)
    result = append_generated_research(store, PRODUCT, normalized, clock=clock())
    record = result["records"][0]
    capture = store.advice_capture(result["capture_ids"][0])
    assert record["source"] == record["upstream"] == source
    assert record["source_url"].startswith(f"rmb-research://{namespace}/")
    assert record["observed_at"] == START and record["available_at"] == END
    assert capture["status"] == "ok" and capture["metadata"]["human_reviewed"] is False
    assert capture["available_at"] == END and "reviewed_at" not in capture["metadata"]
    assert json.loads(capture["raw_bytes"]) == normalized
    assert b"UNADOPTED_DOCUMENT_TEXT" not in capture["raw_bytes"]
    again = append_generated_research(store, PRODUCT, normalized, clock=clock(LATER, LATER))
    assert again["status"] == "existing" and again["records_added"] == again["captures_added"] == 0
    assert again["records"][0]["available_at"] == END
    records = store.advice_evidence(PRODUCT.id, LATER)
    assert build_research_thesis(records, product_id=PRODUCT.id, as_of=EARLY)["record_id"] == prior["id"]
    current = build_research_thesis(records, product_id=PRODUCT.id, as_of=END)
    assert current["record_id"] == record["id"] and not current["needed_review"]
    assert any("AI" in warning for warning in current["warnings"])
    invalid = deepcopy(normalized)
    invalid["assessment"]["summary"] = "Changed content cannot overwrite the same revision."
    with pytest.raises(ValidationError, match="修订标识"):
        append_generated_research(store, PRODUCT, invalid, clock=clock(LATER, LATER))
    with pytest.raises(ValidationError, match="append_generated_research"):
        import_local_research_note(store, PRODUCT, normalized, clock=clock())
    independent = deepcopy(normalized)
    independent.update(note_id="independent-thesis", revision={"id": "independent", "supersedes": None})
    append_generated_research(store, PRODUCT, independent, clock=clock(LATER, LATER))
    assert build_research_thesis(store.advice_evidence(PRODUCT.id, LATER), product_id=PRODUCT.id,
                                 as_of=LATER)["status"] == "contested"


def test_actions_follow_specific_fact_reasons_not_posture_and_keep_personal_constraints():
    value = bound()
    value["posture"] = "unresolved"  # Compatibility summary is not the decision engine.
    value["assessment"]["supporting"] = []
    value["assessment"]["priced_in"] = {
        "status": "partly_reflected", "reason": "合成已核对价格变化，仅作定性判断。", "fact_ids": ["f1"],
    }
    value["action_cases"]["new_buy"].update(action="buy", reason="合成新买的具体依据。")
    value["action_cases"]["held"].update(action="redeem", reason="决定性反方推翻原持有理由。")
    value["action_cases"]["addition"].update(action="add", reason="合成追加的独立依据。")
    thesis = projected(value)
    before = deepcopy(thesis)
    result = build_investor_decision(product_id=PRODUCT.id, research_thesis=thesis)
    assert result["status"] == "generated_research" and result["needed_review"] == []
    assert {key: row["action"] for key, row in result["branches"].items()} == {
        "flat": "buy", "held": "redeem", "addition": "add"}
    assert result["branches"]["held"]["reasons"][0] == "决定性反方推翻原持有理由。"
    assert result["branches"]["held"]["citation_ids"] == ["c1"]
    assert result["action_basis"] == "independent_cases"
    for key, case in (("flat", "new_buy"), ("held", "held"), ("addition", "addition")):
        assert result["branches"][key]["reasons"] == [value["action_cases"][case]["reason"]]
    critical = deepcopy(value)
    critical["key_unknowns"] = [{"question": "决定新买价格的资料缺失。", "affects": ["new_buy"], "critical": True}]
    partial = build_investor_decision(product_id=PRODUCT.id, research_thesis=projected(critical))
    assert partial["branches"]["flat"]["action"] == "wait"
    assert partial["branches"]["held"]["action"] == "redeem"
    assert partial["branches"]["addition"]["action"] == "add"
    fit = {"status": "not_suitable_for_new_money", "near_term_required": "yes", "reason": "近期确实要用钱。"}
    personal = build_investor_decision(product_id=PRODUCT.id, research_thesis=thesis, cash_use_fit=fit)
    assert personal["branches"]["flat"]["action"] == personal["branches"]["addition"]["action"] == "wait"
    assert personal["branches"]["held"]["action"] == "redeem" and personal["cash_use_fit"] == fit
    blocked = build_investor_decision(product_id=PRODUCT.id, research_thesis=thesis, cash_use_fit=fit,
                                      execution_conditions={"subscription": "closed", "redemption": "closed"})
    assert blocked["branches"]["flat"]["action"] == "blocked"
    assert blocked["branches"]["held"]["action"] == "redeem"
    assert any("暂停赎回" in reason for reason in blocked["execution_notes"])
    risk = deepcopy(value)
    risk["action_cases"]["held"]["action"] = "hold"
    reduced = build_investor_decision(product_id=PRODUCT.id, research_thesis=projected(risk),
                                      sizing_plan={"reduction": {"minimum_gross_redemption": "10"}})
    assert reduced["branches"]["held"]["action"] == "reduce"
    assert reduced["branches"]["addition"]["action"] == "wait"
    assert reduced["branches"]["held"]["research_action"] == "hold"
    assert reduced["branches"]["held"]["research_reason"] == value["action_cases"]["held"]["reason"]
    assert reduced["branches"]["held"]["fact_ids"] == value["action_cases"]["held"]["fact_ids"]
    assert reduced["branches"]["held"]["constraint_reasons"]
    assert blocked["branches"]["flat"]["research_action"] == "buy"
    assert blocked["branches"]["flat"]["constraint_reasons"]
    assert thesis == before


def _import(store, family, payload, *, at=EARLY, url="https://example.com/report", report_date="2026-06-30"):
    return import_reviewed_document(
        store, PRODUCT, raw_bytes=json.dumps({"report": payload}).encode(), source_url=url,
        records=[{"family": family, "payload": payload, "report_date": report_date, "locator": "json:/report"}],
        reviewed=True, clock=clock(at, at),
    )


def test_calculation_and_archived_citations_reuse_real_inputs_without_pretending_json_is_original(store):
    archived = _import(store, "scale", {"total_assets_cny": "100", "combined_share_classes": True})
    record_id = archived["record_ids"][0]
    calculation = {
        "id": "calculation-1", "product_id": PRODUCT.id, "method": "existing_scale_calculation",
        "inputs": {"assets_cny": "100", "report_date": "2026-06-30"}, "result": {"assets_cny": "100"},
        "available_at": GENERATED, "record_ids": [record_id], "input_evidence_ids": [],
        "sections": [{"locator": "calculation:assets", "text": "Program result: assets CNY 100."}],
    }
    value = note()
    value["citations"] = [
        {"id": "c1", "kind": "calculation", "evidence_id": "calculation-1", "locator": "calculation:assets",
         "quote": "Program result: assets CNY 100.", "claim": "合成程序结果，不是发布者原文。", "result": "model-fabricated"},
        {"id": "c2", "kind": "archived_evidence", "record_id": record_id,
         "capture_id": archived["capture_ids"][0], "locator": "json:/report", "claim": "Synthetic archived assets."},
    ]
    value["facts"][0]["citation_ids"] = ["c1", "c2"]
    result = normalize_generated_research(value, evidence={
        "records": archived["records"], "calculations": [calculation],
    }, as_of=GENERATED)
    assert result["citations"][0]["result"] == calculation["result"]
    assert "sections" not in result["citations"][0]
    appended = append_generated_research(store, PRODUCT, result, clock=clock())
    assert appended["records_added"] == 1
    assert research_explained_record_ids(result) == {record_id}
    candidate = Product("synthetic:alternative:A", "PEER", "合成候选基金", "open_fund", share_class="A")
    store.add_product(candidate)
    candidate_fees = {"management_rate": "0.01", "custody_rate": "0.001", "sales_service_rate": "0",
                      "subscription_rate": "0", "redemption_bands": [],
                      "holding_day_basis": "unknown", "channel_verified": False}
    peer_evidence = import_reviewed_document(
        store, candidate, raw_bytes=json.dumps({"fees": candidate_fees}).encode(),
        source_url="https://example.com/candidate-fees", reviewed=True, clock=clock(EARLY, EARLY),
        records=[{"family": "fees", "payload": candidate_fees, "report_date": "2026-06-30",
                  "locator": "json:/fees"}],
    )
    with_candidate = deepcopy(result)
    with_candidate["origin"]["generated_at"] = LATER
    with_candidate["research_date"] = "2026-10-28"
    with_candidate["revision"] = {"id": "candidate-fees", "supersedes": appended["record_ids"][0]}
    with_candidate["alternatives"] = [{"product_id": candidate.id, "reason": "合成候选明确费率可比较。", "fact_ids": ["f1"]}]
    with_candidate["citations"].append({
        "id": "candidate-fees", "kind": "archived_evidence", "record_id": peer_evidence["record_ids"][0],
        "capture_id": peer_evidence["capture_ids"][0], "locator": "json:/fees", "claim": "合成候选真实存档费率。",
    })
    with_candidate["facts"][0]["citation_ids"].append("candidate-fees")
    assert append_generated_research(store, PRODUCT, with_candidate, clock=clock(LATER, LATER))["records_added"] == 1
    invalid = deepcopy(result)
    invalid["citations"][0]["record_ids"] = ["a" * 64]
    invalid["revision"] = {"id": "bad-input", "supersedes": appended["record_ids"][0]}
    with pytest.raises(ValidationError, match="计算依据"):
        append_generated_research(store, PRODUCT, invalid, clock=clock(LATER, LATER))


def _release(start="2026-04-01", end="2026-06-30", quarter=2, value="12"):
    return {"kind": "financial_release", "issuer_id": "synthetic-issuer", "issuer_name": "Synthetic issuer",
            "security_id": "synthetic-security", "release_id": "release-" + end, "revision_id": "original",
            "revision_kind": "original", "metrics": [{
                "name": "revenue", "category": "actual", "currency": "USD", "unit": "currency", "scale": "1",
                "value": value, "accounting_basis": "GAAP", "period_role": "primary", "locator": "json:/report",
                "period": {"start": start, "end": end, "basis": "quarter", "fiscal_year": 2026, "fiscal_quarter": quarter},
            }]}


def test_adopted_online_record_basis_survives_without_source_record_storage(store):
    doc = document()
    doc["sections"][0]["text"] = "Assets CNY 100."
    record = normalize_record({
        "product_id": PRODUCT.id, "family": "scale", "source": "synthetic-online",
        "upstream": "synthetic-online", "source_url": doc["source_url"], "report_date": "2026-10-27",
        "observed_at": doc["observed_at"], "available_at": doc["available_at"],
        "payload": {"total_assets_cny": "100", "combined_share_classes": True},
    })
    value = note()
    value["facts"][0]["claim"] = "合成原文资产100元。"
    value["citations"][0].update(quote="Assets CNY 100.", claim="合成资产100元。")
    calculation = {"id": "calc-assets", "product_id": PRODUCT.id, "method": "existing_assets",
                   "inputs": {"assets_cny": "100"}, "result": "100", "available_at": GENERATED,
                   "record_ids": [record["id"]], "input_evidence_ids": [],
                   "sections": [{"locator": "program_result", "text": "Program assets CNY 100."}]}
    value["citations"].append({"id": "c2", "kind": "calculation", "evidence_id": "calc-assets",
                               "locator": "program_result", "quote": "Program assets CNY 100.",
                               "claim": "合成程序结果100元。"})
    value["facts"][0]["citation_ids"].append("c2")
    normalized = normalize_generated_research(value, evidence={
        "documents": [doc], "records": [record], "calculations": [calculation],
        "record_links": [{"record_id": record["id"], "evidence_id": doc["id"],
                          "locator": "page:3;section:results", "quote": "Assets CNY 100."}],
    }, as_of=GENERATED)
    append_generated_research(store, PRODUCT, normalized, clock=clock())
    stored = store.advice_evidence(PRODUCT.id, LATER)
    assert len(stored) == 1 and stored[0]["family"] == "research_thesis"
    basis = stored[0]["payload"]["citations"][0]["record_basis"][0]
    assert basis == record and basis["available_at"] == doc["available_at"]
    candidate = {key: item for key, item in record.items() if key != "id"}
    candidate.update(report_date="2026-10-28", observed_at=LATER, available_at=LATER)
    candidate["payload"] = {"total_assets_cny": "80", "combined_share_classes": True}
    updated = build_research_thesis([*stored, normalize_record(candidate)], product_id=PRODUCT.id, as_of=LATER)
    assert any("较新经济内容" in reason for reason in updated["needed_review"])
    invalid = deepcopy(normalized)
    invalid["citations"][0]["record_basis"] = []
    with pytest.raises(ValidationError, match="record_ids"):
        normalize_generated_research(invalid)


def test_economic_content_compatibility_keeps_descriptions_and_unknown_fee_fields_non_conflicting():
    mandate = {"product_id": PRODUCT.id, "family": "mandate", "payload": {
        "fund_id": "fund", "symbol": "SYNTH", "share_class": "C", "name": "Full official name",
        "benchmark": {"id": "b", "name": "Benchmark"},
    }}
    short_name = deepcopy(mandate)
    short_name["payload"]["name"] = "Short name QDII"
    assert economic_content_compatible(mandate, short_name)
    fees = {"product_id": PRODUCT.id, "family": "fees", "payload": {
        "management_rate": "0.01", "custody_rate": "0.001", "sales_service_rate": "0.004",
        "subscription_rate": "0", "redemption_bands": [{"min_days": 0, "max_days": None, "rate": "0"}],
        "holding_day_basis": "unknown", "channel_verified": False,
    }}
    missing_subscription = deepcopy(fees)
    missing_subscription["payload"].pop("subscription_rate")
    assert economic_content_compatible(fees, missing_subscription)
    known_difference = deepcopy(fees)
    known_difference["payload"]["subscription_rate"] = "0.001"
    assert not economic_content_compatible(fees, known_difference)
    required_difference = deepcopy(fees)
    required_difference["payload"]["management_rate"] = "0.02"
    assert not economic_content_compatible(fees, required_difference)


def test_cross_publisher_new_quarter_and_unaddressed_conflict_require_review_only_for_same_economic_scope(store):
    old = _import(store, "issuer_fundamentals", _release())
    value = note()
    value["citations"] = [{"id": "c1", "kind": "archived_evidence", "record_id": old["record_ids"][0],
                           "capture_id": old["capture_ids"][0], "locator": "json:/report", "claim": "Earlier quarter."}]
    first = append_generated_research(store, PRODUCT, normalize_generated_research(value), clock=clock())["records"][0]
    new = _import(store, "issuer_fundamentals", _release("2026-07-01", "2026-09-30", 3, "9"),
                  at=LATER, url="https://another.example.com/release", report_date="2026-09-30")
    cited, candidate = old["records"][0], new["records"][0]
    assert cited["upstream"] != candidate["upstream"] and _same_source_basis(cited, candidate)
    current = build_research_thesis(store.advice_evidence(PRODUCT.id, LATER), product_id=PRODUCT.id, as_of=LATER)
    assert any("较新经济内容" in reason for reason in current["needed_review"])
    different_security = deepcopy(candidate)
    different_security["payload"]["security_id"] = "other-security"
    assert not _same_source_basis(cited, different_security)
    annual = deepcopy(candidate)
    annual["payload"]["metrics"][0]["period"]["basis"] = "year"
    assert not _same_source_basis(cited, annual)
    portfolio = {"product_id": PRODUCT.id, "family": "portfolio", "payload": {"denominator": "nav", "holdings_scope": "top10"}}
    assert not _same_source_basis(portfolio, {**portfolio, "payload": {"denominator": "equity_assets", "holdings_scope": "top10"}})
    assert not _same_source_basis(portfolio, {**portfolio, "payload": {"denominator": "nav", "holdings_scope": "full"}})
    changed = deepcopy(value)
    changed["research_date"] = "2026-10-28"
    changed["origin"]["generated_at"] = LATER
    changed["revision"] = {"id": "new-quarter-explained", "supersedes": first["id"]}
    changed["citations"][0].update(record_id=candidate["id"], capture_id=new["capture_ids"][0])
    changed["action_cases"]["held"].update(action="reduce", reason="合成新季度数据削弱原持有理由。")
    changed["change_evaluation"] = {"summary": "合成新季度削弱旧盈利依据，因此减少而非继续持有。", "fact_ids": ["f1"]}
    append_generated_research(store, PRODUCT, normalize_generated_research(changed), clock=clock(LATER, LATER))
    updated = build_research_thesis(store.advice_evidence(PRODUCT.id, LATER), product_id=PRODUCT.id, as_of=LATER)
    assert not updated["needed_review"] and updated["payload"]["action_cases"]["held"]["action"] == "reduce"
    # Another upstream's contradictory same-period report is not chosen by fetch time.
    conflicting = {key: item for key, item in candidate.items() if key != "id"}
    conflicting.update(source="independent-source", upstream="independent-source", source_url="https://third.example.com/report")
    conflicting["payload"] = deepcopy(candidate["payload"])
    conflicting["payload"]["metrics"][0]["value"] = "999"
    conflict = build_research_thesis([*store.advice_evidence(PRODUCT.id, LATER), normalize_record(conflicting)],
                                      product_id=PRODUCT.id, as_of=LATER)
    assert any("冲突" in reason for reason in conflict["needed_review"])


PUBLIC_PRODUCT = Product("open_fund:SYNTH:C", "SYNTH", "合成研究测试基金", "open_fund",
                         share_class="C", exchange="OTC")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from rmbfund.service import Workspace
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: LATER)
    value = Workspace(tmp_path / "synthetic-service", "public")
    value.store.add_product(PUBLIC_PRODUCT)
    value.store.initialize_evidence_schema()
    yield value
    value.store.close()


def _economic(family, payload, *, product=PUBLIC_PRODUCT, report_date="2026-09-30", at=GENERATED):
    return normalize_record({
        "product_id": product.id, "family": family, "source": "synthetic-online",
        "upstream": "synthetic-online", "source_url": "https://example.com/" + family,
        "report_date": report_date, "observed_at": at, "available_at": at, "payload": payload,
    })


def _holdings(weight="0.35"):
    return {"denominator": "nav", "assets": {"equity": "0.8"}, "holdings_scope": "partial",
            "holdings": [{"issuer_id": "synthetic-issuer", "name": "Synthetic issuer",
                          "kind": "corporate_equity", "weight": weight}],
            "countries": [], "sectors": [], "currencies": []}


def _retain(store, record):
    capture = store.record_advice_capture({
        **{key: record[key] for key in
           ("product_id", "family", "source", "upstream", "source_url", "observed_at")},
        "status": "ok", "content_type": "application/json", "raw_bytes": json.dumps(record).encode(),
        "parser_version": "synthetic-v1", "warnings": [],
    })
    return store.add_advice_evidence(record, capture, "json:/record")


def _basis_note(records, *, at=GENERATED, supersedes=None):
    identity = {key: getattr(PUBLIC_PRODUCT, key) for key in IDENTITY}
    value = research_output_example(product=identity, as_of=at)
    value["revision"] = {"id": at, "supersedes": supersedes}
    value["alternatives"] = [{"product_id": identifier, "reason": "合成实际候选依据。", "fact_ids": ["f1"]}
                             for identifier in sorted({row["product_id"] for row in records}
                                                      - {PUBLIC_PRODUCT.id})]
    documents, citations, links = [], [], []
    for index, row in enumerate(records):
        evidence_id, citation_id = f"doc-{index}", f"c{index + 1}"
        documents.append({"id": evidence_id, "product_id": row["product_id"],
            **{key: row[key] for key in ("source_url", "observed_at", "available_at", "published_at", "publication_precision")},
            "title": "Synthetic disclosure", "content_status": "full_text",
            "sections": [{"locator": "json:/record", "text": "Adopted economic basis."}]})
        citations.append({"id": citation_id, "kind": "online_evidence", "evidence_id": evidence_id,
                          "locator": "json:/record", "quote": "Adopted economic basis.", "claim": "合成披露依据。"})
        links.append({"record_id": row["id"], "evidence_id": evidence_id,
                      "locator": "json:/record", "quote": "Adopted economic basis."})
    value["citations"] = citations
    value["facts"][0]["citation_ids"] = [row["id"] for row in citations]
    return normalize_generated_research(value, evidence={"documents": documents, "records": records,
                                                        "record_links": links}, as_of=at)


def _empty_current(products, *, at=LATER, records=None):
    return {"schema_version": 1, "kind": "current_online_context", "status": "ready",
            "evaluated_at": at, "checked_at": at, "source_checks": [], "nav_by_product": {},
            "records_by_product": records or {}, "messages_by_product": {},
            "public_subjects": [{key: getattr(product, key) for key in IDENTITY} for product in products]}


def test_selected_v2_basis_drives_original_workspace_profile_portfolio_issuer_and_current(workspace):
    pid, store = PUBLIC_PRODUCT.id, workspace.store
    old = _retain(store, _economic("portfolio", _holdings("0.2"), report_date="2026-06-30", at=EARLY))
    v1 = _v1_note()
    v1["product"] = {key: getattr(PUBLIC_PRODUCT, key) for key in IDENTITY}
    prior = import_local_research_note(store, PUBLIC_PRODUCT, v1, clock=clock(EARLY, EARLY))["records"][0]
    historical = workspace.fund_detail(pid, EARLY)
    portfolio = _economic("portfolio", _holdings())
    financials = _economic("issuer_fundamentals", _release("2026-07-01", "2026-09-30", 3, "9"))
    note = _basis_note([portfolio, financials], supersedes=prior["id"])
    saved = append_generated_research(store, PUBLIC_PRODUCT, note, clock=clock())["records"][0]
    detail = workspace.fund_detail(pid, END)
    assert detail["research_thesis"]["record_id"] == saved["id"]
    assert detail["research_thesis"]["payload"]["origin"]["human_reviewed"] is False
    assert detail["investment_profile"]["portfolio"]["record_id"] == portfolio["id"]
    assert detail["portfolio"]["holdings"] == portfolio["payload"]["holdings"]
    assert detail["portfolio"]["available_at"] == GENERATED
    assert next(row for row in detail["background_records"] if row["family"] == "portfolio") == portfolio
    analysis = detail["issuer_analysis"]
    assert analysis == workspace.issuer_fundamentals(pid, END)
    assert analysis["profile_reference"]["portfolio_record_id"] == portfolio["id"]
    assert analysis["coverage"]["disclosed_corporate_equity_weight"] == "0.35"
    assert analysis["issuers"][0]["quarter_actuals"][0]["record_id"] == financials["id"]
    assert analysis["issuers"][0]["disclosed_exposure"]["record_id"] == portfolio["id"]
    projected = workspace.current_evaluation(pid, _empty_current([PUBLIC_PRODUCT], at=END))["details"][pid]
    assert projected["investment_profile"] == detail["investment_profile"]
    assert projected["issuer_analysis"] == analysis and projected["portfolio"] == detail["portfolio"]
    # A later thin parse of the very same report must not discard the adopted holdings.
    thin = _economic("portfolio", {**_holdings(), "holdings": []}, at=LATER)
    current = workspace.current_evaluation(pid, _empty_current([PUBLIC_PRODUCT], records={pid: [thin]}))["details"][pid]
    assert current["investment_profile"]["portfolio"]["record_id"] == portfolio["id"]
    assert current["portfolio"] == detail["portfolio"]
    assert current["issuer_analysis"]["coverage"]["disclosed_corporate_equity_weight"] == "0.35"
    assert workspace.fund_detail(pid, EARLY) == historical
    assert historical["portfolio"]["report_date"] == old["report_date"] == "2026-06-30"
    assert store.advice_evidence(pid, LATER)[0]["id"] == old["id"]
    assert {row["family"] for row in store.advice_evidence(pid, LATER)} == {"portfolio", "research_thesis"}
    assert store.advice_evidence(pid, LATER)[-1]["payload"] == saved["payload"] == note


def test_basis_uses_only_selected_head_and_original_availability(workspace):
    old = _economic("portfolio", _holdings("0.2"), report_date="2026-06-30", at=EARLY)
    new = _economic("portfolio", _holdings())
    first = append_generated_research(workspace.store, PUBLIC_PRODUCT, _basis_note([old], at=EARLY),
                                      clock=clock(EARLY, EARLY))["records"][0]
    latest = append_generated_research(workspace.store, PUBLIC_PRODUCT, _basis_note([new], supersedes=first["id"]),
                                       clock=clock())["records"][0]
    records = workspace.store.advice_evidence(PUBLIC_PRODUCT.id, LATER)
    def basis(rows, at=END):
        thesis = build_research_thesis(rows, product_id=PUBLIC_PRODUCT.id, as_of=at)
        return with_research_basis([], thesis, product_id=PUBLIC_PRODUCT.id, as_of=at)
    assert basis(records, EARLY) == [old]
    assert basis(records) == [new]
    assert basis([latest]) == []  # Orphan successor is contested, not an adopted source.
    independent = deepcopy(first)
    independent.pop("id")
    independent["payload"]["note_id"] = "independent"
    independent["source_url"] = research_thesis_source_uri(PUBLIC_PRODUCT.id, independent["payload"])
    assert basis([first, latest, normalize_record(independent)]) == []
    selected = build_research_thesis(records, product_id=PUBLIC_PRODUCT.id, as_of=END)
    assert with_research_basis([], selected, product_id=PUBLIC_PRODUCT.id, as_of=EARLY) == []
    future = deepcopy(selected)
    future["payload"]["citations"][0]["record_basis"][0]["available_at"] = LATER
    assert with_research_basis([], future, product_id=PUBLIC_PRODUCT.id, as_of=END) == []
    unadopted = deepcopy(selected)
    unadopted["payload"]["citations"][0]["record_ids"] = []
    assert with_research_basis([], unadopted, product_id=PUBLIC_PRODUCT.id, as_of=END) == []
    assert with_research_basis([], selected, product_id="open_fund:OTHER:C", as_of=END) == []


def test_target_adopted_alternatives_feed_current_and_stored_comparisons_only(workspace):
    peers = [Product(f"open_fund:PEER:{share}", "PEER", f"Synthetic peer {share}", "open_fund",
                     share_class=share, exchange="OTC") for share in ("A", "C")]
    fees = {"management_rate": "0.01", "custody_rate": "0.001", "sales_service_rate": "0.004",
            "redemption_bands": [], "holding_day_basis": "unknown", "channel_verified": False}
    adopted = []
    for peer in peers:
        workspace.store.add_product(peer)
        for family, payload in (("portfolio", _holdings("0.2")), ("fees", fees)):
            _retain(workspace.store, _economic(family, payload, product=peer, report_date="2026-06-30", at=EARLY))
        adopted.extend([_economic("portfolio", _holdings(), product=peer),
                        _economic("fees", {**fees, "sales_service_rate": "0" if peer.share_class == "A" else "0.003"},
                                  product=peer, report_date="2026-10-27")])
    append_generated_research(workspace.store, PUBLIC_PRODUCT, _basis_note(adopted), clock=clock())
    peer_set = {"target_product_id": PUBLIC_PRODUCT.id, "selection_reason": "Synthetic alternatives",
                "evidence_refs": ["synthetic:comparison"], "peers": [{"product_id": peer.id,
                "relation": "style_or_cost_alternative", "selection_reason": "Synthetic disclosed fees",
                "evidence_refs": ["synthetic:comparison"]} for peer in peers]}
    context = _empty_current([*peers, PUBLIC_PRODUCT])  # Target need not come first.
    evaluation = workspace.current_evaluation(PUBLIC_PRODUCT.id, context, peer_set=peer_set)
    stored_comparison = workspace.compare_funds(PUBLIC_PRODUCT.id, peer_set=peer_set, as_of=LATER)
    for peer in peers:
        detail = evaluation["details"][peer.id]
        assert detail["portfolio"]["report_date"] == "2026-09-30"
        assert detail["investment_profile"]["portfolio"]["payload"]["holdings"] == _holdings()["holdings"]
        current_fee = next(row for row in adopted if row["product_id"] == peer.id and row["family"] == "fees")
        for comparison in (evaluation["fund_comparison"], stored_comparison):
            fund = next(row for row in comparison["funds"] if row["product"]["id"] == peer.id)
            assert fund["fees"]["as_of"] == "2026-10-27"
            assert current_fee["id"] in fund["fees"]["evidence_refs"]
        # No global adoption into the candidate's own page or economic tables.
        assert workspace.fund_detail(peer.id, LATER)["portfolio"]["report_date"] == "2026-06-30"
        assert all(row["report_date"] == "2026-06-30" for row in workspace.store.advice_evidence(peer.id, LATER))
    historical = workspace.compare_funds(PUBLIC_PRODUCT.id, peer_set=peer_set, as_of=EARLY)
    assert all(row["fees"]["as_of"] == "2026-06-30" for row in historical["funds"]
               if row["product"]["id"] != PUBLIC_PRODUCT.id)


def test_share_class_cost_display_uses_adopted_standard_fees_not_stale_unknowns(workspace, monkeypatch):
    from contextlib import nullcontext
    from rmbfund.buyer_research import comparison_fund
    from rmbfund.ui import buyer

    peer = Product("open_fund:PEER:A", "PEER", "Synthetic peer A", "open_fund", share_class="A", exchange="OTC")
    workspace.store.add_product(peer)
    fees = {"management_rate": "0.01", "custody_rate": "0.001", "sales_service_rate": "0.004",
            "subscription_rate": "0", "redemption_bands": [{"min_days": 0, "max_days": None, "rate": "0"}],
            "holding_day_basis": "unknown", "channel_verified": False}
    target_fee = _economic("fees", fees)
    peer_fee = _economic("fees", {**fees, "subscription_rate": "0.016", "sales_service_rate": "0",
        "redemption_bands": [{"min_days": 0, "max_days": 6, "rate": "0.015"},
                             {"min_days": 7, "max_days": None, "rate": "0"}]}, product=peer)
    append_generated_research(workspace.store, PUBLIC_PRODUCT, _basis_note([target_fee, peer_fee]), clock=clock())
    target = workspace.fund_detail(PUBLIC_PRODUCT.id, END)
    candidate = workspace.fund_detail(peer.id, END, adopted_research=target["research_thesis"])
    metadata = {"underlying_fund_id": "same-synthetic-underlying"}
    value = {"funds": [comparison_fund(detail, metadata) for detail in (target, candidate)],
             "comparisons": [{"target_product_id": PUBLIC_PRODUCT.id, "product_id": peer.id,
                              "selection_reason": "A申赎阶梯及渠道折扣缺失，不断言哪类更便宜。"}]}
    projected_fee = value["funds"][1]["fees"]
    assert projected_fee["subscription_rate"] == "0.016"
    assert projected_fee["redemption_bands"] == peer_fee["payload"]["redemption_bands"]
    assert projected_fee["available_at"] == peer_fee["available_at"]
    before = deepcopy(value)
    shown = []
    monkeypatch.setattr(buyer, "_write", shown.append)
    for method in ("subheader", "markdown", "caption", "json"):
        monkeypatch.setattr(buyer.st, method, lambda *args, **kwargs: None)
    monkeypatch.setattr(buyer.st, "expander", lambda *args, **kwargs: nullcontext())
    buyer.render_comparison(value)
    text = "\n".join(shown)
    assert "公示标准申购费率 1.60%" in text and "公示赎回费分段：" in text
    assert "0～6天（含） 1.50%" in text and "公示标准申购费率 0.00%" in text
    assert "支付宝等渠道的实际申购折扣及适用赎回费仍未核实" in text
    assert "持有天数计算口径尚未核实" in text
    assert "A申赎阶梯及渠道折扣缺失" not in text
    assert value == before
    annual_only = deepcopy(value)
    for fund in annual_only["funds"]:
        fund["fees"].pop("subscription_rate")
        fund["fees"]["redemption_bands"] = []
    assert buyer._share_class_cost_basis(annual_only, annual_only["comparisons"][0]) == []
    other_fund = deepcopy(value)
    other_fund["funds"][1]["product"]["underlying_fund_id"] = "different-underlying"
    assert buyer._share_class_cost_basis(other_fund, other_fund["comparisons"][0]) == []

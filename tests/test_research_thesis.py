"""Research evidence contracts, with no network, app, or SQLite connections."""

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy

import pytest

from rmbfund.advice_evidence import (
    FAMILIES,
    normalize_capture,
    normalize_record,
    normalize_research_thesis,
    normalize_source_url,
    research_thesis_source_uri,
)
from rmbfund.domain import Product, ValidationError, digest, timestamp
from rmbfund.providers.advice_evidence import (
    import_local_research_note,
    import_reviewed_document,
)
from rmbfund.research_thesis import build_research_thesis

PRODUCT = Product("public:test-fund:C", "012345", "Contract Test Fund C", "open_fund",
                  exchange="OTC", share_class="C")
EARLY = "2026-10-20T10:00:00+08:00"
START = "2026-10-26T11:00:00+08:00"
END = "2026-10-26T11:01:00+08:00"
LATER = "2026-10-27T11:00:00+08:00"


def _clock(start=START, end=END):
    return iter((start, end)).__next__


def _note():
    return {
        "product": {key: getattr(PRODUCT, key) for key in
                    ("id", "symbol", "name", "kind", "share_class", "exchange")},
        "note_id": "emerging-markets-thesis", "author": "Research author",
        "reviewer": "Research reviewer", "research_date": "2026-10-25",
        "horizon": {"start": "2026-10-25", "end": "2027-04-25"}, "posture": "maintain",
        "facts": [{
            "id": "fact-1", "claim": "The announcement reports a change in demand.",
            "status": "factual", "effective_date": "2026-10-24", "issuer_ids": ["issuer-1"],
            "channels": ["earnings", "valuation"], "citation_ids": ["citation-1"],
        }],
        "citations": [{
            "id": "citation-1", "kind": "reviewed_excerpt",
            "publisher_url": "https://example.com/announcement", "title": "Announcement",
            "publication_precision": "date", "published_at": "2026-10-24",
            "checked_at": "2026-10-25T12:00:00+08:00", "locator": "section:outlook",
            "claim": "Reported demand improved.", "excerpt": "Reported demand improved.",
        }],
        "assessment": {
            "summary": "Maintain the research posture pending additional evidence.",
            "supporting": [{"reason": "Demand may support issuer earnings.", "fact_ids": ["fact-1"]}],
            "counterevidence": [],
            "priced_in": {"status": "unknown", "reason": "No price-reaction analysis is available."},
            "missing": ["Current complete fund exposures are not known."],
            "falsifiers": ["Subsequent orders contradict the reported demand improvement."],
            "change_conditions": ["Reassess if demand or currency transmission changes."],
            "review_date": "2026-11-05",
        },
        "revision": {"id": "original", "supersedes": None},
    }


def _record(payload=None, *, available=END):
    payload = _note() if payload is None else payload
    return normalize_record({
        "product_id": PRODUCT.id, "family": "research_thesis",
        "source": "local_authored_research", "upstream": "local_authored_research",
        "source_url": research_thesis_source_uri(PRODUCT.id, payload),
        "report_date": payload["research_date"], "observed_at": START,
        "available_at": available, "payload": payload,
    })


class MemoryEvidenceStore:
    """The existing evidence write/read protocol, without a database connection."""

    def __init__(self):
        self.product = PRODUCT
        self.records = {}
        self.captures = {}
        self.links = []
        self.schema = True

    def get_product(self, product_id):
        return self.product if product_id == self.product.id else None

    def has_fund_review_schema(self):
        return self.schema

    @contextmanager
    def transaction(self):
        before = deepcopy((self.records, self.captures, self.links))
        try:
            yield
        except BaseException:
            self.records, self.captures, self.links = before
            raise

    def record_advice_capture(self, capture):
        value = normalize_capture(capture)
        self.captures[value["id"]] = value
        return value["id"]

    def add_advice_evidence(self, record, capture_id, locator):
        value = normalize_record(record)
        capture = self.captures[capture_id]
        assert all(value[key] == capture[key] for key in
                   ("product_id", "family", "source", "upstream", "source_url"))
        assert timestamp(value["available_at"]) >= timestamp(capture["available_at"])
        self.records.setdefault(value["id"], value)
        self.links.append({
            "record_id": value["id"], "capture_id": capture_id, "locator": locator,
            "linked_at": value["available_at"],
        })
        return deepcopy(self.records[value["id"]])

    def advice_evidence(self, product_id, as_of):
        return [deepcopy(value) for value in self.records.values()
                if value["product_id"] == product_id
                and timestamp(value["available_at"]) <= timestamp(as_of)]

    def advice_evidence_links(self, product_id, record_id, as_of):
        if record_id not in {row["id"] for row in self.advice_evidence(product_id, as_of)}:
            return []
        return [deepcopy(link) for link in self.links
                if link["record_id"] == record_id
                and timestamp(link["linked_at"]) <= timestamp(as_of)]


def _legacy_inputs():
    return {
        "mandate": {
            "symbol": "012345", "share_class": "C", "name": "Contract Test Fund C",
            "fund_id": "fund-1", "base_currency": "CNY", "asset_class": "equity",
            "principal_protected": False, "sector_mandate": False, "single_country_mandate": False,
            "benchmark": {"id": "bench", "name": "Benchmark", "equity_weight": "1",
                          "region": "global", "currency": "USD"},
        },
        "portfolio": {"denominator": "nav", "assets": {"equity": "0.8"},
                      "holdings_scope": "partial", "holdings": [], "countries": [],
                      "sectors": [], "currencies": []},
        "management": {"managers": [{"name": "Manager", "start_date": "2025-01-01"}]},
        "scale": {"total_assets_cny": "100", "combined_share_classes": True},
        "fees": {"management_rate": "0.01", "custody_rate": "0.001",
                 "sales_service_rate": "0.004", "redemption_bands": [],
                 "holding_day_basis": "unknown", "channel_verified": False},
        "operations": {"subscription_state": "open", "redemption_state": "open", "channel_verified": False},
        "events": {"events": [{"event_id": "e1", "type": "other", "effective_date": "2026-10-20",
                               "official": True, "reassessed": False, "description": "Announcement"}]},
        "valuation": {"benchmark_id": "b1", "region": "global", "pe_ttm": "20", "pb": "2",
                      "pe_definition": "trailing", "pb_definition": "reported", "compatible_scope": True},
        "macro": {"kind": "yield", "series_id": "DGS10", "currency": "USD", "unit": "percent",
                  "tenor_years": "10", "observations": [{"date": "2026-10-20", "value": "4"}]},
        "issuer_fundamentals": {
            "kind": "security_mapping", "issuer_id": "issuer-1", "issuer_name": "Issuer One",
            "security_id": "security-1", "symbol": "ONE", "exchange": "NYSE",
            "quote_currency": "USD", "share_basis": "ordinary", "locator": "json:/mapping",
        },
    }


def test_existing_family_identity_algorithm_and_public_url_contract_are_unchanged():
    frozen_ids = {
        "mandate": "b4e46a70e8bf87874aa85ca6fe173869188f7c52088c6bcadbb2c3541ec45f97",
        "portfolio": "3577539e8fefe9b6f9755efffc0809978d3cb26ca27bd4f20690e795065aff90",
        "management": "50e95a0973c883b6b83d93b05b96838b3f569f6e2ca36d9d113d708a78d21b81",
        "scale": "01f9a24681a071e6d4aba3e7172547ef39a8d481b81d1decb29699d9b3a37592",
        "fees": "4f4e85b5ec728398e710ad6390b4478be25fb518c4ea2d316d097614cb5a4d2f",
        "operations": "1108efd0bf6bfa9a98897850d71e0e7cddd9a1be251b572c009aa1597e673a2c",
        "events": "44e9d4b892a96091d904e7aa1b109dc9e2559f8d5327ae9b62f925cb4ef181a8",
        "valuation": "1d2cea7f2709fa6a5e2e3d6a2a512ca2fcec7ad440d25257d35f79d9cee04ce0",
        "macro": "d5000ce6af422aad741b72db636541d67f43780fff837f68a7f82ff2dcf0f171",
        "issuer_fundamentals": "ed119f8f93f48e702110bd477bb828aeaf8c96897abed09f056663be174372af",
    }
    payloads = _legacy_inputs()
    assert FAMILIES == set(payloads) | {"research_thesis"}
    for family, payload in payloads.items():
        original = {
            "product_id": PRODUCT.id, "family": family, "source": "source", "upstream": "publisher",
            "source_url": "https://example.com/document", "report_date": "2026-10-20",
            "observed_at": START, "available_at": END, "payload": payload,
        }
        # The literal pre-thesis v1 identity recipe, including all existing null
        # fields and public URL, freezes every legacy family, not just one fixture.
        legacy_identity = {key: value for key, value in original.items()
                           if key not in {"observed_at", "available_at"}}
        legacy_identity.update(schema_version=1, normalization_version="1", period_start=None,
                               effective_from=None, effective_to=None, published_at=None,
                               publication_precision="unknown")
        record = normalize_record(original)
        assert record["id"] == digest(legacy_identity) == frozen_ids[family]
        legacy_capture = {
            "product_id": PRODUCT.id, "family": family, "source": "source", "upstream": "publisher",
            "source_url": "https://example.com/document", "observed_at": START, "status": "ok",
            "content_type": "application/json", "raw_bytes": b'{"retained":"public source"}',
            "parser_version": "legacy-parser-v1", "warnings": [],
        }
        captured = normalize_capture(legacy_capture)
        capture_identity = {key: value for key, value in legacy_capture.items() if key != "raw_bytes"}
        capture_identity.update(metadata={}, available_at=START,
                                raw_sha256=hashlib.sha256(legacy_capture["raw_bytes"]).hexdigest())
        assert captured["id"] == digest(capture_identity)
        original["source_url"] = "rmb-research://local/" + "a" * 64
        with pytest.raises(ValidationError):
            normalize_record(original)
    with pytest.raises(ValidationError):
        normalize_source_url(_record()["source_url"])


def test_research_uri_is_exact_content_bound_and_operational_clock_independent():
    first = _record()
    assert first["source_url"].startswith("rmb-research://local/")
    later_note = _note()
    later_note["citations"][0]["checked_at"] = LATER
    later = _record(later_note, available=LATER)
    assert later["source_url"] == first["source_url"]
    assert later["id"] == first["id"]
    changed = _note()
    changed["assessment"]["summary"] += " Changed reasoning."
    assert _record(changed)["source_url"] != first["source_url"]
    for url in ("https://example.com/note", "rmb-research://LOCAL/" + "a" * 64,
                "rmb-research://local/" + "A" * 64, first["source_url"] + "?x=1",
                first["source_url"] + "#locator", "rmb-research://local/" + "a" * 64):
        bad = {key: value for key, value in first.items() if key != "id"}
        bad["source_url"] = url
        with pytest.raises(ValidationError):
            normalize_record(bad)


@pytest.mark.parametrize("mutate", [
    lambda n: n.update(personal_holdings={"units": 1}),
    lambda n: n["assessment"].update(target="0.2"),
    lambda n: n["assessment"].update(score=99),
    lambda n: n["assessment"].update(probability=float("nan")),
    lambda n: n["facts"][0].update(status="forecast"),
    lambda n: n["facts"][0].update(effective_date="2027-01-01"),
    lambda n: n["facts"][0].update(citation_ids=["nonexistent"]),
    lambda n: n["facts"][0].update(channels=[]),
    lambda n: n["assessment"]["supporting"][0].update(fact_ids=["nonexistent"]),
    lambda n: n["assessment"]["priced_in"].update(status="yes"),
    lambda n: n["assessment"].update(falsifiers=[]),
    lambda n: n["assessment"].update(change_conditions=[]),
    lambda n: n["assessment"].update(review_date="2026-10-01"),
    lambda n: n["citations"][0].update(excerpt_sha256="a" * 64),
    lambda n: n["citations"][0].update(publication_precision="unknown"),
    lambda n: n["citations"][0].update(excerpt="x" * 2001),
    lambda n: n.update(facts=n["facts"] * 65),
    lambda n: n.update(author="x" * 201),
])
def test_strict_payload_rejects_personal_numeric_and_ambiguous_research_fields(mutate):
    note = _note()
    mutate(note)
    with pytest.raises(ValidationError):
        normalize_research_thesis(note)


def test_url_only_and_manual_excerpt_have_explicit_different_limitations():
    note = _note()
    citation = note["citations"][0]
    citation.pop("excerpt")
    citation["kind"] = "linked_not_archived"
    with pytest.raises(ValidationError, match="只有URL"):
        normalize_research_thesis(note)
    note["facts"][0]["status"] = "unresolved"
    with pytest.raises(ValidationError, match="未确证"):
        normalize_research_thesis(note)
    note["posture"] = "unresolved"
    assert normalize_research_thesis(note)["facts"][0]["status"] == "unresolved"
    note["citations"][0]["excerpt"] = "Cannot label entered text a URL-only reference."
    with pytest.raises(ValidationError):
        normalize_research_thesis(note)
    reviewed = normalize_research_thesis(_note())["citations"][0]
    assert reviewed["excerpt_sha256"] == hashlib.sha256(reviewed["excerpt"].encode()).hexdigest()


def test_local_import_retains_actual_clocks_raw_bytes_first_availability_and_revisions():
    store = MemoryEvidenceStore()
    note = _note()
    raw = json.dumps(note, ensure_ascii=False, indent=2).encode()
    before = deepcopy(note)
    first = import_local_research_note(store, PRODUCT, raw, clock=_clock())
    record = first["records"][0]
    capture = store.captures[first["capture_ids"][0]]
    assert note == before
    assert record["observed_at"] == START and record["available_at"] == END
    assert record["report_date"] == "2026-10-25"
    assert record["source"] == "local_authored_research"
    assert capture["raw_bytes"] == raw
    assert capture["raw_sha256"] == first["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert capture["metadata"]["reviewed_at"] == END
    assert capture["source_url"] == record["source_url"]
    rechecked = deepcopy(note)
    rechecked["citations"][0]["checked_at"] = LATER
    again = import_local_research_note(store, PRODUCT, rechecked, clock=_clock(LATER, LATER))
    assert again["status"] == "existing"
    assert again["records"][0] == record
    assert again["captures_added"] == again["records_added"] == 0
    assert len(store.captures) == len(store.records) == 1
    revised = deepcopy(note)
    revised["revision"] = {"id": "revision-2", "supersedes": record["id"]}
    revised["assessment"]["summary"] = "Trim posture after a changed interpretation."
    revised["posture"] = "trim"
    second = import_local_research_note(store, PRODUCT, revised, clock=_clock(LATER, LATER))
    assert second["records_added"] == second["captures_added"] == 1
    assert second["records"][0]["id"] != record["id"]
    assert second["records"][0]["payload"]["revision"]["supersedes"] == record["id"]
    assert len(store.records) == 2
    historical = build_research_thesis(list(store.records.values()), product_id=PRODUCT.id, as_of=END)
    assert historical["record_id"] == record["id"]
    current = build_research_thesis(list(store.records.values()), product_id=PRODUCT.id, as_of=LATER)
    assert current["payload"]["posture"] == "trim"
    assert not current["needed_review"]
    assert any("人工录入摘录" in warning for warning in current["warnings"])
    assert current["source"]["source"] == "local_authored_research"
    assert build_research_thesis([record], product_id=PRODUCT.id, as_of=START)["status"] == "missing"


def _seed_archived(store):
    return import_reviewed_document(
        store, PRODUCT, raw_bytes=b"%PDF-1.4\nContract source\n%%EOF",
        source_url="https://example.com/source.pdf", reviewed=True, clock=_clock(EARLY, EARLY),
        records=[{"family": "scale", "report_date": "2026-10-20",
                  "payload": {"total_assets_cny": "100", "combined_share_classes": True},
                  "locator": "page:1;section:assets"}],
    )


def test_archived_citations_require_existing_product_capture_locator_and_as_of():
    store = MemoryEvidenceStore()
    archived = _seed_archived(store)
    note = _note()
    note["citations"] = [{
        "id": "citation-1", "kind": "archived_evidence", "record_id": archived["record_ids"][0],
        "capture_id": archived["capture_ids"][0], "locator": "page:1;section:assets",
        "claim": "Archived source states the assets.",
    }]
    result = import_local_research_note(store, PRODUCT, note, clock=_clock())
    assert result["records_added"] == 1
    for field, bad in (("record_id", "a" * 64), ("capture_id", "b" * 64),
                       ("locator", "page:2;section:assets")):
        changed = deepcopy(note)
        changed["citations"][0][field] = bad
        with pytest.raises(ValidationError):
            import_local_research_note(store, PRODUCT, changed, clock=_clock())
    store.links[0]["linked_at"] = LATER
    with pytest.raises(ValidationError, match="链接"):
        import_local_research_note(store, PRODUCT, note, clock=_clock())
    store.records[archived["record_ids"][0]]["available_at"] = LATER
    with pytest.raises(ValidationError, match="已可用"):
        import_local_research_note(store, PRODUCT, note, clock=_clock())


def test_import_rejects_identity_backclock_fabricated_revision_and_public_import_conflation():
    store = MemoryEvidenceStore()
    for field, value in (("id", "other-product"), ("symbol", "987654"),
                         ("share_class", "A"), ("exchange", "OTHER")):
        note = _note()
        note["product"][field] = value
        with pytest.raises(ValidationError, match="身份"):
            import_local_research_note(store, PRODUCT, note, clock=_clock())
    with pytest.raises(ValidationError, match="完成时钟"):
        import_local_research_note(store, PRODUCT, _note(), clock=_clock(END, START))
    assert not store.records and not store.captures
    note = _note()
    note["revision"]["supersedes"] = "a" * 64
    with pytest.raises(ValidationError, match="研究修订"):
        import_local_research_note(store, PRODUCT, note, clock=_clock())
    first = import_local_research_note(store, PRODUCT, _note(), clock=_clock())
    changed = _note()
    changed["assessment"]["summary"] += " Changed."
    with pytest.raises(ValidationError, match="修订标识"):
        import_local_research_note(store, PRODUCT, changed, clock=_clock(LATER, LATER))
    assert len(store.records) == len(store.captures) == 1
    with pytest.raises(ValidationError, match="单独导入"):
        import_reviewed_document(
            store, PRODUCT, raw_bytes=b'{"note":"not a publisher original"}',
            source_url="https://example.com/note", reviewed=True, clock=_clock(),
            records=[{"family": "research_thesis", "report_date": "2026-10-25",
                      "payload": _note(), "locator": "json:/note"}],
        )
    assert first["records_added"] == 1


def test_raw_note_json_finite_duplicates_and_capture_uri_are_checked():
    store = MemoryEvidenceStore()
    for raw in (b'{"note_id":"first","note_id":"duplicate"}', b'{"score":NaN}',
                b'{"score":1e999}', b'{"a":Infinity}', b'\xff', b'[]'):
        with pytest.raises(ValidationError):
            import_local_research_note(store, PRODUCT, raw, clock=_clock())
    first = import_local_research_note(store, PRODUCT, _note(), clock=_clock())
    capture = deepcopy(store.captures[first["capture_ids"][0]])
    for key in ("id", "raw_sha256", "available_at"):
        capture.pop(key)
    bad = deepcopy(capture)
    bad["source_url"] = "rmb-research://local/" + "a" * 64
    with pytest.raises(ValidationError):
        normalize_capture(bad)
    legacy = deepcopy(capture)
    legacy["family"] = "scale"
    with pytest.raises(ValidationError):
        normalize_capture(legacy)
    capture["metadata"].pop("reviewed_at")
    with pytest.raises(ValidationError, match="实际取得"):
        normalize_capture(capture)


def test_new_archived_economic_basis_requires_review_but_refetches_and_other_news_do_not():
    store = MemoryEvidenceStore()
    archived = _seed_archived(store)
    note = _note()
    note["citations"] = [{
        "id": "citation-1", "kind": "archived_evidence", "record_id": archived["record_ids"][0],
        "capture_id": archived["capture_ids"][0], "locator": "page:1;section:assets",
        "claim": "Archived source states the assets.",
    }]
    thesis = import_local_research_note(store, PRODUCT, note, clock=_clock())["records"][0]
    cited = archived["records"][0]
    records = [cited, thesis]
    assert not build_research_thesis(records, product_id=PRODUCT.id, as_of=LATER)["needed_review"]
    rechecked = {**cited, "checked_at": LATER}
    assert not build_research_thesis([rechecked, thesis], product_id=PRODUCT.id, as_of=LATER)["needed_review"]
    updated = {key: value for key, value in cited.items() if key != "id"}
    updated.update(observed_at=LATER, available_at=LATER)
    updated["payload"] = {"total_assets_cny": "80", "combined_share_classes": True}
    revised = normalize_record(updated)
    current = build_research_thesis([*records, revised], product_id=PRODUCT.id, as_of=LATER)
    assert len(current["needed_review"]) == 1
    assert "较新经济内容" in current["needed_review"][0]
    historical = build_research_thesis([*records, revised], product_id=PRODUCT.id, as_of=END)
    assert not historical["needed_review"]
    unrelated = {**updated, "family": "events", "payload": {"events": [
        {"event_id": "unrelated", "type": "other", "effective_date": "2026-10-27",
         "official": True, "reassessed": False, "description": "Unrelated news"},
    ]}}
    assert not build_research_thesis([*records, normalize_record(unrelated)],
                                     product_id=PRODUCT.id, as_of=LATER)["needed_review"]


def test_planned_facts_and_missing_data_are_information_not_automatic_review_gates():
    note = _note()
    note["facts"][0].update(status="plan", effective_date="2027-01-01")
    projected = build_research_thesis([_record(note)], product_id=PRODUCT.id, as_of=END)
    assert not projected["needed_review"]
    assert any("计划或未确证" in warning for warning in projected["warnings"])
    assert any("资料缺口" in warning for warning in projected["warnings"])


def test_projection_exposes_review_due_contested_heads_and_never_selects_future_links():
    record = _record()
    due = build_research_thesis([record], product_id=PRODUCT.id, as_of="2026-11-06T12:00:00+08:00")
    assert due["status"] == "available" and "review_date_elapsed" in due["needed_review"]
    checked_future = {**record, "checked_at": LATER}
    assert build_research_thesis([checked_future], product_id=PRODUCT.id, as_of=END)["status"] == "missing"
    other_note = _note()
    other_note["note_id"] = "separate-independent-thesis"
    other = _record(other_note)
    contested = build_research_thesis([record, other], product_id=PRODUCT.id, as_of=END)
    assert contested["status"] == "contested" and contested["payload"] is None
    assert "multiple_visible_thesis_heads" in contested["needed_review"]

"""Offline tests only: injected public bytes, never live source recovery."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pytest

from rmbfund.domain import ValidationError
from rmbfund.providers.followed import FetchResponse, refresh_followed, review_followed_capture

CLOCK = "2026-10-26T12:00:00+08:00"
SUBJECT = {
    "product_id": "open_fund:123456:A", "symbol": "123456", "name": "公开测试基金",
    "share_class": "A", "currency": "CNY", "public": True,
}


def _millis(day: str) -> int:
    return int(datetime.fromisoformat(day + "T00:00:00+08:00").timestamp() * 1000)


def _body(symbol: str = "123456", *, rows: list | None = None) -> bytes:
    rows = rows if rows is not None else [
        {"x": _millis("2026-10-22"), "y": 1.25},
        {"x": _millis("2026-10-23"), "y": 1.3},
    ]
    return (f'fS_code = "{symbol}";\nData_netWorthTrend = ' + json.dumps(rows) + ";\n").encode()


def _job(identifier="nav-123456", *, family="nav", parser="eastmoney_nav_js_v1", url=None,
         config=None):
    return {
        "job_id": identifier, "product_id": SUBJECT["product_id"], "family": family,
        "source_url": url or "https://public.example.org/123456.js", "parser": parser,
        "config": config if config is not None else {"start": "2026-10-22", "end": "2026-10-23"},
    }


def _refresh(tmp_path, jobs, *, families=("nav",), budget=20, fetch=None, previous=None,
             clock=None, blocked=(), max_bytes=8 * 1024 * 1024):
    return refresh_followed(
        [SUBJECT], list(families), jobs, budget, tmp_path / "run",
        fetch=fetch or (lambda *_: FetchResponse(200, _body())), clock=clock or (lambda: CLOCK),
        previous_hashes=previous, blocked_urls=blocked, max_bytes=max_bytes,
    )


def _review_config(raw, text, *, pdf=False):
    low = 0 if pdf else text.index("capacity")
    quote = text if pdf else "capacity plans"
    payload = {"event": "capacity_plan", "interpretation_status": "review_needed"}
    if pdf:
        payload = {"family": "portfolio", "data": {"denominator": "nav",
                   "assets": {"equity": "0.1"}, "holdings_scope": "top10", "holdings": [],
                   "countries": [], "sectors": [], "currencies": []}, "evidence_refs": ["ref"]}
    config = {
        "reviewed": True, "reviewed_at": CLOCK, "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "extracts": [{"start": low, "end": low + len(quote), "quote": quote,
                      "locator": "page:45;table:7.4;row:1" if pdf else "html:body:span",
                      "report_date": "2026-06-30" if pdf else "2026-10-21",
                      "published_at": "2026-08-31" if pdf else "2026-10-21", "payload": payload}],
    }
    if pdf:
        config.update(review_text=text, review_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    return config


def test_parsed_nav_and_unparsed_pdf_are_independent(tmp_path):
    calls = []

    def fetch(url, timeout, max_bytes):
        calls.append(url)
        return FetchResponse(200, b"%PDF-1.7 inert\n%%EOF" if url.endswith(".pdf") else _body())

    result = _refresh(tmp_path, [
        _job(), _job("report", family="disclosure", parser="capture_only_v1",
                     url="https://official.example.org/report.pdf", config={}),
    ], families=("nav", "disclosure"), budget=2, fetch=fetch)
    nav, report = result["outcomes"]
    assert len(calls) == result["requests_made"] == 2
    assert result["implicit_retries"] == result["implicit_fallbacks"] == 0
    assert nav["status"] == "ok" and nav["parse_status"] == "parsed"
    assert [(row["date"], row["value"]) for row in nav["records"]] == [
        ("2026-10-22", "1.25"), ("2026-10-23", "1.3"),
    ]
    assert nav["records"][0]["published_at"] is None
    assert nav["records"][0]["basis"] == "unit_nav"
    assert report["status"] == "captured_unparsed" and report["records"] == []
    assert [(item["family"], item["status"], item["record_count"])
            for item in result["family_outcomes"]] == [
                ("nav", "ok", 2), ("disclosure", "captured_unparsed", 0),
            ]
    assert (tmp_path / "run" / "report" / "response.bin").read_bytes().startswith(b"%PDF-")
    receipt = json.loads((tmp_path / "run" / "nav-123456" / "receipt.json").read_text())
    assert receipt["request_count"] == receipt["attempt"] == 1


@pytest.mark.parametrize("same", [True, False])
def test_previous_content_hash_change_is_explicit(tmp_path, same):
    prior = hashlib.sha256(_body() if same else b"older response").hexdigest()
    result = _refresh(tmp_path, [_job()], previous={"nav-123456": prior}, budget=1)
    assert result["outcomes"][0]["changed"] is (not same)
    assert result["requests_made"] == 1


@pytest.mark.parametrize("failure", [403, 429, "tls"])
def test_denial_or_tls_stops_host_but_not_other_families(tmp_path, failure):
    calls = []

    def fetch(url, *_):
        calls.append(url)
        if "official.example.org" in url:
            if failure == "tls":
                raise OSError("certificate verification failed")
            return FetchResponse(failure, b"denied")
        return FetchResponse(200, _body())

    result = _refresh(tmp_path, [
        _job("news", family="policy_news", parser="capture_only_v1",
             url="https://official.example.org/first", config={}),
        _job("same-host", family="policy_news", parser="capture_only_v1",
             url="https://official.example.org/second", config={}),
        _job(),
    ], families=("nav", "policy_news"), fetch=fetch)
    assert len(calls) == result["requests_made"] == 2
    assert [o["status"] for o in result["outcomes"]] == ["error", "stopped", "ok"]


def test_real_completion_clocks_are_not_nav_dates(tmp_path):
    start = datetime.fromisoformat(CLOCK)
    tick = iter((start + timedelta(seconds=i)).isoformat() for i in range(10))
    result = _refresh(tmp_path, [_job()], clock=lambda: next(tick))
    record = result["outcomes"][0]["records"][0]
    assert record["observed_at"].startswith("2026-10-26")
    assert datetime.fromisoformat(record["available_at"]) > datetime.fromisoformat(record["observed_at"])
    assert record["date"] == "2026-10-22"


@pytest.mark.parametrize("body", [_body("654321"), _body(rows=[
    {"x": _millis("2026-10-23"), "y": 1.3}, {"x": _millis("2026-10-23"), "y": 1.3},
]), _body(rows=[{"x": _millis("2026-10-27"), "y": 1.3}]), b"<html>login required</html>"])
def test_invalid_nav_is_not_structured_success(tmp_path, body):
    result = _refresh(tmp_path, [_job()], fetch=lambda *_: FetchResponse(200, body))
    assert result["outcomes"][0]["status"] == "error"
    assert result["outcomes"][0]["records"] == []


def test_reviewed_news_uses_exact_raw_character_positions(tmp_path):
    raw = b"<html><body>Published 2026-10-21: capacity plans are forward-looking.</body></html>"
    config = _review_config(raw, raw.decode())
    result = _refresh(tmp_path, [
        _job("news", family="policy_news", parser="reviewed_extract_v1", config=config),
    ], families=("policy_news",), fetch=lambda *_: FetchResponse(200, raw))
    outcome = result["outcomes"][0]
    assert outcome["status"] == "ok" and outcome["parse_status"] == "reviewed"
    assert outcome["records"][0]["payload"]["interpretation_status"] == "review_needed"


@pytest.mark.parametrize("changed", ["hash", "quote"])
def test_reviewed_news_invalidates_changed_source_or_position(tmp_path, changed):
    raw = b"<html><body>capacity plans</body></html>"
    config = _review_config(raw, raw.decode())
    if changed == "hash":
        config["expected_sha256"] = "0" * 64
    else:
        config["extracts"][0]["start"] += 1
    result = _refresh(tmp_path, [
        _job("news", family="policy_news", parser="reviewed_extract_v1", config=config),
    ], families=("policy_news",), fetch=lambda *_: FetchResponse(200, raw))
    assert result["outcomes"][0]["status"] == "error"
    assert result["outcomes"][0]["records"] == []


def test_retained_pdf_review_emits_import_ready_records_with_no_fresh_fetch():
    raw = b"%PDF-1.7 retained public document\n%%EOF"
    text = "2026-06-30 holdings: Example Corp 10.00%"
    config = _review_config(raw, text, pdf=True)
    capture = {
        "job_id": "report", "product_id": SUBJECT["product_id"], "family": "disclosure",
        "source_url": "https://official.example.org/report.pdf", "status": "captured_unparsed",
        "http_status": 200, "raw_sha256": hashlib.sha256(raw).hexdigest(), "observed_at": CLOCK,
    }
    result = review_followed_capture(
        SUBJECT, capture, raw, parser="reviewed_pdf_extract_v1", config=config, clock=lambda: CLOCK,
    )
    record = result["records"][0]
    assert result["request_count"] == 0
    assert result["retrieval"] == "retained_public_capture_review_not_live_fetch"
    assert record["observed_at"] == CLOCK
    assert record["evidence_import"]["family"] == "portfolio"
    assert record["evidence_import"]["locator"] == "page:45;table:7.4;row:1"
    with pytest.raises(ValidationError, match="retained bytes"):
        review_followed_capture(
            SUBJECT, capture, b"different", parser="reviewed_pdf_extract_v1", config=config,
            clock=lambda: CLOCK,
        )


@pytest.mark.parametrize("url", [
    "https://fundf10.eastmoney.com/jjjz_123456.html", "https://ccbfund.cn/",
    "http://public.example.org/", "https://127.0.0.1/", "https://user:pass@public.example.org/",
    "https://public.example.org/?token=secret", "https://localhost/",
])
def test_stopped_private_and_authenticated_routes_are_rejected_before_request(tmp_path, url):
    calls = []
    with pytest.raises(ValidationError):
        _refresh(tmp_path, [_job(url=url)], fetch=lambda *args: calls.append(args))
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_private_subject_and_private_config_fields_are_rejected(tmp_path):
    private = dict(SUBJECT, account="not-accepted")
    with pytest.raises(ValidationError):
        refresh_followed([private], ["nav"], [], 0, tmp_path / "private")
    with pytest.raises(ValidationError):
        _refresh(tmp_path, [_job("x", family="policy_news", parser="unknown", config={"cookie": "x"})])


def test_budget_unsupported_and_unconfigured_are_distinct(tmp_path):
    calls = []
    result = _refresh(tmp_path, [
        _job("unsupported", family="policy_news", parser="unknown_v1", config={}), _job(),
    ], families=("nav", "policy_news", "disclosure"), budget=0,
        fetch=lambda *args: calls.append(args))
    assert calls == [] and result["requests_made"] == 0
    assert [o["status"] for o in result["outcomes"]] == [
        "unsupported", "budget_exhausted", "not_configured",
    ]


def test_fresh_directory_and_response_byte_limit(tmp_path):
    result = _refresh(tmp_path, [_job()], max_bytes=10)
    assert result["outcomes"][0]["status"] == "error"
    assert not (tmp_path / "run" / "nav-123456" / "response.bin").exists()
    with pytest.raises(FileExistsError):
        _refresh(tmp_path, [_job()])


def test_duplicate_url_is_not_requested_twice(tmp_path):
    calls = []
    result = _refresh(tmp_path, [_job(), _job("duplicate")],
                      fetch=lambda *args: (calls.append(args[0]) or FetchResponse(200, _body())))
    assert len(calls) == 1 and result["requests_made"] == 1
    assert result["outcomes"][1]["status"] == "stopped"


def test_nav_preserves_unitmoney_presence_without_synthetic_empty_marker(tmp_path):
    body = _body(rows=[
        {"x": _millis("2026-10-22"), "y": 1.25, "unitMoney": ""},
        {"x": _millis("2026-10-23"), "y": 1.3},
    ])
    result = _refresh(tmp_path, [_job()], fetch=lambda *_: FetchResponse(200, body))
    first, second = result["outcomes"][0]["records"]
    assert "unitMoney" in first and first["unitMoney"] == ""
    assert "unitMoney" not in second


def test_live_reviewed_pdf_has_import_record_and_top_level_provenance(tmp_path):
    raw = b"%PDF-1.7 retained public document\n%%EOF"
    text = "2026-06-30 holdings: Example Corp 10.00%"
    config = _review_config(raw, text, pdf=True)
    result = _refresh(tmp_path, [
        _job("report", family="disclosure", parser="reviewed_pdf_extract_v1", config=config),
    ], families=("disclosure",), fetch=lambda *_: FetchResponse(200, raw))
    outcome = result["outcomes"][0]
    assert outcome["status"] == "ok" and outcome["parse_status"] == "reviewed"
    assert outcome["records"][0]["evidence_import"]["family"] == "portfolio"
    assert outcome["raw_path"] == str(tmp_path / "run" / "report" / "response.bin")
    for key in ("source_url", "raw_sha256", "observed_at", "available_at", "job_id"):
        assert outcome[key] == outcome["records"][0][key]

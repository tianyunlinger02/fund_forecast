"""RAM current-source contracts; injected bytes, never network or Store writes."""
from datetime import datetime, timedelta
import json

from rmbfund.providers.current import (
    collect_current,
    default_current_source_plan,
    merge_current_source_plan,
)
from rmbfund.providers.followed import FetchResponse

CLOCK = "2026-10-26T14:00:00+08:00"
SUBJECT = {"product_id": "open_fund:018147:C", "symbol": "018147", "name": "公开样本",
           "share_class": "C", "currency": "CNY", "public": True}


def _job(parser="eastmoney_nav_js_v1", *, url="https://public.example.org/nav.js", config=None,
         product_id=None, identifier="nav"):
    return {"job_id": identifier, "product_id": product_id or SUBJECT["product_id"],
            "family": "nav" if parser == "eastmoney_nav_js_v1" else "policy_news",
            "source_url": url, "parser": parser, "config": config or {}}


def _nav(days=("2026-10-23", "2026-10-26")):
    rows = [{"x": int(datetime.fromisoformat(day + "T00:00:00+08:00").timestamp() * 1000),
             "y": 2.04 + index / 100} for index, day in enumerate(days)]
    return ('fS_code="018147";Data_netWorthTrend=' + json.dumps(rows) + ';').encode()


def _rss(identifier="new", *, description="New capacity is planned."):
    return f'''<rss><channel><item><title>New {identifier}</title>
    <link>https://article.example.org/{identifier}</link>
    <pubDate>Mon, 26 Oct 2026 02:00:00 GMT</pubDate>
    <description>{description}</description></item></channel></rss>'''.encode()


def test_ram_rolling_nav_ignores_old_fixed_end_and_has_real_observation_clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tick = datetime.fromisoformat(CLOCK)
    clocks = iter((tick + timedelta(seconds=i)).isoformat() for i in range(20))
    result = collect_current([dict(SUBJECT, shares="999", cost="123")], source_plan=[
        _job(config={"start": "2026-01-01", "end": "2026-10-23"}),
    ], fetch=lambda *_: FetchResponse(200, _nav()), clock=lambda: next(clocks))
    rows = result["nav_by_product"][SUBJECT["product_id"]]
    assert [row["date"] for row in rows] == ["2026-10-23", "2026-10-26"]
    assert rows[-1]["value"] == "2.05"
    assert rows[-1]["nav_type"] == "final_unit_nav_candidate"
    assert rows[-1]["published_at"] is None
    assert datetime.fromisoformat(rows[-1]["available_at"]) > datetime.fromisoformat(rows[-1]["observed_at"])
    assert result["source_checks"][0]["requested_window"]["end"] == "2026-10-26"
    assert result["records_by_product"][SUBJECT["product_id"]] == []
    assert "cost" not in result["public_subjects"][0] and "shares" not in result["public_subjects"][0]
    assert list(tmp_path.iterdir()) == []
    assert json.loads(json.dumps(result)) == result


def test_raw_nav_collector_retains_same_date_source_disagreement():
    first, second = "https://public.example.org/first.js", "https://public.example.org/second.js"
    result = collect_current([SUBJECT], source_plan=[
        _job(url=first, identifier="first"), _job(url=second, identifier="second"),
    ], fetch=lambda url, *_: FetchResponse(200, _nav() if url == first else _nav().replace(b"2.04", b"3.04")),
       clock=lambda: CLOCK)
    rows = result["nav_by_product"][SUBJECT["product_id"]]
    same_day = [row for row in rows if row["date"] == "2026-10-23"]
    assert len(rows) == 4
    assert {row["value"] for row in same_day} == {"2.04", "3.04"}
    assert {row["source_url"] for row in same_day} == {first, second}


def test_new_feed_items_discovered_and_article_excerpt_is_not_reviewed_research():
    article = b'<html><article><div class="entry-content"><p>Plans are not revenue.</p></div></article></html>'
    plan = [_job("nvidia_rss_v1", url="https://feed.example.org/releases.xml",
                 config={"article_limit": 1, "issuer_ids": ["issuer:nvidia"]})]
    results = []
    for identifier in ("first", "later"):
        calls = []

        def fetch(url, *_):
            calls.append(url)
            return FetchResponse(200, article if "article.example.org" in url else _rss(identifier))

        result = collect_current([SUBJECT], source_plan=plan, fetch=fetch, clock=lambda: CLOCK)
        message = result["messages_by_product"][SUBJECT["product_id"]][0]
        assert len(calls) == result["requests_made"] == 2
        assert message["excerpt"] == "Plans are not revenue."
        assert message["content_status"] == "online_excerpt"
        assert message["interpretation_status"] == "unreviewed"
        assert message["event_status"] == "unresolved"
        assert message["publication_precision"] == "timestamp"
        assert message["published_at"] == "2026-10-26T10:00:00+08:00"
        results.append(message)
    assert results[0]["id"] != results[1]["id"]
    assert results[0]["source_url"].endswith("first")
    assert results[1]["source_url"].endswith("later")


def test_sina_gb18030_announcement_date_and_unparsed_report_not_holdings():
    url = "https://finance.sina.com.cn/fund/quotes/018147/bc.shtml"
    raw = '''<html><meta charset="gb2312"><div>018147</div>
    <table><tr><td><a title="基金2026年第三季度报告"
    href="http://vip.stock.finance.sina.com.cn/fund/go.php/vAkFundInfo_JJGKXX/q/5600668.phtml">季度报告</a>
    </td><td>投资组合</td><td>2026-10-26</td></tr></table></html>'''.encode("gb18030")
    calls = []

    def fetch(target, *_):
        calls.append(target)
        return FetchResponse(200, raw if target == url else b"<html>Loading bulletin</html>")

    result = collect_current([SUBJECT], source_plan=[_job("sina_fund_announcements_v1", url=url)],
                             fetch=fetch, clock=lambda: CLOCK)
    message = result["messages_by_product"][SUBJECT["product_id"]][0]
    assert message["published_at"] == "2026-10-26" and message["publication_precision"] == "date"
    assert message["title"] == "基金2026年第三季度报告"
    assert message["content_status"] == "unparsed_document" and message["excerpt"] == ""
    assert calls[1].startswith("https://vip.stock.finance.sina.com.cn/")
    assert result["records_by_product"][SUBJECT["product_id"]] == []
    assert result["status"] == "partial"


def test_shared_feed_and_articles_requested_once_for_two_public_funds():
    peer = dict(SUBJECT, product_id="open_fund:539002:A", symbol="539002", share_class="A")
    plan = [_job("nvidia_rss_v1", url="https://feed.example.org/releases.xml",
                 product_id=item["product_id"], identifier=f"feed-{i}")
            for i, item in enumerate((SUBJECT, peer))]
    calls = []

    def fetch(url, *_):
        calls.append(url)
        return FetchResponse(200, b'<div class="entry-content">Shared source body.</div>'
                             if "article.example.org" in url else _rss())

    result = collect_current([SUBJECT, SUBJECT, peer], source_plan=plan, fetch=fetch, clock=lambda: CLOCK)
    assert len(calls) == len(set(calls)) == result["requests_made"] == 2
    assert len(result["public_subjects"]) == 2
    a, b = (result["messages_by_product"][row["product_id"]][0] for row in (SUBJECT, peer))
    assert {key: value for key, value in a.items() if key != "evidence_id"} == {
        key: value for key, value in b.items() if key != "evidence_id"
    }
    assert a["evidence_id"] != b["evidence_id"] and a["content_status"] == "online_excerpt"
    assert sum(check["request_made"] for check in result["source_checks"]) == 2


def test_failure_and_prior_stop_do_not_refetch_or_erase_other_source_dates():
    calls = []

    def fetch(url, *_):
        calls.append(url)
        return FetchResponse(403, b"denied") if "news.example.org" in url else FetchResponse(200, _nav())

    plan = [_job("nvidia_rss_v1", url="https://news.example.org/feed", identifier="failed"),
            _job("nvidia_rss_v1", url="https://news.example.org/another", identifier="same-host"),
            _job()]
    result = collect_current([SUBJECT], source_plan=plan, fetch=fetch, clock=lambda: CLOCK)
    assert result["requests_made"] == len(calls) == 2
    assert [row["status"] for row in result["source_checks"]] == ["error", "stopped", "ok"]
    assert result["source_checks"][0]["observed_at"] is None
    assert result["source_checks"][2]["source_date"] == "2026-10-26"
    assert result["status"] == "partial"
    again = collect_current([SUBJECT], source_plan=plan[:1], fetch=fetch, clock=lambda: CLOCK,
                            blocked_urls=result["stopped_urls"])
    assert again["requests_made"] == 0 and len(calls) == 2
    assert again["source_checks"][0]["status"] == "skipped_blocked"
    assert again["source_checks"][0]["observed_at"] is None


def test_motir_dated_index_and_scoped_body_not_site_slogan():
    index = b'''<a href="/eng/article/EATCLdfa319ada/2754/view">Read more</a>
    <a href="/eng/article/EATCLdfa319ada/2765/view">Trade and robotics cooperation 2026-10-23</a>'''
    article = b'''<meta property="og:description" content="Generic ministry slogan">
    <div class="detail-cont"><p>Officials met. Cooperation is planned.</p></div>'''
    result = collect_current([SUBJECT], source_plan=[
        _job("motir_press_index_v1", url="https://english.motir.go.kr/eng/"),
    ], fetch=lambda url, *_: FetchResponse(200, index if url.endswith("/eng/") else article),
        clock=lambda: CLOCK)
    messages = result["messages_by_product"][SUBJECT["product_id"]]
    assert len(messages) == 1
    assert messages[0]["published_at"] == "2026-10-23"
    assert messages[0]["title"] == "Trade and robotics cooperation"
    assert messages[0]["excerpt"] == "Officials met. Cooperation is planned."
    assert messages[0]["event_status"] == "unresolved"


def test_budget_reserves_indexes_and_keeps_feed_summary_when_body_unavailable():
    result = collect_current([SUBJECT], source_plan=[
        _job("nvidia_rss_v1", url="https://feed.example.org/releases.xml"), _job(),
    ], fetch=lambda url, *_: FetchResponse(200, _nav() if url.endswith("nav.js") else _rss()),
        clock=lambda: CLOCK, request_budget=2)
    assert result["requests_made"] == 2
    assert [check["status"] for check in result["source_checks"]] == ["ok", "ok", "budget_exhausted"]
    message = result["messages_by_product"][SUBJECT["product_id"]][0]
    assert message["content_status"] == "publisher_summary"
    assert message["excerpt"] == "New capacity is planned."
    assert result["nav_by_product"][SUBJECT["product_id"]]


def test_defaults_are_changing_sources_without_pinned_historical_dates():
    jobs = default_current_source_plan([SUBJECT])
    assert len(jobs) == 4
    assert {row["parser"] for row in jobs} == {
        "eastmoney_nav_js_v1", "sina_fund_announcements_v1", "nvidia_rss_v1", "motir_press_index_v1",
    }
    assert all("end" not in row["config"] and "expected_sha256" not in row["config"] for row in jobs)
    foreign = dict(SUBJECT, product_id="open_fund:000001:A", symbol="000001", share_class="A")
    assert len(default_current_source_plan([foreign])) == 2


def test_explicit_empty_plan_still_means_no_requests_not_implicit_defaults():
    def fetch(*_):
        raise AssertionError("an explicit empty plan must not request a default source")

    result = collect_current([SUBJECT], source_plan=[], fetch=fetch, clock=lambda: CLOCK)
    assert result["requests_made"] == 0 and result["source_checks"] == []
    assert result["status"] == "unavailable"
    assert result["evidence_by_product"][SUBJECT["product_id"]] == []
    assert result["record_evidence_by_product"][SUBJECT["product_id"]] == {}


def test_merge_plan_preserves_generic_nav_announcements_for_every_six_digit_input():
    peers = [dict(SUBJECT, product_id=f"open_fund:{code}:A", symbol=code, share_class="A")
             for code in ("000300", "110006")]
    overridden = _job(url="https://public.example.org/replacement.js", identifier="configured-nav")
    extra = _job("public_document_v1", url="https://public.example.org/observed-report.pdf",
                 identifier="known-report")
    jobs = merge_current_source_plan([SUBJECT, *peers], [overridden, extra])
    for peer in (SUBJECT, *peers):
        own = [job for job in jobs if job["product_id"] == peer["product_id"]]
        assert any(row["parser"] == "eastmoney_nav_js_v1" for row in own)
        assert any(row["parser"] == "sina_fund_announcements_v1" for row in own)
    assert len([row for row in jobs if row["product_id"] == SUBJECT["product_id"]
                and row["family"] == "nav"]) == 1
    assert overridden in jobs and extra in jobs
    assert merge_current_source_plan(peers, []) == default_current_source_plan(peers)
    assert merge_current_source_plan(peers, [overridden]) == default_current_source_plan(peers)


def test_full_body_evidence_has_late_paragraph_but_messages_stay_compact():
    end = "The final paragraph contains the important counterargument."
    article = ('<div class="entry-content"><p>' + "Background. " * 350
               + '</p><p>' + end + '</p></div>').encode()
    result = collect_current([SUBJECT], source_plan=[
        _job("nvidia_rss_v1", url="https://feed.example.org/releases.xml"),
    ], fetch=lambda url, *_: FetchResponse(200, article if "article.example.org" in url else _rss()),
        clock=lambda: CLOCK)
    message = result["messages_by_product"][SUBJECT["product_id"]][0]
    doc = result["evidence_by_product"][SUBJECT["product_id"]][0]
    assert message["evidence_id"] == doc["id"]
    assert end not in message["excerpt"] and len(message["excerpt"]) <= 2400
    assert any(end in section["text"] for section in doc["sections"])
    assert doc["product_id"] == SUBJECT["product_id"]
    assert doc["published_at"] == "2026-10-26T10:00:00+08:00"
    assert doc["content_status"] == "online_html_sections"
    assert result["records_by_product"][SUBJECT["product_id"]] == []  # News is not a financial release.


def test_collector_follows_only_actual_announced_attachment_and_returns_located_records(monkeypatch):
    import subprocess

    index_url = "https://finance.sina.com.cn/fund/quotes/018147/bc.shtml"
    detail_url = "https://vip.stock.finance.sina.com.cn/fund/go.php/vAkFundInfo_JJGKXX/q/42.phtml"
    pdf_url = "https://files.example.org/the-original-42.pdf"
    index = (f'<div>018147</div><a title="基金产品资料概要" href="{detail_url}">概要</a>'
             '<span>2026-09-21</span>').encode()
    detail = f'<table id="Table1"><tr><td>信息全文</td><td><a href="{pdf_url}">附件</a></td></tr></table>'.encode()
    text = '''建信新兴市场优选混合型证券投资基金(C类份额)基金产品资料概要更新
编制日期：2026年9月17日
下属基金交易代码 018147
基金经理 程星烨 开始担任本基金基金经理的日期 2019年12月11日
'''
    monkeypatch.setattr("rmbfund.providers.documents.subprocess.run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, text.encode() + b"\f", b""))
    calls = []

    def fetch(url, *_):
        calls.append(url)
        return FetchResponse(200, {index_url: index, detail_url: detail, pdf_url: b"%PDF-synthetic"}[url])

    result = collect_current([SUBJECT], source_plan=[
        dict(_job("sina_fund_announcements_v1", url=index_url), family="disclosure"),
    ], fetch=fetch, clock=lambda: CLOCK)
    assert calls == [index_url, detail_url, pdf_url]
    records = result["records_by_product"][SUBJECT["product_id"]]
    assert len(records) == 1 and records[0]["family"] == "management"
    assert records[0]["source_url"] == pdf_url
    refs = result["record_evidence_by_product"][SUBJECT["product_id"]][records[0]["id"]]
    docs = {row["id"]: row for row in result["evidence_by_product"][SUBJECT["product_id"]]}
    assert refs[0]["locator"] == "pdf:page:1"
    assert all(any(ref["quote"] in sec["text"] and ref["locator"] == sec["locator"]
                   for sec in docs[ref["evidence_id"]]["sections"]) for ref in refs)
    assert result["messages_by_product"][SUBJECT["product_id"]][0]["attachment_urls"] == [pdf_url]
    assert json.loads(json.dumps(result)) == result

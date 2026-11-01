"""Offline document contracts: synthetic text/bytes, no network or original archive."""
from datetime import datetime, timedelta
import subprocess

import pytest

from rmbfund.advice_evidence import normalize_record
from rmbfund.domain import ValidationError
from rmbfund.providers.documents import (
    _select_sections,
    read_public_document,
    records_from_document,
)
from rmbfund.providers.followed import FetchResponse

AT = "2026-10-27T12:00:00+08:00"
SUBJECT = {"product_id": "open_fund:018147:C", "symbol": "018147", "share_class": "C"}
TITLE = "建信新兴市场优选混合型证券投资基金(C类份额)基金产品资料概要更新"
SUMMARY = """建信新兴市场优选混合型证券投资基金(C类份额)
基金产品资料概要更新
编制日期：2026年9月17日
送出日期：2026年9月21日
一、产品概况
下属基金简称 建信新兴市场混合（QDII）C 下属基金交易代码 018147
基金经理 程星烨 开始担任本基金基金经理的日期 2019年12月11日
基金经理 李博涵 开始担任本基金基金经理的日期 2017年11月13日
基金经理 房乐 开始担任本基金基金经理的日期 2024年6月18日
运作方式 普通开放式 开放频率 每个开放日
二、投资范围
股票投资占基金资产的比例不低于60%，不是实际披露持仓。
三、基金费用
赎回费 N＜7天 1.5%
7天≤N＜30天 0.5%
N≥30天 0%
管理费 1.2% 基金管理人和销售机构
托管费 0.2% 基金托管人
销售服务费 0.40% 销售机构
四、重要提示
本基金不提供任何保证。
"""


def bulletin(text=SUMMARY):
    return ('<meta charset="utf-8"><div>outside slogan</div><table id="Table1">'
            '<tr><td>信息全文</td><td>' + text.replace("\n", "<br>")
            + '</td></tr></table><footer>not evidence</footer>').encode()


def document(raw=None, *, url="https://public.example.org/summary", fund=True):
    return read_public_document(
        FetchResponse(200, raw or bulletin()), product_id=SUBJECT["product_id"],
        source_url=url, title=TITLE, published_at="2026-09-21", publication_precision="date",
        observed_at=AT, available_at=AT, fund=fund,
    )


def assert_citations(records, refs, doc):
    sections = {row["locator"]: row["text"] for row in doc["sections"]}
    for record in records:
        assert normalize_record(record) == record
        assert refs[record["id"]]
        for ref in refs[record["id"]]:
            assert ref["evidence_id"] == doc["id"]
            assert ref["quote"] in sections[ref["locator"]]


def test_real_shape_static_summary_normalizes_dated_fees_managers_not_mandate_or_dealing():
    doc, links = document()
    records, refs = records_from_document(doc, SUBJECT)
    assert links == [] and doc["content_status"] == "online_html_sections"
    assert {row["family"] for row in records} == {"management", "fees"}
    assert {row["report_date"] for row in records} == {"2026-09-17"}
    assert {row["published_at"] for row in records} == {"2026-09-21"}
    assert all(row["observed_at"] == row["available_at"] == AT for row in records)
    fees = next(row["payload"] for row in records if row["family"] == "fees")
    assert fees["management_rate"] == "0.012"
    assert fees["custody_rate"] == "0.002" and fees["sales_service_rate"] == "0.004"
    assert fees["redemption_bands"] == [
        {"min_days": 0, "max_days": 6, "rate": "0.015"},
        {"min_days": 7, "max_days": 29, "rate": "0.005"},
        {"min_days": 30, "max_days": None, "rate": "0"},
    ]
    assert fees["holding_day_basis"] == "unknown" and fees["channel_verified"] is False
    assert_citations(records, refs, doc)
    assert all("outside slogan" not in row["text"] for row in doc["sections"])
    assert records_from_document(doc, dict(SUBJECT, share_class="A")) == ([], {})


def test_late_html_fee_section_is_real_body_not_first_2400_characters():
    text = SUMMARY.replace("三、基金费用", "\n" + "Background sentence. " * 350 + "\n三、基金费用")
    doc, _ = document(bulletin(text))
    assert sum(len(row["text"]) for row in doc["sections"]) > 7000
    records, refs = records_from_document(doc, SUBJECT)
    fees = next(row for row in records if row["family"] == "fees")
    quotes = [row["quote"] for row in refs[fees["id"]]]
    assert "销售服务费 0.40%" in quotes
    assert_citations(records, refs, doc)


def test_same_public_document_url_has_product_bound_evidence_ids():
    first, _ = document(url="https://public.example.org/shared-report")
    second, _ = read_public_document(
        FetchResponse(200, bulletin()), product_id="open_fund:539002:A",
        source_url="https://public.example.org/shared-report", title=TITLE,
        published_at="2026-09-21", publication_precision="date",
        observed_at=AT, available_at=AT, fund=True,
    )
    assert first["source_url"] == second["source_url"]
    assert first["id"] != second["id"]
    assert first["product_id"] != second["product_id"]


def test_placeholder_metadata_and_navigation_are_not_report_text_or_invented_links():
    doc, links = document(bulletin("公告内容详见附件"))
    assert doc["sections"] == [] and doc["content_status"] == "unparsed_document"
    assert links == [] and records_from_document(doc, SUBJECT) == ([], {})
    raw = ('<meta property="og:description" content="A generic slogan">'
           '<a href="/background.pdf">Company in brief</a><p>Page navigation</p>').encode()
    doc, links = document(raw)
    assert doc["sections"] == [] and links == []


def test_pdf_attachment_href_must_actually_be_advertised_in_bulletin():
    doc, links = document(bulletin('公告内容详见附件 <a href="/actual-report-42.pdf">附件</a>'))
    assert links == ["/actual-report-42.pdf"]
    assert all("actual-report-42.pdf" not in row["text"] for row in doc["sections"])


def test_pdf_poppler_uses_ram_page_breaks_and_wrapped_product_manager(monkeypatch, tmp_path):
    page = SUMMARY.split("基金经理 程星烨")[0] + """开始担任本基金基金经
2019 年 12 月 11 日
基金经理 程星烨 理的日期
证券从业日期 2012 年 7 月 1 日
"""
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, (page + "\fLate policy paragraph.\f").encode(), b"")

    monkeypatch.setattr("rmbfund.providers.documents.subprocess.run", run)
    monkeypatch.chdir(tmp_path)
    doc, _ = document(b"%PDF-synthetic", url="https://public.example.org/actual.pdf")
    assert [row["locator"] for row in doc["sections"]] == ["pdf:page:1", "pdf:page:2"]
    assert calls[0][0] == ["/usr/bin/pdftotext", "-layout", "-", "-"]
    assert calls[0][1]["input"] == b"%PDF-synthetic"
    assert list(tmp_path.iterdir()) == []
    records, refs = records_from_document(doc, SUBJECT)
    assert records[0]["payload"] == {"managers": [{"name": "程星烨", "start_date": "2019-12-11"}]}
    assert_citations(records, refs, doc)


def _small_pdf():
    # A two-page ASCII PDF built here, not a downloaded/executed fixture.
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>"]
    for content_id, text in ((4, "First page body."), (6, "Second page full relevant text.")):
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] "
                        f"/Resources << /Font << /F1 7 0 R >> >> /Contents {content_id} 0 R >>").encode())
        stream = f"BT /F1 12 Tf 40 350 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode())
    return bytes(data)


def test_actual_installed_poppler_reads_two_pages_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc, _ = document(_small_pdf())
    assert doc["content_status"] == "online_pdf_pages"
    assert doc["sections"] == [
        {"locator": "pdf:page:1", "text": "First page body."},
        {"locator": "pdf:page:2", "text": "Second page full relevant text."},
    ]
    assert list(tmp_path.iterdir()) == []


def test_unreadable_pdf_is_not_claimed_as_economic_success(monkeypatch):
    monkeypatch.setattr("rmbfund.providers.documents.subprocess.run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 1, b"", b"not readable"))
    with pytest.raises(ValidationError, match="Poppler"):
        document(b"%PDF-broken")


def test_relevant_late_report_tables_are_not_displaced_by_early_nav_mentions():
    rows = [{"locator": f"pdf:page:{i}", "text": f"净资产 accounting note {i}"}
            for i in range(1, 58)]
    rows[2]["text"] = "\n".join("7.4 所有权益投资明细 .................. 45" for _ in range(25))
    rows[44]["text"] = "7.4 所有权益投资明细\n股票名称 NVIDIA 股票代码 US67066G1040 9.38"
    selected = _select_sections(rows, fund=True)
    assert len(selected) <= 32
    assert rows[44] in selected and rows[45] in selected
    assert rows[2] not in selected


def periodic_document():
    doc, _ = document()
    doc["title"] = "建信新兴市场优选混合型证券投资基金2026年中期报告"
    doc["published_at"] = "2026-08-28"
    doc["sections"] = [
        {"locator": "pdf:page:2", "text": "建信新兴市场混合（QDII）2026 年中期报告\n本报告期自 2026 年 1 月 1 日起至 6 月 30 日止。"},
        {"locator": "pdf:page:5", "text": "下属分级基金的基\n建信新兴市场混合（QDII）A 建信新兴市场混合（QDII）C\n金简称\n下属分级基金的交\n539002 018147\n易代码"},
        {"locator": "pdf:page:44", "text": "7.2 期末在各个国家（地区）证券市场的权益投资分布\n国家（地区） 公允价值 占基金资产净值比例（%）\n合计 6,193,364,803.48 81.60\n7.3 期末按行业分类"},
        {"locator": "pdf:page:45", "text": "7.4 期末按公允价值占基金资产净值比例大小排序的所有权益投资明细\nNVIDIA 英伟达 US6706\n2 克证券 美国 522,385 711,902,611.38 9.38\nCorp 公司 6G1040\n交易所"},
    ]
    return doc


def test_narrow_periodic_portfolio_has_actual_report_period_nav_and_combined_scope():
    doc = periodic_document()
    records, refs = records_from_document(doc, SUBJECT)
    assert len(records) == 1
    record = records[0]
    assert record["family"] == "portfolio"
    assert record["report_date"] == "2026-06-30" and record["period_start"] == "2026-01-01"
    assert record["payload"]["denominator"] == "nav"
    assert record["payload"]["assets"] == {"equity": "0.816"}
    assert record["payload"]["holdings_scope"] == "partial"
    assert record["payload"]["holdings"][0]["weight"] == "0.0938"
    assert any("539002 018147" in ref["quote"] for ref in refs[record["id"]])
    assert_citations(records, refs, doc)
    later = (datetime.fromisoformat(AT) + timedelta(days=1)).isoformat()
    doc.update(observed_at=later, available_at=later)
    again, _ = records_from_document(doc, SUBJECT)
    assert again[0]["id"] == record["id"]


@pytest.mark.parametrize("changed", ["denominator", "scope", "row"])
def test_missing_nav_denominator_combined_scope_or_actual_row_emits_no_portfolio(changed):
    doc = periodic_document()
    if changed == "denominator":
        doc["sections"][2]["text"] = doc["sections"][2]["text"].replace("资产净值", "总资产")
    elif changed == "scope":
        doc["sections"][1]["text"] = "018147 alone is not combined share-class proof"
    else:
        doc["sections"][3]["text"] = "7.5 本期累计买入金额 NVIDIA 9.38 not current holdings"
    assert records_from_document(doc, SUBJECT) == ([], {})

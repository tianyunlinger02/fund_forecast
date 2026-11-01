"""Bounded public document text in RAM, with page/section evidence locators.

HTML is read only from qualified article regions or Sina's labelled bulletin
cell. PDF text comes from the installed Poppler executable over stdin/stdout;
no fetched content is executed and no original/full text is written to disk.
The narrow structured adapters cover verified 018147 C product-summary facts
and one partial H1 portfolio table, not arbitrary reports or current dealing.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from decimal import Decimal
from html.parser import HTMLParser
from urllib.parse import urlsplit

from ..advice_evidence import normalize_record
from ..domain import ValidationError, digest
from .followed import FetchResponse

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
                   "meta", "param", "source", "track", "wbr"})
_BLOCK = frozenset({"br", "p", "div", "article", "section", "tr", "li", "h1", "h2", "h3",
                    "h4", "h5", "h6", "table", "td", "th", "ul", "ol"})
_REGIONS = frozenset({"entry-content", "detail-cont", "main-content", "bulletin_content",
                      "article-content"})
_HEADING = re.compile(r"^(?:[一二三四五六七八九十]+[、．.]|§\s*\d+)")
_FUND_USEFUL = re.compile(r"产品概况|基金经理|费用|投资组合|股票名称|股票代码|前十|净资产|资产净值|"
                          r"编制日期|报告期|重要提示|申购|赎回")
MAX_TEXT_CHARS = 80_000
MAX_SECTIONS = 32


def _decode(response: FetchResponse) -> str:
    headers = {key.lower(): value for key, value in response.headers.items()}
    match = re.search(r"charset=[\s\"']*([\w-]+)", headers.get("content-type", ""), re.I)
    if match is None:
        match = re.search(br"charset=[\s\"']*([\w-]+)", response.body[:4096], re.I)
    codec = match.group(1) if match else "utf-8-sig"
    if isinstance(codec, bytes):
        codec = codec.decode("ascii")
    try:
        return response.body.decode(codec)
    except (UnicodeError, LookupError):
        return response.body.decode("gb18030", "replace")


def _clean(text: str) -> str:
    text = text.replace("\r", "\n").replace("\x00", "")
    return "\n".join(line for raw in text.splitlines()
                     if (line := re.sub(r"[^\S\n]+", " ", raw).strip()))


class _HTML(HTMLParser):
    """One static parser shared by changing lists and scoped document reads."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.tokens: list[str] = []
        self.links: list[dict] = []
        self.embeds: list[dict] = []
        self.meta: dict[str, str] = {}
        self.regions: list[dict] = []
        self.anchor: dict | None = None
        self.feed(text)
        self.close()
        # Some public bulletins omit closing table/body tags.
        while self.stack:
            self._close_frame(self.stack.pop())
        self.body = [row["text"] for row in self.regions]

    def _break(self):
        for frame in self.stack:
            if "parts" in frame:
                frame["parts"].append("\n")

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in _BLOCK:
            self._break()
        if tag == "meta" and values.get("content"):
            self.meta[values.get("property", values.get("name", ""))] = values["content"]
        scoped = any(frame.get("region") or frame.get("bulletin_cell") for frame in self.stack)
        if tag == "a":
            self.anchor = {"href": values.get("href", ""), "title": values.get("title", ""),
                           "start": len(self.tokens), "parts": [], "scoped": scoped}
        if tag in {"iframe", "object", "embed"}:
            self.embeds.append({"href": values.get("data", values.get("src", "")),
                                "scoped": scoped})
        if tag in _VOID:
            return
        frame = {"tag": tag, "attrs": values}
        qualified = (_REGIONS.intersection(values.get("class", "").split())
                     or values.get("id") in {"bulletin_content", "articleContent"})
        if qualified and not any(item.get("region") for item in self.stack):
            marker = "#" + values["id"] if values.get("id") else "." + next(
                name for name in values.get("class", "").split() if name in _REGIONS)
            frame.update(region=True, parts=[], locator=f"html:{tag}{marker}")
        table = any(item["tag"] == "table" and item["attrs"].get("id") == "Table1"
                    for item in self.stack)
        if table and tag == "tr":
            frame["cells"] = []
        if table and tag == "td":
            frame.update(bulletin_cell=True, parts=[])
        self.stack.append(frame)

    def _close_frame(self, frame):
        if frame.get("region"):
            text = _clean("".join(frame["parts"]))
            if text:
                self.regions.append({"locator": frame["locator"], "text": text})
        if frame.get("bulletin_cell"):
            row = next((item for item in reversed(self.stack) if "cells" in item), None)
            if row is not None:
                row["cells"].append(_clean("".join(frame["parts"])))
        cells = frame.get("cells", [])
        if len(cells) >= 2 and cells[0].strip() == "信息全文" and cells[1]:
            self.regions.append({"locator": "html:table#Table1;row:信息全文;cell:2", "text": cells[1]})

    def handle_endtag(self, tag):
        if tag in _BLOCK:
            self._break()
        if tag == "a" and self.anchor is not None:
            self.anchor.update(end=len(self.tokens), text=" ".join(self.anchor.pop("parts")))
            self.links.append(self.anchor)
            self.anchor = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                while len(self.stack) > index:
                    self._close_frame(self.stack.pop())
                break

    def handle_data(self, data):
        if any(frame["tag"] in {"script", "style", "noscript"} for frame in self.stack):
            return
        value = " ".join(data.split())
        if not value:
            return
        self.tokens.append(value)
        if self.anchor is not None:
            self.anchor["parts"].append(value)
        for frame in self.stack:
            if "parts" in frame:
                frame["parts"].append(data)


def _html_sections(page: _HTML) -> list[dict]:
    result = []
    for region in page.regions:
        text = region["text"]
        if re.fullmatch(r"(?:公告内容详见附件|附件|下载全文|下载附件|点击下载|[。\s])+", text):
            continue  # This is an unavailable attachment notice, not the report.
        parts, heading, lines = [], "正文", []
        for line in text.splitlines():
            if _HEADING.match(line) and lines:
                parts.append((heading, lines))
                heading, lines = line[:120], []
            lines.append(line)
        if lines:
            parts.append((heading, lines))
        for heading, lines in parts:
            # Paragraph-sized windows preserve late fees/risks instead of a head excerpt.
            chunks, chunk, size = [], [], 0
            for line in lines:
                if chunk and size + len(line) > 7000:
                    chunks.append("\n".join(chunk))
                    chunk, size = [], 0
                chunk.append(line)
                size += len(line) + 1
            if chunk:
                chunks.append("\n".join(chunk))
            for index, chunk in enumerate(chunks, 1):
                result.append({"locator": f"{region['locator']};section:{heading};part:{index}",
                               "text": chunk})
    return result


def _select_sections(sections: list[dict], *, fund: bool) -> list[dict]:
    if not sections:
        return []
    if len(sections) <= MAX_SECTIONS and sum(len(row["text"]) for row in sections) <= MAX_TEXT_CHARS:
        return sections
    candidates = list(range(len(sections)))
    if fund:
        scores = {}
        for i, section in enumerate(sections):
            text = section["text"]
            # A contents-page mention is not the actual late report table.
            if len(re.findall(r"\.{6,}", text)) > 5:
                continue
            score = 10 if _FUND_USEFUL.search(text) else 0
            if re.search(r"编制日期|本报告期自|基金基本情况|下属基金交易代码", text):
                score = 80
            if re.search(r"基金经理|基金运作相关费用|赎回费", text):
                score = 90
            if re.search(r"期末基金资产组合|股票名称|股票代码|证券代码|所有权益投资明细|前十名股票", text):
                score = 120
            if score:
                scores[i] = score
        # Preserve the cover, and neighbouring pages where a table may continue.
        for i, score in list(scores.items()):
            if score >= 90:
                for neighbour in (i - 1, i + 1):
                    if 0 <= neighbour < len(sections):
                        scores[neighbour] = max(scores.get(neighbour, 0), score - 5)
        scores[0] = 150
        candidates = sorted(scores, key=lambda i: (-scores[i], i))
    selected, size = [], 0
    for index in candidates:
        section = sections[index]
        if len(selected) < MAX_SECTIONS and size + len(section["text"]) <= MAX_TEXT_CHARS:
            selected.append(index)
            size += len(section["text"])
    return [sections[i] for i in sorted(selected)]


def _attachment_hrefs(page: _HTML) -> list[str]:
    result = []
    for link in page.links + page.embeds:
        href = link["href"]
        pdf = re.search(r"\.pdf(?:[?&#]|$)|/download_pdf/|file_ext=\.?pdf(?:&|$)", href, re.I)
        labelled = re.search(r"附件|下载全文|报告全文|Download Press Release", link.get("text", ""), re.I)
        if href and (pdf and link.get("scoped") or labelled):
            result.append(href)
    return list(dict.fromkeys(result))


def read_public_document(response: FetchResponse, *, product_id: str, source_url: str,
                         title: str, published_at: str | None, publication_precision: str,
                         observed_at: str, available_at: str | None = None,
                         fund: bool = False) -> tuple[dict, list[str]]:
    """Return inert located text and advertised attachment hrefs; perform no requests."""
    attachments = []
    if response.body.startswith(b"%PDF-"):
        process = subprocess.run(
            ["/usr/bin/pdftotext", "-layout", "-", "-"], input=response.body,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=20,
        )
        if process.returncode:
            raise ValidationError("Poppler could not extract this PDF; no OCR or guessed tables")
        if len(process.stdout) > 4 * 1024 * 1024:
            raise ValidationError("PDF extracted text exceeds the document text budget")
        pages = process.stdout.decode("utf-8", "strict").split("\f")
        sections = [{"locator": f"pdf:page:{i}", "text": text}
                    for i, page in enumerate(pages, 1) if (text := _clean(page))]
        status = "online_pdf_pages" if sections else "unparsed_pdf"
    else:
        page = _HTML(_decode(response))
        sections = _html_sections(page)
        attachments = _attachment_hrefs(page)
        status = "online_html_sections" if sections else "unparsed_document"
    sections = _select_sections(sections, fund=fund)
    if not sections and status.startswith("online_"):
        status = "unparsed_document"
    identity = {"product_id": product_id, "source_url": source_url, "title": title,
                "published_at": published_at, "sections": sections}
    document = {"id": digest(identity), "product_id": product_id, "source_url": source_url,
                "title": title, "published_at": published_at,
                "publication_precision": publication_precision, "observed_at": observed_at,
                "available_at": available_at or observed_at, "content_status": status,
                "sections": sections}
    return document, attachments


_CHINESE_DAY = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
_MANAGER = re.compile(r"基金经理\s+([^\s]+)\s+开始担任本基金基金经理的日期\s*" + _CHINESE_DAY)
_PDF_MANAGER = re.compile(r"开始担任本基金基金经\s*" + _CHINESE_DAY
                          + r"\s*基金经理\s+([^\s]+)\s*理的日期")


def _day(groups) -> str:
    return f"{int(groups[0]):04d}-{int(groups[1]):02d}-{int(groups[2]):02d}"


def _percent(text: str) -> str:
    return format(Decimal(text) / 100, "f")


def _periodic_portfolio(document: Mapping, subject: Mapping) -> tuple[list[dict], dict]:
    """One verified wrapped report-table shape, intentionally partial issuer coverage."""
    sections = document["sections"]
    text = "\n".join(section["text"] for section in sections)
    if "建信新兴市场" not in text or "中期报告" not in text:
        return [], {}

    def first(pattern, *, within=None, until=None):
        for section in sections:
            value = section["text"]
            if within:
                start = value.find(within)
                if start < 0:
                    continue
                value = value[start:]
            if until:
                value = value.split(until, 1)[0]
            match = re.search(pattern, value)
            if match:
                return section, match
        return None

    period = first(r"本报告期自\s*" + _CHINESE_DAY
                   + r"\s*起至\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日止")
    scope = first(r"下属分级基金的基\s*\n建信新兴市场混合（QDII）A\s+建信新兴市场混合（QDII）C"
                  r"\s*\n金简称\s*\n下属分级基金的交\s+539002\s+018147\s+易代码")
    equity = first(r"国家（地区）\s+公允价值\s+占基金资产净值比例[（(]%[）)]\s*\n"
                   r"[\s\S]*?合计\s+[\d,]+\.\d{2}\s+([\d.]+)(?=\n|$)",
                   within="7.2 期末在各个国家（地区）证券市场的权益投资分布", until="7.3 ")
    nvidia = first(r"NVIDIA\s+英伟达\s+US6706\s*\n"
                   r"2\s+克证券\s+美国\s+[\d,]+\s+[\d,]+\.\d{2}\s+([\d.]+)\s*\n"
                   r"Corp\s+公司\s+6G1040", within="7.4 期末按公允价值占基金资产净值比例")
    if not all((period, scope, equity, nvidia)):
        return [], {}
    if "占基金资产净值比例" not in equity[0]["text"]:
        return [], {}
    year, month, day, end_month, end_day = period[1].groups()
    record = normalize_record({
        "product_id": subject["product_id"], "family": "portfolio",
        "source": "public_fund_document", "upstream": urlsplit(document["source_url"]).hostname,
        "source_url": document["source_url"], "period_start": _day((year, month, day)),
        "report_date": _day((year, end_month, end_day)),
        "published_at": document["published_at"],
        "publication_precision": document["publication_precision"],
        "observed_at": document["observed_at"], "available_at": document["available_at"],
        "payload": {"denominator": "nav", "assets": {"equity": _percent(equity[1][1])},
                    "holdings_scope": "partial", "holdings": [
                        {"issuer_id": "issuer:nvidia", "name": "NVIDIA Corp", "kind": "corporate_equity",
                         "weight": _percent(nvidia[1][1])}],
                    "countries": [], "sectors": [], "currencies": []},
    })
    evidence = [{"evidence_id": document["id"], "locator": section["locator"], "quote": match.group()}
                for section, match in (period, scope, equity, nvidia)]
    # The table heading establishes the NAV denominator, not total assets (7.1).
    heading = re.search(r"7\.4 期末按公允价值占基金资产净值比例[^\n]+", nvidia[0]["text"])
    evidence.append({"evidence_id": document["id"], "locator": nvidia[0]["locator"],
                     "quote": heading.group()})
    return [record], {record["id"]: evidence}


def records_from_document(document: Mapping, subject: Mapping) -> tuple[list[dict], dict]:
    """Normalize only literal 018147-C product-summary facts we can locate.

    A normal open-ended fund description does not establish today's dealing
    state. Investment limits are not actual portfolio weights. No record is
    emitted for either, or for issuer marketing/forward-looking news.
    """
    if (subject.get("symbol") != "018147" or subject.get("share_class") != "C"
            or document.get("product_id") != subject.get("product_id")):
        return [], {}
    sections = document["sections"]
    text = "\n".join(section["text"] for section in sections)
    if not ("建信新兴市场" in text and "基金产品资料概要" in text
            and re.search(r"下属基金交易代码\s*018147\b", text)
            and re.search(r"(?:C\s*类份额|下属基金简称[^\n]*C\b)", text)):
        return _periodic_portfolio(document, subject)

    def matches(pattern):
        return [(section, match) for section in sections
                for match in re.finditer(pattern, section["text"])]

    compiled = matches(r"编制日期\s*[:：]\s*" + _CHINESE_DAY)
    if len(compiled) != 1:
        return [], {}
    report_date = _day(compiled[0][1].groups())
    records, evidence = [], {}

    def emit(family, payload, citations):
        record = normalize_record({
            "product_id": subject["product_id"], "family": family,
            "source": "public_fund_document", "upstream": urlsplit(document["source_url"]).hostname,
            "source_url": document["source_url"], "report_date": report_date,
            "published_at": document["published_at"],
            "publication_precision": document["publication_precision"],
            "observed_at": document["observed_at"], "available_at": document["available_at"],
            "payload": payload,
        })
        records.append(record)
        evidence[record["id"]] = [{"evidence_id": document["id"], "locator": section["locator"],
                                   "quote": match.group()}
                                  for section, match in compiled + citations]

    managers = []
    for section in sections:
        for match in _MANAGER.finditer(section["text"]):
            managers.append((section, match, match[1], _day(match.groups()[1:])))
        for match in _PDF_MANAGER.finditer(section["text"]):
            managers.append((section, match, match[4], _day(match.groups()[:3])))
    manager_count = text.count("开始担任本基金基金经")
    if managers and len(managers) == manager_count:
        emit("management", {"managers": [{"name": name, "start_date": start}
                                         for _, _, name, start in managers]},
             [(section, match) for section, match, _, _ in managers])

    rates, citations = {}, []
    for label, key in (("管理费", "management_rate"), ("托管费", "custody_rate"),
                       ("销售服务费", "sales_service_rate")):
        flexible = r"\s*".join(map(re.escape, label))
        found = matches(r"(?<![\w])" + flexible + r"\s+([\d.]+)\s*%")
        if len(found) != 1:
            break
        rates[key] = _percent(found[0][1][1])
        citations.extend(found)
    if len(rates) == 3:
        first = matches(r"(?m)^(?:赎回费\s*)?N\s*[<＜]\s*(\d+)\s*天\s*([\d.]+)\s*%")
        middle = matches(r"(\d+)\s*天\s*≤\s*N\s*[<＜]\s*(\d+)\s*天\s*([\d.]+)\s*%")
        last = matches(r"N\s*≥\s*(\d+)\s*天\s*([\d.]+)\s*%")
        if len(first) == len(middle) == len(last) == 1:
            a, b, c = first[0][1], middle[0][1], last[0][1]
            low, high = int(a[1]), int(c[1])
            if 0 < low == int(b[1]) < int(b[2]) == high:
                emit("fees", {**rates, "redemption_bands": [
                    {"min_days": 0, "max_days": low - 1, "rate": _percent(a[2])},
                    {"min_days": low, "max_days": high - 1, "rate": _percent(b[3])},
                    {"min_days": high, "max_days": None, "rate": _percent(c[2])},
                ], "holding_day_basis": "unknown", "channel_verified": False},
                     citations + first + middle + last)
    return records, evidence

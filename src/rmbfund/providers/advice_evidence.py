"""Finite, inert public-document evidence for the experimental fund review.

No known-blocked origin is contacted here. The only automatic financial parser
validated against an actual retained 018147 document is its Eastmoney fee table.
It does not establish the fund's mandate, holdings, managers, combined assets, or
Alipay transaction conditions. A reviewed import is an explicit maintainer
attestation, not authentication by the publisher or a successful current fetch.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ..advice_evidence import (
    MAX_CAPTURE_BYTES,
    MAX_RECORD_BYTES,
    MAX_WEB_BYTES,
    bounded_json,
    _research_raw_note,
    normalize_record,
    normalize_research_thesis,
    normalize_source_url,
    research_thesis_source,
    research_thesis_source_uri,
)
from ..domain import Product, ValidationError, decimal, now_iso, timestamp
from ..research_contract import GENERATED_VALIDATION, normalize_generated_research

PARSER_VERSION = "reviewed-public-document-v1"
RESEARCH_PARSER_VERSION = "local-authored-research-v1"
_RESEARCH_WARNING = "本地作者研究，不是公开文件或来源方认证；链接及人工摘录不等于已归档的发布者原文。"
FEE_PARSER_VERSION = "eastmoney-018147-fee-html-v1"
MAX_IMPORT_RECORDS = 32
_FUND_NAME = "建信新兴市场混合(QDII)C"
_REVIEW_WARNING = "维护者已复核公开文件；导入校验不是来源方认证，也不代表已核实当前支付宝交易条件。"


class _Tables(HTMLParser):
    """Bounded text/table reader; scripts, styles, and hidden data are inert."""

    _VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self.titles: list[str] = []
        self.table: dict | None = None
        self.row: list | None = None
        self.cell: list[str] | None = None
        self.cell_tag = ""
        self.title: list[str] | None = None
        self.depth = 0
        self.ignored: list[str] = []
        self.cells = 0
        self.cell_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        hidden = ("hidden" in values or (values.get("aria-hidden") or "").lower() == "true"
                  or re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
                               values.get("style") or "", re.I))
        if self.ignored or tag in {"script", "style", "template", "noscript"} or hidden:
            if tag not in self._VOID:
                self.ignored.append(tag)
            return
        if tag == "title":
            if self.title is not None:
                raise ValidationError("HTML标题结构不明确")
            self.title = []
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                if len(self.tables) >= 128:
                    raise ValidationError("HTML表格数量超过上限")
                self.table = {"rows": [], "errors": []}
            elif self.table is not None:
                self.table["errors"].append("nested table")
        elif self.table is not None and self.depth == 1:
            if tag == "tr":
                if self.row is not None:
                    self.table["errors"].append("unclosed row")
                    self._finish_row()
                self.row = []
            elif tag in {"td", "th"}:
                if self.row is None:
                    self.table["errors"].append("cell outside row")
                    return
                if self.cell is not None:
                    self.table["errors"].append("unclosed cell")
                    self._finish_cell()
                spans = [values.get(key, "1") for key in ("colspan", "rowspan")]
                if any(not isinstance(span, str) or not re.fullmatch(r"[1-9]\d{0,3}", span)
                       or int(span) > 2048 for span in spans):
                    self.table["errors"].append("invalid cell span")
                elif any(span != "1" for span in spans):
                    self.table["errors"].append("spanning cell")
                self.cells += 1
                if self.cells > 20000:
                    raise ValidationError("HTML单元格数量超过上限")
                self.cell, self.cell_tag, self.cell_chars = [], tag, 0

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.ignored:
            return
        if self.title is not None:
            self.title.append(data)
            if sum(map(len, self.title)) > 1024:
                raise ValidationError("HTML标题超过上限")
        if self.cell is not None and self.depth == 1:
            self.cell_chars += len(data)
            if self.cell_chars > 4096:
                raise ValidationError("HTML单元格文本超过上限")
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.ignored:
            if tag in self.ignored:
                del self.ignored[len(self.ignored) - 1 - self.ignored[::-1].index(tag):]
            return
        if tag == "title" and self.title is not None:
            self.titles.append("".join("".join(self.title).split()))
            self.title = None
        if self.table is None:
            return
        if tag == "table":
            self.depth -= 1
            if self.depth == 0:
                if self.row is not None:
                    self.table["errors"].append("unclosed row")
                    self._finish_row()
                self.tables.append(self.table)
                self.table = None
        elif self.depth == 1:
            if tag in {"td", "th"}:
                if self.cell is None or self.cell_tag != tag:
                    self.table["errors"].append("unmatched cell")
                self._finish_cell()
            elif tag == "tr":
                if self.cell is not None:
                    self.table["errors"].append("unclosed cell")
                self._finish_row()

    def _finish_cell(self) -> None:
        if self.cell is not None and self.row is not None:
            self.row.append((self.cell_tag, "".join("".join(self.cell).split())))
        self.cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self.row is not None and self.table is not None:
            if len(self.table["rows"]) >= 2048:
                raise ValidationError("HTML表格行数超过上限")
            self.table["rows"].append(self.row)
        self.row = None


def _html(raw: bytes) -> _Tables:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_WEB_BYTES:
        raise ValidationError("HTML必须是非空且不超过2MiB的原始字节")
    try:
        text = raw.decode("utf-8-sig", "strict")
    except UnicodeError as exc:
        raise ValidationError("只支持已核实UTF-8口径的HTML，不猜测其他编码") from exc
    parser = _Tables()
    try:
        parser.feed(text)
        parser.close()
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("HTML无法按有界静态表格读取") from exc
    if parser.table is not None or parser.title is not None:
        raise ValidationError("HTML表格或标题未闭合")
    return parser


def _percent(value: str, *, annual: bool = False) -> str:
    suffix = r"%（每年）" if annual else "%"
    match = re.fullmatch(r"(\d+(?:\.\d+)?)" + suffix, value)
    if match is None:
        raise ValidationError("费率缺少已核实的百分比或年度单位")
    number = decimal(match.group(1)) / 100
    if not 0 <= number <= 1:
        raise ValidationError("费率不在0到100%之间")
    result = format(number, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def parse_eastmoney_fee_html(raw_bytes: bytes, *, symbol: str = "018147") -> dict:
    """Parse only the retained, validated 018147 C-share fee-table semantics.

    Returns a payload and one-based table/row locator, not a dated evidence
    record. The document has no verified fee effective/publication date. Neither
    its manager names nor its header assets establish appointment/combined scope.
    """
    if symbol != "018147":
        raise ValidationError("该静态费率适配器只核实过018147的C类页面")
    return _fee_tables(_html(raw_bytes))


def _fee_tables(parser: _Tables) -> dict:
    expected = _FUND_NAME + "(018147)基金费率"
    if len(parser.titles) != 1 or not parser.titles[0].startswith(expected):
        raise ValidationError("费率页面的代码、名称或C类份额身份不匹配")

    annual_matches, subscription_matches, redemption_matches = [], [], []
    for number, table in enumerate(parser.tables, 1):
        rows = table["rows"]
        if not rows:
            continue
        first = [text for _, text in rows[0]]
        if first[::2] == ["管理费率", "托管费率", "销售服务费率"]:
            annual_matches.append((number, table))
        if first == ["适用金额", "费率"]:
            subscription_matches.append((number, table))
        if first == ["适用期限", "赎回费率"]:
            redemption_matches.append((number, table))
    if any(len(matches) != 1 for matches in
           (annual_matches, subscription_matches, redemption_matches)):
        raise ValidationError("必要费率表缺失或有多张冲突表；网页壳不是有效资料")
    annual_number, annual = annual_matches[0]
    subscription_number, subscription = subscription_matches[0]
    redemption_number, redemption = redemption_matches[0]
    for table in (annual, subscription, redemption):
        if table["errors"]:
            raise ValidationError("费率表结构不完整或包含不可解释的合并单元格")
    if (len(annual["rows"]) != 1 or len(annual["rows"][0]) != 6
            or any(tag != "td" for tag, _ in annual["rows"][0])):
        raise ValidationError("年度直接费用表格式不符合已核实口径")
    annual_values = [text for _, text in annual["rows"][0]]
    if (len(subscription["rows"]) != 2
            or [text for _, text in subscription["rows"][1]][:1] != ["---"]
            or len(subscription["rows"][1]) != 2):
        raise ValidationError("申购费不是已核实的单一展示费率；不推断分段费率")
    for table in (subscription, redemption):
        if (any(tag != "th" for tag, _ in table["rows"][0])
                or any(len(row) != 2 or any(tag != "td" for tag, _ in row)
                       for row in table["rows"][1:])):
            raise ValidationError("申赎费率表的列或表头不符合已核实口径")
    periods = ["小于7天", "大于等于7天，小于30天", "大于等于30天"]
    if [row[0][1] for row in redemption["rows"][1:]] != periods:
        raise ValidationError("赎回期限缺失、交叠或不符合已核实的7天/30天边界")
    limits = [(0, 6), (7, 29), (30, None)]  # max_days is inclusive in the shared contract.
    payload = {
        "management_rate": _percent(annual_values[1], annual=True),
        "custody_rate": _percent(annual_values[3], annual=True),
        "sales_service_rate": _percent(annual_values[5], annual=True),
        "subscription_rate": _percent(subscription["rows"][1][1][1]),
        "redemption_bands": [
            {"min_days": low, "max_days": high, "rate": _percent(row[1][1])}
            for (low, high), row in zip(limits, redemption["rows"][1:])
        ],
        "holding_day_basis": "unknown",
        "channel_verified": False,
    }
    return {
        "family": "fees", "payload": payload,
        "identity": {"symbol": "018147", "name": _FUND_NAME, "share_class": "C"},
        "locator": (f"table:{annual_number};row:1|table:{subscription_number};row:2|"
                    f"table:{redemption_number};row:2-4"),
        "parser_version": FEE_PARSER_VERSION,
        "warnings": ["这是东方财富上游的二手费率披露，不是独立官方认证。",
                     "持有日计算口径、当前渠道限制及历史生效版本未核实。"],
    }


def _public_url(value: str) -> str:
    result = normalize_source_url(value)
    parts = urlsplit(result)
    host = parts.hostname or ""
    if (parts.scheme != "https" or parts.port not in (None, 443)
            or host.endswith((".invalid", ".test", ".example"))
            or re.fullmatch(r"[0-9.]+", host)):
        raise ValidationError("公开文件只接受无凭据的标准HTTPS公开来源地址")
    secret_keys = {"token", "access_token", "auth", "authorization", "cookie", "password",
                   "passwd", "session", "sessionid", "signature", "api_key", "apikey"}
    if any(key.lower() in secret_keys for key, _ in parse_qsl(parts.query)):
        raise ValidationError("公开来源地址不能包含认证或会话凭据")
    return result


def _origin(url: str) -> tuple[str, str]:
    host = urlsplit(url).hostname or ""
    upstream = host
    if any(host == domain or host.endswith("." + domain)
           for domain in ("eastmoney.com", "dfcfw.com", "1234567.com.cn")):
        upstream = "eastmoney"
    elif host == "ccbfund.cn" or host.endswith(".ccbfund.cn"):
        upstream = "ccbfund"
    return "reviewed_public_document:" + host, upstream


def _reject_constant(value: str) -> None:
    raise ValidationError(f"原始JSON包含非有限值: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("原始JSON包含重复键")
        result[key] = value
    return result


def _document(raw: bytes, content_type: str | None) -> tuple[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_CAPTURE_BYTES:
        raise ValidationError("公开原件必须是非空bytes，且最多8MiB")
    beginning = raw.lstrip(b"\xef\xbb\xbf \r\n\t")
    inferred = ("application/pdf" if beginning.startswith(b"%PDF-") else
                "text/html" if re.match(br"(?:<!doctype\s+html\b|<html\b)", beginning, re.I)
                else "application/json" if beginning[:1] in (b"{", b"[") else None)
    selected = content_type.split(";", 1)[0].strip().lower() if isinstance(content_type, str) else inferred
    if selected != inferred or selected not in {"application/pdf", "text/html", "application/json"}:
        raise ValidationError("只接受类型一致的公开PDF、完整HTML或JSON原始文件")
    if selected == "application/pdf":
        if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-2048:]:
            raise ValidationError("PDF缺少基本文件头或结束标记；不执行PDF或OCR")
        return selected, None
    if len(raw) > MAX_WEB_BYTES:
        raise ValidationError("HTML/JSON原始文件最多2MiB")
    if selected == "text/html":
        return selected, _html(raw)
    try:
        value = json.loads(raw.decode("utf-8-sig", "strict"),
                           parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        bounded_json(value, max_bytes=MAX_WEB_BYTES)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("原始文件不是有界的有效JSON") from exc
    return selected, value


def _locator(value: Any, content_type: str, parsed: Any) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 1000
            or any(ord(char) < 32 for char in value)):
        raise ValidationError("每条资料必须提供有界的页/表/行或JSON定位")
    if content_type == "application/pdf":
        # PDF content is manually reviewed, never executed or silently OCR'd.
        if not re.fullmatch(r"page:[1-9]\d{0,4};(?:table:[^;|]{1,80};row:[1-9]\d{0,5}(?:-[1-9]\d{0,5})?|section:[^;|]{1,200})", value):
            raise ValidationError("PDF定位格式应为page:3;table:2;row:1-4或page:3;section:费用")
        row = re.search(r";row:(\d+)-(\d+)$", value)
        if row and int(row.group(1)) > int(row.group(2)):
            raise ValidationError("PDF行定位起止倒置")
    elif content_type == "text/html":
        references = value.split("|")
        if len(references) > 32:
            raise ValidationError("HTML定位表数量超过上限")
        for reference in references:
            match = re.fullmatch(r"table:([1-9]\d{0,2});row:([1-9]\d{0,3})(?:-([1-9]\d{0,3}))?", reference)
            if match is None:
                raise ValidationError("HTML定位格式应为table:6;row:1|table:8;row:2-4")
            table_number, low, high = map(int, (match.group(1), match.group(2), match.group(3) or match.group(2)))
            if table_number > len(parsed.tables) or low > high:
                raise ValidationError("HTML表/行定位不存在或起止倒置")
            table = parsed.tables[table_number - 1]
            # Manual references name physical rows, not inferred economic columns.
            # A valid colspan/rowspan does not make such a row incomplete; the
            # automatic fee parser separately requires its exact unmerged layout.
            structural_errors = [error for error in table["errors"] if error != "spanning cell"]
            if high > len(table["rows"]) or structural_errors:
                raise ValidationError("HTML定位没有对应的完整表格行")
            rows = table["rows"][low - 1:high]
            if not any(tag == "td" and text for row in rows for tag, text in row):
                raise ValidationError("HTML定位只有空表或表头，没有实际数据行")
    else:
        if not value.startswith("json:/"):
            raise ValidationError("JSON定位必须使用json:/path形式的JSON Pointer")
        current = parsed
        for token in value[6:].split("/"):
            if re.search(r"~(?![01])", token):
                raise ValidationError("JSON Pointer转义无效")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and re.fullmatch(r"0|[1-9]\d*", token) and int(token) < len(current):
                current = current[int(token)]
            else:
                raise ValidationError("JSON定位在原始文件中不存在")
        if current is None or current == "" or current == [] or current == {}:
            raise ValidationError("JSON定位只有空值，不是实际数据")
    return value


def _actual_product(store: Any, product: Product) -> None:
    if not isinstance(product, Product) or product.is_demo or store.get_product(product.id) != product:
        raise ValidationError("公开资料必须匹配Store内实际登记的非演示产品身份")


def _research_citation_checks(
    store: Any, product: Product, note: dict, as_of: str, records: dict[str, dict],
    *, allowed_product_ids: set[str] | None = None,
) -> None:
    allowed = allowed_product_ids if allowed_product_ids is not None else {product.id}
    for citation in note["citations"]:
        if citation["kind"] in {"online_evidence", "calculation"}:
            if timestamp(citation["available_at"]) > timestamp(as_of):
                raise ValidationError("采用引用不能晚于本次实际追加时间")
            continue
        if citation["kind"] != "archived_evidence":
            if timestamp(citation["checked_at"]) > timestamp(as_of):
                raise ValidationError("引用核对时间不能晚于本次实际导入开始时间")
            continue
        referenced = records.get(citation["record_id"])
        if (referenced is None or referenced["family"] == "research_thesis"
                or referenced["product_id"] not in allowed):
            raise ValidationError("存档引用必须指向本产品或明确候选在实际导入时已可用的原始证据，不是另一篇作者研究")
        links = store.advice_evidence_links(referenced["product_id"], referenced["id"], as_of)
        if not any(link["capture_id"] == citation["capture_id"] and link["locator"] == citation["locator"]
                   for link in links):
            raise ValidationError("引用的记录、采集及定位必须匹配实际时点已经存在的同一证据链接")


def _research_revision_check(note: dict, existing: dict[str, dict]) -> None:
    same_note = [row for row in existing.values()
                 if row["family"] == "research_thesis" and row["payload"]["note_id"] == note["note_id"]]
    revision = note["revision"]
    if any(row["payload"]["revision"]["id"] == revision["id"] for row in same_note):
        raise ValidationError("研究内容修改必须使用新的修订标识；不能覆盖既有版本")
    predecessor = revision["supersedes"]
    if predecessor is None:
        if same_note:
            raise ValidationError("已有论题的新版本必须明确supersedes，不能另造无前序原始版本")
        return
    prior = existing.get(predecessor)
    if (prior is None or prior["family"] != "research_thesis"
            or prior["payload"]["note_id"] != note["note_id"]):
        raise ValidationError("研究修订必须替代本产品同一论题在实际时点已经存在的记录")
    if note["research_date"] < prior["payload"]["research_date"]:
        raise ValidationError("研究修订日期不能倒退到被替代版本之前")
    if any(row["payload"]["revision"]["supersedes"] == predecessor for row in same_note):
        raise ValidationError("被替代版本已有后续修订；新修订必须接续当前研究版本")


def import_local_research_note(
    store: Any, product: Product, note: Mapping | bytes, *, clock: Callable[[], str] = now_iso,
) -> dict:
    """Append a bounded, locally authored thesis using actual import clocks.

    ``note`` is a strict note mapping or its original UTF-8 JSON bytes. A mapping
    is retained as canonical submitted JSON bytes, not as publisher raw content.
    Author/reviewer names are attestations; review completion is assigned here.
    No retrieval, personal data, model outputs or schema initialization occurs.
    An identical semantic reimport keeps the first record, capture and clocks.
    """
    _actual_product(store, product)
    if not store.has_fund_review_schema():
        raise ValidationError("请先显式初始化资料表，再导入本地研究笔记")
    started = timestamp(clock())
    if isinstance(note, bytes):
        raw = note
        payload = _research_raw_note(raw)
    elif isinstance(note, Mapping):
        raw = bounded_json(note).encode("utf-8")
        payload = normalize_research_thesis(note)
    else:
        raise ValidationError("本地研究笔记必须是JSON对象或原始JSON字节")
    if payload.get("schema_version") == 2:
        raise ValidationError("API生成研究必须通过append_generated_research追加，不能作为人工复核笔记导入")
    if any(payload["product"][key] != getattr(product, key)
           for key in ("id", "symbol", "name", "kind", "share_class", "exchange")):
        raise ValidationError("研究笔记的代码、名称、份额及市场身份必须匹配实际登记产品")
    source = "local_authored_research"
    uri = research_thesis_source_uri(product.id, payload)
    record = normalize_record({
        "product_id": product.id, "family": "research_thesis", "source": source,
        "upstream": source, "source_url": uri, "report_date": payload["research_date"],
        "observed_at": started.isoformat(), "available_at": started.isoformat(), "payload": payload,
    })
    # Evidence availability and revision heads are checked under the same writer
    # transaction as insertion; a concurrent identical import still reuses first.
    with store.transaction():
        existing = {row["id"]: row for row in store.advice_evidence(product.id, started.isoformat())}
        _research_citation_checks(store, product, payload, started.isoformat(), existing)
        previous = existing.get(record["id"])
        if previous is None:
            _research_revision_check(payload, existing)
        completed = timestamp(clock())
        if completed < started:
            raise ValidationError("研究复核完成时钟不能早于实际导入开始")
        capture_ids = []
        if previous is not None:
            stored = previous
        else:
            record["available_at"] = completed.isoformat()
            capture = {
                "product_id": product.id, "family": "research_thesis", "source": source,
                "upstream": source, "source_url": uri, "observed_at": started.isoformat(),
                "status": "reviewed", "content_type": "application/json", "raw_bytes": raw,
                "parser_version": RESEARCH_PARSER_VERSION, "warnings": [_RESEARCH_WARNING],
                "metadata": {
                    "reviewed": True, "reviewed_at": completed.isoformat(),
                    "validation": "local_authored_assessment_not_publisher_authentication",
                    "retrieval": "local_note_import_not_live_fetch",
                },
            }
            capture_id = store.record_advice_capture(capture)
            capture_ids.append(capture_id)
            stored = store.add_advice_evidence(record, capture_id, "json:/")
    warnings = [_RESEARCH_WARNING]
    if previous is not None:
        warnings.append("相同研究内容已存在：保留首次可用与复核时间，再导入不续鲜。")
    if any(row["kind"] == "linked_not_archived" for row in payload["citations"]):
        warnings.append("仅有公开链接的主张仍未确证；没有下载或归档来源原件。")
    if any(row["kind"] == "reviewed_excerpt" for row in payload["citations"]):
        warnings.append("人工摘录及文本哈希只证明所录入文字，不能证明完整原件或来源方认证。")
    return {
        "product_id": product.id, "status": "existing" if previous is not None else "imported",
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "capture_ids": capture_ids,
        "record_ids": [stored["id"]], "records": [stored],
        "records_added": int(previous is None), "captures_added": len(capture_ids),
        "imported_at": completed.isoformat(), "warnings": warnings,
    }


def append_generated_research(
    store: Any, product: Product, payload: Mapping, *, clock: Callable[[], str] = now_iso,
) -> dict:
    """Append AI-generated v2 using existing records/captures/links and real clocks.

    Online citations must already be bound by ``normalize_generated_research``
    against the runner's RAM bundle. Only the final research and adopted excerpts
    or calculation inputs are retained. Program checks are not human review.
    Reusing an identical result preserves its first actual availability.
    """
    _actual_product(store, product)
    if not store.has_fund_review_schema():
        raise ValidationError("请先显式初始化资料表，再追加API研究")
    started = timestamp(clock())
    note = normalize_generated_research(payload, as_of=started.isoformat())
    if any(note["product"][key] != getattr(product, key)
           for key in ("id", "symbol", "name", "kind", "share_class", "exchange")):
        raise ValidationError("API研究的代码、名称、份额及市场身份须匹配登记产品")
    source = research_thesis_source(note)
    session_research = source == "assistant_generated_research"
    uri = research_thesis_source_uri(product.id, note)
    record = normalize_record({
        "product_id": product.id, "family": "research_thesis", "source": source,
        "upstream": source, "source_url": uri, "report_date": note["research_date"],
        "observed_at": started.isoformat(), "available_at": started.isoformat(), "payload": note,
    })
    raw = bounded_json(note).encode("utf-8")
    warning = "AI生成研究；程序核对引用/计算依据不是人工复核，不证明投资判断正确或有效。"
    with store.transaction():
        existing = {row["id"]: row for row in store.advice_evidence(product.id, started.isoformat())}
        generated = timestamp(note["origin"]["generated_at"])
        allowed_product_ids = {product.id}
        related_records = dict(existing)
        for alternative in note["alternatives"]:
            candidate = store.get_product(alternative["product_id"])
            _actual_product(store, candidate)
            allowed_product_ids.add(candidate.id)
            related_records.update({row["id"]: row for row in store.advice_evidence(
                candidate.id, generated.isoformat(),
            )})
        visible_at_generation = {identifier: row for identifier, row in related_records.items()
                                 if timestamp(row["available_at"]) <= generated}
        _research_citation_checks(store, product, note, generated.isoformat(), visible_at_generation,
                                 allowed_product_ids=allowed_product_ids)
        calculation_records = dict(visible_at_generation)
        for citation in note["citations"]:
            for basis in citation.get("record_basis", []):
                calculation_records.setdefault(basis["id"], basis)
        online_ids = {identifier for citation in note["citations"]
                      if citation["kind"] == "online_evidence" for identifier in citation["record_ids"]}
        for citation in note["citations"]:
            if citation["kind"] != "calculation":
                continue
            for identifier in citation["record_ids"]:
                referenced = calculation_records.get(identifier)
                if referenced is None:
                    if identifier in online_ids and citation["input_evidence_ids"]:
                        continue
                    raise ValidationError("计算依据须指向生成时实际可用的经济记录或已绑定在线原文")
                if (referenced["family"] == "research_thesis"
                        or timestamp(referenced["available_at"]) > generated):
                    raise ValidationError("不能以其他研究判断或生成后资料冒充原始计算依据")
        previous = existing.get(record["id"])
        if previous is None:
            _research_revision_check(note, existing)
        completed = timestamp(clock())
        if completed < started:
            raise ValidationError("研究程序核对完成时钟不能早于实际追加开始")
        capture_ids = []
        if previous is not None:
            stored = previous
        else:
            record["available_at"] = completed.isoformat()
            capture_id = store.record_advice_capture({
                "product_id": product.id, "family": "research_thesis", "source": source,
                "upstream": source, "source_url": uri, "observed_at": started.isoformat(),
                "status": "ok", "content_type": "application/json", "raw_bytes": raw,
                "parser_version": "assistant-generated-research-v2" if session_research else "api-generated-research-v2",
                "warnings": [warning],
                "metadata": {
                    "ai_generated": True, "human_reviewed": False,
                    "validation": GENERATED_VALIDATION,
                    "validation_completed_at": completed.isoformat(),
                    "retrieval": "assistant_research_with_adopted_evidence_not_publisher_original" if session_research else
                                 "api_research_with_adopted_evidence_not_publisher_original",
                },
            })
            capture_ids.append(capture_id)
            stored = store.add_advice_evidence(record, capture_id, "json:/")
    warnings = [warning]
    if previous is not None:
        warnings.append("相同API研究已存在；保留首次生成与可用时间，不以再次追加续鲜。")
    return {
        "product_id": product.id, "status": "existing" if previous is not None else "imported",
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "capture_ids": capture_ids,
        "record_ids": [stored["id"]], "records": [stored], "records_added": int(previous is None),
        "captures_added": len(capture_ids), "imported_at": completed.isoformat(), "warnings": warnings,
    }


def import_reviewed_document(
    store: Any, product: Product, *, raw_bytes: bytes, source_url: str,
    records: list[Mapping], reviewed: bool, content_type: str | None = None,
    clock: Callable[[], str] = now_iso,
) -> dict:
    """Validate and atomically import an explicitly reviewed public document.

    Each structured item supplies family/report_date/payload/locator and optional
    economic/publication dates. Product/source identity and actual clocks are
    assigned here, never accepted from the JSON. No network or schema migration
    occurs. Reimporting an existing semantic record preserves its availability
    and capture provenance; local re-review is not a current public-source check.
    """
    _actual_product(store, product)
    if not store.has_fund_review_schema():
        raise ValidationError("请先显式初始化多因素资料表，再导入公开文件")
    if reviewed is not True:
        raise ValidationError("公开文件导入必须显式人工复核并设置reviewed=True")
    started = timestamp(clock())
    url = _public_url(source_url)
    media_type, parsed = _document(raw_bytes, content_type)
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_IMPORT_RECORDS:
        raise ValidationError("结构化资料必须是1到32条的列表")
    bounded_json(records, max_bytes=MAX_RECORD_BYTES)
    source, upstream = _origin(url)
    allowed = {"family", "report_date", "payload", "locator", "period_start", "effective_from",
               "effective_to", "published_at", "publication_precision"}
    normalized, locators = [], {}
    fee_page = (urlsplit(url).hostname == "fundf10.eastmoney.com"
                and urlsplit(url).path == "/jjfl_018147.html")
    if fee_page and (product.symbol != "018147" or product.share_class != "C"):
        raise ValidationError("该费率原件只适用于018147的C类份额")
    if fee_page and media_type != "text/html":
        raise ValidationError("该费率来源需要HTML原始表格")
    fee_parsed = _fee_tables(parsed) if fee_page else None
    for value in records:
        if (not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value)
                or set(value) - allowed or not {"family", "report_date", "payload", "locator"} <= set(value)):
            raise ValidationError("导入记录字段无效；不接受调用方提供的身份、采集、可用或复核时间")
        if value["family"] == "research_thesis":
            raise ValidationError("本地作者研究必须通过import_local_research_note单独导入，不能冒充公开原件")
        locator = _locator(value["locator"], media_type, parsed)
        item = {key: item for key, item in value.items() if key != "locator"}
        item.update(product_id=product.id, source=source, upstream=upstream, source_url=url,
                    observed_at=started.isoformat(), available_at=started.isoformat())
        record = normalize_record(item)
        if record["family"] == "mandate":
            mandate = record["payload"]
            if (mandate["symbol"] != product.symbol
                    or (product.share_class and mandate["share_class"] != product.share_class)):
                raise ValidationError("投资范围的代码/份额身份与实际产品不匹配")
        if record["family"] == "portfolio" and record["payload"]["denominator"] != "nav":
            raise ValidationError("首版只接受以净资产为分母的组合；不猜测转换比例")
        if fee_parsed is not None:
            if (record["family"] != "fees" or record["payload"] != fee_parsed["payload"]
                    or locator != fee_parsed["locator"]):
                raise ValidationError("018147费率记录/定位与真实原始表格不一致；该页面不能证明其他资料家族")
        if record["id"] in locators:
            raise ValidationError("同一次导入不能重复同一经济记录")
        normalized.append(record)
        locators[record["id"]] = locator
    completed = timestamp(clock())
    if completed < started:
        raise ValidationError("导入完成时钟不能早于验证开始时钟")
    for item in normalized:
        item["available_at"] = completed.isoformat()
    captures = {}
    for family in sorted({item["family"] for item in normalized}):
        capture = {
            "product_id": product.id, "family": family, "source": source, "upstream": upstream,
            "source_url": url, "observed_at": started.isoformat(), "status": "reviewed",
            "content_type": media_type, "raw_bytes": raw_bytes,
            "parser_version": FEE_PARSER_VERSION if fee_parsed is not None else PARSER_VERSION,
            "warnings": [_REVIEW_WARNING],
            "metadata": {"reviewed": True, "reviewed_at": completed.isoformat(),
                         "validation": "explicit_review_not_publisher_authentication",
                         "retrieval": "local_public_document_import_not_live_fetch"},
        }
        captures[family] = capture
    capture_ids, stored_records, added = [], [], 0
    with store.transaction():
        # Check again under the writer lock: a concurrent identical import must
        # not create a new successful link or change the old record's provenance.
        existing = {item["id"]: item for item in store.advice_evidence(product.id, completed.isoformat())}
        needed_families = {item["family"] for item in normalized if item["id"] not in existing}
        ids_by_family = {}
        for family in sorted(needed_families):
            capture_id = store.record_advice_capture(captures[family])
            ids_by_family[family] = capture_id
            capture_ids.append(capture_id)
        for item in normalized:
            if item["id"] in existing:
                stored_records.append(existing[item["id"]])
            else:
                stored_records.append(store.add_advice_evidence(
                    item, ids_by_family[item["family"]], locators[item["id"]],
                ))
                added += 1
    warnings = [_REVIEW_WARNING]
    if added < len(normalized):
        warnings.append("相同经济记录已存在：保留首次可用时间，不以再次导入本地文件续鲜。")
    return {
        "product_id": product.id, "status": "imported" if added else "existing",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "capture_ids": capture_ids, "record_ids": [item["id"] for item in stored_records],
        "records": stored_records, "records_added": added, "captures_added": len(capture_ids),
        "imported_at": completed.isoformat(), "warnings": warnings,
    }

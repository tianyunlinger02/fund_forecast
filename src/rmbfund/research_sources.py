"""Read-only, bounded public evidence for the approved historical experiment.

The caller must create a fresh run folder before calling ``acquire_sources``.
Open-fund NAV retrieval belongs to the caller. Downloads are retained separately
from code, parsed as inert HTML/JSON, and never evaluated as JavaScript.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain import Fact, Product, ValidationError, decimal, now_iso
from .providers.akshare import SinaDailyProvider

MAX_BYTES = 2 * 1024 * 1024
HTTP_TIMEOUT = 20
MAX_RETRIES = 1
MAX_EXPERIMENT_END = "2026-10-19"
SOURCE_DIR_NAME = "research-sources"
ACTION_URLS = {
    symbol: f"https://fundf10.eastmoney.com/fhsp_{symbol}.html"
    for symbol in ("510300", "000051")
}
FEE_URL = "https://fund.chinaamc.com/fund/000051/jijinfeilv.shtml"
TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh510300,day,,,640,"
DIVIDEND_HEADERS = ("年份", "权益登记日", "除息日", "每10份分红", "分红发放日")
PER_UNIT_HEADERS = ("年份", "权益登记日", "除息日", "每份分红", "分红发放日")
SPLIT_HEADERS = ("年份", "拆分折算日", "拆分类型", "拆分折算比例")
OHLC = ("open", "high", "low", "close")


class SourceError(ValueError):
    """A source failed transport limits or strict schema validation."""


def _write_json(path: Path, value: Any) -> bytes:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(body)
    return body


def _hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _http_get(url: str, timeout: float) -> tuple[int, bytes]:
    """One bounded GET using ordinary certificate verification, without auth."""
    request = Request(url, headers={"User-Agent": "rmbfund-research/0.1", "Accept-Encoding": "identity"})
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as exc:
            response = exc  # Preserve the actual HTTP error body as evidence too.
        with response:
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise SourceError("response exceeds 2 MiB")
                chunks.append(chunk)
            return int(response.status), b"".join(chunks)
    except (URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"network failure: {type(exc).__name__}: {exc}") from exc


def _fetch(
    root: Path, name: str, url: str, filename: str, *, kind: str, source: str,
    evidence: list[dict[str, Any]], warnings: list[str],
) -> tuple[bytes | None, dict[str, Any], Path]:
    """Keep each request attempt in its own newly-created empty directory."""
    for attempt in range(1, MAX_RETRIES + 2):
        directory = root / f"{name}-attempt-{attempt}"
        directory.mkdir(exist_ok=False)
        item: dict[str, Any] = {"kind": kind, "source": source, "source_url": url, "attempt": attempt}
        body: bytes | None = None
        try:
            status, body = _http_get(url, HTTP_TIMEOUT)
            if len(body) > MAX_BYTES:
                raise SourceError("response exceeds 2 MiB")
            item.update(http_status=status, status="downloaded" if status == 200 else "http_error")
            path = directory / filename
            path.write_bytes(body)
        except (SourceError, TimeoutError, OSError) as exc:
            item.update(status="network_error", error=f"{type(exc).__name__}: {exc}")
            path = directory / "error.json"
            body = _write_json(path, {"error": item["error"], "source_url": url})
        item.update(observed_at=now_iso(), path=str(path.resolve()), content_hash=_hash(body))
        evidence.append(item)
        _write_json(directory / "metadata.json", item)
        if item["status"] == "downloaded":
            return body, item, directory
        warnings.append(f"{name} attempt {attempt}: {item.get('error', item.get('http_status'))}")
        # A denial is not a reason to bypass access controls or keep asking.
        if item.get("http_status") in (400, 401, 403, 404, 429):
            break
    return None, item, directory


class _TableParser(HTMLParser):
    """Collect cells and retain structural errors instead of repairing tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self.table: dict[str, Any] | None = None
        self.row: list[tuple[str, str]] | None = None
        self.cell: list[str] | None = None
        self.cell_tag = ""
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.depth += 1
            if self.table is None:
                self.table = {"rows": [], "errors": []}
            else:
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
                else:
                    if self.cell is not None:
                        self.table["errors"].append("unclosed cell")
                        self._finish_cell()
                    self.cell, self.cell_tag = [], tag

    def handle_data(self, data: str) -> None:
        if self.cell is not None and self.depth == 1:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
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
                if self.cell is None or tag != self.cell_tag:
                    self.table["errors"].append("unmatched cell end")
                self._finish_cell()
            elif tag == "tr":
                if self.cell is not None:
                    self.table["errors"].append("unclosed cell")
                    self._finish_cell()
                if self.row is None:
                    self.table["errors"].append("unmatched row end")
                self._finish_row()

    def _finish_cell(self) -> None:
        if self.cell is not None and self.row is not None:
            self.row.append((self.cell_tag, re.sub(r"\s+", "", "".join(self.cell))))
        self.cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self.row is not None and self.table is not None:
            self.table["rows"].append(self.row)
        self.row = None


def _table_rows(
    tables: list[dict[str, Any]], headers: tuple[tuple[str, ...], ...], no_data: str,
) -> tuple[list[list[str]], tuple[str, ...]]:
    matches = [table for table in tables if table["rows"] and
               tuple(text for _, text in table["rows"][0]) in headers]
    if len(matches) != 1:
        raise SourceError("missing or ambiguous required action table headers")
    table = matches[0]
    header = tuple(text for _, text in table["rows"][0])
    if table["errors"] or any(tag != "th" for tag, _ in table["rows"][0]):
        raise SourceError("malformed action table structure")
    rows = table["rows"][1:]
    if not rows or any(any(tag != "td" for tag, _ in row) for row in rows):
        raise SourceError("action table has no explicit data/no-data rows")
    values = [[text for _, text in row] for row in rows]
    if len(values) == 1 and len(values[0]) == 1 and re.fullmatch(re.escape(no_data) + r"[!！。]?", values[0][0]):
        return [], header
    if any(len(row) != len(header) for row in values):
        raise SourceError("action table row width differs from its headers")
    return values, header


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise SourceError(f"{label} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise SourceError(f"invalid {label}: {value}") from exc


def _cash(value: str, *, per_ten: bool) -> str:
    # The header must specify the unit even for a bare numeric amount. A phrase
    # in the cell, if present, must agree with that header; never guess /10.
    number = r"[+]?(?:\d+(?:\.\d*)?|\.\d+)"
    if re.fullmatch(number, value):
        amount = decimal(value)
    else:
        match = re.fullmatch(r"每(10)?份(?:派现金|分红)(" + number + r")元", value)
        if match is None or bool(match.group(1)) != per_ten:
            raise SourceError("cash cell lacks explicit matching per-unit/per-10 units")
        amount = decimal(match.group(2))
    return format(amount / (10 if per_ten else 1), "f")


def _parse_action_tables(
    body: bytes, *, symbol: str, source_url: str, observed_at: str, end: str,
    start: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parser = _TableParser()
    parser.feed(body.decode("utf-8", "strict"))
    parser.close()
    dividend_rows, headers = _table_rows(
        parser.tables, (DIVIDEND_HEADERS, PER_UNIT_HEADERS), "暂无分红信息"
    )
    split_rows, _ = _table_rows(parser.tables, (SPLIT_HEADERS,), "暂无拆分信息")
    actions: list[dict[str, Any]] = []
    provenance = {"source_url": source_url, "observed_at": observed_at, "source": "eastmoney:fhsp_html"}
    for year, record, ex, cash, pay in dividend_rows:
        record, ex, pay = (_day(value, label) for value, label in
                           ((record, "record_date"), (ex, "ex_date"), (pay, "pay_date")))
        if start is not None and pay < start:
            # Preserve old anomalous rows in raw HTML, not in an unrelated
            # recent-window cash-flow model. Never repair their source dates.
            continue
        if not re.fullmatch(r"\d{4}年", year) or year[:4] != ex[:4] or not record <= ex <= pay:
            raise SourceError("dividend year/dates are inconsistent")
        amount = _cash(cash, per_ten=headers == DIVIDEND_HEADERS)
        if ex <= end:
            actions.append({"record_date": record, "ex_date": ex, "pay_date": pay,
                            "cash_per_unit": amount, **provenance})
    splits = []
    for year, ex, split_type, raw_ratio in split_rows:
        ex = _day(ex, "split_date")
        match = re.fullmatch(r"1:([+]?(?:\d+(?:\.\d*)?|\.\d+))", raw_ratio)
        if not re.fullmatch(r"\d{4}年", year) or year[:4] != ex[:4] or not split_type or match is None:
            raise SourceError("malformed split row")
        ratio = decimal(match.group(1))
        if ratio <= 0:
            raise SourceError("split ratio must be positive")
        split = {"kind": "split", "ex_date": ex, "ratio": format(ratio, "f"),
                 "split_type": split_type, **provenance}
        splits.append(split)
        # Keep every observed split, including 2012, without fabricating record
        # or payment dates. The runner must block a split inside its own window.
        actions.append(split)
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    for action in actions:
        key = action.get("kind", "dividend"), action["ex_date"]
        if key in identities and identities[key] != action:
            raise SourceError("conflicting duplicate action date")
        identities[key] = action
    return sorted(identities.values(), key=lambda row: row["ex_date"]), {
        "symbol": symbol, "dividend_table_parsed": True, "split_table_parsed": True,
        "dividend_rows_observed": len(dividend_rows), "split_rows_observed": len(split_rows),
        "splits": splits, "coverage_meaning": "observed secondary tables parsed; not official verification",
    }


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.ignored:
            self.ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def _fee_markers(body: bytes) -> dict[str, Any]:
    parser = _TextParser()
    parser.feed(body.decode("gb18030", "strict"))
    parser.close()
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    markers = {marker: marker in text for marker in ("1.20%", "1.50%", "0.50%")}
    sections = {marker: marker in text for marker in ("申购费率", "基金赎回费率")}
    return {"markers": markers, "sections": sections, "encoding": "gb18030",
            "current_page_only": True, "historical_applicability_verified": False,
            "status": "ok" if all(markers.values()) and all(sections.values()) else "incomplete"}


def _reject_constant(value: str) -> None:
    raise SourceError(f"non-finite JSON constant: {value}")


def _tencent_rows(body: bytes, *, end: str) -> dict[str, dict[str, str]]:
    value = json.loads(body.decode("utf-8", "strict"), parse_constant=_reject_constant)
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise SourceError("Tencent missing data object")
    instrument = value["data"].get("sh510300")
    if not isinstance(instrument, dict) or not isinstance(instrument.get("day"), list):
        raise SourceError("Tencent missing raw data.sh510300.day (never use qfqday)")
    rows = instrument["day"]
    if not rows or len(rows) > 640:
        raise SourceError("Tencent day rows must be nonempty and bounded to 640")
    parsed: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            raise SourceError("Tencent day row must be [date,open,close,high,low,volume]")
        day = _day(row[0], "Tencent date")
        prices = {key: decimal(raw) for key, raw in zip(("open", "close", "high", "low"), row[1:5])}
        if any(price <= 0 for price in prices.values()):
            raise SourceError("Tencent OHLC must be positive")
        if prices["high"] < max(prices.values()) or prices["low"] > min(prices.values()):
            raise SourceError("Tencent OHLC high/low relationship is invalid")
        normalized = {key: format(price, "f") for key, price in prices.items()}
        if day in parsed and any(decimal(parsed[day][key]) != prices[key] for key in OHLC):
            raise SourceError("Tencent duplicate day conflicts")
        if day <= end:
            parsed[day] = normalized
    return parsed


def _compare_ohlc(facts: list[Fact], tencent: dict[str, dict[str, str]]) -> dict[str, Any]:
    sina = {fact.event_date: fact.payload for fact in facts if fact.kind == "bar"}
    common = sorted(sina.keys() & tencent.keys())
    mismatches = [
        {"date": day, "field": field, "sina": str(sina[day][field]), "tencent": tencent[day][field]}
        for day in common for field in OHLC
        if decimal(sina[day][field]) != decimal(tencent[day][field])
    ]
    return {"status": "conflict" if mismatches else ("ok" if common else "no_overlap"),
            "blocked": bool(mismatches), "common_dates": len(common),
            "checked_fields": list(OHLC), "mismatches": mismatches}


def acquire_sources(etf: Product, output_dir: Path, *, end: str) -> dict[str, Any]:
    """Return ETF facts, action candidates, and retained independent evidence.

    Action coverage means both secondary-source tables were observed and parsed;
    it does not certify completeness or official verification. Split rows remain
    explicit ``kind='split'`` evidence so the runner can reject relevant splits.
    Tencent failures are nonfatal, but actual OHLC conflicts set ``blocked=True``
    and retain every mismatch; neither source is substituted or stitched.
    """
    if not isinstance(etf, Product) or etf.kind != "etf" or etf.is_demo or etf.symbol != "510300":
        raise ValidationError("acquire_sources requires the approved non-demo 510300 ETF")
    end = _day(end, "end")
    if end > MAX_EXPERIMENT_END:
        raise ValidationError("experiment end must not exceed 2026-10-19")
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ValidationError("output_dir must already be a fresh run directory")
    root = output_dir / SOURCE_DIR_NAME
    root.mkdir(exist_ok=False)
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = [
        "权益覆盖仅表示二手源表解析成功，不代表官方完整核验。",
        "官方费率页只证明当前页面标记，不证明历史渠道费率；不能重复扣除净值内运作费用。",
    ]

    sina_dir = root / "sina-etf-1023"
    sina_dir.mkdir(exist_ok=False)
    result = SinaDailyProvider(timeout=HTTP_TIMEOUT, retries=MAX_RETRIES, min_interval=0, datalen=1023).fetch(
        etf, now_iso(), start_date="1990-01-01", end_date=end
    )
    body = _write_json(sina_dir / "provider_result.json", result.to_dict())
    evidence.append({"kind": "provider_result", "source": "sina:CN_MarketData.getKLineData",
                     "source_url": result.source_url, "observed_at": now_iso(), "status": result.status,
                     "path": str((sina_dir / "provider_result.json").resolve()), "content_hash": _hash(body),
                     "raw_row_count": len(result.raw_records), "fact_count": len(result.facts)})
    warnings.extend(result.warnings)
    # No lower bound: preserve all <=end facts for the runner's warmup selection.
    facts = [fact for fact in result.facts if fact.event_date <= end]
    actions: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in ACTION_URLS}
    coverage = {symbol: False for symbol in ACTION_URLS}
    for symbol, url in ACTION_URLS.items():
        body, item, directory = _fetch(root, f"eastmoney-actions-{symbol}", url, "raw.html",
                                        kind="actions_html", source="eastmoney:fhsp_html",
                                        evidence=evidence, warnings=warnings)
        if body is None:
            continue
        try:
            actions[symbol], details = _parse_action_tables(
                body, symbol=symbol, source_url=url, observed_at=item["observed_at"], end=end
            )
            coverage[symbol] = True
            item.update(details, status="parsed")
        except (ValueError, TypeError, KeyError) as exc:
            item.update(status="parse_error", error=f"{type(exc).__name__}: {exc}")
            warnings.append(f"{symbol} action table parse failed; coverage false: {exc}")
        _write_json(directory / "metadata.json", item)

    body, fee, directory = _fetch(root, "china-amc-fees-000051", FEE_URL, "raw.html",
                                   kind="official_fee_html", source="china_amc:fee_html",
                                   evidence=evidence, warnings=warnings)
    if body is not None:
        try:
            fee.update(_fee_markers(body))
            if fee["status"] != "ok":
                warnings.append("官方费率页未观察到全部 1.20%、1.50%、0.50% 费率与相应段落标记")
        except (ValueError, TypeError) as exc:
            fee.update(status="parse_error", error=f"{type(exc).__name__}: {exc}")
            warnings.append(f"official fee page parse failed: {exc}")
        _write_json(directory / "metadata.json", fee)

    body, item, directory = _fetch(root, "tencent-510300", TENCENT_URL, "raw.json",
                                   kind="tencent_ohlc", source="tencent:fqkline",
                                   evidence=evidence, warnings=warnings)
    crosscheck: dict[str, Any] = {"status": "unavailable", "blocked": False, "mismatches": [],
                                  "common_dates": 0, "checked_fields": list(OHLC),
                                  "source_url": TENCENT_URL, "observed_at": item["observed_at"]}
    if body is not None:
        try:
            crosscheck.update(_compare_ohlc(facts, _tencent_rows(body, end=end)))
            if crosscheck["blocked"]:
                warnings.append("Sina/Tencent 原价 OHLC 存在冲突；保留差异并阻断，禁止拼源")
            elif crosscheck["status"] == "no_overlap":
                warnings.append("Sina/Tencent 没有共同日期，未完成独立 OHLC 核对")
        except (ValueError, TypeError, KeyError) as exc:
            crosscheck["error"] = f"{type(exc).__name__}: {exc}"
            warnings.append(f"Tencent crosscheck unavailable (nonfatal): {exc}")
    else:
        crosscheck["error"] = item.get("error", f"HTTP {item.get('http_status')}")
        warnings.append("Tencent crosscheck network unavailable (nonfatal)")
    item.update(status=crosscheck["status"], crosscheck=crosscheck)
    _write_json(directory / "metadata.json", item)

    _write_json(root / "manifest.json", {"end": end, "etf_product_id": etf.id,
                "action_coverage": coverage, "evidence": evidence, "crosscheck": crosscheck,
                "warnings": warnings})
    return {"etf_facts": facts, "actions": actions, "action_coverage": coverage,
            "evidence": evidence, "crosscheck": crosscheck, "warnings": warnings}

"""Explicit, bounded refresh of source-qualified public investment factors.

Only the documented Nasdaq history, Sina daily form, and Eastmoney yield routes
are used. Parsers are pure; importing this module never opens a connection.
Current capture clocks are not evidence of historical publication/availability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from http.client import HTTPException
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from zoneinfo import ZoneInfo

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
SINA_PAGE_COUNT = 12
SINA_PAGE_DELAY = 1.05
START_DATE = "2024-04-01"
QUALIFIED_END_DATE = "2026-10-23"
SOXX_SPLIT_DATE = date(2026, 11, 5)
SINA_URL = "https://biz.finance.sina.com.cn/forex/forex.php"
EASTMONEY_YIELD_URL = (
    "https://datacenter.eastmoney.com/api/data/get?type=RPTA_WEB_TREASURYYIELD&"
    "sty=ALL&st=SOLAR_DATE&sr=-1&ps=1000&p=1&source=WEB&client=WEB"
)
_PRICE_BASIS = "publisher_close_adjustment_unverified_not_total_return"
_FX_BASIS = "central_parity_fixing_not_spot"
_YIELD_BASIS = "government_bond_yield_percent_not_price"
_ROLES = ("market", "industry", "fx", "yield_2y", "yield_10y")
_NY, _CN = ZoneInfo("America/New_York"), ZoneInfo("Asia/Shanghai")
_HEADERS = {
    "User-Agent": "PublicSourceQualification/1.0",
    "Accept": "application/json,text/html,text/csv,application/xml;q=0.9,*/*;q=0.5",
}
_SINA_HEADINGS = ["日期", "中行汇买价(元)", "中行钞买价(元)",
                  "中行钞卖价/汇卖价", "央行中间价", "中行折算价"]
_RECONSTRUCTION = (
    "Exploratory current-vintage reconstruction only: economic date <= China NAV date "
    "minus 2 calendar days, maximum age 7 days. These are research assumptions, not "
    "publisher release clocks or verified historical availability. Current decisions "
    "must also require actual available_at <= issue time."
)
_WARNINGS = [
    "Current-vintage observations only; historical_available_at is null, not reconstructed from economic dates.",
    _RECONSTRUCTION,
    "Nasdaq Close/Last has an unverified adjustment convention; neither ETF series is verified total return.",
    "FX is central parity in CNY per USD, not spot, CNH, bank buy/sell, or fund valuation FX.",
    "Yields are published percent levels (5 means 5%), not price returns or decimal rates.",
    "Sparse dates and missing values are omitted without zero fill, interpolation, or backfill.",
    "Coverage is bounded: one Eastmoney page and 12 Sina pages; 2023-present coverage is not claimed.",
    "SOXX issuer discloses split-adjusted trading from 2026-11-05; refresh on/after that date requires requalification.",
]


class FactorProviderError(ValueError):
    """Invalid input, response, or source access; never triggers another source."""


def _date(value: object) -> date:
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FactorProviderError("source date must be YYYY-MM-DD") from exc


def _clock(value: datetime | str) -> datetime:
    try:
        if isinstance(value, str) and len(value) <= 64:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError
        return value.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise FactorProviderError("capture/observation time must include a valid timezone") from exc


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clocks(completed_at: datetime | str, observed_at: datetime | str | None) -> tuple:
    completed = _clock(completed_at)
    observed = None if observed_at is None else _clock(observed_at)
    if observed is not None and observed > completed:
        raise FactorProviderError("observed_at cannot be after actual fetch completion")
    return completed, observed


def _number(value: object, *, positive: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise FactorProviderError("source value must be numeric, not missing or boolean")
    text = str(value)
    if len(text) > 64:
        raise FactorProviderError("source numeric value exceeds the parser bound")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise FactorProviderError("source value is not numeric") from exc
    if (not number.is_finite() or len(number.as_tuple().digits) > 32
            or not -20 <= number.as_tuple().exponent <= 20 or (positive and number <= 0)):
        raise FactorProviderError("source value is nonfinite, out of bounds, or not positive")
    return format(number, "f")


def _body(raw: bytes) -> bytes:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise FactorProviderError("source body must be nonempty bytes, at most 16 MiB")
    return raw


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FactorProviderError("duplicate source JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FactorProviderError(f"nonfinite JSON constant: {value}")


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(_body(raw).decode("utf-8-sig", "strict"), parse_float=Decimal,
                           object_pairs_hook=_unique, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError, InvalidOperation) as exc:
        raise FactorProviderError("source is not valid bounded UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise FactorProviderError("source JSON root must be an object")
    return value


def _nasdaq_url(symbol: str, end: str) -> str:
    if symbol not in {"EEM", "SOXX"}:
        raise FactorProviderError("only qualified EEM and SOXX symbols are supported")
    return (f"https://api.nasdaq.com/api/quote/{symbol}/historical?assetclass=etf&"
            f"fromdate={START_DATE}&todate={end}&limit=5000")


NASDAQ_EEM_URL = _nasdaq_url("EEM", QUALIFIED_END_DATE)
NASDAQ_SOXX_URL = _nasdaq_url("SOXX", QUALIFIED_END_DATE)


def sina_url(page: int, *, end_date: str = QUALIFIED_END_DATE) -> str:
    """Use the successful full-form page scheme, not an untested AJAX variant."""
    if type(page) is not int or not 1 <= page <= SINA_PAGE_COUNT:
        raise FactorProviderError("Sina page must be 1..12")
    _date(end_date)
    return SINA_URL + "?" + urlencode({"money_code": "USD", "type": "0",
                                      "startdate": START_DATE, "enddate": end_date,
                                      "page": str(page)})


def _request_end(url: str, *, symbol: str | None = None, page: int | None = None) -> date:
    try:
        key = "todate" if symbol else "enddate"
        end = parse_qs(urlsplit(url).query)[key]
        if len(end) != 1:
            raise ValueError
        parsed = _date(end[0])
        expected = _nasdaq_url(symbol, end[0]) if symbol else sina_url(page, end_date=end[0])
        if url != expected or not _date(START_DATE) <= parsed < SOXX_SPLIT_DATE:
            raise ValueError
        return parsed
    except (ValueError, TypeError, KeyError) as exc:
        raise FactorProviderError("URL is outside the exact qualified route/query contract") from exc


def _row(day: date, value: str | None, *, digest: str, completed: datetime,
         observed: datetime | None, zones: tuple[ZoneInfo, ...]) -> dict:
    if any(day >= clock.astimezone(zone).date()
           for clock in (completed, observed) if clock is not None for zone in zones):
        raise FactorProviderError("future or uncompleted economic bar")
    result = {"date": day.isoformat(), "value": value, "available_at": completed.isoformat(),
              "historical_available_at": None, "source_sha256": digest}
    if observed is not None:
        result["observed_at"] = observed.isoformat()
    return result


def _series(series_id: str, label: str, source: str, url: str, currency: str,
            unit: str, basis: str, completed: datetime, rows: list[dict], **metadata) -> dict:
    rows.sort(key=lambda row: row["date"])
    if not rows or len({row["date"] for row in rows}) != len(rows):
        raise FactorProviderError("required series is empty or has duplicate economic dates")
    return {"series_id": series_id, "label": label, "source": source, "source_url": url,
            "currency": currency, "unit": unit, "return_basis": basis,
            "captured_at": completed.isoformat(), "historical_available_at": None,
            "metadata": {"reconstruction_policy": _RECONSTRUCTION,
                         "coverage_start": rows[0]["date"], "coverage_end": rows[-1]["date"],
                         "complete_window": True, **metadata}, "rows": rows}


def parse_nasdaq_history(raw: bytes, *, symbol: str, completed_at: datetime | str,
                         observed_at: datetime | str | None = None,
                         source_url: str | None = None) -> dict:
    """Read the validated Close/Last field; never infer an adjustment convention."""
    url = source_url or _nasdaq_url(symbol, QUALIFIED_END_DATE)
    end = _request_end(url, symbol=symbol)
    completed, observed = _clocks(completed_at, observed_at)
    if completed.astimezone(_NY).date() >= SOXX_SPLIT_DATE:
        raise FactorProviderError("ETF source contract expired at SOXX split; requalification required")
    payload = _json(raw)
    digest = hashlib.sha256(raw).hexdigest()
    status, data = payload.get("status"), payload.get("data")
    if not isinstance(status, dict) or type(status.get("rCode")) is not int or status["rCode"] != 200:
        raise FactorProviderError("Nasdaq application status is not successful")
    if not isinstance(data, dict) or data.get("symbol") != symbol:
        raise FactorProviderError("Nasdaq symbol identity mismatch")
    table = data.get("tradesTable")
    headers = table.get("headers") if isinstance(table, dict) else None
    if not isinstance(headers, dict) or headers.get("date") != "Date" or headers.get("close") != "Close/Last":
        raise FactorProviderError("Nasdaq labelled date/close field contract changed")
    records = table.get("rows")
    if (not isinstance(records, list) or not 0 < len(records) <= 5000
            or type(data.get("totalRecords")) is not int or data["totalRecords"] != len(records)):
        raise FactorProviderError("Nasdaq row count is missing, truncated, or inconsistent")
    rows = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("date"), str):
            raise FactorProviderError("Nasdaq daily row/date is missing")
        try:
            day = datetime.strptime(record["date"], "%m/%d/%Y").date()
            if day.strftime("%m/%d/%Y") != record["date"] or not _date(START_DATE) <= day <= end:
                raise ValueError
        except ValueError as exc:
            raise FactorProviderError("Nasdaq date is invalid or outside the request window") from exc
        value = record.get("close")
        if not isinstance(value, str) or not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
            raise FactorProviderError("Nasdaq close numeric-string contract changed")
        close = _number(value.replace(",", ""), positive=True)
        rows.append(_row(day, close, digest=digest, completed=completed,
                         observed=observed, zones=(_NY,)))
    label = "iShares MSCI Emerging Markets ETF" if symbol == "EEM" else "iShares Semiconductor ETF"
    issuer_url = ("https://www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf"
                  if symbol == "EEM" else
                  "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf")
    return _series(f"nasdaq_{symbol.lower()}_close_usd", f"{label} ({symbol})",
                   "Nasdaq historical API + iShares issuer metadata", url, "USD", "price",
                   _PRICE_BASIS, completed, rows, issuer_metadata_url=issuer_url,
                   economic_timezone="America/New_York", published_field="data.tradesTable.rows[].close",
                   coverage_start=START_DATE, coverage_end=end.isoformat(),
                   corporate_action_requalification_date=SOXX_SPLIT_DATE.isoformat())


def parse_eastmoney_yields(raw: bytes, *, completed_at: datetime | str,
                           observed_at: datetime | str | None = None,
                           source_url: str = EASTMONEY_YIELD_URL) -> dict[str, dict]:
    """Preserve US2/US10 percent levels; omit nulls independently, never as zero."""
    if source_url != EASTMONEY_YIELD_URL:
        raise FactorProviderError("unexpected Eastmoney yield URL")
    completed, observed = _clocks(completed_at, observed_at)
    payload = _json(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if payload.get("success") is not True or type(payload.get("code")) is not int or payload["code"] != 0:
        raise FactorProviderError("Eastmoney application status is not successful")
    result = payload.get("result")
    records = result.get("data") if isinstance(result, dict) else None
    if (not isinstance(records, list) or not 0 < len(records) <= 1000
            or type(result.get("count")) is not int or result["count"] < len(records)
            or type(result.get("pages")) is not int or result["pages"] < 1):
        raise FactorProviderError("Eastmoney yield page/count contract changed")
    fields = {"yield_2y": ("EMG00001306", "2"), "yield_10y": ("EMG00001310", "10")}
    rows: dict[str, list[dict]] = {role: [] for role in fields}
    seen, observed_dates, omitted_uncompleted = set(), set(), 0
    cutoff = min(clock.astimezone(zone).date()
                 for clock in (completed, observed) if clock is not None for zone in (_NY, _CN))
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("SOLAR_DATE"), str):
            raise FactorProviderError("Eastmoney observation date is missing")
        stamp = record["SOLAR_DATE"]
        day = _date(stamp[:10])
        if stamp != day.isoformat() + " 00:00:00" or day in observed_dates:
            raise FactorProviderError("Eastmoney date placeholder changed or is duplicated")
        observed_dates.add(day)
        if day >= cutoff:
            omitted_uncompleted += 1
            continue
        seen.add(day)
        for role, (field, _) in fields.items():
            if field not in record:
                raise FactorProviderError(f"Eastmoney tenor field missing: {field}")
            value = record[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, Decimal))):
                raise FactorProviderError("Eastmoney yield must be a JSON number or null")
            # Validate even dates where both tenors are null.
            row = _row(day, None if value is None else _number(value), digest=digest,
                       completed=completed, observed=observed, zones=(_NY, _CN))
            if value is not None:
                rows[role].append(row)
    if not seen:
        raise FactorProviderError("Eastmoney page has no completed observation dates")
    return {role: _series(f"eastmoney_us_treasury_{tenor}y_pct", f"US Treasury yield {tenor}-year",
                          "Eastmoney RPTA_WEB_TREASURYYIELD (publisher-labeled)", source_url,
                          "USD", "percent", _YIELD_BASIS, completed, rows[role],
                          publisher_field=field, null_observations_omitted=len(seen) - len(rows[role]),
                          omitted_uncompleted_observations=omitted_uncompleted,
                          warnings=([f"Omitted {omitted_uncompleted} uncompleted/future economic dates; raw capture retained."]
                                    if omitted_uncompleted else []),
                          coverage_start=min(seen).isoformat(), coverage_end=max(seen).isoformat(),
                          metadata_url="https://data.eastmoney.com/newstatic/js/cjsj/zmgzsyl.js")
            for role, (field, tenor) in fields.items()}


class _SinaHTML(HTMLParser):
    """Read the visible labelled table and form; ignore script/style contents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored = None
        self.depth = 0
        self.table = self.row = self.cell = None
        self.tables = []
        self.inputs = {}
        self.in_money = False
        self.option = None
        self.selected = []
        self.current_links = []
        self.visible = []

    def handle_starttag(self, tag, attrs):
        if self.ignored:
            return
        if tag in {"script", "style", "template", "noscript"}:
            self.ignored = tag
            return
        attrs = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.table is not None:
                raise FactorProviderError("nested Sina data table")
            if "list_table" in (attrs.get("class") or "").split():
                self.table = []
        elif self.table is not None and tag == "tr":
            if self.row is not None:
                raise FactorProviderError("unclosed Sina data row")
            self.row = []
        elif self.table is not None and tag in {"td", "th"}:
            if (self.row is None or self.cell is not None
                    or any(attrs.get(key, "1") != "1" for key in ("rowspan", "colspan"))):
                raise FactorProviderError("invalid Sina data cell")
            self.cell = []
        if tag == "input" and attrs.get("name") in {"startdate", "enddate"}:
            name = attrs["name"]
            if name in self.inputs:
                raise FactorProviderError("duplicate Sina date control")
            self.inputs[name] = attrs.get("value")
        if tag == "select":
            self.in_money = attrs.get("id") == "money_code" and attrs.get("name") == "money_code"
        elif tag == "option" and self.in_money:
            self.option = [attrs, []]
        if tag == "a" and {"page", "nolink"} <= set((attrs.get("class") or "").split()):
            self.current_links.append(attrs.get("href"))

    def handle_data(self, data):
        if self.ignored:
            return
        self.visible.append(data)
        if self.cell is not None:
            self.cell.append(data)
        if self.option is not None:
            self.option[1].append(data)

    def handle_endtag(self, tag):
        if self.ignored:
            if tag == self.ignored:
                self.ignored = None
            return
        if tag in {"td", "th"} and self.table is not None:
            if self.cell is None or self.row is None:
                raise FactorProviderError("unmatched Sina data cell")
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.table is not None:
            if self.row is None or self.cell is not None or len(self.table) >= 51:
                raise FactorProviderError("invalid or oversized Sina data row")
            self.table.append(self.row)
            self.row = None
        elif tag == "table":
            if self.table is not None:
                if self.row is not None or self.cell is not None:
                    raise FactorProviderError("unclosed Sina data table")
                self.tables.append(self.table)
                self.table = None
            self.depth -= 1
        elif tag == "option" and self.option is not None:
            attrs, text = self.option
            if "selected" in attrs:
                self.selected.append((attrs.get("value"), "".join(text).strip()))
            self.option = None
        elif tag == "select":
            self.in_money = False


def parse_sina_page(raw: bytes, *, page: int, completed_at: datetime | str,
                    observed_at: datetime | str | None = None,
                    source_url: str | None = None,
                    include_coverage: bool = False) -> list[dict] | dict:
    """Read 央行中间价; optional coverage includes dates with missing fixings."""
    url = source_url or sina_url(page)
    end = _request_end(url, page=page)
    completed, observed = _clocks(completed_at, observed_at)
    digest = hashlib.sha256(_body(raw)).hexdigest()
    parser = _SinaHTML()
    try:
        parser.feed(_body(raw).decode("gb18030", "strict"))
        parser.close()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise FactorProviderError(f"Sina static HTML is invalid: {exc}") from exc
    expected_link = "/forex/forex.php?" + urlsplit(url).query
    if (parser.depth or parser.table is not None or len(parser.tables) != 1
            or parser.inputs != {"startdate": START_DATE, "enddate": end.isoformat()}
            or parser.selected != [("USD", "美元")] or set(parser.current_links) != {expected_link}
            or "注：100外币兑人民币" not in "".join("".join(parser.visible).split())):
        raise FactorProviderError("Sina currency, units, dates, pagination, or table identity changed")
    table = parser.tables[0]
    if not 1 < len(table) <= 51 or table[0] != _SINA_HEADINGS:
        raise FactorProviderError("Sina labelled central-parity table contract changed")
    rows, seen = [], set()
    for cells in table[1:]:
        if len(cells) != 6:
            raise FactorProviderError("Sina data row has the wrong column count")
        day = _date(cells[0])
        if not _date(START_DATE) <= day <= end or day in seen:
            raise FactorProviderError("Sina date is duplicate or outside the request window")
        seen.add(day)
        value = cells[4]
        if value not in {"", "--", "-"} and not re.fullmatch(r"\d+(?:\.\d+)?", value):
            raise FactorProviderError("Sina central-parity numeric contract changed")
        quoted = None if value in {"", "--", "-"} else Decimal(_number(value, positive=True))
        # A decimal exponent shift is exactly /100, independent of caller precision.
        parts = None if quoted is None else quoted.as_tuple()
        converted = None if parts is None else Decimal((parts.sign, parts.digits, parts.exponent - 2))
        row = _row(day, None if converted is None else _number(converted, positive=True),
                   digest=digest, completed=completed, observed=observed, zones=(_CN,))
        if converted is not None:
            rows.append(row)
    if include_coverage:
        return {"rows": rows, "coverage_start": min(seen).isoformat(),
                "coverage_end": max(seen).isoformat()}
    return rows


def _fx_series(rows: list[dict], completed: datetime, end: str,
               coverage_start: str, coverage_end: str) -> dict:
    return _series("sina_usd_cny_central_parity", "USD/CNY central-parity fixing proxy",
                   "Sina Finance / Bank of China quote table", sina_url(1, end_date=end), "CNY",
                   "CNY_per_USD", _FX_BASIS, completed, rows, published_field="央行中间价",
                   published_unit="CNY per 100 USD", conversion="divide by 100 exactly",
                   pages=SINA_PAGE_COUNT, economic_timezone="Asia/Shanghai",
                   coverage_start=coverage_start, coverage_end=coverage_end)


def parse_factor_bundle(*, market: Mapping, industry: Mapping, fx_pages: Mapping,
                         yields: Mapping, observed_at: datetime | str | None = None) -> dict:
    """Pure offline entry point; each source has body, completed_at and url keys."""
    series = {
        "market": parse_nasdaq_history(market["body"], symbol="EEM", completed_at=market["completed_at"],
                                        source_url=market["url"], observed_at=observed_at),
        "industry": parse_nasdaq_history(industry["body"], symbol="SOXX", completed_at=industry["completed_at"],
                                          source_url=industry["url"], observed_at=observed_at),
    }
    series.update(parse_eastmoney_yields(yields["body"], completed_at=yields["completed_at"],
                                         source_url=yields["url"], observed_at=observed_at))
    if set(fx_pages) != set(range(1, SINA_PAGE_COUNT + 1)):
        raise FactorProviderError("exactly Sina pages 1..12 are required")
    rows, ends, completions, coverage = [], set(), [], []
    for page in range(1, SINA_PAGE_COUNT + 1):
        item = fx_pages[page]
        ends.add(_request_end(item["url"], page=page).isoformat())
        completions.append(_clock(item["completed_at"]))
        parsed = parse_sina_page(item["body"], page=page, completed_at=item["completed_at"],
                                 source_url=item["url"], observed_at=observed_at, include_coverage=True)
        if coverage and parsed["coverage_end"] >= coverage[-1]["coverage_start"]:
            raise FactorProviderError("Sina pages overlap or do not move backwards")
        coverage.append(parsed)
        rows.extend(parsed["rows"])
    if len(ends) != 1:
        raise FactorProviderError("Sina pages use inconsistent request windows")
    series["fx"] = _fx_series(rows, max(completions), ends.pop(),
                               min(part["coverage_start"] for part in coverage),
                               max(part["coverage_end"] for part in coverage))
    return {"schema_version": 1, "series": series, "warnings": list(_WARNINGS)}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url: str, *, timeout: float) -> tuple[bytes, datetime]:
    """One verified HTTPS request, without proxies, cookies, redirects, or retries."""
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    started, chunks, size = time.monotonic(), [], 0
    try:
        with opener.open(Request(url, headers=_HEADERS, method="GET"), timeout=timeout) as response:
            if response.status != 200:
                raise FactorProviderError(f"HTTP {response.status}; stopped without retry")
            while True:
                if time.monotonic() - started > timeout:
                    raise FactorProviderError("response exceeded the bounded request time")
                chunk = response.read(min(64 * 1024, MAX_RESPONSE_BYTES + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise FactorProviderError("response exceeds 16 MiB; stopped")
                chunks.append(chunk)
    except HTTPError as exc:
        exc.close()
        raise FactorProviderError(f"HTTP {exc.code}; stopped without redirects or retry") from exc
    except (HTTPException, URLError, TimeoutError, OSError) as exc:
        raise FactorProviderError(f"source connection failed ({type(exc).__name__}); no retry") from exc
    return _body(b"".join(chunks)), _now()


def fetch_factor_bundle(*, product_id: str, observed_at: datetime | str | None = None,
                         timeout: float = 15.0) -> dict:
    """Explicit live update; return ready/error and retain partial capture evidence.

    Documented date parameters roll only through the preceding New York calendar
    day (weekends may return the previous trading session). The SOXX issuer's
    2026-11-05 split is a hard requalification boundary. The caller owns caching,
    persistence and preservation of first availability for unchanged observations.
    Raw bodies are base64 strings in raw_records, never in the factor bundle.
    Invalid caller arguments raise FactorProviderError before any request.
    """
    if not isinstance(product_id, str) or not product_id.strip() or len(product_id) > 256:
        raise FactorProviderError("product_id must be a nonempty bounded string")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 60):
        raise FactorProviderError("timeout must be finite and in (0, 60] seconds")
    started = _now()
    observed = started if observed_at is None else _clock(observed_at)
    if observed > started:
        raise FactorProviderError("observed_at cannot be in the future")
    captures, raw_records, series = [], [], {}
    warnings, current_source, current_url = list(_WARNINGS), None, None

    def get(source: str, url: str) -> dict:
        nonlocal current_source, current_url
        current_source, current_url = source, url
        try:
            raw, completed = _fetch(url, timeout=float(timeout))
        except FactorProviderError as exc:
            captures.append({"source": source, "url": url, "completed_at": _now().isoformat(),
                             "content_hash": None, "bytes": None, "status": "error",
                             "body_status": "unavailable_or_incomplete_not_retained", "error": str(exc)})
            raise
        capture = {"source": source, "url": url, "completed_at": completed.isoformat(),
                   "content_hash": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        captures.append(capture)
        raw_records.append({**capture, "encoding": "base64",
                            "body_base64": base64.b64encode(raw).decode("ascii")})
        return {"body": raw, "completed_at": completed, "url": url}

    try:
        if started.astimezone(_NY).date() >= SOXX_SPLIT_DATE:
            raise FactorProviderError("SOXX split boundary reached; source requalification required before refresh")
        end = (observed.astimezone(_NY).date() - timedelta(days=1)).isoformat()
        if _date(end) < _date(START_DATE):
            raise FactorProviderError("requested observation time predates qualified history")
        for role, symbol in (("market", "EEM"), ("industry", "SOXX")):
            item = get(f"nasdaq-{symbol.lower()}-history", _nasdaq_url(symbol, end))
            series[role] = parse_nasdaq_history(item["body"], symbol=symbol,
                                               completed_at=item["completed_at"],
                                               observed_at=observed, source_url=item["url"])
        item = get("eastmoney-yield-page1000", EASTMONEY_YIELD_URL)
        series.update(parse_eastmoney_yields(item["body"], completed_at=item["completed_at"],
                                              observed_at=observed, source_url=item["url"]))
        fx_rows, coverage = [], []
        for page in range(1, SINA_PAGE_COUNT + 1):
            if page > 1:
                time.sleep(SINA_PAGE_DELAY)
            item = get(f"sina-fx-dated-page{page}", sina_url(page, end_date=end))
            parsed = parse_sina_page(item["body"], page=page, completed_at=item["completed_at"],
                                     observed_at=observed, source_url=item["url"], include_coverage=True)
            if coverage and parsed["coverage_end"] >= coverage[-1]["coverage_start"]:
                raise FactorProviderError("Sina pages overlap or do not move backwards; stopped")
            coverage.append(parsed)
            fx_rows.extend(parsed["rows"])
        series["fx"] = _fx_series(fx_rows, item["completed_at"], end,
                                    min(part["coverage_start"] for part in coverage),
                                    max(part["coverage_end"] for part in coverage))
    except FactorProviderError as exc:
        if captures and captures[-1]["url"] == current_url and captures[-1].get("status") != "error":
            captures[-1].update(status="error", error=str(exc))
        warnings.append(str(exc))
        return {"status": "error", "bundle": None, "captures": captures,
                "raw_records": raw_records, "warnings": warnings, "error": str(exc),
                "error_source": current_source, "error_url": current_url,
                "missing_roles": [role for role in _ROLES if role not in series]}
    bundle = {"schema_version": 1, "series": series, "warnings": warnings}
    return {"status": "ready", "bundle": bundle, "captures": captures,
            "raw_records": raw_records, "warnings": warnings}

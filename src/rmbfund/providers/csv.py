"""Strict, bounded CSV evidence import. Files are inert, untrusted text.

Row timestamps are provenance only: they cannot backdate observation or availability.
CSV data is user-supplied/unverified, not proof of upstream historical publication.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from ..domain import Fact, Product, ValidationError, decimal
from .akshare import _date, _number, _observation

MAX_CSV_BYTES = 8 * 1024 * 1024
MAX_CSV_ROWS = 50_000
MAX_CELL_CHARS = 16_384

_COMMON = {"product_id", "symbol", "source", "source_url", "observed_at", "available_at",
           "published_at", "metadata"}
_SCHEMAS: dict[str, tuple[set[str], set[str]]] = {
    "bar": ({"event_date", "open", "high", "low", "close", "adjustment", "currency"},
            {"volume", "volume_unit"}),
    "nav": ({"event_date", "unit_nav", "currency"}, {"accumulated_nav"}),
    "calendar": ({"event_date", "is_open"}, {"exchange"}),
    "announcement": ({"event_date", "title"}, {"text", "url"}),
    "status": ({"event_date", "purchase_allowed", "redeem_allowed"}, {"reason"}),
    "dividend": ({"event_date", "record_date", "ex_date", "pay_date", "cash_per_unit", "currency"}, set()),
    "split": ({"event_date", "record_date", "ex_date", "pay_date", "ratio"}, set()),
}


def _boolean(value: str, field: str) -> bool:
    if value not in {"true", "false"}:
        raise ValidationError(f"{field} 只接受小写 true 或 false")
    return value == "true"


def _payload(row: dict[str, str], kind: str) -> dict[str, Any]:
    if "currency" in row and row["currency"] != "CNY":
        raise ValidationError("只支持明确标记 currency=CNY 的人民币数据")
    if kind == "bar":
        if row["adjustment"] != "raw":
            raise ValidationError("bar 必须显式 adjustment=raw，不接受复权价")
        payload = {key: _number(row[key], key, positive=True)
                   for key in ("open", "high", "low", "close")}
        payload.update(currency="CNY", adjustment="raw")
        if row.get("volume"):
            if row.get("volume_unit") != "share":
                raise ValidationError("volume 必须显式 volume_unit=share（份/股），不能猜手/份")
            payload["volume"] = _number(row["volume"], "volume")
            if decimal(payload["volume"]) < 0:
                raise ValidationError("volume 不得为负")
            payload["volume_unit"] = "share"
        elif row.get("volume_unit"):
            raise ValidationError("有 volume_unit 但缺少 volume")
        return payload
    if kind == "nav":
        payload = {"unit_nav": _number(row["unit_nav"], "unit_nav", positive=True),
                   "currency": "CNY", "nav_type": "user_supplied_unit_nav"}
        if row.get("accumulated_nav"):
            payload["accumulated_nav"] = _number(row["accumulated_nav"], "accumulated_nav", positive=True)
        return payload
    if kind == "calendar":
        payload = {"is_open": _boolean(row["is_open"], "is_open")}
        if row.get("exchange"):
            if row["exchange"] not in {"SSE", "SZSE"}:
                raise ValidationError("exchange 只接受 SSE/SZSE")
            payload["exchange"] = row["exchange"]
        return payload
    if kind == "announcement":
        if not row["title"]:
            raise ValidationError("公告 title 不能为空")
        return {"title": row["title"], "text": row.get("text", ""),
                "url": row.get("url", ""), "requires_manual_verification": True,
                "verified": False}
    if kind == "status":
        purchase = _boolean(row["purchase_allowed"], "purchase_allowed")
        redeem = _boolean(row["redeem_allowed"], "redeem_allowed")
        return {"purchase_allowed": purchase, "redeem_allowed": redeem,
                "subscription": "open" if purchase else "closed",
                "redemption": "open" if redeem else "closed",
                "reason": row.get("reason", ""), "verified": False}
    dates = {key: _date(row[key], key) for key in ("record_date", "ex_date", "pay_date")}
    if not dates["record_date"] <= dates["ex_date"] <= dates["pay_date"]:
        raise ValidationError("权益日期必须 record_date <= ex_date <= pay_date")
    if row["event_date"] != dates["ex_date"]:
        raise ValidationError("权益 event_date 必须等于 ex_date")
    if kind == "dividend":
        cash = _number(row["cash_per_unit"], "cash_per_unit")
        if decimal(cash) < 0:
            raise ValidationError("cash_per_unit 不得为负")
        return {**dates, "cash_per_unit": cash, "currency": "CNY", "verified": False}
    return {**dates, "ratio": _number(row["ratio"], "ratio", positive=True), "verified": False}


def load_csv(path: str | Path, product: Product, observed_at: str | None,
             kind: str) -> list[Fact]:
    """Import one homogeneous schema, rejecting the entire file on any bad row.

    ``observed_at`` is the caller's actual import observation (None means now),
    never a CSV column. ``available_at`` is always that same observation. CSV
    published/available/observed timestamps remain strings in payload metadata.
    """
    if kind not in _SCHEMAS:
        raise ValidationError(f"不支持的 CSV kind: {kind}")
    observed = _observation(observed_at)
    file = Path(path).expanduser().resolve()
    try:
        if not file.is_file():
            raise ValidationError("CSV 路径必须指向普通文件")
        with file.open("rb") as handle:
            # Bound the actual read as well as stat: a growing file cannot evade it.
            if file.stat().st_size > MAX_CSV_BYTES:
                raise ValidationError("CSV 超过 8 MiB 上限")
            raw = handle.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            raise ValidationError("CSV 超过 8 MiB 上限")
        text = raw.decode("utf-8-sig", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"无法读取 UTF-8 CSV: {exc}") from exc
    if "\x00" in text:
        raise ValidationError("CSV 含 NUL 字节")
    if not text.strip():
        raise ValidationError("CSV 为空")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = reader.fieldnames
        if not headers or any(not h or h.strip() != h for h in headers):
            raise ValidationError("CSV 表头为空或含首尾空格")
        if len(set(headers)) != len(headers):
            raise ValidationError("CSV 表头重复")
        required, optional = _SCHEMAS[kind]
        missing = required - set(headers)
        unknown = set(headers) - required - optional - _COMMON
        if missing or unknown:
            raise ValidationError(f"CSV schema 不符，缺少 {sorted(missing)}；未知列 {sorted(unknown)}")
        facts: list[Fact] = []
        seen: dict[str, Fact] = {}
        for line_number, row in enumerate(reader, 2):
            if line_number - 1 > MAX_CSV_ROWS:
                raise ValidationError("CSV 超过 50000 行上限")
            if None in row or any(value is None for value in row.values()):
                raise ValidationError(f"CSV 第 {line_number} 行列数不符")
            if any(len(value) > MAX_CELL_CHARS for value in row.values()):
                raise ValidationError(f"CSV 第 {line_number} 行字段过长")
            if row.get("product_id") and row["product_id"] != product.id:
                raise ValidationError(f"CSV 第 {line_number} 行 product_id 与导入产品不符")
            if row.get("symbol") and row["symbol"] != product.symbol:
                raise ValidationError(f"CSV 第 {line_number} 行 symbol 与导入产品不符")
            try:
                event_date = _date(row["event_date"], "event_date")
                if kind not in {"announcement", "calendar"} and event_date > observed[:10]:
                    raise ValidationError("未来事件不能冒充已发生历史事实；预告请用 announcement")
                payload = _payload(row, kind)
                if kind in {"dividend", "split"} and payload["pay_date"] > observed[:10]:
                    raise ValidationError("尚未到账的权益应导入为 announcement 待核验")
                payload["metadata"] = {
                    "import_format": "csv", "verification": "user_supplied_unverified",
                    "csv_provenance": {key: row[key] for key in sorted(_COMMON)
                                       if key in row and row[key]},
                }
                fact = Fact(product_id=product.id, kind=kind, event_date=event_date,
                            payload=payload, source=f"csv:{file.name}", observed_at=observed,
                            source_url=file.as_uri(), available_at=observed,
                            published_at=None, quality="unverified")
            except (ValidationError, ValueError, KeyError) as exc:
                raise ValidationError(f"CSV 第 {line_number} 行: {exc}") from exc
            # Several different announcements on the same day are legitimate.
            # Single daily NAV/bar/status values, however, must not conflict.
            identity = fact.id if kind == "announcement" else event_date
            if identity in seen:
                if seen[identity].payload != payload:
                    raise ValidationError(f"CSV 同一日期 {event_date} 存在冲突记录")
                continue
            seen[identity] = fact
            facts.append(fact)
    except csv.Error as exc:
        raise ValidationError(f"CSV 语法错误: {exc}") from exc
    if not facts:
        raise ValidationError("CSV 没有数据行")
    return sorted(facts, key=lambda fact: fact.event_date)

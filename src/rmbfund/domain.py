"""Shared data contracts. Monetary values cross persistence boundaries as strings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
SUPPORTED_KINDS = {"etf", "open_fund", "monitor_only"}
FACT_KINDS = {"bar", "nav", "calendar", "status", "announcement", "dividend", "split"}


class ValidationError(ValueError):
    """Invalid input or an unsupported assumption; never replace with a guessed value."""


def decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValidationError("布尔值不是金额或份额")
    raw = str(value)
    if len(raw) > 128:
        raise ValidationError("数值文本过长")
    try:
        result = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError(f"无效十进制数: {value!r}") from exc
    if not result.is_finite():
        raise ValidationError("金额/份额必须为有限数")
    if abs(result.as_tuple().exponent) > 24 or abs(result) > Decimal("1e24"):
        raise ValidationError("数值或精度超出本地研究工具支持范围")
    return result


def timestamp(value: str | datetime) -> datetime:
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValidationError(f"无效时间: {value}") from exc
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValidationError("时间必须包含时区，例如 2026-01-05T16:00:00+08:00")
    return result.astimezone(SHANGHAI)


def now_iso() -> str:
    return datetime.now(SHANGHAI).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Product:
    id: str
    symbol: str
    name: str
    kind: str
    is_demo: bool = False
    exchange: str = ""
    category: str = "未分类"
    share_class: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.symbol or not self.name:
            raise ValidationError("产品代码、名称不能为空")
        if self.kind not in SUPPORTED_KINDS:
            raise ValidationError(f"不支持产品类型: {self.kind}")
        if len(self.id) > 80 or len(self.name) > 200:
            raise ValidationError("产品字段过长")


@dataclass(frozen=True)
class Profile:
    id: str
    product_id: str
    effective_from: str
    observed_at: str
    rules: dict[str, Any]
    source_url: str
    verified: bool = False
    effective_to: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool) or not isinstance(self.rules, dict):
            raise ValidationError("verified必须是布尔值，rules必须是对象")
        if not self.id or not self.product_id:
            raise ValidationError("档案版本与产品标识不能为空")
        date.fromisoformat(self.effective_from)
        timestamp(self.observed_at)
        if self.effective_to:
            if date.fromisoformat(self.effective_to) < date.fromisoformat(self.effective_from):
                raise ValidationError("规则有效期倒置")
        if not self.source_url:
            raise ValidationError("档案必须记录依据来源，演示可用 demo://")


@dataclass(frozen=True)
class Fact:
    product_id: str
    kind: str
    event_date: str
    payload: dict[str, Any]
    source: str
    observed_at: str
    source_url: str = ""
    published_at: str | None = None
    available_at: str | None = None
    quality: str = "observed"
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.kind not in FACT_KINDS:
            raise ValidationError(f"不支持事实类型: {self.kind}")
        date.fromisoformat(self.event_date)
        observed = timestamp(self.observed_at)
        published = timestamp(self.published_at) if self.published_at else None
        available = timestamp(self.available_at) if self.available_at else max(
            observed, published or observed
        )
        if available < observed or (published and available < published):
            raise ValidationError("可用时间不能早于实际观察时间或来源发布时间")
        object.__setattr__(self, "observed_at", observed.isoformat())
        object.__setattr__(self, "available_at", available.isoformat())
        if self.published_at:
            object.__setattr__(self, "published_at", published.isoformat())
        self._validate_payload()
        if not self.id:
            # Identical re-fetches preserve the first observed version, not a new fact.
            identity = {k: v for k, v in asdict(self).items() if k not in {
                "id", "observed_at", "available_at"
            }}
            object.__setattr__(self, "id", digest(identity))

    def _validate_payload(self) -> None:
        if self.kind == "bar":
            values = {key: decimal(self.payload[key]) for key in ("open", "high", "low", "close")}
            if any(value <= 0 for value in values.values()):
                raise ValidationError("行情价格必须大于零")
            if values["high"] < max(values["open"], values["close"], values["low"]) or values["low"] > min(values["open"], values["close"]):
                raise ValidationError("行情高低价关系错误")
            if self.payload.get("adjustment", "raw") != "raw":
                raise ValidationError("执行行情只接受不复权价格")
            if "volume" in self.payload and decimal(self.payload["volume"]) < 0:
                raise ValidationError("成交量不得为负")
        elif self.kind == "nav":
            if decimal(self.payload["unit_nav"]) <= 0:
                raise ValidationError("单位净值必须大于零")
        elif self.kind == "calendar":
            if not isinstance(self.payload.get("is_open"), bool):
                raise ValidationError("日历必须明确 is_open 布尔值")
        elif self.kind == "dividend":
            for key in ("record_date", "ex_date", "pay_date"):
                date.fromisoformat(self.payload[key])
            if self.payload["record_date"] > self.payload["ex_date"] or self.payload["ex_date"] > self.payload["pay_date"]:
                raise ValidationError("分红日期顺序错误")
            if decimal(self.payload["cash_per_unit"]) < 0:
                raise ValidationError("分红不得为负")
        elif self.kind == "split":
            if decimal(self.payload["ratio"]) <= 0:
                raise ValidationError("拆分比例必须为正")

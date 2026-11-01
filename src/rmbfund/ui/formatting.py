"""Neutral presentation helpers; no model, service, or personal-state dependencies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from html import escape
from typing import Any
from zoneinfo import ZoneInfo


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(str(item) for item in value if isinstance(item, str) and item))


def markdown_text(value: Any) -> str:
    """Source text is never interpreted as HTML, images, or Markdown links."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|~\-])", r"\\\1", escape(str(value), quote=False))


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def percent(value: Any, *, signed: bool = False, digits: int = 1) -> str:
    result = number(value)
    if result is None:
        return "未提供"
    return f"{result * 100:+.{digits}f}%" if signed else f"{result * 100:.{digits}f}%"


def when(value: Any) -> str:
    if not value:
        return "未提供"
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is not None:
            return parsed.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        pass
    return str(value)

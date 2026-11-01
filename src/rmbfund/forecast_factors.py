"""Offline CBOE close-date features, never a historical availability assertion.

The two-calendar-day lag is a declared exploration assumption. Neither that lag
nor a recent download reconstructs historical publication times or vintages.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .domain import ValidationError
from .forecast import _digest, _normalise_series

VIX_TRANSFORM = {
    "series_id": "VIX",
    "source_id": "cboe_vix_historical",
    "field": "close",
    "unit": "index_points",
    "return_basis": "index_level_close",
    "features": ["log_vix_level", "vix_log_change_5_observations"],
    "alignment": "latest source close date <= NAV valuation date minus 2 calendar days",
    "lag_calendar_days": 2,
    "maximum_age_calendar_days": 7,
    "missing": "NaN feature; candidate falls back explicitly; never zero-fill",
    "availability_basis": "valuation_date_exploration",
    "point_in_time_verified": False,
    "historical_publication_times": "not supplied; date lag is an assumption, not a timestamp",
    "vintage_policy": "downloaded historical close levels; past revisions not reconstructed",
    "economic_role": "global risk proxy, not a direct emerging-market exposure or return overlay",
}


def _validated_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows or len(rows) > 50000:
        raise ValidationError("VIX历史输入须为非空、有界JSON列表")
    result, previous = [], ""
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValidationError("VIX观察必须为对象")
        try:
            day = date.fromisoformat(item["date"]).isoformat()
            value = float(item["value"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("VIX日期或数值无效") from exc
        if (
            day != item["date"] or day <= previous
            or isinstance(item["value"], bool) or not math.isfinite(value) or value <= 0
            or item.get("series_id") != "VIX"
            or item.get("source_id") != "cboe_vix_historical"
            or item.get("field") != "close"
            or item.get("unit") != "index_points"
            or item.get("currency") is not None
            or item.get("return_basis") != "index_level_close"
            or item.get("observation_time_kind") != "daily_close_date_only"
        ):
            raise ValidationError("VIX来源身份、顺序或日收盘口径不匹配")
        # This input contract does not accept inferred historical availability.
        if item.get("published_at") is not None or item.get("available_at") is not None:
            raise ValidationError("日收盘日期输入不能混入未经验证的历史发布时间")
        result.append(dict(item))
        previous = day
    _digest(result)
    return result


def load_vix_factors(path: str | Path) -> dict[str, Any]:
    """Read the explicit normalized JSON only; no paths in it are followed."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValidationError("VIX冻结输入必须是16MiB以内的普通JSON文件")
    raw = path.read_bytes()
    try:
        rows = _validated_rows(json.loads(raw))
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("无法读取VIX冻结JSON输入") from exc
    definition = dict(VIX_TRANSFORM)
    return {
        "rows": rows,
        "input_hash": _digest(rows),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "definition": definition,
    }


def align_vix_features(
    series: Sequence[Mapping[str, Any]], factor_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """As-of align lagged closes without using any same-day overseas close."""
    navs = _normalise_series(series)
    rows = _validated_rows(list(factor_rows))
    days = [item["date"] for item in rows]
    values = np.asarray([float(item["value"]) for item in rows])
    features = np.full((len(navs), 2), np.nan)
    for index, item in enumerate(navs):
        origin = date.fromisoformat(item["date"])
        cutoff = (origin - timedelta(days=VIX_TRANSFORM["lag_calendar_days"])).isoformat()
        chosen = bisect.bisect_right(days, cutoff) - 1
        if chosen < 5:
            continue
        age = (origin - date.fromisoformat(days[chosen])).days
        if age > VIX_TRANSFORM["maximum_age_calendar_days"]:
            continue
        features[index] = (
            math.log(values[chosen]), math.log(values[chosen] / values[chosen - 5]),
        )
    return features

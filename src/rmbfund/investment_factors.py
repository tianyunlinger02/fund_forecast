"""Qualified public factor inputs; retrospective date alignment is NOT system PIT."""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .domain import ValidationError, timestamp
from .forecast import _canonical, _digest, _normalise_series

DATA_CONTRACT_VERSION = 1
# Identity and units are deliberately fixed, not inferred from a file's title.
SERIES_CONTRACT = {
    "market": ("nasdaq_eem_close_usd", "USD", "price",
               "publisher_close_adjustment_unverified_not_total_return"),
    "industry": ("nasdaq_soxx_close_usd", "USD", "price",
                 "publisher_close_adjustment_unverified_not_total_return"),
    "fx": ("sina_usd_cny_central_parity", "CNY", "CNY_per_USD",
           "central_parity_fixing_not_spot"),
    "yield_2y": ("eastmoney_us_treasury_2y_pct", "USD", "percent",
                 "government_bond_yield_percent_not_price"),
    "yield_10y": ("eastmoney_us_treasury_10y_pct", "USD", "percent",
                  "government_bond_yield_percent_not_price"),
}
ALIGNMENT = {
    "lag_calendar_days": 2, "maximum_age_calendar_days": 7,
    "rule": "latest source date <= NAV date minus 2 calendar days; age measured from NAV date",
    "availability_basis": "retrospective_date_lag_reconstruction_not_system_point_in_time",
    "historical_publication_times_verified": False, "historical_vintages_verified": False,
    "missing": "NaN; retain every NAV date; required current features must be finite",
    "actual_issuance": "selected observations must have actual available_at <= generated_at",
}


def _definition(name, label, unit, role, roles, description):
    return {"name": name, "label": label, "unit": unit, "economic_role": role,
            "series_ids": [SERIES_CONTRACT[key][0] for key in roles],
            "description": description}


FEATURE_DEFINITIONS = (
    *(_definition(f"eem_log_return_{n}", f"新兴市场股票近{n}笔涨跌", "log_return",
                  "market", ["market"], f"EEM raw USD close log change over {n} source observations; "
                  "price association, not total return or this fund's exact holdings") for n in (5, 20)),
    *(_definition(f"soxx_minus_eem_log_return_{n}", f"半导体相对新兴市场近{n}笔强弱",
                  "log_return_difference", "industry", ["industry", "market"],
                  f"SOXX minus EEM USD log change over {n} common dated source observations; "
                  "relative industry proxy, not a disclosed portfolio weight") for n in (5, 20)),
    _definition("usd_cny_fixing_log_change_20", "美元兑人民币中间价近20笔变化", "log_change",
                "fx", ["fx"], "20-source-observation log change in CNY per USD central parity "
                "fixing (publisher CNY per 100 USD divided by 100); NOT spot or an FX return overlay"),
    _definition("us10y_yield_percent", "美国十年期国债收益率", "percent", "discount_rate",
                ["yield_10y"], "Yield in percent units (4 means 4%), not a bond price or return"),
    _definition("us10y_yield_change_20_pp", "美国十年期收益率近20笔变化", "percentage_points",
                "discount_rate", ["yield_10y"], "Current minus 20-source-observation-prior yield, "
                "in percentage points; no duration or stock valuation identity assumed"),
)
FEATURE_DEFINITION_HASH = _digest(FEATURE_DEFINITIONS)


def normalize_factor_bundle(bundle: Mapping[str, Any], *, as_of: str | None = None) -> dict:
    """Detach/validate frozen JSON; optionally select actually acquired rows by cutoff.

    Actual capture clocks never become economic dates. No paths in the input are
    followed, and no implicit current clock or historical first-seen test is used.
    """
    point = timestamp(as_of) if as_of is not None else None
    if not isinstance(bundle, Mapping) or bundle.get("schema_version") != DATA_CONTRACT_VERSION:
        raise ValidationError("unsupported investment factor data contract")
    result = json.loads(_canonical(bundle))
    sources = result.get("series")
    if not isinstance(sources, dict) or not sources or sources.keys() - SERIES_CONTRACT.keys():
        raise ValidationError("investment factors require explicit qualified series roles")
    for role, source in sources.items():
        if not isinstance(source, dict) or tuple(source.get(key) for key in (
            "series_id", "currency", "unit", "return_basis"
        )) != SERIES_CONTRACT[role]:
            raise ValidationError(f"investment factor identity/unit mismatch: {role}")
        if any(not isinstance(source.get(key), str) or not source[key].strip()
               for key in ("label", "source", "source_url")):
            raise ValidationError(f"investment factor missing source identity: {role}")
        captured = timestamp(source.get("captured_at"))
        coverage = source.get("metadata", {})
        if not isinstance(coverage, Mapping):
            raise ValidationError(f"investment factor metadata must be an object: {role}")
        if coverage.get("complete_window") is True:
            try:
                start = date.fromisoformat(coverage["coverage_start"]).isoformat()
                end = date.fromisoformat(coverage["coverage_end"]).isoformat()
                if start > end:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(f"invalid complete source coverage window: {role}") from exc
        rows = source.get("rows")
        if not isinstance(rows, list) or len(rows) > 50000:
            raise ValidationError(f"investment factor rows must be bounded lists: {role}")
        selected, previous = [], ""
        for row in rows:
            try:
                day = date.fromisoformat(row["date"]).isoformat()
                value = float(row["value"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(f"invalid investment factor observation: {role}") from exc
            if (day != row["date"] or day <= previous or isinstance(row["value"], bool)
                    or not math.isfinite(value) or (role in ("market", "industry", "fx") and value <= 0)
                    or row.get("historical_available_at") is not None):
                raise ValidationError(f"invalid investment factor date/value/vintage: {role}")
            available = timestamp(row.get("available_at"))
            if available > captured or available.date().isoformat() < day:
                raise ValidationError(f"investment factor acquisition clock inconsistent: {role}")
            if row.get("observed_at") is not None and timestamp(row["observed_at"]) > available:
                raise ValidationError(f"investment factor observed_at exceeds available_at: {role}")
            row["value"] = value
            previous = day
            if point is None or available <= point:
                selected.append(row)
        source["rows"] = selected
    return result


def load_factor_bundle(path: str | Path) -> dict:
    """Read only the caller's bounded normalized JSON, never referenced files."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValidationError("investment factors require a regular JSON file <=16 MiB")
    try:
        return normalize_factor_bundle(json.loads(path.read_bytes()))
    except (ValueError, UnicodeError) as exc:
        raise ValidationError(f"invalid investment factor JSON: {exc}") from exc


def factor_input_hash(bundle: Mapping[str, Any]) -> str:
    """Hash semantic values/versions/clocks, not receipt or URL query refresh metadata.

    URL identity is scheme/host/path; the full request URL stays in the snapshot.
    Fixed role IDs and exact dated values—not query parameters—identify inputs.
    """
    normalized = normalize_factor_bundle(bundle)
    semantic = {"schema_version": normalized["schema_version"], "series": {}}
    for role, source in normalized["series"].items():
        try:
            url = urlsplit(source["source_url"])
        except ValueError as exc:
            raise ValidationError("invalid investment factor source URL") from exc
        semantic["series"][role] = {
            **{key: source[key] for key in (
                "series_id", "label", "source", "currency", "unit", "return_basis"
            )},
            "source_url_identity": [url.scheme.lower(), url.netloc.lower(), url.path],
            **{key: source[key] for key in ("version_id", "source_version") if key in source},
            "rows": [{key: row[key] for key in (
                "date", "value", "available_at", "historical_available_at", "observed_at",
                "version_id", "fact_id"
            ) if key in row} for row in source["rows"]],
        }
    return _digest(semantic)


def align_investment_features(
    series: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[dict, ...], dict]:
    """Return N x 7 external features, definitions, and compact alignment provenance."""
    navs, factors = _normalise_series(series), normalize_factor_bundle(bundle)
    sources = factors["series"]
    required = ("market", "industry", "fx", "yield_10y")
    if any(role not in sources or not sources[role]["rows"] for role in required):
        raise ValidationError("missing current required investment factor role")
    lookup = {role: {row["date"]: row for row in source["rows"]}
              for role, source in sources.items()}
    days = {role: list(rows) for role, rows in lookup.items()}
    common = sorted(lookup["market"].keys() & lookup["industry"].keys())
    matrix, current = np.full((len(navs), 7), np.nan), {}
    for i, nav in enumerate(navs):
        origin = date.fromisoformat(nav["date"])
        cutoff = (origin - timedelta(days=ALIGNMENT["lag_calendar_days"])).isoformat()
        for j, definition in enumerate(FEATURE_DEFINITIONS):
            roles = {0: ("market",), 1: ("market",), 2: ("industry", "market"),
                     3: ("industry", "market"), 4: ("fx",), 5: ("yield_10y",),
                     6: ("yield_10y",)}[j]
            dates = common if len(roles) == 2 else days[roles[0]]
            end = bisect.bisect_right(dates, cutoff) - 1
            lookback = (5, 20, 5, 20, 20, 0, 20)[j]
            if end < lookback or (origin - date.fromisoformat(dates[end])).days > 7:
                continue
            ends, starts, changes, acquired = {}, {}, [], []
            for role in roles:
                last, first = lookup[role][dates[end]], lookup[role][dates[end - lookback]]
                sid = sources[role]["series_id"]
                ends[sid], starts[sid] = last["date"], first["date"]
                window = dates[end - lookback:end + 1]
                acquired.extend(lookup[role][day]["available_at"] for day in window)
                changes.append(last["value"] - first["value"] if role == "yield_10y"
                               else math.log(last["value"]) - math.log(first["value"]))
            matrix[i, j] = (lookup["yield_10y"][dates[end]]["value"] if j == 5
                            else changes[0] - changes[1] if len(changes) == 2 else changes[0])
            if i == len(navs) - 1:
                current[definition["name"]] = {
                    "observation_ends": ends, "observation_starts": starts,
                    "availability_basis": ALIGNMENT["availability_basis"],
                    "actual_available_at_max": max(acquired, key=timestamp),
                }
    if not np.all(np.isfinite(matrix[-1])):
        raise ValidationError("missing/stale current required investment factor feature (no fallback)")
    manifest = [{**{key: source[key] for key in (
        "series_id", "label", "source", "source_url", "currency", "unit", "return_basis", "captured_at"
    )}, "role": role, "observation_count": len(source["rows"]),
        "first_date": source["rows"][0]["date"] if source["rows"] else None,
        "latest_date": source["rows"][-1]["date"] if source["rows"] else None,
        "actual_available_at_max": max((row["available_at"] for row in source["rows"]),
                                       key=timestamp, default=None)}
        for role, source in sources.items()]
    return matrix, tuple(json.loads(_canonical(FEATURE_DEFINITIONS))), {
        "data_contract_version": DATA_CONTRACT_VERSION, "alignment": dict(ALIGNMENT),
        "factor_input_hash": factor_input_hash(factors), "series_manifest": manifest,
        "current_features": current, "complete_factor_row_count": int(np.isfinite(matrix).all(axis=1).sum()),
        "missing_factor_row_count": int((~np.isfinite(matrix).all(axis=1)).sum()),
    }

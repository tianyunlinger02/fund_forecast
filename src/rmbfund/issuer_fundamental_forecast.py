"""Frozen18 offline qualification and research, never a runtime/catalog model.

``freeze_protocol18(nav, market, issuer, prices, frozen_at=..., as_of=...)``
seals the exact five-input experiment before ``qualify_then_run18`` fits anything.
Persist that protocol before calling the runner. Both functions are pure: no
network, paths, databases, clock, issuance, holdings, or automatic promotion.

Input contracts:
* nav: existing frozen forecast input object with ``series`` (all dates retained).
* market: existing investment_factors v1 bundle (the unchanged seven features).
* issuer: {"issuers": [{"issuer_id": ..., "quarter_actuals": [...],
  "basis_qualification": {"status": "qualified", "method": "native_currency"
  or "documented_per_release_translation", "reference": {"source_url": ...,
  "locator": ...}}}]}. Quarter observations use the existing issuer-analysis
  ``metric``/scaled ``value``/release/revision/publication/capture contract.
* prices: {"securities": [same-listing price-history objects]}. Each has security,
  currency, share_basis, price_basis, source_url, observed_at, available_at,
  captured_at, closes, and currency_qualification.status="verified" with reference.
  price_basis.kind is unadjusted or split_adjusted, with an explicit split_basis
  and a documentary reference. Per-close clocks override the series clocks.

Documentary qualifications are caller-reviewed assertions, not authentication.
Missing/unknown assertions cannot be inferred from prices or current quotes.
Historical reconstruction uses public release dates under an explicit lag, NOT
actual acquisition dates relabeled as historical availability or immutable PIT.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np

from . import forecast
from .domain import ValidationError, timestamp
from .forecast_research import _disjoint, _metrics, evaluate_forecasts
from .investment_factors import (
    ALIGNMENT, FEATURE_DEFINITIONS as MARKET_FEATURES, SERIES_CONTRACT,
    normalize_factor_bundle,
)
from .investment_forecast import (
    MODEL_HASH as FROZEN14_HASH, MODEL_ID as FROZEN14_ID,
)

MODEL_ID = "issuer_fundamental_frozen18_research_v1"
ISSUERS = (("issuer:tsmc", "tsmc", "IFRS"), ("issuer:nvidia", "nvidia", "GAAP"))
ISSUER_FEATURES = tuple(
    definition
    for issuer_id, prefix, _ in ISSUERS
    for definition in (
        {"name": f"{prefix}_seasonal_eps_baseline_error_div_pre_release_price",
         "unit": "earnings_per_price", "issuer_id": issuer_id,
         "definition": "(original quarterly diluted EPS - same-basis original EPS of fiscal "
                       "quarter one year earlier) / latest qualified close strictly before "
                       "the current public release date; carry only this latest released event; "
                       "not analyst surprise or management-guidance error"},
        {"name": f"{prefix}_ttm_eps_div_price", "unit": "earnings_per_price",
         "issuer_id": issuer_id,
         "definition": "sum four distinct contiguous same-basis original quarterly diluted "
                       "EPS ending at latest released fiscal quarter / qualified as-of close; "
                       "not total return, constant-FX earnings, or whole-fund earnings yield"},
    )
)
FEATURE_DEFINITIONS = tuple(json.loads(forecast._canonical(forecast.FEATURE_DEFINITIONS))) + (
    tuple(json.loads(forecast._canonical(MARKET_FEATURES))) + ISSUER_FEATURES
)
FEATURE_DEFINITION_HASH = forecast._digest(FEATURE_DEFINITIONS)
RESEARCH_RULES = {
    "horizons": [1, 5, 20], "primary_horizon": 20, "secondary_horizons": [1, 5],
    "market_alignment": dict(ALIGNMENT),
    "issuer_release_lag_calendar_days": 2,
    "issuer_release_rule": "date-only release <= NAV date minus 2 calendar days; timestamp "
                           "release <= end of that cutoff date in UTC; never same-day inferred",
    "security_price_lag_calendar_days": 2,
    "security_price_maximum_age_calendar_days": 7,
    "pre_release_price": "latest close date strictly before public release calendar date; "
                         "maximum 7 calendar days old at release; no same-day close",
    "quarter_selection": "original actual primary diluted GAAP(NVDA)/IFRS(TSMC) EPS only; "
                         "no comparative, guidance, revisions, YTD subtraction, or single-quarter x4",
    "training": "one expanding generic _fit_head per unique model/horizon/lag; only matured "
                "labels/errors; all ridge candidates use the SAME complete18 origin mask",
    "empirical_training": "all valid matured NAV targets as in unchanged _fit_head; not fitted18",
    "comparison": "intersection of genuinely fitted18/14/NAV7 coordinates and identical "
                  "origin/entry/target/horizon/lag; same empirical/no-change summaries",
    "missing": "NaN features; no NAV date removal/reindex, zero fill, weight or holding backfill",
    "metrics": ["up_brier", "multiclass_brier", "mae", "rmse", "up_calibration",
                "up_calibration_ece", "interval_coverage", "mean_interval_width", "nonoverlap"],
    "periods": {"development_2025": ["2025-01-01", "2025-12-31"],
                "inspected_2026": ["2026-01-01", "2026-12-31"]},
    "actual_availability": "every selected input acquired/available by explicit frozen as_of; "
                           "preserve public date precision, actual capture, availability and "
                           "historical_available_at separately",
    "strict_unseen": False, "system_point_in_time": False, "hyperparameter_search": False,
    "runtime_promotion": False, "profit_claim": False,
}
MODEL_DEFINITION = {
    "model_id": MODEL_ID, "features": list(FEATURE_DEFINITIONS),
    "feature_definition_hash": FEATURE_DEFINITION_HASH,
    "numerical_core_model_hash": forecast.MODEL_HASH,
    "target_definition": json.loads(forecast._canonical(forecast.RUNTIME_TARGET_DEFINITION)),
    "ridge_alpha": forecast.RIDGE_ALPHA,
    "minimum_training_samples": forecast.MIN_TRAINING_SAMPLES,
    "minimum_residual_samples": forecast.MIN_RESIDUAL_SAMPLES,
    "fallback": "none; empirical is comparator only, never relabeled fitted18",
    "research_rules": RESEARCH_RULES,
}
MODEL_HASH = forecast._digest(MODEL_DEFINITION)
_PROTOCOL_ID = "frozen18_common_origin_protocol_v1"
_BASIS_FIELDS = ("security_id", "currency", "unit", "accounting_basis", "accounting_basis_detail",
                 "eps_basis", "share_basis", "split_basis")


def _copy(value):
    return json.loads(forecast._canonical(value))


def _hashes(nav, market, issuer, prices):
    return {key: forecast._digest(value) for key, value in (
        ("nav", nav), ("market", market), ("issuer", issuer), ("security_prices", prices))}


def freeze_protocol18(frozen_nav, frozen_market, frozen_issuer, frozen_security_prices, *,
                      frozen_at: str, as_of: str, entry_lags=(0, 3)) -> dict:
    """Construct the pre-fit JSON seal; caller must persist it before running research."""
    if timestamp(frozen_at) < timestamp(as_of):
        raise ValidationError("protocol frozen_at must be at or after its input availability cutoff")
    lags = list(forecast._entry_lags(entry_lags))
    if not lags or 0 not in lags:
        raise ValidationError("frozen18 protocol requires anchor lag0 and explicit nonnegative lags")
    body = {"protocol_id": _PROTOCOL_ID, "frozen_at": frozen_at, "as_of": as_of,
            "model_id": MODEL_ID, "model_hash": MODEL_HASH,
            "feature_definition_hash": FEATURE_DEFINITION_HASH,
            "input_hashes": _hashes(frozen_nav, frozen_market, frozen_issuer,
                                   frozen_security_prices),
            "entry_lags": lags, "rules": _copy(RESEARCH_RULES),
            "unchanged_models": {"nav_v1": forecast.MODEL_HASH,
                                 "nav_v2": forecast.RUNTIME_MODEL_HASH,
                                 "frozen14": FROZEN14_HASH}}
    return {**body, "protocol_hash": forecast._digest(body)}


def _protocol(protocol, nav, market, issuer, prices):
    if not isinstance(protocol, Mapping):
        raise ValidationError("frozen18 requires a sealed protocol, never an implicit default")
    expected = freeze_protocol18(nav, market, issuer, prices,
                                frozen_at=protocol.get("frozen_at"), as_of=protocol.get("as_of"),
                                entry_lags=protocol.get("entry_lags", ()))
    if _copy(protocol) != expected:
        raise ValidationError("frozen18 protocol/input/definition seal mismatch; freeze before fitting")
    return expected


def _reference(value):
    return isinstance(value, Mapping) and all(isinstance(value.get(key), str) and value[key].strip()
                                             for key in ("source_url", "locator"))


def _day(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date must be canonical YYYY-MM-DD")
    return parsed


def _number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("finite numeric value required")
    return number


def _clock(row, point, *, parent=None):
    """Capture is not historical release time. All three real clocks remain intact."""
    parent = parent or {}
    available = row.get("available_at", parent.get("available_at"))
    observed = row.get("observed_at", parent.get("observed_at"))
    captured = row.get("captured_at", parent.get("captured_at", observed))
    if not available or not observed or not captured:
        return "missing_actual_capture_or_availability"
    # Review availability can legitimately follow the actual capture. Neither
    # becomes a historical publisher-availability claim.
    if timestamp(observed) > timestamp(available) or timestamp(captured) > timestamp(available):
        return "inconsistent_actual_capture_or_availability"
    if timestamp(available) > point or timestamp(captured) > point:
        return "actually_unavailable_at_frozen_cutoff"
    return None


def _qualification(reasons=(), **fields):
    reasons = list(dict.fromkeys(reasons))
    return {"status": "unavailable" if reasons else "qualified", "reasons": reasons, **fields}


def _market_matrix(navs, bundle, as_of):
    """Same frozen seven market definitions, with partial rows rather than fallback."""
    count = len(navs)
    matrix = np.full((count, 7), np.nan)
    qualifications = {}
    try:
        selected = normalize_factor_bundle(bundle, as_of=as_of)
    except (ValidationError, TypeError, ValueError) as exc:
        return matrix, {"bundle": _qualification([str(exc)])}, {}
    sources = selected["series"]
    required = ("market", "industry", "fx", "yield_10y")
    for role in required:
        source = sources.get(role, {})
        reasons = [] if source.get("rows") else ["missing_actually_available_required_market_series"]
        if source.get("captured_at") and timestamp(source["captured_at"]) > timestamp(as_of):
            reasons.append("market_source_captured_after_frozen_cutoff")
        qualifications[role] = _qualification(
            reasons,
            series_id=SERIES_CONTRACT[role][0], row_count=len(source.get("rows", [])),
            input_row_count=len(bundle.get("series", {}).get(role, {}).get("rows", [])),
            captured_at=source.get("captured_at"), return_basis=source.get("return_basis"),
            historical_available_at="preserved_per_row_unknown_not_inferred",
        )
    lookup = {role: {row["date"]: row for row in sources.get(role, {}).get("rows", [])}
              for role in required}
    days = {role: sorted(rows) for role, rows in lookup.items()}
    common = sorted(lookup["market"].keys() & lookup["industry"].keys())
    for i, nav in enumerate(navs):
        origin = _day(nav["date"])
        cutoff = (origin - timedelta(days=2)).isoformat()
        for j, lookback in enumerate((5, 20, 5, 20, 20, 0, 20)):
            roles = (("market",), ("market",), ("industry", "market"),
                     ("industry", "market"), ("fx",), ("yield_10y",), ("yield_10y",))[j]
            dates = common if len(roles) == 2 else days[roles[0]]
            end = bisect.bisect_right(dates, cutoff) - 1
            if end < lookback or (origin - _day(dates[end])).days > 7:
                continue
            changes = []
            for role in roles:
                last, first = lookup[role][dates[end]], lookup[role][dates[end - lookback]]
                changes.append(last["value"] - first["value"] if role == "yield_10y"
                               else math.log(last["value"]) - math.log(first["value"]))
            matrix[i, j] = (lookup["yield_10y"][dates[end]]["value"] if j == 5 else
                            changes[0] - changes[1] if len(changes) == 2 else changes[0])
    return matrix, qualifications, selected


def _price_series(payload, point):
    reasons = []
    if payload.get("price_basis", {}).get("kind") not in {"unadjusted", "split_adjusted"}:
        reasons.append("historical_price_adjustment_unknown_or_not_eps_compatible")
    if not payload.get("price_basis", {}).get("split_basis"):
        reasons.append("missing_explicit_price_split_basis")
    if not _reference(payload.get("price_basis", {}).get("reference")):
        reasons.append("missing_documented_adjustment_reference")
    currency = payload.get("currency_qualification", {})
    if currency.get("status") != "verified" or not _reference(currency.get("reference")):
        reasons.append("historical_price_currency_not_explicitly_qualified")
    if any(not payload.get(key) for key in ("security_id", "symbol", "exchange", "currency",
                                            "share_basis", "source_url")):
        reasons.append("missing_same_listing_price_identity")
    clock = _clock(payload, point)
    if clock:
        reasons.append(clock)
    rows, rejected, previous = [], [], ""
    for raw in payload.get("closes", []):
        try:
            day = _day(raw["date"]).isoformat()
            close = _number(raw["close"], positive=True)
            if day <= previous:
                raise ValueError("price dates must be unique and ascending")
            previous = day
            reason = _clock(raw, point, parent=payload)
            if day > point.date().isoformat():
                reason = "price_economic_date_after_frozen_cutoff"
            if reason:
                rejected.append({"date": day, "reason": reason})
                continue
            rows.append({**raw, "close": close,
                         **{key: raw.get(key, payload.get(key)) for key in
                            ("observed_at", "available_at", "captured_at", "historical_available_at")}})
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            rejected.append({"date": raw.get("date"), "reason": str(exc)})
            reasons.append("invalid_security_price_rows")
    if not rows:
        reasons.append("no_actually_available_security_prices")
    qualification = _qualification(reasons, symbol=payload.get("symbol"),
        security_id=payload.get("security_id"), input_rows=len(payload.get("closes", [])),
        actually_available_rows=len(rows), basis_qualified_rows=len(rows) if not reasons else 0,
        price_basis=_copy(payload.get("price_basis", {})),
        currency_qualification=_copy(currency), rejected_rows=rejected)
    return {"payload": payload, "rows": rows if not reasons else [],
            "dates": [row["date"] for row in rows] if not reasons else [],
            "qualification": qualification}


def _basis(row):
    return tuple(row["metric"].get(key) for key in _BASIS_FIELDS)


def _fiscal(row):
    period = row["metric"]["period"]
    return period["fiscal_year"] * 4 + period["fiscal_quarter"]


def _published(row):
    if row.get("publication_precision") == "date":
        return _day(row["published_at"]).isoformat()
    if row.get("publication_precision") == "timestamp":
        return datetime.fromisoformat(row["published_at"]).date().isoformat()
    raise ValueError("missing_explicit_public_release_precision")


def _quarter_rows(company, accounting, point):
    rows, rejected = [], []
    for raw in company.get("quarter_actuals", []):
        metric = raw.get("metric", {})
        # Other currencies/metrics are inventory, not substitutions for exact18.
        if metric.get("name") != "eps" or metric.get("currency") != "USD":
            continue
        reason = None
        try:
            period = metric["period"]
            start, end = _day(period["start"]), _day(period["end"])
            if (metric.get("category") != "actual" or metric.get("period_role") != "primary"
                    or period.get("basis") != "quarter" or metric.get("eps_basis") != "diluted"
                    or metric.get("accounting_basis") != accounting
                    or metric.get("unit") != "currency_per_share"
                    or raw.get("revision_kind") != "original" or raw.get("quarantined") is True):
                raise ValueError("not_original_primary_comparable_quarter_eps")
            if (isinstance(period["fiscal_year"], bool) or not isinstance(period["fiscal_year"], int)
                    or type(period["fiscal_quarter"]) is not int
                    or period["fiscal_quarter"] not in (1, 2, 3, 4)
                    or not 70 <= (end - start).days + 1 <= 110):
                raise ValueError("invalid_independent_fiscal_quarter")
            if any(not raw.get(key) for key in ("release_id", "revision_id", "source_url")):
                raise ValueError("missing_release_revision_source_identity")
            if any(not metric.get(key) for key in ("security_id", "share_basis", "split_basis")):
                raise ValueError("missing_explicit_eps_security_share_or_split_basis")
            published = _published(raw)
            if published < period["end"]:
                raise ValueError("release_precedes_financial_period_end")
            if raw["publication_precision"] == "timestamp":
                if timestamp(raw["published_at"]) > point:
                    reason = "release_after_frozen_cutoff"
            elif published > point.date().isoformat():
                reason = "release_after_frozen_cutoff"
            reason = reason or _clock(raw, point)
            value = _number(raw["value"])  # existing observation value is ALREADY scaled
            if reason:
                rejected.append({"release_id": raw.get("release_id"), "reason": reason})
            else:
                rows.append({**raw, "value": value, "public_date": published})
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            rejected.append({"release_id": raw.get("release_id"), "reason": str(exc)})
    # Multiple originals for one quarter cannot manufacture events or select a
    # convenient newer/current-vintage observation. Ambiguity is explicit.
    grouped = defaultdict(list)
    for row in rows:
        grouped[_fiscal(row)].append(row)
    accepted = []
    for group in grouped.values():
        unique = {forecast._digest(row): row for row in group}
        if len(unique) != 1:
            rejected.extend({"release_id": row["release_id"], "reason": "ambiguous_original_quarter"}
                            for row in unique.values())
        else:
            accepted.append(next(iter(unique.values())))
    return sorted(accepted, key=_fiscal), rejected


def _window(rows, latest):
    indexed = {_fiscal(row): row for row in rows}
    end = _fiscal(latest)
    if any(index not in indexed for index in range(end - 3, end + 1)):
        return []
    four = [indexed[index] for index in range(end - 3, end + 1)]
    if any(_basis(row) != _basis(latest) for row in four):
        return []
    periods = [row["metric"]["period"] for row in four]
    if (any(_day(b["start"]) != _day(a["end"]) + timedelta(days=1)
            for a, b in zip(periods, periods[1:]))
            or not 350 <= (_day(periods[-1]["end"]) - _day(periods[0]["start"])).days + 1 <= 380):
        return []
    return four


def _price_for(prices, quarter, cutoff, age_at):
    payload = prices["payload"]
    metric = quarter["metric"]
    if any(payload.get(key) != metric.get(key) for key in
           ("security_id", "currency", "share_basis")):
        return None, "eps_price_security_currency_or_share_basis_mismatch"
    if payload.get("price_basis", {}).get("split_basis") != metric.get("split_basis"):
        return None, "eps_price_split_basis_mismatch"
    index = bisect.bisect_right(prices["dates"], cutoff) - 1
    if index < 0:
        return None, "no_qualified_price_by_cutoff"
    price = prices["rows"][index]
    if (_day(age_at) - _day(price["date"])).days > 7:
        return None, "stale_qualified_price"
    return price, None


def _issuer_matrix(navs, bundle, prices_bundle, point):
    matrix = np.full((len(navs), 4), np.nan)
    qualifications, inventory, traces, price_qualifications = {}, {}, {}, {}
    companies = bundle.get("issuers", [])
    securities = prices_bundle.get("securities", [])
    for offset, (issuer_id, prefix, accounting) in enumerate(ISSUERS):
        matches = [row for row in companies if row.get("issuer_id") == issuer_id]
        company = matches[0] if len(matches) == 1 else {}
        rows, rejected = _quarter_rows(company, accounting, point)
        basis = company.get("basis_qualification", {})
        reasons = [] if len(matches) == 1 else ["missing_or_ambiguous_issuer"]
        if (basis.get("status") != "qualified"
                or basis.get("method") not in {"native_currency", "documented_per_release_translation"}
                or not _reference(basis.get("reference"))):
            reasons.append("historical_same_security_earnings_basis_unqualified")
        if not rows:
            reasons.append("no_actually_available_original_usd_diluted_quarters")
        security_ids = {row["metric"]["security_id"] for row in rows}
        candidates = [row for row in securities if row.get("security_id") in security_ids]
        if len(candidates) != 1:
            price = {"payload": {}, "rows": [], "dates": [],
                     "qualification": _qualification(["missing_or_ambiguous_same_listing_price_series"])}
        else:
            price = _price_series(candidates[0], point)
            expected_symbol = "TSM" if prefix == "tsmc" else "NVDA"
            expected_shares = "ADS" if prefix == "tsmc" else "ordinary"
            if (candidates[0].get("symbol") != expected_symbol
                    or candidates[0].get("share_basis") != expected_shares):
                price["qualification"] = _qualification(
                    [*price["qualification"]["reasons"], "wrong_issuer_listing_symbol_or_share_unit"],
                    symbol=candidates[0].get("symbol"), basis_qualified_rows=0)
                price["rows"], price["dates"] = [], []
        price_qualifications[issuer_id] = price["qualification"]
        if price["qualification"]["status"] != "qualified":
            reasons.append("historical_security_prices_unqualified")
        indexed = {_fiscal(row): row for row in rows}
        ttm_windows, seasonal_events, seasonal = [], [], {}
        for row in rows:
            four = _window(rows, row)
            if four:
                ttm_windows.append({"end": row["metric"]["period"]["end"],
                                    "eps_sum": sum(q["value"] for q in four),
                                    "release_ids": [q["release_id"] for q in four]})
            prior = indexed.get(_fiscal(row) - 4)
            if prior and _basis(prior) == _basis(row) and prior["public_date"] < row["public_date"]:
                seasonal_events.append(row["release_id"])
                denominator, reason = _price_for(price, row,
                    (_day(row["public_date"]) - timedelta(days=1)).isoformat(), row["public_date"])
                seasonal[row["release_id"]] = {
                    "prior_release_id": prior["release_id"],
                    "eps_baseline_error": row["value"] - prior["value"],
                    "pre_release_price": denominator, "reason": reason,
                    "value": ((row["value"] - prior["value"]) / denominator["close"]
                              if denominator else None),
                }
        qualifications[issuer_id] = _qualification(reasons, basis_qualification=_copy(basis),
            accepted_original_quarters=len(rows), rejected_quarters=rejected)
        inventory[issuer_id] = {
            "original_quarter_events": len({_fiscal(row) for row in rows}),
            "independent_release_event_count": len({row["release_id"] for row in rows}),
            "original_quarter_release_ids": [row["release_id"] for row in rows],
            "same_basis_ttm_window_count": len(ttm_windows), "same_basis_ttm_windows": ttm_windows,
            "same_basis_seasonal_eps_event_count": len(seasonal_events),
            "price_qualified_seasonal_event_count": sum(item["value"] is not None for item in seasonal.values()),
            "daily_carry_is_not_independent_event": True,
            "publication_precisions": sorted({row["publication_precision"] for row in rows}),
            "selected_quarters": _copy(rows), "seasonal_events": seasonal,
        }
        trace = []
        for i, nav in enumerate(navs):
            cutoff = (_day(nav["date"]) - timedelta(days=2)).isoformat()
            public_cutoff = datetime.combine(_day(cutoff), time.max, timezone.utc)
            released = [row for row in rows if row["public_date"] <= cutoff
                        and (row["publication_precision"] != "timestamp"
                             or timestamp(row["published_at"]) <= public_cutoff)]
            if not released or reasons:
                trace.append({"status": "unavailable", "reason": "issuer_or_price_qualification_failed"
                              if reasons else "no_original_quarter_released_by_origin_cutoff"})
                continue
            latest = max(released, key=_fiscal)
            four = _window(released, latest)
            shock = seasonal.get(latest["release_id"], {})
            if shock.get("value") is not None:
                matrix[i, offset * 2] = shock["value"]
            denominator, reason = _price_for(price, latest, cutoff, nav["date"])
            if four and denominator:
                matrix[i, offset * 2 + 1] = sum(row["value"] for row in four) / denominator["close"]
            trace.append({"release_id": latest["release_id"], "published_at": latest["published_at"],
                          "publication_precision": latest["publication_precision"],
                          "observed_at": latest["observed_at"], "available_at": latest["available_at"],
                          "historical_available_at": latest.get("historical_available_at"),
                          "origin_public_cutoff": cutoff,
                          "seasonal_reason": shock.get("reason", "missing_same_basis_prior_year_original"),
                          "ttm_release_ids": [row["release_id"] for row in four],
                          "ttm_reason": reason or (None if four else "missing_contiguous_same_basis_ttm"),
                          "asof_price": denominator})
        traces[prefix] = trace
    return matrix, qualifications, price_qualifications, inventory, traces


def _compare(days, log_nav, invalid, features, joint, protocol):
    matrices = {MODEL_ID: features, FROZEN14_ID: features[:, :14],
                forecast.RUNTIME_MODEL_ID: features[:, :7]}
    histories, latest_results, fitted_count = {}, [], 0
    # The finite18 mask is applied to every ridge candidate, including training.
    # No dates are removed, so H20 still means twenty original NAV observations.
    for model_id, raw in matrices.items():
        matrix = raw.copy()
        matrix[~joint] = np.nan
        definitions = FEATURE_DEFINITIONS[:matrix.shape[1]]
        histories[model_id] = {}
        for horizon in (1, 5, 20):
            for lag in protocol["entry_lags"]:
                head, history, state, _, _ = forecast._fit_head(
                    days, log_nav, matrix, invalid, horizon=horizon, entry_lag=lag,
                    feature_definitions=definitions, capture_baseline=True)
                fitted = [row for row in history
                          if row["prediction"]["method"] == "ridge_prequential_empirical"]
                fitted_count += len(fitted) if model_id == MODEL_ID else 0
                for row in fitted:
                    histories[model_id][row["origin_index"], horizon, lag] = row
                available = head is not None and state is not None
                latest_results.append({
                    "model_id": model_id, "horizon": horizon, "entry_lag": lag,
                    "status": "fitted" if available else "unavailable",
                    "reason": None if available else head.get("fallback_reason") if head
                              else "insufficient_matured_labels",
                    "fallback_used_as_model": False, "fitted_evaluation_rows": len(fitted),
                    "forecast": head if available else None, "model_state": state if available else None,
                })
    identities = set.intersection(*(set(history) for history in histories.values()))
    matched = []
    for key in sorted(identities):
        row = dict(histories[MODEL_ID][key])
        for model_id, name in ((FROZEN14_ID, "frozen14_prediction"),
                               (forecast.RUNTIME_MODEL_ID, "nav7_prediction")):
            other = histories[model_id][key]
            if any(row[field] != other[field] for field in
                   ("origin_index", "entry_index", "target_index", "actual_change", "horizon", "entry_lag")):
                raise ValidationError("common-origin comparison target mismatch")
            row[name] = other["prediction"]
        # _fit_head's legacy period label is not a claim of an untouched holdout.
        row["period"] = "development_2025" if row["origin_date"] < "2026-01-01" else "inspected_2026"
        matched.append(row)
    keys = ("prediction", "frozen14_prediction", "nav7_prediction",
            "historical_baseline", "no_change_baseline")
    comparison = []
    for horizon in (1, 5, 20):
        for lag in protocol["entry_lags"]:
            periods = {}
            for name, (start, end) in RESEARCH_RULES["periods"].items():
                subset = [row for row in matched if row["horizon"] == horizon
                          and row["entry_lag"] == lag and start <= row["origin_date"] <= end]
                blocks = _disjoint(subset)
                periods[name] = {"joint_fitted_origins": len(subset),
                    "rolling": {key: _metrics(subset, key) for key in keys},
                    "nonoverlap": {"count": len(blocks),
                                   "metrics": {key: _metrics(blocks, key) for key in keys}},
                    "status": "evaluated" if subset else "unavailable_no_common_fitted_origins"}
            comparison.append({"horizon": horizon, "entry_lag": lag,
                               "primary": horizon == 20, "periods": periods})
    diagnostic = evaluate_forecasts(matched)
    diagnostic["heldout_period"] = {**diagnostic["heldout_period"], "already_inspected": True}
    return {"latest_heads": latest_results, "evaluation_rows": matched,
            "common_origin_comparison": comparison, "validation": diagnostic,
            "fitted18_evaluation_row_count": fitted_count,
            "common_fitted_evaluation_row_count": len(matched)}


def qualify_then_run18(frozen_nav, frozen_market, frozen_issuer, frozen_security_prices,
                       frozen_protocol) -> dict[str, Any]:
    """Qualify every input, then fit exact18 only if real qualifications pass.

    An unavailable return is a usable qualification result, not an exception or a
    fake fitted head. Definition/input seal violations raise ValidationError.
    Explicit NaNs become JSON null only in returned feature rows, never zeros.
    """
    protocol = _protocol(frozen_protocol, frozen_nav, frozen_market, frozen_issuer,
                         frozen_security_prices)
    point = timestamp(protocol["as_of"])
    qualifications = {"nav": _qualification(), "market": {}, "issuer": {}, "security_prices": {}}
    try:
        navs = forecast._normalise_series(frozen_nav.get("series", []))
    except (ValidationError, AttributeError) as exc:
        return {"model_id": MODEL_ID, "model_hash": MODEL_HASH,
                "protocol": protocol, "status": "qualification_unavailable", "fitted": False,
                "input_qualifications": {**qualifications, "nav": _qualification([str(exc)])},
                "independent_event_counts": {}, "joint_origins": {"count": 0, "origins": []},
                "fit_results": None, "issued": False, "runtime_promoted": False}
    nav_reasons = []
    for row in navs:
        provenance = row.get("provenance") or {}
        reason = _clock(row, point, parent=provenance)
        if reason:
            nav_reasons.append(reason)
        if row["date"] > point.date().isoformat():
            nav_reasons.append("nav_economic_date_after_frozen_cutoff")
    qualifications["nav"] = _qualification(nav_reasons, observation_count=len(navs),
        first_date=navs[0]["date"], last_date=navs[-1]["date"], index_preserved=True)
    external, qualifications["market"], selected_market = _market_matrix(
        navs, frozen_market, protocol["as_of"])
    issuer, qualifications["issuer"], qualifications["security_prices"], inventory, traces = (
        _issuer_matrix(navs, frozen_issuer, frozen_security_prices, point))
    days = [row["date"] for row in navs]
    log_nav = np.log(np.asarray([float(row["nav"]) for row in navs]))
    invalid = np.asarray([row.get("invalid_return", False) or row.get("label_valid") is False
                          for row in navs], dtype=bool)
    matrix = np.column_stack((forecast._feature_matrix(log_nav, invalid), external, issuer))
    joint = np.isfinite(matrix).all(axis=1)
    origins = [{"origin_index": i, "origin_date": days[i]} for i in np.flatnonzero(joint).tolist()]
    input_pass = (qualifications["nav"]["status"] == "qualified"
                  and all(item["status"] == "qualified" for section in
                          ("market", "issuer", "security_prices")
                          for item in qualifications[section].values()))
    blockers = [{"input": name, "component": key, "reasons": value["reasons"]}
                for name in ("market", "issuer", "security_prices")
                for key, value in qualifications[name].items() if value["status"] != "qualified"]
    if nav_reasons:
        blockers.append({"input": "nav", "reasons": qualifications["nav"]["reasons"]})
    if not origins:
        blockers.append({"input": "joint_matrix", "reasons": ["no_complete18_origin"]})
    result = {
        "schema_version": 1, "model_id": MODEL_ID, "model_hash": MODEL_HASH,
        "feature_definition_hash": FEATURE_DEFINITION_HASH,
        "feature_definitions": _copy(FEATURE_DEFINITIONS), "feature_count": 18,
        "protocol": protocol, "input_qualifications": qualifications,
        "independent_event_counts": inventory,
        "independent_earnings_event_count": sum(row["original_quarter_events"] for row in inventory.values()),
        "joint_origins": {"count": len(origins), "origins": origins,
                          "nav_observation_count": len(navs),
                          "missing18_rows": int((~joint).sum()),
                          "not_independent_earnings_events": True},
        "feature_rows": [{"origin_index": i, "date": day,
                          "features": {item["name"]: float(matrix[i, j]) if np.isfinite(matrix[i, j]) else None
                                       for j, item in enumerate(FEATURE_DEFINITIONS)},
                          "issuer_alignment": {prefix: trace[i] for prefix, trace in traces.items()}}
                         for i, day in enumerate(days)],
        "status": "qualification_unavailable", "fitted": False, "fit_results": None,
        "blocking_gaps": blockers, "issued": False, "runtime_promoted": False,
        "strict_unseen": False, "system_point_in_time": False, "profit_claim": False,
        "selected_market_input_hash": forecast._digest(selected_market),
        "unchanged_models": protocol["unchanged_models"],
        "limitations": ["2026 already inspected; no strict unseen or system-PIT claim.",
                        "Daily carried financial values do not add independent quarterly events.",
                        "Same observed NAV index/horizons; no calendar or holdings backfill.",
                        "No automatic runtime model promotion, catalog registration or issuance.",
                        "Unknown price adjustment/earnings translation cannot be solved by values."]}
    if input_pass and origins:
        fits = _compare(days, log_nav, invalid, matrix, joint, protocol)
        fitted = bool(fits["fitted18_evaluation_row_count"] or any(
            item["model_id"] == MODEL_ID and item["status"] == "fitted" for item in fits["latest_heads"]))
        result.update(fit_results=fits, fitted=fitted,
                      status="fitted_research_only" if fitted else "qualified_fit_unavailable")
    # Fail early on non-JSON numbers; this is not a NaN-to-zero cleanup pass.
    forecast._canonical(result)
    return result

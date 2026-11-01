"""Pure presentation projections of an already-decided buyer assessment.

No store, clock, network, model import, fitting, action selection, or ranking lives
here. The brief owns the final action; the calendar plan owns the exact H/L. A
chart retains actual NAV observations and a separate simple-return endpoint, not
a fabricated future NAV path.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any


_BUYER_FIELDS = ("preferred_action", "label", "advice_reason", "next_step", "execution_status")
_IDENTITY_FIELDS = (
    "record_id", "model_id", "model_hash", "input_hash", "basis_hash", "request_hash",
    "issue_id", "generated_at", "target_definition", "horizon", "entry_lag", "role",
)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _day(value: Any) -> str | None:
    if type(value) is date:
        return value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value).isoformat()
    except ValueError:
        return None
    return parsed if parsed == value else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) else None


def _integer(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _anchor(assessment: Mapping) -> Mapping:
    scenario = _mapping(assessment.get("trade_scenario"))
    for value in (scenario.get("anchor"), assessment.get("anchor"),
                  _mapping(assessment.get("anchor")).get("basis"),
                  _mapping(assessment.get("forecast")).get("anchor")):
        if _day(_mapping(value).get("date")):
            return value
    return {}


def _data_date(assessment: Mapping, case: Mapping | None = None) -> Any:
    brief = _mapping(assessment.get("decision_brief"))
    for value in (_mapping(case).get("data_date"), brief.get("data_date")):
        if value is not None:
            return deepcopy(value)
    scenario = _mapping(assessment.get("trade_scenario"))
    for value in (_anchor(assessment).get("date"), _mapping(scenario.get("plan")).get("anchor_date"),
                  _mapping(scenario.get("purchase")).get("latest_date")):
        if _day(value):
            return _day(value)
    return None


def build_buyer_view(assessment: Mapping, *, branch: str = "flat") -> dict:
    """Copy only the chosen brief case, never fall back to a quantitative action.

    ``flat`` and ``held`` are explicit user contexts, not inferred from example
    balances. Missing copy stays missing; neither labels nor reasons are invented
    from research branches, action rows, or another brief case. New brief schemas
    may add internal evidence without changing this whitelist.
    """
    if branch not in {"flat", "held"}:
        raise ValueError("branch must be flat or held")
    assessment = _mapping(assessment)
    brief = _mapping(assessment.get("decision_brief"))
    case = _mapping(_mapping(brief.get("cases")).get(branch))
    result = {key: deepcopy(case.get(key)) for key in _BUYER_FIELDS}
    result["data_date"] = _data_date(assessment, case)
    # A deliberately unresolved action (None) is still the authoritative result.
    missing = [key for key in _BUYER_FIELDS if key not in case]
    missing += [key for key in ("label", "advice_reason", "next_step") if key in case and not case[key]]
    if result["data_date"] is None:
        missing.append("data_date")
    return {
        "schema_version": 1, "kind": "buyer_view", "branch": branch,
        "status": "missing" if not case else "partial" if missing else "ready",
        **result, "missing_fields": missing,
    }


def _history_rows(series: Iterable[Mapping], anchor: Mapping) -> dict:
    """Keep only real date/value/unit fields from the two existing NAV contracts."""
    points, seen = [], set()
    try:
        for raw in series:
            if not isinstance(raw, Mapping):
                raise ValueError
            day = _day(raw.get("date"))
            candidates = [raw[key] for key in ("nav", "value") if raw.get(key) is not None]
            if not day or day in seen or not candidates or any(isinstance(v, bool) for v in candidates):
                raise ValueError
            values = [Decimal(str(value)) for value in candidates]
            if any(not value.is_finite() or value <= 0 for value in values):
                raise ValueError
            if len(set(values)) != 1 or not isfinite(float(values[0])) or float(values[0]) <= 0:
                raise ValueError
            provenance = _mapping(raw.get("provenance"))
            currencies = {value for value in (raw.get("currency"), provenance.get("currency")) if value}
            if len(currencies) > 1:
                raise ValueError
            currency = next(iter(currencies), None) or anchor.get("currency")
            points.append({
                "date": day, "value": str(values[0]), "currency": currency,
                "basis": deepcopy(raw.get("basis") or provenance.get("nav_type") or "unit_nav"),
            })
            seen.add(day)
    except (TypeError, ValueError, InvalidOperation, OverflowError):
        return {"status": "invalid", "points": [], "currency": None,
                "unit": "unit_nav", "date_basis": "nav_date", "missing_fields": ["history.series"]}
    points.sort(key=lambda row: row["date"])
    currencies = {row["currency"] for row in points}
    if len(currencies) > 1:
        return {"status": "invalid", "points": [], "currency": None,
                "unit": "unit_nav", "date_basis": "nav_date", "missing_fields": ["history.currency"]}
    return {
        "status": "ready" if points else "missing", "points": points,
        "currency": next(iter(currencies), None), "unit": "unit_nav", "date_basis": "nav_date",
        "missing_fields": [] if points else ["history.series"],
    }


def build_forecast_chart(assessment: Mapping, *, series: Iterable[Mapping]) -> dict:
    """Project actual NAV history and one exact calendar-plan return distribution.

    ``series`` is the caller-supplied ``scenario_series``/``series`` (date + nav)
    or fund-detail history (date + value); never fetch or use an anchor as fake
    history. ``expected_change``, ``lower`` and ``upper`` are the saved simple
    fractions (.01 = 1%), with lower/upper the model contract's p10/p90. There is
    no interpolation, sample re-quantiling, fee arithmetic, or NAV multiplication.
    """
    assessment = _mapping(assessment)
    scenario = _mapping(assessment.get("trade_scenario"))
    plan = _mapping(scenario.get("plan"))
    decision = _mapping(scenario.get("decision"))
    head = _mapping(decision.get("primary_head"))
    horizon, lag = plan.get("horizon"), plan.get("entry_lag")
    history = _history_rows(series, _anchor(assessment))
    missing, point = [], None
    forecast_status = "ready"
    if not _integer(horizon, 1):
        missing.append("plan.horizon")
        forecast_status = "infeasible_window"
    if not _integer(lag, 0):
        missing.append("plan.entry_lag")
        forecast_status = "missing_plan"
    for key in ("application_date", "sale_date"):
        if not _day(plan.get(key)):
            missing.append(f"plan.{key}")
            forecast_status = "missing_plan"
    if (_day(plan.get("application_date")) and _day(plan.get("sale_date"))
            and plan["sale_date"] <= plan["application_date"]):
        missing.append("plan.sale_date")
        forecast_status = "infeasible_window"
    exact = (head and _integer(head.get("horizon"), 1) and _integer(head.get("entry_lag"), 0)
             and head.get("horizon") == horizon and head.get("entry_lag") == lag
             and head.get("status", "available") == "available")
    if not exact:
        missing.append("decision.primary_head")
        if not missing[:-1]:
            forecast_status = "missing_exact_head"
    if not missing:
        expected = _number(head.get("expected_change"))
        if expected is None or expected < -1:
            missing.append("primary_head.expected_change")
            forecast_status = "invalid_head"
        else:
            lower, upper = _number(head.get("lower")), _number(head.get("upper"))
            bounds_valid = (lower is not None and upper is not None and -1 <= lower <= upper
                            and ("interval_coverage" not in head
                                 or _number(head["interval_coverage"]) == 0.8))
            if not bounds_valid:
                missing.extend(["primary_head.lower", "primary_head.upper"])
                forecast_status = "partial"
            point = {
                "date": plan["sale_date"], "expected_change": expected,
                "lower": lower if bounds_valid else None, "upper": upper if bounds_valid else None,
            }
    forecast = {
        "status": forecast_status, "point": point, "horizon": horizon, "entry_lag": lag,
        "application_date": _day(plan.get("application_date")), "sale_date": _day(plan.get("sale_date")),
        "deadline": _day(plan.get("deadline")), "unit": "simple_return",
        "date_basis": "conditional_plan_sale_date", "interval_quantiles": [0.1, 0.9],
        "missing_fields": missing,
    }
    saved_forecast = _mapping(assessment.get("forecast"))
    model_identity = _mapping(decision.get("model_identity"))
    identity = {key: deepcopy(model_identity[key] if key in model_identity else saved_forecast.get(key))
                for key in _IDENTITY_FIELDS}
    identity["anchor"] = deepcopy(dict(_anchor(assessment)))
    product = _mapping(assessment.get("subject")) or _mapping(scenario.get("subject"))
    return {
        "schema_version": 1, "kind": "buyer_forecast_chart",
        "status": "ready" if history["status"] == forecast_status == "ready" else
                  "partial" if history["points"] or point else "unavailable",
        "product": {key: deepcopy(product.get(key)) for key in ("name", "symbol", "share_class")},
        "data_date": _data_date(assessment) or (history["points"][-1]["date"] if history["points"] else None),
        "history": history, "forecast": forecast,
        "identity": identity,
    }

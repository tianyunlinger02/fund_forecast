"""Pure retail-fund comparisons and conditional switching-cost arithmetic.

Inputs are JSON-ready mappings, not storage rows or personal positions. A fund is
``{product, series?, fees?, exposure?, corporate_action_coverage?, status?, reason?}``.
``product`` contains the public Product fields plus explicit ``currency``,
``underlying_fund_id``, ``evidence_refs`` and, for same-mandate peers, ``mandate_id``.
A usable ``series`` has status, product_id, share_class, currency, evidence_refs,
basis (unit_nav or unadjusted_price), and observations [{date, value, status?}].
Absent observation status means an observed value; explicit missing/blocked rows
are retained but never used. Section statuses must be explicit when supplied.

Fees are dated disclosures: {status, as_of, evidence_refs, items: [{name, rate,
basis, condition?}]}. They are not automatically summed or selected from tiers.
Exposure is a dated disclosure: {status, as_of, denominator, scope, evidence_refs,
holdings: [{security_id, weight}], ...}. Weights are nonnegative fractions of the
stated denominator, never rescaled, filled, capped, or matched by similar names.

Every pair's metrics use that pair's exact common dated observations. The
whole-set intersection is also reported without ranking across different pair
windows. There is no filling, FX conversion, annualization, alpha estimation, or
calculation of total return from unit NAV. All derived numbers are Decimal text.
Only the existing domain validation helpers are imported; there is no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
import re
from typing import Any

from .domain import Product, ValidationError, decimal, timestamp

SCHEMA_VERSION = 1
RELATIONS = frozenset({"same_mandate", "style_or_cost_alternative", "reference_only"})
_STATUSES = frozenset({"available", "partial", "missing", "blocked"})
_BASES = frozenset({"unit_nav", "unadjusted_price"})
_FEE_BASES = frozenset({"annual_nav", "gross_redemption", "gross_subscription", "net_subscription"})
__all__ = ["PeerRelation", "PeerSet", "compare_funds", "compare_peer_set", "switch_scenario"]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{field} must be explicit nonblank text")
    return value


def _day(value: Any, field: str) -> str:
    _text(value, field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValidationError(f"{field} must use YYYY-MM-DD")
    return value


def _refs(value: Any, field: str = "evidence_refs") -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValidationError(f"{field} requires explicit evidence references")
    result = tuple(_text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} contains duplicate references")
    return result


def _mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be a mapping")
    return value


def _rows(value: Any, field: str) -> Sequence:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field} must be an array")
    return value


def _number(value: Decimal) -> str:
    if not value:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _nonnegative(value: Any, field: str) -> Decimal:
    parsed = decimal(value)
    if parsed < 0:
        raise ValidationError(f"{field} must be nonnegative")
    return parsed


def _rate(value: Any) -> Decimal:
    parsed = _nonnegative(value, "fee rate")
    if parsed > 1:
        raise ValidationError("fee rates must be fractions between 0 and 1")
    return parsed


def _status(value: Mapping, field: str) -> str:
    status = value.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValidationError(f"{field}.status must be available, partial, missing or blocked")
    if status in {"missing", "blocked"} and "reason" in value and value.get("reason") is not None:
        _text(value.get("reason"), f"{field}.reason")
    return status


def _section(value: Any, field: str) -> dict:
    if value is None:
        return {"status": "missing", "reason": "not_supplied"}
    result = deepcopy(dict(_mapping(value, field)))
    _status(result, field)
    return result


@dataclass(frozen=True)
class PeerRelation:
    """An evidenced selection relative to one subject, not an inferred peer label."""

    peer_id: str
    relation: str
    selection_reason: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.peer_id, "peer_id")
        if not isinstance(self.relation, str) or self.relation not in RELATIONS:
            raise ValidationError("unknown peer relation")
        _text(self.selection_reason, "selection_reason")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))

    @classmethod
    def from_mapping(cls, value: Mapping) -> PeerRelation:
        value = dict(_mapping(value, "peer relation"))
        if "product_id" in value:
            if "peer_id" in value:
                raise ValidationError("peer relation must use one product identity field")
            value["peer_id"] = value.pop("product_id")
        if set(value) != {"peer_id", "relation", "selection_reason", "evidence_refs"}:
            raise ValidationError("peer relation requires product_id, relation, selection_reason, evidence_refs")
        return cls(**value)

    def to_dict(self) -> dict:
        return {"product_id": self.peer_id, "relation": self.relation,
                "selection_reason": self.selection_reason, "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class PeerSet:
    """An evidenced cohort, optionally limited to a requested inclusive interval."""

    subject_id: str
    peers: tuple[PeerRelation, ...]
    start_date: str | None
    end_date: str | None
    selection_reason: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.subject_id, "subject_id")
        if (self.start_date is None) != (self.end_date is None):
            raise ValidationError("peer interval requires both start_date and end_date")
        if self.start_date is not None:
            _day(self.start_date, "start_date")
            _day(self.end_date, "end_date")
        if self.start_date is not None and self.end_date < self.start_date:
            raise ValidationError("peer interval is reversed")
        _text(self.selection_reason, "selection_reason")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))
        peers = tuple(item if isinstance(item, PeerRelation) else PeerRelation.from_mapping(item)
                      for item in _rows(self.peers, "peers"))
        if not peers:
            raise ValidationError("a peer set requires at least one peer")
        ids = [self.subject_id, *(item.peer_id for item in peers)]
        if len(ids) != len(set(ids)):
            raise ValidationError("peer ids must be distinct from each other and the subject")
        object.__setattr__(self, "peers", peers)

    @classmethod
    def from_mapping(cls, value: Mapping) -> PeerSet:
        value = dict(_mapping(value, "peer set"))
        if "target_product_id" in value:
            if "subject_id" in value:
                raise ValidationError("peer set must use one target identity field")
            value["subject_id"] = value.pop("target_product_id")
        value.setdefault("start_date", None)
        value.setdefault("end_date", None)
        if set(value) != {"subject_id", "peers", "start_date", "end_date",
                          "selection_reason", "evidence_refs"}:
            raise ValidationError("peer set requires explicit identity, peers, reason and evidence")
        return cls(**value)

    def to_dict(self) -> dict:
        return {"target_product_id": self.subject_id, "peers": [item.to_dict() for item in self.peers],
                "start_date": self.start_date, "end_date": self.end_date,
                "selection_reason": self.selection_reason, "evidence_refs": list(self.evidence_refs)}


def _product(value: Mapping) -> dict:
    raw = deepcopy(dict(_mapping(value, "product")))
    base = {"id", "symbol", "name", "kind", "is_demo", "exchange", "category", "share_class"}
    if set(raw) - base - {"currency", "underlying_fund_id", "mandate_id", "evidence_refs"}:
        raise ValidationError("product accepts only public product identity fields")
    for key in ("id", "symbol", "name", "kind"):
        _text(raw.get(key), f"product.{key}")
    if raw["kind"] not in {"open_fund", "etf"}:
        raise ValidationError("comparison requires a public retail fund or ETF, not monitor_only")
    raw.setdefault("share_class", None)
    raw.setdefault("underlying_fund_id", None)
    raw.setdefault("mandate_id", None)
    raw.setdefault("currency", None)
    raw.setdefault("evidence_refs", [])
    Product(**{key: item for key, item in raw.items() if key in base})
    for key in ("share_class", "underlying_fund_id", "mandate_id"):
        if raw.get(key) not in (None, ""):
            _text(raw[key], f"product.{key}")
    if raw.get("currency") not in (None, "") and (
            not isinstance(raw["currency"], str) or not re.fullmatch(r"[A-Z]{3}", raw["currency"])):
        raise ValidationError("product.currency requires an explicit three-letter uppercase currency")
    if "is_demo" in raw and not isinstance(raw["is_demo"], bool):
        raise ValidationError("product.is_demo must be boolean")
    if raw.get("evidence_refs"):
        _refs(raw["evidence_refs"], "product.evidence_refs")
    return raw


def _series(value: Any, product: Mapping) -> tuple[dict, dict[str, Decimal], list[str]]:
    series = _section(value, "series")
    if series["status"] in {"missing", "blocked"}:
        return series, {}, []
    _refs(series.get("evidence_refs"), "series.evidence_refs")
    if not isinstance(series.get("basis"), str) or series["basis"] not in _BASES:
        raise ValidationError("series.basis must be unit_nav or unadjusted_price")
    issues = [f"series_{key}_mismatch_or_missing" for key in ("product_id", "share_class", "currency")
              if product.get("id" if key == "product_id" else key)
              and series.get(key) != product.get("id" if key == "product_id" else key)]
    values, seen = {}, set()
    for row in _rows(series.get("observations"), "series.observations"):
        row = _mapping(row, "observation")
        day = _day(row.get("date"), "observation.date")
        if day in seen:
            raise ValidationError("duplicate observation dates require explicit upstream resolution")
        seen.add(day)
        status = _status(row, "observation") if "status" in row else "available"
        if status == "partial":
            raise ValidationError("an observation must be available, missing or blocked")
        if "evidence_refs" in row:
            _refs(row["evidence_refs"], "observation.evidence_refs")
        for key in ("observed_at", "available_at", "published_at"):
            if row.get(key) is not None:
                timestamp(row[key])
        if status != "available":
            continue
        for key in ("product_id", "share_class", "currency", "basis"):
            expected = series.get(key)
            if key in row and row[key] != expected:
                issues.append(f"observation_{key}_mismatch:{day}")
        number = decimal(row.get("value"))
        if number <= 0:
            raise ValidationError("raw NAV/price observations must be positive")
        values[day] = number
    return series, values, issues


def _fees(value: Any) -> dict:
    fees = _section(value, "fees")
    if fees["status"] in {"available", "partial"}:
        _day(fees.get("as_of"), "fees.as_of")
        _refs(fees.get("evidence_refs"), "fees.evidence_refs")
        for item in _rows(fees.get("items"), "fees.items"):
            item = _mapping(item, "fee item")
            _text(item.get("name"), "fee name")
            if item.get("basis") not in _FEE_BASES:
                raise ValidationError("fee basis must explicitly describe its denominator")
            _rate(item.get("rate"))
    return fees


def _exposure(value: Any) -> dict:
    exposure = _section(value, "exposure")
    if exposure["status"] not in {"available", "partial"}:
        return exposure
    _day(exposure.get("as_of"), "exposure.as_of")
    _refs(exposure.get("evidence_refs"), "exposure.evidence_refs")
    if exposure.get("denominator") not in {"nav", "equity_assets", "total_assets"}:
        raise ValidationError("exposure requires its disclosed denominator")
    if exposure.get("scope") not in {"full", "partial", "top10"}:
        raise ValidationError("exposure.scope must be full, partial or top10")
    ids = set()
    for row in _rows(exposure.get("holdings"), "exposure.holdings"):
        row = _mapping(row, "holding")
        security_id = _text(row.get("security_id"), "holding.security_id")
        if security_id in ids:
            raise ValidationError("duplicate security ids require upstream resolution, not normalization")
        ids.add(security_id)
        _nonnegative(row.get("weight"), "holding.weight")
    return exposure


def _coverage(value: Any) -> dict:
    value = {} if value is None else _mapping(value, "corporate_action_coverage")
    result = {}
    for kind in ("dividends", "splits"):
        section = _section(value.get(kind), f"{kind} coverage")
        if section["status"] in {"available", "partial"}:
            _day(section.get("start_date"), "coverage.start_date")
            _day(section.get("end_date"), "coverage.end_date")
            if section["end_date"] < section["start_date"]:
                raise ValidationError("corporate action coverage interval is reversed")
            _refs(section.get("evidence_refs"), "coverage.evidence_refs")
        result[kind] = section
    return result


def _missing_fund(product_id: str) -> dict:
    """A selected-but-absent product stays visible instead of aborting the cohort."""
    return {"product": {"id": product_id}, "status": "missing", "reason": "fund_not_supplied",
            "series": {"status": "missing", "reason": "fund_not_supplied"}, "values": {},
            "identity_issues": [], "missing_fields": ["product", "series"],
            "fees": {"status": "missing", "reason": "not_supplied"},
            "exposure": {"status": "missing", "reason": "not_supplied"},
            "corporate_action_coverage": _coverage(None)}


def _fund(value: Mapping, *, expected_id: str) -> dict:
    value = _mapping(value, "fund")
    if value.get("product") is None:
        result = _missing_fund(expected_id)
        result["status"] = _status(value, "fund") if "status" in value else "missing"
        result["reason"] = value.get("reason", result["reason"])
        result["series"] = _section(value.get("series"), "series")
        result["fees"] = _fees(value.get("fees"))
        result["exposure"] = _exposure(value.get("exposure"))
        result["corporate_action_coverage"] = _coverage(value.get("corporate_action_coverage"))
        result["missing_fields"] = ["product"]
        return result
    product = _product(value["product"])
    if product["id"] != expected_id:
        raise ValidationError("fund map key does not match public product id")
    missing = [f"product.{key}" for key in ("share_class", "currency", "underlying_fund_id", "evidence_refs")
               if not product.get(key)]
    status = _status(value, "fund") if "status" in value else "available"
    if status in {"missing", "blocked"}:
        series, observations, issues = _section(value.get("series"), "series"), {}, []
    else:
        series, observations, issues = _series(value.get("series"), product)
    return {"product": product, "status": status, "reason": value.get("reason"),
            "series": series, "values": observations, "identity_issues": issues,
            "missing_fields": missing, "fees": _fees(value.get("fees")),
            "exposure": _exposure(value.get("exposure")),
            "corporate_action_coverage": _coverage(value.get("corporate_action_coverage"))}


def _compatibility(subject: dict, peer: dict, relation: PeerRelation) -> list[str]:
    if any(fund["status"] in {"missing", "blocked"} for fund in (subject, peer)):
        return []
    a, b = subject["product"], peer["product"]
    issues = []
    for key in ("kind", "currency"):
        if a[key] != b[key]:
            issues.append(f"incompatible_{key}")
    if a.get("is_demo", False) != b.get("is_demo", False):
        issues.append("demo_and_public_products_mixed")
    same_underlying = a["underlying_fund_id"] == b["underlying_fund_id"]
    if same_underlying and relation.relation != "style_or_cost_alternative":
        issues.append("same_underlying_requires_style_or_cost_alternative_not_independent_peer")
    if relation.relation == "same_mandate":
        if a["share_class"] != b["share_class"]:
            issues.append("incompatible_share_class_for_same_mandate")
        if not a.get("mandate_id") or not b.get("mandate_id"):
            issues.append("missing_explicit_mandate_identity")
        elif a["mandate_id"] != b["mandate_id"]:
            issues.append("unlike_mandates_require_alternative_or_reference_relation")
    elif (a["underlying_fund_id"], a["share_class"]) == (b["underlying_fund_id"], b["share_class"]):
        issues.append("duplicate_underlying_share_class_is_not_a_distinct_alternative")
    if (subject["series"]["status"] in {"available", "partial"}
            and peer["series"]["status"] in {"available", "partial"}
            and subject["series"]["basis"] != peer["series"]["basis"]):
        issues.append("incompatible_observation_basis")
    return issues


def _metrics(fund: dict, dates: list[str], blocked: bool) -> dict:
    count = len(dates)
    reason = "incompatible_or_blocked_peer_set" if blocked else "fewer_than_two_common_observations"
    status = "blocked" if blocked else "missing"
    base = {"basis": fund["series"].get("basis"), "observation_count": count,
            "return_interval_count": max(0, count - 1), "annualized": False}
    if blocked or count < 2:
        return {**base, "status": status, "reason": reason,
                "raw_return": {"status": status, "value": None, "reason": reason},
                "volatility": {"status": status, "value": None, "reason": reason},
                "max_drawdown": {"status": status, "value": None, "reason": reason}}
    values = [fund["values"][day] for day in dates]
    with localcontext() as context:
        context.prec = 40
        returns = [b / a - 1 for a, b in zip(values, values[1:])]
        peak, drawdown = values[0], Decimal(0)
        for value in values:
            peak = max(peak, value)
            drawdown = max(drawdown, 1 - value / peak)
        volatility = {"status": "missing", "value": None,
                      "reason": "sample_volatility_requires_two_return_intervals"}
        if len(returns) >= 2:
            mean = sum(returns, Decimal(0)) / len(returns)
            variance = sum(((value - mean) ** 2 for value in returns), Decimal(0)) / (len(returns) - 1)
            volatility = {"status": "available", "value": _number(variance.sqrt()), "reason": None}
        return {**base, "status": "available" if len(returns) >= 2 else "partial", "reason": None,
                "raw_return": {"status": "available", "value": _number(values[-1] / values[0] - 1),
                               "unit": "fraction", "label": "raw_NAV_or_price_change_not_total_return"},
                "volatility": {**volatility, "unit": "fraction",
                               "method": "sample_std_of_common_observation_simple_returns"},
                "max_drawdown": {"status": "available", "value": _number(drawdown),
                                 "unit": "fraction", "convention": "positive_peak_to_trough_loss"}}


def _total_return(fund: dict, dates: list[str]) -> dict:
    coverage = fund["corporate_action_coverage"]
    covered = bool(dates) and all(section["status"] == "available"
                                 and section["start_date"] <= dates[0]
                                 and section["end_date"] >= dates[-1]
                                 for section in coverage.values())
    return {"status": "total_return_unavailable", "value": None,
            "reason": "raw_series_not_adjusted_for_total_return" if covered else
                      "dividend_or_split_coverage_absent_incomplete_or_blocked",
            "coverage": deepcopy(coverage),
            "caveat": "Unit NAV/price changes exclude distribution reinvestment and split adjustment; "
                      "corporate actions can distort raw return, volatility and drawdown."}


def _overlap(a: dict, b: dict, blocked: bool) -> dict:
    base = {"value": None, "normalization_applied": False,
            "method": "sum_of_minimum_weights_for_exact_common_security_ids"}
    if blocked:
        return {**base, "status": "blocked", "reason": "incompatible_or_blocked_peer_set"}
    unavailable = [exposure for exposure in (a, b) if exposure["status"] in {"missing", "blocked"}]
    if unavailable:
        return {**base, "status": "blocked" if any(item["status"] == "blocked" for item in unavailable) else "missing",
                "reason": "exposure_unavailable", "source_statuses": [
                    {"status": item["status"], "reason": item.get("reason")} for item in (a, b)]}
    for key in ("as_of", "denominator"):
        if a[key] != b[key]:
            return {**base, "status": "blocked", "reason": f"incompatible_exposure_{key}"}
    aa = {row["security_id"]: decimal(row["weight"]) for row in a["holdings"]}
    bb = {row["security_id"]: decimal(row["weight"]) for row in b["holdings"]}
    with localcontext() as context:
        context.prec = 40
        common = [{"security_id": key, "weight": _number(min(aa[key], bb[key]))}
                  for key in sorted(aa.keys() & bb.keys())]
        value = sum((decimal(row["weight"]) for row in common), Decimal(0))
        lower_bound = any(item["scope"] != "full" or item["status"] == "partial" for item in (a, b))
        return {**base, "status": "partial" if lower_bound else "available", "reason": None,
                "value": _number(value), "as_of": a["as_of"], "denominator": a["denominator"],
                "is_disclosed_only_lower_bound": lower_bound, "common_holdings": common,
                "subject_disclosed_weight": _number(sum(aa.values(), Decimal(0))),
                "peer_disclosed_weight": _number(sum(bb.values(), Decimal(0))),
                "evidence_refs": [deepcopy(a["evidence_refs"]), deepcopy(b["evidence_refs"])],
                "caveat": "Undisclosed positions are unknown, not zero; this is a dated disclosure, "
                          "not an estimate of today's full portfolio overlap."}


def _window(dates: list[str], peers: PeerSet) -> dict:
    return {"requested_start": peers.start_date, "requested_end": peers.end_date,
            "start": dates[0] if dates else None, "end": dates[-1] if dates else None,
            "dates": dates, "observation_count": len(dates),
            "return_interval_count": max(0, len(dates) - 1),
            "calendar_day_gaps": [(date.fromisoformat(b) - date.fromisoformat(a)).days
                                  for a, b in zip(dates, dates[1:])]}


def _dates(fund: dict, peers: PeerSet, point) -> set[str]:
    if (fund["status"] in {"missing", "blocked"} or fund["series"]["status"] in {"missing", "blocked"}
            or fund["missing_fields"] or fund["identity_issues"]):
        return set()
    for field in ("available_at", "observed_at", "published_at"):
        if fund["series"].get(field) is not None:
            available = timestamp(fund["series"][field])
            if point is not None and available > point:
                return set()
    dates = {day for day in fund["values"]
             if (peers.start_date is None or peers.start_date <= day <= peers.end_date)
             and (point is None or day <= point.date().isoformat())}
    for row in fund["series"].get("observations", []):
        for field in ("available_at", "observed_at", "published_at"):
            if row.get(field) is not None:
                available = timestamp(row[field])
                if point is not None and available > point:
                    dates.discard(row["date"])
    return dates


def _dated_disclosure(section: dict, point) -> dict:
    if section["status"] not in {"available", "partial"}:
        return section
    available = [timestamp(section[field]) for field in ("available_at", "observed_at", "published_at")
                 if section.get(field) is not None]
    if point is not None and (section["as_of"] > point.date().isoformat()
                              or any(value > point for value in available)):
        return {"status": "missing", "reason": "disclosure_after_as_of", "source": section}
    return section


def compare_peer_set(peer_set: PeerSet | Mapping, funds: Mapping[str, Mapping], *,
                     as_of: str | None = None) -> dict:
    """Return independent descriptive pairs, preserving every selected member.

    Each pair has its own exact common-dated window. A missing or blocked peer is
    visible but cannot suppress other pairs or be silently removed from the
    whole-set common-window count. With no requested interval, use all explicitly
    dated observations through as_of. No app-specific status/identity inference
    or evidence normalization occurs here; callers supply the documented contract.
    """
    peers = peer_set if isinstance(peer_set, PeerSet) else PeerSet.from_mapping(peer_set)
    funds = _mapping(funds, "funds")
    point = timestamp(as_of) if as_of is not None else None
    ids = [peers.subject_id, *(item.peer_id for item in peers.peers)]
    selected = {key: _fund(funds[key], expected_id=key) if funds.get(key) is not None else _missing_fund(key)
                for key in ids}
    dates = {key: _dates(fund, peers, point) for key, fund in selected.items()}
    for fund in selected.values():
        fund["fees"] = _dated_disclosure(fund["fees"], point)
        fund["exposure"] = _dated_disclosure(fund["exposure"], point)
    subject = selected[peers.subject_id]
    comparisons = []
    for relation in peers.peers:
        peer = selected[relation.peer_id]
        missing = [{"product_id": fund["product"]["id"], "field": field}
                   for fund in (subject, peer) for field in fund["missing_fields"]]
        issues = [{"product_id": fund["product"]["id"], "reason": reason}
                  for fund in (subject, peer) for reason in fund["identity_issues"]]
        for fund in (subject, peer):
            if fund["status"] == "blocked" or fund["series"]["status"] == "blocked":
                issues.append({"product_id": fund["product"]["id"], "reason": "source_data_blocked"})
        structural = [] if missing else _compatibility(subject, peer, relation)
        issues.extend({"product_id": relation.peer_id, "reason": reason} for reason in structural)
        common = sorted(dates[peers.subject_id] & dates[relation.peer_id]) if not issues and not missing else []
        status = "blocked" if issues else "missing" if len(common) < 2 else "available"
        if status == "available" and (len(common) < 3 or any(
                fund["status"] == "partial" or fund["series"]["status"] == "partial" for fund in (subject, peer))):
            status = "partial"
        comparisons.append({**relation.to_dict(), "target_product_id": peers.subject_id, "status": status,
                            "compatibility": {"status": "blocked" if issues else "missing" if missing else "available",
                                              "issues": issues, "missing_fields": missing},
                            "common_window": _window(common, peers),
                            "observation_counts_in_requested_interval": {
                                peers.subject_id: len(dates[peers.subject_id]), relation.peer_id: len(dates[relation.peer_id])},
                            "subject_metrics": _metrics(subject, common, bool(issues)),
                            "peer_metrics": _metrics(peer, common, bool(issues)),
                            "subject_total_return": _total_return(subject, common),
                            "peer_total_return": _total_return(peer, common),
                            "disclosed_overlap": _overlap(subject["exposure"], peer["exposure"],
                                                           bool(structural) or bool(missing) or any(
                                                               fund["status"] == "blocked" for fund in (subject, peer)))})
    states = {item["status"] for item in comparisons}
    status = "available" if states == {"available"} else "blocked" if states == {"blocked"} else (
        "missing" if states == {"missing"} else "partial")
    common = (sorted(set.intersection(*(dates[key] for key in ids)))
              if all(item["compatibility"]["status"] == "available" for item in comparisons) else [])
    return {"schema_version": SCHEMA_VERSION, "status": status, "as_of": point.isoformat() if point else None,
            "peer_set": peers.to_dict(), "common_window": _window(common, peers),
            "funds": [{"product": fund["product"], "source_status": fund["status"], "source_reason": fund["reason"],
                       "missing_fields": fund["missing_fields"], "series": fund["series"],
                       "fees": fund["fees"], "exposure": fund["exposure"]} for fund in selected.values()],
            "comparisons": comparisons,
            "ranking": {"status": "not_produced", "reason": "different_pair_windows_and_unlike_funds_are_not_ranked"},
            "manager_alpha": {"status": "not_assessed", "value": None,
                              "reason": "requires_compatible_benchmark_and_manager_tenure_data"},
            "caveats": ["Metrics are pair-specific; use each pair's common_window, never rank across different windows.",
                        "Returns, volatility and drawdown are raw NAV/price-only and unannualized.",
                        "Drawdown is measured on common dates only and can omit intervening extremes.",
                        "Missing dates are not filled; disclosed holdings and fee rates are never normalized.",
                        "as_of checks supplied dates/availability; missing publication timestamps are not invented."]}


def compare_funds(peer_set: PeerSet | Mapping, funds: Mapping[str, Mapping], *,
                  as_of: str | None = None) -> dict:
    """Public snapshot entry point; an alias with the same explicit peer-set contract."""
    return compare_peer_set(peer_set, funds, as_of=as_of)


def _switch_fee(value: Any, field: str, bases: set[str], *, currency: str) -> dict:
    fee = _section(value, field)
    if fee["status"] not in {"available", "partial"}:
        return fee
    _refs(fee.get("evidence_refs"), f"{field}.evidence_refs")
    if fee.get("currency", currency) != currency:
        raise ValidationError("switch fee currency is incompatible")
    if fee.get("basis") not in bases:
        raise ValidationError(f"{field} has an incompatible fee denominator")
    if ("rate" in fee) == ("amount" in fee):
        raise ValidationError(f"{field} requires exactly one explicit rate or amount")
    if "rate" in fee:
        _rate(fee["rate"])
    else:
        _nonnegative(fee["amount"], f"{field}.amount")
    return fee


def _switch_timing(value: Any) -> dict:
    timing = _section(value, "timing")
    fields = ("redemption_date", "cash_available_date", "subscription_date", "target_confirmation_date",
              "source_exposure_end_date", "target_exposure_start_date")
    unknowns = [key for key in fields if timing.get(key) is None]
    if timing["status"] in {"missing", "blocked"}:
        return {"status": timing["status"], "reason": timing.get("reason"), "source": timing,
                "unknowns": unknowns, "calendar_cash_wait_days": None,
                "calendar_redeployment_wait_days": None, "out_of_market_calendar_days": None}
    _refs(timing.get("evidence_refs"), "timing.evidence_refs")
    for key in fields:
        if timing.get(key) is not None:
            _day(timing[key], f"timing.{key}")
    ordering = (("redemption_date", "cash_available_date"),
                ("cash_available_date", "subscription_date"),
                ("subscription_date", "target_confirmation_date"),
                ("redemption_date", "source_exposure_end_date"),
                ("source_exposure_end_date", "cash_available_date"),
                ("subscription_date", "target_exposure_start_date"),
                ("target_exposure_start_date", "target_confirmation_date"),
                ("source_exposure_end_date", "target_exposure_start_date"),
                ("redemption_date", "subscription_date"),
                ("redemption_date", "target_confirmation_date"),
                ("cash_available_date", "target_confirmation_date"))
    problems = [f"{b}_before_{a}" for a, b in ordering
                if timing.get(a) is not None and timing.get(b) is not None and timing[b] < timing[a]]
    before_cutoff = timing.get("cash_before_subscription_cutoff")
    if before_cutoff is not None and not isinstance(before_cutoff, bool):
        raise ValidationError("cash_before_subscription_cutoff must be boolean or unknown")
    if timing.get("cash_available_date") is not None and timing.get("subscription_date") == timing["cash_available_date"]:
        if before_cutoff is False:
            problems.append("cash_not_available_before_subscription_cutoff")
        elif before_cutoff is None:
            unknowns.append("cash_before_subscription_cutoff")
    def gap(a: str, b: str) -> int | None:
        return ((date.fromisoformat(timing[b]) - date.fromisoformat(timing[a])).days
                if timing.get(a) is not None and timing.get(b) is not None and not problems else None)
    return {"status": "blocked" if problems else "partial" if unknowns or timing["status"] == "partial" else "available",
            "reason": "infeasible_cash_or_exposure_order" if problems else None,
            "source": timing, "issues": problems, "unknowns": unknowns,
            "calendar_cash_wait_days": gap("redemption_date", "cash_available_date"),
            "calendar_redeployment_wait_days": gap("cash_available_date", "subscription_date"),
            "out_of_market_calendar_days": gap("source_exposure_end_date", "target_exposure_start_date"),
            "caveat": "Dates are supplied assumptions, not trading-calendar or cutoff verification. "
                      "Confirmation dates alone do not establish market exposure dates."}


def switch_scenario(source: Mapping, target: Mapping, scenario: Mapping) -> dict:
    """Redeem -> wait for cash -> subscribe, using supplied gross proceeds only.

    ``scenario`` requires selection_reason, evidence_refs, currency and
    gross_redemption_amount. Fees are status sections with rate OR fixed amount:
    redemption_fee uses gross_redemption; subscription_fee uses gross_subscription
    (fee = cash * rate) or net_subscription (net = cash / (1 + rate)); other_fees
    uses fixed_cash with an amount. Unknown fees are never treated as zero.
    ``timing`` contains explicit dates described by _switch_timing; nothing is
    inferred from settlement-day counts. All results are conditional arithmetic,
    not orders, realized gains, forecasts, tax estimates, or target-unit estimates.
    """
    source = _mapping(source, "source")
    target = _mapping(target, "target")
    a, b = _product(source.get("product")), _product(target.get("product"))
    scenario = _mapping(scenario, "scenario")
    allowed = {"selection_reason", "evidence_refs", "currency", "gross_redemption_amount",
               "redemption_fee", "subscription_fee", "other_fees", "timing"}
    if set(scenario) - allowed:
        raise ValidationError("switch scenario accepts costs and timing only, never forecasts or gain assumptions")
    _text(scenario.get("selection_reason"), "selection_reason")
    _refs(scenario.get("evidence_refs"))
    currency = scenario.get("currency")
    if not currency or currency != a["currency"] or currency != b["currency"]:
        raise ValidationError("switch scenario requires explicit compatible currencies; no implicit FX conversion")
    if a["id"] == b["id"]:
        raise ValidationError("switch source and target must be distinct public share classes")
    gross = _nonnegative(scenario.get("gross_redemption_amount"), "gross_redemption_amount")
    if not gross:
        raise ValidationError("gross_redemption_amount must be positive")
    redeem = _switch_fee(scenario.get("redemption_fee"), "redemption_fee", {"gross_redemption"}, currency=currency)
    subscribe = _switch_fee(scenario.get("subscription_fee"), "subscription_fee",
                            {"gross_subscription", "net_subscription"}, currency=currency)
    other = _switch_fee(scenario.get("other_fees"), "other_fees", {"fixed_cash"}, currency=currency)
    if "rate" in other:
        raise ValidationError("other_fees requires a fixed amount, including explicit zero if none")
    timing = _switch_timing(scenario.get("timing"))
    fees = {"redemption_fee": redeem, "subscription_fee": subscribe, "other_fees": other}
    unknowns = [key for key, fee in fees.items() if fee["status"] != "available"]
    source_statuses = [(_status(value, name) if "status" in value else "available")
                       for name, value in (("source", source), ("target", target))]
    unknowns.extend(name for name, status in zip(("source_data", "target_data"), source_statuses)
                    if status != "available")
    missing_identity = [f"{name}.product.{key}" for name, product in (("source", a), ("target", b))
                        for key in ("share_class", "currency", "underlying_fund_id", "evidence_refs")
                        if not product.get(key)]
    unknowns.extend(missing_identity)
    blocked = any(fee["status"] == "blocked" for fee in fees.values()) or "blocked" in source_statuses
    def fee_amount(fee: dict, budget: Decimal | None) -> Decimal | None:
        if fee["status"] != "available" or budget is None:
            return None
        if "amount" in fee:
            return decimal(fee["amount"])
        rate = decimal(fee["rate"])
        return budget - budget / (1 + rate) if fee["basis"] == "net_subscription" else budget * rate
    with localcontext() as context:
        context.prec = 40
        redemption_cost = fee_amount(redeem, gross)
        net_cash = gross - redemption_cost if redemption_cost is not None else None
        other_cost = decimal(other["amount"]) if other["status"] == "available" else None
        budget = net_cash - other_cost if net_cash is not None and other_cost is not None else None
        issues = []
        if net_cash is not None and net_cash < 0:
            issues.append("redemption_fee_exceeds_gross_proceeds")
        if budget is not None and budget < 0:
            issues.append("other_fees_exceed_net_cash")
        subscription_cost = fee_amount(subscribe, budget) if not issues else None
        invested = budget - subscription_cost if budget is not None and subscription_cost is not None else None
        if invested is not None and invested < 0:
            issues.append("subscription_fee_exceeds_subscription_budget")
            invested = None
        if missing_identity or any(status in {"missing", "blocked"} for status in source_statuses):
            invested = None
        total_cost = (redemption_cost + other_cost + subscription_cost
                      if not issues and all(value is not None for value in
                                            (redemption_cost, other_cost, subscription_cost)) else None)
        blocked = blocked or bool(issues)
        arithmetic = {"status": "blocked" if blocked else "partial" if unknowns else "available",
                      "issues": issues, "unknowns": unknowns,
                      "gross_redemption_amount": _number(gross),
                      "redemption_fee_amount": _number(redemption_cost) if redemption_cost is not None else None,
                      "cash_after_redemption_fee": _number(net_cash) if net_cash is not None else None,
                      "other_fee_amount": _number(other_cost) if other_cost is not None else None,
                      "subscription_budget": _number(budget) if budget is not None and not issues else None,
                      "subscription_fee_amount": _number(subscription_cost) if subscription_cost is not None else None,
                      "target_net_subscription_amount": _number(invested) if invested is not None and not blocked else None,
                      "total_explicit_cost": _number(total_cost) if total_cost is not None else None}
    status = "blocked" if blocked or timing["status"] == "blocked" else (
        "partial" if unknowns or timing["status"] != "available" else "available")
    return {"schema_version": SCHEMA_VERSION, "status": status, "kind": "conditional_switch_cost_scenario",
            "source_product": a, "target_product": b, "currency": currency,
            "selection_reason": scenario["selection_reason"], "evidence_refs": deepcopy(scenario["evidence_refs"]),
            "fees": fees, "timing": timing, "arithmetic": arithmetic,
            "executable_amount": {"status": "not_assessed", "value": None,
                                  "reason": "scenario_is_not_an_order_or_verified_settlement_quote"},
            "realizable_gain": {"status": "not_estimated", "value": None,
                                "reason": "switching_costs_are_not_forecast_differences_or_future_profit"},
            "caveats": ["No future NAV, units, market return, tax, FX or opportunity-cost assumption is inserted.",
                        "Ongoing fees already embedded in NAV are disclosed separately, not deducted again.",
                        "Unknown fees/timing remain unknown; cash cannot be reused before availability.",
                        "Arithmetic is unrounded; supplied amounts/rates are not verified platform quotes."]}

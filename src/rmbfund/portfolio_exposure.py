"""Pure, dated exposure calculations for a recorded fund cohort, not total wealth.

The caller supplies valuations and public profiles. Nothing is loaded, fetched or
matched by a security's name/ticker here, and disclosed baskets are never scaled
up to 100%. Monetary results cross the boundary as decimal text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from itertools import combinations
from typing import Any

from .domain import ValidationError, decimal, timestamp

SCHEMA_VERSION = 1
__all__ = ["build_portfolio_context", "pairwise_issuer_overlap", "issuer_overlap"]
_DATES = ("report_date", "period_start", "published_at", "publication_precision",
          "observed_at", "available_at", "effective_from", "effective_to")
_ASSUMPTIONS = [
    "分母只是已记录且可估值的人民币基金市值，不是个人总资产；未知金额不按零计。",
    "人民币估值沿用传入的market_value；不换汇、不用成本替代市值、不推算未知份额。",
    "历史NAV口径披露权重保持原值，不归一化，也不代表今天或实际成交时的持仓。",
    "发行人只按明确提供的corporate_equity issuer_id相等归并；不从名称、代码或上市地猜测。",
    "行业采用来源的原分类；发行人与行业是不同切面，不能相加；列表外不叫现金。",
]


def _amount(value: Any) -> Decimal | None:
    # A derived shares * NAV value can exceed domain.decimal's input bound.
    if value is None or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError("基金市值必须是有效十进制数") from exc
    if isinstance(value, bool) or not result.is_finite() or result < 0:
        raise ValidationError("基金市值必须是有限非负数")
    return result


def _text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _optional_text(value: Any, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValidationError(f"{label}必须是文本或未知")
    return value


def _date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise ValidationError("估值和披露日期必须是YYYY-MM-DD") from exc
    return value


def _rows(value: Any, label: str) -> Sequence[Mapping]:
    if value is None:
        return []
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or any(not isinstance(row, Mapping) for row in value)):
        raise ValidationError(f"{label}必须是对象列表")
    return value


def _snapshot(profile: Mapping | None, *, as_of: str | None = None) -> dict:
    """Accept an investment_profile or its portfolio section, without evidence I/O."""
    if profile is not None and not isinstance(profile, Mapping):
        raise ValidationError("投资画像必须是对象或未知")
    section = profile.get("portfolio") if profile and "portfolio" in profile else profile
    if section is not None and not isinstance(section, Mapping):
        raise ValidationError("组合披露必须是对象或未知")
    section = section or {}
    payload = section.get("payload") or {}
    if not isinstance(payload, Mapping):
        raise ValidationError("组合披露payload必须是对象")
    report_date = _date(section.get("report_date"))
    result = {
        "status": "available",
        "record_id": _optional_text(section.get("record_id", section.get("id")), "record_id"),
        **{key: _optional_text(section.get(key), key) for key in _DATES},
        "report_date": report_date,
        "denominator": _optional_text(payload.get("denominator"), "denominator"),
        "holdings_scope": _optional_text(payload.get("holdings_scope"), "holdings_scope"),
        "issuers": {}, "sectors": {}, "known_listed_weight": None,
        "issuer_coverage": None, "sector_coverage": None, "exclusions": [], "unknowns": [],
    }
    reasons = result["unknowns"]
    if not section:
        reasons.append("missing_portfolio_disclosure")
    if report_date is None:
        reasons.append("missing_report_date")
    if payload.get("denominator") != "nav":
        reasons.append("missing_denominator" if not payload.get("denominator")
                       else "non_nav_denominator")
    day = as_of[:10] if as_of else None
    if day and report_date and report_date > day:
        reasons.append("report_after_as_of")
    if as_of:
        for key in ("published_at", "observed_at", "available_at"):
            value = section.get(key)
            if not value:
                continue
            # Public reports may disclose only a day, not a precise instant.
            later = _date(value) > day if key == "published_at" and len(value) == 10 \
                else timestamp(value) > timestamp(as_of)
            if later:
                reasons.append(f"{key}_after_as_of")
    if reasons:
        result["status"] = "unavailable"
        return result

    for source, key, target in (("holdings", "issuer_id", "issuers"),
                                ("sectors", "name", "sectors")):
        rows = _rows(payload.get(source), source)
        if payload.get(source) is None:
            reasons.append(f"missing_{source}")
            continue
        listed = Decimal(0)
        for index, row in enumerate(rows):
            raw_weight = row.get("weight")
            if raw_weight is None or raw_weight == "":
                result["exclusions"].append({"basket": source, "index": index,
                                              "reason": "missing_weight"})
                continue
            weight = decimal(raw_weight)
            if not 0 <= weight <= 1:
                raise ValidationError("披露权重必须在0和1之间；不自动归一化")
            listed += weight
            identity = row.get(key)
            reason = None
            if source == "holdings" and row.get("kind") != "corporate_equity":
                reason = "not_explicit_corporate_equity"
            elif not isinstance(identity, str) or not identity.strip():
                reason = f"missing_{key}"
            if reason:
                result["exclusions"].append({"basket": source, "index": index,
                                              "reason": reason, "weight": str(weight)})
                continue
            basket = result[target]
            basket[identity] = basket.get(identity, Decimal(0)) + weight
        if listed > 1:
            raise ValidationError("同一披露切面合计超过100%；不自动归一化")
        if source == "holdings":
            result["known_listed_weight"] = str(listed)
        result["issuer_coverage" if source == "holdings" else "sector_coverage"] = str(
            sum(result[target].values(), Decimal(0)),
        )
    return result


def _public_snapshot(snapshot: Mapping) -> dict:
    return {**snapshot,
            "issuers": {key: str(value) for key, value in snapshot["issuers"].items()},
            "sectors": {key: str(value) for key, value in snapshot["sectors"].items()}}


def _overlap(left: Mapping, right: Mapping) -> dict:
    available = (left["status"] == right["status"] == "available"
                 and "missing_holdings" not in left["unknowns"]
                 and "missing_holdings" not in right["unknowns"])
    common = []
    if available:
        for issuer in sorted(left["issuers"].keys() & right["issuers"].keys()):
            a, b = left["issuers"][issuer], right["issuers"][issuer]
            common.append({"issuer_id": issuer, "left_weight": str(a),
                           "right_weight": str(b), "overlap_weight": str(min(a, b))})
    return {
        "status": "available" if available else "unavailable",
        "overlap": str(sum((Decimal(row["overlap_weight"]) for row in common), Decimal(0)))
        if available else None,
        "common_issuers": common, "left_report_date": left["report_date"],
        "right_report_date": right["report_date"],
        "left_issuer_coverage": left["issuer_coverage"],
        "right_issuer_coverage": right["issuer_coverage"],
        "left_denominator": left["denominator"], "right_denominator": right["denominator"],
        "exclusions": {"left": list(left["exclusions"]), "right": list(right["exclusions"])},
        "unknowns": {"left": list(left["unknowns"]), "right": list(right["unknowns"])},
        "formula": "sum(min(left_nav_weight[i], right_nav_weight[i])) for common explicit issuer_id",
        "assumptions": [
            "只比较两份历史披露中明确映射的股票发行人，不归一化部分篮子。",
            "这是已披露部分的重叠，不是完整组合重叠或相关性；零不证明没有共同风险。",
            "两份报告日分别保留，不推断它们属于同一时点或今天的持仓。",
        ],
    }


def pairwise_issuer_overlap(left: Mapping | None, right: Mapping | None) -> dict:
    """Disclosed NAV issuer overlap; accept profiles or portfolio sections.

    Only identical supplied issuer IDs on explicit corporate_equity rows match.
    Missing or non-NAV baskets return None, not a manufactured zero overlap.
    """
    with localcontext() as context:
        context.prec = 160
        return _overlap(_snapshot(left), _snapshot(right))


issuer_overlap = pairwise_issuer_overlap


def _look_through(funds: list[dict], snapshots: list[dict], basket: str,
                  total: Decimal | None) -> dict:
    values: dict[str, Decimal] = {}
    sources: dict[str, list[dict]] = {}
    disclosed_value = Decimal(0)
    disclosure_count = 0
    source = "holdings" if basket == "issuers" else "sectors"
    for fund, snapshot in zip(funds, snapshots):
        if (not fund["included"] or snapshot["status"] != "available"
                or f"missing_{source}" in snapshot["unknowns"]):
            continue
        value = Decimal(fund["market_value"])
        disclosed_value += value
        disclosure_count += 1
        for identity, weight in snapshot[basket].items():
            contribution = value * weight
            values[identity] = values.get(identity, Decimal(0)) + contribution
            sources.setdefault(identity, []).append({
                "index": fund["index"], "fund_id": fund["fund_id"],
                "product_id": fund["product_id"], "nav_date": fund["nav_date"],
                "report_date": snapshot["report_date"], "record_id": snapshot["record_id"],
                "nav_weight": str(weight), "known_value_cny": str(contribution),
            })
    covered = sum(values.values(), Decimal(0)) if total is not None else None
    key = "issuer_id" if basket == "issuers" else "name"
    return {
        "exposures": [{key: identity, "known_value_cny": str(value),
                       "weight_of_known_cny_valued_funds": str(value / total)
                       if total is not None and total > 0 else None,
                       "sources": sources[identity]} for identity, value in sorted(values.items())],
        "covered_value_cny": _text(covered),
        "uncovered_value_cny": _text(total - covered) if total is not None else None,
        "coverage": str(covered / total) if total is not None and total > 0 else None,
        "uncovered_weight": str((total - covered) / total)
        if total is not None and total > 0 else None,
        "disclosure_market_value_cny": str(disclosed_value) if disclosure_count else None,
        "disclosure_fund_count": disclosure_count,
        "formula": "sum(known_CNY_fund_market_value * disclosed_NAV_weight)",
        "unknowns": ["列表外、未映射及缺披露部分未知；未估值基金在此人民币分母之外。"],
    }


def build_portfolio_context(
    positions: Sequence[Mapping], *, selected_product_id: str | None = None,
    as_of: str | None = None,
) -> dict:
    """Build a detached JSON-ready, known-CNY-cohort view.

    Each row is {holding: Portfolio-record mapping, valuation: value_holding
    result mapping, profile: investment_profile mapping or None}. No persistence
    module is imported. ``holding.product_id`` (or explicit ``subject.id``) is
    the only public identity link; the fund code/name never creates a match.
    The precomputed valuation is authoritative, as in summary_holdings; cost
    completeness has no bearing on market-value cohort inclusion.
    """
    positions = _rows(positions, "positions")
    selected_product_id = _optional_text(selected_product_id, "selected_product_id")
    point = timestamp(as_of).isoformat() if as_of is not None else None
    day = point[:10] if point else None
    funds, snapshots, exclusions = [], [], []
    unknowns = []
    with localcontext() as context:
        context.prec = 160 + len(str(len(positions)))
        for index, row in enumerate(positions):
            holding, valuation = row.get("holding") or {}, row.get("valuation") or {}
            if not isinstance(holding, Mapping) or not isinstance(valuation, Mapping):
                raise ValidationError("holding和valuation必须是对象")
            subject = holding.get("subject") or {}
            if not isinstance(subject, Mapping):
                raise ValidationError("subject必须是对象")
            product_id = _optional_text(holding.get("product_id") or subject.get("id"), "product_id")
            fund_id = _optional_text(holding.get("id"), "fund_id")
            currency = _optional_text(valuation.get("currency"), "currency")
            valuation_status = _optional_text(valuation.get("valuation_status"), "valuation_status")
            value = _amount(valuation.get("market_value"))
            nav_date = _date(valuation.get("nav_date"))
            reasons = []
            if valuation.get("currency") != "CNY":
                reasons.append("currency_mismatch" if valuation.get("currency")
                               else "missing_currency")
            if value is None:
                reasons.append("missing_market_value")
            if valuation.get("valuation_status") not in (None, "ready"):
                reasons.append(valuation["valuation_status"])
            if day and nav_date and nav_date > day:
                reasons.append("valuation_after_as_of")
            if nav_date is None:
                unknowns.append({"index": index, "reason": "missing_valuation_date"})
            profile = row.get("profile")
            snapshot = _snapshot(profile, as_of=point)
            if (isinstance(profile, Mapping) and profile.get("product_id") is not None
                    and profile["product_id"] != product_id):
                snapshot.update(status="unavailable", issuers={}, sectors={},
                                issuer_coverage=None, sector_coverage=None)
                snapshot["unknowns"].append("profile_product_id_mismatch")
            fund = {
                "index": index, "fund_id": fund_id, "product_id": product_id,
                "currency": currency, "market_value": _text(value),
                "nav_date": nav_date, "valuation_status": valuation_status,
                "included": not reasons, "exclusions": reasons,
                "weight_of_known_cny_valued_funds": None,
                "disclosure": _public_snapshot(snapshot),
            }
            funds.append(fund)
            snapshots.append(snapshot)
            if reasons:
                exclusions.append({"index": index, "fund_id": fund["fund_id"],
                                   "product_id": product_id, "reasons": reasons})
            if snapshot["unknowns"] or snapshot["exclusions"]:
                unknowns.append({"index": index, "disclosure": list(snapshot["unknowns"]),
                                 "unmapped_rows": list(snapshot["exclusions"])})
        markets = [Decimal(fund["market_value"]) for fund in funds if fund["included"]]
        total = sum(markets, Decimal(0)) if markets else None
        for fund in funds:
            if fund["included"] and total is not None and total > 0:
                fund["weight_of_known_cny_valued_funds"] = str(
                    Decimal(fund["market_value"]) / total,
                )
        selected_rows = [fund for fund in funds if selected_product_id is not None
                         and fund["product_id"] == selected_product_id]
        selected_known = bool(selected_rows) and all(fund["included"] for fund in selected_rows)
        selected_value = sum((Decimal(fund["market_value"]) for fund in selected_rows), Decimal(0)) \
            if selected_known else None
        selected_weight = selected_value / total \
            if selected_known and total is not None and total > 0 else None
        selected = {
            "product_id": selected_product_id, "market_value": _text(selected_value),
            "weight_of_known_cny_valued_funds": _text(selected_weight),
            "status": "available" if selected_weight is not None else
            "not_recorded" if not selected_rows else "unknown",
            "nav_dates": sorted({fund["nav_date"] for fund in selected_rows if fund["nav_date"]}),
        }
        look_through = {basket: _look_through(funds, snapshots, basket, total)
                        for basket in ("issuers", "sectors")}
        overlaps = []
        for a, b in combinations(range(len(funds)), 2):
            overlaps.append({"left_index": a, "right_index": b,
                             "left_product_id": funds[a]["product_id"],
                             "right_product_id": funds[b]["product_id"],
                             **_overlap(snapshots[a], snapshots[b])})
        return {
            "schema_version": SCHEMA_VERSION, "as_of": point,
            "status": "available" if total is not None else "unavailable",
            "scope": "recorded_valued_cny_funds", "currency": "CNY",
            "known_cohort": {
                "market_value": _text(total), "currency": "CNY", "fund_count": len(markets),
                "total_count": len(funds), "unknown_funds": exclusions,
                "exclusions": exclusions,
            },
            "selected": selected, "funds": funds, "look_through": look_through,
            "overlaps": overlaps, "exclusions": exclusions,
            "coverage": {
                "valued_fund_count": len(markets), "recorded_fund_count": len(funds),
                "excluded_fund_count": len(exclusions), "total_personal_assets_known": False,
                "issuer_weight": look_through["issuers"]["coverage"],
                "sector_weight": look_through["sectors"]["coverage"],
            },
            "unknowns": unknowns, "assumptions": list(_ASSUMPTIONS),
        }

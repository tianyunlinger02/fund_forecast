"""Detached current-public-data projections; no requests, files, clocks or Store.

A retrieval clock does not renew a disclosure, an authored thesis or a forecast.
Full NAV series are useful in memory; saved judgments retain only their used basis.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import re

from .advice_evidence import normalize_record
from .domain import ValidationError, decimal, timestamp
from .investment_profile import build_investment_profile, prefer_disclosure
from .issuer_fundamentals import build_issuer_analysis
from .research_thesis import (
    _same_source_basis, economic_content_compatible, research_citation_record_ids,
    research_explained_record_ids, with_research_basis,
)

_PUBLIC = ("id", "symbol", "name", "kind", "share_class", "exchange", "is_demo", "category")
_NAV = ("product_id", "symbol", "share_class", "date", "value", "currency", "nav_type",
        "source", "source_url", "observed_at", "available_at", "published_at", "basis")
_RECORD = ("id", "schema_version", "normalization_version", "product_id", "family", "source",
           "upstream", "source_url", "report_date", "period_start", "observed_at", "available_at",
           "published_at", "publication_precision", "effective_from", "effective_to", "checked_at")
_MESSAGE = ("id", "evidence_id", "title", "source_url", "index_url", "published_at", "publication_precision",
            "observed_at", "available_at", "checked_at", "content_observed_at", "excerpt",
            "content_status", "body_status", "issuer_ids", "channels", "relevance_reason",
            "event_status", "interpretation_status")
_CHECK = ("job_id", "product_id", "family", "source_url", "status", "parse_status", "http_status",
          "checked_at", "observed_at", "source_date", "requested_window", "request_made",
          "reused_response", "record_count", "message_count", "reason", "discovered_count",
          "parsed_count")
_PROFILE_FAMILIES = {"mandate", "portfolio", "management", "valuation"}


def _fields(row: Mapping, fields) -> dict:
    return {key: deepcopy(row[key]) for key in fields if key in row}


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _point(context: Mapping):
    value = context.get("evaluated_at") or context.get("checked_at")
    return timestamp(value) if value else None


def _eligible(row: Mapping, point) -> bool:
    if point is None:
        return False
    day = point.date().isoformat()
    return bool(
        timestamp(row["observed_at"]) <= timestamp(row["available_at"]) <= point
        and row["report_date"] <= day
        and (not row.get("effective_from") or row["effective_from"] <= day)
        and (not row.get("effective_to") or row["effective_to"] >= day)
        and (row.get("publication_precision") != "timestamp"
             or timestamp(row["published_at"]) <= point)
        and (row.get("publication_precision") != "date" or row["published_at"] <= day)
    )


def _order(row: Mapping) -> tuple:
    return row["report_date"], timestamp(row["available_at"]), row.get("id", "")


def _compact_record(row: Mapping) -> dict:
    result = _fields(row, _RECORD)
    payload = _mapping(row.get("payload"))
    result["payload"] = deepcopy({key: value for key, value in payload.items()
                                   if key not in {"closes", "observations"}})
    for key in ("closes", "observations"):
        if key in payload:
            rows = payload[key]
            result["payload"]["latest_" + key] = deepcopy(rows[-1]) if rows else None
            result["payload"][key + "_count"] = len(rows)
    return result


def snapshot_online_context(context: Mapping, product_ids=None) -> dict:
    """Compact and deep-copy public facts, not raw pages or full market histories.

    The shape is idempotent so display and save can share the same object. A NAV
    entry here means acquired, not necessarily selected: projections record use
    or rejection separately. All source dates survive unchanged.
    """
    if not isinstance(context, Mapping):
        raise ValidationError("online_context应为本次公开资料查询对象")
    if isinstance(product_ids, str):
        product_ids = [product_ids]
    selected = set(product_ids) if product_ids is not None else None

    def included(identifier):
        return selected is None or identifier in selected

    result = _fields(context, ("schema_version", "kind", "product_id", "status", "started_at", "evaluated_at",
                               "checked_at", "requests_made", "warnings"))
    result["public_subjects"] = [_fields(row, _PUBLIC) for row in context.get("public_subjects", [])
                                 if included(row.get("id"))]
    result["source_checks"] = [_fields(row, _CHECK) for row in context.get("source_checks", [])
                               if included(row.get("product_id"))]
    result["nav_by_product"], result["nav_observation_counts"] = {}, {}
    counts = _mapping(context.get("nav_observation_counts"))
    for identifier, rows in _mapping(context.get("nav_by_product")).items():
        if not included(identifier):
            continue
        latest = max(rows, key=lambda row: (row.get("date", ""), row.get("available_at", "")),
                     default=None)
        result["nav_by_product"][identifier] = [_fields(latest, _NAV)] if latest else []
        result["nav_observation_counts"][identifier] = counts.get(identifier, len(rows))
    result["records_by_product"] = {
        identifier: [_compact_record(row) for row in rows]
        for identifier, rows in _mapping(context.get("records_by_product")).items()
        if included(identifier)
    }
    result["messages_by_product"] = {
        identifier: [_fields(row, _MESSAGE) for row in rows]
        for identifier, rows in _mapping(context.get("messages_by_product")).items()
        if included(identifier)
    }
    for key in ("review_signals_by_product", "execution_conditions_by_product",
                "projections_by_product"):
        if key in context:
            result[key] = deepcopy({identifier: value for identifier, value in context[key].items()
                                    if included(identifier)})
    return result


def derive_review_signals(thesis: Mapping | None, current_records, messages=None) -> list[dict]:
    """Detect changed cited economic content, never title sentiment or fetch times.

    ``current_records`` may contain the old cited records as well as incoming
    ones. Without the old economic basis, a new URL/title cannot prove a change.
    Messages remain visible in current rationale but do not mechanically revoke
    an author's conclusion.
    """
    thesis = _mapping(thesis)
    payload = _mapping(thesis.get("payload"))
    product_id = _mapping(payload.get("product")).get("id")
    by_id = {row.get("id"): row for row in current_records if row.get("product_id") == product_id}
    for citation in payload.get("citations", []):
        for row in citation.get("record_basis", []):
            if row.get("product_id") == product_id:
                by_id.setdefault(row["id"], row)
    records = list(by_id.values())
    source_time = _mapping(thesis.get("source")).get("available_at")
    result, seen = [], set()
    explained = research_explained_record_ids(payload)
    for citation in payload.get("citations", []):
        cited_records = [by_id[identifier] for identifier in research_citation_record_ids(citation)
                         if identifier in by_id]
        for cited in cited_records:
            for candidate in records:
                marker = (citation.get("id"), candidate.get("id"))
                if (marker in seen or candidate.get("id") == cited.get("id")
                        or candidate.get("id") in explained):
                    continue
                if (not _same_source_basis(cited, candidate)
                        or economic_content_compatible(cited, candidate)
                        or candidate["report_date"] < cited["report_date"]
                        or timestamp(candidate["available_at"]) <= timestamp(cited["available_at"])
                        or (source_time and timestamp(candidate["available_at"]) <= timestamp(source_time))):
                    continue
                seen.add(marker)
                label = {"mandate": "基金合同投向", "portfolio": "披露持仓", "management": "经理名单",
                         "fees": "费率", "operations": "申赎状态", "events": "正式事件",
                         "valuation": "估值", "macro": "市场序列",
                         "issuer_fundamentals": "公司盈利或估值依据"}.get(cited["family"], "公开资料")
                result.append({
                    "kind": "cited_economic_basis_changed", "status": "review_required",
                    "product_id": product_id, "family": cited["family"],
                    "citation_id": citation.get("id"), "previous_record_id": cited.get("id"),
                    "current_record_id": candidate.get("id"), "source_url": candidate["source_url"],
                    "report_date": candidate["report_date"], "available_at": candidate["available_at"],
                    "interpretation_status": "unreviewed",
                    "reason": f"原论证引用{citation.get('id')}对应的{label}，"
                              f"已有{candidate['report_date']}的不同经济内容；需核对原理由是否仍成立。",
                })
    return result


def _operation(record: Mapping) -> dict:
    result = _fields(record, ("report_date", "source_url", "observed_at", "available_at",
                              "published_at", "publication_precision"))
    payload = record["payload"]
    result.update(subscription=payload["subscription_state"], redemption=payload["redemption_state"],
                  channel_verified=payload["channel_verified"], record_id=record.get("id"),
                  reasons=["本次在线取得的公开申赎披露；不等于支付宝当前页面已确认。"])
    if "limit_amount" in payload:
        result["limit_amount"] = payload["limit_amount"]
    return result


def _projected_records(context: Mapping, product_id: str | None) -> list:
    rows = _mapping(context.get("records_by_product")).get(product_id, [])
    projection = _mapping(context.get("projections_by_product")).get(product_id)
    if projection is not None:
        accepted_ids = set(projection.get("disclosure_record_ids", []))
        rows = [row for row in rows if row.get("id") in accepted_ids]
    return rows


def current_execution_conditions(context: Mapping | None, product_id: str | None,
                                 existing: Mapping | None = None) -> dict:
    """Select an actually available operations record, keeping dated old fallback."""
    context = _mapping(context)
    result = deepcopy(dict(existing or {}))
    candidate = _mapping(context.get("execution_conditions_by_product")).get(product_id)
    point = _point(context)
    rows = _projected_records(context, product_id)
    operations = []
    for row in rows:
        if row.get("family") != "operations" or row.get("product_id") != product_id:
            continue
        record = normalize_record({key: value for key, value in row.items() if key != "checked_at"})
        if _eligible(record, point):
            operations.append(record)
    if operations:
        latest = _operation(max(operations, key=_order))
        if not candidate or latest["report_date"] >= candidate.get("report_date", ""):
            candidate = latest
    if candidate and candidate.get("report_date", "") >= result.get("report_date", ""):
        result = deepcopy(dict(candidate))
    return result


def _local_records(detail: Mapping, product_id: str, point=None) -> list[dict]:
    rows = [*(detail.get("evidence_records") or []), *(detail.get("background_records") or [])]
    if point is not None:
        rows = with_research_basis(rows, detail.get("research_thesis"),
                                   product_id=product_id, as_of=point.isoformat())
    profile = _mapping(detail.get("investment_profile"))
    for family in _PROFILE_FAMILIES:
        section = profile.get(family)
        if section:
            rows.append({**_fields(section, _RECORD), "id": section["record_id"],
                         "product_id": product_id, "family": family, "payload": section["payload"]})
    result = {}
    for row in rows:
        record = normalize_record({key: value for key, value in row.items() if key != "checked_at"})
        if record["product_id"] != product_id:
            raise ValidationError("本地公开资料与当前基金身份不匹配")
        result.setdefault(record["id"], record)
    return list(result.values())


def _nav_rows(rows, *, product: Mapping, detail: Mapping, point) -> tuple[list[dict], list[str]]:
    accepted, reasons = [], []
    currency = _mapping(detail.get("latest")).get("currency") or product.get("currency")
    if not currency:
        mandate = _mapping(_mapping(detail.get("investment_profile")).get("mandate"))
        currency = _mapping(mandate.get("payload")).get("base_currency")
    for row in rows:
        try:
            if not product.get("id") or point is None:
                raise ValidationError("缺少可核对的公开基金身份或实际查询时间")
            if row.get("product_id", product["id"]) != product["id"]:
                raise ValidationError("净值所属产品不匹配")
            for key in ("symbol", "share_class"):
                if key in row and row[key] != product.get(key, ""):
                    raise ValidationError("净值代码或份额类别不匹配")
            if (not re.fullmatch(r"[A-Z]{3}", str(row.get("currency", "")))
                    or (currency and row["currency"] != currency)):
                raise ValidationError("净值币种与当前基金口径不匹配")
            if row.get("nav_type") not in {"unit_nav", "final_unit_nav", "final_unit_nav_candidate"}:
                raise ValidationError("不是可用的单位净值，估算和累计净值不混用")
            if decimal(row["value"]) <= 0 or date.fromisoformat(row["date"]) > point.date():
                raise ValidationError("单位净值数值或所属日期不可用")
            observed, available = timestamp(row["observed_at"]), timestamp(row["available_at"])
            if observed > available or available > point or row["date"] > observed.date().isoformat():
                raise ValidationError("单位净值尚未实际可用")
            if row.get("published_at") and timestamp(row["published_at"]) > available:
                raise ValidationError("单位净值披露时间晚于可用时间")
            if not row.get("source") or not row.get("source_url"):
                raise ValidationError("单位净值缺少实际来源")
            accepted.append(_fields(row, _NAV))
        except (ValidationError, ValueError, KeyError) as exc:
            reasons.append(f"未采用一条在线净值：{exc}。")
    if len({row["currency"] for row in accepted}) > 1:
        return [], [*reasons, "在线净值币种存在冲突，没有选其中一种混算。"]
    values_by_date = {}
    for row in accepted:
        values_by_date.setdefault(row["date"], set()).add(decimal(row["value"]))
    conflicts = {day for day, values in values_by_date.items() if len(values) > 1}
    if conflicts:
        reasons.append("在线同日单位净值存在不同数值，未默选其中一项：" + "、".join(sorted(conflicts)) + "。")
        accepted = [row for row in accepted if row["date"] not in conflicts]
    return accepted, list(dict.fromkeys(reasons))


def project_current_detail(local_detail: Mapping, context: Mapping, product_id: str) -> dict:
    """Overlay one current result in RAM; never mutate a thesis, forecast or input.

    Optional ``evidence_records`` supplies the full local normalized evidence for
    rebuilding pure profile/issuer calculations. Without incoming structured
    disclosures, old profile dates remain untouched, regardless of new headlines.
    """
    result = deepcopy(dict(local_detail))
    product = _mapping(result.get("product"))
    if product and product.get("id") != product_id:
        raise ValidationError("当前详情与请求的公开基金身份不匹配")
    public = next((row for row in context.get("public_subjects", []) if row.get("id") == product_id), {})
    identity_conflict = any(public.get(key) != product.get(key) for key in
                            ("symbol", "kind", "share_class", "exchange")
                            if key in public and key in product)
    if not product and public:
        product = _fields(public, _PUBLIC)
        result["product"] = product
    point = _point(context)
    reasons = list(result.get("reasons") or [])
    checks = [_fields(row, _CHECK) for row in context.get("source_checks", [])
              if row.get("product_id") == product_id]
    nav_checks = [row for row in checks if row.get("family") == "nav"]
    incoming = _mapping(context.get("nav_by_product")).get(product_id, [])
    if identity_conflict:
        accepted, rejected = [], ["在线公开身份与所选代码或份额类别不匹配，保留原有日期资料。"]
    else:
        accepted, rejected = _nav_rows(incoming, product=product, detail=result, point=point)
    reasons.extend(rejected)
    origin = "local_reference" if result.get("latest") else "missing"
    if accepted:
        currency = accepted[0]["currency"]
        history = {row["date"]: deepcopy(row) for row in result.get("history", [])
                   if row.get("currency") == currency}
        latest_local = _mapping(result.get("latest"))
        for row in sorted(accepted, key=lambda row: (timestamp(row["available_at"]), row["date"])):
            history[row["date"]] = deepcopy(row)
        history = [history[key] for key in sorted(history)]
        online_latest = max(accepted, key=lambda row: (row["date"], timestamp(row["available_at"])))
        if online_latest["date"] >= latest_local.get("date", ""):
            result["latest"] = deepcopy(online_latest)
            result["last_successful_check_at"] = online_latest["observed_at"]
            origin = "current_online"
        result["history"] = history
        result["previous_change"] = (float(decimal(history[-1]["value"]) / decimal(history[-2]["value"]) - 1)
                                     if len(history) > 1 else None)
        result["status"] = "ready"
        reasons = [reason for reason in reasons if reason not in {
            "本地暂无实际可用的单位净值；显式更新后才会请求数据。",
            "显示本地已取得的单位净值；净值日期不是今天的实时估值。",
        }]
        reasons.append("本次在线单位净值直接用于当前估值；净值所属日不等于今天的实时成交价。"
                       if origin == "current_online" else
                       "在线来源返回的净值日早于已有记录；较新记录仍保留原取得日期，未续鲜。")
        if online_latest["nav_type"] == "final_unit_nav_candidate":
            reasons.append("在线来源为公布净值候选，未据此宣称已经官方核实。")
    elif incoming or nav_checks:
        reasons.append("本次未取得可采用的在线净值；如有旧值，仅按原净值日和取得时间参考。")
    if nav_checks:
        latest_check = max(nav_checks, key=lambda row: row.get("checked_at") or "")
        result["last_checked_at"] = latest_check.get("checked_at")
        result["latest_attempt_status"] = latest_check.get("status")

    local_records = _local_records(result, product_id, point)
    online_records = []
    if not identity_conflict:
        for row in _mapping(context.get("records_by_product")).get(product_id, []):
            if row.get("product_id") != product_id:
                reasons.append("未采用属于其他基金的在线披露。")
                continue
            record = normalize_record({key: value for key, value in row.items() if key != "checked_at"})
            if record["family"] != "research_thesis" and _eligible(record, point):
                online_records.append(record)
    records = {row["id"]: row for row in local_records}
    records.update({row["id"]: row for row in online_records})
    eligible = [row for row in records.values() if _eligible(row, point)]
    latest_records = {}
    for record in sorted(eligible, key=_order):
        if prefer_disclosure(record, latest_records.get(record["family"])):
            latest_records[record["family"]] = record
    new_families = {row["family"] for row in online_records
                    if row["id"] not in {item["id"] for item in local_records}}
    if new_families & _PROFILE_FAMILIES:
        result["investment_profile"] = build_investment_profile(eligible, as_of=point.isoformat())
    if (new_families & (_PROFILE_FAMILIES | {"issuer_fundamentals"})
            and ("evidence_records" in result or not result.get("issuer_analysis")
                 or "issuer_fundamentals" in new_families)):
        profile = build_investment_profile(eligible, as_of=point.isoformat())
        result["issuer_analysis"] = build_issuer_analysis(eligible, profile=profile, as_of=point.isoformat())
    if online_records:
        result["background_records"] = [latest_records[key] for key in sorted(latest_records)
                                         if key in {"mandate", "portfolio", "management", "fees", "operations"}]
        if portfolio := latest_records.get("portfolio"):
            result["portfolio"] = {**deepcopy(portfolio["payload"]),
                                   **_fields(portfolio, ("report_date", "source_url", "source", "observed_at", "available_at"))}
    signals = derive_review_signals(result.get("research_thesis"), eligible)
    selected = snapshot_online_context(context, [product_id])
    # Use eligible records only for actions. Rejected records remain represented
    # by source checks/reasons, not promoted through the saved projection.
    selected["records_by_product"][product_id] = [_compact_record(row) for row in online_records]
    selected["review_signals_by_product"] = {product_id: signals}
    conditions = current_execution_conditions(selected, product_id, result.get("execution_conditions"))
    if conditions:
        result["execution_conditions"] = conditions
        selected["execution_conditions_by_product"] = {product_id: deepcopy(conditions)}
    source_context = {
        "checked_at": context.get("checked_at"), "source_checks": checks,
        "nav_origin": origin, "nav_date": _mapping(result.get("latest")).get("date"),
        "nav_observed_at": _mapping(result.get("latest")).get("observed_at"),
        "accepted_nav_observations": len(accepted), "rejected_nav_reasons": rejected,
        "disclosure_record_ids": [row["id"] for row in online_records],
    }
    selected["projections_by_product"] = {product_id: deepcopy(source_context)}
    result.update(online_source_context=source_context, online_context=selected,
                  current_nav_history=deepcopy(accepted), reasons=list(dict.fromkeys(reasons)))
    return result

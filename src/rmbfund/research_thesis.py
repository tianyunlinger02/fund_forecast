"""Pure as-of projection of authored research, not a person's investment advice.

Only an explicit visible successor supersedes a thesis. Independent heads remain
contested; import order never silently chooses an author's position.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .advice_evidence import (
    MAX_EVIDENCE_READ_BYTES,
    MAX_EVIDENCE_RECORDS,
    bounded_json,
    normalize_record,
)
from .domain import ValidationError, timestamp


def _informational_warnings(payload: dict) -> list[str]:
    warnings = []
    if payload["assessment"]["missing"]:
        warnings.append("作者明确列出的资料缺口仍未补齐；不把未知当作零或反向结论。")
    if any(citation["kind"] == "linked_not_archived" for citation in payload["citations"]):
        warnings.append("部分依据只有公开链接，没有归档发布者原件；链接存在不等于已核对全文。")
    if any(citation["kind"] == "reviewed_excerpt" for citation in payload["citations"]):
        warnings.append("部分依据是人工录入摘录及其哈希，只证明所录入文字，不证明完整原件或来源方认证。")
    if any(fact["status"] in {"plan", "unresolved", "assumption", "model_judgment"}
           for fact in payload["facts"]):
        warnings.append("论题包含计划或未确证事实、假设或模型判断；这些不确定性保留，不自动否定其他行动依据。")
    if payload.get("schema_version") == 2:
        warnings.append("本研究由AI生成；程序核对摘录与计算依据不是人工复核，也不证明投资结论有效。")
        if any(row["kind"] == "online_evidence" for row in payload["citations"]):
            warnings.append("在线依据只保留本次采用的定位摘录及原取得日期，没有归档整份网页或PDF。")
    return warnings


def _needed_review(payload: dict, day: str) -> list[str]:
    reasons = []
    if payload.get("schema_version") != 2 and payload["posture"] == "unresolved":
        reasons.append("posture_unresolved")
    if payload["assessment"]["review_date"] <= day:
        reasons.append("review_date_elapsed")
    if payload["horizon"]["end"] < day:
        reasons.append("research_horizon_elapsed")
    return reasons


def _source(record: dict) -> dict:
    return {key: record[key] for key in (
        "source", "upstream", "source_url", "observed_at", "available_at", "checked_at",
    )}


def _normalize_projection_record(record: Mapping[str, Any]) -> dict:
    checked = timestamp(record.get("checked_at", record["available_at"]))
    value = normalize_record({key: item for key, item in record.items() if key != "checked_at"})
    if checked < timestamp(value["available_at"]):
        raise ValidationError("证据复核投影不能早于实际可用时间")
    value["checked_at"] = checked.isoformat()
    return value


def _metric_series(metric: Mapping, release: Mapping) -> tuple:
    """Comparable series, not an assertion that different periods are equal."""
    return (
        metric["name"], metric["category"], metric["period"]["basis"], metric["currency"],
        metric["unit"], metric["accounting_basis"], metric.get("accounting_basis_detail"),
        metric.get("security_id", release.get("security_id")), metric.get("eps_basis"),
        metric.get("share_basis"), metric.get("split_basis"), metric.get("comparison_basis"),
    )


def _same_source_basis(cited: dict, candidate: dict) -> bool:
    """Match an economic basis across publishers; keep the old integration name.

    The same product/object/denominator/scope is required. A later comparable
    fiscal quarter can prompt review; a year/quarter, another security or an
    unrelated benchmark cannot substitute for the cited basis.
    """
    if any(candidate.get(key) != cited.get(key) for key in ("product_id", "family")):
        return False
    before, after = cited["payload"], candidate["payload"]
    family = cited["family"]
    if family == "issuer_fundamentals":
        if any(before.get(key) != after.get(key) for key in ("kind", "issuer_id", "security_id")):
            return False
        if before["kind"] == "financial_release":
            return any(
                _metric_series(old, before) == _metric_series(new, after)
                and new["period"]["start"] >= old["period"]["start"]
                and (old["period"]["end"] is None or new["period"]["end"] is None
                     or new["period"]["end"] >= old["period"]["end"])
                for old in before["metrics"] if old["period_role"] == "primary"
                for new in after["metrics"] if new["period_role"] == "primary"
            )
        keys = (("security_id", "currency", "exchange", "price_basis")
                if before["kind"] == "price_history" else
                ("security_id", "exchange", "quote_currency", "share_basis", "ads_ratio"))
    elif family == "macro":
        keys = ("series_id", "currency", "kind", "unit", "tenor_years")
    elif family == "valuation":
        keys = ("benchmark_id", "region", "pe_definition", "pb_definition", "compatible_scope")
    elif family == "portfolio":
        keys = ("denominator", "holdings_scope")
    elif family == "scale":
        keys = ("combined_share_classes",)
    elif family == "mandate":
        keys = ("fund_id", "symbol", "share_class")
    elif family == "events":
        return bool({row["event_id"] for row in before["events"]}
                    & {row["event_id"] for row in after["events"]})
    else:
        keys = ()
    return all(before.get(key) == after.get(key) for key in keys)


def research_citation_record_ids(citation: Mapping) -> set[str]:
    """Economic records actually adopted by one archived/online/calculated cite."""
    if citation.get("kind") == "archived_evidence":
        return {citation["record_id"]}
    return set(citation.get("record_ids") or [])


def with_research_basis(
    records: list[Mapping[str, Any]], thesis: Mapping | None, *, product_id: str, as_of: str,
) -> list[Mapping[str, Any]]:
    """Merge only one selected v2 note's adopted, actually available economic rows.

    An explicit alternative may reuse its own cited rows during that target's
    comparison. This is a read projection, not an import or a renewed source check.
    Missing/contested notes and other products contribute nothing.
    """
    thesis = thesis or {}
    payload, source = thesis.get("payload") or {}, thesis.get("source") or {}
    point = timestamp(as_of)
    products = {payload.get("product", {}).get("id")} | {
        row["product_id"] for row in payload.get("alternatives", [])
    }
    if (thesis.get("status") != "available" or payload.get("schema_version") != 2
            or product_id not in products or not source.get("available_at")
            or any(timestamp(source[key]) > point for key in
                   ("observed_at", "available_at", "checked_at") if source.get(key))):
        return list(records)
    result = {row["id"]: row for row in records}
    day = point.date().isoformat()
    for citation in payload.get("citations", []):
        adopted = research_citation_record_ids(citation)
        for row in citation.get("record_basis", []):
            if row.get("product_id") != product_id or row.get("id") not in adopted:
                continue
            record = normalize_record({key: value for key, value in row.items()
                                       if key != "checked_at"})
            if (record["family"] == "research_thesis" or timestamp(record["available_at"]) > point
                    or (record["effective_from"] and record["effective_from"] > day)
                    or (record["effective_to"] and record["effective_to"] < day)):
                continue
            previous = result.get(record["id"])
            if previous is None or timestamp(record["available_at"]) < timestamp(previous["available_at"]):
                result[record["id"]] = record
    return list(result.values())


def research_explained_record_ids(payload: Mapping) -> set[str]:
    """Clear only changes connected to the v2 explanation's actual fact citations."""
    if payload.get("schema_version") != 2:
        return set()
    facts = set(payload.get("change_evaluation", {}).get("fact_ids", []))
    citations = {identifier for fact in payload.get("facts", []) if fact["id"] in facts
                 for identifier in fact["citation_ids"]}
    adopted = [row for row in payload.get("citations", []) if row["id"] in citations]
    input_documents = {identifier for row in adopted if row["kind"] == "calculation"
                       for identifier in row.get("input_evidence_ids", [])}
    adopted.extend(row for row in payload.get("citations", [])
                   if row.get("evidence_id") in input_documents)
    return {identifier for row in adopted for identifier in research_citation_record_ids(row)}


def _economic_content(record: Mapping) -> str:
    payload = dict(record["payload"])
    if record["family"] == "issuer_fundamentals":
        payload = {key: item for key, item in payload.items()
                   if key not in {"locator", "release_id", "revision_id", "revision_kind",
                                  "revises_revision_id", "issuer_name"}}
        if "metrics" in payload:
            payload["metrics"] = sorted(
                ({key: value for key, value in row.items() if key != "locator"}
                 for row in payload["metrics"]), key=bounded_json,
            )
    return bounded_json(payload)


def economic_content_compatible(cited: Mapping, candidate: Mapping) -> bool:
    """Compare adopted economic claims, not display names or missing fee data.

    A fuller fund name does not change its mandate. An unknown fee is not zero
    and is not a contradictory rate either; when both rates are known they must
    agree. This only affects change interpretation, never records or their IDs.
    """
    if any(cited.get(key) != candidate.get(key) for key in ("product_id", "family")):
        return False
    family = cited["family"]
    if family not in {"mandate", "fees"}:
        return _economic_content(cited) == _economic_content(candidate)
    before, after = dict(cited["payload"]), dict(candidate["payload"])
    if family == "mandate":
        before.pop("name", None)
        after.pop("name", None)
    else:
        # subscription_rate is the only optional fee field. Missing versus any
        # known value is unknown (not a zero assertion); all required rates and
        # redemption bands remain strict, and two known subscription rates differ.
        if before.get("subscription_rate") is None or after.get("subscription_rate") is None:
            before.pop("subscription_rate", None)
            after.pop("subscription_rate", None)
    return bounded_json(before) == bounded_json(after)


def _same_reporting_period(cited: Mapping, candidate: Mapping) -> bool:
    if candidate["report_date"] != cited["report_date"]:
        return False
    before, after = cited["payload"], candidate["payload"]
    if cited["family"] == "issuer_fundamentals" and before["kind"] == "financial_release":
        return any(_metric_series(old, before) == _metric_series(new, after)
                   and old["period"] == new["period"]
                   for old in before["metrics"] if old["period_role"] == "primary"
                   for new in after["metrics"] if new["period_role"] == "primary")
    return True


def _source_evolution_review(thesis: dict, source_records: Mapping[str, dict]) -> list[str]:
    """Flag unaddressed economic changes and same-report source conflicts."""
    result = []
    explained = research_explained_record_ids(thesis["payload"])
    for citation in thesis["payload"]["citations"]:
        for identifier in sorted(research_citation_record_ids(citation)):
            cited = source_records.get(identifier)
            if cited is None:
                continue
            for candidate in source_records.values():
                if (candidate["id"] == cited["id"] or candidate["id"] in explained
                        or not _same_source_basis(cited, candidate)
                        or economic_content_compatible(cited, candidate)
                        or candidate["report_date"] < cited["report_date"]):
                    continue
                same_report_conflict = (
                    _same_reporting_period(cited, candidate)
                    and not (cited["payload"].get("revision_kind") == "revision"
                             and cited["payload"].get("revises_revision_id")
                             == candidate["payload"].get("revision_id"))
                    and not (candidate["payload"].get("revision_kind") == "revision"
                             and candidate["payload"].get("revises_revision_id")
                             == cited["payload"].get("revision_id"))
                )
                newer = timestamp(candidate["available_at"]) > max(
                    timestamp(cited["available_at"]), timestamp(thesis["available_at"]),
                )
                if not newer and not same_report_conflict:
                    continue
                change = ("较新经济内容（同期经济内容冲突）" if newer else "同期经济内容冲突") if same_report_conflict else "较新经济内容"
                result.append(f"引用{citation['id']}对应的{cited['family']}资料已有{change}，需复核研究判断。")
    return list(dict.fromkeys(result))


def build_research_thesis(
    records: list[Mapping[str, Any]], *, product_id: str, as_of: str,
) -> dict[str, Any]:
    """Expose one visible thesis and explicit citation/review limitations.

    ``records`` can be the existing ``Store.advice_evidence`` projection with
    all families. URL-only references and reviewer-entered excerpts are never
    promoted into archived publisher originals by this projection. Newer source
    records are only a review trigger when they can replace an archived economic
    basis; unrelated news and same-id refetches do not invalidate a thesis.
    """
    if not isinstance(product_id, str) or not product_id.strip() or len(product_id) > 80:
        raise ValidationError("研究论题产品标识无效")
    point = timestamp(as_of)
    if not isinstance(records, list) or len(records) > MAX_EVIDENCE_RECORDS:
        raise ValidationError("研究论题输入必须是有界记录列表")
    thesis_by_id: dict[str, dict] = {}
    source_by_id: dict[str, dict] = {}
    size = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValidationError("研究论题记录必须是对象")
        if record.get("product_id") != product_id:
            continue
        if "family" not in record or "available_at" not in record:
            raise ValidationError("研究论题投影记录缺少家族或可用时间")
        value = _normalize_projection_record(record)
        if (timestamp(value["observed_at"]) > point or timestamp(value["available_at"]) > point
                or timestamp(value["checked_at"]) > point):
            continue
        size += len(bounded_json(value).encode("utf-8"))
        if size > MAX_EVIDENCE_READ_BYTES:
            raise ValidationError("研究论题投影总字节超过读取上限")
        if value["family"] == "research_thesis":
            thesis_by_id.setdefault(value["id"], value)
        else:
            source_by_id.setdefault(value["id"], value)
    missing = {
        "status": "missing", "record_id": None, "payload": None,
        "source": None, "citations": [], "needed_review": ["no_research_thesis_available"],
        "warnings": [],
    }
    if not thesis_by_id:
        return missing
    superseded = set()
    invalid_predecessor = False
    for value in thesis_by_id.values():
        predecessor = value["payload"]["revision"]["supersedes"]
        if predecessor is None:
            continue
        prior = thesis_by_id.get(predecessor)
        if (prior is None or prior["payload"]["note_id"] != value["payload"]["note_id"]
                or timestamp(prior["available_at"]) > timestamp(value["available_at"])
                or prior["payload"]["research_date"] > value["payload"]["research_date"]):
            invalid_predecessor = True
        else:
            superseded.add(predecessor)
    heads = sorted((value for value in thesis_by_id.values() if value["id"] not in superseded),
                   key=lambda value: value["id"])
    if len(heads) != 1 or invalid_predecessor:
        result = {
            "status": "contested", "record_id": None, "payload": None,
            "source": None, "citations": [], "needed_review": [], "warnings": [],
        }
        if len(heads) != 1:
            result["needed_review"].append("multiple_visible_thesis_heads")
        if invalid_predecessor:
            result["needed_review"].append("superseded_record_not_available_as_of")
        return result
    selected = heads[0]
    for citation in selected["payload"]["citations"]:
        for basis in citation.get("record_basis", []):
            source_by_id.setdefault(basis["id"], basis)
    needed = _needed_review(selected["payload"], point.date().isoformat())
    needed.extend(_source_evolution_review(selected, source_by_id))
    return {
        "status": "available", "record_id": selected["id"],
        "payload": selected["payload"], "source": _source(selected),
        "citations": selected["payload"]["citations"],
        "needed_review": needed,
        "warnings": _informational_warnings(selected["payload"]),
    }

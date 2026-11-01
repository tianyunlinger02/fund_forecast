"""Explicit API research over public material; no web, database, or file writes.

Only adopted excerpts and research leave this runner. The model explains existing
facts/calculations, while the program supplies identity, clocks, lineage and source
bindings. A contract check is not a check of investment correctness.
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any

from .advice_evidence import bounded_json
from .domain import ValidationError, now_iso, timestamp
from .providers.research_api import ResearchAPIError, complete_research
from .research_contract import normalize_generated_research, research_output_example
from .research_settings import is_configured, normalize_research_api_settings

MAX_EVIDENCE_ROUNDS = 2  # Additional evidence rounds, after the initial model call.
MAX_DOCUMENT_CHARS = 120_000
MAX_INPUT_BYTES = 1_500_000
_PRIVATE_KEYS = {
    "shares", "share_count", "shares_held", "holding_shares", "units", "quantity",
    "cost", "cost_basis", "cost_cny", "average_cost", "avg_cost", "holding_cost",
    "budget", "budget_cny", "planned_budget", "user_budget", "available_cash",
    "personal", "personal_profile", "personal_assessment", "portfolio_value",
    "position_value", "market_value", "holding_value", "planned_buy", "planned_redemption",
    "amount_to_invest", "amount_to_redeem", "position_size", "api_key", "access_token",
    "authorization", "secret", "password", "api_token",
}
_DETAIL_FIELDS = {
    "status", "reasons", "latest", "previous_change", "investment_profile",
    "execution_conditions", "portfolio", "current_projection", "review_signals",
}
_CONTEXT_FIELDS = {
    "status", "started_at", "evaluated_at", "checked_at", "warnings", "public_subjects",
    "source_checks", "messages_by_product", "review_signals_by_product",
}
_DOC_FIELDS = {
    "id", "product_id", "source_url", "title", "published_at", "publication_precision",
    "observed_at", "available_at", "content_status",
}
_RECORD_FIELDS = {
    "id", "schema_version", "normalization_version", "product_id", "family", "source",
    "upstream", "source_url", "report_date", "period_start", "observed_at", "available_at",
    "published_at", "publication_precision", "effective_from", "effective_to", "payload",
}
_BODY_UNAVAILABLE = {"title_only", "unparsed_document", "unavailable", "error", "empty"}
_TOPIC = re.compile(
    r"持仓|投资组合|经理|费率|费用|申购|赎回|风险|业绩|利润|收入|利率|汇率|政策|"
    r"outlook|revenue|earnings|margin|guidance|valuation|interest|currency|portfolio|risk", re.I,
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|％|亿|万|billion|million)?", re.I)
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _public(value: Any) -> Any:
    """Drop private sizing/credential fields; retain public fund holdings and fees."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {key: _public(item) for key, item in value.items()
                if isinstance(key, str) and key.casefold() not in _PRIVATE_KEYS
                and not key.casefold().startswith("personal_")}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValidationError("公开研究输入须为JSON兼容资料")


def _fields(value: Mapping, fields: set) -> dict:
    return _public({key: item for key, item in value.items() if key in fields})


def _product(value) -> dict:
    raw = _public(value)
    keys = ("id", "symbol", "name", "kind", "share_class", "exchange")
    if not isinstance(raw, Mapping) or any(not isinstance(raw.get(key), str) for key in keys):
        raise ValidationError("研究产品缺少公开身份字段")
    return {key: raw[key] for key in keys}


def _candidate_ids(product: Mapping, comparison) -> set[str]:
    ids = {product["id"]}

    def visit(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"product_id", "peer_id", "target_product_id", "subject_id"} and isinstance(item, str):
                    ids.add(item)
                elif key == "product" and isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
                else:
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for row in value:
                visit(row)
    visit(comparison)
    return ids


def _rows(value) -> list[Mapping]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        result = []
        for row in value.values():
            if isinstance(row, list):
                result.extend(item for item in row if isinstance(item, Mapping))
            elif isinstance(row, Mapping):
                result.append(row)
        return result
    return []


def _previous_payload(previous) -> Mapping:
    value = _mapping(previous)
    return _mapping(value.get("payload", value))


def _source_documents(context: Mapping, previous, products: set,
                      peer_details: Mapping | None = None) -> list[dict]:
    documents = {}
    source_rows = list(_rows(context.get("evidence_by_product")))
    for peer in _rows(peer_details):
        source_rows.extend(_rows(peer.get("evidence_by_product")))
    for row in source_rows:
        if row.get("product_id") not in products or not isinstance(row.get("id"), str):
            continue
        doc = _fields(row, _DOC_FIELDS)
        doc["sections"] = [_fields(section, {"locator", "text"})
                           for section in _rows(row.get("sections"))
                           if isinstance(section.get("locator"), str) and isinstance(section.get("text"), str)]
        documents.setdefault(doc["id"], doc)
    for citation in _previous_payload(previous).get("citations", []):
        if (not isinstance(citation, Mapping) or citation.get("kind") != "online_evidence"
                or citation.get("product_id") not in products
                or not isinstance(citation.get("evidence_id"), str)
                or not isinstance(citation.get("quote"), str)):
            continue
        identifier = citation["evidence_id"]
        if identifier in documents:
            doc = documents[identifier]
            if doc.get("content_status") == "retained_excerpt":
                section = {"locator": citation["locator"], "text": citation["quote"]}
                if not any(existing == section for existing in doc["sections"]):
                    # A saved excerpt is one exact adopted fragment, not a claim
                    # that the whole original section was archived. Keep distinct
                    # fragments even when the publisher locator is the same.
                    doc["sections"].append(section)
            continue
        doc = _fields(citation, _DOC_FIELDS - {"id"})
        doc.update(id=identifier, content_status="retained_excerpt",
                   sections=[{"locator": citation["locator"], "text": citation["quote"]}])
        documents[identifier] = doc  # Original URL and acquisition clocks remain unchanged.
    return list(documents.values())


def _select_documents(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """Send whole material when it fits, otherwise named relevant section windows.

    Selection is explicit coverage, not the UI's first-2400-character excerpt.
    The exact section locator and selected contiguous range remain visible.
    """
    total = sum(len(section["text"]) for doc in documents for section in doc["sections"])
    selected, coverage = [], []
    quota = MAX_DOCUMENT_CHARS // max(len(documents), 1)
    for doc in documents:
        sections = doc["sections"]
        budget = sum(len(row["text"]) for row in sections) if total <= MAX_DOCUMENT_CHARS else quota
        chosen, ranges = [], []
        # Keep document order in the actual prompt after choosing relevant sections.
        ranked = sorted(enumerate(sections), key=lambda item: (-len(_TOPIC.findall(item[1]["text"])), item[0]))
        for index, section in ranked:
            if budget <= 0:
                break
            text = section["text"]
            limit = min(len(text), budget, 12_000 if total > MAX_DOCUMENT_CHARS else len(text))
            if limit <= 0:
                continue
            start = 0
            if len(text) > limit:
                windows = range(0, len(text), max(limit // 2, 1))
                start = max(windows, key=lambda offset: len(_TOPIC.findall(text[offset:offset + limit])))
            excerpt = text[start:start + limit]
            chosen.append((index, {"locator": section["locator"], "text": excerpt}))
            ranges.append({"locator": section["locator"], "start": start, "end": start + len(excerpt),
                           "section_chars": len(text)})
            budget -= len(excerpt)
        chosen.sort(key=lambda item: item[0])
        selected.append({**_fields(doc, _DOC_FIELDS), "sections": [row for _, row in chosen]})
        used = sum(row["end"] - row["start"] for row in ranges)
        original = sum(len(row["text"]) for row in sections)
        coverage.append({"evidence_id": doc["id"], "content_status": doc.get("content_status"),
                         "selected_chars": used, "source_chars": original,
                         "selected_sections": ranges, "total_sections": len(sections),
                         "complete_supplied_text": used == original})
    return selected, coverage


def _records(detail: Mapping, context: Mapping, products: set) -> list[dict]:
    records = {}
    for rows in (detail.get("evidence_records"), detail.get("background_records"),
                 context.get("records_by_product")):
        for row in _rows(rows):
            if (row.get("family") != "research_thesis" and row.get("product_id") in products
                    and isinstance(row.get("id"), str)):
                records.setdefault(row["id"], _fields(row, _RECORD_FIELDS))
    return list(records.values())


def _retained_record_basis(previous, products: set) -> tuple[dict, list[dict]]:
    """Reuse only economic snapshots actually retained by a prior bound citation."""
    records, links = {}, []
    for citation in _previous_payload(previous).get("citations", []):
        if (not isinstance(citation, Mapping) or citation.get("kind") != "online_evidence"
                or citation.get("product_id") not in products
                or not all(isinstance(citation.get(key), str) for key in ("evidence_id", "locator", "quote"))):
            continue
        for row in _rows(citation.get("record_basis")):
            if (row.get("family") == "research_thesis" or row.get("product_id") != citation["product_id"]
                    or row.get("source_url") != citation.get("source_url") or not isinstance(row.get("id"), str)):
                continue
            records.setdefault(row["id"], _fields(row, _RECORD_FIELDS))
            links.append({"record_id": row["id"], "evidence_id": citation["evidence_id"],
                          "locator": citation["locator"], "quote": citation["quote"]})
    return records, links


def _record_prompt(record: Mapping) -> dict:
    result = deepcopy(dict(record))
    payload = _mapping(result.get("payload"))
    for key in ("closes", "observations"):
        if isinstance(payload.get(key), list) and len(payload[key]) > 3:
            payload[key + "_count"] = len(payload[key])
            payload["latest_" + key] = payload.pop(key)[-3:]
    return result


def _record_ids(value) -> set[str]:
    ids = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"record_id", "portfolio_record_id", "selected_record_id"} and isinstance(item, str):
                ids.add(item)
            else:
                ids.update(_record_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(_record_ids(item))
    return ids


def _row_observation(row: Mapping) -> Mapping:
    return _mapping(_mapping(row.get("inputs")).get(row.get("anchor"))
                    or row.get("current") or row.get("actual") or row)


def _metric_basis(row: Mapping, *, include_period: bool = True) -> str:
    metric = _mapping(_row_observation(row).get("metric"))
    basis = {key: metric.get(key) for key in (
        "name", "category", "security_id", "currency", "unit", "accounting_basis",
        "accounting_basis_detail", "share_basis", "split_basis", "eps_basis", "period_role",
        "comparison_basis", "cash_capex", "cash_flow_definition", "debt_definition", "debt_maturity_scope",
    )}
    if "inputs" in row:
        # Do not keep an older success when the current same-basis check is missing.
        basis = {key: row.get(key) for key in (
            "name", "metric_name", "unit", "input_currency", "accounting_basis",
            "accounting_basis_detail", "comparison_basis",
        )} | {"security_id": metric.get("security_id")}
    period = _mapping(row.get("period") or metric.get("period"))
    return bounded_json(basis | ({"period_basis": period.get("basis")} if include_period else {}))


def _latest_rows(rows) -> list[Mapping]:
    """Latest observation per actual economic/measurement basis, not last import."""
    selected = {}
    for row in _rows(rows):
        observation = _row_observation(row)
        period = _mapping(row.get("period") or _mapping(observation.get("metric")).get("period"))
        order = (period.get("end") or period.get("start") or "", period.get("start") or "",
                 observation.get("available_at") or "", observation.get("record_id") or "")
        key = _metric_basis(row)
        if key not in selected or order > selected[key][0]:
            selected[key] = order, row
    return [row for _, row in sorted(selected.values(), key=lambda item: item[0])]


def _latest_financial_checks(rows) -> list[Mapping]:
    rows = _rows(rows)
    latest_balances = {}
    for row in rows:
        if row.get("name") == "working_capital_to_revenue":
            key = _metric_basis(row, include_period=False)
            end = row["period"]["end"]
            latest_balances[key] = max(latest_balances.get(key, ""), end)
    # No-revenue placeholders have instant basis, not another sales window.
    # Keep all explicit revenue windows for the newest balance, including gaps,
    # rather than an old success or an old instant placeholder beside them.
    return _latest_rows([row for row in rows if row.get("name") != "working_capital_to_revenue"
                         or row["period"]["end"] == latest_balances[_metric_basis(row, include_period=False)]])


def _observation_prompt(value) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    result = _fields(value, {"record_id", "locator", "metric", "value", "low", "high",
                             "published_at", "publication_precision", "available_at", "revision_kind"})
    original = _mapping(value.get("original_release"))
    if original:
        result["original_release"] = _fields(original, {"record_id", "published_at", "publication_precision"})
    return result


def _comparison_prompt(row: Mapping) -> dict:
    nested = {"current", "prior", "actual", "prior_guidance"}
    result = _public({key: value for key, value in row.items() if key not in nested})
    result.update({key: _observation_prompt(row[key]) for key in nested if key in row})
    return result


def _financial_check_prompt(row: Mapping) -> dict:
    return {**_public({key: value for key, value in row.items() if key != "inputs"}),
            "inputs": {key: _observation_prompt(value) for key, value in _mapping(row.get("inputs")).items()}}


def _security_prompt(row: Mapping) -> dict:
    result = _public({key: value for key, value in row.items()
                      if key not in {"mapping", "latest_close", "ttm_diluted_eps"}})
    mapping = _mapping(row.get("mapping"))
    result["mapping"] = _fields(mapping, {"record_id", "locator", "payload"}) if mapping else None
    close = _mapping(row.get("latest_close"))
    result["latest_close"] = _fields(close, {"record_id", "locator", "date", "close", "currency",
                                           "price_basis", "available_at"}) if close else None
    ttm = _mapping(row.get("ttm_diluted_eps"))
    result["ttm_diluted_eps"] = {
        **_public({key: value for key, value in ttm.items() if key != "quarters"}),
        "quarters": [_observation_prompt(item) for item in _rows(ttm.get("quarters"))],
    }
    return result


def _analysis_prompt(analysis) -> dict:
    """Current decision basis, not every historical observation nested repeatedly."""
    analysis = _mapping(analysis)
    result = _public({key: value for key, value in analysis.items() if key != "issuers"})
    result["issuers"] = []
    for issuer in _rows(analysis.get("issuers")):
        compact = _fields(issuer, {"issuer_id", "issuer_name", "status", "disclosed_exposure"})
        actuals = issuer.get("financial_observations") or issuer.get("quarter_actuals")
        selections = {
            "latest_financials": [_observation_prompt(row) for row in _latest_rows(actuals)],
            "eps_yoy": [_comparison_prompt(row) for row in _latest_rows(issuer.get("eps_yoy"))],
            "revenue_vs_prior_guidance": [_comparison_prompt(row) for row in _latest_rows(issuer.get("revenue_vs_prior_guidance"))],
            "latest_guidance": [_observation_prompt(row) for row in _latest_rows(issuer.get("latest_guidance"))],
            "capex": [_observation_prompt(row) for row in _latest_rows(issuer.get("capex"))],
            **{key: [_financial_check_prompt(row) for row in _latest_financial_checks(issuer.get(key))]
               for key in ("cash_flow_checks", "balance_sheet_checks", "working_capital_checks")},
        }
        compact.update(selections)
        compact["securities"] = [_security_prompt(row) for row in _rows(issuer.get("securities"))]
        compact["history_coverage"] = {
            "policy": "latest_per_economic_basis_plus_actual_ttm_quarters; older_history_not_fully_sent_or_reviewed",
            "available_counts": {"financial_observations": len(_rows(actuals)),
                                 **{key: len(_rows(issuer.get(key))) for key in selections if key != "latest_financials"}},
            "selected_counts": {key: len(rows) for key, rows in selections.items()},
        }
        result["issuers"].append(compact)
    return result


def _structured_record_prompts(records, analyses, previous):
    prior_ids = _record_ids(_previous_payload(previous))
    needed = _record_ids(analyses) | prior_ids
    metric_locators = {}

    def collect(value):
        if isinstance(value, Mapping):
            identifier, metric = value.get("record_id"), _mapping(value.get("metric"))
            if isinstance(identifier, str) and isinstance(metric.get("locator"), str):
                metric_locators.setdefault(identifier, set()).add(metric["locator"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(analyses)
    selected = [row for row in records if row.get("family") != "issuer_fundamentals" or row["id"] in needed]
    prompts = []
    omitted_metrics = 0
    for row in selected:
        prompt = _record_prompt(row)
        payload = _mapping(prompt.get("payload"))
        locators = metric_locators.get(row["id"])
        if row["id"] not in prior_ids and locators and isinstance(payload.get("metrics"), list):
            metrics = payload["metrics"]
            payload["metrics"] = [metric for metric in metrics if metric.get("locator") in locators]
            omitted_metrics += len(metrics) - len(payload["metrics"])
            payload["metric_selection"] = {"available": len(metrics), "selected": len(payload["metrics"])}
        prompts.append(prompt)
    return prompts, {
        "available_records": len(records), "selected_records": len(selected),
        "omitted_older_issuer_records": len(records) - len(selected), "omitted_metrics": omitted_metrics,
        "policy": "current_selected_financial_basis_and_prior_research_references; complete_records_retained_only_for_program_validation",
    }


def _calculation_prompt(row: Mapping) -> dict:
    # sections already contains the exact result to quote; do not repeat result
    # a second time or serialize all inputs into that quote text as well.
    return _public({key: value for key, value in row.items() if key != "result"})


def _calculations(detail: Mapping, records: list[dict], product: Mapping, started: str) -> list[dict]:
    """Expose only current necessary arithmetic with compact real input bases."""
    available = {row["id"] for row in records}
    result = []
    analysis = _analysis_prompt(detail.get("issuer_analysis"))

    def add(method, inputs, outputs):
        references = sorted(_record_ids(inputs))
        if not references or not set(references) <= available:
            return  # A calculation cannot cite only the available subset of its actual inputs.
        inputs, outputs = _public(inputs), _public(outputs)
        # The contract keeps these compact adopted bases, not whole analysis objects.
        bounded_json(inputs, max_bytes=32 * 1024)
        bounded_json(outputs, max_bytes=16 * 1024)
        identifier = f"calc:{product['id']}:{len(result) + 1}"
        text = bounded_json(outputs)  # Exact compact computed output, not duplicated input history.
        result.append({"id": identifier, "product_id": product["id"], "method": method,
                       "inputs": inputs, "result": outputs, "available_at": started,
                       "record_ids": references, "input_evidence_ids": [],
                       "sections": [{"locator": "program_result", "text": text}]})

    for issuer in _rows(analysis.get("issuers")):
        for row in _rows(issuer.get("eps_yoy")):
            add("issuer_fundamentals.eps_yoy", {"current": row.get("current"), "prior": row.get("prior")},
                _fields(row, {"period", "security_id", "status", "reason", "current_eps", "prior_eps",
                              "change", "absolute_change"}))
        for row in _rows(issuer.get("revenue_vs_prior_guidance")):
            add("issuer_fundamentals.revenue_vs_prior_guidance",
                {"actual": row.get("actual"), "prior_guidance": row.get("prior_guidance")},
                {key: _public(value) for key, value in row.items() if key not in {"actual", "prior_guidance"}})
        for key in ("cash_flow_checks", "balance_sheet_checks", "working_capital_checks"):
            for row in _rows(issuer.get(key)):
                add(f"issuer_fundamentals.{row['name']}", row["inputs"],
                    {"issuer_id": issuer["issuer_id"], **{key: value for key, value in row.items()
                                                        if key not in {"inputs", "anchor"}}})
        for row in _rows(issuer.get("securities")):
            add("issuer_fundamentals.ttm_and_pe",
                _fields(row, {"mapping", "latest_close", "ttm_diluted_eps"}),
                _fields(row, {"security_id", "symbol", "currency", "share_basis", "pe_ttm",
                              "earnings_yield_ttm", "valuation_status", "valuation_reason"}))
    return result


def _verified_detail(detail, records, product, started):
    """Re-run the existing pure arithmetic against the actual supplied records."""
    relevant = [row for row in records if row.get("product_id") == product["id"]]
    if not relevant:
        return detail
    from .investment_profile import build_investment_profile
    from .issuer_fundamentals import build_issuer_analysis
    profile = build_investment_profile(relevant, as_of=started)
    analysis = build_issuer_analysis(relevant, profile=profile, as_of=started)
    return {**detail, "investment_profile": profile, "issuer_analysis": analysis}


def _bundle(product, context, detail, comparison, previous, started):
    products = _candidate_ids(product, comparison)
    peer_details = _mapping(detail.get("peer_details"))
    products.update(identifier for identifier in peer_details if isinstance(identifier, str))
    retained_records, retained_links = _retained_record_basis(previous, products)
    peers = {}
    records = {**retained_records, **{row["id"]: row for row in _records(detail, context, products)}}
    detail = _verified_detail(detail, list(records.values()), product, started)
    calculations = _calculations(detail, list(records.values()), product, started)
    links = [_fields(row, {"record_id", "capture_id", "locator"})
             for row in _rows(detail.get("evidence_links"))]
    for identifier, peer in peer_details.items():
        if not isinstance(peer, Mapping) or not isinstance(identifier, str):
            continue
        peer_product = _product(peer.get("product"))
        if identifier != peer_product["id"] or identifier == product["id"]:
            raise ValidationError("研究候选的产品身份与索引不符")
        products.add(identifier)
        peer_records = {key: row for key, row in retained_records.items() if row["product_id"] == identifier}
        peer_records.update({row["id"]: row for row in _records(peer, context, {identifier})})
        peer_records = list(peer_records.values())
        peer = _verified_detail(peer, peer_records, peer_product, started)
        peers[identifier] = {"product": peer_product, **_fields(peer, _DETAIL_FIELDS),
                             "issuer_analysis": _analysis_prompt(peer.get("issuer_analysis"))}
        records.update({row["id"]: row for row in peer_records})
        calculations.extend(_calculations(peer, peer_records, peer_product, started))
        links.extend(_fields(row, {"record_id", "capture_id", "locator"})
                     for row in _rows(peer.get("evidence_links")))
    # Context may also contain newly acquired candidate records.
    records.update({row["id"]: row for row in _records({}, context, products)})
    docs, coverage = _select_documents(_source_documents(context, previous, products, peer_details))
    record_links = [
        {"record_id": record_id, **link}
        for rows in _mapping(context.get("record_evidence_by_product")).values()
        for record_id, references in _mapping(rows).items()
        for link in _rows(references)
    ]
    record_links.extend(retained_links)
    evidence = {"documents": docs, "records": list(records.values()), "calculations": calculations,
                "record_links": record_links}
    analysis_prompt = _analysis_prompt(detail.get("issuer_analysis"))
    structured_records, record_coverage = _structured_record_prompts(
        list(records.values()), [analysis_prompt, *[row["issuer_analysis"] for row in peers.values()]], previous,
    )
    package = {
        "as_of": started, "product": product, "document_coverage": coverage,
        "structured_records": structured_records, "structured_record_coverage": record_coverage,
        "documents": docs, "archived_evidence_links": links,
        "program_calculations": [_calculation_prompt(row) for row in calculations],
        "public_detail": _fields(detail, _DETAIL_FIELDS), "issuer_analysis": analysis_prompt,
        "peer_details": peers, "public_context": _fields(context, _CONTEXT_FIELDS),
        "comparison": _public(comparison), "previous_research": _public(previous),
    }
    return package, evidence, links, products


_SYSTEM = """你负责用公开材料生成能让金融新手读懂的基金研究。只输出简洁结论和可核查理由，不输出内部思维过程。
输出单个JSON对象：直接输出与 RESEARCH_OUTPUT_EXAMPLE 完全同形的schema_version=2研究；或需要补证时输出
{"evidence_requests":{"evidence_ids":["已有附件id"],"questions":["待检索问题"]},"research":完整研究或null}。
只使用本次材料；API不带联网，不能声称自己已检索。附件请求交程序公开取材，最多两轮。无回调时保留具体缺口。
引用online_evidence只能用documents中实际id、locator和逐字quote；title_only、unparsed和消息标题仅供发现，不能确证事实。
retained_excerpt是旧研究保留的摘录，原时间不是今天；不能说刚取得全文。archived_evidence仅用所供record_id/capture_id/locator。
标准化record是结构化资料，不是发布者原文。程序计算用calculation引用program_calculations中id/locator/quote，不自造计算、URL或输入。
有cash_flow_checks/balance_sheet_checks/working_capital_checks时，实际解释利润是否转成经营现金、明确定义的现金资本开支后剩余多少、同日现金能否覆盖披露有息债务、应收和存货相对收入占用多少；就近写实际期间。亏损下的OCF/净利润不能机械评优劣，缺失不是零。OCF-capex不是FCFF；defined_basis_bridge只是原文定义桥接，保留FCF与OCF各自会计口径，差额原因仍须原文证据。期末余额/期间收入不是周转天数，不自行年化或累计EPS相减。
事实数字须对应引用中的原数值；为便于阅读四舍五入时，在每个近似数前写“约”，按该数字实际展示的小数位四舍五入，不改写原始引用。
严格区分factual/effective/plan/assumption/model_judgment/unresolved。没有共识数据就不能说超市场预期；公司指引不等于市场共识。
先分析宏观/政策/产业事实如何影响收入、利润率、现金流、估值、利率或汇率，再解释通过哪期披露持仓影响基金、覆盖多少和未覆盖什么。
经营改善与现在买入划算分开；给最强反方、价格已反映的实际证据或具体未知。没有依据不编合理PE、目标价、概率、最佳仓位。
分别论证长期基金质量fund_quality、当前入场entry_case、new_buy/held/addition，不由posture推导动作。
每个action_cases.reason须回答“为什么选这个动作而不是另一个”：新买比较现在投入与暂留现金；持有比较继续占用这笔资金与卖出、换成已提供的实际候选；追加比较进一步集中风险与保留现有投入。给出本基金具体的关键证据、取舍及足以改变该动作的可观察事实，不要复制总摘要或列一串风险了事。
持有也在承担从今天开始的风险与机会成本；“没有发现必须卖出的证据”“长期看好”不能单独充当持有优于退出的理由。持有但不追加须说明两者为何不同（实际费用、资金边际集中或其他有据差异）；材料不能支持这种区别时，不为凑三个结论硬写不同动作。
价格未知不等于价格贵，风险未消失不等于不该买；说明现有证据能支持什么判断、哪项未知可能反转比较。不要用“等合理估值”“利好兑现后再看”等无法观察的循环条件替代分析；确实缺决定性依据就说清缺哪项，不能把无依据的hold包装成完整结论。
所有历史权重、利率、盈利及价格数字在理由里就近带实际观察/披露日期；重取旧研究不让这些数字变成今天的数据。
基金任务、经理行为、费用与操作条件也影响结论；年度净值内费用不能再扣一遍。候选只能用实际comparison中的产品，比较授权/底层/费用和换基代价。不能确认费用或价格时说明比较方向和待核对项，不虚构节省金额、目标价或优势概率。
用日常中文直接说明处理这笔钱的得失；共有事实写在assessment，不复制到三个动作。通用免责声明、引用检查口径和“不是涨跌概率”等话不充当投资理由。
旧研究不是确证事实；若接续旧研究须明确哪条理由增强/减弱/失效/未变及如何改变各行动。非关键缺口不自动否定有据结论。
输出不要个人份额、成本、预算；这些不在研究职责内。product/origin/research_date/note_id/revision及URL时间由程序绑定，不要伪造。
示例只有格式没有证据或预设建议；必须替换所有示例内容，用本次事实得出你自己的判断。"""


def _decode(text: str) -> tuple[dict | None, dict | None]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValidationError("模型输出不是单个有效JSON对象") from exc
    if not isinstance(value, dict):
        raise ValidationError("模型输出必须是JSON对象")
    requested = value.pop("evidence_requests", None)
    payload = value.get("research") if "research" in value else value
    if payload is not None and not isinstance(payload, dict):
        raise ValidationError("模型research字段必须是研究对象或null")
    if requested is not None:
        if not isinstance(requested, Mapping):
            raise ValidationError("模型补证请求须为对象")
        for key in ("evidence_ids", "questions"):
            rows = requested.get(key, [])
            if not isinstance(rows, list) or len(rows) > 16 or any(not isinstance(item, str) or len(item) > 1000 for item in rows):
                raise ValidationError("模型补证请求须为有界文本列表")
        requested = {key: list(requested.get(key, [])) for key in ("evidence_ids", "questions")}
        if not any(requested.values()):
            requested = None
    return payload, requested


def _metadata(payload, *, product, previous, run_id, model, generated_at, protocol):
    value = deepcopy(payload)
    prior = _previous_payload(previous)
    value.update(product=product, note_id=prior.get("note_id") or "fund-investment-case",
                 research_date=timestamp(generated_at).date().isoformat(),
                 origin={"provider": protocol, "model": model, "run_id": run_id,
                         "generated_at": generated_at, "human_reviewed": False,
                         "validation": "programmatic_citations_not_investment_verification"},
                 revision={"id": run_id, "supersedes": _mapping(previous).get("record_id")})
    # A new model response may not use the contract's previously-bound reload path.
    for index, citation in enumerate(value.get("citations", [])):
        if isinstance(citation, Mapping):
            keys = {"id", "kind", "locator", "claim", "record_id", "capture_id"} if citation.get("kind") == "archived_evidence" else {
                "id", "kind", "evidence_id", "locator", "quote", "claim"}
            value["citations"][index] = {key: item for key, item in citation.items() if key in keys}
    return value


def _number_tokens(text: str):
    for match in _NUMBER.finditer(text):
        raw = match.group().replace(",", "").strip()
        multiplier = Decimal(1)
        for suffix, scale in (("billion", "1e9"), ("million", "1e6"), ("亿", "1e8"), ("万", "1e4"), ("%", ".01"), ("％", ".01")):
            if raw.lower().endswith(suffix):
                multiplier, raw = Decimal(scale), raw[:-len(suffix)].strip()
                break
        value = Decimal(raw)
        precision = Decimal(1).scaleb(value.as_tuple().exponent) * multiplier
        approximate = bool(re.search(r"(?:约|四舍五入(?:后)?(?:为)?|approximately|about|roughly)\s*$",
                                     text[:match.start()], re.I))
        yield value * multiplier, precision, approximate


def _numbers_supported(claim: str, basis: str) -> bool:
    available = {value for value, _, _ in _number_tokens(basis)}
    for value, precision, approximate in _number_tokens(claim):
        if value in available:
            continue
        # An explicit approximation must be the source rounded to the stated
        # display precision. This is not a free percentage/error tolerance.
        if not approximate or not available:
            return False
        with localcontext() as context:
            context.prec = max([context.prec, *(len(source.as_tuple().digits)
                               + max(0, source.as_tuple().exponent - precision.as_tuple().exponent)
                               + 1 for source in available)])
            if not any(source.quantize(precision, rounding=ROUND_HALF_UP) == value for source in available):
                return False
    return True


def _check_bindings(payload: Mapping, evidence: Mapping, links: list, products: set, generated_at: str):
    record_map = {row["id"]: row for row in evidence["records"]}
    link_set = {(row.get("record_id"), row.get("capture_id"), row.get("locator")) for row in links}
    citation_map = {row["id"]: row for row in payload["citations"]}
    if any(row["product_id"] not in products for row in payload["alternatives"]):
        raise ValidationError("模型使用了未提供的替代候选")
    for citation in citation_map.values():
        if citation["kind"] == "archived_evidence":
            row = record_map.get(citation["record_id"])
            marker = (citation["record_id"], citation["capture_id"], citation["locator"])
            if row is None or marker not in link_set or timestamp(row["available_at"]) > timestamp(generated_at):
                raise ValidationError("模型存档引用不对应本次提供的真实记录和定位")
        if citation["kind"] == "calculation" and not set(citation["record_ids"]) <= record_map.keys():
            raise ValidationError("程序计算引用缺少实际输入记录")
    # Check numbers asserted as facts against their adopted source/actual program
    # basis. Interpretation, scenarios and labelled assumptions are not arithmetic.
    for fact in payload["facts"]:
        if fact["status"] not in {"factual", "effective", "plan"}:
            continue
        basis = []
        for identifier in fact["citation_ids"]:
            citation = citation_map[identifier]
            if citation["kind"] == "archived_evidence":
                record = record_map[citation["record_id"]]
                basis.append(bounded_json({key: record.get(key) for key in
                                           ("payload", "report_date", "published_at")}))
            else:
                # The model-authored citation.claim is not evidence for itself.
                basis.append(bounded_json({key: citation.get(key) for key in
                                           ("quote", "inputs", "result", "published_at")}))
        if not _numbers_supported(fact["claim"], " ".join(basis)):
            raise ValidationError(f"事实{fact['id']}包含未绑定到其采用引用的数值；请区分事实与假设")


def _progress(callback, stage: str, message: str, run_id: str, round_number: int):
    if callback:
        callback({"stage": stage, "message": message, "run_id": run_id, "round": round_number})


def run_research(*, product, context, detail, comparison=None, previous=None, settings,
                 api_key=None, request=None, clock=now_iso, on_progress=None,
                 acquire_more=None, environ=None) -> dict:
    """One initial API request and at most two evidence-acquisition follow-ups.

    ``request(method, url, *, headers, json, timeout)`` is httpx-compatible.
    ``acquire_more({evidence_ids, questions, round, product_id})`` returns a public
    context (``evidence_by_product``/``records_by_product``) or a document list.
    It alone owns public acquisition; an API response cannot execute network I/O.
    ``environ`` permits deterministic tests without inspecting the real environment.
    """
    run_id, started = uuid.uuid4().hex, timestamp(clock()).isoformat()
    result = {"status": "failed", "run_id": run_id, "started_at": started, "completed_at": None,
              "model": None, "usage": {}, "research": None, "human_reviewed": False,
              "validation": {"status": "not_run", "errors": [], "warnings": []},
              "coverage": {}, "evidence_requests": [], "errors": []}
    round_number = 0

    def finish(status, code=None, message=None):
        result["status"], result["completed_at"] = status, timestamp(clock()).isoformat()
        if code:
            result["errors"].append({"code": code, "message": message})
        _progress(on_progress, "complete" if result["research"] else "failed",
                  message or "研究已生成；引用校验不代表投资结论已验证", run_id, round_number)
        return result

    try:
        config = normalize_research_api_settings(settings)
        result["model"] = config["model"] or None
        if not is_configured(config, api_key=api_key, environ=environ):
            return finish("unconfigured", "unconfigured", "请明确配置研究API地址、模型及服务端Key；本次未请求API")
        public_product = _product(product)
        detail, context = _mapping(detail), _mapping(context)
        if previous is None:
            previous = detail.get("research_thesis")
        _progress(on_progress, "preparing", "整理本次公开正文、旧披露及程序计算", run_id, 0)
        package, evidence, links, products = _bundle(public_product, context, detail, comparison, previous, started)
        example = research_output_example(product=public_product, as_of=started)
        payload, generated_at, acquisition_gap = None, None, False
        for round_number in range(1, MAX_EVIDENCE_ROUNDS + 2):
            prompt = {"RESEARCH_OUTPUT_EXAMPLE": example, "INPUT_PACKAGE": package,
                      "follow_up_results": result["evidence_requests"],
                      "remaining_acquisition_rounds": MAX_EVIDENCE_ROUNDS + 1 - round_number}
            content = bounded_json(prompt, max_bytes=MAX_INPUT_BYTES)
            _progress(on_progress, "requesting", "调用用户已配置的研究API", run_id, round_number)
            response = complete_research(settings=config, messages=[{"role": "user", "content": content}],
                                         system=_SYSTEM, api_key=api_key, request=request, environ=environ)
            generated_at = timestamp(clock()).isoformat()  # Actual end of this API call.
            result["model"] = response.model or config["model"]
            for key, amount in response.usage.items():
                result["usage"][key] = result["usage"].get(key, 0) + amount
            payload, requested = _decode(response.text)
            if requested is None:
                break
            log = {**requested, "round": round_number, "status": "not_acquired"}
            result["evidence_requests"].append(log)
            if acquire_more is None or round_number > MAX_EVIDENCE_ROUNDS:
                log["status"] = "no_acquisition_callback" if acquire_more is None else "round_limit_reached"
                acquisition_gap = True
                break
            _progress(on_progress, "acquiring", "通过公开取材回调核对模型提出的问题", run_id, round_number)
            try:
                more = acquire_more({**requested, "round": round_number, "product_id": public_product["id"]})
                if isinstance(more, list):
                    more = {"evidence_by_product": {public_product["id"]: more}}
                more = _mapping(more)
                new_docs = _source_documents(more, None, products)
                new_records = _records({}, more, products)
                merged = deepcopy(dict(context))
                for key in ("evidence_by_product", "records_by_product", "messages_by_product"):
                    indexed = merged.setdefault(key, {})
                    for identifier, rows in _mapping(more.get(key)).items():
                        known = {row["id"]: row for row in _rows(indexed.get(identifier))}
                        known.update({row["id"]: row for row in _rows(rows)})
                        indexed[identifier] = list(known.values())
                indexed = merged.setdefault("record_evidence_by_product", {})
                for identifier, rows in _mapping(more.get("record_evidence_by_product")).items():
                    indexed.setdefault(identifier, {}).update(_mapping(rows))
                context = merged
                material_at = timestamp(clock()).isoformat()
                package, evidence, links, products = _bundle(
                    public_product, context, detail, comparison, previous, material_at,
                )
                log.update(status="acquired" if new_docs or new_records else "unavailable",
                           document_count=len(new_docs), record_count=len(new_records))
                acquisition_gap |= log["status"] != "acquired"
            except Exception:
                log["status"], acquisition_gap = "acquisition_failed", True
        result["coverage"] = {"documents": package["document_coverage"], "record_count": len(evidence["records"]),
                              "calculation_count": len(evidence["calculations"]), "api_calls": round_number,
                              "candidate_product_ids": sorted(products - {public_product["id"]})}
        if not payload:
            return finish("partial" if result["evidence_requests"] else "failed", "no_research_payload",
                          "模型仅提出补证请求，未形成可用研究；未替换旧结论")
        _progress(on_progress, "validating", "核对结构、实际引用和事实数值", run_id, round_number)
        if acquisition_gap and isinstance(payload.get("assessment"), Mapping):
            payload = deepcopy(payload)
            missing = payload["assessment"].get("missing")
            if isinstance(missing, list):
                missing.append("本次提出的补证请求未全部取得材料；未把未执行的补查视为已经完成。")
        payload = _metadata(payload, product=public_product, previous=previous, run_id=run_id,
                            model=result["model"], generated_at=generated_at, protocol=config["protocol"])
        normalized = normalize_generated_research(payload, evidence=evidence, as_of=generated_at)
        _check_bindings(normalized, evidence, links, products, generated_at)
        result["research"] = normalized
        warnings = []
        if acquisition_gap:
            warnings.append("部分补证请求未取得材料；没有把未执行的补查说成已完成。")
        if any(not row["complete_supplied_text"] for row in package["document_coverage"]):
            warnings.append("本次采用有限相关段落，不代表已覆盖报告全部正文。")
        result["validation"] = {"status": "validated", "errors": [], "warnings": warnings}
        return finish("partial" if acquisition_gap else "generated")
    except ResearchAPIError as exc:
        return finish("unconfigured" if exc.code == "unconfigured" else "failed", exc.code, str(exc))
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        # Contract errors are short diagnostics, not raw response dumps.
        message = str(exc)[:300]
        if api_key:
            message = message.replace(str(api_key), "[Key omitted]")
        result["validation"] = {"status": "failed", "errors": [message], "warnings": []}
        return finish("failed", "research_validation_failed", message)

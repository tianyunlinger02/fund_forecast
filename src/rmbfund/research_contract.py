"""API research v2: adopted evidence, causal reasoning and distinct action cases.

Binding checks that an excerpt occurs at the supplied RAM locator; it does not
certify the model's interpretation or investment effectiveness. Only the adopted
excerpt/calculation is returned, never the source document or its full sections.
The v1 authored-note normalizer and content identity remain separate.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from .advice_evidence import (
    _RESEARCH_CHANNELS, _day, _enum, _list, _object, _research_citation, _sha256,
    _text, _unique_texts, bounded_json, normalize_record, normalize_source_url,
)
from .domain import ValidationError, timestamp

GENERATED_VALIDATION = "programmatic_citations_not_investment_verification"
QUOTE_VALIDATION = "quote_matches_supplied_section"
ACTION_CHOICES = {
    "new_buy": frozenset({"buy", "wait"}),
    "held": frozenset({"hold", "reduce", "redeem", "review"}),
    "addition": frozenset({"add", "wait"}),
}
_NO_BODY = frozenset({"title_only", "unparsed_document", "unavailable", "error", "empty"})
_BOUND_DOCUMENT = {
    "product_id", "source_url", "title", "published_at", "publication_precision",
    "observed_at", "available_at", "content_status", "validation",
}
_CITATION_INPUT = {"id", "kind", "evidence_id", "locator", "quote", "claim"}

# Prompt/test example, not a second JSON-schema framework. All content is
# illustrative; the runner replaces product, dates, origin and revision itself.
RESEARCH_OUTPUT_EXAMPLE = {
    "schema_version": 2,
    "product": {"id": "synthetic:fund:C", "symbol": "SYNTH", "name": "合成示例基金",
                "kind": "open_fund", "share_class": "C", "exchange": "OTC"},
    "note_id": "fund-investment-case", "research_date": "2026-10-27",
    "horizon": {"start": "2026-10-27", "end": "2027-10-27"}, "posture": "maintain",
    "origin": {"provider": "runner-supplied", "model": "runner-supplied",
               "run_id": "runner-supplied", "generated_at": "2026-10-27T12:00:00+08:00"},
    "facts": [{"id": "f1", "claim": "示例：仅采用所供原文能支持的事实，不填虚构数字。",
               "status": "factual", "effective_date": None, "issuer_ids": [],
               "channels": ["earnings"], "citation_ids": ["c1"]}],
    "citations": [{"id": "c1", "kind": "online_evidence", "evidence_id": "actual-document-id",
                   "locator": "actual-section-locator", "quote": "实际原文逐字短摘录",
                   "claim": "该段实际支持的主张"}],
    "assessment": {
        "summary": "示例：区分基金长期质量、当前价格吸引力和各行动依据。",
        "supporting": [{"reason": "示例：解释事实为什么支持保留。", "fact_ids": ["f1"]}],
        "counterevidence": [],
        "priced_in": {"status": "unknown", "reason": "缺少可核对的价格反映依据。"},
        "missing": ["当前完整持仓覆盖未知。"], "falsifiers": ["后续事实推翻原盈利依据。"],
        "change_conditions": ["取得价格或决定性反方后重新评价相应动作。"],
        "review_date": "2026-11-27",
    },
    "revision": {"id": "runner-supplied", "supersedes": None},
    "action_cases": {
        "new_buy": {"action": "wait", "reason": "格式占位：用本基金事实比较现在买入与保留现金的得失，说明选择及改判事件。", "fact_ids": ["f1"]},
        "held": {"action": "hold", "reason": "格式占位：比较继续占用这笔资金与退出、换入实际候选的机会和代价，说明为何选此动作。", "fact_ids": ["f1"]},
        "addition": {"action": "wait", "reason": "格式占位：解释新增投入的收益机会与边际风险；若与保留现有投入不同，说明有据差异。", "fact_ids": ["f1"]},
    },
    "drivers": [{"channel": "earnings", "mechanism": "解释需求/政策如何影响收入、利润或现金流。",
                 "fund_transmission": "说明哪期披露持仓承接影响、已覆盖和未覆盖什么。",
                 "fact_ids": ["f1"]}],
    "fund_quality": {"conclusion": "示例：保留长期质量判断。", "reason": "用合同、经理行为、持仓和费用解释。",
                     "fact_ids": ["f1"]},
    "entry_case": {"conclusion": "示例：不优先新买。", "reason": "价格及最强反方与经营质量分开。",
                   "fact_ids": ["f1"]},
    "alternatives": [],
    "key_unknowns": [{"question": "价格是否已反映利好？", "affects": ["new_buy", "addition"],
                      "critical": False}],
    "change_evaluation": {"summary": "首次研究；后续须解释理由增强、减弱、失效或未变。", "fact_ids": ["f1"]},
}


def research_output_example(*, product: Mapping | None = None, as_of: str | None = None) -> dict:
    """Return a detached concrete prompt example; it is not evidence or advice."""
    result = json.loads(bounded_json(RESEARCH_OUTPUT_EXAMPLE))
    if product is not None:
        result["product"] = {key: product[key] for key in result["product"]}
    if as_of is not None:
        point = timestamp(as_of)
        day = point.date()
        result["research_date"] = result["horizon"]["start"] = day.isoformat()
        result["horizon"]["end"] = (day + timedelta(days=365)).isoformat()
        result["assessment"]["review_date"] = (day + timedelta(days=30)).isoformat()
        result["origin"]["generated_at"] = point.isoformat()
    return result


def _rows(value: Any, label: str) -> list[Mapping]:
    if isinstance(value, Mapping):
        result = []
        for key, row in value.items():
            if not isinstance(row, Mapping) or row.get("id", key) != key:
                raise ValidationError(f"{label}标识与对象不匹配")
            result.append({"id": key, **row})
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(row, Mapping) for row in value):
            raise ValidationError(f"{label}必须包含对象")
        return list(value)
    raise ValidationError(f"{label}必须是列表或按标识索引的对象")


def _index(value: Any, label: str) -> dict[str, Mapping]:
    result = {}
    for row in _rows(value, label):
        identifier = _text(row.get("id"), f"{label}id", maximum=200)
        if identifier in result:
            raise ValidationError(f"{label}标识重复")
        result[identifier] = row
    return result


def _record_link_index(value: Any) -> dict[tuple[str, str, str], list[dict]]:
    if not isinstance(value, list):
        raise ValidationError("record_links必须是平铺引用映射列表")
    result = {}
    for row in value:
        item = _object(row, "精确经济引用映射",
                       required={"record_id", "evidence_id", "locator", "quote"})
        for key in ("record_id", "evidence_id"):
            item[key] = _text(item[key], key, maximum=200)
        item["locator"] = _text(item["locator"], "经济引用定位", maximum=1000)
        item["quote"] = _quote(item["quote"])
        marker = (item["record_id"], item["evidence_id"], item["locator"])
        result.setdefault(marker, []).append(item)
    return result


def _evidence_parts(evidence: Mapping | Sequence | None) -> tuple[dict, dict, dict, dict | None]:
    if evidence is None:
        return {}, {}, {}, None
    if isinstance(evidence, Mapping) and any(
        key in evidence for key in ("documents", "records", "calculations", "record_links")
    ):
        links = None
        if "record_links" in evidence:
            links = _record_link_index(evidence["record_links"])
        return (*tuple(_index(evidence.get(key, []), key)
                       for key in ("documents", "records", "calculations")), links)
    return _index(evidence, "documents"), {}, {}, None


def _publication(result: dict, available) -> None:
    precision = _enum(result["publication_precision"], {"unknown", "date", "timestamp"},
                      "在线引用披露时间精度")
    published = result["published_at"]
    if precision == "unknown":
        if published is not None:
            raise ValidationError("披露日期未知时不能补造日期")
    elif precision == "date":
        published = _day(published, "在线引用披露日")
        if published > available.date().isoformat():
            raise ValidationError("引用尚未公开披露")
    else:
        published = timestamp(published).isoformat()
        if timestamp(published) > available:
            raise ValidationError("引用尚未公开披露")
    result["published_at"] = published


def _quote(value: Any) -> str:
    if not isinstance(value, str) or any(ord(char) < 32 and char not in "\r\n\t" for char in value):
        raise ValidationError("引用摘录必须是原文短文本")
    return _text(" ".join(value.split()), "引用原文摘录", maximum=4000)


def _match_quote(document: Mapping, citation: dict) -> None:
    sections = document.get("sections")
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        raise ValidationError("在线引用缺少可定位正文，不能用标题替代")
    found = [row for row in sections if isinstance(row, Mapping)
             and row.get("locator") == citation["locator"]]
    if (not found or any(not isinstance(row.get("text"), str) for row in found)
            or (len(found) != 1 and document.get("content_status") != "retained_excerpt")):
        raise ValidationError("在线引用定位不存在或不唯一")
    if not any(citation["quote"] in " ".join(row["text"].split()) for row in found):
        raise ValidationError("引用摘录与对应定位的实际正文不匹配")


def _economic_record(identifier: str, records: dict, point, products: set) -> dict:
    row = records.get(identifier)
    if row is None:
        raise ValidationError("采用的经济记录不在本次实际资料包中")
    value = normalize_record({key: item for key, item in row.items() if key != "checked_at"})
    if (identifier != value["id"] or value["family"] == "research_thesis"
            or value["product_id"] not in products or timestamp(value["available_at"]) > point):
        raise ValidationError("采用记录须为生成时已可用的相关产品原始经济依据")
    return value


def _quote_overlap(document: Mapping, citation: Mapping, link: Mapping) -> bool:
    left, right = citation["quote"], link["quote"]
    if not (left in right or right in left):
        return False
    return any(left in text and right in text for text in (
        " ".join(row["text"].split()) for row in document["sections"]
        if row["locator"] == citation["locator"]
    ))


def _online_citation(value: Mapping, documents: dict, records: dict, record_links: dict | None,
                     point, products: set, *, bind: bool) -> dict:
    result = _object(value, "在线研究引用", required=_CITATION_INPUT - {"kind"},
                     optional={"kind", *_BOUND_DOCUMENT, "record_ids", "record_basis"})
    result["kind"] = "online_evidence"
    for key, maximum in (("id", 80), ("evidence_id", 200), ("locator", 1000), ("claim", 1000)):
        result[key] = _text(result[key], key, maximum=maximum)
    result["quote"] = _quote(result["quote"])
    document = documents.get(result["evidence_id"])
    if bind and document is None:
        raise ValidationError("在线引用evidence_id不在本次实际证据包中")
    if document is not None:
        _match_quote(document, result)
        for key in _BOUND_DOCUMENT - {"validation"}:
            if key not in document:
                raise ValidationError(f"在线证据缺少{key}")
            result[key] = document[key]
        result["record_basis"] = []
        if record_links is None:
            # Compatibility for older RAM packages: no exact record-to-section
            # map means the online citation remains valid but does not claim a
            # whole document's economic records were adopted.
            result["record_ids"] = []
        else:
            matching = []
            for (record_id, evidence_id, locator), links in record_links.items():
                if evidence_id != result["evidence_id"] or locator != result["locator"]:
                    continue
                if any(_quote_overlap(document, result, link) for link in links):
                    record = _economic_record(record_id, records, point, products)
                    if record["product_id"] != document["product_id"] or record["source_url"] != document["source_url"]:
                        raise ValidationError("精确经济引用映射的记录来源与在线文档不匹配")
                    matching.append(record_id)
            result["record_ids"] = sorted(set(matching))
            result["record_basis"] = [
                _economic_record(identifier, records, point, products)
                for identifier in result["record_ids"]
            ]
        result["validation"] = QUOTE_VALIDATION
    elif not _BOUND_DOCUMENT <= result.keys() or result["validation"] != QUOTE_VALIDATION:
        raise ValidationError("未绑定在线引用须提供RAM证据包；重载只接受已绑定的引用")
    result["product_id"] = _text(result["product_id"], "证据产品", maximum=80)
    if result["product_id"] not in products:
        raise ValidationError("引用属于其他产品，且未列为实际比较候选")
    result["source_url"] = normalize_source_url(result["source_url"])
    result["title"] = _text(result["title"], "证据标题", maximum=500)
    result["content_status"] = _text(result["content_status"], "正文状态", maximum=80)
    if result["content_status"] in _NO_BODY:
        raise ValidationError("只有标题或未解析文件不能成为在线正文依据")
    observed, available = timestamp(result["observed_at"]), timestamp(result["available_at"])
    if not observed <= available <= point:
        raise ValidationError("研究引用必须在实际生成时已经取得且可用")
    result.update(observed_at=observed.isoformat(), available_at=available.isoformat())
    _publication(result, available)
    result["record_ids"] = sorted(_sha256(item, "引用经济记录")
                                  for item in _unique_texts(result.get("record_ids", []),
                                                            "引用经济记录", maximum=64))
    if "record_basis" in result:
        basis = _index(_list(result["record_basis"], "采用经济基线", 64), "采用经济基线")
        if set(basis) != set(result["record_ids"]):
            raise ValidationError("采用经济基线须与实际绑定record_ids一致")
        result["record_basis"] = [_economic_record(identifier, basis, point, {result["product_id"]})
                                  for identifier in result["record_ids"]]
        if any(row["source_url"] != result["source_url"] for row in result["record_basis"]):
            raise ValidationError("采用经济基线来源须匹配实际引用文档")
    return result


def _calculation_citation(value: Mapping, calculations: dict, point, products: set,
                          *, bind: bool) -> dict:
    fields = {"product_id", "method", "inputs", "result", "available_at",
              "input_evidence_ids", "record_ids", "validation"}
    result = _object(value, "程序计算引用", required=_CITATION_INPUT,
                     optional=fields)
    for key, maximum in (("id", 80), ("evidence_id", 200), ("locator", 1000), ("claim", 1000)):
        result[key] = _text(result[key], key, maximum=maximum)
    result["quote"] = _quote(result["quote"])
    calculation = calculations.get(result["evidence_id"])
    if bind and calculation is None:
        raise ValidationError("计算引用不在本次实际计算依据中")
    if calculation is not None:
        _match_quote(calculation, result)
        for key in fields - {"validation", "record_ids", "input_evidence_ids"}:
            if key not in calculation:
                raise ValidationError(f"计算依据缺少{key}")
            result[key] = calculation[key]
        for key in ("record_ids", "input_evidence_ids"):
            result[key] = calculation.get(key, [])
        result["validation"] = "program_calculation"
    elif not fields <= result.keys() or result["validation"] != "program_calculation":
        raise ValidationError("程序计算引用必须先从本次计算包绑定")
    if result["product_id"] not in products:
        raise ValidationError("计算依据属于其他产品")
    result["method"] = _text(result["method"], "计算方法", maximum=2000)
    if not isinstance(result["inputs"], (Mapping, list)) or not result["inputs"]:
        raise ValidationError("程序计算必须保留实际采用的输入依据")
    result["inputs"] = json.loads(bounded_json(result["inputs"], max_bytes=32 * 1024))
    result["result"] = json.loads(bounded_json(result["result"], max_bytes=16 * 1024))
    result["available_at"] = timestamp(result["available_at"]).isoformat()
    if timestamp(result["available_at"]) > point:
        raise ValidationError("计算结果尚未实际可用")
    result["input_evidence_ids"] = sorted(_unique_texts(result["input_evidence_ids"],
                                                         "计算输入证据", maximum=64))
    result["record_ids"] = sorted(_sha256(item, "计算输入记录") for item in
                                  _unique_texts(result["record_ids"], "计算输入记录", maximum=64))
    if not result["input_evidence_ids"] and not result["record_ids"]:
        raise ValidationError("程序计算必须保留原始输入证据引用")
    return result


def _fact_refs(value: Any, facts: set, label: str, *, empty: bool = False) -> list[str]:
    references = _unique_texts(value, label, maximum=64, text_maximum=80)
    if (not references and not empty) or not set(references) <= facts:
        raise ValidationError(f"{label}必须引用本研究实际存在的事实")
    return sorted(references)


def _reason(value: Any, facts: set, label: str, *, extra: set = frozenset(),
            optional: set = frozenset(), empty: bool = False) -> dict:
    result = _object(value, label, required={"reason", "fact_ids"} | extra, optional=optional)
    result["reason"] = _text(result["reason"], label, maximum=4000)
    result["fact_ids"] = _fact_refs(result["fact_ids"], facts, label, empty=empty)
    return result


def normalize_generated_research(
    payload: Any, *, evidence: Mapping | Sequence | None = None, as_of: str | None = None,
) -> dict:
    """Bind model references to RAM evidence, or validate a previously bound v2.

    ``evidence`` accepts ``{documents, records, calculations}``, a document
    sequence, or an id-to-document mapping. Model citations contain only an id,
    kind (online_evidence/calculation), evidence_id, locator, quote and claim.
    URL and acquisition/publication clocks always come from the supplied bundle.
    ``as_of`` is the caller's real clock, not a model-provided research date.
    """
    bounded_json(payload)
    result = _object(payload, "API研究v2", required={
        "schema_version", "product", "note_id", "research_date", "horizon", "posture", "origin",
        "facts", "citations", "assessment", "revision", "action_cases", "drivers", "fund_quality",
        "entry_case", "alternatives", "key_unknowns", "change_evaluation",
    })
    if type(result["schema_version"]) is not int or result["schema_version"] != 2:
        raise ValidationError("API研究须显式使用payload schema_version=2")
    product = _object(result["product"], "研究产品", required={
        "id", "symbol", "name", "kind", "share_class", "exchange",
    })
    for key in product:
        product[key] = _text(product[key], f"产品{key}", maximum=200 if key in {"name", "symbol"} else 80,
                             empty=key in {"share_class", "exchange"})
    product["kind"] = _enum(product["kind"], {"etf", "open_fund", "monitor_only"}, "产品类型")
    result["product"] = product
    result["note_id"] = _text(result["note_id"], "研究论题标识", maximum=200)
    origin = _object(result["origin"], "研究生成来源", required={
        "provider", "model", "run_id", "generated_at",
    }, optional={"human_reviewed", "validation"})
    for key in ("provider", "model", "run_id"):
        origin[key] = _text(origin[key], f"生成{key}", maximum=200)
    if origin.get("human_reviewed", False) is not False:
        raise ValidationError("API生成研究不能标为人工复核")
    origin.update(human_reviewed=False, validation=GENERATED_VALIDATION)
    generated = timestamp(origin["generated_at"])
    origin["generated_at"] = generated.isoformat()
    if as_of is not None and generated > timestamp(as_of):
        raise ValidationError("研究生成时间不能晚于实际处理时钟")
    result["origin"] = origin
    result["research_date"] = _day(result["research_date"], "研究日")
    if result["research_date"] != generated.date().isoformat():
        raise ValidationError("API研究日期必须是实际生成日期，不能补造历史研究")
    horizon = _object(result["horizon"], "研究期限", required={"start", "end"})
    horizon = {key: _day(value, f"研究期限{key}") for key, value in horizon.items()}
    if not horizon["start"] <= result["research_date"] <= horizon["end"]:
        raise ValidationError("研究日期必须位于明确的研究期限内")
    if (date.fromisoformat(horizon["end"]) - date.fromisoformat(horizon["start"])).days > 3660:
        raise ValidationError("研究期限不能超过十年")
    result["horizon"] = horizon
    result["posture"] = _enum(result["posture"], {"accumulate", "maintain", "trim", "exit", "unresolved"},
                               "兼容研究姿态")
    alternatives = _list(result["alternatives"], "真实候选比较", 16)
    products = {product["id"]} | {_text(row.get("product_id"), "候选产品id", maximum=80)
                                for row in alternatives if isinstance(row, Mapping)}
    documents, records, calculations, record_links = _evidence_parts(evidence)
    citations = []
    for row in _list(result["citations"], "研究引用", 64):
        if not isinstance(row, Mapping):
            raise ValidationError("研究引用必须是对象")
        kind = row.get("kind", "online_evidence")
        if kind == "online_evidence":
            citation = _online_citation(row, documents, records, record_links, generated, products,
                                        bind=evidence is not None)
        elif kind == "calculation":
            citation = _calculation_citation(row, calculations, generated, products,
                                             bind=evidence is not None)
            if evidence is not None:
                for identifier in citation["record_ids"]:
                    _economic_record(identifier, records, generated, products)
        elif kind == "archived_evidence":
            citation = _research_citation(row)
        else:
            raise ValidationError("API研究采用online_evidence、程序计算或实际存档引用，不能冒称人工摘录")
        citations.append(citation)
    citation_ids = {row["id"] for row in citations}
    if len(citation_ids) != len(citations):
        raise ValidationError("研究引用标识不能重复")
    adopted_documents = {row["evidence_id"] for row in citations if row["kind"] == "online_evidence"}
    for citation in citations:
        if citation["kind"] == "calculation" and not set(citation["input_evidence_ids"]) <= adopted_documents:
            raise ValidationError("计算输入原文必须同时保留实际采用的引用摘录")
    result["citations"] = sorted(citations, key=lambda row: row["id"])
    facts = []
    for row in _list(result["facts"], "研究事实与假设", 96):
        fact = _object(row, "事实与假设", required={
            "id", "claim", "status", "effective_date", "issuer_ids", "channels", "citation_ids",
        })
        fact["id"] = _text(fact["id"], "事实id", maximum=80)
        fact["claim"] = _text(fact["claim"], "事实主张", maximum=4000)
        fact["status"] = _enum(fact["status"], {
            "factual", "effective", "plan", "unresolved", "assumption", "model_judgment",
        }, "事实/计划/假设/模型判断")
        if fact["effective_date"] is not None:
            fact["effective_date"] = _day(fact["effective_date"], "事实生效日")
            if fact["status"] in {"factual", "effective"} and fact["effective_date"] > result["research_date"]:
                raise ValidationError("未来计划不能当作已发生事实")
        elif fact["status"] == "effective":
            raise ValidationError("已生效事实须有实际生效日")
        fact["issuer_ids"] = sorted(_unique_texts(fact["issuer_ids"], "发行人", maximum=32,
                                                  text_maximum=300))
        channels = _unique_texts(fact["channels"], "传导渠道", maximum=8)
        if not channels:
            raise ValidationError("事实须说明至少一种实际传导渠道")
        fact["channels"] = sorted(_enum(key, _RESEARCH_CHANNELS, "传导渠道") for key in channels)
        fact["citation_ids"] = _fact_refs(fact["citation_ids"], citation_ids, "事实引用")
        facts.append(fact)
    fact_ids = {row["id"] for row in facts}
    if len(fact_ids) != len(facts) or not facts:
        raise ValidationError("研究须有非重复的事实或明确假设")
    result["facts"] = sorted(facts, key=lambda row: row["id"])
    assessment = _object(result["assessment"], "研究判断", required={
        "summary", "supporting", "counterevidence", "priced_in", "missing", "falsifiers",
        "change_conditions", "review_date",
    })
    assessment["summary"] = _text(assessment["summary"], "结论", maximum=4000)
    for key in ("supporting", "counterevidence"):
        assessment[key] = [_reason(row, fact_ids, key) for row in _list(assessment[key], key, 32)]
    priced = _object(assessment["priced_in"], "价格反映程度", required={"status", "reason"},
                     optional={"fact_ids"})
    priced["status"] = _enum(priced["status"], {
        "unknown", "contested", "not_reflected", "partly_reflected", "largely_reflected",
    }, "有依据的定性价格反映程度")
    priced["reason"] = _text(priced["reason"], "价格反映依据", maximum=4000)
    priced["fact_ids"] = _fact_refs(priced.get("fact_ids", []), fact_ids, "价格反映事实",
                                    empty=priced["status"] == "unknown")
    assessment["priced_in"] = priced
    for key in ("missing", "falsifiers", "change_conditions"):
        assessment[key] = _unique_texts(assessment[key], key, maximum=32, text_maximum=2000)
    if not assessment["falsifiers"] or not assessment["change_conditions"]:
        raise ValidationError("研究须说明什么会推翻理由或改变动作")
    assessment["review_date"] = _day(assessment["review_date"], "复核日")
    if not result["research_date"] <= assessment["review_date"] <= horizon["end"]:
        raise ValidationError("复核日须位于研究日至期限结束之间")
    result["assessment"] = assessment
    cases = _object(result["action_cases"], "分情况行动", required=set(ACTION_CHOICES))
    actual_facts = {row["id"] for row in facts if row["status"] in {"factual", "effective", "plan"}}
    for key, choices in ACTION_CHOICES.items():
        case = _reason(cases[key], fact_ids, f"{key}行动理由", extra={"action"},
                       optional={"summary", "next_step", "comparison"})
        case["action"] = _enum(case["action"], choices, f"{key}行动")
        if case["action"] not in {"wait", "review"} and not set(case["fact_ids"]) & actual_facts:
            raise ValidationError("实际买入/持有/增减/赎回须有带引用的事实或明确计划，不能只靠模型假设")
        for field in ("summary", "next_step"):
            if field in case:
                case[field] = _text(case[field], f"{key}{field}", maximum=500)
        if "comparison" in case:
            comparison = _reason(case["comparison"], fact_ids, f"{key}主要替代取舍", extra={"action"})
            comparison["action"] = _enum(comparison["action"],
                {"buy", "wait", "hold", "add", "reduce", "redeem"}, "主要替代动作")
            case["comparison"] = comparison
        cases[key] = case
    result["action_cases"] = cases
    drivers = []
    for row in _list(result["drivers"], "影响链", 24):
        driver = _object(row, "影响链", required={"channel", "mechanism", "fact_ids"},
                         optional={"fund_transmission"})
        driver["channel"] = _enum(driver["channel"], _RESEARCH_CHANNELS, "影响渠道")
        driver["mechanism"] = _text(driver["mechanism"], "经济影响链", maximum=4000)
        driver["fact_ids"] = _fact_refs(driver["fact_ids"], fact_ids, "影响链事实")
        if "fund_transmission" in driver:
            driver["fund_transmission"] = _text(driver["fund_transmission"], "披露持仓传导", maximum=4000)
        drivers.append(driver)
    if not drivers:
        raise ValidationError("研究须说明事实如何影响盈利/估值/汇率并传到基金")
    result["drivers"] = drivers
    for key in ("fund_quality", "entry_case"):
        section = _reason(result[key], fact_ids, key, extra={"conclusion"})
        section["conclusion"] = _text(section["conclusion"], key, maximum=2000)
        result[key] = section
    result["alternatives"] = []
    for row in alternatives:
        alternative = _reason(row, fact_ids, "实际替代选择", extra={"product_id"})
        alternative["product_id"] = _text(alternative["product_id"], "候选产品id", maximum=80)
        if alternative["product_id"] == product["id"]:
            raise ValidationError("替代选择不能冒充另一只相同产品")
        result["alternatives"].append(alternative)
    unknowns = []
    for row in _list(result["key_unknowns"], "关键未知", 32):
        unknown = _object(row, "具体未知", required={"question", "affects", "critical"})
        unknown["question"] = _text(unknown["question"], "具体未知", maximum=2000)
        unknown["affects"] = sorted(_unique_texts(unknown["affects"], "受影响行动", maximum=3))
        if not set(unknown["affects"]) <= ACTION_CHOICES.keys() or not unknown["affects"]:
            raise ValidationError("未知须说明影响哪种行动")
        if not isinstance(unknown["critical"], bool):
            raise ValidationError("是否决定性未知须明确布尔值")
        unknowns.append(unknown)
    result["key_unknowns"] = unknowns
    change = _object(result["change_evaluation"], "理由变化", required={"summary", "fact_ids"})
    change["summary"] = _text(change["summary"], "理由变化说明", maximum=4000)
    change["fact_ids"] = _fact_refs(change["fact_ids"], fact_ids, "变化依据", empty=True)
    result["change_evaluation"] = change
    revision = _object(result["revision"], "研究修订", required={"id", "supersedes"})
    revision["id"] = _text(revision["id"], "修订id", maximum=200)
    if revision["supersedes"] is not None:
        revision["supersedes"] = _sha256(revision["supersedes"], "被接续研究记录")
        if not change["fact_ids"]:
            raise ValidationError("接续研究须用实际引用解释理由增强、减弱、失效或未变")
    result["revision"] = revision
    bounded_json(result)
    return result

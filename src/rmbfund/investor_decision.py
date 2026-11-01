"""Fund-investor synthesis of reviewed research and explicit personal constraints.

This does not fit a model, score news, select internal stocks or alter a forecast.
The research posture is an authored conclusion with citations, not a probability.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal

from .current_online import (
    _eligible, _point, _projected_records, current_execution_conditions, snapshot_online_context,
)
from .domain import ValidationError, timestamp
from .research_contract import normalize_generated_research
from .research_thesis import research_explained_record_ids

POLICY_VERSION = "fund_buyer_action_cases_v3"


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _branch(action: str, label: str, reasons: list[str]) -> dict:
    return {"action": action, "label": label, "reasons": list(reasons), "constraint_reasons": []}


def _constrain(branch: dict, action: str, label: str, reason: str) -> None:
    branch.update(action=action, label=label)
    if reason not in branch["constraint_reasons"]:
        branch["constraint_reasons"].append(reason)
    if reason not in branch["reasons"]:
        branch["reasons"].insert(0, reason)


_CHANNELS = {"earnings": "盈利", "valuation": "估值", "fx": "汇率", "rates": "利率",
             "liquidity": "流动性", "operations": "申赎", "mandate": "基金投向", "other": "其他关系",
             "fund_disclosure": "基金披露", "policy": "政策"}


def _structured_reason(row: Mapping) -> str:
    payload = _mapping(row.get("payload"))
    family = row.get("family")
    if family == "operations":
        states = {"open": "开放", "closed": "暂停", "limited": "限额", "unknown": "未知"}
        detail = (f"申购{states.get(payload.get('subscription_state'), '未知')}、"
                  f"赎回{states.get(payload.get('redemption_state'), '未知')}")
    elif family == "management":
        detail = "经理名单：" + "、".join(item["name"] for item in payload.get("managers", []))
    elif family == "portfolio":
        denominator = {"nav": "净资产", "equity_assets": "股票资产", "total_assets": "总资产"}.get(
            payload.get("denominator"), "原披露口径")
        detail = (f"披露{len(payload.get('holdings', []))}项持仓，"
                  f"权重按{denominator}计算，不是今天的实时持仓")
    elif family == "fees":
        detail = "公开费率：" + "、".join(
            f"{label}{payload[key]}" for key, label in (("management_rate", "年管理费率"),
            ("custody_rate", "年托管费率"), ("sales_service_rate", "年销售服务费率"),
            ("subscription_rate", "申购费率")) if payload.get(key) is not None)
    elif family == "mandate":
        detail = f"基金投向与合同基准：{_mapping(payload.get('benchmark')).get('name', '未注明')}"
    elif family == "events":
        detail = "；".join(item["description"] for item in payload.get("events", []))
    elif family == "issuer_fundamentals":
        kind = {"financial_release": "财报或经营指引", "price_history": "市场价格",
                "security_mapping": "证券与发行人对应关系"}.get(payload.get("kind"), "财务资料")
        detail = f"{payload.get('issuer_name', '相关发行人')}的{kind}"
    else:
        detail = "公开结构化资料已更新，仍需分别核对盈利、估值或汇率含义"
    return f"本次在线资料（{row.get('report_date', '原报告日')}）：{detail}。"


def _online_basis(context: Mapping, product_id: str | None,
                  payload: Mapping | None = None) -> tuple[list[dict], list[str], list[dict]]:
    evidence, reasons = [], []
    payload = _mapping(payload)
    citations = payload.get("citations", []) if payload.get("schema_version") == 2 else []
    adopted_records = {identifier for row in citations for identifier in row.get("record_ids", [])}
    adopted_records.update(row["record_id"] for row in citations if row["kind"] == "archived_evidence")
    adopted_documents = {row["evidence_id"] for row in citations if row["kind"] == "online_evidence"}
    explained = research_explained_record_ids(payload)
    signals = deepcopy(_mapping(context.get("review_signals_by_product")).get(product_id, []))
    for signal in signals:
        if (signal.get("current_record_id") in explained
                and (not signal.get("available_at") or timestamp(signal["available_at"])
                     <= timestamp(payload["origin"]["generated_at"]))):
            signal.update(status="addressed_by_generated_research",
                          interpretation_status="ai_generated",
                          reason="本次AI研究已实际引用并解释这项经济依据变化；程序引用核对不是人工复核。")
    point = _point(context)
    for row in _projected_records(context, product_id):
        if row.get("product_id") != product_id or not _eligible(row, point):
            continue
        item = {key: deepcopy(row[key]) for key in (
            "id", "family", "source_url", "report_date", "published_at", "observed_at", "available_at",
        ) if key in row}
        changed = any(signal.get("current_record_id") == row.get("id") for signal in signals)
        adopted = row.get("id") in adopted_records
        item.update(
            basis_relationship="adopted_in_generated_research" if adopted else
                               "basis_change_requires_review" if changed else
                               "execution_condition" if row.get("family") == "operations" else
                               "current_structured_fact",
            interpretation_status="ai_generated" if adopted else "unreviewed",
            reason=_structured_reason(row) + ("本次AI研究已采用并解释；程序核对不是人工复核。" if adopted else
                                            "用于对应净值以外的披露或执行计算；不是新编的涨跌概率。"),
        )
        item["payload"] = deepcopy(row.get("payload", {}))
        evidence.append(item)
        reasons.append(item["reason"])
    for row in _mapping(context.get("messages_by_product")).get(product_id, []):
        item = deepcopy(dict(row))
        channels = "、".join(_CHANNELS.get(key, str(key)) for key in row.get("channels", []))
        relation = row.get("relevance_reason") or "来源将该条目关联到本基金，尚需核对实际传导关系"
        title = row.get("title") or "未注明标题的消息"
        content = row.get("content_status")
        adopted = row.get("evidence_id", row.get("id")) in adopted_documents
        if adopted:
            explanation = "本次AI研究已引用实际定位摘录并解释传导；程序核对引用不证明投资判断正确。"
        elif content in {"title_only", "unparsed_document"}:
            explanation = "只取得标题或附件入口，尚未核对正文，不据标题改变买卖动作。"
        else:
            explanation = "已取得在线摘录，但尚未判断它支持还是反对原论证，也未核对价格已反映多少。"
        if row.get("event_status") == "plan":
            explanation += "这是计划，不当作已经实现的盈利。"
        elif row.get("event_status") == "unresolved":
            explanation += "事件内容仍有未确认部分。"
        reason = f"本次消息《{title}》：{relation}。" + (f"可能涉及{channels}。" if channels else "") + explanation
        item.update(basis_relationship="adopted_in_generated_research" if adopted else "uninterpreted",
                    interpretation_status="ai_generated" if adopted else "unreviewed", reason=reason)
        evidence.append(item)
        reasons.append(reason)
    reasons.extend(signal["reason"] for signal in signals if signal.get("reason"))
    if evidence and not reasons:
        reasons.append("本次在线结构化资料已用于当前披露或执行条件；原作者的研究姿态没有被自动改写。")
    return evidence, list(dict.fromkeys(reasons)), signals


def _generated_branch(payload: Mapping, key: str) -> dict:
    case = payload["action_cases"][key]
    labels = {"buy": "有条件新买", "wait": "暂不新买" if key == "new_buy" else "暂不追加",
              "hold": "继续持有", "add": "具备追加依据，先核对个人资金条件",
              "reduce": "倾向减少这只基金", "redeem": "倾向赎回这只基金",
              "review": "保留原计划，先核对已有持仓理由"}
    result = _branch(case["action"], labels[case["action"]], [case["reason"]])
    result.update(
        research_action=case["action"], research_reason=case["reason"],
        fact_ids=list(case["fact_ids"]),
        citation_ids=sorted({identifier for fact in payload["facts"] if fact["id"] in case["fact_ids"]
                             for identifier in fact["citation_ids"]}),
    )
    unknowns = [row["question"] for row in payload["key_unknowns"]
                if row["critical"] and key in row["affects"]]
    if unknowns:
        action, label = (("review", "已有持仓的决定性未知须先核对") if key == "held" else
                         ("wait", "决定性条件未明确，暂不新增资金"))
        for question in unknowns:
            _constrain(result, action, label, question)
        result["needed_review"] = unknowns
    return result


def build_investor_decision(
    *, product_id: str | None, holding_status: str = "unknown",
    research_thesis: Mapping | None = None, fund_comparison: Mapping | None = None,
    portfolio_context: Mapping | None = None, sizing_plan: Mapping | None = None,
    quantitative_decision: Mapping | None = None, forecast_status: str = "absent",
    execution_conditions: Mapping | None = None, online_context: Mapping | None = None,
    cash_use_fit: Mapping | None = None, preserve_research_cases: bool = False,
) -> dict:
    """Separate new buying, retention and additions; preserve disagreement.

    Missing evidence means review, never an invented sale. A personal reduction
    can arise only from an explicitly supplied loss budget and coherent scenario.
    Neither cost-to-break-even nor an arbitrary concentration cutoff is used.
    """
    if holding_status not in {"held", "flat", "unknown"}:
        raise ValidationError("持有状态须为held、flat或unknown")
    thesis = _mapping(research_thesis)
    payload = _mapping(thesis.get("payload"))
    if payload and _mapping(payload.get("product")).get("id") != product_id:
        raise ValidationError("研究论证与当前基金身份不匹配")
    if payload.get("schema_version") == 2:
        payload = normalize_generated_research(payload)
    assessment = _mapping(payload.get("assessment"))
    review = list(thesis.get("needed_review") or [])
    online = snapshot_online_context(online_context, [product_id]) if online_context is not None else {}
    online_evidence, online_reasons, online_signals = _online_basis(online, product_id, payload)
    review.extend(row["reason"] for row in online_signals
                  if row.get("status") == "review_required" and row.get("reason"))
    posture = payload.get("posture", "unresolved")
    generated = payload.get("schema_version") == 2 and "action_cases" in payload
    supported = (bool(payload.get("facts") and payload.get("citations") and assessment.get("supporting"))
                 if not generated else
                 bool(payload.get("facts") and payload.get("citations") and payload.get("action_cases")))
    usable = thesis.get("status") == "available" and supported and (
        not review or (preserve_research_cases and generated)) and (
        True if generated else posture != "unresolved")
    missing = list(assessment.get("missing") or [])
    if not payload:
        missing.append("尚无带日期和引用的基金买卖论证；短期净值向上不能代替买入理由。")
    elif thesis.get("status") == "contested":
        review.append("存在尚未解决的研究分歧，需要复核后再采用新买姿态。")
    elif posture == "unresolved" and not generated:
        review.append("研究尚未形成足够明确的买入或退出依据。")
    elif not supported:
        review.append("研究姿态没有完整的事实、引用与论证支持。")

    if not usable:
        held = _branch("review", "尚未形成可采用的持仓判断", [])
        flat = _branch("wait", "尚未形成可采用的新买判断", [])
        addition = _branch("wait", "尚未形成可采用的追加判断", [])
    elif generated:
        flat = _generated_branch(payload, "new_buy")
        held = _generated_branch(payload, "held")
        addition = _generated_branch(payload, "addition")
    else:
        # A posture alone is not three separately argued investment decisions.
        held = _branch("review", "旧研究未单独论证持有或卖出", [])
        flat = _branch("wait", "旧研究未单独论证现在新买", [])
        addition = _branch("wait", "旧研究未单独论证追加", [])

    sizing = _mapping(sizing_plan)
    reduction = _mapping(sizing.get("reduction"))
    # The sizing module certifies feasibility; an unknown amount cannot force sale.
    minimum = reduction.get("minimum_gross_redemption", reduction.get("minimum_reduction"))
    if minimum is not None and Decimal(str(minimum)) > 0:
        reason = "在你明确的损失预算和同一压力假设下，现有投入已超出预算；减少额见金额方案。"
        if held["action"] != "redeem":
            _constrain(held, "reduce", "按个人风险预算减少投入", reason)
        _constrain(flat, "wait", "先处理已有风险，再考虑新买", reason)
        _constrain(addition, "wait", "目前没有追加容量", reason)
    whole = _mapping(sizing.get("whole_fund"))
    capacities = [whole.get(key) for key in ("gross_cash_cap", "cash_constraint_cap")]
    capacities.extend(_mapping(whole.get("constraints")).values())
    zero_capacity = any(value is not None and Decimal(str(value)) == 0 for value in capacities)
    if zero_capacity:
        reason = "明确资金或损失约束下，可新增金额上限为0。"
        _constrain(flat, "wait", "当前约束下不新增资金", reason)
        _constrain(addition, "wait", "当前约束下不追加", reason)

    if preserve_research_cases:
        for branch in (flat, held, addition):
            branch["economic_action"] = branch["action"]
    conditions = current_execution_conditions(online, product_id, execution_conditions)
    execution_notes = list(conditions.get("reasons") or [])
    if conditions.get("subscription") == "closed":
        reason = "当前公开申购状态为暂停，不能执行新增投入。"
        _constrain(flat, "blocked", "暂停申购，暂不能执行新买", reason)
        _constrain(addition, "blocked", "暂停申购，暂不能执行追加", reason)
    if conditions.get("redemption") == "closed" and held["action"] in {"reduce", "redeem"}:
        execution_notes.append("研究有减少或退出依据，但当前披露为暂停赎回；不能假定今天可成交。")

    cash_fit = _mapping(cash_use_fit)
    if (cash_fit.get("status") == "not_suitable_for_new_money"
            and cash_fit.get("near_term_required") == "yes"):
        reason = cash_fit.get("reason") or "你明确这笔钱近期要用，不宜继续增加需要跨境申赎等待的投入。"
        for branch, label in ((flat, "这笔钱近期要用，暂不新买"),
                              (addition, "先留出近期用钱资金，暂不追加")):
            if preserve_research_cases and branch.get("economic_action") in {"buy", "add"}:
                branch["economic_action"] = "wait"
            _constrain(branch, "wait" if branch["action"] in {"buy", "add"} else branch["action"],
                       label if branch["action"] in {"buy", "add"} else branch["label"], str(reason))
        execution_notes.append("近期用钱需提前核对可赎份额和到账安排；这本身不是自动卖出已有持仓的理由。")

    quantitative = _mapping(quantitative_decision)
    conflicts = []
    if forecast_status == "ready":
        qbranches = _mapping(quantitative.get("branches"))
        qheld = _mapping(qbranches.get("held")).get("action")
        qflat = _mapping(qbranches.get("flat")).get("action")
        if held["action"] == "hold" and qheld in {"sell", "conditional_sell"}:
            conflicts.append("基金研究支持继续持有，但短期费用比较偏向退出。两者期限和依据不同，不平均成一个分数；新增资金仍单独评估。")
        if held["action"] in {"reduce", "redeem"} and qheld == "hold":
            conflicts.append("短期净值比较偏向持有，但研究或明确个人风险约束支持减少；短期反弹可能性不能修复失效的长期理由。")
        if (flat["action"] == "wait" and qflat in {"buy", "conditional_buy"}
                and (flat.get("research_action") or flat["constraint_reasons"])):
            conflicts.append("短期模型存在上涨空间，但尚不等于这只基金现在适合新增；价格、替代选择和个人风险容量仍需成立。")
        if flat["action"] == "buy" and qflat == "wait":
            conflicts.append("研究有条件支持买入，但短期净值比较不支持立即新增；不要把长期理由当成下一笔净值上涨保证。")
    else:
        conflicts.append("目前没有可采用的所选短期预测；研究倾向不含新编的上涨概率。")

    comparison = _mapping(fund_comparison)
    portfolio = _mapping(portfolio_context)
    result = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "product_id": product_id,
        "status": "review_required" if not usable else
                  "generated_research" if generated else "legacy_opinion",
        "action_basis": "independent_cases" if generated and usable else
                        "legacy_opinion" if payload and not generated else "unavailable",
        "research_summary": assessment.get("summary"),
        "research_date": payload.get("research_date"),
        "holding_status": holding_status, "posture": posture,
        "research_record_id": thesis.get("record_id"), "horizon": deepcopy(payload.get("horizon")),
        "branches": {"held": held, "flat": flat, "addition": addition},
        "supporting": deepcopy(assessment.get("supporting") or []),
        "counterevidence": deepcopy(assessment.get("counterevidence") or []),
        "priced_in": deepcopy(assessment.get("priced_in") or {"status": "unknown", "reason": "未核对价格已反映多少。"}),
        "missing": missing, "needed_review": list(dict.fromkeys(review)),
        "falsifiers": list(assessment.get("falsifiers") or []),
        "change_conditions": list(assessment.get("change_conditions") or []),
        "quantitative_conflicts": conflicts, "forecast_status": forecast_status,
        "comparison_status": comparison.get("status", "missing"),
        "portfolio_status": portfolio.get("status", "missing"),
        "sizing_status": sizing.get("status", "missing"), "execution_notes": execution_notes,
        "amount_policy": "金额上限是条件风险容量，不是建议用满；不替用户改变基金内部持仓。",
        "probability_relationship": "generated research changes investor branches, never market distributions" if generated else
                                   "reviewed research changes investor branches, never market distributions",
    }
    if generated:
        result.update(research_schema_version=2, origin=deepcopy(payload["origin"]))
        for key in ("action_cases", "fund_quality", "entry_case", "drivers", "alternatives",
                    "key_unknowns", "change_evaluation"):
            result[key] = deepcopy(payload[key])
    if preserve_research_cases:
        result["execution_conditions"] = deepcopy(dict(conditions))
    if cash_use_fit is not None:
        result["cash_use_fit"] = deepcopy(dict(cash_fit))
    if online_context is not None:
        result.update(online_status=online.get("status"), online_checked_at=online.get("checked_at"),
                      online_evidence=online_evidence, online_reasons=online_reasons,
                      online_review_signals=online_signals)
    return result

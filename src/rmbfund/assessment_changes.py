"""Pure differences between supplied assessment snapshots; no rebuilding or IO.

``current`` and ``prior`` are assessment payloads, not save envelopes. Deltas
retain the two saved values and presentation text. ``user_inputs`` describes
simulation assumptions and optional personal inputs, not a market event;
``market_data`` describes adopted heads/data or explicit execution conditions.
An online fetch clock or message title is never an economic interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from .current_online import derive_review_signals


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _percent(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "未提供"
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return "未提供"
    return f"{parsed * 100:.2f}%" if math.isfinite(parsed) else "未提供"


def _is_buyer(payload: Mapping) -> bool:
    return payload.get("schema_version") == 7 and payload.get("mode") == "buyer_decision"


def _is_calendar(payload: Mapping) -> bool:
    return (_is_buyer(payload) or
            (payload.get("schema_version") == 6 and payload.get("mode") == "calendar_trade"))


def _scope(payload: Mapping) -> str:
    if _is_buyer(payload):
        return "schema7（基金首选、自然日计划、计划卖出后净额）"
    if _is_calendar(payload):
        return "schema6（自然日计划、计划卖出后净额）"
    if payload.get("schema_version") == 5 and payload.get("mode") == "standard_trade":
        return "schema5（净值笔数、期末账面值，不自动赎回）"
    return f"schema{payload.get('schema_version')}（旧版个人判断）"


def extract_assessment_changes(current: Mapping, prior: Mapping) -> list[dict[str, Any]]:
    """Return ordered, detached field deltas, or ``[]`` when none are visible.

    Each delta has ``group``, ``field`` (a payload path or related field group), ``before``,
    ``after``, ``text`` and ``level`` (``write`` or ``caption``). Research
    continuation additionally has ``explained``: only an explicit
    ``change_evaluation.summary`` whose ``revision.supersedes`` names the prior
    research record can explain that transition. Missing older fields remain
    missing; in particular an old trade branch is not a saved decision brief.

    Calendar-only comparisons retain the schema6 history's original wording.
    Schema7 also separates authored brief, personal inputs, adopted forecast
    data and neutral provenance changes for reuse on the current decision card.
    """
    changes: list[dict[str, Any]] = []
    buyer = _is_buyer(current) or _is_buyer(prior)

    def add(group, field, before, after, text, *, level="write", **extra):
        changes.append({"group": group, "field": field,
                        "before": deepcopy(before), "after": deepcopy(after),
                        "text": text, "level": level, **extra})

    def decision_changes(after: Mapping, before: Mapping, *, field: str, label: str):
        if after == before:
            return
        changed_branch = False
        for key, name in (("flat", "新买"), ("held", "持有"), ("addition", "追加")):
            old = _mapping(_mapping(before.get("branches")).get(key))
            new = _mapping(_mapping(after.get("branches")).get(key))
            if old == new:
                continue
            changed_branch = True
            path = f"{field}.branches.{key}"
            text = (f"{label} · {name}：{old.get('label') or '未记录'} → {new.get('label') or '未记录'}。"
                    if old.get("action") != new.get("action") or old.get("label") != new.get("label") else
                    f"{label} · {name}标签未变，保存的理由或比较数值有变化。")
            add("decision", path, old, new, text)
            if old.get("action") != new.get("action") and old.get("label") == new.get("label"):
                add("decision", path + ".action", old.get("action"), new.get("action"),
                    f"原保存动作：{old.get('action') or '未记录'} → {new.get('action') or '未记录'}。")
            old_reasons = [*_lines(old.get("reasons")), *_lines(old.get("constraint_reasons"))]
            for reason in [*_lines(new.get("reasons")), *_lines(new.get("constraint_reasons"))]:
                if reason not in old_reasons:
                    add("decision", path + ".reasons", old_reasons, reason, "本次保存的理由：" + reason)
            if new.get("research_reason") and new.get("research_reason") != old.get("research_reason"):
                add("research", path + ".research_reason", old.get("research_reason"), new["research_reason"],
                    "本次保存的研究理由：" + new["research_reason"])
        if not changed_branch:
            add("decision", field, before, after,
                f"{label}的原预测数值、费用比较或延迟情景有变化；没有因此生成新的买卖理由。")

    current_trade, prior_trade = _is_calendar(current), _is_calendar(prior)
    scenario, old_scenario = _mapping(current.get("trade_scenario")), _mapping(prior.get("trade_scenario"))
    if not current_trade or not prior_trade:
        add("provenance", "schema_version", prior.get("schema_version"), current.get("schema_version"),
            f"比较口径变化：{_scope(prior)} → {_scope(current)}。")
        add("provenance", "trade_scenario", prior.get("trade_scenario"), current.get("trade_scenario"),
            "新旧快照不按同一收益或费用口径直接比较。旧版未保存的自然日天数、截止日、确认日及未来分批赎回费均为未记录；"
            "不补成30天、今天日期、零费率或示例费用，也不把旧版账面金额改称卖出后净额。", level="caption")
    else:
        if buyer and (current.get("schema_version"), current.get("mode")) != (prior.get("schema_version"), prior.get("mode")):
            add("provenance", "schema_version", prior.get("schema_version"), current.get("schema_version"),
                f"比较口径变化：{_scope(prior)} → {_scope(current)}；只比较双方实际保存的共同自然日字段。")
        if not scenario or not old_scenario:
            add("provenance", "trade_scenario", prior.get("trade_scenario"), current.get("trade_scenario"),
                "至少一份快照没有保存自然日交易情景，不能重建缺失的计划、费用或动作。", level="caption")
        else:
            plan, old_plan = _mapping(scenario.get("plan")), _mapping(old_scenario.get("plan"))
            for key, label, suffix in (
                ("calendar_days", "自然日窗口", "天"), ("decision_at", "测算时刻", ""),
                ("decision_date", "测算日期", ""), ("deadline", "计划截止日", ""),
                ("application_date", "本次假定申请日", ""), ("sale_date", "期末假定卖出申请日", ""),
                ("entry_lag", "日程映射的净值延迟", "笔净值"), ("horizon", "日程映射的预测期限", "笔净值"),
                ("new_confirmation_date", "新份额假定确认日", ""),
            ):
                before, after = old_plan.get(key), plan.get(key)
                if before != after:
                    add("provenance" if key in {"decision_at", "decision_date"} else "user_inputs",
                        f"trade_scenario.plan.{key}", before, after,
                        f"{label}：{str(before) + suffix if before is not None else '未记录'} → "
                        f"{str(after) + suffix if after is not None else '未记录'}。")
            if plan != old_plan:
                add("user_inputs", "trade_scenario.plan", old_plan, plan,
                    "自然日与净值笔数的日期映射、日程依据或保存假设有变化；各次原计划分别保留，不按今天日历重排。", level="caption")
            for key, label, suffix in (("horizon", "实际采用的预测期限", "笔净值"),
                                       ("entry_lag", "实际采用的执行延迟", "笔净值"),
                                       ("base_amount", "标准单笔示例本金", "金额单位")):
                before, after = old_scenario.get(key), scenario.get(key)
                if before != after:
                    add("user_inputs", f"trade_scenario.{key}", before, after,
                        f"{label}：{str(before) + suffix if before is not None else '未记录'} → "
                        f"{str(after) + suffix if after is not None else '未记录'}。")
            if scenario.get("purchase") != old_scenario.get("purchase"):
                purchase, old_purchase = _mapping(scenario.get("purchase")), _mapping(old_scenario.get("purchase"))
                if not buyer:
                    add("user_inputs", "trade_scenario.purchase", old_scenario.get("purchase"), scenario.get("purchase"),
                        f"假设买入净值日：{old_purchase.get('date') or '未记录'} → {purchase.get('date') or '未记录'}；"
                        "过去净值区间的保存值有变化，不把该日期当成旧、新份额的确认日。")
                else:
                    if purchase.get("date") != old_purchase.get("date"):
                        add("user_inputs", "trade_scenario.purchase.date", old_purchase.get("date"), purchase.get("date"),
                            f"假设买入净值日：{old_purchase.get('date') or '未记录'} → {purchase.get('date') or '未记录'}；"
                            "这是历史买入情景调整，不冒充旧、新份额的真实确认日。")
                    data_keys = ("latest_date", "latest_nav")
                    if purchase.get("date") == old_purchase.get("date"):
                        data_keys += ("nav",)
                    before = {key: old_purchase[key] for key in data_keys if key in old_purchase}
                    after = {key: purchase[key] for key in data_keys if key in purchase}
                    if before != after:
                        add("market_data", "trade_scenario.purchase.nav_basis", before, after,
                            "采用的已公布净值日期或数值有变化；这是已发生的净值依据，不拼接成未来回报。")
                    add("decision", "trade_scenario.purchase", old_scenario.get("purchase"), scenario.get("purchase"),
                        "历史买入以来区间的保存值或适用状态有变化；不把它当作本窗口预测。", level="caption")
            fees, old_fees = _mapping(scenario.get("fees")), _mapping(old_scenario.get("fees"))
            for key, label in (("subscription_fee", "假设申购费"),
                               ("current_redemption_fee", "旧份额当前赎回费"),
                               ("future_old_redemption_fee", "旧份额计划卖出赎回费"),
                               ("future_new_redemption_fee", "新份额计划卖出赎回费")):
                if old_fees.get(key) != fees.get(key):
                    add("user_inputs", f"trade_scenario.fees.{key}", old_fees.get(key), fees.get(key),
                        f"{label}：{_percent(old_fees.get(key))} → {_percent(fees.get(key))}。")
            for key, label in (("old_confirmation_date", "旧份额模拟确认日起点"),
                               ("new_confirmation_date", "新份额模拟确认日起点")):
                if old_fees.get(key) != fees.get(key):
                    add("user_inputs", f"trade_scenario.fees.{key}", old_fees.get(key), fees.get(key),
                        f"{label}：{old_fees.get(key) or '未记录'} → {fees.get(key) or '未记录'}。")
            if old_fees != fees:
                add("user_inputs", "trade_scenario.fees", old_fees, fees,
                    "分批费用上下文有变化：包括各批次确认日起点、费龄、费率来源或当前费表未来不变假设；"
                    "详见各次原费用对象，不借用历史买入日替新份额免手续费。", level="caption")
            if old_scenario.get("end_value_basis") != scenario.get("end_value_basis"):
                add("provenance", "trade_scenario.end_value_basis", old_scenario.get("end_value_basis"), scenario.get("end_value_basis"),
                    f"金额比较口径：{old_scenario.get('end_value_basis') or '未记录'} → {scenario.get('end_value_basis') or '未记录'}。")
            decision, old_decision = _mapping(scenario.get("decision")), _mapping(old_scenario.get("decision"))
            if decision.get("primary_head") != old_decision.get("primary_head"):
                head, old_head = _mapping(decision.get("primary_head")), _mapping(old_decision.get("primary_head"))
                target_changed = any(head.get(key) != old_head.get(key) for key in ("horizon", "entry_lag"))
                only_reasons = ({key: value for key, value in head.items() if key != "reasons"}
                                == {key: value for key, value in old_head.items() if key != "reasons"})
                group = "user_inputs" if buyer and target_changed else "decision" if buyer and only_reasons else "market_data"
                text = "原保存的准确预测头有变化；只核对各次保存的期限、延迟及分布，不借用其他期限或重新拟合。"
                if buyer and target_changed:
                    text = "期限或执行延迟调整后，采用的准确预测头有变化；不同窗口的分布不当作同一窗口的行情改判。"
                elif buyer and only_reasons:
                    text = "准确预测头的解释文字有变化；保存的期限、延迟与预测数值未变，不因此宣称市场数据改变。"
                add(group, "trade_scenario.decision.primary_head", old_decision.get("primary_head"), decision.get("primary_head"), text)
            decision_changes(decision, old_decision, field="trade_scenario.decision", label="自然日计划交易判断")
            if scenario.get("rows") != old_scenario.get("rows"):
                add("decision", "trade_scenario.rows", old_scenario.get("rows"), scenario.get("rows"),
                    "原保存的计划卖出净额、范围或相对对照比较有变化；没有按当前费用重新计算。")
        if (current.get("policy_version") != prior.get("policy_version")
                or current.get("calculation") != prior.get("calculation")
                or current.get("forecast") != prior.get("forecast")):
            add("provenance", "calculation", {key: prior.get(key) for key in ("policy_version", "calculation", "forecast")},
                {key: current.get(key) for key in ("policy_version", "calculation", "forecast")},
                "原计算来源、预测依据或本地计算合同有变化；原版本、目标单位与完整合同按各次保存内容核对，不据此认定已实际发行。")
        if current.get("personal_conversion") != prior.get("personal_conversion"):
            add("user_inputs", "personal_conversion", prior.get("personal_conversion"), current.get("personal_conversion"),
                "可选个人金额换算有变化；它与标准单笔本金和市场预测分开保存。")

    investor, old_investor = _mapping(current.get("investor_decision")), _mapping(prior.get("investor_decision"))
    if investor and old_investor:
        decision_changes(investor, old_investor, field="investor_decision", label="基金研究判断")
    thesis, old_thesis = _mapping(current.get("research_thesis")), _mapping(prior.get("research_thesis"))
    if thesis != old_thesis:
        payload, old_payload = _mapping(thesis.get("payload")), _mapping(old_thesis.get("payload"))
        authored_changed = payload != old_payload or thesis.get("record_id") != old_thesis.get("record_id")
        if buyer and not authored_changed:
            add("provenance", "research_thesis", prior.get("research_thesis"), current.get("research_thesis"),
                "研究的取得状态或保存来源有变化，作者正文与研究记录未变；不因此宣称研究理由已被重新解释。")
        else:
            add("research", "research_thesis", prior.get("research_thesis"), current.get("research_thesis"),
                f"采用的研究版本或保存内容有变化：研究日 {old_payload.get('research_date') or '未记录'} → "
                f"{payload.get('research_date') or '未记录'}；记录 {old_thesis.get('record_id') or '未记录'} → {thesis.get('record_id') or '未记录'}。")
            explanation = _mapping(payload.get("change_evaluation")).get("summary")
            explained = bool(isinstance(explanation, str) and explanation.strip()
                             and bool(old_thesis.get("record_id"))
                             and _mapping(payload.get("revision")).get("supersedes") == old_thesis["record_id"])
            add("research", "research_thesis.payload.change_evaluation", _mapping(old_payload.get("change_evaluation")),
                _mapping(payload.get("change_evaluation")),
                "本次研究对上一版的解释：" + explanation if explained else
                "两次快照没有可对应的显式接续说明，不把版本不同自行解释成原理由已失效。",
                level="write" if explained else "caption", explained=explained)
    if current.get("online_context") != prior.get("online_context"):
        add("online_context", "online_context", prior.get("online_context"), current.get("online_context"),
            "两次保存采用的在线依据有变化；只回放各次已保存内容，不联网判断消息是否已经失效。")
        if buyer:
            add("online_context", "online_context", prior.get("online_context"), current.get("online_context"),
                "重新取得的时间、新标题或尚未解释的正文，不自动说明原买卖理由增强、减弱或失效。", level="caption")
    if buyer:
        _buyer_changes(current, prior, add)
    return changes


def _buyer_changes(current: Mapping, prior: Mapping, add) -> None:
    """Schema7 additions, without upgrading either supplied snapshot."""
    inputs, old_inputs = _mapping(current.get("inputs")), _mapping(prior.get("inputs"))
    for key, label in (("holding", "份额或成本"), ("budget", "拟投入金额")):
        if old_inputs.get(key) != inputs.get(key):
            add("user_inputs", f"inputs.{key}", old_inputs.get(key), inputs.get(key),
                f"{label}填写有变化；这改变个人金额解释，不代表基金本身发生了同样变化。")
    conditions = _mapping(_mapping(current.get("sizing_plan")).get("user_conditions"))
    old_conditions = _mapping(_mapping(prior.get("sizing_plan")).get("user_conditions"))
    if conditions != old_conditions:
        add("user_inputs", "sizing_plan.user_conditions", old_conditions, conditions,
            "保存的资金用途或金额条件有变化，具体值以各次原快照为准。")
    if current.get("sale_comparison") != prior.get("sale_comparison"):
        add("decision", "sale_comparison", prior.get("sale_comparison"), current.get("sale_comparison"),
            "少数卖出时点的保存比较或费用边界有变化；各日期仍独立与现在退出相比，不把跨期限结果合成最佳日概率。")
    forecast, old_forecast = _mapping(current.get("forecast")), _mapping(prior.get("forecast"))
    for key, label in (("anchor", "采用的预测净值起点"), ("input_hash", "预测输入标识"),
                       ("basis_hash", "预测数据依据标识"), ("factor_input_hash", "因素输入标识"),
                       ("features", "采用的预测特征"), ("factor_input_snapshot", "因素输入依据")):
        if forecast.get(key) != old_forecast.get(key):
            add("market_data", f"forecast.{key}", old_forecast.get(key), forecast.get(key),
                f"{label}有变化；核对各次实际采用的数据，不从计算或取得时间推断市场方向。")
    execution, old_execution = _mapping(current.get("execution_conditions")), _mapping(prior.get("execution_conditions"))
    keys = ("subscription", "redemption", "limit_amount", "channel_verified")
    before = {key: old_execution[key] for key in keys if key in old_execution}
    after = {key: execution[key] for key in keys if key in execution}
    if before != after:
        add("market_data", "execution_conditions", before, after,
            "公开申赎状态、限额或渠道核实条件有变化；这是实际执行条件，不把窗口内往返不可比较当作真实暂停申购。")
    online, old_online = _mapping(current.get("online_context")), _mapping(prior.get("online_context"))
    thesis, old_thesis = _mapping(current.get("research_thesis")), _mapping(prior.get("research_thesis"))
    product_id = _mapping(_mapping(thesis.get("payload")).get("product")).get("id")
    if product_id:
        old_records = _mapping(old_online.get("records_by_product")).get(product_id, [])
        records = [*old_records, *_mapping(online.get("records_by_product")).get(product_id, [])]
        signal_keys = ("kind", "product_id", "family", "citation_id", "previous_record_id", "current_record_id")
        previous_signals = {tuple(signal.get(key) for key in signal_keys)
                            for signal in derive_review_signals(old_thesis, old_records)}
        for signal in derive_review_signals(thesis, records):
            if tuple(signal.get(key) for key in signal_keys) in previous_signals:
                continue
            add("market_data", "online_context.review_signals", signal.get("previous_record_id"),
                signal.get("current_record_id"), signal["reason"] + "这是待解释的资料变化，不是自动改判。")

    brief, old_brief = _mapping(current.get("decision_brief")), _mapping(prior.get("decision_brief"))
    if not brief or not old_brief:
        if brief != old_brief:
            add("decision", "decision_brief", prior.get("decision_brief"), current.get("decision_brief"),
                "至少一份快照未保存当前首选摘要；不补写旧版的首选、备选取舍、反方或观察条件。", level="caption")
        return
    if brief.get("rationale") != old_brief.get("rationale"):
        add("research", "decision_brief.rationale", old_brief.get("rationale"), brief.get("rationale"),
            "采用的作者补充说明或其出处有变化；这是作者论证与裁决依据，不是新市场数据或新增模型概率。")
    if brief.get("research_record_id") != old_brief.get("research_record_id"):
        add("research", "decision_brief.research_record_id", old_brief.get("research_record_id"), brief.get("research_record_id"),
            "当前首选引用的研究记录有变化；是否解释了上一版，只按明确接续关系及研究变更说明核对。")
    for key, name in (("flat", "准备新买"), ("held", "已有基金与备用现金")):
        old = _mapping(_mapping(old_brief.get("cases")).get(key))
        new = _mapping(_mapping(brief.get("cases")).get(key))
        if old == new:
            continue
        path = f"decision_brief.cases.{key}"
        if not old or not new:
            add("decision", path, old, new, f"{name}：至少一份快照没有保存该情况的首选，不从原交易分支重建。")
            continue
        if (old.get("preferred_action"), old.get("label")) != (new.get("preferred_action"), new.get("label")):
            add("decision", path + ".preferred_action", old.get("preferred_action"), new.get("preferred_action"),
                f"{name}首选：{old.get('label') or '未记录'} → {new.get('label') or '未记录'}。")
            if old.get("label") == new.get("label"):
                add("decision", path + ".preferred_action", old.get("preferred_action"), new.get("preferred_action"),
                    f"原保存动作：{old.get('preferred_action') or '尚无唯一首选'} → {new.get('preferred_action') or '尚无唯一首选'}。")
        for field, label in (("reasons", "首选理由"), ("why_not", "主要备选的取舍与相对首选金额差"),
                             ("counterevidence", "最强反方"), ("checks", "下一步观察条件"),
                             ("conflict", "研究与本窗口比较的分歧处理")):
            if old.get(field) != new.get(field):
                add("research" if field in {"counterevidence", "checks", "conflict"} else "decision",
                    path + "." + field, old.get(field), new.get(field), f"{name}的{label}有变化；保留各次原文与事实引用，不自动重选。")
                if field == "reasons":
                    for reason in _lines(new.get(field)):
                        if reason not in _lines(old.get(field)):
                            add("decision", path + ".reasons", old.get(field), reason, "本次首选理由：" + reason)
        status_keys = ("research_action", "decision_status", "execution_status", "comparison_status")
        before = {field: old[field] for field in status_keys if field in old}
        after = {field: new[field] for field in status_keys if field in new}
        if before != after:
            add("decision", path + ".status", before, after,
                f"{name}的研究意见、判断状态、执行状态或情景可比性有变化；经济首选与本窗口能否完成往返分开核对。")
    if brief.get("plan_semantics") != old_brief.get("plan_semantics"):
        add("provenance", "decision_brief.plan_semantics", old_brief.get("plan_semantics"), brief.get("plan_semantics"),
            "首选摘要保存的窗口与卖出情景解释有变化；不把窗口截止日变成强制清仓或必须用款日。")

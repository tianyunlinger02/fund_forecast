"""Current and saved fund-buyer views; saved renderers use only supplied JSON."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal

import streamlit as st

from rmbfund.domain import ValidationError, decimal
from rmbfund.ui.formatting import lines, mapping, markdown_text, percent


_REVIEW_LABELS = {
    "no_research_thesis_available": "还没有经过复核的基金买卖论证。",
    "posture_unresolved": "研究尚未形成明确买卖倾向。",
    "review_date_elapsed": "已到这份研究预定的复核时间，需要重新检查依据。",
    "research_horizon_elapsed": "这份研究的适用期限已过。",
    "multiple_visible_thesis_heads": "存在未解决的研究版本分歧。",
    "superseded_record_not_available_as_of": "研究修订的前序依据在该时点不可用。",
}


def _write(value) -> None:
    if isinstance(value, str):
        prefix, reason = ("• ", value[2:]) if value.startswith("• ") else ("", value)
        value = prefix + _REVIEW_LABELS.get(reason, reason)
    st.markdown(markdown_text(value))


def _money(value) -> str:
    return f"¥{Decimal(str(value)):,.2f}" if value is not None else "未提供"


def _render_original_investor(assessment: Mapping) -> None:
    decision = mapping(assessment.get("investor_decision"))
    branches = mapping(decision.get("branches"))
    state = assessment.get("holding_status")
    st.caption("整只基金的投资判断 · 研究依据与个人适配")
    if state == "unknown":
        st.subheader("按买入和持有分别看")
        branch_labels = (("flat", "若准备新买"), ("held", "若已经持有"), ("addition", "若想追加"))
        for key, label in branch_labels:
            branch = mapping(branches.get(key))
            _write(f"{label}：{branch.get('label', '尚待复核')}")
            for reason in lines(branch.get("reasons"))[:4]:
                _write("• " + reason)
        if any(len(lines(mapping(branches.get(key)).get("reasons"))) > 4 for key, _ in branch_labels):
            with st.expander("全部分支理由（含在线资料影响）"):
                for key, label in branch_labels:
                    st.markdown(f"**{label}**")
                    for reason in lines(mapping(branches.get(key)).get("reasons")):
                        _write("• " + reason)
        st.caption("未填份额不等于空仓，不替你选择持有状态。")
    else:
        branch = mapping(branches.get("held" if state == "held" else "flat"))
        st.subheader(markdown_text(branch.get("label", "尚待复核")))
        for reason in lines(branch.get("reasons"))[:4]:
            _write(reason)
        if state == "held":
            _write(f"追加资金：{mapping(branches.get('addition')).get('label', '尚待复核')}")
    horizon = mapping(decision.get("horizon"))
    if horizon:
        st.caption(f"研究适用区间 {horizon.get('start')} ～ {horizon.get('end')}；不是保证持有到期或固定禁售期。")
    for reason in decision.get("quantitative_conflicts") or []:
        st.info(markdown_text(reason))
    for reason in decision.get("needed_review") or []:
        st.warning(markdown_text(_REVIEW_LABELS.get(reason, reason)))
    for reason in decision.get("execution_notes") or []:
        st.warning(markdown_text(reason))
    if decision.get("online_evidence"):
        st.caption("本次在线事实已纳入估值、披露或执行条件；消息对原投资理由的解释程度见下方，不自动改写研究姿态。")
    if not mapping(assessment.get("research_thesis")).get("payload"):
        st.caption("尚无这只基金的已复核研究。下面仍可看组合、比较和短期预测；没有把资料空白解释成卖出理由。")


def _render_original_decision_brief(brief: Mapping, *, replay: bool = False) -> None:
    """Render only the supplied choice; also safe for literal saved replay."""
    st.subheader("当时的首选与取舍" if replay else "当前首选：这笔基金怎么处理")
    st.caption(f"研究依据 {brief.get('research_date') or '尚无独立研究'} · 新买和已有资金分别判断，不要求真实账户资料。")
    if brief.get("research_summary"):
        with st.expander("研究总判断"):
            _write(brief["research_summary"])
    rationale = mapping(brief.get("rationale"))
    task = mapping(rationale.get("scope")).get("capital_task")
    if task:
        _write("这次判断的资金任务：" + task + "。")
    if brief.get("plan_semantics"):
        st.caption(markdown_text(brief["plan_semantics"]))
    for key, title in (("flat", "准备新买一笔"), ("held", "已有基金一笔＋备用现金一笔")):
        case = mapping(mapping(brief.get("cases")).get(key))
        if not case:
            continue
        st.markdown(f"**{title}：{markdown_text(case.get('label') or '尚无唯一首选')}**")
        if case.get("decision_status") in {"unresolved", "unresolved_conflict", "needs_review"}:
            st.caption("原有依据和比较仍在；下面的暂不动作不是已经证明最优。")
        if case.get("expected_value") is not None:
            _write(f"若按所选计划日卖出，首选做法预期净额 {_money(case['expected_value'])}（含应收款）；"
                   f"80%金额范围 {_money(case.get('lower_value'))} ～ {_money(case.get('upper_value'))}。")
        alternatives = case.get("why_not") or []
        for alternative in alternatives:
            difference = alternative.get("expected_difference_vs_preferred")
            if difference is not None:
                _write(f"若改为{alternative.get('label')}，同资本预期净额比首选"
                       f"{'多' if difference > 0 else '少' if difference < 0 else '相同'}"
                       f"{' ' + _money(abs(difference)) if difference else ''}。")
        reasons = lines(case.get("reasons"))
        for reason in reasons[:1]:
            _write(reason)
        conflict = mapping(case.get("conflict"))
        if conflict.get("resolution"):
            _write(conflict["resolution"])
        elif conflict.get("status") == "opposed":
            st.info("研究倾向与本窗口费用后比较相反，还没有针对这份研究的明确裁决；不把两者平均成一个分数。")
        if alternatives:
            st.markdown("为什么不选其他做法")
            # Start with the closest alternative, not a full ranking by mean.
            main = alternatives[0]
            if main.get("reason"):
                _write(f"{main.get('label')}：{main['reason']}")
            elif case.get("preferred_action") is None:
                _write("尚未解决的是原研究动作与本窗口净额的取舍；可以先核对下面对应金额与决定性经营资料。")
            with st.expander(f"{title}：完整理由、备选与引用"):
                for reason in reasons[1:]:
                    _write(reason)
                for alternative in alternatives[1:]:
                    if alternative.get("reason"):
                        _write(f"{alternative.get('label')}：{alternative['reason']}")
                selected = set(case.get("fact_ids") or [])
                st.caption("采用的经济事实与原始来源，见下方“完整研究、反方与来源”。" if selected else "尚无可引用的经济事实。")
        counter = case.get("counterevidence") or []
        if counter:
            st.markdown("最强反方")
            _write(mapping(counter[0]).get("reason"))
        checks = case.get("checks") or []
        if checks:
            st.markdown("什么变化会让我改判")
            for check in checks[:2]:
                _write(f"• {check.get('condition')} → {check.get('effect')}")
        execution = mapping(case.get("execution_status"))
        if execution.get("status") == "blocked":
            st.warning("经济倾向与能否执行分开：当前公开申赎状态暂停，不能按上述倾向假定今天成交。")
        if case.get("comparison_status") != "ready":
            st.caption("部分计划卖出净额尚不可比；只影响相应算账，不自动表示真实禁止申购或必须退出。")
        for reason in case.get("needed_review") or []:
            _write(_REVIEW_LABELS.get(reason, reason))
    st.caption("减仓比例没有被优化；卖一半仅为金额示例。费用后净额是所选日期情景，不是保证收益。")
    if rationale:
        st.caption(f"动作说明生成于 {rationale.get('generated_at')} · 基于原研究的会话助手说明，不是新预测或人工复核。")


def render_decision_brief(brief: Mapping, *, replay: bool = False, branch: str = "flat") -> None:
    """Keep brief1/2 replay literal; brief3 uses the single-context renderer."""
    if brief.get("schema_version") == 3:
        from rmbfund.buyer_view import build_buyer_view
        from rmbfund.ui.buyer_view import render_buyer_view

        render_buyer_view(build_buyer_view({"decision_brief": brief}, branch=branch),
                          key="saved_brief_view" if replay else "brief_view")
        return
    if brief.get("schema_version") != 2:
        _render_original_decision_brief(brief, replay=replay)
        return

    st.subheader("当时的操作建议" if replay else "操作建议")
    cases = mapping(brief.get("cases"))
    titles = (("flat", "还没买"), ("held", "已经持有"))
    action_names = {"buy": "新买一笔", "wait": "先留现金", "hold": "继续持有，不追加",
                    "add": "保留已有基金，再追加一笔", "reduce": "减少这只基金，比例另定",
                    "redeem": "退出这只基金", "exit": "现在退出", "reduce50": "卖出一半（金额示例）"}
    for key, title in titles:
        case = mapping(cases.get(key))
        if not case:
            continue
        action = case.get("preferred_action")
        label = case.get("label") or action_names.get(action, "暂不操作，先复核依据")
        with st.container(border=True):
            st.markdown(f"**{title}：{markdown_text(label)}**")
            reasons = lines(case.get("reasons"))
            _write("主要原因：" + (case.get("advice_reason") or (reasons[0] if reasons else "当前依据还不足以作出买卖判断。")))
            # Explain the chosen action against its relevant alternative, never a
            # ranking by mean. A rejected addition is compared with keeping cash.
            reference = case.get("comparison_action")
            if not reference:
                reference = {"buy": "wait", "wait": "buy", "add": "hold", "hold": "exit",
                             "reduce": "hold", "redeem": "hold", "exit": "hold"}.get(action)
                if action == "hold" and case.get("research_action") == "add":
                    reference = "add"
            alternative = next((row for row in case.get("why_not") or []
                                if row.get("action") == reference
                                and row.get("expected_difference_vs_preferred") is not None), {})
            if alternative:
                difference = Decimal(str(alternative["expected_difference_vs_preferred"]))
                change = "少" if difference > 0 else "多" if difference < 0 else "相同"
                amount = " " + _money(abs(difference)) if difference else ""
                _write(f"按所选计划和同样本金测算，费用后金额预计比{alternative.get('label') or action_names.get(reference)}{change}{amount}。")
            _write("什么时候再考虑：" + (case.get("next_step") or "先复核当前依据，再考虑下一步操作。"))
            execution = mapping(case.get("execution_status"))
            if execution.get("status") == "blocked":
                notice = {"subscription": "当前暂停申购，暂不提交买入。",
                          "redemption": "当前暂停赎回，暂不提交卖出。"}
                st.warning(notice.get(execution.get("operation"), "当前暂停申赎，暂不提交交易。"))

    with st.expander("完整判断、研究分歧与来源", expanded=False):
        if brief.get("research_summary"):
            _write("原研究总判断：" + brief["research_summary"])
        rationale = mapping(brief.get("rationale"))
        task = mapping(rationale.get("scope")).get("capital_task")
        if task:
            _write("判断的资金任务：" + task + "。")
        for name in ("plan_semantics", "amount_basis"):
            if brief.get(name):
                _write(brief[name])
        for key, title in titles:
            case = mapping(cases.get(key))
            if not case:
                continue
            st.markdown(f"**{title}：原研究与完整比较**")
            _write("原研究动作：" + action_names.get(case.get("research_action"), "尚未形成明确动作"))
            for reason in lines(case.get("reasons")):
                if reason != case.get("advice_reason"):
                    _write(reason)
            conflict = mapping(case.get("conflict"))
            if conflict.get("status") == "opposed":
                _write("原研究与本窗口费用后比较方向相反。")
            if conflict.get("resolution"):
                _write("分歧取舍依据：" + conflict["resolution"])
            if case.get("expected_value") is not None:
                _write(f"若按所选计划日卖出，建议做法预期净额 {_money(case['expected_value'])}（含应收款）；"
                       f"80%金额范围 {_money(case.get('lower_value'))} ～ {_money(case.get('upper_value'))}。")
            for alternative in case.get("why_not") or []:
                label = alternative.get("label") or action_names.get(alternative.get("action"), "备选做法")
                difference = alternative.get("expected_difference_vs_preferred")
                if difference is not None:
                    difference = Decimal(str(difference))
                    _write(f"若改为{label}，同资本预期净额比建议做法"
                           f"{'多' if difference > 0 else '少' if difference < 0 else '相同'}"
                           f"{' ' + _money(abs(difference)) if difference else ''}。")
                if alternative.get("reason"):
                    _write(f"{label}：{alternative['reason']}")
            if case.get("counterevidence"):
                st.markdown("**反方依据**")
                for counter in case["counterevidence"]:
                    _write(mapping(counter).get("reason") if isinstance(counter, Mapping) else counter)
            if case.get("checks"):
                st.markdown("**改判条件**")
                for check in case["checks"]:
                    _write(f"• {check.get('condition')} → {check.get('effect')}")
            if case.get("comparison_status") != "ready":
                st.caption("部分计划卖出净额尚不可比；不等于真实暂停申赎。")
            for reason in case.get("needed_review") or []:
                _write(_REVIEW_LABELS.get(reason, reason))
            for reason in mapping(case.get("execution_status")).get("reasons") or []:
                _write(reason)
        st.caption(f"研究日 {brief.get('research_date') or '未记录'}；以下保留原研究、作者说明、事实引用与范围。")
        st.json(dict(brief), expanded=False)


def render_investor(assessment: Mapping) -> None:
    decision = mapping(assessment.get("investor_decision"))
    if "action_basis" not in decision:
        _render_original_investor(assessment)
        return
    hypothetical = assessment.get("mode") in {"standard_trade", "calendar_trade", "buyer_decision"}
    state = "hypothetical" if hypothetical else assessment.get("holding_status")
    branches = mapping(decision.get("branches"))
    payload = mapping(mapping(assessment.get("research_thesis")).get("payload"))
    basis = decision["action_basis"]
    st.caption(f"整只基金的买卖判断 · 研究日 {decision.get('research_date') or '尚无研究'}")
    if basis == "legacy_opinion":
        st.subheader("这份旧研究还没有完成买卖取舍")
        posture = {"accumulate": "增持", "maintain": "维持", "trim": "减少", "exit": "退出",
                   "unresolved": "未定"}.get(decision.get("posture"), "未定")
        _write(f"上次研究倾向：{posture}。原意见保留在下方，不再自动翻成三个买卖建议。")
        _write("尚未单独回答：现在买入比留现金好在哪里；继续持有比卖出或换基好在哪里；若保留却不追加，两者差别是什么。")
    elif basis == "unavailable":
        st.subheader("尚未形成这次买卖判断")
        if not payload:
            st.caption("更新研究后，这里显示针对这只基金的新买、持有和追加理由。")
    else:
        st.subheader("较长期的基金研究" if hypothetical else "这笔钱怎么处理")
        _write(decision.get("research_summary"))
    if hypothetical and payload.get("horizon"):
        horizon = mapping(payload["horizon"])
        st.caption(f"研究适用区间 {horizon.get('start')} ～ {horizon.get('end')}；不是必须持有到该日。")
    names = (("held", "已有持仓"), ("addition", "追加资金")) if state == "held" else (
        (("flat", "准备新买"),) if state == "flat" else
        (("flat", "准备新买"), ("held", "已有持仓"), ("addition", "追加资金")))
    for key, title in names:
        branch = mapping(branches.get(key))
        constraints = lines(branch.get("constraint_reasons"))
        if basis != "independent_cases" and not constraints:
            continue
        label = branch.get("label", "尚未形成判断")
        if hypothetical and branch.get("action") == "add":
            label = "研究支持追加一笔"
        st.markdown(f"**{title}：{markdown_text(label)}**")
        for reason in constraints:
            _write(reason)
        if branch.get("research_reason"):
            if branch.get("research_action") != branch.get("action"):
                _write("研究原意见：" + _ACTION_NAMES.get(branch.get("research_action"), "先复核"))
            _write(branch["research_reason"])
            selected = set(branch.get("fact_ids") or [])
            with st.expander(f"{title}的依据与日期"):
                for fact in payload.get("facts") or []:
                    if fact["id"] in selected:
                        _write(fact["claim"])
                        if fact.get("effective_date"):
                            st.caption("事实生效日 " + fact["effective_date"])
    if hypothetical:
        st.caption("以上分别回答标准新买、持有和追加，不使用个人份额筛掉任何一种情况。")
    elif state == "unknown":
        st.caption("未填写份额，分别展示新买和已有持仓，不假定你没有持有。")
    if basis == "independent_cases":
        for key, title in (("change_conditions", "什么变化会让我换个做法"),
                           ("falsifiers", "什么证据会让我承认看错")):
            if decision.get(key):
                st.markdown(f"**{title}**")
                for reason in decision[key]:
                    _write("• " + reason)
    notes = [*decision.get("needed_review", []), *decision.get("execution_notes", [])]
    if notes:
        with st.expander("需要处理的资料变化与交易条件", expanded=True):
            for reason in dict.fromkeys(notes):
                _write(_REVIEW_LABELS.get(reason, reason))
    if basis == "independent_cases" and decision.get("forecast_status") == "ready":
        for reason in decision.get("quantitative_conflicts") or []:
            st.info(markdown_text(reason))


_CASE_NAMES = {"new_buy": "准备新买", "held": "已经持有", "addition": "想要追加"}
_ACTION_NAMES = {"buy": "有条件买入", "wait": "暂不新增", "hold": "继续持有", "reduce": "减少投入",
                 "redeem": "赎回退出", "review": "先复核持有理由", "add": "有追加依据"}
_CHANNEL_NAMES = {"earnings": "盈利", "valuation": "价格与估值", "fx": "汇率", "rates": "利率",
                  "liquidity": "流动性", "operations": "申赎条件", "mandate": "基金投向", "other": "其他影响"}


def _fact_ids(row: Mapping) -> None:
    if row.get("fact_ids"):
        st.caption("对应下方事实 " + "、".join(row["fact_ids"]))


def _render_generated_reasoning(payload: Mapping, *, include_actions: bool = True) -> None:
    for key, title in (("fund_quality", "基金本身是否值得长期关注"), ("entry_case", "现在的买入价格是否合适")):
        section = mapping(payload.get(key))
        st.markdown(f"**{title}**")
        _write(section.get("conclusion") or "尚无结论")
        _write(section.get("reason") or "理由尚未提供")
        _fact_ids(section)
    if include_actions:
        st.markdown("**不同动作，理由分别看**")
        for key, name in _CASE_NAMES.items():
            case = mapping(mapping(payload.get("action_cases")).get(key))
            _write(f"{name}：{_ACTION_NAMES.get(case.get('action'), '尚待研究')}")
            _write(case.get("reason") or "没有可用的动作论证。")
            _fact_ids(case)
    if payload.get("drivers"):
        st.markdown("**这些材料怎样影响这只基金**")
        for driver in payload["drivers"]:
            _write(f"{_CHANNEL_NAMES.get(driver.get('channel'), '影响渠道')}：{driver.get('mechanism')}")
            if driver.get("fund_transmission"):
                _write("传到基金的路径：" + driver["fund_transmission"])
            _fact_ids(driver)
    if payload.get("alternatives"):
        st.markdown("**实际候选的得失**")
        for alternative in payload["alternatives"]:
            _write(f"{alternative.get('product_id')}：{alternative.get('reason')}")
            _fact_ids(alternative)
    if payload.get("key_unknowns"):
        st.markdown("**哪些未知会影响我的动作**")
        for unknown in payload["key_unknowns"]:
            affected = "、".join(_CASE_NAMES.get(key, key) for key in unknown.get("affects", []))
            _write(f"{unknown.get('question')}（影响{affected}；{'决定性缺口' if unknown.get('critical') else '非决定性缺口'}）")
    change = mapping(payload.get("change_evaluation"))
    if change:
        st.markdown("**原来的理由增强了，还是被改变了**")
        _write(change.get("summary"))
        _fact_ids(change)


def render_research(assessment: Mapping) -> None:
    thesis = mapping(assessment.get("research_thesis"))
    payload = mapping(thesis.get("payload"))
    updates = thesis.get("source_updates") or []
    if updates:
        with st.expander("定向资料最近检查与变化"):
            for update in updates:
                _write(f"{update.get('source_url')} · {update.get('completed_at')} · "
                       + ("有未复核变化" if update.get("unreviewed_change_at") else "保留原资料日期"))
                for record in update.get("reviewed_facts") or []:
                    fact = mapping(record.get("payload"))
                    _write(f"{fact.get('event_date', '')} {fact.get('title', '')}：{fact.get('facts', '')}")
            st.caption("资料取得、事实复核和投资意见更新分开；这不是每天自动生成的新建议。")
    if not payload:
        return
    view = mapping(payload.get("assessment"))
    generated = payload.get("schema_version") == 2
    decision = mapping(assessment.get("investor_decision"))
    current_view = "action_basis" in decision
    active_cases = decision.get("action_basis") == "independent_cases"
    panel = (st.expander("完整研究、反方与来源" if active_cases else
                         f"上次研究原文 · {payload.get('research_date') or '日期未记录'}")
             if current_view else st.container(border=True))
    with panel:
        if not current_view:
            st.markdown("**为什么买这只，而不是只看它涨了多少**")
        if not active_cases:
            _write(view.get("summary", "尚待复核"))
        if generated:
            _render_generated_reasoning(payload, include_actions=not active_cases)
        columns = st.columns(2)
        for column, key, label in ((columns[0], "supporting", "支持依据"), (columns[1], "counterevidence", "反对依据 / 可能看错的地方")):
            with column:
                st.markdown(f"**{label}**")
                for row in view.get(key) or []:
                    _write("• " + row["reason"])
        priced = mapping(view.get("priced_in"))
        _write("价格是否已反映：" + str(priced.get("reason") or "未知，不能把利好当作便宜的证明。"))
        for key, title in (("change_conditions", "什么变化会改变判断"), ("falsifiers", "什么证据会推翻原理由"), ("missing", "目前还缺什么")):
            if active_cases and key in {"change_conditions", "falsifiers"}:
                continue
            if view.get(key):
                st.markdown(f"**{title}**")
                for text in view[key]:
                    _write("• " + text)
        if generated:
            origin = mapping(payload.get("origin"))
            method = "会话助手研究" if origin.get("provider") == "interactive_assistant" else "AI生成研究"
            st.caption(f"{method} · 模型 {origin.get('model') or '未注明'} · 实际生成于 {origin.get('generated_at') or '未记录'} · "
                       f"研究日 {payload.get('research_date')}；不是人工复核，也不是发布者原文。引用与数值检查不证明投资结论正确。")
        else:
            st.caption(f"本地作者研究 · {payload.get('author')} · 复核 {payload.get('reviewer')} · 研究日 {payload.get('research_date')}；不是发布者原文，也不是每天自动生成的意见。")
        with st.expander("关键事实与来源"):
            fact_labels = {"factual": "已披露事实", "effective": "已生效事实", "plan": "计划", "unresolved": "尚未确定",
                           "assumption": "明确假设", "model_judgment": "模型判断"}
            for fact in payload.get("facts") or []:
                _write(f"{fact['id']} · {fact['claim']}")
                if generated:
                    st.caption(f"{fact_labels.get(fact.get('status'), '事实状态未注明')} · 生效日 {fact.get('effective_date') or '未注明'} · "
                               "引用 " + "、".join(fact.get("citation_ids") or []))
            for citation in payload.get("citations") or []:
                _write(f"{citation['id']} · {citation.get('title') or citation.get('claim')}")
                if citation.get("claim") and citation.get("title"):
                    _write(citation["claim"])
                url = citation.get("source_url") or citation.get("publisher_url")
                if url:
                    st.link_button("查看原公开来源", url)
                    if generated:
                        st.caption(str(url))
                if citation.get("quote"):
                    _write("原文摘录：" + citation["quote"])
                if citation.get("kind") == "online_evidence":
                    st.caption(f"原发布日期 {citation.get('published_at') or '未知'} · {citation.get('locator') or '定位未注明'} · "
                               f"实际取得 {citation.get('observed_at') or '未记录'}；仅保存本次采用的摘录，未归档全文。")
                elif citation.get("kind") == "calculation":
                    _write("程序计算方法：" + str(citation.get("method") or "未记录"))
                    st.caption(f"定位 {citation.get('locator') or '未注明'} · 结果可用时间 {citation.get('available_at') or '未记录'}")
                    st.json({key: citation.get(key) for key in ("inputs", "result")}, expanded=False)
                else:
                    st.caption(f"{citation.get('published_at') or '原发布日期未知'} · {citation.get('locator')} · "
                               + ("只保存已核对摘录，未归档全文" if citation.get("kind") == "reviewed_excerpt" else
                                  "仅来源链接，未归档全文" if citation.get("kind") == "linked_not_archived" else "引用本地已留存原件"))


def _peer_subject(value: Mapping, row: Mapping) -> dict | None:
    peer = mapping(row.get("peer"))
    identifier = row.get("peer_product_id") or row.get("product_id") or peer.get("id")
    for item in value.get("funds") or []:
        product = mapping(item.get("product"))
        if product.get("id") == identifier:
            return dict(product)
    if peer.get("symbol"):
        return dict(peer)
    parts = str(identifier or "").split(":", 2)
    if len(parts) == 3 and parts[0] == "open_fund" and len(parts[1]) == 6 and parts[1].isdigit():
        return {"id": identifier, "kind": parts[0], "symbol": parts[1], "name": parts[1],
                "share_class": parts[2], "exchange": "", "is_demo": False}
    return None


def _share_class_cost_basis(value: Mapping, row: Mapping) -> list[str]:
    """Use dated fee evidence, not a configuration-time missing-fee assertion."""
    funds = {mapping(item.get("product")).get("id"): item for item in value.get("funds") or []}
    target_id = row.get("target_product_id") or mapping(value.get("peer_set")).get("target_product_id")
    peer_id = row.get("peer_product_id") or row.get("product_id")
    target, peer = funds.get(target_id, {}), funds.get(peer_id, {})
    before, after = mapping(target.get("product")), mapping(peer.get("product"))
    if (not before.get("underlying_fund_id")
            or before.get("underlying_fund_id") != after.get("underlying_fund_id")
            or not before.get("share_class") or not after.get("share_class")
            or before["share_class"] == after["share_class"]):
        return []
    usable = [fund for fund in (target, peer)
              if mapping(fund.get("fees")).get("status") in {"available", "partial"}]
    if not any(mapping(fund.get("fees")).get("subscription_rate") is not None
               or mapping(fund.get("fees")).get("redemption_bands") for fund in usable):
        return []
    result = [f"同一底层基金的{before['share_class']}类与{after['share_class']}类费用比较，"
              "不是新增一套投资组合或独立的分散化。"]
    for fund in (target, peer):
        product, fees = mapping(fund.get("product")), mapping(fund.get("fees"))
        label = f"{product.get('share_class')}类"
        if fees.get("status") not in {"available", "partial"}:
            result.append(f"{label}：尚无可用的公示费用记录，不能按零费用比较。")
            continue
        parts = [f"{item['name']} {percent(item['rate'], digits=2)}/年"
                 for item in fees.get("items") or [] if item.get("basis") == "annual_nav"]
        subscription = fees.get("subscription_rate")
        parts.append(f"公示标准申购费率 {percent(subscription, digits=2)}" if subscription is not None
                     else "公示标准申购费率尚未取得")
        bands = fees.get("redemption_bands") or []
        if bands:
            tiers = [f"{band['min_days']}～{band['max_days']}天（含）" if band.get("max_days") is not None
                     else f"至少{band['min_days']}天" for band in bands]
            parts.append("公示赎回费分段：" + "、".join(
                f"{days} {percent(band['rate'], digits=2)}" for days, band in zip(tiers, bands)))
        else:
            parts.append("公示赎回费分段尚未取得")
        result.append(f"{label}（费用资料日{fees.get('as_of') or '未注明'}）：" + "；".join(parts) + "。")
        if bands and fees.get("holding_day_basis") != "confirmation_calendar_days":
            result.append(f"{label}的持有天数计算口径尚未核实，不能据此替你选择当前赎回费率。")
        if not fees.get("channel_verified"):
            result.append(f"{label}在支付宝等渠道的实际申购折扣及适用赎回费仍未核实，公示标准费率不等于渠道实际费率。")
    result.append("年度运作费用通常已计入净值，不在净值表现上重复扣除；哪类更省还取决于购买金额、持有期和实际申赎费。")
    return result


def render_comparison(value: Mapping, *, preview: Callable[[Mapping], None] | None = None) -> None:
    st.subheader("和替代基金相比")
    if value.get("basis") == "current_online":
        st.caption("当前比较仅使用本次在线取得的共同净值日期，不混用旧本地序列。")
    comparisons = value.get("comparisons") or []
    if not comparisons:
        st.caption("还没有这只基金的合格替代比较；不会把所有QDII当同类或编造排名。")
        if value.get("status") == "unavailable":
            st.caption("比较资料暂不能使用，基金研究与保存仍可继续。")
        return
    for index, row in enumerate(comparisons):
        peer = _peer_subject(value, row) or mapping(row.get("peer"))
        label = peer.get("name") or row.get("peer_product_id") or row.get("product_id") or "候选基金"
        st.markdown(f"**{markdown_text(label)}**")
        if preview is not None and peer.get("symbol"):
            st.button("公开预览这只候选", key=f"buyer_preview_{peer.get('id') or peer['symbol']}_{index}",
                      help="只打开公开资料，不加入持仓，也不转入你的份额或成本。",
                      on_click=preview, args=(peer,))
        cost_basis = _share_class_cost_basis(value, row)
        if cost_basis:
            for text in cost_basis:
                _write(text)
        else:
            _write(row.get("selection_reason") or mapping(row.get("relation")).get("selection_reason") or "选取理由见已保存候选集合。")
        for key in ("reasons", "interpretation", "tradeoffs", "missing"):
            items = row.get(key)
            for text in ([items] if isinstance(items, str) else items or []):
                if isinstance(text, str):
                    _write(text)
        window = mapping(row.get("common_window"))
        if window:
            start, end = window.get("start_date") or window.get("start"), window.get("end_date") or window.get("end")
            st.caption(f"共同实际观察区间 {start} ～ {end}；不补周末。" if start and end else "共同区间尚不可用。")
        table = []
        for key, title in (("subject_metrics", "所选基金"), ("peer_metrics", "替代基金")):
            metrics = mapping(row.get(key))
            if metrics:
                table.append({"基金": title, "单位净值变化（非总回报）": percent(mapping(metrics.get("raw_return")).get("value"), signed=True),
                              "最大回撤": percent(mapping(metrics.get("max_drawdown")).get("value")),
                              "观察数": metrics.get("observation_count", metrics.get("count"))})
        if table:
            st.dataframe(table, hide_index=True, width="stretch")
    historical = mapping(value.get("historical_reference"))
    if historical:
        with st.expander("历史参考（不是本次完整在线比较）"):
            _write(historical.get("label") or "保留原日期，仅作历史参考。")
            past = mapping(historical.get("comparison"))
            for row in past.get("comparisons") or []:
                window = mapping(row.get("common_window"))
                _write(f"{row.get('product_id') or '候选基金'}：{window.get('start_date') or window.get('start') or '起日未知'} ～ "
                       f"{window.get('end_date') or window.get('end') or '止日未知'}")
            st.json(dict(past), expanded=False)
    with st.expander("比较口径与完整依据"):
        st.json(dict(value), expanded=False)


def render_portfolio(context: Mapping) -> None:
    st.subheader("对我的组合有什么影响")
    cohort, selected = mapping(context.get("known_cohort")), mapping(context.get("selected"))
    if cohort:
        _write(f"已记录且可估值的人民币基金合计 {_money(cohort.get('market_value'))}；"
               f"这只基金占其中 {percent(selected.get('weight_of_known_cny_valued_funds'), digits=2)}。")
        st.caption("这不是你的总资产比例；未登记基金、现金和无法估值的项目不在分母里。")
    look = mapping(context.get("look_through"))
    issuers = mapping(look.get("issuers")).get("exposures") or []
    if issuers:
        st.markdown("**按披露持仓穿透的公司暴露**")
        st.dataframe([{"公司": row.get("name") or row.get("issuer_id"),
                       "披露篮子对应金额": _money(row.get("known_value_cny")),
                       "占已知基金市值": percent(row.get("weight_of_known_cny_valued_funds"), digits=2)}
                      for row in issuers[:8]], hide_index=True, width="stretch")
    basket = mapping(look.get("issuers"))
    if basket:
        st.caption(f"发行人披露覆盖已知基金市值的 {percent(basket.get('coverage'), digits=2)}；"
                   f"未覆盖金额 {_money(basket.get('uncovered_value_cny'))}，不当作现金，也不补成完整持仓。")
    for fund in context.get("funds") or []:
        disclosure = mapping(fund.get("disclosure"))
        st.caption(f"{fund.get('product_id')}：估值日 {fund.get('nav_date') or '未知'}；披露日 {disclosure.get('report_date') or '未知'}。")
    if context.get("unknowns") or context.get("exclusions"):
        st.caption("部分基金缺少估值、披露或可靠映射；未覆盖部分没有按0计算风险。")
        with st.expander("哪些基金或披露还不完整"):
            for item in context.get("unknowns") or []:
                if isinstance(item, str):
                    _write(item)
                else:
                    index = item.get("index")
                    _write(f"第{index + 1 if isinstance(index, int) else '?'}只记录："
                           + ("披露不完整或映射未确认" if item.get("disclosure") or item.get("unmapped_rows") else "估值日期或相关资料未知"))
            for item in context.get("exclusions") or []:
                _write(f"{item.get('product_id')}未计入可估值人民币基金分母；币种或市值资料尚不完整。")
    if context.get("overlaps"):
        with st.expander("两只基金是否买了同样的公司"):
            st.dataframe([{"基金一": row.get("left_product_id"), "基金二": row.get("right_product_id"),
                           "披露发行人重合下限": percent(row.get("overlap"), digits=2),
                           "基金一披露日": row.get("left_report_date"), "基金二披露日": row.get("right_report_date")}
                          for row in context["overlaps"]], hide_index=True, width="stretch")
            st.caption("只比对已核实的发行人ID；披露篮子重合不是当前完整持仓重合，也不是证券级同一份资产。")


def sizing_inputs(saved: dict, *, portfolio: Mapping, current: Mapping,
                  fees: Mapping, requested: Mapping | None) -> dict:
    """Optional local constraints. No forecast quantile is called a stress scenario."""
    from rmbfund.position_sizing import conditional_size, planned_buy, planned_redemption

    raw = dict(saved.get("sizing", {}))
    cash_need_labels = {"unknown": "还不确定", "no": "没有明确的近期必须用款", "yes": "是，明确近期必须用这笔钱"}
    cash_need = raw.get("near_term_required", raw.get("cash_need", "unknown"))
    raw["near_term_required"] = st.selectbox("是否明确近期必须用这笔钱（可选）", list(cash_need_labels),
        index=list(cash_need_labels).index(cash_need if cash_need in cash_need_labels else "unknown"),
        format_func=cash_need_labels.get, key="buyer_near_term_required")
    if raw["near_term_required"] == "yes":
        raw["near_term_use_time"] = st.text_input("明确用款的时间与用途（可留空）",
            value=raw.get("near_term_use_time", ""), key="buyer_near_term_use_time", placeholder="例如下月房租；不知道日期不必猜")
        st.caption("近期必须用款会影响资金适配：QDII确认和赎回到账较慢，不能把预计到账当作保证。")
    with st.expander("想知道能买多少、需要减少多少？（可选）"):
        st.caption("这里计算你明确条件下的金额上限，不是任意给一个最佳仓位。留空仍可看基金判断。")
        scopes = ["只看这只基金", "已记录的全部人民币基金"]
        raw["budget_scope"] = st.selectbox("损失预算适用范围", scopes,
            index=scopes.index(raw.get("budget_scope", scopes[0])), key="buyer_budget_scope")
        budget_label = "这只基金最多愿意承受的情景损失（元，可留空）" if raw["budget_scope"] == scopes[0] else "全部已记录基金合计可承受的情景损失（元，可留空）"
        for key, label in (("spendable_cash", "可投入现金（元，可留空）"),
                           ("max_investment", "本次最高投入（元，可留空）"),
                           ("loss_budget", budget_label),
                           ("scenario_percent", "假设整只基金下跌多少（%，可留空）"),
                           ("planned_redemption", "想预演赎回的当前毛额（元，可留空）")):
            raw[key] = st.text_input(label, value=raw.get(key, ""), key=f"buyer_{key}")
        raw["plan_horizon"] = st.text_input("这笔资金计划使用时间（可留空）", value=raw.get("plan_horizon", ""), key="buyer_plan_horizon")
        st.caption("下跌条件是你填写的整基金压力假设，不是预测、最大亏损或止损保证；不拿部分股票冲击充当整基金损失。")
        other_returns = dict(raw.get("other_returns", {}))
        if raw["budget_scope"] == scopes[1]:
            st.caption("请为每只已记录基金填同一情景下的涨跌幅；有基金不能估值或条件留空，就不计算完整组合上限。现金及未登记资产不在此预算内。")
            selected_id = mapping(portfolio.get("selected")).get("product_id")
            for fund in portfolio.get("funds") or []:
                identifier = fund.get("product_id")
                if identifier != selected_id:
                    other_returns[identifier] = st.text_input(f"{identifier}：同一情景下涨跌幅（%）",
                        value=other_returns.get(identifier, ""), key=f"buyer_scenario_{identifier}")
        raw["other_returns"] = other_returns
    saved["sizing"] = raw
    inputs = {
        "subscription_fee": fees.get("subscription_fee"), "redemption_fee": fees.get("redemption_fee"),
        "existing_market_value": current.get("market_value"),
        "known_cohort_market_value": mapping(portfolio.get("known_cohort")).get("market_value"),
        "selected_market_value": current.get("market_value"),
        "requested_gross_cash": mapping(requested).get("amount"),
        "budget_scope": "fund" if raw["budget_scope"] == scopes[0] else "covered_portfolio", "scenario_scope": "whole_fund",
        "scenario_label": "用户明确填写的整基金压力情景", "scenario_source": "local_user_condition",
    }
    valid = True
    try:
        for key in ("spendable_cash", "max_investment", "loss_budget"):
            if raw.get(key, "").strip():
                amount = decimal(raw[key].strip())
                if amount < 0:
                    raise ValidationError("金额不得小于0")
                inputs[key] = str(amount)
        if raw.get("scenario_percent", "").strip():
            magnitude = decimal(raw["scenario_percent"].strip())
            if not 0 <= magnitude <= 100:
                raise ValidationError("整基金下跌幅度须在0至100之间")
            d = -magnitude / 100
            inputs["scenario_return"] = str(d)
            if current.get("market_value") is not None:
                inputs["existing_scenario_loss"] = str(-d * decimal(current["market_value"]))
            if raw["budget_scope"] == scopes[1]:
                inputs["budget_scope"] = "covered_portfolio"
                inputs["scenario_label"] = "用户为每只已记录基金指定的同一完整情景"
                selected_id = mapping(portfolio.get("selected")).get("product_id")
                total_loss, complete = Decimal(0), bool(portfolio.get("funds"))
                for fund in portfolio.get("funds") or []:
                    identifier = fund.get("product_id")
                    change = d if identifier == selected_id else (
                        decimal(other_returns[identifier]) / 100 if other_returns.get(identifier, "").strip() else None)
                    if change is not None and change < -1:
                        raise ValidationError("每只基金的情景涨跌不能低于-100%")
                    if not fund.get("included") or change is None or fund.get("market_value") is None:
                        complete = False
                    else:
                        total_loss -= decimal(fund["market_value"]) * change
                inputs["existing_scenario_loss"] = str(total_loss) if complete else None
                if not complete:
                    st.caption("整组合情景尚未覆盖每只记录，已有组合损失和完整组合金额上限暂不能计算。")
        result = conditional_size(inputs)
        if requested:
            result["planned_buy"] = planned_buy({
                "gross_cash": inputs.get("requested_gross_cash"),
                **{key: inputs.get(key) for key in ("subscription_fee", "known_cohort_market_value", "selected_market_value")},
            })
        if raw.get("planned_redemption", "").strip():
            amount = decimal(raw["planned_redemption"].strip())
            if amount < 0:
                raise ValidationError("拟赎回毛额不得小于0")
            result["planned_redemption"] = planned_redemption({"gross_redemption": str(amount),
                **{key: inputs.get(key) for key in ("redemption_fee", "known_cohort_market_value", "selected_market_value", "available_redemption_value")}})
    except ValidationError as exc:
        st.warning(markdown_text(str(exc)))
        valid, result = False, {"status": "invalid_inputs"}
    result["inputs"] = inputs
    result["user_conditions"] = raw
    result["inputs_valid"] = valid
    return result


def render_sizing(plan: Mapping) -> None:
    whole, reduction = mapping(plan.get("whole_fund")), mapping(plan.get("reduction"))
    if whole.get("gross_cash_cap") is not None:
        _write(f"明确条件下，本次含费投入上限 {_money(whole['gross_cash_cap'])}。这是风险容量，不是建议全部买满。")
        names = {"spendable_cash": "可投入现金", "max_investment": "本次最高投入", "exposure_cap": "明确暴露限额",
                 "max_fund_weight": "用户比例上限", "scenario_loss_budget": "同一情景的损失预算"}
        st.caption("生效限制：" + "、".join(names.get(key, "明确限制") for key in whole.get("binding_constraints", [])))
    elif whole.get("cash_constraint_cap") is not None:
        _write(f"仅按已知资金限制可用 {_money(whole['cash_constraint_cap'])}；风险情景或预算尚不完整，这不是风险允许买入额。")
    if reduction:
        minimum = reduction.get("minimum_gross_redemption", reduction.get("minimum_reduction"))
        if minimum is not None:
            if reduction.get("feasible") is False:
                st.warning(f"同一情景下，理论上需减少 {_money(minimum)}，但超过可售金额或你的赎回限制；仅卖出这只基金不能满足该损失预算。")
            else:
                _write(f"同一情景下，为满足损失预算至少需要减少当前毛额 {_money(minimum)}。")
    buy = mapping(plan.get("post_buy"))
    if buy.get("weight_after") is not None:
        _write(f"按最近净值的静态买后占比 {percent(buy['weight_after'], digits=2)}（仅已知人民币基金子集）。")
    labels = {"scenario_return": "整基金压力涨跌幅", "existing_scenario_loss": "同一情景的已有损失",
              "loss_budget": "明确的损失预算", "subscription_fee": "适用申购费", "spendable_cash": "可投入现金",
              "whole_fund_scenario_required": "整只基金的情景口径", "explicit_budget_scope_required": "损失预算范围",
              "scenario_label_or_source": "情景依据"}
    missing = [labels[text] for text in plan.get("unknowns") or [] if text in labels]
    if missing:
        st.caption("金额方案还缺：" + "、".join(missing) + "。")
    if plan.get("inputs") or plan.get("planned_redemption"):
        with st.expander("金额计算的条件与静态预演"):
            st.json(dict(plan), expanded=False)
        st.caption("含费现金不等于买入基金金额；赎回毛额不等于到账。这里只作静态预演，不改份额、不登记交易。")


def _online_rows(value, product_id: str | None = None) -> list[Mapping]:
    if isinstance(value, (list, tuple)):
        return [row for row in value if isinstance(row, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if product_id is not None:
        return _online_rows(value.get(product_id))
    return [row for rows in value.values() if isinstance(rows, (list, tuple))
            for row in rows if isinstance(row, Mapping)]


def _online_source_label(row: Mapping) -> str:
    family = {"nav": "净值", "disclosure": "基金披露", "policy_news": "产业/政策", "news": "产业/政策"}.get(
        str(row.get("family") or ""), str(row.get("family") or "资料"))
    raw_status = str(row.get("status") or row.get("parse_status") or "")
    status = {"ok": "已取得", "ready": "可用", "partial": "部分可用", "error": "未取得",
              "failed": "未取得", "unavailable": "不可用", "stopped": "失败后已停止",
              "skipped_blocked": "失败来源未重试", "not_configured": "未配置来源",
              "unsupported": "暂不支持", "parsed": "已解析", "captured_unparsed": "文档尚未解析"}.get(raw_status, "待核对")
    return f"{family} · {status}"


def _online_record_summary(row: Mapping) -> str:
    payload = mapping(row.get("payload"))
    family = row.get("family")
    if family == "operations":
        states = {"open": "开放", "closed": "暂停", "unknown": "尚未确认", "limited": "限额"}
        return (f"公开申购状态：{states.get(payload.get('subscription_state'), '尚未确认')}；"
                f"赎回状态：{states.get(payload.get('redemption_state'), '尚未确认')}。"
                "这项公开操作条件用于本次可执行性判断，不代表支付宝页面已确认。")
    if family == "fees":
        names = (("management_rate", "管理费"), ("custody_rate", "托管费"), ("sales_service_rate", "销售服务费"))
        rates = [f"{label} {percent(payload[key], digits=2)}/年" for key, label in names if payload.get(key) is not None]
        return "；".join(rates) + "。年费已经体现在净值，不在收益比较中重复扣减。" if rates else "在线费率资料用于费用口径核对，未填写的渠道申赎费不按0。"
    if family == "portfolio":
        return "这份披露进入本次持仓穿透和组合比较；披露日不是今天的完整实时持仓。"
    if family == "management":
        return "这份经理资料进入本次基金背景，不把经理变更自动等同于买入或卖出。"
    if family == "mandate":
        return "这份合同任务资料用于核对基金投向、基准与币种，不把合同上限当实际仓位。"
    if family == "issuer_fundamentals":
        return "公司盈利资料按所属期进入公司分析；它不等于整只基金的合理估值。"
    return "这份在线事实保留原所属期；对买卖的完整解释以当前论证说明为准。"


def render_online(value: Mapping | None, *, saved: bool = False, product_id: str | None = None) -> None:
    """Explain the compact online facts used by the displayed assessment.

    This renderer intentionally accepts only the saved compact projection. It never
    fetches, opens a URL, or turns a title into a buy/sell rule.
    """
    context = mapping(value)
    if not context:
        return
    heading = "本次在线依据" if not saved else "当时保存的在线依据"
    with st.container(border=True):
        st.markdown(f"**{heading}**")
        checked = context.get("checked_at") or context.get("evaluated_at") or context.get("started_at")
        st.caption(("保存时记录的在线检查时间 " if saved else "在线检查时间 ") + str(checked or "未记录"))
        if saved:
            st.caption("这里展示保存时真正使用的紧凑依据；历史回放不会联网，也不会用今天的消息改写原判断。")
        else:
            st.caption("在线事实参与本次估值、比较或条件判断；它们不改写作者研究日期，也不自动生成买卖概率。")

        checks = [row for row in _online_rows(context.get("source_checks"))
                  if product_id is None or row.get("product_id") in (None, product_id)]
        if checks:
            st.markdown("**来源检查状态**")
            st.dataframe([{
                "资料家族": _online_source_label(row),
                "来源日期": row.get("source_date") or "未注明",
                "本次检查": row.get("checked_at") or checked or "未记录",
                "结果": row.get("reason") or "已按本次响应记录，详见来源",
                "来源": row.get("source_url") or "未注明",
            } for row in checks], hide_index=True, width="stretch")
            failures = [row for row in checks if str(row.get("status") or "").lower() in {"error", "failed", "unavailable", "stopped", "skipped_blocked"}]
            if failures:
                st.warning("部分在线来源本次未取得；下面保留旧资料日期，不把旧资料说成最新。")

        nav = context.get("nav_by_product") or context.get("nav") or context.get("latest_nav")
        nav_rows = _online_rows(nav, product_id)
        projection = mapping(mapping(context.get("projections_by_product")).get(product_id))
        if projection.get("nav_origin") == "local_reference":
            st.caption(f"当前估值仍用旧参考：净值日 {projection.get('nav_date') or '未注明'}，取得于 {projection.get('nav_observed_at') or '未注明'}；本次检查没有替旧值续鲜。")
        for reason in projection.get("rejected_nav_reasons") or []:
            _write(reason)
        if nav_rows:
            st.markdown("**本次来源返回的净值**")
            st.dataframe([{
                "净值所属日": row.get("date") or row.get("event_date") or "未注明",
                "单位净值": row.get("value") or row.get("unit_nav") or "未提供",
                "币种": row.get("currency") or "未注明",
                "来源": row.get("source_url") or row.get("source") or "未注明",
            } for row in nav_rows[-5:]], hide_index=True, width="stretch")

        records = context.get("records_by_product") or context.get("records")
        record_rows = _online_rows(records, product_id)
        if record_rows:
            with st.expander("结构化披露与实际影响", expanded=True):
                for row in record_rows:
                    title = row.get("title") or {"operations": "申赎条件", "fees": "费用", "portfolio": "披露持仓",
                            "management": "基金经理", "mandate": "合同任务", "issuer_fundamentals": "公司盈利"}.get(row.get("family"), "在线结构化资料")
                    _write(f"{title} · 所属日 {row.get('report_date') or row.get('source_date') or '未注明'}")
                    claim = row.get("claim") or row.get("facts") or row.get("summary") or row.get("reason")
                    _write(str(claim) if claim else _online_record_summary(row))
                    relation = row.get("relation") or row.get("interpretation") or row.get("relevance_reason")
                    if relation:
                        _write("与原判断的关系：" + str(relation))
                    url = row.get("source_url")
                    if url:
                        st.link_button("查看来源", str(url))

        messages = context.get("messages_by_product") or context.get("messages")
        message_rows = _online_rows(messages, product_id)
        if message_rows:
            message_rows = sorted(message_rows, key=lambda row: str(row.get("published_at") or ""), reverse=True)
            st.caption(f"发现{len(message_rows)}条相关消息；以下先看最近3条，完整依据仍保存在这次判断中。")
            for row in message_rows[:3]:
                status = {"title_only": "仅标题", "publisher_summary": "发布者摘要",
                          "online_excerpt": "在线正文摘录", "unparsed_document": "文档尚未解析"}.get(
                              row.get("content_status"), "内容程度未注明")
                _write(f"• {row.get('published_at') or '日期未注明'} · {row.get('title') or '未命名消息'}（{status}）")
            with st.expander("新消息与目前能解释到的程度"):
                for row in message_rows:
                    title = row.get("title") or "未命名消息"
                    _write(f"{title} · {row.get('published_at') or '发布日期未注明'}")
                    excerpt = row.get("excerpt") or row.get("summary")
                    if excerpt:
                        _write(str(excerpt))
                    status = {"title_only": "仅标题", "publisher_summary": "发布者摘要",
                              "online_excerpt": "在线正文摘录", "unparsed_document": "文档尚未解析"}.get(
                                  row.get("content_status"), "内容程度未注明")
                    reason = row.get("relevance_reason") or row.get("interpretation")
                    _write(f"资料状态：{status}。" + (f" 相关性：{reason}" if reason else ""))
                    url = row.get("source_url") or row.get("index_url")
                    if url:
                        st.link_button("查看公开来源", str(url))
                st.caption("标题或摘要只说明发现了什么，不等于已经完成对盈利、估值和基金组合的完整解释；中性消息不会机械改变买卖分支。")

        signals = _online_rows(context.get("review_signals_by_product"), product_id)
        if signals:
            st.markdown("**对原判断的关系**")
            for signal in signals:
                _write("• " + str(signal.get("reason") or "新资料与原论证的关系尚待解释。"))
        for warning in lines(context.get("warnings")):
            _write(warning)


def render_saved_buyer(payload: Mapping) -> None:
    render_investor(payload)
    render_online(mapping(payload.get("online_context")), saved=True,
                  product_id=mapping(payload.get("subject")).get("id"))
    render_research(payload)
    render_comparison(mapping(payload.get("fund_comparison")))
    render_portfolio(mapping(payload.get("portfolio_context")))
    render_sizing(mapping(payload.get("sizing_plan")))

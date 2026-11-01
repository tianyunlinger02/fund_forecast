"""Standard one-trade decisions; personal money is an optional conversion only."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal

import streamlit as st

from rmbfund.decision_brief import POLICY_VERSION
from rmbfund.domain import ValidationError, now_iso
from rmbfund.forecast_models import model_family
from rmbfund.timing_scenario import build_calendar_scenario, resolve_plan_fees
from rmbfund.trade_calendar import adjust_new_confirmation_date, build_trade_plan
from rmbfund.ui import forecast as forecast_ui
from rmbfund.ui.formatting import lines, mapping, markdown_text, percent, when


def _write(value) -> None:
    st.markdown(markdown_text(value))


def _money(value, *, signed=False) -> str:
    if value is None:
        return "未算出"
    return f"{Decimal(str(value)):+,.2f}" if signed else f"{Decimal(str(value)):,.2f}"


def render_plan(plan: Mapping) -> None:
    """Show the saved date mapping, never infer today's dates during replay."""
    if not plan:
        st.caption("这次没有保存日期计划。")
        return
    _write(f"计划 {plan.get('calendar_days')} 个自然日内考虑买卖，截止 {plan.get('deadline')}。")
    st.caption(f"假设 {plan.get('application_date') or '待定'} 申请买入或当前卖出，"
               f"{plan.get('sale_date') or '待定'} 申请到期卖出；"
               f"按该日程预计 {plan.get('horizon') if plan.get('horizon') is not None else '未知'} 笔净值测算。")
    st.caption(f"{plan.get('calendar_label') or '采用保存的模拟日程'}；未来净值笔数按此日程假设。")
    if plan.get("status") != "ready":
        st.info("所选窗口内没有足够时间完成这项买卖模拟，可以增加天数；已有研究仍保留。")
    elif plan.get("new_shares_sellable") is False:
        st.info("新买份额的模拟确认日不在申请日至计划卖出日之间；请调整日期，已有份额比较仍保留。")


def render_plan_fees(fees: Mapping) -> None:
    st.caption(f"模拟费用：申购 {percent(fees.get('subscription_fee'), digits=2)} · "
               f"已有份额现在卖 {percent(fees.get('current_redemption_fee'), digits=2)} · "
               f"已有份额计划卖 {percent(fees.get('future_old_redemption_fee'), digits=2)} · "
               f"新买份额计划卖 {percent(fees.get('future_new_redemption_fee'), digits=2)}。")
    st.caption(markdown_text(mapping(fees.get("subscription")).get("label") or "申购费依据未保存"))
    contexts = mapping(fees.get("contexts"))
    for key, label in (("current", "已有份额本次退出"), ("future_old", "已有份额到计划日卖出"),
                       ("future_new", "新买或追加后到计划日卖出")):
        item = mapping(contexts.get(key))
        if item:
            source = {"public_standard": "按公示阶梯", "override": "按手动费率", "illustrative": "按示例费率",
                      "unavailable": "未确认，不能卖出"}.get(mapping(item.get("redemption")).get("source"), "费率依据未保存")
            st.caption(f"{label}：模拟持有 {item.get('holding_days') if item.get('holding_days') is not None else '未知'} 天，{source}。")
    step = mapping(fees.get("next_lower_fee"))
    if step.get("status") == "available":
        _write(f"只看手续费：再等 {step.get('days_to_wait')} 天，到 {step.get('date')} 的下一较低档，"
               f"每 {_money(step.get('base_amount'))} 元相同赎回市值可少付约 {_money(step.get('saving_per_base'))} 元。")
        st.caption("等待期间净值可能上涨或下跌；不是必须等到该日再卖。")


def render_scenario(scenario: Mapping, *, evidence_only: bool = False) -> None:
    """Render the supplied object, also used by history without rebuilding it."""
    cash_plan = scenario.get("kind") == "calendar_cash"
    if cash_plan:
        render_plan(mapping(scenario.get("plan")))
    purchase = mapping(scenario.get("purchase"))
    if purchase.get("change") is not None:
        _write(f"假设在 {purchase.get('date')} 买入，到已公布的 {purchase.get('latest_date')}，"
               f"基金净值变化 {percent(purchase['change'], signed=True)}。")
        st.caption("这是过去已经发生的区间涨跌，不是未来预测；不因为亏损或想回本而改变下面的判断。")
    decision = mapping(scenario.get("decision"))
    head = mapping(decision.get("primary_head"))
    if not head:
        st.info("这批数据还没有对应周期和成交起点的预测。基金研究保留在下方；可明确点击计算，不拿其他周期代替。")
        if not cash_plan:
            return
    horizon, lag = scenario.get("horizon"), scenario.get("entry_lag")
    if head:
        st.caption(f"成交之后{horizon}笔净值 · 从预测净值起点向后数{lag}笔，作为假定执行起点；净值笔数不是自然日。")
        forecast_ui._metrics(head)
    if head.get("expected_change", 0) < 0 and head.get("up_probability", 0) > 0.5:
        st.caption("上涨概率超过一半，但平均变化仍为负：模型认为下跌时可能跌得更多，所以不能只按上涨次数决定买入。")
    branches = mapping(decision.get("branches"))
    st.subheader("本窗口净额比较依据" if evidence_only else "这一周期，现在偏向怎么做")
    for key, label in (("flat", "准备新买"), ("held", "已经持有")):
        _write(f"{label}：{mapping(branches.get(key)).get('label', '尚无比较')}")
    rows = scenario.get("rows") or []
    by_action = {row.get("action"): row for row in rows}
    for action, label in (("buy", "新买一笔，和留现金比"), ("hold", "留着已有一笔，和现在退出比")):
        row = by_action.get(action, {})
        if row.get("advantage") is not None:
            _write(f"{label}：期末预期{'多' if row['advantage'] >= 0 else '少'} {_money(abs(row['advantage']))} 元；"
                   f"前者更有利的估计概率 {percent(row.get('better_probability'))}。")
    if cash_plan and head:
        buy = by_action.get("buy") or {}
        if buy.get("expected_value") is not None:
            base = scenario.get("base_amount", 10000)
            gross_gain = base * head["expected_change"]
            drag = base + gross_gain - buy["expected_value"]
            if drag > 0:
                _write(f"新买这笔：预期净值变化对应 {_money(gross_gain, signed=True)} 元，"
                       f"申购及计划赎回费用约 {_money(drag)} 元；"
                       f"扣费后相对本金 {_money(buy['expected_value'] - base, signed=True)} 元。")
    st.markdown("**每种做法，钱会有什么不同**")
    st.dataframe([{
        "做法": row.get("label"),
        "共同起点": _money(row.get("initial_value")),
        ("预期卖出净额（含应收款）" if cash_plan else "预期期末账面值"): _money(row.get("expected_value")),
        "80%金额范围": f"{_money(row.get('lower_value'))} ～ {_money(row.get('upper_value'))}",
        "预期收益率": percent(row.get("expected_change"), signed=True, digits=2),
        "对照": row.get("comparison_label"),
        "相对对照多/少": _money(row.get("advantage"), signed=True),
        "胜过对照的估计概率": percent(row.get("better_probability")),
    } for row in rows], hide_index=True, width="stretch")
    if cash_plan:
        reasons = dict.fromkeys(reason for row in rows for reason in lines(row.get("reasons")))
        for reason in reasons:
            st.caption(markdown_text(reason))
    st.caption("新买与留现金各按一笔比较；已有基金的四种做法都从一笔基金＋一笔备用现金开始。追加不是凭投入翻倍就算更好；卖一半只是示例，不是最佳仓位。金额按统一示例单位换算。")
    if cash_plan:
        st.caption("剩余份额按计划申请卖出并扣届时各批次费用；净额包含赎回应收款，不等于截止日已到账。净值内年费不重复扣。")
        render_plan_fees(mapping(scenario.get("fees")))
    else:
        st.caption("期末保留的基金按净值记账，不假设届时一定赎回；清仓指在假定执行点退出，到账时间另计。已计入净值的年费不重复扣。")
    if decision.get("delay_sensitive"):
        st.info("其他成交延迟会改变方向；这里仍用原定起点，不挑更有利的一项。")


def render_sale_comparison(comparison: Mapping, *, expanded: bool = True) -> None:
    """Display independent old-lot date comparisons, including saved fee-only rows."""
    if not comparison:
        return
    with st.expander("已有份额：现在卖、等降费、按计划日卖", expanded=expanded):
        st.caption("只比较同一笔旧份额；各日期分别和现在退出相比，不是寻找全局最佳日。共同本金为1万元基金＋1万元备用现金。")
        for row in comparison.get("rows") or []:
            st.markdown(f"**{markdown_text(row.get('label'))} · {markdown_text(row.get('sale_date'))}** · "
                        f"赎回费 {percent(row.get('redemption_fee'), digits=2)}")
            if "now" in row.get("candidate_ids", []):
                _write(f"按当前费率退出后，基金净额加备用现金共 {_money(row.get('expected_value'))} 元（含赎回应收款）。")
                continue
            saving, boundary = row.get("fee_saving"), row.get("break_even_change")
            if saving is not None:
                _write(f"净值不变时，{'少付' if saving >= 0 else '多付'}手续费 {_money(abs(saving))} 元。")
            if boundary is not None:
                if boundary < 0:
                    _write(f"等待期间净值若下跌约 {percent(abs(boundary), digits=2)}，就会抵消这点节费；再跌则不如现在退出。")
                elif boundary > 0:
                    _write(f"这天费用更高，净值需上涨约 {percent(boundary, digits=2)}，才追平现在退出的净额。")
                else:
                    _write("费用相同；等待的得失取决于净值变化，不能靠等几天省钱。")
            if row.get("advantage") is not None:
                _write(f"当前准确期限预测：相对现在退出，预期净额 {_money(row['advantage'], signed=True)} 元；"
                       f"80%差额范围 {_money(row.get('lower_advantage'), signed=True)} ～ {_money(row.get('upper_advantage'), signed=True)} 元。")
                if row.get("better_probability") is not None:
                    st.caption(f"这个日期严格胜过现在退出的估计频率 {percent(row['better_probability'])}；不是它胜过其他未来日期的概率。")
            else:
                st.caption("尚无对应日期的准确预测；可点上方“比较这几个卖出时点”，先保留费用边界，不借其他期限。")
            for reason in lines(row.get("reasons")):
                st.caption(markdown_text(reason))
        tier = mapping(comparison.get("next_lower_fee"))
        if tier.get("date") and tier.get("within_sale_window") is False:
            st.caption(f"下一公示降费日 {tier['date']} 未落在本次可比较的卖出窗口内，不把它提前成尚未降费的日期。")
        st.caption("节费盈亏边界是算术，不是预测或必须等待；未来费表、开放日与应收款口径采用本次明示假设。")


def render_reconstructed(history: Mapping) -> None:
    st.markdown("**历史回算：当时这样判断，后来怎样**")
    if history.get("kind") == "calendar_cash_fixed_fee_nav_history":
        st.caption(f"统一费用与批次年龄假设下的 {history.get('horizon')} 笔实际净值回算；"
                   "不是历史渠道费率，也不是严格自然日交易策略。")
    counts = mapping(history.get("counts"))
    if counts:
        st.caption(" · ".join(f"{name} {counts.get(key, 0)}" for key, name in (
            ("mature", "已能核对"), ("immature", "后续未走完"),
            ("nowcast", "入场早于可知时点"), ("invalid", "跨无效净值"), ("unavailable", "当时尚不能预测"),
            ("nonoverlapping_windows", "不重叠窗口"))))
    metrics = mapping(history.get("metrics"))
    if counts:
        _write(f"当时判断可以买的窗口 {counts.get('predicted_buys', 0)} 个，其中实际亏损 {counts.get('wrong_buys', 0)} 个；"
               f"当时判断退出的窗口 {counts.get('predicted_exits', 0)} 个，避开损失 {counts.get('avoided_losses', 0)} 个，错过上涨 {counts.get('missed_profits', 0)} 个。")
    paired = mapping(history.get("paired"))
    if paired.get("count"):
        for branch, label in (("flat", "新买/等待"), ("held", "持有/退出")):
            delta = mapping(mapping(paired.get("delta")).get(branch)).get("mean_realized_payoff")
            if delta is not None:
                _write(f"{label}：这段回算中，模型所选操作的单窗口平均盈亏比简单方法{'高' if delta > 0 else '低' if delta < 0 else '相同'}"
                       f"{' ' + _money(abs(delta)) + ' 元示例' if delta else ''}；{'尚未胜过简单方法' if delta <= 0 else '仅说明这一批历史窗口的结果'}。")
    if metrics:
        st.dataframe([{
            "情况": label,
            "可核对窗口": mapping(metrics.get(branch)).get("count"),
            "所选动作平均实际差额（元示例）": _money(mapping(metrics.get(branch)).get("mean_realized_advantage"), signed=True),
            "同批模型平均盈亏": _money(mapping(mapping(paired.get("model")).get(branch)).get("mean_realized_payoff"), signed=True),
            "同批简单方法平均盈亏": _money(mapping(mapping(paired.get("historical_baseline")).get(branch)).get("mean_realized_payoff"), signed=True),
        } for branch, label in (("flat", "准备新买"), ("held", "已经持有"))], hide_index=True, width="stretch")
        st.caption(f"与简单经验方法可同起点配对的窗口 {paired.get('count', 0)} 个。动作差额对照各自比较项；模型与简单方法的平均盈亏都相对相同本金计算，不是累计策略收益。")
    rows = history.get("rows") or []
    if rows:
        with st.expander("逐次看：预测、操作与后续实际结果"):
            st.dataframe([{
                "当时净值起点": row.get("origin_date"),
                "假设最早可知日": row.get("assumed_known_date"),
                "状态": {"mature": "已核对", "immature": "未走完", "nowcast": "入场早于可知时点", "invalid": "无效净值", "unavailable": "无预测"}.get(row.get("status"), row.get("status")),
                "预测涨跌": percent(mapping(row.get("prediction")).get("expected_change"), signed=True),
                "实际涨跌": percent(row.get("actual_change"), signed=True),
                "新买判断": {"buy": "新买", "wait": "留现金"}.get(row.get("flat_action"), "无预测"),
                "已有判断": {"hold": "持有", "sell": "退出"}.get(row.get("held_action"), "无预测"),
                "新买/等待的实际差额": _money(mapping(mapping(row.get("actions")).get("flat")).get("realized_advantage"), signed=True),
                "持有/退出的实际差额": _money(mapping(mapping(row.get("actions")).get("held")).get("realized_advantage"), signed=True),
            } for row in rows], hide_index=True, width="stretch")
    st.caption("这是按当时已成熟净值样本重建的检验，不是当年真正保存的预测，也不包含今天的AI研究。费用使用本次统一假设；重叠窗口不连乘成年化收益。")


def _connection(assessment: Mapping, scenario: Mapping) -> dict:
    thesis = mapping(mapping(assessment.get("research_thesis")).get("payload"))
    cases = mapping(thesis.get("action_cases"))
    if not cases:
        return {}
    branches = mapping(mapping(scenario.get("decision")).get("branches"))
    explanations = []
    for case_key, branch_key, label, positive in (
        ("new_buy", "flat", "新买", {"buy"}),
        ("held", "held", "持有", {"hold"}),
        ("addition", "flat", "追加", {"add"}),
    ):
        case = mapping(cases.get(case_key))
        action = mapping(branches.get(branch_key)).get("action")
        if not action or action == "unavailable":
            continue
        short_positive = action in {"buy", "hold", "conditional_buy"}
        agrees = (case.get("action") in positive) == short_positive
        explanations.append(f"{label}：{'方向一致' if agrees else '短期与研究不同向'}；具体经营与价格理由见下面研究。")
    return {"research_date": thesis.get("research_date"), "horizon": deepcopy(thesis.get("horizon")),
            "lines": explanations}


def render_connection(value: Mapping) -> None:
    if not value:
        return
    horizon = mapping(value.get("horizon"))
    st.markdown("**基金研究与这个周期，是否同向**")
    st.caption(f"研究日 {value.get('research_date')} · 适用区间 {horizon.get('start')} ～ {horizon.get('end')}；它不是上面短期概率的另一种说法。")
    for explanation in lines(value.get("lines")):
        _write(explanation)
    st.caption("发生冲突时，不把较长期研究直接套成每个短期窗口都该买，也不把一次短期下跌当作经营理由已经失效。")


def _load_conditions(subject, saved, load_conditions):
    scope = st.session_state["forecast_subject_scope"]
    loaded = st.session_state.setdefault("forecast_conditions_loaded", set())
    if load_conditions is not None and scope not in loaded:
        try:
            stored = load_conditions(subject)
        except Exception:
            stored = None
        if isinstance(stored, Mapping):
            st.session_state.setdefault("forecast_condition_receipts", {})[scope] = dict(stored)
            conditions = stored.get("conditions", stored)
            if isinstance(conditions, Mapping):
                saved.update(deepcopy(dict(conditions)))
        loaded.add(scope)


def _render_current_changes(assessment: Mapping, previous: Mapping, branch: str) -> None:
    """Keep daily change copy on the chosen situation, not an evidence dump."""
    from rmbfund.assessment_changes import extract_assessment_changes

    changes = extract_assessment_changes(assessment, mapping(previous.get("assessment")))
    allowed = {
        f"decision_brief.cases.{branch}.preferred_action",
        f"decision_brief.cases.{branch}.reasons",
        "research_thesis.payload.change_evaluation", "execution_conditions",
        "trade_scenario.plan.calendar_days", "trade_scenario.purchase.nav_basis",
        "trade_scenario.fees.subscription_fee", "trade_scenario.fees.current_redemption_fee",
        "trade_scenario.fees.future_old_redemption_fee", "trade_scenario.fees.future_new_redemption_fee",
    }
    action_field = f"decision_brief.cases.{branch}.preferred_action"
    action_changes = [row["text"] for row in changes if row["field"] == action_field]
    selected = [row["text"] for row in changes if row["field"] in allowed and row["field"] != action_field]
    current_case = mapping(mapping(mapping(assessment.get("decision_brief")).get("cases")).get(branch))
    previous_case = mapping(mapping(mapping(mapping(previous.get("assessment")).get("decision_brief")).get("cases")).get(branch))
    advice_changes = []
    for field, label in (("advice_reason", "主要理由"), ("next_step", "下一步")):
        if current_case.get(field) != previous_case.get(field) and current_case.get(field):
            advice_changes.append(f"{label}更新：{current_case[field]}")
    selected = action_changes + advice_changes + selected
    st.markdown("**和上次保存相比**")
    st.caption(f"对照 {when(previous.get('saved_at'))} 保存的判断。")
    for text in selected[:2]:
        _write(text)
    if not selected:
        st.caption("当前情境的建议、主要理由和费用没有上述变化；新计算时间本身不代表改判。")


def _render_current_basis(assessment: Mapping, branch: str) -> None:
    """A few dated, cited facts; the complete research stays in the assessment."""
    brief = mapping(assessment.get("decision_brief"))
    case = mapping(mapping(brief.get("cases")).get(branch))
    thesis = mapping(mapping(assessment.get("research_thesis")).get("payload"))
    selected = set(case.get("fact_ids") or [])
    facts = [row for row in thesis.get("facts") or [] if row.get("id") in selected][:2]
    if not facts:
        return
    with st.expander("这次判断的依据"):
        st.caption(f"研究日期 {brief.get('research_date') or thesis.get('research_date') or '未注明'}；不是未来收益保证。")
        cited = set()
        for fact in facts:
            _write(fact.get("claim"))
            cited.update(fact.get("citation_ids") or [])
        for citation in [row for row in thesis.get("citations") or [] if row.get("id") in cited][:2]:
            url = citation.get("source_url") or citation.get("publisher_url")
            if url:
                st.link_button(citation.get("title") or "查看公开依据", url)


def render_standard(
    subject: Mapping, *, report: Mapping, build: Callable, save_displayed: Callable,
    holding=None, current_value=None, summary=None, save=None, preview=None,
    load_conditions=None, save_conditions=None, calculate=None, previous_assessment=None,
    theme: str = "light",
) -> dict | None:
    """Build a complete assessment, then render and save its two projections."""
    from rmbfund.buyer_view import build_buyer_view
    from rmbfund.ui.buyer_view import render_buyer_view, render_forecast_chart
    from rmbfund.ui.charts import build_forecast_chart
    from rmbfund.ui.decision import _save_conditions, _summary, anchor_currency
    from rmbfund.ui.investment import build_investment_analysis

    saved = forecast_ui._scope(subject)
    _load_conditions(subject, saved, load_conditions)
    scope = st.session_state["forecast_subject_scope"]
    report = dict(report)
    series = report.get("scenario_series") or []
    days = sorted({row["date"] for row in series if row.get("date")})
    chosen = st.session_state.get("trade_purchase_date") or saved.get("standard_purchase_date")
    purchase_date = chosen if chosen in days else days[max(0, len(days) - 21)] if days else None
    saved["standard_purchase_date"] = purchase_date

    with st.container(key="trade_controls"):
        st.session_state.setdefault("trade_selected_branch", saved.get("selected_branch", "flat"))
        branch = st.segmented_control("当前情境", ["flat", "held"], key="trade_selected_branch",
            format_func=lambda value: {"flat": "还没买", "held": "已经持有"}[value],
            selection_mode="single", required=True)
        saved["selected_branch"] = branch
        st.session_state.setdefault("trade_calendar_days", saved.get("calendar_days", 30))
        calendar_days = st.slider("往后看几天（自然日）", min_value=1, max_value=90,
                                  step=1, key="trade_calendar_days")
        saved["calendar_days"] = calendar_days
        if calculate is not None:
            try:
                report.update(calculate(calendar_days=calendar_days, lookup_only=True))
                if report.get("series") is not None:
                    series = report["series"]
            except (ValidationError, ValueError, TypeError, KeyError):
                report.update(status="unavailable", prediction=None,
                              reasons=["这个日期计划暂不能计算；原研究保留。"])
        plan = deepcopy(mapping(report.get("trade_plan")))
        if not plan:
            plan = build_trade_plan(decision_at=now_iso(), calendar_days=calendar_days,
                anchor_date=days[-1] if days else None)
        horizon = plan.get("horizon") or 0
        saved["horizon"] = horizon
        st.session_state["forecast_selected_horizon"] = horizon
        st.caption(f"本次判断 {plan.get('decision_date')} · 观察至 {plan.get('deadline')}；到时复核，不是必须卖出。")

        with st.popover("日期与费用设置"):
            if branch == "held":
                if days:
                    st.session_state["trade_purchase_date"] = purchase_date
                    purchase_date = st.select_slider("假设买入净值日", options=days, key="trade_purchase_date")
                    saved["standard_purchase_date"] = purchase_date
                else:
                    st.caption("还没有可选的实际净值日；不编造买入价格。")
            st.caption("假设日期不代表真实成交记录。费用从份额确认日起算；公示费率不等于购买渠道的实际折扣。")
            for key, initial, label in (
                ("old_confirmation_date", purchase_date or plan["decision_date"], "已有份额模拟确认日"),
                ("new_confirmation_date", plan["new_confirmation_date"], "本次新买或追加的模拟确认日"),
            ):
                widget_key, auto_key = f"trade_{key}", f"trade_auto_{key}"
                previous = st.session_state.get(auto_key)
                if widget_key not in st.session_state:
                    chosen = saved.get(key) if saved.get(key) != saved.get(f"{key}_auto") else None
                    st.session_state[widget_key] = date.fromisoformat(chosen or initial)
                elif st.session_state[widget_key] == previous:
                    st.session_state[widget_key] = date.fromisoformat(initial)
                st.session_state[auto_key] = date.fromisoformat(initial)
                saved[f"{key}_auto"] = initial
                saved[key] = st.date_input(label, key=widget_key).isoformat()
            plan = adjust_new_confirmation_date(plan, saved["new_confirmation_date"])
            overrides = {}
            for key, label in (("subscription_fee", "模拟申购费率（%，留空用公示或示例）"),
                               ("redemption_fee", "已有份额现在卖的费率（%，留空按阶梯）"),
                               ("future_old_redemption_fee", "已有份额计划卖的费率（%，留空按阶梯）"),
                               ("future_new_redemption_fee", "新买份额计划卖的费率（%，留空按阶梯）")):
                st.session_state.setdefault(f"trade_{key}", saved.get(f"trade_{key}", ""))
                value = st.text_input(label, key=f"trade_{key}")
                saved[f"trade_{key}"] = value
                try:
                    overrides[key] = forecast_ui.parse_fee(value)
                except ValueError as exc:
                    st.warning(str(exc))
                    overrides[key] = None
            if save_conditions is not None:
                st.button("记住这些设置", key="forecast_save_conditions", on_click=_save_conditions,
                    args=(save_conditions, dict(subject), deepcopy(saved), scope))
        fees = resolve_plan_fees(mapping(report.get("scenario_fees")), plan=plan,
            old_confirmation_date=saved["old_confirmation_date"],
            new_confirmation_date=saved["new_confirmation_date"],
            subscription_override=overrides["subscription_fee"],
            current_redemption_override=overrides["redemption_fee"],
            future_old_redemption_override=overrides["future_old_redemption_fee"],
            future_new_redemption_override=overrides["future_new_redemption_fee"])
        if calculate is not None and st.button("计算本周期判断", key="trade_calculate", type="primary",
                disabled=plan.get("status") != "ready",
                help="只用已有资料本地计算这个日期计划；不联网、不调用研究API，也不提交买卖。"):
            try:
                with st.spinner("用已有资料计算这个日期计划…"):
                    calculated = calculate(calendar_days=calendar_days, include_history=False)
                report.update(dict(calculated))
                if calculated.get("series") is not None:
                    series = calculated["series"]
            except (ValidationError, ValueError, TypeError, KeyError):
                st.error("这次本地计算未完成；原研究和数据日期保留。")

    prediction = mapping(report.get("prediction"))
    status = "ready" if report.get("status") == "ready" and prediction else "unavailable"
    if prediction and report.get("selected_model_id") and model_family(report["selected_model_id"]) != model_family(prediction.get("model_id")):
        status = "model_mismatch"
    if status != "ready":
        prediction = {}
    evidence = _summary(summary, horizon) if prediction and horizon > 0 else {"status": "missing"}
    try:
        scenario = build_calendar_scenario(prediction, subject=subject, plan=plan, series=series,
            purchase_date=purchase_date, fee_basis=fees)
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        st.info(markdown_text(str(exc)))
        return None
    calculation = mapping(report.get("calculation"))
    if not calculation and prediction:
        calculation = {"kind": "saved_public_forecast" if prediction.get("record_id") else "provided_forecast",
                       "actual_issue": bool(prediction.get("record_id")),
                       "record_id": prediction.get("record_id")}
    sale_options = {"fee_record": mapping(report.get("scenario_fees")),
                    "old_confirmation_date": saved["old_confirmation_date"],
                    "current_redemption_override": overrides["redemption_fee"],
                    "future_old_redemption_override": overrides["future_old_redemption_fee"]}
    if calculate is not None:
        try:
            report["sale_comparison"] = calculate(calendar_days=calendar_days,
                sale_options=sale_options, lookup_only=True).get("sale_comparison")
        except (ValidationError, ValueError, TypeError, KeyError):
            report["sale_comparison"] = None
    if not report.get("sale_comparison"):
        from rmbfund.timing_scenario import build_sale_date_comparison
        report["sale_comparison"] = build_sale_date_comparison(plan=plan, **sale_options)
    history = None
    historical = mapping(report.get("timing_history_report"))
    if historical.get("historical_forecasts"):
        from rmbfund.timing_scenario import summarize_calendar_history

        history = summarize_calendar_history(historical.get("series") or series,
            historical["historical_forecasts"], plan=plan, fee_basis=fees)
    investment_analysis = build_investment_analysis(report)
    assessment = build(prediction, subject=subject, holding=None, horizon=horizon,
        current_value=current_value, anchor_currency=anchor_currency(prediction), evidence_summary=evidence,
        investment_analysis=investment_analysis, research_thesis=report.get("research_thesis"),
        fund_comparison=report.get("fund_comparison"), forecast_status=status,
        execution_conditions=report.get("execution_conditions"), online_context=report.get("online_context"),
        subscription_fee=fees.get("subscription_fee"), redemption_fee=fees.get("current_redemption_fee"),
        trade_scenario=scenario, decision_rationale=report.get("decision_rationale"), include_decision_brief=True)
    assessment.update(schema_version=7, policy_version=POLICY_VERSION, mode="buyer_decision",
                      trade_scenario=scenario, calculation=deepcopy(calculation),
                      reconstructed_history=deepcopy(history), personal_conversion=None,
                      sale_comparison=deepcopy(report.get("sale_comparison")), selected_branch=branch)
    assessment["trade_research_connection"] = _connection(assessment, scenario)
    from rmbfund.assessment_replay import freeze_simulation_reference
    cases = mapping(mapping(assessment.get("decision_brief")).get("cases"))
    assessment["simulation_reference"] = freeze_simulation_reference(scenario, branches={
        "flat": {"action": mapping(cases.get("flat")).get("preferred_action"), "reference_action": "wait"},
        "held": {"action": {"redeem": "exit"}.get(mapping(cases.get("held")).get("preferred_action"),
                    mapping(cases.get("held")).get("preferred_action")), "reference_action": "hold"},
    })
    assessment["buyer_view"] = build_buyer_view(assessment, branch=branch)
    assessment["forecast_chart"] = build_forecast_chart(assessment, series=series)

    render_buyer_view(assessment["buyer_view"], key="decision_card")
    if branch == "held":
        purchase = mapping(scenario.get("purchase"))
        if purchase.get("change") is not None:
            st.caption(f"假设 {purchase.get('date')} 买入，截至 {purchase.get('latest_date')} 净值变化 "
                       f"{percent(purchase['change'], signed=True)}；这是过去涨跌，不是个人盈亏。")
    render_forecast_chart(assessment["forecast_chart"], key="decision_forecast_chart", theme=theme)
    _render_current_basis(assessment, branch)
    if previous_assessment is not None:
        try:
            previous = previous_assessment(subject)
        except Exception:
            previous = None
        if previous:
            _render_current_changes(assessment, previous, branch)
    if save is not None:
        st.button("保存这次判断", key="forecast_save_assessment", on_click=save_displayed,
                  args=(save, assessment, scope), help="保存刚才展示的建议、图与依据，不提交买卖。")
    for name in ("forecast_save_notice", "forecast_conditions_notice"):
        notice = st.session_state.pop(name, None)
        if notice and notice["scope"] == scope:
            (st.success if notice["ok"] else st.error)(notice["message"])
    st.caption("只有点击保存才写入判断历史；不自动新增持仓或提交买卖。")
    return assessment

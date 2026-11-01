"""Decision-first views of one saved forecast and one local assessment object.

The host reads the public report once. This component only interprets that report;
its explicit save button passes the same displayed object to the private writer.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any

import streamlit as st

from rmbfund.personal_assessment import build_assessment
from rmbfund.ui import forecast as forecast_ui
from rmbfund.ui.formatting import lines, mapping, markdown_text, percent, when


_ACTIONS = {
    "buy": "倾向买入", "wait": "暂不买入", "hold": "倾向继续持有", "sell": "倾向退出",
    "conditional_buy": "有买入空间，需对照费用", "conditional_sell": "倾向退出，需对照费用",
    "unavailable": "无对应比较",
}


def _write(value: Any) -> None:
    st.markdown(markdown_text(value))


def _money(value: Any, currency: str | None, *, signed: bool = False) -> str:
    if value is None:
        return "未提供"
    amount = Decimal(str(value))
    prefix = "¥" if currency == "CNY" else f"{currency} " if currency else ""
    return prefix + (f"{amount:+,.2f}" if signed else f"{amount:,.2f}")


def anchor_currency(prediction: Mapping) -> str | None:
    """Use this forecast's anchor provenance, never today's NAV or assumed CNY."""
    anchor = mapping(prediction.get("anchor"))
    for record in (anchor, mapping(anchor.get("source")), mapping(anchor.get("provenance"))):
        currency = record.get("currency")
        if isinstance(currency, str) and len(currency) == 3 and currency.isupper():
            return currency
    return None


def _summary(read: Callable[..., Mapping] | Mapping | None, horizon: int) -> dict:
    if read is None:
        return {"status": "missing"}
    try:
        return dict(read(horizon=horizon, role="primary") if callable(read) else read)
    except Exception:
        return {"status": "error"}


def _evidence_line(summary: Mapping) -> None:
    counts = mapping(summary.get("counts"))
    if not counts:
        st.caption("仍在验证 · 暂无可读的真实前向成绩汇总，不用历史回算代替。")
        return
    count = counts.get("recorded_forward", 0)
    windows = counts.get("forward_windows", 0)
    st.caption(f"仍在验证 · 所选期限真实前向样本 {windows} 个去重窗口（已记录发行结果 {count} 次）；不是投资胜率。")


def _anchor_and_amounts(assessment: Mapping) -> None:
    st.subheader("对我的金额意味着什么")
    current = mapping(mapping(assessment.get("amounts")).get("current"))
    if current:
        _write(f"当前估算市值 {_money(current.get('market_value'), current.get('currency'))}，"
               f"账面浮盈亏 {_money(current.get('profit'), current.get('currency'), signed=True)}；"
               f"使用最近公布净值 {current.get('nav_date') or '未提供'}。")
    elif assessment.get("holding_status") == "unknown":
        st.caption("还没有个人持仓金额；补充份额与成本后可作本地换算。")
    anchor = mapping(assessment.get("anchor"))
    head = mapping(anchor.get("head"))
    basis = mapping(anchor.get("basis"))
    amounts = mapping(mapping(assessment.get("amounts")).get("anchor"))
    with st.container(border=True, key="forecast_anchor_amounts"):
        st.markdown("**另一起点：从已公布净值看持仓金额**")
        st.caption(f"净值所属日 {basis.get('date') or '未提供'} · 起点净值 {basis.get('nav') or '未提供'} · "
                   f"{basis.get('currency') or '币种未注明'} · 之后{assessment['horizon']}笔净值。不是上方执行情景，也不是从今天起算。")
        if head:
            _write(f"这一起点的上涨概率 {percent(head.get('up_probability'))}，预期涨跌 {percent(head.get('expected_change'), signed=True)}，"
                   f"80%范围 {percent(head.get('lower'), signed=True)} ～ {percent(head.get('upper'), signed=True)}。")
        else:
            st.caption("没有对应的净值起点预测，不借用执行情景或其他期限。")
        if amounts.get("status") == "ready":
            currency = amounts.get("currency")
            expected = mapping(amounts.get("expected"))
            _write(f"假设份额不变，按这个净值起点的持仓价值 {_money(amounts.get('basis_value'), currency)}，"
                   f"预期期末账面值 {_money(expected.get('value'), currency)}，"
                   f"相对这个起点变动 {_money(expected.get('change'), currency, signed=True)}。")
            lower, upper = mapping(amounts.get("lower")), mapping(amounts.get("upper"))
            _write(f"80%金额范围 {_money(lower.get('value'), currency)} ～ {_money(upper.get('value'), currency)}。")
            if expected.get("profit") is not None:
                _write(f"与已填写持仓成本比较，预期账面浮盈亏 {_money(expected['profit'], currency, signed=True)}。")
        for reason in lines(amounts.get("reasons")):
            st.caption(markdown_text(reason))
    execution_holding = mapping(mapping(assessment.get("amounts")).get("execution_holding"))
    for reason in lines(execution_holding.get("reasons")):
        st.caption(markdown_text(reason))
    budget = mapping(mapping(assessment.get("inputs")).get("budget"))
    if budget:
        buy = mapping(mapping(assessment.get("amounts")).get("new_buy"))
        cash_plan = buy.get("end_value_basis") == "cash_and_redemption_receivable"
        st.markdown("**如果按填写金额新买后按计划卖出**" if cash_plan else "**如果按填写金额新买**")
        if buy.get("status") == "ready":
            currency = buy.get("currency")
            expected = mapping(buy.get("expected"))
            if cash_plan:
                _write(f"拟投入 {_money(buy.get('basis_value'), currency)}；计入申购与计划赎回费后，"
                       f"预期卖出净额 {_money(expected.get('value'), currency)}（含赎回应收款），"
                       f"相对投入变动 {_money(expected.get('change'), currency, signed=True)}。")
            else:
                _write(f"拟投入 {_money(buy.get('basis_value'), currency)}；计入申购费后，"
                       f"成交之后{assessment['horizon']}笔净值的预期账面值 {_money(expected.get('value'), currency)}，"
                       f"相对投入变动 {_money(expected.get('change'), currency, signed=True)}。")
            _write(f"80%金额范围 {_money(mapping(buy.get('lower')).get('value'), currency)} ～ "
                   f"{_money(mapping(buy.get('upper')).get('value'), currency)}。")
        for reason in lines(buy.get("reasons")):
            st.caption(markdown_text(reason))


def _details(assessment: Mapping, prediction: Mapping) -> None:
    with st.expander("判断依据与历史检验"):
        st.markdown("**这个周期的预测根据什么**")
        for reason in lines(mapping(assessment.get("execution")).get("reasons")):
            _write(reason)
        st.caption(markdown_text(forecast_ui.model_source_text({"prediction": mapping(assessment.get("forecast"))})))
        saved_model = mapping(assessment.get("forecast"))
        st.caption(markdown_text(f"原方法ID {saved_model.get('model_id') or '未注明'} · "
                                 f"模型版本 {saved_model.get('model_hash') or '未注明'} · "
                                 f"输入版本 {saved_model.get('input_hash') or '未注明'}。"))
        forecast_ui.render_factor_details(saved_model, mapping(assessment.get("execution")), key="decision_factor_details")
        decision = mapping(assessment.get("decision"))
        scenarios = decision.get("scenarios") or []
        if scenarios:
            st.markdown("**全部已保存执行情景**")
            st.dataframe([{
                "假定延迟（笔净值）": row.get("entry_lag"),
                "采用情景": "主判断" if row.get("is_primary") else "其他延迟",
                "成交后期限（笔）": row.get("horizon"),
                "上涨概率": percent(row.get("up_probability")),
                "预期净值变化": percent(row.get("market_expected_change"), signed=True),
                "新买相对现金": percent(row.get("buy_advantage_vs_cash"), signed=True, digits=2),
                "持有相对退出": percent(row.get("hold_advantage_vs_exit"), signed=True, digits=2),
                "若准备买入": _ACTIONS.get(row.get("flat_action"), "无对应比较"),
                "若已持有": _ACTIONS.get(row.get("held_action"), "无对应比较"),
            } for row in scenarios], hide_index=True, width="stretch")
        for assumption in lines(decision.get("assumptions")):
            _write(assumption)
        if decision.get("execution_note"):
            st.caption(markdown_text(decision["execution_note"]))
        if assessment.get("entry_lag") is not None:
            forecast_ui.render_validation(prediction, horizon=assessment["horizon"],
                entry_lag=assessment["entry_lag"], key="decision_validation")
        else:
            st.caption("原预测未注明主要延迟，没有对应的执行头检验；不借用20笔净值起点的检验状态。")
        st.markdown("**净值起点预测的理由（不是执行判断理由）**")
        for reason in lines(mapping(assessment.get("anchor")).get("reasons")):
            _write(reason)


def _save_displayed(save: Callable[[dict], Mapping], assessment: dict, scope: tuple) -> None:
    """The callback captures the previous render's object before Streamlit reruns."""
    try:
        receipt = save(assessment)
    except Exception:
        st.session_state["forecast_save_notice"] = {
            "scope": scope, "ok": False,
            "message": "这次判断没有保存成功；当前展示和输入仍在，没有提交买卖。",
        }
    else:
        st.session_state["forecast_save_notice"] = {
            "scope": scope, "ok": True,
            "message": f"已保存刚才展示的判断 · {when(receipt.get('saved_at'))}（北京时间）。没有提交买卖。",
        }


def _save_conditions(save: Callable, subject: Mapping, conditions: dict, scope: tuple) -> None:
    try:
        receipt = save(subject, conditions)
    except Exception:
        st.session_state["forecast_conditions_notice"] = {
            "scope": scope, "ok": False, "message": "这次条件没有保存成功；当前填写仍在，可稍后再试。",
        }
    else:
        st.session_state.setdefault("forecast_condition_receipts", {})[scope] = receipt
        st.session_state["forecast_conditions_notice"] = {
            "scope": scope, "ok": True,
            "message": "已记住这次日期与费用设置。下次可改，不代表渠道当日条件已经确认。",
        }


def render_decision(
    subject: Mapping, *, report: Mapping, holding: Mapping | None = None,
    current_value: Mapping | None = None,
    summary: Callable[..., Mapping] | Mapping | None = None,
    save: Callable[[dict], Mapping] | None = None,
    models: Sequence[Mapping] | None = None,
    load_conditions: Callable[[Mapping], Mapping | None] | None = None,
    save_conditions: Callable[[Mapping, Mapping], Mapping] | None = None,
    preview: Callable[[Mapping], None] | None = None,
    calculate: Callable[..., Mapping] | None = None,
    previous_assessment: Callable[..., Mapping | None] | None = None,
    theme: str = "light",
) -> dict | None:
    """Return the standard scenario object actually displayed, not a next-rerun rebuild."""
    from rmbfund.ui.trade import render_standard

    return render_standard(subject, report=report, build=build_assessment,
        save_displayed=_save_displayed, holding=holding, current_value=current_value,
        summary=summary, save=save, preview=preview, load_conditions=load_conditions,
        save_conditions=save_conditions, calculate=calculate, previous_assessment=previous_assessment,
        theme=theme)

"""Compact views of saved NAV forecasts, with one passive read per render.

The host owns the daily update action. A writer is used only when an explicit
callback is supplied and its button is clicked; holding and fee inputs never
change the saved market distribution or leave this component.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

import streamlit as st

from rmbfund.forecast_decision import build_decision
from rmbfund.forecast_models import INVESTMENT_MODEL_ID, NAV_MODEL_ID, model_catalog, model_family
from rmbfund.personal_assessment import explain_forecast_head, format_factor_value, model_name
from rmbfund.ui.formatting import lines, mapping, markdown_text, number, percent, when
_HORIZONS = {
    1: "之后1笔净值",
    5: "之后5笔净值",
    20: "之后20笔净值（通常约一个月）",
}
_VALIDATION_SUMMARIES = {
    "better": "历史表现：这版之后20笔净值的预测，在所检验的历史中误差较小，但不代表买卖会盈利。",
    "not_better": "历史表现：这版之后20笔净值（约一个月）的预测，尚未胜过简单方法。",
    "unverified": "历史表现：之后20笔净值的已检验样本还不够，暂不清楚是否比简单方法更准。",
}


def _write(value: Any) -> None:
    st.markdown(markdown_text(value))


def _scope(subject: Mapping | None) -> dict:
    """Keep unpersisted local inputs per public fund, including tab round trips."""
    item = subject or {}
    identity = tuple(str(item.get(key, "")) for key in ("id", "symbol", "kind", "share_class", "exchange"))
    states = st.session_state.setdefault("forecast_fund_inputs", {})
    if st.session_state.get("forecast_subject_scope") != identity:
        for key in ("forecast_horizon", "forecast_selected_horizon", "forecast_subscription_fee",
                    "forecast_redemption_fee", "forecast_budget", "forecast_fee_values"):
            st.session_state.pop(key, None)
        for key in list(st.session_state):
            if key.startswith(("buyer_", "trade_")):
                st.session_state.pop(key, None)
        st.session_state["forecast_subject_scope"] = identity
    saved = states.setdefault(identity, {"horizon": 20, "subscription_fee": "", "redemption_fee": "", "budget": ""})
    st.session_state["forecast_fee_values"] = saved
    return saved


def parse_fee(value: str) -> float | None:
    """Percent input -> proportional fee; a blank is unknown, never zero."""
    if not value.strip():
        return None
    try:
        rate = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("请填写百分数，例如0.15；未知可留空。") from exc
    if not rate.is_finite() or not 0 <= rate < 100:
        raise ValueError("费率须在0%（含）到100%（不含）之间；未知可留空。")
    return float(rate / 100)


def _empty(subject: Mapping | None, as_of: str, reason: str, *, status: str = "missing") -> dict:
    return {"status": status, "product": subject, "evaluated_at": as_of,
            "prediction": None, "reasons": [reason], "evidence": {}, "refresh": {}}


def read_forecast(read: Callable[[], dict] | None, subject: Mapping | None, as_of: str) -> dict:
    """One passive callback, with no stale success substituted for a read failure."""
    if read is None:
        return _empty(subject, as_of, "还没有保存的预测，可点击“更新数据与预测”后查看。")
    try:
        result = read()
        if not isinstance(result, Mapping):
            raise TypeError("Forecast reader must return a mapping")
        return dict(result)
    except Exception:
        return _empty(subject, as_of, "暂时读不到已保存的预测。请稍后重试，未自动联网或重新计算。", status="error")


def public_receipt(report: Mapping) -> dict:
    """Only public operation metadata enters the host log, never the full model."""
    prediction = mapping(report.get("prediction"))
    refresh = mapping(report.get("refresh"))
    nav_refresh = mapping(refresh.get("nav"))
    return {key: str(value)[:160] for key, value in {
        "status": report.get("status"), "evaluated_at": report.get("evaluated_at"),
        "generated_at": prediction.get("generated_at"), "model_id": prediction.get("model_id"),
        "nav_date": mapping(prediction.get("anchor")).get("date"),
        "refresh_status": refresh.get("status") or nav_refresh.get("status"),
    }.items() if value is not None}


def _update(update: Callable[[], dict], action: Callable | None) -> dict | None:
    captured: dict[str, dict] = {}

    def run() -> dict:
        report = update()
        if not isinstance(report, Mapping):
            raise TypeError("Forecast update must return a mapping")
        captured["report"] = dict(report)
        return public_receipt(report)

    if action is not None:
        ok, _ = action("更新数据与预测", run)
        return captured.get("report") if ok else None
    try:
        with st.spinner("更新净值并计算预测…"):
            run()
    except Exception:
        st.error("这次预测更新没有完成，请稍后重试。没有提交买卖。")
        return None
    return captured["report"]


def _refresh_failed(report: Mapping) -> bool:
    refresh = mapping(report.get("refresh"))
    nav = mapping(refresh.get("nav"))
    return any(item.get("status") in {"error", "failed", "blocked", "unavailable"}
               for item in (refresh, nav))


def _unavailable(report: Mapping, subject: Mapping | None, *, update_label: str | None = None) -> None:
    if subject is None:
        st.info("请先选择一只基金，再查看预测。")
        return
    status = report.get("status", "missing")
    input_status = report.get("input_status")
    next_step = update_label or "更新数据与预测"
    defaults = {
        "missing": f"还没有保存的预测，可点击“{next_step}”后查看。",
        "insufficient": "可用净值记录还不足，尚不能估算上涨概率和涨跌范围。",
        "unsupported": "这种产品目前尚未建模，不会套用其他产品的预测。",
        "stale": "已保存预测的数据基础已改变，需要更新后重新计算。",
        "error": "暂时读不到可用预测，或这次更新没有完成。请稍后重试；没有把旧预测当成新结果。",
    }
    if input_status == "missing_investment_inputs":
        message = "尚无成功保存的市场、行业、汇率和利率数据；请明确更新投资因素。"
    elif input_status == "unsupported_mapping":
        message = "这只基金尚未核对投资因素与实际投向的对应关系，不会套用其他基金。"
    elif report_model_id(report) == INVESTMENT_MODEL_ID and status in {"insufficient", "unsupported"}:
        message = ("投资因素资料或可检验的样本尚不足；需要核对市场、行业、汇率、利率及本基金投向映射，"
                   "不会借用净值模型的概率或把未知因素按零处理。")
    else:
        message = defaults.get(status, defaults["missing"])
        if status == "insufficient":
            count = number(mapping(mapping(report.get("evidence")).get("nav")).get("observation_count"))
            if count is not None:
                message = f"当前有{count:g}笔可用净值，还不足以估算上涨概率和涨跌范围。"
    # Keep source diagnostics/exception details out of daily failure copy.
    show = st.error if status == "error" else st.info
    show(message)


def _catalog(models: Sequence[Mapping] | None) -> list[Mapping]:
    """Return every advertised model family and its availability metadata."""
    source = model_catalog() if models is None else models
    result = {}
    for raw in source:
        item = dict(mapping(raw))
        model_id = item.get("model_id") or item.get("id")
        if not model_id:
            continue
        family = item.get("family") or model_family(model_id)
        # v1/v2 are output versions of one NAV method, not extra algorithms.
        if family in result and model_id != NAV_MODEL_ID:
            continue
        label = item.get("label") or item.get("name") or model_name(model_id)
        result[family] = {**item, "id": model_id, "label": label, "family": family,
                          "available": item.get("available", True)}
    return list(result.values())


def _anchor_heads(prediction: Mapping) -> dict[int, Mapping]:
    return {item["horizon"]: item for raw in prediction.get("forecasts", [])
            if (item := mapping(raw)).get("horizon") in _HORIZONS and item.get("entry_lag") == 0}


def _metrics(head: Mapping) -> None:
    columns = st.columns(3)
    columns[0].metric("上涨概率", percent(head.get("up_probability")), border=True,
                      help="在这个预测范围里，净值比起点更高的估计概率；不是账户赚钱的概率。")
    columns[1].metric("预期涨跌", percent(head.get("expected_change"), signed=True), border=True,
                      help="这个预测分布的平均净值变化，不是保证收益，也不是你的持仓盈亏。")
    interval = f"{percent(head.get('lower'), signed=True)} ～ {percent(head.get('upper'), signed=True)}"
    columns[2].metric("80%预测范围", interval, border=True,
                      help="预测分布中间80%的涨跌范围，实际仍可能越界；不是平均涨跌的置信区间。")
    st.caption(f"同一预测范围：下跌概率 {percent(head.get('down_probability'))} · "
               f"持平概率 {percent(head.get('flat_probability'))}。净值涨跌不是含分红总回报，也不是账户收益。")


def format_driver_value(feature: Mapping | str, value: Any = None) -> str:
    """Shared saved-factor formatter for current and history views."""
    return format_factor_value(feature, value)


# Backward-compatible private spelling used by older view tests.
def _feature_value(name: str, value: Any) -> str:
    return format_driver_value(name, value)


def report_model_id(report: Mapping) -> str | None:
    """Saved prediction identity wins; a selection label cannot relabel its fit."""
    return mapping(report.get("prediction")).get("model_id") or report.get("selected_model_id")


def model_source_text(report: Mapping) -> str:
    """Truthful source gate for the selected report, including unavailable fits."""
    prediction = mapping(report.get("prediction"))
    model_id = report_model_id(report)
    if model_family(model_id) == model_family(NAV_MODEL_ID):
        return ("净值趋势预测只参考本基金已保存净值的涨跌、波动和回撤；资料少时参考相同期限的历史分布。"
                "市场、行业、汇率、利率、盈利、估值和经理资料没有作为额外特征进入这次概率；"
                "披露权重可用于独立的条件计算，但不会改写上方分布。")
    if model_id == INVESTMENT_MODEL_ID:
        metadata = [*(mapping(prediction.get("model_definition")).get("features") or []),
                    *(prediction.get("features") or []), *(prediction.get("drivers") or [])]
        roles = {row.get("economic_role") for row in metadata if isinstance(row, Mapping)}
        names = [label for role, label in (
            ("nav_history", "本基金净值历史"), ("market", "新兴市场股票价格"),
            ("industry", "半导体相对市场的强弱"), ("fx", "美元兑人民币中间价"),
            ("discount_rate", "美国十年期国债收益率"),
        ) if role in roles]
        fitted = ("这份投资因素预测实际拟合了" + "、".join(names) + "。" if names else
                  "投资因素预测需要实际对齐净值、市场、行业、汇率和利率资料；当前没有可据以解释的已保存特征，不套用净值模型。")
        return (fitted + "投向披露只用于核对参照是否相关，不是历史仓位，权重也没有直接当作拟合特征。"
                "盈利、估值和经理能力没有进入拟合，仍是待核对的条件；下方条件计算不生成概率，也不能覆盖上方分布。")
    return (f"本次方法为{model_name(model_id)}，只按这份记录保存的特征解释，不因基金名称自动补写输入。"
            "独立条件计算不生成模型概率，也不覆盖已保存的市场分布。")


def _source_gate(prediction: Mapping) -> str:
    return model_source_text({"prediction": prediction})


def render_factor_details(prediction: Mapping, view: Mapping, *, key: str) -> None:
    """Optional full table; the novice view keeps at most three dated reasons."""
    head = mapping(view.get("head"))
    identity = mapping(view.get("identity"))
    if not head:
        return
    explanation = explain_forecast_head(
        prediction, {**head, "drivers": view.get("drivers", [])},
        horizon=identity.get("horizon", head.get("horizon")),
        entry_lag=identity.get("entry_lag", head.get("entry_lag")),
        role=identity.get("role", "anchor"), all_factors=True,
    )
    factors = explanation.get("factor_reasons", [])
    if not factors:
        return
    st.markdown("**全部已保存因素（同一期限与起点）**")
    st.dataframe([{
        "因素": row["label"], "当时数值": row["formatted_value"], "模型关联方向": row["direction"],
        "对数位置贡献（不是收益率）": f"{float(row['contribution_log_return']):+.6f}",
        "数据日期": row["observation_dates"], "原始单位": row.get("unit") or "未注明",
        "与本基金的关系": row["connection_to_fund"],
    } for row in factors], hide_index=True, width="stretch", key=key)
    st.caption("贡献是各因素标准化后乘以该预测头的拟合系数，针对对数预测位置；还包含常数项和历史误差，"
               "不能把这些数字相加当作净值预期涨跌。相对行业项是对数变化差，不是基金收益或两项涨跌的百分点差。")


def _drivers(prediction: Mapping, head: Mapping, *, entry_lag: int = 0,
             title: str = "模型怎么看") -> None:
    st.markdown(f"**{title}**")
    explanation = explain_forecast_head(
        prediction, head, horizon=head.get("horizon"), entry_lag=entry_lag,
        role="anchor" if entry_lag == 0 else "execution",
    )
    if head.get("method") == "empirical_historical":
        _write("这次先参考历史上相同期限的涨跌分布，没有把近期走势单独拆成理由。")
        explanations = {
            "feature_window_crosses_invalid_event_or_shorter_than_60": "近期可用于比较的净值记录还不完整，暂不把它们当成可靠的近期规律。",
            "insufficient_matured_training_labels": "能核对结果的历史样本还不够，等更多净值公布后才可能用近期走势作判断。",
            "insufficient_matured_prequential_residuals": "近期走势方法还缺少足够的已实现检验结果，所以这次先用历史分布。",
        }
        st.caption(explanations.get(head.get("fallback_reason"), "这是同一模型在资料不足时的处理，不是另一个独立模型。"))
        return
    shown = explanation.get("factor_reasons", [])
    for item in shown:
        reason = item.get("reason")
        if reason:
            st.markdown(f"- {markdown_text(reason)}")
    if shown:
        st.caption("支持和反对方向均来自这个具体期限、成交延迟和模型版本保存的拟合关联，不是涨跌因果；相关因素可能重叠。")
    else:
        st.caption("这次没有保存与所选期限对应的单项理由，不借用其他期限或成交情景的理由，也不另编市场原因。")


def _validation_head(prediction: Mapping, horizon: int, entry_lag: int = 0) -> Mapping:
    validation = mapping(prediction.get("validation"))
    return next((head for raw in validation.get("heads", [])
                 if (head := mapping(raw)).get("horizon") == horizon
                 and head.get("entry_lag") == entry_lag), {})


def _validation_summary(prediction: Mapping, horizon: int = 20, entry_lag: int = 0) -> str:
    prefix = f"假定延迟{entry_lag}笔后成交、" if entry_lag else ""
    if not _validation_head(prediction, horizon, entry_lag):
        return f"历史表现：还没有{prefix}之后{horizon}笔净值的验证资料，暂不能判断这个期限是否比简单方法更准。"
    if horizon != 20 or entry_lag != 0:
        return f"历史表现：{prefix}之后{horizon}笔净值的检验数值见下方详情，不能据此直接判断买卖会盈利。"
    validation = mapping(prediction.get("validation"))
    return _VALIDATION_SUMMARIES.get(validation.get("status"), "历史表现：还没有保存的检验结果，不能据此认为预测比简单方法更准。")


def render_validation(
    prediction: Mapping | None, *, horizon: int = 20, entry_lag: int = 0,
    key: str = "forecast_validation",
) -> None:
    """Show the exact saved validation head; the global status is H20 lag0 only."""
    st.markdown("**过去预测得怎么样**")
    prediction = mapping(prediction)
    validation = mapping(prediction.get("validation"))
    _write(_validation_summary(prediction, horizon, entry_lag))
    st.caption("以下是所选模型的历史回算检验，不是实际发出预测后的成绩。")
    if prediction.get("model_id") == INVESTMENT_MODEL_ID:
        st.caption("投资因素历史采用来源日期滞后对齐，历史披露时点及修订版本未经验证；"
                   "当前投向映射也是事后核对，不是历史仓位。检验不证明买卖获利，也不证明优于净值模型。")
    primary = _validation_head(prediction, horizon, entry_lag)
    period = mapping(mapping(primary.get("periods")).get("heldout_2026"))
    if number(period.get("forecast_count")):
        rows = []
        for name, label in (("model", model_name(prediction.get("model_id"))), ("historical_frequency", "按历史涨跌频率估计"), ("no_change", "假设净值不变")):
            metric = mapping(period.get(name))
            if metric:
                error = number(metric.get("up_brier"))
                rows.append({"比较方法": label,
                             "上涨概率误差（越低越好）": f"{error:.3f}" if error is not None else "未提供",
                             "平均涨跌误差": percent(metric.get("mae"), digits=2),
                             "80%范围实际覆盖率": percent(metric.get("interval_coverage"))})
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", key=key)
        start, end = period.get("start"), period.get("end")
        if start and end:
            st.caption(f"设定检验区间：{start} 至 {end}；这里只比较起点之后{horizon}笔净值的结果，不代表每个日期都有样本。")
        count = period["forecast_count"]
        blocks = mapping(period.get("disjoint_blocks")).get("count", "未提供")
        st.caption(f"这组有{count}次历史检验、{blocks}段不重叠区间；相邻预测会覆盖同一段行情，次数不等于独立证据。")
        st.caption("平均涨跌误差是预测与实际相差多少；概率误差0代表完全准确，它不是亏损比例。80%范围覆盖率是实际涨跌落入该范围的比例。")
    elif primary:
        st.caption(f"之后{horizon}笔净值尚无可展示的2026年分项检验数值。")
    if validation.get("strict_unseen") is False:
        st.caption("这段历史已用于检查模型，不是事先完全没看过的新行情。")


def _holding_note(holding: str) -> None:
    st.markdown("**对你的持仓意味着什么**")
    notes = {
        "held": "你已记录持有：这里预测的是基金净值，不是你的账户盈亏；买入成本不会改变上面的上涨概率。",
        "flat": "你明确记录为未持有：上涨概率不等于今天申购能赚钱的概率，还要考虑实际成交净值与费用。",
        "unknown": "还没有明确持有情况，暂不判断该买还是该卖；未填份额不等于没有持有。",
    }
    _write(notes.get(holding, notes["unknown"]))


def fee_inputs(saved: dict) -> dict[str, float | None]:
    """Unpersisted inputs; no fee is inferred from missing or invalid text."""
    fees = {}
    for column, name, label in zip(st.columns(2), ("subscription_fee", "redemption_fee"),
                                   ("实际申购费率（%，可留空）", "现在卖出的赎回费率（%，可留空）")):
        with column:
            widget_key = f"forecast_{name}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = str(saved.get(name, ""))
            value = st.text_input(label, key=widget_key, placeholder="例如0.15；确认免费才填0")
            saved[name] = value
            try:
                fees[name] = parse_fee(value)
            except ValueError as exc:
                st.warning(str(exc))
                fees[name] = None
    return fees


def _fees(prediction: Mapping, holding: str, *, horizon: int) -> None:
    with st.expander("费用会改变判断吗", expanded=False):
        st.caption("不知道实际费率可留空，未知不会按0%；这里的条件比较不会改变上面的市场概率，也不会提交买卖。")
        fees = fee_inputs(st.session_state.setdefault("forecast_fee_values", {}))
        decision = build_decision(prediction, holding=holding, horizon=horizon, **fees)
        primary = mapping(decision.get("primary_head"))
        if not primary:
            st.info("缺少对应成交时间的预测，目前不能比较买卖费用；不会拿已公布净值起点的涨幅替代。")
            st.caption(f"需要之后{horizon}笔净值及原定主要成交延迟的执行预测；可用已保存净值显式更新。")
            return
        st.caption(f"这是另一起点：从已公布净值向后延迟{primary.get('entry_lag')}笔净值作为假定成交点，再看成交之后{horizon}笔净值。"
                   "期限与上方一致，但起点不同；不承诺实际成交日期。")
        branches = mapping(decision.get("branches"))
        selected = (holding,) if holding in {"held", "flat"} else ("held", "flat")
        for name in selected:
            branch = mapping(branches.get(name))
            if holding == "unknown":
                st.markdown("**若已持有**" if name == "held" else "**若准备买入**")
            _write(branch.get("label", "暂不能比较"))
            for reason in lines(branch.get("reasons")):
                _write(reason)
        if all(value is None for value in fees.values()):
            st.caption("还没填实际费率，暂不计算扣费后的个人比较；上面的费用条件不等于费率为0。")
        st.caption("以上按基金净值计价，不假设期末卖出；不同成交时间可能改变比较。实际费率、可否申赎和到账时间以渠道为准，已计入净值的运作费用不重复扣除。")


def render_forecast(
    subject: Mapping | None, *, as_of: str, read: Callable[[], dict] | None,
    update: Callable[[], dict] | None = None, action: Callable | None = None,
    live: bool = True, mode: str = "public", holding: str = "unknown",
    models: Sequence[Mapping] | None = None,
) -> None:
    saved_inputs = _scope(subject)
    holding = holding if holding in {"held", "flat", "unknown"} else "unknown"
    st.subheader("预测怎么看")
    catalog = _catalog(models)
    if not any(item.get("available") for item in catalog):
        st.caption("当前没有适用于这只基金的可用模型。")
    if not live or mode != "public":
        st.caption("演示只读：不更新预测。" if mode == "demo" else "只读查看已保存预测，不联网、不重新计算。")
    report = read_forecast(read, subject, as_of)
    if update is not None:
        allowed = bool(live and mode == "public" and subject and not subject.get("is_demo"))
        if st.button("更新数据与预测", key="forecast_update", type="primary", disabled=not allowed,
                     help="明确点击才获取净值并保存预测；不提交买卖。") and allowed:
            completed = _update(update, action)
            report = completed if completed is not None else _empty(subject, as_of, "这次更新没有完成。请稍后重试，未把此前预测当作新结果。", status="error")
            if report.get("status") == "ready" and not _refresh_failed(report):
                st.success("预测已更新；没有提交买卖。")
    if _refresh_failed(report):
        st.warning("最近更新没有成功。下面若有已保存预测，其净值起点和生成时间仍是原来的，不是这次的新结果。")
    prediction = mapping(report.get("prediction"))
    selected_model_id = report_model_id(report)
    if selected_model_id:
        selected = next((item for item in catalog if item.get("id") == selected_model_id), {})
        selected_label = selected.get("label") or model_name(selected_model_id)
        availability = " · 可用" if selected.get("available", True) else " · 当前不可用"
        st.caption(markdown_text(f"当前选择：{selected_label}{availability} · 方法ID {selected_model_id}"))
        st.caption(markdown_text(model_source_text(report)))
    if subject is None or report.get("status") != "ready" or not prediction:
        _unavailable(report, subject)
        return
    anchor = mapping(prediction.get("anchor"))
    st.caption(markdown_text(f"净值起点：{anchor.get('date', '未提供')}（净值所属日期） · 单位净值 {anchor.get('nav', '未提供')}；"
                             f"原预测生成于 {when(prediction.get('generated_at'))}（北京时间）。"))
    heads = _anchor_heads(prediction)
    if "forecast_horizon" not in st.session_state:
        st.session_state["forecast_horizon"] = saved_inputs["horizon"]
    horizon = st.selectbox("预测范围", list(_HORIZONS), format_func=_HORIZONS.get, key="forecast_horizon")
    saved_inputs["horizon"] = horizon
    st.session_state["forecast_selected_horizon"] = horizon
    st.caption("范围从这笔净值往后数，不是从今天起算；1笔指下一笔净值，不保证是明天，5笔和20笔也不是自然日。")
    head = heads.get(horizon)
    if head:
        _metrics(head)
    else:
        st.info(f"{_HORIZONS[horizon]}的预测暂不可用，没有用其他期限、成交情景或50%概率替代。")
    st.caption(_validation_summary(prediction, horizon))
    quality = mapping(prediction.get("data_quality"))
    if any(mapping(item).get("known_invalid") is False for item in quality.get("large_nav_moves", [])):
        st.warning("部分历史净值曾大幅变化、原因尚未确认，这次预测可能受影响。")
    if head:
        _drivers(prediction, head)
    _holding_note(holding)
    with st.expander("模型说明与历史表现", expanded=False):
        st.caption(markdown_text(_source_gate(prediction)))
        st.caption("这仍是实验预测，目前尚未证明能稳定超过简单方法。80%预测范围描述可能的净值涨跌，不是平均值的置信区间。")
        if head:
            view = explain_forecast_head(prediction, head, horizon=horizon, entry_lag=0, role="anchor")
            render_factor_details(prediction, view, key="forecast_factor_details")
        render_validation(prediction, horizon=horizon)
    _fees(prediction, holding, horizon=horizon)

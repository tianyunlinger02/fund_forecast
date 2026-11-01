"""Opt-in public estimate snapshots; no workspace, portfolio, or persistent writes."""
from __future__ import annotations

import streamlit as st

from rmbfund.domain import decimal
from rmbfund.providers.intraday import SOURCES, fetch_estimate
from rmbfund.ui.charts import intraday_figure
from rmbfund.ui.formatting import markdown_text, percent, when


@st.cache_data(ttl=60, persist=None, max_entries=256, show_spinner=False)
def cached_estimate(source: str, symbol: str) -> dict:
    """Failures share the same cooldown; button clicks never bypass this cache."""
    return fetch_estimate(source, symbol)


def remember_sample(state: dict, result: dict) -> None:
    """Retain the last success and only actual, same-session source observations."""
    state["attempt"] = result
    if result["status"] != "ok":
        return
    state["good"] = result
    if result["source_time_kind"] != "estimate":
        return
    point = {key: result[key] for key in (
        "source", "source_name", "source_time", "source_time_kind", "checked_at",
        "change", "baseline_date", "method",
    )}
    series = (result["source"], result["source_time"][:10], result["baseline_date"], result["method"])
    points = state.setdefault("points", [])
    if state.get("series") != series or (points and point["source_time"] < points[-1]["source_time"]):
        points = state["points"] = []
    state["series"] = series
    if points and points[-1]["source_time"] == point["source_time"]:
        points[-1] = point
    else:
        points.append(point)
    state["points"] = points[-1440:]


def _auto_changed() -> None:
    st.session_state["fund_intraday_request"] = st.session_state["fund_intraday_auto"]


def _display(state: dict, source: str, symbol: str) -> None:
    attempt, good = state.get("attempt"), state.get("good")
    if attempt and attempt["status"] != "ok":
        reason = attempt.get("reason") or "这次没有取得可显示的估算。"
        st.info(markdown_text(reason) + (" 上次成功的估算仍保留原时间。" if good else ""))
        st.caption(f"本次取数时间（北京时间）{when(attempt['checked_at'])}")
    if not good:
        if attempt is None:
            st.caption("点击“刷新估算”查看这只基金最近可取得的估算；打开页面不会自动联网。")
        elif attempt.get("source_time") and attempt["source_time_kind"] == "snapshot":
            st.caption(f"来源快照生成时间 {markdown_text(attempt['source_time'])}，不代表这只基金有估值。")
        return

    metrics = st.columns(2)
    metrics[0].metric("来源估算涨跌", percent(good["change"], signed=True, digits=2),
                      help="这是来源报告的涨跌，不是你的盈亏；来源未说明比较基准日期。")
    if good.get("estimated_nav") is not None:
        metrics[1].metric("来源估算值", f"{decimal(good['estimated_nav']):,.4f}",
                          help="不是正式公布净值。来源币种未核实，不用于计算持仓市值。")
    else:
        metrics[1].metric("单只基金行情时间", "未提供")

    if good["source_time_kind"] == "estimate":
        st.caption(f"来源估算时间（时区未标明）{good['source_time']}")
    else:
        st.caption(f"来源快照生成时间 {markdown_text(good['source_time'])} · 不是单只基金行情时间")
    st.caption(f"最近取数（北京时间）{when(good['checked_at'])}")
    session = {"closed": "休市", "open": "开市"}.get(good.get("market_session"))
    if session:
        st.caption(f"来源标注：{session}；不据此判断所有海外市场的交易状态。")
    st.caption("涨跌比较基准日期：来源未说明。第三方估算，不是最终净值，也不是当前行情保证。")

    points = state.get("points", [])
    if good["source_time_kind"] == "estimate" and points:
        st.plotly_chart(intraday_figure(points, product_name=symbol), width="stretch", theme="streamlit",
                        config={"displayModeBar": False, "scrollZoom": False})
        st.caption("本次打开后采集的估算，非完整分时历史；同一来源时点不重复画点，不补开盘零点。")
        with st.expander("查看估算数据表"):
            st.dataframe([{"来源估算时间（时区未明）": row["source_time"],
                           "来源估算涨跌": percent(row["change"], signed=True, digits=2),
                           "实际取得时间（北京时间）": when(row["checked_at"])}
                          for row in reversed(points)], hide_index=True, width="stretch")
    else:
        st.caption("来源未提供单只基金行情时间，暂不绘制分时曲线。")
    st.caption(f"来源：{SOURCES[source]['name']} · 估算不改变下方的正式净值、持仓盈亏或预测。")


def render_intraday(*, symbol: str, kind: str, mode: str) -> None:
    """Only strings cross this boundary; timer reruns cannot touch a closed SQLite connection."""
    if kind != "open_fund":
        return
    with st.container(border=True, key="fund_intraday_card"):
        st.subheader("最近估算")
        if mode != "public":
            st.caption("演示模式不连接外部估算来源。")
            return
        controls = st.columns([2, 1, 1.4], vertical_alignment="bottom")
        source = controls[0].selectbox("估算来源", list(SOURCES),
            format_func=lambda item: SOURCES[item]["name"], key="fund_intraday_source")
        active = (symbol, source)
        if st.session_state.get("fund_intraday_active") != active:
            st.session_state["fund_intraday_auto"] = False
            st.session_state["fund_intraday_request"] = False
        st.session_state["fund_intraday_active"] = active
        if st.session_state.pop("fund_intraday_stop", False):
            st.session_state["fund_intraday_auto"] = False
        blocked = st.session_state.setdefault("fund_intraday_blocked", {})
        disabled = source in blocked
        manual = controls[1].button("刷新估算", key="fund_intraday_refresh", disabled=disabled)
        auto = controls[2].toggle("每分钟检查", key="fund_intraday_auto", disabled=disabled,
                                  on_change=_auto_changed,
                                  help="只检查当前基金。每60秒检查不代表来源每分钟更新；失败即暂停。")
        if disabled:
            st.caption("该来源已拒绝访问或要求验证，本会话不再请求它，也不会自动切换来源。")
        st.caption("可手动刷新，或开启每分钟检查；60秒内共用一次取数结果，来源本身可能更新更慢。")
        requested = st.session_state.pop("fund_intraday_request", False) or manual
        records = st.session_state.setdefault("fund_intraday_records", {})
        state = records.setdefault(active, {})
        first = True

        @st.fragment(run_every=60 if auto else None)
        def card() -> None:
            nonlocal first
            # Streamlit also executes fragments on full-app reruns. Only timer
            # ticks or explicit controls may fetch, never editing/browsing.
            request = requested if first else auto
            first = False
            if request and not disabled:
                with st.spinner("正在读取公开估算…"):
                    result = cached_estimate(source, symbol)
                remember_sample(state, result)
                if result.get("blocked"):
                    blocked[source] = True
                if result["status"] != "ok" and auto:
                    st.session_state["fund_intraday_stop"] = True
                    st.rerun()
                if result.get("blocked"):
                    st.rerun()
            _display(state, source, symbol)

        card()

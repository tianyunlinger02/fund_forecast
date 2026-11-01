"""Read-only presentation of a validated ResearchLibrary bundle.

No Workspace, path resolution, artifact loading, registration, or network belongs
here. The one mutation boundary is the explicitly confirmed reproduction callback.
Research-window selectors never change the application's observation clock.
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
import streamlit as st

from rmbfund.domain import ValidationError, timestamp
from rmbfund.reports import safe_cell
from rmbfund.research_artifacts import presentation_decimal
from rmbfund.ui.charts import (
    RESEARCH_LABELS,
    RESEARCH_STRATEGIES,
    price_figure,
    price_rows,
    research_curve_rows,
    research_figure,
)

SCENARIOS = (
    "C_zero_entry_cap100_min10_lag2_confirm2_cash7",
    "C_zero_entry_cap100_min10_lag2_confirm2_cash10_stress",
)
SCENARIO_LABELS = {
    SCENARIOS[0]: "到账 7 个观测日（固定假设）",
    SCENARIOS[1]: "到账 10 个观测日（压力假设）",
}
SOURCE_STATUS = {
    "ok": "来源返回可用，仍非规则核验",
    "partial": "部分覆盖 / 未完整核验",
    "blocked": "受阻",
    "error": "错误",
    "network_error": "采集失败",
    "downloaded": "已下载，未核验",
    "parsed": "已解析，未独立核验",
}


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value) if isinstance(value, Decimal) else value


def _table(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    empty: str = "暂无记录",
    height: int | None = None,
) -> None:
    """Share typed numeric cells and formula-safe text between table and CSV."""
    original = [{safe_cell(str(k)): _cell(v) for k, v in row.items()} for row in rows]
    if not original:
        st.info(empty)
        return
    display = [
        {k: safe_cell(v) if isinstance(v, str) else v for k, v in row.items()} for row in original
    ]
    if display != original:
        st.caption("表格与 CSV 的公式样式文本带安全单引号；真实数值保持数值，原始报告未改。")
    options: dict[str, Any] = {"hide_index": True, "use_container_width": True}
    if height is not None:
        options["height"] = height
    st.dataframe(display, **options)
    st.download_button(
        "下载此表 CSV",
        pd.DataFrame.from_records(display).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"zhiji-research-{key}.csv",
        mime="text/csv",
        key=f"research_{key}_csv",
        help="与上表同一批记录；UTF-8 BOM，文本作电子表格公式注入防护。",
    )


def _percent(value: Any) -> float | None:
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 90
        result = presentation_decimal(value) * 100
    return float(result)


def _money(value: Any) -> str:
    return "" if value is None else str(presentation_decimal(value))


def _windows(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        study["windows"], key=lambda window: (window["start"], window["end"], window["id"])
    )


def _results(window: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    lookup = {
        result["strategy"]: result
        for result in window["report"]["results"]
        if result["scenario"] == scenario
    }
    if set(lookup) != set(RESEARCH_STRATEGIES):
        raise ValidationError("留存研究缺少固定四路径，未按表现筛选或补造结果")
    return [lookup[strategy] for strategy in RESEARCH_STRATEGIES]


def _selection(study: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    windows = _windows(study)
    if not windows:
        raise ValidationError("留存研究没有可用窗口")
    if st.session_state.get("research_selection_study") != study["id"]:
        st.session_state["research_selection_study"] = study["id"]
        for key in ("research_window", "research_scenario", "research_strategy"):
            st.session_state.pop(key, None)
    valid = {
        "research_window": [window["id"] for window in windows],
        "research_scenario": list(SCENARIOS),
        "research_strategy": list(RESEARCH_STRATEGIES),
        "research_theme": ["light", "dark"],
    }
    for key, options in valid.items():
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]
    window = next(item for item in windows if item["id"] == st.session_state["research_window"])
    return window, st.session_state["research_scenario"], st.session_state["research_strategy"]


def selection_context(study: dict) -> dict:
    """Stable, small note context shared with the host; no report/session copy."""
    window, scenario, strategy = _selection(study)
    return {
        "study_id": study["id"],
        "window_id": window["id"],
        "scenario": scenario,
        "strategy": strategy,
        "input_hash": window["input_hash"],
        "economic_result_hash": window["economic_result_hash"],
    }


def _model_complete(result: Mapping[str, Any]) -> bool:
    return result.get("metrics", {}).get("model_complete") is not False and all(
        row.get("model_complete") is not False and row.get("model_assets") is not None
        for row in result.get("daily", [])
    )


def research_summary_rows(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    """All 2 × 2 × 4 retained results, chronologically, never ranked by return."""
    rows = []
    for window in _windows(study):
        for scenario in SCENARIOS:
            for result in _results(window, scenario):
                metrics = result["metrics"]
                complete = _model_complete(result)
                rows.append(
                    {
                        "窗口": window["label"],
                        "起始日": window["start"],
                        "结束日": window["end"],
                        "情景": SCENARIO_LABELS[scenario],
                        "路径": RESEARCH_LABELS[result["strategy"]],
                        "条件净收益（%）": _percent(metrics.get("conditional_net_return"))
                        if complete
                        else None,
                        "条件最大回撤（%）": _percent(metrics.get("conditional_max_drawdown"))
                        if complete
                        else None,
                        "仅价格变化（%）": _percent(metrics.get("price_only_change"))
                        if complete
                        else None,
                        "期末模型资产（元）": _money(metrics.get("end_assets")),
                        "累计费用（元）": _money(metrics.get("fees")),
                        "舍入损失（元）": _money(metrics.get("rounding_loss")),
                        "持仓观测占比（%）": _percent(metrics.get("exposure_fraction")),
                        "成交 / 拒绝笔数": f"{metrics.get('filled_orders', 0)} / {metrics.get('rejected_orders', 0)}",
                        "日级样本数": metrics.get("samples"),
                        "来源完整性": "已核验"
                        if metrics.get("verified_data_complete") is True
                        else "未完整核验",
                        "实绩性质": "条件模型，非已核验实绩",
                    }
                )
    return rows


def research_ending_state_rows(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Visible four-path close-state table for the selected window and scenario."""
    fields = (
        ("cash_available", "可用现金（元）"),
        ("cash_reserved", "冻结现金（元）"),
        ("cash_receivable", "应收现金（元）"),
        ("market_value", "持仓市值（元）"),
        ("units", "总份额"),
        ("available_units", "可用份额"),
        ("unconfirmed_units", "未确认份额"),
    )
    rows: list[dict[str, Any]] = []
    order = {strategy: index for index, strategy in enumerate(RESEARCH_STRATEGIES)}
    for result in sorted(
        results, key=lambda item: order.get(str(item.get("strategy")), len(order))
    ):
        daily = result.get("daily", [])
        metrics = result.get("metrics", {})
        ending = metrics.get("ending_state")
        if not isinstance(ending, Mapping):
            ending = daily[-1] if daily and isinstance(daily[-1], Mapping) else {}
        row = {
            "路径": RESEARCH_LABELS.get(
                str(result.get("strategy")), str(result.get("strategy", ""))
            ),
            "期末估值日": ending.get("date", ""),
            "模型资产（元）": _money(ending.get("model_assets")),
        }
        for key, label in fields:
            value = ending.get(key)
            row[label] = _money(value) if value is not None else ""
        row["日级模型完整"] = "是" if ending.get("model_complete") is True else "否 / 空白"
        rows.append(row)
    return rows


def _assumption_rows(window: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    inputs = window["inputs"]
    base = inputs["assumptions"]
    effective = _results(window, scenario)[0]["assumptions"]
    fields = (
        ("initial_cash", "初始现金（元）"),
        ("fund_purchase_min", "最低申购（元）"),
        ("fund_daily_purchase_cap", "单日申购上限（元）"),
        ("fund_subscription_rate", "申购费率（小数）"),
        ("nav_lag_days", "净值假设可知滞后（观测日）"),
        ("confirm_days", "申请后确认（观测日）"),
        ("redemption_cash_days", "赎回款到账（观测日）"),
        ("fund_redemption_tiers", "赎回费档（days_lt 为自然日上界，不含）"),
        ("ma_period", "MA 回看观测数"),
        ("warmup", "原输入预热观测数"),
    )
    rows = [
        {"假设项": label, "输入基础值": base.get(key), "当前情景有效值": effective.get(key)}
        for key, label in fields
    ]
    definition = inputs.get("strategy_definitions", {}).get("monthly_mom60_hold30", {})
    rows.extend(
        [
            {
                "假设项": "外部追加资金",
                "输入基础值": "不追加",
                "当前情景有效值": "不追加；有仓不补仓",
            },
            {
                "假设项": "月度动量回看（观测数）",
                "输入基础值": definition.get("lookback_observations"),
                "当前情景有效值": definition.get("lookback_observations"),
            },
            {
                "假设项": "月度最短持有（自然日，自确认日起）",
                "输入基础值": definition.get("min_holding_calendar_days"),
                "当前情景有效值": definition.get("min_holding_calendar_days"),
            },
            {
                "假设项": "月度策略所需预热（含 NAV 滞后）",
                "输入基础值": inputs.get("required_strategy_warmup"),
                "当前情景有效值": inputs.get("required_strategy_warmup"),
            },
        ]
    )
    return rows


def _warnings(study: Mapping[str, Any]) -> list[str]:
    warnings = [str(item) for item in study.get("warnings", [])]
    for window in _windows(study):
        warnings.extend(str(item) for item in window["report"].get("gaps", []))
        warnings.extend(str(item) for item in window["report"].get("forward_blockers", []))
        warnings.extend(str(item) for item in window["inputs"].get("warnings", []))
    return list(dict.fromkeys(warnings))


def _safe_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        _ = parsed.port
    except ValueError:
        return None
    return value


def visible_nav_rows(
    study: Mapping[str, Any], *, as_of: str, stored_rows: list | None = None
) -> list[dict[str, Any]]:
    """Use current Store data when supplied, never mix/fill it with the archive.

    Store.price rows are already as-of filtered; older Store projections omit
    available_at but retain observed_at. Retained rows carry both actual times.
    Missing or malformed actual times are not historical-visibility evidence.
    """
    cutoff = timestamp(as_of)
    using_store = bool(stored_rows)
    candidates = stored_rows if using_store else study.get("nav_rows", [])
    visible = []
    for row in candidates:
        try:
            observed = timestamp(row["observed_at"])
            actual_available = row.get("available_at")
            if actual_available is None and not using_store:
                continue
            available = timestamp(actual_available or row["observed_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(observed, available) <= cutoff and str(row["date"]) <= cutoff.date().isoformat():
            visible.append(dict(row))
    return sorted(visible, key=lambda row: row["date"])


def render_overview(study: dict, *, as_of: str, stored_product: dict | None = None) -> None:
    source, product = study["source"], study["product"]
    st.subheader("公开研究样本 · 数据状态")
    st.text(f"{product.get('symbol', '')} · {product.get('name', '')}")
    st.caption("研究样本不是用户真实持仓。留存来源不因选择旧研究窗口而变成历史时点档案。")
    columns = st.columns(3)
    columns[0].metric("留存档案净值截至", str(source.get("latest_date", "未知")))
    columns[1].metric("实际采集日期", str(source.get("observed_at", "未知"))[:10])
    columns[2].metric(
        "留存覆盖",
        "部分 / 未核验" if source.get("status") == "partial" else str(source.get("status", "未知")),
    )
    rows = visible_nav_rows(study, as_of=as_of)
    current = stored_product or {}
    latest_date = (
        current.get("latest_date") if stored_product else rows[-1]["date"] if rows else None
    )
    latest_value = (
        current.get("latest_value") if stored_product else rows[-1]["value"] if rows else None
    )
    _table(
        [
            {
                "代码": product.get("symbol"),
                "份额类别": product.get("share_class"),
                "分类": product.get("category"),
                "观察时刻": as_of,
                "当前可见净值日期": latest_date,
                "当前可见单位净值": latest_value,
                "当前截面来源": "当前 Store（非留存档案）"
                if stored_product
                else "留存档案，按实际可用时间筛选",
                "档案实际采集时间": source.get("observed_at"),
                "档案来源状态": SOURCE_STATUS.get(
                    source.get("status"), source.get("status", "未知")
                ),
            }
        ],
        "overview",
    )
    st.warning(
        "历史申赎状态、限额/费率生效版本、境内外交易日历与实际披露/到账时点仍有缺口；研究假设不能解锁严格模拟成交。"
    )
    if not stored_product and not rows:
        st.info(
            "截至当前观察时刻没有本机已知的留存净值。历史研究页仍可查看事后回放，但不是当时可得信息。"
        )


def render_snapshot(study: dict, *, as_of: str, stored_rows: list | None = None) -> None:
    st.subheader("产品详情 · 单位净值截面")
    st.text(f"{study['product'].get('symbol', '')} · {study['product'].get('name', '')}")
    using_store = bool(stored_rows)
    st.caption(
        "当前 Store 日级值：使用本次 Store 查询，不以旧研究档案覆盖或补齐。"
        if using_store
        else "留存 NAV 档案：仅展示实际观察/可用时间不晚于全局观察时刻的记录。"
    )
    st.caption(
        "单位净值不是红利再投总回报；缺失日期不插值、不补值。研究窗口不会改动这里的观察时钟。"
    )
    visible = visible_nav_rows(study, as_of=as_of, stored_rows=stored_rows)
    if not visible:
        st.warning(
            "当前观察时刻没有可见净值；若早于实际采集时间，本机当时并不知道这些历史 NAV。不会把事后采集记录伪装成历史时点已知数据。"
        )
        return
    prepared = price_rows(visible)
    theme = st.selectbox(
        "净值图表主题",
        ["light", "dark"],
        format_func=lambda value: "浅色" if value == "light" else "深色",
        key="research_snapshot_theme",
    )
    st.plotly_chart(
        price_figure(visible, product_name=study["product"]["name"], kind="open_fund", theme=theme),
        use_container_width=True,
        theme=None,
        config={"displaylogo": False},
        key="research_snapshot_chart",
    )
    provenance = {row["date"]: row for row in visible}
    _table(
        [
            {
                "日期": row["date"],
                "单位净值（元）": row["value"],
                "图中值": row["plot_value"],
                "口径": row["basis"],
                "来源": row["source"],
                "实际观察时间": row["observed_at"],
                "实际可用时间": provenance[row["date"]].get(
                    "available_at", "Store 已按观察时刻过滤"
                ),
                "数据集": "当前 Store" if using_store else "留存 NAV 档案",
            }
            for row in prepared
        ],
        "snapshot",
        height=350,
    )


def render_source_context(study: dict) -> None:
    source = study["source"]
    st.subheader("研究来源 · 假设与缺口")
    st.caption("以下为留存研究档案上下文，不冒充当前来源刷新或历史时点原始档案。")
    _table(
        [
            {
                "来源状态": SOURCE_STATUS.get(source.get("status"), source.get("status", "未知")),
                "数据最新日期": source.get("latest_date"),
                "实际采集时间": source.get("observed_at"),
                "留存 NAV 条数": len(study.get("nav_rows", [])),
                "来源事实条数": len(source.get("facts", [])),
                "来源摘要 SHA256": source.get("sourcehash"),
                "原文 URL": source.get("url"),
            }
        ],
        "source_context",
    )
    url = _safe_url(source.get("url"))
    if url:
        st.link_button("打开留存 NAV 的公开来源", url)
    elif source.get("url"):
        st.caption("来源不是可安全打开的 HTTP(S) 链接。")
        st.text(str(source["url"]))
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for window in _windows(study):
        for item in window["inputs"].get("source_evidence", []):
            identity = (
                str(item.get("source_url")),
                str(item.get("observed_at")),
                str(item.get("content_hash")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            evidence.append(
                {
                    "来源": item.get("source"),
                    "类型": item.get("kind", "NAV"),
                    "状态": item.get("status"),
                    "实际观察": item.get("observed_at"),
                    "URL": item.get("source_url"),
                    "内容 SHA256": item.get("content_hash"),
                    "已核支付宝规则": item.get("alipay_verified", False),
                }
            )
    with st.expander("采集证据与固定有效假设", expanded=False):
        _table(evidence, "source_evidence")
        window, scenario, _ = _selection(study)
        st.text(f"{window['label']} · {SCENARIO_LABELS[scenario]}")
        _table(_assumption_rows(window, scenario), "source_assumptions")
    st.warning(
        "来源 partial ≠ 全部净值/权益/规则已核验；二手页面没有观察到分红或拆分，不等于管理人完整核验。"
    )
    with st.expander("全部留存警告与前向模拟阻塞项", expanded=False):
        _table(
            [{"待核验 / 适用边界": item} for item in _warnings(study)], "source_gaps", height=380
        )


def _compact_reproduction(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValidationError("复现服务没有返回结构化结果，不能标记成功")
    windows = []
    for row in result.get("windows", [])[:2]:
        windows.append(
            {
                "id": str(row.get("id", "")),
                "ok": row.get("ok") is True,
                "hash_matched": row.get("hash_matched") is True,
                "output_dir": str(row.get("output_dir", ""))[:600],
                "error": str(row.get("error", ""))[:500],
            }
        )
    return {
        "ok": result.get("ok") is True,
        "operation_id": str(result.get("operation_id", ""))[:160],
        "output_dir": str(result.get("output_dir", ""))[:600],
        "windows": windows,
        "error": str(result.get("error", ""))[:700],
    }


def _render_reproduce(
    study: dict, reproduce: Callable[[str], dict], action: Callable | None
) -> None:
    states = st.session_state.setdefault("research_reproduction", {})
    if study["id"] not in states:
        states[study["id"]] = {"request_id": uuid4().hex, "status": "idle"}
    state = states[study["id"]]
    with st.expander("显式复现 · 核对留存经济结果哈希", expanded=False):
        st.caption(
            "只在确认后同步运行两个完整窗口；不联网、不登记产品、不更新公共库、不提交模拟订单。切换筛选与页面重跑不会自动运行或重试。"
        )
        st.caption(f"本次请求 ID：{state['request_id']}")
        with st.form("research_reproduce_form"):
            confirmed = st.checkbox(
                "我确认运行本地研究复现，输出不是实盘成交或未来收益",
                key="research_reproduce_confirm",
            )
            submitted = st.form_submit_button(
                "确认复现留存研究",
                disabled=state["status"] != "idle",
                key="research_reproduce_submit",
            )
        if submitted and not confirmed:
            st.warning("请先明确确认；尚未运行复现。")
        if submitted and confirmed and state["status"] == "idle":
            # Mark before entering the callback. An interrupted rerun retains
            # the request and never silently issues the operation again.
            state["status"] = "running"

            def operation() -> dict[str, Any]:
                return _compact_reproduction(reproduce(state["request_id"]))

            try:
                with st.spinner("正在复现两个留存研究窗口…"):
                    if action is None:
                        result = operation()
                        ok = True
                    else:
                        ok, result = action("复现留存研究（简要结果）", operation)
                if not ok or not isinstance(result, dict):
                    state.update(
                        status="failed", error="操作未成功返回；请查看操作日志，未自动重试。"
                    )
                else:
                    state["result"] = _compact_reproduction(result)
                    matches = result.get("windows", [])
                    complete = (
                        result.get("ok") is True
                        and len(matches) == len(study["windows"])
                        and all(
                            row.get("ok") is True and row.get("hash_matched") is True
                            for row in matches
                        )
                    )
                    state["status"] = "succeeded" if complete else "failed"
            except Exception as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:700]}")
        if state["status"] == "succeeded":
            st.success(
                "复现完成：两个窗口的经济结果哈希均一致。这只证明固定输入与模型的可复现性，不证明真实可成交或数据完整。"
            )
        elif state["status"] == "failed":
            st.error("本次复现未完成或哈希不一致，未将其标记为成功；不会在筛选重跑时自动重试。")
        elif state["status"] == "running":
            st.warning(
                "此请求尚无已确认结果，可能被中断；不会自动重复调用。请先核验已有输出，再决定是否创建新请求。"
            )
        if state.get("error"):
            st.text(state["error"])
        result = state.get("result", {})
        if result:
            st.text(f"操作 ID：{result.get('operation_id', '')}")
            st.text(f"输出目录：{result.get('output_dir', '')}")
            if result.get("error"):
                st.text(result["error"])
            _table(
                [
                    {
                        "窗口": row["id"],
                        "运行完成": row["ok"],
                        "经济哈希一致": row["hash_matched"],
                        "输出目录": row["output_dir"],
                        "错误": row["error"],
                    }
                    for row in result["windows"]
                ],
                "reproduction",
            )
        if state["status"] != "idle" and st.button(
            "创建新的复现请求（不立即运行）", key="research_reproduce_new"
        ):
            states[study["id"]] = {"request_id": uuid4().hex, "status": "idle"}
            st.rerun()


def render_research(
    study: dict, *, reproduce: Callable[[str], dict], action: Callable | None = None
) -> None:
    window, scenario, _ = _selection(study)
    source = study["source"]
    st.subheader("历史条件研究 · 固定四路径")
    st.text(study["title"])
    st.warning(
        "这是事后历史研究，不是用户持仓、支付宝真实成交或已运行的前向模拟盘。窗口已经看过，不是事先封存的严格样本外验证。"
    )
    st.caption(
        f"实际采集：{source.get('observed_at', '未知')}；留存净值截至 {source.get('latest_date', '未知')}。历史研究始终为回顾视图，不随全局观察日期伪装成当时可知。"
    )
    st.caption(
        "固定列出两窗口 × 两到账假设 × 四路径共 16 行，不按收益选优；窗口时长不同，累计收益不可直接排名。收益与回撤均为条件模型，非已核验实绩。"
    )
    _table(research_summary_rows(study), "summary", height=600)
    windows = _windows(study)
    window_map = {item["id"]: item for item in windows}
    controls = st.columns([3, 3, 1])
    with controls[0]:
        window_id = st.selectbox(
            "研究窗口（不改变观察时钟）",
            list(window_map),
            format_func=lambda key: (
                f"{window_map[key]['label']} · {window_map[key]['start']} — {window_map[key]['end']}"
            ),
            key="research_window",
        )
    with controls[1]:
        scenario = st.selectbox(
            "到账情景（不是已核交易规则）",
            list(SCENARIOS),
            format_func=lambda key: SCENARIO_LABELS[key],
            key="research_scenario",
        )
    with controls[2]:
        theme = st.selectbox(
            "研究图表主题",
            ["light", "dark"],
            format_func=lambda value: "浅色" if value == "light" else "深色",
            key="research_theme",
        )
    window = window_map[window_id]
    results = _results(window, scenario)
    if any(not _model_complete(result) for result in results):
        st.warning(
            "当前情景有不完整模型估值：缺失点留空，不补零或插值；不完整路径的条件收益留空。缺失点后的真实历史峰值未知，其后回撤也留空。"
        )
    st.caption(
        "同一初始现金指数：model_assets / initial_cash × 100；不是把首日资产或首日 NAV 重置为 100。期末不清仓，不含未来退出费；应收资金并非可立即支取现金。"
    )
    st.caption(
        "单位净值已含基金运作费用与人民币汇率折算；此处不再重复扣除运作费或叠加汇率收益。回撤是日收盘采样相对历史峰值，不是盘中最差回撤。"
    )
    title = f"{window['label']} · {SCENARIO_LABELS[scenario]}"
    for measure, key in (("assets_index", "assets"), ("drawdown_pct", "drawdown")):
        st.plotly_chart(
            research_figure(results, measure=measure, title=title, theme=theme),
            use_container_width=True,
            theme=None,
            config={"displaylogo": False},
            key=f"research_{key}_chart",
        )
    st.caption(
        "现金灰色点线 / 持有蓝色实线 / MA20 橙色虚线 / 月度动量青绿色点划线；身份不会因筛选或表现改变。悬停含全部路径，右端标签以引线避让；下表是同一数据的可访问视图。"
    )
    st.subheader("四路径期末状态（当前窗口 / 情景）")
    _table(
        research_ending_state_rows(results),
        "ending_state",
        empty="当前窗口 / 情景没有可见期末状态。",
    )
    with st.expander("曲线等价日级表与 CSV", expanded=False):
        _table(
            [
                {
                    "日期": row["date"],
                    "路径": row["label"],
                    "情景": row["scenario"],
                    "产品": row["product_name"],
                    "初始现金（元）": row["initial_cash"],
                    "模型资产（元）": row["model_assets"],
                    "图中资产指数": row["assets_index"],
                    "图中回撤（%）": row["drawdown_pct"],
                }
                for row in research_curve_rows(results)
            ],
            "curves",
            height=380,
        )
    with st.expander("当前窗口：输入基础值与情景有效假设", expanded=False):
        _table(_assumption_rows(window, scenario), "assumptions")
        st.caption(
            "观测日不是已核境内/境外交易日历。月度策略只在每月首个窗口内观测日检查：60 期单位净值动量，确认日起至少持有 30 自然日；不足持有期/未确认/未到账则等下个月，不月中补单，不保证及时止损。"
        )
    st.subheader("单一路径审计 · 不影响上方四条曲线")
    strategy = st.selectbox(
        "查看交易 / 信号的路径",
        list(RESEARCH_STRATEGIES),
        format_func=lambda key: RESEARCH_LABELS[key],
        key="research_strategy",
    )
    result = next(item for item in results if item["strategy"] == strategy)
    st.caption(
        "以下均为模型审计记录。假设可知、申请、确认、到账时间与实际采集时间不同；不是当时本机已知或支付宝成交凭证。"
    )
    _table(result["orders"], "orders", empty="此路径在该窗口/情景没有模型订单。", height=320)
    with st.expander("信号、结算事件与原始日级账本", expanded=False):
        _table(result["signals"], "signals", height=320)
        _table(result["events"], "events")
        _table(result["daily"], "daily", height=350)
    with st.expander("留存警告与适用边界", expanded=False):
        _table([{"警告 / 缺口": item} for item in _warnings(study)], "gaps", height=380)
        st.caption(f"输入哈希：{window['input_hash']}")
        st.caption(f"经济结果哈希：{window['economic_result_hash']}")
    _render_reproduce(study, reproduce, action)

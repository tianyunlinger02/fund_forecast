"""Standard-trade fund decisions with optional holdings and unchanged saved history."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import streamlit as st

from rmbfund.domain import ValidationError, decimal, now_iso
from rmbfund.forecast_models import NAV_MODEL_ID
from rmbfund.portfolio import Portfolio, summary_holdings, value_holding
from rmbfund.ui import forecast as forecast_ui
from rmbfund.ui.charts import price_figure
from rmbfund.ui.decision import render_decision
from rmbfund.ui.formatting import markdown_text, percent, when
from rmbfund.ui.fund_context import render_fund_context
from rmbfund.ui.history import render_history
from rmbfund.ui.intraday import render_intraday
from rmbfund.ui.portfolio import (
    add_fund_dialog, edit_fund_dialog, fund_label, money, public_subject, render_summary,
    shares_text,
)
from rmbfund.ui.theme import apply_theme


PUBLIC_KEYS = ("id", "symbol", "name", "kind", "share_class", "exchange", "category", "is_demo")


def _workspace(root: Path, mode: str) -> Any:
    from rmbfund.service import Workspace

    return Workspace(root, mode=mode, initialize=False)


def _empty_detail(subject: Mapping[str, Any]) -> dict:
    return {"status": "missing", "product": dict(subject), "history": [], "latest": None,
            "previous_change": None, "last_checked_at": None, "portfolio": None, "reasons": []}


def _public_identity(subject: Mapping[str, Any]) -> dict:
    identity = {key: value for key, value in subject.items() if key in PUBLIC_KEYS and key != "name"}
    identity["name"] = str(identity["symbol"])
    return identity


def _product_id(subject: Mapping) -> str:
    return subject.get("id") or f"{subject.get('kind', 'open_fund')}:{subject['symbol']}:{subject.get('share_class', '')}"


def _research_time(detail: Mapping) -> str:
    thesis = detail.get("research_thesis") or {}
    payload = thesis.get("payload") or {}
    return str((payload.get("origin") or {}).get("generated_at")
               or (thesis.get("source") or {}).get("available_at")
               or payload.get("research_date") or "")


def _fund_detail(workspace: Any, subject: Mapping[str, Any], as_of: str) -> dict:
    subject = _public_identity(subject)
    local = _empty_detail(subject)
    if workspace is not None:
        try:
            resolved = workspace.resolve_fund(subject, register=False)
            if resolved:
                local = workspace.fund_detail(resolved["id"], as_of)
        except Exception:
            local.update(status="error", reasons=["这只基金资料暂时打不开，已填写的持仓仍保留。"])
    identifier = _product_id(local.get("product") or subject)
    current = st.session_state.get("fund_current_details", {}).get(identifier)
    if current is None:
        return local
    # Keep this session's NAV/disclosure batch across pages, but let a newly
    # persisted research version appear without fetching or renewing old dates.
    result = dict(current)
    if _research_time(local) > _research_time(result):
        for key in ("research_thesis", "investment_profile", "issuer_analysis", "execution_conditions"):
            if key in local:
                result[key] = local[key]
    return result


def _remember_evaluation(product_id: str, evaluation: Mapping, context: Mapping) -> None:
    st.session_state.setdefault("fund_current_evaluations", {})[product_id] = dict(evaluation)
    details = st.session_state.setdefault("fund_current_details", {})
    contexts = st.session_state.setdefault("fund_current_contexts", {})
    for identifier, detail in (evaluation.get("details") or {}).items():
        details[identifier] = detail
        contexts[identifier] = context


def _update_fund(root: Path, mode: str, workspace: Any, subject: Mapping[str, Any], *,
                 refresh: bool = True, model_id: str = "nav_ridge_empirical_v2") -> tuple[dict, dict]:
    """Only an explicit command can fit; refresh controls network access."""
    from rmbfund.service import Workspace

    owned = workspace is None
    current = Workspace(root, mode=mode) if owned else workspace
    try:
        # Private display names never cross the public resolver boundary. Known
        # products supply their resolved public identity; unknown products use code.
        identity = _public_identity(subject)
        known = current.resolve_fund(identity, register=False)
        if known:
            identity = {key: value for key, value in known.items() if key in PUBLIC_KEYS}
        resolved = current.resolve_fund(identity, register=True)
        if resolved is None:
            raise ValueError("公开基金身份无法解析")
        report = current.update_forecast(resolved["id"], refresh=refresh, model_id=model_id)
        return report, resolved
    finally:
        if owned:
            current.store.close()


def _refresh_button(root: Path, mode: str, workspace: Any, subject: dict,
                    repository: Portfolio, fund: dict | None, *, model_id: str) -> None:
    """Render explicit local-fit and explicit network-refresh actions."""
    investment = model_id == "exposure_factor_ridge_empirical_v1"
    local = st.button("用已保存资料更新预测" if investment else "用已保存净值更新预测",
                      key="fund_local_update", type="primary",
                      disabled=mode != "public" or workspace is None,
                      help="只用本地已经保存的净值与所选模型输入重新计算；不会联网。")
    network = st.button("更新净值、投资因素与预测" if investment else "更新净值与预测",
                        key="fund_update", disabled=mode != "public",
                        help="明确点击才获取所选模型的免费公开输入；不读取金融账户或上传持仓。")
    if not (local or network):
        return
    refresh = bool(network)
    try:
        with st.spinner("正在获取已公布净值并计算预测…" if refresh else "正在用已保存净值匹配并计算预测…"):
            report, resolved = _update_fund(root, mode, workspace, subject, refresh=refresh, model_id=model_id)
    except Exception:
        st.error("这次更新没有完成，请稍后再试。原净值与预测仍保留原来的时间。")
        return
    if fund is not None and fund.get("product_id") != resolved["id"]:
        try:
            # Keep the private display name and private quantities while binding
            # only the public product identity.
            private_subject = {key: value for key, value in resolved.items() if key in PUBLIC_KEYS}
            private_subject["name"] = fund["subject"].get("name") or fund["subject"]["symbol"]
            repository.save_fund(private_subject, shares=fund.get("shares"), cost=fund.get("cost"), fund_id=fund["id"])
        except Exception:
            st.warning("公开数据已处理，但这次未能关联到本地持仓。你填写的持仓仍保留，请稍后重试。")
    status = report.get("status")
    refreshed = report.get("refresh") or {}
    if not refresh:
        if status == "ready":
            message = ("已有预测与保存资料匹配，保留原预测时间；没有联网检查新净值或投资因素。" if report.get("reused")
                       else "已用保存资料更新所选模型；没有联网检查新净值或投资因素。")
        else:
            message = "已检查本地保存资料，但预测尚未就绪；没有联网检查新净值，原资料仍保留。"
    elif status == "ready":
        message = ("所选模型的输入与成交假设没有变化，保留原预测时间。" if report.get("reused")
                   else "所选模型的数据与预测已更新。")
    elif refreshed.get("status") in {"ok", "partial", "cached"}:
        message = "净值已检查；预测暂未就绪，已有资料仍可查看。"
    else:
        message = "这次更新未完成，旧资料仍保留原时间；请稍后再试。"
    st.session_state["fund_flash"] = message
    st.session_state["fund_update_result"] = {"status": status, "refresh_status": refreshed.get("status"), "refresh": refresh}
    st.rerun()


def _trend(detail: Mapping[str, Any], subject: Mapping[str, Any], theme: str) -> None:
    latest = detail.get("latest") or {}
    header = st.columns([2, 1])
    with header[0]:
        st.subheader("净值走势")
        value = latest.get("value")
        st.metric("最新公布净值", f"{decimal(value):,.4f}" if value is not None else "—",
                  delta=(percent(detail['previous_change'], signed=True, digits=2)
                         if detail.get("previous_change") is not None else None), delta_color="off",
                  help="涨跌幅与上一笔已公布净值比较，不是今天的盘中涨跌或你的持仓收益。")
    with header[1]:
        st.caption("净值所属日期")
        st.write(latest.get("date") or "暂无净值")
        st.caption(f"最近检查 {when(detail.get('last_checked_at'))}")
    if latest:
        source = str(latest.get("source") or "").lower()
        source_name = "天天基金公开资料" if "eastmoney" in source or "1234567" in source else "公开净值资料"
        st.caption(f"{source_name} · 单位净值（{latest.get('currency') or '币种未注明'}） · "
                   f"取得于 {when(latest.get('observed_at'))}")
    st.caption("本图只显示已公布净值；QDII通常公布得更晚。第三方估算在上方单独查看。")
    if detail.get("status") == "error":
        for reason in detail.get("reasons", []):
            st.warning(markdown_text(reason))
    elif latest.get("nav_type") == "final_unit_nav_candidate":
        st.caption("来自公开网站的净值记录，尚未逐笔与官方披露核对。")
    rows = list(detail.get("history") or [])
    if not rows:
        st.info("还没有可显示的净值。点击“更新净值与预测”获取公开数据；你的持仓无需重填。")
        return
    if "fund_range" not in st.session_state:
        st.session_state["fund_range"] = st.session_state.get("fund_selected_range", "近3月")
    period = st.segmented_control("走势范围", ["近1月", "近3月", "近1年", "全部"], key="fund_range")
    st.session_state["fund_selected_range"] = period
    days = {"近1月": 30, "近3月": 90, "近1年": 365}.get(period)
    if days is not None:
        end = date.fromisoformat(rows[-1]["date"])
        rows = [row for row in rows if date.fromisoformat(row["date"]) >= end - timedelta(days=days)]
    if len(rows) >= 2:
        change = decimal(rows[-1]["value"]) / decimal(rows[0]["value"]) - 1
        st.caption(f"{rows[0]['date']} — {rows[-1]['date']} · 区间净值变化 {percent(change, signed=True, digits=2)}，不是你的持仓收益")
    figure = price_figure(rows, product_name=str(subject.get("name") or subject["symbol"]),
                          kind="open_fund", theme=theme, compact=True, currency=latest.get("currency"),
                          native_theme=True)
    st.plotly_chart(figure, width="stretch", theme="streamlit",
                    config={"displayModeBar": False, "scrollZoom": False})
    with st.expander("查看净值数据表"):
        st.dataframe([{"日期": row["date"], "单位净值": row["value"], "币种": row.get("currency", "未注明")}
                      for row in reversed(rows)], hide_index=True, width="stretch")


def _stop_intraday() -> None:
    st.session_state["fund_intraday_auto"] = False
    st.session_state["fund_intraday_request"] = False
    st.session_state["fund_intraday_active"] = None


def _open_fund(fund_id: str) -> None:
    _stop_intraday()
    st.session_state.pop("fund_public_selection", None)
    st.session_state.pop("fund_preview", None)
    st.session_state["fund_selection"] = fund_id
    st.session_state["fund_picker"] = fund_id
    st.session_state["fund_page"] = "detail"
    st.session_state.pop("fund_navigation", None)


def _switch_fund(fund_ids: tuple[str, ...]) -> None:
    """Only an actual new fund id may reset navigation.

    Selectbox sends formatted labels on the wire. After a local rename the
    browser can still submit the previous label on an unrelated tab click;
    Streamlit returns that unknown label as a string before rerendering options.
    Restore the current id so the native widget receives its updated label,
    without interpreting stale display text as a fund switch.
    """
    selected = st.session_state.get("fund_picker")
    current = st.session_state.get("fund_selection")
    if selected not in fund_ids:
        if current in fund_ids:
            st.session_state["fund_picker"] = current
        return
    if selected != current or st.session_state.get("fund_public_selection"):
        _open_fund(selected)


def _overview() -> None:
    _stop_intraday()
    st.session_state["fund_page"] = "overview"
    st.session_state.pop("fund_navigation", None)
    st.session_state.pop("fund_preview", None)
    st.session_state.pop("fund_public_selection", None)


def _preview_public(subject: Mapping) -> None:
    _stop_intraday()
    st.session_state["fund_public_selection"] = {
        key: value for key, value in subject.items() if key in PUBLIC_KEYS
    }
    st.session_state["fund_page"] = "detail"
    st.session_state.pop("fund_navigation", None)


def _public_code_preview(products: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("输入基金代码，看现在的买卖判断")
        st.caption("不用先填金额或持仓。选“还没买”或“已经持有”，看一个对应的操作建议。")
        code = st.text_input("想研究的基金代码", key="fund_public_code", placeholder="例如 018147").strip()
        matches = [row for row in products if row.get("kind") == "open_fund" and row.get("symbol") == code]
        subject = matches[0] if len(matches) == 1 else None
        if len(matches) > 1:
            choice = st.selectbox("选择公开资料注明的份额类别", range(len(matches)),
                format_func=lambda index: f"{matches[index].get('name') or code} · {matches[index].get('share_class') or '类别未注明'}",
                key="fund_public_code_class")
            subject = matches[choice]
        if st.button("公开预览", key="fund_public_code_open"):
            if len(code) != 6 or not code.isascii() or not code.isdigit():
                st.warning("请填写六位基金代码，保留开头的0。")
            else:
                _preview_public(subject or {"id": f"open_fund:{code}:", "symbol": code, "name": code,
                    "kind": "open_fund", "share_class": "", "exchange": "", "is_demo": False})


def _online_cache_key(scope: Mapping) -> str:
    subjects = sorted(scope["public_subjects"],
                       key=lambda row: str(row.get("id") or row.get("product_id") or row["symbol"]))
    return json.dumps({"subjects": subjects, "source_plan": scope.get("source_plan")},
                      ensure_ascii=False, sort_keys=True)


def _cached_online_context(scope: Mapping) -> Mapping | None:
    return st.session_state.get("fund_online_cache", {}).get(_online_cache_key(scope))


def _online_scope(workspace: Any, subject: Mapping, funds: list[dict]) -> dict:
    identity = _public_identity(subject)
    if not identity.get("id"):
        identity["id"] = f"{identity.get('kind', 'open_fund')}:{identity['symbol']}:{identity.get('share_class', '')}"
    additional = [_public_identity(public_subject(item)) for item in funds]
    if workspace is not None and hasattr(workspace, "current_subjects"):
        return workspace.current_subjects(identity, additional=additional)
    subjects = {}
    for row in [identity, *additional]:
        if not row.get("id"):
            row["id"] = f"{row.get('kind', 'open_fund')}:{row['symbol']}:{row.get('share_class', '')}"
        subjects[row["id"]] = row
    return {"public_subjects": list(subjects.values()), "source_plan": None, "peer_set": None, "blocked_urls": []}


def _current_online(workspace: Any, scope: Mapping, *, force: bool = False) -> tuple[dict, bool]:
    """Reuse only public response data in this browser session, never a Store handle."""
    subjects = sorted(scope["public_subjects"], key=lambda row: str(row.get("id") or row.get("product_id") or row["symbol"]))
    key = _online_cache_key(scope)
    cache = st.session_state.setdefault("fund_online_cache", {})
    cached = cache.get(key)
    if cached is not None and not force:
        return cached, True
    blocked = set(st.session_state.get("fund_online_stopped_urls", [])) | set(scope.get("blocked_urls") or [])
    try:
        if workspace is not None and hasattr(workspace, "current_online"):
            fetch_current = workspace.current_online
        else:
            from rmbfund.service import Workspace
            fetch_current = Workspace.current_online
        context = fetch_current(subjects, source_plan=scope.get("source_plan"),
                                request_budget=20, blocked_urls=sorted(blocked))
    except Exception:
        checked = now_iso()
        context = {"schema_version": 1, "kind": "current_online_context", "status": "unavailable",
                   "checked_at": checked, "evaluated_at": checked, "source_checks": [],
                   "nav_by_product": {}, "records_by_product": {}, "messages_by_product": {},
                   "warnings": ["这次在线检查没有完成；旧资料保留原日期，没有改称本次消息。"]}
    for check in context.get("source_checks") or []:
        if check.get("status") in {"error", "failed"} and check.get("source_url"):
            blocked.add(check["source_url"])
    blocked.update(context.get("stopped_urls") or [])
    st.session_state["fund_online_stopped_urls"] = sorted(blocked)
    cache[key] = context
    return context, False


def _online_evaluation(workspace: Any, product_id: str, detail: dict, context: Mapping, scope: Mapping) -> dict:
    if workspace is not None and hasattr(workspace, "current_evaluation"):
        return workspace.current_evaluation(product_id, context, public_subjects=scope["public_subjects"],
                                            peer_set=scope.get("peer_set"))
    from rmbfund.current_online import project_current_detail, snapshot_online_context

    details = {}
    for subject in scope["public_subjects"]:
        identifier = subject.get("id") or subject.get("product_id")
        original = detail if identifier == product_id else _empty_detail(subject)
        if not (original.get("product") or {}).get("id"):
            original = {**original, "product": dict(subject)}
        details[identifier] = project_current_detail(original, context, identifier)
    return {"details": details, "online_context": snapshot_online_context(context),
            "evaluated_at": context.get("evaluated_at") or context.get("checked_at"),
            "fund_comparison": {"status": "missing", "comparisons": [], "reasons": ["没有经过核对的替代候选集合"]}}


def _research_settings(root: Path) -> tuple[dict, str | None, bool]:
    from rmbfund.research_settings import is_configured, read_research_api_settings

    if "fund_research_settings" not in st.session_state:
        try:
            settings = read_research_api_settings(root / "research-api.json")
        except (OSError, ValidationError, ValueError):
            settings = read_research_api_settings()
            st.warning("research-api.json暂不可用；请在下方填写研究API地址和模型，不会自动猜测供应商。")
        st.session_state["fund_research_settings"] = settings
    settings = dict(st.session_state["fund_research_settings"])
    with st.expander("研究API设置"):
        st.caption("只有明确点击“更新研究”才会调用API并使用你的现有额度；填写设置本身不测试连接。")
        settings["base_url"] = st.text_input("API地址", value=settings.get("base_url", ""),
            key="fund_research_base_url", placeholder="填写你的实际API地址，不预设供应商").strip()
        settings["model"] = st.text_input("模型名", value=settings.get("model", ""),
            key="fund_research_model", placeholder="填写该API实际支持的模型名").strip()
        protocols = ["openai_chat_completions", "anthropic_messages"]
        settings["protocol"] = st.selectbox("API协议", protocols,
            index=protocols.index(settings.get("protocol")) if settings.get("protocol") in protocols else 0,
            format_func=lambda value: {"openai_chat_completions": "OpenAI Chat Completions",
                                       "anthropic_messages": "Anthropic Messages"}[value],
            key="fund_research_protocol")
        if "fund_research_api_key" not in st.session_state:
            st.session_state["fund_research_api_key"] = st.session_state.get("fund_research_session_key", "")
        key = st.text_input("API Key（仅本会话，可留空使用服务端环境）", type="password",
            key="fund_research_api_key", help="仅在本页输入；不需要发到聊天，也不会写进JSON、判断历史或公开研究材料。")
        st.session_state["fund_research_session_key"] = key
        st.caption("本页设置只在当前会话使用。重开可从本机research-api.json读取地址、协议和模型；"
                   "Key只接受本会话输入或服务端RMBFUND_RESEARCH_API_KEY，不读取其他终端配置。")
    st.session_state["fund_research_settings"] = settings
    api_key = key.strip() or None
    configured = is_configured(settings, api_key=api_key)
    if not configured:
        st.caption("自动更新研究需配置已有API；不影响查看下方已完成的研究。")
    return settings, api_key, configured


def _research_saved(result: Mapping | None) -> bool:
    result = result or {}
    return (result.get("status") in {"generated", "partial"}
            and bool((result.get("research_run") or {}).get("research"))
            and bool(result.get("receipt")) and bool(result.get("evaluation")))


def _update_research(root: Path, mode: str, workspace: Any, subject: Mapping, *,
                     settings: Mapping, api_key: str | None, context: Mapping | None,
                     scope: Mapping) -> dict | None:
    from rmbfund.service import Workspace

    owned = workspace is None
    current = None
    try:
        with st.status("更新研究 · 取材", expanded=True) as progress:
            phase = "取材"

            def on_progress(event) -> None:
                nonlocal phase
                stage = event.get("stage", "") if isinstance(event, Mapping) else str(event)
                if stage in {"requesting", "validating"} or "分析" in stage:
                    phase = "分析"
                elif stage in {"saving", "complete"} or "保存" in stage:
                    phase = "保存"
                else:
                    phase = "取材"
                progress.update(label=f"更新研究 · {phase}")

            current = Workspace(root, mode=mode) if owned else workspace
            result = current.update_research(_public_identity(subject), settings=settings,
                api_key=api_key, context=context, public_subjects=scope["public_subjects"],
                peer_set=scope.get("peer_set"), on_progress=on_progress)
            success = _research_saved(result)
            progress.update(label="取材 → 分析 → 保存 · 已完成" if success else "更新研究未完成 · 旧研究保留原日期",
                            state="complete" if success else "error", expanded=False)
        return result
    except Exception:
        st.error("这次研究没有生成或保存成功。已有研究保留原结论和原日期；没有提交买卖，也没有用固定建议替代。")
        return None
    finally:
        if owned and current is not None:
            current.store.close()


def _compact_position(value: Mapping, *, preview: bool) -> None:
    if preview:
        st.caption("正在查看公开基金资料，还没有把它加入你的持仓。")
        return
    st.caption(f"确认份额 {shares_text(value.get('shares'))} · 市值 {money(value.get('market_value'))} · "
               f"浮盈亏 {money(value.get('profit'), signed=True)}")
    if value.get("valuation_status") == "currency_mismatch":
        st.caption("净值币种与人民币成本不相容，未换汇或混算市值和盈亏。")
    elif value.get("shares") is None:
        st.caption("还没有填写份额；“管理与更新”中可编辑持仓，未填不等于未持有。")
    elif value.get("market_value") is None:
        st.caption("还没有可估值的净值，已填份额与成本保留。")
    else:
        st.caption(f"按最近公布净值 {value.get('nav_date') or '日期未提供'} 估算，未扣赎回费。")


def _timing_view(workspace: Any, product_id: str, model_id: str, context: Mapping | None,
                 detail: Mapping, report: dict) -> Callable | None:
    """Read inputs once; fitting is reachable only from an explicit local button."""
    inputs = {}
    decision_at = now_iso()
    if workspace is not None and hasattr(workspace, "timing_inputs"):
        try:
            inputs = workspace.timing_inputs(product_id, context=context, model_id=model_id, decision_at=decision_at)
        except (ValidationError, ValueError, TypeError, KeyError):
            inputs = {"status": "unavailable", "series": [], "reasons": ["这批数据暂不能用于本地测算。"]}
    # This fallback is for choosing a past known price, never a fitting input.
    display_series = inputs.get("series") if inputs else (detail.get("current_nav_history") or detail.get("history") or [])
    report["scenario_series"] = [{**row, "nav": row.get("nav", row.get("value"))} for row in display_series or []]
    fee_records = [row for row in detail.get("background_records") or [] if row.get("family") == "fees"]
    if fee_records:
        fee = max(fee_records, key=lambda row: (row.get("report_date") or "", row.get("available_at") or ""))
        report["scenario_fees"] = {**(fee.get("payload") or {}),
            **{key: fee[key] for key in ("observed_at", "available_at", "report_date") if key in fee},
            "evidence_refs": [fee.get("id"), fee.get("source_url")]}
    prediction = report.get("prediction") or {}
    if inputs.get("status") == "error":
        report["status"] = "error"
        report["reasons"] = inputs.get("reasons") or ["当前输入存在冲突，未把旧预测当成本批判断。"]
    if inputs.get("status") == "ready" and inputs.get("series") and prediction:
        latest = inputs["series"][-1]
        anchor = prediction.get("anchor") or {}
        if latest.get("date") != anchor.get("date") or decimal(latest["nav"]) != decimal(anchor.get("nav", "0")):
            report["status"] = "stale"
            report["reasons"] = ["这批数据已不同于已保存预测；请点击本地计算，旧概率没有冒充当前结果。"]
    if workspace is None or not hasattr(workspace, "calculate_timing"):
        return None
    cache = st.session_state.setdefault("fund_timing_cache", {})
    curve_display = st.session_state.setdefault("fund_timing_curve_display", {})
    display_scope = st.session_state.get("forecast_subject_scope") or (product_id, model_id)

    def calculate(*, calendar_days, include_history=False, include_curve=False, lookup_only=False,
                  sale_options=None, compare_sales=False):
        selected = workspace.timing_inputs(product_id, context=context, model_id=model_id,
                                           calendar_days=calendar_days, decision_at=decision_at)
        plan = selected.get("trade_plan") or {}
        base_key = (product_id, model_id, selected.get("cache_key") or "unavailable")
        curve_key = ("curve", plan.get("horizon")) if include_curve else None
        key = base_key + ((curve_key,) if curve_key is not None else ())
        sale_report = None
        if sale_options is not None:
            if hasattr(workspace, "sale_timing_inputs"):
                prepared = workspace.sale_timing_inputs(product_id, context=context, model_id=model_id,
                    calendar_days=calendar_days, decision_at=decision_at, **sale_options)
                def cached_sale_prediction(request):
                    base_sale_key = (product_id, model_id, request["cache_key"])
                    candidate_keys = [base_sale_key, base_sale_key + (("curve", request["horizon"]),)]
                    saved = next((cache.get(candidate, {}).get("current") for candidate in candidate_keys
                                  if cache.get(candidate, {}).get("current")), None)
                    if not isinstance(saved, Mapping) or not isinstance(saved.get("prediction"), Mapping):
                        return None
                    raw = saved["prediction"]
                    horizon = request["horizon"]
                    compact = deepcopy(dict(raw))
                    compact["forecasts"] = [head for head in raw.get("forecasts", [])
                                             if head.get("horizon") == horizon and head.get("entry_lag") == 0]
                    compact["execution_forecasts"] = [head for head in raw.get("execution_forecasts", [])
                                                       if head.get("horizon") == horizon]
                    compact["requested_horizons"] = [horizon]
                    return compact

                cached = {row["cache_key"]: cached_sale_prediction(row)
                          for row in prepared.get("sale_requests", [])}
                sale_report = (workspace.calculate_sale_timing(prepared, cached_predictions=cached)
                               if compare_sales else workspace.rebuild_sale_timing(prepared, cached_predictions=cached))
                for cache_key, value in sale_report.get("sale_reports", {}).items():
                    cache.setdefault((product_id, model_id, cache_key), {})["current"] = dict(value)
            else:
                from rmbfund.timing_scenario import build_sale_date_comparison
                current_prediction = cache.get(key, {}).get("current", {}).get("prediction")
                predictions = [current_prediction] if current_prediction else []
                sale_report = {"sale_comparison": build_sale_date_comparison(predictions,
                    plan=selected.get("trade_plan") or {}, **sale_options)}
        entry = cache.setdefault(key, {})
        if include_curve and "history" not in entry:
            # History is an independent report, not a prediction bundle. Share
            # only that report with the curve flavor so a history-tool run is
            # visible after the page restores its already-displayed curve.
            saved_history = cache.get(base_key, {}).get("history")
            if saved_history is not None:
                entry["history"] = deepcopy(saved_history)
        name = "history" if include_history else "current"
        if sale_options is None and not lookup_only and name not in entry:
            result = workspace.calculate_timing(product_id, context=context, model_id=model_id,
                                                 calendar_days=calendar_days, decision_at=decision_at,
                                                 include_history=include_history,
                                                 include_curve=include_curve)
            entry[name] = dict(result)
            if include_history and result.get("status") == "ready":
                entry.setdefault("current", {key: value for key, value in result.items() if key != "historical_forecasts"})
        # Numerical H/L heads can be reused by another day plan, but dates and
        # fee ages always come from this selection, not from its cached caller.
        result = dict(entry.get("current") or {})
        result["trade_plan"] = deepcopy(selected.get("trade_plan"))
        if selected.get("series") is not None:
            result["series"] = selected["series"]
        if entry.get("history"):
            result["timing_history_report"] = entry["history"]
        if include_history and not lookup_only:
            result.update(entry.get("history") or {})
            result["trade_plan"] = deepcopy(selected.get("trade_plan"))
        if result.get("prediction"):
            result["prediction"] = deepcopy(result["prediction"])
            result["prediction"]["entry_assumptions"] = deepcopy(selected.get("entry_assumptions") or {})
        if selected.get("calculation"):
            result["calculation"] = {**(result.get("calculation") or {}), **selected["calculation"]}
        if selected.get("status") != "ready":
            result.update(status=selected.get("status", "unavailable"), prediction=None,
                          reasons=selected.get("reasons") or [])
        elif not result.get("prediction"):
            result.update(status="unavailable", prediction=None,
                          reasons=["这个日期计划还没有对应预测；点击本地计算，不借用其他期限。"])
        if include_curve and not lookup_only and result.get("prediction"):
            curve_display[display_scope] = True
        elif include_curve and lookup_only and not result.get("prediction"):
            curve_display.pop(display_scope, None)
        if sale_report is not None:
            result["sale_comparison"] = deepcopy(sale_report.get("sale_comparison"))
        return result

    return calculate


def _replay_assessment(workspace: Any, product_id: str, saved: Mapping, *, note=None) -> dict:
    """An explicit history action reads only stored NAVs, never the live collector."""
    from rmbfund.assessment_replay import replay_saved_assessment

    point = now_iso()
    if workspace is None or not hasattr(workspace, "stored_nav_series"):
        return {"status": "unavailable", "reasons": ["暂时没有可读的本地净值依据；没有联网更新或借用其他基金。"]}
    data = workspace.stored_nav_series(product_id, as_of=point)
    if data.get("status") not in {"ready", "available"}:
        return {"status": "unavailable", "reasons": data.get("reasons") or ["已有净值暂不能用于原模拟核对。"]}
    return replay_saved_assessment(saved, series=data.get("series") or [], note=note, as_of=point)


def _research_tools(workspace: Any, repository: Portfolio, subject: Mapping, detail: Mapping,
                    fund: Mapping | None, *, model_id: str, product_id: str) -> None:
    """Keep optional research/calculators off the current operation screen."""
    from rmbfund.personal_assessment import build_assessment
    from rmbfund.portfolio_exposure import build_portfolio_context
    from rmbfund.timing_scenario import (
        build_calendar_scenario, resolve_plan_fees, summarize_calendar_history,
    )
    from rmbfund.ui.buyer import (
        render_comparison, render_investor, render_online, render_portfolio, render_research,
        render_sizing, sizing_inputs,
    )
    from rmbfund.ui.decision import _anchor_and_amounts, anchor_currency
    from rmbfund.ui.investment import render_investment_analysis
    from rmbfund.ui.trade import render_reconstructed, render_sale_comparison

    assessment = st.session_state.get("fund_displayed_assessments", {}).get(product_id) or {}
    context = st.session_state.get("fund_current_contexts", {}).get(product_id)
    prediction = deepcopy(assessment.get("forecast") or {})
    if prediction:
        anchor_head = (assessment.get("anchor") or {}).get("head")
        execution_head = ((assessment.get("trade_scenario") or {}).get("decision") or {}).get("primary_head")
        prediction["forecasts"] = [anchor_head] if anchor_head else []
        prediction["execution_forecasts"] = [execution_head] if execution_head else []
    report = {"prediction": prediction, "selected_model_id": model_id,
              "evaluated_at": now_iso(), **{key: detail.get(key) for key in (
                  "investment_profile", "issuer_analysis", "research_thesis")}}
    with st.expander("完整研究与条件计算工具"):
        if assessment:
            render_investor(assessment)
        render_research(assessment or {"research_thesis": detail.get("research_thesis"),
            "investor_decision": {"action_basis": "independent_cases"}})
        render_comparison(assessment.get("fund_comparison") or {}, preview=_preview_public)
        render_online(assessment.get("online_context"), product_id=product_id)
        render_investment_analysis(report)
    scenario = assessment.get("trade_scenario") or {}
    plan, fees = scenario.get("plan") or {}, scenario.get("fees") or {}
    if plan and workspace is not None:
        calculate = _timing_view(workspace, product_id, model_id, context, detail, report)
        if calculate is not None:
            days = plan["calendar_days"]
            with st.expander("历史检验与卖出日期工具"):
                st.caption("独立的本地核查工具，不改变已展示或已保存的操作建议。")
                if st.button("运行历史检验", key="trade_history_calculate"):
                    result = calculate(calendar_days=days, include_history=True)
                else:
                    result = calculate(calendar_days=days, lookup_only=True)
                historical = result.get("timing_history_report") or result
                if historical.get("historical_forecasts"):
                    render_reconstructed(summarize_calendar_history(historical.get("series") or [],
                        historical["historical_forecasts"], plan=plan, fee_basis=fees))
                sale_options = {"fee_record": report.get("scenario_fees") or {},
                    "old_confirmation_date": fees.get("old_confirmation_date"),
                    "current_redemption_override": fees.get("current_redemption_fee"),
                    "future_old_redemption_override": fees.get("future_old_redemption_fee")}
                compare = st.button("比较这几个卖出时点", key="trade_compare_sales")
                comparison = calculate(calendar_days=days, sale_options=sale_options,
                    compare_sales=compare, lookup_only=not compare)
                render_sale_comparison(comparison.get("sale_comparison") or {}, expanded=False)
    with st.expander("个人金额换算与资金工具"):
        saved = forecast_ui._scope(subject)
        personal_fees = forecast_ui.fee_inputs(saved)
        st.session_state.setdefault("forecast_budget", saved.get("budget", ""))
        budget_text = st.text_input("拟投入金额（可留空）", key="forecast_budget")
        saved["budget"] = budget_text
        budget = None
        try:
            if budget_text.strip():
                if decimal(budget_text) < 0:
                    raise ValidationError("negative budget")
                budget = {"amount": budget_text, "currency": (fund or {}).get("currency")
                          or anchor_currency(assessment.get("forecast") or {})}
        except ValidationError:
            st.warning("拟投入金额须是非负数；这里只影响个人换算，不改变基金建议。")
            return
        current = value_holding(fund, detail.get("latest")) if fund else {}
        sizing = None
        if st.toggle("展开个人资金规划（可选）", key="trade_personal_plan"):
            try:
                positions = []
                for item in repository.list_funds():
                    item_subject = public_subject(item)
                    row_detail = _fund_detail(workspace, item_subject, report["evaluated_at"])
                    positions.append({"holding": {**item, "product_id": _product_id(item_subject)},
                        "valuation": value_holding(item, row_detail.get("latest")),
                        "profile": row_detail.get("investment_profile")})
                portfolio = build_portfolio_context(positions, selected_product_id=product_id,
                                                    as_of=report["evaluated_at"])
            except (ValidationError, ValueError, TypeError, KeyError):
                portfolio = {}
            render_portfolio(portfolio)
            sizing = sizing_inputs(saved, portfolio=portfolio, current=current or {},
                                   fees=personal_fees, requested=budget)
            render_sizing(sizing)
        personal_scenario = None
        if plan:
            personal_fee_basis = resolve_plan_fees(report.get("scenario_fees") or {}, plan=plan,
                old_confirmation_date=fees.get("old_confirmation_date"),
                new_confirmation_date=fees.get("new_confirmation_date"),
                subscription_override=personal_fees.get("subscription_fee"),
                current_redemption_override=personal_fees.get("redemption_fee"),
                future_old_redemption_override=fees.get("future_old_redemption_fee"),
                future_new_redemption_override=fees.get("future_new_redemption_fee"))
            personal_scenario = build_calendar_scenario(prediction,
                subject=subject, plan=plan, series=report.get("scenario_series") or [],
                purchase_date=(scenario.get("purchase") or {}).get("date"),
                fee_basis=personal_fee_basis)
        personal = build_assessment(prediction, subject=subject, holding=fund,
            horizon=assessment.get("horizon") or 20, current_value=current,
            anchor_currency=anchor_currency(assessment.get("forecast") or {}), budget=budget,
            trade_scenario=personal_scenario, sizing_plan=sizing, **personal_fees)
        _anchor_and_amounts(personal)


def _selected_view(workspace: Any, root: Path, mode: str, repository: Portfolio,
                   subject: dict, detail: dict, fund: dict | None, theme: str) -> None:
    identity = tuple(subject.get(key) for key in ("id", "symbol", "share_class", "exchange"))
    if st.session_state.get("fund_detail_scope") != identity:
        _stop_intraday()
        st.session_state["fund_detail_scope"] = identity
        st.session_state.pop("fund_navigation", None)
        st.session_state.pop("fund_range", None)
        st.session_state.pop("fund_selected_range", None)
    name = (fund["subject"].get("name") if fund else subject.get("name")) or subject["symbol"]
    st.subheader(markdown_text(name))
    suffix = f" · {subject['share_class']}类" if subject.get("share_class") else ""
    st.caption(f"{subject['symbol']}{suffix}")
    current_value = value_holding(fund, detail.get("latest")) if fund else None
    product_id = subject.get("id") or f"{subject.get('kind', 'open_fund')}:{subject['symbol']}:{subject.get('share_class', '')}"
    subject = {**subject, "id": product_id}
    available = workspace is not None and product_id is not None
    if available:
        models = workspace.forecast_models(product_id)
    else:
        from rmbfund.forecast_models import model_catalog
        models = model_catalog(subject.get("kind", "open_fund"), symbol=subject.get("symbol"))
    # The current buyer flow uses the established NAV algorithm. Research-only
    # model families stay available to services and historical records, not an
    # automatic promotion or a decision-page model picker.
    model_id = NAV_MODEL_ID
    with st.popover("管理与更新"):
        _compact_position(current_value or {}, preview=fund is None)
        if fund is not None and st.button("编辑持仓", key="fund_edit"):
            st.session_state["fund_dialog"] = ("edit", fund["id"])
        st.caption(f"最近净值检查 {when(detail.get('last_checked_at'))}；检查时间不等于净值所属日。")
        _refresh_button(root, mode, workspace, subject, repository, fund, model_id=model_id)
        st.caption("短期预测只计算净值统计，不会生成新的基金买卖研究。研究更新和轻量在线检查在“当前判断”中分开进行。")
    if detail.get("status") == "error":
        st.caption("这只基金资料暂时打不开或最近检查失败；已填持仓保留，旧资料仍标记原时间。")
    tabs = st.tabs(["当前判断", "走势与资料", "历史记录"], key="fund_navigation", on_change="rerun")
    if not tabs[1].open:
        _stop_intraday()
    with tabs[0]:
        if tabs[0].open:
            evaluation = {}
            context = None
            if mode == "public":
                with st.popover("更新研究与在线资料"):
                    settings, api_key, configured = _research_settings(root)
                    research_requested = st.button("更新研究", key="fund_update_research", type="primary",
                        help="明确点击才执行一次取材、API分析与保存；使用你的模型额度，不上传个人份额、成本或预算。")
                    force_online = st.button("重新在线检查", key="fund_current_online_refresh",
                        help="只检查公开净值与定向资料，不调用研究API，也不重算短期预测。")
                    st.caption("更新研究：读材料并生成新的基金论证，使用API额度。重新在线检查：只刷新资料。"
                               "短期预测：单独计算净值统计。普通切页、改费用或保存判断都不调用研究API。")
                context = None
                scope = _online_scope(workspace, subject, [])
                try:
                    cached = st.session_state.get("fund_current_evaluations", {}).get(product_id)
                    context = (st.session_state.get("fund_current_contexts", {}).get(product_id)
                               or _cached_online_context(scope))
                    if cached is not None and not force_online:
                        prior_detail = (cached.get("details") or {}).get(product_id) or {}
                        if _research_time(detail) > _research_time(prior_detail) and context is not None:
                            # Another session may have saved research after this RAM
                            # acquisition. Re-evaluate locally, never renew source clocks.
                            context = {**context, "evaluated_at": now_iso()}
                            evaluation = _online_evaluation(workspace, product_id, detail, context, scope)
                            _remember_evaluation(product_id, evaluation, context)
                        else:
                            evaluation = dict(cached)
                            evaluation["details"] = {**(evaluation.get("details") or {}), product_id: detail}
                    elif context is not None and not force_online:
                        evaluation = _online_evaluation(workspace, product_id, detail, context, scope)
                        _remember_evaluation(product_id, evaluation, context)
                    elif force_online or (research_requested and configured):
                        with st.spinner("在线检查当前净值与定向资料…"):
                            context, _ = _current_online(workspace, scope, force=force_online)
                            evaluation = _online_evaluation(workspace, product_id, detail, context, scope)
                        _remember_evaluation(product_id, evaluation, context)
                    # Plain render, settings edits and save callbacks only reuse
                    # supplied/local data; no collector or research API runs here.
                except (ValidationError, ValueError, TypeError, KeyError):
                    st.warning("本次在线资料暂不能合并；旧资料保留原日期，研究和保存仍可继续。")
                if research_requested:
                    if not configured:
                        st.warning("还缺API配置，未启动研究任务；请展开“研究API设置”填写。")
                    else:
                        result = _update_research(root, mode, workspace, subject, settings=settings,
                            api_key=api_key, context=context, scope=scope)
                        if _research_saved(result):
                            evaluation = result["evaluation"]
                            context = result.get("context") or context or {}
                            _remember_evaluation(product_id, evaluation, context)
                            st.session_state.setdefault("fund_online_cache", {})[_online_cache_key(scope)] = context
                            st.success("新研究已保存，下面已改用本次研究与同批资料。" +
                                       ("部分材料仍未取得，具体缺口见研究说明。" if result["status"] == "partial" else ""))
                        elif result:
                            st.warning("还缺API配置，未生成新研究。" if result.get("status") == "unconfigured" else
                                       "这次没有形成合格的新研究；已有研究仍保留原结论和原日期。")
            details = evaluation.get("details") or {}
            selected_detail = details.get(product_id, detail)
            current_value = value_holding(fund, selected_detail.get("latest")) if fund else None
            report = forecast_ui.read_forecast((lambda: workspace.forecast(product_id, model_id=model_id)) if available else None,
                                               subject, now_iso())
            if evaluation.get("evaluated_at"):
                report["evaluated_at"] = evaluation["evaluated_at"]
            for key in ("investment_profile", "issuer_analysis", "research_thesis", "execution_conditions"):
                report[key] = selected_detail.get(key)
            report["selected_model_id"] = model_id
            from rmbfund.decision_brief import load_decision_rationale
            report["decision_rationale"] = load_decision_rationale(
                product_id, (report.get("research_thesis") or {}).get("record_id"))
            report["online_context"] = evaluation.get("online_context") or selected_detail.get("online_context")
            if evaluation.get("fund_comparison") is not None:
                report["fund_comparison"] = evaluation["fund_comparison"]
            elif mode != "public" and available and hasattr(workspace, "compare_funds"):
                try:
                    report["fund_comparison"] = workspace.compare_funds(product_id, as_of=report.get("evaluated_at"))
                except (ValidationError, ValueError, TypeError, KeyError) as exc:
                    report["fund_comparison"] = {"status": "unavailable", "comparisons": [], "reasons": [str(exc)]}
            calculate = _timing_view(workspace, product_id, model_id, context, selected_detail, report)
            assessment = render_decision(subject, report=report, holding=fund, current_value=current_value,
                calculate=calculate, theme=theme,
                summary=(lambda **kwargs: workspace.forecast_summary(product_id, model_id=model_id, **kwargs)) if available else None,
                save=repository.save_assessment, models=models, preview=_preview_public,
                previous_assessment=lambda selected: next(iter(repository.assessment_history(selected, limit=1)), None),
                load_conditions=getattr(repository, "research_preferences", None),
                save_conditions=getattr(repository, "save_research_preferences", None))
            if assessment is not None:
                st.session_state.setdefault("fund_displayed_assessments", {})[product_id] = assessment
    with tabs[1]:
        if tabs[1].open:
            render_intraday(symbol=str(subject["symbol"]), kind=str(subject.get("kind", "")), mode=mode)
            display_detail = (st.session_state.get("fund_current_details", {}).get(product_id)
                              or detail)
            _trend(display_detail, {**subject, "name": name}, theme)
            render_fund_context(display_detail)
            _research_tools(workspace, repository, subject, display_detail, fund,
                            model_id=model_id, product_id=product_id)
    with tabs[2]:
        if tabs[2].open:
            render_history(product_id,
                read=(lambda **kwargs: workspace.forecast_history(product_id, **kwargs)) if available else None,
                detail=(lambda record_id: workspace.forecast_history_detail(product_id, record_id)) if available else None,
                summary=(lambda **kwargs: workspace.forecast_summary(product_id, **kwargs)) if available else None,
                assessments=repository.assessment_history, assessment_detail=repository.assessment_detail,
                action_notes=repository.action_notes, save_action_note=repository.save_action_note,
                replay_assessment=lambda saved, note=None: _replay_assessment(workspace, product_id, saved, note=note),
                subject=subject, models=models, theme=theme)


def _fund_overview(workspace: Any, funds: list[dict]) -> None:
    """Portfolio-wide values belong only to the overview, never above a decision."""
    details = {fund["id"]: _fund_detail(workspace, public_subject(fund), now_iso()) for fund in funds}
    valued = {fund["id"]: value_holding(fund, details[fund["id"]].get("latest")) for fund in funds}
    render_summary(summary_holdings(list(valued.values())))
    st.subheader("选择基金，查看当前判断")
    for fund in funds:
        value = valued[fund["id"]]
        with st.container(border=True):
            st.markdown(f"**{markdown_text(fund_label(fund))}**")
            st.caption(f"市值 {money(value.get('market_value'))} · 成本 {money(value.get('cost'))} · "
                       f"浮盈亏 {money(value.get('profit'), signed=True)}")
            st.caption(f"份额 {shares_text(value.get('shares'))} · 最近净值 {value.get('nav_date') or '尚未取得'}")
            st.button("查看当前判断", key=f"fund_open_{fund['id']}",
                      on_click=_open_fund, args=(fund["id"],))
    st.caption(f"{len(funds)} 只基金 · 手动记录于本机 · 公开样本不会自动加入")


def main(root: Path | str | None = None, mode: str | None = None, *,
         workspace: Any = None, portfolio: Portfolio | None = None) -> None:
    st.set_page_config(page_title="我的基金 · 知基", page_icon="◈", layout="wide")
    root = Path(root or os.environ.get("RMBFUND_ROOT", "./data")).expanduser()
    mode = mode or os.environ.get("RMBFUND_MODE", "public")
    if mode not in {"public", "demo"}:
        st.error("请选择公开数据或演示模式。")
        return
    identity = (str(root.resolve()), mode)
    if st.session_state.get("fund_workspace") != identity:
        for key in list(st.session_state):
            if key.startswith(("fund_", "history_", "forecast_")):
                del st.session_state[key]
        st.session_state["fund_workspace"] = identity
    theme = apply_theme()
    st.caption("知基 · 把基金看明白" + (" · 独立演示空间" if mode == "demo" else ""))
    header = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
    owns_workspace = workspace is None
    repository = None
    products = []
    try:
        if workspace is None:
            try:
                workspace = _workspace(root, mode)
            except FileNotFoundError:
                workspace = None
            except Exception:
                st.warning("公开资料暂时打不开；你仍可查看或填写本地持仓。")
        repository = portfolio if portfolio is not None else Portfolio(root, mode=mode)
        try:
            funds = repository.list_funds()
        except Exception:
            funds = []
            st.warning("暂时读不到本地持仓；仍可按代码研究基金和使用标准交易假设。原记录没有覆盖。")
        try:
            products = workspace.products() if workspace is not None else []
        except Exception:
            products = []
            st.warning("已保存基金目录暂时打不开，仍可按代码添加。")
        detail_open = bool(st.session_state.get("fund_select_next") or
                           st.session_state.get("fund_public_selection") or
                           (funds and st.session_state.get("fund_page") == "detail") or
                           (not funds and st.session_state.get("fund_preview")))
        with header:
            if detail_open:
                st.button("返回总览", key="fund_back", on_click=_overview)
                if len(funds) > 1:
                    with st.popover("切换基金"):
                        choices = {item["id"]: item for item in funds}
                        current = st.session_state.get("fund_select_next") or st.session_state.get("fund_selection")
                        if current in choices:
                            st.session_state["fund_picker"] = current
                        st.selectbox("选择基金", list(choices),
                            format_func=lambda key: fund_label(choices[key]), key="fund_picker",
                            on_change=_switch_fund, args=(tuple(choices),))
            else:
                st.title("我的基金")
            if st.button("＋ 添加基金", key="fund_add", type="secondary" if detail_open else "primary"):
                st.session_state["fund_dialog"] = "add"
        if message := st.session_state.pop("fund_flash", None):
            st.info(message)
        if not detail_open:
            _stop_intraday()
            _public_code_preview(products)
        if preview_subject := st.session_state.get("fund_public_selection"):
            detail = _fund_detail(workspace, preview_subject, now_iso())
            _selected_view(workspace, root, mode, repository, detail.get("product") or preview_subject,
                           detail, None, theme)
            return
        if not funds:
            if not st.session_state.get("fund_preview"):
                with st.container(border=True, key="empty_portfolio"):
                    st.subheader("不填持仓，也能先判断这只基金。")
                    st.write("输入基金代码，选择是否已经持有和往后看的天数，查看操作建议及理由。")
                    st.caption("目前还没有个人持仓记录；这不影响查看基金建议，不需要登录金融账户。")
                    if st.button("添加第一只基金", key="fund_add_first"):
                        st.session_state["fund_dialog"] = "add"
            public_funds = [item for item in products if item.get("kind") == "open_fund"]
            if public_funds:
                mapping = {item["id"]: item for item in public_funds}
                preview = st.selectbox("也可以先看看本地已有基金（不加入持仓）", [None, *mapping],
                    format_func=lambda key: "选择一只基金预览" if key is None else f"{mapping[key]['name']} · {mapping[key]['symbol']}",
                    key="fund_preview", on_change=_stop_intraday)
                if preview:
                    subject = mapping[preview]
                    _selected_view(workspace, root, mode, repository, subject,
                                   _fund_detail(workspace, subject, now_iso()), None, theme)
            return
        by_id = {fund["id"]: fund for fund in funds}
        next_selection = st.session_state.pop("fund_select_next", None)
        if next_selection in by_id:
            st.session_state["fund_selection"] = next_selection
            st.session_state["fund_page"] = "detail"
        selected = st.session_state.get("fund_selection")
        if selected not in by_id:
            selected = next(iter(by_id))
            st.session_state["fund_selection"] = selected
        page = st.session_state.get("fund_page", "overview")
        if page == "detail":
            fund = by_id[selected]
            subject = public_subject(fund)
            detail = _fund_detail(workspace, subject, now_iso())
            _selected_view(workspace, root, mode, repository, detail.get("product") or subject,
                           detail, fund, theme)
        else:
            _fund_overview(workspace, funds)
    finally:
        if owns_workspace and workspace is not None:
            workspace.store.close()
        dialog = st.session_state.get("fund_dialog")
        if repository is not None and dialog == "add":
            add_fund_dialog(repository, products, mode)
        elif repository is not None and isinstance(dialog, tuple) and dialog[0] == "edit":
            selected_fund = repository.get_fund(dialog[1])
            if selected_fund is not None:
                edit_fund_dialog(repository, selected_fund)


if __name__ == "__main__":
    main()

"""Personal fund forms and clear, local-only holding values."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

import streamlit as st

from rmbfund.domain import ValidationError, decimal
from rmbfund.portfolio import Portfolio
from rmbfund.ui.formatting import markdown_text, percent


PUBLIC_KEYS = ("id", "symbol", "name", "kind", "share_class", "exchange", "is_demo")


def public_subject(fund: Mapping[str, Any]) -> dict[str, Any]:
    subject = {key: value for key, value in fund["subject"].items() if key in PUBLIC_KEYS}
    if fund.get("product_id"):
        subject["id"] = fund["product_id"]
    return subject


def money(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    parsed = decimal(value)
    return f"¥{parsed:+,.2f}" if signed else f"¥{parsed:,.2f}"


def shares_text(value: Any) -> str:
    if value is None:
        return "未填写"
    return f"{decimal(value):,.4f}".rstrip("0").rstrip(".")


def fund_label(fund: Mapping[str, Any]) -> str:
    subject = fund["subject"]
    name = str(subject.get("name") or subject["symbol"])
    code = str(subject["symbol"])
    label = f"{name} · {code}" if name != code else code
    if subject.get("share_class"):
        label += f" · {subject['share_class']}类"
    return label


def _save_error(exc: Exception) -> None:
    if isinstance(exc, ValidationError):
        st.error("还没有保存。请核对基金代码、份额和成本，数值不能为负数。")
    else:
        st.error("还没有保存。暂时写不进本地持仓文件，请稍后再试；填写内容仍在这里。")


def dismiss_dialog() -> None:
    st.session_state.pop("fund_dialog", None)


@st.dialog("添加基金", width="small", on_dismiss=dismiss_dialog)
def add_fund_dialog(repository: Portfolio, products: list[dict[str, Any]], mode: str) -> None:
    st.caption("只记在本机，不需要支付宝账号。可以先添加基金，份额和成本稍后补齐。")
    code = st.text_input("基金代码", placeholder="例如 018147", key="fund_add_code").strip()
    matches = [product for product in products
               if product.get("symbol") == code and product.get("kind") == "open_fund"]
    if not matches:
        matches = [public_subject(item) for item in repository.list_funds()
                   if item["subject"]["symbol"] == code and item["subject"]["kind"] == "open_fund"]
    selected = None
    if len(matches) == 1:
        selected = matches[0]
        st.caption(markdown_text(f"已找到：{selected['name']}" +
                   (f" · {selected['share_class']}类" if selected.get("share_class") else "")))
    elif len(matches) > 1:
        choice = st.selectbox("选择这只基金的份额类别", range(len(matches)),
                              format_func=lambda i: f"{matches[i]['name']} · {matches[i].get('share_class') or '未注明'}类",
                              key="fund_add_match")
        selected = matches[choice]
    with st.form("fund_add_form"):
        name = st.text_input("本机显示名（可不填）", value=selected["name"] if selected else "",
                             key=f"fund_add_name_{code}", help="只保存在本机，不修改公开基金名称。")
        share_class = st.text_input("份额类别（可不填，如 A / C）",
                                    value=selected.get("share_class", "") if selected else "",
                                    disabled=selected is not None, key=f"fund_add_class_{code}")
        units = st.text_input("当前已确认的剩余份额（可不填）", key="fund_add_shares")
        cost = st.text_input("这些份额的持仓总成本（元，可不填）", key="fund_add_cost")
        st.caption("成本是剩余份额对应的买入成本，不是当前市值，也不是历年累计投入。")
        submitted = st.form_submit_button("保存到我的基金", type="primary", width="stretch")
    if submitted:
        subject = dict(selected) if selected else {
            "symbol": code, "name": name.strip() or code, "kind": "open_fund",
            "share_class": share_class.strip().upper(), "exchange": "", "is_demo": mode == "demo",
        }
        subject = {key: value for key, value in subject.items() if key in PUBLIC_KEYS}
        # A catalog match fixes the public identity, not the user's local label.
        subject["name"] = name.strip() or str(subject["symbol"])
        try:
            existing = next((item for item in repository.list_funds()
                             if item["subject"]["symbol"] == subject["symbol"]
                             and item["subject"]["kind"] == subject["kind"]
                             and item["subject"]["exchange"] == subject["exchange"]
                             and ((subject.get("id") is not None and item.get("product_id") == subject["id"])
                                  or (item.get("product_id") is None and
                                      (item["subject"]["share_class"] == subject["share_class"]
                                       or (not item["subject"]["share_class"] and subject.get("id") is not None))))), None)
            target_id = existing["id"] if existing else None
            # Adding an already-known fund is not an implicit reset. Editing is the
            # explicit way to clear a previously entered share or cost.
            saved = repository.save_fund(
                subject,
                shares=units if units.strip() else (existing.get("shares") if existing else None),
                cost=cost if cost.strip() else (existing.get("cost") if existing else None),
                fund_id=target_id,
            )
        except Exception as exc:
            _save_error(exc)
            return
        st.session_state["fund_select_next"] = saved["id"]
        st.session_state["fund_flash"] = "已保存到我的基金。没有联网更新，也没有提交买卖。"
        dismiss_dialog()
        st.rerun()


@st.dialog("编辑持仓", width="small", on_dismiss=dismiss_dialog)
def edit_fund_dialog(repository: Portfolio, fund: Mapping[str, Any]) -> None:
    st.write(markdown_text(fund_label(fund)))
    with st.form("fund_edit_form"):
        name = st.text_input("本机显示名", value=fund["subject"].get("name") or fund["subject"]["symbol"],
                             key=f"fund_edit_name_{fund['id']}", help="只改变本机列表和标题，不修改公开基金名称。")
        units = st.text_input("当前已确认的剩余份额", value=fund.get("shares") or "",
                              key=f"fund_edit_shares_{fund['id']}")
        cost = st.text_input("这些份额的持仓总成本（元，可不填）", value=fund.get("cost") or "",
                             key=f"fund_edit_cost_{fund['id']}")
        st.caption("只记录当前剩余份额；成本不是累计投入。空白代表不知道，填 0 代表确实为零。")
        submitted = st.form_submit_button("保存持仓", type="primary", width="stretch")
    if submitted:
        try:
            subject = public_subject(fund) | {"name": name.strip() or fund["subject"]["symbol"]}
            repository.save_fund(subject, shares=units, cost=cost, fund_id=fund["id"])
        except Exception as exc:
            _save_error(exc)
            return
        st.session_state["fund_flash"] = "持仓已保存。走势和模型概率没有因填写金额而改变。"
        dismiss_dialog()
        st.rerun()
    st.divider()
    with st.popover("移出我的基金"):
        st.write("只移除本机个人记录，净值和历史预测会保留。")
        if st.button("确认移除", key=f"fund_remove_{fund['id']}", type="primary"):
            try:
                repository.remove_fund(fund["id"])
            except Exception:
                st.error("没有移除成功，请稍后重试。")
                return
            st.session_state.pop("fund_selection", None)
            st.session_state["fund_flash"] = "已移出我的基金，公开净值和预测记录没有删除。"
            dismiss_dialog()
            st.rerun()


def render_summary(summary: Mapping[str, Any]) -> None:
    count = summary["total_count"]
    columns = st.columns(3)
    items = (("已记录持仓市值", "market_value", "market_value_count"),
             ("已填写持仓成本", "cost", "cost_count"),
             ("可计算账面浮盈亏", "profit", "profit_count"))
    for column, (label, value_key, count_key) in zip(columns, items):
        with column:
            st.metric(label, money(summary.get(value_key), signed=value_key == "profit"))
            covered = summary.get(count_key, 0)
            st.caption(f"{covered} / {count} 只基金" + (" · 仅已记录部分" if covered < count else ""))
    st.caption("按各基金最近公布净值估算，日期可能不同；未扣赎回费。浮盈亏仅统计资料齐全的持仓，不含历次卖出和现金分红。")


def render_position(value: Mapping[str, Any], *, preview: bool = False) -> None:
    if preview:
        st.caption("正在查看公开基金资料，还没有把它加入你的持仓。")
        return
    with st.container(border=True, key="position_card"):
        columns = st.columns(3)
        columns[0].metric("我的确认份额", shares_text(value.get("shares")))
        columns[1].metric("按最近净值估算市值", money(value.get("market_value")))
        columns[2].metric("账面浮盈亏", money(value.get("profit"), signed=True),
                          delta=percent(value["profit_rate"], signed=True) if value.get("profit_rate") is not None else None,
                          delta_color="off")
        if value.get("valuation_status") == "currency_mismatch":
            st.caption("净值币种与人民币成本不相容，未换汇或混算市值和盈亏。")
        elif value.get("shares") is None:
            st.caption("还没有填写份额。点“编辑持仓”补充后，就能按最近公布净值估算市值。")
        elif value.get("market_value") is None:
            st.caption("还没有可用于估算的净值；你填写的份额和成本已保留。")
        elif value.get("cost") is None:
            st.caption("未填写持仓成本，因此还不能计算盈亏。")
        elif decimal(value["cost"]) == Decimal(0):
            st.caption("持仓成本为0，不计算盈亏百分比。")

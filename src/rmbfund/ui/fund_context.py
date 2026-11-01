"""Already-available product disclosures, separate from the NAV prediction."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import streamlit as st

from rmbfund.ui.formatting import mapping, markdown_text, percent, when


_DENOMINATORS = {"nav": "基金净资产", "equity_assets": "股票资产", "total_assets": "报告中的总资产"}
_ASSETS = {"equity": "股票", "bond": "债券", "cash": "现金", "other": "其他资产"}


def _write(value: Any) -> None:
    st.markdown(markdown_text(value))


def _provenance(record: Mapping) -> None:
    st.caption(markdown_text(f"报告日期 {record.get('report_date') or '未提供'} · 来源 {record.get('source') or '未提供'}"))
    published = record.get("published_at")
    precision = record.get("publication_precision")
    label = "披露日期（仅日期）" if precision == "date" else "来源披露时间"
    st.caption(markdown_text(f"{label} {when(published)} · 取得时间 {when(record.get('observed_at'))} · "
                             f"本地可用时间 {when(record.get('available_at'))}"))
    url = record.get("source_url")
    if isinstance(url, str):
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if valid:
            st.link_button("查看披露来源", url)
            return
    st.caption("没有可用的公开来源链接。")


def _mandate(payload: Mapping) -> None:
    asset_class = {"equity": "股票", "bond": "债券", "mixed": "混合资产", "other": "其他"}.get(payload.get("asset_class"), "未披露")
    _write(f"披露投向类型：{asset_class} · 基金计价币种：{payload.get('base_currency') or '未披露'}")
    restrictions = []
    if payload.get("sector_mandate") is True:
        restrictions.append("投资范围限定行业")
    if payload.get("single_country_mandate") is True:
        restrictions.append("投资范围限定单一国家")
    if restrictions:
        _write("；".join(restrictions) + "。具体范围以披露原文为准。")
    benchmark = mapping(payload.get("benchmark"))
    if benchmark:
        _write(f"业绩比较基准：{benchmark.get('name') or '名称未披露'}")
        st.caption(markdown_text(f"基准地区 {benchmark.get('region') or '未披露'} · "
                                 f"基准币种 {benchmark.get('currency') or '未披露'} · "
                                 f"基准的股票部分占比 {percent(benchmark.get('equity_weight'), digits=2)}"))
        st.caption("业绩比较基准是用于比较表现的参照，不是基金实际持仓；基准占比不能当成真实股票仓位。")
    else:
        st.caption("本地资料没有披露业绩比较基准，不根据基金名称猜测投向。")


def _management(payload: Mapping) -> None:
    managers = payload.get("managers") or []
    if managers:
        st.dataframe([{"报告列示经理": item.get("name") or "未披露", "任职起始日期": item.get("start_date") or "未披露",
                       "任职结束日期": item.get("end_date") or "报告未注明"}
                      for item in managers], hide_index=True, width="stretch")
    else:
        st.caption("这份资料没有列示经理任职信息。")
    st.caption("这是报告当时列示的经理和任职日期，未核实今天是否仍在任；旧报告不能当作今天的经理名单。")


def _portfolio(payload: Mapping) -> None:
    denominator = _DENOMINATORS.get(payload.get("denominator"), "未说明的原披露分母")
    scope = {"top10": "前十大持仓", "partial": "部分持仓", "full": "报告披露的完整持仓"}.get(payload.get("holdings_scope"), "覆盖范围未说明")
    st.caption(f"披露范围：{scope} · 以下权重保留原分母：{denominator}。")
    st.caption("报告反映过去的持仓，不是实时仓位；部分披露没有重新凑成100%，未披露部分不代表现金或零持仓。")
    assets = mapping(payload.get("assets"))
    if assets:
        st.markdown("**披露的资产类别**")
        st.dataframe([{"资产类别": _ASSETS.get(key, key), f"占{denominator}": percent(value, digits=2)}
                      for key, value in assets.items()], hide_index=True, width="stretch")
    holdings = payload.get("holdings") or []
    st.markdown("**披露的持仓**")
    if holdings:
        st.dataframe([{"名称": row.get("name") or "未披露", f"占{denominator}": percent(row.get("weight"), digits=2),
                       "国家或地区": row.get("country") or "未披露", "行业": row.get("sector") or "未披露"}
                      for row in holdings], hide_index=True, width="stretch")
    else:
        st.caption("没有可展示的持仓明细，不推断基金当前持有或未持有某只股票。")
    for key, title in (("sectors", "行业分布"), ("countries", "国家或地区分布"), ("currencies", "币种敞口")):
        rows = payload.get(key) or []
        st.markdown(f"**{title}**")
        if not rows:
            st.caption(f"这份资料未披露{title}，缺失不表示占比为0，也不从基金名称或少数持仓推算。")
        elif key == "currencies":
            st.dataframe([{"币种": row.get("currency") or "未披露",
                           f"未对冲净敞口占{denominator}": percent(row.get("unhedged_net_weight"), digits=2)}
                          for row in rows], hide_index=True, width="stretch")
        else:
            st.dataframe([{title: row.get("name") or "未披露", f"占{denominator}": percent(row.get("weight"), digits=2)}
                          for row in rows], hide_index=True, width="stretch")


def _fees(payload: Mapping) -> None:
    st.caption("以下是报告公示费率，不是你在支付宝等渠道的实际费率，不会自动填入个人判断。")
    rates = (("management_rate", "管理费（年费率）"), ("custody_rate", "托管费（年费率）"),
             ("sales_service_rate", "销售服务费（年费率）"), ("subscription_rate", "公示申购费率"))
    st.dataframe([{"费用项目": label, "公示费率": percent(payload.get(key), digits=2) if payload.get(key) is not None else "未披露"}
                  for key, label in rates], hide_index=True, width="stretch")
    st.caption("年度运作费用通常已计入净值，不在净值涨跌上重复扣除。实际申购折扣和当前持有期赎回费请以你的渠道显示为准。")
    bands = payload.get("redemption_bands") or []
    if bands:
        st.markdown("**公示赎回费分段**")
        st.dataframe([{"最低持有天数（含）": row.get("min_days"),
                       "最高持有天数（含）": str(row["max_days"]) if row.get("max_days") is not None else "未设上限",
                       "公示赎回费率": percent(row.get("rate"), digits=2)}
                      for row in bands], hide_index=True, width="stretch")
        st.caption("持有天数口径：从确认日起按自然日计。" if payload.get("holding_day_basis") == "confirmation_calendar_days"
                   else "持有天数的计算口径未核实，不能据此替你选择当前赎回费率。")
    else:
        st.caption("这份资料没有赎回费分段，不能把缺失当作免费。")


def render_fund_context(detail: Mapping | None) -> None:
    """Render supplied disclosures only: no fetching, private fee edits or model calls."""
    detail = mapping(detail)
    st.subheader("基金资料")
    st.caption("用于了解基金投向、经理、公开持仓与费用。哪些资料实际进入预测，以当前判断页所选模型的"
               "已保存输入和来源说明为准；披露权重也可用于独立的条件情景，不能自动当作今天仓位或模型概率。")
    records = {record["family"]: record for record in detail.get("background_records", [])
               if record.get("family") in {"mandate", "management", "portfolio", "fees"}}
    # Earlier fund_detail callers supplied one flattened portfolio record.
    legacy = mapping(detail.get("portfolio"))
    if "portfolio" not in records and legacy:
        records["portfolio"] = {**legacy, "payload": legacy}
    if not records:
        st.info("本地还没有可展示的产品背景披露。缺资料不代表经理、持仓或费用为零；浏览不会联网补齐。")
        return
    sections = (("mandate", "投向与业绩比较基准", _mandate), ("management", "报告中的经理任职", _management),
                ("portfolio", "披露的持仓与行业", _portfolio), ("fees", "公示费用（非个人费率）", _fees))
    for family, title, render in sections:
        with st.expander(title):
            record = records.get(family)
            if record is None:
                st.caption("本地还没有这一类披露资料，不用其他资料猜测或补齐。")
                continue
            _provenance(record)
            render(mapping(record.get("payload")))

"""Small user-facing renderers for saved projections; no service or model calls."""
from __future__ import annotations

from collections.abc import Mapping
from html import escape

import streamlit as st

from rmbfund.ui.charts import forecast_figure, forecast_table_rows
from rmbfund.ui.formatting import markdown_text


def render_buyer_view(view: Mapping, *, key: str = "buyer_view") -> None:
    """Render the final brief copy, without interpreting internal evidence fields."""
    with st.container(border=True, key=key):
        label = view.get("label")
        if isinstance(label, str) and label:
            st.subheader(markdown_text(label))
        else:
            st.info("这次还没有可展示的建议。")
        for field, prefix in (("advice_reason", ""), ("next_step", "接下来：")):
            value = view.get(field)
            if isinstance(value, str) and value:
                st.markdown(markdown_text(prefix + value))
        execution = view.get("execution_status")
        status = execution.get("status") if isinstance(execution, Mapping) else execution
        if status == "blocked":
            st.caption("当前相关买卖暂不能办理；这里没有提交交易。")
        elif status == "unknown":
            st.caption("操作前请在购买渠道确认是否可以办理。")
        day = view.get("data_date")
        st.caption(markdown_text(f"参考净值截至 {day}。" if isinstance(day, str) and day else
                                 "参考净值日期未提供。"))


def _accessible_table(rows: list[dict[str, str]], *, caption: str) -> None:
    """A real HTML table, usable with keyboard/screen readers, not a canvas grid.

    Only whitelist table rows produced by ``forecast_table_rows`` enter here.
    HTML is escaped including labels; no model identities or raw JSON are sent.
    """
    if not rows:
        return
    columns = list(rows[0])
    header = "".join(f'<th scope="col">{escape(column)}</th>' for column in columns)
    body = "".join('<tr tabindex="0">' + "".join(
        f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns
    ) + "</tr>" for row in rows)
    st.html(
        '<style>.buyer-chart-table{overflow-x:auto;max-width:100%;margin-block:1rem}'
        '.buyer-chart-table table{width:100%;border-collapse:collapse;font-size:.9rem;'
        'font-variant-numeric:tabular-nums}.buyer-chart-table caption{text-align:left;'
        'font-weight:600;margin-bottom:.6rem}.buyer-chart-table th,'
        '.buyer-chart-table td{text-align:left;padding:.55rem .65rem;border-bottom:1px solid '
        'color-mix(in srgb,currentColor 16%,transparent)}.buyer-chart-table tr:focus{'
        'outline:2px solid currentColor;outline-offset:-2px}'
        '@media(forced-colors:active){.buyer-chart-table tr:focus{outline:2px solid Highlight}}'
        '</style><div class="buyer-chart-table"><table>'
        f'<caption>{escape(caption)}</caption><thead><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def render_forecast_chart(snapshot: Mapping, *, key: str = "forecast_chart", theme: str = "light") -> None:
    """Show real NAV and one return endpoint, with the same public data in tables.

    The host passes the current light/dark theme. Plot data, tooltip customdata,
    alt text, and accessible tables never receive ``snapshot.identity``.
    """
    st.subheader("净值历史与计划期末估计")
    forecast = snapshot.get("forecast") or {}
    point = forecast.get("point")
    st.plotly_chart(
        forecast_figure(snapshot, theme=theme), width="stretch", theme=None, key=key,
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
        alt="上图为已公布单位净值；下图菱形为计划期末预期涨跌，竖线为中间80%范围，横线为0%。完整数值可展开图表数据查看。",
    )
    start, end = forecast.get("application_date"), forecast.get("sale_date")
    if start and end:
        st.caption(markdown_text(f"下图估计 {start} 至 {end} 的净值变化；日期按这次计划假定，不代表未来实际净值日。"))
    if not point:
        st.info("还没有与这次计划的起点和期限对应的涨跌估计；不拿其他期限补画。")
    elif point.get("lower") is None or point.get("upper") is None:
        st.caption("这次只保存了预期涨跌，尚无对应的中间80%范围。")
    st.caption("历史模型参考，尚未验证择时优势。净值未计分红再投资，估计未扣申赎费；竖线是分布范围，不是平均值的误差或保证。")
    with st.expander("图表数据（可用键盘查看）"):
        tables = forecast_table_rows(snapshot)
        _accessible_table(tables["history"], caption="已公布的单位净值")
        _accessible_table(tables["forecast"], caption="计划期末的涨跌估计")
        if not tables["history"] and not tables["forecast"]:
            st.caption("这次还没有可展示的数值。")

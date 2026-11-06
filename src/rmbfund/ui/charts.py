"""Daily price/NAV charts: one measure, one axis, and a table-first data contract.

The blue series always means the selected product's unadjusted daily value, not
performance including distributions. There is deliberately no multi-asset raw-
price overlay: different scales belong in separate charts or a common index.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, localcontext
from html import escape
from math import isfinite
from typing import Any, Iterable, Mapping

import plotly.graph_objects as go

from rmbfund.buyer_view import build_forecast_chart as build_forecast_chart
from rmbfund.domain import ValidationError, decimal

# Reference dataviz palette. Light/dark first three slots were validated with
# validate_palette.js --pairs all. This single-measure chart uses slot 1 only.
PALETTES = {
    "light": {
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "series": ("#3987e5", "#d95926", "#199e70"),
        "surface": "#1a1a19",
        "text": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def price_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    indexed: bool = False,
    base_date: str | None = None,
) -> list[dict[str, Any]]:
    """Validate, sort, and optionally index a series without forward filling.

    A requested base must be an actual observation, not a fabricated trading day.
    All original observations remain in the accessible table, including dates
    before the chosen base. Indexing is a price/NAV ratio, never total return.
    """
    result: list[dict[str, Any]] = []
    dates: set[str] = set()
    for row in rows:
        try:
            day = date.fromisoformat(str(row["date"])).isoformat()
            value = decimal(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("日级数据缺少有效的日期或数值，请先核验来源") from exc
        if value <= 0 or not isfinite(float(value)) or float(value) <= 0:
            raise ValidationError("日级价格/净值必须为大于零的有限数")
        if day in dates:
            raise ValidationError(f"日期 {day} 存在多条值，请先核验版本，图表未擅自选取")
        dates.add(day)
        result.append(
            {
                "date": day,
                "value": str(value),
                "basis": str(row.get("basis", "未注明口径")),
                "source": str(row.get("source", "未注明来源")),
                "observed_at": str(row.get("observed_at", "未注明")),
            }
        )
    result.sort(key=lambda row: row["date"])
    if not result:
        return result
    base_date = base_date or result[0]["date"]
    bases = [row for row in result if row["date"] == base_date]
    if indexed and not bases:
        raise ValidationError("指数化基期必须为当前可见数据中的实际日期")
    base = decimal(bases[0]["value"]) if bases else decimal(result[0]["value"])
    for row in result:
        plotted = decimal(row["value"]) / base * 100 if indexed else decimal(row["value"])
        if not isfinite(float(plotted)) or float(plotted) <= 0:
            raise ValidationError("数值超出图表可显示范围，请核验原始数据")
        row["plot_value"] = float(plotted)
        row["base_date"] = base_date if indexed else ""
    return result


def price_figure(
    rows: Iterable[Mapping[str, Any]],
    *,
    product_name: str,
    kind: str = "etf",
    indexed: bool = False,
    base_date: str | None = None,
    theme: str = "light",
    corporate_events: Iterable[Mapping[str, Any]] = (),
    compact: bool = False,
    currency: str | None = "CNY",
    native_theme: bool = False,
) -> go.Figure:
    """Build a single-axis interactive line with a stable semantic identity.

    Plotly supports limited HTML in labels, so every external string is escaped.
    The host UI provides the same rows as a keyboard-accessible table and CSV.
    Compact mode lets that host own the title and shows only the endpoint marker;
    dates, values, missing observations, and the original default remain unchanged.
    """
    if theme not in PALETTES:
        raise ValidationError("图表主题只支持 light / dark")
    prepared = price_rows(rows, indexed=indexed, base_date=base_date)
    palette = PALETTES[theme]
    label = "日级原价" if kind == "etf" else "单位净值"
    unit = "元" if currency == "CNY" else escape(str(currency or "币种未注明"))
    scale = f"{label}指数（基期 = 100）" if indexed else f"{label}（{unit}）"
    title = f"{escape(str(product_name))} · {label}变化"
    subtitle = "未含分红，不是总回报；缺失日期不补值"
    if indexed and prepared:
        subtitle += f"；基期 {prepared[0]['base_date']} = 100"
    figure = go.Figure()
    if prepared:
        figure.add_trace(
            go.Scatter(
                x=[row["date"] for row in prepared],
                y=[row["plot_value"] for row in prepared],
                name=escape(str(product_name)),
                mode="lines+markers",
                line={"color": palette["series"][0], "width": 2, "dash": "solid"},
                marker={
                    "color": palette["series"][0],
                    "size": ([0] * (len(prepared) - 1) + [8]) if compact else 8,
                    "line": {"color": palette["surface"], "width": 2},
                },
                customdata=[
                    [
                        escape(row["value"]),
                        escape(row["basis"]),
                        escape(row["source"]),
                        escape(row["observed_at"]),
                    ]
                    for row in prepared
                ],
                hovertemplate=(
                    ("%{y:,.4f}<br>日期 %{x|%Y-%m-%d}"
                     f"<br>原始值 %{{customdata[0]}} {unit}<extra></extra>")
                    if compact else
                    ("%{y:,.4f}<br>日期 %{x|%Y-%m-%d}"
                     f"<br>原始值 %{{customdata[0]}} {unit}"
                     "<br>口径 %{customdata[1]}<br>来源 %{customdata[2]}"
                     "<br>实际观察 %{customdata[3]}<extra></extra>")
                ),
                connectgaps=False,
            )
        )
        figure.add_annotation(
            x=prepared[-1]["date"],
            y=prepared[-1]["plot_value"],
            text=f"{prepared[-1]['plot_value']:,.4f}",
            showarrow=False,
            xanchor="right",
            yshift=18,
            font={"color": palette["text"], "size": 12},
            bgcolor=palette["surface"],
        )
    marked_splits: set[str] = set()
    for event in corporate_events:
        if event.get("kind") != "split" or not prepared:
            continue
        try:
            event_day = date.fromisoformat(str(event.get("event_date", ""))).isoformat()
        except ValueError:
            # The accessible event table retains malformed provenance for review.
            continue
        if (
            not prepared[0]["date"] <= event_day <= prepared[-1]["date"]
            or event_day in marked_splits
        ):
            continue
        marked_splits.add(event_day)
        figure.add_shape(
            type="line",
            x0=event_day,
            x1=event_day,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            layer="below",
            line={"color": palette["axis"], "width": 1, "dash": "dot"},
        )
        figure.add_annotation(
            x=event_day,
            y=1,
            xref="x",
            yref="paper",
            text="份额拆分",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=2,
            font={"color": palette["secondary"], "size": 12},
            bgcolor=palette["surface"],
        )
    figure.update_layout(
        title=None if compact else {
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "font": {"size": 17, "color": palette["text"]},
            "x": 0,
        },
        height=300 if compact else 410,
        margin={"l": 54, "r": 20, "t": 28, "b": 44} if compact else {"l": 58, "r": 25, "t": 82, "b": 58},
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        font={"family": "system-ui, sans-serif", "color": palette["text"]},
        showlegend=False,
        hovermode="x unified",
        hoverdistance=-1,
        hoverlabel={
            "bgcolor": palette["surface"],
            "font": {"color": palette["text"]},
        },
        xaxis={
            "title": None if compact else "数据日期（上海）",
            "type": "date",
            "nticks": 5 if compact else None,
            "hoverformat": "%Y-%m-%d",
            "showgrid": False,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "data",
            "spikethickness": 1,
            "spikecolor": palette["axis"],
            "spikedash": "solid",
            "linecolor": palette["axis"],
            "tickfont": {"color": palette["secondary"]},
        },
        yaxis={
            "title": scale,
            "showgrid": True,
            "gridcolor": palette["grid"],
            "gridwidth": 1,
            "zeroline": False,
            "tickformat": ",.4~f",
            "tickfont": {"color": palette["secondary"]},
        },
    )
    if native_theme:
        # Native templates update in the browser; explicit colors do not. Use a
        # blue validated on both surfaces, and leave ink/surfaces to the template.
        blue = PALETTES["light"]["series"][0]
        figure.update_traces(line_color=blue, marker_color=blue,
                             marker_line_color=None, marker_line_width=0)
        figure.update_layout(
            paper_bgcolor=None, plot_bgcolor=None, font_color=None,
            xaxis_tickfont_color=None, xaxis_linecolor=None, xaxis_spikecolor=None,
            yaxis_tickfont_color=None, yaxis_gridcolor=None,
            hoverlabel_bgcolor=None, hoverlabel_font_color=None,
        )
        figure.update_annotations(font_color=None, bgcolor=None)
    return figure


def intraday_figure(
    rows: Iterable[Mapping[str, Any]],
    *,
    product_name: str,
) -> go.Figure:
    """Plot actual estimate samples, not a live quote or a reconstructed history.

    Callers retain one product/source/day/baseline scope and reset on a change;
    mixed scopes, snapshot-generation times, and duplicate times are rejected.
    Unknown baseline dates remain unknown: zero is only an axis reference, not a
    fabricated opening observation. ``checked_at`` never supplies a plotted time.
    Source clock strings stay intact, including naive times with no assumed zone.
    The host supplies the accessible table and Streamlit's native theme/sizing.
    """
    prepared: list[tuple[datetime, str, float]] = []
    seen: set[datetime] = set()
    scope: tuple[Any, ...] | None = None
    source_label = ""
    for row in rows:
        if row.get("source_time_kind") != "estimate":
            raise ValidationError("图表只接受估算时间；整份快照的生成时间不能作为基金行情时间")
        try:
            source = row["source"]
            source_name = row["source_name"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError
            if not isinstance(source_name, str) or not source_name.strip():
                raise ValueError
            source_time = row["source_time"]
            clock = datetime.fromisoformat(source_time)
            # Require an actual time in the calendar-date form Plotly accepts,
            # without normalizing a naive source clock to UTC or local time.
            if (source_time[:10] != clock.date().isoformat()
                    or source_time[10:11] not in {"T", " "}):
                raise ValueError
            baseline = row.get("baseline_date")
            baseline = date.fromisoformat(baseline) if baseline is not None else None
            change = float(decimal(row["change"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("估算记录缺少有效来源、估算时间、基准日期或有限涨跌幅") from exc
        # Plotly's date axis uses source wall-clock strings. Do not combine
        # different clock offsets (or naive and aware clocks) into one path.
        row_scope = (source, clock.date(), baseline, clock.utcoffset())
        if scope is None:
            scope = row_scope
            source_label = source_name
        elif row_scope != scope:
            raise ValidationError("估算记录须为同一来源、日期、基准日期与时区；切换时请重置记录")
        if clock in seen:
            raise ValidationError("估算记录存在重复来源时间，图表未擅自选取版本")
        seen.add(clock)
        prepared.append((clock, source_time, change))
    prepared.sort(key=lambda point: point[0])

    figure = go.Figure()
    if prepared:
        blue = PALETTES["light"]["series"][0]
        figure.add_trace(
            go.Scatter(
                x=[point[1] for point in prepared],
                y=[point[2] for point in prepared],
                name=escape(str(product_name)),
                mode="markers" if len(prepared) == 1 else "lines+markers",
                line={"color": blue, "width": 2, "shape": "linear"},
                marker={"color": blue, "size": [0] * (len(prepared) - 1) + [8],
                        "line": {"width": 0}},
                customdata=[[escape(point[1])] for point in prepared],
                hovertemplate=(
                    f"{escape(source_label)} · %{{y:+.2%}}<br>"
                    "%{customdata[0]}<extra></extra>"
                ),
                connectgaps=False,
            )
        )
        figure.add_annotation(
            x=prepared[-1][1], y=prepared[-1][2],
            text=f"{prepared[-1][2]:+.2%}",
            showarrow=False, xanchor="right", yshift=16,
            font={"size": 12},
        )
    else:
        figure.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="还没有估算记录", showarrow=False, font={"size": 12},
        )
    # Leave all ink, surfaces, grid/zero-line colors, and hover chrome to the
    # native template. Blue has been validated on both native light/dark surfaces.
    figure.update_layout(
        template=None,
        height=260,
        margin={"l": 64, "r": 20, "t": 28, "b": 44},
        font={"family": "system-ui, sans-serif"},
        showlegend=False,
        hovermode="x unified",
        hoverdistance=-1,
        xaxis={
            "type": "date", "nticks": 5,
            "tickmode": "array" if len(prepared) == 1 else "auto",
            "tickvals": [prepared[0][1]] if len(prepared) == 1 else None,
            "visible": bool(prepared),
            "hoverformat": "%Y-%m-%d %H:%M:%S",
            "showgrid": False, "showspikes": True,
            "spikemode": "across", "spikesnap": "data",
            "spikethickness": 1, "spikedash": "solid",
        },
        yaxis={
            "title": "来源估算涨跌幅", "tickformat": ".2%", "rangemode": "tozero",
            # Give an all-zero series room around the actual dots instead of
            # Plotly's 0..100% fallback, which also clips a dot at the bottom.
            "range": [-0.001, 0.001] if prepared and not any(p[2] for p in prepared) else None,
            "visible": bool(prepared),
            "showgrid": True, "gridwidth": 1,
            "zeroline": True, "zerolinewidth": 1,
        },
    )
    return figure


# Fixed research identities, independent of filters, input order, or performance.
# Cash is a neutral reference, not a fourth categorical/status hue. The three
# colored paths reuse the validated first three slots above, with dashes as a
# second identity channel in both themes and in monochrome exports.
RESEARCH_STRATEGIES = ("cash", "buyhold", "ma20", "monthly_mom60_hold30")
RESEARCH_LABELS = {
    "cash": "现金不投",
    "buyhold": "买入持有",
    "ma20": "MA20",
    "monthly_mom60_hold30": "月度动量60 · 持有30天",
}
RESEARCH_DASHES = {
    "cash": "dot",
    "buyhold": "solid",
    "ma20": "dash",
    "monthly_mom60_hold30": "dashdot",
}


def _research_float(value: Decimal) -> float:
    plotted = float(value)
    if not value.is_finite() or abs(value) > Decimal("1e24") or not isfinite(plotted):
        raise ValidationError("研究数值超出图表可显示范围，请核验原始报告")
    return plotted


def research_curve_rows(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prepare the exact chart/table rows without changing the retained report.

    Only one window/scenario belongs on a chart. Zero model assets are valid;
    prices still use the stricter existing price helpers. The denominator is
    each result's *initial cash*, not its first daily valuation. Drawdown starts
    with that same initial-cash peak, so a first-day fee/loss is not erased.
    Explicitly incomplete valuations stay as gaps; after a missing peak, later
    drawdown stays unknown rather than being calculated against a guessed peak.
    """
    from rmbfund.research_artifacts import presentation_decimal

    series: dict[str, Mapping[str, Any]] = {}
    for result in results:
        strategy = str(result.get("strategy", ""))
        if strategy not in RESEARCH_STRATEGIES or strategy in series:
            raise ValidationError("研究曲线须为唯一的固定四路径，请先选择单一窗口与情景")
        series[strategy] = result
    prepared: list[dict[str, Any]] = []
    for strategy in RESEARCH_STRATEGIES:
        if strategy not in series:
            continue
        result = series[strategy]
        try:
            initial = presentation_decimal(result["assumptions"]["initial_cash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("研究曲线缺少有效初始现金") from exc
        if initial <= 0:
            raise ValidationError("研究曲线初始现金必须为正")
        daily: list[tuple[str, Decimal | None, bool]] = []
        seen: set[str] = set()
        for row in result.get("daily", []):
            try:
                day = date.fromisoformat(str(row["date"])).isoformat()
                complete = row.get("model_complete") is not False
                raw_assets = row["model_assets"]
                assets = (
                    None
                    if raw_assets is None and not complete
                    else presentation_decimal(raw_assets)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError("研究日级记录缺少有效日期或模型资产") from exc
            if assets is not None and assets < 0:
                raise ValidationError("研究模型资产不得为负")
            if day in seen:
                raise ValidationError(f"研究路径 {strategy} 在 {day} 有重复记录，未擅自选取")
            seen.add(day)
            daily.append((day, assets if complete else None, complete))
        peak = initial
        peak_known = True
        for day, assets, complete in sorted(daily):
            indexed = drawdown = None
            if assets is None or not complete:
                peak_known = False
            else:
                # A missing valuation may hide a higher peak. Later asset
                # indices remain usable, but subsequent drawdown is unknown.
                # Do not silently bridge that uncertainty or reset the peak.
                with localcontext() as context:
                    context.prec = 90
                    peak = max(peak, assets)
                    indexed = _research_float(assets / initial * 100)
                    if peak_known:
                        drawdown = _research_float((assets / peak - 1) * 100)
            prepared.append(
                {
                    "date": day,
                    "strategy": strategy,
                    "label": RESEARCH_LABELS[strategy],
                    "scenario": str(result.get("scenario", "")),
                    "product_name": str(result.get("product", {}).get("name", "")),
                    "model_assets": str(assets) if assets is not None else None,
                    "initial_cash": str(initial),
                    "assets_index": indexed,
                    "drawdown_pct": drawdown,
                    "model_complete": complete,
                }
            )
    return prepared


def _research_label_positions(values: list[float], low: float, high: float) -> list[float]:
    """Reserve separate endpoint label lanes; leaders keep each label attached."""
    targets = [(value - low) / (high - low) for value in values]
    order = sorted(range(len(targets)), key=lambda index: (targets[index], index))
    placed: dict[int, float] = {}
    previous = -1.0
    for index in order:
        placed[index] = max(0.06, min(0.94, targets[index]), previous + 0.10)
        previous = placed[index]
    following = 1.04
    for index in reversed(order):
        placed[index] = min(placed[index], following - 0.10)
        following = placed[index]
    return [placed[index] for index in range(len(values))]


def research_figure(
    results: Iterable[Mapping[str, Any]],
    *,
    measure: str = "assets_index",
    title: str = "固定四路径 · 条件研究回放",
    theme: str = "light",
) -> go.Figure:
    """One indexed-assets OR drawdown axis, with stable legend/hover/direct labels.

    There is no winning-path selection, re-ranking, smoothing, or gap filling.
    Hosts must expose ``research_curve_rows`` as the equivalent table/CSV view.
    """
    if theme not in PALETTES:
        raise ValidationError("图表主题只支持 light / dark")
    if measure not in {"assets_index", "drawdown_pct"}:
        raise ValidationError("研究图表只支持模型资产指数或单独的回撤图")
    prepared = research_curve_rows(results)
    palette = PALETTES[theme]
    colors = {
        "cash": palette["secondary"],
        "buyhold": palette["series"][0],
        "ma20": palette["series"][1],
        "monthly_mom60_hold30": palette["series"][2],
    }
    figure = go.Figure()
    endpoints: list[tuple[str, dict[str, Any]]] = []
    for strategy in RESEARCH_STRATEGIES:
        rows = [row for row in prepared if row["strategy"] == strategy]
        if not rows:
            continue
        endpoint = next((row for row in reversed(rows) if row[measure] is not None), None)
        if endpoint is not None:
            endpoints.append((strategy, endpoint))
        figure.add_trace(
            go.Scatter(
                x=[row["date"] for row in rows],
                y=[row[measure] for row in rows],
                name=escape(RESEARCH_LABELS[strategy]),
                legendgroup=strategy,
                mode="lines+markers",
                line={
                    "color": colors[strategy],
                    "width": 2,
                    "dash": RESEARCH_DASHES[strategy],
                    "shape": "linear",
                },
                marker={
                    "color": colors[strategy],
                    "size": [8 if row is endpoint else 0 for row in rows],
                    "line": {"color": palette["surface"], "width": 2},
                },
                customdata=[
                    [
                        escape(str(row["model_assets"])) if row["model_assets"] is not None else "",
                        escape(row["initial_cash"]),
                        row["assets_index"],
                        row["drawdown_pct"],
                        escape(row["scenario"]),
                        escape(row["product_name"]),
                    ]
                    for row in rows
                ],
                hovertemplate=(
                    escape(RESEARCH_LABELS[strategy])
                    + (
                        " · %{y:,.2f} 指数<br>模型资产 %{customdata[0]} 元 · 回撤 %{customdata[3]:,.2f}%"
                        if measure == "assets_index"
                        else " · %{y:,.2f}% 回撤<br>模型资产 %{customdata[0]} 元 · 指数 %{customdata[2]:,.2f}"
                    )
                    + "<extra></extra>"
                ),
                connectgaps=False,
            )
        )
    baseline = 100.0 if measure == "assets_index" else 0.0
    values = [row[measure] for row in prepared if row[measure] is not None] + [baseline]
    low, high = min(values), max(values)
    span = high - low or (20.0 if measure == "assets_index" else 2.0)
    low, high = low - span * 0.12, high + span * 0.15
    label_positions = _research_label_positions([row[measure] for _, row in endpoints], low, high)
    # The plot area is 260px high. Leaders connect individually spaced labels
    # to the actual endpoints, including coincident cash/strategy endpoints.
    for (strategy, row), position in zip(endpoints, label_positions):
        endpoint_position = (row[measure] - low) / (high - low)
        suffix = "%" if measure == "drawdown_pct" else ""
        figure.add_annotation(
            x=row["date"],
            y=row[measure],
            text=f"{escape(RESEARCH_LABELS[strategy])}  {row[measure]:,.2f}{suffix}",
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=palette["axis"],
            ax=20,
            ay=(endpoint_position - position) * 260,
            xanchor="left",
            font={"color": palette["text"], "size": 12},
            bgcolor=palette["surface"],
            borderpad=2,
        )
    subtitle = (
        "模型资产 / 初始现金 × 100；含现金、持仓与应收，非已核验实绩"
        if measure == "assets_index"
        else "相对历史模型资产峰值；初始现金参与峰值，保留首日费用损失"
    )
    figure.add_hline(
        y=baseline,
        line_width=1,
        line_color=palette["axis"],
        layer="below",
    )
    figure.update_layout(
        title={
            "text": f"{escape(str(title))}<br><sup>{subtitle}</sup>",
            "font": {"size": 17, "color": palette["text"]},
            "x": 0,
        },
        height=440,
        margin={"l": 64, "r": 235, "t": 85, "b": 95},
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        font={"family": "system-ui, sans-serif", "color": palette["text"]},
        showlegend=len(figure.data) >= 2,
        legend={
            "orientation": "h",
            "y": -0.23,
            "x": 0,
            "font": {"color": palette["text"]},
            "traceorder": "normal",
        },
        hovermode="x unified",
        hoverdistance=-1,
        hoverlabel={"bgcolor": palette["surface"], "font": {"color": palette["text"]}},
        xaxis={
            "title": "经济估值日期（历史回看）",
            "type": "date",
            "hoverformat": "%Y-%m-%d",
            "showgrid": False,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "data",
            "spikethickness": 1,
            "spikecolor": palette["axis"],
            "spikedash": "solid",
            "linecolor": palette["axis"],
            "tickfont": {"color": palette["secondary"]},
        },
        yaxis={
            "title": "模型资产指数（初始现金 = 100）"
            if measure == "assets_index"
            else "日级回撤（%）",
            "range": [low, high],
            "showgrid": True,
            "gridcolor": palette["grid"],
            "gridwidth": 1,
            "zeroline": False,
            "tickformat": ",.2~f",
            "ticksuffix": "%" if measure == "drawdown_pct" else "",
            "tickfont": {"color": palette["secondary"]},
        },
    )
    return figure


def forecast_figure(snapshot: Mapping[str, Any], *, theme: str = "light") -> go.Figure:
    """Show actual NAV separately from exact conditional return targets.

    Each future target is an independent H/L forecast. The p10/p90 segment is
    independent of its mean marker, and different horizons are never connected
    into a fabricated future NAV path. ``forecast_table_rows`` is the public,
    equivalent table contract.
    """
    from plotly.subplots import make_subplots

    if theme not in PALETTES:
        raise ValidationError("图表主题只支持 light / dark")
    palette = PALETTES[theme]
    history = snapshot.get("history") or {}
    forecast = snapshot.get("forecast") or {}
    rows = history.get("points") or []
    point = forecast.get("point")
    figure = make_subplots(
        rows=2, cols=1, vertical_spacing=0.24,
        subplot_titles=("已公布的单位净值", "计划期末的涨跌估计"),
    )
    if rows:
        figure.add_trace(go.Scatter(
            x=[row["date"] for row in rows], y=[float(row["value"]) for row in rows],
            name="已公布单位净值", mode="lines+markers", legendrank=1,
            line={"color": palette["series"][0], "width": 2, "shape": "linear"},
            marker={"color": palette["series"][0], "size": [0] * (len(rows) - 1) + [8],
                    "symbol": "circle", "line": {"color": palette["surface"], "width": 2}},
            customdata=[[escape(str(row["value"]))] for row in rows],
            hovertemplate="<b>%{customdata[0]}</b> · 单位净值<br>净值日期 %{x|%Y-%m-%d}<extra></extra>",
            connectgaps=False,
        ), row=1, col=1)
        figure.add_annotation(
            x=rows[-1]["date"], y=float(rows[-1]["value"]), xref="x", yref="y",
            text=f"{float(rows[-1]['value']):,.4f}", showarrow=False,
            xanchor="right", yshift=17,
            font={"color": palette["text"], "size": 12}, bgcolor=palette["surface"],
        )
    else:
        figure.add_annotation(
            x=0.5, y=0.83, xref="paper", yref="paper", showarrow=False,
            text="还没有可展示的实际净值记录", font={"color": palette["secondary"], "size": 13},
        )
    points = forecast.get("points")
    all_points = list(points or ())
    all_labels = [str(row.get("label") or f"H={row.get('horizon')}") for row in all_points]
    plotted_points = [row for row in all_points
                      if row.get("status") in {"available", "partial"}
                      and row.get("expected_change") is not None]
    if points is not None:
        labels = [str(row.get("label") or f"H={row.get('horizon')}") for row in plotted_points]
        means = [float(row["expected_change"]) for row in plotted_points]
        if means:
            figure.add_trace(go.Scatter(
                x=labels, y=means, mode="markers", name="各目标点预期涨跌", legendrank=2,
                marker={"color": palette["series"][1], "size": 12, "symbol": "diamond",
                        "line": {"color": palette["surface"], "width": 2}},
                customdata=[[row.get("horizon"), row.get("date") or "条件日期未映射",
                             row.get("status"), row.get("expected_change")]
                            for row in plotted_points],
                hovertemplate=("<b>%{y:+.2%}</b> · %{customdata[0]}笔净值"
                               "<br>条件日期 %{customdata[1]}<br>预期涨跌 %{customdata[3]:+.2%}"
                               "<br>状态 %{customdata[2]}<extra></extra>"),
            ), row=2, col=1)
        interval_index = 0
        values = [0, *means]
        for row in plotted_points:
            lower, upper = row.get("lower"), row.get("upper")
            if lower is None or upper is None:
                continue
            label = str(row.get("label") or f"H={row.get('horizon')}")
            figure.add_trace(go.Scatter(
                x=[label, label], y=[lower, upper], mode="lines+markers",
                name="各目标点中间80%范围" if interval_index == 0 else "各目标点中间80%范围",
                legendrank=3, showlegend=interval_index == 0,
                line={"color": palette["series"][1], "width": 2},
                marker={"color": palette["series"][1], "size": 12, "symbol": "line-ew"},
                customdata=[[row.get("horizon"), row.get("date") or "条件日期未映射",
                             row.get("status")] ] * 2,
                hovertemplate=("<b>%{y:+.2%}</b><br>%{customdata[0]}笔净值 · "
                               "条件日期 %{customdata[1]}<br>中间80%范围端点"
                               "<br>状态 %{customdata[2]}<extra></extra>"),
                connectgaps=False,
            ), row=2, col=1)
            values.extend([lower, upper])
            interval_index += 1
        if means:
            figure.add_annotation(
                x=labels[-1], y=means[-1], xref="x2", yref="y2", showarrow=False,
                text=f"H={plotted_points[-1].get('horizon')} {means[-1]:+.2%}",
                xanchor="left", xshift=19,
                font={"color": palette["text"], "size": 12}, bgcolor=palette["surface"],
            )
            low, high = min(values), max(values)
            span = high - low or 0.02
            figure.update_yaxes(range=[low - span * 0.18, high + span * 0.18], row=2, col=1)
        else:
            figure.add_annotation(
                x=0.5, y=0.19, xref="paper", yref="paper", showarrow=False,
                text="没有可绘制的精确目标点；缺口见下方表格。",
                font={"color": palette["secondary"], "size": 13},
            )
            figure.update_yaxes(range=[-0.01, 0.01], row=2, col=1)
    elif point:
        day, mean = point["date"], point["expected_change"]
        lower, upper = point.get("lower"), point.get("upper")
        if lower is not None and upper is not None:
            figure.add_trace(go.Scatter(
                x=[day, day], y=[lower, upper], mode="lines+markers",
                name="中间80%范围", legendrank=3,
                line={"color": palette["series"][1], "width": 2},
                marker={"color": palette["series"][1], "size": 12, "symbol": "line-ew"},
                customdata=[[lower, upper], [lower, upper]],
                hovertemplate="<b>%{customdata[0]:+.2%} ～ %{customdata[1]:+.2%}</b><br>中间80%范围<extra></extra>",
                connectgaps=False,
            ), row=2, col=1)
        figure.add_trace(go.Scatter(
            x=[day], y=[mean], mode="markers", name="预期涨跌", legendrank=2,
            marker={"color": palette["series"][1], "size": 12, "symbol": "diamond",
                    "line": {"color": palette["surface"], "width": 2}},
            hovertemplate="<b>%{y:+.2%}</b> · 预期涨跌<extra></extra>",
        ), row=2, col=1)
        figure.add_annotation(
            x=day, y=mean, xref="x2", yref="y2", showarrow=False,
            text=f"预期 {mean:+.2%}", xanchor="left", xshift=19,
            font={"color": palette["text"], "size": 12}, bgcolor=palette["surface"],
        )
        values = [0, mean, *([lower, upper] if lower is not None and upper is not None else [])]
        low, high = min(values), max(values)
        span = high - low or 0.02
        figure.update_yaxes(range=[low - span * 0.18, high + span * 0.18], row=2, col=1)
    else:
        figure.add_annotation(
            x=0.5, y=0.19, xref="paper", yref="paper", showarrow=False,
            text="缺少与这次计划对应的涨跌估计", font={"color": palette["secondary"], "size": 13},
        )
        figure.update_yaxes(range=[-0.01, 0.01], row=2, col=1)
    figure.add_hline(y=0, line_width=1, line_color=palette["axis"], row=2, col=1)
    figure.update_layout(
        height=630, margin={"l": 70, "r": 25, "t": 50, "b": 115},
        paper_bgcolor=palette["surface"], plot_bgcolor=palette["surface"],
        font={"family": "system-ui, sans-serif", "color": palette["text"], "size": 12},
        showlegend=len(figure.data) >= 2,
        legend={"orientation": "h", "x": 0, "y": -0.19,
                "entrywidth": 105, "entrywidthmode": "pixels", "font": {"color": palette["text"]}},
        hovermode="closest", hoverdistance=32,
        hoverlabel={"bgcolor": palette["surface"], "font": {"color": palette["text"]}},
    )
    figure.update_annotations(font_color=palette["text"], font_size=14)
    for row in (1, 2):
        figure.update_xaxes(
            showgrid=False, linecolor=palette["axis"], tickfont_color=palette["secondary"],
            showspikes=True, spikemode="across", spikesnap="data", spikethickness=1,
            spikecolor=palette["axis"], spikedash="solid", automargin=True, row=row, col=1,
        )
        figure.update_yaxes(
            showgrid=True, gridcolor=palette["grid"], gridwidth=1, zeroline=False,
            tickfont_color=palette["secondary"], automargin=True, row=row, col=1,
        )
    currency = {"CNY": "元", "USD": "美元", "HKD": "港元"}.get(
        history.get("currency"), str(history.get("currency") or "币种未注明"))
    figure.update_yaxes(
        title_text=f"单位净值（{escape(currency)}）", tickformat=",.4~f", row=1, col=1,
    )
    figure.update_xaxes(type="date", nticks=3, hoverformat="%Y-%m-%d", row=1, col=1)
    figure.update_yaxes(title_text="净值涨跌", tickformat="+.1%", row=2, col=1)
    point_labels = all_labels if points is not None else []
    point_ticktext = [
        f"{label}<br>{row.get('date')}" if row.get("date") else f"{label}<br>日期未映射"
        for label, row in zip(point_labels, all_points, strict=True)
    ]
    figure.update_xaxes(
        type="category", showspikes=False,
        tickmode="array", categoryorder="array",
        categoryarray=point_labels if points is not None else [],
        tickvals=point_labels if points is not None else [point["date"]] if point else [],
        ticktext=point_ticktext if points is not None else [point["date"]] if point else [],
        title_text="申请后目标净值点<br>日期为条件日程，不是已知未来净值日" if points is not None
        else "条件计划期末<br>不是已知的未来净值日", row=2, col=1,
    )
    return figure


def forecast_table_rows(snapshot: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Only chart-visible facts, never arbitrary snapshot keys or model identity."""
    history = snapshot.get("history") or {}
    forecast = snapshot.get("forecast") or {}
    currency = {"CNY": "人民币", "USD": "美元", "HKD": "港元"}.get(
        history.get("currency"), str(history.get("currency") or "未注明"))
    history_rows = [{"净值日期": row["date"], "单位净值": str(row["value"]), "币种": currency}
                    for row in history.get("points") or []]
    point = forecast.get("point")
    points = forecast.get("points")
    forecast_rows = []
    if points is not None:
        for row in points:
            expected = row.get("expected_change")
            forecast_rows.append({
                "目标": str(row.get("label") or f"H={row.get('horizon')}"),
                "条件日期": str(row.get("date") or "未映射"),
                "状态": {"available": "可用", "partial": "只有预期涨跌",
                         "unavailable": "暂不可用", "stale": "已过期"}.get(
                             row.get("status"), "未提供"),
                "预期涨跌": f"{expected:+.2%}" if expected is not None else "未提供",
                "中间80%范围下限": f"{row['lower']:+.2%}" if row.get("lower") is not None else "未提供",
                "中间80%范围上限": f"{row['upper']:+.2%}" if row.get("upper") is not None else "未提供",
                "说明": str(row.get("reason") or ""),
            })
    elif point:
        forecast_rows = [{
            "估计区间起点（条件日期）": forecast.get("application_date") or "未提供",
            "计划期末（条件日期）": point["date"],
            "预期涨跌": f"{point['expected_change']:+.2%}",
            "中间80%范围下限": f"{point['lower']:+.2%}" if point.get("lower") is not None else "未提供",
            "中间80%范围上限": f"{point['upper']:+.2%}" if point.get("upper") is not None else "未提供",
        }]
    return {"history": history_rows, "forecast": forecast_rows}

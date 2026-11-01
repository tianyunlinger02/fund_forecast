"""Dated investment reasoning and explicit what-if calculations, not forecasts."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal

import streamlit as st

from rmbfund.domain import ValidationError
from rmbfund.investment_profile import (
    build_investment_profile, disclosure_shock_scenario, earnings_valuation_fx_scenario,
)
from rmbfund.ui.forecast import model_source_text, report_model_id
from rmbfund.ui.formatting import mapping, markdown_text, percent, when
from rmbfund.ui.issuer_fundamentals import (
    render_issuer_fundamentals, render_saved_issuer_fundamentals,
)


def _write(value: str) -> None:
    st.markdown(markdown_text(value))


def _profile_summary(profile: Mapping) -> None:
    mandate, portfolio = mapping(profile.get("mandate")), mapping(profile.get("portfolio"))
    if mandate.get("explanation"):
        _write(mandate["explanation"])
    if not portfolio:
        st.caption("还没有可计算组合影响的披露。经理任期或基金名称不能代替仓位。")
        return
    payload = mapping(portfolio.get("payload"))
    denominator = portfolio.get("denominator_label") or {
        "nav": "基金净资产", "equity_assets": "股票资产", "total_assets": "总资产",
    }.get(payload.get("denominator"), "原披露分母")
    equity = mapping(payload.get("assets")).get("equity")
    if equity is not None:
        _write(f"{portfolio['report_date']}披露：股票占{denominator}的{percent(equity, digits=2)}。")
    sectors = payload.get("sectors") or []
    if sectors:
        main = sorted(sectors, key=lambda row: Decimal(row["weight"]), reverse=True)[:3]
        _write("主要行业：" + "；".join(
            f"{row['name']} {percent(row['weight'], digits=2)}" for row in main
        ) + f"（占{denominator}）。这些行业的盈利和估值变化，是需要重点研究的驱动。")
    st.caption(f"这是报告时点的比例，不是今天的仓位；部分持仓没有补成100%。资料取得于"
               f"{when(portfolio.get('observed_at'))}。未覆盖的衍生品和调仓影响没有算成零。")


def _result(scenario: Mapping) -> None:
    st.markdown("**按这些条件计算的影响**")
    conditions = mapping(scenario.get("conditions"))
    if conditions:
        _write(f"每股盈利变化 {percent(conditions['earnings_per_share_change'], signed=True)}，"
               f"估值倍数变化 {percent(conditions['valuation_multiple_change'], signed=True)}，"
               f"相乘后的股价条件为 {percent(conditions['local_price_change'], signed=True)}。")
        st.caption("例如盈利上涨并不保证股价上涨：市场给的估值倍数也可能下降。这里要求正的可比每股盈利、"
                   "股数不变和一致的估值口径；输入是条件，不是已取得的盈利预测。")
    _write(str(scenario["explanation"]))
    if scenario.get("nav_return_contribution") is not None:
        _write(f"覆盖部分占净资产 {percent(scenario['covered_weight'], digits=2)}；"
               f"对基金净值的条件贡献 {percent(scenario['nav_return_contribution'], signed=True, digits=2)}。")
    else:
        st.caption("原披露的分母不是基金净资产，因此没有把计算值换成基金净值涨跌。")
    st.caption("只计算选中部分，不是整只基金的预期涨跌；不与其他重叠持仓情景相加，也不赋发生概率。"
               "没有再把汇率或费用加到这项价格冲击上。")


def build_investment_analysis(report: Mapping) -> dict | None:
    """Build the dated analysis without rendering controls, fitting, or fetching.

    The decision page still saves the same profile and issuer calculations even
    when the optional condition calculator is not part of its presentation.
    """
    issuer = deepcopy(dict(mapping(report.get("issuer_analysis"))))
    issuer_fundamentals = ({"analysis": issuer, "scenario": None, "condition_inputs": None}
                           if issuer.get("issuers") else None)
    analysis = {
        "schema_version": 3, "kind": "dated_conditional_investment_analysis",
        "profile": {}, "scenario": None, "condition_inputs": None,
        "issuer_fundamentals": issuer_fundamentals,
        "forecast_model_id": report_model_id(report),
        "forecast_record_id": mapping(report.get("prediction")).get("record_id"),
        "source_gate": model_source_text(report),
        "prediction_relationship": "separate_from_market_forecast; no probability or action override",
    }
    profile = mapping(report.get("investment_profile"))
    if not profile:
        records = mapping(report.get("evidence")).get("records") or []
        if records:
            try:
                profile = build_investment_profile(records, as_of=report["evaluated_at"])
            except (ValidationError, KeyError, TypeError, ValueError):
                profile = {}
    if profile.get("status") == "available":
        analysis["profile"] = deepcopy(dict(profile))
    return analysis if analysis["profile"] or issuer_fundamentals else None


def render_investment_analysis(report: Mapping) -> dict | None:
    """Optional research-tool controls, separate from the current advice page."""
    analysis = build_investment_analysis(report)
    st.info(model_source_text(report))
    issuer_fundamentals = render_issuer_fundamentals(mapping(report.get("issuer_analysis")))
    if analysis is None:
        return None
    analysis["issuer_fundamentals"] = issuer_fundamentals
    profile = mapping(analysis.get("profile"))
    if not profile:
        return analysis
    st.markdown("**从投资逻辑看这只基金**")
    _profile_summary(profile)
    st.caption("下方是按明确条件作的账面影响计算，不是另一份盈利或估值预测；"
               "没有条件发生概率，不产生经理加分，不修改市场预测；基金级研究倾向与短期预测分开。")
    portfolio = mapping(profile.get("portfolio"))
    if not portfolio:
        return analysis
    payload = mapping(portfolio.get("payload"))
    choices = {}
    if "equity" in mapping(payload.get("assets")):
        choices[("equity", "equity")] = "披露的全部股票部分"
    for row in payload.get("sectors") or []:
        choices[("sectors", row["name"])] = f"行业：{row['name']}"
    if not choices:
        return analysis
    prefix = f"investment_{profile['product_id']}"
    with st.expander("如果盈利、估值或行业价格变了，会怎样？"):
        st.caption("填写你想检验的条件，不是让你替模型预测。留空就不计算，不把未知当0。"
                   "用报告权重说明影响，不会改变上方概率或自动保存。")
        selected = st.selectbox("计算哪一部分", list(choices), format_func=choices.get,
                                key=f"{prefix}_scope")
        method = st.selectbox("条件怎么变化", ["价格变化", "盈利与估值变化"],
                              key=f"{prefix}_method")
        scenario = None
        scope, member = selected
        analysis["condition_inputs"] = {
            "scope": scope, "member": member, "method": method, "unit": "percent",
            "price_change": None, "earnings_change": None, "valuation_change": None,
            "profile_as_of": profile.get("as_of"), "report_date": portfolio.get("report_date"),
        }
        try:
            if method == "价格变化":
                change = st.number_input("所选部分价格变化（%，可留空）", value=None,
                    min_value=-100.0, step=1.0, key=f"{prefix}_price", placeholder="例如 -10 表示下跌10%")
                analysis["condition_inputs"]["price_change"] = str(change) if change is not None else None
                if change is not None:
                    scenario = disclosure_shock_scenario(profile, scope=scope,
                        shocks={member: str(Decimal(str(change)) / 100)},
                        label=f"你填写的{choices[selected]}价格条件")
            else:
                columns = st.columns(2)
                earnings = columns[0].number_input("每股盈利变化（%，可留空）", value=None,
                    min_value=-100.0, step=1.0, key=f"{prefix}_earnings")
                valuation = columns[1].number_input("估值倍数变化（%，可留空）", value=None,
                    min_value=-100.0, step=1.0, key=f"{prefix}_valuation")
                analysis["condition_inputs"]["earnings_change"] = str(earnings) if earnings is not None else None
                analysis["condition_inputs"]["valuation_change"] = str(valuation) if valuation is not None else None
                if earnings is not None and valuation is not None:
                    scenario = earnings_valuation_fx_scenario(profile, scope=scope, members=[member],
                        earnings_change=str(Decimal(str(earnings)) / 100),
                        valuation_change=str(Decimal(str(valuation)) / 100),
                        label=f"你填写的{choices[selected]}盈利与估值条件")
            if scenario:
                _result(scenario)
                analysis["scenario"] = scenario
        except ValidationError as exc:
            st.warning(markdown_text(str(exc)))
        if not payload.get("currencies"):
            st.caption("这份披露没有核实各币种未对冲权重，暂不计算基金汇率贡献；海外持仓不等于100%美元敞口。")
    with st.expander("为什么要看这些因素"):
        for item in profile.get("factor_mapping", []):
            st.markdown(f"**{markdown_text(item['label'])}**")
            _write(item["reason"])
            st.caption(markdown_text(item["not_established"]))
    return analysis


def render_saved_investment_analysis(analysis: Mapping) -> None:
    """Render the stored object only; no new data, inference, or editable controls."""
    if not analysis:
        return
    if issuer := mapping(analysis.get("issuer_fundamentals")):
        render_saved_issuer_fundamentals(issuer)
    st.markdown("**当时的投资条件分析**")
    if analysis.get("source_gate"):
        _write(str(analysis["source_gate"]))
    profile = mapping(analysis.get("profile"))
    _profile_summary(profile)
    if scenario := mapping(analysis.get("scenario")):
        _write(str(scenario.get("label", "当时填写的条件")))
        _result(scenario)
    else:
        inputs = mapping(analysis.get("condition_inputs"))
        supplied = [(label, inputs.get(key)) for key, label in (
            ("price_change", "价格变化"), ("earnings_change", "每股盈利变化"), ("valuation_change", "估值倍数变化"),
        ) if inputs.get(key) is not None]
        if supplied:
            _write("当时填写的条件：" + "；".join(f"{label} {value}%" for label, value in supplied)
                   + "；其余留空，条件不完整，未计算情景。")
        else:
            st.caption("当时没有填写条件情景，不用今天的假设补写；空白没有按0处理。")
    st.caption(f"分析所用资料截止 {when(profile.get('as_of'))}；与原市场预测的发布时间分开保存。"
               "这项条件分析没有改变当时的市场预测概率。")

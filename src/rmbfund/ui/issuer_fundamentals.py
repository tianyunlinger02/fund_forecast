"""Company facts, reverse valuation and exact replay of displayed conditions."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal

import streamlit as st

from rmbfund.domain import ValidationError
from rmbfund.issuer_fundamentals import issuer_valuation_scenario
from rmbfund.ui.formatting import mapping, markdown_text, percent, when

_NAMES = {
    "eps": "每股盈利", "revenue": "营业收入", "gross_margin": "毛利率",
    "capex": "资本开支", "operating_cash_flow": "经营现金流", "free_cash_flow": "自由现金流",
}
_BASES = {"GAAP": "GAAP", "IFRS": "IFRS", "non_GAAP": "非GAAP", "unknown": "未核对"}
_REASONS = {
    "missing_latest_matching_close": "缺少已核对口径的对应证券收盘价。",
    "missing_security_mapping": "对应股类、交易市场或报价币种尚未核对。",
    "price_adjustment_basis_unknown": "这份历史价格的调整方式尚不明确，暂不据此计算估值。",
    "missing_four_distinct_contiguous_comparable_quarters": "还缺四个连续、口径一致的季度摊薄每股盈利。",
    "price_mapping_currency_mismatch": "证券报价币种与映射不一致，暂不混算。",
    "eps_price_currency_mismatch": "盈利与股价不是同一币种，暂不混算。",
    "eps_price_share_basis_mismatch": "盈利与股价不是同一股类，普通股与ADS不能直接混算。",
    "nonpositive_ttm_eps_no_pe": "四季合计盈利不为正，不把负市盈率解读成便宜。",
    "eps_price_split_basis_mismatch": "盈利与价格的每股或拆股口径不一致，暂不混算。",
    "eps_mapping_currency_or_share_basis_mismatch": "盈利与证券映射的币种或普通股／ADS口径不一致。",
    "price_mapping_security_symbol_exchange_or_currency_mismatch": "价格与证券代码、市场或币种映射不一致。",
    "ambiguous_comparable_eps_basis": "有多组不能互换的盈利口径，暂不替你选其中一组。",
}


def _number(value, digits=2) -> str:
    return "未知" if value is None else f"{Decimal(str(value)):,.{digits}f}"


def _metric_value(metric: Mapping) -> str:
    unit, scale = metric.get("unit"), Decimal(str(metric.get("scale", "1")))

    def value(number):
        if unit == "fraction":
            return percent(number, digits=2)
        amount = Decimal(str(number)) * scale
        if unit == "currency_per_share":
            share = "ADS" if metric.get("share_basis") == "ADS" else "普通股"
            return f"{_number(amount, 4)} {metric.get('currency')}/{share}"
        if abs(amount) >= 100000000:
            return f"{_number(amount / 100000000)} 亿{metric.get('currency')}"
        return f"{_number(amount, 0)} {metric.get('currency')}"

    if metric.get("category") == "guidance":
        return f"{value(metric['low'])} 至 {value(metric['high'])}"
    return value(metric["value"])


def _period(metric: Mapping) -> str:
    period = mapping(metric.get("period"))
    quarter = period.get("fiscal_quarter")
    fiscal = f"财年{period['fiscal_year']} Q{quarter} · " if quarter else ""
    basis = {"ytd": "年初累计 · ", "year": "全年 · ", "month": "单月 · "}.get(period.get("basis"), "")
    return fiscal + basis + f"{period.get('start') or '起点未注明'}—{period.get('end') or '期末未注明'}"


def _latest(observations: list) -> list:
    """Keep the latest displayed period for each distinct accounting/share basis."""
    selected = {}
    for observation in observations:
        metric = mapping(observation.get("metric"))
        key = tuple(metric.get(name) for name in (
            "name", "category", "accounting_basis", "security_id", "share_basis", "currency",
            "eps_basis", "split_basis", "accounting_basis_detail",
        )) + (mapping(metric.get("period")).get("basis"),)
        period = mapping(metric.get("period"))
        order = (period.get("fiscal_year", 0), period.get("fiscal_quarter") or 0,
                 period.get("end") or "", observation.get("published_at") or "",
                 observation.get("available_at") or "", observation.get("record_id") or "")
        if key not in selected or order > selected[key][0]:
            selected[key] = order, observation
    return [deepcopy(value[1]) for value in selected.values()]


def _snapshot(analysis: Mapping) -> dict:
    result = deepcopy(dict(analysis))
    for issuer in result.get("issuers", []):
        issuer["cash_flows"] = _latest([row for row in issuer.get("financial_observations") or []
            if mapping(row.get("metric")).get("name") in {"operating_cash_flow", "free_cash_flow"}
            and mapping(row.get("metric")).get("category") == "actual"
            and mapping(row.get("metric")).get("period_role") == "primary"])
        issuer.pop("financial_observations", None)
        for key in ("quarter_actuals", "latest_guidance", "capex"):
            issuer[key] = _latest(issuer.get(key) or [])
        yoy = issuer.get("eps_yoy") or []
        latest_period = max((mapping(row.get("period")).get("end", "") for row in yoy), default="")
        issuer["eps_yoy"] = [row for row in yoy if mapping(row.get("period")).get("end") == latest_period]
        comparisons = issuer.get("revenue_vs_prior_guidance") or []
        completed = [row for row in comparisons if row.get("status") == "available"]
        issuer["revenue_vs_prior_guidance"] = sorted(completed or comparisons,
            key=lambda row: mapping(row.get("period")).get("end", ""))[-1:]
        issuer.pop("revision_selection", None)
    result["issuers"] = [row for row in result.get("issuers", [])
                         if row.get("quarter_actuals") or row.get("latest_guidance") or row.get("capex")]
    result["snapshot_basis"] = "displayed_latest_facts_and_valuation; full_history_remains_in_public_evidence"
    return result


def _facts(observations: list) -> None:
    if not observations:
        return
    rows = []
    for observation in observations:
        metric = mapping(observation.get("metric"))
        basis = _BASES.get(metric.get("accounting_basis"), metric.get("accounting_basis", "未知"))
        if metric.get("share_basis"):
            basis += " · " + ("每ADS" if metric["share_basis"] == "ADS" else "普通股")
        if metric.get("eps_basis"):
            basis += " · " + ("摊薄" if metric["eps_basis"] == "diluted" else "基本")
        rows.append({"指标": _NAMES.get(metric.get("name"), metric.get("name")),
                     "财务期间": _period(metric), "数值": _metric_value(metric), "口径": basis,
                     "公开日期": str(observation.get("published_at") or "未核实")})
    st.dataframe(rows, hide_index=True, width="stretch")


def _company(issuer: Mapping) -> None:
    exposure = mapping(issuer.get("disclosed_exposure"))
    name = markdown_text(str(issuer.get("issuer_name", issuer.get("issuer_id", "公司"))))
    st.markdown(f"**{name}**")
    if exposure.get("weight") is not None:
        st.caption(f"{exposure.get('report_date', '未知日期')}披露占比 {percent(exposure['weight'], digits=2)}；"
                   "仅是该报告时点的权重，不是今天仓位。")
    displayed_facts = [*(issuer.get("quarter_actuals") or []), *(issuer.get("capex") or []),
                       *(issuer.get("cash_flows") or [])]
    if any("preliminary" in mapping(row.get("metric")).get("accounting_basis_detail", "")
           for row in displayed_facts):
        st.caption("这份初步合并业绩尚未完成外部复核；公开日期未核实的部分只按本机实际取得时间使用。")
    _facts(issuer.get("quarter_actuals") or [])
    for comparison in issuer.get("eps_yoy") or []:
        if comparison.get("status") == "available":
            current = mapping(mapping(comparison.get("current")).get("metric"))
            share = "ADS" if current.get("share_basis") == "ADS" else "普通股"
            st.write(f"{_period(current)}{current.get('currency')}每{share}盈利较上年同财季变化 "
                     f"{percent(comparison.get('change'), signed=True)}。")
            st.caption("这是已报告盈利的同比变化，不是市场预期差；原季度与证券口径一致才比较。")
    for comparison in issuer.get("revenue_vs_prior_guidance") or []:
        if comparison.get("status") == "available":
            st.write(f"实际收入相对原管理层指引中点：{percent(comparison['midpoint_error_ratio'], signed=True)}。")
            st.caption("比较的是同一财务期间的收入与此前公司指引，不是分析师一致预期，也不是EPS超预期。")
            if comparison.get("comparison_caveat"):
                st.caption("原指引未单独注明会计准则；这里只按公司明确相同的合并收入定义比较，不把指引改称IFRS财报。")
    guidance = issuer.get("latest_guidance") or []
    if guidance:
        st.markdown("**公司自己公布的指引**")
        _facts(guidance)
    capex = issuer.get("capex") or []
    cash_flows = issuer.get("cash_flows") or []
    if capex or cash_flows:
        st.markdown("**资本开支与现金投入**")
        _facts([*capex, *cash_flows])
        definitions = " ".join(mapping(row.get("metric")).get("accounting_basis_detail", "")
                               for row in capex).lower()
        if "intangible" in definitions:
            st.caption("这里的资本开支是购置固定资产及无形资产的现金支出，不是客户的投资额。")
        elif "cash-flow summary purchase of pp&e" in definitions:
            st.caption("这里的资本开支是购置固定资产的现金支出，不包括研发费用。")
        elif "tsmc capital expenditure table" in definitions:
            st.caption("这里采用公司资本开支表的定义，不直接与其他公司同名指标互换。")
        st.caption("增加投资可能支持增长，也可能消耗现金；没有自动转换成利好分数。")
    if not issuer.get("securities"):
        st.caption("已接入经营资料，但还没有口径完整的四季摊薄盈利及匹配价格，暂不计算这家公司的估值。")
    for security in issuer.get("securities") or []:
        close = mapping(security.get("latest_close"))
        eps = mapping(security.get("ttm_diluted_eps"))
        st.markdown(f"**价格与盈利是否匹配 · {markdown_text(str(security.get('symbol', '证券')))}**")
        if close:
            description = mapping(close.get("price_basis")).get("description", "")
            label = "收盘后最近报价" if description.startswith("Direct last-sale") else "收盘价"
            st.write(f"{close.get('date')}{label} {_number(close.get('close'))} {security.get('currency')}。")
            if "raw currency=null" in description:
                st.caption("报价按美国上市证券的美元报价惯例解释；来源币种字段为空，不是逐字段核实的USD声明，亦非实时价。")
        if eps.get("status") == "available":
            quarters = eps.get("quarters") or []
            end = mapping(mapping(quarters[-1].get("metric")).get("period")).get("end") if quarters else "未知"
            st.write(f"已核对四季合计摊薄每股盈利（截至{end}）：{_number(eps.get('value'), 4)} {eps.get('currency')}。")
            if any("issuer-translated" in mapping(row.get("metric")).get("accounting_basis_detail", "")
                   for row in quarters):
                st.caption("美元每ADS盈利是公司各季度披露的折算值之和，不是统一汇率重算；没有将台币普通股盈利直接除以美股价格。")
        if security.get("valuation_status") == "available":
            st.write(f"市盈率约 {_number(security.get('pe_ttm'))} 倍；"
                     f"盈利收益率 {percent(security.get('earnings_yield_ttm'), digits=2)}。")
            st.caption("盈利收益率是已报告盈利相对股价的比例，不是预计投资收益率；未使用未来盈利冒充历史值。")
        else:
            reason = security.get("valuation_reason") or eps.get("reason")
            st.caption(_REASONS.get(reason, "缺少同证券、同币种的可比盈利或价格，暂不计算估值。"))
    observations = [*(issuer.get("quarter_actuals") or []), *(issuer.get("latest_guidance") or []),
                    *(issuer.get("capex") or []), *(issuer.get("cash_flows") or [])]
    for security in issuer.get("securities") or []:
        observations.extend(mapping(security.get("ttm_diluted_eps")).get("quarters") or [])
        if security.get("latest_close"):
            observations.append(security["latest_close"])
    sources = {}
    for observation in observations:
        url = observation.get("source_url")
        if url:
            sources[url] = {"来源": observation.get("source", "公开原件"), "地址": url,
                            "公开时间": observation.get("published_at") or "精度未核实",
                            "本机可用时间": when(observation.get("available_at"))}
    if sources:
        with st.expander("查看这一家公司所用来源与日期"):
            st.dataframe(list(sources.values()), hide_index=True, width="stretch")


def _scenario_result(scenario: Mapping) -> None:
    currency = scenario.get("currency", "")
    st.write(f"如果终值市盈率为 {_number(scenario.get('terminal_multiple'))} 倍，"
             f"支撑当前股价需要每股盈利 {_number(scenario.get('required_eps'), 4)} {currency}，"
             f"相对当前四季盈利需要变化 {percent(scenario.get('required_growth'), signed=True)}。")
    if scenario.get("conditional_price") is not None:
        st.write(f"若每股盈利变化为 {percent(scenario.get('earnings_change'), signed=True)}，"
                 f"按所填盈利条件计算，股价为 {_number(scenario['conditional_price'])} {currency}，"
                 f"相对基准价格变化 {percent(scenario['conditional_price_change'], signed=True)}。")
    st.caption("这是所填倍数和盈利条件下的账，不是目标价预测，没有发生概率；"
               "未把外币股价变化冒充人民币基金净值变化，也未与行业或期货情景相加。")


def render_issuer_fundamentals(analysis: Mapping | None) -> dict | None:
    if not analysis or not analysis.get("issuers"):
        return None
    shown = _snapshot(analysis)
    if not shown.get("issuers"):
        return None
    st.markdown("**公司赚钱能力与当前价格**")
    coverage = sum((Decimal(str(mapping(row.get("disclosed_exposure")).get("weight", "0")))
                    for row in shown["issuers"]), Decimal(0))
    denominator = mapping(shown.get("coverage")).get("denominator")
    if denominator == "nav":
        valued = mapping(shown.get("coverage")).get("analyzed_disclosed_weight")
        valuation_text = f"其中盈利与价格可匹配部分为{percent(valued, digits=2)}。" if valued is not None else ""
        st.caption(f"已接入公司财报对应报告时点净资产的{percent(coverage, digits=2)}，未归一为100%；"
                   + valuation_text + "公司经营分析与上方短期概率分开计算。")
    else:
        st.caption("这里使用公司正式披露，不以净值上涨代替盈利改善。与上方短期概率分开计算。")
    for issuer in shown["issuers"]:
        with st.container(border=True):
            _company(issuer)
    saved = {"analysis": shown, "scenario": None, "condition_inputs": None}
    eligible = {row["issuer_id"]: row for row in shown["issuers"] if any(
        security.get("valuation_status") == "available" and security.get("pe_ttm") is not None
        for security in row.get("securities") or [])}
    if not eligible:
        return saved
    prefix = f"issuer_valuation_{shown.get('product_id')}"
    with st.expander("现在的价格，需要多高盈利才能支撑？"):
        selected = st.selectbox("选择公司", list(eligible),
            format_func=lambda value: eligible[value]["issuer_name"], key=f"{prefix}_issuer")
        columns = st.columns(2)
        multiple = columns[0].number_input("终值市盈率条件（倍，可留空）", value=None,
            min_value=0.01, step=1.0, key=f"{prefix}_multiple", placeholder="你想检验的倍数，不预填合理值")
        growth = columns[1].number_input("每股盈利变化条件（%，可留空）", value=None,
            min_value=-100.0, step=1.0, key=f"{prefix}_growth", placeholder="例如10表示增加10%")
        saved["condition_inputs"] = {"issuer_id": selected,
            "terminal_multiple": str(multiple) if multiple is not None else None,
            "earnings_change_percent": str(growth) if growth is not None else None}
        if multiple is not None:
            try:
                scenario = issuer_valuation_scenario(shown, issuer_id=selected,
                    terminal_multiple=str(multiple),
                    earnings_change=str(Decimal(str(growth)) / 100) if growth is not None else None,
                    label="你填写的公司盈利与终值估值条件")
                _scenario_result(scenario)
                saved["scenario"] = scenario
            except ValidationError as exc:
                st.warning(markdown_text(str(exc)))
        else:
            st.caption("不填写倍数就不计算；不会替你编出合理市盈率或未来股价。")
    return saved


def render_saved_issuer_fundamentals(saved: Mapping) -> None:
    analysis = mapping(saved.get("analysis"))
    if not analysis.get("issuers"):
        return
    st.markdown("**当时的公司盈利与估值分析**")
    for issuer in analysis["issuers"]:
        with st.container(border=True):
            _company(issuer)
    if scenario := mapping(saved.get("scenario")):
        name = next((row["issuer_name"] for row in analysis["issuers"]
                     if row["issuer_id"] == scenario.get("issuer_id")), "所选公司")
        st.markdown(f"**{markdown_text(str(name))} · 当时填写的估值条件**")
        _scenario_result(scenario)
    else:
        st.caption("当时没有完整估值条件；没有用今天的财报、价格或假设补算。")

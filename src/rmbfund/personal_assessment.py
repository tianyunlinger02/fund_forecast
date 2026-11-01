"""Pure, local display-and-save snapshots; no fitting, requests, clocks, or writes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
from math import expm1, isfinite
import re
from typing import Any

from .domain import SUPPORTED_KINDS, ValidationError, decimal
from .forecast_decision import _fee, build_decision
from .forecast_models import model_name as _catalog_model_name
from .portfolio import _holding_text, _text, _valid_nav, value_holding

SCHEMA_VERSION = 4
POLICY_VERSION = "personal_assessment_v4_fund_buyer"
# Labels/units for old NAV records only, not an allowlist of explainable inputs.
_LEGACY_NAV_LABELS = {
    "log_return_1": "最近1笔净值涨跌",
    "log_return_5": "最近5笔净值累计涨跌",
    "log_return_20": "近20笔净值累计涨跌",
    "log_return_60": "近60笔净值累计涨跌",
    "volatility_20": "近20笔净值的单笔波动",
    "volatility_60": "近60笔净值的单笔波动",
    "drawdown_60": "当前净值相对近60笔高点的涨跌",
}


def _subject(value: Mapping) -> dict:
    allowed = {"id", "symbol", "name", "kind", "share_class", "exchange", "is_demo", "category"}
    if not isinstance(value, Mapping) or value.keys() - allowed:
        raise ValidationError("subject只接受公开产品身份字段，个人资料请单独传入holding")
    symbol = _text(value.get("symbol"), "基金代码", 30)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", symbol):
        raise ValidationError("基金代码格式无效")
    kind = _text(value.get("kind", "open_fund"), "产品类型", 30)
    if kind not in SUPPORTED_KINDS:
        raise ValidationError("不支持的产品类型")
    demo = value.get("is_demo", False)
    if not isinstance(demo, bool):
        raise ValidationError("is_demo必须是布尔值")
    public_id, name = value.get("id"), value.get("name")
    share_class, exchange = value.get("share_class"), value.get("exchange")
    return {
        "id": _text(public_id, "公开产品标识", 80) if public_id is not None else None,
        "symbol": symbol,
        "name": _text("" if name is None else name, "基金名称", 200, blank=True) or symbol,
        "kind": kind,
        "share_class": _text("" if share_class is None else share_class, "份额类别", 80, blank=True),
        "exchange": _text("" if exchange is None else exchange, "交易场所", 80, blank=True),
        "is_demo": demo,
    }


def _currency(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value):
        raise ValidationError("币种须明确填写三位大写代码，例如CNY或USD；未知时留空")
    return value


def _fee_text(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    _fee(value)
    return str(value).strip()


def model_name(model_id: str | None) -> str:
    """Name the actual saved method; unknown ids must not turn into NAV models."""
    return _catalog_model_name(model_id) or (str(model_id) if model_id else "未注明预测方法")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) else None


def _feature_metadata(row: Mapping, definitions: Mapping | None = None) -> dict:
    name = str(row.get("feature") or row.get("name") or "未注明因素")
    metadata = dict((definitions or {}).get(name, {}))
    metadata.update(row)
    metadata.setdefault("label", _LEGACY_NAV_LABELS.get(name, name))
    if name in _LEGACY_NAV_LABELS:
        metadata.setdefault("economic_role", "nav_history")
        metadata.setdefault("unit", "log_return" if name.startswith("log_return_") else
                            "daily_log_return_std" if name.startswith("volatility_") else "return_ratio")
    return metadata


def format_factor_value(feature: Mapping | str, value: Any = None) -> str:
    """Render a saved factor in its declared unit, never as portfolio profit.

    Public for current/history views. A percent level of 4 means 4%, while a
    percentage-point change of .1 means +.10pp. Relative log differences remain
    relative factor measurements, not a fund's simple return.
    """
    metadata = _feature_metadata(feature if isinstance(feature, Mapping) else {"feature": feature})
    parsed = _number(metadata.get("value") if value is None else value)
    if parsed is None:
        return "未提供"
    unit = metadata.get("unit")
    if unit in {"log_return", "log_change"}:
        try:
            return f"{expm1(parsed):+.2%}"
        except OverflowError:
            return "数值超出可显示范围"
    if unit == "log_return_difference":
        return f"{parsed:+.4f}（相对强弱的对数差，正数更强；不是基金收益）"
    if unit == "percent":
        return f"{parsed:.2f}%"
    if unit == "percentage_points":
        return f"{parsed:+.2f}pp（百分点）"
    if unit == "daily_log_return_std":
        return f"{parsed:.2%}（单笔波动，非年化）"
    if unit in {"return_ratio", "ratio"}:
        return f"{parsed:+.2%}"
    return f"{parsed:+.4g}" + (f"（{unit}）" if unit else "（原值，未注明单位）")


def factor_observation_dates(driver: Mapping) -> str:
    """Retain each actual source's dates for optional full-factor tables."""
    ends, starts = driver.get("observation_ends"), driver.get("observation_starts")
    if not isinstance(ends, Mapping):
        return "未保存因素日期"
    starts = starts if isinstance(starts, Mapping) else {}
    return "；".join(f"{key}：{starts[key]} ～ {day}" if key in starts and starts[key] != day
                    else f"{key}：{day}" for key, day in ends.items())


def _fund_connection(role: str | None, profile: Mapping) -> str:
    if role == "nav_history":
        return "来自本基金自己的净值历史，不是其他产品的走势。"
    portfolio, mandate = profile.get("portfolio"), profile.get("mandate")
    portfolio = portfolio if isinstance(portfolio, Mapping) else {}
    mandate = mandate if isinstance(mandate, Mapping) else {}
    payload, contract = portfolio.get("payload", {}), mandate.get("payload", {})
    denominator = {"nav": "净资产", "equity_assets": "股票资产", "total_assets": "总资产"}.get(
        payload.get("denominator"), "原披露分母")
    equity = _number(payload.get("assets", {}).get("equity"))
    date = portfolio.get("report_date") or "原报告"
    if role == "market" and equity is not None:
        benchmark = contract.get("benchmark", {})
        region = benchmark.get("name") or benchmark.get("region")
        region = "新兴市场" if region == "emerging_markets" else region
        return (f"{date}披露股票占{denominator}{equity:.2%}"
                + (f"，合同基准为{region}" if region else "")
                + "；市场序列是参照，不是基金的实际组合。")
    if role == "industry":
        sectors = [row for row in payload.get("sectors", []) if isinstance(row, Mapping)
                   and any(term in str(row.get("name", "")).lower() for term in (
                       "信息技术", "半导体", "information technology", "semiconductor"))
                   and _number(row.get("weight")) is not None]
        if sectors:
            shown = "；".join(f"{row['name']}占{denominator}{float(row['weight']):.2%}" for row in sectors[:2])
            return f"{date}披露{shown}；行业序列只是代理，不是逐股持仓复现。"
    if role == "fx" and contract.get("base_currency"):
        return (f"合同载明基金以{contract['base_currency']}计价；中间价是独立统计参照，"
                "不是按实际未对冲仓位计算的汇率收益，也没有重复叠加到人民币净值。")
    if role == "discount_rate" and equity is not None:
        return (f"{date}披露股票占{denominator}{equity:.2%}；收益率作为估值环境的统计参照，"
                "不等于基金的债券收益，也没有套用久期公式。")
    return "未保存足以核对这项因素与本基金投向关系的映射，不据名称猜测实际敞口。"


def explain_forecast_head(prediction: Mapping, head: Mapping | None, *, horizon: int,
                          entry_lag: int | None, role: str, all_factors: bool = False) -> dict:
    """Explain only this saved head, with strongest supporting AND opposing inputs.

    No fitting, data reads or current profile substitutions. Head-local drivers
    take precedence even when empty. Metadata enriches explanations, never the
    raw drivers, values, coefficients or market probabilities being saved.
    """
    identity = {
        "record_id": prediction.get("record_id"), "model_id": prediction.get("model_id"),
        "model_hash": prediction.get("model_hash"),
        "horizon": horizon, "entry_lag": entry_lag, "role": role,
    }
    if (head is None or head.get("horizon") != horizon or head.get("entry_lag") != entry_lag):
        return {"identity": identity, "head": None, "drivers": [], "factor_reasons": [],
                "reasons": ["没有保存这个起点与期限的预测，不借用其他期限或成交情景的理由。"]}
    source_drivers = head["drivers"] if "drivers" in head else prediction.get("drivers", [])
    drivers = [deepcopy(dict(row)) for row in (source_drivers or [])
               if isinstance(row, Mapping) and row.get("horizon") == horizon
               and row.get("entry_lag") == entry_lag and row.get("role", role) == role]
    definitions = {}
    definition = prediction.get("model_definition")
    definition = definition if isinstance(definition, Mapping) else {}
    for row in [*(definition.get("features") or []), *(prediction.get("features") or [])]:
        if isinstance(row, Mapping) and row.get("name"):
            definitions[row["name"]] = {**definitions.get(row["name"], {}), **row}
    profile = prediction.get("profile_mapping")
    profile = profile if isinstance(profile, Mapping) else {}
    factor_reasons, reasons, chosen = [], [], []
    if head.get("method") == "empirical_historical":
        reasons.append("这次参考历史上相同期限、相同起点假设的涨跌分布，没有把近期走势单独拆成理由。")
    else:
        for row in drivers:
            value, effect = _number(row.get("value")), _number(row.get("contribution_log_return"))
            if value is None or effect is None:
                continue
            metadata = _feature_metadata(row, definitions)
            direction = "支持上涨" if effect > 0 else "反对上涨" if effect < 0 else "中性"
            connection = _fund_connection(metadata.get("economic_role"), profile)
            ends = metadata.get("observation_ends")
            dates = sorted({str(day) for day in ends.values()}) if isinstance(ends, Mapping) else []
            dated = f"（因素数据截至{dates[0]}" + (f" ～ {dates[-1]}" if len(dates) > 1 else "") + "）" if dates else ""
            factor_reasons.append({
                "feature": row.get("feature"), "label": metadata["label"], "unit": metadata.get("unit"),
                "economic_role": metadata.get("economic_role"), "value": row["value"],
                "formatted_value": format_factor_value(metadata, row["value"]),
                "contribution_log_return": row["contribution_log_return"], "direction": direction,
                "connection_to_fund": connection, "observation_dates": factor_observation_dates(metadata),
                "reason": f"{direction}的信息：{metadata['label']}为{format_factor_value(metadata, row['value'])}{dated}。"
                          f"这项信息在本次拟合中{'偏向上涨' if effect > 0 else '偏向下跌' if effect < 0 else '没有推高或压低预期'}。{connection}",
            })
        factor_reasons.sort(key=lambda row: abs(float(row["contribution_log_return"])), reverse=True)
        chosen = []
        # Reserve a place for the strongest contrary signal, even if several
        # larger correlated inputs all point the same way.
        for sign in (1, -1):
            candidate = next((row for row in factor_reasons if float(row["contribution_log_return"]) * sign > 0), None)
            if candidate is not None:
                chosen.append(candidate)
        for row in factor_reasons:
            if len(chosen) >= 3:
                break
            if row not in chosen:
                chosen.append(row)
        chosen.sort(key=lambda row: abs(float(row["contribution_log_return"])), reverse=True)
        reasons.extend(row["reason"] for row in chosen)
        if factor_reasons and not any(float(row["contribution_log_return"]) < 0 for row in factor_reasons):
            reasons.append("这份预测保存的单项因素没有反对上涨的方向；这不等于没有下跌风险，仍要看预测范围。")
        if factor_reasons and not any(float(row["contribution_log_return"]) > 0 for row in factor_reasons):
            reasons.append("这份预测保存的单项因素没有支持上涨的方向；这不等于一定下跌，仍要看预测范围。")
        reasons.append(
            "这些是模型从历史中学到的关联，不是涨跌的因果解释；相关因素可能重叠。单项贡献针对模型的对数位置，"
            "不能相加当作基金预期收益或经理能力；单笔波动也不是全年涨跌。"
            if factor_reasons else
            "这次没有保存与所选期限和成交延迟对应的单项理由，不借用其他情景，也不另编市场原因。"
        )
    return {
        "identity": identity,
        "head": deepcopy({key: value for key, value in head.items() if key not in {"samples", "drivers"}}),
        "drivers": drivers, "factor_reasons": factor_reasons if all_factors else chosen, "reasons": reasons,
    }


# Kept for callers of the earlier local-only snapshot helper.
_head_view = explain_forecast_head


def _basis(prediction: Mapping, head: Mapping | None, supplied_currency: str | None) -> dict:
    """A head-local anchor takes precedence; today's valuation is never an input."""
    head = head or {}
    own_anchor = head.get("anchor")
    anchor = own_anchor if isinstance(own_anchor, Mapping) else prediction.get("anchor", {})
    if not isinstance(anchor, Mapping):
        anchor = {}
    anchor_currency, head_currency = anchor.get("currency"), head.get("currency")
    if isinstance(own_anchor, Mapping):
        stored_currency = anchor_currency or head_currency
        source = "head.anchor" if anchor_currency else "head"
    else:
        stored_currency = head_currency or anchor_currency
        source = "head" if head_currency else "prediction.anchor"
    conflict = bool(
        (anchor_currency and head_currency and anchor_currency != head_currency)
        or (stored_currency and supplied_currency and stored_currency != supplied_currency)
    )
    currency = stored_currency or supplied_currency
    if not stored_currency and supplied_currency:
        source = "anchor_currency_argument"
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        currency, source = None, None
    result = {
        "date": anchor.get("date"), "nav": anchor.get("nav"), "currency": currency,
        "currency_source": source, "currency_conflict": conflict,
    }
    evidence = head.get("currency_source") if source == "head" else anchor.get("currency_source")
    if evidence is not None:
        result["currency_evidence"] = deepcopy(evidence)
    return result


def _return(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError("预测净值变化必须是有效数值") from exc
    if isinstance(value, bool) or not result.is_finite() or result < -1:
        raise ValidationError("预测净值变化必须为有限数且不能低于-100%")
    return result


def _point(value: Decimal, basis: Decimal, cost: Decimal | None) -> dict:
    return {"value": str(value), "change": str(value - basis),
            "profit": str(value - cost) if cost is not None else None}


def _empty_amounts(basis: Mapping, reasons: list[str]) -> dict:
    return {
        "status": "ready", "currency": basis.get("currency"),
        "basis_date": basis.get("date"), "basis_nav": basis.get("nav"), "basis_value": None,
        "expected": None, "lower": None, "upper": None, "reasons": reasons,
    }


def _anchor_amounts(head: Mapping | None, basis: Mapping, holding: Mapping) -> dict:
    result = _empty_amounts(basis, [
        "假设份额保持不变，以下金额从这份预测自己的净值起点计算，不是从今日市值起算的可兑现收益。",
        "期末金额是按净值计算的账面值，尚未扣除未来赎回费；净值内的年度运作费用不重复扣。",
    ])
    reason = None
    if head is None:
        result["status"], reason = "missing_forecast", "缺少所选期限的净值起点预测，暂不换算持仓金额。"
    elif holding["shares"] is None:
        result["status"], reason = "missing_shares", "尚未填写确认份额，不能推算持仓金额；未填份额不等于零份额。"
    elif basis["currency_conflict"]:
        result["status"], reason = "currency_mismatch", "传入币种与预测净值来源记载不符，不能混算金额。"
    elif basis["currency"] is None or holding["currency"] is None:
        result["status"], reason = "missing_currency", "预测净值或持仓的币种尚不明确，不默认按人民币换算。"
    elif basis["currency"] != holding["currency"]:
        result["status"], reason = "currency_mismatch", "预测净值与持仓成本的币种不同，没有汇率依据，不换算或混减金额。"
    nav = {"value": basis["nav"], "date": basis["date"], "currency": basis["currency"]}
    if reason is None and _valid_nav(nav) is None:
        result["status"], reason = "missing_nav", "这份预测自己的净值起点不完整，不能用今日净值代替。"
    if reason:
        result["reasons"].insert(0, reason)
        return result
    with localcontext() as context:
        context.prec = 128
        # Reuse the existing CNY valuation convention. Matching foreign amounts
        # remain in their stated currency; no exchange-rate conversion is made.
        if basis["currency"] == "CNY":
            baseline = Decimal(value_holding(holding, nav)["market_value"])
        else:
            baseline = decimal(holding["shares"]) * decimal(basis["nav"])
        result["basis_value"] = str(baseline)
        cost = decimal(holding["cost"]) if holding["cost"] is not None else None
        for field, key in (("expected", "expected_change"), ("lower", "lower"), ("upper", "upper")):
            change = _return(head.get(key))
            if change is not None:
                result[field] = _point(baseline * (1 + change), baseline, cost)
    result["interval_coverage"] = head.get("interval_coverage")
    if result["expected"] is None:
        result["status"] = "missing_return"
        result["reasons"].insert(0, "所选预测未保存预期涨跌，暂不计算预期金额。")
    return result


def _budget(value: Any, holding_currency: str | None) -> dict | None:
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        if value.keys() - {"amount", "currency"}:
            raise ValidationError("拟投入金额仅接受amount和currency")
        amount, currency = value.get("amount"), _currency(value.get("currency"))
    else:
        amount, currency = value, holding_currency
    amount = _holding_text(amount, "拟投入金额")
    return {"amount": amount, "currency": currency} if amount is not None else None


def _buy_amounts(head: Mapping | None, basis: Mapping, budget: Mapping | None,
                 subscription_fee: str | None) -> dict:
    result = _empty_amounts({"currency": budget.get("currency") if budget else None}, [
        "按你明确填写的拟投入金额和申购费，计算新买后的期末账面值；不需要假定今天公布的净值就是成交价。",
        "这里没有扣除期末可能发生的赎回费，也不代表实际到账金额；不会根据现有持仓推测预算。",
    ])
    reason = None
    if budget is None:
        result["status"], reason = "missing_budget", "未填写拟投入金额，不推算新买金额情景。"
    elif head is None:
        result["status"], reason = "missing_forecast", "缺少所选期限与主要成交延迟的预测，不借用净值起点预测计算新买金额。"
    elif subscription_fee is None:
        result["status"], reason = "unknown_subscription_fee", "申购费尚未知，不能把它按零计算；这里只保留费用条件比较。"
    elif basis["currency_conflict"]:
        result["status"], reason = "currency_mismatch", "传入币种与预测净值来源记载不符，不能混算金额。"
    elif budget["currency"] is None or basis["currency"] is None:
        result["status"], reason = "missing_currency", "拟投入金额或预测净值的币种未知，不默认按人民币计算。"
    elif budget["currency"] != basis["currency"]:
        result["status"], reason = "currency_mismatch", "拟投入金额与预测净值的币种不同，没有汇率依据，暂不换算。"
    if reason:
        result["reasons"].insert(0, reason)
        return result
    with localcontext() as context:
        context.prec = 128
        amount, fee = decimal(budget["amount"]), decimal(subscription_fee)
        result["basis_value"] = str(amount)
        for field, key in (("expected", "expected_change"), ("lower", "lower"), ("upper", "upper")):
            change = _return(head.get(key))
            if change is not None:
                result[field] = _point(amount * (1 + change) / (1 + fee), amount, amount)
    result["interval_coverage"] = head.get("interval_coverage")
    return result


def cash_use_suitability(conditions: Mapping | None, *, profile: Mapping | None = None) -> dict:
    """Apply an explicit near-term cash need without inventing a settlement date."""
    conditions = conditions or {}
    need = conditions.get("near_term_required", conditions.get("cash_need", "unknown"))
    if need not in {"yes", "no", "unknown"}:
        need = "unknown"
    profile = profile or {}
    mandate = (profile.get("mandate") or {}).get("payload") or {}
    qdii = "QDII" in str(mandate.get("name", "")).upper()
    planned = conditions.get("near_term_use_time") or conditions.get("plan_horizon") or None
    if need == "yes":
        reason = ("你明确近期必须使用这笔钱，先不把这笔资金新增投入基金；"
                  + ("QDII涉及境内外交易日与确认、赎回安排，" if qdii else "基金净值可能波动，赎回也并非即时到账，")
                  + "目前没有确认到账日期。已有持仓仍按研究理由判断，同时应按实际渠道安排预留用钱时间。")
        status = "not_suitable_for_new_money"
    elif need == "no":
        status, reason = "no_near_term_need", "你填写近期没有必须用钱的安排；这不改变基金自身研究结论，也不等于已确定合适投入额。"
    else:
        status, reason = "unknown", "还没有明确这笔资金的用钱安排；未据此推定长期闲钱或承诺赎回到账日。"
    return {"status": status, "near_term_required": need, "planned_use_time": planned,
            "qdii": qdii, "reason": reason}


def build_assessment(
    prediction: Mapping | None, *, subject: Mapping, holding: Mapping | None = None,
    horizon: int = 20, subscription_fee: str | float | None = None,
    redemption_fee: str | float | None = None, budget: str | Mapping | None = None,
    anchor_currency: str | None = None, current_value: Mapping | None = None,
    evidence_summary: Mapping | None = None, entry_lag: int | None = None,
    investment_analysis: Mapping | None = None,
    research_thesis: Mapping | None = None, fund_comparison: Mapping | None = None,
    portfolio_context: Mapping | None = None, sizing_plan: Mapping | None = None,
    forecast_status: str | None = None, execution_conditions: Mapping | None = None,
    online_context: Mapping | None = None, trade_scenario: Mapping | None = None,
    decision_rationale: Mapping | None = None, include_decision_brief: bool = False,
) -> dict:
    """Build exactly one JSON-ready private object for both display and explicit save.

    ``subject`` contains public Product fields, never a private fund id. Holding
    shares/cost and monetary outputs are decimal text. A scalar budget uses the
    explicitly supplied holding currency; {amount, currency} is also accepted.
    ``anchor_currency`` must come from this forecast's NAV source, not an assumed
    CNY default. ``current_value`` is today's independent ``value_holding`` result.
    No current valuation, personal cost, or fee changes the market distribution.
    Optional dated investment conditions are saved separately, never blended into
    market probabilities or used to regenerate a past prediction.
    """
    subject = _subject(subject)
    forecast_status = forecast_status or ("ready" if prediction else "absent")
    if forecast_status not in {"ready", "absent", "unavailable", "model_mismatch"}:
        raise ValidationError("未知预测可用状态")
    if forecast_status == "ready" and not prediction:
        raise ValidationError("ready状态需要实际预测")
    prediction = prediction if forecast_status == "ready" else {}
    if (subject["id"] is not None and prediction.get("product_id") is not None
            and subject["id"] != prediction["product_id"]):
        raise ValidationError("公开产品身份与所用预测不匹配，不能保存为同一基金的判断")
    if holding is not None and not isinstance(holding, Mapping):
        raise ValidationError("holding应为本机基金记录，未知时留空")
    holding = holding or {}
    if (subject["id"] and holding.get("product_id")
            and subject["id"] != holding["product_id"]):
        raise ValidationError("本机持仓与所选公开产品不匹配，不能混用份额和成本")
    held_inputs = {
        "shares": _holding_text(holding.get("shares"), "份额"),
        "cost": _holding_text(holding.get("cost"), "持仓总成本"),
        "currency": _currency(holding.get("currency")),
    }
    state = "unknown" if held_inputs["shares"] is None else (
        "held" if decimal(held_inputs["shares"]) > 0 else "flat"
    )
    buy_fee, sell_fee = _fee_text(subscription_fee), _fee_text(redemption_fee)
    anchor_currency = _currency(anchor_currency)
    budget = _budget(budget, held_inputs["currency"])
    if current_value is not None and not isinstance(current_value, Mapping):
        raise ValidationError("current_value应为独立的今日估值对象，不能充当预测净值起点")
    if investment_analysis is not None:
        if not isinstance(investment_analysis, Mapping):
            raise ValidationError("investment_analysis应为本次展示的独立条件分析")
        profile = investment_analysis.get("profile", {})
        if profile.get("product_id") not in (None, subject["id"]):
            raise ValidationError("投资条件分析与当前基金身份不匹配")
        issuer = investment_analysis.get("issuer_fundamentals") or {}
        for part in (issuer.get("analysis"), issuer.get("scenario")):
            if part and part.get("product_id") not in (None, subject["id"]):
                raise ValidationError("公司盈利或估值条件与当前基金身份不匹配")
    if trade_scenario is not None:
        if not isinstance(trade_scenario, Mapping) or trade_scenario.get("kind") != "calendar_cash":
            raise ValidationError("按天计划需要实际展示的卖出净额情景")
        scenario_subject = trade_scenario.get("subject") or {}
        if scenario_subject.get("id") not in (None, subject["id"]):
            raise ValidationError("交易情景与当前基金身份不匹配")
        if trade_scenario.get("horizon") != horizon:
            raise ValidationError("交易情景与所选净值期限不匹配")
    if prediction:
        decision = (deepcopy(dict(trade_scenario["decision"])) if trade_scenario is not None else
                    build_decision(prediction, holding=state, horizon=horizon, entry_lag=entry_lag,
                                   subscription_fee=buy_fee, redemption_fee=sell_fee))
        lag = decision["primary_entry_lag"]
        if trade_scenario is not None:
            head = decision.get("primary_head")
            if head is not None and (head.get("horizon") != horizon or head.get("entry_lag") != lag):
                raise ValidationError("交易情景的预测起点或期限不匹配")
            matched = [row for row in prediction.get("execution_forecasts", [])
                       if row.get("horizon") == horizon and row.get("entry_lag") == lag]
            if head is not None and (len(matched) != 1 or dict(head) != dict(matched[0])):
                raise ValidationError("交易情景必须采用这份预测的准确执行头")
        execution_head = decision["primary_head"]
        anchors = [head for head in prediction.get("forecasts", [])
                   if head.get("horizon") == horizon and head.get("entry_lag") == 0]
        if len(anchors) > 1:
            raise ValidationError("同一期限有重复净值起点预测，无法确定金额依据")
        anchor_head = anchors[0] if anchors else None
        execution = _head_view(prediction, execution_head, horizon=horizon, entry_lag=lag, role="execution")
        anchor = _head_view(prediction, anchor_head, horizon=horizon, entry_lag=0, role="anchor")
        anchor["basis"] = _basis(prediction, anchor_head, anchor_currency)
        # Strip only legacy action aliases. The raw head's expected_change still
        # means its unchanged market return, never a fee-adjusted comparison.
        compact_decision = {key: deepcopy(value) for key, value in decision.items()
                            if key not in {"primary_head", "primary"}}
        for row in [*compact_decision["branches"].values(), *compact_decision["scenarios"]]:
            for alias in ("expected_change", "buy_net_change", "hold_advantage"):
                row.pop(alias, None)
    else:
        decision, compact_decision = {}, {}
        lag, execution_head, anchor_head = None, None, None
        execution, anchor = {}, {"basis": _basis({}, None, anchor_currency)}
    forecast = {key: deepcopy(prediction.get(key)) for key in (
        "record_id", "product_id", "model_id", "model_hash", "schema_version",
        "input_hash", "basis_hash", "generated_at", "entry_assumptions", "anchor",
    )}
    forecast["model_name"] = model_name(prediction.get("model_id"))
    # A bounded, exact explanation snapshot. Full time-series rows stay in the
    # immutable public forecast record, identified by record/input hashes. No
    # refetch is needed to replay the private judgment, its mapping or reasons.
    for key in ("request_hash", "issue_id", "model_definition", "features", "target_definition",
                "factor_input_hash", "factor_provenance", "profile_mapping", "profile_mapping_hash",
                "feature_definition_hash", "provenance"):
        if key in prediction:
            forecast[key] = deepcopy(prediction[key])
    if "model_state" in prediction:
        forecast["model_state"] = deepcopy([
            row for row in prediction["model_state"]
            if row.get("horizon") == horizon and row.get("entry_lag") in {0, lag}
        ])
    if isinstance(prediction.get("factor_inputs"), Mapping):
        forecast["factor_input_snapshot"] = {
            "basis": "compact_current_features_and_provenance; full_series_in_original_public_record",
            "record_id": prediction.get("record_id"), "input_hash": prediction.get("input_hash"),
            "factor_input_hash": prediction.get("factor_input_hash"),
            "schema_version": prediction["factor_inputs"].get("schema_version"),
        }
    from .current_online import current_execution_conditions, snapshot_online_context
    from .investor_decision import build_investor_decision
    online = snapshot_online_context(online_context) if online_context is not None else None
    if online is not None:
        execution_conditions = current_execution_conditions(online, subject["id"], execution_conditions)
    conditions = (sizing_plan or {}).get("user_conditions") or {}
    cash_fit = cash_use_suitability(conditions, profile=(investment_analysis or {}).get("profile"))
    investor_decision = build_investor_decision(
        product_id=subject["id"], holding_status=state, research_thesis=research_thesis,
        fund_comparison=fund_comparison, portfolio_context=portfolio_context,
        sizing_plan=sizing_plan, quantitative_decision=compact_decision,
        forecast_status=forecast_status, execution_conditions=execution_conditions, online_context=online,
        cash_use_fit=cash_fit, preserve_research_cases=include_decision_brief,
    )
    result = {
        "schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
        "subject": subject, "forecast": forecast if prediction else {}, "holding_status": state,
        "forecast_status": forecast_status,
        "research_thesis": deepcopy(research_thesis), "fund_comparison": deepcopy(fund_comparison),
        "portfolio_context": deepcopy(portfolio_context), "sizing_plan": deepcopy(sizing_plan),
        "investor_decision": investor_decision, "execution_conditions": deepcopy(execution_conditions),
        "cash_use_fit": cash_fit,
        "inputs": {"holding": held_inputs, "subscription_fee": buy_fee,
                   "redemption_fee": sell_fee, "budget": budget, "anchor_currency": anchor_currency},
        "horizon": horizon, "entry_lag": lag,
        "decision": compact_decision, "execution": execution, "anchor": anchor,
        "amounts": {
            "current": deepcopy(dict(current_value)) if current_value is not None else None,
            "anchor": _anchor_amounts(anchor_head, anchor["basis"], held_inputs),
            "execution_holding": {
                "status": "unknown_entry_nav",
                "reasons": ["实际成交净值尚未知，已有持仓只比较费用后的比例，不把最新公布净值冒充成交价，也不拼接两项预测造出价格路径。"],
            },
            "new_buy": _buy_amounts(execution_head, _basis(prediction, execution_head, anchor_currency),
                                    budget, buy_fee),
        },
        "evidence_summary": deepcopy(evidence_summary),
        "investment_analysis": deepcopy(investment_analysis),
    }
    if trade_scenario is not None:
        result["trade_scenario"] = deepcopy(dict(trade_scenario))
        planned_buy = result["amounts"]["new_buy"]
        planned_buy["end_value_basis"] = "cash_and_redemption_receivable"
        planned_buy["reasons"] = [
            "按所选自然日计划的新份额申购费与未来赎回费换算，净额包含赎回应收款，不是已到账现金。",
            "日期采用本次明确的模拟日程；已有份额与新份额的持有天数分别计算。",
        ] + ([planned_buy["reasons"][0]] if planned_buy["status"] != "ready" else [])
        buy_action = next(row for row in trade_scenario["rows"] if row.get("action") == "buy")
        future_fee = _fee((trade_scenario.get("fees") or {}).get("future_new_redemption_fee"))
        if buy_action.get("status") != "ready":
            planned_buy.update(status="unavailable_trade_plan", expected=None, lower=None, upper=None)
            planned_buy["reasons"].extend(buy_action.get("reasons") or ["当前日期计划不能完成新买后卖出，未计算个人净额。"])
        elif planned_buy["status"] == "ready":
            if future_fee is None:
                planned_buy.update(status="unknown_future_redemption_fee", expected=None, lower=None, upper=None)
                planned_buy["reasons"].append("新份额未来赎回费尚未知，未按零扣除。")
            else:
                with localcontext() as context:
                    context.prec = 128
                    fee, amount = decimal(str(future_fee)), decimal(planned_buy["basis_value"])
                    for key in ("expected", "lower", "upper"):
                        point = planned_buy.get(key)
                        if point is not None:
                            planned_buy[key] = _point(decimal(point["value"]) * (1 - fee), amount, amount)
    if include_decision_brief:
        if trade_scenario is None:
            raise ValidationError("当前首选需要同一份实际展示的自然日情景")
        from .decision_brief import build_decision_brief
        result["decision_brief"] = build_decision_brief(
            product_id=subject["id"], research_thesis=research_thesis,
            investor_decision=investor_decision, trade_scenario=trade_scenario,
            model_id=prediction.get("model_id"), rationale=decision_rationale,
        )
    if online is not None:
        result["online_context"] = online
    return result

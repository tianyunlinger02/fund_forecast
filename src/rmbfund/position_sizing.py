"""Pure conditional cash limits, not forecasts, orders or an optimal allocation.

All returns/fees/weights are decimal ratios, not percentages. No fee, budget,
existing loss, target weight or scenario probability is silently supplied. Amounts
are decimal text; a non-terminating upper bound is rounded downward at 160 digits,
and a required minimum redemption upward. No currency-unit rounding is assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext
from typing import Any

from .domain import ValidationError, decimal

SCHEMA_VERSION = 1
__all__ = ["conditional_size", "post_buy_weight", "planned_buy", "planned_redemption"]
_PRECISION = 160
_FIELDS = {
    "scenario_return", "subscription_fee", "existing_scenario_loss", "loss_budget",
    "spendable_cash", "max_investment", "exposure_cap", "max_fund_weight",
    "requested_gross_cash", "budget_scope", "scenario_scope", "scenario_label",
    "scenario_source", "existing_market_value", "redemption_fee",
    "available_redemption_value", "max_redemption", "known_cohort_market_value",
    "selected_market_value",
}
_NUMBERS = _FIELDS - {"budget_scope", "scenario_scope", "scenario_label", "scenario_source"}
_ASSUMPTIONS = [
    "这是用户明确假设和约束下的容量上限，不是推荐用满，不是唯一最优配置。",
    "d必须是整只目标基金的收益变动；部分披露股票冲击不能冒充整基金收益。",
    "L0和B必须属于同一明确预算范围和同一情景；未覆盖资产风险不按零补齐。",
    "A含申购外扣费，入基金金额为A/(1+f)；d已包含净值内费用，不重复扣年费。",
    "情景不是最大可能亏损、保证止损或实际成交预测；未使用概率或Kelly下注。",
]


def _inputs(inputs: Mapping | None, kwargs: Mapping, allowed: set[str]) -> dict:
    if inputs is not None and not isinstance(inputs, Mapping):
        raise ValidationError("条件金额输入必须是对象")
    result = dict(inputs or {})
    if result.keys() & kwargs.keys():
        raise ValidationError("同一条件不能同时在对象和关键字中重复提供")
    result.update(kwargs)
    if result.keys() - allowed:
        raise ValidationError(f"不支持的条件字段: {', '.join(sorted(result.keys() - allowed))}")
    return result


def _number(value: Any, name: str, *, signed: bool = False) -> Decimal | None:
    if value is None or value == "":
        return None
    if name in {"known_cohort_market_value", "selected_market_value", "existing_market_value",
                "available_redemption_value"}:
        # These may be derived shares * NAV products, outside the user-input
        # magnitude bound but still exact, finite decimal text.
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValidationError(f"{name}必须是有效十进制数") from exc
        if isinstance(value, bool) or not result.is_finite():
            raise ValidationError(f"{name}必须是有限数")
    else:
        result = decimal(value)
    if not signed and result < 0:
        raise ValidationError(f"{name}不得为负")
    return result


def _ratio(value: Decimal | None, name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValidationError(f"{name}必须在0和1之间")


def _text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _divide(numerator: Decimal, denominator: Decimal, *, minimum: bool = False) -> Decimal:
    with localcontext() as context:
        context.prec = _PRECISION
        context.rounding = ROUND_CEILING if minimum else ROUND_FLOOR
        return numerator / denominator


def _cohort(total: Decimal | None, selected: Decimal | None) -> None:
    if total is not None and selected is not None and selected > total:
        raise ValidationError("目标基金市值不能大于已知人民币基金市值分母")


def _static_buy(data: Mapping) -> dict:
    names = ("gross_cash", "subscription_fee", "known_cohort_market_value", "selected_market_value")
    values = {key: _number(data.get(key), key) for key in names}
    gross, fee, total, selected = (values[key] for key in names)
    _ratio(fee, "subscription_fee")
    _cohort(total, selected)
    net = gross / (1 + fee) if gross is not None and fee is not None else None
    post_total = total + net if total is not None and net is not None else None
    post_selected = selected + net if selected is not None and net is not None else None
    weight = post_selected / post_total \
        if post_selected is not None and post_total is not None and post_total > 0 else None
    return {
        "status": "available" if weight is not None else "incomplete",
        "gross_cash": _text(gross), "subscription_fee": _text(fee),
        "subscription_fee_amount": _text(gross - net) if net is not None else None,
        "net_subscription": _text(net), "known_cohort_market_value": _text(total),
        "selected_market_value": _text(selected), "post_buy_market_value": _text(post_selected),
        "post_buy_cohort_market_value": _text(post_total),
        "weight_before": str(selected / total)
        if selected is not None and total is not None and total > 0 else None,
        "weight_after": _text(weight),
        "unknowns": [key for key in names if values[key] is None],
        "formula": "net=A/(1+f); post_weight=(selected_value+net)/(known_cohort_value+net)",
        "assumptions": [
            "按传入估值静态预演；未记录基金不自动当作零持仓，需明确填0。",
            "仅把新增入基金金额加入已知基金市值分母；未估值基金和其他个人资产不在分母中。",
            "不预测实际申购净值、确认份额或到账；外部现金减少A，不把费用计为基金资产。",
        ],
    }


def post_buy_weight(inputs: Mapping | None = None, **kwargs: Any) -> dict:
    """Static whole-fund weight from explicit known values and gross new cash.

    Accept {gross_cash, subscription_fee, known_cohort_market_value,
    selected_market_value}, either as one mapping or keywords. Unknown selected
    value is not zero, including when the selected fund was not recorded.
    """
    data = _inputs(inputs, kwargs, {"gross_cash", "subscription_fee",
                                    "known_cohort_market_value", "selected_market_value"})
    with localcontext() as context:
        context.prec = _PRECISION
        return _static_buy(data)


planned_buy = post_buy_weight


def planned_redemption(inputs: Mapping | None = None, **kwargs: Any) -> dict:
    """Static gross sale, fee, net proceeds and remaining fund-cohort weight.

    Redemption cash leaves the *fund* cohort, so its denominator falls by gross
    redemption, not net proceeds. No actual order, sale eligibility or timing is
    inferred. A known selected value also bounds a static proposed sale.
    """
    names = {"gross_redemption", "redemption_fee", "known_cohort_market_value",
             "selected_market_value", "available_redemption_value"}
    data = _inputs(inputs, kwargs, names)
    values = {key: _number(data.get(key), key) for key in names}
    gross, fee = values["gross_redemption"], values["redemption_fee"]
    total, selected = values["known_cohort_market_value"], values["selected_market_value"]
    available = values["available_redemption_value"]
    _ratio(fee, "redemption_fee")
    _cohort(total, selected)
    if gross is not None and any(gross > bound for bound in (selected, available)
                                 if bound is not None):
        raise ValidationError("拟赎回毛额超过明确的目标基金市值或可售金额")
    with localcontext() as context:
        context.prec = _PRECISION
        net = gross * (1 - fee) if gross is not None and fee is not None else None
        remaining = selected - gross if selected is not None and gross is not None else None
        denominator = total - gross if total is not None and gross is not None else None
        if denominator is not None and denominator < 0:
            raise ValidationError("拟赎回毛额超过已知基金市值分母")
        weight = remaining / denominator \
            if remaining is not None and denominator is not None and denominator > 0 else None
        required = ("gross_redemption", "redemption_fee", "known_cohort_market_value",
                    "selected_market_value")
        return {
            "status": "available" if net is not None and remaining is not None
            and denominator is not None else "incomplete",
            "gross_redemption": _text(gross), "redemption_fee": _text(fee),
            "redemption_fee_amount": _text(gross - net) if net is not None else None,
            "net_proceeds": _text(net), "remaining_market_value": _text(remaining),
            "remaining_cohort_market_value": _text(denominator), "weight_after": _text(weight),
            "unknowns": [key for key in required if values[key] is None],
            "formula": "net=x*(1-r); post_weight=(selected_value-x)/(known_cohort_value-x)",
            "assumptions": ["静态预演，不预测实际成交净值或资金可用时间；现金已不在基金市值分母。"],
        }


def _scenario_unknowns(data: Mapping, values: Mapping) -> list[str]:
    unknowns = [key for key in ("scenario_return", "existing_scenario_loss", "loss_budget")
                if values[key] is None]
    if data.get("scenario_scope") != "whole_fund":
        unknowns.append("whole_fund_scenario_required")
    if data.get("budget_scope") not in ("fund", "covered_portfolio"):
        unknowns.append("explicit_budget_scope_required")
    if not data.get("scenario_label") and not data.get("scenario_source"):
        unknowns.append("scenario_label_or_source")
    return unknowns


def _reduction(data: Mapping, values: Mapping, scenario_unknowns: list[str]) -> dict:
    loss, budget, change = (values[key] for key in
                            ("existing_scenario_loss", "loss_budget", "scenario_return"))
    fee = values["redemption_fee"]
    available = values["available_redemption_value"]
    if available is None:
        available = values["existing_market_value"]
    if (available is not None and values["existing_market_value"] is not None
            and available > values["existing_market_value"]):
        raise ValidationError("可售金额不能超过明确的现有目标基金市值")
    limits = {"available_redemption_value": available, "max_redemption": values["max_redemption"]}
    known_limits = {key: value for key, value in limits.items() if value is not None}
    maximum = min(known_limits.values()) if available is not None else None
    ell = -change if change is not None else None
    relief = ell - fee if ell is not None and fee is not None else None
    excess = max(Decimal(0), loss - budget) if loss is not None and budget is not None else None
    unknowns = list(scenario_unknowns)
    if fee is None:
        unknowns.append("redemption_fee")
    if available is None:
        unknowns.append("available_redemption_value_or_existing_market_value")
    result = {
        "status": "incomplete", "loss_fraction": _text(ell), "redemption_fee": _text(fee),
        "loss_relief_per_gross_redemption": _text(relief), "budget_excess": _text(excess),
        "minimum_gross_redemption": None, "feasible_minimum_gross_redemption": None,
        "maximum_gross_redemption": _text(maximum), "feasible": None,
        "available_value_basis": "caller_confirmed_redeemable"
        if values["available_redemption_value"] is not None else
        "current_market_value_upper_bound" if available is not None else "unknown",
        "net_proceeds_at_minimum": None, "fee_at_minimum": None,
        "loss_after_minimum_redemption": None,
        "loss_after_maximum_redemption": _text(loss - maximum * relief)
        if loss is not None and maximum is not None and relief is not None else None,
        "binding_constraints": [key for key, value in known_limits.items() if value == maximum],
        "unknowns": unknowns,
        "formula": "loss_after=L0-x*(ell-r); min_x=max(0,L0-B)/(ell-r) only if ell>r",
        "assumptions": [
            "x为当前赎回毛额，r为该笔实际适用的赎回费；不从持有天数或份额类别猜费率。",
            "卖出后保留现金，现金在此情景中不再承担目标基金风险；未计未知税费与再投资。",
            "情景所需最少减少额不等于研究上应退出或唯一卖出方案。",
            "若仅有现有市值，可行性只指静态金额不超过市值；未确认实际可赎回资格或到账时间。",
        ],
    }
    if scenario_unknowns:
        return result
    if excess == 0:
        result.update(status="not_required", minimum_gross_redemption="0",
                      feasible_minimum_gross_redemption="0", feasible=True,
                      loss_after_minimum_redemption=str(loss))
        return result
    if relief is None:
        return result
    if relief <= 0:
        result.update(status="no_loss_relief", feasible=False)
        return result
    minimum = _divide(excess, relief, minimum=True)
    result["minimum_gross_redemption"] = str(minimum)
    if maximum is None:
        return result
    feasible = minimum <= maximum
    result.update(status="available" if feasible else "infeasible", feasible=feasible)
    if feasible:
        result.update(
            feasible_minimum_gross_redemption=str(minimum),
            net_proceeds_at_minimum=str(minimum * (1 - fee)),
            fee_at_minimum=str(minimum * fee),
            loss_after_minimum_redemption=str(loss - minimum * relief),
        )
    return result


def conditional_size(inputs: Mapping | None = None, **kwargs: Any) -> dict:
    """Return conditional whole-fund buy capacity and same-scenario reduction.

    Required for a complete buy limit: scenario_return=d (>=-1), explicit
    subscription_fee=f, existing_scenario_loss=L0, loss_budget=B, spendable_cash,
    scenario_scope='whole_fund', budget_scope='fund' or 'covered_portfolio', and
    a scenario_label or scenario_source. L0 is a supplied *signed* net loss (a
    negative value is a supplied scenario gain), not inferred from holdings.

    Optional max_investment/exposure_cap are explicit gross-cash limits.
    Optional max_fund_weight is a caller-chosen static target-fund weight cap,
    requiring known_cohort_market_value and selected_market_value. Redemption
    uses redemption_fee and available_redemption_value (or existing_market_value),
    optionally max_redemption. A requested_gross_cash produces a separate static
    post-buy preview even when risk-budget inputs are unknown.
    """
    data = _inputs(inputs, kwargs, _FIELDS)
    values = {key: _number(data.get(key), key,
                           signed=key in {"scenario_return", "existing_scenario_loss"})
              for key in _NUMBERS}
    change, fee = values["scenario_return"], values["subscription_fee"]
    if change is not None and change < -1:
        raise ValidationError("整基金情景收益不得低于-100%")
    for key in ("subscription_fee", "redemption_fee", "max_fund_weight"):
        _ratio(values[key], key)
    if data.get("budget_scope") not in (None, "", "fund", "covered_portfolio"):
        raise ValidationError("预算范围必须明确为fund或covered_portfolio")
    for key in ("scenario_scope", "scenario_label", "scenario_source"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValidationError(f"{key}必须是文本")
    total, selected = values["known_cohort_market_value"], values["selected_market_value"]
    _cohort(total, selected)
    with localcontext() as context:
        context.prec = _PRECISION
        scenario_unknowns = _scenario_unknowns(data, values)
        unknowns = list(scenario_unknowns)
        if fee is None:
            unknowns.append("subscription_fee")
        if values["spendable_cash"] is None:
            unknowns.append("spendable_cash")
        constraints = {key: values[key] for key in ("spendable_cash", "max_investment", "exposure_cap")
                       if values[key] is not None}
        weight_cap = values["max_fund_weight"]
        exposure_already_exceeded = False
        if weight_cap is not None:
            missing = [key for key in ("known_cohort_market_value", "selected_market_value",
                                      "subscription_fee") if values[key] is None]
            unknowns.extend(key for key in missing if key not in unknowns)
            if not missing:
                exposure_already_exceeded = selected > weight_cap * total
                if weight_cap < 1:
                    constraints["max_fund_weight"] = _divide(
                        max(Decimal(0), weight_cap * total - selected) * (1 + fee), 1 - weight_cap,
                    )
        cash_cap = min(constraints.values()) if values["spendable_cash"] is not None else None
        numerator = fee - change if fee is not None and change is not None else None
        denominator = 1 + fee if fee is not None else None
        k = numerator / denominator if numerator is not None else None
        loss, budget = values["existing_scenario_loss"], values["loss_budget"]
        loss_cap = None
        loss_bound_status = "incomplete"
        loss_unknowns = scenario_unknowns + (["subscription_fee"] if fee is None else [])
        if not loss_unknowns:
            if numerator > 0:
                # Algebraically cancel (1+f), rather than divide by a rounded k.
                loss_cap = _divide(max(Decimal(0), budget - loss) * denominator, numerator)
                constraints["scenario_loss_budget"] = loss_cap
                loss_bound_status = "available"
            else:
                loss_bound_status = "no_positive_loss_bound"
        minimum_cash = None
        if not loss_unknowns:
            if numerator < 0:
                minimum_cash = _divide(max(Decimal(0), loss - budget) * denominator,
                                       -numerator, minimum=True)
            elif loss <= budget:
                minimum_cash = Decimal(0)
        cap = min(constraints.values()) if not unknowns else None
        binding = [key for key, value in constraints.items() if value == cap]
        loss_at_cap = loss + cap * numerator / denominator \
            if cap is not None and not loss_unknowns else None
        budget_feasible = (loss - budget) * denominator + cap * numerator <= 0 \
            if cap is not None and not loss_unknowns else None
        # A zero buy cap does not repair an already-breached exposure constraint.
        feasible = budget_feasible and not exposure_already_exceeded \
            if budget_feasible is not None else None
        whole_fund = {
            "status": "incomplete" if unknowns else "infeasible" if not feasible else "available",
            "scenario_return": _text(change), "subscription_fee": _text(fee),
            "existing_scenario_loss": _text(loss), "loss_budget": _text(budget),
            "budget_scope": data.get("budget_scope"), "scenario_scope": data.get("scenario_scope"),
            "scenario_label": data.get("scenario_label"), "scenario_source": data.get("scenario_source"),
            "k": _text(k), "k_numerator": _text(numerator), "k_denominator": _text(denominator),
            "gross_cash_cap": _text(cap), "cash_constraint_cap": _text(cash_cap),
            "loss_bound_cap": _text(loss_cap), "loss_bound_status": loss_bound_status,
            "minimum_gross_cash_for_budget": _text(minimum_cash),
            "constraints": {key: str(value) for key, value in constraints.items()},
            "binding_constraints": binding, "scenario_loss_at_cap": _text(loss_at_cap),
            "budget_feasible_at_cap": budget_feasible, "feasible": feasible,
            "existing_loss_over_budget": loss > budget if loss is not None and budget is not None else None,
            "existing_weight_over_cap": exposure_already_exceeded if weight_cap is not None else None,
            "unknowns": unknowns,
            "formula": "k=1-(1+d)/(1+f)=(f-d)/(1+f); cap=min(available constraints,max(0,B-L0)/k) if k>0",
            "weight_cap_formula": "A<=(w*known_cohort_value-selected_value)*(1+f)/(1-w) for w<1",
            "assumptions": list(_ASSUMPTIONS),
        }
        reduction = _reduction(data, values, scenario_unknowns)
        requested = values["requested_gross_cash"]
        post_buy = None
        if requested is not None:
            post_buy = _static_buy({"gross_cash": requested, "subscription_fee": fee,
                                    "known_cohort_market_value": total,
                                    "selected_market_value": selected})
            post_buy["within_conditional_cap"] = requested <= cap if cap is not None else None
            post_buy["budget_scope_feasible"] = (
                (loss - budget) * denominator + requested * numerator <= 0
                if not loss_unknowns else None
            )
        return {
            "schema_version": SCHEMA_VERSION, "status": whole_fund["status"],
            "gross_cash_cap": whole_fund["gross_cash_cap"], "k": whole_fund["k"],
            "whole_fund": whole_fund, "reduction": reduction, "post_buy": post_buy,
            "unknowns": list(unknowns), "assumptions": list(_ASSUMPTIONS),
        }

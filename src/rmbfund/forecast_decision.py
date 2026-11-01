"""Compare forecasted portfolio value and actual transaction costs, not eligibility."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

from .domain import ValidationError


def _fee(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("交易费率应填写比例，未知时留空") from exc
    if isinstance(value, bool) or not isfinite(result) or not 0 <= result <= 1:
        raise ValidationError("交易费率必须在0%至100%之间")
    return result


def _percent(value: float) -> str:
    if 0 < abs(value) < 0.0001:
        return "不足0.01%" if value > 0 else "负向不足0.01%"
    return f"{value:+.2%}"


def _primary_lag(prediction: Mapping, entry_lag: int | None) -> int | None:
    if entry_lag is None:
        assumptions = prediction.get("entry_assumptions")
        entry_lag = assumptions.get("baseline_entry_lag") if isinstance(assumptions, Mapping) else None
    if entry_lag is not None and (
        isinstance(entry_lag, bool) or not isinstance(entry_lag, int) or entry_lag < 0
    ):
        raise ValidationError("主要成交延迟必须是非负整数笔净值，未知时留空")
    return entry_lag


def build_decision(
    prediction: Mapping, *, holding: str = "unknown", horizon: int = 20,
    entry_lag: int | None = None,
    subscription_fee: str | float | None = None,
    redemption_fee: str | float | None = None,
    pending_order: str = "unknown",
) -> dict:
    """Use the issued primary execution head, without changing its distribution.

    The common origin is a hypothetical application NAV, not the last published
    NAV and not the user's acquisition cost. Future holdings are marked to NAV,
    not assumed liquidated. Cash interest is zero in this comparison. A missing
    primary head never promotes a shorter or more favourable stress scenario.
    """
    if holding not in {"unknown", "flat", "held"}:
        raise ValidationError("持有情况必须为unknown、flat或held")
    if pending_order not in {"unknown", "none", "buy", "sell"}:
        raise ValidationError("在途状态必须为unknown、none、buy或sell")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in (1, 5, 20):
        raise ValidationError("预测期限必须为1、5或20笔净值")
    primary_lag = _primary_lag(prediction, entry_lag)
    buy_fee, sell_fee = _fee(subscription_fee), _fee(redemption_fee)
    heads = sorted(
        (head for head in prediction.get("execution_forecasts", [])
         if head.get("horizon") == horizon
         and isinstance(head.get("entry_lag"), int)
         and not isinstance(head["entry_lag"], bool) and head["entry_lag"] >= 0),
        key=lambda head: (head["entry_lag"] != primary_lag, head["entry_lag"]),
    )
    if len({head["entry_lag"] for head in heads}) != len(heads):
        raise ValidationError("同一期限和成交延迟有重复预测，无法确定原始主情景")
    assumptions = [
        f"以同一假定申请净值为起点，比较之后{horizon}笔净值的基金账面价值与现金；现金收益按0计算。",
        "期限和延迟按实际净值记录的笔数计算，不承诺交易日、成交日期或到账日期。",
        "申购费按外扣法计算；已经付过的买入费用不再扣，净值内的年度运作费用不重复扣。",
        "持有相对退出的优势，是两种做法的预期终值差额占起点持仓市值的比例；新买相对现金的优势以含申购费的总投入为基准。",
        "继续持有和新买后的终值按净值计，不假设届时一定赎回；若计划届时取现，还需计入那时的费用与到账时间。",
    ]
    execution_note = {
        "unknown": "实际申赎、费用和到账安排以当前渠道显示为准。",
        "none": "这只是动作比较，不会提交买卖。",
        "buy": "有买入尚未确认；预测照常展示，先核对这笔在途交易，避免重复申请。",
        "sell": "有赎回尚未完成；预测照常展示，先核对在途份额和到账安排。",
    }[pending_order]
    scenarios = []
    for head in heads:
        change = float(head["expected_change"])
        if not isfinite(change) or change < -1:
            raise ValidationError("预测净值变化必须为有限数且不能低于-100%")
        buy_net = (1 + change) / (1 + buy_fee) - 1 if buy_fee is not None else None
        hold_advantage = change + sell_fee if sell_fee is not None else None
        flat_action = ("buy" if buy_net > 0 else "wait") if buy_net is not None else (
            "conditional_buy" if change > 0 else "wait"
        )
        held_action = ("sell" if hold_advantage < 0 else "hold") if hold_advantage is not None else (
            "conditional_sell" if change < 0 else "hold"
        )
        samples = head.get("samples", [])
        buy_probability = (sum(value > buy_fee for value in samples) / len(samples)
                           if samples and buy_fee is not None else None)
        hold_probability = (sum(value > -sell_fee for value in samples) / len(samples)
                            if samples and sell_fee is not None else None)
        sell_probability = (sum(value < -sell_fee for value in samples) / len(samples)
                            if samples and sell_fee is not None else None)
        scenarios.append({
            "entry_lag": head["entry_lag"], "horizon": horizon,
            "is_primary": head["entry_lag"] == primary_lag,
            "market_expected_change": change,
            "buy_advantage_vs_cash": buy_net, "hold_advantage_vs_exit": hold_advantage,
            # Kept for existing callers; new personal assessments use the explicit names.
            "expected_change": change, "buy_net_change": buy_net, "hold_advantage": hold_advantage,
            "lower": head.get("lower"), "upper": head.get("upper"),
            "up_probability": head.get("up_probability"),
            "down_probability": head.get("down_probability"),
            "flat_probability": head.get("flat_probability"),
            "flat_action": flat_action, "held_action": held_action,
            "buy_profit_probability": buy_probability,
            "hold_better_probability": hold_probability, "sell_better_probability": sell_probability,
            "buy_break_even_fee": change if change >= 0 else None,
            "sell_break_even_fee": -change if change <= 0 else None,
            "subscription_fee_known": buy_fee is not None, "redemption_fee_known": sell_fee is not None,
        })
    primary = next((item for item in scenarios if item["is_primary"]), None)
    primary_head = next((head for head in heads if head["entry_lag"] == primary_lag), None)
    common = {
        "primary_horizon": horizon, "primary_entry_lag": primary_lag, "entry_lag": primary_lag,
        "primary_head": deepcopy(primary_head), "primary": primary,
        "scenarios": scenarios, "assumptions": assumptions, "execution_note": execution_note,
    }
    if primary is None:
        reason = (
            "这份预测没有保存主要成交延迟，不能擅自选择最短或最有利的情景；请显式更新预测或指定原定延迟。"
            if primary_lag is None else
            f"缺少延迟{primary_lag}笔净值成交、再看之后{horizon}笔净值的主情景预测；请显式更新匹配期限的预测。"
        )
        reasons = [reason, "已公布净值之后的走势不等于今天申请能获得的收益，其他期限或延迟情景也不能代替主判断。"]
        branch = {
            "action": "unavailable", "label": "还没有可用于操作比较的预测", "reasons": reasons,
            "market_expected_change": None, "expected_change": None, "break_even_fee": None,
        }
        return {
            **common, "status": "missing_primary_lag" if primary_lag is None else "missing_primary_head",
            "branches": {
                "flat": {**deepcopy(branch), "buy_advantage_vs_cash": None, "buy_profit_probability": None},
                "held": {**deepcopy(branch), "hold_advantage_vs_exit": None,
                         "hold_better_probability": None, "sell_better_probability": None},
            },
            "market_expected_change": None, "buy_advantage_vs_cash": None,
            "hold_advantage_vs_exit": None, "delay_sensitive": False,
            "delay_sensitive_branches": {"flat": False, "held": False}, "break_even": {},
        }
    change = primary["market_expected_change"]
    window = (f"按延迟{primary_lag}笔净值成交的情景，成交之后{horizon}笔净值的预期变化"
              f"为{_percent(change)}。")
    flat_action, held_action = primary["flat_action"], primary["held_action"]
    if flat_action == "buy":
        flat_label = "倾向买入，费用后仍有正向空间"
        flat_reasons = [window, f"计入你填写的申购费后，新买相对保持现金的预期优势为{_percent(primary['buy_advantage_vs_cash'])}。"]
    elif flat_action == "conditional_buy":
        flat_label = "有买入空间，先对照实际费用"
        flat_reasons = [window, f"申购费低于{primary['buy_break_even_fee']:.2%}时，这项比较才优于保持现金；达到或超过这个费率时倾向等待。实际费率尚未填写。"]
    else:
        flat_label = "暂不买入"
        flat_reasons = [window, f"计入你填写的申购费后，新买相对保持现金的预期差额比例为{_percent(primary['buy_advantage_vs_cash'])}，没有正向空间。" if buy_fee is not None
                        else "即使先不计申购费，这个窗口也没有正向预期。"]
    if held_action == "sell":
        held_label = "倾向退出"
        held_reasons = [window, f"你填写的当前赎回费为{sell_fee:.2%}，继续持有相对退出的预期差额比例为{_percent(primary['hold_advantage_vs_exit'])}；按此比较，转为现金的预期终值更高。"]
    elif held_action == "conditional_sell":
        held_label = "倾向退出，先比较赎回费"
        held_reasons = [window, f"当前赎回费低于{primary['sell_break_even_fee']:.2%}时，退出的预期终值高于继续持有；达到或超过这个费率时倾向继续持有。实际费率尚未填写。"]
    else:
        held_label = "倾向继续持有"
        held_reasons = [window,
                        f"计入你填写的当前赎回费后，继续持有相对退出的预期优势为{_percent(primary['hold_advantage_vs_exit'])}，预期终值不低于转为现金。"
                        if sell_fee is not None else
                        "即使当前赎回免费，继续持有的预期终值也不低于转为现金；未知的赎回费没有按零扣除。"]
    sensitive = {}
    for branch, name, reasons in (("flat", "flat_action", flat_reasons),
                                  ("held", "held_action", held_reasons)):
        sensitive[branch] = any(item[name] != primary[name] for item in scenarios if not item["is_primary"])
        if sensitive[branch]:
            reasons.append("不同成交延迟会改变动作比较，主判断仍采用原定情景；具体差异见情景表。")
    return {
        **common, "status": "ready",
        "market_expected_change": change,
        "buy_advantage_vs_cash": primary["buy_advantage_vs_cash"],
        "hold_advantage_vs_exit": primary["hold_advantage_vs_exit"],
        "branches": {
            "flat": {"action": flat_action, "label": flat_label, "reasons": flat_reasons,
                     "market_expected_change": change,
                     "buy_advantage_vs_cash": primary["buy_advantage_vs_cash"],
                     "buy_profit_probability": primary["buy_profit_probability"],
                     "expected_change": primary["buy_net_change"],
                     "break_even_fee": primary["buy_break_even_fee"]},
            "held": {"action": held_action, "label": held_label, "reasons": held_reasons,
                     "market_expected_change": change,
                     "hold_advantage_vs_exit": primary["hold_advantage_vs_exit"],
                     "hold_better_probability": primary["hold_better_probability"],
                     "sell_better_probability": primary["sell_better_probability"],
                     "expected_change": change, "break_even_fee": primary["sell_break_even_fee"]},
        },
        "delay_sensitive": any(sensitive.values()), "delay_sensitive_branches": sensitive,
        "break_even": {"subscription_fee": primary["buy_break_even_fee"],
                       "redemption_fee": primary["sell_break_even_fee"]},
    }

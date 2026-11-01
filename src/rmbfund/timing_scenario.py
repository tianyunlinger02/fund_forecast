"""Pure equal-capital comparisons and explicitly reconstructed NAV replay.

Legacy comparisons mark to NAV; calendar-cash comparisons include each lot's
planned redemption fee and receivable. Neither is an order, an optimal position,
or an actual channel-fee history. Forecasts select actions before future NAVs
measure what those actions would have done.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
from math import fsum, isfinite
from typing import Any

from .domain import ValidationError
from .forecast_decision import build_decision


_ACTIONS = (
    ("buy", "新买入", "new", "wait", "保持现金"),
    ("wait", "保持现金", "new", "wait", "保持现金"),
    ("hold", "继续持有", "existing", "exit", "全部退出"),
    ("add", "追加同额", "existing", "hold", "继续持有"),
    ("reduce50", "减仓一半", "existing", "hold", "继续持有"),
    ("exit", "全部退出", "existing", "hold", "继续持有"),
)


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label}必须是有限数") from exc
    if isinstance(value, bool) or not isfinite(result):
        raise ValidationError(f"{label}必须是有限数")
    return result


def _base(value: Any) -> float:
    result = _number(value, "模拟本金")
    if result <= 0 or not isfinite(2 * result):
        raise ValidationError("模拟本金必须为正数，且同额备用金合计可表示")
    return result


def _fee(value: Any) -> float | None:
    if value is None:
        return None
    result = _number(value, "交易费率")
    if not 0 <= result <= 1:
        raise ValidationError("交易费率必须在0%至100%之间")
    return result


def _day(value: Any) -> str:
    if type(value) is date:
        return value.isoformat()
    try:
        result = date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValidationError("净值及买入日期必须是实际YYYY-MM-DD日期") from exc
    if result != value:
        raise ValidationError("净值及买入日期必须是实际YYYY-MM-DD日期")
    return result


def _series(series: Sequence[Mapping]) -> list[dict]:
    rows, previous = [], ""
    for raw in series:
        day = _day(raw.get("date"))
        nav = _number(raw.get("nav"), "净值")
        if day <= previous or nav <= 0:
            raise ValidationError("实际净值必须为正数，日期严格递增且不可重复")
        for flag in ("invalid_return", "label_valid"):
            if flag in raw and not isinstance(raw[flag], bool):
                raise ValidationError("净值有效性标记必须是布尔值")
        rows.append({**raw, "date": day, "nav": nav})
        previous = day
    return rows


def resolve_fees(
    fee_record: Mapping | None, *, purchase_date, valuation_date,
    subscription_override=None, redemption_override=None,
) -> dict:
    """Prefer explicit overrides, then direct public rates, then labeled examples.

    Redemption ``max_days`` is inclusive, matching the public fee payload. An
    unknown holding-day convention uses calendar days ONLY AS A SIMULATION.
    Metadata is passed through, never upgraded to channel verification.
    """
    record = fee_record if isinstance(fee_record, Mapping) else {}
    valuation = _day(valuation_date) if valuation_date is not None else None
    purchase = _day(purchase_date) if purchase_date is not None else None
    days = ((date.fromisoformat(valuation) - date.fromisoformat(purchase)).days
            if purchase is not None and valuation is not None else None)
    if days is not None and days < 0:
        raise ValidationError("买入日期不能晚于估值日期")
    confirmed_basis = record.get("holding_day_basis") == "confirmation_calendar_days"
    basis = "confirmation_calendar_days" if confirmed_basis else "calendar_days_simulation"
    basis_label = ("按指定买入日作为确认日起点，按自然日计数" if confirmed_basis else
                   "自然日模拟：公开资料未确认持有日口径，不是已确认的实际赎回费")
    subscription, redemption = None, None
    band = None
    if subscription_override is not None:
        subscription = {"rate": _fee(subscription_override), "source": "override",
                        "label": "手动指定的申购费假设（非渠道核实）"}
    elif record.get("subscription_rate") is not None:
        subscription = {"rate": _fee(record["subscription_rate"]), "source": "public_standard",
                        "label": "公开标准申购费率（不等于当前渠道折扣）"}
    if redemption_override is not None:
        redemption = {"rate": _fee(redemption_override), "source": "override",
                      "label": "手动指定的赎回费假设（非渠道核实）"}
    else:
        bands = []
        for raw in record.get("redemption_bands") or ():
            low, high = raw.get("min_days"), raw.get("max_days")
            if (isinstance(low, bool) or not isinstance(low, int) or low < 0
                    or (high is not None and (isinstance(high, bool)
                        or not isinstance(high, int) or high < low))):
                raise ValidationError("公开赎回费分段的自然日范围无效")
            bands.append({"min_days": low, "max_days": high, "rate": _fee(raw.get("rate"))})
            if bands[-1]["rate"] is None:
                raise ValidationError("公开赎回费分段缺少费率")
        bands.sort(key=lambda item: item["min_days"])
        for left, right in zip(bands, bands[1:]):
            if left["max_days"] is None or right["min_days"] <= left["max_days"]:
                raise ValidationError("公开赎回费分段交叠，max_days应含端点")
        matches = [item for item in bands if days is not None and item["min_days"] <= days
                   and (item["max_days"] is None or days <= item["max_days"])]
        if matches:
            band = matches[0]
            redemption = {"rate": band["rate"], "source": "public_standard",
                          "label": "公开标准赎回分段费率；" + basis_label}
        elif (days is None and len(bands) == 1 and bands[0]["min_days"] == 0
              and bands[0]["max_days"] is None):
            band = bands[0]
            redemption = {"rate": band["rate"], "source": "public_standard",
                          "label": "公开统一赎回费率（不依赖持有日分段，非渠道核实）"}
    if subscription is None:
        subscription = {"rate": 0.0015, "source": "illustrative",
                        "label": "示例申购费0.15%：未取得适用公开费率，不是实际渠道费率"}
    if redemption is None:
        redemption = {"rate": 0.005, "source": "illustrative",
                      "label": "示例赎回费0.50%：未确定适用公开分段，不是实际渠道费率"}
    return {
        "subscription_fee": subscription["rate"], "redemption_fee": redemption["rate"],
        "subscription": subscription, "redemption": {**redemption, "band": band},
        "purchase_date": purchase, "valuation_date": valuation, "holding_days": days,
        "holding_day_basis": basis, "holding_day_label": basis_label,
        "observed_at": record.get("observed_at"), "available_at": record.get("available_at"),
        "evidence_refs": deepcopy(record.get("evidence_refs", [])),
        "channel_verified": record.get("channel_verified") is True,
        "assumptions": ["申购采用外扣法；年度管理费、托管费和销售服务费已在净值中，不重复扣除。",
                        "这些费率用于动作模拟；不据此确认当前渠道费率或历史生效版本。"],
    }


def next_lower_fee_tier(
    fee_record: Mapping | None, *, confirmation_date, application_date,
    current_rate=None, base_amount=10000, deadline_date=None,
) -> dict:
    """Explain the next cheaper public band, not a requirement to keep holding.

    Savings hold the example redemption value constant at B. They do not predict
    the NAV on that later date, or verify that an application will be accepted.
    """
    base = _base(base_amount)
    confirmation = _day(confirmation_date) if confirmation_date is not None else None
    application = _day(application_date) if application_date is not None else None
    deadline = _day(deadline_date) if deadline_date is not None else None
    result = {"status": "unavailable", "date": None, "days_to_wait": None,
              "current_rate": _fee(current_rate), "next_rate": None,
              "saving_per_base": None, "base_amount": base, "within_deadline": None,
              "label": "按同一示例赎回市值比较费率，不预测等待期间的涨跌，也不要求持有到下一档。"}
    if confirmation is None or application is None:
        return {**result, "reason": "缺少模拟确认日或申请日，不能计算下一较低费率日期。"}
    if confirmation > application:
        return {**result, "reason": "申请日早于模拟确认日，这笔份额当日尚不能赎回。"}
    current = resolve_fees(fee_record, purchase_date=confirmation, valuation_date=application)
    if result["current_rate"] is None:
        result["current_rate"] = current["redemption_fee"]
    if current["redemption"]["source"] != "public_standard":
        return {**result, "reason": "没有适用的公开费率档位，不从示例费率推算下一档。"}
    candidates = []
    record = fee_record if isinstance(fee_record, Mapping) else {}
    for band in record.get("redemption_bands") or ():
        if float(band["rate"]) >= result["current_rate"]:
            continue
        try:
            day = (date.fromisoformat(confirmation) + timedelta(days=band["min_days"])).isoformat()
        except OverflowError:
            continue
        if day > application:
            candidates.append((day, float(band["rate"])))
    if not candidates:
        return {**result, "status": "no_lower_tier", "reason": "公开费表中没有更晚的较低费率档。"}
    day, rate = min(candidates)
    return {**result, "status": "available", "date": day,
            "days_to_wait": (date.fromisoformat(day) - date.fromisoformat(application)).days,
            "next_rate": rate, "saving_per_base": base * (result["current_rate"] - rate),
            "within_deadline": day <= deadline if deadline is not None else None,
            "reason": "假设当前公开费表未来不变；还需核对该日是否可申请，等待可能上涨也可能下跌。"}


def resolve_plan_fees(
    fee_record: Mapping | None, *, plan: Mapping, old_confirmation_date,
    new_confirmation_date=None, subscription_override=None, current_redemption_override=None,
    future_old_redemption_override=None, future_new_redemption_override=None,
) -> dict:
    """Resolve each simulated lot/date independently using the legacy fee rules.

    Confirmation dates are hypotheses, never assertions about a real holding.
    Ordinary date infeasibility is returned with its context rather than raised;
    an override of today's redemption fee does not leak into either future lot.
    """
    if not isinstance(plan, Mapping):
        raise ValidationError("自然日计划应为日期与净值笔数映射")
    old = _day(old_confirmation_date) if old_confirmation_date is not None else None
    new_value = (new_confirmation_date if new_confirmation_date is not None
                 else plan.get("new_confirmation_date"))
    new = _day(new_value) if new_value is not None else None
    application = _day(plan["application_date"]) if plan.get("application_date") is not None else None
    sale = _day(plan["sale_date"]) if plan.get("sale_date") is not None else None
    deadline_value = plan.get("deadline_date", plan.get("deadline"))
    deadline = _day(deadline_value) if deadline_value is not None else None
    contexts = {}
    for key, confirmation, valuation, override, label in (
        ("current", old, application, current_redemption_override, "旧份额在本次申请日退出"),
        ("future_old", old, sale, future_old_redemption_override, "旧份额在计划卖出申请日赎回"),
        ("future_new", new, sale, future_new_redemption_override, "新买或追加份额在计划卖出申请日赎回"),
    ):
        negative = confirmation is not None and valuation is not None and confirmation > valuation
        context = resolve_fees(
            fee_record, purchase_date=None if negative else confirmation, valuation_date=valuation,
            subscription_override=subscription_override, redemption_override=override,
        )
        context.update(label=label, confirmation_date=confirmation,
                       confirmation_date_basis="hypothetical_not_actual_record",
                       requested_redemption_override=_fee(override),
                       application_date=valuation, status="ready", reasons=[])
        context["holding_day_label"] = ("模拟确认日起点，不是实际交易确认记录；"
                                        + context["holding_day_label"])
        context["assumptions"].append("确认日起点是单独填写的模拟假设，不由历史买入净值日认定。")
        if key != "current":
            context["assumptions"].append("假设当前公开费表在计划卖出时仍适用；不代表届时渠道核实费率。")
        if negative:
            reason = f"{label}的申请日{valuation}早于模拟确认日{confirmation}，该笔份额尚未确认。"
            context.update(purchase_date=confirmation,
                           holding_days=(date.fromisoformat(valuation)
                                         - date.fromisoformat(confirmation)).days,
                           redemption_fee=None, status="confirmation_after_application", reasons=[reason])
            context["redemption"] = {"rate": None, "source": "unavailable", "band": None,
                                     "label": reason}
        elif valuation is None:
            context.update(status="missing_application_date", reasons=[f"{label}缺少申请日。"])
        if override is None and not negative:
            context["next_lower_fee"] = next_lower_fee_tier(
                fee_record, confirmation_date=confirmation, application_date=valuation,
                current_rate=context["redemption_fee"], deadline_date=deadline,
            )
        else:
            context["next_lower_fee"] = {
                "status": "unavailable", "date": None, "saving_per_base": None,
                "reason": "手动费率不自动延伸为未来费率阶梯。" if override is not None else context["reasons"][0],
            }
        contexts[key] = context
    return {
        "subscription_fee": contexts["current"]["subscription_fee"],
        "current_redemption_fee": contexts["current"]["redemption_fee"],
        "future_old_redemption_fee": contexts["future_old"]["redemption_fee"],
        "future_new_redemption_fee": contexts["future_new"]["redemption_fee"],
        "subscription": deepcopy(contexts["current"]["subscription"]), "contexts": contexts,
        "old_confirmation_date": old, "new_confirmation_date": new,
        "next_lower_fee": deepcopy(contexts["current"]["next_lower_fee"]),
        "assumptions": [
            "旧份额的当前赎回、旧份额的未来赎回、新份额的未来赎回分别计费，不共用手动覆盖。",
            "新份额从自己的模拟确认日计龄，不能继承假设历史买入日或旧份额的费龄。",
            "未来公开费率按当前费表不变模拟；费用缺资料时使用明示示例，不改变市场预测。",
        ],
    }


def _validate_head(head: Mapping) -> None:
    for key in ("expected_change", "lower", "upper"):
        if key == "expected_change" or head.get(key) is not None:
            if _number(head.get(key), "预测净值变化") < -1:
                raise ValidationError("预测净值变化不能低于-100%")
    if head.get("lower") is not None and head.get("upper") is not None:
        if float(head["lower"]) > float(head["upper"]):
            raise ValidationError("预测区间上下界颠倒")
    for key in ("up_probability", "down_probability", "flat_probability"):
        if head.get(key) is not None and not 0 <= _number(head[key], "预测概率") <= 1:
            raise ValidationError("预测概率必须在0至1之间")
    for sample in head.get("samples") or ():
        if _number(sample, "预测样本") < -1:
            raise ValidationError("预测样本不能低于-100%")


def _value(action: str, change: float | None, base: float,
           buy_fee: float | None, sell_fee: float | None) -> float | None:
    if action == "wait":
        result = base
    elif action == "exit":
        result = None if sell_fee is None else base * (1 - sell_fee) + base
    elif change is None:
        return None
    elif action == "buy":
        result = None if buy_fee is None else base * (1 + change) / (1 + buy_fee)
    elif action == "hold":
        result = base * (1 + change) + base
    elif action == "add":
        result = (None if buy_fee is None else
                  base * (1 + change) + base * (1 + change) / (1 + buy_fee))
    else:  # Sell one half now, leave the other half marked to NAV, keep reserve B.
        result = (None if sell_fee is None else
                  base * ((1 + change) / 2 + (1 - sell_fee) / 2) + base)
    if result is not None and not isfinite(result):
        raise ValidationError("模拟金额超出可表示范围")
    return result


def _advantage(action: str, change: float | None, base: float,
               buy_fee: float | None, sell_fee: float | None) -> float | None:
    # Direct differences preserve the exact decision thresholds, including ties.
    if action == "wait":
        return 0.0
    if change is None:
        return None
    if action in {"buy", "add"}:
        result = None if buy_fee is None else base * ((1 + change) / (1 + buy_fee) - 1)
    else:
        result = None if sell_fee is None else base * (change + sell_fee)
        if result is not None and action in {"reduce50", "exit"}:
            result *= -0.5 if action == "reduce50" else -1
    if result is not None and not isfinite(result):
        raise ValidationError("模拟差额超出可表示范围")
    return result


def _action_rows(head: Mapping, base: float, buy_fee: float | None,
                 sell_fee: float | None, primary: Mapping) -> list[dict]:
    expected = _number(head["expected_change"], "预测净值变化")
    lower = float(head["lower"]) if head.get("lower") is not None else None
    upper = float(head["upper"]) if head.get("upper") is not None else None
    probabilities = {
        "buy": primary["buy_profit_probability"], "add": primary["buy_profit_probability"],
        "hold": primary["hold_better_probability"], "wait": 0.0,
        "reduce50": primary["sell_better_probability"], "exit": primary["sell_better_probability"],
    }
    rows = []
    for action, label, family, reference, comparison in _ACTIONS:
        initial = base if family == "new" else 2 * base
        value = _value(action, expected, base, buy_fee, sell_fee)
        limits = [_advantage(action, change, base, buy_fee, sell_fee) for change in (lower, upper)]
        rows.append({
            "action": action, "label": label, "family": family,
            "status": "ready" if value is not None else "unknown_fee",
            "initial_value": initial, "expected_value": value,
            "lower_value": _value(action, lower, base, buy_fee, sell_fee),
            "upper_value": _value(action, upper, base, buy_fee, sell_fee),
            "expected_change": (value / initial - 1) if value is not None else None,
            "comparison_action": reference, "comparison_label": comparison,
            "advantage": _advantage(action, expected, base, buy_fee, sell_fee),
            "lower_advantage": min(limits) if all(item is not None for item in limits) else None,
            "upper_advantage": max(limits) if all(item is not None for item in limits) else None,
            "better_probability": probabilities[action],
        })
    return rows


def _purchase(rows: list[dict], purchase_date, base: float) -> dict:
    latest = rows[-1] if rows else {}
    day = _day(purchase_date) if purchase_date is not None else None
    result = {"date": day, "nav": None, "latest_date": latest.get("date"),
              "latest_nav": latest.get("nav"), "change": None, "profit": None,
              "holding_days": None, "status": "not_specified" if day is None else "missing_nav"}
    selected = next((index for index, row in enumerate(rows) if row["date"] == day), None)
    if selected is None:
        return result
    result.update(nav=rows[selected]["nav"],
                  holding_days=(date.fromisoformat(latest["date"]) - date.fromisoformat(day)).days)
    if any(row.get("invalid_return", False) or row.get("label_valid") is False
           for row in rows[selected + 1:]):
        return {**result, "status": "invalid_window"}
    change = latest["nav"] / rows[selected]["nav"] - 1
    profit = base * change
    if not isfinite(change) or not isfinite(profit):
        raise ValidationError("买入以来的净值变化超出可表示范围")
    return {**result, "status": "available", "change": change, "profit": profit}


def build_scenario(
    prediction: Mapping, *, subject, series=(), purchase_date=None, horizon=20,
    base_amount=10000, subscription_fee=None, redemption_fee=None, fee_basis=None,
) -> dict:
    """Use build_decision's exact saved primary head; never choose another head.

    New buy/cash starts with B. Every existing-position action starts with the
    SAME fund B + cash reserve B. Purchase history stops at the latest known NAV;
    an unknown future application NAV is not compounded onto that past return.
    Unknown fees stay unknown here. ``resolve_fees`` is an explicit opt-in source
    of labeled illustrative/public assumptions, not a hidden builder default.
    """
    base, buy_fee, sell_fee = _base(base_amount), _fee(subscription_fee), _fee(redemption_fee)
    rows = _series(series)
    if not rows and isinstance(prediction.get("anchor"), Mapping):
        rows = _series([prediction["anchor"]])
    decision = build_decision(prediction, horizon=horizon, subscription_fee=buy_fee,
                              redemption_fee=sell_fee)
    head = decision["primary_head"]
    if head is not None:
        _validate_head(head)
    return {
        "kind": "standard_trade", "status": decision["status"],
        "subject": deepcopy(subject), "horizon": horizon, "entry_lag": decision["entry_lag"],
        "base_amount": base, "purchase": _purchase(rows, purchase_date, base),
        "fees": {"subscription_fee": buy_fee, "redemption_fee": sell_fee,
                 "subscription_fee_known": buy_fee is not None,
                 "redemption_fee_known": sell_fee is not None, "basis": deepcopy(fee_basis)},
        "decision": decision,
        "market": {key: head.get(key) if head is not None else None for key in
                   ("expected_change", "lower", "upper", "up_probability",
                    "down_probability", "flat_probability")},
        "rows": _action_rows(head, base, buy_fee, sell_fee, decision["primary"])
                if head is not None else [],
        "assumptions": [
            "新买入与现金各以B为本金；已有持仓的持有、追加、减半、退出均以基金B加备用金B（共2B）比较。",
            "追加使用同一笔备用金B；减仓和退出所得现金留存，现金收益按0计算。",
            "优势为相对比较动作的终值差额（金钱单位），不是年化收益或最优仓位。",
            "买入以来的变化只算至最新已知净值；未来预测从未知的申请成交净值起算，两段不直接相乘。",
            "未来终值按净值计，不自动赎回；年度运作费用不重复扣除，未模拟到账日或交易日。",
        ],
    }


_CALENDAR_FEE_KEYS = ("subscription_fee", "current_redemption_fee",
                      "future_old_redemption_fee", "future_new_redemption_fee")


def _calendar_fees(fee_basis: Mapping | None) -> dict:
    if fee_basis is not None and not isinstance(fee_basis, Mapping):
        raise ValidationError("计划费用应为各批次费用与来源的映射")
    return {key: _fee((fee_basis or {}).get(key)) for key in _CALENDAR_FEE_KEYS}


def _calendar_conditions(plan: Mapping, fee_basis: Mapping | None) -> tuple[dict, list[str]]:
    """Keep infeasible actions separate; cash and old lots need not disappear."""
    if not isinstance(plan, Mapping):
        raise ValidationError("自然日计划应为日期与净值笔数映射")
    reasons = {action: [] for action, *_ in _ACTIONS}
    common = []

    def reject(actions, reason):
        for action in actions:
            if reason not in reasons[action]:
                reasons[action].append(reason)
        if reason not in common:
            common.append(reason)

    future = ("buy", "hold", "add", "reduce50")
    horizon, lag = plan.get("horizon"), plan.get("entry_lag")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        reject(future, f"该日程预计净值笔数H={horizon}，不足以完成正期限的计划卖出模拟；不借用其他期限。")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        reject(future, f"该日程的净值延迟L={lag}不可用，不能另选成交起点。")
    dates = {key: _day(plan[key]) if plan.get(key) is not None else None
             for key in ("decision_date", "application_date", "sale_date", "new_confirmation_date")}
    deadline_value = plan.get("deadline", plan.get("deadline_date"))
    deadline = _day(deadline_value) if deadline_value is not None else None
    application, sale = dates["application_date"], dates["sale_date"]
    if application is None:
        reject((*future, "exit"), "计划没有本次申请日，不能假定现在已能交易。")
    if sale is None:
        reject(future, "截止日前没有计划卖出申请日。")
    if application is not None and deadline is not None and application > deadline:
        reject((*future, "exit"), f"本次申请日{application}晚于截止日{deadline}。")
    if sale is not None and deadline is not None and sale > deadline:
        reject(future, f"计划卖出申请日{sale}晚于截止日{deadline}，不顺延出窗口。")
    if application is not None and sale is not None and sale < application:
        reject(future, f"计划卖出申请日{sale}早于本次申请日{application}。")
    if (application is not None and dates["decision_date"] is not None
            and application < dates["decision_date"]):
        reject((*future, "exit"), "本次申请日早于测算日，不能把过去的申请当作现在可执行。")
    basis = fee_basis or {}
    new_value = basis.get("new_confirmation_date", dates["new_confirmation_date"])
    new = _day(new_value) if new_value is not None else None
    old_value = basis.get("old_confirmation_date")
    old = _day(old_value) if old_value is not None else None
    if new is None:
        reject(("buy", "add"), "没有新份额的模拟确认日，尚不能确认它能在计划卖出前赎回。")
    elif sale is not None and new > sale:
        reject(("buy", "add"), f"新份额模拟确认日{new}晚于计划卖出申请日{sale}，只影响新买与追加。")
    elif sale is not None and application is not None and new < application:
        reject(("buy", "add"), "新份额模拟确认日早于本次申购申请日，不能继承旧份额费龄。")
    elif plan.get("new_shares_sellable") is False and "new_confirmation_date" not in basis:
        reject(("buy", "add"), "日程未能确认新份额可在计划卖出前赎回，只影响新买与追加。")
    if old is not None and application is not None and old > application:
        reject(("reduce50", "exit"), f"旧份额模拟确认日{old}晚于本次申请日{application}，当日不能减仓或退出。")
    if old is not None and sale is not None and old > sale:
        reject(("hold", "add", "reduce50"), f"旧份额模拟确认日{old}晚于计划卖出申请日{sale}。")
    for context_key, actions in (("current", ("reduce50", "exit")),
                                 ("future_old", ("hold", "add", "reduce50")),
                                 ("future_new", ("buy", "add"))):
        context = (basis.get("contexts") or {}).get(context_key) or {}
        if context.get("status", "ready") != "ready":
            for reason in context.get("reasons") or [context.get("label") or "该批次费用日程不可用。"]:
                reject(actions, str(reason))
    return reasons, common


def _calendar_value(action: str, change: float | None, base: float, fees: Mapping) -> float | None:
    buy, current, old, new = (fees[key] for key in _CALENDAR_FEE_KEYS)
    if action == "wait":
        result = base
    elif action == "exit":
        result = base * (1 - current) + base if current is not None else None
    elif change is None:
        return None
    elif action == "buy":
        result = (base * (1 + change) * (1 - new) / (1 + buy)
                  if buy is not None and new is not None else None)
    elif action == "hold":
        result = base * (1 + change) * (1 - old) + base if old is not None else None
    elif action == "add":
        result = (base * (1 + change) * (1 - old)
                  + base * (1 + change) * (1 - new) / (1 + buy)
                  if buy is not None and old is not None and new is not None else None)
    else:
        result = (0.5 * base * (1 - current) + 0.5 * base * (1 + change) * (1 - old) + base
                  if current is not None and old is not None else None)
    if result is not None and not isfinite(result):
        raise ValidationError("计划卖出净额超出可表示范围")
    return result


def _calendar_decimal_advantage(action: str, change, base: float, fees: Mapping) -> Decimal | None:
    if action == "wait":
        return Decimal(0)
    if change is None:
        return None
    buy, current, old, new = (fees[key] for key in _CALENDAR_FEE_KEYS)
    required = (buy, new) if action in {"buy", "add"} else (current, old)
    if any(value is None for value in required):
        return None
    # Keep the strict comparison in Decimal too, before conversion to money
    # floats. No epsilon turns genuine small advantages into ties.
    with localcontext() as context:
        context.prec = 64
        k, amount = 1 + Decimal(str(change)), Decimal(str(base))
        if action in {"buy", "add"}:
            net = k * (1 - Decimal(str(new))) / (1 + Decimal(str(buy)))
            return amount * (net - 1)
        net = k * (1 - Decimal(str(old))) - (1 - Decimal(str(current)))
        multiplier = Decimal("-0.5") if action == "reduce50" else -1 if action == "exit" else 1
        return amount * net * multiplier


def fee_adjusted_actions(
    change, *, subscription_fee, current_redemption_fee,
    future_old_redemption_fee, future_new_redemption_fee,
) -> dict:
    """Compare fixed same-capital net payoffs without constructing a calendar.

    Missing returns/required fees leave the affected branch ``unavailable``.
    Fees follow this module's inclusive 0..1 convention; returns cannot be below
    -100%. Strict Decimal comparisons preserve ties and genuine small margins.
    This is a daily-review rule, not a dynamic optimum or a holding-period rule.
    """
    raw_fees = dict(zip(_CALENDAR_FEE_KEYS, (
        subscription_fee, current_redemption_fee,
        future_old_redemption_fee, future_new_redemption_fee,
    ), strict=True))
    _calendar_fees(raw_fees)
    for value in raw_fees.values():
        if value is not None and not 0 <= Decimal(str(value)) <= 1:
            raise ValidationError("交易费率必须在0%至100%之间")
    if change is not None:
        _number(change, "预测净值变化")
        if Decimal(str(change)) < -1:
            raise ValidationError("预测净值变化不能低于-100%")
    buy = _calendar_decimal_advantage("buy", change, 1, raw_fees)
    hold = _calendar_decimal_advantage("hold", change, 1, raw_fees)
    return {
        "flat": ("buy" if buy > 0 else "wait") if buy is not None else "unavailable",
        "held": ("redeem" if hold < 0 else "hold") if hold is not None else "unavailable",
        "buy_advantage_vs_cash": float(buy) if buy is not None else None,
        "hold_advantage_vs_redeem": float(hold) if hold is not None else None,
    }


def _calendar_advantage(action: str, change: float | None, base: float, fees: Mapping) -> float | None:
    margin = _calendar_decimal_advantage(action, change, base, fees)
    result = float(margin) if margin is not None else None
    if result is not None and not isfinite(result):
        raise ValidationError("计划卖出差额超出可表示范围")
    return result


def _calendar_sample_probability(action: str, samples, base: float, fees: Mapping) -> float | None:
    margins = [_calendar_decimal_advantage(action, sample, base, fees) for sample in samples]
    return (sum(margin > 0 for margin in margins) / len(margins)
            if margins and all(margin is not None for margin in margins) else None)


def _calendar_projection(head: Mapping, fees: Mapping, conditions: Mapping) -> dict:
    _validate_head(head)
    change = float(head["expected_change"])
    margins, probabilities = {}, {}
    samples = head.get("samples") or ()
    for action, reference in (("buy", "wait"), ("hold", "exit"), ("exit", "hold")):
        usable = not conditions[action] and not conditions[reference]
        margin = _calendar_advantage(action, change, 1, fees) if usable else None
        margins[action] = margin
        probabilities[action] = _calendar_sample_probability(action, samples, 1, fees) if usable else None
    buy, current, old, new = (fees[key] for key in _CALENDAR_FEE_KEYS)
    buy_break_even = (1 + change) * (1 - new) - 1 if new is not None else None
    sell_break_even = 1 - (1 + change) * (1 - old) if old is not None else None
    return {
        "entry_lag": head["entry_lag"], "horizon": head["horizon"],
        "market_expected_change": change,
        **{key: head.get(key) for key in ("lower", "upper", "up_probability",
                                         "down_probability", "flat_probability")},
        "buy_advantage_vs_cash": margins["buy"], "hold_advantage_vs_exit": margins["hold"],
        "buy_profit_probability": probabilities["buy"],
        "hold_better_probability": probabilities["hold"],
        "sell_better_probability": probabilities["exit"],
        "flat_action": ("buy" if margins["buy"] > 0 else "wait")
                       if margins["buy"] is not None else "unavailable",
        "held_action": ("sell" if margins["hold"] < 0 else "hold")
                       if margins["hold"] is not None else "unavailable",
        "buy_break_even_fee": buy_break_even if buy_break_even is not None and buy_break_even >= 0 else None,
        "sell_break_even_fee": sell_break_even if sell_break_even is not None and sell_break_even >= 0 else None,
        "subscription_fee_known": buy is not None,
        "redemption_fee_known": current is not None,
        "future_old_redemption_fee_known": old is not None,
        "future_new_redemption_fee_known": new is not None,
    }


def _calendar_head_reasons(prediction: Mapping, head: Mapping | None, horizon, lag) -> dict:
    # Import locally: personal_assessment may itself accept a prebuilt scenario.
    from .personal_assessment import explain_forecast_head

    result = explain_forecast_head(prediction, head, horizon=horizon, entry_lag=lag,
                                   role="execution", all_factors=True)
    result["identity"].update({key: deepcopy(prediction.get(key)) for key in
                               ("input_hash", "request_hash", "target_definition", "generated_at")})
    return result


def _calendar_heads(prediction: Mapping, horizon, lag) -> tuple[list, Mapping | None, list]:
    valid_horizon = not isinstance(horizon, bool) and isinstance(horizon, int) and horizon > 0
    valid_lag = not isinstance(lag, bool) and isinstance(lag, int) and lag >= 0
    heads = [head for head in prediction.get("execution_forecasts", ())
             if valid_horizon and isinstance(head, Mapping) and head.get("horizon") == horizon
             and isinstance(head.get("horizon"), int) and not isinstance(head["horizon"], bool)
             and isinstance(head.get("entry_lag"), int) and not isinstance(head["entry_lag"], bool)
             and head["entry_lag"] >= 0]
    if len({head["entry_lag"] for head in heads}) != len(heads):
        raise ValidationError("同一期限和成交延迟有重复预测，无法确定日历计划的主情景")
    matched = next((head for head in heads if valid_lag and head["entry_lag"] == lag), None)
    primary = matched if matched is not None and matched.get("status", "available") == "available" else None
    unavailable = [row for row in prediction.get("unavailable_forecasts", ())
                   if isinstance(row, Mapping) and row.get("horizon") == horizon
                   and row.get("entry_lag") == lag]
    if matched is not None and primary is None:
        unavailable.insert(0, matched)
    return heads, primary, unavailable


def build_timing_decision(prediction: Mapping, *, plan: Mapping, fee_basis=None) -> dict:
    """Exact positive H/L calendar-cash comparison; legacy build_decision is untouched."""
    fees = _calendar_fees(fee_basis)
    conditions, plan_reasons = _calendar_conditions(plan, fee_basis)
    horizon, lag = plan.get("horizon"), plan.get("entry_lag")
    valid_horizon = not isinstance(horizon, bool) and isinstance(horizon, int) and horizon > 0
    valid_lag = not isinstance(lag, bool) and isinstance(lag, int) and lag >= 0
    heads, primary_head, unavailable = _calendar_heads(prediction, horizon, lag)
    unavailable_reasons = [str(row.get("reason") or row.get("fallback_reason") or row.get("status"))
                           for row in unavailable]
    scenarios = []
    for head in sorted(heads, key=lambda row: (row["entry_lag"] != lag, row["entry_lag"])):
        if head.get("status", "available") == "available":
            scenarios.append({**_calendar_projection(head, fees, conditions),
                              "is_primary": valid_lag and head["entry_lag"] == lag})
    primary = next((row for row in scenarios if row["is_primary"]), None)
    explanation = _calendar_head_reasons(prediction, primary_head, horizon, lag)
    missing = []
    if primary_head is None:
        missing = [f"缺少与本次日程精确匹配的H={horizon}、L={lag}可用预测，不借用其他期限或延迟。",
                   *unavailable_reasons]
    action_reasons = {key: [*value] for key, value in conditions.items()}
    for action in ("buy", "hold", "add", "reduce50"):
        action_reasons[action].extend(missing)
    change = primary["market_expected_change"] if primary is not None else None
    branches = {}
    for branch, action, reference, advantage_key, probability_keys in (
        ("flat", "buy", "wait", "buy_advantage_vs_cash", ("buy_profit_probability",)),
        ("held", "hold", "exit", "hold_advantage_vs_exit",
         ("hold_better_probability", "sell_better_probability")),
    ):
        reasons = list(dict.fromkeys([*action_reasons[action], *action_reasons[reference]]))
        selected = primary[f"{branch}_action"] if primary is not None else "unavailable"
        margin = primary[advantage_key] if primary is not None else None
        if selected == "unavailable":
            if not reasons:
                reasons = ["该比较涉及的当前或未来批次费率未知，未按零费率计算。"]
            label = "本窗口无法比较新买后卖出" if branch == "flat" else "本窗口尚不能比较持有与退出"
        else:
            label = {"buy": "倾向新买，计划卖出费用后仍有正向空间", "wait": "暂不新买，计划卖出净额未胜过现金",
                     "hold": "倾向继续持有后按计划卖出", "sell": "倾向当前退出"}[selected]
            reasons = [f"按本次日程预计H={horizon}笔净值、L={lag}笔延迟；期末包含赎回应收款，不承诺到账。",
                       f"按各批次申赎费用计算，相对{'保持现金' if branch == 'flat' else '当前退出'}的预期净额差额为每笔本金的{margin:+.2%}。"]
            if margin == 0:
                reasons.append("预期净额相同；不把持平算作严格胜过对照。")
        branches[branch] = {
            "action": selected, "label": label, "reasons": reasons,
            "market_expected_change": change, advantage_key: margin,
            **{key: primary[key] if primary is not None else None for key in probability_keys},
            "break_even_fee": primary["buy_break_even_fee" if branch == "flat" else "sell_break_even_fee"]
                              if primary is not None else None,
        }
    sensitive = {branch: primary is not None and any(
        row[f"{branch}_action"] != primary[f"{branch}_action"] for row in scenarios if not row["is_primary"])
        for branch in ("flat", "held")}
    status = ("infeasible_window" if not valid_horizon else
              "missing_primary_lag" if lag is None else "invalid_primary_lag" if not valid_lag else
              "unavailable_primary_head" if unavailable and primary_head is None else
              "missing_primary_head" if primary_head is None else "ready")
    if primary is not None and all(conditions[action] for action in ("buy", "hold", "reduce50")):
        status = "infeasible_window"
    return {
        "kind": "calendar_cash", "status": status,
        "primary_horizon": horizon, "primary_entry_lag": lag, "entry_lag": lag,
        "primary_head": deepcopy(primary_head), "primary": primary, "scenarios": scenarios,
        "branches": branches, "market_expected_change": change,
        "buy_advantage_vs_cash": primary["buy_advantage_vs_cash"] if primary is not None else None,
        "hold_advantage_vs_exit": primary["hold_advantage_vs_exit"] if primary is not None else None,
        "delay_sensitive": any(sensitive.values()), "delay_sensitive_branches": sensitive,
        "break_even": {"subscription_fee": primary["buy_break_even_fee"],
                       "redemption_fee": primary["sell_break_even_fee"]} if primary is not None else {},
        "execution": explanation, "model_identity": deepcopy(explanation["identity"]),
        "head_reasons": deepcopy(explanation["reasons"]), "unavailable_head_reasons": unavailable_reasons,
        "plan_reasons": list(dict.fromkeys([*(plan.get("reasons") or []), *plan_reasons])),
        "action_reasons": action_reasons,
        "execution_note": "这是同一计划下的净额比较，不会提交交易；实际申赎安排以渠道为准。",
        "end_value_basis": "cash_and_redemption_receivable",
        "assumptions": ["概率对同一个精确H/L预测头的原始样本逐一应用各批次净额公式，再严格比较对照。",
                        "没有保存样本时不推算费用后胜率；原头的市场涨跌概率保持不变。",
                        "期限仍按预计净值笔数测算，不宣称已验证的精确自然日概率或最优买卖日。"],
    }


def _calendar_action_rows(head: Mapping | None, base: float, fees: Mapping,
                          decision: Mapping) -> list[dict]:
    expected = float(head["expected_change"]) if head is not None else None
    lower = float(head["lower"]) if head is not None and head.get("lower") is not None else None
    upper = float(head["upper"]) if head is not None and head.get("upper") is not None else None
    samples = head.get("samples") or () if head is not None else ()
    conditions = decision["action_reasons"]
    rows = []
    for action, label, family, reference, comparison in _ACTIONS:
        initial = base if family == "new" else 2 * base
        reasons = list(conditions[action])
        value = _calendar_value(action, expected, base, fees) if not reasons else None
        if value is None and not reasons:
            reasons.append("该动作涉及的批次费率未知，未按零费率计算。")
        comparison_reasons = list(conditions[reference])
        comparable = not reasons and not comparison_reasons
        margin = _calendar_advantage(action, expected, base, fees) if comparable else None
        limits = [_calendar_advantage(action, change, base, fees) if comparable else None
                  for change in (lower, upper)]
        probability = (_calendar_sample_probability(action, samples, base, fees)
                       if comparable else None)
        if action == "wait":
            probability = 0.0
        rows.append({
            "action": action, "label": label, "family": family,
            "status": "ready" if value is not None else "unavailable", "reasons": reasons,
            "initial_value": initial, "expected_value": value,
            "lower_value": _calendar_value(action, lower, base, fees) if not reasons else None,
            "upper_value": _calendar_value(action, upper, base, fees) if not reasons else None,
            "expected_change": value / initial - 1 if value is not None else None,
            "comparison_action": reference, "comparison_label": comparison,
            "comparison_reasons": comparison_reasons,
            "advantage": margin,
            "lower_advantage": min(limits) if all(item is not None for item in limits) else None,
            "upper_advantage": max(limits) if all(item is not None for item in limits) else None,
            "better_probability": probability,
            "end_value_basis": "cash_and_redemption_receivable",
        })
    return rows


def build_calendar_scenario(
    prediction: Mapping, *, subject, plan: Mapping, series=(), purchase_date=None,
    base_amount=10000, fee_basis=None,
) -> dict:
    """Compare six actions' planned net cash plus redemption receivables, not NAV marks."""
    base, fees = _base(base_amount), _calendar_fees(fee_basis)
    navs = _series(series)
    if not navs and isinstance(prediction.get("anchor"), Mapping):
        navs = _series([prediction["anchor"]])
    decision = build_timing_decision(prediction, plan=plan, fee_basis=fee_basis)
    head = decision["primary_head"]
    return {
        "kind": "calendar_cash", "status": decision["status"],
        "subject": deepcopy(subject), "plan": deepcopy(dict(plan)),
        "horizon": plan.get("horizon"), "entry_lag": plan.get("entry_lag"),
        "base_amount": base, "purchase": _purchase(navs, purchase_date, base),
        "fees": deepcopy(dict(fee_basis)) if fee_basis is not None else fees,
        "decision": decision, "execution": deepcopy(decision["execution"]),
        "market": {key: head.get(key) if head is not None else None for key in
                   ("expected_change", "lower", "upper", "up_probability",
                    "down_probability", "flat_probability")},
        "rows": _calendar_action_rows(head, base, fees, decision),
        "end_value_basis": "cash_and_redemption_receivable",
        "assumptions": [
            "新买与现金共同本金B；旧份额四种动作共同本金为基金B加备用现金B（共2B）。",
            "各批次按计划申请赎回后的净额比较，期末包含赎回应收款，不代表已经到账现金。",
            "旧份额当前退出、旧份额计划卖出与新份额计划卖出分别扣各自费率；申购采用外扣法。",
            "已计入净值的年度费用不重复扣；现金收益按零计算，不预测到账日期。",
            "过去假设买入日只解释截至最新已知净值的变化，不是实际确认日，也不与未来收益直接相乘。",
            "这是截止日前的一份日程假设，不是在窗口内自动寻找最佳买卖日，也不要求等满某个费率档。",
        ],
    }


def build_sale_date_comparison(
    predictions=(), *, plan: Mapping, fee_record: Mapping | None = None,
    old_confirmation_date=None, current_redemption_override=None,
    future_old_redemption_override=None, base_amount=10000,
) -> dict:
    """Independently compare now A, the next cheaper open day D and plan day S.

    No fitting, I/O, date search for an optimum, or joint path across heads. The
    original B fund plus unchanged B reserve matches calendar-cash comparisons.
    Public tiers determine D even with a manual A fee; manual S fees only apply
    at S. Missing exact heads leave the pure fee saving/boundary visible.
    """
    from .trade_calendar import project_sale_date

    base = _base(base_amount)
    forecasts = [predictions] if isinstance(predictions, Mapping) else list(predictions)
    application = _day(plan["application_date"])
    sale = _day(plan["sale_date"]) if plan.get("sale_date") is not None else None
    confirmation = _day(old_confirmation_date) if old_confirmation_date is not None else None
    fees = resolve_plan_fees(
        fee_record, plan=plan, old_confirmation_date=confirmation,
        current_redemption_override=current_redemption_override,
        future_old_redemption_override=future_old_redemption_override,
    )
    # Determine the next PUBLIC tier, never a staircase extrapolated from a
    # manually supplied current rate. Its actual comparison still uses A's fee.
    tier = next_lower_fee_tier(
        fee_record, confirmation_date=confirmation, application_date=application,
        deadline_date=sale, base_amount=base,
    )
    candidates = {application: {"candidate_id": "now", "candidate_ids": ["now"],
                                "sale_date": application, "horizon": 0}}
    notes = []
    if tier["status"] == "available":
        projected = project_sale_date(plan, not_before=tier["date"])
        tier.update(projected_sale_date=projected["sale_date"], projection=projected,
                    within_sale_window=projected["status"] == "ready")
        if projected["status"] == "ready":
            day = projected["sale_date"]
            projected_context = resolve_plan_fees(
                fee_record, plan={**plan, "sale_date": day}, old_confirmation_date=confirmation,
            )["contexts"]["future_old"]
            tier["projected_rate"] = (projected_context["redemption_fee"]
                                      if projected_context["redemption"]["source"] == "public_standard" else None)
            candidates[day] = {"candidate_id": "next_lower_fee", "candidate_ids": ["next_lower_fee"],
                               "sale_date": day, "horizon": projected["horizon"]}
        else:
            notes.extend(projected["reasons"])
    if sale is not None and sale >= application:
        if sale in candidates:
            candidates[sale]["candidate_ids"].append("plan")
        else:
            candidates[sale] = {"candidate_id": "plan", "candidate_ids": ["plan"],
                                "sale_date": sale, "horizon": plan.get("horizon")}
    else:
        notes.append("当前情景没有不早于本次申请日的计划卖出日，保留现在退出及可用的费用说明。")
    labels = {"now": "现在申请退出", "next_lower_fee": "下一较低费档的开放日", "plan": "当前情景卖出日"}

    def usable_rate(context):
        return (context["redemption_fee"] if context.get("status") == "ready"
                and context["redemption"]["source"] in {"public_standard", "override"} else None)

    current_context = fees["contexts"]["current"]
    current = usable_rate(current_context)
    rows = []
    for day, candidate in sorted(candidates.items()):
        now = day == application
        if now:
            context = current_context
        elif "plan" in candidate["candidate_ids"]:
            context = fees["contexts"]["future_old"]
        else:
            context = resolve_plan_fees(
                fee_record, plan={**plan, "sale_date": day}, old_confirmation_date=confirmation,
                current_redemption_override=current_redemption_override,
            )["contexts"]["future_old"]
        rate = usable_rate(context)
        cash_fees = _calendar_fees({"current_redemption_fee": current, "future_old_redemption_fee": rate})
        horizon, lag = candidate["horizon"], plan.get("entry_lag")
        selected, primary, unavailable = {}, None, []
        if not now:
            for forecast in forecasts:
                _, matched, missing = _calendar_heads(forecast, horizon, lag)
                unavailable.extend(missing)
                if matched is not None:
                    if primary is not None:
                        raise ValidationError("卖出候选有重复的精确H/L预测，不能混合多份样本。")
                    selected, primary = forecast, matched
            if primary is not None:
                _validate_head(primary)
        expected = float(primary["expected_change"]) if primary is not None else None
        lower = float(primary["lower"]) if primary is not None and primary.get("lower") is not None else None
        upper = float(primary["upper"]) if primary is not None and primary.get("upper") is not None else None
        reasons = list(context.get("reasons") or [])
        if rate is None and not reasons:
            reasons.append("该卖出日没有适用的公开或手动赎回费率；未把未知费率当作零或示例费率。")
        if current is None and not now:
            reasons.append("现在退出的费率或旧份额确认条件不可用，暂不能计算相对现在的净额差。")
        if not now and primary is None:
            reasons.append(f"尚缺本次申请到该日的{horizon}笔净值预测（成交起点向后{lag}笔）；目前只比较费用。")
            reasons.extend(str(row.get("reason") or row.get("fallback_reason") or row.get("status"))
                           for row in unavailable)
        value = (_calendar_value("exit", None, base, cash_fees) if now else
                 _calendar_value("hold", expected, base, cash_fees))
        saving = _calendar_advantage("hold", 0, base, cash_fees)
        with localcontext() as decimal_context:
            decimal_context.prec = 64
            boundary = (float((1 - Decimal(str(current))) / (1 - Decimal(str(rate))) - 1)
                        if current is not None and rate is not None and rate < 1 else None)
        margin = (0.0 if current is not None else None) if now else _calendar_advantage(
            "hold", expected, base, cash_fees,
        )
        limits = ([_calendar_advantage("hold", change, base, cash_fees) for change in (lower, upper)]
                  if not now else [margin, margin])
        identity = (_calendar_head_reasons(selected, primary, horizon, lag)["identity"]
                    if primary is not None else None)
        rows.append({
            **candidate, "label": " / ".join(labels[key] for key in candidate["candidate_ids"]),
            "entry_lag": lag, "decision_at": plan.get("decision_at"),
            "status": "ready" if value is not None and (now or margin is not None) else "fee_only"
                      if saving is not None else "unavailable",
            "reasons": list(dict.fromkeys(reasons)), "fee_context": deepcopy(context),
            "redemption_fee": rate, "fee_known": rate is not None,
            "fee_status": context["redemption"]["source"] if rate is not None else "unknown",
            "fee_label": context["redemption"]["label"] if rate is not None else
                         "该日费率未知；不使用示例或零费率代替",
            "current_redemption_fee": current,
            "fee_saving": saving, "break_even_change": boundary,
            "break_even_status": "available" if boundary is not None else
                                 "no_finite_boundary" if rate == 1 and current is not None else "unknown_fee",
            "initial_value": 2 * base, "base_amount": base, "reserve_cash": base,
            "expected_value": value,
            "lower_value": value if now else _calendar_value("hold", lower, base, cash_fees),
            "upper_value": value if now else _calendar_value("hold", upper, base, cash_fees),
            "expected_change": expected, "advantage": margin,
            "lower_advantage": min(limits) if all(item is not None for item in limits) else None,
            "upper_advantage": max(limits) if all(item is not None for item in limits) else None,
            "better_probability": 0.0 if now and current is not None else
                                  _calendar_sample_probability("hold", primary.get("samples") or (), base, cash_fees)
                                  if primary is not None else None,
            "comparison_candidate_id": "now", "primary_head": deepcopy(primary),
            "model_identity": deepcopy(identity), "end_value_basis": "cash_and_redemption_receivable",
        })
    if application == sale and future_old_redemption_override is not None:
        notes.append("计划卖出日与现在申请日相同，已去重；现在费率只采用当前覆盖，未套用未来覆盖。")
    return {
        "kind": "old_share_sale_dates", "status": "ready" if all(row["status"] == "ready" for row in rows)
                else "partial", "plan": deepcopy(dict(plan)), "rows": rows,
        "base_amount": base, "initial_value": 2 * base, "reserve_cash": base,
        "next_lower_fee": tier, "notes": list(dict.fromkeys(notes)),
        "end_value_basis": "cash_and_redemption_receivable",
        "assumptions": [
            "只比较同一笔旧份额的现在、下一低费档开放日与当前情景卖出日；日期相同只算一次，不寻找全局最优日期。",
            "各行本金均为旧基金B加备用现金B；备用现金相同且收益为零，差额只来自旧份额净额。",
            "每个未来日期只与现在退出独立比较，范围和严格胜过频率来自该日期同一精确H/L头；不同期限样本不是联合路径。",
            "纯费用节省固定赎回前市值为B；盈亏边界r*=(1-fA)/(1-fD)-1，不是涨跌预测或必须等待指令。",
            "降费日期向后投影到原日程中的开放日并重算费率；公开费表未来不变只是情景假设。",
            "当前手动费率只用于现在，未来手动旧份额费率只用于S；未知费率明确保留，不外推为阶梯。",
            "期末含赎回应收款，不承诺到账；不制造候选之间的交叉胜率或最佳日期概率。",
        ],
    }


def _history_decision(head: Any, horizon: int, lag: int, buy_fee, sell_fee) -> dict | None:
    if (not isinstance(head, Mapping) or head.get("horizon") != horizon
            or head.get("entry_lag") != lag or head.get("status", "available") != "available"):
        return None
    _validate_head(head)
    return build_decision({"entry_assumptions": {"baseline_entry_lag": lag},
                           "execution_forecasts": [head]}, horizon=horizon,
                          subscription_fee=buy_fee, redemption_fee=sell_fee)


def _realized(rows: list[dict], actual: float, base: float, buy_fee, sell_fee) -> list[dict]:
    return _realized_rows(
        rows, actual, lambda row: _value(row["action"], actual, base, buy_fee, sell_fee),
        lambda row: _advantage(row["action"], actual, base, buy_fee, sell_fee),
    )


def _realized_rows(rows: list[dict], actual: float, value_for, advantage_for) -> list[dict]:
    result = []
    for row in rows:
        action = row["action"]
        value = value_for(row)
        advantage = advantage_for(row)
        payoff = value - row["initial_value"] if value is not None else None
        label = "unknown_fee" if advantage is None else "better" if advantage > 0 else (
            "worse" if advantage < 0 else "tie")
        if action == "buy" and payoff is not None:
            label = "wrong_buy" if payoff < 0 else "profitable_buy" if payoff > 0 else "break_even"
        if action in {"reduce50", "exit"} and advantage is not None:
            label = ("avoided_loss" if advantage > 0 else "missed_profit" if advantage < 0
                     and actual > 0 else "cost_exceeds_avoided_loss" if advantage < 0 else "tie")
        result.append({**row, "actual_value": value, "realized_payoff": payoff,
                       "actual_change": value / row["initial_value"] - 1 if value is not None else None,
                       "realized_advantage": advantage, "outcome_label": label})
    return result


def _selected(decision: Mapping, comparisons: list[dict]) -> dict:
    by_action = {row["action"]: row for row in comparisons}
    return {branch: deepcopy(by_action.get(
                "exit" if decision["branches"][branch]["action"] == "sell"
                else decision["branches"][branch]["action"],
                {"action": decision["branches"][branch]["action"], "realized_advantage": None}))
            for branch in ("flat", "held")}


def _metrics(rows: list[dict], key: str = "actions") -> dict:
    result = {}
    for branch in ("flat", "held"):
        actions = [row[key][branch] for row in rows if key in row
                   and row[key][branch].get("realized_payoff") is not None]
        margins = [item for item in actions if item["realized_advantage"] is not None]
        count, comparison_count = len(actions), len(margins)
        result[branch] = {
            "count": count, "comparison_count": comparison_count,
            "mean_expected_payoff": fsum((item["expected_value"] - item["initial_value"]) / count
                                         for item in actions) if count else None,
            "mean_realized_payoff": fsum(item["realized_payoff"] / count for item in actions)
                                    if count else None,
            "mean_expected_advantage": fsum(item["advantage"] / comparison_count for item in margins)
                                       if comparison_count else None,
            "mean_realized_advantage": fsum(item["realized_advantage"] / comparison_count
                                           for item in margins) if comparison_count else None,
            "better_count": sum(item["realized_advantage"] > 0 for item in margins),
            "worse_count": sum(item["realized_advantage"] < 0 for item in margins),
        }
    return result


def summarize_history(
    series, historical_forecasts, *, horizon, entry_lag, subscription_fee,
    redemption_fee, base_amount=10000,
) -> dict:
    """Replay chronological, pre-outcome heads, including unavailable/tail rows.

    This is a reconstruction, NEVER saved forward issuance or a compounded
    strategy return. Fixed supplied fees apply to every row; historical channel
    fees and actual NAV availability/confirmation times are not reconstructed.
    Only ``prediction`` chooses actions. No outcome/evaluation input is read.
    """
    base, buy_fee, sell_fee = _base(base_amount), _fee(subscription_fee), _fee(redemption_fee)
    if isinstance(entry_lag, bool) or not isinstance(entry_lag, int) or entry_lag < 0:
        raise ValidationError("历史重建成交延迟必须是非负整数笔净值")
    # Reuse the same supported horizon validation as the live comparison.
    build_decision({"entry_assumptions": {"baseline_entry_lag": entry_lag}}, horizon=horizon,
                   subscription_fee=buy_fee, redemption_fee=sell_fee)
    return _summarize_history_engine(
        series, historical_forecasts, horizon=horizon, entry_lag=entry_lag, base=base,
        decide=lambda head: _history_decision(head, horizon, entry_lag, buy_fee, sell_fee),
        compare=lambda head, decision: _action_rows(head, base, buy_fee, sell_fee, decision["primary"]),
        realize=lambda rows, actual: _realized(rows, actual, base, buy_fee, sell_fee),
        kind="reconstructed",
        fees={"subscription_fee": buy_fee, "redemption_fee": sell_fee,
              "basis": "fixed_across_replay", "label": "整段重建使用同一费率假设，非历史渠道费率"},
        assumptions=[
            "这是逐时点预测的历史重建，不是当时真实保存的发行、下单或前向业绩。",
            "仅使用该时点的同期限同延迟预测选动作；未来净值只用于计算结果，未成熟和无效窗口保留标签。",
            "起止净值索引为原点加延迟、再加期限；按净值笔数计，不补造交易日、公布时刻或确认日期。",
            "进场早于假定可知时点的零延迟窗口标为nowcast，仅作锚点诊断，不计入可执行动作指标。",
            "所有已有持仓动作均含同一备用金B；未来按净值计，不自动赎回，不重复扣年度费用。",
            "指标是每个固定本金窗口的简单平均差额；重叠窗口不连乘，不称为累计策略收益。",
            "每个动作的优势相对其各自标注的比较动作；模型与基线的增量只比较同资本本金盈亏，不能把相反动作的优势直接相减。",
            "不重叠窗口数量不代表相互独立样本；历史频率基线只比较同原点同期限同延迟的可用配对。",
        ],
    )


def _summarize_history_engine(
    series, historical_forecasts, *, horizon, entry_lag, base, decide, compare, realize,
    kind, fees, assumptions, retain_details=False,
) -> dict:
    """Shared chronology and same-capital aggregation; payoff policy is supplied."""
    navs = _series(series)
    histories = [row for row in historical_forecasts
                 if row.get("horizon") == horizon and row.get("entry_lag") == entry_lag]
    rows, seen = [], set()
    for record in sorted(histories, key=lambda row: row["origin_index"]):
        origin = record["origin_index"]
        if isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin < len(navs):
            raise ValidationError("历史判断原点必须对应实际净值序列")
        if origin in seen:
            raise ValidationError("同一原点、期限和延迟有重复历史判断")
        seen.add(origin)
        if record.get("origin_date") != navs[origin]["date"]:
            raise ValidationError("历史判断原点日期与实际净值索引不一致")
        if (record.get("knowledge_cutoff_index") != origin
                or record.get("knowledge_cutoff_date") != navs[origin]["date"]):
            raise ValidationError("历史判断的信息截止必须对应原点，不能使用后续净值")
        known = record.get("assumed_known_index")
        if (isinstance(known, bool) or not isinstance(known, int) or known != origin + 1
                or record.get("assumed_known_date") !=
                (navs[known]["date"] if known < len(navs) else None)):
            raise ValidationError("历史判断假定可知日期与下一笔实际净值索引不一致")
        start, end = origin + entry_lag, origin + entry_lag + horizon
        row = (deepcopy(dict(record)) if retain_details else
               {key: deepcopy(record.get(key)) for key in (
                   "origin_index", "origin_date", "knowledge_cutoff_index", "knowledge_cutoff_date",
                   "assumed_known_index", "assumed_known_date", "prediction")})
        if retain_details:
            # Retain source metadata, not supplied evaluation results. In
            # particular, an input baseline_actions must never create a pair.
            for key in ("baseline_actions", "baseline_comparisons", "baseline_decision", "decision",
                        "flat_action", "held_action", "outcome", "evaluation_rows"):
                row.pop(key, None)
        row.update(horizon=horizon, entry_lag=entry_lag, entry_index=start, target_index=end,
                   entry_date=navs[start]["date"] if start < len(navs) else None,
                   target_date=navs[end]["date"] if end < len(navs) else None,
                   actual_change=None, actions={}, comparisons=[], is_nonoverlapping=False)
        immature = end >= len(navs) or known >= len(navs)
        invalid = not immature and any(item.get("invalid_return", False)
                                      or item.get("label_valid") is False
                                      for item in navs[start + 1:end + 1])
        row["outcome_status"] = "immature" if immature else "invalid" if invalid else "mature"
        try:
            decision = decide(record.get("prediction"))
            row["prediction_status"] = "available" if decision is not None else "unavailable"
        except ValidationError:
            decision = None
            row["prediction_status"] = "invalid"
        row["status"] = row["outcome_status"] if decision is not None else row["prediction_status"]
        if row["status"] == "mature" and start < known:
            row["status"] = "nowcast"
        if decision is not None:
            row["flat_action"] = decision["branches"]["flat"]["action"]
            row["held_action"] = decision["branches"]["held"]["action"]
            row["comparisons"] = compare(record["prediction"], decision)
            if retain_details:
                row["decision"] = deepcopy(decision)
        if row["outcome_status"] == "mature":
            actual = navs[end]["nav"] / navs[start]["nav"] - 1
            if not isfinite(actual):
                row.update(status="invalid", outcome_status="invalid")
            else:
                row["actual_change"] = actual
                if decision is not None:
                    row["comparisons"] = realize(row["comparisons"], actual)
                    row["actions"] = _selected(decision, row["comparisons"])
                    baseline = record.get("historical_baseline")
                    if baseline is None:
                        baseline = record["prediction"].get("historical_baseline")
                    try:
                        baseline_decision = decide(baseline)
                    except ValidationError:
                        baseline_decision = None
                    if baseline_decision is not None:
                        baseline_rows = compare(baseline, baseline_decision)
                        baseline_comparisons = realize(baseline_rows, actual)
                        row["baseline_actions"] = _selected(baseline_decision, baseline_comparisons)
                        if retain_details:
                            row["historical_baseline"] = deepcopy(baseline)
                            row["baseline_decision"] = deepcopy(baseline_decision)
                            row["baseline_comparisons"] = baseline_comparisons
        rows.append(row)
    mature = [row for row in rows if row["status"] == "mature"]
    nonoverlapping, previous_end = [], -1
    for row in mature:
        if row["entry_index"] >= previous_end:
            row["is_nonoverlapping"] = True
            nonoverlapping.append(row)
            previous_end = row["target_index"]
    paired = [row for row in mature if "baseline_actions" in row]
    paired_metrics = {"count": len(paired), "model": {}, "historical_baseline": {}, "delta": {}}
    for branch in ("flat", "held"):
        comparable = [row for row in paired if all(
            row[key][branch].get("realized_payoff") is not None
            for key in ("actions", "baseline_actions"))]
        model = _metrics(comparable)[branch]
        baseline = _metrics(comparable, "baseline_actions")[branch]
        paired_metrics["model"][branch] = model
        paired_metrics["historical_baseline"][branch] = baseline
        paired_metrics["delta"][branch] = {
            key: model[key] - baseline[key] if comparable else None
            for key in ("mean_expected_payoff", "mean_realized_payoff")}
    buys = [row for row in mature if row["flat_action"] == "buy"]
    exits = [row for row in mature if row["held_action"] == "sell"]
    return {
        "kind": kind, "status": "ready" if mature else "no_mature_predictions",
        "horizon": horizon, "entry_lag": entry_lag, "base_amount": base,
        "fees": fees,
        "counts": {
            "total": len(rows), "available": sum(row["prediction_status"] == "available" for row in rows),
            "unavailable": sum(row["prediction_status"] == "unavailable" for row in rows),
            "invalid_predictions": sum(row["prediction_status"] == "invalid" for row in rows),
            "mature": len(mature), "nowcast": sum(row["status"] == "nowcast" for row in rows),
            "immature": sum(row["outcome_status"] == "immature" for row in rows),
            "invalid": sum(row["outcome_status"] == "invalid" for row in rows),
            "nonoverlapping_windows": len(nonoverlapping),
            "overlapping_windows": len(mature) - len(nonoverlapping), "baseline_pairs": len(paired),
            "predicted_buys": len(buys),
            "wrong_buys": sum(row["actions"]["flat"].get("realized_payoff", 0) < 0 for row in buys),
            "predicted_exits": len(exits),
            "avoided_losses": sum(row["actions"]["held"]["outcome_label"] == "avoided_loss" for row in exits),
            "missed_profits": sum(row["actions"]["held"]["outcome_label"] == "missed_profit" for row in exits),
        },
        "metrics": _metrics(mature), "nonoverlapping_metrics": _metrics(nonoverlapping),
        "paired": paired_metrics,
        "rows": rows, "assumptions": assumptions,
    }


def summarize_calendar_history(
    series, historical_forecasts, *, plan: Mapping, fee_basis, base_amount=10000,
) -> dict:
    """Same H/L NAV replay with ONE fixed fee scenario for every origin and baseline.

    Today's hypothetical fee ages are not backfilled onto historical NAV dates.
    A saved summary without samples has no invented fee-adjusted probability.
    This tests NAV-observation forecasts, not a calendar-window trading strategy.
    """
    base, fees = _base(base_amount), _calendar_fees(fee_basis)
    horizon, lag = plan.get("horizon"), plan.get("entry_lag")
    valid = (not isinstance(horizon, bool) and isinstance(horizon, int) and horizon > 0
             and not isinstance(lag, bool) and isinstance(lag, int) and lag >= 0)
    build_timing_decision({}, plan=plan, fee_basis=fee_basis)

    def decide(head):
        if (not isinstance(head, Mapping) or head.get("horizon") != horizon
                or head.get("entry_lag") != lag or head.get("status", "available") != "available"):
            return None
        return build_timing_decision({"execution_forecasts": [head]}, plan=plan, fee_basis=fee_basis)

    def realize(rows, actual):
        return _realized_rows(
            rows, actual,
            lambda row: _calendar_value(row["action"], actual, base, fees)
                        if not row["reasons"] else None,
            lambda row: _calendar_advantage(row["action"], actual, base, fees)
                        if not row["reasons"] and not row["comparison_reasons"] else None,
        )

    result = _summarize_history_engine(
        series, historical_forecasts if valid else (), horizon=horizon, entry_lag=lag, base=base,
        decide=decide, compare=lambda head, decision: _calendar_action_rows(head, base, fees, decision),
        realize=realize, kind="calendar_cash_fixed_fee_nav_history",
        fees=deepcopy(dict(fee_basis)) if fee_basis is not None else fees,
        retain_details=True,
        assumptions=[
            "这是逐时点、同H笔净值与同L延迟的历史重建，不是严格自然日择时策略，也不是当年真实发行或下单。",
            "整段历史和同原点简单基准使用本次同一组批次费用；模拟确认日只解释固定费用情景，不是各历史窗口的真实确认日或费表。",
            "动作只由当时保存的预测决定；未来净值只核对结果，保留未成熟、无效、无预测与nowcast窗口。",
            "没有保存预测样本的历史头只使用已有均值和区间，不从市场涨跌概率编造费用后胜率。",
            "每个窗口新买与现金本金B；已有份额动作本金均为基金B加现金B（共2B）。",
            "期末按计划赎回净额含赎回应收款，不承诺到账；净值内年度费用不重复扣除。",
            "模型与同H/L简单基准只在共同有效窗口比较同资本本金盈亏，不把各自动作的相反优势直接相减。",
            "重叠窗口不连乘，不重叠窗口不等于独立；零延迟早于假定可知时点的nowcast不计入可执行指标。",
        ],
    )
    result.update(plan=deepcopy(dict(plan)), fee_context_basis="fixed_across_origins_and_baseline",
                  end_value_basis="cash_and_redemption_receivable")
    if not valid:
        result["status"] = "infeasible_window"
    return result

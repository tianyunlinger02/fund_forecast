"""Open-fund arithmetic; all inputs are validated, point-in-time engine inputs."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Callable

from ..domain import ValidationError, decimal


class UnfillableSubscription(ValidationError):
    """A fully observed, valid subscription cannot purchase a minimum share."""


def money(value: Decimal, precision: int) -> Decimal:
    try:
        return value.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)
    except DecimalException as exc:
        raise ValidationError("金额超过支持的十进制精度") from exc


def application_date(
    at: datetime, rules: dict[str, Any], is_open: Callable[[date], bool]
) -> date:
    """The cutoff itself belongs to the next application day, never to today's NAV."""
    day = at.date()
    cutoff = time.fromisoformat(rules["cutoff"])
    if is_open(day) and at.time().replace(tzinfo=None) < cutoff:
        return day
    for _ in range(370):
        day += timedelta(days=1)
        if is_open(day):
            return day
    raise ValidationError("一年内没有可用申请日")


def subscription(amount: Decimal, nav: Decimal, rules: dict[str, Any]) -> dict[str, Decimal]:
    """External subscription fee: gross / (1 + rate), not gross * (1 - rate)."""
    net = money(amount / (1 + decimal(rules["subscription_rate"])), rules["cash_precision"])
    fee = amount - net
    units = (net / nav).quantize(
        Decimal(1).scaleb(-rules["share_precision"]), rounding=ROUND_DOWN
    )
    if units <= 0:
        raise UnfillableSubscription("申购金额不足以取得最小份额，整单拒绝")
    return {"gross": amount, "net": net, "fee": fee, "units": units, "price": nav}


def redemption(
    lots: list[dict[str, Any]], nav: Decimal, applied: date, rules: dict[str, Any]
) -> dict[str, Any]:
    fee = Decimal(0)
    units = Decimal(0)
    details = []
    for lot in lots:
        quantity = decimal(lot["units"])
        holding_days = (applied - date.fromisoformat(lot["confirmed_date"])).days
        if holding_days < 0:
            raise ValidationError("赎回申请不能早于份额确认日")
        rate = None
        for tier in rules["redemption_tiers"]:
            if tier["days_lt"] is None or holding_days < tier["days_lt"]:
                rate = decimal(tier["rate"])
                break
        if rate is None:
            raise ValidationError("赎回费率没有覆盖持有期")
        lot_fee = money(quantity * nav * rate, rules["cash_precision"])
        fee += lot_fee
        units += quantity
        details.append({"lot_id": lot["lot_id"], "holding_days": holding_days,
                        "rate": str(rate), "fee": str(lot_fee)})
    gross = money(units * nav, rules["cash_precision"])
    if fee > gross:
        raise ValidationError("赎回费用超过赎回金额")
    return {"gross": gross, "net": gross - fee, "fee": fee, "units": units,
            "price": nav, "fee_details": details}

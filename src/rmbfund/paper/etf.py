"""Conservative daily ETF fills: next open, observed raw bar, fixed slippage.

Availability offsets are model assumptions measured in trading days from fill date.
They must be verified for the actual product; neither volume nor daily extremes are
used to select a more favourable fill. Daily bars cannot establish queue priority.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..domain import ValidationError, decimal
from .fund import money


def execution(
    side: str, units: Decimal, bar: dict[str, Any], rules: dict[str, Any]
) -> dict[str, Decimal]:
    if bar.get("adjustment") != "raw":
        raise ValidationError("ETF 执行必须使用明确的不复权行情")
    if "volume" not in bar:
        raise ValidationError("ETF 日线缺少成交量，无法确认有交易")
    if decimal(bar["volume"]) <= 0:
        raise ValidationError("ETF 执行日无成交，不能假造成交")
    reference = decimal(bar["open"])
    slip = decimal(rules["slippage_bps"]) / 10000
    price = reference * (1 + slip if side == "buy" else 1 - slip)
    if reference <= 0 or price <= 0:
        raise ValidationError("ETF 执行价格必须为正")
    gross = money(units * price, rules["cash_precision"])
    fee = money(max(gross * decimal(rules["commission_rate"]),
                    decimal(rules["commission_min"])), rules["cash_precision"])
    net = gross + fee if side == "buy" else gross - fee
    if net < 0:
        raise ValidationError("ETF 费用超过卖出金额")
    return {"units": units, "price": price, "reference_price": reference,
            "gross": gross, "fee": fee, "net": net}

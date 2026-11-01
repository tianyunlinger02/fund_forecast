"""Descriptive paper-account evaluation; no annualized claims from a short fixture."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .domain import decimal, timestamp


def account_history(engine, account_id: str, clocks: list[str]) -> list[dict[str, Any]]:
    rows = []
    for at in sorted(set(clocks), key=timestamp):
        summary = engine.summary(account_id, at)
        rows.append({"at": at, "date": at[:10], "total_assets": summary["total_assets"],
                     "fees": summary["fees"], "complete": summary["complete"],
                     "warnings": summary["warnings"]})
    return rows


def describe(history: list[dict[str, Any]], initial_cash: str) -> dict[str, Any]:
    capital = decimal(initial_cash)
    valid = [row for row in history if row["total_assets"] is not None and row["complete"]]
    complete = len(valid) == len(history) and bool(history)
    if not complete or capital <= 0:
        return {"complete": False, "net_return": None, "max_drawdown": None,
                "samples": len(history), "warning": "估值或权益数据不完整，不计算貌似精确的收益指标。"}
    peak = capital
    max_drawdown = Decimal(0)
    for row in valid:
        value = decimal(row["total_assets"])
        peak = max(peak, value)
        if peak:
            max_drawdown = min(max_drawdown, value / peak - 1)
    return {"complete": True, "net_return": str(decimal(valid[-1]["total_assets"]) / capital - 1),
            "max_drawdown": str(max_drawdown), "samples": len(valid),
            "fees": valid[-1]["fees"],
            "warning": "描述性、未年化；短期模拟不证明预测能力或真实可成交性。"}

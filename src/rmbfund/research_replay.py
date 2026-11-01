"""Isolated historical research: real observations, explicitly assumed old visibility.

This is deliberately not PaperEngine. It never changes a Fact's availability,
creates verified profiles, or writes to a public/demo account.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any

from .domain import Fact, Product, ValidationError, canonical_json, decimal, digest, now_iso, timestamp
from .paper.fund import UnfillableSubscription, money, redemption, subscription
from .reports import csv_text


def endpoint_target(latest: Any, base: Any, previous_target: str = "cash") -> str:
    """Exact endpoint comparison retained only for the frozen historical study."""
    if previous_target not in {"long", "cash"}:
        raise ValidationError("previous_target 只能是 long 或 cash")
    last, first = decimal(latest), decimal(base)
    if last <= 0 or first <= 0:
        raise ValidationError("动量端点必须为正数")
    return "long" if last > first else "cash" if last < first else previous_target


DEFAULT_START = "2024-10-21"
DEFAULT_END = "2026-10-19"
STRATEGIES = ("cash", "buyhold", "ma20")
MONTHLY_STRATEGY = "monthly_mom60_hold30"
MONTHLY_DEFINITION = {
    "version": 1,
    "product_kind": "open_fund",
    "lookback_observations": 60,
    "min_holding_calendar_days": 30,
    "check_schedule": "first_in_window_observation_each_calendar_month",
    "momentum_basis": "unit_nav_ratio",
    "positive_target": "long",
    "negative_target": "cash",
    "zero_target": "retain_previous_target",
    "initial_target": "cash",
    "order_attempts": "monthly_check_only",
    "holding_period_origin": "confirmed_date",
    "nav_lag": "scenario_assumption",
}
WARNING = (
    "真实历史数据＋显式假设的研究回放；不是历史时点原始档案，"
    "不是已运行两年的前向模拟盘，不代表未来收益，不是荐基。"
)


@dataclass(frozen=True)
class Assumptions:
    initial_cash: str = "100000"
    ma_period: int = 20
    warmup: int = 60
    etf_buffer: str = "0.02"
    commission_rate: str = "0.0003"
    commission_min: str = "5"
    slippage_bps: str = "5"
    lot_size: int = 100
    fund_subscription_rate: str = "0.012"
    fund_daily_purchase_cap: str | None = None
    fund_purchase_min: str | None = None
    fund_redemption_tiers: list[dict[str, Any]] | None = None
    nav_lag_days: int = 1
    confirm_days: int = 1
    redemption_cash_days: int = 3

    def __post_init__(self) -> None:
        if decimal(self.initial_cash) <= 0:
            raise ValidationError("研究初始现金必须为正")
        for value in (self.ma_period, self.warmup, self.lot_size, self.nav_lag_days,
                      self.confirm_days, self.redemption_cash_days):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError("研究周期、份额和延迟必须是正整数")
        if self.warmup < self.ma_period:
            raise ValidationError("预热观测不足")
        for value in (self.etf_buffer, self.commission_rate, self.commission_min,
                      self.slippage_bps, self.fund_subscription_rate):
            if decimal(value) < 0:
                raise ValidationError("费用和缓冲不得为负")
        if self.fund_daily_purchase_cap is not None and decimal(self.fund_daily_purchase_cap) <= 0:
            raise ValidationError("场外单日申购上限必须为正")
        if self.fund_purchase_min is not None:
            if decimal(self.fund_purchase_min) <= 0:
                raise ValidationError("场外最低申购额必须为正")
            if self.fund_daily_purchase_cap is not None and decimal(self.fund_purchase_min) > decimal(self.fund_daily_purchase_cap):
                raise ValidationError("最低申购额不得超过单日上限")
        if decimal(self.slippage_bps) >= 10000 or decimal(self.etf_buffer) >= 1:
            raise ValidationError("滑点或预算缓冲过大")
        if self.redemption_cash_days < self.confirm_days or self.confirm_days < self.nav_lag_days:
            raise ValidationError("确认不得早于净值假设可知日，到账不得早于确认")
        if self.fund_redemption_tiers is not None:
            tiers = self.fund_redemption_tiers
            if not isinstance(tiers, list) or not tiers or not isinstance(tiers[-1], dict) or tiers[-1].get("days_lt") is not None:
                raise ValidationError("赎回费档必须含最终全覆盖档")
            previous = 0
            for index, tier in enumerate(tiers):
                if not isinstance(tier, dict) or not 0 <= decimal(tier.get("rate")) <= 1:
                    raise ValidationError("赎回费率必须在0和1之间")
                bound = tier.get("days_lt")
                if bound is None:
                    if index != len(tiers) - 1:
                        raise ValidationError("全覆盖费档只能位于最后")
                elif isinstance(bound, bool) or not isinstance(bound, int) or bound <= previous:
                    raise ValidationError("赎回持有期费档必须严格递增")
                else:
                    previous = bound

    def fund_rules(self) -> dict[str, Any]:
        tiers = self.fund_redemption_tiers or [
            {"days_lt": 7, "rate": "0.015"},
            {"days_lt": 365, "rate": "0.005"},
            {"days_lt": None, "rate": "0"},
        ]
        return {"subscription_rate": self.fund_subscription_rate, "cash_precision": 2,
                "share_precision": 2, "redemption_tiers": tiers}


def selected_strategies(inputs: dict[str, Any]) -> tuple[str, ...]:
    """Validate the opt-in frozen definition without changing legacy defaults."""
    if "strategies" not in inputs:
        return STRATEGIES
    selected = inputs["strategies"]
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(item, str) or item not in (*STRATEGIES, MONTHLY_STRATEGY)
                   for item in selected)
            or len(set(selected)) != len(selected)
            or not set(STRATEGIES).issubset(selected)):
        raise ValidationError("strategies 必须是非空、唯一、受支持的字符串列表，且包含三个固定对照")
    if MONTHLY_STRATEGY in selected:
        definitions = inputs.get("strategy_definitions")
        expected = {MONTHLY_STRATEGY: MONTHLY_DEFINITION}
        try:
            matches = isinstance(definitions, dict) and canonical_json(definitions) == canonical_json(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValidationError("strategy_definitions 必须严格匹配固定月度策略定义（含字段类型）")
        base = Assumptions(**inputs["assumptions"])
        warmups = [
            MONTHLY_DEFINITION["lookback_observations"] + Assumptions(
                **{**asdict(base), **scenario.get("assumptions", {})}
            ).nav_lag_days
            for item in inputs["products"]
            for scenario in item.get("scenarios", [{"id": "standard", "assumptions": {}}])
        ]
        required = max(warmups, default=MONTHLY_DEFINITION["lookback_observations"] + base.nav_lag_days)
        warmup = inputs.get("required_strategy_warmup")
        if type(warmup) is not int or warmup != required:
            raise ValidationError("required_strategy_warmup 必须严格等于60加各情景净值滞后的最大值")
    return tuple(selected)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def public_snapshot(path: Path) -> dict[str, Any]:
    """A transactionally consistent, read-only logical fingerprint of all tables."""
    if not path.is_file():
        raise ValidationError(f"真实输入库不存在: {path}")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        mode = conn.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()
        if not mode or mode[0] != "public":
            raise ValidationError("研究只接受 public 源库，不接受 demo")
        names = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        tables = {}
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            rows = sorted((list(row) for row in conn.execute(f"SELECT * FROM {quoted}")),
                          key=canonical_json)
            tables[name] = {"rows": len(rows), "hash": digest(rows)}
        products = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM products ORDER BY id")]
        facts = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM facts ORDER BY event_date,observed_at,id")]
        conn.row_factory = sqlite3.Row
        snapshots = [dict(row) for row in conn.execute("SELECT * FROM snapshots ORDER BY id")]
    return {"path": str(path.resolve()), "tables": tables, "hash": digest(tables),
            "products": products, "facts": facts, "snapshots": snapshots}


def price_inputs(product: Product, facts: list[Fact], *, start: str, end: str,
                 warmup: int = 60, nav_lag_days: int = 1) -> list[dict[str, Any]]:
    """Keep the latest acquired version, never pretend it was observed historically."""
    by_day: dict[str, Fact] = {}
    for fact in sorted(facts, key=lambda item: (item.observed_at, item.id)):
        if fact.product_id == product.id and fact.kind == ("bar" if product.kind == "etf" else "nav"):
            if fact.event_date <= end:
                by_day[fact.event_date] = fact
    previous = [day for day in sorted(by_day) if day < start]
    selected = previous[-warmup:] + [day for day in sorted(by_day) if day >= start]
    if len(previous) < warmup or not any(start <= day <= end for day in selected):
        raise ValidationError(f"{product.symbol} 缺少固定窗口或 {warmup} 个预热观测")
    rows = []
    for index, day in enumerate(selected):
        fact = by_day[day]
        next_day = selected[index + nav_lag_days] if index + nav_lag_days < len(selected) else None
        rows.append({**asdict(fact), "parent_fact_id": fact.id,
                     "input_hash": digest(asdict(fact)), "price_basis": "raw" if product.kind == "etf" else "unit_nav",
                     "assumed_known_at": (f"{day}T16:00:00+08:00" if product.kind == "etf" else
                                           f"{next_day}T09:00:00+08:00" if next_day else None),
                     "warmup": day < start})
    return rows


def _close(row: dict[str, Any], kind: str) -> Decimal:
    return decimal(row["payload"]["close" if kind == "etf" else "unit_nav"])


@dataclass
class _Account:
    cash: Decimal
    lots: list[dict[str, Any]] = field(default_factory=list)
    receivables: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    fees: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    rounding_loss: Decimal = Decimal(0)
    dividends: Decimal = Decimal(0)

    def units(self) -> Decimal:
        return sum((lot["units"] for lot in self.lots), Decimal(0))

    def available(self, index: int) -> Decimal:
        return sum((lot["units"] for lot in self.lots if lot["confirmed_index"] <= index), Decimal(0))


def _metrics(daily: list[dict[str, Any]], orders: list[dict[str, Any]], initial: Decimal,
             actions_covered: bool) -> dict[str, Any]:
    peak, drawdown = initial, Decimal(0)
    for row in daily:
        value = decimal(row["model_assets"])
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
    last = daily[-1]
    change = (decimal(last["model_assets"]) / initial - 1).quantize(Decimal("1e-12"))
    return {"model_complete": True, "verified_data_complete": False,
            "valuation_basis": "cash_distribution_total_return_under_assumptions" if actions_covered else "price_only",
            "conditional_net_return": str(change) if actions_covered else None,
            "conditional_max_drawdown": str(drawdown.quantize(Decimal("1e-12"))) if actions_covered else None,
            "price_only_change": str(change) if not actions_covered else None,
            "price_only_drawdown": str(drawdown.quantize(Decimal("1e-12"))) if not actions_covered else None,
            "verified_total_return": None, "net_return": None,
            "end_assets": last["model_assets"], "fees": last["fees"],
            "slippage_cost": last["slippage_cost"], "rounding_loss": last["rounding_loss"],
            "cash_dividends": last["cash_dividends"], "samples": len(daily),
            "actual_start": daily[0]["date"], "actual_end": daily[-1]["date"],
            "filled_orders": sum(order["state"] in {"filled", "priced_unconfirmed"} for order in orders),
            "rejected_orders": sum(order["state"] == "rejected" for order in orders),
            "exposure_fraction": str(sum(decimal(row["units"]) > 0 for row in daily) / Decimal(len(daily))),
            "ending_state": last,
            "warning": "条件费用/结算/权益覆盖下的模型估值；日收盘采样回撤，未强制清仓，未含期末退出费用。"}


def replay_one(product: Product, rows: list[dict[str, Any]], actions: list[dict[str, Any]],
               assumptions: Assumptions, *, start: str, end: str, strategy: str,
               actions_covered: bool) -> dict[str, Any]:
    """Economic close valuations are retrospective; decisions never use that day's fund NAV."""
    if strategy not in (*STRATEGIES, MONTHLY_STRATEGY) or product.kind not in {"etf", "open_fund"}:
        raise ValidationError("不支持的研究策略或产品")
    monthly = strategy == MONTHLY_STRATEGY
    if monthly and product.kind != "open_fund":
        raise ValidationError("固定月度策略仅支持 open_fund 场外基金")
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end:
        raise ValidationError("研究窗口倒置")
    dates = [row["event_date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValidationError("研究价格必须按唯一日期升序")
    if any(row["product_id"] != product.id for row in rows):
        raise ValidationError("价格与研究产品不符")
    if product.kind == "etf" and any(row["payload"].get("adjustment") != "raw" for row in rows):
        raise ValidationError("ETF 研究执行拒绝复权价格")
    period = assumptions.ma_period
    chosen = [index for index, day in enumerate(dates) if start <= day <= end]
    lookback = MONTHLY_DEFINITION["lookback_observations"]
    if monthly:
        if not chosen or chosen[0] < lookback + assumptions.nav_lag_days:
            raise ValidationError("月度策略窗口内无样本或60个观测加净值滞后的预热不足")
    elif not chosen or chosen[0] < period + assumptions.nav_lag_days - 1:
        raise ValidationError("窗口内无样本或 MA 预热不足")
    if dates[chosen[0]] != start or dates[chosen[-1]] != end:
        raise ValidationError("固定研究窗口起止日缺少价格，拒绝将截短路径标为完整")
    earliest_signal_index = chosen[0] if product.kind == "etf" else chosen[0] - assumptions.nav_lag_days
    history_start_index = earliest_signal_index - lookback if monthly else earliest_signal_index - period + 1
    signal_history_start = dates[history_start_index]
    if monthly and any(_close(row, product.kind) <= 0 for row in rows[history_start_index:chosen[-1] + 1]):
        raise ValidationError("月度策略历史及估值所用单位净值必须为正，动量基准不得为零")
    action_dates: set[tuple[str, str]] = set()
    for event in actions:
        kind = event.get("kind", "dividend")
        key = (kind, event.get("ex_date", event.get("event_date", "")))
        if key in action_dates:
            raise ValidationError("重复或冲突权益事件，拒绝重复计款")
        action_dates.add(key)
        if monthly and signal_history_start <= key[1] <= end:
            raise ValidationError("月度单位净值动量的实际历史范围含已知分红或拆分权益事件，拒绝回放")
        if kind != "dividend":
            if signal_history_start <= event.get("ex_date", event.get("event_date", "9999")) <= end:
                raise ValidationError("研究窗口存在尚不支持的拆分，拒绝继续持仓回放")
            continue
        if event["pay_date"] < start or event["record_date"] > end:
            continue
        if product.kind == "open_fund" and actions_covered:
            raise ValidationError("场外基金权益资格/申请先后尚未建模；含分红窗口拒绝完整回放")
        if not event["record_date"] <= event["ex_date"] <= event["pay_date"]:
            raise ValidationError("分红日期顺序错误")
        if event["record_date"] == event["ex_date"] and start <= event["ex_date"] <= end:
            raise ValidationError("登记/除息同日的权益先后顺序尚未建模，拒绝静默漏记")
        if decimal(event["cash_per_unit"]) < 0:
            raise ValidationError("分红不得为负")
        for key in ("record_date", "ex_date"):
            if start <= event[key] <= end and event[key] not in dates:
                raise ValidationError("权益日期缺少行情，不能默认为正常休市")
    initial = decimal(assumptions.initial_cash)
    account = _Account(initial)
    daily, orders, signals, events = [], [], [], []
    entitlements: dict[str, Decimal] = {}
    desired = False
    ever_bought = False
    checked_month: str | None = None
    rules = assumptions.fund_rules()

    def due_day(index: int) -> str | None:
        return dates[index] if index < len(dates) else None

    def settle(index: int) -> None:
        kept = []
        for item in account.receivables:
            ready = item["due_index"] <= index if "due_index" in item else item["pay_date"] <= dates[index]
            if ready:
                account.cash += item["amount"]
                events.append({**item, "date": dates[index], "type": "cash_received",
                               "source_type": item["type"]})
            else:
                kept.append(item)
        account.receivables = kept
        for order in orders:
            if order["state"] == "priced_unconfirmed" and order["confirmation_index"] <= index:
                order["state"] = "filled"
                order["confirmed_at"] = f"{dates[index]}T09:00:00+08:00"

    def record_order(side: str, index: int, known: dict[str, Any]) -> dict[str, Any]:
        day = dates[index]
        order = {"id": f"{product.symbol}-{strategy}-{len(orders) + 1:04d}",
                 "product_id": product.id, "strategy": strategy, "side": side,
                 "signal_at": f"{day}T{'16:00' if product.kind == 'etf' else '09:05'}:00+08:00",
                 "signal_price_date": known["event_date"], "signal_fact_id": known["parent_fact_id"],
                 "state": "submitted", "requested_units": None, "gross_budget": None,
                 "application_date": None, "economic_date": None, "priced_at": None,
                 "fee": "0", "reason": None}
        orders.append(order)
        return order

    for index in chosen:
        day, row = dates[index], rows[index]
        settle(index)
        for event in actions if actions_covered else []:
            if event.get("kind", "dividend") == "dividend" and event["ex_date"] == day:
                key = digest(event)
                quantity = entitlements.get(key, Decimal(0))
                value = money(quantity * decimal(event["cash_per_unit"]), 2)
                if value:
                    account.receivables.append({"amount": value, "pay_date": event["pay_date"],
                                                "type": "dividend", "units": quantity})
                    account.dividends += value
                    events.append({"date": day, "type": "dividend_receivable", "amount": str(value),
                                   "record_date": event["record_date"], "pay_date": event["pay_date"]})
        # A same-day pay date is still only cash after the receivable exists.
        settle(index)
        if product.kind == "etf" and account.pending:
            order = account.pending
            budget = decimal(order["gross_budget"]) if order["side"] == "buy" else Decimal(0)
            quantity = decimal(order["requested_units"])
            raw = decimal(row["payload"]["open"])
            slip = decimal(assumptions.slippage_bps) / 10000
            price = raw * (1 + slip if order["side"] == "buy" else 1 - slip)
            gross = money(quantity * price, 2)
            fee = money(max(gross * decimal(assumptions.commission_rate), decimal(assumptions.commission_min)), 2)
            volume = row["payload"].get("source_volume", row["payload"].get("volume"))
            rejected = "源成交量缺失或非正，参考成交无法计算" if volume is None or decimal(volume) <= 0 else None
            if order["side"] == "buy" and gross + fee > budget:
                rejected = "跳空后固定数量超过事前预算；拒单，不按开盘价反算份额"
            if order["side"] == "sell" and (quantity > account.available(index) or gross < fee):
                rejected = "可卖份额或成交金额不足"
            if rejected:
                account.cash += budget
                order.update(state="rejected", reason=rejected, attempted_date=day)
            else:
                if order["side"] == "buy":
                    account.cash += budget - gross - fee
                    account.lots.append({"lot_id": order["id"], "units": quantity,
                                         "confirmed_index": index + 1, "confirmed_date": due_day(index + 1)})
                    ever_bought = True
                else:
                    account.lots.clear()
                    account.receivables.append({"amount": gross - fee, "due_index": index + 1,
                                                "due_date": due_day(index + 1), "type": "sale"})
                account.fees += fee
                account.slippage += quantity * abs(price - raw)
                order.update(state="filled", economic_date=day, application_date=day,
                             price_fact_id=row["parent_fact_id"], price_observed_at=row["observed_at"],
                             price_available_at=row["available_at"],
                             priced_at=f"{day}T09:30:00+08:00", reference_open=str(raw), price=str(price),
                             units=str(quantity), gross=str(gross), fee=str(fee))
            account.pending = None
        side = None
        monthly_signal = None
        if monthly:
            # A partial first month is checked immediately. A blocked check has no
            # queued order: settlement and valuation continue, but attempts wait.
            if day[:7] != checked_month:
                checked_month = day[:7]
                known_index = index - assumptions.nav_lag_days
                known = rows[known_index]
                base = rows[known_index - lookback]
                visible = rows[known_index - lookback:known_index + 1]
                last, base_price = _close(known, product.kind), _close(base, product.kind)
                signal_at = f"{day}T09:05:00+08:00"
                if any(not item.get("assumed_known_at") or timestamp(item["assumed_known_at"]) > timestamp(signal_at)
                       for item in visible):
                    raise ValidationError("月度动量全部61个输入的研究可见性假设必须不晚于信号时刻")
                # The report is rounded, but the target uses exact endpoint order.
                # A tiny nonzero change must not become a zero-momentum decision.
                desired = endpoint_target(last, base_price, "long" if desired else "cash") == "long"
                try:
                    momentum = (last / base_price - 1).quantize(Decimal("1e-12"))
                except InvalidOperation as exc:
                    raise ValidationError("月度动量比率超过支持的十进制报告精度") from exc
                monthly_signal = {
                    "date": day, "at": signal_at, "known_price_date": known["event_date"],
                    "known_fact_id": known["parent_fact_id"], "known_price": str(last),
                    "lookback_start_date": base["event_date"],
                    "lookback_start_fact_id": base["parent_fact_id"],
                    "lookback_start_price": str(base_price), "lookback_observations": lookback,
                    "momentum": str(momentum), "target": "long" if desired else "cash",
                    "action": "none", "action_reason": "already_long" if desired else "already_cash",
                }
                units = account.units()
                if desired and units == 0:
                    if account.cash > 0:
                        side = "buy"
                        monthly_signal.update(action="buy", action_reason="buy_flat_known_cash")
                    else:
                        monthly_signal.update(action="blocked", action_reason="no_available_cash")
                elif not desired and units > 0:
                    if account.available(index) != units:
                        monthly_signal.update(action="blocked", action_reason="unconfirmed_units")
                    elif any(lot["confirmed_date"] is None or
                             (date.fromisoformat(day) - date.fromisoformat(lot["confirmed_date"])).days
                             < MONTHLY_DEFINITION["min_holding_calendar_days"] for lot in account.lots):
                        monthly_signal.update(action="blocked", action_reason="min_holding_not_met")
                    else:
                        side = "sell"
                        monthly_signal.update(action="sell", action_reason="sell_all_confirmed_min_holding_met")
                signals.append(monthly_signal)
        else:
            known_index = index if product.kind == "etf" else index - assumptions.nav_lag_days
            known = rows[known_index]
            visible = rows[known_index - period + 1:known_index + 1]
            last = _close(known, product.kind)
            mean = sum((_close(item, product.kind) for item in visible), Decimal(0)) / period
            if strategy == "cash":
                desired = False
            elif strategy == "buyhold":
                desired = True
            elif last != mean:
                desired = last > mean
            signal_at = f"{day}T{'16:00' if product.kind == 'etf' else '09:05'}:00+08:00"
            if any(not item.get("assumed_known_at") or timestamp(item["assumed_known_at"]) > timestamp(signal_at)
                   for item in visible):
                raise ValidationError("均线输入的研究可见性假设晚于信号时刻，拒绝提前使用")
            signals.append({"date": day, "at": signal_at, "known_price_date": known["event_date"],
                            "known_fact_id": known["parent_fact_id"], "known_price": str(last),
                            "ma20": str(mean), "target": "long" if desired else "cash"})
            side = None
            if desired and account.units() == 0 and account.cash > 0 and not (strategy == "buyhold" and ever_bought):
                side = "buy"
            elif not desired and account.available(index) > 0 and account.available(index) == account.units():
                side = "sell"
        if side and not account.pending:
            order = record_order(side, index, known)
            if side == "buy":
                budget = account.cash
                if product.kind == "open_fund" and assumptions.fund_daily_purchase_cap is not None:
                    budget = min(budget, decimal(assumptions.fund_daily_purchase_cap))
                order["gross_budget"] = str(budget)
                if product.kind == "open_fund" and assumptions.fund_purchase_min is not None and budget < decimal(assumptions.fund_purchase_min):
                    order.update(state="rejected", reason="可用预算不足最低申购额，未冻结资金且不使用未知净值")
                    if monthly_signal is not None:
                        monthly_signal.update(action="rejected", action_reason="budget_below_purchase_min")
                elif product.kind == "etf":
                    unit_budget = last * (1 + decimal(assumptions.etf_buffer)) * (1 + decimal(assumptions.commission_rate))
                    lots = ((budget - decimal(assumptions.commission_min)) / unit_budget / assumptions.lot_size).to_integral_value(rounding=ROUND_DOWN)
                    quantity = max(Decimal(0), lots * assumptions.lot_size)
                    order["requested_units"] = str(quantity)
                    if quantity == 0:
                        order.update(state="rejected", reason="事前预算不足一手")
                    else:
                        account.cash -= budget
                        account.pending = order
                else:
                    order.update(application_date=day, application_at=f"{day}T14:00:00+08:00")
                    try:
                        result = subscription(budget, _close(row, product.kind), rules)
                    except UnfillableSubscription as exc:
                        raise ValidationError("申请日未知净值导致零份额；研究模型尚不支持延迟拒单，停止而非提前退款重试") from exc
                    else:
                        account.cash -= budget
                        account.lots.append({"lot_id": order["id"], "units": result["units"],
                                             "confirmed_index": index + assumptions.confirm_days,
                                             "confirmed_date": due_day(index + assumptions.confirm_days)})
                        account.fees += result["fee"]
                        rounding = result["net"] - result["units"] * result["price"]
                        account.rounding_loss += rounding
                        order.update(state="priced_unconfirmed", economic_date=day,
                                     price_fact_id=row["parent_fact_id"], price_observed_at=row["observed_at"],
                                     price_available_at=row["available_at"],
                                     priced_at=f"{due_day(index + assumptions.nav_lag_days)}T09:00:00+08:00" if due_day(index + assumptions.nav_lag_days) else None,
                                     units=str(result["units"]), price=str(result["price"]), fee=str(result["fee"]),
                                     rounding_loss=str(rounding), confirmation_index=index + assumptions.confirm_days,
                                     confirmation_date=due_day(index + assumptions.confirm_days))
                        ever_bought = True
            else:
                quantity = account.available(index)
                order["requested_units"] = str(quantity)
                if product.kind == "etf":
                    order["gross_budget"] = "0"
                    account.pending = order
                else:
                    result = redemption(account.lots, _close(row, product.kind), date.fromisoformat(day), rules)
                    account.lots.clear()
                    account.receivables.append({"amount": result["net"],
                                                "due_index": index + assumptions.redemption_cash_days,
                                                "due_date": due_day(index + assumptions.redemption_cash_days), "type": "redemption"})
                    account.fees += result["fee"]
                    order.update(state="priced_unconfirmed", application_date=day,
                                 application_at=f"{day}T14:00:00+08:00", economic_date=day,
                                 price_fact_id=row["parent_fact_id"], price_observed_at=row["observed_at"],
                                 price_available_at=row["available_at"],
                                 priced_at=f"{due_day(index + assumptions.nav_lag_days)}T09:00:00+08:00" if due_day(index + assumptions.nav_lag_days) else None,
                                 units=str(result["units"]), price=str(result["price"]), gross=str(result["gross"]),
                                 fee=str(result["fee"]), fee_details=result["fee_details"],
                                 confirmation_index=index + assumptions.confirm_days,
                                 confirmation_date=due_day(index + assumptions.confirm_days),
                                 cash_due_date=due_day(index + assumptions.redemption_cash_days),
                                 cash_due_index=index + assumptions.redemption_cash_days)
        for event in actions if actions_covered else []:
            if event.get("kind", "dividend") == "dividend" and event["record_date"] == day:
                entitlements[digest(event)] = account.units()
        reserved = decimal(account.pending["gross_budget"]) if account.pending and account.pending["side"] == "buy" else Decimal(0)
        receivable = sum((item["amount"] for item in account.receivables), Decimal(0))
        market = account.units() * _close(row, product.kind)
        assets = account.cash + reserved + receivable + market
        if min(account.cash, reserved, receivable, account.units()) < 0:
            raise ValidationError("研究账户出现负现金/份额，停止而非掩盖")
        daily.append({"date": day, "product_id": product.id, "strategy": strategy,
                      "model_assets": str(assets), "cash_available": str(account.cash),
                      "cash_reserved": str(reserved), "cash_receivable": str(receivable),
                      "units": str(account.units()), "available_units": str(account.available(index)),
                      "unconfirmed_units": str(account.units() - account.available(index)),
                      "price": str(_close(row, product.kind)), "market_value": str(market),
                      "fees": str(account.fees), "slippage_cost": str(account.slippage),
                      "rounding_loss": str(account.rounding_loss), "cash_dividends": str(account.dividends),
                      "pending_orders": int(account.pending is not None) + sum(item["state"] == "priced_unconfirmed" for item in orders),
                      "model_complete": True})
    return {"product": asdict(product), "strategy": strategy, "daily": daily,
            "orders": orders, "signals": signals, "events": events,
            "metrics": _metrics(daily, orders, initial, actions_covered)}


def _unit_money_check(snapshots: list[dict[str, Any]], product_id: str) -> dict[str, Any]:
    records = []
    for snapshot in snapshots:
        if snapshot["product_id"] == product_id and snapshot["source"] == "eastmoney_js":
            payload = json.loads(snapshot["payload"])
            if isinstance(payload, list):
                records.extend(item["record"] for item in payload if isinstance(item, dict)
                               and item.get("function") == "Data_netWorthTrend" and isinstance(item.get("record"), dict))
    return {"records": len(records), "all_have_unitMoney": bool(records) and all("unitMoney" in item for item in records),
            "nonempty": sum(item.get("unitMoney") not in (None, "") for item in records),
            "warning": "空标记仅证明该源未编码权益，不代替基金公司公告完整性核验。"}


def evaluate_frozen(inputs: dict[str, Any]) -> dict[str, Any]:
    start, end = inputs["start"], inputs["end"]
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end or end >= now_iso()[:10]:
        raise ValidationError("冻结研究输入的实际窗口必须已经结束")
    strategies = selected_strategies(inputs)
    assumptions = Assumptions(**inputs["assumptions"])
    results, gaps = [], list(inputs.get("warnings", []))
    for item in inputs["products"]:
        product = Product(**item["product"])
        if item.get("blocked"):
            gaps.append(f"{product.symbol}: {item['blocked']}")
            continue
        for scenario in item.get("scenarios", [{"id": "standard", "assumptions": {}}]):
            scenario_id = scenario["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", scenario_id):
                raise ValidationError("费用情景标识不合法")
            model = Assumptions(**{**asdict(assumptions), **scenario.get("assumptions", {})})
            for strategy in strategies:
                result = replay_one(product, item["prices"], item["actions"], model,
                                    start=inputs["start"], end=inputs["end"], strategy=strategy,
                                    actions_covered=item["action_coverage"])
                result.update(scenario=scenario_id, assumptions=asdict(model))
                results.append(result)
    comparisons = []
    for symbol, scenario_id in sorted({(item["product"]["symbol"], item["scenario"]) for item in results}):
        selected = {item["strategy"]: item["metrics"] for item in results
                    if item["product"]["symbol"] == symbol and item["scenario"] == scenario_id}
        if selected and selected["ma20"]["conditional_net_return"] is not None:
            comparisons.append({"symbol": symbol, "scenario": scenario_id, "comparison_basis": "conditional_model_only",
                                "ma20_minus_hold_return": str(decimal(selected["ma20"]["conditional_net_return"]) - decimal(selected["buyhold"]["conditional_net_return"])),
                                "ma20_minus_hold_drawdown": str(decimal(selected["ma20"]["conditional_max_drawdown"]) - decimal(selected["buyhold"]["conditional_max_drawdown"])),
                                "extra_fees": str(decimal(selected["ma20"]["fees"]) - decimal(selected["buyhold"]["fees"]))})
    report = {"mode": "research_replay", "warning": WARNING, "start": inputs["start"], "end": inputs["end"],
              "input_hash": digest(inputs), "assumptions": inputs["assumptions"],
              "results": results, "comparisons": comparisons, "gaps": gaps,
              "forward_ready": False, "forward_blockers": [
                  "public 库暂无已核验产品规则版本、交易日历、历史/当前申赎状态和完整权益覆盖。",
                  "ETF 标准成交量单位未核验；历史研究的参考成交不能绕过 PaperEngine 门槛。",
                  "本次未创建严格 paper 委托或虚构未来收益，今天不能观察未来表现。",
              ]}
    report.update(_strategy_metadata(inputs))
    if MONTHLY_STRATEGY in strategies:
        strategy_comparisons = []
        for symbol, scenario_id in sorted({(item["product"]["symbol"], item["scenario"]) for item in results}):
            selected = {item["strategy"]: item["metrics"] for item in results
                        if item["product"]["symbol"] == symbol and item["scenario"] == scenario_id}
            if selected and selected[MONTHLY_STRATEGY]["conditional_net_return"] is not None:
                monthly_metrics, hold = selected[MONTHLY_STRATEGY], selected["buyhold"]
                strategy_comparisons.append({
                    "symbol": symbol, "scenario": scenario_id, "strategy": MONTHLY_STRATEGY,
                    "comparison_basis": "conditional_model_only",
                    "minus_hold_return": str(decimal(monthly_metrics["conditional_net_return"]) - decimal(hold["conditional_net_return"])),
                    "minus_hold_drawdown": str(decimal(monthly_metrics["conditional_max_drawdown"]) - decimal(hold["conditional_max_drawdown"])),
                    "extra_fees": str(decimal(monthly_metrics["fees"]) - decimal(hold["fees"])),
                })
        report["strategy_comparisons"] = strategy_comparisons
    return report


def _strategy_metadata(inputs: dict[str, Any]) -> dict[str, Any]:
    """An explicit opt-in whitelist shared by reports and both manifest paths."""
    if "strategies" not in inputs:
        return {}
    return {key: inputs[key] for key in (
        "strategies", "strategy_definitions", "required_strategy_warmup", "input_preparation",
    ) if key in inputs}


def _save_results(output: Path, report: dict[str, Any]) -> None:
    _write_json(output / "report.json", report)
    for result in report["results"]:
        symbol = result["product"]["symbol"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", symbol):
            raise ValidationError("研究产品代码不适合作为安全导出目录")
        directory = output / f"{symbol}-{result['scenario']}-{result['strategy']}"
        directory.mkdir()
        for name in ("daily", "orders", "signals", "events"):
            (directory / f"{name}.csv").write_text(csv_text(result[name]), encoding="utf-8")
    lines = ["# 真实数据历史研究回放", "", WARNING, "",
             f"窗口：{report['start']} 至 {report['end']}；默认每条路径初始 {decimal(report['assumptions']['initial_cash']):,.2f} 元（各情景配置见JSON）；未强制期末清仓。", "",
             "| 标的 | 费用情景 | 方法 | 模型期末资产 | 条件模型收益 | 日收盘回撤 | 显式费用 | 成交笔数 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for result in report["results"]:
        metric = result["metrics"]
        gain = metric["conditional_net_return"]
        drawdown = metric["conditional_max_drawdown"]
        lines.append(f"| {result['product']['symbol']} | {result['scenario']} | {result['strategy']} | {decimal(metric['end_assets']):,.2f} |"
                     f"{decimal(gain):.2%} | {decimal(drawdown):.2%} | {decimal(metric['fees']):,.2f} | {metric['filled_orders']} |"
                     if gain is not None else
                     f"| {result['product']['symbol']} | {result['scenario']} | {result['strategy']} | {decimal(metric['end_assets']):,.2f} |权益不足，留空 | 留空 | {metric['fees']} | {metric['filled_orders']} |")
    if MONTHLY_STRATEGY in report.get("strategies", []):
        lines += ["", "月度方法仅用于场外基金：每个日历月首个窗口内观测日（含首个不完整月份）09:05检查，",
                  "以情景滞后后的单位净值相对60个观测前的变化定目标；正值持有、负值现金、零值保留上次目标（初始现金）。",
                  "只在检查日尝试申赎；赎回须全部份额已确认且每笔自确认日起满30自然日，阻塞后等下月，不排队或月中重试。",
                  "单位净值变化不是总收益；每日确认、到账和回看估值继续，动量依赖范围内已知分红/拆分会拒绝回放。"]
    lines += ["", "上述不是可核验的实际投资实绩；条件模型收益计入源表分红与假设费用，不含期末退出成本。",
              "基金日级估值使用经济日 NAV 回看：国内样本决策滞后1个观测日，QDII情景滞后2个观测日；未确认份额不能卖、应收款不能提前再投。",
              "ETF 使用原价信号，除息可触发 MA；没有容量、开盘盘口或成交可得性证明。", "", "## 数据与假设缺口", ""]
    lines += [f"- {gap}" for gap in report["gaps"] + report["forward_blockers"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research_replay(public_db: Path, output_dir: Path, *, start: str = DEFAULT_START,
                        end: str = DEFAULT_END, frozen: Path | None = None,
                        include_us_example: bool = False) -> dict[str, Any]:
    """Collect once or replay a frozen input; never silently overwrite an old run."""
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end or end >= now_iso()[:10]:
        raise ValidationError("研究区间必须已结束，且开始日不得晚于结束日")
    if output_dir.exists():
        raise ValidationError("研究输出目录已存在；请换新目录保留旧证据")
    if frozen is not None:
        if not frozen.is_file() or frozen.stat().st_size > 32 * 1024 * 1024:
            raise ValidationError("冻结研究输入不存在或超过 32MiB")
        inputs = json.loads(frozen.read_text(encoding="utf-8"))
        if inputs.get("mode") != "research_replay":
            raise ValidationError("不是 research_replay 冻结输入")
        report = evaluate_frozen(inputs)
        output_dir.mkdir(parents=True)
        _write_json(output_dir / "inputs.json", inputs)
        _save_results(output_dir, report)
        _write_json(output_dir / "manifest.json", {"mode": "research_replay", "warning": WARNING,
                    "created_at": now_iso(), "replayed_from": str(frozen.resolve()),
                    "start": inputs["start"], "end": inputs["end"], "assumptions": inputs["assumptions"],
                    "sources": inputs.get("source_evidence", []), "source_public_hash": inputs.get("source_public_hash"),
                    "input_hash": digest(inputs), "economic_result_hash": digest(report),
                    **_strategy_metadata(inputs)})
        return report
    before = public_snapshot(public_db)
    assumptions = Assumptions()
    products = [Product(**value) for value in before["products"]]
    fund = next((item for item in products if item.symbol == "000051" and item.kind == "open_fund" and not item.is_demo), None)
    etf = next((item for item in products if item.symbol == "510300" and item.kind == "etf" and not item.is_demo), None)
    if not fund or not etf:
        raise ValidationError("真实库需含 000051 场外基金和 510300 ETF；不自动添加或改类型")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "public-before.json", {key: before[key] for key in ("path", "tables", "hash")})
    from .research_sources import acquire_sources
    sources = acquire_sources(etf, output_dir, end=end)
    facts = [Fact(**item) for item in before["facts"]]
    fund_check = _unit_money_check(before["snapshots"], fund.id)
    source_dir = output_dir / "public-nav-snapshot"
    source_dir.mkdir()
    _write_json(source_dir / "snapshots.json", before["snapshots"])
    _write_json(source_dir / "unit-money-check.json", fund_check)
    items = []
    for product, values in ((fund, facts), (etf, sources["etf_facts"])):
        coverage = sources["action_coverage"].get(product.symbol, False)
        if product.kind == "open_fund":
            coverage = coverage and fund_check["all_have_unitMoney"] and fund_check["nonempty"] == 0
        rows = price_inputs(product, values, start=start, end=end, warmup=assumptions.warmup,
                            nav_lag_days=assumptions.nav_lag_days)
        crosscheck = sources.get("crosscheck", {})
        blocked = ("原价交叉核对存在冲突，拒绝静默拼源" if product.kind == "etf" and crosscheck.get("mismatches") else None)
        items.append({"product": asdict(product), "prices": rows,
                      "actions": sources["actions"].get(product.symbol, []),
                      "action_coverage": bool(coverage), "blocked": blocked})
    gaps = sources["warnings"] + [
        "源表权益覆盖是一项公开二手数据假设，非基金公司全部公告逐笔核验。",
        "日历使用各产品有效报价观测日序；未核验历史开放状态，缺失日不插值、不伪造休市。",
        "000051 NAV(T) 假设下一观测日09:00可知；申请日后1日确认、赎回后3日到账均为研究假设。",
        "000051 当前官网费率1.20%作历史费用情景，非实际渠道历史费用；一年以365自然日近似。",
        "ETF 佣金万三/最低5元、单边滑点5bp、2%预算缓冲、T+1可卖及卖出款次日可用均为假设。",
        "期末盯市不强平；现金零利息；现金分红不自动再投资；未确认份额按经济日回看估值。",
        "000051与510300均为沪深300相关暴露，不应将这两个独立账户当作分散组合。",
    ]
    # Fees on a distributor screen are channel-specific; a one-tenth scenario
    # is a sensitivity, never an assertion about the user's Alipay fee.
    if include_us_example:
        items[0]["scenarios"] = [
            {"id": "current_standard_fee", "assumptions": {}},
            {"id": "hyp_discount_0p12", "assumptions": {"fund_subscription_rate": "0.0012"}},
        ]
        from .research_qdii import acquire_qdii_sample
        qdii = acquire_qdii_sample(output_dir, start=start, end=end)
        model = Assumptions(nav_lag_days=2, confirm_days=3, redemption_cash_days=7,
                            fund_subscription_rate="0.013", fund_redemption_tiers=[
                                {"days_lt": 7, "rate": "0.015"}, {"days_lt": 365, "rate": "0.005"},
                                {"days_lt": 730, "rate": "0.003"}, {"days_lt": None, "rate": "0"}])
        prices = price_inputs(qdii["product"], qdii["facts"], start=start, end=end,
                              warmup=model.warmup, nav_lag_days=model.nav_lag_days)
        items.append({"product": asdict(qdii["product"]), "prices": prices, "actions": qdii["actions"],
                      "action_coverage": qdii["action_coverage"], "blocked": None,
                      "scenarios": [
                          {"id": "hyp_standard_1p30_lag2_3_7", "assumptions": asdict(model)},
                          {"id": "hyp_discount_0p13_lag2_3_7", "assumptions": {
                              **asdict(model), "fund_subscription_rate": "0.0013"}},
                      ]})
        gaps.extend(qdii["warnings"])
        sources["evidence"].extend(qdii["evidence"])
    gaps.append("场外折扣情景仅用于费用敏感性；0.12%/0.13%不是已核验支付宝费率，不能选结果较好的情景当事实。")
    inputs = {"mode": "research_replay", "warning": WARNING, "start": start, "end": end,
              "assumptions": asdict(assumptions), "products": items, "warnings": gaps,
              "source_evidence": sources["evidence"], "crosscheck": sources.get("crosscheck", {}),
              "source_public_hash": before["hash"], "unit_money_check": fund_check}
    _write_json(output_dir / "inputs.json", inputs)
    report = evaluate_frozen(inputs)
    economic_result_hash = digest(report)
    after = public_snapshot(public_db)
    _write_json(output_dir / "public-after.json", {key: after[key] for key in ("path", "tables", "hash")})
    if before["hash"] != after["hash"]:
        raise ValidationError("public 源库在研究期间发生变化；保留输入证据，拒绝宣称只读一致验收")
    report["source_public_unchanged"] = True
    _save_results(output_dir, report)
    _write_json(output_dir / "manifest.json", {"mode": "research_replay", "warning": WARNING,
                "created_at": now_iso(), "start": start, "end": end, "input_hash": digest(inputs),
                "source_public_hash_before": before["hash"], "source_public_hash_after": after["hash"],
                "source_public_unchanged": True, "economic_result_hash": economic_result_hash,
                "sources": sources["evidence"], "assumptions": asdict(assumptions),
                **_strategy_metadata(inputs)})
    return report

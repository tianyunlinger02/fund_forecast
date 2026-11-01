"""Append-only, point-in-time paper accounting; no live trading interface.

Verified profiles certify this model's conventions, not universal market rules:
all *_days offsets count open calendar days from the application/fill date;
redemption_cash_days is measured from application, not confirmation. The cash
settlement offset cannot precede confirmation. Fund holding periods are natural
days from confirmation to redemption application. Unsupported conventions must
not be enabled by marking an incompatible public profile verified.

Economic dates describe ownership; recorded_at describes when the simulator
actually learned/booked an event. Spending and historical summaries use the
latter, so a late NAV/bar never makes funds retroactively spendable.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from ..domain import (
    Product, Profile, ValidationError, canonical_json, decimal, digest, now_iso, timestamp,
)
from . import etf, fund

ZERO = Decimal(0)
TERMINAL = {"settled", "rejected"}
CASH_FIELDS = ("available", "reserved", "receivable", "inflight")
COMMON_RULES = {
    "calendar_id", "events_complete_through", "status_max_age_days",
    "cash_precision", "share_precision",
}
FUND_RULES = {
    "cutoff", "subscription_rate", "subscription_min", "confirm_days",
    "redemption_confirm_days", "redemption_cash_days", "redemption_tiers",
}
ETF_RULES = {
    "lot_size", "commission_rate", "commission_min", "slippage_bps",
    "buy_available_days", "sale_cash_days",
}


def text(value: Decimal) -> str:
    return format(value, "f")


def safe(value: Any) -> Any:
    """Only JSON primitives leave the API or enter event payloads."""
    if isinstance(value, Decimal):
        return text(value)
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


class PaperEngine:
    def __init__(self, store: Any) -> None:
        self.store = store
        if store.mode not in {"demo", "public"}:
            raise ValidationError("未知账本模式")

    def create_account(
        self, name: str, initial_cash: Any, at: str | datetime, account_id: str | None = None
    ) -> str:
        point = timestamp(at)
        self._write_time(point)
        cash = decimal(initial_cash)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValidationError("账户名不能为空且不能超过 200 字")
        if cash < 0 or fund.money(cash, 2) != cash:
            raise ValidationError("初始资金须为非负人民币分金额")
        account_id = account_id or "acct_" + uuid.uuid4().hex
        if not isinstance(account_id, str) or not account_id or len(account_id) > 100:
            raise ValidationError("无效账户标识")
        with self.store.transaction():
            if self.store.conn.execute("SELECT 1 FROM accounts WHERE id=?", (account_id,)).fetchone():
                raise ValidationError("账户标识已存在")
            self.store.conn.execute(
                "INSERT INTO accounts VALUES(?,?,?,?,?,?,?)",
                (account_id, name.strip(), point.isoformat(), text(cash), self.store.mode,
                 point.isoformat(), canonical_json({"currency": "CNY"})),
            )
            self._emit(account_id, "account:" + account_id, "initial_cash", point.date(), point,
                       {"cash": {"available": cash}})
        return account_id

    def accounts(self) -> list[dict[str, Any]]:
        return [self._row(row) for row in self.store.conn.execute(
            "SELECT * FROM accounts WHERE mode=? ORDER BY created_at,id", (self.store.mode,)
        )]

    def submit(
        self, account_id: str, product_id: str, side: str, at: str | datetime,
        amount: Any = None, units: Any = None, budget: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        point = timestamp(at)
        self._write_time(point)
        if side not in {"buy", "sell"}:
            raise ValidationError("side 仅支持 buy / sell")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not idempotency_key or len(idempotency_key) > 200
        ):
            raise ValidationError("幂等键须为 1 至 200 字符")
        request = {"product_id": product_id, "side": side, "at": point.isoformat(),
                   "amount": self._input_number(amount), "units": self._input_number(units),
                   "budget": self._input_number(budget)}
        order_id = ("ord_" + digest([account_id, idempotency_key])[:40]
                    if idempotency_key else "ord_" + uuid.uuid4().hex)
        with self.store.transaction():
            self._account(account_id)
            existing = self.store.conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            if existing:
                order = self._row(existing)
                if order["payload"]["request"] != request:
                    raise ValidationError("同一幂等键不能用于不同订单")
                return self._public_order(order)
            self._advance(account_id, point)
            product = self._product(product_id)
            profile = self._profile(product, point)
            rules = profile.rules
            self._coverage(rules, point.date())
            self._status(product_id, side, point.date(), point, rules)
            def is_open(day: date) -> bool:
                return self._is_open(rules["calendar_id"], day, point)

            if product.kind == "open_fund":
                applied = fund.application_date(point, rules, is_open)
            else:
                applied = self._shift(point.date(), 1, rules, point)
            payload: dict[str, Any] = {
                "request": request, "kind": product.kind, "profile": asdict(profile),
                "application_date": applied.isoformat(), "idempotency_key": idempotency_key,
                "model": "daily-open/application-T-plus-cash-v1",
            }
            state = self._state(account_id, point)
            changes: dict[str, Any] = {}
            if side == "buy":
                if product.kind == "open_fund":
                    if amount is None or units is not None or budget is not None:
                        raise ValidationError("场外申购仅接受 gross amount")
                    reserve = self._positive(amount, rules["cash_precision"], "申购金额")
                    if reserve < decimal(rules["subscription_min"]):
                        raise ValidationError("申购金额低于档案最低申购额")
                    payload["amount"] = reserve
                else:
                    if amount is not None or units is None or budget is None:
                        raise ValidationError("ETF 买入必须提供 units 和冻结 budget")
                    quantity = self._positive(units, rules["share_precision"], "买入份额")
                    if quantity % decimal(rules["lot_size"]) != 0:
                        raise ValidationError("ETF 买入份额不是一手的整数倍")
                    reserve = self._positive(budget, rules["cash_precision"], "冻结预算")
                    payload.update(units=quantity, budget=reserve)
                if state["available"] < reserve:
                    raise ValidationError("可用现金不足；冻结资金和应收款不能再次使用")
                payload["reserved_cash"] = reserve
                changes["cash"] = {"available": -reserve, "reserved": reserve}
            else:
                if units is None or amount is not None or budget is not None:
                    raise ValidationError("卖出仅接受 units")
                quantity = self._positive(units, rules["share_precision"], "卖出份额")
                allocated = self._allocate(state, product_id, quantity, point)
                if product.kind == "etf" and quantity % decimal(rules["lot_size"]) != 0:
                    available = sum((lot["units"] - lot["reserved"] for lot in state["lots"].values()
                                     if lot["product_id"] == product_id
                                     and lot["available_date"] <= point.date().isoformat()), ZERO)
                    if quantity != available:
                        raise ValidationError("ETF 零股只支持一次性卖出全部可卖余额")
                payload.update(units=quantity, reserved_lots=allocated)
                changes["lot_reservations"] = [
                    {"lot_id": lot["lot_id"], "units": lot["units"]} for lot in allocated
                ]
            self.store.conn.execute(
                "INSERT INTO orders VALUES(?,?,?,?,?,?,?,?)",
                (order_id, account_id, product_id, side, "pending", point.isoformat(),
                 point.isoformat(), canonical_json(safe(payload))),
            )
            self._emit(account_id, order_id + ":reserve", "order_reserved", point.date(), point,
                       changes, order_id, product_id)
            self._state(account_id, point)
            return self._get_order(order_id)

    def advance(self, account_id: str, at: str | datetime) -> list[dict[str, Any]]:
        point = timestamp(at)
        self._write_time(point)
        with self.store.transaction():
            last = self.store.conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM ledger_events WHERE account_id=?", (account_id,)
            ).fetchone()[0]
            self._advance(account_id, point)
            return [self._row(row) for row in self.store.conn.execute(
                "SELECT * FROM ledger_events WHERE account_id=? AND id>? ORDER BY id",
                (account_id, last),
            )]

    def orders(self, account_id: str) -> list[dict[str, Any]]:
        self._account(account_id)
        return [self._public_order(self._row(row)) for row in self.store.conn.execute(
            "SELECT * FROM orders WHERE account_id=? ORDER BY submitted_at,id", (account_id,)
        )]

    def ledger(self, account_id: str) -> list[dict[str, Any]]:
        self._account(account_id)
        return [self._row(row) for row in self.store.conn.execute(
            "SELECT * FROM ledger_events WHERE account_id=? ORDER BY id", (account_id,)
        )]

    def summary(self, account_id: str, at: str | datetime) -> dict[str, Any]:
        point = timestamp(at)
        account = self._account(account_id)
        if point < timestamp(account["created_at"]):
            raise ValidationError("查询时点早于账户创建")
        state = self._state(account_id, point)
        warnings: list[str] = []
        positions = []
        total_market = ZERO
        missing_price = False
        priced_redemptions: dict[str, str] = {}
        for event in self.store.conn.execute(
            "SELECT order_id,product_id,event_type,recorded_at FROM ledger_events WHERE account_id=? "
            "AND event_type IN ('redemption_priced','redemption_confirmed') ORDER BY id", (account_id,)
        ):
            if timestamp(event["recorded_at"]) > point:
                continue
            if event["event_type"] == "redemption_priced":
                priced_redemptions[event["order_id"]] = event["product_id"]
            else:
                priced_redemptions.pop(event["order_id"], None)
        locked_products = set(priced_redemptions.values())
        grouped: dict[str, list[dict[str, Any]]] = {}
        for lot in state["lots"].values():
            if lot["units"] > 0:
                grouped.setdefault(lot["product_id"], []).append(lot)
        for product_id, lots in sorted(grouped.items()):
            product = self._product(product_id)
            quantity = sum((lot["units"] for lot in lots), ZERO)
            reserved = sum((lot["reserved"] for lot in lots), ZERO)
            available = sum((lot["units"] - lot["reserved"] for lot in lots
                             if lot["available_date"] <= point.date().isoformat()), ZERO)
            kind, key = ("bar", "close") if product.kind == "etf" else ("nav", "unit_nav")
            price_fact = self.store.latest_fact(product_id, kind, point.isoformat())
            price = decimal(price_fact.payload[key]) if price_fact else None
            if kind == "bar" and price_fact and price_fact.payload.get("adjustment") != "raw":
                price = None
            if product_id in locked_products:
                # The operational lots stay reserved until confirmation, but their
                # redemption proceeds are already locked. Do not reprice those
                # units at a later NAV or invent a second receivable/inflight asset.
                price = None
                warnings.append(f"{product_id}: 赎回已锁价待确认，不能按最新净值重估；暂不提供完整资产合计")
            for corporate_kind in ("split", "dividend"):
                for identity, action in self._corporate_facts(product_id, corporate_kind, point).items():
                    boundary = action.event_date if corporate_kind == "split" else action.payload["ex_date"]
                    if boundary > point.date().isoformat():
                        continue
                    if price_fact and price_fact.event_date < boundary:
                        price = None
                        warnings.append(f"{product_id}: 除权/除息前价格不能与权益变动后的账本混合估值")
                    if corporate_kind == "split" and any(lot["confirmed_date"] < boundary for lot in lots):
                        key = f"{account_id}:split:{product_id}:{identity}"
                        applied = self.store.conn.execute(
                            "SELECT recorded_at,payload FROM ledger_events WHERE event_key=?", (key,)
                        ).fetchone()
                        if (applied is None or timestamp(applied["recorded_at"]) > point
                                or json.loads(applied["payload"])["source_payload"] != action.payload):
                            price = None
            if price is None:
                warnings.append(f"{product_id}: 缺少查询时点已观察的未复权估值数据")
                missing_price = True
                value = None
            else:
                value = quantity * price
                total_market += value
            try:
                profile = self._profile(product, point)
                self._coverage(profile.rules, point.date())
                expected = (point.date() if self._is_open(profile.rules["calendar_id"], point.date(), point)
                            else self._previous_open(point.date(), profile.rules, point))
                if price_fact and price_fact.event_date < expected.isoformat():
                    warnings.append(f"{product_id}: 估值仍为 {price_fact.event_date}，缺少查询日已观察估值")
            except ValidationError as exc:
                warnings.append(f"{product_id}: {exc}")
            positions.append({"product_id": product_id, "units": text(quantity),
                              "available_units": text(available), "reserved_units": text(reserved),
                              "price": text(price) if price is not None else None,
                              "price_date": price_fact.event_date if price_fact else None,
                              "market_value": text(value) if value is not None else None})
        if point > timestamp(account["last_advanced_at"]) and (
            positions or any(state[key] for key in ("reserved", "receivable", "inflight"))
        ):
            warnings.append("账本尚未推进至查询时点；本次只读查询不会自动确认、到账或处理权益")
        historical_products = {row["product_id"] for row in self.store.conn.execute(
            "SELECT product_id,submitted_at FROM orders WHERE account_id=?", (account_id,)
        ) if timestamp(row["submitted_at"]) <= point}
        unsettled_corporate = False
        for product_id in sorted(historical_products):
            corporate_warnings = self._corporate_warnings(account_id, product_id, point)
            unsettled_corporate |= bool(corporate_warnings)
            warnings.extend(corporate_warnings)
        if self.store.mode == "public" and point > timestamp(now_iso()) + timedelta(minutes=1):
            warnings.append("未来时点仅显示已记账状态，不是预测或完整未来资产")
        total = (None if missing_price or unsettled_corporate else
                 total_market + sum((state[key] for key in CASH_FIELDS), ZERO))
        return {"account_id": account_id, "at": point.isoformat(),
                "cash_available": text(state["available"]), "cash_reserved": text(state["reserved"]),
                "cash_receivable": text(state["receivable"]), "inflight": text(state["inflight"]),
                "market_value": None if missing_price else text(total_market),
                "total_assets": None if total is None else text(total), "fees": text(state["fees"]),
                "positions": positions, "complete": not warnings, "warnings": warnings,
                "last_advanced_at": account["last_advanced_at"], "mode": self.store.mode}

    def _advance(self, account_id: str, point: datetime) -> None:
        account = self._account(account_id)
        if point < timestamp(account["last_advanced_at"]):
            raise ValidationError("账本时间不可回拨；历史 summary 仅可只读查询")
        rows = [self._row(row) for row in self.store.conn.execute(
            "SELECT * FROM orders WHERE account_id=? ORDER BY submitted_at,id", (account_id,)
        )]
        products = {row["product_id"] for row in rows}
        product_rules: dict[str, dict[str, Any]] = {}
        for product_id in products:
            self._product(product_id)
            # Existing obligations keep the profile accepted at submission. A later
            # expired/unverified profile can block new orders, not rewrite old ones.
            rules = dict(next(row["payload"]["profile"]["rules"] for row in reversed(rows)
                              if row["product_id"] == product_id))
            current = self.store.profiles(product_id, point.isoformat())
            if current and current[0].verified is True:
                profile = self._profile(self._product(product_id), point)
                rules["events_complete_through"] = max(
                    rules["events_complete_through"], profile.rules["events_complete_through"],
                    key=date.fromisoformat,
                )
            self._coverage(rules, point.date())
            product_rules[product_id] = rules
            self._splits(account_id, product_id, rules, point, rows)
        for order in rows:
            if order["state"] in TERMINAL:
                continue
            self._advance_order(order, point)
        for product_id in products:
            self._dividends(account_id, product_id, product_rules[product_id], point)
        self._state(account_id, point)
        self.store.conn.execute("UPDATE accounts SET last_advanced_at=? WHERE id=?",
                                (point.isoformat(), account_id))

    def _advance_order(self, order: dict[str, Any], point: datetime) -> None:
        payload = order["payload"]
        rules = payload["profile"]["rules"]
        day = date.fromisoformat(payload["application_date"])
        if day > point.date():
            return
        if order["state"] in {"pending", "awaiting_nav", "awaiting_bar"}:
            if not self._is_open(rules["calendar_id"], day, point):
                raise ValidationError("已接受订单的申请/成交日日历被更正为休市，需人工核验")
            if not self._status(order["product_id"], order["side"], day, point, rules, reject_closed=True):
                self._reject(order, point, "申请/成交日状态关闭")
                return
            kind = "bar" if payload["kind"] == "etf" else "nav"
            fact = self.store.latest_fact(order["product_id"], kind, point.isoformat(), day.isoformat())
            if fact is None:
                self._update(order, "awaiting_bar" if kind == "bar" else "awaiting_nav", point)
                return
            payload["pricing_fact_id"] = fact.id
            payload["pricing_available_at"] = fact.available_at
            if kind == "bar":
                result = etf.execution(order["side"], decimal(payload["units"]), fact.payload, rules)
                if order["side"] == "buy" and result["net"] > decimal(payload["reserved_cash"]):
                    self._reject(order, point, "开盘价、滑点与佣金合计超过冻结预算，整单拒绝")
                    return
                payload["pricing"] = safe(result)
                self._etf_fill(order, day, point)
            else:
                nav = decimal(fact.payload["unit_nav"])
                if order["side"] == "buy":
                    try:
                        result = fund.subscription(decimal(payload["amount"]), nav, rules)
                    except fund.UnfillableSubscription as exc:
                        self._reject(order, point, str(exc))
                        return
                else:
                    result = fund.redemption(payload["reserved_lots"], nav, day, rules)
                payload["pricing"] = safe(result)
                offset = rules["confirm_days"] if order["side"] == "buy" else rules["redemption_confirm_days"]
                payload["confirmation_date"] = self._shift(day, offset, rules, point).isoformat()
                if order["side"] == "buy":
                    self._emit_order(order, "priced", "fund_priced", day, point, {
                        "cash": {"reserved": -decimal(payload["reserved_cash"]),
                                 "inflight": result["net"]}, "fees": result["fee"],
                        "pricing": result, "fact_id": fact.id,
                    })
                else:
                    payload["cash_date"] = self._shift(day, rules["redemption_cash_days"], rules, point).isoformat()
                    self._emit_order(order, "priced", "redemption_priced", day, point,
                                     {"pricing": result, "fact_id": fact.id})
                self._update(order, "pending_confirmation", point)
        if order["state"] == "pending_confirmation" and payload["confirmation_date"] <= point.date().isoformat():
            confirmed = date.fromisoformat(payload["confirmation_date"])
            result = payload["pricing"]
            if order["side"] == "buy":
                self._emit_order(order, "confirmed", "fund_confirmed", confirmed, point, {
                    "cash": {"inflight": -decimal(result["net"])},
                    "lots_add": [self._new_lot(order, result["units"], confirmed, confirmed)],
                })
                self._update(order, "settled", point)
            else:
                self._emit_order(order, "confirmed", "redemption_confirmed", confirmed, point, {
                    "cash": {"receivable": decimal(result["net"])},
                    "fees": decimal(result["fee"]), "lots_consume": payload["reserved_lots"],
                })
                self._update(order, "pending_cash", point)
        if order["state"] == "pending_cash" and payload["cash_date"] <= point.date().isoformat():
            net = decimal(payload["pricing"]["net"])
            self._emit_order(order, "cash", "sale_cash_received", date.fromisoformat(payload["cash_date"]),
                             point, {"cash": {"receivable": -net, "available": net}})
            self._update(order, "settled", point)

    def _etf_fill(self, order: dict[str, Any], day: date, point: datetime) -> None:
        payload, side = order["payload"], order["side"]
        rules, result = payload["profile"]["rules"], payload["pricing"]
        net = decimal(result["net"])
        changes: dict[str, Any] = {"fees": decimal(result["fee"]), "pricing": result,
                                   "fact_id": payload["pricing_fact_id"]}
        if side == "buy":
            available = self._shift(day, rules["buy_available_days"], rules, point)
            payload["available_date"] = available.isoformat()
            budget = decimal(payload["reserved_cash"])
            changes.update(cash={"reserved": -budget, "available": budget - net},
                           lots_add=[self._new_lot(order, result["units"], day, available)])
            state = "settled"
        else:
            payload["cash_date"] = self._shift(day, rules["sale_cash_days"], rules, point).isoformat()
            changes.update(cash={"receivable": net}, lots_consume=payload["reserved_lots"])
            state = "pending_cash"
        self._emit_order(order, "filled", "etf_filled", day, point, changes)
        self._update(order, state, point)

    def _reject(self, order: dict[str, Any], point: datetime, reason: str) -> None:
        payload = order["payload"]
        changes: dict[str, Any] = {"reason": reason}
        if order["side"] == "buy":
            reserve = decimal(payload["reserved_cash"])
            changes["cash"] = {"reserved": -reserve, "available": reserve}
        else:
            changes["lot_reservations"] = [{"lot_id": lot["lot_id"], "units": -decimal(lot["units"])}
                                            for lot in payload["reserved_lots"]]
        payload["rejection_reason"] = reason
        self._emit_order(order, "rejected", "order_rejected", point.date(), point, changes)
        self._update(order, "rejected", point)

    def _splits(
        self, account_id: str, product_id: str, rules: dict[str, Any], point: datetime,
        orders: list[dict[str, Any]],
    ) -> None:
        facts = self._corporate_facts(product_id, "split", point)
        for identity, fact in facts.items():
            key = f"{account_id}:split:{product_id}:{identity}"
            if self._already_corporate(key, fact.payload):
                continue
            state = self._state(account_id, point)
            lots = [lot for lot in state["lots"].values() if lot["product_id"] == product_id
                    and lot["units"] > 0 and lot["confirmed_date"] < fact.event_date]
            for order in orders:
                if order["product_id"] != product_id:
                    continue
                order_payload = order["payload"]
                applied = order_payload["application_date"]
                confirmed = order_payload.get("confirmation_date")
                if (order["state"] not in TERMINAL and applied <= fact.event_date) or (
                    order["state"] == "settled" and order["side"] == "buy"
                    and applied <= fact.event_date and confirmed and confirmed >= fact.event_date
                ):
                    raise ValidationError(
                        "拆分跨越订单申请/确认区间；已结订单也缺少跨日调整规则，本模型阻断处理"
                    )
            # A correction observed after post-split sales cannot be repaired by multiplying remaining lots.
            if any(event["product_id"] == product_id and event["event_type"] in {
                "etf_filled", "redemption_confirmed"
            } and event["economic_date"] >= fact.event_date and event["payload"].get("lots_consume")
                   for event in self.ledger(account_id)):
                raise ValidationError("迟到的拆分影响已记卖出，禁止静默重写账本")
            ratio = decimal(fact.payload["ratio"])
            updates = []
            for lot in lots:
                quantity = lot["units"] * ratio
                if lot["reserved"] or quantity != quantity.quantize(Decimal(1).scaleb(-rules["share_precision"])):
                    raise ValidationError("拆分涉及冻结份额或零碎份额补偿，缺少规则，已阻断")
                updates.append({"lot_id": lot["lot_id"], "ratio": ratio})
            self._emit(account_id, key, "split_applied", date.fromisoformat(fact.event_date), point,
                       {"lot_multipliers": updates, "source_payload": fact.payload, "fact_id": fact.id},
                       product_id=product_id)

    def _dividends(
        self, account_id: str, product_id: str, rules: dict[str, Any], point: datetime
    ) -> None:
        for identity, fact in self._corporate_facts(product_id, "dividend", point).items():
            data = fact.payload
            if data.get("method", "cash") != "cash" or data.get("reinvest", False):
                raise ValidationError("仅支持现金分红；红利再投资需要独立份额确认规则")
            if data["ex_date"] > point.date().isoformat():
                continue
            if data["record_date"] >= point.date().isoformat():
                # record_date == ex_date is legal input, but the complete record
                # day holdings are not final during that day. Never pay early.
                continue
            key = f"{account_id}:dividend:{product_id}:{identity}"
            existed = self._already_corporate(key, data)
            if not existed:
                for order in self.orders(account_id):
                    if order["product_id"] != product_id or order["state"] in TERMINAL | {"pending_cash"}:
                        continue
                    payload = order["payload"]
                    ownership_day = payload.get("confirmation_date", payload["application_date"])
                    if ownership_day <= data["record_date"]:
                        raise ValidationError("登记日之前仍有未确认订单，无法可靠确定现金分红权益")
                snapshot = self._state(account_id, point, economic=data["record_date"])
                units = sum((lot["units"] for lot in snapshot["lots"].values()
                             if lot["product_id"] == product_id), ZERO)
                amount = fund.money(units * decimal(data["cash_per_unit"]), rules["cash_precision"])
                self._emit(account_id, key, "dividend_entitlement", date.fromisoformat(data["ex_date"]), point,
                           {"cash": {"receivable": amount}, "eligible_units": units, "amount": amount,
                            "source_payload": data, "fact_id": fact.id}, product_id=product_id)
            if data["pay_date"] <= point.date().isoformat():
                row = self.store.conn.execute("SELECT payload FROM ledger_events WHERE event_key=?", (key,)).fetchone()
                amount = decimal(json.loads(row[0])["amount"])
                self._emit(account_id, key + ":paid", "dividend_cash_received", date.fromisoformat(data["pay_date"]),
                           point, {"cash": {"receivable": -amount, "available": amount}}, product_id=product_id)

    def _corporate_facts(self, product_id: str, kind: str, point: datetime) -> dict[str, Any]:
        values = {}
        for fact in self.store.facts(product_id, kind, point.isoformat(), end=point.date().isoformat()):
            # A changed pay_date is a revision, not a second cash entitlement.
            values[fact.event_date] = fact
        return values

    def _corporate_warnings(self, account_id: str, product_id: str, point: datetime) -> list[str]:
        warnings = []
        for kind in ("split", "dividend"):
            for identity, fact in self._corporate_facts(product_id, kind, point).items():
                if kind == "dividend" and fact.payload["ex_date"] > point.date().isoformat():
                    continue
                key = f"{account_id}:{kind}:{product_id}:{identity}"
                row = self.store.conn.execute(
                    "SELECT recorded_at,payload FROM ledger_events WHERE event_key=?", (key,)
                ).fetchone()
                if row is None or timestamp(row["recorded_at"]) > point:
                    warnings.append(f"{product_id}: 查询时点的 {kind} 权益尚未入账，不是完整资产估值")
                elif json.loads(row["payload"])["source_payload"] != fact.payload:
                    warnings.append(f"{product_id}: 已入账 {kind} 权益存在更正，需人工对账")
                elif kind == "dividend" and fact.payload["pay_date"] <= point.date().isoformat():
                    paid = self.store.conn.execute(
                        "SELECT recorded_at FROM ledger_events WHERE event_key=?", (key + ":paid",)
                    ).fetchone()
                    if paid is None or timestamp(paid["recorded_at"]) > point:
                        warnings.append(f"{product_id}: 分红到账尚未记账")
        return warnings

    def _already_corporate(self, key: str, payload: dict[str, Any]) -> bool:
        row = self.store.conn.execute("SELECT payload FROM ledger_events WHERE event_key=?", (key,)).fetchone()
        if row and json.loads(row[0])["source_payload"] != payload:
            raise ValidationError("已入账权益事实被更正；需人工对账，禁止重复入账或覆盖历史")
        return bool(row)

    def _state(self, account_id: str, point: datetime, economic: str | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {key: ZERO for key in (*CASH_FIELDS, "fees")}
        state["lots"] = {}
        for row in self.store.conn.execute("SELECT * FROM ledger_events WHERE account_id=? ORDER BY id", (account_id,)):
            if timestamp(row["recorded_at"]) > point or (economic and row["economic_date"] > economic):
                continue
            payload = json.loads(row["payload"])
            for key, value in payload.get("cash", {}).items():
                state[key] += decimal(value)
            state["fees"] += decimal(payload.get("fees", "0"))
            for data in payload.get("lots_add", []):
                lot = dict(data)
                lot["units"], lot["reserved"] = decimal(lot["units"]), ZERO
                state["lots"][lot["lot_id"]] = lot
            for reservation in payload.get("lot_reservations", []):
                if reservation["lot_id"] in state["lots"]:
                    state["lots"][reservation["lot_id"]]["reserved"] += decimal(reservation["units"])
            for consumed in payload.get("lots_consume", []):
                lot = state["lots"][consumed["lot_id"]]
                quantity = decimal(consumed["units"])
                lot["units"] -= quantity
                lot["reserved"] -= quantity
            for change in payload.get("lot_multipliers", []):
                lot = state["lots"][change["lot_id"]]
                lot["units"] *= decimal(change["ratio"])
                lot["reserved"] *= decimal(change["ratio"])
        if economic is None:
            self._check(state)
        return state

    @staticmethod
    def _check(state: dict[str, Any]) -> None:
        if any(state[key] < 0 for key in CASH_FIELDS):
            raise ValidationError("账本不变量失败：现金、冻结、应收或在途余额为负")
        if any(lot["units"] < 0 or lot["reserved"] < 0 or lot["reserved"] > lot["units"]
               for lot in state["lots"].values()):
            raise ValidationError("账本不变量失败：份额或冻结份额无效")

    @staticmethod
    def _allocate(state: dict[str, Any], product_id: str, quantity: Decimal, point: datetime) -> list[dict[str, Any]]:
        result = []
        for lot in sorted(state["lots"].values(), key=lambda item: (
            item["confirmed_date"], item.get("submitted_at", ""), item["lot_id"]
        )):
            if lot["product_id"] != product_id or lot["available_date"] > point.date().isoformat():
                continue
            taken = min(quantity, lot["units"] - lot["reserved"])
            if taken > 0:
                result.append({"lot_id": lot["lot_id"], "units": text(taken),
                               "confirmed_date": lot["confirmed_date"]})
                quantity -= taken
            if quantity == 0:
                return result
        raise ValidationError("可卖份额不足；未确认、未到可卖日或已冻结份额不能卖出")

    @staticmethod
    def _new_lot(order: dict[str, Any], units: Any, confirmed: date, available: date) -> dict[str, Any]:
        return {"lot_id": order["id"] + ":lot", "product_id": order["product_id"], "units": str(units),
                "confirmed_date": confirmed.isoformat(), "available_date": available.isoformat(),
                "submitted_at": order["submitted_at"]}

    def _profile(self, product: Product, point: datetime) -> Profile:
        profiles = self.store.profiles(product.id, point.isoformat())
        if not profiles:
            raise ValidationError(f"{product.id}: 查询时点没有有效且已观察的规则档案")
        profile = profiles[0]
        if profile.verified is not True:
            raise ValidationError(f"{product.id}: 最新适用档案未经核验，不回退旧版猜测")
        if not isinstance(profile.rules, dict):
            raise ValidationError("档案 rules 必须为规则对象")
        if profile.rules.get("dividend_mode", "cash") != "cash":
            raise ValidationError("仅支持现金分红档案；红利再投资缺少确认规则")
        if self.store.mode == "public" and profile.source_url.startswith("demo:"):
            raise ValidationError("真实模式禁止演示档案")
        rules = profile.rules
        required = COMMON_RULES | (ETF_RULES if product.kind == "etf" else FUND_RULES)
        missing = required - rules.keys()
        if missing:
            raise ValidationError("档案缺少必要规则: " + ", ".join(sorted(missing)))
        if not isinstance(rules["calendar_id"], str) or not rules["calendar_id"]:
            raise ValidationError("档案缺少日历标识")
        try:
            date.fromisoformat(rules["events_complete_through"])
        except (ValueError, TypeError) as exc:
            raise ValidationError("权益完整性日期无效") from exc
        integer_fields = {"status_max_age_days", "cash_precision", "share_precision"}
        integer_fields |= ({"buy_available_days", "sale_cash_days"} if product.kind == "etf" else
                           {"confirm_days", "redemption_confirm_days", "redemption_cash_days"})
        for key in integer_fields:
            if type(rules[key]) is not int or not 0 <= rules[key] <= (8 if "precision" in key else 366):
                raise ValidationError(f"规则 {key} 须为有效非负整数")
        if product.kind == "etf":
            for key in ("commission_rate", "commission_min", "slippage_bps", "lot_size"):
                if decimal(rules[key]) < 0:
                    raise ValidationError(f"规则 {key} 不得为负")
            if decimal(rules["lot_size"]) <= 0 or decimal(rules["lot_size"]) % 1:
                raise ValidationError("ETF lot_size 须为正整数")
            if decimal(rules["slippage_bps"]) >= 10000 or decimal(rules["commission_rate"]) > 1:
                raise ValidationError("ETF 滑点/佣金超出有效范围")
        else:
            try:
                cutoff = time.fromisoformat(rules["cutoff"])
                if cutoff.tzinfo is not None or len(rules["cutoff"]) != 5:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValidationError("cutoff 须为上海时间 HH:MM") from exc
            if not 0 <= decimal(rules["subscription_rate"]) <= 1 or decimal(rules["subscription_min"]) < 0:
                raise ValidationError("申购费率/起点无效")
            if rules["redemption_cash_days"] < rules["redemption_confirm_days"]:
                raise ValidationError("赎回到账日不能早于确认日（均从申请日计）")
            if rules.get("redemption_cash_basis", "application") != "application":
                raise ValidationError("不支持此赎回到账起算约定")
            tiers = rules["redemption_tiers"]
            if not isinstance(tiers, list) or not tiers:
                raise ValidationError("缺少分段赎回费率")
            previous = 0
            for index, tier in enumerate(tiers):
                if not isinstance(tier, dict) or "days_lt" not in tier or "rate" not in tier:
                    raise ValidationError("赎回费率分段字段不完整")
                boundary = tier["days_lt"]
                if boundary is None:
                    if index != len(tiers) - 1:
                        raise ValidationError("无上限赎回费率必须为最后一段")
                elif type(boundary) is not int or boundary <= previous:
                    raise ValidationError("赎回费率持有日边界须严格递增")
                else:
                    previous = boundary
                if not 0 <= decimal(tier["rate"]) <= 1:
                    raise ValidationError("赎回费率须在 0 到 1 之间")
            if tiers[-1]["days_lt"] is not None:
                raise ValidationError("赎回费率必须覆盖所有持有期")
        return profile

    @staticmethod
    def _coverage(rules: dict[str, Any], day: date) -> None:
        if date.fromisoformat(rules["events_complete_through"]) < day:
            raise ValidationError("权益事件完整性未覆盖该日，禁止假设没有分红/拆分")

    def _status(
        self, product_id: str, side: str, day: date, point: datetime, rules: dict[str, Any],
        reject_closed: bool = False,
    ) -> bool:
        facts = self.store.facts(product_id, "status", point.isoformat(), end=day.isoformat())
        if not facts:
            raise ValidationError("缺少当时可观察的申赎状态")
        fact = facts[-1]
        if (day - date.fromisoformat(fact.event_date)).days > rules["status_max_age_days"]:
            raise ValidationError("申赎状态已过期，不能假定继续开放")
        value = fact.payload.get("subscription" if side == "buy" else "redemption")
        if value not in {"open", "closed"}:
            raise ValidationError("申赎状态字段不完整")
        if value == "closed" and not reject_closed:
            raise ValidationError("该方向申赎状态关闭")
        return value == "open"

    def _is_open(self, calendar_id: str, day: date, point: datetime) -> bool:
        fact = self.store.latest_fact(calendar_id, "calendar", point.isoformat(), day.isoformat())
        if fact is None or type(fact.payload.get("is_open")) is not bool:
            raise ValidationError(f"日历缺失或尚未可观察: {calendar_id} {day}")
        return fact.payload["is_open"]

    def _shift(self, day: date, count: int, rules: dict[str, Any], point: datetime) -> date:
        self._is_open(rules["calendar_id"], day, point)
        for _ in range(740):
            if count == 0:
                return day
            day += timedelta(days=1)
            if self._is_open(rules["calendar_id"], day, point):
                count -= 1
        raise ValidationError("交易日偏移超出支持范围")

    def _previous_open(self, day: date, rules: dict[str, Any], point: datetime) -> date:
        for _ in range(370):
            day -= timedelta(days=1)
            if self._is_open(rules["calendar_id"], day, point):
                return day
        raise ValidationError("没有已知估值交易日")

    def _product(self, product_id: str) -> Product:
        product = self.store.get_product(product_id)
        if product is None:
            raise ValidationError("产品不存在")
        if product.is_demo != (self.store.mode == "demo"):
            raise ValidationError("真实与演示产品不可混用")
        if product.kind not in {"etf", "open_fund"}:
            raise ValidationError(f"不支持模拟交易产品类型: {product.kind}")
        return product

    def _write_time(self, point: datetime) -> None:
        if self.store.mode == "public" and point > timestamp(now_iso()) + timedelta(minutes=1):
            raise ValidationError("真实模式禁止将账本推进或提交到未来时间")

    def _account(self, account_id: str) -> dict[str, Any]:
        row = self.store.conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
        if row is None or row["mode"] != self.store.mode:
            raise ValidationError("账户不存在或数据模式不匹配")
        return self._row(row)

    @staticmethod
    def _positive(value: Any, precision: int, label: str) -> Decimal:
        result = decimal(value)
        if result <= 0 or result != fund.money(result, precision):
            raise ValidationError(f"{label}须为正数，最多 {precision} 位小数")
        return result

    @staticmethod
    def _input_number(value: Any) -> str | None:
        if value is None:
            return None
        number = decimal(value)
        if number and number.adjusted() >= 20:
            raise ValidationError("输入数值超过支持的十进制精度")
        return text(number.normalize())

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    @staticmethod
    def _public_order(order: dict[str, Any]) -> dict[str, Any]:
        return {**order["payload"], **order}

    def _get_order(self, order_id: str) -> dict[str, Any]:
        return self._public_order(self._row(self.store.conn.execute(
            "SELECT * FROM orders WHERE id=?", (order_id,)
        ).fetchone()))

    def _update(self, order: dict[str, Any], state: str, point: datetime) -> None:
        order["state"] = state
        order["updated_at"] = point.isoformat()
        self.store.conn.execute("UPDATE orders SET state=?,updated_at=?,payload=? WHERE id=?",
                                (state, point.isoformat(), canonical_json(safe(order["payload"])), order["id"]))

    def _emit_order(
        self, order: dict[str, Any], suffix: str, event_type: str, day: date, point: datetime,
        payload: dict[str, Any],
    ) -> None:
        self._emit(order["account_id"], order["id"] + ":" + suffix, event_type, day, point,
                   payload, order["id"], order["product_id"])

    def _emit(
        self, account_id: str, key: str, event_type: str, day: date, point: datetime,
        payload: dict[str, Any], order_id: str | None = None, product_id: str | None = None,
    ) -> None:
        if day > point.date():
            raise ValidationError("不能在经济日之前入账")
        self.store.conn.execute(
            "INSERT OR IGNORE INTO ledger_events(event_key,account_id,order_id,product_id,event_type,"
            "economic_date,recorded_at,payload) VALUES(?,?,?,?,?,?,?,?)",
            (key, account_id, order_id, product_id, event_type, day.isoformat(), point.isoformat(),
             canonical_json(safe(payload))),
        )

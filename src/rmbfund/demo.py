"""Explicitly fictional fixtures. Future observations remain invisible until the demo clock."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .domain import Fact, Product, Profile, ValidationError
from .storage import Store

INITIAL_AT = "2026-01-05T16:00:00+08:00"
END_AT = "2026-01-31T22:00:00+08:00"
CALENDAR_ID = "DEMO-CALENDAR"
DAYS = [
    "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
]
KNOWN_AT = "2026-01-04T12:00:00+08:00"


def initialize(store: Store) -> dict:
    if store.mode != "demo":
        raise ValidationError("演示只能初始化到独立 demo 数据库")
    products = [
        Product("DEMO-ETF", "DEMO-ETF", "虚构宽基 ETF · 非真实产品", "etf", True,
                "DEMO", "股票", "模拟份额"),
        Product("DEMO-FUND", "DEMO-FUND", "虚构开放式基金 · 非真实产品", "open_fund", True,
                "", "混合", "模拟 A"),
    ]
    common = {
        "calendar_id": CALENDAR_ID, "events_complete_through": "2026-01-31",
        "status_max_age_days": 4, "cash_precision": 2, "share_precision": 2,
        "settlement_convention": "示例所有延迟从申请/成交日计交易日，不代表任何真实产品",
        "dividend_mode": "cash",
    }
    rules = {
        "DEMO-ETF": {
            **common, "lot_size": 100, "commission_rate": "0.0003", "commission_min": "1",
            "slippage_bps": "5", "buy_available_days": 1, "sale_cash_days": 0,
        },
        "DEMO-FUND": {
            **common, "cutoff": "15:00", "subscription_rate": "0.0015",
            "subscription_min": "10", "confirm_days": 1, "redemption_confirm_days": 1,
            "redemption_cash_days": 3,
            "redemption_tiers": [
                {"days_lt": 7, "rate": "0.015"},
                {"days_lt": 30, "rate": "0.005"},
                {"days_lt": None, "rate": "0"},
            ],
        },
    }
    count = 0
    with store.transaction():
        for product in products:
            store.add_product(product)
            store.add_profile(Profile(
                f"{product.id}-v2", product.id, "2026-01-02", KNOWN_AT,
                rules[product.id], "demo://fictional-contract-v1", True,
            ))
        for offset in range(45):
            day = (date(2026, 1, 1) + timedelta(days=offset)).isoformat()
            count += store.add_fact(Fact(
                CALENDAR_ID, "calendar", day, {"is_open": day in DAYS},
                "demo", KNOWN_AT, source_url="demo://fictional-calendar", quality="demo",
            ))
        etf_prices = ["4.00", "4.03", "4.01", "3.97", "4.02", "4.06", "4.10", "4.07",
                      "4.09", "4.03", "4.05", "4.10", "4.08", "4.13", "4.16", "2.09",
                      "2.08", "2.11", "2.12", "2.10"]
        fund_navs = ["1.0000", "1.0060", "1.0030", "0.9980", "1.0040", "1.0080",
                     "1.0100", "1.0070", "1.0120", "1.0090", "1.0130", "0.9950",
                     "0.9970", "1.0010", "1.0040", "1.0060", "1.0040", "1.0090",
                     "1.0110", "1.0080"]
        for day, etf_value, fund_value in zip(DAYS, etf_prices, fund_navs, strict=True):
            open_value = Decimal(etf_value) - Decimal("0.005")
            count += store.add_fact(Fact(
                "DEMO-ETF", "bar", day,
                {"open": str(open_value), "high": str(Decimal(etf_value) + Decimal("0.02")),
                 "low": str(open_value - Decimal("0.02")), "close": etf_value,
                 "volume": "1000000", "adjustment": "raw"},
                "demo", f"{day}T16:05:00+08:00", quality="demo",
            ))
            observed_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
            count += store.add_fact(Fact(
                "DEMO-FUND", "nav", day, {"unit_nav": fund_value},
                "demo", f"{observed_day}T20:00:00+08:00", quality="demo",
            ))
            for product in products:
                count += store.add_fact(Fact(
                    product.id, "status", day,
                    {"subscription": "open", "redemption": "open"},
                    "demo", f"{day}T08:00:00+08:00", quality="demo",
                ))
        for code, payload, known in [
            ("DEMO-ETF", {"record_date": "2026-01-15", "ex_date": "2026-01-16",
                          "pay_date": "2026-01-20", "cash_per_unit": "0.04"},
             "2026-01-12T08:00:00+08:00"),
            ("DEMO-FUND", {"record_date": "2026-01-19", "ex_date": "2026-01-20",
                           "pay_date": "2026-01-22", "cash_per_unit": "0.02"},
             "2026-01-14T08:00:00+08:00"),
        ]:
            count += store.add_fact(Fact(
                code, "dividend", payload["ex_date"], payload, "demo", known, quality="demo",
            ))
            count += store.add_fact(Fact(
                code, "announcement", known[:10],
                {"title": "【虚构公告】现金分红示例", "url": "demo://dividend-example",
                 "coverage": "仅演示，不对应真实基金"}, "demo", known, quality="demo",
            ))
        count += store.add_fact(Fact(
            "DEMO-ETF", "split", "2026-01-26", {"ratio": "2"}, "demo",
            "2026-01-19T08:00:00+08:00", quality="demo",
        ))
    from .paper.engine import PaperEngine

    paper = PaperEngine(store)
    account_id = "demo-main"
    if not any(account["id"] == account_id for account in paper.accounts()):
        paper.create_account("虚构演示账户", "100000", INITIAL_AT, account_id=account_id)
    return {
        "initial_at": INITIAL_AT, "end_at": END_AT, "account_id": account_id,
        "inserted": count, "dates": [f"{day}T{hour}:00:00+08:00" for day in DAYS
                                     for hour in ("16", "22")] + [END_AT],
        "warning": "虚构产品、资金、净值及日历。不是投资推荐或真实回测。",
    }

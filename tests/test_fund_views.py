"""Synthetic-only fund views: identity, actual availability and saved issue history."""

from __future__ import annotations


import pytest


from rmbfund.domain import Fact, Product


from rmbfund.service import Workspace


NOW = "2026-10-23T12:00:00+08:00"


FIRST = "2026-10-21T23:00:00+08:00"


URL = "https://example.invalid/synthetic/nav"


PRODUCT = Product("existing:018147:C", "018147", "SYNTHETIC fund C", "open_fund", share_class="C")


def forbidden(*args, **kwargs):
    pytest.fail("passive fund views must not fit, refresh, score, research or use accounts")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: NOW)
    item = Workspace(tmp_path / "synthetic", "public")
    item.store.add_product(PRODUCT)
    for name in ("refresh", "_score_forecasts", "forecast_research", "overview", "prices"):
        monkeypatch.setattr(item, name, forbidden)
    monkeypatch.setattr("rmbfund.forecast.fit_forecast", forbidden)
    monkeypatch.setattr(item.paper, "submit", forbidden)
    monkeypatch.setattr(item.paper, "create_account", forbidden)
    try:
        yield item
    finally:
        item.store.close()


def seed_nav(workspace, *, product_id=PRODUCT.id, observed_at=FIRST, rows=None):
    values = rows or [("2026-10-16", "1.00"), ("2026-10-19", "1.10"), ("2026-10-21", "1.21")]
    raw = []
    for day, nav in values:
        fact = Fact(
            product_id,
            "nav",
            day,
            {
                "unit_nav": nav,
                "currency": "CNY",
                "nav_type": "final_unit_nav_candidate",
            },
            "synthetic-source",
            observed_at,
            URL,
            published_at="2026-10-21T22:00:00+08:00",
        )
        workspace.store.add_fact(fact)
        raw.append(
            {"source": fact.source, "source_url": URL, "record": {"date": day, "unit_nav": nav}}
        )
    workspace.store.record_snapshot("synthetic-source", product_id, raw, observed_at, URL)
    return raw


def add_portfolio(workspace, *, report_date="2026-09-30", available_at=FIRST):
    workspace.store.initialize_fund_review_schema()
    common = {
        "product_id": PRODUCT.id,
        "family": "portfolio",
        "source": "synthetic-disclosure",
        "upstream": "synthetic-report",
        "source_url": "https://example.invalid/synthetic/report",
        "observed_at": FIRST,
        "available_at": available_at,
    }
    capture = workspace.store.record_advice_capture(
        {
            **{key: value for key, value in common.items() if key != "available_at"},
            "status": "ok",
            "content_type": "application/json",
            "raw_bytes": b'{"synthetic":true}',
            "parser_version": "synthetic-v1",
            "warnings": [],
        }
    )
    return workspace.store.add_advice_evidence(
        {
            **common,
            "report_date": report_date,
            "payload": {
                "denominator": "equity_assets",
                "assets": {"equity": "0.8", "cash": "0.2"},
                "holdings_scope": "partial",
                "holdings": [
                    {
                        "issuer_id": "synthetic-issuer",
                        "name": "SYNTHETIC stock",
                        "kind": "corporate_equity",
                        "weight": "0.07",
                        "country": "SYNTHETIC country",
                    }
                ],
                "countries": [],
                "sectors": [],
                "currencies": [],
            },
        },
        capture,
        "synthetic holdings table",
    )


def test_public_portfolio_preserves_disclosure_scope_denominator_weight_and_date(workspace):
    seed_nav(workspace)
    original = add_portfolio(workspace)
    add_portfolio(workspace, report_date="2026-10-01", available_at="2026-10-24T12:00:00+08:00")
    before = list(workspace.store.conn.iterdump())
    portfolio = workspace.fund_detail(PRODUCT.id)["portfolio"]
    assert portfolio["report_date"] == "2026-09-30" and portfolio["holdings_scope"] == "partial"
    assert (
        portfolio["denominator"] == "equity_assets" and portfolio["holdings"][0]["weight"] == "0.07"
    )
    assert portfolio["holdings"] == original["payload"]["holdings"]
    assert portfolio["source_url"] == original["source_url"]
    assert workspace.fund_detail(PRODUCT.id, "2026-10-21T22:00:00+08:00")["portfolio"] is None
    assert list(workspace.store.conn.iterdump()) == before


def test_dated_conditions_preserve_denominator_and_do_not_use_future_disclosure(workspace):
    from rmbfund.investment_profile import (
        build_investment_profile, disclosure_shock_scenario, earnings_valuation_fx_scenario,
    )

    record = add_portfolio(workspace)
    profile = workspace.fund_detail(PRODUCT.id)["investment_profile"]
    assert profile["portfolio"]["report_date"] == record["report_date"]
    condition = disclosure_shock_scenario(profile, scope="equity", shocks={"equity": "-0.1"},
                                           label="SYNTHETIC explicit condition")
    assert condition["covered_contribution"] == "-0.08"
    assert condition["nav_return_contribution"] is None  # Original denominator is equity_assets.
    assert condition["whole_fund_return"] is None
    earnings = earnings_valuation_fx_scenario(profile, scope="holdings",
        members=["synthetic-issuer"], earnings_change="0.1", valuation_change="-0.05",
        label="SYNTHETIC explicit EPS and multiple conditions")
    assert earnings["conditions"]["local_price_change"] == "0.045"
    assert earnings["covered_contribution"] == "0.00315"
    assert earnings["fx_condition"] is None
    assert "probability" not in condition
    early = build_investment_profile([record], as_of="2026-10-21T22:00:00+08:00")
    assert early["portfolio"] is None

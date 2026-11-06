"""Offline service boundaries, actual availability, cache identity and outcome maturity."""

from __future__ import annotations


import math


from datetime import date, timedelta


from decimal import Decimal


import pytest


from rmbfund.domain import Fact, timestamp


from rmbfund.service import Workspace


NOW = "2026-10-22T12:00:00+08:00"


CAPTURED = "2026-10-21T23:00:00+08:00"


URL = "https://example.invalid/synthetic/nav"


def seed_nav(workspace, product_id, *, count=90, end="2026-10-21", observed_at=CAPTURED):
    end_day = date.fromisoformat(end)
    rows = []
    for index in range(count):
        day = (end_day - timedelta(days=count - index - 1)).isoformat()
        nav = format(Decimal(str(1 + index * 0.001 + math.sin(index / 4) * 0.01)), "f")
        fact = Fact(
            product_id,
            "nav",
            day,
            {"unit_nav": nav, "currency": "CNY"},
            "synthetic-offline",
            observed_at,
            URL,
        )
        workspace.store.add_fact(fact)
        rows.append(
            {
                "source": fact.source,
                "source_url": URL,
                "record": {"date": day, "unit_nav": nav},
                "observed_at": observed_at,
            }
        )
    workspace.store.record_snapshot("synthetic-offline", product_id, rows, observed_at, URL)
    return rows


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: NOW)
    workspace = Workspace(tmp_path / "store", "public")
    product_id = workspace.add_product(
        "270042", "SYNTHETIC generic fund", "open_fund", share_class="A"
    )
    seed_nav(workspace, product_id)

    def forbidden(*args, **kwargs):
        raise AssertionError("network/account/order work is forbidden")

    monkeypatch.setattr(workspace, "refresh", forbidden)
    monkeypatch.setattr(workspace.paper, "submit", forbidden)
    monkeypatch.setattr(workspace.paper, "create_account", forbidden)
    try:
        yield workspace, product_id
    finally:
        workspace.store.close()


def test_reader_never_fits_initializes_or_writes(context, monkeypatch):
    workspace, product_id = context
    monkeypatch.setattr(
        "rmbfund.forecast.fit_runtime_forecast", lambda *a, **k: pytest.fail("passive fit")
    )
    before = list(workspace.store.conn.iterdump())
    assert workspace.forecast(product_id)["status"] == "missing"
    assert not workspace.store.has_forecast_schema()
    assert list(workspace.store.conn.iterdump()) == before
    assert not hasattr(workspace, "advice")
    assert not hasattr(workspace, "update_advice")
    assert not hasattr(workspace, "fund_review")
    assert not hasattr(workspace, "update_fund_review")


def test_explicit_local_update_whole_history_weekends_and_dedup(context, monkeypatch):
    from rmbfund import forecast

    workspace, product_id = context
    fit = forecast.fit_runtime_forecast
    calls = []

    def counted(series, **kwargs):
        calls.append((series, kwargs))
        return fit(series, **kwargs)

    monkeypatch.setattr(forecast, "fit_runtime_forecast", counted)
    first = workspace.update_forecast(product_id, refresh=False)
    assert first["status"] == "ready"
    assert len(calls) == 1
    assert len(calls[0][0]) == 90
    assert any(date.fromisoformat(row["date"]).weekday() >= 5 for row in calls[0][0])
    assert first["prediction"]["generated_at"] == NOW
    assert first["prediction"]["anchor"]["date"] == "2026-10-21"
    assert first["prediction"]["entry_assumptions"]["entry_lags"] == [1, 2, 3]
    assert first["prediction"]["entry_assumptions"]["calendar_basis"] == "weekday_research_proxy"
    changes = workspace.store.conn.total_changes
    repeated = workspace.update_forecast(product_id, refresh=False)
    assert repeated["prediction"] == first["prediction"]
    assert len(calls) == 1
    assert workspace.store.conn.total_changes == changes
    assert workspace.store.conn.execute("SELECT count(*) FROM forecast_records").fetchone()[0] == 1
    for table in ("accounts", "orders", "ledger_events", "journal"):
        assert workspace.store.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_local_curve_requests_exact_horizons_without_official_issue_or_write(context):
    workspace, product_id = context
    before = list(workspace.store.conn.iterdump())
    result = workspace.calculate_timing(
        product_id, calendar_days=20, decision_at=NOW, include_curve=True,
    )
    assert result["status"] == "ready"
    prediction = result["prediction"]
    plan = result["trade_plan"]
    requested = list(range(1, plan["horizon"] + 1))
    assert prediction["requested_horizons"] == requested
    assert prediction["calculation"]["include_curve"] is True
    assert prediction["calculation"]["requested_horizons"] == requested
    assert {(head["horizon"], head["entry_lag"])
            for head in prediction["execution_forecasts"] if head["entry_lag"] == plan["entry_lag"]} == {
                (horizon, plan["entry_lag"]) for horizon in requested
            }
    assert result["calculation"]["actual_issue"] is False
    assert result["calculation"]["persisted"] is False
    assert list(workspace.store.conn.iterdump()) == before
    assert not workspace.store.has_forecast_schema()

    single = workspace.calculate_timing(product_id, calendar_days=20, decision_at=NOW)
    assert single["prediction"]["requested_horizons"] == [plan["horizon"]]
    assert single["prediction"]["execution_forecasts"]
    assert {head["horizon"] for head in single["prediction"]["execution_forecasts"]} == {plan["horizon"]}


def test_failed_refresh_never_masquerades_as_success_and_read_sees_actual_snapshot(
    context, monkeypatch
):
    workspace, product_id = context
    original = workspace.update_forecast(product_id, refresh=False)
    later = "2026-10-22T12:03:00+08:00"
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: later)

    def failed(product):
        workspace.store.record_snapshot(
            "synthetic-offline",
            product,
            [],
            later,
            URL,
            status="error",
            warnings=["synthetic source failure"],
        )
        return {"status": "error", "warnings": ["synthetic source failure"]}

    monkeypatch.setattr(workspace, "refresh", failed)
    result = workspace.update_forecast(product_id)
    assert result["status"] == "error"
    assert result["prediction"] == original["prediction"]
    changes = workspace.store.conn.total_changes
    read = workspace.forecast(product_id)
    assert read["status"] == "error"
    assert read["prediction"]["generated_at"] == NOW
    assert "synthetic source failure" in read["refresh"]["warnings"]
    assert workspace.store.conn.total_changes == changes


def test_forward_results_append_only_when_mature_and_never_on_read(context, monkeypatch):
    workspace, product_id = context
    issued = workspace.update_forecast(product_id, refresh=False)
    record_id = issued["prediction"]["record_id"]
    assert workspace.store.forecast_outcomes(product_id, NOW) == []
    for offset in range(1, 25):
        day = date(2026, 10, 21) + timedelta(days=offset)
        available = f"{day.isoformat()}T23:00:00+08:00"
        workspace.store.add_fact(
            Fact(
                product_id,
                "nav",
                day.isoformat(),
                {"unit_nav": str(1.1 + offset * 0.001)},
                "synthetic-offline",
                available,
                URL,
            )
        )
    mature_at = "2026-11-15T12:00:00+08:00"
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: mature_at)
    before = workspace.store.conn.total_changes
    workspace.forecast(product_id)
    assert workspace.store.conn.total_changes == before
    assert workspace.store.forecast_outcomes(product_id, mature_at) == []
    result = workspace.update_forecast(product_id, refresh=False)
    assert result["status"] == "ready"
    outcomes = [
        row
        for row in workspace.store.forecast_outcomes(product_id, mature_at)
        if row["forecast_id"] == record_id
    ]
    expected_heads = {
        (head["horizon"], head["entry_lag"])
        for head in [
            *issued["prediction"]["forecasts"],
            *issued["prediction"]["execution_forecasts"],
        ]
    }
    assert {(row["horizon"], row["entry_lag"]) for row in outcomes} == expected_heads
    assert len(outcomes) == 12
    assert any(row["kind"] == "forward" for row in outcomes)
    assert all(row["kind"] == "nowcast" for row in outcomes if row["entry_lag"] == 0)
    assert all(timestamp(row["scored_at"]) >= timestamp(row["issued_at"]) for row in outcomes)
    before_rows = workspace.store.forecast_outcomes(product_id, mature_at)
    workspace.update_forecast(product_id, refresh=False)
    assert workspace.store.forecast_outcomes(product_id, mature_at) == before_rows


def test_factor_versions_preserve_first_seen_and_change_request_without_nav_change(context):
    from copy import deepcopy
    from rmbfund.investment_factors import SERIES_CONTRACT

    workspace, product_id = context
    product = workspace.store.get_product(product_id)
    data = workspace._forecast_data(product, NOW)
    source = {"schema_version": 1, "series": {role: {
        "series_id": identity[0], "currency": identity[1], "unit": identity[2],
        "return_basis": identity[3], "label": role, "source": "synthetic-offline",
        "source_url": "https://example.invalid/synthetic/factors", "captured_at": CAPTURED,
        "rows": [{"date": "2026-10-20", "value": 4, "available_at": CAPTURED,
                  "historical_available_at": None}],
    } for role, identity in SERIES_CONTRACT.items()}}
    first = workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": source}, observed_at=NOW)
    refreshed = deepcopy(source)
    later = "2026-10-22T12:05:00+08:00"
    for series in refreshed["series"].values():
        series["captured_at"] = later
        series["rows"][0]["available_at"] = later
    second = workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": refreshed}, observed_at=later)
    assert first["input_hash"] == second["input_hash"]
    assert second["bundle"]["series"]["market"]["rows"][0]["available_at"] == CAPTURED
    changed = deepcopy(refreshed)
    changed["series"]["market"]["rows"][0]["value"] = 5
    third = workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": changed}, observed_at=later)
    assert third["input_hash"] != first["input_hash"]
    revised = deepcopy(changed)
    revised["series"]["market"]["rows"][0]["version_id"] = "new-publisher-version"
    fourth = workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": revised}, observed_at=later)
    assert fourth["input_hash"] != third["input_hash"]
    extended = deepcopy(revised)
    extended["series"]["market"]["rows"].insert(0, {
        **extended["series"]["market"]["rows"][0], "date": "2026-10-19"})
    workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": extended}, observed_at=later)
    withdrawn = deepcopy(revised)
    withdrawn["series"]["market"]["rows"] = []
    withdrawn["series"]["market"]["metadata"] = {
        "complete_window": True, "coverage_start": "2026-10-20", "coverage_end": "2026-10-20"}
    final = workspace.store.add_factor_capture(product_id, {"status": "ready", "bundle": withdrawn}, observed_at=later)
    assert [row["date"] for row in final["bundle"]["series"]["market"]["rows"]] == ["2026-10-19"]
    assert final["input_hash"] != fourth["input_hash"]
    assert workspace.store.factor_capture(product_id, as_of=NOW)["id"] == first["id"]
    def request(value):
        return workspace._forecast_request(product, NOW, data,
            model_id="exposure_factor_ridge_empirical_v1",
            investment={"factor_input_hash": value, "profile_mapping_hash": "synthetic-profile"})[0]
    assert request(first["input_hash"]) == request(second["input_hash"])
    assert request(first["input_hash"]) != request(third["input_hash"])
    before = workspace.store.conn.total_changes
    unsupported = workspace.forecast(product_id, model_id="exposure_factor_ridge_empirical_v1")
    assert unsupported["status"] == "unsupported" and unsupported["prediction"] is None
    assert workspace.store.conn.total_changes == before


def test_equal_nav_values_with_contradictory_source_currency_block_forecast(context):
    workspace, product_id = context
    product = workspace.store.get_product(product_id)
    latest = workspace._forecast_data(product, NOW)["series"][-1]
    workspace.store.add_fact(
        Fact(
            product_id,
            "nav",
            latest["date"],
            {"unit_nav": latest["nav"], "currency": "USD"},
            "contradictory-currency-source",
            "2026-10-21T22:00:00+08:00",
            "https://example.invalid/synthetic/usd-nav",
        )
    )
    data = workspace._forecast_data(product, NOW)
    assert data["nav"]["nav_conflicts"] == []
    assert {row["provenance"]["currency"] for row in data["series"]} == {"CNY"}
    assert any("币种或净值口径不一致" in reason for reason in data["reasons"])
    before = workspace.store.conn.total_changes
    assert workspace.forecast(product_id)["status"] == "error"
    assert workspace.update_forecast(product_id, refresh=False)["status"] == "error"
    assert workspace.store.conn.total_changes == before
    assert not workspace.store.has_forecast_schema()

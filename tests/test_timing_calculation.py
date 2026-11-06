"""Synthetic RAM-only timing/chronology contracts, not investment performance tests."""
from copy import deepcopy
from datetime import date, timedelta
import socket

import numpy as np
import pytest

from rmbfund import forecast as core
from rmbfund.current_online import snapshot_online_context
from rmbfund.domain import Fact, Profile, ValidationError
from rmbfund.service import Workspace
from rmbfund.trade_calendar import build_trade_plan, project_sale_date

NOW = "2026-10-28T12:00:00+08:00"
LOCAL = "2026-10-26T18:00:00+08:00"
STARTED = "2026-10-28T11:00:00+08:00"
OBSERVED = "2026-10-28T11:01:00+08:00"
AVAILABLE = "2026-10-28T11:02:00+08:00"
COMPLETED = "2026-10-28T11:03:00+08:00"
URL = "https://example.invalid/synthetic/nav"
PID = "open_fund:018147:C"
SUBJECT = {"id": PID, "symbol": "018147", "share_class": "C", "kind": "open_fund",
           "exchange": "", "is_demo": False, "name": "SYNTHETIC fund"}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("timing calculation must not request network or issue/write anything")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: NOW)
    return forbidden


def _rows(count=340, start="2025-01-02"):
    rng = np.random.default_rng(811)
    navs = np.exp(np.cumsum(rng.normal(0.0002, 0.006, count)))
    first = date.fromisoformat(start)
    return [{"date": (first + timedelta(days=i)).isoformat(), "nav": str(value),
             "provenance": {"source": "synthetic", "source_url": URL,
                            "available_at": LOCAL, "observed_at": LOCAL, "currency": "CNY"}}
            for i, value in enumerate(navs)]


def _fit(rows, **kwargs):
    return core.fit_runtime_forecast(rows, product_id=PID, generated_at=NOW, **kwargs)


def _compact(head):
    return {key: value for key, value in head.items()
            if key not in {"samples", "drivers", "historical_baseline"}}


def _online(rows):
    navs = [{"date": row["date"], "value": row["nav"], "currency": "CNY",
             "product_id": PID, "symbol": "018147", "share_class": "C",
             "nav_type": "final_unit_nav_candidate", "source": "synthetic",
             "source_url": URL, "observed_at": OBSERVED, "available_at": AVAILABLE,
             "published_at": None, "unitMoney": ""} for row in rows]
    return {"schema_version": 1, "kind": "current_online_context", "status": "ready",
            "started_at": STARTED, "checked_at": COMPLETED, "evaluated_at": COMPLETED,
            "public_subjects": [deepcopy(SUBJECT)], "nav_by_product": {PID: navs},
            "records_by_product": {}, "messages_by_product": {}, "source_checks": [
                {"product_id": PID, "family": "nav", "source_url": URL, "status": "ok",
                 "parse_status": "parsed", "observed_at": OBSERVED, "checked_at": OBSERVED,
                 "record_count": len(navs)}]}


@pytest.fixture
def workspace(tmp_path, offline, monkeypatch):
    root = tmp_path / "store"
    seeded = Workspace(root, "public")
    seeded.add_product("018147", "SYNTHETIC fund", "open_fund", share_class="C")
    rows = _rows(90, "2026-07-29")
    for row in rows:
        seeded.store.add_fact(Fact(PID, "nav", row["date"], {"unit_nav": row["nav"], "currency": "CNY"},
                                   "synthetic", LOCAL, URL))
    raw = [{"record": {"date": row["date"], "unit_nav": row["nav"],
                       "unitMoney": "synthetic-known-event" if i == 40 else ""},
            "source": "synthetic", "source_url": URL, "observed_at": LOCAL}
           for i, row in enumerate(rows)]
    seeded.store.record_snapshot("synthetic", PID, raw, LOCAL, URL)
    seeded.store.add_fact(Fact(PID, "split", "2026-10-27", {"ratio": "2"},
                               "synthetic-event", LOCAL, URL + "/split"))
    seeded.store.close()
    work = Workspace(root, "public", read_only=True, initialize=False)
    for name in ("refresh", "current_online", "update_forecast", "_score_forecasts", "forecast"):
        monkeypatch.setattr(work, name, offline)
    for name in ("add_forecast_record", "add_forecast_outcome", "add_fact", "record_snapshot"):
        if hasattr(work.store, name):
            monkeypatch.setattr(work.store, name, offline)
    monkeypatch.setattr(work.store, "forecast_record", offline)
    try:
        yield work, rows
    finally:
        work.store.close()


def test_complete_history_keeps_every_exact_head_origin_and_immature_invalid_tail():
    rows = _rows(310, "2023-09-01")  # Outside the old 2025/26 research window.
    rows[-4]["invalid_return"] = True
    report = _fit(rows, entry_lags=(0, 2, 4), include_history=True, include_evaluation=True)
    history = report["historical_forecasts"]
    assert report["issued_forecasts"] == report["evaluation_rows"] == []
    assert len(history) == len(rows) * 9
    assert [(r["origin_index"], r["horizon"], r["entry_lag"]) for r in history] == sorted({
        (i, h, lag) for i in range(len(rows)) for h in (1, 5, 20) for lag in (0, 2, 4)})
    assert history[0]["prediction"]["status"] == "insufficient_history"
    for row in history:
        i, horizon, lag = row["origin_index"], row["horizon"], row["entry_lag"]
        assert row["origin_date"] == rows[i]["date"]
        assert row["assumed_known_index"] == i + 1
        assert row["assumed_known_date"] == (rows[i + 1]["date"] if i + 1 < len(rows) else None)
        assert row["prediction"]["horizon"] == horizon and row["prediction"]["entry_lag"] == lag
        assert not {"actual_change", "target_date", "issued_at", "generated_at", "samples"} & row.keys()
        for head in (row["prediction"], row.get("historical_baseline", {})):
            assert "samples" not in head
            for kind in ("training", "residual"):
                endpoint = head.get(f"{kind}_target_end_index")
                assert endpoint is None or endpoint <= i
    assert all(r["prediction"]["status"] == "available" for r in history[-9:])
    final_heads = {(h["horizon"], h["entry_lag"]): h
                   for h in report["forecasts"] + report["execution_forecasts"]}
    for row in history[-9:]:
        head = final_heads[row["horizon"], row["entry_lag"]]
        assert row["prediction"] == _compact(head)
        assert row["historical_baseline"] == head["historical_baseline"]


def test_history_prefix_and_future_independence_without_changing_legacy_diagnostics():
    rows, origin = _rows(340), 280
    original = _fit(rows, entry_lags=(1, 3), include_evaluation=True)
    complete = _fit(rows, entry_lags=(1, 3), include_evaluation=True, include_history=True)
    assert {key: value for key, value in complete.items() if key != "historical_forecasts"} == original
    assert {r["horizon"] for r in original["issued_forecasts"]} == {20}
    prefix = _fit(rows[:origin + 1], entry_lags=(1, 3), include_history=True)
    changed = deepcopy(rows)
    for row in changed[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.15)
    changed[origin + 2]["invalid_return"] = True
    after = _fit(changed, entry_lags=(1, 3), include_history=True)
    before_rows = [r for r in complete["historical_forecasts"] if r["origin_index"] <= origin]
    assert before_rows == [r for r in after["historical_forecasts"] if r["origin_index"] <= origin]
    for before, cut in zip(before_rows, prefix["historical_forecasts"], strict=True):
        assert before["prediction"] == cut["prediction"]
        assert before.get("historical_baseline") == cut.get("historical_baseline")
    # A target later crossing the new event stays in the action stream, even
    # though it can no longer be included among matured evaluation labels.
    invalid = next(r for r in after["historical_forecasts"]
                   if (r["origin_index"], r["horizon"], r["entry_lag"]) == (origin, 20, 1))
    assert invalid["prediction"]["status"] == "available"


def test_historical_baselines_use_exact_horizons_lags_and_event_payoff_masks():
    rows, origin = _rows(120), 95
    rows[70]["invalid_return"] = True
    result = _fit(rows, entry_lags=(2,), include_history=True)
    for row in result["historical_forecasts"]:
        if row["origin_index"] != origin:
            continue
        h, lag = row["horizon"], row["entry_lag"]
        labels = [float(rows[j + lag + h]["nav"]) / float(rows[j + lag]["nav"]) - 1
                  for j in range(origin - lag - h + 1)
                  if not any(r.get("invalid_return") for r in rows[j + lag + 1:j + lag + h + 1])]
        baseline = row["historical_baseline"]
        assert baseline["sample_count"] == len(labels)
        assert baseline["expected_change"] == pytest.approx(np.mean(labels), abs=1e-15)
        assert [baseline["lower"], baseline["median_change"], baseline["upper"]] == pytest.approx(
            np.quantile(labels, [.1, .5, .9]), abs=1e-15)


def test_explicit_short_history_keeps_unavailable_states_but_default_contract_stays():
    rows = _rows(8)
    with pytest.raises(ValueError, match="insufficient_history"):
        _fit(rows)
    report = _fit(rows, entry_lags=(1,), include_history=True)
    assert len(report["historical_forecasts"]) == len(rows) * 6
    assert report["forecasts"] == report["execution_forecasts"] == []
    assert all(r["prediction"]["status"] == "insufficient_history" for r in report["historical_forecasts"])


def test_timing_inputs_is_lightweight_and_calculation_is_ram_only_on_readonly_store(workspace, monkeypatch):
    work, rows = workspace
    original_fit = core.fit_runtime_forecast
    before = list(work.store.conn.iterdump())
    monkeypatch.setattr(core, "fit_runtime_forecast", lambda *a, **k: pytest.fail("timing_inputs fitted"))
    inputs = work.timing_inputs(PID)
    assert inputs["status"] == "ready" and len(inputs["series"]) == len(rows)
    assert inputs["cache_key"] == work.timing_inputs(PID)["cache_key"]
    monkeypatch.setattr(core, "fit_runtime_forecast", original_fit)
    report = work.calculate_timing(PID, include_history=True)
    assert report["status"] == "ready" and report["cache_key"] == inputs["cache_key"]
    assert report["evaluated_at"] == NOW and report["prediction"]["generated_at"] == NOW
    assert report["calculation"]["kind"] == "local_hypothetical"
    assert report["calculation"]["persisted"] is report["calculation"]["actual_issue"] is False
    assert "record_id" not in report["prediction"]
    assert len(report["historical_forecasts"]) == len(rows) * 12
    assert list(work.store.conn.iterdump()) == before
    assert work.store.conn.total_changes == 0 and not work.store.has_forecast_schema()
    assert "historical_forecasts" not in work.calculate_timing(PID)


def test_raw_merge_preserves_local_clock_all_provenance_and_events_in_online_extension(workspace):
    work, local_rows = workspace
    online_rows = deepcopy(local_rows[-3:]) + [{"date": "2026-10-27", "nav": "1.2"}]
    context = _online(online_rows)
    context["nav_by_product"][PID][-1]["unitMoney"] = "synthetic-dividend"
    original = deepcopy(context)
    inputs = work.timing_inputs(PID, context=context)
    assert inputs["status"] == "ready" and len(inputs["series"]) == len(local_rows) + 1
    assert inputs["cache_key"] == work.timing_inputs(PID, context=context)["cache_key"]
    assert inputs["cache_key"] != work.timing_inputs(PID)["cache_key"]
    overlapped, latest = inputs["series"][-2:]
    assert overlapped["provenance"]["available_at"] == LOCAL
    assert overlapped["provenance"]["online_observations"][0]["available_at"] == AVAILABLE
    assert latest["provenance"]["observed_at"] == OBSERVED
    assert latest["provenance"]["available_at"] == AVAILABLE
    assert latest["invalid_return"] is True
    assert latest["provenance"]["corporate_events"][0]["kind"] == "split"
    assert latest["provenance"]["source_markers"][0]["unitMoney"] == "synthetic-dividend"
    assert inputs["series"][40]["invalid_return"] is True
    assert work.calculate_timing(PID, context=context)["prediction"]["anchor"]["date"] == "2026-10-27"
    assert context == original


@pytest.mark.parametrize("change", [
    {"value": "99"}, {"currency": "USD"}, {"share_class": "A"},
    {"nav_type": "intraday_estimate"}, {"available_at": "2026-10-29T11:02:00+08:00"},
    {"observed_at": LOCAL}, {"label_valid": "false"},
])
def test_invalid_or_conflicting_online_nav_never_silently_uses_local_fallback(workspace, change, monkeypatch):
    work, rows = workspace
    context = _online(rows[-1:])
    context["nav_by_product"][PID][0].update(change)
    monkeypatch.setattr(core, "fit_runtime_forecast", lambda *a, **k: pytest.fail("fit rejected NAV"))
    report = work.calculate_timing(PID, context=context, include_history=True)
    assert report["status"] == "error" and report["prediction"] is None
    assert report["reasons"] and report["historical_forecasts"] == []


def test_gap_filling_moves_known_event_mask_to_first_real_observation(workspace, monkeypatch):
    work, _ = workspace
    data = work._forecast_data(work.store.get_product(PID), NOW)
    earlier = deepcopy(data["series"][0])
    later = deepcopy(data["series"][1])
    earlier["date"], later["date"] = "2026-10-22", "2026-10-26"
    later["invalid_return"] = True
    event = {"fact_id": "synthetic-gap-split", "kind": "split", "event_date": "2026-10-24",
             "effective_date": "2026-10-24", "payload": {"ratio": "2"}, "source": "synthetic",
             "source_url": URL, "observed_at": LOCAL, "available_at": LOCAL}
    data.update(series=[earlier, later])
    data["nav"]["corporate_events"] = [event]
    data["nav"]["unit_money"]["markers"] = []
    monkeypatch.setattr(work, "_forecast_data", lambda *a: deepcopy(data))
    context = _online([{"date": "2026-10-25", "nav": "1"}])
    report = work.timing_inputs(PID, context=context)
    assert report["status"] == "ready"
    first, inserted, last = report["series"]
    assert not first.get("invalid_return") and inserted["invalid_return"] is True
    assert not last.get("invalid_return")
    assert inserted["provenance"]["corporate_events"] == [event]
    assert last["provenance"]["corporate_events"] == []


def test_raw_same_batch_conflicts_and_compacted_context_are_rejected(workspace):
    work, rows = workspace
    context = _online(rows[-1:])
    context["nav_by_product"][PID].append({**context["nav_by_product"][PID][0], "value": "4"})
    report = work.calculate_timing(PID, context=context)
    assert report["status"] == "error" and report["evidence"]["nav"]["nav_conflicts"]
    for invalid in (snapshot_online_context(_online(rows[-3:])), {"history": rows}):
        report = work.calculate_timing(PID, context=invalid)
        assert report["status"] == "error" and report["prediction"] is None


def test_unregistered_public_identity_can_calculate_without_registration(workspace):
    work, rows = workspace
    other = "open_fund:539002:C"
    context = _online(rows)
    context["public_subjects"] = [{**SUBJECT, "id": other, "symbol": "539002"}]
    context["nav_by_product"][other] = context["nav_by_product"].pop(PID)
    for row in context["nav_by_product"][other]:
        row.update(product_id=other, symbol="539002")
    context["source_checks"][0]["product_id"] = other
    before = list(work.store.conn.iterdump())
    report = work.calculate_timing(other, context=context)
    assert report["status"] == "ready" and report["product"]["id"] == other
    assert work.store.get_product(other) is None and list(work.store.conn.iterdump()) == before


def _investment_inputs(rows):
    from rmbfund.investment_factors import SERIES_CONTRACT, factor_input_hash
    from rmbfund.investment_forecast import profile_mapping_input_hash

    t = np.arange(len(rows))
    values = {"market": 50 * np.exp(t * .0001 + .02 * np.sin(t / 13)),
              "industry": 100 * np.exp(t * .0002 + .03 * np.sin(t / 17)),
              "fx": 7 * np.exp(.01 * np.sin(t / 31)),
              "yield_2y": 4 + .2 * np.sin(t / 19), "yield_10y": 4 + .3 * np.sin(t / 23)}
    bundle = {"schema_version": 1, "series": {}}
    for role, (sid, currency, unit, basis) in SERIES_CONTRACT.items():
        bundle["series"][role] = {
            "series_id": sid, "label": "SYNTHETIC " + role, "source": "synthetic",
            "source_url": URL + "/" + role, "currency": currency, "unit": unit,
            "return_basis": basis, "captured_at": LOCAL,
            "rows": [{"date": row["date"], "value": float(value), "available_at": LOCAL,
                      "historical_available_at": None} for row, value in zip(rows, values[role], strict=True)]}
    profile = {"schema_version": 1, "product_id": PID, "as_of": NOW, "factor_mapping": [],
               "portfolio": {"record_id": "synthetic-portfolio", "report_date": "2025-06-30",
                             "available_at": LOCAL, "payload": {"denominator": "nav",
                                 "assets": {"equity": "0.8"}, "holdings": [],
                                 "sectors": [{"name": "信息技术", "weight": "0.7"}]}},
               "mandate": {"record_id": "synthetic-mandate", "report_date": "2025-06-30",
                           "available_at": LOCAL, "payload": {"asset_class": "mixed", "base_currency": "CNY",
                               "benchmark": {"region": "emerging_markets"}}}}
    return {"bundle": bundle, "profile": profile, "factor_input_hash": factor_input_hash(bundle),
            "profile_mapping_hash": profile_mapping_input_hash(profile, product_id=PID, generated_at=NOW)}


def test_selected_investment_model_is_genuine_or_explicitly_unavailable(workspace, monkeypatch):
    work, local_rows = workspace
    model = "exposure_factor_ridge_empirical_v1"
    assert work.calculate_timing(PID, model_id=model)["status"] == "insufficient"
    short = _investment_inputs(local_rows)
    monkeypatch.setattr(work, "_investment_inputs", lambda *a: short)
    monkeypatch.setattr(core, "fit_runtime_forecast", lambda *a, **k: pytest.fail("investment fell back to NAV"))
    missing = work.calculate_timing(PID, model_id=model, include_history=True)
    assert missing["status"] == "insufficient" and missing["prediction"] is None
    assert len(missing["historical_forecasts"]) == len(local_rows) * 12
    assert all(row["prediction"]["status"] != "available" for row in missing["historical_forecasts"])
    # Qualify a synthetic complete 14-feature model through the real existing fitter.
    rows = _rows(340, "2025-11-21")
    inputs = _investment_inputs(rows)
    monkeypatch.setattr(work, "_investment_inputs", lambda *a: inputs)
    original_data = work._forecast_data
    data = original_data(work.store.get_product(PID), NOW)
    data.update(series=rows, basis_hash="synthetic-qualified-full-series")
    monkeypatch.setattr(work, "_forecast_data", lambda *a: deepcopy(data))
    report = work.calculate_timing(PID, model_id=model, include_history=True)
    assert report["status"] == "ready" and report["prediction"]["model_id"] == model
    assert len(report["prediction"]["model_state"][0]["feature_names"]) == 14
    history = report["historical_forecasts"]
    assert len(history) == len(rows) * 12
    assert any(row["prediction"]["status"] == "unavailable" and row.get("historical_baseline") for row in history)
    assert all(row["prediction"].get("method") == "ridge_prequential_empirical"
               for row in history if row["prediction"]["status"] == "available")
    for row in history[-12:]:
        head = next(head for head in report["prediction"]["forecasts"] + report["prediction"]["execution_forecasts"]
                    if (head["horizon"], head["entry_lag"]) == (row["horizon"], row["entry_lag"]))
        assert row["prediction"] == _compact(head)


def _calendar(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return {(first + timedelta(days=i)).isoformat(): (first + timedelta(days=i)).weekday() < 5
            for i in range((last - first).days + 1)}


@pytest.mark.parametrize(("at", "days", "application", "sale", "horizon", "status"), [
    ("2026-10-29T14:59:59+08:00", 1, "2026-10-29", "2026-10-30", 1, "ready"),
    ("2026-10-29T15:00:00+08:00", 1, "2026-10-30", "2026-10-30", 0, "too_short"),
    ("2026-10-30T12:00:00+08:00", 2, "2026-10-30", "2026-10-30", 0, "too_short"),
    ("2026-10-30T15:00:00+08:00", 2, "2026-11-02", "2026-10-30", 0, "too_short"),
    ("2026-10-31T12:00:00+08:00", 1, "2026-11-02", None, 0, "too_short"),
    ("2026-10-31T12:00:00+08:00", 3, "2026-11-02", "2026-11-03", 1, "ready"),
])
def test_calendar_schedule_cutoff_weekend_and_too_short_never_rolls_past_deadline(
    at, days, application, sale, horizon, status,
):
    plan = build_trade_plan(decision_at=at, anchor_date="2026-10-28", calendar_days=days)
    assert plan["decision_at"] == at
    assert plan["application_date"] == application and plan["sale_date"] == sale
    assert plan["horizon"] == horizon and plan["status"] == status
    assert sale is None or sale <= plan["deadline"]
    assert plan["calendar_basis"] == "weekday_scenario"
    assert plan["confirmation_basis"] == "illustrative_2_workdays"
    assert "示例" in plan["confirmation_label"]
    assert plan["target_unit"] == "nav_observations"


def test_new_confirmation_does_not_hide_existing_share_comparisons_or_require_nav():
    plan = build_trade_plan(decision_at="2026-10-29T12:00:00+08:00",
                            anchor_date=None, calendar_days=1)
    assert plan["status"] == "ready" and plan["horizon"] == 1
    assert plan["entry_lag"] is None
    assert plan["new_confirmation_date"] == "2026-11-02"
    assert plan["new_shares_sellable"] is False
    changed = build_trade_plan(decision_at=plan["decision_at"], anchor_date=None,
                               calendar_days=1, confirmation_days=1)
    assert changed["new_confirmation_date"] == "2026-10-30"
    assert changed["new_shares_sellable"] is True
    assert changed["confirmation_basis"] == "manual_workday_scenario"
    assert changed["horizon"] == plan["horizon"]


def test_complete_calendar_is_conditional_and_missing_confirmation_switches_whole_schedule():
    calendar = _calendar("2026-10-28", "2026-11-06")
    calendar["2026-10-29"] = False  # Known business holiday, not inferred from NAV weekdays.
    arguments = {"decision_at": "2026-10-29T12:00:00+08:00", "anchor_date": "2026-10-28",
                 "calendar_days": 6, "entry_assumptions": {"cutoff": "15:00", "calendar_id": "test"}}
    original = deepcopy(calendar)
    plan = build_trade_plan(**arguments, calendar=calendar)
    assert calendar == original
    assert plan["calendar_basis"] == "available_profile_business_calendar"
    assert plan["application_date"] == "2026-10-30" and plan["sale_date"] == "2026-11-04"
    assert plan["entry_lag"] == 1 and plan["horizon"] == 3
    assert plan["assumed_open_dates"] == ["2026-11-02", "2026-11-03", "2026-11-04"]
    assert any("假设" in note and "净值" in note for note in plan["assumptions"])
    missing_window = deepcopy(calendar)
    missing_window.pop("2026-11-04")
    fallback = build_trade_plan(**arguments, calendar=missing_window)
    assert fallback["calendar_basis"] == "weekday_scenario"
    assert fallback["application_date"] == "2026-10-29"  # No hybrid business/weekday schedule.
    assert fallback["horizon"] == 4
    # Confirmation may extend beyond a short requested window; coverage there is required too.
    short = {**arguments, "calendar_days": 1}
    missing_confirmation = deepcopy(calendar)
    missing_confirmation.pop("2026-11-03")
    fallback = build_trade_plan(**short, calendar=missing_confirmation)
    assert fallback["calendar_basis"] == "weekday_scenario"
    assert fallback["first_missing_calendar_date"] == "2026-11-03"
    assert fallback["new_confirmation_date"] == "2026-11-02"


@pytest.mark.parametrize("days", [True, False, 0, -1, 91, 2.5, "30"])
def test_calendar_day_input_must_be_integer_within_control_range(days):
    with pytest.raises(ValidationError, match="calendar_days"):
        build_trade_plan(decision_at=NOW, anchor_date=None, calendar_days=days)


def test_calendar_inputs_use_actual_now_not_stale_anchor_and_exact_nonstandard_h(workspace, monkeypatch):
    work, rows = workspace
    called = []
    real_fit = core.fit_local_timing_forecast

    def counted(series, **kwargs):
        called.append(kwargs)
        return real_fit(series, **kwargs)

    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("passive local fit"))
    now_calls = []
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: now_calls.append(True) or NOW)
    inputs = work.timing_inputs(PID, calendar_days=9)
    assert len(now_calls) == 1 and inputs["status"] == "ready"
    plan = inputs["trade_plan"]
    assert plan["decision_date"] == "2026-10-28" and plan["deadline"] == "2026-11-06"
    assert plan["anchor_date"] == rows[-1]["date"] == "2026-10-26"
    assert plan["application_date"] == "2026-10-28"
    assert plan["sale_date"] == "2026-11-06" and plan["horizon"] == 7
    assert plan["entry_lag"] == inputs["entry_assumptions"]["baseline_entry_lag"] == 2
    monkeypatch.setattr(core, "fit_local_timing_forecast", counted)
    monkeypatch.setattr(core, "fit_runtime_forecast", lambda *a, **k: pytest.fail("legacy heads substituted"))
    before = list(work.store.conn.iterdump())
    now_calls.clear()
    result = work.calculate_timing(PID, calendar_days=9, include_history=True)
    assert len(now_calls) == 1 and len(called) == 1
    assert called[0]["horizons"] == (7,) and called[0]["entry_lags"] == (2, 3, 4)
    assert result["status"] == "ready" and result["cache_key"] == inputs["cache_key"]
    prediction = result["prediction"]
    assert prediction["requested_horizons"] == [7]
    assert prediction["model_hash"] == core.LOCAL_RUNTIME_MODEL_HASH != core.RUNTIME_MODEL_HASH
    assert prediction["algorithm_model_hash"] == core.RUNTIME_MODEL_HASH
    assert prediction["target_unit"] == result["calculation"]["target_unit"] == "nav_observations"
    assert result["calculation"]["local_contract"] == core.LOCAL_TIMING_CONTRACT
    assert {(h["horizon"], h["entry_lag"]) for h in
            prediction["forecasts"] + prediction["execution_forecasts"]} == {(7, lag) for lag in (0, 2, 3, 4)}
    assert len(result["historical_forecasts"]) == len(rows) * 4
    assert {r["horizon"] for r in result["historical_forecasts"]} == {7}
    assert prediction["entry_assumptions"] == result["entry_assumptions"]
    assert list(work.store.conn.iterdump()) == before and work.store.conn.total_changes == 0
    assert not work.store.has_forecast_schema()


def test_calendar_numeric_cache_reuses_same_h_l_not_ticking_time_or_window_dates(workspace, monkeypatch):
    work, _ = workspace
    friday = "2026-10-30T12:00:00.123456+08:00"
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: friday)
    one = work.timing_inputs(PID, calendar_days=7)  # Following Friday.
    same = work.timing_inputs(PID, calendar_days=8)  # Saturday rolls backward, not out of window.
    assert one["trade_plan"]["deadline"] != same["trade_plan"]["deadline"]
    assert one["trade_plan"]["horizon"] == same["trade_plan"]["horizon"] == 5
    assert one["cache_key"] == same["cache_key"]
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-30T12:00:01.654321+08:00")
    refreshed = work.timing_inputs(PID, calendar_days=7)
    assert refreshed["cache_key"] == one["cache_key"]
    assert refreshed["trade_plan"]["decision_at"] != one["trade_plan"]["decision_at"]
    changed_h = work.timing_inputs(PID, calendar_days=10)
    assert changed_h["cache_key"] != one["cache_key"]
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-30T15:00:00+08:00")
    after_cutoff = work.timing_inputs(PID, calendar_days=7)
    assert after_cutoff["entry_assumptions"]["baseline_entry_lag"] == one["trade_plan"]["entry_lag"] + 1
    assert after_cutoff["cache_key"] != one["cache_key"]
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: friday)
    assert work.timing_inputs(PID)["cache_key"] != one["cache_key"]  # Original official contract.
    assert work.timing_inputs(PID, model_id="exposure_factor_ridge_empirical_v1", calendar_days=7)["status"] == "insufficient"


@pytest.mark.parametrize("later", ["2026-10-28T15:00:00+08:00", "2026-10-29T00:00:01+08:00"])
def test_input_lookup_and_fit_share_explicit_captured_clock_across_cutoff_or_day(workspace, monkeypatch, later):
    work, _ = workspace
    captured = "2026-10-28T14:59:59+08:00"
    selected = work.timing_inputs(PID, calendar_days=9, decision_at=captured)
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: later)
    advanced = work.timing_inputs(PID, calendar_days=9)
    assert advanced["trade_plan"] != selected["trade_plan"]
    assert advanced["cache_key"] != selected["cache_key"]
    result = work.calculate_timing(PID, calendar_days=9, decision_at=captured)
    assert result["status"] == "ready"
    assert result["cache_key"] == selected["cache_key"]
    assert result["trade_plan"] == selected["trade_plan"]
    assert result["entry_assumptions"] == selected["entry_assumptions"]
    assert result["evaluated_at"] == result["prediction"]["generated_at"] == captured
    assert result["series"] == selected["series"]
    assert all(row["provenance"]["available_at"] == LOCAL for row in result["series"])


def test_too_short_window_keeps_dates_and_evidence_without_fitting_any_head(workspace, monkeypatch):
    work, _ = workspace
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-30T15:00:00+08:00")
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("too-short fit"))
    report = work.calculate_timing(PID, calendar_days=2, include_history=True)
    assert report["status"] == report["trade_plan"]["status"] == "too_short"
    assert report["prediction"] is None and report["historical_forecasts"] == []
    assert report["trade_plan"]["application_date"] == "2026-11-02"
    assert report["trade_plan"]["deadline"] == "2026-11-01"
    assert report["trade_plan"]["sale_date"] == "2026-10-30" and report["trade_plan"]["horizon"] == 0
    assert report["evidence"]["nav"]["observation_count"] > 0
    assert report["entry_assumptions"]["entry_lags"] == [5, 6, 7]


def test_service_reads_asof_calendar_through_deadline_confirmation_and_profile_rules(workspace, monkeypatch):
    work, _ = workspace
    product = work.store.get_product(PID)
    profile = Profile("test-calendar", PID, "2026-01-01", LOCAL,
                      {"cutoff": "14:00", "calendar_id": "synthetic-business", "confirm_days": 3},
                      URL + "/rules", verified=True)
    monkeypatch.setattr(work.store, "profiles", lambda *a: [profile])
    original_facts = work.store.facts
    read_calls = []
    records = [Fact("synthetic-business", "calendar", day, {"is_open": opened}, "synthetic",
                    LOCAL, URL + "/calendar") for day, opened in _calendar("2026-10-26", "2026-12-15").items()]

    def facts(pid=None, kind=None, as_of=None, start=None, end=None):
        if kind == "calendar":
            read_calls.append({"as_of": as_of, "start": start, "end": end})
            return [fact for fact in records if (not start or fact.event_date >= start)
                    and (not end or fact.event_date <= end)]
        return original_facts(pid, kind, as_of, start, end)

    monkeypatch.setattr(work.store, "facts", facts)
    report = work.timing_inputs(PID, calendar_days=30)
    plan = report["trade_plan"]
    assert report["status"] == "ready" and plan["deadline"] == "2026-11-27"
    assert plan["calendar_basis"] == "available_profile_business_calendar"
    assert plan["cutoff"] == "14:00" and plan["confirmation_days"] == 3
    assert plan["confirmation_basis"] == "verified_profile" and plan["new_confirmation_date"] == "2026-11-02"
    assert read_calls[-1]["as_of"] == NOW
    assert read_calls[-1]["end"] is None or read_calls[-1]["end"] >= plan["deadline"]
    # A verified confirmation rule remains useful when cutoff is unspecified.
    confirmation_only = Profile("confirmation-only", PID, "2026-01-01", LOCAL,
                                {"confirm_days": 4}, URL + "/confirmation", verified=True)
    monkeypatch.setattr(work.store, "profiles", lambda *a: [confirmation_only])
    plan, assumptions = work._calendar_trade_plan(product, NOW, "2026-10-26", 30)
    assert plan["confirmation_days"] == 4 and plan["confirmation_basis"] == "verified_profile"
    assert assumptions["cutoff_basis"] == "15:00_scenario"


def test_calendar_future_economic_records_use_actual_availability_not_today_cutoff(tmp_path, monkeypatch):
    root = tmp_path / "calendar-store"
    writer = Workspace(root, "public")
    writer.add_product("018147", "SYNTHETIC calendar fund", "open_fund", share_class="C")
    writer.store.add_profile(Profile("calendar-rule", PID, "2026-01-01", LOCAL,
                                     {"cutoff": "15:00", "calendar_id": "test-cal", "confirm_days": 2},
                                     URL + "/rule", verified=True))
    for row in _rows(90, "2026-07-29"):
        writer.store.add_fact(Fact(PID, "nav", row["date"], {"unit_nav": row["nav"]},
                                   "synthetic", LOCAL, URL))
    for day, opened in _calendar("2026-10-26", "2026-12-01").items():
        # A closure known only tomorrow must not change today's calendar plan.
        writer.store.add_fact(Fact("test-cal", "calendar", day, {"is_open": opened},
                                   "synthetic", LOCAL, URL + "/calendar"))
    writer.store.add_fact(Fact("test-cal", "calendar", "2026-11-04", {"is_open": False},
                               "synthetic", "2026-10-29T09:00:00+08:00", URL + "/calendar"))
    writer.store.close()
    reader = Workspace(root, "public", read_only=True, initialize=False)
    try:
        before = list(reader.store.conn.iterdump())
        result = reader.timing_inputs(PID, calendar_days=30)
        assert result["trade_plan"]["calendar_basis"] == "available_profile_business_calendar"
        assert result["trade_plan"]["horizon"] == 22
        assert result["trade_plan"]["deadline"] == "2026-11-27"
        assert list(reader.store.conn.iterdump()) == before and reader.store.conn.total_changes == 0
        monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-29T12:00:00+08:00")
        later = reader.timing_inputs(PID, calendar_days=29)
        assert later["trade_plan"]["deadline"] == "2026-11-27"
        assert later["trade_plan"]["horizon"] == 20  # One elapsed business day and the now-known closure.
        assert later["cache_key"] != result["cache_key"]
    finally:
        reader.store.close()


def test_calendar_factor_model_uses_actual_h_14_features_without_nav_fallback(workspace, monkeypatch):
    work, _ = workspace
    from rmbfund import investment_forecast as factor_core

    rows = _rows(340, "2025-11-21")
    inputs = _investment_inputs(rows)
    monkeypatch.setattr(work, "_investment_inputs", lambda *a: inputs)
    data = work._forecast_data(work.store.get_product(PID), NOW)
    data.update(series=rows, basis_hash="synthetic-qualified-calendar")
    monkeypatch.setattr(work, "_forecast_data", lambda *a: deepcopy(data))
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("factor fell back to NAV"))
    result = work.calculate_timing(PID, model_id=factor_core.MODEL_ID, calendar_days=9, include_history=True)
    assert result["status"] == "ready"
    prediction = result["prediction"]
    assert prediction["requested_horizons"] == [7]
    assert prediction["model_id"] == factor_core.MODEL_ID
    assert prediction["model_hash"] == factor_core.LOCAL_MODEL_HASH != factor_core.MODEL_HASH
    assert prediction["algorithm_model_hash"] == factor_core.MODEL_HASH
    assert prediction["feature_definition_hash"] == factor_core.FEATURE_DEFINITION_HASH
    assert all(len(state["feature_names"]) == 14 for state in prediction["model_state"])
    assert all(head["method"] == "ridge_prequential_empirical" for head in
               prediction["forecasts"] + prediction["execution_forecasts"])
    history = result["historical_forecasts"]
    assert len(history) == len(rows) * 4 and {r["horizon"] for r in history} == {7}
    assert history[-1]["prediction"]["status"] == "available"
    assert any(r["prediction"]["status"] == "unavailable" and r.get("historical_baseline") for r in history)
    for row in history[-4:]:
        head = next(h for h in prediction["forecasts"] + prediction["execution_forecasts"]
                    if (h["horizon"], h["entry_lag"]) == (row["horizon"], row["entry_lag"]))
        assert row["prediction"] == _compact(head)
    # Even a qualified input bundle cannot relabel immature empirical fallback as factors.
    short = rows[:90]
    inputs = _investment_inputs(short)
    data.update(series=short, basis_hash="synthetic-short-calendar")
    missing = work.calculate_timing(PID, model_id=factor_core.MODEL_ID, calendar_days=9, include_history=True)
    assert missing["status"] == "insufficient"
    assert not missing["prediction"]["forecasts"] and not missing["prediction"]["execution_forecasts"]
    assert len(missing["historical_forecasts"]) == len(short) * 4
    assert all(row["prediction"]["status"] != "available" for row in missing["historical_forecasts"])


def _sale_options():
    return {"calendar_days": 9, "old_confirmation_date": "2026-10-24",
            "fee_record": {"holding_day_basis": "confirmation_calendar_days", "redemption_bands": [
                {"min_days": 0, "max_days": 6, "rate": 0.015},
                {"min_days": 7, "max_days": 29, "rate": 0.005},
                {"min_days": 30, "max_days": None, "rate": 0},
            ]}}


def test_project_sale_uses_original_calendar_forward_and_counts_only_a_to_d():
    calendar = _calendar("2026-10-26", "2026-11-06")
    calendar["2026-11-02"] = False
    plan = build_trade_plan(decision_at=NOW, anchor_date="2026-10-26", calendar_days=9,
                            calendar=calendar)
    result = project_sale_date(plan, not_before="2026-10-31")
    assert result["sale_date"] == "2026-11-03"  # Not Friday Oct 30, nor closed Monday Nov 2.
    assert result["entry_lag"] == plan["entry_lag"] == 2
    assert result["horizon"] == 3  # Oct 29/30 and Nov 3; not anchor-to-D L+H.
    assert result["calendar_basis"] == "available_profile_business_calendar"
    assert "2026-11-02" not in plan["assumed_open_dates"]
    assert project_sale_date(plan, not_before="2026-11-07")["sale_date"] is None
    legacy = {key: value for key, value in plan.items() if key != "assumed_open_dates"}
    assert project_sale_date(legacy, not_before="2026-10-31")["status"] == "missing_calendar_projection"
    calendar.pop("2026-11-04")
    fallback = build_trade_plan(decision_at=NOW, anchor_date="2026-10-26", calendar_days=9,
                                calendar=calendar)
    assert fallback["calendar_basis"] == "weekday_scenario"
    assert project_sale_date(fallback, not_before="2026-10-31")["sale_date"] == "2026-11-02"
    assert "2026-11-02" in fallback["assumed_open_dates"]  # Whole-plan fallback, not a hybrid.


def test_sale_input_lookup_is_pure_exact_requests_match_existing_timing_keys(workspace, monkeypatch):
    work, _ = workspace
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("passive sale fit"))
    calls = []
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: calls.append(True) or NOW)
    before = list(work.store.conn.iterdump())
    inputs = work.sale_timing_inputs(PID, **_sale_options())
    assert len(calls) == 1
    requests = inputs["sale_requests"]
    assert [(row["horizon"], row["entry_lag"], row["sale_dates"]) for row in requests] == [
        (3, 2, ["2026-11-02"]), (7, 2, ["2026-11-06"])]
    assert all(row["entry_lags"] == [2, 3, 4] and not row["cached"] for row in requests)
    assert requests[0]["cache_key"] == work.timing_inputs(PID, calendar_days=5)["cache_key"]
    assert requests[1]["cache_key"] == inputs["cache_key"]
    original = deepcopy(inputs)
    rebuilt = work.rebuild_sale_timing(inputs, cached_predictions={"unrelated-key": {"execution_forecasts": []}})
    assert rebuilt == inputs == original
    assert list(work.store.conn.iterdump()) == before and work.store.conn.total_changes == 0
    row = rebuilt["sale_comparison"]["rows"][1]
    assert row["status"] == "fee_only" and row["fee_saving"] == 100
    assert row["break_even_change"] == pytest.approx(.985 / .995 - 1)


def test_sale_explicit_fit_only_missing_h_reuses_shared_cache_and_never_rereads_inputs(workspace, monkeypatch):
    work, _ = workspace
    current_plan = work.calculate_timing(PID, calendar_days=9, decision_at=NOW)
    inputs = work.sale_timing_inputs(PID, decision_at=NOW, **_sale_options())
    cache = {current_plan["cache_key"]: current_plan}
    original = deepcopy((inputs, cache))
    calls = []
    real_fit = core.fit_local_timing_forecast

    def counted(series, **kwargs):
        calls.append(kwargs)
        return real_fit(series, **kwargs)

    monkeypatch.setattr(core, "fit_local_timing_forecast", counted)
    monkeypatch.setattr(core, "fit_runtime_forecast", lambda *a, **k: pytest.fail("sale official fit"))
    monkeypatch.setattr(work, "timing_inputs", lambda *a, **k: pytest.fail("sale reread"))
    monkeypatch.setattr(work, "_calendar_trade_plan", lambda *a, **k: pytest.fail("sale calendar reread"))
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: pytest.fail("sale clock reread"))
    result = work.calculate_sale_timing(inputs, cached_predictions=cache)
    assert len(calls) == 1 and calls[0]["horizons"] == (3,)
    assert calls[0]["entry_lags"] == (2, 3, 4) and calls[0]["generated_at"] == NOW
    assert result["sale_calculation"]["fit_count"] == result["sale_calculation"]["reused_count"] == 1
    assert result["sale_comparison"]["status"] == "ready"
    assert len(result["sale_predictions"]) == len(result["sale_reports"]) == 2
    assert all(row["cached"] for row in result["sale_requests"])
    assert all(row["primary_head"]["horizon"] == row["horizon"] for row in result["sale_comparison"]["rows"][1:])
    assert all(prediction["generated_at"] == NOW for prediction in result["sale_predictions"].values())
    assert (inputs, cache) == original
    repeat = work.calculate_sale_timing(inputs, cached_predictions=result["sale_reports"])
    assert repeat["sale_calculation"]["fit_count"] == 0 and len(calls) == 1
    raw = work.rebuild_sale_timing(inputs, cached_predictions=result["sale_predictions"])
    assert raw["sale_comparison"] == repeat["sale_comparison"]
    assert work.store.conn.total_changes == 0 and not work.store.has_forecast_schema()


def test_sale_fee_edits_and_same_projected_day_reuse_keys_new_h_waits_for_explicit_fit(workspace, monkeypatch):
    work, _ = workspace
    inputs = work.sale_timing_inputs(PID, **_sale_options())
    fitted = work.calculate_sale_timing(inputs)
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("sale edit fitted"))
    edited = work.sale_timing_inputs(PID, **{**_sale_options(), "old_confirmation_date": "2026-10-25"},
                                      current_redemption_override=0.03,
                                      future_old_redemption_override=0.02, base_amount=5000)
    lookup = work.rebuild_sale_timing(edited, cached_predictions=fitted["sale_reports"])
    assert [row["cache_key"] for row in lookup["sale_requests"]] == [row["cache_key"] for row in fitted["sale_requests"]]
    assert all(row["cached"] for row in lookup["sale_requests"])
    assert [row["redemption_fee"] for row in lookup["sale_comparison"]["rows"]] == [0.03, .005, .02]
    changed = work.sale_timing_inputs(PID, **{**_sale_options(), "old_confirmation_date": "2026-10-27"})
    pending = work.rebuild_sale_timing(changed, cached_predictions=fitted["sale_reports"])
    assert [(row["horizon"], row["cached"]) for row in pending["sale_requests"]] == [(4, False), (7, True)]
    assert pending["sale_comparison"]["rows"][1]["better_probability"] is None
    assert pending["sale_comparison"]["rows"][1]["fee_saving"] == 100
    assert pending["sale_comparison"]["rows"][2]["better_probability"] is not None


def test_sale_d_equal_s_requests_one_head_and_too_short_or_missing_nav_never_fit(workspace, monkeypatch):
    work, _ = workspace
    inputs = work.sale_timing_inputs(PID, **{**_sale_options(), "calendar_days": 5})
    assert len(inputs["sale_requests"]) == 1
    assert inputs["sale_requests"][0]["candidate_ids"] == ["next_lower_fee", "plan"]
    fitted = work.calculate_sale_timing(inputs)
    assert fitted["sale_calculation"]["fit_count"] == 1
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("infeasible sale fitted"))
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-30T15:00:00+08:00")
    short = work.sale_timing_inputs(PID, **{**_sale_options(), "calendar_days": 2})
    assert short["status"] == "too_short" and short["sale_requests"] == []
    assert work.calculate_sale_timing(short)["sale_calculation"]["fit_count"] == 0
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: NOW)
    data = work._timing_data(work.store.get_product(PID), NOW, None)
    data["series"] = []
    monkeypatch.setattr(work, "_timing_data", lambda *a: deepcopy(data))
    missing = work.sale_timing_inputs(PID, **_sale_options())
    assert missing["status"] == "insufficient" and missing["sale_requests"] == []
    assert missing["sale_comparison"]["rows"][1]["fee_saving"] == 100
    assert missing["sale_comparison"]["rows"][1]["entry_lag"] is None
    assert work.calculate_sale_timing(missing)["sale_calculation"]["fit_count"] == 0


def test_sale_factor_heads_keep_original_qualification_no_nav_fallback(workspace, monkeypatch):
    work, rows = workspace
    from rmbfund import investment_forecast as factor_core

    investment = _investment_inputs(rows)
    monkeypatch.setattr(work, "_investment_inputs", lambda *a: investment)
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("sale factor fell back to NAV"))
    inputs = work.sale_timing_inputs(PID, model_id=factor_core.MODEL_ID, **_sale_options())
    result = work.calculate_sale_timing(inputs)
    assert result["sale_calculation"]["fit_count"] == 2
    for report in result["sale_reports"].values():
        assert report["status"] == "insufficient"
        assert report["selected_model_id"] == factor_core.MODEL_ID
        assert report["calculation"]["local_contract_hash"] == factor_core.LOCAL_MODEL_HASH
        assert report["prediction"] is None  # Existing qualification failure, not a substituted NAV model.
        assert any(reason.startswith("insufficient_history:") for reason in report["reasons"])
    assert all(row["status"] == "fee_only" and row["better_probability"] is None
               for row in result["sale_comparison"]["rows"][1:])
    assert result["sale_comparison"]["rows"][1]["fee_saving"] == 100
    assert work.calculate_sale_timing(inputs, cached_predictions=result["sale_reports"])["sale_calculation"]["fit_count"] == 0


def test_stored_nav_series_is_read_only_keeps_original_event_provenance_and_cutoff(workspace, monkeypatch):
    work, _ = workspace
    original = work._forecast_data(work.store.get_product(PID), NOW)
    before = list(work.store.conn.iterdump())
    monkeypatch.setattr(work, "_timing_data", lambda *a: pytest.fail("stored replay acquired online NAV"))
    monkeypatch.setattr(work, "timing_inputs", lambda *a, **k: pytest.fail("stored replay rebuilt forecast"))
    monkeypatch.setattr(core, "fit_local_timing_forecast", lambda *a, **k: pytest.fail("stored replay fitted"))
    result = work.stored_nav_series(PID, as_of=NOW)
    assert result["status"] == "ready" and result["evaluated_at"] == NOW
    assert result["series"] == original["series"]
    assert result["basis_hash"] == original["basis_hash"]
    assert result["series"][40]["invalid_return"] is True
    assert result["series"][0]["provenance"]["available_at"] == LOCAL
    earlier = work.stored_nav_series(PID, as_of="2026-10-25T12:00:00+08:00")
    assert earlier["status"] == "insufficient" and earlier["series"] == []
    future = work.stored_nav_series(PID, as_of="2026-10-29T12:00:00+08:00")
    assert future["status"] == "error" and future["series"] == []
    assert work.stored_nav_series("open_fund:missing:C")["status"] == "missing"
    assert list(work.store.conn.iterdump()) == before and work.store.conn.total_changes == 0

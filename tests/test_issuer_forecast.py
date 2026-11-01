"""Frozen18 pure offline regressions; synthetic fits are not investment evidence."""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund import forecast
from rmbfund.domain import ValidationError
from rmbfund.forecast_models import model_catalog
from rmbfund.investment_factors import SERIES_CONTRACT
from rmbfund.investment_forecast import MODEL_HASH as FACTOR_HASH
from rmbfund.issuer_fundamental_forecast import (
    FEATURE_DEFINITIONS, FEATURE_DEFINITION_HASH, MODEL_DEFINITION, MODEL_HASH, MODEL_ID,
    freeze_protocol18, qualify_then_run18,
)

CAPTURED = "2026-10-24T12:00:00+00:00"
AS_OF = "2026-10-25T12:00:00+00:00"
FROZEN = "2026-10-25T12:01:00+00:00"
LATER = "2026-10-26T12:00:00+00:00"
REFERENCE = {"source_url": "https://example.invalid/synthetic-methodology",
             "locator": "SYNTHETIC explicit test basis, not real financial evidence"}


def _inputs(count=660):
    rng = np.random.default_rng(8118)
    days, day = [], date(2024, 1, 2)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    t = np.arange(count)
    nav = {"product": {"id": "synthetic:issuer:C"}, "series": [
        {"date": day, "nav": str(value), "observed_at": CAPTURED,
         "available_at": CAPTURED, "historical_available_at": None}
        for day, value in zip(days, np.exp(np.cumsum(
            0.0004 + 0.003 * np.sin(t / 11) + rng.normal(0, 0.004, count))), strict=True)]}
    market = {"schema_version": 1, "series": {}}
    for role, (sid, currency, unit, basis) in SERIES_CONTRACT.items():
        values = (4 + 0.3 * np.sin(t / 13) if role.startswith("yield") else
                  50 * np.exp(0.0004 * t + 0.03 * np.sin(t / (11 if role == "industry" else 17))))
        market["series"][role] = {
            "series_id": sid, "label": f"SYNTHETIC {role}", "source": "synthetic_test",
            "source_url": f"https://example.invalid/{role}", "currency": currency,
            "unit": unit, "return_basis": basis, "captured_at": CAPTURED,
            "rows": [{"date": day, "value": float(value), "observed_at": CAPTURED,
                      "available_at": CAPTURED, "historical_available_at": None}
                     for day, value in zip(days, values, strict=True)],
        }
    issuer, prices = {"issuers": []}, {"securities": []}
    for prefix, accounting, shares in (("tsmc", "IFRS", "ADS"), ("nvidia", "GAAP", "ordinary")):
        sid = f"synthetic:{prefix}:USD"
        quarters = []
        for year in range(2022, 2027):
            for q in (1, 2, 3, 4):
                start = date(year, (q - 1) * 3 + 1, 1)
                next_start = date(year + 1, 1, 1) if q == 4 else date(year, q * 3 + 1, 1)
                published = next_start + timedelta(days=20)
                if published > date(2026, 10, 1):
                    continue
                release = f"SYNTHETIC:{prefix}:{year}Q{q}"
                quarters.append({
                    "release_id": release, "revision_id": "original-synthetic",
                    "revision_kind": "original", "record_id": release,
                    "published_at": published.isoformat(), "publication_precision": "date",
                    "observed_at": CAPTURED, "available_at": CAPTURED,
                    "historical_available_at": None, "source_url": REFERENCE["source_url"],
                    "value": str(0.1 * (year - 2020) + 0.04 * q + (0.2 if prefix == "tsmc" else 0)),
                    "metric": {"name": "eps", "category": "actual", "period_role": "primary",
                               "period": {"start": start.isoformat(),
                                          "end": (next_start - timedelta(days=1)).isoformat(),
                                          "basis": "quarter", "fiscal_year": year, "fiscal_quarter": q},
                               "currency": "USD", "unit": "currency_per_share", "scale": "1",
                               "eps_basis": "diluted", "accounting_basis": accounting,
                               "accounting_basis_detail": "synthetic comparable earnings",
                               "security_id": sid, "share_basis": shares,
                               "split_basis": "synthetic_unchanged_share_unit"}})
        issuer["issuers"].append({"issuer_id": f"issuer:{prefix}", "quarter_actuals": quarters,
            "basis_qualification": {"status": "qualified", "method": "native_currency",
                                    "reference": REFERENCE}})
        p_days, p_day = [], date(2022, 1, 1)
        while p_day.isoformat() <= days[-1]:
            if p_day.weekday() < 5:
                p_days.append(p_day.isoformat())
            p_day += timedelta(days=1)
        prices["securities"].append({
            "security_id": sid, "symbol": "TSM" if prefix == "tsmc" else "NVDA",
            "exchange": "SYNTHETIC", "currency": "USD", "share_basis": shares,
            "source_url": REFERENCE["source_url"], "captured_at": CAPTURED,
            "observed_at": CAPTURED, "available_at": CAPTURED, "historical_available_at": None,
            "currency_qualification": {"status": "verified", "reference": REFERENCE},
            "price_basis": {"kind": "unadjusted", "split_basis": "synthetic_unchanged_share_unit",
                            "reference": REFERENCE},
            "closes": [{"date": day, "close": 50 + i * 0.1}
                       for i, day in enumerate(p_days)]})
    return nav, market, issuer, prices


def _run(inputs):
    return qualify_then_run18(*inputs, freeze_protocol18(*inputs, frozen_at=FROZEN, as_of=AS_OF))


@pytest.fixture(scope="module")
def qualified():
    inputs = _inputs()
    return inputs, _run(inputs)


def test_exact18_hash_and_frozen_original_models(qualified):
    inputs, result = qualified
    assert len(FEATURE_DEFINITIONS) == 18
    assert len({item["name"] for item in FEATURE_DEFINITIONS}) == 18
    assert FEATURE_DEFINITION_HASH == forecast._digest(FEATURE_DEFINITIONS)
    assert MODEL_HASH == forecast._digest(MODEL_DEFINITION)
    assert MODEL_HASH == "13acc78278d3e1573563a685ca42bf4e3a7813246dec48727965e4e03b5c2618"
    assert FEATURE_DEFINITION_HASH == "1c613abdbf0f26aaf5c1a12c487eb54b22a56c53a4ce9a6fcf5944433aace114"
    assert MODEL_ID not in {row["id"] for row in model_catalog()}
    assert forecast.RUNTIME_MODEL_HASH == "7e9688b1549455c7e145eae6826a983b1857db835f2524e10ff2013522916474"
    assert FACTOR_HASH == "9e072f2994db03855c13ae2e66c04a1d0d639b4379aec36fdb5c53e4d6d0c39d"
    assert MODEL_HASH not in {forecast.MODEL_HASH, forecast.RUNTIME_MODEL_HASH, FACTOR_HASH}
    before = deepcopy(inputs)
    protocol = freeze_protocol18(*inputs, frozen_at=FROZEN, as_of=AS_OF)
    changed = deepcopy(protocol)
    changed["rules"]["primary_horizon"] = 5
    with pytest.raises(ValidationError, match="seal mismatch"):
        qualify_then_run18(*inputs, changed)
    mutated = deepcopy(inputs)
    mutated[0]["series"][-1]["nav"] = "2.0"
    with pytest.raises(ValidationError, match="seal mismatch"):
        qualify_then_run18(*mutated, protocol)
    assert inputs == before
    assert result["runtime_promoted"] is result["issued"] is False


def test_synthetic_qualified_run_real_fits_common_targets_and_maturity(qualified):
    inputs, result = qualified
    assert result["status"] == "fitted_research_only" and result["fitted"]
    assert result["strict_unseen"] is result["system_point_in_time"] is False
    assert result["joint_origins"]["nav_observation_count"] == len(inputs[0]["series"])
    assert result["independent_earnings_event_count"] < result["joint_origins"]["count"]
    fit = result["fit_results"]
    assert fit["evaluation_rows"] and len(fit["latest_heads"]) == 18
    for head in fit["latest_heads"]:
        if head["status"] == "fitted":
            assert len(head["model_state"]["feature_names"]) == {
                MODEL_ID: 18, "exposure_factor_ridge_empirical_v1": 14,
                "nav_ridge_empirical_v2": 7}[head["model_id"]]
            assert head["forecast"]["method"] == "ridge_prequential_empirical"
    for row in fit["evaluation_rows"]:
        assert row["target_index"] - row["entry_index"] == row["horizon"]
        for key in ("prediction", "frozen14_prediction", "nav7_prediction"):
            assert row[key]["method"] == "ridge_prequential_empirical"
            assert row[key]["training_target_end_index"] <= row["origin_index"]
            assert row[key]["residual_target_end_index"] <= row["origin_index"]
            # Same complete18 training origin mask, not only matching test rows.
            assert row[key]["training_sample_count"] == row["prediction"]["training_sample_count"]
        assert row["period"] != "heldout_2026"
    primary = next(h for h in fit["common_origin_comparison"]
                   if h["horizon"] == 20 and h["entry_lag"] == 0)
    metrics = primary["periods"]["inspected_2026"]
    assert 0 < metrics["nonoverlap"]["count"] < metrics["joint_fitted_origins"]
    assert all(name in metrics["rolling"]["prediction"]
               for name in ("up_brier", "mae", "up_calibration", "interval_coverage"))


def test_actual_capture_and_availability_never_become_historical(qualified):
    inputs, result = qualified
    late = deepcopy(inputs)
    for company in late[2]["issuers"]:
        for row in company["quarter_actuals"]:
            row["available_at"] = LATER
    unavailable = _run(late)
    assert unavailable["status"] == "qualification_unavailable"
    assert unavailable["fit_results"] is None and not unavailable["fitted"]
    assert unavailable["independent_earnings_event_count"] == 0
    assert all(not value["selected_quarters"] for value in unavailable["independent_event_counts"].values())
    # Today's downloaded historical releases remain useful for declared
    # reconstruction, while the actual clocks stay in every selected quarter.
    quarter = result["independent_event_counts"]["issuer:tsmc"]["selected_quarters"][0]
    assert quarter["observed_at"] == CAPTURED and quarter["available_at"] == CAPTURED
    assert quarter["publication_precision"] == "date" and quarter["historical_available_at"] is None
    missing_capture = deepcopy(inputs)
    del missing_capture[0]["series"][0]["observed_at"]
    failed = _run(missing_capture)
    assert failed["input_qualifications"]["nav"]["status"] == "unavailable"
    assert failed["fit_results"] is None


def test_unknown_adjustment_translation_or_missing_market_stops_before_any_fit(qualified, monkeypatch):
    inputs, _ = qualified
    def forbidden(*args, **kwargs):
        raise AssertionError("qualification failure must not trigger any old/new model race")
    monkeypatch.setattr(forecast, "_fit_head", forbidden)
    unknown = deepcopy(inputs)
    unknown[3]["securities"][0]["price_basis"]["kind"] = "unknown"
    failed = _run(unknown)
    assert not failed["fitted"] and failed["fit_results"] is None
    assert failed["input_qualifications"]["security_prices"]["issuer:tsmc"]["basis_qualified_rows"] == 0
    unknown = deepcopy(inputs)
    unknown[2]["issuers"][0]["basis_qualification"]["status"] = "unresolved"
    assert _run(unknown)["fit_results"] is None
    missing = deepcopy(inputs)
    del missing[1]["series"]["fx"]
    assert _run(missing)["fit_results"] is None


def test_no_lookahead_missing_quarter_not_zero_or_older_complete_ttm(qualified):
    inputs, original = qualified
    changed = deepcopy(inputs)
    company = changed[2]["issuers"][0]
    # Remove the preceding quarter while keeping the newest eligible release.
    company["quarter_actuals"] = [row for row in company["quarter_actuals"]
                                 if row["release_id"] != "SYNTHETIC:tsmc:2025Q4"]
    late_eps = next(row for row in company["quarter_actuals"]
                    if row["release_id"] == "SYNTHETIC:tsmc:2026Q2")
    late_eps["value"] = "99.0"
    changed_result = _run(changed)
    assert len(changed_result["feature_rows"]) == len(original["feature_rows"])
    key = "tsmc_ttm_eps_div_price"
    for before, after in zip(original["feature_rows"], changed_result["feature_rows"], strict=True):
        if before["date"] < "2026-01-22":
            assert before["features"] == after["features"]
        if "2026-04-24" <= before["date"] <= "2026-07-20":
            assert after["features"][key] is None
    for row in original["feature_rows"]:
        trace = row["issuer_alignment"]["tsmc"]
        if trace.get("release_id") == "SYNTHETIC:tsmc:2026Q2":
            assert row["date"] >= "2026-07-23"  # release date21 + two calendar days
            assert trace["asof_price"]["date"] <= trace["origin_public_cutoff"]
    assert any(row["features"][key] is None for row in changed_result["feature_rows"])


def test_qualified_but_short_history_is_not_a_fake_fit():
    result = _run(_inputs(count=100))
    assert result["status"] == "qualified_fit_unavailable"
    assert not result["fitted"] and result["fit_results"] is not None
    assert result["fit_results"]["common_fitted_evaluation_row_count"] == 0
    assert all(head["forecast"] is None for head in result["fit_results"]["latest_heads"])

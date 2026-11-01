"""Synthetic pure-core regressions; no public artifacts or investment-skill claims."""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund import forecast as nav_core
from rmbfund.investment_factors import (
    SERIES_CONTRACT, align_investment_features, factor_input_hash, normalize_factor_bundle,
)
from rmbfund.investment_forecast import MODEL_ID, fit_investment_forecast

PRODUCT = "synthetic:emerging-tech:C"
ACQUIRED = "2026-10-24T12:00:00+00:00"
ISSUED = "2026-10-25T12:00:00+00:00"
LATER = "2026-10-26T12:00:00+00:00"


def _inputs(count=340):
    rng, days, day = np.random.default_rng(7021), [], date(2025, 1, 2)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    t = np.arange(count)
    market = 50 * np.exp(np.cumsum(0.001 + 0.006 * np.sin(t / 9) + rng.normal(0, 0.004, count)))
    values = {
        "market": market,
        "industry": 100 * np.exp(np.cumsum(0.001 + 0.009 * np.sin(t / 11) + rng.normal(0, 0.006, count))),
        "fx": 7 * np.exp(0.01 * np.sin(t / 30)),
        "yield_2y": 4 + 0.2 * np.sin(t / 17),
        "yield_10y": 4 + 0.3 * np.sin(t / 23),
    }
    navs = np.exp(np.cumsum(0.0003 + 0.004 * np.sin(t / 9) + rng.normal(0, 0.003, count)))
    series = [{"date": day, "nav": str(value), "currency": "CNY", "available_at": ACQUIRED}
              for day, value in zip(days, navs, strict=True)]
    bundle = {"schema_version": 1, "series": {}}
    for role, (sid, currency, unit, basis) in SERIES_CONTRACT.items():
        bundle["series"][role] = {
            "series_id": sid, "label": f"SYNTHETIC {role}", "source": "synthetic_test",
            "source_url": f"https://example.invalid/{role}", "currency": currency,
            "unit": unit, "return_basis": basis, "captured_at": ACQUIRED,
            "rows": [{"date": day, "value": float(value), "available_at": ACQUIRED,
                      "historical_available_at": None, "source_sha256": "a" * 64}
                     for day, value in zip(days, values[role], strict=True)],
        }
    profile = {
        "schema_version": 1, "product_id": PRODUCT, "as_of": ISSUED, "factor_mapping": [],
        "portfolio": {"record_id": "synthetic-portfolio", "report_date": "2026-06-30",
                      "available_at": ACQUIRED, "payload": {
                          "denominator": "nav", "assets": {"equity": "0.8"}, "holdings": [],
                          "sectors": [{"name": "信息技术", "weight": "0.7"}]}},
        "mandate": {"record_id": "synthetic-mandate", "report_date": "2026-06-30",
                    "available_at": ACQUIRED, "payload": {
                        "asset_class": "mixed", "base_currency": "CNY",
                        "benchmark": {"region": "emerging_markets"}}},
    }
    return series, bundle, profile


def _fit(series, factors, profile, **kwargs):
    return fit_investment_forecast(series, factors=factors, profile=profile, product_id=PRODUCT,
                                   generated_at=ISSUED, **kwargs)


def _heads(result):
    return {(h["horizon"], h["entry_lag"]): h
            for h in result["forecasts"] + result["execution_forecasts"]}


def test_factors_change_fitted_numbers_with_own_metadata_without_changing_nav_default():
    series, bundle, profile = _inputs()
    nav_before = nav_core.fit_runtime_forecast(series, product_id=PRODUCT, generated_at=ISSUED)
    result = _fit(series, bundle, profile)
    diagnostic = _fit(series, bundle, profile, include_evaluation=True)
    assert result["validation"] == diagnostic["validation"]
    assert result["validation"]["forecast_count"] > 0
    assert "evaluation_rows" not in result and "issued_forecasts" not in result
    perturbed = deepcopy(bundle)
    cutoff = (date.fromisoformat(series[-1]["date"]) - timedelta(days=2)).isoformat()
    last = next(r for r in reversed(perturbed["series"]["market"]["rows"]) if r["date"] <= cutoff)
    last["value"] *= 1.1
    changed = _fit(series, perturbed, profile)
    assert result["model_id"] == MODEL_ID and len(_heads(result)) == 12
    assert not result["unavailable_forecasts"] and result["input_hash"] != changed["input_hash"]
    assert any(abs(h["expected_change"] - _heads(changed)[identity]["expected_change"]) > 1e-6
               for identity, h in _heads(result).items())
    states = {(s["horizon"], s["entry_lag"]): s for s in result["model_state"]}
    for identity, head in _heads(result).items():
        assert head["method"] == "ridge_prequential_empirical"
        assert len(head["drivers"]) == len(states[identity]["feature_names"]) == 14
        assert all({"label", "unit", "economic_role", "series_ids", "observation_ends",
                    "availability_basis", "current_feature_value"} <= d.keys() for d in head["drivers"])
        assert states[identity]["intercept"] + sum(d["contribution_log_return"] for d in head["drivers"]) == (
            pytest.approx(head["ridge_log_location"], abs=1e-13))
        assert head["historical_baseline"]["method"] == "empirical_historical"
        for kind in ("training", "residual"):
            assert head[f"{kind}_target_end_index"] <= len(series) - 1
            assert head[f"{kind}_target_end_date"] == series[head[f"{kind}_target_end_index"]]["date"]
    nav_after = nav_core.fit_runtime_forecast(series, product_id=PRODUCT, generated_at=ISSUED)
    assert nav_before == nav_after
    assert nav_after["model_hash"] == "7e9688b1549455c7e145eae6826a983b1857db835f2524e10ff2013522916474"
    assert nav_after["model_id"] == "nav_ridge_empirical_v2"
    assert len(nav_after["features"]) == 7 and all(len(h["drivers"]) == 7 for h in _heads(nav_after).values())


def test_actual_availability_excludes_future_rows_and_hash_ignores_only_refresh_receipts():
    series, bundle, profile = _inputs()
    future = deepcopy(bundle)
    for role, source in future["series"].items():
        bundle["series"][role]["rows"] = bundle["series"][role]["rows"][:-3]
        source["captured_at"] = LATER
        for row in source["rows"][-3:]:
            row.update(value=row["value"] * 1.2, available_at=LATER, version_id="new-version")
        for row in source["rows"]:
            row["source_sha256"] = "b" * 64
    original = deepcopy((series, bundle, profile))
    selected = normalize_factor_bundle(future, as_of=ISSUED)
    assert factor_input_hash(selected) == factor_input_hash(bundle)
    assert all(len(s["rows"]) == len(series) - 3 for s in selected["series"].values())
    assert factor_input_hash(normalize_factor_bundle(future, as_of=LATER)) != factor_input_hash(selected)
    first, changed = align_investment_features(series, bundle)[0], align_investment_features(series, selected)[0]
    np.testing.assert_allclose(first, changed, equal_nan=True)
    before, after = _fit(series, bundle, profile, entry_lags=()), _fit(series, future, profile, entry_lags=())
    assert before["input_hash"] == after["input_hash"] and _heads(before) == _heads(after)
    version = deepcopy(selected)
    version["series"]["market"]["rows"][0]["version_id"] = "actual-new-version"
    assert factor_input_hash(version) != factor_input_hash(selected)
    query_refresh = deepcopy(selected)
    query_refresh["series"]["market"]["source_url"] += "?todate=2026-10-25"
    assert factor_input_hash(query_refresh) == factor_input_hash(selected)
    path_change = deepcopy(selected)
    path_change["series"]["market"]["source_url"] += "/different-endpoint"
    assert factor_input_hash(path_change) != factor_input_hash(selected)
    host_change = deepcopy(selected)
    host_change["series"]["market"]["source_url"] = "https://other.invalid/market"
    assert factor_input_hash(host_change) != factor_input_hash(selected)
    assert (series, bundle, profile) == original


def test_local_factor_exact_h_keeps_qualification_features_and_fixed_span_history():
    from rmbfund import investment_forecast as factors_core

    series, bundle, profile = _inputs()
    original = _fit(series, bundle, profile, entry_lags=(0, 2))
    local = factors_core.fit_local_investment_forecast(
        series, factors=bundle, profile=profile, product_id=PRODUCT, generated_at=ISSUED,
        horizons=(7,), entry_lags=(0, 2), include_history=True,
    )
    assert original == _fit(series, bundle, profile, entry_lags=(0, 2))
    assert original["model_hash"] == factors_core.MODEL_HASH
    assert local["model_hash"] == factors_core.LOCAL_MODEL_HASH != original["model_hash"]
    assert local["feature_definition_hash"] == original["feature_definition_hash"]
    assert len(local["features"]) == 14 and all(len(state["feature_names"]) == 14 for state in local["model_state"])
    assert {(head["horizon"], head["entry_lag"]) for head in _heads(local).values()} == {(7, 0), (7, 2)}
    assert all(head["method"] == "ridge_prequential_empirical" for head in _heads(local).values())
    assert len(local["historical_forecasts"]) == len(series) * 2
    origin = 280
    prefix = factors_core.fit_local_investment_forecast(
        series[:origin + 1], factors=bundle, profile=profile, product_id=PRODUCT, generated_at=ISSUED,
        horizons=(7,), entry_lags=(0, 2), include_history=True,
    )
    changed = deepcopy(series)
    for row in changed[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.2)
    changed[origin + 3]["invalid_return"] = True
    after = factors_core.fit_local_investment_forecast(
        changed, factors=bundle, profile=profile, product_id=PRODUCT, generated_at=ISSUED,
        horizons=(7,), entry_lags=(0, 2), include_history=True,
    )
    before_rows = [r for r in local["historical_forecasts"] if r["origin_index"] <= origin]
    assert before_rows == [r for r in after["historical_forecasts"] if r["origin_index"] <= origin]
    for full, cut in zip(before_rows, prefix["historical_forecasts"], strict=True):
        assert full["prediction"] == cut["prediction"]
        assert full.get("historical_baseline") == cut.get("historical_baseline")
        for kind in ("training", "residual"):
            endpoint = full["prediction"].get(f"{kind}_target_end_index")
            assert endpoint is None or endpoint <= full["origin_index"]
    broken = deepcopy(profile)
    broken["mandate"]["payload"]["benchmark"]["region"] = "unrelated"
    with pytest.raises(ValueError, match="does not match"):
        factors_core.fit_local_investment_forecast(
            series, factors=bundle, profile=broken, product_id=PRODUCT, generated_at=ISSUED,
            horizons=(7,), include_history=True,
        )

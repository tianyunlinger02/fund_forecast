"""Synthetic recovery-feature checks, not evidence of investment performance."""

from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pytest

from rmbfund import forecast as core
from rmbfund import forecast_candidates as candidates


def _series(count=420, *, start=date(2024, 6, 1), seed=61031):
    rng = np.random.default_rng(seed)
    increments = 0.0003 + 0.003 * np.sin(np.arange(count) / 11) + rng.normal(0, 0.008, count)
    return [
        {"date": (start + timedelta(days=i)).isoformat(), "nav": str(nav)}
        for i, nav in enumerate(np.exp(np.cumsum(increments)))
    ]


def _identity(row):
    return row["origin_index"], row["horizon"], row["entry_lag"]


def _head_key(row):
    return row["horizon"], row["entry_lag"]


def _valid_label(rows, origin, horizon, lag):
    endpoint = origin + horizon + lag
    return endpoint < len(rows) and not any(
        row.get("invalid_return", False) or row.get("label_valid") is False
        for row in rows[origin + lag + 1:endpoint + 1]
    )


def _training_origins(rows, features, horizon, lag):
    return np.asarray([
        origin for origin in range(len(rows) - horizon - lag)
        if np.all(np.isfinite(features[origin])) and _valid_label(rows, origin, horizon, lag)
    ], dtype=int)


def test_recovery_feature_preserves_nav7_nan_event_policy_and_maximum_slice():
    rows = _series(145)
    nav = np.ones(len(rows))
    nav[0] = 2.0  # At origin 60, the existing maximum includes origin - 60.
    nav[56:61] = [1.02, 1.05, 1.10, 1.15, 1.20]
    nav[65], nav[70] = 0.9, 3.0
    for row, value in zip(rows, nav, strict=True):
        row["nav"] = str(value)
    rows[0]["invalid_return"] = True  # Incoming move at the start boundary is excluded.
    rows[75]["invalid_return"] = True
    rows[80]["label_valid"] = False
    days, logs, original, invalid = candidates._inputs(rows)
    recovery_days, recovery_logs, features, recovery_invalid = candidates._inputs(rows, recovery=True)
    assert recovery_days == days
    np.testing.assert_array_equal(recovery_logs, logs)
    np.testing.assert_array_equal(recovery_invalid, invalid)
    np.testing.assert_array_equal(features[:, :7], original)
    np.testing.assert_array_equal(original, core._feature_matrix(logs, invalid))
    assert features.shape == (len(rows), 8)
    assert features[60, 1] == pytest.approx(np.log(1.2))
    assert features[60, 6] == pytest.approx(-0.4)
    assert features[60, 7] == pytest.approx(np.log(1.2) * 0.4)
    assert features[65, 1] < 0 and features[65, 6] < 0 and features[65, 7] == 0
    assert features[70, 1] > 0 and features[70, 6] == 0 and features[70, 7] == 0
    assert np.all(np.isnan(features[:60]))
    assert np.all(np.isfinite(features[74]))
    assert np.all(np.isnan(features[75:140]))
    assert np.all(np.isfinite(features[140]))
    np.testing.assert_array_equal(
        features[:, 7], np.maximum(original[:, 1], 0) * np.maximum(-original[:, 6], 0),
    )


def test_recovery_full_history_prefix_and_future_nav_event_invariance():
    rows, origin = _series(), 350
    rows[270]["invalid_return"] = True
    rows[275]["label_valid"] = False
    original = candidates.issue_recovery_forecasts(rows)
    prefix = candidates.issue_recovery_forecasts(rows[:origin + 1])
    changed = copy.deepcopy(rows)
    for row in changed[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.4)
    changed[origin + 5]["invalid_return"] = True
    changed[origin + 7]["label_valid"] = False
    future = candidates.issue_recovery_forecasts(changed)
    history = original["historical_forecasts"]
    assert len(history) == len(rows) * len(candidates.SHRINK_HEADS)
    assert [_identity(row) for row in history] == sorted(_identity(row) for row in history)
    assert len({_identity(row) for row in history}) == len(history)
    past = [row for row in history if row["origin_index"] <= origin]
    assert past == [row for row in future["historical_forecasts"] if row["origin_index"] <= origin]
    for full, cut in zip(past, prefix["historical_forecasts"], strict=True):
        assert full["prediction"] == cut["prediction"]
        assert full.get("historical_baseline") == cut.get("historical_baseline")
    assert [row for row in original["issued_forecasts"] if row["origin_index"] <= origin] == [
        row for row in future["issued_forecasts"] if row["origin_index"] <= origin
    ]
    for row in history:
        assert not {"actual_change", "actual_log_return", "target_date", "issued_at"} & row.keys()
        for kind in ("training", "residual"):
            endpoint = row["prediction"][f"{kind}_target_end_index"]
            assert endpoint is None or endpoint <= row["origin_index"]
    assert history[-1]["origin_index"] == len(rows) - 1
    assert len(future["historical_forecasts"]) == len(history)


def test_own_nav8_fit_shrunk_residual_maturity_and_head_specific_nonzero_counts():
    rows = _series(480, start=date(2025, 1, 1))
    _, _, clean_features, _ = candidates._inputs(rows, recovery=True)
    event_origin = next(i for i in range(280, 300) if clean_features[i, 7] > 0)
    rows[event_origin + 4]["invalid_return"] = True
    rows[355]["label_valid"] = False
    _, logs, features, _ = candidates._inputs(rows, recovery=True)
    result = candidates.issue_recovery_forecasts(rows)
    old = candidates.issue_shrinkage_forecasts(rows)
    old_heads = {_head_key(head): head for head in old["forecasts"]}
    assert _valid_label(rows, event_origin, 20, 4)
    assert not _valid_label(rows, event_origin, 20, 3)
    for head in result["forecasts"]:
        horizon, lag = _head_key(head)
        span = horizon + lag
        history = [row for row in result["historical_forecasts"] if _head_key(row) == (horizon, lag)]
        qualified = _training_origins(rows, features, horizon, lag)
        x = features[qualified]
        y = logs[qualified + span] - logs[qualified + lag]
        means, scales = x.mean(axis=0), x.std(axis=0)
        scales[scales <= 1e-12] = 1
        z = (x - means) / scales
        coefficients = np.linalg.solve(z.T @ z + 10 * np.eye(8), z.T @ (y - y.mean()))
        unshrunk = float(y.mean() + ((features[-1] - means) / scales) @ coefficients)
        assert coefficients[-1] < 0  # A negative recovery coefficient must not be clipped away.
        assert head["unshrunk_recovery_log_location"] == pytest.approx(unshrunk, abs=1e-13)
        assert head["unshrunk_recovery_log_location"] != pytest.approx(
            old_heads[horizon, lag]["old_ridge_log_location"], abs=1e-8,
        )
        assert head["training_sample_count"] == len(qualified)
        assert head["training_log_mean"] == pytest.approx(y.mean(), abs=1e-15)
        assert head["conditional_log_contribution"] == (
            head["unshrunk_recovery_log_location"] - head["training_log_mean"]
        )
        assert head["candidate_log_location"] == (
            head["training_log_mean"] + 0.5 * head["conditional_log_contribution"]
        )
        for row in history:
            origin = row["origin_index"]
            matured = qualified[qualified + span <= origin]
            assert row["prediction"]["training_sample_count"] == len(matured)
            assert row["prediction"]["matured_nonzero_recovery_training_count"] == int(
                np.count_nonzero(features[matured, 7] > 0)
            )
        admission = event_origin + span
        count_key = "matured_nonzero_recovery_training_count"
        assert (history[admission]["prediction"][count_key]
                - history[admission - 1]["prediction"][count_key]) == (1 if lag == 4 else 0)
        errors = [
            logs[row["origin_index"] + span] - logs[row["origin_index"] + lag]
            - row["prediction"]["candidate_log_location"]
            for row in history
            if row["prediction"]["candidate_log_location"] is not None
            and _valid_label(rows, row["origin_index"], horizon, lag)
        ]
        samples = np.expm1(head["candidate_log_location"] + np.asarray(errors))
        np.testing.assert_array_equal(head["samples"], samples)
        assert head["sample_count"] == head["residual_sample_count"] == len(errors)
        assert head["residual_log_mean"] == float(np.mean(errors))
        assert head["expected_change"] == float(np.sum(samples / len(samples)))
        assert head["up_probability"] == np.mean(samples > 0)
        assert head["down_probability"] == np.mean(samples < 0)
        assert head["flat_probability"] == np.mean(samples == 0)
        np.testing.assert_array_equal(
            [head["lower"], head["median_change"], head["upper"]],
            np.quantile(samples, [0.1, 0.5, 0.9]),
        )
        first_location = next(row["origin_index"] for row in history
                              if row["prediction"]["candidate_log_location"] is not None)
        assert history[first_location + span - 1]["prediction"]["residual_sample_count"] == 0
        assert history[first_location + span]["prediction"]["residual_sample_count"] == 1
        assert history[first_location + span + 38]["prediction"]["status"] == "unavailable"
        first_available = history[first_location + span + 39]["prediction"]
        assert first_available["status"] == "available"
        assert first_available["residual_sample_count"] == 40


def test_recovery_identity_metadata_baselines_and_no_volatility_or_fallback():
    definition = candidates.candidate_definition(candidates.RECOVERY_ID)
    assert candidates.RECOVERY_ID == "nav_ridge_recovery_shrink50_empirical_v1"
    assert candidates.RECOVERY_METHOD == "ridge_recovery_shrink50_prequential_empirical"
    assert definition["method"] == candidates.RECOVERY_METHOD
    assert definition["shrink_weight"] == 0.5
    assert definition["minimum_training_samples"] == 120
    assert definition["minimum_residual_samples"] == 40
    assert definition["location_definition"]["ridge_alpha"] == 10.0
    assert definition["location_definition"]["features"][:7] == list(core.FEATURE_DEFINITIONS)
    assert definition["location_model_hash"] == core._digest(definition["location_definition"])
    assert definition["location_model_hash"] != core.MODEL_HASH
    assert definition["runtime_issuance"] is definition["promotion"] is False
    assert definition["hyperparameter_search"] is False
    assert candidates.candidate_hash(candidates.RECOVERY_ID) not in (
        candidates.candidate_hash(candidates.SHRINK_ID), core.MODEL_HASH, core.RUNTIME_MODEL_HASH,
    )
    assert candidates.RECOVERY_ID not in core._canonical(core.RUNTIME_MODEL_DEFINITION)
    rows = _series(320)
    result = candidates.issue_recovery_forecasts(rows)
    old = candidates.issue_shrinkage_forecasts(rows)
    assert result.keys() == old.keys()
    assert result["candidate_hash"] == candidates.candidate_hash(candidates.RECOVERY_ID)
    assert candidates.issue_candidate_forecasts(
        rows, candidate_id=candidates.RECOVERY_ID, heads=candidates.SHRINK_HEADS,
    ) == result
    predictions = [*result["forecasts"], *result["unavailable_forecasts"]]
    for section in ("historical_forecasts", "issued_forecasts", "evaluation_rows"):
        for row, reference in zip(result[section], old[section], strict=True):
            assert _identity(row) == _identity(reference)
            assert row.get("historical_baseline") == reference.get("historical_baseline")
            assert row["prediction"].get("historical_baseline") == reference["prediction"].get(
                "historical_baseline"
            )
            predictions.append(row["prediction"])
    for head, reference in zip(result["forecasts"], old["forecasts"], strict=True):
        assert head["historical_baseline"] == reference["historical_baseline"]
    for prediction in predictions:
        assert prediction["method"] == candidates.RECOVERY_METHOD
        assert prediction["status"] in ("available", "unavailable")
        assert prediction["fallback_used"] is False
        assert "old_ridge_log_location" not in prediction
        assert "risk_scale" not in prediction
        assert {"unshrunk_recovery_log_location", "log_return_5", "simple_drawdown_60",
                "recovery_feature", "matured_nonzero_recovery_training_count"} <= prediction.keys()
    for length in (1, 180):
        short = candidates.issue_recovery_forecasts(_series(length, start=date(2025, 1, 1)))
        assert not short["forecasts"] and len(short["unavailable_forecasts"]) == 2
        assert len(short["historical_forecasts"]) == length * 2
        for prediction in [*short["unavailable_forecasts"], *(
            row["prediction"] for row in short["historical_forecasts"]
        )]:
            assert prediction["method"] == candidates.RECOVERY_METHOD
            assert prediction["status"] == "unavailable"
            assert prediction["fallback_used"] is False and "expected_change" not in prediction
    for row in rows:
        row["nav"] = "1"
    flat = candidates.issue_recovery_forecasts(rows)
    assert len(flat["forecasts"]) == 2
    assert all(head["expected_change"] == head["candidate_log_location"] == 0
               and head["matured_nonzero_recovery_training_count"] == 0
               for head in flat["forecasts"])
    rows[-1]["invalid_return"] = True
    invalid = candidates.issue_recovery_forecasts(rows)
    assert not invalid["forecasts"]
    assert all(head["method"] == candidates.RECOVERY_METHOD
               and head["reason"] == "feature_window_crosses_invalid_event_or_shorter_than_60"
               and head["recovery_feature"] is None
               and head["historical_baseline"]["method"] == "empirical_historical"
               for head in invalid["unavailable_forecasts"])


def test_existing_outputs_and_shared_ridge_kernel_contract_unchanged(monkeypatch):
    rows = _series(330, start=date(2025, 1, 1), seed=61030)
    definitions = core._canonical((core.MODEL_DEFINITION, core.RUNTIME_MODEL_DEFINITION))
    calls = []
    original_fit_head = candidates._fit_head

    def fit_head(*args, **kwargs):
        calls.append((args[2].shape[1], kwargs.copy()))
        return original_fit_head(*args, **kwargs)

    monkeypatch.setattr(candidates, "_fit_head", fit_head)
    heads = ((1, 0), (5, 0), (20, 0), *candidates.SHRINK_HEADS)
    before = {
        candidate_id: candidates.issue_ridge_distribution_forecasts(
            rows, candidate_id=candidate_id, heads=heads,
        ) for candidate_id in (core.MODEL_ID, candidates.RISK_ID, candidates.SHRINK_ID)
    }
    assert all(width == 7 and options["feature_definitions"] == core.FEATURE_DEFINITIONS
               for width, options in calls)
    trend = candidates.issue_candidate_forecasts(
        rows, candidate_id=candidates.TREND_ID, heads=candidates.SHRINK_HEADS, include_history=True,
    )
    calls.clear()
    candidates.issue_recovery_forecasts(rows)
    assert len(calls) == 2
    for width, options in calls:
        assert width == 8
        assert options["feature_definitions"] == candidates.RECOVERY_FEATURE_DEFINITIONS
        assert options["shrink_location"] is True and options["normalize_residuals"] is False
        assert "fitted_history_only" not in options
    after = {
        candidate_id: candidates.issue_ridge_distribution_forecasts(
            rows, candidate_id=candidate_id, heads=heads,
        ) for candidate_id in before
    }
    assert after == before
    assert candidates.issue_candidate_forecasts(
        rows, candidate_id=candidates.TREND_ID, heads=candidates.SHRINK_HEADS, include_history=True,
    ) == trend
    assert core._canonical((core.MODEL_DEFINITION, core.RUNTIME_MODEL_DEFINITION)) == definitions
    official = core.fit_forecast(
        rows, product_id="synthetic:recovery", generated_at="2026-10-30T00:00:00Z",
        entry_lags=(3, 4), include_evaluation=True,
    )
    assert before[core.MODEL_ID]["evaluation_rows"] == official["evaluation_rows"]
    assert before[core.MODEL_ID]["issued_forecasts"] == official["issued_forecasts"]

"""SYNTHETIC contract/chronology tests; no claim of improved investment forecasts."""

from __future__ import annotations


import copy


import socket


from datetime import date, timedelta


import numpy as np


import pytest


from rmbfund import forecast as core


GENERATED_AT = "2026-10-22T12:00:00+08:00"


RUNTIME_HEAD_FIELDS = {
    "historical_baseline",
    "drivers",
    "target_definition_id",
    "training_target_end_date",
    "residual_target_end_date",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure runtime forecast must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _series(count=340):
    # Fixed synthetic NAVs keep the daily calculation reproducible.
    rng = np.random.default_rng(731)
    increments = 0.0002 + 0.004 * np.sin(np.arange(count) / 11) + rng.normal(0, 0.008, count)
    navs = 2 * np.exp(np.cumsum(increments))
    rows, day = [], date(2023, 4, 3)
    while len(rows) < count:
        if day.weekday() < 5:
            rows.append(
                {
                    "date": day.isoformat(),
                    "nav": f"{navs[len(rows)]:.12f}",
                    "source": "synthetic_test_data",
                    "available_at": GENERATED_AT,
                }
            )
        day += timedelta(days=1)
    return rows


def _fit(series, **kwargs):
    return core.fit_runtime_forecast(
        series,
        product_id="synthetic:nav",
        generated_at=GENERATED_AT,
        **kwargs,
    )


def _heads(bundle):
    return {
        (head["horizon"], head["entry_lag"]): head
        for head in [*bundle["forecasts"], *bundle["execution_forecasts"]]
    }


def _legacy_summary(head, *, samples=True):
    excluded = RUNTIME_HEAD_FIELDS | (set() if samples else {"samples"})
    return {key: value for key, value in head.items() if key not in excluded}


def test_each_runtime_head_seals_its_own_distribution_baseline_drivers_and_maturity():
    rows = _series()
    result = _fit(rows)
    logs = np.log([float(row["nav"]) for row in rows])
    states = {(row["horizon"], row["entry_lag"]): row for row in result["model_state"]}
    assert len(states) == 12
    for identity, head in _heads(result).items():
        horizon, lag = identity
        baseline = head["historical_baseline"]
        origins = np.arange(len(rows) - horizon - lag)
        baseline_samples = np.expm1(logs[origins + lag + horizon] - logs[origins + lag])
        samples = np.asarray(head["samples"])
        assert head["method"] == "ridge_prequential_empirical"
        assert head["sample_count"] == len(samples)
        for summary, values in ((head, samples), (baseline, baseline_samples)):
            assert (summary["horizon"], summary["entry_lag"]) == identity
            assert summary["target_definition_id"] == result["target_definition"]["id"]
            assert summary["sample_count"] == len(values)
            assert summary["up_probability"] == np.mean(values > 0)
            assert summary["down_probability"] == np.mean(values < 0)
            assert summary["flat_probability"] == np.mean(values == 0)
            assert summary["expected_change"] == pytest.approx(np.mean(values), abs=1e-15)
            assert [summary["lower"], summary["median_change"], summary["upper"]] == (
                pytest.approx(np.quantile(values, [0.1, 0.5, 0.9]), abs=1e-15)
            )
            for kind in ("training", "residual"):
                endpoint = summary[f"{kind}_target_end_index"]
                assert endpoint is None or endpoint <= len(rows) - 1
                assert summary[f"{kind}_target_end_date"] == (
                    rows[endpoint]["date"] if endpoint is not None else None
                )
        assert "samples" not in baseline and "historical_baseline" not in baseline
        assert baseline["method"] == "empirical_historical"
        assert baseline["training_start"] == rows[lag]["date"]
        assert baseline["training_end"] == rows[-1]["date"]
        assert baseline["training_sample_count"] == len(baseline_samples)
        assert baseline["residual_sample_count"] == 0
        assert baseline["residual_target_end_index"] is None
        assert head["drivers"] == [
            row for row in result["drivers"] if (row["horizon"], row["entry_lag"]) == identity
        ]
        assert len(head["drivers"]) == 7
        assert states[identity]["intercept"] + sum(
            driver["contribution_log_return"] for driver in head["drivers"]
        ) == pytest.approx(head["ridge_log_location"], abs=1e-14)


def test_future_nav_and_event_changes_do_not_rewrite_past_runtime_heads_or_baselines():
    rows, origin, lags = _series(580), 540, (0, 2, 4)
    before = _fit(rows, entry_lags=lags, include_evaluation=True)
    prefix = _fit(rows[: origin + 1], entry_lags=lags)
    future = copy.deepcopy(rows)
    for row in future[origin + 1 :]:
        row["nav"] = str(float(row["nav"]) * 1.2)
    future[origin + 10]["invalid_return"] = True
    after = _fit(future, entry_lags=lags, include_evaluation=True)
    before_rows = {
        (row["horizon"], row["entry_lag"]): row
        for row in before["evaluation_rows"]
        if row["origin_index"] == origin
    }
    after_rows = {
        (row["horizon"], row["entry_lag"]): row
        for row in after["evaluation_rows"]
        if row["origin_index"] == origin
    }
    assert len(before_rows) == 9
    for identity, head in _heads(prefix).items():
        old = before_rows[identity]
        assert old["prediction"] == _legacy_summary(head, samples=False)
        assert old["historical_baseline"] == _legacy_summary(head["historical_baseline"])
        if identity in after_rows:
            assert after_rows[identity]["prediction"] == old["prediction"]
            assert after_rows[identity]["historical_baseline"] == old["historical_baseline"]
        else:
            assert identity[0] == 20  # Future-invalid H20 labels leave evaluation only.
    assert [row for row in before["issued_forecasts"] if row["origin_index"] <= origin] == [
        row for row in after["issued_forecasts"] if row["origin_index"] <= origin
    ]
    identities = [
        (row["origin_index"], row["horizon"], row["entry_lag"]) for row in before["evaluation_rows"]
    ]
    assert len(identities) == len(set(identities))
    for row in before["evaluation_rows"]:
        entry = rows[row["origin_index"] + row["entry_lag"]]
        target = rows[row["origin_index"] + row["entry_lag"] + row["horizon"]]
        assert row["actual_change"] == pytest.approx(float(target["nav"]) / float(entry["nav"]) - 1)
        for summary in (row["prediction"], row["historical_baseline"]):
            for kind in ("training", "residual"):
                endpoint = summary[f"{kind}_target_end_index"]
                assert endpoint is None or endpoint <= row["origin_index"]


def _local_fit(rows, **kwargs):
    return core.fit_local_timing_forecast(
        rows, product_id="synthetic:nav", generated_at=GENERATED_AT, **kwargs,
    )


def test_local_nonstandard_heads_reuse_exact_kernel_without_changing_official_identity():
    rows = _series()
    official = _fit(rows, entry_lags=(0, 2))
    local = _local_fit(rows, horizons=(1, 7, 13), entry_lags=(0, 2), include_history=True)
    assert official == _fit(rows, entry_lags=(0, 2))
    assert official["model_hash"] == core.RUNTIME_MODEL_HASH
    assert "local_contract" not in official and "target_unit" not in official
    assert local["model_hash"] == core.LOCAL_RUNTIME_MODEL_HASH != core.RUNTIME_MODEL_HASH
    assert local["algorithm_model_hash"] == official["model_hash"]
    assert local["local_contract"] == core.LOCAL_TIMING_CONTRACT
    assert local["target_unit"] == "nav_observations" and local["requested_horizons"] == [1, 7, 13]
    assert {(h["horizon"], h["entry_lag"]) for h in local["forecasts"] + local["execution_forecasts"]} == {
        (h, lag) for h in (1, 7, 13) for lag in (0, 2)}
    # On an official H the exact numerical result is unchanged; only the local
    # output target metadata is additional. Zero-lag aliases are still exact.
    for key, head in _heads(local).items():
        if key[0] == 1:
            clean = {k: v for k, v in head.items() if k not in {"local_contract", "target_unit", "historical_baseline"}}
            baseline = {k: v for k, v in head["historical_baseline"].items()
                        if k not in {"local_contract", "target_unit"}}
            assert {**clean, "historical_baseline": baseline} == _heads(official)[key]
    for head in local["forecasts"]:
        assert next(h for h in local["execution_forecasts"]
                    if h["horizon"] == head["horizon"] and h["entry_lag"] == 0) is head
    assert len(local["historical_forecasts"]) == len(rows) * 6
    logs = np.log([float(row["nav"]) for row in rows])
    for (horizon, lag), head in _heads(local).items():
        labels = np.expm1(logs[lag + horizon:] - logs[lag:len(rows) - horizon])
        assert head["historical_baseline"]["sample_count"] == len(rows) - horizon - lag
        assert head["historical_baseline"]["expected_change"] == pytest.approx(np.mean(labels))
        assert head["training_target_end_index"] <= len(rows) - 1
        assert head["residual_target_end_index"] <= len(rows) - 1


def test_local_exact_h_history_is_complete_prefix_consistent_and_not_future_filtered():
    rows, origin = _series(340), 280
    original = _local_fit(rows, horizons=(7,), entry_lags=(2, 4), include_history=True)
    prefix = _local_fit(rows[:origin + 1], horizons=(7,), entry_lags=(2, 4), include_history=True)
    future = copy.deepcopy(rows)
    for row in future[origin + 1:]:
        row["nav"] = str(float(row["nav"]) * 1.2)
    future[origin + 3]["invalid_return"] = True
    after = _local_fit(future, horizons=(7,), entry_lags=(2, 4), include_history=True)
    before_rows = [r for r in original["historical_forecasts"] if r["origin_index"] <= origin]
    assert before_rows == [r for r in after["historical_forecasts"] if r["origin_index"] <= origin]
    for full, cut in zip(before_rows, prefix["historical_forecasts"], strict=True):
        assert full["prediction"] == cut["prediction"]
        assert full.get("historical_baseline") == cut.get("historical_baseline")
        for kind in ("training", "residual"):
            endpoint = full["prediction"].get(f"{kind}_target_end_index")
            assert endpoint is None or endpoint <= full["origin_index"]
    assert len(after["historical_forecasts"]) == len(rows) * 3
    assert after["historical_forecasts"][-1]["origin_index"] == len(rows) - 1
    assert after["historical_forecasts"][-1]["prediction"]["status"] == "available"
    assert not {"issued_forecasts", "evaluation_rows"} & original.keys()
    assert not {"actual_change", "target_date", "issued_at"} & original["historical_forecasts"][-1].keys()


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (True,), (1.5,), "7"])
def test_local_horizon_requires_positive_exact_integer_and_never_clamps(horizons):
    with pytest.raises(ValueError, match="horizons"):
        _local_fit(_series(30), horizons=horizons)

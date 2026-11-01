"""Reusable routing checks; synthetic paths are not investment evidence."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from rmbfund.forecast_reentry_validation import compare_reentry_paths


def _series(count=100):
    start = date(2024, 7, 1)
    return [{"date": (start + timedelta(days=i)).isoformat(), "nav": str(1 + i / 1000)}
            for i in range(count)]


def _case():
    return {"id": "main", "buy_fee": .0015, "redemption_fee": .005,
            "entry_lag": 3, "redemption_lag": 3, "confirmation_lag": 1,
            "cash_lag": 3, "nav_availability_lag": 1}


def _histories(series, *, exit_signal=True):
    histories = {name: [] for name in ("baseline", "recovery", "empirical")}
    for i, point in enumerate(series):
        for lag in (3, 4):
            for name, rows in histories.items():
                mean = -.02 if name == "baseline" and exit_signal and 6 <= i <= 15 else .02
                rows.append({"origin_index": i, "origin_date": point["date"],
                             "horizon": 20, "entry_lag": lag,
                             "prediction": {"status": "available", "method": "synthetic", "expected_change": mean}})
    return histories


def _paths(series, histories):
    return {r["variant_id"]: r for r in compare_reentry_paths(series, histories, [_case()], start_origin=0)["scenarios"]}


def test_reentry_changes_only_after_actual_sale_and_spendable_cash_with_unchanged_sells():
    series = _series()
    paths = _paths(series, _histories(series))
    old, new = paths["baseline"], paths["recovery"]
    assert all(r["status"] == "available" for r in paths.values())
    assert old["replay"]["strategy"]["trades"][:2] == new["replay"]["strategy"]["trades"][:2]
    diagnostics = new["diagnostics"]
    first = next(r for r in diagnostics if r["reentry_gate"])
    assert first["origin_index"] == 11
    assert first["decision_index"] == 12  # sale priced 9, cash settled 9+3
    assert first["base_mean"] < 0
    assert first["action"] == "buy"  # no hidden base AND recovery gate
    assert all(r["selected_model"] == "baseline" and r["action"] == r["base_action"]
               for r in diagnostics if not r["reentry_gate"])
    assert not any(r["ever_redeemed"] for r in diagnostics if r["decision_index"] < 9)
    assert all(not r["reentry_gate"] for r in diagnostics if r["state"]["cash_in_transit"])
    assert [t["execution_index"] for t in new["replay"]["strategy"]["trades"]][:4] == [3, 9, 14, 17]
    assert new["short_reentry_round_trip_count"] == 1
    assert new["attribution"]["identity_residual"] == pytest.approx(0, abs=1e-14)
    assert _paths(series, _histories(series)) == paths  # no closure state shared between runs


def test_no_exit_never_requires_replacement_and_matches_baseline_exactly():
    series = _series()
    histories = _histories(series, exit_signal=False)
    histories["recovery"] = []
    paths = _paths(series, histories)
    assert all(r["status"] == "available" and not r["reentry_triggered"] for r in paths.values())
    assert all(r["replay"]["strategy"] == paths["baseline"]["replay"]["strategy"] for r in paths.values())


def test_missing_required_reentry_head_invalidates_path_without_moving_start_or_dropping_tail():
    series = _series()
    histories = _histories(series)
    histories["recovery"] = [r for r in histories["recovery"]
                              if (r["origin_index"], r["entry_lag"]) != (11, 3)]
    paths = _paths(series, histories)
    assert paths["recovery"]["status"] == "unavailable"
    assert "required reentry forecast" in paths["recovery"]["reason"]
    assert paths["baseline"]["status"] == paths["empirical"]["status"] == "available"
    assert len(paths["baseline"]["replay"]["strategy"]["action_trace"]) == len(series) - 1
    assert paths["baseline"]["replay"]["start_date"] == series[1]["date"]


def test_retrospective_targets_and_unknown_current_nav_fields_cannot_enter_decisions():
    series = _series()
    histories = _histories(series)
    changed = deepcopy(histories)
    for rows in changed.values():
        for row in rows:
            row.update(actual_change=1e9, future_nav=0, target_date="1900-01-01")
            row["prediction"]["future_nav"] = 0
    assert _paths(series, histories) == _paths(series, changed)

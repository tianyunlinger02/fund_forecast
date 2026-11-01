"""Synthetic saved-choice replay, not performance evidence or personal P&L."""
from copy import deepcopy
import socket
import sqlite3

import pytest

from rmbfund.assessment_replay import freeze_simulation_reference, replay_saved_assessment
from rmbfund.domain import ValidationError, digest
from rmbfund.timing_scenario import build_calendar_scenario

PID = "open_fund:synthetic:C"
SUBJECT = {"id": PID, "symbol": "synthetic", "kind": "open_fund", "share_class": "C",
           "exchange": "", "is_demo": False, "name": "合成基金"}
NOW = "2026-10-01T20:30:00+08:00"
TARGET = {"id": "raw_unit_nav_simple_return_v1", "target_kind": "raw_nav_simple_return",
          "nav_basis": "raw unit NAV; no dividend reinvestment or split adjustment",
          "not_total_return": True}
FEES = {"subscription_fee": .01, "current_redemption_fee": .02,
        "future_old_redemption_fee": .005, "future_new_redemption_fee": .015}


@pytest.fixture(autouse=True)
def no_io_or_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved replay must not read a DB, fit, request, issue, or score")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for name in ("fit_runtime_forecast", "fit_local_timing_forecast", "fit_forecast"):
        monkeypatch.setattr("rmbfund.forecast." + name, forbidden)
    monkeypatch.setattr("rmbfund.investment_forecast.fit_local_investment_forecast", forbidden)
    monkeypatch.setattr("rmbfund.service.Workspace._score_forecasts", forbidden)
    monkeypatch.setattr("rmbfund.service.Workspace.update_forecast", forbidden)


def _fixture(*, flat="buy", held="add", currency="CNY"):
    days = ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"]
    series = [{"date": day, "nav": nav, "provenance": {
        "currency": currency, "nav_type": "unit_nav", "available_at": day + "T18:00:00+08:00",
        "source": "synthetic", "source_url": "https://example.invalid/synthetic",
    }} for day, nav in zip(days, ("0.9", "1.0", "1.1", "1.155", "1.21", "1.31"), strict=True)]
    plan = {"kind": "calendar_trade_plan", "decision_at": "2026-10-01T20:00:00+08:00",
            "decision_date": "2026-10-01", "application_date": "2026-10-02",
            "sale_date": "2026-10-06", "deadline": "2026-10-06", "anchor_date": "2026-10-01",
            "new_confirmation_date": "2026-10-05", "new_shares_sellable": True,
            "horizon": 2, "entry_lag": 1, "target_unit": "nav_observations"}
    head = {"horizon": 2, "entry_lag": 1, "expected_change": .03, "lower": -.04, "upper": .08,
            "up_probability": .7, "samples": [-.04, .03, .08]}
    prediction = {"product_id": PID, "generated_at": "2026-10-01T20:00:00+08:00",
                  "anchor": {"date": "2026-10-01", "nav": "1.0", "currency": currency},
                  "basis_hash": digest(series[:2]), "target_definition": deepcopy(TARGET),
                  "execution_forecasts": [head]}
    scenario = build_calendar_scenario(prediction, subject=SUBJECT, plan=plan, series=series[:2],
                                       fee_basis=FEES, base_amount=1000)
    reference = freeze_simulation_reference(scenario, branches={
        "flat": {"action": flat, "reference_action": "wait"},
        "held": {"action": held, "reference_action": "hold"},
    })
    assessment = {"schema_version": 7, "policy_version": "synthetic", "subject": deepcopy(SUBJECT),
                  "forecast": prediction, "trade_scenario": scenario, "horizon": 2, "entry_lag": 1,
                  "decision_brief": {"not_interpreted": "原作者选择"}, "simulation_reference": reference}
    return {"id": "assessment-original", "saved_at": NOW, "assessment": assessment}, series


def _note(*, kind="simulated_choice", branch="flat", action="buy", occurred_on="2026-10-02",
          recorded_at=NOW, **fields):
    return {"id": "note-original", "assessment_id": "assessment-original", "product_id": PID,
            "kind": kind, "branch": branch, "action": action, "occurred_on": occurred_on,
            "recorded_at": recorded_at, "rationale": "合成笔记", **fields}


def test_mature_saved_actions_use_original_exact_lag_fees_capital_and_reference_without_mutation():
    saved, series = _fixture()
    before = deepcopy((saved, series))
    result = replay_saved_assessment(saved, series=series)
    assert result["kind"] == "saved_calendar_simulation" and result["status"] == "mature"
    assert result["horizon"] == 2 and result["entry_lag"] == 1
    assert result["window"]["entry_date"] == "2026-10-02"
    assert result["window"]["target_date"] == "2026-10-06"
    assert result["actual_nav_change"] == pytest.approx(.1)  # Not target / anchor - 1.
    flat, held = result["comparisons"]
    buy = 1000 * 1.1 * .985 / 1.01
    hold = 1000 * 1.1 * .995 + 1000
    assert flat["action"] == "buy" and flat["reference_action"] == "wait"
    assert flat["chosen"]["actual_value"] == pytest.approx(buy)
    assert flat["reference"]["actual_value"] == 1000
    assert flat["realized_difference"] == pytest.approx(buy - 1000)
    assert held["action"] == "add" and held["reference_action"] == "hold"
    assert held["chosen"]["initial_value"] == held["reference"]["initial_value"] == 2000
    assert held["chosen"]["actual_value"] == pytest.approx(hold - 1000 + buy)
    assert held["realized_difference"] == pytest.approx(buy - 1000)
    assert result["selection_timing"] == "prospective"
    assert result["personal_actual_pnl"] is None
    assert result["end_value_basis"] == "cash_and_redemption_receivable"
    assert all("better_probability" not in row["chosen"] for row in result["comparisons"])
    assert all("expected_value" not in row["chosen"] for row in result["comparisons"])
    assert (saved, series) == before


@pytest.mark.parametrize("action,branch,expected", [
    ("buy", "flat", 1000 * 1.1 * .985 / 1.01), ("wait", "flat", 1000),
    ("no_action", "flat", 1000), ("hold", "held", 1000 * 1.1 * .995 + 1000),
    ("add", "held", 1000 * 1.1 * .995 + 1000 * 1.1 * .985 / 1.01),
    ("reduce50", "held", 500 * .98 + 500 * 1.1 * .995 + 1000),
    ("exit", "held", 1980), ("redeem", "held", 1980),
    ("no_action", "held", 1000 * 1.1 * .995 + 1000),
])
def test_explicit_choice_uses_frozen_same_capital_reference_for_all_actions(action, branch, expected):
    saved, series = _fixture(flat=None, held=None)
    note = _note(action=action, branch=branch)
    result = replay_saved_assessment(saved, series=series, note=note)
    assert result["status"] == "mature"
    assert len(result["comparisons"]) == 1
    row = result["comparisons"][0]
    assert row["branch"] == branch and row["action"] == action
    assert row["chosen"]["actual_value"] == pytest.approx(expected)
    assert row["reference_action"] == ("wait" if branch == "flat" else "hold")
    assert row["choice_context"] == "prospective"
    assert saved["assessment"]["simulation_reference"]["branches"][branch]["action"] is None


def test_preferences_are_not_rechosen_from_any_current_model_or_outcomes():
    saved, series = _fixture(flat="wait", held="exit")
    saved["assessment"]["forecast"]["execution_forecasts"][0]["expected_change"] = 50
    saved["assessment"]["trade_scenario"]["decision"]["branches"]["held"]["action"] = "hold"
    saved["assessment"]["decision_brief"] = {"today_preferred_action": "add"}
    saved["assessment"]["reconstructed_history"] = {"best_action": "add", "fit": "never read"}
    result = replay_saved_assessment(saved, series=series)
    assert [row["action"] for row in result["comparisons"]] == ["wait", "exit"]
    assert result["comparisons"][1]["realized_difference"] < 0


@pytest.mark.parametrize("kind", ["user_report", "observation"])
def test_user_report_and_observation_are_only_notes_without_trade_or_personal_pnl(kind):
    saved, _ = _fixture()
    note = _note(kind=kind, recorded_at="2026-10-07T09:00:00+08:00")
    result = replay_saved_assessment(saved, series=[], note=note)
    assert result["status"] == "note_only"
    assert result["comparisons"] == [] and result["actual_nav_change"] is None
    assert result["personal_actual_pnl"] is None
    assert result["note"] == note
    assert result["choice_context"] == "historical"


@pytest.mark.parametrize("recorded,occurred,context,timing", [
    (NOW, "2026-10-02", "prospective", "prospective"),
    ("2026-10-05T09:00:00+08:00", "2026-10-02", "historical", "pre_target"),
    ("2026-10-06T00:01:00+08:00", "2026-10-02", "historical", "post_hoc"),
    ("2026-10-07T09:00:00+08:00", "2026-10-02", "historical", "post_hoc"),
    ("2026-10-07T09:00:00+08:00", "2026-10-09", "prospective", "post_hoc"),
])
def test_actual_record_clock_never_backdates_a_historical_choice(recorded, occurred, context, timing):
    saved, series = _fixture()
    result = replay_saved_assessment(saved, series=series, note=_note(
        recorded_at=recorded, occurred_on=occurred))
    assert result["status"] == "mature"
    assert result["choice_context"] == context and result["selection_timing"] == timing
    assert result["comparisons"][0]["selection_timing"] == timing
    assert result["window"]["entry_date"] == "2026-10-02"  # Notes do not move the old plan.


def test_plan_is_a_labelled_simulation_not_an_executed_choice_and_missing_clock_is_unknown():
    saved, series = _fixture()
    result = replay_saved_assessment(saved, series=series, note=_note(kind="plan"))
    assert result["status"] == "mature" and result["selection"]["source"] == "plan"
    payload_only = replay_saved_assessment(saved["assessment"], series=series)
    assert payload_only["status"] == "mature" and payload_only["selection_timing"] == "unknown"


@pytest.mark.parametrize("note", [
    _note(assessment_id="other"), _note(product_id="other"), _note(branch="held", action="buy"),
    _note(recorded_at="2020-01-01T00:00:00+08:00"), _note(recorded_at="2026-10-01"),
])
def test_unrelated_or_inconsistent_note_cannot_attach_to_saved_simulation(note):
    saved, series = _fixture()
    result = replay_saved_assessment(saved, series=series, note=note)
    assert result["status"] == "invalid" and result["comparisons"] == []


def test_absent_preference_and_unquantified_reduce_are_explicit_not_half_sale():
    saved, series = _fixture(flat=None, held="reduce")
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == "unavailable"
    assert [row["action"] for row in result["comparisons"]] == [None, "reduce"]
    assert all(row["chosen"] is None for row in result["comparisons"])
    selected = replay_saved_assessment(saved, series=series,
                                      note=_note(branch="held", action="reduce50"))
    assert selected["status"] == "mature"
    assert selected["comparisons"][0]["action"] == "reduce50"


def test_original_schema6_and7_share_calendar_contract_and_reference_may_live_in_brief():
    saved, series = _fixture()
    saved["assessment"]["schema_version"] = 6
    original = replay_saved_assessment(saved, series=series)
    saved["assessment"]["schema_version"] = 7
    saved["assessment"]["decision_brief"]["simulation_reference"] = saved["assessment"].pop("simulation_reference")
    assert replay_saved_assessment(saved, series=series) == original


def test_reference_freezer_retains_original_actions_none_and_simple_bases_without_guessing():
    saved, _ = _fixture()
    scenario = saved["assessment"]["trade_scenario"]
    expected = {"branches": {"flat": {"action": "buy", "reference_action": "wait"},
                             "held": {"action": "hold", "reference_action": "hold"}}}
    assert freeze_simulation_reference(scenario) == expected
    scenario["decision"]["branches"]["held"]["action"] = "sell"
    assert freeze_simulation_reference(scenario)["branches"]["held"]["action"] == "exit"
    pairs = {"flat": {"action": None, "reference_action": "wait"},
             "held": {"action": "reduce", "reference_action": "hold"}}
    assert freeze_simulation_reference(scenario, branches=pairs) == {"branches": pairs}
    for bad in ({}, {**pairs, "flat": {"action": "buy", "reference_action": "hold"}},
                {**pairs, "held": {"action": "buy", "reference_action": "hold"}}):
        with pytest.raises(ValidationError):
            freeze_simulation_reference(scenario, branches=bad)


def test_immature_window_retains_original_indices_no_calendar_filling_or_shift():
    saved, series = _fixture()
    result = replay_saved_assessment(saved, series=series[:4])
    assert result["status"] == "immature"
    assert result["window"]["entry_index"] == 2 and result["window"]["target_index"] == 4
    assert result["window"]["target_date"] is None
    assert result["comparisons"] == [] and result["actual_nav_change"] is None
    # Inserting an unavailable intermediate observation cannot move a window onto a later date.
    series[3]["provenance"]["available_at"] = "2026-11-01T00:00:00+08:00"
    bounded = replay_saved_assessment(saved, series=series, as_of="2026-10-08T00:00:00+08:00")
    assert bounded["status"] == "immature"
    assert bounded["window"]["target_date"] == "2026-10-06"
    del series[3]["provenance"]["available_at"]
    assert replay_saved_assessment(saved, series=series)["status"] == "immature"


@pytest.mark.parametrize("flag,value", [("invalid_return", True), ("label_valid", False)])
def test_only_incoming_moves_after_entry_are_invalid_return_events(flag, value):
    saved, series = _fixture()
    series[2][flag] = value  # Incoming to entry is outside the realized return.
    assert replay_saved_assessment(saved, series=series)["status"] == "mature"
    series[3][flag] = value
    invalid = replay_saved_assessment(saved, series=series)
    assert invalid["status"] == "invalid" and invalid["reason_codes"] == ["invalid_event"]
    assert invalid["comparisons"] == []


@pytest.mark.parametrize("mutation,code,status", [
    (lambda s, r: s["assessment"].update(schema_version=5), "unsupported_schema", "unavailable"),
    (lambda s, r: s["assessment"].pop("simulation_reference"), "missing_reference", "unavailable"),
    (lambda s, r: s["assessment"]["forecast"].pop("target_definition"), "missing_target", "unavailable"),
    (lambda s, r: s["assessment"]["forecast"].update(target_definition={"id": "total_return"}), "target_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["forecast"]["target_definition"].update(nav_basis="adjusted"), "target_basis_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["forecast"]["anchor"].update(nav="1.01"), "anchor_mismatch", "invalid"),
    (lambda s, r: r.pop(1), "missing_anchor", "invalid"),
    (lambda s, r: r[3]["provenance"].update(currency="USD"), "currency_mismatch", "invalid"),
    (lambda s, r: r[3].update(currency="USD"), "currency_mismatch", "invalid"),
    (lambda s, r: r[3]["provenance"].pop("currency"), "missing_currency", "unavailable"),
    (lambda s, r: r[3]["provenance"].update(nav_type="accumulated_nav"), "nav_basis_mismatch", "invalid"),
    (lambda s, r: r[3]["provenance"].pop("nav_type"), "missing_nav_basis", "unavailable"),
    (lambda s, r: r[3].update(product_id="other"), "product_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["forecast"].update(product_id="other"), "product_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["subject"].update(share_class="A"), "product_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["plan"].update(horizon=3), "window_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"].update(horizon=0), "invalid_window", "unavailable"),
    (lambda s, r: s["assessment"]["trade_scenario"]["plan"].update(anchor_date="2026-09-30"), "anchor_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"].update(end_value_basis="nav"), "end_basis_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["plan"].update(target_unit="calendar_days"), "target_unit_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["decision"]["primary_head"].update(horizon=3), "window_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["decision"]["primary_head"].update(target_definition_id="total_return"), "target_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["decision"]["primary_head"].update(currency="USD"), "currency_mismatch", "invalid"),
    (lambda s, r: s["assessment"]["trade_scenario"]["decision"]["primary_head"].update(anchor={"date": "2026-09-30", "nav": "1"}), "anchor_mismatch", "invalid"),
])
def test_inconsistent_original_or_actual_basis_fails_explicitly_without_values(mutation, code, status):
    saved, series = _fixture()
    mutation(saved, series)
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == status and result["reason_codes"] == [code]
    assert result["comparisons"] == [] and result["actual_nav_change"] is None


def test_historical_provenance_and_pre_anchor_values_are_not_reconstructed_or_required_for_replay():
    saved, series = _fixture()
    baseline = replay_saved_assessment(saved, series=series)
    original_basis = saved["assessment"]["forecast"]["basis_hash"]
    series[0]["nav"] = "0.8"  # The old training prefix is not refitted.
    series[0]["provenance"]["source"] = "revised-pre-anchor-provenance"
    series[1]["provenance"]["online_observations"] = [{"value": "1.0", "source": "same-value"}]
    series.pop(0)  # No effect on dates/values after the original anchor.
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == "mature"
    assert result["original_basis_hash"] == original_basis
    assert result["basis_relationship"] == "original_anchor_verified; historical_forecast_inputs_not_reconstructed"
    assert result["actual_nav_change"] == baseline["actual_nav_change"]
    assert result["comparisons"] == baseline["comparisons"]
    saved["assessment"]["forecast"].pop("basis_hash")
    limited = replay_saved_assessment(saved, series=series)
    assert limited["status"] == "mature" and limited["original_basis_hash"] is None
    assert limited["provenance_limitations"]
    assert limited["comparisons"] == baseline["comparisons"]


def test_same_value_online_merged_original_replays_with_stored_only_provenance():
    saved, stored = _fixture()
    merged = deepcopy(stored[:2])
    for row in merged:
        row["provenance"]["online_observations"] = [{"date": row["date"], "value": row["nav"],
            "currency": "CNY", "observed_at": NOW, "available_at": NOW, "source": "same-value online"}]
    saved["assessment"]["forecast"]["basis_hash"] = digest(merged)
    assert digest(stored[:2]) != saved["assessment"]["forecast"]["basis_hash"]
    result = replay_saved_assessment(saved, series=stored)
    assert result["status"] == "mature"
    assert result["actual_nav_change"] == pytest.approx(.1)
    assert result["original_basis_hash"] == digest(merged)


def test_original_fee_and_plan_limitations_are_not_repaired_from_actual_outcome():
    saved, series = _fixture()
    saved["assessment"]["trade_scenario"]["fees"]["future_new_redemption_fee"] = None
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == "unavailable"
    assert all(row["chosen"]["actual_value"] is None for row in result["comparisons"])
    saved["assessment"]["simulation_reference"]["branches"]["held"]["action"] = "hold"
    partial = replay_saved_assessment(saved, series=series)
    assert partial["status"] == "mature" and partial["has_unavailable_comparisons"]
    assert partial["comparisons"][0]["status"] == "unavailable"
    assert partial["comparisons"][1]["status"] == "mature"
    saved["assessment"]["trade_scenario"]["fees"]["future_new_redemption_fee"] = .015
    saved["assessment"]["trade_scenario"]["plan"]["new_confirmation_date"] = "2026-10-07"
    assert replay_saved_assessment(saved, series=series)["comparisons"][0]["status"] == "unavailable"


def test_real_observation_horizon_is_not_silently_recast_as_a_calendar_strategy():
    saved, series = _fixture(currency="USD")
    series[4]["date"] = "2026-10-09"
    series[4]["provenance"]["available_at"] = "2026-10-09T18:00:00+08:00"
    series.pop()
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == "mature" and result["currency"] == "USD"
    assert result["window"]["planned_sale_date"] == "2026-10-06"
    assert result["window"]["target_date"] == "2026-10-09"
    assert result["actual_nav_change"] == pytest.approx(.1)


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.reverse(), lambda rows: rows[3].update(nav=0),
    lambda rows: rows[3].update(label_valid="false"), lambda rows: rows[3].update(date="bad"),
    lambda rows: rows[3].update(provenance=[]),
])
def test_invalid_actual_series_is_not_cleaned_filtered_or_imputed(mutate):
    saved, series = _fixture()
    mutate(series)
    result = replay_saved_assessment(saved, series=series)
    assert result["status"] == "invalid" and result["comparisons"] == []

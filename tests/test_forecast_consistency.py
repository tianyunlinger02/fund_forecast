"""Reusable source-routing checks; synthetic paths are not investment evidence."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

import rmbfund.forecast_consistency_validation as consistency
from rmbfund.forecast_consistency_validation import (
    VARIANTS,
    compare_consistency_path,
    consistency_packets,
    consistency_policy,
)
from rmbfund.forecast_economic_validation import _decision_policy, _ledger_args
from rmbfund.forecast_reentry_validation import (
    baseline_parity,
    reentry_packets,
    reentry_policy,
)
from rmbfund.forecast_research import replay_actions


def _series(count=45):
    start = date(2025, 1, 1)
    return [{"date": (start + timedelta(days=i)).isoformat(), "nav": "1.0"}
            for i in range(count)]


def _case(lag=3):
    return {"id": "main" if lag == 3 else "slow", "buy_fee": .0015,
            "redemption_fee": .005, "entry_lag": lag, "redemption_lag": lag,
            "confirmation_lag": 1, "cash_lag": 3, "nav_availability_lag": 1}


def _histories(series, *, exit_signal=True):
    histories = {source: [] for source in ("baseline", "empirical")}
    for origin, point in enumerate(series):
        for lag in (3, 4):
            for source, rows in histories.items():
                mean = -.02 if source == "baseline" and exit_signal and 6 <= origin <= 15 else .02
                rows.append({"origin_index": origin, "origin_date": point["date"],
                             "horizon": 20, "entry_lag": lag,
                             "target_index": origin + lag + 20,
                             "outcome_status": ("not_mature" if origin + lag + 20 >= len(series)
                                                else "available"),
                             "prediction": {"status": "available", "method": "synthetic",
                                            "expected_change": mean}})
    return histories


def _path(series, histories, variant, *, case=None, start_origin=0):
    return compare_consistency_path(series, histories, _case() if case is None else case,
                                    variant=variant, start_origin=start_origin)


def _missing(histories, source, origin, lag):
    result = deepcopy(histories)
    result[source] = [row for row in result[source]
                      if (row["origin_index"], row["horizon"], row["entry_lag"]) != (origin, 20, lag)]
    return result


def _state(**changes):
    return {"can_buy": False, "can_redeem": False, "has_units": False,
            "order_pending": False, "unconfirmed_units": False,
            "cash_in_transit": False, **changes}


def _assert_parity(current, original):
    result = baseline_parity(current, original)
    assert result["matched"], result["differences"]


def test_permanent_switch_waits_for_executed_exit_prevents_switch_churn_but_allows_later_sale():
    series = _series()
    histories = _histories(series)
    paths = {variant: _path(series, histories, variant) for variant in VARIANTS}
    assert VARIANTS == ("baseline", "hybrid", "post_exit_empirical", "uniform_empirical")
    assert all(row["status"] == "available" for row in paths.values())
    base, hybrid, permanent = [paths[key] for key in VARIANTS[:3]]
    for other in (hybrid, permanent):
        assert other["replay"]["strategy"]["trades"][:2] == base["replay"]["strategy"]["trades"][:2]
        assert other["replay"]["strategy"]["action_trace"][:8] == base["replay"]["strategy"]["action_trace"][:8]
    assert [row["execution_index"] for row in hybrid["replay"]["strategy"]["trades"]][:4] == [3, 9, 14, 17]
    assert [row["execution_index"] for row in permanent["replay"]["strategy"]["trades"]] == [3, 9, 14]
    assert hybrid["short_reentry_round_trip_count"] == 1
    assert permanent["short_reentry_round_trip_count"] == 0
    assert len(permanent["exit_episodes"]) == 1
    assert permanent["holding_episodes"][-1]["sale"] is None

    diagnostics = permanent["diagnostics"]
    first = next(row for row in diagnostics if row["actual_exit_observed"])
    assert (first["origin_index"], first["decision_index"], first["episode"]) == (8, 9, 1)
    assert all(row["selected_source"] == "baseline" and not row["actual_exit_observed"]
               for row in diagnostics[:8])  # Signal at 6 and pending sale at 7 do not switch.
    assert all(row["selected_source"] == "empirical" for row in diagnostics[8:])
    for row in diagnostics:
        assert row["reentry_gate"] == (row["actual_exit_observed"] and row["state"]["can_buy"])
        assert "buy_advantage_vs_cash" in row and "hold_advantage_vs_redeem" in row
    transit = next(row for row in permanent["replay"]["strategy"]["action_trace"]
                   if row["state"]["cash_in_transit"])
    assert transit["action"] == "buy" and transit["reason"] == "not_spendable_cash"

    later_negative = deepcopy(histories)
    for row in later_negative["empirical"]:
        if 24 <= row["origin_index"] <= 28:
            row["prediction"]["expected_change"] = -.02
    later = _path(series, later_negative, "post_exit_empirical")
    assert later["status"] == "available"
    sales = [trade for trade in later["replay"]["strategy"]["trades"] if trade["action"] == "redeem"]
    assert [(trade["origin_index"], trade["execution_index"]) for trade in sales] == [(6, 9), (24, 27)]
    assert later["diagnostics"][24]["selected_mean"] < 0
    assert later["diagnostics"][24]["action"] == "redeem"  # Not an implicit hold/cooldown policy.


@pytest.mark.parametrize("lag", [3, 4])
def test_no_exit_needs_no_empirical_uniform_needs_no_baseline_and_closures_are_fresh(lag):
    series, case = _series(), _case(lag)
    histories = _histories(series, exit_signal=False)
    only_baseline = {"baseline": histories["baseline"]}
    base = _path(series, only_baseline, "baseline", case=case)
    permanent = _path(series, only_baseline, "post_exit_empirical", case=case)
    uniform = _path(series, {"empirical": histories["empirical"]}, "uniform_empirical", case=case)
    assert all(row["status"] == "available" for row in (base, permanent, uniform))
    assert permanent["replay"] == base["replay"]
    _assert_parity(uniform["replay"], base["replay"])
    assert all(not row["actual_exit_observed"] and row["selected_source"] == "baseline"
               for row in permanent["diagnostics"])
    assert all(row["selected_source"] == "empirical" for row in uniform["diagnostics"])
    assert permanent == _path(series, only_baseline, "post_exit_empirical", case=case)

    packets = consistency_packets(series, histories, 0, len(series) - 1, case)
    used, fresh = [], []
    old_policy = consistency_policy("post_exit_empirical", case, used)
    old_policy(packets[0], _state(has_units=True, can_redeem=True))
    old_policy(packets[1], _state(cash_in_transit=True))
    new_policy = consistency_policy("post_exit_empirical", case, fresh)
    assert new_policy(packets[2], _state(can_buy=True)) == "buy"
    assert [row["selected_source"] for row in used] == ["baseline", "empirical"]
    assert used[-1]["actual_exit_observed"] and used[-1]["episode"] == 1
    assert fresh[-1]["selected_source"] == "baseline"
    assert fresh[-1]["actual_exit_observed"] is False and fresh[-1]["episode"] == 0


@pytest.mark.parametrize("lag", [3, 4])
@pytest.mark.parametrize("variant,source", [("baseline", "baseline"), ("uniform_empirical", "empirical")])
@pytest.mark.parametrize("position,scope", [("first", "account_eligible_signal_trace"),
                                            ("pending", "blocked_signal_trace"),
                                            ("immature_tail", "account_eligible_signal_trace")])
def test_missing_exact_selected_head_never_moves_start_substitutes_or_drops_tail(lag, variant, source,
                                                                                position, scope):
    series, case, start = _series(), _case(lag), 2
    histories = _histories(series, exit_signal=False)
    end = len(series) - 1
    complete = _path(series, histories, variant, case=case, start_origin=start)
    assert complete["status"] == "available"
    assert complete["replay"]["start_date"] == series[start + 1]["date"]
    assert complete["replay"]["end_date"] == series[end]["date"]
    assert [row["origin_index"] for row in complete["diagnostics"]] == list(range(start, end))
    assert len(complete["replay"]["strategy"]["action_trace"]) == end - start
    packets = consistency_packets(series, histories, start, end, case)
    assert [row["origin_index"] for row in packets] == list(range(start, end))
    assert all((row["horizon"], row["entry_lag"]) == (20, lag) for row in packets)

    gap = {"first": start, "pending": start + 1, "immature_tail": end - 1}[position]
    broken = _missing(histories, source, gap, lag)
    # The other lag already exists; an available wrong-horizon head cannot substitute either.
    wrong_horizon = deepcopy(next(row for row in histories[source]
                                  if row["origin_index"] == gap and row["entry_lag"] == lag))
    wrong_horizon["horizon"] = 10
    broken[source].append(wrong_horizon)
    failed = _path(series, broken, variant, case=case, start_origin=start)
    assert failed["status"] == "unavailable" and scope in failed["reason"]
    assert [row["origin_index"] for row in failed["diagnostics"]] == list(range(start, gap + 1))
    diagnostic = failed["diagnostics"][-1]
    assert diagnostic["selected_source"] == source and diagnostic["availability_scope"] == scope
    assert bool(diagnostic["state"]["can_buy"] or diagnostic["state"]["can_redeem"]) == (scope == "account_eligible_signal_trace")


@pytest.mark.parametrize("lag", [3, 4])
def test_a_b_d_match_original_net_payoff_and_reentry_callbacks_including_blocked_traces(lag):
    series, case, start = _series(), _case(lag), 2
    histories = _histories(series)
    end = len(series) - 1
    controls = {}
    for variant, source in (("baseline", "baseline"), ("uniform_empirical", "empirical")):
        actions = [row for row in histories[source]
                   if start <= row["origin_index"] < end and row["horizon"] == 20 and row["entry_lag"] == lag]
        controls[variant] = replay_actions(series, actions, **_ledger_args(case),
                                           start_index=start + 1, end_index=end,
                                           decision_policy=_decision_policy("net_payoff", case))
    old_packets = reentry_packets({**histories, "recovery": []}, start, end, case)
    controls["hybrid"] = replay_actions(series, old_packets, **_ledger_args(case),
                                        start_index=start + 1, end_index=end,
                                        decision_policy=reentry_policy("empirical", case, []))
    for variant, original in controls.items():
        result = _path(series, histories, variant, case=case, start_origin=start)
        assert result["status"] == "available"
        _assert_parity(result["replay"], original)
        for row in result["diagnostics"]:
            expected = "empirical" if variant == "uniform_empirical" or (variant == "hybrid" and row["reentry_gate"]) else "baseline"
            assert row["selected_source"] == expected


def test_c_drops_baseline_dependency_after_sale_b_keeps_it_and_blocked_selected_heads_are_required():
    series, case = _series(), _case()
    histories = _histories(series)
    original = _path(series, histories, "post_exit_empirical")
    switch = next(row["origin_index"] for row in original["diagnostics"] if row["actual_exit_observed"])
    gate = next(row["origin_index"] for row in original["diagnostics"] if row["reentry_gate"])
    independent = {"baseline": [row for row in histories["baseline"] if row["origin_index"] < switch],
                   "empirical": [row for row in histories["empirical"] if row["origin_index"] >= switch]}
    result = _path(series, independent, "post_exit_empirical")
    assert result["status"] == "available"
    _assert_parity(result["replay"], original["replay"])

    missing_base_at_gate = _missing(histories, "baseline", gate, case["entry_lag"])
    assert _path(series, missing_base_at_gate, "post_exit_empirical")["status"] == "available"
    hybrid = _path(series, missing_base_at_gate, "hybrid")
    assert hybrid["status"] == "unavailable" and "account_eligible_signal_trace" in hybrid["reason"]
    assert hybrid["diagnostics"][-1]["origin_index"] == gate  # S is required even when B selects E.

    for flag in ("cash_in_transit", "order_pending", "unconfirmed_units", "can_redeem"):
        diagnostic = next(row for row in original["diagnostics"]
                          if row["actual_exit_observed"] and row["state"][flag])
        origin = diagnostic["origin_index"]
        broken = _missing(histories, "empirical", origin, case["entry_lag"])
        failed = _path(series, broken, "post_exit_empirical")
        scope = "account_eligible_signal_trace" if flag == "can_redeem" else "blocked_signal_trace"
        assert failed["status"] == "unavailable" and scope in failed["reason"]
        assert failed["diagnostics"][-1]["origin_index"] == origin
        assert failed["diagnostics"][-1]["selected_source"] == "empirical"
        assert failed["diagnostics"][-1]["availability_scope"] == scope
        assert _path(series, broken, "hybrid")["status"] == "available"  # Not an E reentry gate.


def test_outcomes_and_future_payloads_are_whitelisted_away_and_callbacks_run_once(monkeypatch):
    series, case, start = _series(), _case(4), 2
    histories = _histories(series)
    untouched = deepcopy(histories)
    polluted, polluted_series = deepcopy(histories), deepcopy(series)
    for rows in polluted.values():
        for row in rows:
            row.update(actual_change=-999, future_nav=999, target_index=-999,
                       target_date="1900-01-01", outcome_status="invalid", action="redeem")
            row["prediction"].update(actual_change=999, future_nav=0, future_profit=-999,
                                     action="redeem", nav=999)
    for row in polluted_series:
        row.update(future_nav=0, actual_change=-999, future_profit=999)
    end = len(series) - 1
    packets = consistency_packets(series, histories, start, end, case)
    assert packets == consistency_packets(polluted_series, polluted, start, end, case)
    for packet in packets:
        assert set(packet) == {"origin_index", "origin_date", "horizon", "entry_lag", "source_predictions"}
        assert set(packet["source_predictions"]) == {"baseline", "empirical"}
        assert all(set(prediction) == {"status", "method", "expected_change"}
                   for prediction in packet["source_predictions"].values())

    factory, calls = consistency.consistency_policy, []

    def inspect_policy(variant, parameters, diagnostics):
        policy = factory(variant, parameters, diagnostics)
        seen = []
        calls.append(seen)

        def inspect(packet, state):
            assert packet == packets[packet["origin_index"] - start]
            assert set(state) == set(_state()) and all(type(value) is bool for value in state.values())
            seen.append(packet["origin_index"])
            return policy(packet, state)

        return inspect

    monkeypatch.setattr(consistency, "consistency_policy", inspect_policy)
    changed_nav = deepcopy(series)
    for row in changed_nav[20:]:
        row["nav"] = "5.0"
    for variant in VARIANTS:
        original = _path(series, histories, variant, case=case, start_origin=start)
        contaminated = _path(polluted_series, polluted, variant, case=case, start_origin=start)
        assert original["status"] == "available" and original == contaminated
        other = _path(changed_nav, histories, variant, case=case, start_origin=start)
        assert other["status"] == "available"
        fields = ("origin_index", "decision_index", "selected_source", "selected_mean", "action", "state")
        assert [{key: row[key] for key in fields} for row in original["diagnostics"]] == [
            {key: row[key] for key in fields} for row in other["diagnostics"]]
    assert len(calls) == len(VARIANTS) * 3
    assert all(origins == list(range(start, end)) for origins in calls)
    assert histories == untouched

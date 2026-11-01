"""Hand-computable post-settlement forks; synthetic checks, not skill evidence."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

import rmbfund.forecast_reentry_forks as forks
from rmbfund.forecast_economic_validation import _ledger_args
from rmbfund.forecast_research import replay_actions


def _series(count=70, nav=2.0):
    start = date(2025, 1, 1)
    return [{"date": (start + timedelta(days=i)).isoformat(), "nav": nav}
            for i in range(count)]


def _case(lag=3, **changes):
    return {"id": "main" if lag == 3 else "slow", "buy_fee": 0.1, "redemption_fee": 0.2,
            "entry_lag": lag, "redemption_lag": lag, "confirmation_lag": 1,
            "cash_lag": 3, "nav_availability_lag": 1, **changes}


def _baseline(series, case):
    actions = [{"origin_index": 0, "action": "buy"},
               {"origin_index": 4, "action": "redeem"}]
    return replay_actions(series, actions, **_ledger_args(case), initial_cash=110.0,
                          decision_policy=lambda item, state: item["action"])


def _histories(series, mean=0.5):
    return {arm: [{"origin_index": origin, "horizon": 20, "entry_lag": lag,
                   "prediction": {"status": "available", "expected_change": mean}}
                  for origin in range(len(series)) for lag in (3, 4)]
            for arm in ("baseline", "recovery", "empirical")}


def _episode(result):
    assert len(result["episodes"]) == 1
    return result["episodes"][0]


def test_settlement_uses_s_minus_a_external_subscription_and_terminal_redemption_once():
    series, case = _series(), _case()
    baseline = _baseline(series, case)
    untouched = deepcopy(baseline)
    episode = _episode(forks.compare_reentry_forks(series, baseline, _histories(series), case))
    sale = baseline["strategy"]["trades"][1]
    # 110/(1+.1)=100 fund at entry; 100*(1-.2)=80 net sale cash.
    assert sale["gross_notional"] == pytest.approx(100)
    assert sale["fee"] == pytest.approx(20)
    assert episode["initial_cash"] == episode["net_exit_cash"] == pytest.approx(80)
    assert (episode["sale_execution_index"], episode["settlement_index"],
            episode["known_origin_index"], episode["first_pricing_index"],
            episode["common_end_index"]) == (7, 10, 9, 12, 32)
    assert episode["sale"] == sale
    assert episode["first_pricing_date"] == series[12]["date"]
    assert episode["matched_four_arm"] is True
    for arm in episode["arms"].values():
        path, summary = arm["replay"]["strategy"], arm["strategy_summary"]
        buy, = path["trades"]
        assert (buy["origin_index"], buy["decision_index"], buy["execution_index"],
                buy["confirmation_index"]) == (9, 10, 12, 13)
        assert buy["gross_notional"] == pytest.approx(80 / 1.1)
        assert buy["fee"] == pytest.approx(80 * .1 / 1.1)
        assert buy["units"] == pytest.approx(80 / 1.1 / 2)
        assert path["terminal_liquidation_fee"] == pytest.approx(80 / 1.1 * .2)
        assert path["terminal_liquidated_wealth"] == pytest.approx(80 / 1.1 * .8)
        assert summary["return"] == pytest.approx(.8 / 1.1 - 1)
        assert summary["terminal_wealth_per_initial_cash"] == pytest.approx(.8 / 1.1)
        assert summary["total_fees_per_initial_cash"] == pytest.approx((.1 + .2) / 1.1)
        assert summary["turnover"] == pytest.approx(1 / 1.1)
        assert "cash=1" not in summary["turnover_basis"]
        assert arm["attribution"]["initial_cash"] == 80
        assert arm["attribution"]["market_pnl"] == 0
        assert arm["attribution"]["identity_residual"] == pytest.approx(0, abs=1e-12)
        assert arm["first_qualifying"]["origin_index"] == 9
        assert arm["first_accepted_application"]["decision_index"] == 10
        assert arm["pricing"] == {"index": 12, "date": series[12]["date"]}
        assert arm["confirmation"] == {"index": 13, "date": series[13]["date"],
                                       "status": "confirmed"}
        assert (path["daily"][0]["index"], path["daily"][-1]["index"]) == (10, 32)
    assert baseline == untouched
    assert set(episode["paired_terminal_return_deltas"].values()) == {0.0}


def test_one_buy_then_hold_ignores_negative_and_missing_held_heads_main_and_slow_endpoints():
    series = _series()
    observed = []
    for lag in (3, 4):
        case = _case(lag)
        baseline = _baseline(series, case)
        settlement = baseline["strategy"]["trades"][1]["execution_index"] + case["cash_lag"]
        origin = settlement - 1
        history = _histories(series, mean=-.5)
        for arm, rows in history.items():
            history[arm] = [row for row in rows if row["origin_index"] <= origin + 1]
            for row in history[arm]:
                if row["origin_index"] == origin:
                    row["prediction"]["expected_change"] = .5
        episode = _episode(forks.compare_reentry_forks(series, baseline, history, case))
        assert episode["common_end_index"] == settlement + (22 if lag == 3 else 23)
        assert episode["first_pricing_index"] == settlement + lag - 1
        observed.append(episode["common_end_index"] - settlement)
        for arm in episode["arms"].values():
            assert arm["status"] == "available"
            assert arm["strategy_summary"]["buy_count"] == 1
            assert arm["strategy_summary"]["redemption_count"] == 0
            assert arm["pending_application_count"] == 0
            assert len(arm["replay"]["strategy"]["trades"]) == 1
            assert not any(row["action"] == "redeem"
                           for row in arm["replay"]["strategy"]["action_trace"])
    assert observed == [22, 23]


def test_missing_exact_head_unavailable_but_waiting_cash_is_complete_and_late_buy_is_rejected():
    series, case = _series(), _case()
    baseline = _baseline(series, case)
    histories = _histories(series, mean=0)
    # Settlement s=10, initial origin s-A=9. L4 cannot substitute for L3.
    histories["baseline"] = [row for row in histories["baseline"]
                             if (row["origin_index"], row["entry_lag"]) != (9, 3)]
    for row in histories["recovery"]:
        if row["entry_lag"] == 3 and row["origin_index"] == 30:
            row["prediction"]["expected_change"] = .5
    result = forks.compare_reentry_forks(series, baseline, histories, case)
    episode = _episode(result)
    missing, late, cash, immediate = (episode["arms"][key] for key in
                                     ("baseline", "recovery", "empirical", "immediate"))
    assert missing["status"] == "unavailable"
    assert missing["unavailable_origin_indices"] == [9]
    assert "replay" not in missing
    assert episode["matched_four_arm"] is False
    assert episode["paired_terminal_return_deltas"] is None
    assert cash["status"] == "available" and cash["entry_status"] == "cash_not_entered"
    assert cash["application_status"] == "not_applied"
    assert cash["strategy_summary"]["return"] == 0
    assert cash["strategy_summary"]["terminal_liquidated_wealth"] == 80
    assert cash["attribution"]["initial_cash"] == 80
    assert cash["strategy_summary"]["trade_count"] == 0
    assert cash["replay"]["strategy"]["daily"][-1]["index"] == 32
    assert late["status"] == "available" and late["entry_status"] == "cash_not_entered"
    assert late["application_status"] == "qualifying_but_pricing_after_end"
    assert late["first_qualifying"]["origin_index"] == 30
    assert late["first_qualifying"]["scheduled_pricing_index"] == 33
    assert late["first_accepted_application"] is late["pricing"] is late["confirmation"] is None
    assert late["pending_application_count"] == 0
    assert late["rejected_applications"] == [{"origin_index": 30, "decision_index": 31,
                                               "scheduled_pricing_index": 33,
                                               "reason": "execution_after_replay_end"}]
    assert late["replay"]["strategy"]["trades"] == []
    assert late["strategy_summary"]["return"] == 0
    assert immediate["head"] is None and immediate["entry_status"] == "executed"
    assert result["counts"]["arms"]["recovery"]["cash_not_entered"] == 1
    assert result["counts"]["arms"]["baseline"]["unavailable"] == 1


def test_every_executed_sale_is_retained_and_unobserved_endpoints_never_replay(monkeypatch):
    case = _case()
    full = _series()
    baseline = _baseline(full, case)
    # Explicit sale anchors may settle outside the supplied observations.
    sale = deepcopy(baseline["strategy"]["trades"][1])
    sale.update(origin_index=62, decision_index=63, execution_index=65,
                execution_date=full[65]["date"], settlement_index=68,
                settlement_date=full[68]["date"])
    baseline["strategy"]["trades"].append(sale)
    short = full[:67]
    result = forks.compare_reentry_forks(short, baseline, _histories(short, 0), case)
    assert len(result["episodes"]) == 2
    assert len({row["episode_id"] for row in result["episodes"]}) == 2
    assert result["episodes"][0]["status"] == "available"
    assert result["episodes"][1]["status"] == "right_censored"
    assert result["episodes"][1]["reason"] == "settlement_after_last_observation"
    assert result["counts"]["right_censored"] == 1
    assert result["counts"]["arms"]["empirical"]["cash_not_entered"] == 1
    assert result["counts"]["arms"]["empirical"]["right_censored"] == 1

    def must_not_replay(*args, **kwargs):
        pytest.fail("censored episode must not run an invalid or shortened replay")

    monkeypatch.setattr(forks, "replay_actions", must_not_replay)
    for count, reason in ((9, "settlement_after_last_observation"),
                          (32, "common_end_after_last_observation")):
        result = forks.compare_reentry_forks(full[:count], baseline, {}, case)
        assert len(result["episodes"]) == 2
        first = result["episodes"][0]
        assert first["status"] == "right_censored" and first["reason"] == reason
        assert first["common_end_index"] == 32 and first["common_end_date"] is None
        for episode in result["episodes"]:
            assert not episode["matched_four_arm"]
            assert all(arm["status"] == "right_censored" and "replay" not in arm
                       for arm in episode["arms"].values())


def test_outcome_pollution_never_enters_callback_and_future_nav_cannot_change_choices(monkeypatch):
    series, case = _series(), _case(nav_availability_lag=2)
    baseline = _baseline(series, case)
    histories = _histories(series)
    original = deepcopy(histories)
    polluted = deepcopy(histories)
    for rows in polluted.values():
        for row in rows:
            row.update(actual_change=-999, future_nav=999, target_index=999,
                       outcome_status="invalid", realized_profit=-999)
            row["prediction"].update(actual_change=999, actual_nav=999, future_profit=999)
    real_replay, calls = forks.replay_actions, []

    def inspect_replay(series, actions, **kwargs):
        policy = kwargs["decision_policy"]

        def inspect_policy(item, state):
            assert "nav" not in item and "actual_change" not in item
            if "prediction" in item:
                assert set(item) == {"origin_index", "horizon", "entry_lag", "prediction"}
                assert set(item["prediction"]) == {"status", "expected_change"}
            else:
                assert set(item) == {"origin_index"}
            calls.append(deepcopy(item))
            return policy(item, state)

        return real_replay(series, actions, **{**kwargs, "decision_policy": inspect_policy})

    monkeypatch.setattr(forks, "replay_actions", inspect_replay)
    result = forks.compare_reentry_forks(series, baseline, histories, case)
    assert result == forks.compare_reentry_forks(series, baseline, polluted, case)
    episode = _episode(result)
    assert episode["settlement_index"] == 10
    assert episode["known_origin_index"] == 8
    assert episode["first_pricing_index"] == 11
    changed = deepcopy(series)
    for row in changed[10:]:
        row["nav"] = 100
    other = _episode(forks.compare_reentry_forks(changed, baseline, histories, case))
    for name in episode["arms"]:
        left, right = episode["arms"][name], other["arms"][name]
        assert left["first_qualifying"] == right["first_qualifying"]
        assert left["first_accepted_application"] == right["first_accepted_application"]
        assert [(row["origin_index"], row["action"], row["status"], row["reason"])
                for row in left["replay"]["strategy"]["action_trace"]] == [
                    (row["origin_index"], row["action"], row["status"], row["reason"])
                    for row in right["replay"]["strategy"]["action_trace"]]
    assert calls
    assert histories == original

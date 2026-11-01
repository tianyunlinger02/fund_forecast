"""Matched, post-redemption reentry episodes on the existing cash/units ledger.

Episodes share a net settled cash endowment and a fixed common endpoint. They
are overlapping counterfactuals, not independent trades or a compounded path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .domain import ValidationError
from .forecast_economic_validation import _fees, _ledger_args
from .forecast_research import ledger_attribution, replay_actions
from .timing_scenario import fee_adjusted_actions


_FORECAST_ARMS = ("baseline", "recovery", "empirical")
_ARMS = (*_FORECAST_ARMS, "immediate")
_HORIZON = 20


class _RequiredHeadUnavailable(Exception):
    def __init__(self, origin: int):
        self.origin = origin


def _date_at(series: Sequence[Mapping], index: int) -> str | None:
    return series[index]["date"] if 0 <= index < len(series) else None


def _history_map(rows: Sequence[Mapping], lag: int) -> dict:
    result = {}
    for row in rows:
        if (row["horizon"], row["entry_lag"]) != (_HORIZON, lag):
            continue
        origin = row["origin_index"]
        if origin in result:
            raise ValidationError("duplicate reentry-fork origin/exact head")
        result[origin] = row
    return result


def _known_packet(origin: int, lag: int, record: Mapping | None) -> dict:
    """Whitelist forecast-time inputs; never copy an outcome-bearing record."""
    prediction = record.get("prediction", {}) if record is not None else {}
    mean = prediction.get("expected_change")
    available = (prediction.get("status") == "available"
                 and not isinstance(mean, bool) and isinstance(mean, (int, float))
                 and math.isfinite(mean) and mean >= -1)
    return {"origin_index": origin, "horizon": _HORIZON, "entry_lag": lag,
            "prediction": {"status": "available" if available else "unavailable",
                           "expected_change": mean if available else None}}


def _summary(path: Mapping, initial_cash: float) -> dict:
    daily = path["daily"]
    buys = [trade for trade in path["trades"] if trade["action"] == "buy"]
    fees = path["fees_paid_before_terminal"] + path["terminal_liquidation_fee"]
    notional = math.fsum(trade["gross_notional"] for trade in path["trades"])
    peak, drawdown = initial_cash, 0.0
    for wealth in [row["marked_wealth"] for row in daily] + [path["terminal_liquidated_wealth"]]:
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1)
    return {
        "initial_cash": initial_cash,
        "terminal_liquidated_wealth": path["terminal_liquidated_wealth"],
        "terminal_wealth_per_initial_cash": path["terminal_liquidated_wealth"] / initial_cash,
        "return": path["terminal_liquidated_wealth"] / initial_cash - 1,
        "max_drawdown": drawdown, "trade_count": len(path["trades"]),
        "buy_count": len(buys),
        "redemption_count": sum(trade["action"] == "redeem" for trade in path["trades"]),
        "fees_paid_before_terminal": path["fees_paid_before_terminal"],
        "terminal_liquidation_fee": path["terminal_liquidation_fee"],
        "total_fees": fees, "total_fees_per_initial_cash": fees / initial_cash,
        "subscription_fees": math.fsum(trade["fee"] for trade in buys),
        "turnover_notional": notional, "turnover": notional / initial_cash,
        "turnover_basis": "executed one-way gross notional / episode net exit cash B; "
                          "excludes the prior sale and hypothetical terminal liquidation",
        "terminal_cash_available": daily[-1]["cash_available"],
        "terminal_cash_reserved": daily[-1]["cash_reserved"],
        "terminal_cash_in_transit": daily[-1]["cash_in_transit"],
    }


def _timing(series: Sequence[Mapping], origin: int, case: Mapping) -> dict:
    decision = origin + case["nav_availability_lag"]
    pricing = origin + case["entry_lag"]
    return {"origin_index": origin, "origin_date": _date_at(series, origin),
            "decision_index": decision, "decision_date": _date_at(series, decision),
            "scheduled_pricing_index": pricing, "scheduled_pricing_date": _date_at(series, pricing)}


def _replay_arm(series: Sequence[Mapping], history: Mapping, case: Mapping,
                *, arm: str, origin: int, settlement: int, end: int, initial_cash: float) -> dict:
    lag, availability = case["entry_lag"], case["nav_availability_lag"]
    immediate = arm == "immediate"
    packets = ([{"origin_index": origin}] if immediate else
               [_known_packet(i, lag, history.get(i)) for i in range(origin, end - availability + 1)])
    diagnostics = {}
    fees = _fees(case)

    def decide(item, state):
        # Once an application is accepted, pending or priced units prevent a
        # second buy. Held forecasts (including negative/missing heads) do not
        # trigger another exit in this isolated reentry experiment.
        if not state["can_buy"]:
            return "hold"
        if immediate:
            qualifies, margin = True, None
        else:
            prediction = item["prediction"]
            if prediction["status"] != "available":
                raise _RequiredHeadUnavailable(item["origin_index"])
            choices = fee_adjusted_actions(prediction["expected_change"], **fees)
            if choices["flat"] == "unavailable":
                raise _RequiredHeadUnavailable(item["origin_index"])
            qualifies, margin = choices["flat"] == "buy", choices["buy_advantage_vs_cash"]
        if qualifies:
            if "first_qualifying" not in diagnostics:
                # The engine deep-copies callback packets; retain this known
                # input diagnostic separately, then read acceptance from trace.
                diagnostics["first_qualifying"] = {
                    "origin_index": item["origin_index"],
                    "criterion": "immediate" if immediate else "fee_adjusted_flat_buy",
                    "expected_change": None if immediate else item["prediction"]["expected_change"],
                    "buy_advantage_vs_cash": margin,
                }
            return "buy"
        return "hold"

    result = {"arm": arm, "head": None if immediate else [_HORIZON, lag]}
    try:
        replay = replay_actions(series, packets, **_ledger_args(case), initial_cash=initial_cash,
                                start_index=settlement, end_index=end, decision_policy=decide)
    except _RequiredHeadUnavailable as exc:
        result.update(status="unavailable", reason="missing_or_unavailable_required_exact_head",
                      unavailable_origin_indices=[exc.origin], entry_status="unavailable")
    except ValidationError as exc:
        result.update(status="unavailable", reason=str(exc), entry_status="unavailable")
    else:
        path = replay["strategy"]
        applications = [row for row in path["action_trace"] if row["action"] == "buy"]
        accepted = next((row for row in applications if row["status"] in {"executed", "pending"}), None)
        bought = next((trade for trade in path["trades"] if trade["action"] == "buy"), None)
        after_end = any(row["reason"] == "execution_after_replay_end" for row in applications)
        result.update(
            status="available", entry_status="executed" if bought is not None else "cash_not_entered",
            application_status=("executed" if bought is not None else "pending" if accepted is not None
                                else "qualifying_but_pricing_after_end" if after_end else "not_applied"),
            first_accepted_application=(
                {**_timing(series, accepted["origin_index"], case), "status": accepted["status"]}
                if accepted is not None else None),
            pricing=({"index": bought["execution_index"], "date": bought["execution_date"]}
                     if bought is not None else None),
            confirmation=({"index": bought["confirmation_index"], "date": bought["confirmation_date"],
                           "status": bought["confirmation_status"]} if bought is not None else None),
            pending_application_count=sum(row["status"] == "pending" for row in applications),
            rejected_applications=[{"origin_index": row["origin_index"],
                                    "decision_index": row["decision_index"],
                                    "scheduled_pricing_index": row["scheduled_execution_index"],
                                    "reason": row["reason"]}
                                   for row in applications if row["status"] == "skipped"],
            replay=replay, strategy_summary=_summary(path, initial_cash),
            attribution=ledger_attribution(path, series, initial_cash=initial_cash),
        )
    qualifying = diagnostics.get("first_qualifying")
    result["first_qualifying"] = ({**qualifying, **_timing(series, qualifying["origin_index"], case)}
                                  if qualifying is not None else None)
    return result


def compare_reentry_forks(series: Sequence[Mapping], baseline_replay: Mapping,
                          histories: Mapping[str, Sequence[Mapping]], case: Mapping) -> dict:
    """Fork all executed baseline sales into three forecast arms and immediate buy.

    At sale pricing e, B = gross notional - its already-paid fee. Settlement
    s=e+cash_lag supplies B; the first known origin is s-A, not s. All arms end
    at (s-A+entry_lag)+20, regardless of when (or whether) they enter. Missing
    exact heads matter only while flat and eligible to buy. Unobserved common
    endpoints remain censored episodes, never shortened or replayed.
    """
    maps = {arm: _history_map(histories.get(arm, ()), case["entry_lag"])
            for arm in _FORECAST_ARMS}
    last = len(series) - 1
    episodes = []
    for trade_index, sale in enumerate(baseline_replay["strategy"]["trades"]):
        if sale["action"] != "redeem":
            continue
        execution = sale["execution_index"]
        settlement = execution + case["cash_lag"]
        origin = settlement - case["nav_availability_lag"]
        pricing = origin + case["entry_lag"]
        end = pricing + _HORIZON
        initial_cash = sale["gross_notional"] - sale["fee"]
        episode = {
            "episode_id": f"{case['id']}:sale:{trade_index}:execution:{execution}",
            "sale_trade_index": trade_index, "sale": dict(sale),
            "sale_execution_index": execution, "sale_execution_date": _date_at(series, execution),
            "settlement_index": settlement, "settlement_date": _date_at(series, settlement),
            "known_origin_index": origin, "known_origin_date": _date_at(series, origin),
            "first_pricing_index": pricing, "first_pricing_date": _date_at(series, pricing),
            "common_end_index": end, "common_end_date": _date_at(series, end),
            "gross_exit_notional": sale["gross_notional"], "prior_exit_fee": sale["fee"],
            "initial_cash": initial_cash, "net_exit_cash": initial_cash,
        }
        if settlement > last or end > last:
            reason = ("settlement_after_last_observation" if settlement > last
                      else "common_end_after_last_observation")
            episode.update(status="right_censored", reason=reason, matched_four_arm=False,
                           arms={arm: {"arm": arm, "status": "right_censored", "reason": reason,
                                       "entry_status": "not_evaluated_right_censored"} for arm in _ARMS},
                           paired_terminal_return_deltas=None)
        else:
            arms = {arm: _replay_arm(series, maps.get(arm, {}), case, arm=arm, origin=origin,
                                     settlement=settlement, end=end, initial_cash=initial_cash)
                    for arm in _ARMS}
            matched = all(arm["status"] == "available" for arm in arms.values())
            episode.update(status="available" if matched else "with_unavailable_arms",
                           arms=arms, matched_four_arm=matched,
                           paired_terminal_return_deltas=(
                               {f"{arm}_minus_{reference}": arms[arm]["strategy_summary"]["return"]
                                - arms[reference]["strategy_summary"]["return"]
                                for reference in ("baseline", "immediate") for arm in _ARMS
                                if arm != reference} if matched else None))
            def nav(i):
                return float(series[i]["nav"])

            episode["evaluation_only"] = {
                "nav_changes": {"sale_to_settlement": nav(settlement) / nav(execution) - 1,
                                "settlement_to_first_pricing": nav(pricing) / nav(settlement) - 1,
                                "first_pricing_to_common_end": nav(end) / nav(pricing) - 1},
                "basis": "realized NAV diagnostics only; never passed to decision callbacks",
            }
        episodes.append(episode)
    counts = {
        "episodes": len(episodes),
        "matched_four_arm": sum(row["matched_four_arm"] for row in episodes),
        "right_censored": sum(row["status"] == "right_censored" for row in episodes),
        "with_unavailable_arms": sum(row["status"] == "with_unavailable_arms" for row in episodes),
        "arms": {arm: {status: sum(row["arms"][arm]["status"] == status for row in episodes)
                       for status in ("available", "unavailable", "right_censored")}
                 for arm in _ARMS},
    }
    for arm in _ARMS:
        counts["arms"][arm].update({
            status: sum(row["arms"][arm].get("entry_status") == status for row in episodes)
            for status in ("executed", "cash_not_entered")})
    return {
        "schema": "rmbfund.reentry-forks.v1", "case": case["id"], "parameters": dict(case),
        "episodes": episodes, "counts": counts,
        "basis": "matched four-arm post-exit episodes; at most one buy then hold; terminal/B-1; "
                 "prior sale fee already removed from B; cash earns zero interest; overlapping "
                 "episodes are not independent, bootstrapped, or compounded",
        "skill_certified": False,
    }

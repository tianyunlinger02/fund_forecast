"""A fixed post-redemption forecast replacement, evaluated on the existing ledger.

Only reentry changes. An earlier reentry can change later executed sales even
though the original held-position decision function is never changed.
"""
from __future__ import annotations

import json
import math
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

from .domain import ValidationError, now_iso
from .forecast_candidates import RECOVERY_ID, issue_recovery_forecasts
from .forecast_comparison import (
    EVIDENCE, _json_write, _path_summary, audit_availability,
    compare_paired_losses, evaluate_candidate_rows,
)
from .forecast_economic_validation import (
    _decision_policy, _fees, _ledger_args, empirical_history,
    prepare_economic_history,
)
from .forecast_reentry_forks import compare_reentry_forks
from .forecast_research import ledger_attribution, paired_delta_bootstrap, replay_actions
from .timing_scenario import _calendar_value, fee_adjusted_actions

VARIANTS = ("baseline", "recovery", "empirical", "immediate")
# These fields exist at the forecast origin. Outcomes never enter policy packets.
PREDICTION_FIELDS = (
    "status", "method", "expected_change", "reason", "fallback_used",
    "training_log_mean", "conditional_log_contribution", "candidate_log_location",
    "residual_log_mean", "training_sample_count", "residual_sample_count",
    "training_target_end_index", "residual_target_end_index",
    "unshrunk_recovery_log_location", "old_ridge_log_location", "log_return_5",
    "simple_drawdown_60", "recovery_feature", "matured_nonzero_recovery_training_count",
)


def _known_prediction(row: Mapping | None) -> dict:
    if row is None:
        return {"status": "unavailable", "reason": "missing_exact_head"}
    prediction = row["prediction"]
    return {key: prediction[key] for key in PREDICTION_FIELDS if key in prediction}


def _mean(prediction: Mapping) -> float:
    value = prediction.get("expected_change")
    if (prediction.get("status") != "available" or prediction.get("fallback_used") is True
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < -1):
        raise ValidationError("required reentry forecast unavailable or invalid; no substitution")
    return value


def _maps(histories: Mapping[str, Sequence[Mapping]]) -> dict:
    result = {}
    for variant, rows in histories.items():
        mapping = {}
        for row in rows:
            key = (row["origin_index"], row["horizon"], row["entry_lag"])
            if key in mapping:
                raise ValidationError("duplicate reentry origin/head")
            mapping[key] = row
        result[variant] = mapping
    return result


def reentry_packets(histories: Mapping[str, Sequence[Mapping]], start_origin: int,
                    end_index: int, case: Mapping) -> list[dict]:
    """Complete daily packets, including outcomes that have not matured yet."""
    maps = _maps(histories)
    packets = []
    for origin in range(start_origin, end_index):
        key = origin, 20, case["entry_lag"]
        row = maps["baseline"].get(key)
        base = _known_prediction(row)
        _mean(base)  # Baseline is required in every state, not just while flat.
        packets.append({
            "origin_index": origin, "origin_date": row["origin_date"],
            "horizon": 20, "entry_lag": case["entry_lag"], "prediction": base,
            "reentry_predictions": {variant: _known_prediction(maps[variant].get(key))
                                    for variant in ("recovery", "empirical")},
        })
    return packets


def reentry_policy(variant: str, case: Mapping, diagnostics: list[dict]):
    """Fresh state for one complete, single-callback-per-observation replay."""
    if variant not in VARIANTS:
        raise ValidationError("unknown reentry variant")
    base_policy = _decision_policy("net_payoff", case)
    previous_has_units = ever_redeemed = False
    episode = 0

    def decide(item, state):
        nonlocal previous_has_units, ever_redeemed, episode
        if previous_has_units and not state["has_units"]:
            ever_redeemed = True
            episode += 1
        previous_has_units = state["has_units"]
        base_mean = _mean(item["prediction"])
        base_action = base_policy(item, state)
        gate = ever_redeemed and state["can_buy"]
        selected = "baseline"
        prediction, selected_mean, action = item["prediction"], base_mean, base_action
        if gate:
            selected = variant
            if variant == "immediate":
                prediction, selected_mean, action = {}, None, "buy"
            else:
                prediction = (item["prediction"] if variant == "baseline"
                              else item["reentry_predictions"][variant])
                try:
                    selected_mean = _mean(prediction)
                except ValidationError as exc:
                    raise ValidationError(f"{exc}; variant={variant}, origin={item['origin_index']}") from exc
                choice = fee_adjusted_actions(selected_mean, **_fees(case))["flat"]
                action = "buy" if choice == "buy" else "hold"
        diagnostics.append({
            "origin_index": item["origin_index"], "origin_date": item["origin_date"],
            "decision_index": item["origin_index"] + case["nav_availability_lag"],
            "state": dict(state), "ever_redeemed": ever_redeemed, "episode": episode,
            "reentry_gate": gate, "base_mean": base_mean, "base_action": base_action,
            "selected_model": selected, "selected_mean": selected_mean,
            "selected_prediction": dict(prediction), "action": action,
            "entry_advantage": (fee_adjusted_actions(selected_mean, **_fees(case))["buy_advantage_vs_cash"]
                                if selected_mean is not None else None),
        })
        return action
    return decide


def holding_episodes(path: Mapping) -> list[dict]:
    """Actual executed holding periods, not an imposed minimum holding rule."""
    result, entry = [], None
    for trade in path["trades"]:
        if trade["action"] == "buy":
            entry = trade
        elif entry is not None:
            span = trade["execution_index"] - entry["execution_index"]
            result.append({"entry": entry, "sale": trade, "observations_held": span,
                           "is_reentry": bool(result), "short_round_trip": span <= 20,
                           "executed_fees": entry["fee"] + trade["fee"]})
            entry = None
    if entry is not None:
        result.append({"entry": entry, "sale": None, "observations_held": None,
                       "is_reentry": bool(result), "short_round_trip": False,
                       "executed_fees": entry["fee"]})
    return result


def exit_episodes(path: Mapping, diagnostics: Sequence[Mapping], series: Sequence[Mapping],
                  case: Mapping) -> list[dict]:
    """Inventory every actual exit; retrospective price changes are labels only."""
    trades, result, end = path["trades"], [], len(series) - 1
    for index, sale in enumerate(trades):
        if sale["action"] != "redeem":
            continue
        entry = next((trade for trade in trades[index + 1:] if trade["action"] == "buy"), None)
        settlement = sale["execution_index"] + case["cash_lag"]
        eligible = next((row for row in diagnostics
                         if row["decision_index"] >= sale["execution_index"]
                         and (entry is None or row["decision_index"] <= entry["decision_index"])
                         and row["state"]["can_buy"]), None)
        first_eligible = eligible["decision_index"] if eligible else None
        attempt = next((trace for trace in path["action_trace"]
                        if trace["action"] == "buy" and trace["state"]["can_buy"]
                        and trace["decision_index"] >= settlement
                        and (entry is None or trace["decision_index"] <= entry["decision_index"])), None)
        late_pricing = attempt is not None and attempt["reason"] == "execution_after_replay_end"
        row = {"episode": len(result) + 1, "sale": sale, "reentry": entry,
               "first_qualifying_buy_application": attempt,
               "settlement_index": settlement,
               "settlement_date": series[settlement]["date"] if settlement <= end else None,
               "first_eligible_decision_index": first_eligible,
               "first_eligible_known_origin": eligible["origin_index"] if eligible else None,
               "cash_transit_observations": case["cash_lag"],
               "eligible_signal_wait_observations": (attempt["decision_index"] - first_eligible
                                                     if attempt and eligible else None),
               "application_to_pricing_observations": (entry["execution_index"] - entry["decision_index"]
                                                       if entry else None),
               "status": ("reentered" if entry else "cash_in_transit_at_end" if settlement > end
                          else "qualifying_but_pricing_after_end" if late_pricing else "not_reentered"),
               "net_sale_proceeds": sale["gross_notional"] - sale["fee"]}
        if settlement <= end:
            row["realized_nav_change_sale_to_settlement"] = (float(series[settlement]["nav"])
                                                             / sale["execution_nav"] - 1)
            if entry:
                row["realized_nav_change_settlement_to_reentry"] = (entry["execution_nav"]
                                                                     / float(series[settlement]["nav"]) - 1)
        result.append(row)
    return result


def compare_reentry_paths(series: Sequence[Mapping], histories: Mapping,
                          cases: Sequence[Mapping], *, start_origin: int) -> dict:
    end = len(series) - 1
    result = {"schema": "rmbfund.reentry-validation.continuous.v1", "start_origin": start_origin,
              "scenarios": [], "skill_certified": False, **EVIDENCE}
    for case in cases:
        for variant in VARIANTS:
            row = {"case": case["id"], "variant_id": variant, "parameters": dict(case),
                   "head": [20, case["entry_lag"]], "expected_decision_count": end - start_origin}
            diagnostics = []
            try:
                packets = reentry_packets(histories, start_origin, end, case)
                replay = replay_actions(series, packets, **_ledger_args(case),
                                        start_index=start_origin + case["nav_availability_lag"],
                                        end_index=end,
                                        decision_policy=reentry_policy(variant, case, diagnostics))
                path = replay["strategy"]
                summary, passive = _path_summary(path), _path_summary(replay["buy_and_hold"])
                holdings = holding_episodes(path)
                short = [item for item in holdings if item["is_reentry"] and item["short_round_trip"]]
                row.update(status="available", replay=replay, diagnostics=diagnostics,
                           strategy_summary=summary, buy_and_hold_summary=passive,
                           holding_episodes=holdings,
                           exit_episodes=exit_episodes(path, diagnostics, series, case),
                           short_reentry_round_trip_count=len(short),
                           short_reentry_round_trip_fees=sum(item["executed_fees"] for item in short),
                           reentry_gate_count=sum(item["reentry_gate"] for item in diagnostics),
                           reentry_triggered=any(item["ever_redeemed"] for item in diagnostics),
                           attribution=ledger_attribution(path, series),
                           buy_hold_attribution=ledger_attribution(replay["buy_and_hold"], series))
                trace = [{"origin_index": item["origin_index"], "action": item["action"]}
                         for item in path["action_trace"]]
                if case["buy_fee"] or case["redemption_fee"]:
                    zero = replay_actions(series, trace,
                                          **{**_ledger_args(case), "entry_fee_rate": 0., "redemption_fee_rate": 0.},
                                          start_index=start_origin + case["nav_availability_lag"], end_index=end)
                    row["same_signals_zero_fee"] = {
                        "return": zero["strategy"]["return"],
                        "cost_drag": zero["strategy"]["return"] - summary["return"],
                        "basis": "original selected actions frozen; no zero-fee policy recomputation",
                    }
            except ValidationError as exc:
                row.update(status="unavailable", reason=str(exc), diagnostics=diagnostics)
            result["scenarios"].append(row)
    result["status"] = ("available" if all(r["status"] == "available" for r in result["scenarios"])
                        else "with_unavailable_scenarios")
    return result


def baseline_parity(current: Mapping, stored: Mapping) -> dict:
    """Packets have new annotations; actual accounting and trade tapes must not."""
    differences = []
    for branch in ("strategy", "buy_and_hold", "cash"):
        for key, expected in stored[branch].items():
            if key == "action_trace":
                # Compare lifecycle/state but not the additional known predictors.
                fields = ("origin_index", "action", "decision_index", "state", "status", "reason",
                          "execution_index", "confirmation_index", "settlement_index")
                left = [{name: r[name] for name in fields} for r in current[branch][key]]
                right = [{name: r[name] for name in fields} for r in expected]
                equal = left == right
            else:
                equal = current[branch].get(key) == expected
            if not equal:
                differences.append(f"{branch}.{key}")
    return {"matched": not differences, "differences": differences,
            "basis": "exact original daily accounting, trade tape, skips, fees and terminal values"}


def _baseline_cohort(scenario: Mapping) -> set[int]:
    return {row["origin_index"] for row in scenario["diagnostics"] if row["reentry_gate"]}


def forecast_diagnostics(bundles: Mapping, baseline_scenarios: Sequence[Mapping],
                         periods: Mapping, heads: Sequence, bootstrap: Mapping) -> dict:
    full = evaluate_candidate_rows(bundles, heads=heads, periods=periods,
                                   mature_by_period_end=True,
                                   fitted_candidate_ids=("baseline", "recovery"))
    cohorts = {}
    for scenario in baseline_scenarios:
        if scenario["status"] != "available":
            continue
        indices = _baseline_cohort(scenario)
        selected = {variant: {"evaluation_rows": [r for r in bundle["evaluation_rows"]
                                                   if r["origin_index"] in indices
                                                   and r["entry_lag"] == scenario["head"][1]]}
                    for variant, bundle in bundles.items()}
        cohorts[scenario["case"]] = {
            "origin_indices": sorted(indices), "origin_count_including_immature": len(indices),
            "metrics": evaluate_candidate_rows(selected, heads=(tuple(scenario["head"]),),
                                                periods=periods, mature_by_period_end=True,
                                                fitted_candidate_ids=("baseline", "recovery")),
            "basis": "fixed baseline post-sale spendable-cash cohort; descriptive, no episode CI",
        }
    paired = compare_paired_losses(
        {key: bundles[key] for key in ("baseline", "recovery")}, heads=heads,
        candidate_id="recovery", reference_id="baseline", periods=periods,
        mature_by_period_end=True, replicates=bootstrap["replicates"], seed=bootstrap["seed"])
    return {"full": full, "baseline_flat_cohorts": cohorts, "paired_forecast_losses": paired}


def flat_payoff_diagnostics(bundles: Mapping, cases: Sequence[Mapping], periods: Mapping,
                            bootstrap: Mapping) -> dict:
    maps = {name: {(r["origin_index"], r["horizon"], r["entry_lag"]): r
                   for r in bundle["evaluation_rows"] if r["prediction"].get("status") == "available"}
            for name, bundle in bundles.items()}
    keys = sorted(set.intersection(*(set(rows) for rows in maps.values())))
    result = {"basis": "one flat decision with capital 1 and future liquidation; NOT continuous path CI",
              "cases": []}
    for case in cases:
        rows = []
        for key in keys:
            if key[1:] != (20, case["entry_lag"]):
                continue
            reference = maps["baseline"][key]
            values = {}
            for name, mapping in maps.items():
                pred = mapping[key]["prediction"]
                action = fee_adjusted_actions(pred["expected_change"], **_fees(case))["flat"]
                value = _calendar_value(action, reference["actual_change"], 1., _fees(case))
                values[name] = {"action": action, "actual_terminal_value": value}
            rows.append({key: reference[key] for key in ("origin_index", "origin_date", "horizon", "entry_lag",
                                                       "entry_index", "target_index", "target_date")} |
                        {"values": values})
        scored = {}
        for name, (start, finish) in periods.items():
            selected = [r for r in rows if start <= r["origin_date"] <= finish and r["target_date"] <= finish]
            deltas = {f"recovery_minus_{reference}": [r["values"]["recovery"]["actual_terminal_value"]
                                                     - r["values"][reference]["actual_terminal_value"]
                                                     for r in selected]
                      for reference in ("baseline", "empirical")}
            deltas["recovery_minus_cash"] = [r["values"]["recovery"]["actual_terminal_value"] - 1 for r in selected]
            scored[name] = paired_delta_bootstrap(selected, deltas, replicates=bootstrap["replicates"],
                                                   seed=bootstrap["seed"])
        result["cases"].append({"case": case["id"], "rows": rows, "periods": scored})
    return result


def _summary(row: Mapping) -> dict:
    omitted = {"replay", "diagnostics", "holding_episodes", "attribution", "buy_hold_attribution"}
    result = {key: value for key, value in row.items() if key not in omitted}
    for field in ("attribution", "buy_hold_attribution"):
        if field in row:
            result[field] = {key: value for key, value in row[field].items() if key != "daily"}
    return result


def _reproduce_baseline(source: Mapping, prior: Path, folder: Path, cases: Sequence) -> dict:
    frozen = json.loads(Path(source["path"]).read_text())
    series = frozen["series"]
    old_folder = prior / source["symbol"]
    old_report = json.loads((old_folder / "report.json").read_text())
    original = json.loads((old_folder / "shrink-candidate.json").read_text())
    old_ridge = json.loads((old_folder / "old-ridge.json").read_text())
    base = prepare_economic_history(series, original)
    empirical = prepare_economic_history(series, empirical_history(old_ridge))
    histories = {"baseline": base["historical_forecasts"],
                 "empirical": empirical["historical_forecasts"], "recovery": []}
    reproduction = []
    for case in cases:
        diagnostics = []
        packets = reentry_packets(histories, old_report["start_origin"], len(series) - 1, case)
        replay = replay_actions(series, packets, **_ledger_args(case),
                                start_index=old_report["start_origin"] + case["nav_availability_lag"],
                                decision_policy=reentry_policy("baseline", case, diagnostics))
        stored = json.loads((old_folder / f"{case['id']}-nav_ridge_shrink50_empirical_v1-net_payoff.json").read_text())
        reproduction.append({"case": case["id"], **baseline_parity(replay, stored["replay"])})
    _json_write(folder / "baseline-reproduction.json", reproduction)
    if not all(row["matched"] for row in reproduction):
        raise ValidationError("original baseline replay mismatch; recovery candidate not run")
    return {"frozen": frozen, "base": base, "empirical": empirical,
            "old_report": old_report, "reproduction": reproduction}


def run_reentry_validation(protocol_path: str | Path, output: str | Path) -> dict:
    protocol = json.loads(Path(protocol_path).read_text())
    out, prior = Path(output), Path(protocol["prior_study"])
    out.mkdir(parents=True, exist_ok=False)
    heads = tuple(tuple(pair) for pair in protocol["heads"])
    report = {"schema": "rmbfund.reentry-validation.report.v1", "started_at": now_iso(),
              "candidate_id": RECOVERY_ID, "protocol_path": str(protocol_path),
              "status": "running", "funds": [], "failures": [], "promotion": False,
              "skill_certified": False, "network": False, "database_access": False, **EVIDENCE}
    _json_write(out / "report.json", report)
    # Reproduce all twelve fixed baseline cases before any new candidate fit.
    reproduced = {}
    for source in protocol["inputs"]:
        folder = out / source["symbol"]
        folder.mkdir()
        try:
            reproduced[source["symbol"]] = _reproduce_baseline(source, prior, folder, protocol["execution_cases"])
        except Exception as exc:
            failure = {"symbol": source["symbol"], "stage": "baseline_reproduction",
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            _json_write(folder / "failure.json", failure)
            report["failures"].append(failure)
    if report["failures"]:
        report.update(status="baseline_reproduction_failed", finished_at=now_iso())
        _json_write(out / "report.json", report)
        return report
    for source in protocol["inputs"]:
        folder = out / source["symbol"]
        stage = "recovery_candidate"
        try:
            prepared = reproduced[source["symbol"]]
            frozen, old_report = prepared["frozen"], prepared["old_report"]
            series = frozen["series"]
            base, empirical, reproduction = prepared["base"], prepared["empirical"], prepared["reproduction"]
            recovery_raw = issue_recovery_forecasts(series, heads=heads)
            _json_write(folder / "recovery-candidate.json", recovery_raw)
            recovery = prepare_economic_history(series, recovery_raw)
            bundles = {"baseline": base, "recovery": recovery, "empirical": empirical}
            histories = {name: bundle["historical_forecasts"] for name, bundle in bundles.items()}
            stage = "continuous_paths"
            continuous = compare_reentry_paths(series, histories, protocol["execution_cases"],
                                                start_origin=old_report["start_origin"])
            for row in continuous["scenarios"]:
                _json_write(folder / f"{row['case']}-{row['variant_id']}.json", row)
            baseline_scenarios = [row for row in continuous["scenarios"] if row["variant_id"] == "baseline"]
            stage = "matched_exit_forks"
            forks = {}
            for case in protocol["execution_cases"]:
                baseline = next(row for row in baseline_scenarios if row["case"] == case["id"])
                if baseline["status"] != "available":
                    raise ValidationError("baseline became unavailable after reproduction")
                value = compare_reentry_forks(series, baseline["replay"], histories, case)
                _json_write(folder / f"forks-{case['id']}.json", value)
                forks[case["id"]] = value.get("counts", {})
            stage = "forecast_diagnostics"
            periods = {name: (start, min(finish, series[-1]["date"]))
                       for name, (start, finish) in protocol["folds"].items()}
            periods.update(all=("2024-07-01", series[-1]["date"]),
                           inspected_2026=("2026-01-01", series[-1]["date"]))
            diagnostics = forecast_diagnostics(bundles, baseline_scenarios, periods, heads, protocol["bootstrap"])
            _json_write(folder / "forecast-diagnostics.json", diagnostics)
            payoffs = flat_payoff_diagnostics(bundles, protocol["execution_cases"], periods, protocol["bootstrap"])
            _json_write(folder / "flat-payoffs.json", payoffs)
            fund_report = {"symbol": source["symbol"], "status": "completed",
                           "input_path": source["path"], "availability": audit_availability(frozen),
                           "start_origin": old_report["start_origin"], "start_date": old_report["start_date"],
                           "end_date": series[-1]["date"], "baseline_parity": reproduction,
                           "scenarios": [_summary(row) for row in continuous["scenarios"]],
                           "unavailable_scenario_count": sum(row["status"] != "available" for row in continuous["scenarios"]),
                           "unavailable_scenarios": [{key: row[key] for key in ("case", "variant_id", "reason")}
                                                     for row in continuous["scenarios"] if row["status"] != "available"],
                           "fork_counts": forks, "promotion": False, "skill_certified": False, **EVIDENCE}
            _json_write(folder / "report.json", fund_report)
            report["funds"].append(fund_report)
        except Exception as exc:
            failure = {"symbol": source["symbol"], "stage": stage, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}
            _json_write(folder / "failure.json", failure)
            report["failures"].append(failure)
        _json_write(out / "report.json", report)
    unavailable = sum(fund["unavailable_scenario_count"] for fund in report["funds"])
    report.update(status=("completed_with_failures" if report["failures"]
                          else "completed_with_unavailable_scenarios" if unavailable else "completed"),
                  scenario_count=sum(len(fund["scenarios"]) for fund in report["funds"]),
                  unavailable_scenario_count=unavailable, finished_at=now_iso())
    _json_write(out / "report.json", report)
    return report

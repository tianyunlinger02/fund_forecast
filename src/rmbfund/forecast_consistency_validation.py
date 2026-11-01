"""Fixed forecast-source routing ablation on retained, complete prediction histories.

The payoff rule and ledger are unchanged. Same-origin coherence is not evidence
of timing skill or a dynamic optimum over future entry/exit opportunities.
"""
from __future__ import annotations

import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

from .domain import ValidationError, now_iso
from .forecast_comparison import EVIDENCE, _json_write, _path_summary
from .forecast_economic_validation import (
    _decision_policy, _fees, _ledger_args, empirical_history, prepare_economic_history,
)
from .forecast_reentry_validation import (
    _known_prediction, _maps, _mean, baseline_parity, exit_episodes, holding_episodes,
)
from .forecast_research import ledger_attribution, replay_actions
from .timing_scenario import fee_adjusted_actions

VARIANTS = ("baseline", "hybrid", "post_exit_empirical", "uniform_empirical")
CONTROLS = ("baseline", "hybrid", "uniform_empirical")


def consistency_packets(series: Sequence[Mapping], histories: Mapping, start_origin: int,
                        end_index: int, case: Mapping) -> list[dict]:
    """Whitelist then-known fields; unused sources may be absent or unavailable."""
    maps = _maps(histories)
    return [{
        "origin_index": origin, "origin_date": series[origin]["date"],
        "horizon": 20, "entry_lag": case["entry_lag"],
        "source_predictions": {
            source: _known_prediction(maps.get(source, {}).get((origin, 20, case["entry_lag"])))
            for source in ("baseline", "empirical")
        },
    } for origin in range(start_origin, end_index)]


def consistency_policy(variant: str, case: Mapping, diagnostics: list[dict]):
    """One fresh sticky source regime, preserving blocked attempted signals too."""
    if variant not in VARIANTS:
        raise ValidationError("unknown consistency variant")
    policy = _decision_policy("net_payoff", case)
    previous_has_units = actual_exit_observed = False
    episode = 0

    def decide(item, state):
        nonlocal previous_has_units, actual_exit_observed, episode
        if previous_has_units and not state["has_units"]:
            actual_exit_observed = True
            episode += 1
        previous_has_units = state["has_units"]
        gate = actual_exit_observed and state["can_buy"]
        source = "empirical" if (
            variant == "uniform_empirical"
            or variant == "post_exit_empirical" and actual_exit_observed
            or variant == "hybrid" and gate
        ) else "baseline"
        prediction = item["source_predictions"][source]
        # Account eligibility does not assert that pricing fits the replay end.
        scope = "account_eligible_signal_trace" if state["can_buy"] or state["can_redeem"] else "blocked_signal_trace"
        row = {
            "origin_index": item["origin_index"], "origin_date": item["origin_date"],
            "decision_index": item["origin_index"] + case["nav_availability_lag"],
            "state": dict(state), "actual_exit_observed": actual_exit_observed,
            "ever_redeemed": actual_exit_observed, "episode": episode, "reentry_gate": gate,
            "selected_source": source, "selected_prediction": dict(prediction),
            "availability_scope": scope,
        }
        # The hybrid's old callback evaluates S even at its E-only buy gate.
        required = ("baseline", "empirical") if variant == "hybrid" and gate else (source,)
        means = {}
        for required_source in required:
            try:
                means[required_source] = _mean(item["source_predictions"][required_source])
            except ValidationError as exc:
                row.update(status="unavailable", reason=str(exc), unavailable_source=required_source)
                diagnostics.append(row)
                raise ValidationError(
                    f"required consistency head unavailable; variant={variant}, source={required_source}, "
                    f"origin={item['origin_index']}, scope={scope}: {exc}"
                ) from exc
        mean = means[source]
        choices = fee_adjusted_actions(mean, **_fees(case))
        action = policy({**item, "prediction": prediction}, state)
        row.update(status="available", selected_mean=mean, action=action,
                   buy_advantage_vs_cash=choices["buy_advantage_vs_cash"],
                   hold_advantage_vs_redeem=choices["hold_advantage_vs_redeem"],
                   counterfactual_flat=choices["flat"], counterfactual_held=choices["held"])
        diagnostics.append(row)
        return action
    return decide


def compare_consistency_path(series: Sequence[Mapping], histories: Mapping, case: Mapping,
                             *, variant: str, start_origin: int) -> dict:
    end, diagnostics = len(series) - 1, []
    result = {"case": case["id"], "variant_id": variant, "parameters": dict(case),
              "head": [20, case["entry_lag"]], "start_origin": start_origin,
              "expected_decision_count": end - start_origin, "diagnostics": diagnostics}
    try:
        packets = consistency_packets(series, histories, start_origin, end, case)
        args = _ledger_args(case)
        replay = replay_actions(
            series, packets, **args, start_index=start_origin + case["nav_availability_lag"],
            end_index=end, decision_policy=consistency_policy(variant, case, diagnostics),
        )
        path = replay["strategy"]
        holdings = holding_episodes(path)
        short = [row for row in holdings if row["is_reentry"] and row["short_round_trip"]]
        result.update(
            status="available", replay=replay, strategy_summary=_path_summary(path),
            buy_and_hold_summary=_path_summary(replay["buy_and_hold"]),
            holding_episodes=holdings, exit_episodes=exit_episodes(path, diagnostics, series, case),
            short_reentry_round_trip_count=len(short),
            short_reentry_round_trip_fees=sum(row["executed_fees"] for row in short),
            actual_exit_observed=any(row["actual_exit_observed"] for row in diagnostics),
            reentry_gate_count=sum(row["reentry_gate"] for row in diagnostics),
            attribution=ledger_attribution(path, series),
            buy_hold_attribution=ledger_attribution(replay["buy_and_hold"], series),
        )
        if case["buy_fee"] or case["redemption_fee"]:
            actions = [{"origin_index": row["origin_index"], "action": row["action"]}
                       for row in path["action_trace"]]
            zero = replay_actions(series, actions,
                                  **{**args, "entry_fee_rate": 0., "redemption_fee_rate": 0.},
                                  start_index=start_origin + case["nav_availability_lag"], end_index=end)
            result["same_signals_zero_fee"] = {
                "return": zero["strategy"]["return"],
                "cost_drag": zero["strategy"]["return"] - path["return"],
                "basis": "selected actions frozen, not new zero-fee policy decisions",
            }
    except ValidationError as exc:
        result.update(status="unavailable", reason=str(exc))
    return result


def _shadow_choices(packet: Mapping, case: Mapping) -> dict:
    try:
        means = {source: _mean(prediction) for source, prediction in packet["source_predictions"].items()}
    except ValidationError as exc:
        return {"status": "unavailable", "reason": str(exc), "origin_index": packet["origin_index"]}
    choices = {source: fee_adjusted_actions(mean, **_fees(case)) for source, mean in means.items()}
    return {
        "status": "available", "origin_index": packet["origin_index"], "origin_date": packet["origin_date"],
        "means": means, "choices": choices,
        "hybrid_source_conflict": choices["empirical"]["flat"] == "buy" and choices["baseline"]["held"] == "redeem",
        "same_source_conflicts": {source: item["flat"] == "buy" and item["held"] == "redeem"
                                  for source, item in choices.items()},
    }


def consistency_diagnostics(series: Sequence[Mapping], histories: Mapping, case: Mapping,
                            baseline: Mapping, hybrid: Mapping, *, start_origin: int,
                            folds: Mapping) -> dict:
    packets = consistency_packets(series, histories, start_origin, len(series) - 1, case)
    rows = [_shadow_choices(packet, case) for packet in packets]
    fixed = {row["origin_index"] for row in baseline["diagnostics"] if row["reentry_gate"]}

    def summarize(items):
        available = [row for row in items if row["status"] == "available"]
        count = sum(row["hybrid_source_conflict"] for row in available)
        return {"expected_count": len(items), "available_count": len(available),
                "unavailable_count": len(items) - len(available), "source_conflict_count": count,
                "source_conflict_fraction": count / len(available) if available else None,
                "same_source_conflict_counts": {
                    source: sum(row["same_source_conflicts"][source] for row in available)
                    for source in ("baseline", "empirical")}}

    periods = {name: (start, end) for name, (start, end) in folds.items()}
    periods["all"] = (series[start_origin]["date"], series[-1]["date"])
    cohort = [row for row in rows if row["origin_index"] in fixed]
    summary = {name: {
        "all_origin_hypothetical": summarize([row for row in rows
                                               if start <= series[row["origin_index"]]["date"] <= end]),
        "fixed_baseline_post_exit_can_buy": summarize([row for row in cohort
                                                       if start <= series[row["origin_index"]]["date"] <= end]),
    } for name, (start, end) in periods.items()}
    by_origin = {row["origin_index"]: row for row in rows}
    reentry_sells = []
    for episode in hybrid["holding_episodes"]:
        if not episode["is_reentry"] or episode["sale"] is None:
            continue
        sale = episode["sale"]
        shadow = by_origin[sale["origin_index"]]
        empirical_action = (shadow["choices"]["empirical"]["held"]
                            if shadow["status"] == "available" else None)
        reentry_sells.append({
            "entry": episode["entry"], "sale": sale,
            "observations_held": episode["observations_held"],
            "short_round_trip": episode["short_round_trip"],
            "empirical_held_action_at_same_sale_origin": empirical_action,
            "different_source_sell_not_supported_by_empirical": empirical_action == "hold",
            "shadow": shadow,
        })
    return {"schema": "rmbfund.consistency-validation.diagnostics.v1", "case": case["id"],
            "head": [20, case["entry_lag"]], "periods": summary, "all_origin_rows": rows,
            "fixed_cohort_origin_indices": sorted(fixed), "hybrid_reentry_sales": reentry_sells,
            "interpretation": "Descriptive source conflict, not independent trades or causal profit attribution; "
                              "origins in half-years, no forecast-outcome scoring or confidence intervals"}


def _summary(row: Mapping) -> dict:
    result = {key: value for key, value in row.items()
              if key not in {"replay", "diagnostics", "attribution", "buy_hold_attribution",
                             "holding_episodes", "exit_episodes"}}
    for name in ("attribution", "buy_hold_attribution"):
        if name in row:
            result[name] = {key: value for key, value in row[name].items() if key != "daily"}
    return result


def _control_path(protocol: Mapping, symbol: str, case: str, variant: str) -> Path:
    if variant == "uniform_empirical":
        return Path(protocol["prior_economic_study"]) / symbol / f"{case}-empirical_historical-net_payoff.json"
    suffix = "baseline" if variant == "baseline" else "empirical"
    return Path(protocol["prior_reentry_study"]) / symbol / f"{case}-{suffix}.json"


def _case_comparisons(paths: Mapping, cases: Sequence[Mapping]) -> list[dict]:
    comparisons = []
    for case in cases:
        rows = {variant: paths[case["id"], variant] for variant in VARIANTS}
        if any(row["status"] != "available" for row in rows.values()):
            comparisons.append({"case": case["id"], "status": "unavailable"})
            continue
        new, hybrid = rows["post_exit_empirical"], rows["hybrid"]
        n, h = new["strategy_summary"], hybrid["strategy_summary"]
        comparisons.append({
            "case": case["id"], "status": "available",
            "post_exit_minus_hybrid": {key: n[key] - h[key] for key in
                                       ("return", "max_drawdown", "total_fees", "trade_count", "mean_market_exposure")},
            "short_reentry_trip_delta": new["short_reentry_round_trip_count"] - hybrid["short_reentry_round_trip_count"],
            "historical_pareto_criterion": (
                n["return"] >= h["return"] and n["max_drawdown"] >= h["max_drawdown"]
                and n["total_fees"] <= h["total_fees"]
                and new["short_reentry_round_trip_count"] <= hybrid["short_reentry_round_trip_count"]
                and (n["return"] > h["return"]
                     or new["short_reentry_round_trip_count"] < hybrid["short_reentry_round_trip_count"])
            ),
            "post_exit_minus_baseline_return": n["return"] - rows["baseline"]["strategy_summary"]["return"],
            "post_exit_minus_uniform_return": n["return"] - rows["uniform_empirical"]["strategy_summary"]["return"],
            "post_exit_minus_hold_return": n["return"] - new["buy_and_hold_summary"]["return"],
        })
    return comparisons


def run_consistency_validation(protocol_path: str | Path, output: str | Path) -> dict:
    protocol = json.loads(Path(protocol_path).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    cases = protocol["execution_cases"]
    report = {"schema": "rmbfund.consistency-validation.report.v1", "started_at": now_iso(),
              "protocol_path": str(protocol_path), "status": "running_controls", "funds": [],
              "control_reproduction": [], "failures": [], "promotion": False,
              "skill_certified": False, "network_data_access": False, "database_access": False,
              "runtime_ui_changed": False, **EVIDENCE}
    _json_write(out / "report.json", report)
    prepared = {}
    # Globally reproduce all thirty-six old paths before any new C path.
    for source in protocol["inputs"]:
        symbol, stage = source["symbol"], "control_reproduction"
        folder = out / symbol
        folder.mkdir()
        try:
            frozen = json.loads(Path(source["path"]).read_text())
            series = frozen["series"]
            prior = Path(protocol["prior_economic_study"]) / symbol
            old = json.loads((prior / "report.json").read_text())
            if (old["start_origin"] != source["start_origin"] or old["start_date"] != source["decision_start"]
                    or len(series) != source["count"] or series[-1]["date"] != source["end"]):
                raise ValidationError("original study window/manifest mismatch")
            bundles = {
                "baseline": json.loads((prior / "shrink-candidate.json").read_text()),
                "empirical": empirical_history(json.loads((prior / "old-ridge.json").read_text())),
            }
            histories = {name: prepare_economic_history(series, bundle)["historical_forecasts"]
                         for name, bundle in bundles.items()}
            paths = {}
            for case in cases:
                for variant in CONTROLS:
                    row = compare_consistency_path(series, histories, case, variant=variant,
                                                   start_origin=source["start_origin"])
                    paths[case["id"], variant] = row
                    _json_write(folder / f"{case['id']}-{variant}.json", row)
                    if row["status"] != "available":
                        raise ValidationError(f"control unavailable: {case['id']}/{variant}: {row['reason']}")
                    reference = _control_path(protocol, symbol, case["id"], variant)
                    stored = json.loads(reference.read_text())["replay"]
                    parity = baseline_parity(row["replay"], stored)
                    for key in ("assumptions", "start_date", "end_date"):
                        if row["replay"][key] != stored[key]:
                            parity["differences"].append(key)
                    parity["matched"] = not parity["differences"]
                    report["control_reproduction"].append({
                        "symbol": symbol, "case": case["id"], "variant": variant,
                        "reference": str(reference), **parity,
                    })
                    if not parity["matched"]:
                        raise ValidationError(f"control mismatch: {case['id']}/{variant}: {parity['differences']}")
            prepared[symbol] = series, histories, paths
        except Exception as exc:
            failure = {"symbol": symbol, "stage": stage, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}
            report["failures"].append(failure)
            _json_write(folder / "failure.json", failure)
    _json_write(out / "control-reproduction.json", report["control_reproduction"])
    if report["failures"]:
        report.update(status="control_reproduction_failed", finished_at=now_iso())
        _json_write(out / "report.json", report)
        return report
    report["status"] = "running_consistent_paths"
    _json_write(out / "report.json", report)
    for source in protocol["inputs"]:
        symbol, stage = source["symbol"], "consistent_paths"
        folder = out / symbol
        try:
            series, histories, paths = prepared[symbol]
            for case in cases:
                row = compare_consistency_path(series, histories, case, variant="post_exit_empirical",
                                               start_origin=source["start_origin"])
                paths[case["id"], "post_exit_empirical"] = row
                _json_write(folder / f"{case['id']}-post_exit_empirical.json", row)
            stage = "source_conflict_diagnostics"
            for case in cases:
                value = consistency_diagnostics(
                    series, histories, case, paths[case["id"], "baseline"], paths[case["id"], "hybrid"],
                    start_origin=source["start_origin"], folds=protocol["folds"],
                )
                _json_write(folder / f"{case['id']}-consistency-diagnostics.json", value)
            fund = {"symbol": symbol, "status": "completed", "start_origin": source["start_origin"],
                    "start_date": source["decision_start"], "end_date": series[-1]["date"],
                    "scenarios": [_summary(paths[case["id"], variant]) for case in cases for variant in VARIANTS],
                    "comparisons": _case_comparisons(paths, cases), "promotion": False}
            fund["unavailable_scenario_count"] = sum(row["status"] != "available" for row in fund["scenarios"])
            _json_write(folder / "report.json", fund)
            report["funds"].append(fund)
        except Exception as exc:
            failure = {"symbol": symbol, "stage": stage, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}
            report["failures"].append(failure)
            _json_write(folder / "failure.json", failure)
        _json_write(out / "report.json", report)
    unavailable = sum(fund["unavailable_scenario_count"] for fund in report["funds"])
    report.update(status=("completed_with_failures" if report["failures"]
                          else "completed_with_unavailable_scenarios" if unavailable else "completed"),
                  scenario_count=sum(len(fund["scenarios"]) for fund in report["funds"]),
                  unavailable_scenario_count=unavailable, finished_at=now_iso())
    _json_write(out / "report.json", report)
    return report

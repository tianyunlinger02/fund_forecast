"""Fixed offline forecast/policy ablation on one continuous fund ledger.

The supplied NAV observations are not historical publication/dealing calendars.
This experiment neither reconstructs the author's advice nor promotes a model.
"""

from __future__ import annotations

import json
import math
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .domain import ValidationError, now_iso
from .forecast import MODEL_ID, _distribution
from .forecast_candidates import (
    SHRINK_ID, TREND_ID, candidate_definition, issue_candidate_forecasts,
    issue_ridge_distribution_forecasts, issue_shrinkage_forecasts,
)
from .forecast_comparison import (
    EVIDENCE, _available, _fitted, _identity, _json_write, _path_summary,
    audit_availability, compare_paired_losses, evaluate_candidate_rows,
)
from .forecast_research import ledger_attribution, paired_delta_bootstrap, replay_actions
from .timing_scenario import _calendar_value, fee_adjusted_actions

EMPIRICAL_ID = "empirical_historical"
MODEL_IDS = (MODEL_ID, SHRINK_ID, EMPIRICAL_ID, TREND_ID)
POLICY_MATRIX = ((MODEL_ID, "legacy"), (MODEL_ID, "net_payoff"),
                 (SHRINK_ID, "legacy"), (SHRINK_ID, "net_payoff"),
                 (EMPIRICAL_ID, "legacy"), (EMPIRICAL_ID, "net_payoff"),
                 (TREND_ID, "net_payoff"))


def empirical_history(reference: Mapping) -> dict:
    history = []
    for record in reference["historical_forecasts"]:
        baseline = record.get("historical_baseline")
        prediction = dict(baseline) if baseline is not None else {
            "horizon": record["horizon"], "entry_lag": record["entry_lag"],
            "status": "unavailable", "reason": "no_same_head_matured_empirical_baseline",
        }
        history.append({**record, "prediction": prediction})
    return {"candidate_id": EMPIRICAL_ID, "historical_forecasts": history}


def prepare_economic_history(series: Sequence[Mapping], bundle: Mapping) -> dict:
    """Attach outcomes for scoring only; keep every forecast-time tail record."""
    history, scores, seen = [], [], set()
    navs = np.asarray([float(row["nav"]) for row in series])
    bad = np.r_[0, np.cumsum([row.get("invalid_return", False)
                             or row.get("label_valid") is False for row in series])]
    for record in bundle["historical_forecasts"]:
        identity = _identity(record)
        if identity in seen:
            raise ValidationError("duplicate economic-history origin/head")
        seen.add(identity)
        origin, horizon, lag = identity
        if (not 0 <= origin < len(series) or record["origin_date"] != series[origin]["date"]
                or record["knowledge_cutoff_index"] != origin
                or record["knowledge_cutoff_date"] != series[origin]["date"]
                or record["assumed_known_index"] != origin + 1
                or record.get("assumed_known_date") != (
                    series[origin + 1]["date"] if origin + 1 < len(series) else None)):
            raise ValidationError("economic-history chronology does not match observations")
        for key in ("training_target_end_index", "residual_target_end_index"):
            cutoff = record["prediction"].get(key)
            if cutoff is not None and cutoff > origin:
                raise ValidationError("economic-history head contains future training information")
        entry, target = origin + lag, origin + lag + horizon
        immature = target >= len(series)
        invalid = not immature and bad[target + 1] != bad[entry + 1]
        row = {**record, "entry_index": entry, "target_index": target,
               "entry_date": series[entry]["date"] if entry < len(series) else None,
               "target_date": series[target]["date"] if target < len(series) else None,
               "outcome_status": "immature" if immature else "invalid" if invalid else "mature"}
        if row["outcome_status"] == "mature":
            actual_log = float(np.log(navs[target]) - np.log(navs[entry]))
            row.update(actual_log_return=actual_log, actual_change=float(np.expm1(actual_log)),
                       no_change_baseline=_distribution(np.zeros(1), horizon=horizon,
                                                        entry_lag=lag, method="no_change_point_baseline"))
            scores.append(row)
        history.append(row)
    return {**bundle, "historical_forecasts": history, "evaluation_rows": scores}


def fold_coverage(bundles: Mapping, folds: Mapping) -> dict:
    result = {}
    for candidate_id, bundle in bundles.items():
        entries = {}
        for horizon, lag in sorted({(r["horizon"], r["entry_lag"])
                                    for r in bundle["historical_forecasts"]}):
            periods = {}
            for name, (start, end) in folds.items():
                rows = [r for r in bundle["historical_forecasts"]
                        if (r["horizon"], r["entry_lag"]) == (horizon, lag)
                        and start <= r["origin_date"] <= end]
                mature = [r for r in rows if r["outcome_status"] == "mature"
                          and r["target_date"] <= end]
                periods[name] = {
                    "expected_origins": len(rows), "available_predictions": sum(_available(r) for r in rows),
                    "genuine_method_predictions": sum(_fitted(r) for r in rows)
                    if candidate_id != EMPIRICAL_ID else None,
                    "fallbacks": sum(_available(r) and not _fitted(r) for r in rows)
                    if candidate_id != EMPIRICAL_ID else 0,
                    "unavailable_predictions": sum(not _available(r) for r in rows),
                    "invalid_outcomes": sum(r["outcome_status"] == "invalid"
                                            and r["target_date"] <= end for r in rows),
                    "not_mature_at_fold_end": sum(r["outcome_status"] == "immature"
                                                  or r["target_date"] is not None and r["target_date"] > end
                                                  for r in rows),
                    "mature_valid_outcomes": len(mature),
                }
            entries[f"H{horizon}L{lag}"] = periods
        result[candidate_id] = entries
    return result


def _history_maps(bundles: Mapping) -> dict:
    return {candidate_id: {_identity(row): row for row in bundle["historical_forecasts"]}
            for candidate_id, bundle in bundles.items()}


def _usable(candidate_id: str, record: Mapping | None) -> bool:
    if record is None or not _available(record):
        return False
    return candidate_id == EMPIRICAL_ID or _fitted(record)


def common_replay_origin(series: Sequence[Mapping], bundles: Mapping, heads: Sequence,
                         *, start_date: str = "2024-07-01") -> int | None:
    maps = _history_maps(bundles)
    for origin, nav in enumerate(series[:-1]):
        if nav["date"] < start_date:
            continue
        if all(_usable(candidate_id, mapping.get((origin, horizon, lag)))
               for candidate_id, mapping in maps.items() for horizon, lag in heads):
            return origin
    return None


def _fees(case: Mapping) -> dict:
    return {"subscription_fee": case["buy_fee"], "current_redemption_fee": case["redemption_fee"],
            "future_old_redemption_fee": case["redemption_fee"],
            "future_new_redemption_fee": case["redemption_fee"]}


def _branch_actions(mean: float, policy: str, case: Mapping) -> dict:
    if policy == "legacy":
        return {"flat": "buy" if mean > case["buy_fee"] else "wait",
                "held": "redeem" if mean < -case["redemption_fee"] else "hold"}
    if policy != "net_payoff":
        raise ValidationError("unknown economic-validation policy")
    return fee_adjusted_actions(mean, **_fees(case))


def _decision_policy(policy: str, case: Mapping):
    def decide(item, state):
        mean = item["prediction"]["expected_change"]
        if policy == "legacy":
            # Preserve the original signal even when the ledger must reject it.
            return "buy" if mean > case["buy_fee"] else "redeem" if mean < -case["redemption_fee"] else "hold"
        choices = _branch_actions(mean, policy, case)
        action = choices["held" if state["has_units"] else "flat"]
        return "hold" if action == "wait" else action
    return decide


def _ledger_args(case: Mapping) -> dict:
    return {"entry_fee_rate": case["buy_fee"], "redemption_fee_rate": case["redemption_fee"],
            **{key: case[key] for key in ("entry_lag", "redemption_lag", "confirmation_lag",
                                         "cash_lag", "nav_availability_lag")}}


def compare_continuous_actions(series: Sequence[Mapping], bundles: Mapping,
                               cases: Sequence[Mapping], *, heads=((20, 3), (20, 4)),
                               start_date="2024-07-01") -> dict:
    origin = common_replay_origin(series, bundles, heads, start_date=start_date)
    result = {"schema": "rmbfund.economic-validation.continuous.v1", "start_origin": origin,
              "start_basis": "first then-known common genuine method, not future returns or gaps",
              "scenarios": [], "skill_certified": False, **EVIDENCE}
    if origin is None:
        result.update(status="unavailable", reason="no_common_genuine_start")
        return result
    maps, end = _history_maps(bundles), len(series) - 1
    for case in cases:
        lag = case["entry_lag"]
        for candidate_id, policy in POLICY_MATRIX:
            scenario = {"case": case["id"], "candidate_id": candidate_id, "policy": policy,
                        "head": [20, lag], "parameters": dict(case),
                        "expected_decision_count": end - origin}
            records = [maps[candidate_id].get((i, 20, lag)) for i in range(origin, end)]
            missing = [i for i, r in zip(range(origin, end), records, strict=True)
                       if not _usable(candidate_id, r)]
            if missing:
                scenario.update(status="unavailable", reason="missing_or_unavailable_required_head",
                                unavailable_origin_indices=missing)
                result["scenarios"].append(scenario)
                continue
            actions = []
            for row in records:
                mean = row["prediction"].get("expected_change")
                if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < -1:
                    raise ValidationError("continuous replay needs a finite exact-head expected change")
                # Only forecast-time fields enter the policy. Outcomes stay in
                # scoring records, never in a stateful decision callback.
                actions.append({"origin_index": row["origin_index"],
                                "prediction": {"expected_change": mean},
                                "origin_date": row["origin_date"]})
            args = _ledger_args(case)
            try:
                replay = replay_actions(series, actions, **args, start_index=origin + 1,
                                        end_index=end, decision_policy=_decision_policy(policy, case))
                attribution = ledger_attribution(replay["strategy"], series)
                passive_attribution = ledger_attribution(replay["buy_and_hold"], series)
            except ValidationError as exc:
                scenario.update(status="unavailable", reason=str(exc))
            else:
                summary, passive = _path_summary(replay["strategy"]), _path_summary(replay["buy_and_hold"])
                scenario.update(status="available", replay=replay,
                                strategy_summary=summary, buy_and_hold_summary=passive,
                                excess_return_vs_buy_and_hold=summary["return"] - passive["return"],
                                attribution=attribution, buy_hold_attribution=passive_attribution)
                if case["buy_fee"] or case["redemption_fee"]:
                    trace = [{"origin_index": item["origin_index"], "action": item["action"]}
                             for item in replay["strategy"]["action_trace"]]
                    gross = replay_actions(series, trace, **{**args, "entry_fee_rate": 0.0,
                                                              "redemption_fee_rate": 0.0},
                                           start_index=origin + 1, end_index=end)
                    scenario["same_signals_zero_fee"] = {
                        "actions": trace, "return": gross["strategy"]["return"],
                        "cost_drag": gross["strategy"]["return"] - summary["return"],
                        "basis": "fee-conditioned selected actions frozen; no new zero-fee policy decisions",
                    }
            result["scenarios"].append(scenario)
    result["status"] = "available" if all(r["status"] == "available" for r in result["scenarios"]) else "with_unavailable_scenarios"
    return result


def continuous_effects(scenarios: Sequence[Mapping]) -> list[dict]:
    result = []
    for case in dict.fromkeys(row["case"] for row in scenarios):
        rows = {(r["candidate_id"], r["policy"]): r for r in scenarios
                if r["case"] == case and r["status"] == "available"}
        effects = {"case": case, "model_effect_at_same_policy": {}, "policy_effect_at_same_model": {}}
        for policy in ("legacy", "net_payoff"):
            if all((model, policy) in rows for model in (MODEL_ID, SHRINK_ID)):
                effects["model_effect_at_same_policy"][policy] = {
                    key: rows[SHRINK_ID, policy]["strategy_summary"][key] - rows[MODEL_ID, policy]["strategy_summary"][key]
                    for key in ("return", "max_drawdown", "total_fees", "trade_count")}
        for model in (MODEL_ID, SHRINK_ID, EMPIRICAL_ID):
            if all((model, policy) in rows for policy in ("legacy", "net_payoff")):
                effects["policy_effect_at_same_model"][model] = {
                    key: rows[model, "net_payoff"]["strategy_summary"][key] - rows[model, "legacy"]["strategy_summary"][key]
                    for key in ("return", "max_drawdown", "total_fees", "trade_count")}
        model_effects = effects["model_effect_at_same_policy"]
        if len(model_effects) == 2:
            effects["interaction_return"] = model_effects["net_payoff"]["return"] - model_effects["legacy"]["return"]
        result.append(effects)
    return result


def compare_single_decision_payoffs(bundles: Mapping, cases: Sequence[Mapping], periods: Mapping,
                                    *, replicates=2000, seed=20261030) -> dict:
    """Same arithmetic as calendar history, without invented historical dates.

    There is no dynamic capital here: flat owns B cash, held owns B fund+B
    reserve. Actual terminal values, not unlike-reference advantages, are paired.
    """
    maps, outputs = _history_maps(bundles), []
    for case in cases:
        lag, fees = case["entry_lag"], _fees(case)
        keys = sorted(key for key, row in maps[SHRINK_ID].items() if key[1:] == (20, lag)
                      and row["outcome_status"] == "mature"
                      and all(_usable(model, maps[model].get(key)) for model in (MODEL_ID, SHRINK_ID, EMPIRICAL_ID)))
        details = []
        for key in keys:
            source = maps[SHRINK_ID][key]
            item = {name: source[name] for name in ("origin_index", "origin_date", "entry_index", "target_index",
                                                    "entry_date", "target_date", "horizon", "entry_lag", "actual_change")}
            values = {}
            for model in (MODEL_ID, SHRINK_ID, EMPIRICAL_ID):
                for policy in ("legacy", "net_payoff"):
                    mean = maps[model][key]["prediction"]["expected_change"]
                    selected = _branch_actions(mean, policy, case)
                    branches = {}
                    for branch, base in (("flat", 1), ("held", 2)):
                        action = selected[branch]
                        value_action = "exit" if action == "redeem" else action
                        expected = _calendar_value(value_action, mean, 1.0, fees)
                        actual = _calendar_value(value_action, source["actual_change"], 1.0, fees)
                        branches[branch] = {"action": action, "expected_terminal_value": expected,
                                            "actual_terminal_value": actual,
                                            "initial_capital": base, "net_return": actual / base - 1}
                    values[f"{model}:{policy}"] = branches
            item["values"] = values
            details.append(item)
        period_results = {}
        for name, (start, end) in periods.items():
            selected = [r for r in details if start <= r["origin_date"] <= end and r["target_date"] <= end]
            deltas = {}
            for policy in ("legacy", "net_payoff"):
                for branch in ("flat", "held"):
                    new = [r["values"][f"{SHRINK_ID}:{policy}"][branch]["net_return"] for r in selected]
                    for ref in (MODEL_ID, EMPIRICAL_ID):
                        old = [r["values"][f"{ref}:{policy}"][branch]["net_return"] for r in selected]
                        deltas[f"{branch}:{policy}:shrink_minus_{ref}"] = [a - b for a, b in zip(new, old, strict=True)]
                    baseline = [0.0 if branch == "flat" else
                                _calendar_value("hold", r["actual_change"], 1.0, fees) / 2 - 1 for r in selected]
                    deltas[f"{branch}:{policy}:shrink_minus_passive"] = [a - b for a, b in zip(new, baseline, strict=True)]
            for model in (MODEL_ID, SHRINK_ID, EMPIRICAL_ID):
                for branch in ("flat", "held"):
                    deltas[f"{branch}:{model}:net_minus_legacy"] = [
                        r["values"][f"{model}:net_payoff"][branch]["net_return"] -
                        r["values"][f"{model}:legacy"][branch]["net_return"] for r in selected]
            period_results[name] = paired_delta_bootstrap(selected, deltas, replicates=replicates, seed=seed)
        outputs.append({"case": case["id"], "head": [20, lag], "rows": details, "periods": period_results})
    return {"schema": "rmbfund.economic-validation.single-decisions.v1", "cases": outputs,
            "basis": "per-origin actual terminal differences / equal initial capital; NOT a compounded path or PnL confidence interval",
            "skill_certified": False}


def _summary_scenario(row: Mapping) -> dict:
    result = {key: value for key, value in row.items() if key not in {"replay", "same_signals_zero_fee"}}
    for key in ("attribution", "buy_hold_attribution"):
        if key in result:
            result[key] = {name: value for name, value in result[key].items() if name != "daily"}
    if "same_signals_zero_fee" in row:
        result["same_signals_zero_fee"] = {key: value for key, value in row["same_signals_zero_fee"].items()
                                           if key != "actions"}
    return result


def run_economic_validation(protocol_path: str | Path, output: str | Path) -> dict:
    protocol_path, out = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    heads = tuple(tuple(pair) for pair in protocol["heads"])
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rmbfund.economic-validation.report.v1", "status": "running",
              "started_at": now_iso(), "protocol_path": str(protocol_path),
              "candidate_definition": candidate_definition(SHRINK_ID),
              "funds": [], "failures": [], "promotion": False, "skill_certified": False,
              "network": False, "database_access": False, **EVIDENCE}
    _json_write(out / "report.json", report)
    for source in protocol["inputs"]:
        folder = out / source["symbol"]
        folder.mkdir()
        stage = "input"
        try:
            frozen = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
            series = frozen["series"]
            stage = "old_ridge"
            old = issue_ridge_distribution_forecasts(series, candidate_id=MODEL_ID, heads=heads)
            _json_write(folder / "old-ridge.json", old)
            stage = "shrink_candidate"
            shrink = issue_shrinkage_forecasts(series, heads=heads)
            _json_write(folder / "shrink-candidate.json", shrink)
            stage = "trend_control"
            trend = issue_candidate_forecasts(series, candidate_id=TREND_ID, heads=heads, include_history=True)
            _json_write(folder / "trend-control.json", trend)
            bundles = {model: prepare_economic_history(series, bundle)
                       for model, bundle in ((MODEL_ID, old), (SHRINK_ID, shrink),
                                             (EMPIRICAL_ID, empirical_history(old)), (TREND_ID, trend))}
            periods = {name: (start, min(end, series[-1]["date"]))
                       for name, (start, end) in protocol["folds"].items()}
            # Aggregate diagnostics do not replace the predeclared cutoff-purged folds.
            periods.update(all=("2024-07-01", series[-1]["date"]),
                           development_2025=("2025-01-01", "2025-12-31"),
                           inspected_2026=("2026-01-01", series[-1]["date"]))
            stage = "matched_scores"
            scores = evaluate_candidate_rows({model: bundles[model] for model in (MODEL_ID, SHRINK_ID)},
                                             heads=heads, periods=periods, mature_by_period_end=True)
            _json_write(folder / "metrics.json", scores)
            _json_write(folder / "fold-coverage.json", fold_coverage(bundles, periods))
            stage = "paired_forecast_losses"
            bootstrap = compare_paired_losses(
                {model: bundles[model] for model in (MODEL_ID, SHRINK_ID)}, heads=heads,
                candidate_id=SHRINK_ID, periods=periods, mature_by_period_end=True,
                replicates=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"])
            _json_write(folder / "forecast-bootstrap.json", bootstrap)
            stage = "continuous_ledger"
            continuous = compare_continuous_actions(series, bundles, protocol["execution_cases"], heads=heads)
            for row in continuous["scenarios"]:
                name = f"{row['case']}-{row['candidate_id']}-{row['policy']}.json"
                _json_write(folder / name, row)
            effects = continuous_effects(continuous["scenarios"])
            _json_write(folder / "effects.json", effects)
            stage = "single_decision_payoffs"
            payoffs = compare_single_decision_payoffs(bundles, protocol["execution_cases"], periods,
                                                       replicates=protocol["bootstrap"]["replicates"],
                                                       seed=protocol["bootstrap"]["seed"])
            _json_write(folder / "single-decision-payoffs.json", payoffs)
            fund_report = {"product": frozen["product"], "input_path": source["path"],
                           "availability": audit_availability(frozen), "output_directory": str(folder),
                           "start_origin": continuous["start_origin"],
                           "start_date": series[continuous["start_origin"] + 1]["date"] if continuous["start_origin"] is not None else None,
                           "end_date": series[-1]["date"],
                           "scenarios": [_summary_scenario(row) for row in continuous["scenarios"]],
                           "effects": effects, "primary_forecast_periods": scores["heads"][0]["periods"],
                           "promotion": False, "skill_certified": False, **EVIDENCE}
            _json_write(folder / "report.json", fund_report)
            report["funds"].append({"symbol": source["symbol"], "status": "completed",
                                    "output_directory": str(folder), "start_date": fund_report["start_date"],
                                    "end_date": fund_report["end_date"],
                                    "scenarios": fund_report["scenarios"], "effects": effects})
        except Exception as exc:
            failure = {"symbol": source["symbol"], "stage": stage, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}
            _json_write(folder / "failure.json", failure)
            report["failures"].append(failure)
        _json_write(out / "report.json", report)
    report.update(status="completed" if not report["failures"] else "completed_with_failures",
                  finished_at=now_iso())
    _json_write(out / "report.json", report)
    return report

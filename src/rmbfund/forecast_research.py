"""Pure chronological forecast evaluation and cash/units action replay.

These are research calculations, not historically available Fact timestamps or
claims of an executable strategy. There is no network, persistence, or fitting
in this module. Overlapping rolling targets are never treated as independent
trades, and forecast accuracy is not reported as investment profit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date
from typing import Any

import numpy as np

from .domain import ValidationError, digest

PERIODS = {
    "development_2025": ("2025-01-01", "2025-12-31"),
    "heldout_2026": ("2026-01-01", "2026-12-31"),
}


def _metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    actual = np.asarray([row["actual_change"] for row in rows], dtype=float)
    predictions = [row[key] for row in rows]
    up = np.asarray([item["up_probability"] for item in predictions], dtype=float)
    probabilities = np.asarray(
        [
            [item["up_probability"], item["down_probability"], item["flat_probability"]]
            for item in predictions
        ],
        dtype=float,
    )
    outcomes = np.column_stack((actual > 0, actual < 0, actual == 0)).astype(float)
    expected = np.asarray([item["expected_change"] for item in predictions], dtype=float)
    lower = np.asarray([item["lower"] for item in predictions], dtype=float)
    upper = np.asarray([item["upper"] for item in predictions], dtype=float)
    if not all(
        np.all(np.isfinite(value))
        for value in (
            actual,
            probabilities,
            expected,
            lower,
            upper,
        )
    ):
        raise ValidationError("forecast evaluation requires finite numbers")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        or np.any(lower > upper)
    ):
        raise ValidationError("invalid forecast distribution summary")
    calibration = []
    absolute_calibration_error = 0.0
    for index in range(5):
        low, high = index / 5, (index + 1) / 5
        mask = (up >= low) & ((up <= high) if index == 4 else (up < high))
        count = int(mask.sum())
        mean_probability = float(up[mask].mean()) if count else None
        observed_frequency = float((actual[mask] > 0).mean()) if count else None
        if count:
            absolute_calibration_error += count * abs(mean_probability - observed_frequency)
        calibration.append(
            {
                "lower": low,
                "upper": high,
                "count": count,
                "mean_probability": mean_probability,
                "observed_up_frequency": observed_frequency,
            }
        )
    errors = expected - actual
    error_scale = float(np.max(np.abs(errors)))
    rmse = (
        error_scale * float(np.sqrt(np.mean((errors / error_scale) ** 2))) if error_scale else 0.0
    )
    alpha = 0.2
    interval_scores = (upper - lower) + (2 / alpha) * (
        np.maximum(lower - actual, 0.0) + np.maximum(actual - upper, 0.0)
    )
    interval_metrics = {"interval_score_80": float(np.mean(interval_scores))}
    if all(item.get("median_change") is not None for item in predictions):
        median = np.asarray([item["median_change"] for item in predictions], dtype=float)
        if not np.all(np.isfinite(median)) or np.any(median < lower) or np.any(median > upper):
            raise ValidationError("forecast median must be finite and inside its interval")
        interval_metrics["wis_80_median"] = float(np.mean(
            (0.5 * np.abs(actual - median) + 0.1 * interval_scores) / 1.5
        ))
    return {
        "count": len(rows),
        **interval_metrics,
        "up_brier": float(np.mean((up - (actual > 0)) ** 2)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - outcomes) ** 2, axis=1))),
        "mae": float(np.sum(np.abs(errors) / len(rows))),
        "rmse": rmse,
        "mean_error": float(np.sum(errors / len(rows))),
        "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width": float(np.sum((upper - lower) / len(rows))),
        "nominal_interval_coverage": 0.8,
        "up_calibration_ece": float(absolute_calibration_error / len(rows)),
        "up_calibration": calibration,
    }


def forecast_losses(
    rows: Sequence[Mapping[str, Any]], key: str = "prediction",
) -> dict[str, np.ndarray]:
    """Per-origin losses for paired chronological research, not profit series."""
    actual = np.asarray([row["actual_change"] for row in rows], dtype=float)
    if key == "always_up":
        return {"direction_error": (actual <= 0).astype(float)}
    if key == "no_change":
        return {"absolute_error": np.abs(actual), "squared_error": actual ** 2,
                "signed_error": -actual}
    _metrics(rows, key)
    heads = [row[key] for row in rows]
    up = np.asarray([head["up_probability"] for head in heads])
    expected = np.asarray([head["expected_change"] for head in heads])
    lower = np.asarray([head["lower"] for head in heads])
    upper = np.asarray([head["upper"] for head in heads])
    errors = expected - actual
    score = upper - lower + 10 * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
    result = {
        "up_brier": (up - (actual > 0)) ** 2,
        "absolute_error": np.abs(errors), "squared_error": errors ** 2,
        "signed_error": errors,
        "direction_error": ((up >= 0.5) != (actual > 0)).astype(float),
        "interval_score_80": score, "interval_width": upper - lower,
        "interval_miss": ((actual < lower) | (actual > upper)).astype(float),
    }
    if all(head.get("median_change") is not None for head in heads):
        median = np.asarray([head["median_change"] for head in heads])
        result["wis_80_median"] = (0.5 * np.abs(actual - median) + 0.1 * score) / 1.5
    return result


def paired_loss_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, prediction_key: str = "prediction",
    reference_key: str = "historical_baseline", replicates: int = 2000,
    seed: int = 20261029,
) -> dict[str, Any]:
    """Paired moving-block loss deltas at H+L and twice H+L.

    Blocks never cross a missing origin. Percentile intervals are exploratory,
    particularly with few nonoverlap blocks; no p-value or certification flag.
    Signed bias, width and miss losses need joint interpretation with scores.
    """
    if not rows:
        return {"count": 0, "blocks": [], "skill_certified": False}
    identities = {(row["horizon"], row["entry_lag"]) for row in rows}
    origins = [row["origin_index"] for row in rows]
    if len(identities) != 1 or origins != sorted(set(origins)):
        raise ValidationError("paired bootstrap requires one exact head in original unique time order")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValidationError("bootstrap replicates must be a positive integer")
    candidate = forecast_losses(rows, prediction_key)
    reference = forecast_losses(rows, reference_key)
    result = paired_delta_bootstrap(
        rows, {name: candidate[name] - reference[name] for name in candidate if name in reference},
        replicates=replicates, seed=seed,
    )
    result["delta_direction"] = "candidate minus reference; negative lower loss, signed bias/width require context"
    result["interpretation"] = "Exploratory percentile intervals, not a significance test. Few blocks, dependence, multiple diagnostics and inspected 2026 history prevent skill certification."
    return result


def paired_delta_bootstrap(
    rows: Sequence[Mapping[str, Any]], deltas: Mapping[str, Sequence[float]], *,
    replicates: int = 2000, seed: int = 20261030,
) -> dict[str, Any]:
    """Block-resample paired per-origin deltas, never a compounded wealth path.

    Each row supplies one exact H/L, origin_index/date and entry/target indices.
    Values can be loss differences or same-capital terminal-value differences.
    Missing origins split runs; no unsupported run is silently discarded.
    """
    if not rows:
        return {"count": 0, "blocks": [], "skill_certified": False}
    identities = {(row["horizon"], row["entry_lag"]) for row in rows}
    origins = [row["origin_index"] for row in rows]
    if len(identities) != 1 or origins != sorted(set(origins)):
        raise ValidationError("paired bootstrap requires one exact head in original unique time order")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValidationError("bootstrap replicates must be a positive integer")
    horizon, lag = next(iter(identities))
    if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            or isinstance(lag, bool) or not isinstance(lag, int) or lag < 0):
        raise ValidationError("paired bootstrap requires a positive horizon and nonnegative lag")
    names = list(deltas)
    try:
        columns = [np.asarray(deltas[name], dtype=float) for name in names]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("paired bootstrap deltas must be finite aligned sequences") from exc
    if not columns or any(column.shape != (len(rows),) or not np.all(np.isfinite(column))
                          for column in columns):
        raise ValidationError("paired bootstrap deltas must be finite aligned sequences")
    deltas = np.column_stack(columns)
    boundaries = [0] + [i for i in range(1, len(origins)) if origins[i] != origins[i - 1] + 1] + [len(rows)]
    runs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    prefix = np.vstack((np.zeros(len(names)), np.cumsum(deltas, axis=0)))
    blocks = []
    for length in (horizon + lag, 2 * (horizon + lag)):
        starts = np.asarray([i for start, end in runs for i in range(start, end - length + 1)], dtype=int)
        nonoverlap_count = sum((end - start) // length for start, end in runs)
        unsupported = sum(end - start for start, end in runs if end - start < length)
        available = len(starts) > 0 and unsupported == 0
        result = {
            "block_length": length, "available_block_starts": len(starts),
            "origins_in_too_short_runs": unsupported,
            "nonoverlap_block_count_upper": nonoverlap_count,
            "few_blocks": nonoverlap_count < 10,
            "status": "available" if available else "unavailable",
            "intervals": {},
        }
        if available:
            rng = np.random.default_rng(seed)
            full, remainder = divmod(len(rows), length)
            draws = rng.choice(starts, size=(replicates, full + bool(remainder)), replace=True)
            sampled = (prefix[draws[:, :full] + length] - prefix[draws[:, :full]]).sum(axis=1)
            if remainder:
                sampled += prefix[draws[:, -1] + remainder] - prefix[draws[:, -1]]
            sampled /= len(rows)
            lower, upper = np.quantile(sampled, [0.025, 0.975], axis=0)
            result["intervals"] = {
                name: {"mean_delta": float(np.mean(deltas[:, index])),
                       "lower_95": float(lower[index]), "upper_95": float(upper[index])}
                for index, name in enumerate(names)
            }
        else:
            result["reason"] = "some_origins_in_runs_shorter_than_block; no silent exclusion from the estimand"
        blocks.append(result)
    return {
        "count": len(rows), "origin_start": rows[0]["origin_date"],
        "origin_end": rows[-1]["origin_date"], "horizon": horizon, "entry_lag": lag,
        "contiguous_run_lengths": [end - start for start, end in runs],
        "target_nonoverlap_count": len(_disjoint(rows)),
        "mean_deltas": {name: float(np.mean(deltas[:, i])) for i, name in enumerate(names)},
        "blocks": blocks, "replicates": replicates, "seed": seed,
        "delta_direction": "supplied paired per-origin deltas; sign follows the caller's comparison",
        "interpretation": "Exploratory per-origin percentile intervals, not a continuous-strategy return interval or a significance test. Few blocks, dependence, multiple diagnostics and inspected 2026 history prevent skill certification.",
        "skill_certified": False,
    }


def _disjoint(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Greedy non-overlapping return blocks; sharing a NAV boundary is fine."""
    selected = []
    last_endpoint = -1
    for row in sorted(rows, key=lambda item: (item["entry_index"], item["target_index"])):
        if row["entry_index"] >= last_endpoint:
            selected.append(row)
            last_endpoint = row["target_index"]
    return selected


def evaluate_forecasts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare matched, matured chronological predictions, split by origin year.

    Each row has origin/entry/target dates and indices, actual_change, and three
    identically shaped summaries: prediction, historical_baseline, and
    no_change_baseline. 2026 is a chronological replay with expanding updates,
    NOT a strict untouched holdout: this history was already inspected. The
    headline status uses only the anchor H20 head, not a favourable head search.
    """
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["entry_index"] < row["origin_index"]:
            raise ValidationError("forecast evaluation entry precedes origin")
        if row["target_index"] - row["entry_index"] != row["horizon"]:
            raise ValidationError("forecast evaluation target horizon mismatch")
        for key in ("prediction", "historical_baseline", "no_change_baseline"):
            for field in ("training_target_end_index", "residual_target_end_index"):
                cutoff = row[key].get(field)
                if cutoff is not None and cutoff > row["origin_index"]:
                    raise ValidationError("forecast evaluation contains a future target")
        groups[(row["horizon"], row["entry_lag"])].append(row)
    heads = []
    for (horizon, lag), group in sorted(groups.items()):
        periods = {}
        for name, (start, end) in PERIODS.items():
            subset = [row for row in group if start <= row["origin_date"] <= end]
            blocks = _disjoint(subset)
            periods[name] = {
                "start": start,
                "end": end,
                "forecast_count": len(subset),
                "ridge_forecast_count": sum(
                    row["prediction"]["method"] == "ridge_prequential_empirical" for row in subset
                ),
                "model": _metrics(subset, "prediction"),
                "historical_frequency": _metrics(subset, "historical_baseline"),
                "no_change": _metrics(subset, "no_change_baseline"),
                "disjoint_blocks": {
                    "count": len(blocks),
                    "model": _metrics(blocks, "prediction"),
                    "historical_frequency": _metrics(blocks, "historical_baseline"),
                    "no_change": _metrics(blocks, "no_change_baseline"),
                    "selection": "earliest entry, next entry >= preceding target endpoint",
                },
            }
        heads.append({"horizon": horizon, "entry_lag": lag, "periods": periods})
    primary = next(
        (head for head in heads if head["horizon"] == 20 and head["entry_lag"] == 0), None
    )
    heldout = primary["periods"]["heldout_2026"] if primary else None
    status = "unverified"
    if (
        heldout
        and heldout["ridge_forecast_count"] >= 30
        and heldout["disjoint_blocks"]["count"] >= 5
    ):
        model = heldout["model"]
        historical = heldout["historical_frequency"]
        no_change = heldout["no_change"]
        status = (
            "better"
            if (
                model["up_brier"] < historical["up_brier"]
                and model["mae"] < min(historical["mae"], no_change["mae"])
            )
            else "not_better"
        )
    summaries = {
        "unverified": "Insufficient rows for the legacy H20 point-estimate comparison; no skill certification.",
        "better": "H20 point estimates improve the specified historical Brier/error comparisons only; this is not statistical significance, skill certification or proof of profit.",
        "not_better": "H20 point estimates do not beat all specified chronological baseline comparisons; no skill certification.",
    }
    return {
        "status": status,
        "summary": summaries[status] + " 2026 was inspected and is not strict unseen data.",
        "strict_unseen": False,
        "development_period": {
            "start": PERIODS["development_2025"][0],
            "end": PERIODS["development_2025"][1],
        },
        "heldout_period": {"start": PERIODS["heldout_2026"][0], "end": PERIODS["heldout_2026"][1]},
        "forecast_count": sum(
            PERIODS["development_2025"][0] <= row["origin_date"] <= PERIODS["heldout_2026"][1]
            for row in rows
        ),
        "heads": heads,
        "headline_rule": "legacy anchor H20 descriptive point-estimate screen only; >=30 ridge forecasts and >=5 disjoint blocks are reporting thresholds, not certification; "
        "up Brier < historical; MAE < both historical and no-change",
        "status_interpretation": "better/not_better describes point estimates only; overlapping outcomes and few disjoint blocks do not certify skill or statistical significance",
        "skill_certified": False,
        "chronology": "expanding fit; only labels and prequential residuals matured by origin",
        "split_basis": "origin NAV date; targets may cross a year boundary",
        "overlap_warning": "Rolling H20 outcomes overlap; rolling count is not independent evidence. "
        "Disjoint blocks reduce overlap but do not establish independence.",
        "availability_assumption": {
            "nav_lag_observations": 1,
            "description": "Research decision assumes origin NAV known at the next supplied NAV "
            "observation. Actual acquired/available_at timestamps are not rewritten.",
        },
        "profit_claim": False,
    }


def replay_actions(
    series: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    *,
    entry_fee_rate: float,
    redemption_fee_rate: float,
    entry_lag: int = 2,
    confirmation_lag: int = 1,
    redemption_lag: int = 2,
    cash_lag: int = 3,
    nav_availability_lag: int = 1,
    initial_cash: float = 1.0,
    start_index: int = 0,
    end_index: int | None = None,
    decision_policy: Callable[[Mapping[str, Any], Mapping[str, bool]], str] | None = None,
) -> dict[str, Any]:
    """Replay buy/redeem decisions once on a cash/units ledger.

    By default actions contain origin_index and action ('buy', 'redeem', 'hold',
    'cash'); cash is a no-op, not an exit. With decision_policy, supply only the
    known prediction item at each origin. The callback receives that item and
    boolean eligibility flags, never the decision-index NAV, units, or wealth.
    Its selected actions and lifecycle are retained in strategy.action_trace;
    these rows can be replayed explicitly for a same-signal zero-fee diagnostic.
    The passive buy-and-hold reference never calls the policy.

    origin_index is the LAST KNOWN NAV, not a tradable same-day NAV. Buy/sell
    pricing is origin + entry_lag/redemption_lag respectively; decisions occur
    at origin + nav_availability_lag. All lags count supplied observations, not
    invented exchange days. Only one all-in position/order is allowed at once.
    Cash reserved for entry and cash in transit cannot fund another entry.

    Terminal wealth uses hypothetical common-end liquidation at the final NAV
    with the supplied scenario redemption fee; it is not immediately spendable
    cash. Fees are explicit research scenarios, not inferred product rules.
    No daily overlapping forecast return is ever compounded.
    """
    lags = (entry_lag, confirmation_lag, redemption_lag, cash_lag, nav_availability_lag)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in lags):
        raise ValidationError("replay lags must be nonnegative integer observation counts")
    if min(entry_lag, redemption_lag) < nav_availability_lag:
        raise ValidationError("replay cannot execute before assumed NAV availability")
    if (
        not math.isfinite(initial_cash)
        or initial_cash <= 0
        or any(
            not math.isfinite(rate) or not 0 <= rate < 1
            for rate in (entry_fee_rate, redemption_fee_rate)
        )
    ):
        raise ValidationError("invalid replay cash or fee scenario")
    if not series:
        raise ValidationError("replay requires NAV observations")
    days, navs = [], []
    for item in series:
        try:
            day = date.fromisoformat(item["date"]).isoformat()
            nav = float(item["nav"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("invalid replay NAV observation") from exc
        if (
            not math.isfinite(nav)
            or nav <= 0
            or (days and day <= days[-1])
            or isinstance(item["nav"], bool)
        ):
            raise ValidationError("replay requires positive finite NAVs and unique ascending dates")
        days.append(day)
        navs.append(nav)
    end_index = len(days) - 1 if end_index is None else end_index
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (start_index, end_index)
    ) or not 0 <= start_index <= end_index < len(days):
        raise ValidationError("invalid replay window")
    if any(
        item.get("invalid_return", False) or item.get("label_valid") is False
        for item in series[start_index + 1 : end_index + 1]
    ):
        raise ValidationError(
            "replay requires adjusted NAV/action mechanics across known invalid "
            "split/dividend returns; raw units cannot be replayed honestly"
        )
    if decision_policy is not None and not callable(decision_policy):
        raise ValidationError("replay decision_policy must be callable")
    decisions: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    action_names = {"buy", "redeem", "hold", "cash"}
    for item in actions:
        origin = item.get("origin_index")
        if isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin < len(days):
            raise ValidationError("invalid replay action origin")
        if decision_policy is None and item.get("action") not in action_names:
            raise ValidationError("unknown replay action")
        decisions[origin + nav_availability_lag].append(item)

    def run(
        schedule: Mapping[int, Sequence[Mapping[str, Any]]],
        policy: Callable[[Mapping[str, Any], Mapping[str, bool]], str] | None = None,
    ) -> dict[str, Any]:
        cash, reserved, units, receivable = float(initial_cash), 0.0, 0.0, 0.0
        confirmed_at, settles_at = -1, -1
        pending: tuple[str, int, int] | None = None
        pending_trace: dict[str, Any] | None = None
        daily, trades, skipped, action_trace = [], [], [], []
        fees = 0.0

        def observed_date(index: int) -> str | None:
            return days[index] if index < len(days) else None

        def execute(index: int) -> None:
            nonlocal pending, cash, reserved, units, receivable, confirmed_at, settles_at, fees
            nonlocal pending_trace
            if pending is None or pending[1] != index:
                return
            kind, _, origin = pending
            if kind == "buy":
                fee = reserved * entry_fee_rate / (1.0 + entry_fee_rate)
                gross_notional = reserved - fee
                units = gross_notional / navs[index]
                traded_units = units
                reserved = 0.0
                confirmed_at = index + confirmation_lag
            else:
                gross = units * navs[index]
                gross_notional, traded_units = gross, units
                fee = gross * redemption_fee_rate
                receivable += gross - fee
                units = 0.0
                settles_at = index + cash_lag
                if cash_lag == 0:
                    cash += receivable
                    receivable = 0.0
            fees += fee
            trade = {
                "action": kind,
                "origin_index": origin,
                "execution_index": index,
                "execution_date": days[index],
                "fee": fee,
                "gross_notional": gross_notional,
                "units": traded_units,
                "execution_nav": navs[index],
            }
            if policy is not None:
                trade.update(
                    origin_date=days[origin],
                    decision_index=origin + nav_availability_lag,
                    decision_date=days[origin + nav_availability_lag],
                    confirmation_index=confirmed_at if kind == "buy" else None,
                    confirmation_date=observed_date(confirmed_at) if kind == "buy" else None,
                    confirmation_status=("confirmed" if confirmed_at <= end_index else "unconfirmed")
                    if kind == "buy" else "not_applicable",
                    settlement_index=settles_at if kind == "redeem" else None,
                    settlement_date=observed_date(settles_at) if kind == "redeem" else None,
                    settlement_status=("settled" if settles_at <= end_index else "in_transit")
                    if kind == "redeem" else "not_applicable",
                )
                pending_trace.update(trade, status="executed")
            trades.append(trade)
            pending = None
            pending_trace = None

        for index in range(start_index, end_index + 1):
            if receivable and settles_at <= index:
                cash += receivable
                receivable = 0.0
            execute(index)
            for item in schedule.get(index, []):
                origin = item["origin_index"]
                trace = None
                if policy is None:
                    kind = item["action"]
                else:
                    state = {
                        "can_buy": not (pending or units or reserved or receivable or cash <= 0),
                        "can_redeem": pending is None and units > 0 and index >= confirmed_at,
                        "has_units": units > 0,
                        "order_pending": pending is not None,
                        "unconfirmed_units": units > 0 and index < confirmed_at,
                        "cash_in_transit": receivable > 0,
                    }
                    kind = policy(deepcopy(dict(item)), dict(state))
                    if not isinstance(kind, str) or kind not in action_names:
                        raise ValidationError("unknown replay decision_policy action")
                    trace = {
                        **deepcopy(dict(item)), "action": kind,
                        "origin_date": days[origin],
                        "decision_index": index, "decision_date": days[index],
                        "state": state, "status": "noop", "reason": None,
                        "scheduled_execution_index": None, "scheduled_execution_date": None,
                        "execution_index": None, "execution_date": None,
                        "confirmation_index": None, "confirmation_date": None,
                        "confirmation_status": "not_applicable",
                        "settlement_index": None, "settlement_date": None,
                        "settlement_status": "not_applicable",
                    }
                    action_trace.append(trace)
                if kind in {"hold", "cash"}:
                    continue
                execution = origin + (entry_lag if kind == "buy" else redemption_lag)
                if trace is not None:
                    trace.update(scheduled_execution_index=execution,
                                 scheduled_execution_date=observed_date(execution))
                reason = None
                if execution > end_index:
                    reason = "execution_after_replay_end"
                elif pending:
                    reason = "order_pending"
                elif kind == "buy" and (units or reserved or receivable or cash <= 0):
                    reason = "not_spendable_cash"
                elif kind == "redeem" and (units <= 0 or index < confirmed_at):
                    reason = "no_confirmed_units"
                if reason:
                    skipped.append({"origin_index": origin, "action": kind, "reason": reason})
                    if trace is not None:
                        trace.update(status="skipped", reason=reason)
                    continue
                if kind == "buy":
                    reserved, cash = cash, 0.0
                pending = (kind, execution, origin)
                pending_trace = trace
                if trace is not None:
                    trace["status"] = "pending"
                execute(index)
            market_value = units * navs[index]
            daily.append(
                {
                    "index": index,
                    "date": days[index],
                    "cash_available": cash,
                    "cash_reserved": reserved,
                    "cash_in_transit": receivable,
                    "units": units,
                    "confirmed_units": units if index >= confirmed_at else 0.0,
                    "market_value": market_value,
                    "marked_wealth": cash + reserved + receivable + market_value,
                    "fees_paid": fees,
                }
            )
        terminal_fee = units * navs[end_index] * redemption_fee_rate
        wealth = cash + reserved + receivable + units * navs[end_index] - terminal_fee
        result = {
            "terminal_liquidated_wealth": wealth,
            "return": wealth / initial_cash - 1.0,
            "terminal_liquidation_fee": terminal_fee,
            "fees_paid_before_terminal": fees,
            "trades": trades,
            "skipped_actions": skipped,
            "daily": daily,
        }
        if policy is not None:
            result["action_trace"] = action_trace
        return result

    # Use the same first research knowledge/entry opportunity for both paths.
    first_origin = max(0, start_index - nav_availability_lag)
    first_decision = first_origin + nav_availability_lag
    buy_hold_schedule = {first_decision: [{"origin_index": first_origin, "action": "buy"}]}
    strategy = run(decisions, decision_policy)
    buy_hold = run(buy_hold_schedule)
    return {
        "schema_version": 1,
        "assumptions": {
            "entry_fee_rate": float(entry_fee_rate),
            "redemption_fee_rate": float(redemption_fee_rate),
            "entry_lag": entry_lag,
            "confirmation_lag": confirmation_lag,
            "redemption_lag": redemption_lag,
            "cash_lag": cash_lag,
            "nav_availability_lag": nav_availability_lag,
            "lag_unit": "supplied NAV observations; not an exchange calendar",
            "cash_interest": 0.0,
            "terminal_basis": "hypothetical common-final-NAV liquidation net scenario "
            "redemption fee; proceeds not necessarily settled",
        },
        "start_date": days[start_index],
        "end_date": days[end_index],
        "strategy": strategy,
        "buy_and_hold": buy_hold,
        "cash": {"terminal_liquidated_wealth": float(initial_cash), "return": 0.0},
        "excess_return_vs_cash": strategy["return"],
        "excess_return_vs_buy_and_hold": strategy["return"] - buy_hold["return"],
        "warning": "Scenario replay, not evidence of historical executability or future profit.",
    }


def ledger_attribution(
    path: Mapping[str, Any], series: Sequence[Mapping[str, Any]], initial_cash: float = 1.0,
) -> dict[str, Any]:
    """Attribute one replay path to NAV exposure and fees, without resetting it.

    ``path`` is replay_actions(...)["strategy"] (or ["buy_and_hold"]). Priced,
    unconfirmed units are economic units; reserved entry cash and redemption
    receivables are not. The previous observation's units earn the NAV move,
    including the move into a sale execution, but not into a buy execution.
    Terminal hypothetical redemption fees occur only on the final observation.
    Year/half-year opening wealth and units carry over from the preceding mark.
    This is a bookkeeping identity, not causal timing attribution.
    """
    observations = path["daily"]
    if not observations:
        raise ValidationError("ledger attribution requires daily replay marks")
    contributions = []
    previous_wealth, previous_units, previous_fees = float(initial_cash), 0.0, 0.0
    previous_index = None
    terminal_fee = path["terminal_liquidation_fee"]
    for position, row in enumerate(observations):
        index = row["index"]
        if (previous_index is not None and index != previous_index + 1
                or series[index]["date"] != row["date"]):
            raise ValidationError("ledger attribution requires matching consecutive NAV marks")
        nav_move = (float(series[index]["nav"]) - float(series[previous_index]["nav"])) \
            if previous_index is not None else 0.0
        market_pnl = previous_units * nav_move
        fee = row["fees_paid"] - previous_fees
        final_fee = terminal_fee if position == len(observations) - 1 else 0.0
        net_contribution = market_pnl - fee - final_fee
        wealth_change = row["marked_wealth"] - previous_wealth - final_fee
        contributions.append({
            "index": index, "date": row["date"], "nav_change": nav_move,
            "opening_economic_units": previous_units, "closing_economic_units": row["units"],
            "opening_wealth": previous_wealth, "closing_marked_wealth": row["marked_wealth"],
            "closing_wealth": row["marked_wealth"] - final_fee,
            "market_pnl": market_pnl, "executed_fees": fee,
            "terminal_liquidation_fee": final_fee,
            "net_contribution": net_contribution, "wealth_change": wealth_change,
            "identity_residual": wealth_change - net_contribution,
        })
        previous_index, previous_units = index, row["units"]
        previous_wealth, previous_fees = row["marked_wealth"], row["fees_paid"]

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        market = math.fsum(row["market_pnl"] for row in rows)
        paid = math.fsum(row["executed_fees"] for row in rows)
        liquidation = math.fsum(row["terminal_liquidation_fee"] for row in rows)
        change = rows[-1]["closing_wealth"] - rows[0]["opening_wealth"]
        explained = market - paid - liquidation
        return {
            "start_date": rows[0]["date"], "end_date": rows[-1]["date"],
            "opening_wealth": rows[0]["opening_wealth"],
            "closing_marked_wealth": rows[-1]["closing_marked_wealth"],
            "closing_wealth": rows[-1]["closing_wealth"],
            "opening_economic_units": rows[0]["opening_economic_units"],
            "closing_economic_units": rows[-1]["closing_economic_units"],
            "market_pnl": market, "executed_fees": paid,
            "terminal_liquidation_fee": liquidation,
            "wealth_change": change, "explained_change": explained,
            "net_contribution": explained, "identity_residual": change - explained,
        }

    years, halves = defaultdict(list), defaultdict(list)
    for row in contributions:
        year, month = row["date"][:4], int(row["date"][5:7])
        years[year].append(row)
        halves[f"{year}_h{1 if month <= 6 else 2}"].append(row)
    result = aggregate(contributions)
    # Use the reported liquidated endpoint too, so a terminal-accounting error
    # cannot disappear merely because the marked path itself reconciles.
    result["wealth_change"] = path["terminal_liquidated_wealth"] - initial_cash
    result["identity_residual"] = result["wealth_change"] - result["explained_change"]
    result.update(
        initial_cash=float(initial_cash),
        terminal_liquidated_wealth=path["terminal_liquidated_wealth"],
        daily=contributions,
        years={key: aggregate(rows) for key, rows in years.items()},
        half_years={key: aggregate(rows) for key, rows in halves.items()},
        interpretation="Bookkeeping NAV-exposure/fee attribution, not causal timing alpha; "
        "terminal wealth includes redemption receivables, not immediately spendable cash.",
    )
    return result


STUDY_DEFINITION = {
    "definition_id": "nav_forecast_action_replay_v1",
    "periods": [2025, 2026],
    "period_basis": "actual assumed decision date = next supplied NAV date after origin",
    "horizon": 20,
    "entry_lags": [2, 3, 4],
    "entry_fee_rate": 0.0,
    "redemption_fee_rates": [0.0, 0.005, 0.015],
    "decision_rule": "buy iff expected_change > entry_fee_rate; otherwise redeem iff "
    "expected_change < -redemption_fee_rate; otherwise hold",
    "matching_head": "H20 with entry_lag identical to the scenario; no substitute head",
    "confirmation_lag": 1,
    "cash_lag": 3,
    "nav_availability_lag": 1,
    "lag_unit": "supplied NAV observations, not invented trading days",
    "redemption_lag": "same as scenario entry_lag, measured from known NAV origin",
    "initial_cash": 1.0,
    "window": "first through last supplied NAV date of each year; restart with cash each year",
    "terminal_basis": "all paths hypothetically liquidated at common final NAV, net scenario "
    "redemption fee; cash may not have settled",
    "coverage": "every decision-day matching issued head required, including immature targets; "
    "otherwise whole scenario unavailable",
    "event_policy": "known invalid raw-NAV transition requires adjusted/action-aware replay input",
    "fee_provenance": "public-context stress scenarios only; not verified channel/product fees",
    "strict_unseen": False,
    "hyperparameter_search": False,
}
STUDY_DEFINITION_HASH = digest(STUDY_DEFINITION)


def study_actions(
    series: Sequence[Mapping[str, Any]],
    issued_forecasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the fixed 2025/2026 fee/lag study on complete issued H20 predictions.

    Never supply matured evaluation_rows here: that would silently remove the
    last H20+lag decisions and forecasts whose future labels become invalid.
    Complete matching issuance is required for EVERY decision date of a year.
    Outcomes/target validity are never read when constructing actions. Periods
    are split on assumed decision dates, not on forecast origin/target dates.
    """
    if not series:
        raise ValidationError("action study requires frozen NAV observations")
    days = []
    for item in series:
        try:
            day = date.fromisoformat(item["date"]).isoformat()
            nav = float(item["nav"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("invalid action study NAV observation") from exc
        if (
            item["date"] != day
            or (days and day <= days[-1])
            or not math.isfinite(nav)
            or nav <= 0
            or isinstance(item["nav"], bool)
        ):
            raise ValidationError(
                "action study requires positive NAVs and canonical ascending dates"
            )
        days.append(day)
    issued: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in issued_forecasts:
        if row.get("horizon") != 20 or row.get("entry_lag") not in (2, 3, 4):
            continue
        origin, known = row.get("origin_index"), row.get("assumed_known_index")
        if (
            isinstance(origin, bool)
            or not isinstance(origin, int)
            or not 0 <= origin < len(days)
            or known != origin + 1
            or row.get("origin_date") != days[origin]
        ):
            raise ValidationError("invalid issued forecast chronology in action study")
        if (
            known < len(days)
            and row.get("assumed_known_date") != days[known]
            or known == len(days)
            and row.get("assumed_known_date") is not None
        ):
            raise ValidationError("issued forecast assumed decision date mismatches frozen NAVs")
        prediction = row.get("prediction", {})
        mean = prediction.get("expected_change")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or mean < -1
            or prediction.get("method")
            not in {"ridge_prequential_empirical", "empirical_historical"}
        ):
            raise ValidationError("action study requires a finite matching issued NAV forecast")
        for field in ("training_target_end_index", "residual_target_end_index"):
            cutoff = prediction.get(field)
            if cutoff is not None and cutoff > origin:
                raise ValidationError("action study issued forecast contains a future target")
        identity = known, row["entry_lag"]
        if identity in issued:
            raise ValidationError("duplicate issued forecast for an action-study decision/head")
        issued[identity] = row

    scenarios = []
    for year in STUDY_DEFINITION["periods"]:
        indices = [index for index, day in enumerate(days) if day.startswith(f"{year}-")]
        for lag in STUDY_DEFINITION["entry_lags"]:
            missing = [days[index] for index in indices if (index, lag) not in issued]
            for redemption_rate in STUDY_DEFINITION["redemption_fee_rates"]:
                scenario = {
                    "scenario_id": f"{STUDY_DEFINITION['definition_id']}:{year}:lag{lag}:"
                    f"entry0bp:redemption{round(redemption_rate * 10000)}bp",
                    "period": str(year),
                    "horizon": 20,
                    "entry_lag": lag,
                    "entry_fee_rate": 0.0,
                    "redemption_fee_rate": redemption_rate,
                    "start_date": days[indices[0]] if indices else None,
                    "end_date": days[indices[-1]] if indices else None,
                    "expected_decision_count": len(indices),
                    "issued_forecast_count": len(indices) - len(missing),
                    "action_count": 0,
                    "executed_action_count": 0,
                }
                if not indices or missing:
                    scenario.update(
                        status="unavailable",
                        reason=(
                            "no_nav_observations_in_period"
                            if not indices
                            else "missing_matching_issued_forecasts"
                        ),
                        missing_decision_dates=missing,
                    )
                    scenarios.append(scenario)
                    continue
                actions, counts = [], {"buy": 0, "redeem": 0, "hold": 0}
                for index in indices:
                    row = issued[index, lag]
                    prediction = row["prediction"]
                    mean = float(prediction["expected_change"])
                    action = (
                        "buy" if mean > 0.0 else "redeem" if mean < -redemption_rate else "hold"
                    )
                    counts[action] += 1
                    actions.append(
                        {
                            "origin_index": row["origin_index"],
                            "origin_date": row["origin_date"],
                            "decision_index": index,
                            "decision_date": days[index],
                            "horizon": 20,
                            "entry_lag": lag,
                            "expected_change": mean,
                            "method": prediction["method"],
                            "action": action,
                        }
                    )
                scenario.update(
                    actions=actions, signal_action_counts=counts, action_count=len(actions)
                )
                try:
                    replay = replay_actions(
                        series,
                        actions,
                        entry_fee_rate=0.0,
                        redemption_fee_rate=redemption_rate,
                        entry_lag=lag,
                        redemption_lag=lag,
                        confirmation_lag=1,
                        cash_lag=3,
                        nav_availability_lag=1,
                        initial_cash=1.0,
                        start_index=indices[0],
                        end_index=indices[-1],
                    )
                except ValidationError as exc:
                    scenario.update(status="unavailable", reason=str(exc))
                else:
                    scenario.update(
                        status="available",
                        replay=replay,
                        executed_action_count=len(replay["strategy"]["trades"]),
                    )
                scenarios.append(scenario)
    return {
        "schema_version": 1,
        "definition_id": STUDY_DEFINITION["definition_id"],
        "definition_hash": STUDY_DEFINITION_HASH,
        "definition": {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in STUDY_DEFINITION.items()
        },
        "scenarios": scenarios,
        "strict_unseen": False,
        "profit_claim": False,
        "warning": "Fixed scenario comparison, not a verified channel-fee backtest or future "
        "profit claim. Full issued forecasts are used, never matured-label selection.",
    }

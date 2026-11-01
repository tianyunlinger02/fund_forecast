"""Frozen-file-only comparisons; no Workspace, database, network, or account access.

Dates describe NAV observations, not a reconstructed publication calendar. Candidates
are scored on matched origins, and their immature issued tails remain in action replay.
This research module does not select or promote a runtime model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .domain import ValidationError, now_iso
from .forecast import MODEL_DEFINITION, MODEL_HASH, MODEL_ID, _digest, fit_forecast, forecast_input_hash
from .forecast_candidates import (
    DEFAULT_HEADS, RISK_HEADS, RISK_ID, TREE_ID, TREND_ID, candidate_definition, candidate_hash,
    issue_candidate_forecasts, issue_ridge_distribution_forecasts, issue_risk_forecasts, require_tree,
)
from .forecast_research import (
    _disjoint, _metrics as distribution_metrics, paired_loss_bootstrap, replay_actions,
)

INPUT_SCHEMA = "rmbfund.forecast.inputs.v1"
PROTOCOL_SCHEMA = "rmbfund.forecast-comparison.protocol.v1"
MAX_INPUT_BYTES = 16 * 1024 * 1024
PERIODS = {
    "all": ("2025-01-01", "2026-12-31"),
    "development_2025": ("2025-01-01", "2025-12-31"),
    "heldout_2026": ("2026-01-01", "2026-12-31"),
    "development_2025_h1": ("2025-01-01", "2025-06-30"),
    "development_2025_h2": ("2025-07-01", "2025-12-31"),
    "heldout_2026_h1": ("2026-01-01", "2026-06-30"),
    "heldout_2026_h2": ("2026-07-01", "2026-12-31"),
}
ENTRY_FEES = (0.0, 0.0015)
REDEMPTION_FEES = (0.0, 0.005, 0.015)
VIX_ID = "nav_vix_ridge_empirical_v1"
LABELS = {
    MODEL_ID: "现有净值Ridge", TREND_ID: "固定60笔趋势", TREE_ID: "标准浅梯度树",
    VIX_ID: "净值＋VIX线性模型", RISK_ID: "同Ridge位置＋20笔波动缩放分布", "historical_frequency": "历史经验分布",
    "always_up": "一直猜涨", "no_change": "净值不变",
}
EVIDENCE = {
    "availability_basis": "valuation_date_exploration", "point_in_time_verified": False,
    "strict_unseen": False, "strict_forward_outcome_count": 0, "profit_claim": False,
}


def _reject_constant(value: str) -> None:
    raise ValidationError(f"JSON不接受非有限数值：{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("JSON包含重复字段")
        result[key] = value
    return result


def _read_json(path: str | Path, *, limit: int = MAX_INPUT_BYTES) -> tuple[Any, bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > limit:
        raise ValidationError("冻结输入必须是大小范围内的普通JSON文件")
    raw = source.read_bytes()
    if len(raw) > limit:
        raise ValidationError("冻结输入超过大小上限")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("冻结输入不是有效UTF-8 JSON") from exc
    return value, raw


def load_frozen_inputs(path: str | Path, *, product_id: str | None = None) -> tuple[dict, bytes]:
    """Verify the original v1 input identity without following any embedded paths."""
    value, raw = _read_json(path)
    if not isinstance(value, dict) or value.get("schema") != INPUT_SCHEMA:
        raise ValidationError("未知冻结输入格式")
    product = value.get("product")
    if not isinstance(product, dict) or not isinstance(product.get("id"), str) or not product["id"]:
        raise ValidationError("冻结输入产品标识无效")
    if product_id is not None and product["id"] != product_id:
        raise ValidationError("冻结输入产品不匹配")
    if product.get("kind") != "open_fund" or product.get("is_demo") is not False:
        raise ValidationError("比较需要公开场外基金冻结输入")
    if value.get("model_hash") != MODEL_HASH:
        raise ValidationError("冻结输入模型哈希不匹配")
    if value.get("input_hash") != forecast_input_hash(value.get("series")):
        raise ValidationError("冻结输入内容哈希不匹配")
    return value, raw


def audit_availability(frozen: Mapping[str, Any]) -> dict[str, Any]:
    series = frozen["series"]
    times: dict[str, list[str]] = {name: [] for name in ("published_at", "observed_at", "available_at")}
    for row in series:
        provenance = row.get("provenance", {})
        for name, values in times.items():
            value = row.get(name) or (provenance.get(name) if isinstance(provenance, Mapping) else None)
            if isinstance(value, str) and value:
                values.append(value)
    return {
        **EVIDENCE, "observation_count": len(series),
        "valuation_date_start": series[0]["date"], "valuation_date_end": series[-1]["date"],
        **{f"{name}_count": len(values) for name, values in times.items()},
        "actual_available_at_min": min(times["available_at"], default=None),
        "actual_available_at_max": max(times["available_at"], default=None),
        "invalid_return_count": sum(row.get("invalid_return", False) or row.get("label_valid") is False for row in series),
        "event_coverage": "not established by absence of event flags",
        "note": "真实取得时钟原样保留；按估值日期的历史研究不等于真实历史可交易回放。",
    }


def _versions(*, research_dependency: bool) -> dict[str, str]:
    names = ["numpy"] + (["scikit-learn", "scipy", "threadpoolctl"] if research_dependency else [])
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValidationError(f"缺少显式研究依赖：{name}") from exc
    return result


def build_comparison_protocol(
    *, include_tree: bool = False, inputs: Sequence[Mapping[str, Any]] = (),
    factor: Mapping[str, Any] | None = None, protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    definitions = {
        MODEL_ID: {"candidate_hash": MODEL_HASH, "definition": MODEL_DEFINITION},
        TREND_ID: {"candidate_hash": candidate_hash(TREND_ID), "definition": candidate_definition(TREND_ID)},
    }
    if include_tree:
        definitions[TREE_ID] = {"candidate_hash": candidate_hash(TREE_ID), "definition": candidate_definition(TREE_ID)}
    if factor is not None:
        from .forecast_candidates import factor_candidate_definition

        definition = factor_candidate_definition(factor["definition"])
        definitions[VIX_ID] = {"candidate_hash": _digest(definition), "definition": definition}
    result = {
        "schema": PROTOCOL_SCHEMA, "definitions": definitions,
        "input_identities": list(inputs),
        "factor_input": ({"input_hash": factor["input_hash"], "file_sha256": factor["file_sha256"]} if factor else None),
        "candidate_ids": list(definitions),
        "baseline_ids": ["historical_frequency", "no_change", "always_up"],
        "heads": [list(pair) for pair in DEFAULT_HEADS], "primary_head": [20, 0],
        "periods": {name: list(period) for name, period in PERIODS.items()},
        "period_labels": {
            "development_2025": "2025开发历史", "heldout_2026": "2026已查看历史验证", "all": "全部历史验证",
            "development_2025_h1": "2025上半年开发历史", "development_2025_h2": "2025下半年开发历史",
            "heldout_2026_h1": "2026上半年已查看历史", "heldout_2026_h2": "2026下半年已查看历史",
        },
        "training": "expanding at every origin; only full targets matured by origin; training-only preprocessing",
        "residuals": "each candidate's own raw prequential errors, admitted only at full target maturity",
        "matching": "same fund, origin, horizon and entry lag; full pipelines and all-fitted common samples reported separately",
        "direction": "up_probability >= 0.5 predicts up; actual_change > 0 is up; flat is non-up",
        "metrics": ["direction_accuracy", "up_brier", "multiclass_brier", "mae", "rmse", "mean_error", "interval_coverage", "mean_interval_width", "up_calibration"],
        "probability_bins": 5, "interval_nominal_coverage": 0.8,
        "overlap": "rolling count and greedy nonoverlapping blocks separately; neither establishes independence",
        "action_rule": "buy iff expected_change > entry_fee_rate; redeem iff expected_change < -redemption_fee_rate; otherwise hold",
        "entry_fee_rates": list(ENTRY_FEES), "redemption_fee_rates": list(REDEMPTION_FEES),
        "entry_lags": [2, 3, 4], "nav_availability_lag": 1, "confirmation_lag": 1, "cash_lag": 3,
        "redemption_lag": "same as scenario entry_lag", "initial_cash": 1.0,
        "lag_unit": "supplied NAV observations, not an exchange calendar",
        "cost_periods": [2025, 2026], "cost_window": "all supplied NAV dates in each calendar year; start from cash each year",
        "terminal_basis": "hypothetical common-final-NAV liquidation net scenario redemption fee; cash need not be settled",
        "fee_provenance": "fixed stress scenarios, not verified channel/product fees; NAV operating fees not charged twice",
        "promotion": "none; historical comparison does not promote a runtime model",
        "dependencies": _versions(research_dependency=include_tree or factor is not None),
        "hyperparameter_search": False, **EVIDENCE,
    }
    result["protocol_hash"] = _digest(result)
    if protocol is not None and (not isinstance(protocol, Mapping) or dict(protocol) != result):
        raise ValidationError("研究协议、输入、依赖版本或候选定义不匹配，不静默修改协议")
    return result


def _identity(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return row["origin_index"], row["horizon"], row["entry_lag"]


def _available(row: Mapping[str, Any]) -> bool:
    return row["prediction"].get("status") == "available"


def _fitted(row: Mapping[str, Any]) -> bool:
    return _available(row) and row["prediction"].get("method") != "empirical_historical"


def _row_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for row in rows:
        if not _available(row):
            reason = row["prediction"].get("reason", "unavailable")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "eligible_count": len(rows), "available_count": sum(_available(row) for row in rows),
        "fitted_count": sum(_fitted(row) for row in rows),
        "fallback_count": sum(_available(row) and not _fitted(row) for row in rows),
        "unavailable_count": sum(not _available(row) for row in rows),
        "unavailable_reasons": reasons,
    }


def _metrics(rows: Sequence[Mapping[str, Any]], key: str = "prediction") -> dict[str, Any]:
    result = distribution_metrics(rows, key)
    if not rows:
        return {**result, "direction_accuracy": None, "always_up_accuracy": None, "disjoint_count": 0}
    result.update(
        direction_accuracy=float(np.mean([(row[key]["up_probability"] >= 0.5) == (row["actual_change"] > 0) for row in rows])),
        always_up_accuracy=float(np.mean([row["actual_change"] > 0 for row in rows])),
        disjoint_count=len(_disjoint(rows)),
    )
    result["direction_minus_always_up"] = result["direction_accuracy"] - result["always_up_accuracy"]
    return result


def _baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    historical = _metrics(rows, "historical_baseline")
    point = _metrics(rows, "no_change_baseline")
    # A point prediction is an amplitude/zero-width baseline, not a useful
    # calibrated probability model. No arbitrary 50% probability is invented.
    point = {key: value for key, value in point.items() if key in {
        "count", "mae", "rmse", "mean_error", "interval_coverage", "mean_interval_width", "disjoint_count"
    }}
    return {
        "historical_frequency": historical, "no_change": point,
        "always_up": {"count": len(rows), "direction_accuracy": float(np.mean([row["actual_change"] > 0 for row in rows])) if rows else None,
                      "disjoint_count": len(_disjoint(rows))},
    }


def _loss_deltas(model: Mapping[str, Any], baselines: Mapping[str, Any]) -> dict[str, Any]:
    return {name: {metric: model[metric] - baseline[metric] for metric in ("up_brier", "mae", "rmse", "direction_accuracy")
                   if metric in model and metric in baseline and model[metric] is not None and baseline[metric] is not None}
            for name, baseline in baselines.items()}


def evaluate_candidate_rows(
    bundles: Mapping[str, Mapping[str, Any]], *,
    heads: Sequence[tuple[int, int]] = DEFAULT_HEADS,
    periods: Mapping[str, Sequence[str]] | None = None,
    mature_by_period_end: bool = False,
    fitted_candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Match origins, optionally purging targets beyond each diagnostic cutoff.

    Passive empirical references need not be regressors in the fitted cohort.
    The default periods and all-candidate fitted rule retain the old protocol.
    """
    selected_periods = PERIODS if periods is None else periods
    fitted_ids = tuple(bundles) if fitted_candidate_ids is None else tuple(fitted_candidate_ids)
    if any(candidate_id not in bundles for candidate_id in fitted_ids):
        raise ValidationError("拟合队列指定了不存在的候选")
    maps = {}
    for candidate_id, bundle in bundles.items():
        mapping = {}
        for row in bundle.get("evaluation_rows", []):
            identity = _identity(row)
            if identity in mapping:
                raise ValidationError("候选存在重复评分起点")
            for field in ("training_target_end_index", "residual_target_end_index"):
                cutoff = row["prediction"].get(field)
                if cutoff is not None and cutoff > row["origin_index"]:
                    raise ValidationError("候选评分包含未来训练或残差终点")
            if row["target_index"] - row["entry_index"] != row["horizon"]:
                raise ValidationError("候选评分目标期限不匹配")
            mapping[identity] = row
        maps[candidate_id] = mapping
    common_eligible = set.intersection(*(set(mapping) for mapping in maps.values())) if maps else set()
    for identity in common_eligible:
        reference = next(iter(maps.values()))[identity]
        fields = ("origin_date", "entry_index", "entry_date", "target_index", "target_date", "actual_change")
        if any(any(mapping[identity][field] != reference[field] for field in fields) for mapping in maps.values()):
            raise ValidationError("共同起点的实际目标不一致")
    common = {identity for identity in common_eligible if all(_available(mapping[identity]) for mapping in maps.values())}
    fitted_common = {identity for identity in common
                     if all(_fitted(maps[candidate_id][identity]) for candidate_id in fitted_ids)}
    head_results = []
    for horizon, lag in heads:
        period_results = {}
        for period, (start, end) in selected_periods.items():
            def in_period(row):
                return (start <= row["origin_date"] <= end
                        and (not mature_by_period_end or row["target_date"] <= end))

            keys = sorted(identity for identity in common if identity[1:] == (horizon, lag)
                          and in_period(next(iter(maps.values()))[identity]))
            fitted_keys = [identity for identity in keys if identity in fitted_common]
            reference = [next(iter(maps.values()))[identity] for identity in keys] if maps else []
            baselines = _baselines(reference)
            candidates = {}
            fitted_results = {}
            for candidate_id, mapping in maps.items():
                rows = [mapping[identity] for identity in keys]
                metrics = _metrics(rows)
                candidates[candidate_id] = {
                    "metrics": metrics, "fitted_count": sum(_fitted(row) for row in rows),
                    "fallback_count": sum(not _fitted(row) for row in rows),
                    "paired_deltas": _loss_deltas(metrics, baselines),
                    "disjoint_metrics": _metrics(_disjoint(rows)),
                }
                fitted_results[candidate_id] = _metrics([mapping[identity] for identity in fitted_keys])
            period_results[period] = {
                "configured_start": start, "configured_end": end, "common_count": len(keys),
                "common_eligible_count": sum(identity[1:] == (horizon, lag)
                                             and in_period(next(iter(maps.values()))[identity])
                                             for identity in common_eligible),
                "candidate_coverage": {
                    candidate_id: _row_coverage([row for row in mapping.values()
                                                if (row["horizon"], row["entry_lag"]) == (horizon, lag)
                                                and in_period(row)])
                    for candidate_id, mapping in maps.items()
                },
                "actual_origin_start": reference[0]["origin_date"] if reference else None,
                "actual_origin_end": reference[-1]["origin_date"] if reference else None,
                "last_target_date": max((row["target_date"] for row in reference), default=None),
                "baselines": baselines, "candidates": candidates,
                "all_fitted_common": {"count": len(fitted_keys), "candidates": fitted_results,
                                      "baselines": _baselines([next(iter(maps.values()))[identity] for identity in fitted_keys]) if maps else _baselines([])},
            }
        head_results.append({"horizon": horizon, "entry_lag": lag, "periods": period_results})
    return {
        "schema": "rmbfund.forecast-comparison.metrics.v1", "heads": head_results,
        "coverage": {candidate_id: {"evaluation_count": len(mapping),
                                   **_row_coverage(list(mapping.values())),
                                   "common_count": len(common), "common_eligible_count": len(common_eligible),
                                   "outside_common_count": len(set(mapping) - common),
                                   "unavailable_heads": bundles[candidate_id].get("unavailable_forecasts", [])}
                     for candidate_id, mapping in maps.items()},
        "overlap_warning": "滚动行不是独立试验，不重叠区段也不保证独立。", **EVIDENCE,
    }


def compare_paired_losses(
    bundles: Mapping[str, Mapping[str, Any]], *,
    heads: Sequence[tuple[int, int]] = RISK_HEADS,
    replicates: int = 2000, seed: int = 20261029,
    candidate_id: str = RISK_ID, reference_id: str = MODEL_ID,
    periods: Mapping[str, Sequence[str]] | None = None,
    mature_by_period_end: bool = False,
) -> dict[str, Any]:
    """Same-origin candidate-minus-reference block uncertainty, never promotion."""
    selected_periods = PERIODS if periods is None else periods
    maps = {candidate_id: {_identity(row): row for row in bundle["evaluation_rows"]}
            for candidate_id, bundle in bundles.items()}
    common = sorted(set.intersection(*(
        {key for key, row in mapping.items() if _available(row)} for mapping in maps.values()
    ))) if maps else []
    fitted = {key for key in common if all(_fitted(mapping[key]) for mapping in maps.values())}
    result = []
    for horizon, lag in heads:
        period_results = {}
        for period, (start, end) in selected_periods.items():
            keys = [key for key in common if key[1:] == (horizon, lag)
                    and start <= maps[candidate_id][key]["origin_date"] <= end
                    and (not mature_by_period_end or maps[candidate_id][key]["target_date"] <= end)]
            cohorts = {}
            previous_keys = None
            for cohort, selected in (("common_available", keys), ("common_fitted", [key for key in keys if key in fitted])):
                if previous_keys == selected:
                    cohorts[cohort] = {"same_origins_as": "common_available", "count": len(selected)}
                    continue
                paired = [{**maps[candidate_id][key], "old_ridge": maps[reference_id][key]["prediction"]} for key in selected]
                cohorts[cohort] = {
                    reference: paired_loss_bootstrap(
                        paired, reference_key=reference, replicates=replicates, seed=seed,
                    ) for reference in ("old_ridge", "historical_baseline", "no_change", "always_up")
                }
                previous_keys = selected
            period_results[period] = cohorts
        result.append({"horizon": horizon, "entry_lag": lag, "periods": period_results})
    schema = ("rmbfund.risk-distribution.bootstrap.v1" if candidate_id == RISK_ID
              else "rmbfund.forecast-comparison.bootstrap.v1")
    return {"schema": schema, "heads": result,
            "candidate": candidate_id, "skill_certified": False, **EVIDENCE}


def _path_summary(path: Mapping[str, Any]) -> dict[str, Any]:
    wealth = [float(row["marked_wealth"]) for row in path["daily"]] + [float(path["terminal_liquidated_wealth"])]
    peak, drawdown = 0.0, 0.0
    for value in wealth:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1 if peak else 0.0)
    exposure = [row["market_value"] / row["marked_wealth"] if row["marked_wealth"] else 0.0 for row in path["daily"]]
    buys = [trade for trade in path["trades"] if trade["action"] == "buy"]
    redemptions = [trade for trade in path["trades"] if trade["action"] == "redeem"]
    daily = path["daily"]
    return {
        "return": path["return"], "max_drawdown": drawdown, "trade_count": len(path["trades"]),
        "buy_count": len(buys), "redemption_count": len(redemptions),
        "reentry_count": max(0, len(buys) - 1), "round_trip_count": len(redemptions),
        "turnover": sum(trade["gross_notional"] for trade in path["trades"]),
        "turnover_basis": "one-way gross executed buy+redemption notional / initial cash=1; excludes hypothetical terminal liquidation",
        "terminal_liquidation_notional": daily[-1]["market_value"] if daily else 0.0,
        "total_fees": path["fees_paid_before_terminal"] + path["terminal_liquidation_fee"],
        "repeat_entry_fees": sum(trade["fee"] for trade in buys[1:]),
        "round_trip_redemption_fees": sum(trade["fee"] for trade in redemptions),
        "max_cash_reserved": max((row["cash_reserved"] for row in daily), default=0.0),
        "max_cash_in_transit": max((row["cash_in_transit"] for row in daily), default=0.0),
        "unconfirmed_observation_count": sum(row["units"] > row["confirmed_units"] for row in daily),
        "in_transit_observation_count": sum(row["cash_in_transit"] > 0 for row in daily),
        "terminal_cash_available": daily[-1]["cash_available"] if daily else 0.0,
        "terminal_cash_in_transit": daily[-1]["cash_in_transit"] if daily else 0.0,
        "mean_market_exposure": float(np.mean(exposure)) if exposure else 0.0,
    }


def compare_action_scenarios(
    series: Sequence[Mapping[str, Any]], bundles: Mapping[str, Mapping[str, Any]], *,
    entry_lags: Sequence[int] = (2, 3, 4),
    include_cost_diagnostics: bool = False,
) -> dict[str, Any]:
    days = [row["date"] for row in series]
    scenarios = []
    for candidate_id, bundle in bundles.items():
        issued = {}
        for row in bundle.get("issued_forecasts", []):
            if row.get("horizon") != 20 or row.get("entry_lag") not in entry_lags:
                continue
            origin, known = row.get("origin_index"), row.get("assumed_known_index")
            if isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin < len(days) or known != origin + 1:
                raise ValidationError("候选已发出序列的时间顺序无效")
            if row.get("origin_date") != days[origin] or row.get("assumed_known_date") != (days[known] if known < len(days) else None):
                raise ValidationError("候选已发出序列的日期不匹配")
            prediction = row["prediction"]
            mean = prediction.get("expected_change")
            if _available(row) and (isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < -1):
                raise ValidationError("候选执行预测须为有限收益")
            if (prediction.get("horizon"), prediction.get("entry_lag")) != (20, row["entry_lag"]):
                raise ValidationError("候选执行预测头不匹配")
            if any(prediction.get(field) is not None and prediction[field] > origin
                   for field in ("training_target_end_index", "residual_target_end_index")):
                raise ValidationError("候选执行预测含未来训练终点")
            key = known, row["entry_lag"]
            if key in issued:
                raise ValidationError("候选存在重复执行预测头")
            issued[key] = row
        for year in (2025, 2026):
            indices = [index for index, day in enumerate(days) if day.startswith(f"{year}-")]
            for lag in entry_lags:
                missing = [days[index] for index in indices if (index, lag) not in issued]
                unavailable = [days[index] for index in indices if (index, lag) in issued and not _available(issued[index, lag])]
                for entry_fee in ENTRY_FEES:
                    for redemption_fee in REDEMPTION_FEES:
                        scenario = {
                            "candidate_id": candidate_id, "year": year, "horizon": 20, "entry_lag": lag,
                            "entry_fee_rate": entry_fee, "redemption_fee_rate": redemption_fee,
                            "expected_decision_count": len(indices), "issued_decision_count": len(indices) - len(missing),
                            "scenario_id": f"{candidate_id}:{year}:lag{lag}:entry{round(entry_fee * 10000)}bp:redeem{round(redemption_fee * 10000)}bp",
                        }
                        scenario["unavailable_decision_count"] = len(unavailable)
                        if not indices or missing or unavailable:
                            scenario.update(status="unavailable", reason="no_nav_in_year" if not indices else "missing_matching_issued_forecasts" if missing else "unavailable_matching_issued_forecasts", missing_decision_dates=missing, unavailable_decision_dates=unavailable)
                            scenarios.append(scenario)
                            continue
                        actions = []
                        for index in indices:
                            row = issued[index, lag]
                            mean = row["prediction"]["expected_change"]
                            action = "buy" if mean > entry_fee else "redeem" if mean < -redemption_fee else "hold"
                            actions.append({"origin_index": row["origin_index"], "action": action,
                                            "decision_date": days[index], "expected_change": mean,
                                            "method": row["prediction"].get("method")})
                        scenario["actions"] = actions
                        try:
                            replay = replay_actions(series, actions, entry_fee_rate=entry_fee,
                                                    redemption_fee_rate=redemption_fee, entry_lag=lag, redemption_lag=lag,
                                                    confirmation_lag=1, cash_lag=3, nav_availability_lag=1,
                                                    initial_cash=1.0, start_index=indices[0], end_index=indices[-1])
                        except ValidationError as exc:
                            scenario.update(status="unavailable", reason=str(exc))
                        else:
                            scenario.update(status="available", replay=replay,
                                            strategy_summary=_path_summary(replay["strategy"]),
                                            buy_and_hold_summary=_path_summary(replay["buy_and_hold"]),
                                            cash_summary={"return": 0.0, "trade_count": 0, "turnover": 0.0, "total_fees": 0.0, "max_drawdown": 0.0})
                            if include_cost_diagnostics:
                                gross = replay_actions(
                                    series, actions, entry_fee_rate=0.0, redemption_fee_rate=0.0,
                                    entry_lag=lag, redemption_lag=lag, confirmation_lag=1,
                                    cash_lag=3, nav_availability_lag=1, initial_cash=1.0,
                                    start_index=indices[0], end_index=indices[-1],
                                )
                                scenario["same_signals_zero_fee_replay"] = gross
                                scenario["cost_diagnostics"] = {
                                    "strategy_same_signals_zero_fee_return": gross["strategy"]["return"],
                                    "strategy_total_cost_drag": gross["strategy"]["return"] - replay["strategy"]["return"],
                                    "buy_hold_total_cost_drag": gross["buy_and_hold"]["return"] - replay["buy_and_hold"]["return"],
                                    "strategy_minus_buy_hold_return": replay["strategy"]["return"] - replay["buy_and_hold"]["return"],
                                    "basis": "identical fee-conditioned signals on a zero-fee ledger; includes wealth effects, not pure causal timing loss",
                                }
                        scenarios.append(scenario)
    return {"schema": "rmbfund.forecast-comparison.scenarios.v1", "scenarios": scenarios, **EVIDENCE}


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _headline(metrics: Mapping[str, Any]) -> dict[str, Any]:
    head = next((row for row in metrics["heads"] if (row["horizon"], row["entry_lag"]) == (20, 0)), None)
    return head["periods"] if head else {}


def _markdown(product: Mapping[str, Any], metrics: Mapping[str, Any], scenarios: Mapping[str, Any]) -> str:
    lines = [f"# {product.get('symbol', product['id'])} 离线模型对照", "",
             "## 结论", "", "这次是历史探索，不是新模型已具备买卖能力的证明；没有自动选冠军或替换线上模型。",
             "逐日真实公布时间缺失，2026历史也已查看。下面的准确率是方向命中率，不是账户盈利概率。", ""]
    for period, title in (("development_2025", "2025开发历史"), ("heldout_2026", "2026已查看历史验证")):
        item = _headline(metrics).get(period, {})
        lines += [f"## {title}：未来20笔（主任务）", "",
                  "| 方法 | 共同样本 | 方向命中率 | 上涨概率误差（越低越好） | 平均涨跌误差 | 80%范围覆盖率 |",
                  "|---|---:|---:|---:|---:|---:|"]
        values = {key: value["metrics"] for key, value in item.get("candidates", {}).items()}
        values.update(item.get("baselines", {}))
        for candidate_id, value in values.items():
            def pct(key: str) -> str:
                return f"{value[key]:.2%}" if value.get(key) is not None else "—"
            brier = f"{value['up_brier']:.4f}" if value.get("up_brier") is not None else "—"
            lines.append(f"| {LABELS.get(candidate_id, candidate_id)} | {value.get('count', 0)} | {pct('direction_accuracy')} | {brier} | {pct('mae')} | {pct('interval_coverage')} |")
        lines += ["", f"各候选均真正运行而非经验回退的共同样本：{item.get('all_fitted_common', {}).get('count', 0)}；滚动预测会重叠，不能把行数当独立试验。", ""]
    lines += ["## 费用和延迟情景", "", "申购0/15bp、赎回0/50/150bp，入场延迟2/3/4笔，均是假设而非支付宝已核实规则。",
              "每条路径使用同一现金/份额/在途账本，预测收益没有逐日连乘。成本后完整结果在 scenarios.json。", "",
              "## 对你意味着什么", "", "若方法没有稳定超过历史频率和一直猜涨，不能把一个较高命中率理解成值得据此买卖。",
              "即使历史指标改善，也须在封存协议后用真正新增的前向资料检验，再决定是否进入日常模型。", ""]
    available = [row for row in scenarios["scenarios"] if row["status"] == "available"]
    lines.append(f"本次可计算成本情景：{len(available)}；这些共享同段行情，不是独立试验。")
    return "\n".join(lines) + "\n"


def _run_one(frozen: Mapping[str, Any], raw: bytes, out: Path, *, protocol: Mapping[str, Any], factor: Mapping[str, Any] | None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs.json").write_bytes(raw)
    _json_write(out / "protocol.json", protocol)
    series, product = frozen["series"], frozen["product"]
    ridge = fit_forecast(series, product_id=product["id"], generated_at=now_iso(), entry_lags=(2, 3, 4), include_evaluation=True)
    bundles = {MODEL_ID: {"candidate_id": MODEL_ID, "candidate_hash": MODEL_HASH,
                          "definition": MODEL_DEFINITION, "evaluation_rows": ridge["evaluation_rows"],
                          "issued_forecasts": ridge["issued_forecasts"], "unavailable_forecasts": ridge["unavailable_forecasts"]}}
    for candidate_id in (TREND_ID, TREE_ID):
        if candidate_id in protocol["candidate_ids"]:
            bundles[candidate_id] = issue_candidate_forecasts(series, candidate_id=candidate_id, heads=DEFAULT_HEADS)
    if factor is not None:
        from .forecast_candidates import issue_factor_forecasts
        from .forecast_factors import align_vix_features

        bundles[VIX_ID] = issue_factor_forecasts(series, factor_features=align_vix_features(series, factor["rows"]),
                                               factor_definition=factor["definition"], heads=DEFAULT_HEADS)
    predictions = {"schema": "rmbfund.forecast-comparison.predictions.v1", "product_id": product["id"],
                   "input_hash": frozen["input_hash"], "factor_input_hash": factor["input_hash"] if factor else None,
                   "candidates": bundles, **EVIDENCE}
    metrics = evaluate_candidate_rows(bundles)
    scenarios = compare_action_scenarios(series, bundles)
    report = {"schema": "rmbfund.forecast-comparison.report.v1", "product": product,
              "input_hash": frozen["input_hash"], "factor_input_hash": factor["input_hash"] if factor else None,
              "availability": audit_availability(frozen), "headlines": _headline(metrics),
              "winner": None, "promotion": False, "conclusion": "historical_comparison_only",
              "strict_forward_outcome_count": 0, **EVIDENCE}
    for name, value in (("predictions.json", predictions), ("metrics.json", metrics), ("scenarios.json", scenarios), ("report.json", report)):
        _json_write(out / name, value)
    (out / "report.md").write_text(_markdown(product, metrics, scenarios), encoding="utf-8")
    return {"product_id": product["id"], "symbol": product.get("symbol"), "output_directory": out.name,
            "input_hash": frozen["input_hash"], "observation_count": len(series),
            "headlines": _headline(metrics), "candidate_ids": list(bundles)}


def _empirical_bundle(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Original same-fit same-head baseline, including immature issued tails."""
    history = {_identity(row): row for row in reference["historical_forecasts"]}
    issued = []
    for row in reference["issued_forecasts"]:
        baseline = history[_identity(row)].get("historical_baseline")
        prediction = dict(baseline) if baseline is not None else {
            "horizon": row["horizon"], "entry_lag": row["entry_lag"],
            "status": "unavailable", "reason": "missing_same_fit_empirical_baseline",
        }
        issued.append({**row, "prediction": prediction})
    return {
        "candidate_id": "historical_frequency",
        "evaluation_rows": [{**row, "prediction": row["historical_baseline"]}
                            for row in reference["evaluation_rows"]],
        "issued_forecasts": issued,
    }


def run_risk_distribution_comparison(
    protocol_path: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Run the approved single risk candidate from existing offline inputs only.

    The protocol must already exist before this call. It is not reselected from
    results; no database, network, frozen input copies or file-hash guards are
    added. Each completed stage and every failed fund/stage stays in output.
    """
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    heads = tuple(tuple(head) for head in protocol["heads"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "rmbfund.risk-distribution.report.v1", "status": "running",
        "protocol_path": str(Path(protocol_path).resolve()), "started_at": now_iso(),
        "dependencies": _versions(research_dependency=False),
        "primary_head": protocol["head_priority"]["primary"],
        "candidate_definition": candidate_definition(RISK_ID),
        "funds": [], "failures": [], "winner": None, "promotion": False,
        "network": False, "database_access": False, **EVIDENCE,
    }
    _json_write(out / "report.json", summary)
    for index, source in enumerate(protocol["inputs"]):
        fund_out = out / ("primary" if index == 0 else f"auxiliary-{index}")
        fund_out.mkdir()
        stage = "read_existing_offline_input"
        try:
            frozen = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
            series, product = frozen["series"], frozen["product"]
            stage = "old_ridge_kernel"
            reference = issue_ridge_distribution_forecasts(series, candidate_id=MODEL_ID, heads=heads)
            _json_write(fund_out / "old-ridge.json", reference)
            stage = "risk_candidate_kernel"
            risk = issue_risk_forecasts(series, heads=heads)
            _json_write(fund_out / "risk-candidate.json", risk)
            bundles = {MODEL_ID: reference, RISK_ID: risk}
            stage = "matched_metrics"
            metrics = evaluate_candidate_rows(bundles, heads=heads)
            _json_write(fund_out / "metrics.json", metrics)
            stage = "paired_block_bootstrap"
            bootstrap = compare_paired_losses(
                bundles, heads=heads, replicates=protocol["bootstrap"]["replicates"],
                seed=protocol["bootstrap"]["seed"],
            )
            _json_write(fund_out / "bootstrap.json", bootstrap)
            stage = "quant_rule_ledger_scenarios"
            empirical = _empirical_bundle(reference)
            _json_write(fund_out / "empirical-predictions.json", empirical)
            scenarios = compare_action_scenarios(
                series, {**bundles, "historical_frequency": empirical},
                entry_lags=(3,), include_cost_diagnostics=True,
            )
            _json_write(fund_out / "scenarios.json", scenarios)
            old_rows = {_identity(row): row for row in reference["historical_forecasts"]}
            pairs = [(row["prediction"]["ridge_log_location"], old_rows[_identity(row)]["prediction"]["ridge_log_location"])
                     for row in risk["historical_forecasts"]
                     if row["prediction"].get("ridge_log_location") is not None
                     and old_rows[_identity(row)]["prediction"].get("ridge_log_location") is not None]
            primary = tuple(protocol["head_priority"]["primary"])
            report = {
                "product": product, "input_path": source["path"], "output_directory": str(fund_out),
                "availability": audit_availability(frozen),
                "head_priority": protocol["head_priority"], "coverage": metrics["coverage"],
                "primary_periods": next(head["periods"] for head in metrics["heads"]
                                        if (head["horizon"], head["entry_lag"]) == primary),
                "log_location_invariance": {
                    "compared_origins": len(pairs), "exact_equal_count": sum(a == b for a, b in pairs),
                    "max_absolute_difference": max((abs(a - b) for a, b in pairs), default=None),
                    "quantity": "fixed Ridge log location only; simple-return means need not match",
                },
                "action_summaries": [{key: value for key, value in scenario.items()
                                      if key not in {"replay", "same_signals_zero_fee_replay", "actions"}}
                                     for scenario in scenarios["scenarios"]],
                "conclusion": "single_candidate_historical_comparison_only; interpret metrics jointly, never promote automatically",
                "winner": None, "promotion": False, "skill_certified": False, **EVIDENCE,
            }
            _json_write(fund_out / "report.json", report)
            summary["funds"].append({"product_id": product["id"], "symbol": product.get("symbol"),
                                     "output_directory": str(fund_out), "status": "completed",
                                     "coverage": metrics["coverage"],
                                     "log_location_invariance": report["log_location_invariance"]})
        except Exception as exc:
            failure = {"input_path": source["path"], "stage": stage,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()}
            _json_write(fund_out / "failure.json", failure)
            summary["failures"].append(failure)
        _json_write(out / "report.json", summary)
    summary.update(status="completed" if not summary["failures"] else "completed_with_failures",
                   finished_at=now_iso())
    _json_write(out / "report.json", summary)
    return summary


def run_forecast_comparison(
    primary_frozen: str | Path, aux_frozen: Sequence[str | Path], output: str | Path,
    include_tree: bool = False, *, product_id: str | None = None,
    protocol_path: str | Path | None = None, vix_frozen: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a protocol, then compute independent fund comparisons in a new directory."""
    loaded = [load_frozen_inputs(primary_frozen, product_id=product_id)]
    loaded += [load_frozen_inputs(path) for path in aux_frozen]
    ids = [frozen["product"]["id"] for frozen, _ in loaded]
    if len(ids) != len(set(ids)):
        raise ValidationError("主样本和辅助样本产品重复")
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise ValidationError("研究输出目录已存在；不会覆盖")
    factor = None
    if vix_frozen is not None:
        from .forecast_factors import load_vix_factors

        factor = load_vix_factors(vix_frozen)
    if include_tree or factor is not None:
        require_tree()
    supplied, _ = _read_json(protocol_path, limit=2 * 1024 * 1024) if protocol_path else (None, None)
    identities = [{"product_id": frozen["product"]["id"], "input_hash": frozen["input_hash"],
                   "file_sha256": hashlib.sha256(raw).hexdigest()} for frozen, raw in loaded]
    protocol = build_comparison_protocol(include_tree=include_tree, inputs=identities, factor=factor, protocol=supplied)
    out.mkdir(parents=True, exist_ok=False)
    _json_write(out / "protocol.json", protocol)
    if vix_frozen is not None:
        (out / "vix-inputs.json").write_bytes(Path(vix_frozen).read_bytes())
    funds = []
    for index, (frozen, raw) in enumerate(loaded):
        # A data field is never used as a path: symbolic IDs stay inside JSON.
        funds.append(_run_one(frozen, raw, out / ("primary" if index == 0 else f"auxiliary-{index}"), protocol=protocol, factor=factor))
    summary = {"schema": "rmbfund.forecast-comparison.report.v1", "status": "ready", "network": False,
               "database_writes": False, "protocol_hash": protocol["protocol_hash"], "include_tree": include_tree,
               "factor_input_hash": factor["input_hash"] if factor else None,
               "funds": funds, "winner": None, "promotion": False, **EVIDENCE}
    _json_write(out / "report.json", summary)
    root_text = "# 离线量化候选比较\n\n各基金独立训练和报告，不池化，不把相关公开样本合并为独立证据。\n\n"
    root_text += "\n".join(f"- {fund['product_id']}：[{fund['output_directory']}/report.md]({fund['output_directory']}/report.md)" for fund in funds)
    (out / "report.md").write_text(root_text + "\n\n历史探索不能自动晋级为买卖模型。\n", encoding="utf-8")
    manifest = {"schema": "rmbfund.forecast-comparison.manifest.v1", "protocol_hash": protocol["protocol_hash"],
                "runtime_generated_at": now_iso(), "dependencies": protocol["dependencies"],
                "inputs": identities, "factor_input": protocol["factor_input"],
                "files": {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(out.rglob("*")) if path.is_file()}, **EVIDENCE}
    _json_write(out / "manifest.json", manifest)
    return {"status": "ready", "output": str(out.resolve()), "protocol_hash": protocol["protocol_hash"],
            "funds": [{key: value for key, value in fund.items() if key != "headlines"} for fund in funds],
            "network": False, "database_writes": False, "winner": None, **EVIDENCE}

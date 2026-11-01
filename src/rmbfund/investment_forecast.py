"""Pure factor-augmented ridge issuance; fitted associations, not holding replication."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from .domain import ValidationError, timestamp
from .forecast import (
    FEATURE_DEFINITIONS as NAV_FEATURES,
    LOCAL_TIMING_CONTRACT,
    MODEL_DEFINITION as NAV_MODEL_DEFINITION,
    RUNTIME_TARGET_DEFINITION,
    _canonical,
    _digest,
    _entry_lags,
    _feature_matrix,
    _fit_head,
    _horizons,
    _local_target_metadata,
    _normalise_series,
    _runtime_currency,
)
from .forecast_research import evaluate_forecasts
from .investment_factors import (
    ALIGNMENT,
    DATA_CONTRACT_VERSION,
    FEATURE_DEFINITIONS as EXTERNAL_FEATURES,
    align_investment_features,
    normalize_factor_bundle,
)

MODEL_ID = "exposure_factor_ridge_empirical_v1"
_NAV_LABELS = ("净值近1笔涨跌", "净值近5笔涨跌", "净值近20笔涨跌", "净值近60笔涨跌",
               "净值近20笔波动", "净值近60笔波动", "净值近60笔回撤")
FEATURE_DEFINITIONS = tuple(
    dict(item, label=label, economic_role="nav_history", series_ids=["raw_unit_nav"])
    for item, label in zip(NAV_FEATURES, _NAV_LABELS, strict=True)
) + EXTERNAL_FEATURES
FEATURE_DEFINITION_HASH = _digest(FEATURE_DEFINITIONS)
DRIVER_INTERPRETATION = (
    "Each standardized feature times its head's fitted ridge coefficient contributes to log "
    "location; intercept + contributions = ridge_log_location, NOT expected simple return. "
    "Predictive simple returns are expm1(log location + matured prequential errors). "
    "Correlated-factor contributions are associations, not causal effects, weights or manager alpha."
)
MODEL_DEFINITION = {
    **{key: NAV_MODEL_DEFINITION[key] for key in (
        "feature_lookback", "ridge_alpha", "standardization", "feature_std_tolerance", "intercept",
        "minimum_training_samples", "minimum_residual_samples", "minimum_baseline_samples",
        "residuals", "predictive_samples", "interval", "probabilities", "event_policy", "outlier_policy",
        "development_period", "chronological_heldout_period", "strict_unseen", "hyperparameter_search",
    )},
    "model_id": MODEL_ID, "schema_version": 2, "data_contract_version": DATA_CONTRACT_VERSION,
    "features": list(FEATURE_DEFINITIONS), "feature_definition_hash": FEATURE_DEFINITION_HASH,
    "horizons": [1, 5, 20], "execution_horizons": [1, 5, 20],
    "target_definition": RUNTIME_TARGET_DEFINITION, "alignment": ALIGNMENT,
    "fallback": "none: unfitted/uncalibrated heads explicitly unavailable, never relabeled empirical",
    "zero_lag": "execution aliases corresponding anchor object; one fit per (horizon,entry_lag)",
    "profile_mapping": "current dated/ex-post equity+IT mandate/exposure match selects fixed proxies; "
    "not historical weights, feature multipliers, manager-alpha scores or exact holdings",
    "factor_basis": "EEM/SOXX USD price changes already include their quoted currency basis; "
    "fixing is an independent predictor, never an FX overlay on CNY NAV; US2Y retained but not fitted",
    "drivers": DRIVER_INTERPRETATION,
    "evaluation": "own expanding prequential fitted-only origins vs matched empirical/no-change "
    "baselines; date-lag reconstruction, no historical publication vintages or trading-profit claim",
}
MODEL_HASH = _digest(MODEL_DEFINITION)
LOCAL_MODEL_DEFINITION = {
    **MODEL_DEFINITION,
    "local_contract": LOCAL_TIMING_CONTRACT,
    "official_model_hash": MODEL_HASH,
    "horizons": "explicit requested positive NAV-observation horizon",
    "execution_horizons": "the same requested horizon at each selected entry lag",
    "target_unit": "nav_observations",
    "calendar_mapping": "conditional business-day schedule is separate from numerical inputs",
    "actual_issue": False,
    "persisted": False,
}
LOCAL_MODEL_HASH = _digest(LOCAL_MODEL_DEFINITION)


def _profile_mapping(profile: Mapping[str, Any], product_id: str, generated_at: str) -> dict:
    if not isinstance(profile, Mapping) or profile.get("product_id") != product_id:
        raise ValidationError("investment forecast requires a matching dated product profile")
    point = timestamp(generated_at)
    if timestamp(profile.get("as_of")) > point:
        raise ValidationError("investment profile as_of exceeds issuance cutoff")
    portfolio, mandate = profile.get("portfolio"), profile.get("mandate")
    if not isinstance(portfolio, Mapping) or not isinstance(mandate, Mapping):
        raise ValidationError("investment factor mapping requires mandate and actual portfolio disclosure")
    try:
        payload, contract = portfolio["payload"], mandate["payload"]
        equity = float(payload["assets"]["equity"])
        it = [row for row in payload["sectors"] if any(term in row["name"].lower()
              for term in ("信息技术", "半导体", "information technology", "semiconductor"))
              and math.isfinite(float(row["weight"])) and float(row["weight"]) > 0]
        matched = (payload["denominator"] == "nav" and math.isfinite(equity) and 0 < equity <= 1
                   and it and contract["asset_class"] in ("equity", "mixed")
                   and contract["benchmark"]["region"] == "emerging_markets"
                   and contract["base_currency"] == "CNY")
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ValidationError("invalid investment mandate/exposure mapping") from exc
    if not matched:
        raise ValidationError("investment profile does not match emerging-market equity/IT proxies")
    sections = {}
    for name, section in (("portfolio", portfolio), ("mandate", mandate)):
        try:
            report_date = date.fromisoformat(section.get("report_date")).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValidationError("invalid investment mapping report date") from exc
        if (not section.get("record_id") or report_date != section["report_date"]
                or section["report_date"] > point.date().isoformat()
                or timestamp(section.get("available_at")) > point):
            raise ValidationError("investment mapping disclosure is unavailable at issuance")
        sections[name] = {key: section.get(key) for key in (
            "record_id", "report_date", "available_at", "observed_at", "source", "source_url", "payload"
        )}
    return json.loads(_canonical({
        "product_id": product_id, "schema_version": profile.get("schema_version"),
        "as_of": profile["as_of"], **sections, "factor_mapping": profile.get("factor_mapping", []),
        "selected_roles": ["market", "industry", "fx", "yield_10y"],
        "basis": "current_ex_post_mapping_not_historical_portfolio_weights",
        "weights_used_as_numeric_features": False,
    }))


def profile_mapping_input_hash(
    profile: Mapping[str, Any], *, product_id: str, generated_at: str,
) -> str:
    """Stable selected mapping identity; a refreshed as_of is not a new disclosure."""
    mapping = _profile_mapping(profile, product_id, generated_at)
    return _digest({key: value for key, value in mapping.items() if key != "as_of"})


def fit_investment_forecast(
    series: Sequence[Mapping[str, Any]], *, factors: Mapping[str, Any], profile: Mapping[str, Any],
    product_id: str, generated_at: str, entry_lags: Sequence[int] = (2, 3, 4),
    include_evaluation: bool = False,
) -> dict[str, Any]:
    """Fit all independent v2 heads with 7 NAV + 7 qualified external predictors.

    Actual acquisition is gated at issuance, not retrospectively at economic
    origins. Historical features use the declared two-calendar-day assumption.
    Missing training features remain NaN without removing/reindexing NAV dates.
    Each available head must really have fitted ridge and matured OOS errors.
    """
    return _fit_investment_forecast_core(
        series, factors=factors, profile=profile, product_id=product_id,
        generated_at=generated_at, entry_lags=entry_lags,
        include_evaluation=include_evaluation,
    )


def fit_local_investment_forecast(
    series: Sequence[Mapping[str, Any]], *, factors: Mapping[str, Any], profile: Mapping[str, Any],
    product_id: str, generated_at: str, horizons: Sequence[int],
    entry_lags: Sequence[int] = (2, 3, 4), include_history: bool = False,
) -> dict[str, Any]:
    """Explicit RAM-only exact-H factor heads; original 14-feature qualification.

    The date schedule does not enter features, change maturity or excuse a NAV
    fallback. With history requested, qualified but immature heads remain visible
    as unavailable at every origin, including the complete immature target tail.
    """
    if not isinstance(include_history, bool):
        raise ValidationError("include_history must be boolean")
    return _fit_investment_forecast_core(
        series, factors=factors, profile=profile, product_id=product_id,
        generated_at=generated_at, entry_lags=entry_lags,
        include_evaluation=False, horizons=_horizons(horizons), include_history=include_history,
    )


def _fit_investment_forecast_core(
    series: Sequence[Mapping[str, Any]], *, factors: Mapping[str, Any], profile: Mapping[str, Any],
    product_id: str, generated_at: str, entry_lags: Sequence[int], include_evaluation: bool,
    horizons: Sequence[int] | None = None, include_history: bool = False,
) -> dict[str, Any]:
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValidationError("forecast product_id must be nonempty")
    if not isinstance(generated_at, str):
        raise ValidationError("forecast generated_at must be an explicit timestamp string")
    issued_at = timestamp(generated_at)
    rows, lags = _normalise_series(series), _entry_lags(entry_lags)
    mapping = _profile_mapping(profile, product_id, generated_at)
    mapping_hash = _digest({key: value for key, value in mapping.items() if key != "as_of"})
    selected = normalize_factor_bundle(factors, as_of=generated_at)
    for row in rows:
        if row["date"] > issued_at.date().isoformat():
            raise ValidationError("NAV economic date exceeds issuance cutoff")
        provenance = row.get("provenance")
        for source in (row, provenance if isinstance(provenance, Mapping) else {}):
            if source.get("available_at") and timestamp(source["available_at"]) > issued_at:
                raise ValidationError("NAV observation is unavailable at issuance")
    external, _, factor_provenance = align_investment_features(rows, selected)
    days = [row["date"] for row in rows]
    log_nav = np.log(np.asarray([float(row["nav"]) for row in rows]))
    invalid = np.asarray([row.get("invalid_return", False) or row.get("label_valid") is False
                          for row in rows], dtype=bool)
    features = np.column_stack((_feature_matrix(log_nav, invalid), external))
    selected_horizons = (1, 5, 20) if horizons is None else _horizons(horizons)
    execution_heads = [(h, lag) for lag in lags for h in selected_horizons]
    results, unavailable, states, all_drivers, evaluation, issued = {}, [], [], [], [], []
    chronological = [] if include_history else None
    definitions = {item["name"]: item for item in FEATURE_DEFINITIONS}
    for horizon, lag in dict.fromkeys([(h, 0) for h in selected_horizons] + execution_heads):
        head, history, state, drivers, issuance = _fit_head(
            days, log_nav, features, invalid, horizon=horizon, entry_lag=lag,
            capture_baseline=True, collect_issued=include_evaluation,
            feature_definitions=FEATURE_DEFINITIONS,
            historical_forecasts=chronological, fitted_history_only=True,
        )
        # Empirical historical targets remain a comparator only, never a factor forecast.
        available = head is not None and state is not None and head["method"] == "ridge_prequential_empirical"
        results[horizon, lag] = head if available else None
        if not available:
            unavailable.append({
                "horizon": horizon, "entry_lag": lag, "status": "unavailable",
                "reason": head.get("fallback_reason") if head else "insufficient_matured_labels",
                "required_method": "ridge_prequential_empirical", "fallback_used": False,
            })
        else:
            for driver in drivers:
                name = driver["feature"]
                driver.update({key: definitions[name][key] for key in (
                    "label", "unit", "description", "economic_role", "series_ids"
                )})
                driver.update(factor_provenance["current_features"].get(name, {
                    "observation_ends": {"raw_unit_nav": days[-1]},
                    "availability_basis": "supplied_NAV_history_actual_issuance_cutoff",
                }))
                driver["series_ids"] = list(driver["series_ids"])
                driver["current_feature_value"] = driver["value"]
                driver["contribution_interpretation"] = "additive_to_ridge_log_location_not_simple_return"
            head["drivers"] = drivers
            for summary in (head, head["historical_baseline"]):
                summary["target_definition_id"] = RUNTIME_TARGET_DEFINITION["id"]
                if horizons is not None:
                    _local_target_metadata(summary)
                for kind in ("training", "residual"):
                    endpoint = summary[f"{kind}_target_end_index"]
                    summary[f"{kind}_target_end_date"] = days[endpoint] if endpoint is not None else None
            state["feature_definition_hash"] = FEATURE_DEFINITION_HASH
            states.append(state)
            all_drivers.extend(drivers)
        evaluation.extend(row for row in history
                          if row["prediction"]["method"] == "ridge_prequential_empirical")
        if include_evaluation:
            issued.extend(row for row in issuance
                          if row["prediction"]["method"] == "ridge_prequential_empirical")
    forecasts = [results[h, 0] for h in selected_horizons if results[h, 0] is not None]
    execution = [results[key] for key in execution_heads if results[key] is not None]
    if not forecasts and not execution and not include_history:
        raise ValidationError("insufficient_history: no genuinely fitted/calibrated investment factor head")
    validation = evaluate_forecasts(evaluation)
    validation.update(model_id=MODEL_ID, profit_claim=False,
                      factor_availability_assumption=dict(ALIGNMENT),
                      comparison_basis="fitted-only factor prequential origins; same-origin baselines")
    moves = [{"date": days[i + 1], "log_return": float(change), "known_invalid": bool(invalid[i + 1])}
             for i, change in enumerate(np.diff(log_nav)) if abs(change) > 0.25]
    bundle = {
        "schema_version": 2, "model_id": MODEL_ID, "model_hash": MODEL_HASH,
        "model_definition": json.loads(_canonical(MODEL_DEFINITION)),
        "input_hash": _digest({"nav": rows, "factor_input_hash": factor_provenance["factor_input_hash"],
                               "profile_mapping_hash": mapping_hash, "feature_definition_hash": FEATURE_DEFINITION_HASH}),
        "product_id": product_id, "generated_at": generated_at,
        "anchor": {"date": days[-1], "nav": rows[-1]["nav"], **_runtime_currency(rows[-1])},
        "target_definition": json.loads(_canonical(RUNTIME_TARGET_DEFINITION)),
        "observation_count": len(rows), "training_start": days[0], "training_end": days[-1],
        "training_window_basis": "all supplied NAV dates retained; matured finite-feature rows per head",
        "forecasts": forecasts, "execution_forecasts": execution, "unavailable_forecasts": unavailable,
        "entry_lags": list(lags), "validation": validation,
        "features": [dict(item, series_ids=list(item["series_ids"]), value=float(features[-1, i]))
                     for i, item in enumerate(FEATURE_DEFINITIONS)
                     if np.isfinite(features[-1, i])],
        "drivers": all_drivers, "model_state": states,
        "factor_inputs": selected, "factor_input_hash": factor_provenance["factor_input_hash"],
        "factor_provenance": factor_provenance, "profile_mapping": mapping,
        "profile_mapping_hash": mapping_hash, "feature_definition_hash": FEATURE_DEFINITION_HASH,
        "data_quality": {"invalid_return_dates": [days[i] for i in np.flatnonzero(invalid)],
                         "large_nav_moves": moves, "warnings": [
            "Raw unit NAV and publisher USD closes are not verified reinvested total returns.",
            "Historical factor publication times/vintages are unknown; reconstruction is not system PIT.",
            "Current dated profile selects proxies ex post, not historical portfolio weights or manager alpha.",
            "FX is central parity, not spot; no FX overlay or exact-holdings return decomposition.",
            "Observation horizons are not guaranteed trading days; rolling targets overlap.",
            "Fitted association and forecast diagnostics do not establish valid trading or profit.",
        ] + (["Unmarked large NAV moves require event review; no automatic adjustment."]
             if any(not move["known_invalid"] for move in moves) else [])},
        "provenance": {"numerical_library": f"numpy {np.__version__}",
                       "actual_availability_cutoff": generated_at, "fact_timestamps_rewritten": False,
                       "historical_first_seen_before_origin_required": False,
                       "profile_mapping_basis": mapping["basis"],
                       "driver_interpretation": DRIVER_INTERPRETATION},
    }
    if horizons is not None:
        bundle.update(
            local_contract=LOCAL_TIMING_CONTRACT, local_contract_hash=LOCAL_MODEL_HASH,
            target_unit="nav_observations", requested_horizons=list(selected_horizons),
            algorithm_model_hash=MODEL_HASH, model_hash=LOCAL_MODEL_HASH,
            model_definition=json.loads(_canonical(LOCAL_MODEL_DEFINITION)),
        )
        for head in unavailable:
            _local_target_metadata(head)
        for row in chronological or []:
            _local_target_metadata(row)
            for head in (row["prediction"], row.get("historical_baseline")):
                if head is not None:
                    _local_target_metadata(head)
    if include_history:
        bundle["historical_forecasts"] = sorted(
            chronological, key=lambda r: (r["origin_index"], r["horizon"], r["entry_lag"])
        )
    if include_evaluation:
        bundle["evaluation_rows"] = sorted(evaluation, key=lambda r: (r["origin_index"], r["horizon"], r["entry_lag"]))
        bundle["issued_forecasts"] = sorted(issued, key=lambda r: (r["origin_index"], r["horizon"], r["entry_lag"]))
    _canonical(bundle)
    return bundle

"""Fixed offline candidates; not a runtime model catalogue or a promotion rule.

As in the original NAV core, chronology here is in supplied NAV observations.
Actual historical publication times are not inferred from valuation dates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np

from .domain import ValidationError
from .forecast import (
    FEATURE_DEFINITIONS,
    FEATURE_LOOKBACK,
    MIN_BASELINE_SAMPLES,
    MIN_RESIDUAL_SAMPLES,
    MIN_TRAINING_SAMPLES,
    MODEL_DEFINITION,
    MODEL_HASH,
    MODEL_ID,
    _canonical,
    _fit_head,
    _digest,
    _distribution,
    _feature_matrix,
    _historical_row,
    _normalise_series,
)

TREND_ID = "nav_trend60_empirical_v1"
TREE_ID = "nav_histgb_empirical_v1"
FACTOR_ID = "nav_vix_ridge_empirical_v1"
RISK_ID = "nav_ridge_vol20_empirical_v1"
RISK_METHOD = "ridge_vol20_prequential_empirical"
RISK_HEADS = ((20, 3), (20, 0), (1, 0), (5, 0))
SHRINK_ID = "nav_ridge_shrink50_empirical_v1"
SHRINK_METHOD = "ridge_shrink50_prequential_empirical"
SHRINK_HEADS = ((20, 3), (20, 4))
RECOVERY_ID = "nav_ridge_recovery_shrink50_empirical_v1"
RECOVERY_METHOD = "ridge_recovery_shrink50_prequential_empirical"
RECOVERY_FEATURE_DEFINITIONS = (
    *FEATURE_DEFINITIONS,
    {
        "name": "recovery_feature",
        "unit": "log_return_times_return_ratio",
        "description": "max(existing log_return_5, 0) * max(-existing simple drawdown_60, 0)",
    },
)
DEFAULT_HEADS = ((1, 0), (5, 0), (20, 0), (20, 2), (20, 3), (20, 4))
FACTOR_FEATURE_NAMES = ("loglevel", "logchange5")
FACTOR_PARAMETERS = {
    "alpha": 10.0,
    "fit_intercept": True,
}
TREE_PARAMETERS = {
    "loss": "squared_error",
    "max_depth": 2,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 30,
    "max_iter": 100,
    "learning_rate": 0.05,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 0,
    "categorical_features": None,
    "max_bins": 255,
    "warm_start": False,
}


def candidate_definition(candidate_id: str) -> dict[str, Any]:
    if candidate_id == RECOVERY_ID:
        definition = candidate_definition(SHRINK_ID)
        location_definition = json.loads(_canonical(MODEL_DEFINITION))
        location_definition.update(
            model_id="nav_ridge_recovery_v1",
            features=list(RECOVERY_FEATURE_DEFINITIONS),
            coefficient_constraints="none; the recovery coefficient is not forced positive",
        )
        definition.update(
            candidate_id=RECOVERY_ID,
            method=RECOVERY_METHOD,
            location_model_id=location_definition["model_id"],
            location_model_hash=_digest(location_definition),
            location_definition=location_definition,
            location_changes="own NAV8 Ridge fit, then fixed shrinkage of its conditional log contribution",
            log_location="mean_y + 0.5 * (unshrunk_recovery_log_location - mean_y)",
            conditional_log_contribution="unshrunk_recovery_log_location - training_log_mean; unshrunk",
            recovery_feature="max(existing log_return_5, 0) * max(-existing simple drawdown_60, 0)",
            feature_policy="preserve NAV7, NaNs, incoming-event exclusions and the current-plus-60 NAV maximum slice",
            training_log_mean="same-origin, exact-H/L, matured finite NAV8 Ridge training target mean; not the empirical baseline mean",
            matured_nonzero_recovery_training_count="count j with j+H+L <= origin, all features_j finite, x8_j > 0, and no invalid incoming move in entry+1..endpoint",
        )
        return json.loads(_canonical(definition))
    if candidate_id == SHRINK_ID:
        return json.loads(_canonical({
            "candidate_id": SHRINK_ID,
            "method": SHRINK_METHOD,
            "location_model_id": MODEL_ID,
            "location_model_hash": MODEL_HASH,
            "location_definition": MODEL_DEFINITION,
            "location_changes": "fixed shrinkage of the conditional log contribution only",
            "shrink_weight": 0.5,
            "log_location": "mean_y + 0.5 * (old_ridge_log_location - mean_y)",
            "training_log_mean": "same-origin, exact-H/L, matured feature-qualified Ridge training target mean; not the empirical baseline mean",
            "conditional_log_contribution": "old_ridge_log_location - training_log_mean; unshrunk",
            "minimum_training_samples": MIN_TRAINING_SAMPLES,
            "minimum_residual_samples": MIN_RESIDUAL_SAMPLES,
            "residuals": "own raw prequential y_i - stored shrunk_mu_i; exact H/L; admit only at origin_i + H + L",
            "predictive_samples": "equal-weight expm1(current shrunk log location + every matured own raw log residual)",
            "residual_weighting": "equal; all matured same-head errors; no fitted window or vol20 scaling",
            "summary": "mean, probabilities, median and linear 0.1/0.9 quantiles from the same simple-return samples",
            "fallback": "none: explicit unavailable without fitted location, valid current features, or 40 own matured residuals; historical baseline remains separate",
            "assumed_nav_availability_lag_observations": 1,
            "hyperparameter_search": False,
            "strict_unseen": False,
            "point_in_time_verified": False,
            "runtime_issuance": False,
            "promotion": False,
        }))
    if candidate_id == RISK_ID:
        return json.loads(_canonical({
            "candidate_id": RISK_ID,
            "method": RISK_METHOD,
            "location_model_id": MODEL_ID,
            "location_model_hash": MODEL_HASH,
            "location_definition": MODEL_DEFINITION,
            "location_changes": False,
            "preserved_quantity": "fixed Ridge log location, not simple-return mean",
            "risk_scale": "population std of last 20 known one-observation log returns * sqrt(exact H)",
            "risk_scale_lookback": 20,
            "risk_scale_ddof": 0,
            "residuals": "same exact H/L prequential z_i=(y_i-mu_i)/stored s_i; admit only when full original target matures",
            "predictive_samples": "equal-weight expm1(mu_now+s_now*z_i) for every matured usable same-head z_i",
            "minimum_residual_samples": MIN_RESIDUAL_SAMPLES,
            "summary": "mean, probabilities, median and linear 0.1/0.9 quantiles from the same simple-return samples",
            "fallback": "none: unavailable on absent mu, nonpositive/invalid scale, or insufficient usable z; historical baseline remains separate",
            "invalid_scale": "no epsilon, filling, zero substitution, neighboring head or silent fallback",
            "residual_weighting": "equal; all matured usable z, no fitted window",
            "hyperparameter_search": False,
            "strict_unseen": False,
            "point_in_time_verified": False,
            "runtime_issuance": False,
            "promotion": False,
        }))
    if candidate_id not in (TREND_ID, TREE_ID):
        raise ValidationError("unknown offline forecast candidate")
    definition = {
        "candidate_id": candidate_id,
        "target": "raw log NAV[i+entry_lag+horizon] - log NAV[i+entry_lag]",
        "feature_lookback": FEATURE_LOOKBACK,
        "minimum_baseline_samples": MIN_BASELINE_SAMPLES,
        "minimum_residual_samples": MIN_RESIDUAL_SAMPLES,
        "minimum_training_samples": MIN_TRAINING_SAMPLES if candidate_id == TREE_ID else 0,
        "residuals": "candidate's own raw prequential log errors, admitted at full target maturity",
        "predictive_samples": "expm1(current log location + all matured candidate residuals)",
        "fallback": "empirical historical matured target log returns; never training residuals",
        "event_policy": "exclude labels crossing invalid_return=True or label_valid=False; "
        "exclude trailing 60-return feature windows crossing these incoming moves",
        "interval": "linear 0.1/0.9 simple-return quantiles; nominal coverage 0.8",
        "probabilities": "finite empirical frequencies >0, <0, =0; not certainty claims",
        "research_period": ["2025-01-01", "2026-12-31"],
        "assumed_nav_availability_lag_observations": 1,
        "strict_unseen": False,
        "point_in_time_verified": False,
        "hyperparameter_search": False,
    }
    if candidate_id == TREE_ID:
        definition.update(
            estimator="sklearn.ensemble.HistGradientBoostingRegressor",
            parameters=dict(TREE_PARAMETERS),
            features=list(FEATURE_DEFINITIONS),
            preprocessing="numeric NAV features; no global scaling or random validation split",
            fitting="fresh expanding fit at every eligible origin using matured valid labels only",
            method="histgb_prequential_empirical",
        )
    else:
        definition.update(
            estimator="fixed trailing log-return mean, not a fitted regression",
            location="horizon * (log NAV[i] - log NAV[i-60]) / 60; does not multiply by entry_lag",
            features=["trailing_60_observation_mean_log_return"],
            fitting="no supervised fit; own raw forecasts issued from the first valid feature window",
            method="trend60_prequential_empirical",
        )
    return json.loads(_canonical(definition))


def candidate_hash(candidate_id: str) -> str:
    return _digest(candidate_definition(candidate_id))


def factor_candidate_definition(factor_definition: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(factor_definition, Mapping) or not factor_definition:
        raise ValidationError("factor candidate requires a nonempty vetted factor definition")
    definition = candidate_definition(TREE_ID)
    definition.update(
        candidate_id=FACTOR_ID,
        estimator="sklearn.linear_model.Ridge",
        parameters=dict(FACTOR_PARAMETERS),
        features=[
            *list(FEATURE_DEFINITIONS),
            {"name": "vix_loglevel", "unit": "log_index_level"},
            {"name": "vix_logchange5", "unit": "five_VIX_observation_log_change"},
        ],
        factor_definition=dict(factor_definition),
        preprocessing="expanding matured training rows only; per-feature population mean/std; "
        "std <= 1e-12 -> 1; same fitted transform applied to current row; no missing-value fill",
        fitting="fresh Ridge fit each eligible origin using matured valid labels and finite "
        "as-of factor features; missing current factors trigger empirical fallback",
        method="vix_ridge_prequential_empirical",
    )
    return json.loads(_canonical(definition))


def require_tree() -> None:
    """Fail explicitly without installing or substituting the optional dependency."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
        from threadpoolctl import threadpool_limits  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "浅树研究需要可选 research 依赖 scikit-learn；尚未运行，不会替换为其他算法。"
        ) from exc


def _heads(values: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValidationError("candidate heads must be a nonempty sequence of horizon/lag pairs")
    result = []
    for pair in values:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
            or pair[0] <= 0
            or pair[1] < 0
        ):
            raise ValidationError("candidate heads require positive horizons and nonnegative lags")
        key = tuple(pair)
        if key not in result:
            result.append(key)
    return tuple(result)


def _issue_head(
    days: list[str],
    log_nav: np.ndarray,
    features: np.ndarray,
    invalid: np.ndarray,
    *,
    candidate_id: str,
    horizon: int,
    entry_lag: int,
    estimator_class: Any = None,
    historical_forecasts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    count, span = len(days), horizon + entry_lag
    method = (
        "trend60_prequential_empirical" if candidate_id == TREND_ID
        else "histgb_prequential_empirical" if candidate_id == TREE_ID
        else "vix_ridge_prequential_empirical"
    )
    bad_prefix = np.r_[0, np.cumsum(invalid)]
    targets = np.full(count, np.nan)
    for origin in range(max(0, count - span)):
        entry, endpoint = origin + entry_lag, origin + span
        if bad_prefix[endpoint + 1] == bad_prefix[entry + 1]:
            targets[origin] = log_nav[endpoint] - log_nav[entry]
    historical, historical_origins = [], []
    training_origins, training_targets = [], []
    residuals, residual_endpoints = [], []
    raw_locations = np.full(count, np.nan)
    evaluation, issued = [], []
    final_available = False
    for origin in range(count):
        matured = origin - span
        if matured >= 0 and np.isfinite(targets[matured]):
            target = float(targets[matured])
            historical.append(target)
            historical_origins.append(matured)
            if np.isfinite(raw_locations[matured]):
                residuals.append(target - float(raw_locations[matured]))
                residual_endpoints.append(origin)
            if np.all(np.isfinite(features[matured])):
                training_origins.append(matured)
                training_targets.append(target)

        feature_valid = np.all(np.isfinite(features[origin]))
        location = None
        if feature_valid:
            if candidate_id == TREND_ID:
                location = float(horizon * (log_nav[origin] - log_nav[origin - 60]) / 60)
            elif len(training_origins) >= MIN_TRAINING_SAMPLES:
                training_x = features[training_origins]
                current_x = features[origin : origin + 1]
                if candidate_id == FACTOR_ID:
                    means = np.mean(training_x, axis=0)
                    scales = np.std(training_x, axis=0)
                    scales[scales <= 1e-12] = 1.0
                    training_x = (training_x - means) / scales
                    current_x = (current_x - means) / scales
                    estimator = estimator_class(**FACTOR_PARAMETERS)
                else:
                    estimator = estimator_class(**TREE_PARAMETERS)
                estimator.fit(training_x, np.asarray(training_targets))
                location = float(estimator.predict(current_x)[0])
            if location is not None:
                if not math.isfinite(location):
                    raise ValidationError("candidate log forecast is not finite")
                raw_locations[origin] = location

        final = origin == count - 1
        in_evaluation = (
            origin + span < count
            and np.isfinite(targets[origin])
            and "2025-01-01" <= days[origin] <= "2026-12-31"
        )
        in_issuance = horizon == 20 and (
            "2025-01-01" <= days[origin] <= "2026-12-31"
            or (origin + 1 < count and "2025-01-01" <= days[origin + 1] <= "2026-12-31")
        )
        if historical_forecasts is not None:
            is_fitted = candidate_id in (TREE_ID, FACTOR_ID)
            own_state = {
                "horizon": horizon, "entry_lag": entry_lag, "status": "unavailable",
                "method": method, "fallback_used": False,
                "candidate_log_location": location,
                "training_sample_count": len(training_origins) if is_fitted else 0,
                "residual_sample_count": len(residuals),
                "training_start": days[training_origins[0]] if is_fitted and training_origins else None,
                "training_end": days[training_origins[-1] + span] if is_fitted and training_origins else None,
                "training_target_end_index": training_origins[-1] + span if is_fitted and training_origins else None,
                "residual_target_end_index": residual_endpoints[-1] if residual_endpoints else None,
                "reason": (
                    "missing_asof_factors_or_invalid_NAV_feature_window"
                    if candidate_id == FACTOR_ID and not feature_valid
                    else "feature_window_crosses_invalid_event_or_shorter_than_60"
                    if not feature_valid
                    else "insufficient_matured_training_labels" if location is None
                    else "insufficient_matured_prequential_residuals"
                ),
            }
            if len(historical) < MIN_BASELINE_SAMPLES:
                historical_forecasts.append(_historical_row(days, origin, own_state))
        if len(historical) < MIN_BASELINE_SAMPLES or not (
            final or in_evaluation or in_issuance or historical_forecasts is not None
        ):
            continue
        final_available = final_available or final
        baseline = _distribution(
            np.asarray(historical),
            horizon=horizon,
            entry_lag=entry_lag,
            method="empirical_historical",
        )
        baseline.update(
            training_sample_count=len(historical),
            residual_sample_count=0,
            training_start=days[historical_origins[0] + entry_lag],
            training_end=days[historical_origins[-1] + span],
            training_target_end_index=historical_origins[-1] + span,
            residual_target_end_index=None,
        )
        if location is not None and len(residuals) >= MIN_RESIDUAL_SAMPLES:
            prediction = _distribution(
                location + np.asarray(residuals),
                horizon=horizon,
                entry_lag=entry_lag,
                method=method,
            )
            is_fitted = candidate_id in (TREE_ID, FACTOR_ID)
            prediction.update(
                candidate_log_location=location,
                training_sample_count=len(training_origins) if is_fitted else 0,
                residual_sample_count=len(residuals),
                training_start=days[training_origins[0]] if is_fitted else None,
                training_end=days[training_origins[-1] + span] if is_fitted else None,
                training_target_end_index=training_origins[-1] + span if is_fitted else None,
                residual_target_end_index=residual_endpoints[-1],
            )
        else:
            prediction = dict(baseline)
            prediction["fallback_reason"] = (
                "missing_asof_factors_or_invalid_NAV_feature_window"
                if candidate_id == FACTOR_ID and not feature_valid
                else "feature_window_crosses_invalid_event_or_shorter_than_60"
                if not feature_valid
                else "insufficient_matured_training_labels"
                if location is None
                else "insufficient_matured_prequential_residuals"
            )
        prediction.update(
            candidate_training_sample_count=(
                len(training_origins) if candidate_id in (TREE_ID, FACTOR_ID) else 0
            ),
            candidate_residual_sample_count=len(residuals),
            raw_candidate_log_location=location,
            feature_start=days[origin - FEATURE_LOOKBACK] if feature_valid else None,
            feature_end=days[origin] if feature_valid else None,
        )
        if historical_forecasts is not None:
            history_prediction = prediction if prediction["method"] == method else own_state
            historical_forecasts.append(_historical_row(days, origin, history_prediction, baseline))
        chronology = {
            "horizon": horizon,
            "entry_lag": entry_lag,
            "origin_index": origin,
            "origin_date": days[origin],
            "knowledge_cutoff_index": origin,
            "knowledge_cutoff_date": days[origin],
            "assumed_known_index": origin + 1,
            "assumed_known_date": days[origin + 1] if origin + 1 < count else None,
            "assumed_nav_availability_lag_observations": 1,
        }
        if in_issuance:
            issued.append({**chronology, "prediction": dict(prediction)})
        if in_evaluation:
            evaluation.append(
                {
                    **chronology,
                    "entry_index": origin + entry_lag,
                    "entry_date": days[origin + entry_lag],
                    "target_index": origin + span,
                    "target_date": days[origin + span],
                    "actual_log_return": float(targets[origin]),
                    "actual_change": float(np.expm1(targets[origin])),
                    "period": "development_2025" if days[origin] < "2026-01-01" else "heldout_2026",
                    "prediction": dict(prediction),
                    "historical_baseline": baseline,
                    "no_change_baseline": _distribution(
                        np.zeros(1),
                        horizon=horizon,
                        entry_lag=entry_lag,
                        method="no_change_point_baseline",
                    ),
                }
            )
    return evaluation, issued, final_available


def _inputs(
    rows: Sequence[Mapping[str, Any]], *, recovery: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    days = [row["date"] for row in rows]
    log_nav = np.log(np.asarray([float(row["nav"]) for row in rows]))
    invalid = np.asarray(
        [row.get("invalid_return", False) or row.get("label_valid") is False for row in rows],
        dtype=bool,
    )
    features = _feature_matrix(log_nav, invalid)
    if recovery:
        # Reuse both existing columns exactly, including their missing/event
        # windows and current-plus-60 NAV maximum; no alternate drawdown slice.
        x8 = np.maximum(features[:, 1], 0.0) * np.maximum(-features[:, 6], 0.0)
        features = np.column_stack((features, x8))
    return days, log_nav, features, invalid


def _collect_forecasts(
    days: list[str],
    log_nav: np.ndarray,
    features: np.ndarray,
    invalid: np.ndarray,
    *,
    definition: Mapping[str, Any],
    selected_heads: tuple[tuple[int, int], ...],
    estimator_class: Any = None,
    include_history: bool = False,
) -> dict[str, Any]:
    candidate_id = definition["candidate_id"]
    evaluation, issued, unavailable = [], [], []
    historical = [] if include_history else None
    for horizon, lag in selected_heads:
        scores, emissions, available = _issue_head(
            days,
            log_nav,
            features,
            invalid,
            candidate_id=candidate_id,
            horizon=horizon,
            entry_lag=lag,
            estimator_class=estimator_class,
            historical_forecasts=historical,
        )
        evaluation.extend(scores)
        issued.extend(emissions)
        if not available:
            unavailable.append(
                {
                    "horizon": horizon,
                    "entry_lag": lag,
                    "status": "insufficient_history",
                    "reason": "fewer than 20 valid matured historical labels for this head",
                }
            )

    def order(row: Mapping[str, Any]) -> tuple[int, int, int]:
        return row["origin_index"], row["horizon"], row["entry_lag"]

    bundle = {
        "schema": "rmbfund.forecast-candidates.v1",
        "candidate_id": candidate_id,
        "candidate_hash": _digest(definition),
        "definition": dict(definition),
        "evaluation_rows": sorted(evaluation, key=order),
        "issued_forecasts": sorted(issued, key=order),
        "unavailable_forecasts": unavailable,
    }
    if include_history:
        bundle["historical_forecasts"] = sorted(historical, key=order)
    return bundle


def _recovery_head_metadata(
    features: np.ndarray,
    invalid: np.ndarray,
    *,
    horizon: int,
    entry_lag: int,
    head: dict[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Decorate only own predictions; leave every empirical baseline untouched."""
    count, span = len(features), horizon + entry_lag
    nonzero_at_maturity = np.zeros(count, dtype=int)
    if span < count:
        origins = np.arange(count - span)
        bad_prefix = np.r_[0, np.cumsum(invalid)]
        label_valid = (
            bad_prefix[origins + span + 1] == bad_prefix[origins + entry_lag + 1]
        )
        qualified = np.all(np.isfinite(features[origins]), axis=1) & (features[origins, 7] > 0)
        nonzero_at_maturity[span:] = qualified & label_valid
    matured_counts = np.cumsum(nonzero_at_maturity)

    def annotate(prediction: dict[str, Any], origin: int) -> None:
        prediction["method"] = RECOVERY_METHOD
        if "old_ridge_log_location" in prediction:
            prediction["unshrunk_recovery_log_location"] = prediction.pop("old_ridge_log_location")
        for name, column in (("log_return_5", 1), ("simple_drawdown_60", 6), ("recovery_feature", 7)):
            value = features[origin, column]
            prediction[name] = float(value) if np.isfinite(value) else None
        prediction["matured_nonzero_recovery_training_count"] = int(matured_counts[origin])

    if head is not None:
        annotate(head, count - 1)
    for row in rows:
        annotate(row["prediction"], row["origin_index"])


def issue_ridge_distribution_forecasts(
    series: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str = RISK_ID,
    heads: Sequence[tuple[int, int]] = RISK_HEADS,
) -> dict[str, Any]:
    """Research-only fixed Ridge distributions using the original training loop.

    New risk/shrink/recovery candidates never substitute empirical fallback;
    the old Ridge reference retains its exact original fallback contract. Each
    has full chronology independent of the frozen 2025/26 issued/evaluation
    windows. These research identities do not enter the runtime catalogue.
    """
    if candidate_id not in (MODEL_ID, RISK_ID, SHRINK_ID, RECOVERY_ID):
        raise ValidationError("fixed Ridge comparison only accepts old Ridge, risk, shrink50 and recovery")
    normalize = candidate_id == RISK_ID
    recovery = candidate_id == RECOVERY_ID
    selected = _heads(heads)
    days, log_nav, features, invalid = _inputs(_normalise_series(series), recovery=recovery)
    definition = MODEL_DEFINITION if candidate_id == MODEL_ID else candidate_definition(candidate_id)
    evaluation, issued, history, forecasts, unavailable = [], [], [], [], []
    for horizon, lag in selected:
        head_history = []
        head, scores, _, _, emissions = _fit_head(
            days, log_nav, features, invalid, horizon=horizon, entry_lag=lag,
            collect_issued=True, capture_baseline=True,
            historical_forecasts=head_history, normalize_residuals=normalize,
            feature_definitions=RECOVERY_FEATURE_DEFINITIONS if recovery else FEATURE_DEFINITIONS,
            shrink_location=candidate_id in (SHRINK_ID, RECOVERY_ID),
        )
        if recovery:
            _recovery_head_metadata(
                features, invalid, horizon=horizon, entry_lag=lag, head=head,
                rows=[*head_history, *scores, *emissions],
            )
        history.extend(head_history)
        evaluation.extend(scores)
        issued.extend(emissions)
        if head is not None and head.get("status") == "available":
            forecasts.append(head)
        else:
            unavailable.append(head or {
                "horizon": horizon, "entry_lag": lag, "status": "unavailable",
                "reason": "insufficient_history", "fallback_used": False,
            })

    def order(row: Mapping[str, Any]) -> tuple[int, int, int]:
        return row["origin_index"], row["horizon"], row["entry_lag"]

    return {
        "schema": "rmbfund.forecast-candidates.v1", "candidate_id": candidate_id,
        "candidate_hash": _digest(definition), "definition": definition,
        "forecasts": forecasts, "unavailable_forecasts": unavailable,
        "evaluation_rows": sorted(evaluation, key=order),
        "issued_forecasts": sorted(issued, key=order),
        "historical_forecasts": sorted(history, key=order),
    }


def issue_risk_forecasts(
    series: Sequence[Mapping[str, Any]], *,
    heads: Sequence[tuple[int, int]] = RISK_HEADS,
) -> dict[str, Any]:
    """Issue the sole new distribution candidate, without baseline substitution."""
    return issue_ridge_distribution_forecasts(series, candidate_id=RISK_ID, heads=heads)


def issue_shrinkage_forecasts(
    series: Sequence[Mapping[str, Any]], *,
    heads: Sequence[tuple[int, int]] = SHRINK_HEADS,
) -> dict[str, Any]:
    """Fixed shrink50, own matured raw errors and all origins; no fallback.

    Full historical_forecasts are the forecast-time input for custom research
    folds, including 2024 and the immature tail. The legacy-shaped evaluation
    and issued fields still use the original 2025/26 diagnostic windows.
    """
    return issue_ridge_distribution_forecasts(series, candidate_id=SHRINK_ID, heads=heads)


def issue_recovery_forecasts(
    series: Sequence[Mapping[str, Any]], *,
    heads: Sequence[tuple[int, int]] = SHRINK_HEADS,
) -> dict[str, Any]:
    """Own NAV8 Ridge with fixed shrink50 and matured raw errors; no fallback."""
    return issue_ridge_distribution_forecasts(series, candidate_id=RECOVERY_ID, heads=heads)


def issue_candidate_forecasts(
    series: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    heads: Sequence[tuple[int, int]] = DEFAULT_HEADS,
    include_history: bool = False,
) -> dict[str, Any]:
    """Issue fixed research predictions without I/O, persistence or model selection.

    H20 issuance does not consult future label validity or maturity. Scoring rows
    are a distinct, retrospective subset. Native thread limits are scoped to this
    call, not installed as process-wide environment configuration. Optional full
    history has explicit own-method unavailability and a separate empirical
    baseline; it never changes the legacy issued/evaluation fields or identity.
    """
    if not isinstance(include_history, bool):
        raise ValidationError("include_history must be boolean")
    if candidate_id == RECOVERY_ID:
        return issue_recovery_forecasts(series, heads=heads)
    if candidate_id == SHRINK_ID:
        return issue_shrinkage_forecasts(series, heads=heads)
    if candidate_id == RISK_ID:
        return issue_risk_forecasts(series, heads=heads)
    definition = candidate_definition(candidate_id)
    selected_heads = _heads(heads)
    days, log_nav, features, invalid = _inputs(_normalise_series(series))
    estimator_class = None
    limits = nullcontext()
    if candidate_id == TREE_ID:
        require_tree()
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        estimator_class = HistGradientBoostingRegressor
        limits = threadpool_limits(limits=1)
    with limits:
        return _collect_forecasts(
            days,
            log_nav,
            features,
            invalid,
            definition=definition,
            selected_heads=selected_heads,
            estimator_class=estimator_class,
            include_history=include_history,
        )


def issue_factor_forecasts(
    series: Sequence[Mapping[str, Any]],
    *,
    factor_features: np.ndarray,
    factor_definition: Mapping[str, Any],
    heads: Sequence[tuple[int, int]] = DEFAULT_HEADS,
) -> dict[str, Any]:
    """Fit NAV+VIX Ridge on already vetted, as-of-aligned two-column features.

    The caller owns alignment and frozen factor input identity. This function
    never reads source data or invents availability. Missing (NaN) feature values
    remain missing: their training rows are excluded and current rows fall back.
    """
    definition = factor_candidate_definition(factor_definition)
    selected_heads = _heads(heads)
    rows = _normalise_series(series)
    if (
        not isinstance(factor_features, np.ndarray)
        or factor_features.shape != (len(rows), 2)
        or not np.issubdtype(factor_features.dtype, np.number)
        or not np.isrealobj(factor_features)
        or np.any(np.isinf(factor_features))
    ):
        raise ValidationError("factor features require a real (NAV count, 2) array; NaN allowed, inf not allowed")
    try:
        from sklearn.linear_model import Ridge
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValidationError(
            "VIX增强研究需要可选 research 依赖 scikit-learn；尚未运行，不会静默替换模型。"
        ) from exc
    days, log_nav, nav_features, invalid = _inputs(rows)
    features = np.column_stack((nav_features, factor_features))
    with threadpool_limits(limits=1):
        result = _collect_forecasts(
            days,
            log_nav,
            features,
            invalid,
            definition=definition,
            selected_heads=selected_heads,
            estimator_class=Ridge,
        )
    result["aligned_factor_features_hash"] = _digest(
        [[float(value) if np.isfinite(value) else None for value in row] for row in factor_features]
    )
    return result

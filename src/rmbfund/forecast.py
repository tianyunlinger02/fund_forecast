"""Deterministic NAV forecasts from fixed ridge heads and empirical OOS errors.

Service callers own source/version selection and the actual available_at cutoff.
This pure core consumes strictly increasing, unique *real* NAV observations,
including legitimate weekend dates. Horizons and lags count those observations,
not fabricated calendar or trading sessions. No model selection is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

from .domain import ValidationError, timestamp
from .forecast_research import evaluate_forecasts

SCHEMA_VERSION = 1
MODEL_ID = "nav_ridge_empirical_v1"
FEATURE_LOOKBACK = 60
RIDGE_ALPHA = 10.0
FEATURE_STD_TOLERANCE = 1e-12
MIN_TRAINING_SAMPLES = 120
MIN_RESIDUAL_SAMPLES = 40
MIN_BASELINE_SAMPLES = 20
MIN_OBSERVATIONS = MIN_BASELINE_SAMPLES + 1
FEATURE_DEFINITIONS = (
    {"name": "log_return_1", "unit": "log_return", "description": "1-observation log NAV return"},
    {"name": "log_return_5", "unit": "log_return", "description": "5-observation log NAV return"},
    {"name": "log_return_20", "unit": "log_return", "description": "20-observation log NAV return"},
    {"name": "log_return_60", "unit": "log_return", "description": "60-observation log NAV return"},
    {
        "name": "volatility_20",
        "unit": "daily_log_return_std",
        "description": "Population standard deviation of 20 one-observation log returns; not annualized",
    },
    {
        "name": "volatility_60",
        "unit": "daily_log_return_std",
        "description": "Population standard deviation of 60 one-observation log returns; not annualized",
    },
    {
        "name": "drawdown_60",
        "unit": "return_ratio",
        "description": "Current NAV / maximum of the current and preceding 60 NAVs - 1",
    },
)
MODEL_DEFINITION = {
    "model_id": MODEL_ID,
    "feature_lookback": FEATURE_LOOKBACK,
    "features": list(FEATURE_DEFINITIONS),
    "horizons": [1, 5, 20],
    "execution_horizon": 20,
    "target": "raw log NAV[i+entry_lag+horizon] - log NAV[i+entry_lag]",
    "ridge_alpha": RIDGE_ALPHA,
    "standardization": "expanding matured training rows only; population std; std <= 1e-12 -> 1",
    "feature_std_tolerance": FEATURE_STD_TOLERANCE,
    "intercept": "unpenalized training target mean",
    "minimum_training_samples": MIN_TRAINING_SAMPLES,
    "minimum_residual_samples": MIN_RESIDUAL_SAMPLES,
    "minimum_baseline_samples": MIN_BASELINE_SAMPLES,
    "residuals": "all earlier prequential ridge errors; admitted only when full target matures",
    "predictive_samples": "expm1(current ridge log location + matured prequential log errors)",
    "fallback": "empirical historical matured target log returns; no fitted residual substitution",
    "interval": "0.1 and 0.9 linear empirical simple-return quantiles; nominal coverage 0.8",
    "probabilities": "finite sample proportions >0, <0, =0; 0/1 are not certainty claims",
    "event_policy": "invalid_return=True or label_valid=False marks incoming NAV move; "
    "exclude crossing labels and 60-return feature windows; never adjust NAVs",
    "outlier_policy": "flag unmarked abs one-observation log return >0.25; no clipping or winsorizing",
    "research_availability_lag_observations": 1,
    "development_period": ["2025-01-01", "2025-12-31"],
    "chronological_heldout_period": ["2026-01-01", "2026-12-31"],
    "strict_unseen": False,
    "hyperparameter_search": False,
    "factors": "NAV only; optional supplied factors retained in input provenance but not fitted",
}


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "forecast input provenance must be finite JSON-compatible values"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


MODEL_HASH = _digest(MODEL_DEFINITION)

# A new issuance contract, not a new algorithm or a new research candidate.
# Keep the v1 definition/hash above frozen for historical study replay.
RUNTIME_SCHEMA_VERSION = 2
RUNTIME_MODEL_ID = "nav_ridge_empirical_v2"
RUNTIME_TARGET_DEFINITION = {
    "id": "raw_unit_nav_simple_return_v1",
    "target_kind": "raw_nav_simple_return",
    "simple_return": "NAV[i+entry_lag+horizon] / NAV[i+entry_lag] - 1",
    "fitted_log_return": MODEL_DEFINITION["target"],
    "nav_basis": "raw unit NAV; no dividend reinvestment or split adjustment",
    "origin": "i is the last supplied NAV observation known at issuance",
    "horizon": "number of supplied NAV observations from entry to target",
    "entry_lag": "number of supplied NAV observations from origin to scenario entry",
    "observation_unit": "actual supplied NAV observations, not calendar or trading days",
    "not_total_return": True,
    "transaction_fees": "subscription/redemption fees excluded; NAV-embedded fees not re-deducted",
}
RUNTIME_MODEL_DEFINITION = {
    **{
        key: value
        for key, value in json.loads(_canonical(MODEL_DEFINITION)).items()
        if key != "execution_horizon"
    },
    "model_id": RUNTIME_MODEL_ID,
    "schema_version": RUNTIME_SCHEMA_VERSION,
    "algorithm_model_id": MODEL_ID,
    "algorithm_model_hash": MODEL_HASH,
    "algorithm_changes": False,
    "horizons": [1, 5, 20],
    "execution_horizons": [1, 5, 20],
    "target_definition": RUNTIME_TARGET_DEFINITION,
    "zero_lag": "execution aliases the exact corresponding anchor head; fit each identity once",
    "historical_baseline": "each final head seals its same-fit matured empirical historical "
    "summary and target cutoffs, without samples or another fit",
    "drivers": "each final head retains its own fixed-ridge contributions; empty on fallback",
    "maturity": "training/residual target endpoint indices and dates remain head-specific",
    "anchor_currency": "only explicit final-row currency or provenance.currency, with source "
    "reference; missing or conflicting currency is not inferred",
}
RUNTIME_MODEL_HASH = _digest(RUNTIME_MODEL_DEFINITION)

# An explicit RAM-only contract for a conditional calendar plan's exact H.
# Neither this definition nor its hash replaces the official issuance identity.
LOCAL_TIMING_CONTRACT = "calendar_trade_nav_observations_v1"
LOCAL_RUNTIME_MODEL_DEFINITION = {
    **RUNTIME_MODEL_DEFINITION,
    "local_contract": LOCAL_TIMING_CONTRACT,
    "official_model_hash": RUNTIME_MODEL_HASH,
    "horizons": "explicit requested positive NAV-observation horizon",
    "execution_horizons": "the same requested horizon at each selected entry lag",
    "target_unit": "nav_observations",
    "calendar_mapping": "conditional business-day schedule is separate from numerical inputs",
    "actual_issue": False,
    "persisted": False,
}
LOCAL_RUNTIME_MODEL_HASH = _digest(LOCAL_RUNTIME_MODEL_DEFINITION)


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in values)):
        raise ValidationError("horizons must be positive integer NAV observation counts")
    return tuple(dict.fromkeys(values))


def local_curve_horizons(plan_horizon: int) -> tuple[int, ...]:
    """Return every exact target horizon used by a local plan curve."""
    if isinstance(plan_horizon, bool) or not isinstance(plan_horizon, int) or plan_horizon <= 0:
        raise ValidationError("curve horizon must be a positive integer NAV observation count")
    return tuple(range(1, plan_horizon + 1))


def _local_target_metadata(head: dict[str, Any]) -> None:
    head.update(local_contract=LOCAL_TIMING_CONTRACT, target_unit="nav_observations")


def _entry_lags(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError("entry_lags must be a sequence of nonnegative integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValidationError("entry_lags must be nonnegative integer observation counts")
    return tuple(dict.fromkeys(values))


def minimum_observations(entry_lags: Sequence[int] = (2, 3, 4)) -> int:
    """Theoretical count for ALL requested baseline heads, before event exclusions."""
    return max((20, *(20 + lag for lag in _entry_lags(entry_lags)))) + MIN_BASELINE_SAMPLES


def _normalise_series(series: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(series, (str, bytes)) or not isinstance(series, Sequence) or not series:
        raise ValidationError("insufficient_history: forecast requires dated NAV observations")
    result = []
    previous = ""
    for index, item in enumerate(series):
        if not isinstance(item, Mapping):
            raise ValidationError(f"forecast observation {index} must be a mapping")
        raw_date = item.get("date")
        try:
            day = date.fromisoformat(raw_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid forecast date at observation {index}") from exc
        if raw_date != day:
            raise ValidationError("forecast dates must be canonical YYYY-MM-DD")
        if day <= previous:
            raise ValidationError("forecast dates must be strictly increasing and unique")
        raw_nav = item.get("nav")
        if isinstance(raw_nav, bool):
            raise ValidationError("forecast NAV cannot be boolean")
        try:
            number = Decimal(str(raw_nav))
            nav = float(number)
        except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
            raise ValidationError(f"invalid forecast NAV at {day}") from exc
        if not number.is_finite() or number <= 0 or not math.isfinite(nav) or nav <= 0:
            raise ValidationError(
                f"forecast NAV must be positive, finite and representable at {day}"
            )
        for flag in ("invalid_return", "label_valid"):
            if flag in item and not isinstance(item[flag], bool):
                raise ValidationError(f"forecast {flag} must be boolean")
        row = dict(item)
        row.update(date=day, nav=str(raw_nav))
        result.append(row)
        previous = day
    # Also makes the complete provenance digest well-defined before fitting.
    _canonical(result)
    return result


def forecast_input_hash(series: Sequence[Mapping[str, Any]]) -> str:
    """Hash the normalized dated NAVs AND all supplied JSON provenance metadata."""
    return _digest(_normalise_series(series))


def _feature_matrix(log_nav: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    result = np.full((len(log_nav), len(FEATURE_DEFINITIONS)), np.nan, dtype=float)
    changes = np.diff(log_nav)
    for index in range(FEATURE_LOOKBACK, len(log_nav)):
        if np.any(invalid[index - FEATURE_LOOKBACK + 1 : index + 1]):
            continue
        result[index] = [
            *(log_nav[index] - log_nav[index - horizon] for horizon in (1, 5, 20, 60)),
            float(np.std(changes[index - 20 : index])),
            float(np.std(changes[index - 60 : index])),
            float(np.expm1(log_nav[index] - np.max(log_nav[index - 60 : index + 1]))),
        ]
    return result


def _distribution(
    log_samples: np.ndarray,
    *,
    horizon: int,
    entry_lag: int,
    method: str,
    include_samples: bool = False,
) -> dict[str, Any]:
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.expm1(log_samples)
    if not len(samples) or not np.all(np.isfinite(samples)):
        raise ValidationError("forecast predictive returns are not finite and representable")
    lower, median, upper = np.quantile(samples, [0.1, 0.5, 0.9], method="linear")
    # Dividing before summing avoids overflow of a representable finite-sample mean.
    expected = float(np.sum(samples / len(samples)))
    if not math.isfinite(expected):
        raise ValidationError("forecast mean return is not finite and representable")
    result = {
        "horizon": horizon,
        "entry_lag": entry_lag,
        "status": "available",
        "up_probability": float(np.count_nonzero(samples > 0) / len(samples)),
        "down_probability": float(np.count_nonzero(samples < 0) / len(samples)),
        "flat_probability": float(np.count_nonzero(samples == 0) / len(samples)),
        "expected_change": expected,
        "median_change": float(median),
        "lower": float(lower),
        "upper": float(upper),
        "interval_coverage": 0.8,
        "sample_count": len(samples),
        "method": method,
        "probability_interpretation": "finite empirical frequency; 0 or 1 does not mean certainty",
    }
    if include_samples:
        result["samples"] = samples.tolist()
    return result


def _historical_row(
    days: list[str], origin: int, prediction: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A reconstructed decision origin, never a persisted issue or an outcome."""
    def summary(head):
        result = {key: value for key, value in head.items()
                  if key not in {"samples", "historical_baseline", "drivers"}}
        result["target_definition_id"] = RUNTIME_TARGET_DEFINITION["id"]
        for kind in ("training", "residual"):
            key = f"{kind}_target_end_index"
            if key in result:
                endpoint = result[key]
                result[f"{kind}_target_end_date"] = days[endpoint] if endpoint is not None else None
        return result

    row = {
        "horizon": prediction["horizon"], "entry_lag": prediction["entry_lag"],
        "origin_index": origin, "origin_date": days[origin],
        "knowledge_cutoff_index": origin, "knowledge_cutoff_date": days[origin],
        "assumed_known_index": origin + 1,
        "assumed_known_date": days[origin + 1] if origin + 1 < len(days) else None,
        "assumed_nav_availability_lag_observations": 1,
        "prediction": summary(prediction),
    }
    if baseline is not None:
        row["historical_baseline"] = summary(baseline)
    return row


def _risk_scales(log_nav: np.ndarray, invalid: np.ndarray, horizon: int) -> np.ndarray:
    """Known-at-origin 20-return population volatility times sqrt(exact H).

    No epsilon, filling or neighboring-head scale. A zero/invalid scale remains
    unavailable, including when its otherwise valid label later matures.
    """
    scales = np.full(len(log_nav), np.nan)
    changes = np.diff(log_nav)
    for origin in range(20, len(log_nav)):
        if np.any(invalid[origin - 19 : origin + 1]):
            continue
        scale = float(np.std(changes[origin - 20 : origin]) * math.sqrt(horizon))
        if math.isfinite(scale) and scale > 0:
            scales[origin] = scale
    return scales


def _shrink50_log_location(old_ridge_log_location: float, training_log_mean: float) -> float:
    """The one fixed research hypothesis; not a fitted or configurable weight."""
    return float(training_log_mean + 0.5 * (old_ridge_log_location - training_log_mean))


def _fit_head(
    days: list[str],
    log_nav: np.ndarray,
    features: np.ndarray,
    invalid: np.ndarray,
    *,
    horizon: int,
    entry_lag: int,
    collect_issued: bool = False,
    capture_baseline: bool = False,
    historical_forecasts: list[dict[str, Any]] | None = None,
    fitted_history_only: bool = False,
    feature_definitions: Sequence[Mapping[str, Any]] = FEATURE_DEFINITIONS,
    normalize_residuals: bool = False,
    shrink_location: bool = False,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """One expanding origin loop; every admission is endpoint <= known origin.

    normalize_residuals is an offline research distribution only. It leaves the
    fixed ridge fit/log location unchanged and never falls back to a baseline.
    shrink_location instead halves the conditional log contribution and stores
    its own raw prequential errors; it cannot be combined with risk scaling.
    The default branch and all official issuance identities are unchanged.
    """
    if normalize_residuals and shrink_location:
        raise ValidationError("fixed shrink50 uses raw residuals, not volatility scaling")
    count, width = features.shape
    span = horizon + entry_lag
    if span >= count and historical_forecasts is None:
        return None, [], None, [], []
    bad_prefix = np.r_[0, np.cumsum(invalid)]
    targets = np.full(count, np.nan)
    for origin in range(count - span):
        entry, endpoint = origin + entry_lag, origin + span
        if bad_prefix[endpoint + 1] == bad_prefix[entry + 1]:
            targets[origin] = log_nav[endpoint] - log_nav[entry]
    historical, historical_origins, residuals, residual_endpoints = [], [], [], []
    prequential = np.full(count, np.nan)
    risk_scales = _risk_scales(log_nav, invalid, horizon) if normalize_residuals else None
    training_count = 0
    mean_x, centered_xy = np.zeros(width), np.zeros(width)
    centered_xx = np.zeros((width, width))
    mean_y = 0.0
    first_training_origin = None
    training_endpoint = None
    evaluation_rows, issued_rows = [], []
    latest, latest_state, drivers = None, None, []
    identity = np.eye(width)
    for origin in range(count):
        # A prior forecast and its full label become eligible only NOW, never at
        # their original forecast date. Future invalid-event labels are also
        # excluded only at maturity, not used to alter an earlier prediction.
        matured = origin - span
        if matured >= 0 and np.isfinite(targets[matured]):
            target = float(targets[matured])
            historical.append(target)
            historical_origins.append(matured)
            if np.isfinite(prequential[matured]):
                error = target - float(prequential[matured])
                if not normalize_residuals:
                    residuals.append(error)
                    residual_endpoints.append(origin)
                elif np.isfinite(risk_scales[matured]):
                    standardized_error = error / float(risk_scales[matured])
                    if not math.isfinite(standardized_error):
                        raise ValidationError("standardized prequential error is not finite; no substitution")
                    residuals.append(standardized_error)
                    residual_endpoints.append(origin)
            if np.all(np.isfinite(features[matured])):
                x = features[matured]
                # Welford centered moments avoid cancellation on flat/trending
                # NAV histories; these are exactly the expanding ridge sums.
                training_count += 1
                delta_x, delta_y = x - mean_x, target - mean_y
                mean_x += delta_x / training_count
                mean_y += delta_y / training_count
                centered_xx += np.outer(delta_x, x - mean_x)
                centered_xy += delta_x * (target - mean_y)
                if first_training_origin is None:
                    first_training_origin = matured
                training_endpoint = origin

        location = old_ridge_location = None
        coefficients = means = scales = standardized = None
        if training_count >= MIN_TRAINING_SAMPLES and np.all(np.isfinite(features[origin])):
            means = mean_x.copy()
            variance = np.maximum(np.diag(centered_xx) / training_count, 0.0)
            scales = np.sqrt(variance)
            scales[scales <= FEATURE_STD_TOLERANCE] = 1.0
            gram = centered_xx / np.outer(scales, scales)
            coefficients = np.linalg.solve(gram + RIDGE_ALPHA * identity, centered_xy / scales)
            standardized = (features[origin] - means) / scales
            location = float(mean_y + standardized @ coefficients)
            if not math.isfinite(location):
                raise ValidationError("ridge forecast is not finite")
            if shrink_location:
                old_ridge_location = location
                location = _shrink50_log_location(old_ridge_location, float(mean_y))
            prequential[origin] = location

        final = origin == count - 1
        in_evaluation = (
            origin + span < count
            and np.isfinite(targets[origin])
            and "2025-01-01" <= days[origin] <= "2026-12-31"
        )
        # Issuance is a forecast-time record, not an outcome-time filter. It
        # includes immature and later-invalid targets, and the prior-year origin
        # needed for the first research-year decision. No target value/validity
        # is consulted to admit an issued prediction.
        in_issuance = (
            collect_issued
            and horizon == 20
            and (
                "2025-01-01" <= days[origin] <= "2026-12-31"
                or (origin + 1 < count and "2025-01-01" <= days[origin + 1] <= "2026-12-31")
            )
        )
        if shrink_location:
            shrink_metadata = {
                "old_ridge_log_location": old_ridge_location,
                "training_log_mean": float(mean_y) if training_count else None,
                "conditional_log_contribution": (
                    float(old_ridge_location - mean_y) if old_ridge_location is not None else None
                ),
                "candidate_log_location": location,
                "residual_log_mean": float(np.mean(residuals)) if residuals else None,
                "training_sample_count": training_count,
                "residual_sample_count": len(residuals),
                "training_start": days[first_training_origin] if first_training_origin is not None else None,
                "training_end": days[training_endpoint] if training_endpoint is not None else None,
                "training_target_end_index": training_endpoint,
                "residual_target_end_index": residual_endpoints[-1] if residual_endpoints else None,
                "fallback_used": False,
            }
            shrink_unavailable_reason = (
                "feature_window_crosses_invalid_event_or_shorter_than_60"
                if not np.all(np.isfinite(features[origin]))
                else "insufficient_matured_training_labels" if location is None
                else "insufficient_matured_prequential_residuals"
            )
        if shrink_location and len(historical) < MIN_BASELINE_SAMPLES:
            early = {
                "horizon": horizon, "entry_lag": entry_lag, "status": "unavailable",
                "method": "ridge_shrink50_prequential_empirical",
                "reason": shrink_unavailable_reason, **shrink_metadata,
            }
            if historical_forecasts is not None:
                historical_forecasts.append(_historical_row(days, origin, early))
            if final:
                latest = early
            continue
        if historical_forecasts is not None and len(historical) < MIN_BASELINE_SAMPLES:
            early = {
                "horizon": horizon, "entry_lag": entry_lag, "status": "insufficient_history",
                "reason": "fewer than 20 valid matured historical labels for this head",
            }
            if normalize_residuals:
                early.update(
                    method="ridge_vol20_prequential_empirical", ridge_log_location=location,
                    risk_scale=float(risk_scales[origin]) if np.isfinite(risk_scales[origin]) else None,
                    fallback_used=False,
                )
            historical_forecasts.append(_historical_row(days, origin, early))
        if len(historical) < MIN_BASELINE_SAMPLES or not (
            final or in_evaluation or in_issuance or historical_forecasts is not None
        ):
            continue
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
        if shrink_location:
            if location is None or len(residuals) < MIN_RESIDUAL_SAMPLES:
                prediction = {
                    "horizon": horizon, "entry_lag": entry_lag, "status": "unavailable",
                    "method": "ridge_shrink50_prequential_empirical",
                    "reason": shrink_unavailable_reason,
                }
            else:
                prediction = _distribution(
                    location + np.asarray(residuals), horizon=horizon, entry_lag=entry_lag,
                    method="ridge_shrink50_prequential_empirical", include_samples=final,
                )
            prediction.update(shrink_metadata)
        elif normalize_residuals:
            scale = float(risk_scales[origin]) if np.isfinite(risk_scales[origin]) else None
            metadata = {
                "ridge_log_location": location,
                "risk_scale": scale,
                "risk_scale_lookback": 20,
                "risk_scale_horizon": horizon,
                "training_sample_count": training_count,
                "residual_sample_count": len(residuals),
                "training_start": days[first_training_origin] if first_training_origin is not None else None,
                "training_end": days[training_endpoint] if training_endpoint is not None else None,
                "training_target_end_index": training_endpoint,
                "residual_target_end_index": residual_endpoints[-1] if residual_endpoints else None,
                "fallback_used": False,
            }
            if location is None or scale is None or len(residuals) < MIN_RESIDUAL_SAMPLES:
                prediction = {
                    "horizon": horizon, "entry_lag": entry_lag, "status": "unavailable",
                    "method": "ridge_vol20_prequential_empirical",
                    "reason": (
                        "invalid_or_nonpositive_risk_scale" if scale is None
                        else "feature_window_crosses_invalid_event_or_shorter_than_60"
                        if not np.all(np.isfinite(features[origin]))
                        else "insufficient_matured_training_labels" if location is None
                        else "insufficient_matured_standardized_residuals"
                    ),
                    **metadata,
                }
            else:
                prediction = _distribution(
                    location + scale * np.asarray(residuals),
                    horizon=horizon, entry_lag=entry_lag,
                    method="ridge_vol20_prequential_empirical", include_samples=final,
                )
                prediction.update(metadata)
        elif location is not None and len(residuals) >= MIN_RESIDUAL_SAMPLES:
            prediction = _distribution(
                location + np.asarray(residuals),
                horizon=horizon,
                entry_lag=entry_lag,
                method="ridge_prequential_empirical",
                include_samples=final,
            )
            prediction.update(
                ridge_log_location=location,
                training_sample_count=training_count,
                residual_sample_count=len(residuals),
                training_start=days[first_training_origin],
                training_end=days[training_endpoint],
                training_target_end_index=training_endpoint,
                residual_target_end_index=residual_endpoints[-1],
            )
        else:
            prediction = dict(baseline)
            prediction["fallback_reason"] = (
                "feature_window_crosses_invalid_event_or_shorter_than_60"
                if not np.all(np.isfinite(features[origin]))
                else "insufficient_matured_training_labels"
                if location is None
                else "insufficient_matured_prequential_residuals"
            )
            if final:
                prediction["samples"] = np.expm1(np.asarray(historical)).tolist()
        if historical_forecasts is not None:
            head = prediction
            if fitted_history_only and prediction["method"] != "ridge_prequential_empirical":
                head = {
                    "horizon": horizon, "entry_lag": entry_lag, "status": "unavailable",
                    "reason": prediction["fallback_reason"],
                    "required_method": "ridge_prequential_empirical", "fallback_used": False,
                }
            historical_forecasts.append(_historical_row(days, origin, head, baseline))
        if in_issuance:
            issued_rows.append(
                {
                    "horizon": horizon,
                    "entry_lag": entry_lag,
                    "origin_index": origin,
                    "origin_date": days[origin],
                    "knowledge_cutoff_index": origin,
                    "knowledge_cutoff_date": days[origin],
                    "assumed_known_index": origin + 1,
                    "assumed_known_date": days[origin + 1] if origin + 1 < count else None,
                    "assumed_nav_availability_lag_observations": 1,
                    "prediction": {
                        key: value for key, value in prediction.items() if key != "samples"
                    },
                }
            )
        if in_evaluation:
            no_change = _distribution(
                np.zeros(1), horizon=horizon, entry_lag=entry_lag, method="no_change_point_baseline"
            )
            # Deliberately no full samples/model copies for each historical row.
            evaluation_rows.append(
                {
                    "horizon": horizon,
                    "entry_lag": entry_lag,
                    "origin_index": origin,
                    "origin_date": days[origin],
                    "entry_index": origin + entry_lag,
                    "entry_date": days[origin + entry_lag],
                    "target_index": origin + span,
                    "target_date": days[origin + span],
                    "knowledge_cutoff_index": origin,
                    "knowledge_cutoff_date": days[origin],
                    "assumed_known_index": origin + 1,
                    "assumed_known_date": days[origin + 1],
                    "assumed_nav_availability_lag_observations": 1,
                    "actual_log_return": float(targets[origin]),
                    "actual_change": float(np.expm1(targets[origin])),
                    "period": "development_2025" if days[origin] < "2026-01-01" else "heldout_2026",
                    "prediction": prediction,
                    "historical_baseline": baseline,
                    "no_change_baseline": no_change,
                }
            )
        if final:
            if capture_baseline:
                # Seal the baseline already calculated at this origin. In the
                # fallback case it must be a separate summary, not a self-link.
                prediction["historical_baseline"] = dict(baseline)
            latest = prediction
            if prediction["method"] == "ridge_prequential_empirical":
                latest_state = {
                    "horizon": horizon,
                    "entry_lag": entry_lag,
                    "alpha": RIDGE_ALPHA,
                    "training_sample_count": training_count,
                    "training_start": days[first_training_origin],
                    "training_end": days[training_endpoint],
                    "feature_names": [item["name"] for item in feature_definitions],
                    "feature_means": means.tolist(),
                    "feature_scales": scales.tolist(),
                    "coefficients": coefficients.tolist(),
                    "intercept": float(mean_y),
                    "ridge_log_location": location,
                }
                drivers = sorted(
                    [
                        {
                            "horizon": horizon,
                            "entry_lag": entry_lag,
                            "feature": item["name"],
                            "value": float(features[origin, index]),
                            "standardized_value": float(standardized[index]),
                            "coefficient": float(coefficients[index]),
                            "contribution_log_return": float(
                                standardized[index] * coefficients[index]
                            ),
                        }
                        for index, item in enumerate(feature_definitions)
                    ],
                    key=lambda item: abs(item["contribution_log_return"]),
                    reverse=True,
                )
    return latest, evaluation_rows, latest_state, drivers, issued_rows


def fit_forecast(
    series: Sequence[Mapping[str, Any]],
    *,
    product_id: str,
    generated_at: str,
    entry_lags: Sequence[int] = (2, 3, 4),
    include_evaluation: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready forecast bundle without I/O or mutable model state.

    Per-row invalid_return=True (or label_valid=False) marks the incoming NAV
    change as unsuitable for return labels/features, e.g. a known unadjusted
    split/dividend. Do not invent such events just because coverage is unknown.
    Other metadata, including acquisition times and factors, is retained in the
    input digest but never reinterpreted as historical availability.

    Available heads contain numeric forecasts and simple-return ``samples``.
    Insufficient heads are listed separately in unavailable_forecasts; there is
    no guessed 50% probability. If no head is available, raise ValidationError.
    include_evaluation adds matured evaluation_rows and compact H20 issued_forecasts
    for all 2025/2026 research origins, even when their outcomes are not yet known.
    Issuance also includes the preceding origin needed for a first research-year
    decision; it is never filtered on future target maturity or event validity.
    This v1 research contract is frozen; daily issuance uses fit_runtime_forecast.
    """
    return _fit_forecast_core(
        series,
        product_id=product_id,
        generated_at=generated_at,
        entry_lags=entry_lags,
        include_evaluation=include_evaluation,
        runtime=False,
    )


def fit_runtime_forecast(
    series: Sequence[Mapping[str, Any]],
    *,
    product_id: str,
    generated_at: str,
    entry_lags: Sequence[int] = (2, 3, 4),
    include_evaluation: bool = False,
    include_history: bool = False,
) -> dict[str, Any]:
    """Issue v2 heads with the unchanged v1 ridge/fallback calculations.

    Anchor and each requested execution lag have H1/H5/H20 heads where history
    permits. A zero-lag execution head aliases its anchor exactly. Every final
    head seals its same-fit historical_baseline (without samples), own drivers,
    and target maturity dates. The bundle identifies the raw unit-NAV return
    target; anchor currency is retained only when explicit in its source row.

    Input validation/event policy and optional chronological diagnostics are
    inherited from fit_forecast. Optional issued_forecasts remain the original
    H20 research diagnostics, not a new candidate study or real issuance history.
    Explicit include_history instead adds ALL H1/H5/H20 chronological origins,
    including unavailable early heads and immature/later-invalid target windows.
    These hypothetical rows contain no outcomes or fabricated issuance clocks.
    With that option an entirely insufficient bundle retains its unavailable rows.
    """
    if not isinstance(include_history, bool):
        raise ValidationError("include_history must be boolean")
    return _fit_forecast_core(
        series,
        product_id=product_id,
        generated_at=generated_at,
        entry_lags=entry_lags,
        include_evaluation=include_evaluation,
        runtime=True,
        include_history=include_history,
    )


def fit_local_timing_forecast(
    series: Sequence[Mapping[str, Any]], *, product_id: str, generated_at: str,
    horizons: Sequence[int], entry_lags: Sequence[int] = (2, 3, 4),
    include_history: bool = False,
) -> dict[str, Any]:
    """RAM-only exact-H heads using the unchanged fixed-span maturity/kernel.

    This is not official issuance or a validated calendar-day probability model.
    The caller's conditional schedule determines H/L separately. No interpolation,
    neighboring head, mutable model state or I/O is used.
    """
    if not isinstance(include_history, bool):
        raise ValidationError("include_history must be boolean")
    return _fit_forecast_core(
        series, product_id=product_id, generated_at=generated_at,
        entry_lags=entry_lags, include_evaluation=False, runtime=True,
        include_history=include_history, horizons=_horizons(horizons),
    )


def reconstruct_investment_history(
    series: Sequence[Mapping[str, Any]], *, factors: Mapping[str, Any],
    entry_lags: Sequence[int] = (2, 3, 4), horizons: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Use the existing factor features/head loop after normal input qualification.

    The service first calls fit_investment_forecast to validate the actual model,
    mapping and acquisition cutoff, and passes its selected factor_inputs here.
    Empirical fallback is a baseline only, never a relabeled factor prediction.
    No outcomes, issue ledger, new algorithm or historical first-seen claim.
    """
    from .investment_factors import align_investment_features
    from .investment_forecast import FEATURE_DEFINITIONS as definitions

    rows, lags = _normalise_series(series), _entry_lags(entry_lags)
    days = [row["date"] for row in rows]
    log_nav = np.log(np.asarray([float(row["nav"]) for row in rows]))
    invalid = np.asarray([row.get("invalid_return", False) or row.get("label_valid") is False
                          for row in rows], dtype=bool)
    external, _, _ = align_investment_features(rows, factors)
    features = np.column_stack((_feature_matrix(log_nav, invalid), external))
    history = []
    selected_horizons = (1, 5, 20) if horizons is None else _horizons(horizons)
    for horizon, lag in dict.fromkeys(
        [(h, 0) for h in selected_horizons]
        + [(h, lag) for lag in lags for h in selected_horizons]
    ):
        _fit_head(days, log_nav, features, invalid, horizon=horizon, entry_lag=lag,
                  historical_forecasts=history, fitted_history_only=True,
                  feature_definitions=definitions)
    if horizons is not None:
        for row in history:
            _local_target_metadata(row)
            for head in (row["prediction"], row.get("historical_baseline")):
                if head is not None:
                    _local_target_metadata(head)
    return sorted(history, key=lambda row: (row["origin_index"], row["horizon"], row["entry_lag"]))


def _runtime_currency(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only a final observation's explicit, unambiguous currency claim."""
    provenance = row.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    claims = [
        (field, value.strip())
        for field, value in (
            ("currency", row.get("currency")),
            ("provenance.currency", provenance.get("currency")),
        )
        if isinstance(value, str) and value.strip()
    ]
    if not claims or len({value for _, value in claims}) != 1:
        return {}
    field, currency = claims[0]
    source = {"field": field, "date": row["date"]}
    primary, secondary = (row, provenance) if field == "currency" else (provenance, row)
    for key in ("fact_id", "source", "source_url", "observed_at", "available_at", "snapshot_id"):
        value = primary.get(key)
        if value is None:
            value = secondary.get(key)
        if value is not None:
            source[key] = value
    return {"currency": currency, "currency_source": json.loads(_canonical(source))}


def _fit_forecast_core(
    series: Sequence[Mapping[str, Any]],
    *,
    product_id: str,
    generated_at: str,
    entry_lags: Sequence[int],
    include_evaluation: bool,
    runtime: bool,
    include_history: bool = False,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Shared numerical core; only the explicit output contract varies."""
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValidationError("forecast product_id must be nonempty")
    if not isinstance(generated_at, str):
        raise ValidationError("forecast generated_at must be an ISO timestamp string with timezone")
    timestamp(generated_at)
    lags = _entry_lags(entry_lags)
    rows = _normalise_series(series)
    days = [row["date"] for row in rows]
    log_nav = np.log(np.asarray([float(row["nav"]) for row in rows]))
    invalid = np.asarray(
        [row.get("invalid_return", False) or row.get("label_valid") is False for row in rows],
        dtype=bool,
    )
    features = _feature_matrix(log_nav, invalid)
    forecasts, execution, unavailable, evaluation, states, drivers = [], [], [], [], [], []
    issued = []
    chronological = [] if include_history else None
    selected_horizons = (1, 5, 20) if horizons is None else _horizons(horizons)
    execution_horizons = selected_horizons if runtime else (20,)
    execution_heads = [(horizon, lag) for lag in lags for horizon in execution_horizons]
    # Reuse every exact anchor result for explicitly supplied zero-lag execution.
    heads = dict.fromkeys([(horizon, 0) for horizon in selected_horizons] + execution_heads)
    results = {}
    for horizon, lag in heads:
        forecast, history, state, head_drivers, issued_rows = _fit_head(
            days,
            log_nav,
            features,
            invalid,
            horizon=horizon,
            entry_lag=lag,
            collect_issued=include_evaluation,
            capture_baseline=runtime,
            historical_forecasts=chronological,
        )
        if runtime and forecast is not None:
            forecast["drivers"] = head_drivers
            for summary in (forecast, forecast["historical_baseline"]):
                summary["target_definition_id"] = RUNTIME_TARGET_DEFINITION["id"]
                if horizons is not None:
                    _local_target_metadata(summary)
                for kind in ("training", "residual"):
                    endpoint = summary[f"{kind}_target_end_index"]
                    summary[f"{kind}_target_end_date"] = (
                        days[endpoint] if endpoint is not None else None
                    )
        results[horizon, lag] = forecast
        evaluation.extend(history)
        issued.extend(issued_rows)
        drivers.extend(head_drivers)
        if state:
            states.append(state)
        if forecast is None:
            unavailable.append(
                {
                    "horizon": horizon,
                    "entry_lag": lag,
                    "status": "insufficient_history",
                    "reason": "fewer than 20 valid matured historical labels for this head",
                    "minimum_observations_without_events": horizon + lag + MIN_BASELINE_SAMPLES,
                }
            )
    forecasts = [results[horizon, 0] for horizon in selected_horizons
                 if results[horizon, 0] is not None]
    execution = [results[head] for head in execution_heads if results[head] is not None]
    if not forecasts and not execution and not include_history:
        raise ValidationError(
            "insufficient_history: no head has 20 valid matured historical labels"
        )
    feature_summary = []
    if np.all(np.isfinite(features[-1])):
        feature_summary = [
            dict(item, value=float(features[-1, index]))
            for index, item in enumerate(FEATURE_DEFINITIONS)
        ]
    moves = np.diff(log_nav)
    large_moves = [
        {
            "date": days[index + 1],
            "log_return": float(change),
            "known_invalid": bool(invalid[index + 1]),
        }
        for index, change in enumerate(moves)
        if abs(change) > 0.25
    ]
    warnings = [
        "Unit-NAV changes are not verified reinvested total returns.",
        "Horizon/lag counts are supplied NAV observations, not a guaranteed exchange calendar.",
        "No external factor is fitted; factors metadata is provenance only.",
    ]
    if any(not item["known_invalid"] for item in large_moves):
        warnings.append(
            "Unmarked large NAV moves require source/event review; no automatic adjustment."
        )
    bundle = {
        "schema_version": RUNTIME_SCHEMA_VERSION if runtime else SCHEMA_VERSION,
        "model_id": RUNTIME_MODEL_ID if runtime else MODEL_ID,
        "model_hash": RUNTIME_MODEL_HASH if runtime else MODEL_HASH,
        "model_definition": json.loads(
            _canonical(RUNTIME_MODEL_DEFINITION if runtime else MODEL_DEFINITION)
        ),
        "input_hash": _digest(rows),
        "product_id": product_id,
        "generated_at": generated_at,
        "anchor": {"date": days[-1], "nav": rows[-1]["nav"]},
        "observation_count": len(rows),
        "training_start": days[0],
        "training_end": days[-1],
        "training_window_basis": "available NAV history; fitted label ranges are per-head",
        "forecasts": forecasts,
        "execution_forecasts": execution,
        "unavailable_forecasts": unavailable,
        "entry_lags": list(lags),
        "validation": evaluate_forecasts(evaluation),
        "features": feature_summary,
        "drivers": drivers,
        "model_state": states,
        "data_quality": {
            "invalid_return_dates": [days[index] for index in np.flatnonzero(invalid)],
            "large_nav_moves": large_moves,
            "warnings": warnings,
        },
        "provenance": {
            "numerical_library": f"numpy {np.__version__}",
            "actual_availability_cutoff_owner": "service",
            "source_version_selection_owner": "service",
            "input_metadata_retained_in_hash": True,
            "fact_timestamps_rewritten": False,
        },
    }
    if runtime:
        bundle["target_definition"] = json.loads(_canonical(RUNTIME_TARGET_DEFINITION))
        bundle["anchor"].update(_runtime_currency(rows[-1]))
    if horizons is not None:
        bundle.update(
            local_contract=LOCAL_TIMING_CONTRACT, local_contract_hash=LOCAL_RUNTIME_MODEL_HASH,
            target_unit="nav_observations", requested_horizons=list(selected_horizons),
            algorithm_model_hash=RUNTIME_MODEL_HASH, model_hash=LOCAL_RUNTIME_MODEL_HASH,
            model_definition=json.loads(_canonical(LOCAL_RUNTIME_MODEL_DEFINITION)),
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
            chronological, key=lambda row: (row["origin_index"], row["horizon"], row["entry_lag"])
        )
    if include_evaluation:
        bundle["issued_forecasts"] = sorted(
            issued, key=lambda row: (row["origin_index"], row["horizon"], row["entry_lag"])
        )
        bundle["evaluation_rows"] = sorted(
            evaluation,
            key=lambda row: (
                row["origin_index"],
                row["horizon"],
                row["entry_lag"],
            ),
        )
    return bundle

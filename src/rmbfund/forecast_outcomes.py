"""Read-only evidence from actual saved issues, never a historical replay.

The caller supplies compact store projections and, optionally, an already selected
as-of sequence of real NAV dates and availability times. That sequence can locate
an economic window, but cannot create an outcome or a performance observation.
All issues survive in ``issues``. Within each algorithm family, the earliest
issue of a window is chosen before checking whether it has an outcome or a
same-issue historical baseline. Distinct algorithms never share a score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from .domain import ValidationError, canonical_json, timestamp
from .forecast_models import NAV_MODEL_ID, model_family, model_name

RAW_NAV_TARGET = "raw_unit_nav_simple_return_v1"
_LEGACY_TARGET = "raw log NAV[i+entry_lag+horizon] - log NAV[i+entry_lag]"
_NAV_MODELS = {"nav_ridge_empirical_v1", NAV_MODEL_ID}
_METRICS = ("brier_score", "mae", "interval_coverage", "interval_width", "direction_accuracy")


def _target_key(definition: Any, model_id: str | None = None) -> str | None:
    # The old model fits log returns but publishes simple-return distributions.
    # A changed description/contract version is not a new economic target.
    if definition is None:
        return RAW_NAV_TARGET if model_id in _NAV_MODELS else None
    if isinstance(definition, Mapping):
        return str(definition["id"]) if definition.get("id") else canonical_json(definition)
    if definition in (RAW_NAV_TARGET, _LEGACY_TARGET):
        return RAW_NAV_TARGET
    return str(definition) if definition else None


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("真实前向汇总需要原发行的有限数值")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValidationError("真实前向汇总需要原发行的有限数值") from exc
    if not math.isfinite(number):
        raise ValidationError("真实前向汇总需要原发行的有限数值")
    return number


def _distribution(head: Mapping[str, Any]) -> tuple[float, float, float, float]:
    probability, expected, lower, upper = (
        _number(head.get(key)) for key in ("up_probability", "expected_change", "lower", "upper")
    )
    if not 0 <= probability <= 1 or lower > upper:
        raise ValidationError("真实前向汇总的原发行概率或区间无效")
    return probability, expected, lower, upper


def _baseline(item: dict) -> Mapping[str, Any] | None:
    row = item["row"]
    baseline = row["head"].get("historical_baseline")
    if (not isinstance(baseline, Mapping) or baseline.get("status") != "available"
            or baseline.get("horizon") != row["horizon"]
            or baseline.get("entry_lag") != row["entry_lag"]):
        return None
    definition = baseline.get("target_definition_id", baseline.get("target_definition"))
    if definition is not None and _target_key(definition) != item["target_definition"]:
        return None
    try:
        _distribution(baseline)
    except ValidationError:
        return None
    return baseline


def _identical(head: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    # This is identity of saved summaries, not proof of identical latent samples.
    for key in ("up_probability", "down_probability", "flat_probability", "expected_change",
                "median_change", "lower", "upper", "sample_count"):
        left, right = head.get(key), baseline.get(key)
        if left is None or right is None:
            if left != right:
                return False
        elif not math.isclose(_number(left), _number(right), rel_tol=1e-12, abs_tol=1e-12):
            return False
    return True


def _empty_metrics(count: int = 0) -> dict[str, Any]:
    return {"count": count, **dict.fromkeys(_METRICS)}


def _metrics(items: Sequence[dict], *, baseline: bool = False) -> dict[str, Any]:
    if not items:
        return _empty_metrics()
    values = {key: [] for key in _METRICS}
    for item in items:
        head = _baseline(item) if baseline else item["row"]["head"]
        probability, expected, lower, upper = _distribution(head)
        actual = _number(item["outcome"]["actual_change"])
        values["brier_score"].append((probability - int(actual > 0)) ** 2 if baseline else
                                      _number(item["outcome"]["brier_score"]))
        values["mae"].append(abs(expected - actual))
        values["interval_coverage"].append(float(lower <= actual <= upper) if baseline else
                                           float(item["outcome"]["interval_covered"]))
        values["interval_width"].append(upper - lower)
        values["direction_accuracy"].append(float((probability >= 0.5) == (actual > 0)))
    return {"count": len(items), **{
        key: math.fsum(value / len(items) for value in numbers) for key, numbers in values.items()
    }}


def _paired(items: Sequence[dict], *, with_metrics: bool = True) -> dict[str, Any]:
    pairs = [item for item in items if _baseline(item) is not None]
    model = _metrics(pairs) if with_metrics else _empty_metrics(len(pairs))
    baseline = _metrics(pairs, baseline=True) if with_metrics else _empty_metrics(len(pairs))
    identical = sum(_identical(item["row"]["head"], _baseline(item)) for item in pairs)
    return {
        "count": len(pairs), "model": model, "baseline": baseline,
        "metrics_available": with_metrics and bool(pairs),
        "delta": {key: (0.0 if math.isclose(model[key], baseline[key], rel_tol=1e-12, abs_tol=1e-12)
                        else model[key] - baseline[key]) if pairs and with_metrics else None
                  for key in _METRICS},
        "identical_to_baseline_count": identical,
        # Non-identical predictions allow a comparison; this is NOT a count of wins
        # or demonstrated improvements. Fallback copies contribute no increment.
        "incremental_count": len(pairs) - identical,
        "comparison_state": ("not_pooled" if not with_metrics else "no_pairs" if not pairs
                             else "identical_to_baseline" if identical == len(pairs)
                             else "paired_comparison"),
    }


def _series_dates(series: Sequence[Mapping[str, Any]]) -> tuple[list, dict]:
    dates, positions = [], {}
    previous = ""
    for row in series:
        try:
            day = date.fromisoformat(row["date"]).isoformat()
            raw_available = row.get("provenance", {}).get("available_at", row.get("available_at"))
            available = timestamp(raw_available) if raw_available is not None else None
        except (TypeError, KeyError, ValueError) as exc:
            raise ValidationError("窗口定位需要实际净值日期和可用时间") from exc
        if day != row["date"] or day <= previous:
            raise ValidationError("窗口定位需要按日期严格递增的实际净值序列")
        positions[day] = len(dates)
        dates.append((day, available))
        previous = day
    return dates, positions


def _coordinate(item: dict) -> tuple | None:
    row = item["row"]
    anchor = row.get("anchor")
    if item["target_definition"] is None or not isinstance(anchor, Mapping):
        return None
    anchor_date = anchor.get("date")
    if not isinstance(anchor_date, str):
        return None
    return (item["model_family"], row["product_id"], item["target_definition"], row["horizon"],
            anchor_date, row["entry_lag"])


def _window(item: dict, dates: list, positions: dict) -> tuple | None:
    row, outcome, point = item["row"], item["outcome"], item["point"]
    source = None
    if outcome is not None:
        entry, target = outcome["entry"]["date"], outcome["target"]["date"]
        source = "outcome"
    else:
        anchor = positions.get((row.get("anchor") or {}).get("date"))
        if anchor is None:
            return None
        start, end = anchor + row["entry_lag"], anchor + row["entry_lag"] + row["horizon"]
        if end >= len(dates):
            return None
        # Do not drop unavailable intermediate observations and silently shift a
        # horizon to another date. Do not infer a future date from a calendar.
        span = dates[anchor:end + 1]
        if any(available is None or (point is not None and (
                available > point or day > point.date().isoformat())) for day, available in span):
            return None
        entry, target = dates[start][0], dates[end][0]
        source = "series"
    item.update(entry_date=entry, target_date=target, window_source=source)
    if item["target_definition"] is None:
        return None
    return (item["model_family"], row["product_id"], item["target_definition"], row["horizon"],
            entry, target)


def _resolve_shared_windows(items: Sequence[dict]) -> None:
    """An absent current NAV sequence must not promote a later scored refresh.

    An original outcome supplies dates, not a new result, for another issue with
    exactly the same product/target/origin/horizon/lag coordinates. Conflicting
    known windows cannot locate an earlier unrecorded issue; every window it
    could precede stays pending rather than being graded from a later issue.
    This runs after role/model filtering and never maps across those boundaries.
    """
    coordinates = defaultdict(list)
    for item in items:
        coordinate = _coordinate(item)
        if coordinate is not None:
            coordinates[coordinate].append(item)
    uncertain = defaultdict(list)
    for group in coordinates.values():
        originals = [item for item in group
                     if item["outcome"] is not None and item["window_key"] is not None]
        known = {item["window_key"] for item in group if item["window_key"] is not None}
        for item in group:
            if item["outcome"] is not None:
                continue
            point = item["point"]
            peers = [peer for peer in originals if point is None
                     or timestamp(peer["outcome"]["scored_at"]) <= point]
            if len(known) > 1:
                # A current sequence contradicting an original saved window is
                # not permission to attach the missing first result elsewhere.
                item.update(window_key=None, entry_date=None, target_date=None,
                            window_source="conflicting_windows")
                for key in known:
                    uncertain[key].append(item)
            elif item["window_key"] is None:
                if peers:
                    peer = peers[0]
                    item.update(window_key=peer["window_key"], entry_date=peer["entry_date"],
                                target_date=peer["target_date"], window_source="same_origin_outcome")
                else:
                    for key in known:
                        uncertain[key].append(item)
    for item in items:
        predecessors = [peer["row"]["record_id"] for peer in uncertain.get(item["window_key"], ())
                        if (peer["issued"], peer["row"]["record_id"]) <
                        (item["issued"], item["row"]["record_id"])]
        item["pending_predecessors"] = predecessors


def _saved_kind(outcome: Mapping[str, Any] | None, issued) -> str:
    if outcome is None:
        return "unrecorded"
    try:
        entry, target = (date.fromisoformat(outcome[key]["date"]) for key in ("entry", "target"))
        entry_available, target_available = (
            timestamp(outcome[key]["provenance"]["available_at"]) for key in ("entry", "target")
        )
        scored = timestamp(outcome["scored_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("已记录结果缺少实际窗口或可用时间") from exc
    eligible = entry > issued.date() and target >= issued.date() and target_available > issued
    kind = "forward" if eligible else "nowcast"
    if (target <= entry or target > scored.date() or scored < issued
            or max(entry_available, target_available) > scored
            or outcome.get("kind") != kind or outcome.get("eligible_forward") is not eligible):
        raise ValidationError("已记录结果的成熟时间或前向分类不一致")
    return kind


def _audit(item: dict, *, selected: bool) -> dict[str, Any]:
    row = item["row"]
    return {
        **{key: row.get(key) for key in ("record_id", "product_id", "model_id", "model_hash",
                                        "input_hash", "generated_at", "horizon", "entry_lag")},
        "role": item["role"], "model_family": item["model_family"],
        "target_definition": item["target_definition"],
        "entry_date": item.get("entry_date"), "target_date": item.get("target_date"),
        "window_source": item.get("window_source"),
        "window_key": list(item["window_key"]) if item["window_key"] is not None else None,
        "outcome_status": item["kind"], "selected": selected,
        "window_status": ("unknown" if item["window_key"] is None else
                          "pending_earliest_issue" if item.get("pending_predecessors") else "resolved"),
        "pending_predecessors": item.get("pending_predecessors", []),
        "baseline_available": _baseline(item) is not None,
    }


def _nonoverlapping(items: Sequence[dict]) -> int:
    last_target: dict[tuple, str] = {}
    count = 0
    for item in sorted(items, key=lambda value: (
            value["target_date"], value["entry_date"], value["issued"], value["row"]["record_id"])):
        scope = item["window_key"][:4]
        if item["entry_date"] >= last_target.get(scope, ""):
            count += 1
            last_target[scope] = item["target_date"]
    return count


def _summarize(items: Sequence[dict], *, horizon: int, role: str,
               model_id: str | None, family: str | None = None,
               with_metrics: bool = True) -> dict[str, Any]:
    selected: dict[tuple, dict] = {}
    for item in items:  # already ordered by actual issuance, never by result availability
        if item["window_key"] is not None:
            selected.setdefault(item["window_key"], item)
    windows = list(selected.values())
    representatives = [item for item in windows if not item.get("pending_predecessors")]
    forward = [item for item in representatives if item["kind"] == "forward"]
    paired = _paired(forward, with_metrics=with_metrics)
    products = {item["row"]["product_id"] for item in items}
    targets = {item["target_definition"] for item in items}
    return {
        "product_id": next(iter(products)) if len(products) == 1 else None,
        "target_definition": next(iter(targets)) if len(targets) == 1 else None,
        "horizon": horizon, "role": role, "model_id": model_id, "model_family": family,
        "model_name": model_name(model_id),
        "model_ids": sorted({item["row"]["model_id"] for item in items}),
        "metrics_available": with_metrics and bool(forward),
        "counts": {
            "issued": len(items),
            "recorded_forward": sum(item["kind"] == "forward" for item in items),
            "recorded_nowcast": sum(item["kind"] == "nowcast" for item in items),
            "unrecorded": sum(item["kind"] == "unrecorded" for item in items),
            "unknown_window": sum(item["window_key"] is None for item in items),
            "unique_windows": len(windows), "nonoverlapping_windows": _nonoverlapping(windows),
            "unresolved_windows": len(windows) - len(representatives),
            "forward_windows": len(forward),
            "forward_nonoverlapping_windows": _nonoverlapping(forward),
            "paired": sum(_baseline(item) is not None for item in forward),
            "selected_unrecorded": sum(item["kind"] == "unrecorded" for item in representatives),
            "selected_nowcast": sum(item["kind"] == "nowcast" for item in representatives),
        },
        "forward_metrics": _metrics(forward) if with_metrics else _empty_metrics(len(forward)),
        "paired": paired,
        "windows": [_audit(item, selected=not item.get("pending_predecessors")) for item in windows],
        "issues": [_audit(item, selected=selected.get(item["window_key"]) is item
                          and not item.get("pending_predecessors")) for item in items],
        "groups": [],
    }


def summarize_outcomes(
    rows: Sequence[Mapping[str, Any]], *, series: Sequence[Mapping[str, Any]] = (),
    horizon: int = 20, role: str = "primary", model_id: str | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Summarize one role/horizon without fitting, scoring new outcomes or I/O.

    ``rows`` is Store.forecast_evidence_heads output. Its per-row ``as_of`` bounds
    visibility; an optional explicit as_of can only narrow that bound. Unrecorded
    windows remain unrecorded even when series contains every actual date.

    Counts of issued/recorded heads precede economic-window deduplication. Metrics
    use only the earliest issue's recorded strict-forward result per family/window.
    Hash changes, NAV output versions and different lags do not create new evidence
    for the same algorithm/window. Selecting a model selects its whole family.
    ``by_model`` separates algorithm families; ``groups`` separates stress lags and
    product/target scopes within a family. Grouped accuracy metrics are unavailable,
    not pooled; their counts still describe the actual selected windows.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValidationError("真实前向汇总期限必须是正整数")
    if role not in {"anchor", "primary", "stress"}:
        raise ValidationError("真实前向汇总情景类别无效")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValidationError("真实前向汇总模型标识无效")
    cutoff = timestamp(as_of) if as_of is not None else None
    dates, positions = _series_dates(series)
    items, identities = [], set()
    for row in sorted(rows, key=lambda value: (timestamp(value["generated_at"]),
                                              value["record_id"], value["entry_lag"])):
        family = model_family(row.get("model_id"))
        if row.get("horizon") != horizon or (model_id is not None and family != model_family(model_id)):
            continue
        roles = row.get("roles", [row.get("role")])
        if role not in roles:
            continue
        lag, primary = row["entry_lag"], row.get("baseline_entry_lag")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ValidationError("真实前向汇总延迟必须是非负整数")
        if role != "anchor" and (isinstance(primary, bool) or not isinstance(primary, int)
                                 or primary < 0 or (lag == primary) != (role == "primary")):
            continue  # Missing saved assumptions never authorize a guessed primary.
        issued = timestamp(row["generated_at"])
        point = timestamp(row["as_of"]) if row.get("as_of") is not None else None
        if cutoff is not None:
            point = min(point, cutoff) if point is not None else cutoff
        if point is not None and issued > point:
            continue
        identity = (row["product_id"], row["record_id"], horizon, lag)
        if identity in identities:
            continue  # lag-0 aliases are one issue, even in caller-supplied projections.
        identities.add(identity)
        outcome = row.get("outcome")
        if outcome is not None and point is not None and timestamp(outcome["scored_at"]) > point:
            outcome = None
        item = {
            "row": row, "outcome": outcome, "point": point, "issued": issued, "role": role,
            "model_family": family,
            "target_definition": _target_key(row.get("target_definition"), row.get("model_id")),
            "kind": _saved_kind(outcome, issued),
        }
        item["window_key"] = _window(item, dates, positions)
        items.append(item)
    _resolve_shared_windows(items)
    families = defaultdict(list)
    for item in items:
        families[item["model_family"]].append(item)
    by_model = []
    for family, family_items in sorted(families.items(), key=lambda value: value[0] or ""):
        # NAV v1/v2 describe saved output contracts of one algorithm, not contenders.
        family_model_id = NAV_MODEL_ID if family == model_family(NAV_MODEL_ID) else family
        scopes = defaultdict(list)
        for item in family_items:
            key = (item["row"]["product_id"], item["target_definition"],
                   item["row"]["entry_lag"] if role == "stress" else None)
            scopes[key].append(item)
        grouped = len(scopes) > 1 or role == "stress"
        summary = _summarize(family_items, horizon=horizon, role=role, model_id=family_model_id,
                             family=family, with_metrics=not grouped)
        if grouped:
            summary["groups"] = [
                {**_summarize(group, horizon=horizon, role=role, model_id=family_model_id,
                             family=family), "entry_lag": key[2], "metrics_scope": "single_target"}
                for key, group in sorted(scopes.items(), key=lambda value: (
                    value[0][0], value[0][1] or "", value[0][2] if value[0][2] is not None else -1))
            ]
        summary["metrics_scope"] = "groups_only" if grouped else "single_target"
        by_model.append(summary)
    if len(by_model) == 1:
        result = {**by_model[0], "model_id": model_id}
    else:
        result = _summarize(items, horizon=horizon, role=role, model_id=model_id,
                            family=model_family(model_id), with_metrics=not by_model)
        result["metrics_scope"] = "models_only" if by_model else "single_target"
    result["by_model"] = by_model
    result["window_count_definition"] = "distinct algorithm-family/economic-window pairs"
    result["direction_definition"] = (
        "up_probability >= 0.5 predicts up; actual_change > 0 is up; flat is non-up"
    )
    result["delta_definition"] = "model minus same-issue baseline on the identical paired windows"
    result["overlap_note"] = "nonoverlapping windows are not claimed to be independent observations"
    return result

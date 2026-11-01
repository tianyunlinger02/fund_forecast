"""Point-in-time issuer financial arithmetic, not a forecast or a data adapter.

No network, clocks, storage, fitting, scores or probabilities. Inputs use the
existing evidence normalizer/capture contract and investment-profile projection.
Release/revision identifiers identify imported content, not publisher authentication.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, localcontext
from typing import Any

from .advice_evidence import normalize_record
from .domain import ValidationError, decimal, timestamp

SCHEMA_VERSION = 1
__all__ = ["build_issuer_analysis", "issuer_valuation_scenario"]


def _text(value: Decimal) -> str:
    if not value:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _source(record: Mapping, *, locator: str | None = None) -> dict:
    result = {"record_id": record["id"], **{key: record.get(key) for key in (
        "source", "upstream", "source_url", "report_date", "published_at",
        "publication_precision", "observed_at", "available_at",
    )}}
    if locator is not None:
        result["locator"] = locator
    return result


def _available_order(row: Mapping) -> tuple:
    return timestamp(row["available_at"]), row.get("record_id", row.get("id", ""))


def _public_order(row: Mapping) -> tuple | None:
    precision, published = row.get("publication_precision"), row.get("published_at")
    if precision == "timestamp":
        return timestamp(published).date().isoformat(), timestamp(published).isoformat()
    if precision == "date":
        return published, ""
    return None


def _publicly_before(earlier: Mapping, later: Mapping) -> bool:
    """Do not turn today's import order into historical management guidance."""
    a, b = _public_order(earlier), _public_order(later)
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return a[0] < b[0]
    return (earlier.get("publication_precision") == later.get("publication_precision") == "timestamp"
            and timestamp(earlier["published_at"]) < timestamp(later["published_at"]))


def _eligible(record: Mapping, point) -> bool:
    day = point.date().isoformat()
    if (timestamp(record["available_at"]) > point or record["report_date"] > day
            or (record["effective_from"] and record["effective_from"] > day)
            or (record["effective_to"] and record["effective_to"] < day)):
        return False
    if record["publication_precision"] == "timestamp":
        return timestamp(record["published_at"]) <= point
    return record["publication_precision"] != "date" or record["published_at"] <= day


def _period_key(period: Mapping) -> tuple:
    return tuple(period[key] for key in ("start", "end", "basis", "fiscal_year", "fiscal_quarter"))


def _fiscal_index(period: Mapping) -> int:
    return period["fiscal_year"] * 4 + period["fiscal_quarter"]


def _basis_key(metric: Mapping) -> tuple:
    return tuple(metric.get(key) for key in (
        "security_id", "currency", "unit", "accounting_basis", "accounting_basis_detail",
        "eps_basis", "share_basis", "split_basis",
    ))


def _metric_key(metric: Mapping) -> tuple:
    capex, fcf = metric.get("cash_capex", {}), metric.get("cash_flow_definition", {})
    return (metric["name"], metric["category"], _period_key(metric["period"]),
            metric["period_role"], _basis_key(metric), metric.get("comparison_basis"),
            capex.get("definition"), capex.get("sign_convention"), metric.get("debt_definition"),
            metric.get("debt_maturity_scope"),
            tuple(fcf.get(key) for key in ("definition", "ocf_accounting_basis",
                                          "ocf_accounting_basis_detail", "cash_capex_definition")))


def _observation(record: Mapping, metric: Mapping) -> dict:
    payload = record["payload"]
    result = {
        "metric": deepcopy(metric), **_source(record, locator=metric["locator"]),
        **{key: payload[key] for key in ("release_id", "revision_id", "revision_kind")},
    }
    # metric preserves the original accounting scale and value. Convenient derived
    # fields are in currency/fraction units; never apply that scale a second time.
    scale = Decimal(metric["scale"])
    for key in (("value",) if metric["category"] == "actual" else ("low", "high")):
        result[key] = _text(Decimal(metric[key]) * scale)
    return result


def _observation_order(row: Mapping) -> tuple:
    metric, period = row["metric"], row["metric"]["period"]
    return (period["end"] or period["start"], period["start"], metric["name"],
            str(_basis_key(metric)), _available_order(row))


def _financial_observations(records: list[dict]) -> tuple[list[dict], list[dict]]:
    lineages = defaultdict(list)
    for record in records:
        if record["payload"]["kind"] == "financial_release":
            lineages[record["payload"]["release_id"]].append(record)
    observations, revisions = [], []
    for release_id, versions in sorted(lineages.items()):
        versions.sort(key=_available_order)
        selected = versions[-1]
        revisions.append({
            "release_id": release_id, "selected_record_id": selected["id"],
            "selected_revision_id": selected["payload"]["revision_id"],
            "versions": [{**_source(row), "revision_id": row["payload"]["revision_id"],
                          "revision_kind": row["payload"]["revision_kind"],
                          "revises_revision_id": row["payload"].get("revises_revision_id")}
                         for row in versions],
        })
        originals = [row for row in versions if row["payload"]["revision_kind"] == "original"]
        original = min(originals, key=_available_order) if originals else None
        for metric in selected["payload"]["metrics"]:
            observation = _observation(selected, metric)
            observation["original_release"] = _source(original) if original else None
            observations.append(observation)
    # A later release's comparative column is never an original-quarter release.
    # If two originals redundantly cover a quarter, prefer its earliest public
    # release; revisions are selected within their own release lineage above.
    selected_metrics = {}
    for row in observations:
        key = _metric_key(row["metric"])
        if (row["metric"]["category"] == "guidance"
                or (row["metric"]["period"]["basis"] == "instant"
                    and row["metric"]["period_role"] == "comparative")):
            key += (row["release_id"],)
        previous = selected_metrics.get(key)
        original_release = row.get("original_release") or row
        order = (_public_order(original_release)
                 or (original_release["available_at"][:10], original_release["available_at"]),
                 _available_order(original_release))
        if previous is None or order < previous[0]:
            selected_metrics[key] = order, row
    return sorted((item[1] for item in selected_metrics.values()), key=_observation_order), revisions


def _ttm_result(rows: list[dict], *, currency: str | None, share_basis: str | None) -> dict:
    result = {"status": "missing", "value": None,
              "reason": "missing_four_distinct_contiguous_comparable_quarters",
              "currency": currency, "accounting_basis": None, "share_basis": share_basis,
              "split_basis": None, "quarters": []}
    if not rows:
        return result
    # Anchor on the latest actual quarter, not the latest older complete window.
    latest = max(_fiscal_index(row["metric"]["period"]) for row in rows)
    groups = defaultdict(list)
    for row in rows:
        metric = row["metric"]
        if metric.get("eps_basis") == "diluted" and metric["accounting_basis"] != "unknown":
            groups[_basis_key(metric)].append(row)
    windows = []
    for group in groups.values():
        by_quarter = defaultdict(list)
        for row in group:
            by_quarter[_fiscal_index(row["metric"]["period"])].append(row)
        required = list(range(latest - 3, latest + 1))
        if any(len(by_quarter[index]) != 1 for index in required):
            continue
        window = [by_quarter[index][0] for index in required]
        periods = [row["metric"]["period"] for row in window]
        if any(date.fromisoformat(b["start"]) != date.fromisoformat(a["end"]) + timedelta(days=1)
               for a, b in zip(periods, periods[1:])):
            continue
        # Four fiscal labels with gapped/overlapping dates do not make a TTM.
        if not 350 <= (date.fromisoformat(periods[-1]["end"]) - date.fromisoformat(periods[0]["start"])).days + 1 <= 380:
            continue
        windows.append(window)
    if not windows:
        return result
    compatible = [window for window in windows
                  if (currency is None or window[0]["metric"]["currency"] == currency)
                  and (share_basis is None or window[0]["metric"]["share_basis"] == share_basis)]
    if not compatible:
        result["reason"] = "eps_mapping_currency_or_share_basis_mismatch"
        return result
    # Prefer an actual GAAP/IFRS earnings basis over a non-GAAP alternative. Two
    # incompatible candidates of the same priority are not silently combined.
    priority = min(1 if window[0]["metric"]["accounting_basis"] == "non_GAAP" else 0
                   for window in compatible)
    compatible = [window for window in compatible
                  if (1 if window[0]["metric"]["accounting_basis"] == "non_GAAP" else 0) == priority]
    if len(compatible) != 1:
        result["reason"] = "ambiguous_comparable_eps_basis"
        return result
    window = compatible[0]
    metric = window[-1]["metric"]
    result.update(status="available", value=_text(sum((Decimal(row["value"]) for row in window), Decimal(0))),
                  reason=None, currency=metric["currency"], accounting_basis=metric["accounting_basis"],
                  share_basis=metric["share_basis"], split_basis=metric["split_basis"], quarters=window)
    return result


def _latest_price(price_records: list[dict], security_id: str, mapping: Mapping | None) -> tuple[dict | None, str | None]:
    records = [record for record in price_records if record["payload"]["security_id"] == security_id]
    if mapping is not None:
        payload = mapping["payload"]
        matched = [record for record in records if all(record["payload"][key] == payload[other]
                   for key, other in (("symbol", "symbol"), ("exchange", "exchange"), ("currency", "quote_currency")))]
        if records and not matched:
            return None, "price_mapping_security_symbol_exchange_or_currency_mismatch"
        records = matched
    candidates = []
    for record in records:
        payload = record["payload"]
        close = payload["closes"][-1]
        candidates.append({**deepcopy(close), "security_id": security_id,
                           "symbol": payload["symbol"], "exchange": payload["exchange"],
                           "currency": payload["currency"], "price_basis": deepcopy(payload["price_basis"]),
                           **_source(record, locator=payload["price_basis"]["locator"])})
    if not candidates:
        return None, "missing_latest_matching_close"
    # Adjustment uncertainty on the newest quote is visible, not hidden by
    # silently falling back to an older quote with a convenient basis.
    return max(candidates, key=lambda row: (row["date"], _available_order(row))), None


def _security_result(security_id: str, mapping: dict | None, actuals: list[dict], prices: list[dict]) -> dict:
    payload = mapping["payload"] if mapping else {}
    eps = [row for row in actuals if row["metric"]["name"] == "eps"
           and row["metric"].get("security_id") == security_id]
    ttm = _ttm_result(eps, currency=payload.get("quote_currency"), share_basis=payload.get("share_basis"))
    close, reason = _latest_price(prices, security_id, mapping)
    if mapping is None:
        reason = "missing_security_mapping"
    elif ttm["status"] != "available":
        reason = ttm["reason"]
    elif close is not None and reason is None:
        if close["price_basis"]["kind"] not in {"unadjusted", "split_adjusted"}:
            reason = "price_adjustment_basis_unknown"
        elif close["currency"] != ttm["currency"]:
            reason = "eps_price_currency_mismatch"
        elif close["price_basis"]["split_basis"] != ttm["split_basis"]:
            reason = "eps_price_split_basis_mismatch"
    pe = earnings_yield = None
    status = "missing"
    if reason is None:
        if Decimal(ttm["value"]) <= 0:
            status, reason = "nonpositive_eps", "nonpositive_ttm_eps_no_pe"
        else:
            status = "available"
            pe = _text(Decimal(close["close"]) / Decimal(ttm["value"]))
            earnings_yield = _text(Decimal(ttm["value"]) / Decimal(close["close"]))
    return {
        "security_id": security_id, "symbol": payload.get("symbol"), "exchange": payload.get("exchange"),
        "currency": payload.get("quote_currency"), "share_basis": payload.get("share_basis"),
        "mapping": mapping, "latest_close": close, "ttm_diluted_eps": ttm,
        "pe_ttm": pe, "earnings_yield_ttm": earnings_yield,
        "valuation_status": status, "valuation_reason": reason,
    }


def _eps_yoy(actuals: list[dict]) -> list[dict]:
    rows = [row for row in actuals if row["metric"]["name"] == "eps"]
    results = []
    for current in rows:
        metric, period = current["metric"], current["metric"]["period"]
        prior = [row for row in rows if _basis_key(row["metric"]) == _basis_key(metric)
                 and row["metric"]["period"]["fiscal_year"] == period["fiscal_year"] - 1
                 and row["metric"]["period"]["fiscal_quarter"] == period["fiscal_quarter"]]
        previous = prior[0] if len(prior) == 1 else None
        a, b = Decimal(current["value"]), Decimal(previous["value"]) if previous else None
        reason = ("missing_or_incompatible_prior_year_original_quarter" if previous is None else
                  "nonpositive_prior_eps_growth_not_meaningful" if b <= 0 else None)
        results.append({
            "period": deepcopy(period), "security_id": metric["security_id"],
            "status": "available" if reason is None else "missing", "reason": reason,
            "current_eps": _text(a), "prior_eps": _text(b) if b is not None else None,
            "change": _text(a / b - 1) if reason is None else None,
            "absolute_change": _text(a - b) if b is not None else None,
            "current": current, "prior": previous,
        })
    return results


def _target_matches(actual: Mapping, guidance: Mapping) -> bool:
    a, b = actual["period"], guidance["period"]
    if any(a[key] != b[key] for key in ("start", "basis", "fiscal_year", "fiscal_quarter")):
        return False
    return b["end"] is None or a["end"] == b["end"]


def _revenue_basis_matches(actual: Mapping, guidance: Mapping) -> bool:
    if any(actual[key] != guidance[key] for key in ("currency", "unit")):
        return False
    if actual["accounting_basis"] == guidance["accounting_basis"] != "unknown":
        return actual.get("accounting_basis_detail") == guidance.get("accounting_basis_detail")
    # Narrow documented metric identity, not a blanket unknown-accounting bypass.
    return (guidance["accounting_basis"] == "unknown" and actual["accounting_basis"] != "unknown"
            and bool(actual.get("comparison_basis"))
            and actual["comparison_basis"] == guidance.get("comparison_basis"))


def _revenue_comparisons(actuals: list[dict], guidance: list[dict]) -> list[dict]:
    result = []
    for actual in [row for row in actuals if row["metric"]["name"] == "revenue"]:
        # Only original, earlier public guidance. A later revision imported today
        # cannot manufacture a beat against a backfilled management expectation.
        candidates = [row for row in guidance if row["metric"]["name"] == "revenue"
                      and row["revision_kind"] == "original"
                      and _target_matches(actual["metric"], row["metric"])
                      and _revenue_basis_matches(actual["metric"], row["metric"])
                      and _publicly_before(row, actual.get("original_release") or actual)]
        prior = max(candidates, key=lambda row: (_public_order(row), _available_order(row))) if candidates else None
        row = {"period": deepcopy(actual["metric"]["period"]), "status": "available" if prior else "missing",
               "actual": actual, "prior_guidance": prior, "actual_value": actual["value"],
               "guidance_low": None, "guidance_high": None, "guidance_midpoint": None,
               "midpoint_error": None, "midpoint_error_ratio": None, "range_position": None,
               "comparison_caveat": None,
               "reason": None if prior else "missing_prior_original_public_guidance_same_target_currency_and_basis"}
        if prior:
            value, low, high = Decimal(actual["value"]), Decimal(prior["low"]), Decimal(prior["high"])
            midpoint = (low + high) / 2
            row.update(guidance_low=_text(low), guidance_high=_text(high), guidance_midpoint=_text(midpoint),
                       midpoint_error=_text(value - midpoint),
                       midpoint_error_ratio=_text((value - midpoint) / abs(midpoint)) if midpoint else None,
                       range_position="below" if value < low else "above" if value > high else "within")
            if prior["metric"]["accounting_basis"] == "unknown":
                row["comparison_caveat"] = "guidance_accounting_unknown; explicit_same_publisher_revenue_metric_definition_only"
        result.append(row)
    return result


def _latest_guidance(guidance: list[dict]) -> list[dict]:
    if not guidance:
        return []
    latest = max(guidance, key=lambda row: (_public_order(row) or ("", ""), _available_order(row)))
    return [row for row in guidance if row["record_id"] == latest["record_id"]]


def _same_financial_period(a: Mapping, b: Mapping) -> bool:
    if a["basis"] == b["basis"] == "instant":
        return a["end"] == b["end"]  # Fiscal labels do not change a dated snapshot.
    return _period_key(a) == _period_key(b)


def _financial_input(anchor: Mapping, rows: list[dict], name: str,
                     *, period: Mapping | None = None) -> tuple[dict | None, str | None]:
    """Exact flow windows or same-day balances; scales are already normalized."""
    metric = anchor["metric"]
    target = period or metric["period"]
    if metric["period"]["basis"] == "instant" and target["basis"] != "instant":
        fiscal_quarter = metric["period"]["fiscal_quarter"]
        flow_quarter = 4 if target["basis"] == "year" else target["fiscal_quarter"]
        if (metric["period"]["fiscal_year"] != target["fiscal_year"]
                or (fiscal_quarter is not None and flow_quarter is not None and fiscal_quarter != flow_quarter)):
            return None, "incompatible_balance_and_revenue_fiscal_labels"
    candidates = [row for row in rows if row["metric"]["name"] == name
                  and _same_financial_period(row["metric"]["period"], target)]
    if not candidates:
        return None, f"missing_same_period_{name}"
    if metric["accounting_basis"] == "unknown":
        return None, "unknown_accounting_basis"
    matched = [row for row in candidates if _basis_key(row["metric"]) == _basis_key(metric)]
    if not matched:
        return None, f"incompatible_currency_unit_security_or_accounting_basis_{name}"
    if len(matched) != 1:
        return None, f"ambiguous_same_period_{name}"
    return matched[0], None


def _disclosed_fcf_input(ocf: Mapping, capex: Mapping | None,
                         actuals: list[dict]) -> tuple[dict | None, str | None, str | None]:
    fcf, reason = _financial_input(ocf, actuals, "free_cash_flow")
    if reason is None:
        return fcf, None, "same_reported_accounting_basis"
    if (not reason.startswith("incompatible_") or capex is None
            or not capex["metric"].get("cash_capex")):
        return None, reason, None
    metric, definition = ocf["metric"], capex["metric"]["cash_capex"]["definition"]
    bridged = []
    for row in actuals:
        other = row["metric"]
        bridge = other.get("cash_flow_definition", {})
        if (other["name"] == "free_cash_flow"
                and _same_financial_period(metric["period"], other["period"])
                and all(metric.get(key) == other.get(key) for key in ("currency", "unit", "security_id"))
                and bridge.get("ocf_accounting_basis") == metric["accounting_basis"]
                and bridge.get("ocf_accounting_basis_detail") == metric.get("accounting_basis_detail")
                and bridge.get("cash_capex_definition") == definition):
            bridged.append(row)
    if len(bridged) == 1:
        return bridged[0], None, "defined_basis_bridge"
    return None, "ambiguous_explicit_FCF_definition_bridge" if bridged else reason, None


def _financial_check(anchor: Mapping, name: str, inputs: dict, *, formula: str,
                     value: Decimal | None = None, reason: str | None = None,
                     unit: str = "fraction", period: Mapping | None = None, **details) -> dict:
    metric = anchor["metric"]
    if not any(row is anchor for row in inputs.values()):
        inputs = {"basis": anchor, **inputs}
    anchor_key = next(key for key, row in inputs.items() if row is anchor)
    return {
        "name": name, "anchor": anchor_key, "period": deepcopy(period or metric["period"]),
        "status": "missing" if reason else "available", "reason": reason,
        "value": _text(value) if value is not None else None, "unit": unit,
        "currency": metric["currency"] if unit == "currency" else None,
        "input_currency": metric["currency"], "accounting_basis": metric["accounting_basis"],
        "accounting_basis_detail": metric.get("accounting_basis_detail"),
        "comparison_basis": metric.get("comparison_basis"),
        "formula": formula, "inputs": inputs, **details,
    }


def _ratio_check(anchor: Mapping, name: str, inputs: dict, numerator: Decimal | None,
                 denominator: Mapping | None, reason: str | None, *, formula: str,
                 period: Mapping | None = None, **details) -> dict:
    if reason is None and Decimal(denominator["value"]) == 0:
        reason = "zero_denominator"
    value = numerator / Decimal(denominator["value"]) if reason is None else None
    return _financial_check(anchor, name, inputs, formula=formula, value=value,
                            reason=reason, period=period, **details)


def _cash_flow_checks(actuals: list[dict]) -> list[dict]:
    """No annualization, YTD subtraction, guessed capex signs or FCFF identity."""
    checks, bridged_fcfs = [], set()
    for ocf in [row for row in actuals if row["metric"]["name"] == "operating_cash_flow"]:
        income, income_reason = _financial_input(ocf, actuals, "net_income")
        checks.append(_ratio_check(
            ocf, "ocf_to_net_income", {"operating_cash_flow": ocf, "net_income": income},
            Decimal(ocf["value"]), income, income_reason,
            formula="operating_cash_flow / net_income (same explicit flow period)",
            caveat="signed_loss_ratio_not_a_cash_conversion_quality_score"
            if income and Decimal(income["value"]) < 0 else None,
        ))
        capex, capex_reason = _financial_input(ocf, actuals, "capex")
        definition = capex["metric"].get("cash_capex") if capex else None
        if capex_reason is None and definition is None:
            capex_reason = "missing_explicit_cash_capex_definition_and_sign_convention"
        cash_outflow = residual = None
        if capex_reason is None:
            direction = 1 if definition["sign_convention"] == "outflow_positive" else -1
            cash_outflow = Decimal(capex["value"]) * direction
            residual = Decimal(ocf["value"]) - cash_outflow
        inputs = {"operating_cash_flow": ocf, "capex": capex}
        checks.append(_financial_check(
            ocf, "ocf_minus_cash_capex", inputs,
            formula="operating_cash_flow - cash_capex_outflow; outflow_negative is negated, never abs",
            value=residual, reason=capex_reason, unit="currency",
            cash_capex_outflow=_text(cash_outflow) if cash_outflow is not None else None,
            caveat="residual_after_explicit_cash_capex_not_FCFF_or_automatically_publisher_FCF",
        ))
        fcf, fcf_reason, comparison_mode = _disclosed_fcf_input(ocf, capex, actuals)
        difference_reason = capex_reason or fcf_reason
        checks.append(_financial_check(
            ocf, "ocf_minus_cash_capex_vs_disclosed_fcf", {**inputs, "free_cash_flow": fcf},
            formula="(operating_cash_flow - cash_capex_outflow) - disclosed_free_cash_flow",
            value=residual - Decimal(fcf["value"]) if difference_reason is None else None,
            reason=difference_reason, unit="currency", comparison_mode=comparison_mode,
            caveat="difference_only_not_proof_of_error_or_identical_FCF_definitions",
        ))
        revenue, revenue_reason = _financial_input(ocf, actuals, "revenue")
        checks.append(_ratio_check(
            ocf, "ocf_margin", {"operating_cash_flow": ocf, "revenue": revenue},
            Decimal(ocf["value"]), revenue, revenue_reason,
            formula="operating_cash_flow / revenue (same explicit flow period)",
        ))
        checks.append(_ratio_check(
            ocf, "ocf_minus_cash_capex_margin", {**inputs, "revenue": revenue}, residual,
            revenue, capex_reason or revenue_reason,
            formula="(operating_cash_flow - cash_capex_outflow) / revenue (same explicit flow period)",
            caveat="not_FCFF_margin",
        ))
        if fcf and comparison_mode == "defined_basis_bridge":
            checks.append(_ratio_check(
                fcf, "disclosed_fcf_margin", {"free_cash_flow": fcf, "revenue": revenue,
                                              "operating_cash_flow": ocf, "capex": capex},
                Decimal(fcf["value"]), revenue, revenue_reason or capex_reason,
                formula="disclosed_free_cash_flow / revenue (explicit OCF/cash-capex definition bridge)",
                comparison_mode=comparison_mode,
                caveat="publisher_FCF_measure_not_assumed_FCFF; original_accounting_bases_retained",
            ))
            bridged_fcfs.add((fcf["record_id"], _metric_key(fcf["metric"])))
    for fcf in [row for row in actuals if row["metric"]["name"] == "free_cash_flow"
                and (row["record_id"], _metric_key(row["metric"])) not in bridged_fcfs]:
        revenue, reason = _financial_input(fcf, actuals, "revenue")
        checks.append(_ratio_check(
            fcf, "disclosed_fcf_margin", {"free_cash_flow": fcf, "revenue": revenue},
            Decimal(fcf["value"]), revenue, reason,
            formula="disclosed_free_cash_flow / revenue (same explicit flow period)",
            caveat="publisher_FCF_measure_not_assumed_FCFF",
        ))
    return checks


def _balance_sheet_checks(actuals: list[dict]) -> list[dict]:
    balances = [row for row in actuals if row["metric"]["name"] in {
        "cash_and_cash_equivalents", "total_debt", "short_term_debt", "long_term_debt",
    }]
    anchors = {}
    for row in balances:
        anchors.setdefault((row["metric"]["period"]["end"], _basis_key(row["metric"])), row)
    checks = []
    for anchor in anchors.values():
        cash, cash_reason = _financial_input(anchor, actuals, "cash_and_cash_equivalents")
        debt, debt_reason = _financial_input(anchor, actuals, "total_debt")
        inputs = {"cash_and_cash_equivalents": cash, "total_debt": debt}
        debt_total = Decimal(debt["value"]) if debt else None
        debt_basis = "reported_total" if debt else None
        # Only an absent total can be replaced by the explicitly disclosed,
        # non-overlapping short/current and long/noncurrent interest-bearing debt.
        if debt_reason == "missing_same_period_total_debt":
            short, short_reason = _financial_input(anchor, actuals, "short_term_debt")
            long, long_reason = _financial_input(anchor, actuals, "long_term_debt")
            inputs.update(short_term_debt=short, long_term_debt=long)
            debt_reason = short_reason or long_reason
            if debt_reason is None and (short["metric"]["debt_maturity_scope"] != "current"
                                        or long["metric"]["debt_maturity_scope"] != "noncurrent"):
                debt_reason = "debt_components_not_explicitly_nonoverlapping_current_and_noncurrent"
            if debt_reason is None:
                debt_total = Decimal(short["value"]) + Decimal(long["value"])
                debt_basis = "explicit_short_plus_long"
        reason = cash_reason or debt_reason
        checks.append(_financial_check(
            anchor, "net_cash", inputs,
            formula="cash_and_cash_equivalents - interest_bearing_debt; reported total OR explicit short + long (same date)",
            value=Decimal(cash["value"]) - debt_total if reason is None else None,
            reason=reason, unit="currency", debt_basis=debt_basis,
            interest_bearing_debt_total=_text(debt_total) if debt_total is not None else None,
            caveat="cash_equivalents_only; debt_scope_is_the_explicit_disclosed_debt_definition",
        ))
    return checks


def _prior_balance(current: Mapping, observations: list[dict]) -> tuple[dict | None, str | None]:
    metric = current["metric"]
    candidates = [row for row in observations if row["metric"]["category"] == "actual"
                  and row["metric"]["name"] == metric["name"]
                  and row["metric"]["period"]["end"] < metric["period"]["end"]]
    # The current release's comparable column can disclose restated balances.
    comparative = [row for row in candidates if row["release_id"] == current["release_id"]
                   and row["metric"]["period_role"] == "comparative"]
    candidates = comparative or candidates
    if not candidates:
        return None, "missing_prior_instant_balance"
    latest = max(row["metric"]["period"]["end"] for row in candidates)
    candidates = [row for row in candidates if row["metric"]["period"]["end"] == latest]
    if not metric.get("comparison_basis"):
        return None, "missing_explicit_balance_comparison_basis"
    if metric["accounting_basis"] == "unknown":
        return None, "unknown_accounting_basis"
    matched = [row for row in candidates if _basis_key(row["metric"]) == _basis_key(metric)
               and row["metric"].get("comparison_basis") == metric["comparison_basis"]]
    if not matched:
        return None, "incompatible_prior_balance_currency_accounting_or_comparison_basis"
    # Prefer the original observation if a different release repeats the same
    # prior date as a comparative. Conflicting candidates of that role stay missing.
    primary = [row for row in matched if row["metric"]["period_role"] == "primary"]
    matched = primary or matched
    if len(matched) != 1:
        return None, "ambiguous_prior_instant_balance"
    return matched[0], None


def _working_capital_checks(actuals: list[dict], observations: list[dict]) -> list[dict]:
    checks = []
    balances = [row for row in actuals if row["metric"]["name"] in {"accounts_receivable", "inventory"}]
    for current in balances:
        metric = current["metric"]
        previous, reason = _prior_balance(current, observations)
        change = Decimal(current["value"]) - Decimal(previous["value"]) if reason is None else None
        ratio_reason = reason or ("zero_prior_balance" if Decimal(previous["value"]) == 0 else None)
        checks.append(_financial_check(
            current, "working_capital_change", {"current": current, "prior": previous},
            formula="current_balance - comparable_prior_balance; change_ratio = current / prior - 1",
            value=change, reason=reason, unit="currency", metric_name=metric["name"],
            comparison_basis=metric.get("comparison_basis"),
            comparison_period=deepcopy(previous["metric"]["period"]) if previous else None,
            change_ratio=_text(Decimal(current["value"]) / Decimal(previous["value"]) - 1)
            if ratio_reason is None else None,
            change_ratio_status="missing" if ratio_reason else "available", change_ratio_reason=ratio_reason,
            caveat="change_between_the_two_explicit_dates_not_automatically_YoY_or_cash_flow",
        ))
        revenues = [row for row in actuals if row["metric"]["name"] == "revenue"
                    and row["metric"]["period"]["end"] == metric["period"]["end"]]
        periods = {_period_key(row["metric"]["period"]): row["metric"]["period"] for row in revenues}
        if not periods:
            checks.append(_financial_check(
                current, "working_capital_to_revenue", {"balance": current, "revenue": None},
                formula="ending_balance / revenue_for_the_explicit_flow_period",
                reason="missing_revenue_flow_ending_at_balance_date", metric_name=metric["name"],
            ))
        for period in periods.values():
            revenue, revenue_reason = _financial_input(current, actuals, "revenue", period=period)
            checks.append(_ratio_check(
                current, "working_capital_to_revenue", {"balance": current, "revenue": revenue},
                Decimal(current["value"]), revenue, revenue_reason,
                formula="ending_balance / revenue_for_the_explicit_flow_period",
                period=period, metric_name=metric["name"], balance_period=deepcopy(metric["period"]),
                caveat="ending_stock_to_period_sales_not_turnover_days_or_annualized_sales",
            ))
    return checks


def _profile_context(profile: Mapping, point) -> tuple[dict, dict, dict]:
    if not isinstance(profile, Mapping):
        raise ValidationError("必须提供现有build_investment_profile结果")
    portfolio = profile.get("portfolio") or {}
    same_point = bool(profile.get("as_of") and timestamp(profile["as_of"]) == point)
    eligible = bool(portfolio and same_point and timestamp(portfolio["available_at"]) <= point
                    and portfolio["report_date"] <= point.date().isoformat())
    payload = portfolio.get("payload", {}) if eligible else {}
    weights, names = {}, {}
    for holding in payload.get("holdings", []):
        if holding["kind"] == "corporate_equity":
            issuer_id = holding["issuer_id"]
            weights[issuer_id] = weights.get(issuer_id, Decimal(0)) + Decimal(holding["weight"])
            names[issuer_id] = holding["name"]
    disclosed = sum(weights.values(), Decimal(0))
    listed = sum((Decimal(row["weight"]) for row in payload.get("holdings", [])), Decimal(0))
    coverage = {
        "denominator": payload.get("denominator"), "holdings_scope": payload.get("holdings_scope"),
        "disclosed_corporate_equity_weight": _text(disclosed) if eligible else None,
        "analyzed_disclosed_weight": "0" if eligible else None, "missing_issuer_ids": [],
        "outside_disclosed_holdings_weight": _text(1 - listed) if eligible else None,
        "uncovered_disclosed_corporate_equity_weight": _text(disclosed) if eligible else None,
        "profile_matches_as_of": same_point,
        "coverage_basis": "selected_historical_disclosure_not_current_holdings" if eligible else "missing_matching_selected_profile",
    }
    reference = {"as_of": profile.get("as_of"), "portfolio_record_id": portfolio.get("record_id"),
                 "portfolio_report_date": portfolio.get("report_date"), "matching_analysis_as_of": same_point}
    exposures = {issuer_id: {
        "weight": _text(weight), "denominator": payload["denominator"], "record_id": portfolio["record_id"],
        "report_date": portfolio["report_date"], "available_at": portfolio["available_at"],
        "profile_as_of": profile["as_of"], "profile_product_id": profile.get("product_id"),
        "holdings_scope": payload["holdings_scope"], "current_position_verified": False,
    } for issuer_id, weight in weights.items()}
    return coverage, reference, {key: {"name": names[key], "exposure": exposures[key]} for key in weights}


def build_issuer_analysis(records: Sequence[Mapping], *, profile: Mapping, as_of: str) -> dict:
    """Select only evidence actually available by an explicit aware timestamp.

    ``quarter_actuals`` is a flat observation list. Original metric values/scales
    remain in ``metric``; observation ``value``/``low``/``high`` are scaled numeric
    strings. Comparative columns are retained separately, never substituted for
    original quarterly EPS. Product IDs must match the supplied existing profile.
    """
    point = timestamp(as_of)
    by_record_id, product_ids = {}, set()
    for row in records:
        record = normalize_record({key: value for key, value in row.items() if key != "checked_at"})
        product_ids.add(record["product_id"])
        if record["family"] == "issuer_fundamentals" and _eligible(record, point):
            previous = by_record_id.get(record["id"])
            if previous is None or _available_order(record) < _available_order(previous):
                by_record_id[record["id"]] = record
    normalized = list(by_record_id.values())
    if len(product_ids) > 1 or (product_ids and profile.get("product_id") not in product_ids):
        raise ValidationError("发行人资料必须与投资画像属于同一基金产品")
    coverage, reference, disclosed = _profile_context(profile, point)
    prices = [row for row in normalized if row["payload"]["kind"] == "price_history"]
    by_issuer, names = defaultdict(list), {}
    for row in sorted(normalized, key=_available_order):
        payload = row["payload"]
        if payload["kind"] != "price_history":
            by_issuer[payload["issuer_id"]].append(row)
            names[payload["issuer_id"]] = payload["issuer_name"]
    # A security may not be attributed to different issuers using the same ID.
    owners = defaultdict(set)
    for issuer_id, rows in by_issuer.items():
        for row in rows:
            payload = row["payload"]
            securities = ([payload["security_id"]] if payload.get("security_id") else [])
            securities += [metric["security_id"] for metric in payload.get("metrics", []) if metric.get("security_id")]
            for security_id in securities:
                owners[security_id].add(issuer_id)
    if any(len(value) > 1 for value in owners.values()):
        raise ValidationError("同一证券标识不能映射到不同发行人")
    issuers, covered, missing = [], Decimal(0), []
    with localcontext() as context:
        context.prec = 64
        for issuer_id in sorted(set(by_issuer) | set(disclosed)):
            rows = by_issuer[issuer_id]
            observations, revisions = _financial_observations(rows)
            primary_actuals = [row for row in observations if row["metric"]["category"] == "actual"
                               and row["metric"]["period_role"] == "primary"]
            flow_actuals = [row for row in primary_actuals if row["metric"]["period"]["basis"] != "instant"]
            actuals = [row for row in flow_actuals if row["metric"]["period"]["basis"] == "quarter"]
            guidance = [row for row in observations if row["metric"]["category"] == "guidance"
                        and row["metric"]["period_role"] == "primary"]
            original_guidance = [_observation(record, metric) for record in rows
                                 if record["payload"]["kind"] == "financial_release"
                                 and record["payload"]["revision_kind"] == "original"
                                 for metric in record["payload"]["metrics"]
                                 if metric["category"] == "guidance" and metric["period_role"] == "primary"]
            mappings = {}
            for row in rows:
                if row["payload"]["kind"] == "security_mapping":
                    key = row["payload"]["security_id"]
                    previous = mappings.get(key)
                    if previous is None or (row["report_date"], _available_order(row)) > (previous["report_date"], _available_order(previous)):
                        mappings[key] = {"payload": deepcopy(row["payload"]),
                                         **_source(row, locator=row["payload"]["locator"])}
            security_ids = {key for key, value in owners.items() if issuer_id in value}
            securities = [_security_result(key, mappings.get(key), actuals, prices) for key in sorted(security_ids)]
            available = any(row["valuation_status"] == "available" for row in securities)
            status = "available" if available else "partial" if rows else "missing"
            if issuer_id in disclosed:
                if available:
                    covered += Decimal(disclosed[issuer_id]["exposure"]["weight"])
                else:
                    missing.append(issuer_id)
            issuers.append({
                "issuer_id": issuer_id, "issuer_name": names.get(issuer_id, disclosed.get(issuer_id, {}).get("name")),
                "status": status, "disclosed_exposure": deepcopy(disclosed.get(issuer_id, {}).get("exposure")),
                "quarter_actuals": [row for row in actuals if row["metric"]["name"] in {"eps", "revenue", "gross_margin"}],
                "eps_yoy": _eps_yoy(actuals), "revenue_vs_prior_guidance": _revenue_comparisons(actuals, original_guidance),
                "latest_guidance": _latest_guidance(guidance),
                "capex": [row for row in observations if row["metric"]["name"] == "capex"],
                "cash_flow_checks": _cash_flow_checks(flow_actuals),
                "balance_sheet_checks": _balance_sheet_checks(primary_actuals),
                "working_capital_checks": _working_capital_checks(primary_actuals, observations),
                "financial_observations": observations, "securities": securities, "revision_selection": revisions,
            })
    if coverage["analyzed_disclosed_weight"] is not None:
        coverage["analyzed_disclosed_weight"] = _text(covered)
        coverage["uncovered_disclosed_corporate_equity_weight"] = _text(Decimal(coverage["disclosed_corporate_equity_weight"]) - covered)
    coverage["missing_issuer_ids"] = missing
    status = ("available" if issuers and all(row["status"] == "available" for row in issuers)
              else "partial" if normalized else "missing")
    return deepcopy({
        "schema_version": SCHEMA_VERSION, "kind": "point_in_time_issuer_fundamentals",
        "product_id": profile.get("product_id"), "as_of": point.isoformat(), "status": status,
        "profile_reference": reference, "coverage": coverage, "issuers": issuers,
        "unmapped_price_security_ids": sorted({row["payload"]["security_id"] for row in prices} - owners.keys()),
        "limitations": [
            "这是实际已取得资料的逐发行人计算，不是预测、概率、投资建议或你的持仓。",
            "只取as_of前实际可用且已公开的记录；原公告比较期不能代替当时原始季度公告。修订按实际可用时间选择。",
            "TTM须有四个不同、连续、同证券/币种/会计/每股及拆股口径的季度实际稀释EPS；缺失不是零，不用单季乘四或相减累计EPS。",
            "财务流量只按相同起止日、quarter/ytd/year等原期间及币种/单位/会计口径计算，倍率先转为原币单位；不将累计数冒充单季或年化。",
            "现金资本开支必须有原文现金支出定义及符号方向；无口径的旧capex只展示，OCF-capex不是FCFF。披露FCF若不同会计标签，须显式核对原文OCF及capex定义桥接并保留双方口径。",
            "净现金只减同日有息债务，不用总负债或缺项补零；应收/存货变化须同原文可比定义，期末余额占收入明确对应收入期间，不是周转天数。",
            "普通股与ADS不换算；来源调整未知的价格不能用于可比P/E，也不被当作含股息总回报。",
            "权重只对应同一as_of的选定历史基金披露，不代表今天持仓；未知部分不补齐，不与行业、整体股票或衍生品重复累计。",
            "release/revision标识仅识别导入文件内容，非发行人认证；原文、采集及定位沿用已有证据合同。",
        ],
    })


def issuer_valuation_scenario(
    analysis: Mapping, *, issuer_id: str, earnings_change: Any = None,
    terminal_multiple: Any, label: str = "caller_supplied_condition",
) -> dict:
    """Required/conditional EPS and local price, from the exact saved analysis.

    No default earnings growth or probability. Required growth is the earnings
    needed to maintain the selected price at the supplied terminal multiple.
    """
    if not isinstance(label, str) or not label.strip() or len(label) > 2000:
        raise ValidationError("情景须有明确有界的假设说明")
    issuers = [row for row in analysis.get("issuers", []) if row.get("issuer_id") == issuer_id]
    if len(issuers) != 1:
        raise ValidationError("分析中找不到唯一发行人")
    eligible = [row for row in issuers[0]["securities"] if row["valuation_status"] == "available"]
    if len(eligible) != 1:
        raise ValidationError("情景要求唯一的正TTM EPS与匹配价格证券；普通股/ADS不能默选")
    security = eligible[0]
    base_eps, price = Decimal(security["ttm_diluted_eps"]["value"]), Decimal(security["latest_close"]["close"])
    multiple = decimal(terminal_multiple)
    if base_eps <= 0 or price <= 0 or multiple <= 0:
        raise ValidationError("情景要求正的可比TTM EPS、价格与终值估值倍数")
    growth = decimal(earnings_change) if earnings_change is not None else None
    if growth is not None and growth < -1:
        raise ValidationError("盈利条件变化不能低于-100%")
    with localcontext() as context:
        context.prec = 64
        required_eps = price / multiple
        required_growth = required_eps / base_eps - 1
        conditional_eps = base_eps * (1 + growth) if growth is not None else None
        conditional_price = conditional_eps * multiple if conditional_eps is not None else None
        price_change = conditional_price / price - 1 if conditional_price is not None else None
        base_pe = price / base_eps
    exposure = issuers[0].get("disclosed_exposure")
    if exposure and (exposure.get("profile_product_id") != analysis.get("product_id")
                     or timestamp(exposure["profile_as_of"]) != timestamp(analysis["as_of"])
                     or exposure.get("record_id") != analysis.get("profile_reference", {}).get("portfolio_record_id")):
        exposure = None
    return deepcopy({
        "schema_version": SCHEMA_VERSION, "kind": "issuer_valuation_conditional_scenario",
        "label": label, "assumption_source": "caller_supplied_not_a_forecast",
        "as_of": analysis["as_of"], "product_id": analysis["product_id"],
        "issuer_id": issuer_id, "security_id": security["security_id"], "currency": security["currency"],
        "base_eps": _text(base_eps), "base_price": _text(price), "base_pe": _text(base_pe),
        "terminal_multiple": _text(multiple), "required_eps": _text(required_eps), "required_growth": _text(required_growth),
        "earnings_change": _text(growth) if growth is not None else None,
        "conditional_eps": _text(conditional_eps) if conditional_eps is not None else None,
        "conditional_price": _text(conditional_price) if conditional_price is not None else None,
        "conditional_price_change": _text(price_change) if price_change is not None else None,
        "disclosed_exposure": exposure, "nav_return_contribution": None, "whole_fund_return": None,
        "additive_with_other_scenarios": False,
        "conditions": {"identity": "price = EPS * terminal_multiple",
                       "required_growth_identity": "(selected_price / terminal_multiple) / selected_TTM_EPS - 1",
                       "eps_basis": "selected positive comparable TTM diluted EPS",
                       "fx": "not applied; no verified joint FX assumptions"},
        "source_references": {"latest_close": security["latest_close"], "ttm_diluted_eps": security["ttm_diluted_eps"],
                              "mapping": security["mapping"]},
        "limitations": [
            "盈利变化与终值倍数是调用方填写的条件，不是预测、概率或已经实现的事实。",
            "只计算本地证券价格效果；没有验证联合汇率条件，不自动折算人民币基金净值回报。",
            "披露权重不是当前持仓；未覆盖部分、期货及行业切面不补零、不重复计入。",
        ],
    })

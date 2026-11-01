"""Small permanent business-core checks for issuer evidence and arithmetic."""

import json
from copy import deepcopy

import pytest

from rmbfund.advice_evidence import normalize_record
from rmbfund.domain import ValidationError
from rmbfund.issuer_fundamentals import build_issuer_analysis, issuer_valuation_scenario
from rmbfund.providers.advice_evidence import _html, _locator

PRODUCT = "synthetic:issuer-fund:C"
AS_OF = "2026-10-01T12:00:00+00:00"
URL = "https://example.com/issuer-document"


def _period(start, end, year, quarter):
    return {"start": start, "end": end, "basis": "quarter", "fiscal_year": year,
            "fiscal_quarter": quarter}


def _record(payload, *, report_date="2026-09-30", available="2026-10-01T09:00:00+00:00",
            published="2026-10-01T08:00:00+00:00", precision="timestamp"):
    value = {
        "product_id": PRODUCT, "family": "issuer_fundamentals", "source": "test-source",
        "upstream": "test-upstream", "source_url": URL, "report_date": report_date,
        "observed_at": available, "available_at": available, "payload": payload,
        "publication_precision": precision,
    }
    if precision != "unknown":
        value["published_at"] = published
    return normalize_record(value)


def _profile(weight="0.2"):
    return {
        "product_id": PRODUCT, "as_of": AS_OF,
        "portfolio": {
            "record_id": "portfolio-record", "report_date": "2026-09-30",
            "available_at": "2026-10-01T09:00:00+00:00",
            "payload": {
                "denominator": "nav", "holdings_scope": "top10",
                "holdings": [{"issuer_id": "issuer-1", "name": "Issuer One",
                              "kind": "corporate_equity", "weight": weight}],
            },
        },
    }


def _mapping(*, share_basis="ordinary", currency="USD"):
    return _record({
        "kind": "security_mapping", "issuer_id": "issuer-1", "issuer_name": "Issuer One",
        "security_id": "security-1", "symbol": "ISS", "exchange": "NYSE",
        "quote_currency": currency, "share_basis": share_basis, "locator": "json:/mapping",
    }, report_date="2026-09-30")


def _price(*, currency="USD", split_basis="ordinary-reported"):
    return _record({
        "kind": "price_history", "security_id": "security-1", "symbol": "ISS",
        "currency": currency, "exchange": "NYSE",
        "price_basis": {"kind": "unadjusted", "split_basis": split_basis,
                        "description": "direct current close", "locator": "json:/close"},
        "closes": [{"date": "2026-09-30", "close": "40"}],
    }, report_date="2026-09-30")


def _eps_release(values, *, split_basis="ordinary-reported", accounting_basis="GAAP"):
    metrics = []
    periods = [
        _period("2025-10-01", "2025-12-31", 2025, 4),
        _period("2026-01-01", "2026-03-31", 2026, 1),
        _period("2026-04-01", "2026-06-30", 2026, 2),
        _period("2026-07-01", "2026-09-30", 2026, 3),
    ]
    for quarter, value in enumerate(values, 1):
        metrics.append({
            "name": "eps", "category": "actual", "period": periods[quarter - 1],
            "currency": "USD", "unit": "currency_per_share", "scale": "1",
            "accounting_basis": accounting_basis, "period_role": "primary",
            "locator": f"json:/eps/{quarter}", "value": str(value), "security_id": "security-1",
            "eps_basis": "diluted", "share_basis": "ordinary", "split_basis": split_basis,
        })
    return _record({
        "kind": "financial_release", "issuer_id": "issuer-1", "issuer_name": "Issuer One",
        "release_id": "release-2026", "revision_id": "original-2026", "revision_kind": "original",
        "security_id": "security-1", "metrics": metrics,
    }, report_date="2026-09-30")


def test_signed_financials_and_future_guidance_are_explicitly_preserved_and_gated():
    signed = _record({
        "kind": "financial_release", "issuer_id": "issuer-1", "issuer_name": "Issuer One",
        "release_id": "release-loss", "revision_id": "original-loss", "revision_kind": "original",
        "security_id": "security-1", "metrics": [{
            "name": "free_cash_flow", "category": "actual",
            "period": _period("2026-07-01", "2026-09-30", 2026, 3),
            "currency": "USD", "unit": "currency", "scale": "1000000",
            "accounting_basis": "GAAP", "period_role": "primary", "locator": "json:/fcf",
            "value": "-2.5",
        }],
    })
    guidance = _record({
        "kind": "financial_release", "issuer_id": "issuer-1", "issuer_name": "Issuer One",
        "release_id": "release-guidance", "revision_id": "original-guidance", "revision_kind": "original",
        "security_id": "security-1", "metrics": [{
            "name": "revenue", "category": "guidance",
            "period": {"start": "2026-10-01", "end": None, "basis": "quarter",
                       "fiscal_year": 2026, "fiscal_quarter": 4},
            "currency": "USD", "unit": "currency", "scale": "1000000",
            "accounting_basis": "GAAP", "period_role": "primary", "locator": "json:/guidance",
            "low": "10", "high": "12",
        }],
    }, available="2026-10-02T09:00:00+00:00", published="2026-10-02T08:00:00+00:00")
    assert signed["payload"]["metrics"][0]["value"] == "-2.5"
    assert guidance["payload"]["metrics"][0]["period"]["end"] is None
    analysis = build_issuer_analysis([signed, guidance], profile=_profile(),
                                     as_of="2026-10-01T12:00:00+00:00")
    assert analysis["status"] == "partial"
    assert analysis["issuers"][0]["latest_guidance"] == []
    assert analysis["issuers"][0]["financial_observations"][0]["value"] == "-2500000"
    future_actual = deepcopy(signed["payload"])
    future_actual["metrics"][0]["period"] = _period("2026-10-01", "2026-12-31", 2026, 4)
    with pytest.raises(ValidationError, match="实际财务期间"):
        _record(future_actual)
    eligible_guidance = _record(guidance["payload"])
    selected = build_issuer_analysis([signed, eligible_guidance], profile=_profile(), as_of=AS_OF)
    assert selected["issuers"][0]["latest_guidance"][0]["low"] == "10000000"

    # Guidance imported after the actual release is still prior guidance only
    # because the original public release is earlier, never because of import order.
    actual_payload = deepcopy(signed["payload"])
    actual_payload.update(release_id="actual-revenue", revision_id="original-actual-revenue")
    actual_metric = actual_payload["metrics"][0]
    actual_metric.update(name="revenue", value="13", comparison_basis="issuer consolidated revenue")
    actual = _record(actual_payload)
    prior_payload = deepcopy(guidance["payload"])
    prior_payload.update(release_id="prior-guidance", revision_id="original-prior-guidance")
    prior_payload["metrics"][0].update(
        period=_period("2026-07-01", "2026-09-30", 2026, 3), accounting_basis="unknown",
        comparison_basis="issuer consolidated revenue",
    )
    prior = _record(prior_payload, report_date="2026-06-30",
                    available="2026-10-01T10:00:00+00:00", published="2026-07-01T08:00:00+00:00")
    corrected_payload = deepcopy(prior_payload)
    corrected_payload.update(revision_id="later-local-content-version", revision_kind="revision",
                             revises_revision_id="original-prior-guidance")
    corrected_payload["metrics"][0].update(low="30", high="40")
    corrected = _record(corrected_payload, published="2026-10-01T09:00:00+00:00",
                        available="2026-10-01T11:00:00+00:00")
    compared = build_issuer_analysis([actual, corrected, prior], profile=_profile(), as_of=AS_OF)
    comparison = compared["issuers"][0]["revenue_vs_prior_guidance"][0]
    assert comparison["prior_guidance"]["record_id"] == prior["id"]
    assert comparison["guidance_midpoint"] == "11000000"
    assert comparison["midpoint_error"] == "2000000"
    assert comparison["range_position"] == "above"
    assert comparison["comparison_caveat"] is not None
    reversed_publication = _record(prior_payload, report_date="2026-06-30",
                                   published="2026-10-01T08:30:00+00:00")
    absent = build_issuer_analysis([actual, reversed_publication], profile=_profile(), as_of=AS_OF)
    assert absent["issuers"][0]["revenue_vs_prior_guidance"][0]["status"] == "missing"

    # Real issuer releases use legitimate merged cells, unlike the narrow fee
    # parser's exact columns; a locator still needs a complete, nonempty row.
    merged = _html(b'<html><table><tr><td colspan="2">Revenue 13</td></tr></table></html>')
    assert _locator("table:1;row:1", "text/html", merged) == "table:1;row:1"
    for invalid in (
        b'<html><table><tr><td colspan="0">Revenue 13</td></tr></table></html>',
        b'<html><table><tr><td>Revenue 13</td></table></html>',
    ):
        with pytest.raises(ValidationError):
            _locator("table:1;row:1", "text/html", _html(invalid))


def test_ttm_never_floors_signed_loss_and_rejects_ads_or_currency_mismatch():
    release = _eps_release(["-1", "-1", "-1", "-1"])
    analysis = build_issuer_analysis([release, _mapping(), _price()], profile=_profile(), as_of=AS_OF)
    security = analysis["issuers"][0]["securities"][0]
    assert security["ttm_diluted_eps"]["value"] == "-4"
    assert security["valuation_status"] == "nonpositive_eps"
    assert security["pe_ttm"] is None

    ads_analysis = build_issuer_analysis([_eps_release(["1", "1", "1", "1"]),
                                          _mapping(share_basis="ADS"), _price()],
                                         profile=_profile(), as_of=AS_OF)
    ads_security = ads_analysis["issuers"][0]["securities"][0]
    assert ads_security["valuation_status"] == "missing"
    assert ads_security["valuation_reason"] == "eps_mapping_currency_or_share_basis_mismatch"
    currency_analysis = build_issuer_analysis([_eps_release(["1", "1", "1", "1"]),
                                               _mapping(currency="TWD"), _price(currency="TWD")],
                                              profile=_profile(), as_of=AS_OF)
    assert currency_analysis["issuers"][0]["securities"][0]["valuation_status"] == "missing"
    split_analysis = build_issuer_analysis([_eps_release(["1", "1", "1", "1"]),
                                           _mapping(), _price(split_basis="incompatible-current-basis")],
                                          profile=_profile(), as_of=AS_OF)
    assert split_analysis["issuers"][0]["securities"][0]["valuation_reason"] == "eps_price_split_basis_mismatch"
    missing = _eps_release(["1", "1", "1"])
    incomplete = build_issuer_analysis([missing, _mapping(), _price()], profile=_profile(), as_of=AS_OF)
    assert incomplete["issuers"][0]["securities"][0]["ttm_diluted_eps"]["value"] is None

    original = _eps_release(["1", "1", "1", "1"])
    comparison_payload = deepcopy(original["payload"])
    comparison_payload.update(release_id="later-comparative", revision_id="original-comparative")
    comparison_payload["metrics"] = [comparison_payload["metrics"][-1]]
    comparison_payload["metrics"][0].update(value="999", period_role="comparative")
    comparative = _record(comparison_payload)
    revision_payload = deepcopy(original["payload"])
    revision_payload.update(revision_id="revised-quarter-values", revision_kind="revision",
                            revises_revision_id="original-2026")
    for metric in revision_payload["metrics"]:
        metric["value"] = "2"
    revision = _record(revision_payload, available="2026-10-02T09:00:00+00:00",
                       published="2026-10-02T08:00:00+00:00")
    unchanged = build_issuer_analysis([original, comparative, revision, _mapping(), _price()],
                                      profile=_profile(), as_of=AS_OF)
    revised = build_issuer_analysis([original, comparative, revision, _mapping(), _price()],
                                    profile=_profile(), as_of="2026-10-02T12:00:00+00:00")
    assert unchanged["issuers"][0]["securities"][0]["ttm_diluted_eps"]["value"] == "4"
    assert revised["issuers"][0]["securities"][0]["ttm_diluted_eps"]["value"] == "8"
    assert revised["issuers"][0]["disclosed_exposure"] is None


def test_saved_analysis_supports_exact_conditional_price_arithmetic(tmp_path):
    analysis = build_issuer_analysis([_eps_release(["1", "1", "1", "1"]), _mapping(), _price()],
                                     profile=_profile(), as_of=AS_OF)
    path = tmp_path / "exact-analysis.json"
    path.write_text(json.dumps(analysis), encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == analysis
    before = deepcopy(saved)
    scenario = issuer_valuation_scenario(saved, issuer_id="issuer-1", terminal_multiple="20",
                                         earnings_change="0.1", label="explicit test condition")
    assert scenario["base_eps"] == "4"
    assert scenario["base_price"] == "40"
    assert scenario["base_pe"] == "10"
    assert scenario["required_eps"] == "2"
    assert scenario["required_growth"] == "-0.5"
    assert scenario["conditional_eps"] == "4.4"
    assert scenario["conditional_price"] == "88"
    assert scenario["conditional_price_change"] == "1.2"
    assert scenario["nav_return_contribution"] is None
    assert saved == before
    assert scenario["source_references"]["ttm_diluted_eps"] == saved["issuers"][0]["securities"][0]["ttm_diluted_eps"]
    assert scenario["disclosed_exposure"]["weight"] == "0.2"
    required_only = issuer_valuation_scenario(saved, issuer_id="issuer-1", terminal_multiple="20")
    assert required_only["earnings_change"] is None and required_only["conditional_price"] is None
    assert required_only["required_growth"] == "-0.5"
    with pytest.raises(ValidationError):
        issuer_valuation_scenario(saved, issuer_id="issuer-1", terminal_multiple="0")

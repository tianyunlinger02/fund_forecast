"""Offline financial arithmetic and author-visible evidence, with synthetic amounts."""
from copy import deepcopy
import json

import pytest

from rmbfund.advice_evidence import bounded_json, normalize_record
from rmbfund.domain import ValidationError
from rmbfund.issuer_fundamentals import build_issuer_analysis
from rmbfund.research_contract import normalize_generated_research, research_output_example
from rmbfund.research_runner import _bundle, _calculations, _check_bindings

PRODUCT = {"id": "synthetic:financial:C", "symbol": "FIN", "name": "合成财务测试基金",
           "kind": "open_fund", "share_class": "C", "exchange": "OTC"}
AS_OF = "2026-10-01T12:00:00+00:00"
CAPEX_DEFINITION = "Cash payments for property, plant and equipment, excluding debt principal"


def _period(basis="quarter"):
    start = {"quarter": "2026-07-01", "ytd": "2026-01-01", "year": "2025-10-01"}[basis]
    return {"start": start, "end": "2026-09-30", "basis": basis, "fiscal_year": 2026,
            "fiscal_quarter": None if basis == "year" else 3}


def _instant(day="2026-09-30"):
    return {"start": day, "end": day, "basis": "instant", "fiscal_year": 2026,
            "fiscal_quarter": None}


def _metric(name, value, *, period=None, **overrides):
    balance = name in {"cash_and_cash_equivalents", "accounts_receivable", "inventory",
                       "total_debt", "short_term_debt", "long_term_debt"}
    period = period or (_instant() if balance else _period())
    result = {"name": name, "category": "actual", "period": period, "currency": "USD",
              "unit": "currency", "scale": "1000000", "accounting_basis": "GAAP",
              "period_role": "primary", "value": str(value),
              "locator": f"table:{name}:{period['basis']}:{period['end']}"}
    if name == "capex":
        result["cash_capex"] = {"definition": CAPEX_DEFINITION, "sign_convention": "outflow_negative"}
    if name in {"total_debt", "short_term_debt", "long_term_debt"}:
        result["debt_definition"] = "Interest-bearing borrowing; short/current and long/noncurrent do not overlap"
    if name in {"short_term_debt", "long_term_debt"}:
        result["debt_maturity_scope"] = "current" if name == "short_term_debt" else "noncurrent"
    if name in {"accounts_receivable", "inventory"}:
        result["comparison_basis"] = f"consolidated {name}, net carrying amount, unchanged scope"
    result.update(overrides)
    return result


def _record(metrics, release="current", *, available="2026-10-01T09:00:00+00:00", **revision):
    return normalize_record({
        "product_id": PRODUCT["id"], "family": "issuer_fundamentals", "source": "synthetic",
        "upstream": "synthetic", "source_url": f"https://example.com/{release}",
        "report_date": max(metric["period"]["end"] for metric in metrics),
        "published_at": available, "publication_precision": "timestamp",
        "observed_at": available, "available_at": available,
        "payload": {"kind": "financial_release", "issuer_id": "issuer-financial",
                    "issuer_name": "Synthetic Financial Issuer", "release_id": release,
                    "revision_id": "original", "revision_kind": "original", "metrics": metrics,
                    **revision},
    })


def _analysis(records, *, as_of=AS_OF):
    return build_issuer_analysis(records, profile={"product_id": PRODUCT["id"], "as_of": as_of},
                                 as_of=as_of)["issuers"][0]


def _find(issuer, family, name, metric_name=None):
    rows = [row for row in issuer[family] if row["name"] == name
            and (metric_name is None or row.get("metric_name") == metric_name)]
    assert len(rows) == 1
    return rows[0]


def _flow_metrics(period=None):
    return [_metric(name, value, period=period) for name, value in (
        ("operating_cash_flow", "120"), ("net_income", "100"), ("capex", "-20"),
        ("free_cash_flow", "95"), ("revenue", "200"),
    )]


def test_cash_checks_preserve_signed_source_scale_and_explain_reported_fcf_difference():
    record = _record(_flow_metrics())
    before = deepcopy(record)
    issuer = _analysis([record])
    expected = {"ocf_to_net_income": "1.2", "ocf_minus_cash_capex": "100000000",
                "ocf_minus_cash_capex_vs_disclosed_fcf": "5000000", "ocf_margin": "0.6",
                "ocf_minus_cash_capex_margin": "0.5", "disclosed_fcf_margin": "0.475"}
    assert {row["name"]: row["value"] for row in issuer["cash_flow_checks"]} == expected
    residual = _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex")
    assert residual["cash_capex_outflow"] == "20000000"
    assert residual["inputs"]["capex"]["metric"]["value"] == "-20"
    assert residual["inputs"]["capex"]["value"] == "-20000000"
    assert "not_FCFF" in residual["caveat"]
    assert record == before and normalize_record(record) == record


@pytest.mark.parametrize("basis", ["quarter", "ytd", "year"])
def test_cash_checks_use_exact_reported_flow_period_without_annualization(basis):
    metrics = _flow_metrics(_period(basis))
    issuer = _analysis([_record(metrics)])
    assert _find(issuer, "cash_flow_checks", "ocf_to_net_income")["value"] == "1.2"
    assert _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex")["period"]["basis"] == basis
    metrics[1]["period"] = _period("year" if basis == "ytd" else "ytd")
    issuer = _analysis([_record(metrics)])
    check = _find(issuer, "cash_flow_checks", "ocf_to_net_income")
    assert check["status"] == "missing" and check["value"] is None
    assert check["reason"] == "missing_same_period_net_income"
    assert issuer["eps_yoy"] == []


def test_legacy_capex_is_not_guessed_and_sign_convention_is_not_absolute_value():
    metrics = _flow_metrics()
    metrics[2].pop("cash_capex")
    legacy = _record(metrics)
    issuer = _analysis([legacy])
    check = _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex")
    assert check["value"] is None and check["reason"].startswith("missing_explicit_cash_capex")
    assert "cash_capex" not in next(metric for metric in legacy["payload"]["metrics"]
                                    if metric["name"] == "capex")
    metrics[2]["cash_capex"] = {"definition": "Cash purchases net of cash proceeds (refunds stay signed)",
                                 "sign_convention": "outflow_positive"}
    issuer = _analysis([_record(metrics)])
    assert _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex")["value"] == "140000000"
    metrics[0]["value"], metrics[1]["value"], metrics[4]["value"] = "-120", "-100", "0"
    issuer = _analysis([_record(metrics)])
    ratio = _find(issuer, "cash_flow_checks", "ocf_to_net_income")
    assert ratio["value"] == "1.2" and ratio["caveat"] is not None
    assert _find(issuer, "cash_flow_checks", "ocf_margin")["reason"] == "zero_denominator"
    metrics[1]["value"] = "0"
    assert _find(_analysis([_record(metrics)]), "cash_flow_checks", "ocf_to_net_income")["value"] is None


def test_financial_basis_is_explicit_while_reported_scales_are_normalized_once():
    metrics = _flow_metrics()
    metrics[0].update(value="0.12", scale="1000000000")
    assert _find(_analysis([_record(metrics)]), "cash_flow_checks", "ocf_to_net_income")["value"] == "1.2"
    for key, value in (("currency", "EUR"), ("accounting_basis", "IFRS"), ("accounting_basis", "unknown")):
        changed = deepcopy(metrics)
        changed[1][key] = value
        row = _find(_analysis([_record(changed)]), "cash_flow_checks", "ocf_to_net_income")
        assert row["value"] is None and "incompatible" in row["reason"]


@pytest.mark.parametrize("fcf_basis", ["non_GAAP", "unknown"])
def test_reported_fcf_has_narrow_explicit_definition_bridge_not_a_general_accounting_bypass(fcf_basis):
    metrics = _flow_metrics()
    metrics[3].update(accounting_basis=fcf_basis)
    if fcf_basis == "non_GAAP":
        metrics[3]["accounting_basis_detail"] = "OCF minus cash PP&E purchases and financing principal"
    issuer = _analysis([_record(metrics)])
    assert _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex_vs_disclosed_fcf")["value"] is None
    metrics[3]["cash_flow_definition"] = {
        "definition": "GAAP OCF less cash PP&E purchases less financing principal",
        "ocf_accounting_basis": "GAAP", "cash_capex_definition": CAPEX_DEFINITION,
    }
    issuer = _analysis([_record(metrics)])
    difference = _find(issuer, "cash_flow_checks", "ocf_minus_cash_capex_vs_disclosed_fcf")
    assert difference["value"] == "5000000" and difference["comparison_mode"] == "defined_basis_bridge"
    assert difference["inputs"]["free_cash_flow"]["metric"]["accounting_basis"] == fcf_basis
    assert difference["inputs"]["operating_cash_flow"]["metric"]["accounting_basis"] == "GAAP"
    margin = _find(issuer, "cash_flow_checks", "disclosed_fcf_margin")
    assert margin["value"] == "0.475" and margin["comparison_mode"] == "defined_basis_bridge"
    metrics[3]["cash_flow_definition"]["cash_capex_definition"] = "Different net cash asset purchases"
    assert _find(_analysis([_record(metrics)]), "cash_flow_checks",
                 "ocf_minus_cash_capex_vs_disclosed_fcf")["value"] is None


def test_same_day_net_cash_uses_explicit_debt_components_but_never_missing_zero_or_total_liabilities():
    metrics = [_metric("cash_and_cash_equivalents", "50"), _metric("short_term_debt", "10"),
               _metric("long_term_debt", "20")]
    row = _find(_analysis([_record(metrics)]), "balance_sheet_checks", "net_cash")
    assert row["value"] == "20000000" and row["interest_bearing_debt_total"] == "30000000"
    assert row["debt_basis"] == "explicit_short_plus_long"
    assert row["inputs"]["short_term_debt"]["metric"]["debt_definition"]
    overlapping = deepcopy(metrics)
    overlapping[2]["debt_maturity_scope"] = "current_and_noncurrent"
    rejected = _find(_analysis([_record(overlapping)]), "balance_sheet_checks", "net_cash")
    assert rejected["value"] is None and rejected["reason"].startswith("debt_components_not_explicitly_nonoverlapping")
    missing = _find(_analysis([_record(metrics[:2])]), "balance_sheet_checks", "net_cash")
    assert missing["value"] is None and missing["reason"] == "missing_same_period_long_term_debt"
    metrics.append(_metric("total_debt", "35"))
    row = _find(_analysis([_record(metrics)]), "balance_sheet_checks", "net_cash")
    assert row["value"] == "15000000" and row["debt_basis"] == "reported_total"
    metrics[3]["period"] = _instant("2026-06-30")
    rows = _analysis([_record(metrics)])["balance_sheet_checks"]
    assert next(row for row in rows if row["period"]["end"] == "2026-06-30")["value"] is None
    assert next(row for row in rows if row["period"]["end"] == "2026-09-30")["value"] == "20000000"


def test_comparable_balance_changes_use_current_release_comparatives_and_actual_revenue_periods():
    metrics = [_metric("revenue", "200"), _metric("revenue", "600", period=_period("ytd")),
               _metric("accounts_receivable", "60"), _metric("inventory", "40"),
               _metric("accounts_receivable", "50", period=_instant("2026-06-30"), period_role="comparative"),
               _metric("inventory", "0", period=_instant("2026-06-30"), period_role="comparative")]
    older = _record([_metric("accounts_receivable", "999", period=_instant("2026-06-30"),
                             period_role="comparative")], "earlier-comparative")
    issuer = _analysis([older, _record(metrics)])
    change = _find(issuer, "working_capital_checks", "working_capital_change", "accounts_receivable")
    assert change["value"] == "10000000" and change["change_ratio"] == "0.2"
    assert change["inputs"]["prior"]["release_id"] == "current"
    inventory = _find(issuer, "working_capital_checks", "working_capital_change", "inventory")
    assert inventory["value"] == "40000000" and inventory["change_ratio"] is None
    assert inventory["change_ratio_reason"] == "zero_prior_balance"
    ratios = [row for row in issuer["working_capital_checks"]
              if row["name"] == "working_capital_to_revenue" and row["metric_name"] == "accounts_receivable"]
    assert {row["period"]["basis"]: row["value"] for row in ratios} == {"quarter": "0.3", "ytd": "0.1"}
    metrics[4]["comparison_basis"] = "different gross balance definition"
    changed = _find(_analysis([_record(metrics)]), "working_capital_checks",
                    "working_capital_change", "accounts_receivable")
    assert changed["value"] is None and "comparison_basis" in changed["reason"]


def test_revisions_and_actual_availability_apply_to_new_financials():
    original = _record(_flow_metrics())
    metrics = _flow_metrics()
    metrics[0]["value"] = "150"
    revision = _record(metrics, available="2026-10-02T09:00:00+00:00", revision_kind="revision",
                       revision_id="corrected", revises_revision_id="original")
    assert _find(_analysis([original, revision]), "cash_flow_checks", "ocf_to_net_income")["value"] == "1.2"
    latest = _analysis([original, revision], as_of="2026-10-02T12:00:00+00:00")
    ratio = _find(latest, "cash_flow_checks", "ocf_to_net_income")
    assert ratio["value"] == "1.5" and ratio["inputs"]["operating_cash_flow"]["record_id"] == revision["id"]


def test_new_metric_contract_preserves_flow_instant_and_source_definitions():
    for metric in (_metric("net_income", "1", period=_instant()),
                   _metric("accounts_receivable", "1", period=_period()),
                   _metric("cash_and_cash_equivalents", "1", period={**_instant(), "start": "2026-09-29"})):
        with pytest.raises(ValidationError):
            _record([metric])
    debt = _metric("total_debt", "1")
    debt.pop("debt_definition")
    with pytest.raises(ValidationError, match="有息债务"):
        _record([debt])
    wrong = _metric("net_income", "1", cash_capex={"definition": "not capex", "sign_convention": "outflow_positive"})
    with pytest.raises(ValidationError, match="只有capex"):
        _record([wrong])


def test_author_package_keeps_latest_checks_real_prior_inputs_and_bindable_calculations():
    prior = _record([_metric("accounts_receivable", "50", period=_instant("2026-06-30"))], "prior")
    unused = _record([_metric("net_income", "999", period={"start": "2026-04-01", "end": "2026-06-30",
                      "basis": "quarter", "fiscal_year": 2026, "fiscal_quarter": 2})], "unused")
    current = _record([*_flow_metrics(), *_flow_metrics(_period("ytd")), _metric("cash_and_cash_equivalents", "50"),
                       _metric("total_debt", "30"), _metric("accounts_receivable", "60")])
    package, evidence, links, products = _bundle(PRODUCT, {}, {"evidence_records": [unused, prior, current]},
                                               None, None, AS_OF)
    issuer = package["issuer_analysis"]["issuers"][0]
    assert all(issuer[key] for key in ("cash_flow_checks", "balance_sheet_checks", "working_capital_checks"))
    assert {row["period"]["basis"] for row in issuer["cash_flow_checks"]
            if row["name"] == "ocf_to_net_income"} == {"quarter", "ytd"}
    assert {row["period"]["basis"] for row in issuer["working_capital_checks"]
            if row["name"] == "working_capital_to_revenue"} == {"quarter", "ytd"}
    assert {row["id"] for row in package["structured_records"]} == {prior["id"], current["id"]}
    calculation = next(row for row in evidence["calculations"]
                       if row["method"] == "issuer_fundamentals.working_capital_change")
    assert calculation["record_ids"] == sorted([prior["id"], current["id"]])
    complete_analysis = build_issuer_analysis([prior, current],
        profile={"product_id": PRODUCT["id"], "as_of": AS_OF}, as_of=AS_OF)
    incomplete = _calculations({"issuer_analysis": complete_analysis}, [current], PRODUCT, AS_OF)
    assert "issuer_fundamentals.working_capital_change" not in {row["method"] for row in incomplete}
    assert calculation["inputs"]["prior"]["metric"]["comparison_basis"]
    assert calculation["inputs"]["current"]["value"] == "60000000"
    assert calculation["result"]["value"] == "10000000"
    assert calculation["sections"][0]["text"] == bounded_json(calculation["result"])
    assert all(len(bounded_json(row["inputs"]).encode()) < 32 * 1024 for row in evidence["calculations"])
    assert all(len(row["sections"][0]["text"]) < 4000 for row in evidence["calculations"])
    note = research_output_example(product=PRODUCT, as_of=AS_OF)
    note["facts"][0]["claim"] = "合成应收余额的可比变动为10000000。"
    note["citations"] = [{"id": "c1", "kind": "calculation", "evidence_id": calculation["id"],
                          "locator": "program_result", "quote": calculation["sections"][0]["text"],
                          "claim": "Synthetic balance change using the supplied source observations"}]
    normalized = normalize_generated_research(note, evidence=evidence, as_of=AS_OF)
    _check_bindings(normalized, evidence, links, products, AS_OF)
    saved = json.loads(json.dumps(normalized))
    assert saved["citations"][0]["inputs"] == calculation["inputs"]
    assert saved["citations"][0]["result"] == calculation["result"]

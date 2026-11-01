"""Synthetic, pure retail-fund comparisons: identity, common dates and switching costs."""

import json
from copy import deepcopy
from decimal import Decimal, localcontext

import pytest

from rmbfund.domain import ValidationError
from rmbfund.fund_comparison import (
    PeerRelation,
    PeerSet,
    compare_funds,
    compare_peer_set,
    switch_scenario,
)

AS_OF = "2026-10-26T16:00:00+08:00"
REFS = ["synthetic:prospectus#identity"]


def _fund(product_id="subject", *, share_class="C", underlying=None, currency="CNY",
          mandate="global_equity", values=None):
    if values is None:
        values = [("2026-10-20", "1"), ("2026-10-21", "1.1"), ("2026-10-22", "0.99")]
    return {
        "product": {
            "id": product_id, "symbol": product_id, "name": f"Synthetic {product_id}",
            "kind": "open_fund", "share_class": share_class, "currency": currency,
            "underlying_fund_id": underlying or f"underlying:{product_id}",
            "mandate_id": mandate, "evidence_refs": REFS.copy(),
        },
        "series": {
            "status": "available", "product_id": product_id, "share_class": share_class,
            "currency": currency, "basis": "unit_nav", "evidence_refs": ["synthetic:raw-nav"],
            "observations": [{"date": day, "value": value} for day, value in values],
        },
    }


def _relation(product_id="peer", relation="same_mandate"):
    return {"product_id": product_id, "relation": relation,
            "selection_reason": "Explicit synthetic comparison with a documented mandate.",
            "evidence_refs": REFS.copy()}


def _peers(*relations, start="2026-10-01", end="2026-10-26"):
    return {"target_product_id": "subject", "peers": list(relations or [_relation()]),
            "start_date": start, "end_date": end,
            "selection_reason": "Synthetic public-fund cohort, not personal holdings.",
            "evidence_refs": REFS.copy()}


def _pair(subject=None, peer=None, relation="same_mandate"):
    return compare_funds(_peers(_relation(relation=relation)),
                         {"subject": subject or _fund(), "peer": peer or _fund("peer")}, as_of=AS_OF)


def _exposure(weights, *, date="2026-06-30", denominator="nav", scope="top10"):
    return {"status": "available", "as_of": date, "denominator": denominator, "scope": scope,
            "evidence_refs": ["synthetic:semiannual-holdings"],
            "holdings": [{"security_id": key, "weight": weight} for key, weight in weights]}


def _fee(*, rate=None, amount=None, basis="gross_redemption", status="available"):
    if status != "available":
        return {"status": status, "reason": "synthetic unknown fee schedule"}
    return {"status": status, "basis": basis, "currency": "CNY", "evidence_refs": ["synthetic:fee"],
            **({"rate": rate} if rate is not None else {"amount": amount})}


def _switch():
    return {
        "currency": "CNY", "gross_redemption_amount": "1000",
        "selection_reason": "Evaluate costs only; no expected-return comparison.",
        "evidence_refs": ["synthetic:switch-assumptions"],
        "redemption_fee": _fee(rate="0.01"),
        "subscription_fee": _fee(rate="0.02", basis="net_subscription"),
        "other_fees": _fee(amount="3", basis="fixed_cash"),
        "timing": {
            "status": "available", "evidence_refs": ["synthetic:settlement-rules"],
            "redemption_date": "2026-10-20", "source_exposure_end_date": "2026-10-20",
            "cash_available_date": "2026-10-23", "subscription_date": "2026-10-26",
            "target_exposure_start_date": "2026-10-26", "target_confirmation_date": "2026-10-28",
        },
    }


def test_peer_relations_require_explicit_evidence_reason_and_distinct_members():
    for relation in ("same_mandate", "style_or_cost_alternative", "reference_only"):
        item = PeerRelation.from_mapping(_relation(relation=relation))
        assert item.relation == relation
        assert PeerSet.from_mapping(_peers(item.to_dict())).to_dict()["peers"] == [item.to_dict()]
    invalid = [dict(_relation(), relation="best_fund"), dict(_relation(), selection_reason=" "),
               dict(_relation(), evidence_refs=[]), dict(_relation(), evidence_refs="source")]
    for relation in invalid:
        with pytest.raises(ValidationError):
            PeerRelation.from_mapping(relation)
    for peers in (_peers(_relation("subject")), _peers(_relation(), _relation()),
                  _peers(start="2026-10-26", end="2026-10-01"),
                  dict(_peers(), evidence_refs=[]), dict(_peers(), start_date=None)):
        with pytest.raises(ValidationError):
            PeerSet.from_mapping(peers)
    with pytest.raises(ValidationError):
        PeerSet.from_mapping(dict(_peers(), peers=[]))


def test_only_exact_common_dates_are_used_for_raw_return_volatility_and_drawdown():
    subject = _fund(values=[("2026-09-30", "100"), ("2026-10-20", "1"),
                            ("2026-10-21", "100"), ("2026-10-22", "1.2"),
                            ("2026-10-23", "1.08"), ("2026-10-26", "500")])
    peer = _fund("peer", values=[("2026-10-20", "2"), ("2026-10-22", "2.2"),
                                 ("2026-10-23", "2.42"), ("2026-10-24", "999")])
    original = deepcopy((subject, peer))
    result = _pair(subject, peer)
    pair = result["comparisons"][0]
    assert pair["common_window"]["dates"] == ["2026-10-20", "2026-10-22", "2026-10-23"]
    assert pair["common_window"]["calendar_day_gaps"] == [2, 1]
    assert pair["common_window"]["observation_count"] == 3
    assert pair["subject_metrics"]["raw_return"]["value"] == "0.08"
    assert pair["subject_metrics"]["max_drawdown"]["value"] == "0.1"
    assert pair["peer_metrics"]["raw_return"]["value"] == "0.21"
    assert pair["peer_metrics"]["volatility"]["value"] == "0"
    with localcontext() as context:
        context.prec = 40
        expected = (Decimal("0.045")).sqrt()
    assert Decimal(pair["subject_metrics"]["volatility"]["value"]) == expected
    assert pair["subject_metrics"]["annualized"] is False
    assert pair["subject_metrics"]["return_interval_count"] == 2
    assert (subject, peer) == original
    assert json.loads(json.dumps(result)) == result


def test_nav_is_never_called_total_return_even_when_corporate_action_coverage_is_supplied():
    result = _pair()
    pair = result["comparisons"][0]
    assert pair["subject_total_return"]["status"] == "total_return_unavailable"
    assert pair["subject_total_return"]["value"] is None
    assert "coverage_absent" in pair["subject_total_return"]["reason"]
    assert "not_total_return" in pair["subject_metrics"]["raw_return"]["label"]
    covered = _fund()
    covered["corporate_action_coverage"] = {
        key: {"status": "available", "start_date": "2026-10-01", "end_date": "2026-10-26",
              "evidence_refs": ["synthetic:complete-action-log"]} for key in ("dividends", "splits")}
    total = _pair(covered)["comparisons"][0]["subject_total_return"]
    assert total["status"] == "total_return_unavailable"
    assert total["reason"] == "raw_series_not_adjusted_for_total_return"
    assert result["manager_alpha"]["value"] is None
    assert result["ranking"]["status"] == "not_produced"


@pytest.mark.parametrize("mutation,reason", [
    ("currency", "incompatible_currency"),
    ("share_class", "incompatible_share_class_for_same_mandate"),
    ("series_identity", "series_share_class_mismatch_or_missing"),
    ("basis", "incompatible_observation_basis"),
    ("mandate", "unlike_mandates_require_alternative_or_reference_relation"),
])
def test_incompatible_public_currency_share_class_basis_or_mandate_is_blocked(mutation, reason):
    peer = _fund("peer")
    if mutation in {"currency", "share_class"}:
        value = "USD" if mutation == "currency" else "A"
        peer["product"][mutation] = value
        peer["series"][mutation] = value
    elif mutation == "series_identity":
        peer["series"]["share_class"] = "A"
    elif mutation == "basis":
        peer["series"]["basis"] = "unadjusted_price"
    else:
        peer["product"]["mandate_id"] = "unlike_regional_mandate"
    pair = _pair(peer=peer)["comparisons"][0]
    assert pair["status"] == "blocked"
    assert reason in {item["reason"] for item in pair["compatibility"]["issues"]}
    assert pair["peer_metrics"]["raw_return"]["value"] is None


def test_same_underlying_a_c_classes_are_only_cost_alternatives_not_manager_peers():
    subject = _fund(underlying="one-portfolio")
    peer = _fund("peer", underlying="one-portfolio", share_class="A")
    for relation in ("same_mandate", "reference_only"):
        pair = _pair(subject, peer, relation)["comparisons"][0]
        assert pair["status"] == "blocked"
        assert any("same_underlying_requires" in item["reason"] for item in pair["compatibility"]["issues"])
    result = _pair(subject, peer, "style_or_cost_alternative")
    assert result["comparisons"][0]["status"] == "available"
    assert result["manager_alpha"]["status"] == "not_assessed"
    assert result["ranking"]["status"] == "not_produced"
    peer["product"]["mandate_id"] = "different_style"
    assert _pair(peer=peer, relation="reference_only")["comparisons"][0]["status"] == "available"


def test_missing_and_blocked_peers_and_rows_remain_visible_without_poisoning_valid_pairs():
    peer = _fund("peer")
    peer["series"]["observations"].insert(1, {"date": "2026-10-19", "status": "blocked",
                                                  "reason": "conflicting public values", "currency": "USD",
                                                  "value": "not-a-number"})
    blocked = _fund("blocked")
    blocked["series"] = {"status": "blocked", "reason": "currency conflict", "observations": []}
    peers = _peers(_relation(), _relation("absent"), _relation("blocked"))
    result = compare_peer_set(peers, {"subject": _fund(), "peer": peer, "blocked": blocked}, as_of=AS_OF)
    assert result["status"] == "partial"
    assert [item["status"] for item in result["comparisons"]] == ["available", "missing", "blocked"]
    assert result["comparisons"][0]["common_window"]["observation_count"] == 3
    assert result["common_window"]["observation_count"] == 0
    assert result["funds"][1]["series"] == peer["series"]
    assert result["funds"][3]["series"]["reason"] == "currency conflict"
    missing = _fund("peer")
    del missing["product"]["currency"]
    pair = _pair(peer=missing)["comparisons"][0]
    assert pair["status"] == "missing"
    assert {"product_id": "peer", "field": "product.currency"} in pair["compatibility"]["missing_fields"]


def test_as_of_filters_later_dates_and_later_available_observations_without_fill():
    subject = _fund(values=[("2026-10-20", "1"), ("2026-10-21", "1.1"), ("2026-10-27", "9")])
    peer = deepcopy(subject)
    peer["product"].update(id="peer", symbol="peer", underlying_fund_id="another")
    peer["series"]["product_id"] = "peer"
    subject["series"]["observations"][1]["available_at"] = "2026-10-27T12:00:00+08:00"
    result = _pair(subject, peer)
    pair = result["comparisons"][0]
    assert pair["common_window"]["dates"] == ["2026-10-20"]
    assert pair["status"] == "missing"
    assert pair["subject_metrics"]["raw_return"]["value"] is None
    assert pair["peer_metrics"]["volatility"]["value"] is None
    peer["series"]["observations"].append({"date": "2026-10-20", "value": "1"})
    with pytest.raises(ValidationError, match="duplicate observation"):
        _pair(subject, peer)


def test_price_only_metrics_and_two_observation_volatility_caveat():
    subject, peer = _fund(), _fund("peer")
    for fund in (subject, peer):
        fund["product"]["kind"] = "etf"
        fund["product"]["share_class"] = "listed"
        fund["series"]["share_class"] = "listed"
        fund["series"]["basis"] = "unadjusted_price"
        fund["series"]["observations"] = fund["series"]["observations"][:2]
    result = _pair(subject, peer)
    pair = result["comparisons"][0]
    assert pair["status"] == "partial"
    assert pair["peer_metrics"]["raw_return"]["value"] == "0.1"
    assert pair["peer_metrics"]["volatility"]["status"] == "missing"
    assert pair["peer_total_return"]["status"] == "total_return_unavailable"


def test_disclosed_overlap_preserves_denominators_coverage_and_original_weights():
    subject, peer = _fund(), _fund("peer")
    subject["exposure"] = _exposure([("exact-security", "0.10"), ("other", "0.20")])
    peer["exposure"] = _exposure([("exact-security", "0.05"), ("same-name-different-id", "0.30")])
    subject["fees"] = {"status": "available", "as_of": "2026-09-30", "evidence_refs": REFS,
                       "items": [{"name": "sales_service", "rate": "0.0040", "basis": "annual_nav"}]}
    result = _pair(subject, peer)
    overlap = result["comparisons"][0]["disclosed_overlap"]
    assert overlap["value"] == "0.05"
    assert overlap["subject_disclosed_weight"] == "0.3"
    assert overlap["peer_disclosed_weight"] == "0.35"
    assert overlap["is_disclosed_only_lower_bound"] is True
    assert overlap["normalization_applied"] is False
    assert result["funds"][0]["fees"] == subject["fees"]
    assert result["funds"][0]["exposure"] == subject["exposure"]
    for key, value in (("denominator", "equity_assets"), ("as_of", "2026-09-30")):
        changed = deepcopy(peer)
        changed["exposure"][key] = value
        blocked = _pair(subject, changed)["comparisons"][0]["disclosed_overlap"]
        assert blocked["status"] == "blocked"
        assert blocked["value"] is None
    peer["exposure"] = {"status": "blocked", "reason": "ambiguous disclosed denominator"}
    assert _pair(subject, peer)["funds"][1]["exposure"] == peer["exposure"]


def test_future_and_blocked_disclosures_keep_source_statuses_and_do_not_get_normalized():
    subject, peer = _fund(), _fund("peer")
    peer["exposure"] = {"status": "blocked", "reason": "conflicting denominator"}
    overlap = _pair(subject, peer)["comparisons"][0]["disclosed_overlap"]
    assert overlap["status"] == "blocked"
    assert [item["status"] for item in overlap["source_statuses"]] == ["missing", "blocked"]
    subject["exposure"] = _exposure([("same", "1.20")], scope="full")
    peer["exposure"] = _exposure([("same", "1.10")], scope="full")
    assert _pair(subject, peer)["comparisons"][0]["disclosed_overlap"]["value"] == "1.1"
    peer["exposure"]["published_at"] = "2026-10-27T12:00:00+08:00"
    snapshot = _pair(subject, peer)
    assert snapshot["funds"][1]["exposure"]["status"] == "missing"
    assert snapshot["funds"][1]["exposure"]["source"] == peer["exposure"]
    subject["corporate_action_coverage"] = {"dividends": {"status": "missing"},
                                             "splits": {"status": "blocked"}}
    total = _pair(subject, peer)["comparisons"][0]["subject_total_return"]
    assert total["coverage"] == subject["corporate_action_coverage"]
    assert total["status"] == "total_return_unavailable"
    no_interval = _peers()
    del no_interval["start_date"]
    del no_interval["end_date"]
    assert compare_funds(no_interval, {"subject": _fund(), "peer": _fund("peer")},
                         as_of=AS_OF)["comparisons"][0]["common_window"]["observation_count"] == 3


def test_switch_cost_arithmetic_uses_fee_denominators_and_explicit_cash_timing():
    scenario = _switch()
    before = deepcopy(scenario)
    result = switch_scenario(_fund(), _fund("peer"), scenario)
    arithmetic = result["arithmetic"]
    assert result["status"] == "available"
    assert arithmetic["redemption_fee_amount"] == "10"
    assert arithmetic["cash_after_redemption_fee"] == "990"
    assert arithmetic["subscription_budget"] == "987"
    assert Decimal(arithmetic["target_net_subscription_amount"]) == pytest.approx(Decimal("967.6470588235294117647058823529411764706"))
    with localcontext() as context:
        context.prec = 40
        assert Decimal(arithmetic["total_explicit_cost"]) == 1000 - Decimal(arithmetic["target_net_subscription_amount"])
    assert result["timing"]["calendar_cash_wait_days"] == 3
    assert result["timing"]["calendar_redeployment_wait_days"] == 3
    assert result["timing"]["out_of_market_calendar_days"] == 6
    assert result["realizable_gain"]["value"] is None
    assert result["executable_amount"]["value"] is None
    assert scenario == before
    json.dumps(result)
    scenario["subscription_fee"] = _fee(rate="0.02", basis="gross_subscription")
    gross = switch_scenario(_fund(), _fund("peer"), scenario)["arithmetic"]
    assert gross["subscription_fee_amount"] == "19.74"
    assert gross["target_net_subscription_amount"] == "967.26"


def test_switch_unknown_or_blocked_fees_do_not_become_zero_and_forecasts_are_rejected():
    scenario = _switch()
    scenario["redemption_fee"] = _fee(status="missing")
    missing = switch_scenario(_fund(), _fund("peer"), scenario)
    assert missing["status"] == "partial"
    assert missing["arithmetic"]["cash_after_redemption_fee"] is None
    assert missing["arithmetic"]["target_net_subscription_amount"] is None
    assert missing["arithmetic"]["total_explicit_cost"] is None
    assert missing["fees"]["redemption_fee"] == scenario["redemption_fee"]
    scenario["redemption_fee"] = _fee(status="blocked")
    assert switch_scenario(_fund(), _fund("peer"), scenario)["status"] == "blocked"
    scenario = _switch()
    del scenario["other_fees"]
    assert switch_scenario(_fund(), _fund("peer"), scenario)["arithmetic"]["target_net_subscription_amount"] is None
    scenario["forecast_gain"] = "0.2"
    with pytest.raises(ValidationError, match="never forecasts"):
        switch_scenario(_fund(), _fund("peer"), scenario)
    with pytest.raises(ValidationError, match="compatible currencies"):
        switch_scenario(_fund(), _fund("peer", currency="USD"), _switch())


def test_switch_rejects_reuse_before_cash_and_keeps_cutoff_exposure_unknowns():
    scenario = _switch()
    scenario["timing"]["subscription_date"] = "2026-10-22"
    blocked = switch_scenario(_fund(), _fund("peer"), scenario)
    assert blocked["status"] == "blocked"
    assert "subscription_date_before_cash_available_date" in blocked["timing"]["issues"]
    scenario = _switch()
    scenario["timing"]["subscription_date"] = "2026-10-23"
    same_day = switch_scenario(_fund(), _fund("peer"), scenario)
    assert same_day["status"] == "partial"
    assert "cash_before_subscription_cutoff" in same_day["timing"]["unknowns"]
    scenario["timing"]["cash_before_subscription_cutoff"] = False
    assert switch_scenario(_fund(), _fund("peer"), scenario)["status"] == "blocked"
    scenario = _switch()
    del scenario["timing"]["source_exposure_end_date"]
    partial = switch_scenario(_fund(), _fund("peer"), scenario)
    assert partial["timing"]["out_of_market_calendar_days"] is None
    assert "source_exposure_end_date" in partial["timing"]["unknowns"]
    scenario["redemption_fee"] = _fee(amount="1001")
    overdrawn = switch_scenario(_fund(), _fund("peer"), scenario)
    assert overdrawn["status"] == "blocked"
    assert overdrawn["arithmetic"]["target_net_subscription_amount"] is None


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "0", "-1"])
def test_invalid_nav_values_use_existing_domain_validation(value):
    peer = _fund("peer")
    peer["series"]["observations"][0]["value"] = value
    with pytest.raises(ValidationError):
        _pair(peer=peer)

"""Synthetic, persistence-free retail-buyer arithmetic and disclosure boundaries."""

from copy import deepcopy
from decimal import Decimal, localcontext
import json

import pytest

from rmbfund.domain import ValidationError
from rmbfund.portfolio_exposure import build_portfolio_context, pairwise_issuer_overlap
from rmbfund.position_sizing import conditional_size, planned_redemption, post_buy_weight

AS_OF = "2026-10-26T12:00:00+08:00"


def profile(weights=None, *, sectors=None, denominator="nav", report_date="2026-06-30"):
    weights = {"issuer-a": "0.2", "issuer-b": "0.1"} if weights is None else weights
    return {"portfolio": {
        "record_id": "synthetic-disclosure", "report_date": report_date,
        "published_at": "2026-08-20T09:00:00+08:00",
        "payload": {
            "denominator": denominator, "holdings_scope": "partial",
            "holdings": [{"issuer_id": issuer, "kind": "corporate_equity", "weight": weight}
                         for issuer, weight in weights.items()],
            "sectors": [{"name": name, "weight": weight}
                        for name, weight in (sectors or {"source-sector": "0.4"}).items()],
        },
    }}


def position(product_id, value, *, currency="CNY", public_profile=None):
    return {
        "holding": {"id": f"private-{product_id}", "product_id": product_id,
                    "subject": {"symbol": "not-an-identity"}},
        "valuation": {"currency": currency, "market_value": value, "nav_date": "2026-10-23",
                      "cost": None, "valuation_status": "ready"},
        "profile": profile() if public_profile is None else public_profile,
    }


def conditions(**changes):
    return {
        "scenario_return": "-0.15", "subscription_fee": "0.02",
        "existing_scenario_loss": "10", "loss_budget": "30", "spendable_cash": "500",
        "scenario_scope": "whole_fund", "budget_scope": "covered_portfolio",
        "scenario_label": "用户明确条件，非最大损失", **changes,
    }


def test_partial_disclosure_uses_original_nav_weights_and_exact_known_cny_cohort():
    rows = [position("fund-a", "100"), position("fund-b", "300"),
            position("unknown", None), position("foreign", "1000", currency="USD")]
    original = deepcopy(rows)
    result = build_portfolio_context(rows, selected_product_id="fund-a", as_of=AS_OF)
    assert result["known_cohort"]["market_value"] == "400"
    assert result["known_cohort"]["fund_count"] == 2
    assert result["selected"]["weight_of_known_cny_valued_funds"] == "0.25"
    issuer = result["look_through"]["issuers"]
    exposures = {row["issuer_id"]: row for row in issuer["exposures"]}
    assert Decimal(exposures["issuer-a"]["known_value_cny"]) == 80
    assert Decimal(issuer["coverage"]) == Decimal("0.3")
    assert Decimal(issuer["uncovered_value_cny"]) == 280
    assert Decimal(result["look_through"]["sectors"]["coverage"]) == Decimal("0.4")
    assert exposures["issuer-a"]["sources"][0]["report_date"] == "2026-06-30"
    assert exposures["issuer-a"]["sources"][0]["nav_date"] == "2026-10-23"
    assert len(result["exclusions"]) == 2
    assert rows == original
    json.dumps(result, allow_nan=False)


def test_unknown_foreign_zero_and_unrecorded_are_not_fabricated_values():
    rows = [position("missing", None), position("foreign", "10", currency="USD")]
    result = build_portfolio_context(rows, selected_product_id="missing", as_of=AS_OF)
    assert result["known_cohort"]["market_value"] is None
    assert result["selected"]["market_value"] is None
    assert result["look_through"]["issuers"]["coverage"] is None
    zero = build_portfolio_context([position("zero", "0")], selected_product_id="zero", as_of=AS_OF)
    assert zero["known_cohort"]["market_value"] == "0"
    assert zero["selected"]["weight_of_known_cny_valued_funds"] is None
    absent = build_portfolio_context([position("known", "20")],
                                     selected_product_id="absent", as_of=AS_OF)
    assert absent["selected"]["status"] == "not_recorded"
    assert absent["selected"]["market_value"] is None
    currency_unknown = position("no-currency", "12", currency=None)
    assert build_portfolio_context([currency_unknown], as_of=AS_OF)["exclusions"][0]["reasons"] == [
        "missing_currency",
    ]


def test_overlap_only_matches_explicit_issuer_ids_and_keeps_both_dates():
    first = profile({"same": "0.2", "different-id-a": "0.1"})
    second = profile({"same": "0.05", "different-id-b": "0.4"}, report_date="2026-09-30")
    first["portfolio"]["payload"]["holdings"].append(
        {"issuer_id": "same", "kind": "corporate_equity", "weight": "0.1"},
    )
    second["portfolio"]["payload"]["holdings"].append(
        {"issuer_id": "different-id-a", "kind": "fund", "weight": "0.1"},
    )
    result = pairwise_issuer_overlap(first, second)
    assert Decimal(result["overlap"]) == Decimal("0.05")
    assert Decimal(result["common_issuers"][0]["left_weight"]) == Decimal("0.3")
    assert result["left_report_date"] == "2026-06-30"
    assert result["right_report_date"] == "2026-09-30"
    assert len(result["common_issuers"]) == 1
    assert result["exclusions"]["right"][0]["reason"] == "not_explicit_corporate_equity"
    assert pairwise_issuer_overlap(first, profile(denominator="equity_assets"))["overlap"] is None
    assert pairwise_issuer_overlap(None, first)["overlap"] is None
    missing = profile()
    missing["portfolio"]["payload"].pop("holdings")
    assert pairwise_issuer_overlap(missing, first)["overlap"] is None


def test_disclosure_exclusions_keep_unknown_identity_and_non_nav_denominator():
    partial = profile({"known": "0.1"})
    partial["portfolio"]["payload"]["holdings"].extend([
        {"name": "known", "kind": "corporate_equity", "weight": "0.2"},
        {"issuer_id": "unknown-weight", "kind": "corporate_equity", "weight": None},
    ])
    result = build_portfolio_context([
        position("a", "100", public_profile=partial),
        position("b", "100", public_profile=profile(denominator="total_assets")),
    ], as_of=AS_OF)
    assert Decimal(result["look_through"]["issuers"]["covered_value_cny"]) == 10
    assert Decimal(result["look_through"]["issuers"]["uncovered_value_cny"]) == 190
    assert result["funds"][1]["disclosure"]["unknowns"] == ["non_nav_denominator"]
    assert result["funds"][0]["disclosure"]["exclusions"][0]["reason"] == "missing_issuer_id"
    too_much = profile({"a": "0.8", "b": "0.3"})
    with pytest.raises(ValidationError):
        pairwise_issuer_overlap(too_much, partial)
    for invalid in (True, "NaN", "-1"):
        with pytest.raises(ValidationError):
            build_portfolio_context([position("bad", invalid)], as_of=AS_OF)
    dated = profile()
    dated["portfolio"]["published_at"] = "2026-08-20"
    dated["portfolio"]["publication_precision"] = "date"
    assert build_portfolio_context([position("a", "100", public_profile=dated)],
                                   as_of=AS_OF)["look_through"]["issuers"]["coverage"] == "0.3"
    dated["portfolio"]["available_at"] = "2026-10-27T09:00:00+08:00"
    future = build_portfolio_context([position("a", "100", public_profile=dated)], as_of=AS_OF)
    assert future["funds"][0]["disclosure"]["status"] == "unavailable"
    assert "available_at_after_as_of" in future["funds"][0]["disclosure"]["unknowns"]
    mismatched = {**profile(), "product_id": "some-other-explicit-fund"}
    wrong = build_portfolio_context([position("a", "100", public_profile=mismatched)], as_of=AS_OF)
    assert "profile_product_id_mismatch" in wrong["funds"][0]["disclosure"]["unknowns"]
    assert wrong["funds"][0]["disclosure"]["report_date"] == "2026-06-30"


def test_decimal_sizing_cancels_repeating_k_exactly_and_reports_binding_constraints():
    with localcontext() as context:
        context.prec = 7
        result = conditional_size(conditions())
    # k=.17/1.02=1/6 repeats; dividing a rounded k would not give exactly 120.
    assert Decimal(result["gross_cash_cap"]) == 120
    assert result["whole_fund"]["binding_constraints"] == ["scenario_loss_budget"]
    assert result["whole_fund"]["budget_feasible_at_cap"] is True
    assert Decimal(result["whole_fund"]["scenario_loss_at_cap"]) == 30
    assert Decimal(result["whole_fund"]["k_numerator"]) == Decimal("0.17")
    limits = conditional_size(conditions(max_investment="100", exposure_cap="80"))
    assert limits["gross_cash_cap"] == "80"
    assert limits["whole_fund"]["binding_constraints"] == ["exposure_cap"]
    fees_alone = conditional_size(conditions(scenario_return="0", subscription_fee="0.1",
                                             existing_scenario_loss="0", loss_budget="10"))
    assert Decimal(fees_alone["gross_cash_cap"]) == 110
    json.dumps(result, allow_nan=False)


def test_existing_loss_over_budget_and_same_scenario_reduction_including_fee():
    result = conditional_size(conditions(existing_scenario_loss="50", redemption_fee="0.05",
                                         existing_market_value="300"))
    assert Decimal(result["gross_cash_cap"]) == 0
    assert result["whole_fund"]["feasible"] is False
    reduction = result["reduction"]
    assert reduction["status"] == "available"
    assert Decimal(reduction["minimum_gross_redemption"]) == 200
    assert Decimal(reduction["fee_at_minimum"]) == 10
    assert Decimal(reduction["net_proceeds_at_minimum"]) == 190
    assert Decimal(reduction["loss_after_minimum_redemption"]) == 30
    infeasible = conditional_size(conditions(existing_scenario_loss="50", redemption_fee="0.05",
                                              available_redemption_value="199"))["reduction"]
    assert infeasible["status"] == "infeasible"
    assert Decimal(infeasible["minimum_gross_redemption"]) == 200
    assert infeasible["feasible_minimum_gross_redemption"] is None


@pytest.mark.parametrize("fee", ["0.15", "0.16"])
def test_reduction_does_not_invent_benefit_when_loss_fraction_is_not_above_fee(fee):
    result = conditional_size(conditions(existing_scenario_loss="50", redemption_fee=fee,
                                         existing_market_value="300"))["reduction"]
    assert result["status"] == "no_loss_relief"
    assert result["minimum_gross_redemption"] is None
    assert result["feasible"] is False


@pytest.mark.parametrize("change", ["0.02", "0.1"])
def test_no_positive_loss_bound_is_not_an_infinite_cash_recommendation(change):
    result = conditional_size(conditions(scenario_return=change, max_investment="70"))
    assert result["whole_fund"]["loss_bound_status"] == "no_positive_loss_bound"
    assert result["whole_fund"]["loss_bound_cap"] is None
    assert result["gross_cash_cap"] == "70"
    assert result["whole_fund"]["binding_constraints"] == ["max_investment"]
    gain = conditional_size(conditions(scenario_return="0.12", subscription_fee="0.02",
                                      existing_scenario_loss="40", loss_budget="30"))
    assert gain["whole_fund"]["loss_bound_status"] == "no_positive_loss_bound"
    assert Decimal(gain["whole_fund"]["minimum_gross_cash_for_budget"]) == 102


def test_missing_inputs_partial_scenarios_and_fees_remain_explicit():
    result = conditional_size({})
    assert result["gross_cash_cap"] is None
    assert "spendable_cash" in result["unknowns"]
    assert "existing_scenario_loss" in result["unknowns"]
    for field in ("subscription_fee", "loss_budget", "existing_scenario_loss", "spendable_cash"):
        data = conditions()
        data.pop(field)
        missing = conditional_size(data)
        assert missing["gross_cash_cap"] is None
        assert field in missing["unknowns"]
    partial = conditional_size(conditions(scenario_scope="disclosed_basket"))
    assert partial["gross_cash_cap"] is None
    assert "whole_fund_scenario_required" in partial["unknowns"]
    unknown_fee = post_buy_weight(gross_cash="100", known_cohort_market_value="1000",
                                  selected_market_value="100")
    assert unknown_fee["net_subscription"] is None
    assert unknown_fee["weight_after"] is None
    with pytest.raises(ValidationError):
        conditional_size(conditions(subscription_fee=True))


def test_post_buy_weight_is_net_of_external_fee_and_explicit_weight_cap_is_optional():
    result = post_buy_weight(gross_cash="102", subscription_fee="0.02",
                             known_cohort_market_value="900", selected_market_value="100")
    assert Decimal(result["net_subscription"]) == 100
    assert result["subscription_fee_amount"] == "2"
    assert Decimal(result["weight_after"]) == Decimal("0.2")
    unknown = post_buy_weight(gross_cash="102", subscription_fee="0.02",
                              known_cohort_market_value="900", selected_market_value=None)
    assert unknown["weight_after"] is None
    capped = conditional_size(conditions(
        known_cohort_market_value="900", selected_market_value="100", max_fund_weight="0.2",
        requested_gross_cash="102",
    ))
    assert Decimal(capped["gross_cash_cap"]) == 102
    assert capped["whole_fund"]["binding_constraints"] == ["max_fund_weight"]
    assert Decimal(capped["post_buy"]["weight_after"]) == Decimal("0.2")
    assert capped["post_buy"]["within_conditional_cap"] is True
    sold = planned_redemption(gross_redemption="100", redemption_fee="0.01",
                              known_cohort_market_value="900", selected_market_value="200")
    assert sold["net_proceeds"] == "99.00"
    assert sold["remaining_cohort_market_value"] == "800"
    assert Decimal(sold["weight_after"]) == Decimal("0.125")
    with pytest.raises(ValidationError):
        planned_redemption(gross_redemption="201", redemption_fee="0",
                           known_cohort_market_value="900", selected_market_value="200")

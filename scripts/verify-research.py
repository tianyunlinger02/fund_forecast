"""Validate a retained research run without network or modifying its source DB."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rmbfund.domain import Product, ValidationError, decimal, digest
from rmbfund.research_replay import (
    MONTHLY_STRATEGY,
    Assumptions,
    evaluate_frozen,
    public_snapshot,
    replay_one,
    selected_strategies,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def order_intents(result: dict, cutoff: str, *, inclusive: bool = False) -> list[dict]:
    keys = ("id", "product_id", "strategy", "side", "signal_at", "signal_price_date",
            "signal_fact_id", "requested_units", "gross_budget")
    return [{key: order.get(key) for key in keys} for order in result["orders"]
            if order["signal_at"][:10] < cutoff
            or (inclusive and order["signal_at"][:10] == cutoff)]


def economic_fills(result: dict, cutoff: str) -> list[dict]:
    keys = ("id", "side", "economic_date", "application_date", "price", "price_fact_id",
            "units", "gross", "fee", "fee_details", "rounding_loss")
    return [{key: order.get(key) for key in keys} for order in result["orders"]
            if order.get("economic_date") and order["economic_date"] < cutoff]


def short_holding_counts(result: dict) -> dict:
    lots = [sum(detail["holding_days"] < 7 for detail in order.get("fee_details", []))
            for order in result["orders"] if order["side"] == "sell"
            and order["state"] in {"filled", "priced_unconfirmed"}]
    return {"redemptions_under_7_natural_days": sum(count > 0 for count in lots),
            "redemption_lots_under_7_natural_days": sum(lots)}


def monthly_audit(result: dict, model: Assumptions) -> dict:
    first_days = {}
    for day in result["daily"]:
        first_days.setdefault(day["date"][:7], day["date"])
    expected = list(first_days.values())
    check([signal["date"] for signal in result["signals"]] == expected, "Monthly evaluation schedule differs")
    attempts = Counter(order["signal_at"][:7] for order in result["orders"])
    check(max(attempts.values(), default=0) <= 1, "More than one monthly application attempt")
    for order in result["orders"]:
        check(order["signal_at"][:10] in expected, "Intra-month order attempt")
        if order.get("application_date"):
            check(order["application_date"] == order["signal_at"][:10], "Queued monthly application")
        if order["state"] not in {"filled", "priced_unconfirmed"}:
            continue
        if order["side"] == "buy":
            budget = decimal(order["gross_budget"])
            if model.fund_purchase_min is not None:
                check(budget >= decimal(model.fund_purchase_min), "Purchase below minimum")
            if model.fund_daily_purchase_cap is not None:
                check(budget <= decimal(model.fund_daily_purchase_cap), "Purchase above cap")
        else:
            check(bool(order["fee_details"]), "Missing redemption holding-period evidence")
            check(all(detail["holding_days"] >= 30 for detail in order["fee_details"]), "Monthly redemption before 30 natural days")
    return {"symbol": result["product"]["symbol"], "scenario": result["scenario"],
            "strategy": result["strategy"], "checks": len(expected), "orders": len(result["orders"]),
            "monthly_schedule_and_minimum_holding_passed": True}


def unknown_nav_audit(item: dict, original: dict, model: Assumptions, *, start: str, end: str) -> list[dict]:
    product = Product(**item["product"])
    indices = {row["event_date"]: index for index, row in enumerate(item["prices"])}
    checks = []
    for signal in original["signals"]:
        decision = signal["date"]
        index = indices[decision]
        # Each not-yet-known NAV is varied separately, including the application
        # day and earlier NAVs still hidden by the scenario's publication lag.
        for unknown_index in range(index - model.nav_lag_days + 1, index + 1):
            modified = copy.deepcopy(item["prices"])
            row = modified[unknown_index]
            row["payload"]["unit_nav"] = str(decimal(row["payload"]["unit_nav"]) * 2)
            changed = replay_one(product, modified, item["actions"], model, start=start, end=end,
                                 strategy=MONTHLY_STRATEGY, actions_covered=item["action_coverage"])
            prior = [s for s in original["signals"] if s["date"] <= decision]
            changed_prior = [s for s in changed["signals"] if s["date"] <= decision]
            check(prior == changed_prior, "Not-yet-known NAV affects monthly signal")
            check(order_intents(original, decision, inclusive=True) == order_intents(changed, decision, inclusive=True),
                  "Not-yet-known NAV affects monthly intent")
            checks.append({"symbol": product.symbol, "scenario": original["scenario"],
                           "decision_date": decision, "perturbed_unknown_nav_date": row["event_date"],
                           "decision_and_intent_unchanged": True})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--public-db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite existing validation output")
    inputs = json.loads((args.run / "inputs.json").read_text(encoding="utf-8"))
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    rebuilt = evaluate_frozen(inputs)
    check(digest(rebuilt) == digest(report), "Frozen economic output differs")
    public = public_snapshot(args.public_db)
    check(inputs["source_public_hash"] == public["hash"], "Public DB changed")
    checked = []
    for evidence in inputs.get("source_evidence", []):
        path = Path(evidence["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        check(actual == evidence["content_hash"], f"Evidence changed: {path}")
        checked.append({"path": str(path), "sha256": actual})
    fee_audit, monthly_checks = [], []
    for result in report["results"]:
        for day in result["daily"]:
            total = sum((decimal(day[key]) for key in ("cash_available", "cash_reserved", "cash_receivable", "market_value")), decimal(0))
            check(total == decimal(day["model_assets"]), "Asset components do not reconcile")
            check(decimal(day["available_units"]) <= decimal(day["units"]), "Available units exceed total units")
        fees = sum((decimal(order["fee"]) for order in result["orders"]), decimal(0))
        check(fees == decimal(result["metrics"]["fees"]), "Order fees do not reconcile")
        fee_audit.append({"symbol": result["product"]["symbol"], "scenario": result["scenario"],
                          "strategy": result["strategy"], "fees": str(fees), **short_holding_counts(result),
                          "ending_cash_receivable": result["daily"][-1]["cash_receivable"]})
        if result["strategy"] == MONTHLY_STRATEGY:
            monthly_checks.append(monthly_audit(result, Assumptions(**result["assumptions"])))
    causality, own_nav_checks = [], []
    for item in inputs["products"]:
        if item.get("blocked"):
            continue
        product = Product(**item["product"])
        for scenario in item.get("scenarios", [{"id": "standard", "assumptions": {}}]):
            model = Assumptions(**{**inputs["assumptions"], **scenario.get("assumptions", {})})
            for strategy in selected_strategies(inputs):
                original = next(result for result in report["results"]
                                if result["product"]["id"] == product.id
                                and result["scenario"] == scenario["id"] and result["strategy"] == strategy)
                modified = copy.deepcopy(item["prices"])
                active = [row for row in modified if inputs["start"] <= row["event_date"] <= inputs["end"]]
                cutoff = active[len(active) // 2]["event_date"]
                for row in modified:
                    if row["event_date"] >= cutoff:
                        for key in ("open", "close", "high", "low") if product.kind == "etf" else ("unit_nav",):
                            row["payload"][key] = str(decimal(row["payload"][key]) * 2)
                changed = replay_one(product, modified, item["actions"], model, start=inputs["start"],
                                     end=inputs["end"], strategy=strategy, actions_covered=item["action_coverage"])
                for key in ("signals", "daily", "events"):
                    # Retained JSON encodes Decimal event amounts as strings.
                    prior = [row for row in original[key] if row["date"] < cutoff]
                    perturbed_prior = [row for row in changed[key] if row["date"] < cutoff]
                    check(digest(prior) == digest(perturbed_prior), f"Future prices affect past {key}")
                check(order_intents(original, cutoff) == order_intents(changed, cutoff), "Future prices affect past order intents")
                check(economic_fills(original, cutoff) == economic_fills(changed, cutoff), "Future prices affect past fills")
                causality.append({"symbol": product.symbol, "scenario": scenario["id"], "strategy": strategy,
                                  "future_doubled_from": cutoff, "unchanged_prior_signals_and_assets": True,
                                  "unchanged_prior_order_intents_and_fills": True, "assumptions": asdict(model)})
                if strategy == MONTHLY_STRATEGY:
                    own_nav_checks.extend(unknown_nav_audit(item, original, model, start=inputs["start"], end=inputs["end"]))
    check(public_snapshot(args.public_db)["hash"] == public["hash"], "Public DB changed during audit")
    validation = {"mode": "research_replay_validation", "all_checks_passed": True,
                  "input_hash": digest(inputs), "economic_result_hash": digest(rebuilt),
                  "public_unchanged": True, "public_hash": public["hash"],
                  "source_tables": public["tables"], "source_evidence_checked": checked,
                  "cash_components_and_fees_reconciled": True, "fee_audit": fee_audit,
                  "future_perturbation_checks": causality, "monthly_rule_checks": monthly_checks,
                  "unknown_application_nav_checks": own_nav_checks,
                  "warning": "Arithmetic and causal checks do not prove real execution or verified historical rules."}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "public_unchanged": True, "evidence_files": len(checked),
                      "result_paths": len(report["results"]), "future_perturbations": len(causality),
                      "monthly_rule_checks": len(monthly_checks), "unknown_nav_checks": len(own_nav_checks)}, indent=2))


if __name__ == "__main__":
    main()

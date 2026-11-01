"""Synthetic private holdings: passive reads, exact values, and public isolation."""

from __future__ import annotations


from copy import deepcopy


from decimal import Decimal, localcontext


import stat
import sqlite3

import pytest

from rmbfund.domain import ValidationError, timestamp


from rmbfund.personal_assessment import build_assessment


from rmbfund.portfolio import Portfolio, value_holding


NOW = "2026-10-23T10:00:00+08:00"


LATER = "2026-10-23T11:00:00+08:00"


NAV = {"value": "2.161", "currency": "CNY", "date": "2026-10-21"}


SUBJECT = {
    "id": "actual-public-product-id",
    "symbol": "000001",
    "name": "合成测试基金C",
    "kind": "open_fund",
    "share_class": "C",
    "exchange": "",
    "is_demo": False,
}


def pending(**updates):
    subject = {key: value for key, value in SUBJECT.items() if key != "id"}
    subject.update(updates)
    return subject


def holding(shares=None, cost=None, **updates):
    return {"shares": shares, "cost": cost, "currency": "CNY", **updates}


def test_crud_reopens_preserves_private_id_and_exact_text(tmp_path, monkeypatch):
    times = iter((NOW, LATER))
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: next(times))
    root = tmp_path / "personal"
    first = Portfolio(root).save_fund(
        {**SUBJECT, "category": "公开分类"},
        shares=" 0012.3400 ",
        cost="00020.5000",
    )
    assert set(first) == {
        "id",
        "product_id",
        "subject",
        "shares",
        "cost",
        "currency",
        "created_at",
        "updated_at",
    }
    assert first["id"] != SUBJECT["id"]
    assert first["product_id"] == SUBJECT["id"]
    assert first["subject"] == pending()
    assert "id" not in first["subject"]
    assert "category" not in first["subject"]
    assert first["shares"] == "0012.3400"
    assert first["cost"] == "00020.5000"
    assert first["currency"] == "CNY"
    assert first["created_at"] == first["updated_at"] == NOW
    assert timestamp(first["created_at"]).tzinfo is not None
    assert Portfolio(root).list_funds() == [first]
    assert Portfolio(root).get_fund(first["id"]) == first
    assert Portfolio(root).get_fund("absent") is None
    assert stat.S_IMODE((root / "personal-public.sqlite3").stat().st_mode) == 0o600

    corrected = Portfolio(root).save_fund(
        {**SUBJECT, "name": "更正公开名称"},
        shares="0",
        cost="0",
        fund_id=first["id"],
    )
    assert corrected["id"] == first["id"]
    assert corrected["created_at"] == NOW
    assert corrected["updated_at"] == LATER
    assert corrected["subject"]["name"] == "更正公开名称"
    assert corrected["shares"] == corrected["cost"] == "0"
    assert Portfolio(root).list_funds() == [corrected]
    Portfolio(root).remove_fund(first["id"])
    assert Portfolio(root).get_fund(first["id"]) is None
    assert Portfolio(root).list_funds() == []
    Portfolio(root).remove_fund(first["id"])


def test_value_holding_unknown_and_explicit_zero_are_distinct():
    unknown = value_holding(holding(), NAV)
    assert unknown == {
        "shares": None,
        "cost": None,
        "currency": "CNY",
        "market_value": None,
        "profit": None,
        "profit_rate": None,
        "nav_date": "2026-10-21",
        "valuation_status": "missing_shares",
    }
    zero = value_holding(holding("0", "0"), NAV)
    assert Decimal(zero["market_value"]) == Decimal(zero["profit"]) == 0
    assert zero["profit_rate"] is None
    assert zero["valuation_status"] == "ready"
    assert value_holding(holding("0"), None)["market_value"] is None
    positive_cost = value_holding(holding("0", "12.00"), NAV)
    assert positive_cost["profit"] == "-12.000"
    assert Decimal(positive_cost["profit_rate"]) == -1


def test_value_holding_precise_decimal_arithmetic_and_cost_cohort():
    shares = "12345678901234567890123.123456789012345678901234"
    nav = {**NAV, "value": "2.161000000000000000000001"}
    cost = "500.000000000000000000000001"
    with localcontext() as context:
        context.prec = 128
        expected_market = Decimal(shares) * Decimal(nav["value"])
        expected_profit = expected_market - Decimal(cost)
        expected_rate = expected_profit / Decimal(cost)
    with localcontext() as context:
        context.prec = 9
        valued = value_holding(holding(shares, cost), nav)
    assert valued["shares"] == shares
    assert valued["cost"] == cost
    assert valued["market_value"] == str(expected_market)
    assert valued["profit"] == str(expected_profit)
    assert valued["profit_rate"] == str(expected_rate)
    without_cost = value_holding(holding("10"), NAV)
    assert without_cost["market_value"] == "21.610"
    assert without_cost["profit"] is without_cost["profit_rate"] is None


def assessment(subject=None, **kwargs):
    subject = SUBJECT if subject is None else subject
    head = {
        "horizon": 20,
        "entry_lag": 2,
        "expected_change": 0.02,
        "lower": -0.1,
        "upper": 0.1,
        "up_probability": 0.6,
        "samples": [-0.1, 0.02, 0.1],
        "method": "empirical_historical",
    }
    prediction = {
        "record_id": "original-public-forecast",
        "product_id": subject.get("id"),
        "model_id": "original-model",
        "model_hash": "original-hash",
        "input_hash": "input-hash",
        "generated_at": NOW,
        "anchor": {"date": NAV["date"], "nav": NAV["value"], "currency": "CNY"},
        "entry_assumptions": {"baseline_entry_lag": 2},
        "forecasts": [{**head, "entry_lag": 0}],
        "execution_forecasts": [head],
    }
    return build_assessment(prediction, subject=subject, **kwargs)


def test_snapshots_are_immutable_after_current_edits_remove_and_readd(tmp_path):
    portfolio = Portfolio(tmp_path)
    original = portfolio.save_fund(SUBJECT, shares="10", cost="12")
    shown = assessment(holding=original, subscription_fee="0.01")
    saved = portfolio.save_assessment(shown, submission_id="snapshot-original")
    frozen = deepcopy(saved)
    # Neither today's edited inputs nor caller mutation can rewrite the original display.
    portfolio.save_fund(SUBJECT, fund_id=original["id"], shares="1000", cost="2000")
    shown["inputs"]["holding"]["shares"] = "999"
    shown["decision"]["branches"]["held"]["label"] = "changed now"
    saved["assessment"]["inputs"]["subscription_fee"] = "0.9"
    portfolio.remove_fund(original["id"])
    assert portfolio.assessment_history(SUBJECT) == [frozen]
    replaced = portfolio.save_fund(SUBJECT, shares="3", cost="4")
    assert replaced["id"] != original["id"]
    assert portfolio.assessment_detail(frozen["id"]) == frozen
    assert portfolio.assessment_history(SUBJECT) == [frozen]


def _tables(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()


def _note(portfolio, assessment_id, **updates):
    fields = {"kind": "simulated_choice", "branch": "flat", "action": "buy",
              "occurred_on": "2026-10-26", "rationale": "合成选择，不是成交"}
    fields.update(updates)
    return portfolio.save_action_note(assessment_id, **fields)


def test_action_note_reads_do_not_create_private_file_or_migrate_existing_schema(tmp_path):
    root = tmp_path / "absent"
    portfolio = Portfolio(root)
    assert portfolio.action_notes("assessment-absent") == []
    assert not root.exists()
    with pytest.raises(ValidationError, match="原判断不存在"):
        _note(portfolio, "assessment-absent")
    assert not root.exists()
    portfolio.save_fund(SUBJECT)
    before = _tables(portfolio.path)
    raw = portfolio.path.read_bytes()
    assert portfolio.action_notes("assessment-absent") == []
    assert _tables(portfolio.path) == before
    assert portfolio.path.read_bytes() == raw
    with pytest.raises(ValidationError, match="原判断不存在"):
        _note(portfolio, "assessment-absent")
    assert _tables(portfolio.path) == before


def test_action_notes_append_idempotently_and_never_rewrite_snapshots_or_holdings(tmp_path, monkeypatch):
    portfolio = Portfolio(tmp_path)
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: NOW)
    held = portfolio.save_fund(SUBJECT, shares="12.30", cost="10.00")
    saved = portfolio.save_assessment(assessment())
    frozen = deepcopy(saved)
    assert "assessment_action_notes" not in {row[1] for row in _tables(portfolio.path)}
    before = _tables(portfolio.path)
    assert portfolio.action_notes(saved["id"]) == []
    assert _tables(portfolio.path) == before
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: LATER)
    note = _note(portfolio, saved["id"], submission_id="note-click")
    assert note == {
        "id": note["id"], "submission_id": "note-click", "assessment_id": saved["id"],
        "product_id": SUBJECT["id"], "kind": "simulated_choice", "branch": "flat", "action": "buy",
        "occurred_on": "2026-10-26", "recorded_at": LATER, "rationale": "合成选择，不是成交",
    }
    assert timestamp(note["recorded_at"]).tzinfo is not None
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: pytest.fail("retry must keep original clock"))
    assert _note(portfolio, saved["id"], submission_id="note-click") == note
    with pytest.raises(ValidationError, match="不同内容"):
        _note(portfolio, saved["id"], action="wait", submission_id="note-click")
    assert portfolio.action_notes(saved["id"]) == [note]
    assert portfolio.assessment_detail(saved["id"]) == frozen
    assert portfolio.get_fund(held["id"]) == held
    assert stat.S_IMODE(portfolio.path.stat().st_mode) == 0o600
    with sqlite3.connect(portfolio.path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE assessment_action_notes SET action='wait' WHERE id=?", (note["id"],))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM assessment_action_notes WHERE id=?", (note["id"],))
    assert portfolio.action_notes(saved["id"]) == [note]


def test_note_scope_is_original_assessment_and_product_and_default_idempotency(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: NOW)
    portfolio = Portfolio(tmp_path)
    first = portfolio.save_assessment(assessment(), submission_id="first")
    second = portfolio.save_assessment(assessment(), submission_id="same-product-later")
    another_subject = {**SUBJECT, "id": "other-public-id", "symbol": "999999"}
    other = portfolio.save_assessment(assessment(another_subject))
    one = _note(portfolio, first["id"])
    assert _note(portfolio, first["id"]) == one
    two = _note(portfolio, second["id"])
    other_note = _note(portfolio, other["id"])
    assert one["id"] != two["id"]
    assert other_note["product_id"] == another_subject["id"]
    assert Portfolio(tmp_path).action_notes(first["id"]) == [one]
    assert portfolio.action_notes(second["id"]) == [two]
    assert portfolio.action_notes(other["id"]) == [other_note]
    assert portfolio.action_notes("absent") == []
    assert Portfolio(tmp_path, "demo").action_notes(first["id"]) == []
    assert not Portfolio(tmp_path, "demo").path.exists()
    with pytest.raises(ValidationError, match="不同内容"):
        _note(portfolio, first["id"], submission_id=other_note["submission_id"])
    assert not any("journal" in row[1] for row in _tables(portfolio.path))


def test_all_note_kinds_actions_and_historical_occurrence_remain_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.portfolio.now_iso", lambda: NOW)
    portfolio = Portfolio(tmp_path)
    saved = portfolio.save_assessment(assessment())
    notes = []
    for kind in ("plan", "simulated_choice", "user_report", "observation"):
        notes.append(_note(portfolio, saved["id"], kind=kind, action="no_action",
                           occurred_on="2020-01-01", rationale="用户历史描述，不回填记录时钟"))
    for action in ("hold", "add", "reduce50", "exit", "reduce", "redeem", "no_action"):
        notes.append(_note(portfolio, saved["id"], action=action, branch="held"))
    assert all(row["recorded_at"] == NOW for row in notes)
    assert portfolio.action_notes(saved["id"]) == list(reversed(notes))
    assert portfolio.action_notes(saved["id"], limit=2, offset=1) == list(reversed(notes))[1:3]
    assert portfolio.list_funds() == []


@pytest.mark.parametrize("fields", [
    {"kind": "trade"}, {"kind": []}, {"branch": "addition"}, {"branch": []},
    {"action": "hold"}, {"action": []}, {"occurred_on": "2026-02-30"},
    {"occurred_on": "20261026"}, {"occurred_on": None}, {"rationale": "x" * 1001},
])
def test_invalid_action_notes_do_not_initialize_private_schema(tmp_path, fields):
    portfolio = Portfolio(tmp_path / "absent")
    with pytest.raises(ValidationError):
        _note(portfolio, "missing", **fields)
    assert not portfolio.root.exists()


@pytest.mark.parametrize("fields", [{"limit": 0}, {"limit": True}, {"offset": -1}])
def test_action_note_paging_requires_real_integers(tmp_path, fields):
    portfolio = Portfolio(tmp_path / "absent")
    with pytest.raises(ValidationError):
        portfolio.action_notes("missing", **fields)
    assert not portfolio.root.exists()

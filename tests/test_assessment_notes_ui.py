"""Real Streamlit note/replay interactions use only an isolated synthetic portfolio."""
from copy import deepcopy
from pathlib import Path
import sqlite3

from streamlit.testing.v1 import AppTest

from rmbfund.portfolio import Portfolio
from test_assessment_replay import _fixture
from test_ui import click, text


def show_notes(root, identifier, series):
    from pathlib import Path
    import streamlit as st
    from rmbfund.assessment_replay import replay_saved_assessment
    from rmbfund.portfolio import Portfolio
    from rmbfund.ui.assessment_notes import render_assessment_notes

    repository = Portfolio(Path(root))
    saved = repository.assessment_detail(identifier)
    st.session_state.setdefault("replay_calls", 0)

    def replay(original, *, note=None):
        st.session_state["replay_calls"] = st.session_state.get("replay_calls", 0) + 1
        return replay_saved_assessment(original, series=series, note=note)

    render_assessment_notes(saved, read=repository.action_notes, save=repository.save_action_note, replay=replay)


def setup_ui(tmp_path, *, immature=False):
    saved, series = _fixture()
    repository = Portfolio(tmp_path)
    saved["assessment"]["decision"] = deepcopy(saved["assessment"]["trade_scenario"]["decision"])
    receipt = repository.save_assessment(saved["assessment"])
    if immature:
        series = series[:3]
    return repository, receipt, AppTest.from_function(show_notes, args=(str(tmp_path), receipt["id"], series)).run()


def test_note_save_is_optional_idempotent_and_never_changes_original_assessment(tmp_path):
    repository, receipt, at = setup_ui(tmp_path)
    original = deepcopy(repository.assessment_detail(receipt["id"]))
    assert not at.exception and repository.action_notes(receipt["id"]) == []
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='assessment_action_notes'").fetchone() is None
    at.text_area[0].input("合成计划：先留现金，等经营回款资料")
    click(at, "保存这条笔记")
    assert not at.exception
    notes = repository.action_notes(receipt["id"])
    assert len(notes) == 1 and notes[0]["rationale"].startswith("合成计划")
    assert notes[0]["kind"] == "simulated_choice"
    assert notes[0]["action"] == "buy"
    assert notes[0]["recorded_at"] and any("笔记已保存" in message.value for message in at.success)
    assert repository.assessment_detail(receipt["id"]) == original
    assert not at.session_state["replay_calls"]
    click(at, "保存这条笔记")
    assert len(repository.action_notes(receipt["id"])) == 1


def test_explicit_mature_review_uses_original_choice_and_labels_post_hoc(tmp_path):
    repository, receipt, at = setup_ui(tmp_path)
    assert not at.session_state["replay_calls"]
    click(at, "用已有净值查看这次模拟结果")
    assert not at.exception and at.session_state["replay_calls"] == 1
    shown = text(at)
    assert "可以核对" in shown and "模拟卖出净额" in shown
    assert "事后模拟" in shown and "不是个人账户收益" in shown
    assert "1,072.77" in shown  # Original B=1000, fee .01/.015 and exact L1/H2.
    assert not repository.action_notes(receipt["id"])


def test_unmature_review_and_self_report_are_not_fabricated_account_results(tmp_path):
    repository, receipt, at = setup_ui(tmp_path, immature=True)
    click(at, "用已有净值查看这次模拟结果")
    assert not at.exception and "尚未走完" in text(at)
    assert "模拟卖出净额" not in text(at)
    at.selectbox(key=f"note_kind_{receipt['id']}").set_value("user_report")
    click(at, "保存这条笔记")
    notes = repository.action_notes(receipt["id"])
    assert len(notes) == 1 and notes[0]["kind"] == "user_report"
    at.selectbox(key=f"note_replay_choice_{receipt['id']}").set_value(notes[0]["id"]).run()
    click(at, "用已有净值查看这次模拟结果")
    assert not at.exception and "不计算个人真实盈亏" in text(at)
    assert "模拟卖出净额" not in text(at)


def test_app_replay_callback_reads_only_stored_nav_at_explicit_clock(tmp_path, monkeypatch):
    from rmbfund.ui.app import _replay_assessment
    from test_ui import PRODUCT

    saved, series = _fixture()
    calls = []

    class ReadOnlyWorkspace:
        def stored_nav_series(self, product_id, *, as_of):
            calls.append((product_id, as_of))
            return {"status": "ready", "series": series}

    monkeypatch.setattr("rmbfund.ui.app.now_iso", lambda: "2026-10-29T20:30:00+08:00")
    before = deepcopy(saved)
    result = _replay_assessment(ReadOnlyWorkspace(), PRODUCT["id"], saved)
    assert result["status"] == "mature"
    assert calls == [(PRODUCT["id"], "2026-10-29T20:30:00+08:00")]
    assert saved == before
    assert list(Path(tmp_path).iterdir()) == []

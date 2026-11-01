"""Explicit local session research append is not an API or manual-review route."""
from copy import deepcopy
import json

import pytest

from rmbfund.domain import ValidationError
from rmbfund.service import Workspace
from test_generated_research import PRODUCT, bound, document, note


def test_session_import_appends_selected_evidence_without_calling_research_api(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-28T12:00:00+08:00")
    def no_api(*args, **kwargs):
        raise AssertionError("local append must never invoke the API runner")
    monkeypatch.setattr("rmbfund.research_runner.run_research", no_api)
    workspace = Workspace(tmp_path / "workspace", mode="public")
    workspace.store.add_product(PRODUCT)
    workspace.store.initialize_evidence_schema()
    payload = note()
    payload["origin"]["provider"] = "interactive_assistant"
    path, evidence = tmp_path / "session.json", tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    evidence.write_text(json.dumps({"documents": [document()]}), encoding="utf-8")
    result = workspace.import_session_research(PRODUCT.id, path, evidence_path=evidence)
    assert result["research_api_called"] is False
    assert result["receipt"]["records_added"] == 1
    saved = result["research_thesis"]["payload"]
    assert saved["origin"]["provider"] == "interactive_assistant"
    assert saved["origin"]["human_reviewed"] is False
    assert saved["citations"][0]["observed_at"] == document()["observed_at"]
    assert "UNADOPTED_DOCUMENT_TEXT" not in json.dumps(saved)
    again = workspace.import_session_research(PRODUCT.id, path, evidence_path=evidence)
    assert again["receipt"]["records_added"] == 0
    assert len(workspace.store.advice_evidence(PRODUCT.id, "2026-10-28T12:00:00+08:00")) == 1
    workspace.store.close()


def test_already_bound_session_and_rejection_of_other_authorship(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: "2026-10-28T12:00:00+08:00")
    workspace = Workspace(tmp_path / "workspace", mode="public")
    workspace.store.add_product(PRODUCT)
    workspace.store.initialize_evidence_schema()
    payload = bound()
    path = tmp_path / "bound.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="interactive_assistant"):
        workspace.import_session_research(PRODUCT.id, path)
    payload["origin"]["provider"] = "interactive_assistant"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert workspace.import_session_research(PRODUCT.id, path)["research_thesis"]["payload"]["origin"]["provider"] == "interactive_assistant"
    prior = deepcopy(payload)
    prior["revision"] = {"id": "incorrect-provider", "supersedes": None}
    prior["origin"]["provider"] = "runner-supplied"
    path.write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises(ValidationError, match="interactive_assistant"):
        workspace.import_session_research(PRODUCT.id, path)
    workspace.store.close()
    read_only = Workspace(tmp_path / "workspace", mode="public", read_only=True)
    with pytest.raises(ValidationError, match="可写public"):
        read_only.import_session_research(PRODUCT.id, path)
    read_only.store.close()

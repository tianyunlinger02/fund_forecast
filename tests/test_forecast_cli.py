"""Public CLI local reads, explicit fit, frozen evaluation and no account access."""

from __future__ import annotations


import json


import pytest


from rmbfund.cli import main


from rmbfund.service import Workspace


from test_forecast_service import NOW, seed_nav


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "store"
    workspace = Workspace(root, "public")
    product_id = workspace.add_product(
        "270042", "SYNTHETIC generic fund", "open_fund", share_class="A"
    )
    seed_nav(workspace, product_id)
    workspace.store.close()
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: NOW)
    monkeypatch.setattr("rmbfund.cli.now_iso", lambda: NOW)

    def forbidden(*args, **kwargs):
        raise AssertionError("CLI must not use network/account/trades in local tests")

    monkeypatch.setattr(Workspace, "refresh", forbidden)
    monkeypatch.setattr("rmbfund.paper.engine.PaperEngine.submit", forbidden)
    monkeypatch.setattr("rmbfund.paper.engine.PaperEngine.create_account", forbidden)
    return root, product_id


def command(context, name, *extra):
    return ["--root", str(context[0]), "--mode", "public", name, context[1], *map(str, extra)]


def dump(root):
    workspace = Workspace(root, "public", read_only=True)
    try:
        return list(workspace.store.conn.iterdump())
    finally:
        workspace.store.close()


def test_explicit_local_update_then_read_and_duplicate_reuses_record(context, capsys):
    assert main(command(context, "forecast-update", "--local")) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ready"
    assert first["prediction"]["generated_at"] == NOW
    before = dump(context[0])
    assert main(command(context, "forecast", "--local")) == 0
    assert json.loads(capsys.readouterr().out)["prediction"] == first["prediction"]
    assert dump(context[0]) == before
    assert main(command(context, "forecast-update", "--local")) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["prediction"] == first["prediction"]
    assert dump(context[0]) == before

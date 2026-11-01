"""Daily online retrieval, same-context comparison and database-independent CLI."""
from copy import deepcopy
from datetime import datetime
import json

from rmbfund.buyer_research import current_public_identity, install_buyer_config
from rmbfund.cli import main
from rmbfund.portfolio import value_holding
from rmbfund.providers.followed import FetchResponse
from rmbfund.service import Workspace

AT = "2026-10-26T12:00:00+08:00"
TARGET = "open_fund:900001:C"
PEER = "open_fund:900002:A"
PEERS = {"target_product_id": TARGET, "start_date": "2026-07-01", "end_date": "2026-10-21",
         "selection_reason": "Synthetic cost alternative", "evidence_refs": ["synthetic-contract"],
         "peers": [{"product_id": PEER, "relation": "style_or_cost_alternative",
                    "selection_reason": "Synthetic alternative, not a ranking",
                    "evidence_refs": ["synthetic-contract"]}]}


def nav_bytes(code, values=(1, 1.1, 1.2)):
    rows = [{"x": int(datetime.fromisoformat(f"2026-10-{day}T00:00:00+08:00").timestamp() * 1000),
             "y": value} for day, value in zip((21, 22, 23), values)]
    return (f'fS_code="{code}";Data_netWorthTrend=' + json.dumps(rows) + ';').encode()


def test_readonly_workspace_uses_ram_for_current_valuation_and_both_peer_windows(tmp_path, monkeypatch):
    root = tmp_path / "synthetic"
    workspace = Workspace(root, "public")
    workspace.add_product("900001", "synthetic", "open_fund", share_class="C")
    workspace.store.close()
    monkeypatch.setattr("rmbfund.buyer_research.now_iso", lambda: AT)
    install_buyer_config(root, {"peer_sets": {TARGET: PEERS}, "fund_metadata": {
        TARGET: {"currency": "CNY", "underlying_fund_id": "synthetic-one", "evidence_refs": ["synthetic-contract"]},
        PEER: {"currency": "CNY", "underlying_fund_id": "synthetic-two", "evidence_refs": ["synthetic-contract"]}}})
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: AT)
    workspace = Workspace(root, "public", read_only=True, initialize=False)
    try:
        scope = workspace.current_subjects(current_public_identity(TARGET), additional=[current_public_identity(TARGET)])
        assert [row["id"] for row in scope["public_subjects"]] == [TARGET, PEER]
        assert scope["source_plan"] is None  # Not the old archive plan.
        jobs = [{"job_id": str(index), "product_id": row["id"], "family": "nav",
                 "source_url": f'https://public.example.org/{row["symbol"]}.js',
                 "parser": "eastmoney_nav_js_v1", "config": {}}
                for index, row in enumerate(scope["public_subjects"])]
        context = workspace.current_online(scope["public_subjects"], source_plan=jobs,
            fetch=lambda url, *_: FetchResponse(200, nav_bytes(url.rsplit('/', 1)[1][:6])), clock=lambda: AT)
        evaluation = workspace.current_evaluation(TARGET, context, public_subjects=scope["public_subjects"],
                                                   peer_set=scope["peer_set"])
        detail = evaluation["details"][TARGET]
        assert detail["latest"]["value"] == "1.2"
        holding = {"subject": current_public_identity(TARGET), "shares": "100", "cost": "100", "currency": "CNY"}
        assert value_holding(holding, detail["latest"])["market_value"] == "120.0"
        pair = evaluation["fund_comparison"]["comparisons"][0]
        assert pair["common_window"]["end"] == "2026-10-23"
        assert pair["common_window"]["observation_count"] == 3
        assert evaluation["fund_comparison"]["basis"] == "current_online"
        assert len(evaluation["online_context"]["nav_by_product"][TARGET]) == 1
        assert workspace.store.get_product(PEER) is None  # Current queries do not register products.
        assert workspace.fund_detail(TARGET)["latest"] is None
    finally:
        workspace.store.close()


def test_failed_peer_is_not_mixed_with_stored_nav(tmp_path, monkeypatch):
    monkeypatch.setattr("rmbfund.service.now_iso", lambda: AT)
    workspace = Workspace(tmp_path / "synthetic", "public")
    try:
        subjects = list(map(current_public_identity, (TARGET, PEER)))
        jobs = [{"job_id": str(index), "product_id": row["id"], "family": "nav",
                 "source_url": f'https://host{index}.example.org/{row["symbol"]}.js',
                 "parser": "eastmoney_nav_js_v1", "config": {}} for index, row in enumerate(subjects)]
        context = workspace.current_online(subjects, source_plan=jobs, clock=lambda: AT,
            fetch=lambda url, *_: FetchResponse(403, b"denied") if "host1" in url else FetchResponse(200, nav_bytes("900001")))
        evaluation = workspace.current_evaluation(TARGET, context, public_subjects=subjects, peer_set=PEERS)
        pair = evaluation["fund_comparison"]["comparisons"][0]
        assert pair["status"] == "missing"
        assert evaluation["fund_comparison"]["source_context"][PEER]["nav_origin"] == "missing"
        assert "历史参考" in evaluation["fund_comparison"]["historical_reference"]["label"]
    finally:
        workspace.store.close()


def test_current_cli_does_not_need_database_registration_or_output_directory(tmp_path, monkeypatch, capsys):
    calls = []
    result = {"schema_version": 1, "kind": "current_online_context", "status": "ready",
              "evaluated_at": AT, "checked_at": AT, "public_subjects": [current_public_identity(TARGET)],
              "source_checks": [], "nav_by_product": {TARGET: []}, "records_by_product": {},
              "messages_by_product": {}, "requests_made": 1}

    def current(subjects, **options):
        calls.append((subjects, options))
        return deepcopy(result)

    monkeypatch.setattr(Workspace, "current_online", staticmethod(current))
    root = tmp_path / "not-created"
    assert main(["--root", str(root), "--mode", "public", "current-online", TARGET]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready" and output["database_writes"] is False
    assert len(calls) == 1 and calls[0][0][0]["id"] == TARGET
    assert not root.exists()

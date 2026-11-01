"""History renders saved heads/results without issuing or settling forecasts."""

from copy import deepcopy

import pytest


from streamlit.testing.v1 import AppTest


SUBJECT = {
    "id": "synthetic",
    "symbol": "SYNTHETIC",
    "name": "合成基金",
    "kind": "open_fund",
    "share_class": "C",
}


def tree_text(at):
    parts = []

    def visit(node):
        if getattr(node, "type", "") in {
            "markdown",
            "caption",
            "info",
            "warning",
            "error",
            "subheader",
            "metric",
        }:
            parts.append(str(node.value))
        if getattr(node, "type", "") == "dataframe":
            parts.append(str(node.value.to_dict("records")))
        for child in getattr(node, "children", {}).values():
            visit(child)

    visit(at.main)
    return "\n".join(parts).replace("\\", "")


def saved_prediction():
    def head(horizon, lag):
        value = 0.01 if lag == 0 else 0.02 if lag == 2 else -0.03 if lag == 3 else 0.04
        return {
            "horizon": horizon,
            "entry_lag": lag,
            "up_probability": 0.6 if lag == 2 else 0.3,
            "expected_change": value,
            "lower": -0.07,
            "upper": 0.11,
            "interval_coverage": 0.8,
            "method": "ridge_prequential_empirical",
            "reasons": [f"原理由H{horizon}L{lag}"],
            "drivers": [
                {
                    "horizon": horizon,
                    "entry_lag": lag,
                    "feature": "log_return_60" if lag == 2 else "log_return_5",
                    "value": 0.02,
                    "contribution_log_return": 0.01 if lag == 2 else -0.01,
                }
            ],
        }

    return {
        "record_id": "original-record",
        "model_id": "nav_ridge_empirical_v2",
        "model_hash": "abc123ef" * 8,
        "generated_at": "2026-10-01T10:00:00+08:00",
        "product_id": "synthetic",
        "entry_assumptions": {"baseline_entry_lag": 2},
        "anchor": {"date": "2026-09-30", "nav": "1.2345", "currency": "CNY"},
        "forecasts": [head(horizon, 0) for horizon in (1, 5, 20)],
        "execution_forecasts": [head(horizon, lag) for horizon in (1, 5, 20) for lag in (2, 3, 4)],
        "drivers": [
            {
                "horizon": 20,
                "entry_lag": 0,
                "feature": "log_return_1",
                "value": 99,
                "contribution_log_return": 99,
            }
        ],
        "reasons": ["不可借用的全局理由"],
    }


def real_history_entry(root):
    from rmbfund.portfolio import Portfolio
    from rmbfund.service import Workspace
    from rmbfund.ui.history import render_history
    from test_history_ui import SUBJECT

    workspace = Workspace(root, "public", read_only=True, initialize=False)
    private = Portfolio(root)
    try:
        render_history(
            "synthetic",
            read=lambda **kwargs: workspace.forecast_history("synthetic", **kwargs),
            detail=lambda record_id: workspace.forecast_history_detail("synthetic", record_id),
            summary=lambda **kwargs: workspace.forecast_summary("synthetic", **kwargs),
            models=workspace.forecast_models("synthetic"),
            assessments=private.assessment_history,
            assessment_detail=private.assessment_detail,
            subject=SUBJECT,
        )
    finally:
        workspace.store.close()


def test_real_repository_snapshots_and_sql_projected_heads_render_read_only(tmp_path, monkeypatch):
    from rmbfund.domain import Product
    from rmbfund.personal_assessment import build_assessment
    from rmbfund.portfolio import Portfolio
    from rmbfund.service import Workspace

    root = tmp_path / "synthetic-history-only"
    workspace = Workspace(root, "public")
    workspace.store.add_product(
        Product("synthetic", "SYNTHETIC", "合成基金", "open_fund", share_class="C")
    )
    prediction = saved_prediction() | {
        "input_hash": "synthetic-input",
        "basis_hash": "synthetic-basis",
    }
    for head in prediction["forecasts"] + prediction["execution_forecasts"]:
        head["status"] = "available"
    record = {
        "id": prediction["record_id"],
        "request_hash": "synthetic-request",
        "series": [],
        **{
            key: prediction[key]
            for key in (
                "product_id",
                "model_id",
                "model_hash",
                "input_hash",
                "basis_hash",
                "generated_at",
                "entry_assumptions",
            )
        },
        "prediction": prediction,
    }
    workspace.store.add_forecast_record(record)
    legacy = deepcopy(record)
    legacy.update(id="legacy-record", request_hash="legacy-request",
                  model_id="nav_ridge_empirical_v1", generated_at="2026-09-29T10:00:00+08:00")
    legacy["prediction"].update(record_id=legacy["id"], model_id=legacy["model_id"],
                                generated_at=legacy["generated_at"])
    legacy["prediction"]["execution_forecasts"] = [
        head for head in legacy["prediction"]["execution_forecasts"] if head["horizon"] == 20
    ]
    workspace.store.add_forecast_record(legacy)
    factors = deepcopy(record)
    factors.update(id="factor-record", request_hash="factor-request",
                   model_id="exposure_factor_ridge_empirical_v1", generated_at="2026-09-30T10:00:00+08:00")
    factors["prediction"].update(record_id=factors["id"], model_id=factors["model_id"],
                                 generated_at=factors["generated_at"])
    for head in factors["prediction"]["forecasts"] + factors["prediction"]["execution_forecasts"]:
        head["drivers"] = [{"horizon": head["horizon"], "entry_lag": head["entry_lag"],
                            "feature": "us10y_yield_percent", "label": "原十年期国债收益率",
                            "unit": "percent", "value": 4.25, "contribution_log_return": -0.01}]
    workspace.store.add_forecast_record(factors)
    public_path = workspace.store.path
    workspace.store.close()
    private = Portfolio(root)
    payload = build_assessment(
        prediction,
        subject=SUBJECT,
        holding={"shares": "100", "cost": "90", "currency": "CNY"},
        horizon=5,
        subscription_fee="0",
        redemption_fee="0.015",
        budget="200",
        investment_analysis={"profile": {"product_id": "synthetic", "as_of": prediction["generated_at"]},
                             "source_gate": "原投资条件资料，不是今天重算", "scenario": None},
    )
    envelope = private.save_assessment(payload)
    public_before, private_before = public_path.read_bytes(), private.path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("read-only history must not issue, settle, or save")

    monkeypatch.setattr(Workspace, "update_forecast", forbidden)
    monkeypatch.setattr(Workspace, "_score_forecasts", forbidden)
    monkeypatch.setattr(Portfolio, "save_assessment", forbidden)
    at = AppTest.from_function(real_history_entry, args=(str(root),), default_timeout=20).run()
    assert not at.exception and not at.error
    assert "原理由H20L2" in tree_text(at)
    assert "124.68 CNY" in tree_text(at) and "204.00 CNY" in tree_text(at)
    assert "保存时的新买金额情景" in tree_text(at)
    assert "1.50%" in tree_text(at)
    assert "当时的投资条件分析" in tree_text(at) and "原投资条件资料，不是今天重算" in tree_text(at)
    assert "不把准确率和误差混成一个总成绩" in tree_text(at)
    model_options = at.selectbox(key="history_model").options
    assert any("净值趋势预测" in option for option in model_options)
    assert any("投资因素预测" in option for option in model_options)
    at.selectbox(key="history_model").set_value("exposure_factor_ridge_empirical_v1").run()
    assert not at.exception and len(at.selectbox(key="history_selected_head").options) == 1
    assert "原十年期国债收益率" in tree_text(at) and "4.25%" in tree_text(at)
    at.selectbox(key="history_model").set_value("nav_ridge_empirical_v2").run()
    assert not at.exception and len(at.selectbox(key="history_selected_head").options) == 2
    at.selectbox(key="history_role").set_value("stress").run()
    assert not at.exception and len(at.selectbox(key="history_selected_head").options) == 4
    at.selectbox(key="history_horizon").set_value(5).run()
    assert not at.exception and "原理由H5L3" in tree_text(at)
    assert len(at.selectbox(key="history_selected_head").options) == 2
    assert private.assessment_detail(envelope["id"]) == envelope
    assert public_path.read_bytes() == public_before and private.path.read_bytes() == private_before


def test_personal_history_precedes_collapsed_quant_and_compares_only_two_saved_payloads():
    old = {"id": "old", "saved_at": "2026-10-01T10:00:00+08:00", "assessment": {
        "schema_version": 3, "holding_status": "held", "horizon": 20, "forecast": {},
        "decision": {"branches": {"held": {"action": "hold", "label": "原保存继续持有", "reasons": ["原保存保留理由"]}}},
        "inputs": {"holding": {}, "subscription_fee": "0", "redemption_fee": "0.015"},
        "research_thesis": {"record_id": "old-research", "payload": {"research_date": "2026-09-30"}},
    }}
    new = deepcopy(old)
    new.update(id="new", saved_at="2026-10-20T10:00:00+08:00")
    new["assessment"]["decision"]["branches"]["held"] = {
        "action": "reduce", "label": "新保存减少投入", "reasons": ["新保存决定性反方"]}
    new["assessment"]["inputs"]["redemption_fee"] = "0.01"
    new["assessment"]["research_thesis"] = {"record_id": "new-research", "payload": {
        "research_date": "2026-10-20", "revision": {"supersedes": "old-research"},
        "change_evaluation": {"summary": "保存研究明确接续：原现金流理由减弱"}}}
    original = deepcopy([new, old])

    def entry(rows):
        from rmbfund.ui.history import render_history

        render_history("synthetic", read=None, subject={"id": "synthetic", "symbol": "SYNTHETIC", "kind": "open_fund"},
            assessments=lambda *args, **kwargs: rows,
            assessment_detail=lambda record_id: next(row for row in rows if row["id"] == record_id))

    at = AppTest.from_function(entry, args=([new, old],), default_timeout=20).run()
    assert not at.exception and at.subheader[0].value == "我保存的判断"
    quant = next(item for item in at.expander if item.label.startswith("量化预测记录"))
    assert quant.proto.expanded is False
    shown = tree_text(at)
    assert "原保存继续持有 → 新保存减少投入" in shown
    assert "新保存决定性反方" in shown and "原现金流理由减弱" in shown
    assert "1.50% → 1.00%" in shown and "不联网、不重算" in shown
    at.selectbox(key="history_assessment_selected").set_value("old").run()
    assert not at.exception and "原保存保留理由" in tree_text(at)
    assert "新保存决定性反方" not in tree_text(at) and "没有更早的保存判断" in tree_text(at)
    assert [new, old] == original

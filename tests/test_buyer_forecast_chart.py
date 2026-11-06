"""Actual NAV + one exact-plan simple-return endpoint; no IO or model fitting."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
from streamlit.testing.v1 import AppTest

from rmbfund.buyer_view import build_forecast_chart as pure_chart
from rmbfund.ui.charts import build_forecast_chart, forecast_figure, forecast_table_rows


ARTIFACT_SENTINEL = "INTERNAL_MODEL_ID_MUST_NOT_RENDER"


def assessment():
    return {
        "schema_version": 7, "subject": {"name": "合成展示基金C", "symbol": "SYNTHETIC", "share_class": "C"},
        "decision_brief": {"data_date": "2026-10-26"},
        "forecast": {"record_id": "saved-synthetic", "model_id": ARTIFACT_SENTINEL,
                     "model_hash": "INTERNAL_HASH", "generated_at": "2026-10-29T10:00:00+08:00",
                     "anchor": {"date": "2026-10-26", "nav": "400", "currency": "CNY"},
                     "execution_forecasts": [{"horizon": 20, "entry_lag": 2, "expected_change": 5}]},
        "trade_scenario": {
            "plan": {"horizon": 9, "entry_lag": 3, "application_date": "2026-10-29",
                     "sale_date": "2026-11-11", "deadline": "2026-11-12", "calendar_days": 14},
            "decision": {"primary_head": {
                "horizon": 9, "entry_lag": 3, "expected_change": 0.01234,
                "lower": -0.045, "upper": 0.08, "interval_coverage": 0.8,
                "samples": [-0.99, 0.99], "drivers": [{"feature": "SECRET_FEATURE"}],
            }, "model_identity": {"model_id": ARTIFACT_SENTINEL, "model_hash": "INTERNAL_HASH",
                                   "horizon": 9, "entry_lag": 3, "role": "execution"}},
            "rows": [{"action": "buy", "initial_value": 10000, "expected_value": 99999999,
                      "expected_change": 50}],
        },
        "personal_conversion": {"amounts": {"current": {"market_value": "PERSONAL_AMOUNT"}}},
    }


def series():
    return [{"date": day, "nav": value, "provenance": {"currency": "CNY", "nav_type": "unit_nav",
             "source": "INTERNAL_SOURCE", "source_url": "INTERNAL_URL", "fact_id": "INTERNAL_FACT"}}
            for day, value in (("2026-10-23", "350.1234"), ("2026-10-26", "400"))]


def test_exact_head_is_separate_simple_return_not_nav_anchor_or_account_money():
    source, actual = assessment(), series()
    before = deepcopy((source, actual))
    snapshot = build_forecast_chart(source, series=actual)
    assert build_forecast_chart is pure_chart
    assert snapshot["status"] == "ready"
    assert snapshot["history"]["points"] == [
        {"date": "2026-10-23", "value": "350.1234", "currency": "CNY", "basis": "unit_nav"},
        {"date": "2026-10-26", "value": "400", "currency": "CNY", "basis": "unit_nav"},
    ]
    assert snapshot["history"]["date_basis"] == "nav_date"
    assert snapshot["forecast"]["point"] == {
        "date": "2026-11-11", "expected_change": 0.01234, "lower": -0.045, "upper": 0.08,
    }
    assert snapshot["forecast"]["unit"] == "simple_return"
    assert snapshot["forecast"]["date_basis"] == "conditional_plan_sale_date"
    assert snapshot["forecast"]["deadline"] == "2026-11-12"  # Not substituted for sale_date.
    assert snapshot["identity"]["model_id"] == ARTIFACT_SENTINEL
    dumped = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    assert all(term not in dumped for term in ("samples", "SECRET_FEATURE", "INTERNAL_SOURCE", "PERSONAL_AMOUNT"))
    assert (source, actual) == before
    snapshot["history"]["points"][0]["value"] = "1"
    snapshot["identity"]["anchor"]["nav"] = "0"
    assert (source, actual) == before


@pytest.mark.parametrize("horizon,lag", [(20, 3), (9, 2), (True, 3), (9, False)])
def test_no_other_horizon_or_entry_origin_is_substituted(horizon, lag):
    source = assessment()
    source["trade_scenario"]["decision"]["primary_head"].update(horizon=horizon, entry_lag=lag)
    snapshot = build_forecast_chart(source, series=series())
    assert snapshot["forecast"]["status"] == "missing_exact_head"
    assert snapshot["forecast"]["point"] is None
    assert snapshot["history"]["points"]


@pytest.mark.parametrize("change", [None, float("nan"), float("inf"), True, -1.01])
def test_bad_mean_does_not_draw_a_forecast(change):
    source = assessment()
    source["trade_scenario"]["decision"]["primary_head"]["expected_change"] = change
    snapshot = build_forecast_chart(source, series=series())
    assert snapshot["forecast"]["point"] is None
    assert snapshot["forecast"]["status"] == "invalid_head"
    json.dumps(snapshot, allow_nan=False)


def test_h0_and_missing_sale_date_cannot_be_drawn_and_deadline_is_not_a_sale_day():
    source = assessment()
    source["trade_scenario"]["plan"]["horizon"] = 0
    assert build_forecast_chart(source, series=series())["forecast"]["point"] is None
    source = assessment()
    source["trade_scenario"]["plan"].pop("sale_date")
    assert build_forecast_chart(source, series=series())["forecast"]["point"] is None
    source = assessment()
    source["trade_scenario"]["plan"]["sale_date"] = "2026-10-29"
    assert build_forecast_chart(source, series=series())["forecast"]["point"] is None
    source = assessment()
    source["trade_scenario"]["decision"]["primary_head"] = None
    assert build_forecast_chart(source, series=series())["forecast"]["point"] is None


@pytest.mark.parametrize("patch", [{"lower": None}, {"upper": 0.01, "lower": 0.03},
                                   {"lower": -2}, {"interval_coverage": 0.95}])
def test_missing_or_incompatible_interval_keeps_only_saved_mean(patch):
    source = assessment()
    source["trade_scenario"]["decision"]["primary_head"].update(patch)
    snapshot = build_forecast_chart(source, series=series())
    assert snapshot["forecast"]["status"] == "partial"
    assert snapshot["forecast"]["point"]["expected_change"] == 0.01234
    assert snapshot["forecast"]["point"]["lower"] is None
    assert snapshot["forecast"]["point"]["upper"] is None


def test_actual_nav_value_contract_sorted_without_imputation_or_anchor_as_fake_history():
    rows = [{"date": "2026-10-26", "value": "4", "currency": "USD", "basis": "raw"},
            {"date": "2026-10-01", "value": "3", "currency": "USD", "basis": "raw"}]
    snapshot = build_forecast_chart(assessment(), series=rows)
    assert [row["date"] for row in snapshot["history"]["points"]] == ["2026-10-01", "2026-10-26"]
    assert snapshot["history"]["currency"] == "USD"
    assert build_forecast_chart(assessment(), series=[])["history"]["points"] == []
    assert build_forecast_chart({}, series=rows)["status"] == "partial"
    assert build_forecast_chart({}, series=[])["status"] == "unavailable"


def test_missing_prediction_preserves_real_history_and_its_actual_date_without_fake_endpoint():
    source = {"trade_scenario": {"purchase": {"latest_date": "2026-10-26"},
                               "decision": {"primary_head": None}}}
    snapshot = build_forecast_chart(source, series=series())
    assert snapshot["forecast"]["point"] is None
    assert len(snapshot["history"]["points"]) == 2
    assert snapshot["data_date"] == "2026-10-26"
    # Even without the scenario's date projection, passed actual history retains its date.
    source["trade_scenario"].pop("purchase")
    snapshot = build_forecast_chart(source, series=series())
    assert snapshot["data_date"] == "2026-10-26"
    figure = forecast_figure(snapshot)
    assert len(figure.data) == 1 and figure.data[0].yaxis == "y"


@pytest.mark.parametrize("rows", [
    [{"date": "2026-10-26", "nav": "1"}, {"date": "2026-10-26", "nav": "2"}],
    [{"date": "2026-10-26", "value": "1", "nav": "2"}],
    [{"date": "2026-10-26", "value": "0"}],
    [{"date": "2026-10-26", "value": "nan"}],
    [{"date": "2026-10-26", "value": True}],
    [{"date": "2026-10-26", "value": "1", "currency": "USD"},
     {"date": "2026-10-27", "value": "2", "currency": "CNY"}],
])
def test_invalid_history_does_not_silently_choose_versions_or_units(rows):
    snapshot = build_forecast_chart(assessment(), series=rows)
    assert snapshot["history"]["status"] == "invalid"
    assert snapshot["history"]["points"] == []
    assert snapshot["forecast"]["point"] is not None


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_figure_has_separate_panels_one_endpoint_zero_reference_and_no_future_path(theme):
    snapshot = build_forecast_chart(assessment(), series=series())
    figure = forecast_figure(snapshot, theme=theme)
    traces = list(figure.data)
    history = [trace for trace in traces if trace.yaxis == "y"]
    future = [trace for trace in traces if trace.yaxis == "y2"]
    assert len(history) == 1 and list(history[0].y) == [350.1234, 400.0]
    assert len(future) == 2
    assert all(set(trace.x) == {"2026-11-11"} for trace in future)
    assert any(trace.mode == "markers" and list(trace.y) == [0.01234] for trace in future)
    assert any(list(trace.y) == [-0.045, 0.08] for trace in future)
    assert all(trace.type == "scatter" for trace in traces)
    assert figure.layout.yaxis2.tickformat.endswith("%")
    assert figure.layout.xaxis2.type == "category"
    assert figure.layout.showlegend
    assert any(shape.y0 == shape.y1 == 0 and shape.yref == "y2" for shape in figure.layout.shapes)
    for secret in (ARTIFACT_SENTINEL, "INTERNAL_HASH", "INTERNAL_URL", "PERSONAL_AMOUNT", "samples"):
        assert secret not in figure.to_json()


def test_mean_outside_quantile_range_keeps_independent_interval_not_negative_error_bars():
    source = assessment()
    source["trade_scenario"]["decision"]["primary_head"]["expected_change"] = 0.2
    snapshot = build_forecast_chart(source, series=series())
    figure = forecast_figure(snapshot)
    assert any(list(trace.y) == [-0.045, 0.08] for trace in figure.data)
    assert any(list(trace.y) == [0.2] for trace in figure.data)
    assert all(not trace.error_y.visible for trace in figure.data)
    assert figure.layout.yaxis2.range[1] > 0.2


def test_accessible_table_whitelists_public_data_and_preserves_fraction_units():
    snapshot = build_forecast_chart(assessment(), series=series())
    snapshot["forecast"]["point"]["model_id"] = ARTIFACT_SENTINEL
    tables = forecast_table_rows(snapshot)
    assert tables["forecast"][0]["预期涨跌"] == "+1.23%"
    assert tables["forecast"][0]["计划期末（条件日期）"] == "2026-11-11"
    assert tables["history"][0]["单位净值"] == "350.1234"
    text = json.dumps(tables, ensure_ascii=False)
    assert ARTIFACT_SENTINEL not in text and "INTERNAL_HASH" not in text
    assert "expected_value" not in text and "10000" not in text


def test_builder_does_not_need_files_or_fit(monkeypatch):
    source, actual = assessment(), series()
    def forbidden(*args, **kwargs):
        pytest.fail("pure chart projection must not perform IO")
    monkeypatch.setattr("builtins.open", forbidden)
    assert build_forecast_chart(source, series=actual)["forecast"]["point"]


def test_renderers_show_only_final_copy_and_public_chart_data_not_internal_snapshot():
    def app():
        from rmbfund.buyer_view import build_buyer_view, build_forecast_chart
        from rmbfund.ui.buyer_view import render_buyer_view, render_forecast_chart
        source = {
            "decision_brief": {"schema_version": 3, "data_date": "2026-10-26", "cases": {
                "flat": {"preferred_action": "wait", "label": "先保留现金",
                         "advice_reason": "当前价格尚未补足新增风险。", "next_step": "等价格或盈利变化再核对。",
                         "execution_status": {"status": "not_required", "reasons": ["INTERNAL_REASON"]},
                         "decision_basis": {"model_id": "INTERNAL_MODEL"}, "expected_value": 99999999},
            }},
            "forecast": {"model_id": "INTERNAL_MODEL", "anchor": {"date": "2026-10-26", "currency": "CNY"}},
            "trade_scenario": {"plan": {"horizon": 9, "entry_lag": 3,
                "application_date": "2026-10-29", "sale_date": "2026-11-11"},
                "decision": {"primary_head": {"horizon": 9, "entry_lag": 3,
                    "expected_change": .01, "lower": -.03, "upper": .06}}},
        }
        render_buyer_view(build_buyer_view(source))
        render_forecast_chart(build_forecast_chart(source, series=[
            {"date": "2026-10-26", "value": "1.2345", "currency": "CNY"},
        ]))
    at = AppTest.from_function(app).run()
    assert not at.exception
    public = " ".join(str(item.value) for group in (at.markdown, at.caption, at.subheader) for item in group)
    assert "先保留现金" in public
    assert "当前价格尚未补足新增风险。" in public
    assert "参考净值截至 2026" in public
    assert "历史模型参考，尚未验证择时优势" in public
    assert all(secret not in public for secret in ("INTERNAL_MODEL", "INTERNAL_REASON", "99999999", "10000", "H=", "L="))
    assert not at.json and not at.metric
    htmls = [node.proto.body for node in at.get("html")]
    assert any("<table>" in body and 'tabindex="0"' in body for body in htmls)
    assert all("INTERNAL_" not in body for body in htmls)


def _curve_prediction(horizons=(1, 2, 3), lag=3):
    return {
        "generated_at": "2026-10-29T10:00:00+08:00",
        "requested_horizons": list(horizons),
        "calculation": {"include_curve": True, "timestamp_kind": "calculated"},
        "execution_forecasts": [
            {"horizon": horizon, "entry_lag": lag, "status": "available",
             "expected_change": horizon / 100, "lower": -0.01,
             "upper": 0.03, "interval_coverage": 0.8,
             "target_definition_id": "raw_unit_nav_simple_return_v1",
             "samples": ["SECRET_SAMPLE"], "drivers": [{"feature": "SECRET_DRIVER"}]}
            for horizon in horizons
        ],
    }


def test_local_curve_keeps_exact_heads_and_conditional_dates_without_nav_path():
    source = assessment()
    plan = source["trade_scenario"]["plan"]
    plan.update(horizon=3, entry_lag=3, application_date="2026-10-29", sale_date="2026-11-02",
                assumed_open_dates=["2026-10-30", "2026-10-31", "2026-11-02"])
    primary = source["trade_scenario"]["decision"]["primary_head"]
    primary.update(horizon=3, entry_lag=3)
    snapshot = build_forecast_chart(source, series=series(), prediction=_curve_prediction())
    assert snapshot["schema_version"] == 2
    points = snapshot["forecast"]["points"]
    assert [(row["horizon"], row["entry_lag"], row["date"]) for row in points] == [
        (1, 3, "2026-10-30"), (2, 3, "2026-10-31"), (3, 3, "2026-11-02")
    ]
    assert [row["expected_change"] for row in points] == [0.01, 0.02, 0.03]
    assert snapshot["forecast"]["point"]["date"] == "2026-11-02"
    dumped = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    assert all(secret not in dumped for secret in ("SECRET_SAMPLE", "SECRET_DRIVER"))
    figure = forecast_figure(snapshot)
    future = [trace for trace in figure.data if trace.yaxis == "y2"]
    assert any(trace.mode == "markers" and list(trace.x) == ["H=1", "H=2", "H=3"] for trace in future)
    assert all(not (trace.mode == "lines+markers" and set(trace.x) & {"2026-10-29", "2026-10-26"})
               for trace in future)
    assert figure.layout.xaxis2.type == "category"


def test_curve_missing_exact_head_is_a_gap_and_not_another_lag_or_horizon():
    source = assessment()
    source["trade_scenario"]["plan"].update(horizon=3, entry_lag=3,
        application_date="2026-10-29", sale_date="2026-11-02",
        assumed_open_dates=["2026-10-30", "2026-10-31", "2026-11-02"])
    source["trade_scenario"]["decision"]["primary_head"].update(horizon=3, entry_lag=3)
    prediction = _curve_prediction((1, 3), lag=3)
    prediction["requested_horizons"] = [1, 2, 3]
    prediction["execution_forecasts"].append({
        "horizon": 2, "entry_lag": 4, "status": "available", "expected_change": 0.99,
        "lower": 0.9, "upper": 1.0, "interval_coverage": 0.8,
        "target_definition_id": "raw_unit_nav_simple_return_v1",
    })
    snapshot = build_forecast_chart(source, series=series(), prediction=prediction)
    rows = snapshot["forecast"]["points"]
    assert rows[1]["status"] == "unavailable"
    assert rows[1]["expected_change"] is None
    assert rows[1]["reason"]
    assert snapshot["forecast"]["points_status"] == "partial"
    assert forecast_table_rows(snapshot)["forecast"][1]["状态"] == "暂不可用"


@pytest.mark.parametrize("dates", [
    ["2026-10-30", "not-a-date", "2026-11-02"],
    ["2026-10-30", "2026-10-30", "2026-11-02"],
    ["2026-10-30", "2026-11-03", "2026-11-02"],
    ["2026-10-30"],
    [],
])
def test_curve_date_slots_keep_h_positions_and_never_fallback_to_sale_date(dates):
    source = assessment()
    source["trade_scenario"]["plan"].update(
        horizon=3, entry_lag=3, application_date="2026-10-29", sale_date="2026-11-02",
        assumed_open_dates=dates,
    )
    source["trade_scenario"]["decision"]["primary_head"].update(horizon=3, entry_lag=3)
    snapshot = build_forecast_chart(source, series=series(), prediction=_curve_prediction())
    rows = snapshot["forecast"]["points"]
    assert [row["horizon"] for row in rows] == [1, 2, 3]
    expected_dates = []
    for index in range(3):
        raw = dates[index] if index < len(dates) else None
        expected_dates.append(
            raw if raw in {"2026-10-30", "2026-10-31", "2026-11-02"}
            and dates.count(raw) == 1 else None
        )
    assert [row["date"] for row in rows] == expected_dates
    invalid = [row for row in rows if row["status"] == "unavailable"]
    assert invalid
    assert all(row["expected_change"] is None for row in invalid)
    assert all(row["reason"] for row in invalid)
    assert snapshot["forecast"]["points_unavailable_horizons"] == [row["horizon"] for row in invalid]
    assert all(f"H={row['horizon']}" in snapshot["forecast"]["points_missing_fields"]
               for row in invalid)


def test_curve_axis_keeps_unavailable_h_category_slots_and_public_hover_fields():
    source = assessment()
    source["trade_scenario"]["plan"].update(
        horizon=3, entry_lag=3, application_date="2026-10-29", sale_date="2026-11-02",
        assumed_open_dates=["2026-10-30", "bad-date", "2026-11-02"],
    )
    source["trade_scenario"]["decision"]["primary_head"].update(horizon=3, entry_lag=3)
    snapshot = build_forecast_chart(source, series=series(), prediction=_curve_prediction())
    figure = forecast_figure(snapshot)
    assert list(figure.layout.xaxis2.categoryarray) == ["H=1", "H=2", "H=3"]
    assert list(figure.layout.xaxis2.tickvals) == ["H=1", "H=2", "H=3"]
    assert "日期未映射" in " ".join(figure.layout.xaxis2.ticktext)
    marker = next(trace for trace in figure.data if trace.name == "各目标点预期涨跌")
    assert list(marker.x) == ["H=1", "H=3"]
    assert "预期涨跌" in marker.hovertemplate and "状态" in marker.hovertemplate
    interval = next(trace for trace in figure.data if trace.name == "各目标点中间80%范围")
    assert "范围端点" in interval.hovertemplate and "状态" in interval.hovertemplate

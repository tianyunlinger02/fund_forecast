"""Everyday portfolio UI contracts, using isolated synthetic holdings only."""

from __future__ import annotations


from copy import deepcopy


from pathlib import Path


import pytest


from streamlit.testing.v1 import AppTest


from rmbfund.domain import now_iso


from rmbfund.portfolio import Portfolio


AT = "2026-10-23T08:00:00+08:00"


PRODUCT = {
    "id": "open_fund:018147:C",
    "symbol": "018147",
    "name": "合成测试基金C",
    "kind": "open_fund",
    "share_class": "C",
    "exchange": "",
    "is_demo": False,
}


def prediction() -> dict:
    def head(h, lag=0):
        change = {1: -0.03, 5: 0.005, 20: 0.025}[h] if lag else 0.01
        up = {1: 0.35, 5: 0.55, 20: 0.62}[h] if lag else 0.80
        return {
            "horizon": h,
            "entry_lag": lag,
            "up_probability": up,
            "down_probability": 1 - up - 0.01,
            "flat_probability": 0.01,
            "expected_change": change,
            "lower": change - 0.05,
            "upper": change + 0.06,
            "method": "ridge_prequential_empirical",
            "sample_count": 50,
            "samples": [-0.05, 0.02, 0.08],
            "interval_coverage": 0.8,
        }

    return {
        "model_id": "nav_ridge_empirical_v2",
        "schema_version": 2,
        "model_hash": "synthetic-model-hash",
        "record_id": "synthetic-current-record",
        "product_id": PRODUCT["id"],
        "generated_at": AT,
        "anchor": {"date": "2026-10-21", "nav": "2.161", "currency": "CNY"},
        "forecasts": [head(h) for h in (1, 5, 20)],
        "execution_forecasts": [head(h, lag) for h in (1, 5, 20) for lag in (2, 3, 4)],
        "entry_assumptions": {"baseline_entry_lag": 2},
        "validation": {"status": "not_better"},
        "drivers": [
            {
                "horizon": h,
                "entry_lag": 2,
                "feature": f"log_return_{h}",
                "value": h / 1000,
                "contribution_log_return": 0.01,
            }
            for h in (1, 5, 20)
        ],
        "features": [],
    }


def generated_thesis() -> dict:
    return {"status": "available", "record_id": "a" * 64, "needed_review": [], "payload": {
        "schema_version": 2, "product": {key: PRODUCT[key] for key in ("id", "symbol", "name", "kind", "share_class", "exchange")},
        "note_id": "synthetic-ui-thesis",
        "research_date": "2026-10-23", "posture": "maintain",
        "horizon": {"start": "2026-10-23", "end": "2027-10-23"},
        "origin": {"provider": "synthetic", "model": "synthetic-model", "run_id": "test-run",
                   "generated_at": AT, "human_reviewed": False},
        "facts": [{"id": "f1", "claim": "合成经营改善事实", "status": "factual", "effective_date": "2026-10-21",
                   "citation_ids": ["c1"], "channels": ["earnings"], "issuer_ids": []}],
        "citations": [{"id": "c1", "kind": "online_evidence", "claim": "合成材料支持", "title": "合成公告",
            "source_url": "https://example.test/disclosure", "published_at": "2026-10-21", "observed_at": AT,
            "locator": "第3页 / 表2", "quote": "合成原文：经营收入改善。", "evidence_id": "synthetic-document",
            "product_id": PRODUCT["id"], "publication_precision": "date", "available_at": AT,
            "content_status": "online_excerpt", "validation": "quote_matches_supplied_section", "record_ids": []}],
        "assessment": {"summary": "合成新研究摘要", "supporting": [{"reason": "经营改善支持保留", "fact_ids": ["f1"]}],
            "counterevidence": [{"reason": "价格尚未证明划算", "fact_ids": ["f1"]}],
            "priced_in": {"status": "unknown", "reason": "不能把经营改善当作买价便宜"},
            "missing": [], "falsifiers": ["现金流恶化"], "change_conditions": ["价格与盈利重新核对"],
            "review_date": "2027-01-23"},
        "action_cases": {"new_buy": {"action": "wait", "reason": "新买需要更有吸引力的价格", "fact_ids": ["f1"]},
                         "held": {"action": "hold", "reason": "保留理由仍成立但不必加钱", "fact_ids": ["f1"]},
                         "addition": {"action": "wait", "reason": "追加没有新的价格优势", "fact_ids": ["f1"]}},
        "fund_quality": {"conclusion": "基金长期任务可理解", "reason": "披露与任务相符", "fact_ids": ["f1"]},
        "entry_case": {"conclusion": "当前入场不优先", "reason": "盈利改善与买得便宜是两回事", "fact_ids": ["f1"]},
        "drivers": [{"channel": "earnings", "mechanism": "收入先影响现金流，再影响估值", "fact_ids": ["f1"],
                     "fund_transmission": "通过原披露篮子影响基金，未覆盖部分未知"}],
        "alternatives": [{"product_id": "open_fund:539002:A", "reason": "同底层但换基有费用", "fact_ids": ["f1"]}],
        "key_unknowns": [{"question": "现价是否已反映改善", "critical": False, "affects": ["new_buy", "addition"]}],
        "change_evaluation": {"summary": "合成经营理由增强，但入场理由未增强", "fact_ids": ["f1"]},
        "revision": {"id": "test-v2", "supersedes": None},
    }}


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.mode = "public"
        self.updates = []
        self.update_refreshes = []
        self.resolutions = []
        self.history_reads = 0
        self.forecast_reads = 0
        self.summary_reads = []
        self.online_calls = []
        self.online_result = None
        self.research_calls = []
        self.calculations = []
        self.fail_research = False
        self.comparison = {"status": "missing", "comparisons": []}
        self.prediction = prediction()
        self.fail_update = False
        self.product_rows = [deepcopy(PRODUCT)]
        self.detail = {
            "status": "ready",
            "product": deepcopy(PRODUCT),
            "latest": {
                "date": "2026-10-21",
                "value": "2.161",
                "currency": "CNY",
                "observed_at": AT,
                "source": "synthetic",
            },
            "previous_change": 0.01,
            "last_checked_at": AT,
            "reasons": [],
            "history": [
                {
                    "date": day,
                    "value": value,
                    "currency": "CNY",
                    "basis": "synthetic",
                    "source": "synthetic",
                    "observed_at": AT,
                }
                for day, value in (("2026-10-19", "2.1"), ("2026-10-21", "2.161"))
            ],
            "portfolio": None,
        }

    def products(self):
        return deepcopy(self.product_rows)

    def resolve_fund(self, subject, *, register=False):
        if register:
            self.resolutions.append(deepcopy(subject))
        return next(
            (
                deepcopy(row)
                for row in self.product_rows
                if row["symbol"] == subject.get("symbol")
                and (not subject.get("share_class") or row["share_class"] == subject["share_class"])
                and (not subject.get("exchange") or row.get("exchange", "") == subject["exchange"])
            ),
            None,
        )

    def fund_detail(self, product_id, as_of=None):
        product = next((row for row in self.product_rows if row["id"] == product_id), None)
        if product is None:
            return {"status": "missing", "product": None, "history": [], "latest": None}
        return deepcopy(self.detail) | {"product": deepcopy(product)}

    def current_subjects(self, subject, *, additional=()):
        peers = [row["product"] for row in self.comparison.get("funds", [])]
        subjects = {row["id"]: deepcopy(row) for row in [subject, *additional, *peers]}
        return {"public_subjects": list(subjects.values()), "source_plan": [], "peer_set": None, "blocked_urls": []}

    def current_online(self, public_subjects, **kwargs):
        self.online_calls.append((deepcopy(public_subjects), deepcopy(kwargs)))
        if self.online_result is not None:
            return deepcopy(self.online_result)
        checked = now_iso()
        return {"schema_version": 1, "kind": "current_online_context", "status": "unavailable",
                "checked_at": checked, "evaluated_at": checked, "public_subjects": deepcopy(public_subjects),
                "source_checks": [], "nav_by_product": {}, "records_by_product": {},
                "messages_by_product": {}, "warnings": ["合成在线查询：未配置新资料，旧参考保持原日期。"]}

    def current_evaluation(self, product_id, context, *, public_subjects=(), peer_set=None):
        from rmbfund.current_online import project_current_detail, snapshot_online_context

        details = {row["id"]: project_current_detail(self.fund_detail(row["id"]), context, row["id"])
                   for row in public_subjects}
        return {"details": details, "online_context": snapshot_online_context(context),
                "evaluated_at": context["evaluated_at"],
                "fund_comparison": deepcopy(self.comparison)}

    def forecast(self, product_id, as_of=None, *, model_id="nav_ridge_empirical_v2"):
        self.forecast_reads += 1
        product = next(row for row in self.product_rows if row["id"] == product_id)
        saved = deepcopy(self.prediction) | {"product_id": product_id}
        return {
            "status": "ready",
            "product": deepcopy(product),
            "prediction": saved,
            "evaluated_at": now_iso(),
            "evidence": {},
            "reasons": [],
        }

    def timing_inputs(self, product_id, *, context=None, model_id=None, calendar_days=None, decision_at=None):
        from rmbfund.trade_calendar import build_trade_plan

        rows = getattr(self, "series", None)
        if rows is None:
            rows = [{**row, "nav": row["value"]} for row in self.detail.get("history", [])]
        rows = deepcopy(rows)
        result = {"status": "ready" if rows else "insufficient_history", "series": rows,
                  "cache_key": getattr(self, "input_version", "synthetic-input-v1"),
                  "entry_assumptions": {"baseline_entry_lag": 2}}
        if calendar_days is not None:
            plan = build_trade_plan(decision_at=decision_at or now_iso(), calendar_days=calendar_days,
                                    anchor_date=rows[-1]["date"] if rows else None)
            result["trade_plan"] = plan
            result["entry_assumptions"] = {"baseline_entry_lag": plan["entry_lag"]}
            result["cache_key"] += f":{plan['horizon']}:{plan['entry_lag']}"
            if plan["status"] != "ready":
                result["status"] = "infeasible_window"
        return result

    def calculate_timing(self, product_id, *, context=None, model_id=None, calendar_days=None,
                         include_history=False, decision_at=None):
        result = self.timing_inputs(product_id, context=context, model_id=model_id,
                                   calendar_days=calendar_days, decision_at=decision_at)
        self.calculations.append((product_id, model_id, include_history, calendar_days))
        current = deepcopy(self.prediction or {})
        plan = result.get("trade_plan") or {}
        horizon, lag = plan.get("horizon", 20), plan.get("entry_lag", 2)
        if not current or not horizon or lag is None or result["status"] != "ready":
            return {**result, "prediction": None}
        for field in ("forecasts", "execution_forecasts"):
            template = next(head for head in current[field] if head["horizon"] == 20)
            current[field] = [{**deepcopy(template), "horizon": horizon,
                              "entry_lag": 0 if field == "forecasts" else selected}
                             for selected in ([0] if field == "forecasts" else [lag, lag + 1, lag + 2])]
        current["drivers"] = [{"horizon": horizon, "entry_lag": lag,
                               "feature": "log_return_20", "value": 0.02,
                               "contribution_log_return": 0.01}]
        current["generated_at"] = now_iso()
        current["entry_assumptions"] = result["entry_assumptions"]
        current["local_contract"] = "calendar_trade_nav_observations_v1"
        current.pop("record_id", None)
        result.update(prediction=current, selected_model_id=model_id,
                      calculation={"kind": "local_hypothetical", "actual_issue": False, "persisted": False})
        if include_history:
            rows = result["series"]
            result["historical_forecasts"] = [{"horizon": horizon, "entry_lag": lag,
                "origin_index": i, "origin_date": row["date"], "knowledge_cutoff_index": i,
                "knowledge_cutoff_date": row["date"], "assumed_known_index": i + 1,
                "assumed_known_date": rows[i + 1]["date"] if i + 1 < len(rows) else None,
                "prediction": deepcopy(current["execution_forecasts"][0])}
                for i, row in enumerate(rows)]
        return result

    def forecast_models(self, product_id=None):
        return [
            {
                "id": "nav_ridge_empirical_v2",
                "name": "净值趋势预测",
                "available": True,
                "description": "只使用历史净值",
                "horizons": [1, 5, 20],
            }
        ]

    def forecast_summary(self, product_id, **kwargs):
        self.summary_reads.append((product_id, kwargs))
        return {
            "status": "ready",
            "horizon": kwargs.get("horizon", 20),
            "role": kwargs.get("role", "primary"),
            "counts": {
                "recorded_forward": 0,
                "recorded_nowcast": 0,
                "unrecorded": 1,
                "forward_windows": 0,
                "unique_windows": 0,
                "nonoverlapping_windows": 0,
                "paired": 0,
            },
            "forward_metrics": {"count": 0},
            "paired": {"count": 0},
        }

    def update_forecast(self, subject, *, refresh=True, model_id="nav_ridge_empirical_v2"):
        self.updates.append(subject)
        self.update_refreshes.append(refresh)
        if self.fail_update:
            raise RuntimeError("PRIVATE-cost-secret /private/path")
        return self.forecast(subject) | {
            "refresh": {"status": "partial"} if refresh else {},
            "reused": True,
        }

    def update_research(self, subject, *, settings, api_key=None, context=None,
                        public_subjects=(), peer_set=None, on_progress=None, **kwargs):
        self.research_calls.append({"subject": deepcopy(subject), "settings": deepcopy(settings),
                                    "api_key": api_key, "context": deepcopy(context),
                                    "public_subjects": deepcopy(public_subjects)})
        for stage in ("取材", "分析", "保存"):
            if on_progress:
                on_progress(stage)
        if self.fail_research:
            return {"status": "partial" if self.fail_research == "partial" else "failed",
                    "research_run": {"research": None}, "research_thesis": {"status": "available"},
                    "evaluation": self.current_evaluation(subject["id"], context, public_subjects=public_subjects)}
        identifier = subject["id"]
        generated = AT.replace("T08:00", "T08:05")
        thesis = generated_thesis()
        thesis["record_id"] = "synthetic-research-v2"
        thesis["payload"]["product"] = {key: subject[key] for key in ("id", "symbol", "name", "kind", "share_class", "exchange")}
        thesis["payload"]["origin"]["generated_at"] = generated
        evaluation = self.current_evaluation(identifier, context, public_subjects=public_subjects)
        evaluation["details"][identifier]["research_thesis"] = thesis
        evaluation["evaluated_at"] = generated
        return {"status": "generated", "research": thesis, "research_thesis": thesis,
                "research_run": {"status": "generated", "research": thesis},
                "receipt": {"record_id": "synthetic-receipt"},
                "evaluation": evaluation, "context": context or {}}

    def forecast_history(self, product_id, **kwargs):
        self.history_reads += 1
        horizon = kwargs.get("horizon", 20)
        return [
            {
                "record_id": "original-record",
                "product_id": product_id,
                "generated_at": AT,
                "model_id": "nav_ridge_empirical_v1",
                "model_name": "净值趋势预测",
                "anchor": prediction()["anchor"],
                "horizon": horizon,
                "head": prediction()["forecasts"][(1, 5, 20).index(horizon)],
                "outcome": None,
            }
        ]

    def forecast_history_detail(self, product_id, record_id):
        return {
            "status": "ready",
            "record_id": record_id,
            "generated_at": AT,
            "model_name": "净值趋势预测",
            "prediction": prediction(),
            "outcomes": [],
        }


def entry(root, workspace=None):
    from rmbfund.ui.app import main

    main(root=root, mode="public", workspace=workspace)


def run(workspace: FakeWorkspace, *, overview: bool = False) -> AppTest:
    at = AppTest.from_function(
        entry, args=(str(workspace.root), workspace), default_timeout=20
    ).run()
    buttons = [button for button in at.button if button.label == "查看当前判断"]
    if buttons and not overview:
        buttons[0].click().run()
    return at


def text(at) -> str:
    return "\n".join(
        str(element.value)
        for kind in ("markdown", "caption", "info", "warning", "error", "subheader", "title")
        for element in at.get(kind)
    ).replace("\\", "")


def click(at, label):
    return next(button for button in at.button if button.label == label).click().run()


def displayed(at):
    return at.session_state["fund_displayed_assessments"][PRODUCT["id"]]


def choose_branch(at, branch):
    at.get("button_group")[0].set_value(branch).run()
    return at


def open_tools(at):
    at.session_state["fund_navigation"] = "走势与资料"
    at.run()
    return at


def open_decision(at):
    at.session_state["fund_navigation"] = "当前判断"
    at.run()
    return at


def seed(workspace, *, shares="100", cost="200"):
    return Portfolio(workspace.root).save_fund(PRODUCT, shares=shares, cost=cost)


def test_initial_portfolio_is_empty_even_with_research_sample_in_public_catalog(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    at = run(workspace)
    assert not at.exception
    assert "目前还没有个人持仓记录" in text(at)
    assert not at.tabs and not workspace.forecast_reads and not workspace.updates
    assert not (tmp_path / "personal-public.sqlite3").exists()
    assert not (tmp_path / "public.sqlite3").exists()


def test_add_known_code_reuses_actual_class_and_saves_without_network(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    at = run(workspace)
    click(at, "＋ 添加基金")
    at.text_input(key="fund_add_code").input("018147").run()
    at.text_input(key="fund_add_shares").input("123.4567")
    at.text_input(key="fund_add_cost").input("200.01")
    click(at, "保存到我的基金")
    assert not at.exception
    funds = Portfolio(tmp_path).list_funds()
    assert len(funds) == 1 and funds[0]["product_id"] == PRODUCT["id"]
    assert funds[0]["subject"]["share_class"] == "C" and funds[0]["shares"] == "123.4567"
    assert funds[0]["cost"] == "200.01" and not workspace.updates
    assert not workspace.resolutions


def test_edit_persists_explicitly_and_does_not_change_market_or_update(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    fund = seed(workspace)
    at = run(workspace)
    click(at, "编辑持仓")
    at.text_input(key=f"fund_edit_shares_{fund['id']}").input("150.0123")
    assert Portfolio(tmp_path).get_fund(fund["id"])["shares"] == "100"
    click(at, "保存持仓")
    assert not at.exception
    assert Portfolio(tmp_path).get_fund(fund["id"])["shares"] == "150.0123"
    assert not workspace.updates
    reopened = run(workspace)
    assert not reopened.exception and "150.0123" in text(reopened)


def test_update_is_one_explicit_call_and_private_fields_never_cross_boundary(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace, shares="123456.789", cost="987654.321")
    at = run(workspace)
    assert not workspace.updates
    click(at, "更新净值与预测")
    assert not at.exception and workspace.updates == [PRODUCT["id"]]
    assert not any(
        token in str(workspace.resolutions)
        for token in ("shares", "cost", "123456.789", "987654.321")
    )
    at.run()
    assert len(workspace.updates) == 1


@pytest.mark.parametrize("mode", ["public", "demo"])
def test_missing_workspace_browsing_never_creates_files(tmp_path, mode):
    root = tmp_path / mode / "absent"

    def missing_entry(root, mode):
        from rmbfund.ui.app import main

        main(root=root, mode=mode)

    at = AppTest.from_function(missing_entry, args=(str(root), mode), default_timeout=20).run()
    assert not at.exception and not root.exists()


def test_calendar_days_select_exact_execution_head_and_matching_chart(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    at = run(workspace)
    for days in (30, 14, 60):
        at.slider(key="trade_calendar_days").set_value(days).run()
        horizon = workspace.timing_inputs(PRODUCT["id"], calendar_days=days)["trade_plan"]["horizon"]
        click(at, "计算本周期判断")
        assessment = displayed(at)
        head = assessment["trade_scenario"]["decision"]["primary_head"]
        chart = assessment["forecast_chart"]["forecast"]
        assert head["horizon"] == chart["horizon"] == horizon
        assert head["entry_lag"] == chart["entry_lag"]
        assert chart["point"]["expected_change"] == head["expected_change"] == 0.025
        assert "近20笔净值累计涨跌" not in text(at) and not at.json
        assert workspace.summary_reads[-1][1] == {
            "horizon": horizon, "role": "primary", "model_id": "nav_ridge_empirical_v2",
        }
        assert not at.exception
    assert not workspace.updates


def test_personal_tools_keep_valuation_anchor_and_current_advice_separate(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace, shares="100", cost="150")
    workspace.detail["latest"].update(value="3", date="2026-10-23")
    workspace.prediction["anchor"].update(nav="2", date="2026-10-21")
    for head in workspace.prediction["forecasts"]:
        head["expected_change"] = 0.10
    at = run(workspace)
    click(at, "计算本周期判断")
    original = deepcopy(displayed(at))
    assert not any(item.key == "forecast_budget" for item in at.text_input)
    open_tools(at)
    shown = text(at)
    assert "当前估算市值 ¥300.00" in shown
    assert "按这个净值起点的持仓价值 ¥200.00" in shown
    assert "预期期末账面值 ¥220.00" in shown and "变动 ¥+20.00" in shown
    assert "假设份额不变" in shown and "不是从今天起算" in shown
    at.text_input(key="forecast_budget").input("1000").run()
    at.text_input(key="forecast_subscription_fee").input("0").run()
    assert not at.exception
    open_decision(at)
    assert displayed(at)["forecast_chart"] == original["forecast_chart"]
    assert displayed(at)["buyer_view"] == original["buyer_view"]
    assert not workspace.updates


def test_research_is_one_explicit_call_and_reuses_ram_detail_without_touching_holding(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    fund = seed(workspace, shares="123.456", cost="987.654")
    (tmp_path / "research-api.json").write_text(
        '{"protocol":"openai_chat_completions","base_url":"https://api.example.test/v1",'
        '"model":"synthetic-model","timeout":120,"max_tokens":1000}', encoding="utf-8"
    )
    workspace.online_result = online_context()
    at = run(workspace)
    at.text_input(key="fund_research_api_key").input("session-only-key").run()
    click(at, "更新研究")
    assert not at.exception and len(workspace.research_calls) == 1
    call = workspace.research_calls[0]
    assert call["api_key"] == "session-only-key" and call["context"]
    assert "session-only-key" not in str({key: call[key] for key in ("subject", "settings", "context", "public_subjects")})
    assert all(key not in str(call["subject"]) for key in ("shares", "cost", "987.654"))
    assert displayed(at)["research_thesis"]["record_id"] == "synthetic-research-v2"
    assert "合成新研究摘要" not in text(at)
    open_tools(at)
    shown = text(at)
    assert "AI生成研究" in shown and "合成新研究摘要" in shown
    assert shown.count("合成新研究摘要") == 1
    assert shown.count("保留理由仍成立但不必加钱") == 1
    assert shown.count("追加没有新的价格优势") == 1
    for expected in ("基金长期任务可理解", "当前入场不优先", "收入先影响现金流",
                     "同底层但换基有费用", "现价是否已反映改善", "入场理由未增强", "第3页 / 表2", "合成原文"):
        assert expected in shown
    assert "本地作者研究" not in shown and "人工复核" in shown
    open_decision(at)
    saved = Portfolio(tmp_path).get_fund(fund["id"])
    assert saved["shares"] == "123.456" and saved["cost"] == "987.654"
    at.text_input(key="trade_redemption_fee").input("0.5").run()
    click(at, "保存这次判断")
    at.session_state["fund_navigation"] = "走势与资料"
    at.run()
    assert not at.exception and {metric.label: metric.value for metric in at.metric}["最新公布净值"] == "4.0000"
    click(at, "返回总览")
    assert "市值 ¥493.82" in text(at) and len(workspace.online_calls) == 1
    click(at, "查看当前判断")
    assert displayed(at)["research_thesis"]["record_id"] == "synthetic-research-v2"
    workspace.fail_research = True
    click(at, "更新研究")
    assert displayed(at)["research_thesis"]["payload"]["origin"]["generated_at"].endswith("08:05:00+08:00")
    assert "已有研究仍保留原结论和原日期" in text(at)
    at.run()
    assert len(workspace.research_calls) == 2 and len(workspace.online_calls) == 1
    workspace.fail_research = "partial"
    click(at, "更新研究")
    assert "新研究已保存" not in text(at) and "已有研究仍保留原结论和原日期" in text(at)
    assert displayed(at)["research_thesis"]["record_id"] == "synthetic-research-v2"
    assert len(workspace.research_calls) == 3


def test_legacy_opinion_is_not_expanded_and_personal_constraints_still_show():
    from rmbfund.personal_assessment import build_assessment

    legacy = {"status": "available", "needed_review": [], "payload": {
        "product": PRODUCT, "research_date": "2026-10-26", "posture": "maintain",
        "assessment": {"summary": "合成旧研究原意见", "supporting": [{"reason": "合成旧支持理由"}]},
        "facts": [{"id": "f1", "claim": "合成六月披露事实"}], "citations": [],
    }}
    assessment = build_assessment(None, subject=PRODUCT, research_thesis=legacy,
                                 sizing_plan={"whole_fund": {"gross_cash_cap": "0"}})
    before = deepcopy(assessment)

    def entry(value):
        from rmbfund.ui.buyer import render_investor, render_research

        render_investor(value)
        render_research(value)

    at = AppTest.from_function(entry, args=(assessment,), default_timeout=20).run()
    shown = text(at)
    assert not at.exception and shown.count("合成旧研究原意见") == 1
    assert "这份旧研究还没有完成买卖取舍" in shown
    assert "继续持有，但不等于值得加钱" not in shown
    assert "当前约束下不新增资金" in shown and "可新增金额上限为0" in shown
    panel = next(item for item in at.expander if "上次研究原文" in item.label)
    assert "2026-10-26" in panel.label and panel.proto.expanded is False
    assert assessment == before


def test_final_personal_action_is_not_replaced_by_research_and_old_saved_view_is_retained():
    from rmbfund.personal_assessment import build_assessment

    research = generated_thesis()
    research["payload"]["action_cases"]["new_buy"].update(action="buy", reason="合成新买机会强于留现金")
    research["payload"]["action_cases"]["addition"].update(action="add", reason="合成追加的独立取舍")
    value = build_assessment(None, subject=PRODUCT, research_thesis=research,
                             sizing_plan={"user_conditions": {"near_term_required": "yes",
                                                              "near_term_use_time": "下月交房租"}})

    def entry(payload):
        from rmbfund.ui.buyer import render_investor

        render_investor(payload)

    at = AppTest.from_function(entry, args=(value,), default_timeout=20).run()
    shown = text(at)
    assert not at.exception and "这笔钱近期要用，暂不新买" in shown
    assert "先留出近期用钱资金，暂不追加" in shown and "研究原意见：有条件买入" in shown
    assert shown.count("合成新买机会强于留现金") == shown.count("合成追加的独立取舍") == 1
    old = deepcopy(value)
    old["investor_decision"].pop("action_basis")
    old["investor_decision"]["policy_version"] = "fund_buyer_reviewed_thesis_v1"
    old["investor_decision"]["branches"]["held"].update(label="当时保存的持有标签", reasons=["当时保存的原理由"])
    before = deepcopy(old)
    history = AppTest.from_function(entry, args=(old,), default_timeout=20).run()
    assert not history.exception and "当时保存的原理由" in text(history)
    assert "当时保存的持有标签" in text(history) and old == before


def test_candidates_and_arbitrary_code_preview_never_add_or_borrow_holdings(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    peer = {**PRODUCT, "id": "open_fund:539002:A", "symbol": "539002", "name": "合成候选A", "share_class": "A"}
    workspace.product_rows.append(peer)
    workspace.comparison = {"status": "partial", "funds": [{"product": peer}],
        "comparisons": [{"product_id": peer["id"], "selection_reason": "合成同底层费用比较"}]}
    at = run(workspace)
    click(at, "重新在线检查")
    open_tools(at)
    click(at, "公开预览这只候选")
    assert not at.exception and "合成候选A" in text(at)
    assert "还没有把它加入你的持仓" in text(at)
    assert len(Portfolio(tmp_path).list_funds()) == 1 and len(workspace.online_calls) == 1
    assert not workspace.resolutions and not workspace.research_calls
    from rmbfund.service import Workspace

    monkeypatch.setattr(Workspace, "current_online", staticmethod(lambda *args, **kwargs: online_context()))
    code_at = AppTest.from_function(entry, args=(str(tmp_path),), default_timeout=20).run()
    code_at.text_input(key="fund_public_code").input("118001").run()
    click(code_at, "公开预览")
    assert not code_at.exception and "118001" in text(code_at)
    assert code_at.session_state["fund_public_selection"]["share_class"] == ""
    assert len(Portfolio(tmp_path).list_funds()) == 1 and not (tmp_path / "public.sqlite3").exists()


def test_date_fee_settings_load_and_only_explicit_remember_persists(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    records = {"conditions": {"calendar_days": 14, "trade_subscription_fee": "0.15",
                              "trade_redemption_fee": "1.5", "budget": "300"}, "saved_at": AT}
    writes = []

    def load(repository, subject):
        return deepcopy(records)

    def save(repository, subject, conditions):
        writes.append(deepcopy(conditions))
        records["conditions"] = deepcopy(conditions)
        return deepcopy(records)

    monkeypatch.setattr(Portfolio, "research_preferences", load, raising=False)
    monkeypatch.setattr(Portfolio, "save_research_preferences", save, raising=False)
    at = run(workspace)
    assert not at.exception and at.slider(key="trade_calendar_days").value == 14
    assert at.text_input(key="trade_subscription_fee").value == "0.15"
    at.text_input(key="trade_subscription_fee").input("0.5").run()
    assert not writes and not workspace.research_calls and not workspace.online_calls
    click(at, "记住这些设置")
    assert not at.exception and len(writes) == 1
    assert writes[0]["trade_subscription_fee"] == "0.5" and writes[0]["budget"] == "300"
    assert not Portfolio(tmp_path).assessment_history(PRODUCT)
    reopened = run(workspace)
    assert not reopened.exception and reopened.text_input(key="trade_subscription_fee").value == "0.5"
    assert not workspace.research_calls and len(writes) == 1


def test_new_local_thesis_reprojects_ram_without_renewing_source_dates(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    workspace.online_result = online_context()
    at = run(workspace)
    click(at, "重新在线检查")
    original = deepcopy(at.session_state["fund_current_contexts"][PRODUCT["id"]])
    thesis = generated_thesis()
    thesis["record_id"] = "new-local-research-from-another-session"
    thesis["payload"]["origin"]["generated_at"] = now_iso()
    thesis["payload"]["research_date"] = thesis["payload"]["origin"]["generated_at"][:10]
    assert original["evaluated_at"] < thesis["payload"]["origin"]["generated_at"]
    workspace.detail["research_thesis"] = thesis
    workspace.detail["issuer_analysis"] = {"product_id": PRODUCT["id"], "issuers": [
        {"issuer_id": "synthetic-complete-company", "financial_observations": [{"original": "full-data-kept"}]}]}
    at.run()
    evaluation = at.session_state["fund_current_evaluations"][PRODUCT["id"]]
    assert not at.exception and displayed(at)["research_thesis"]["record_id"] == thesis["record_id"]
    assert evaluation["evaluated_at"] >= thesis["payload"]["origin"]["generated_at"]
    assert evaluation["online_context"]["checked_at"] == original["checked_at"]
    assert evaluation["online_context"]["source_checks"] == original["source_checks"]
    assert evaluation["details"][PRODUCT["id"]]["latest"]["value"] == "4"
    assert len(workspace.online_calls) == 1 and not workspace.research_calls
    click(at, "保存这次判断")
    saved = Portfolio(tmp_path).assessment_history(PRODUCT)[0]["assessment"]
    assert saved["research_thesis"]["record_id"] == thesis["record_id"]
    assert saved["investment_analysis"]["issuer_fundamentals"]["analysis"] == workspace.detail["issuer_analysis"]
    assert saved["online_context"]["evaluated_at"] >= thesis["payload"]["origin"]["generated_at"]
    assert saved["online_context"]["checked_at"] == original["checked_at"]


def test_unconfigured_research_does_not_call_workspace_or_replace_old_detail(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    at = run(workspace)
    click(at, "更新研究")
    assert not at.exception and not workspace.research_calls
    assert "还缺API配置" in text(at)


def test_local_update_is_explicit_refresh_false_and_reports_no_network_check(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    at = run(workspace)
    assert not workspace.updates
    click(at, "用已保存净值更新预测")
    assert not at.exception and workspace.updates == [PRODUCT["id"]]
    assert workspace.update_refreshes == [False]
    assert "没有联网" in text(at)
    at.run()
    assert len(workspace.updates) == 1


def test_save_uses_previously_displayed_object_not_next_report_and_only_click_writes(
    tmp_path, monkeypatch
):
    from rmbfund.ui import decision as decision_ui

    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    built, saved = [], []
    original_build, original_save = decision_ui.build_assessment, Portfolio.save_assessment

    def build(*args, **kwargs):
        assessment = original_build(*args, **kwargs)
        built.append(assessment)
        return assessment

    def save(repository, assessment, **kwargs):
        # Button callback runs before the new render/read/build. It receives the
        # object whose numbers and reasons were actually visible at click time.
        assert assessment is built[-1]
        saved.append((assessment, workspace.forecast_reads))
        return original_save(repository, assessment, **kwargs)

    monkeypatch.setattr(decision_ui, "build_assessment", build)
    monkeypatch.setattr(Portfolio, "save_assessment", save)
    at = run(workspace)
    reads = workspace.forecast_reads
    shown_assessment = deepcopy(built[-1])
    workspace.prediction["record_id"] = "NEXT-REPORT-NOT-YET-DISPLAYED"
    workspace.prediction["execution_forecasts"][6]["expected_change"] = -0.2
    click(at, "保存这次判断")
    assert not at.exception and at.success and saved[0][1] == reads
    assert workspace.forecast_reads == reads + 1 and len(built) == 2
    rows = Portfolio(tmp_path).assessment_history(PRODUCT)
    assert len(rows) == 1
    stored = Portfolio(tmp_path).assessment_detail(rows[0]["id"])["assessment"]
    assert stored == saved[0][0] == shown_assessment
    at.text_input(key="trade_redemption_fee").input("1.5").run()
    assert len(Portfolio(tmp_path).assessment_history(PRODUCT)) == 1
    assert Portfolio(tmp_path).assessment_detail(rows[0]["id"])["assessment"] == stored
    assert not workspace.updates


def online_context(value="4", *, title="合成产业消息甲"):
    checked = now_iso()
    identifier = PRODUCT["id"]
    return {"schema_version": 1, "kind": "current_online_context", "status": "partial",
            "checked_at": checked, "evaluated_at": checked, "public_subjects": [deepcopy(PRODUCT)],
            "source_checks": [
                {"product_id": identifier, "family": "nav", "source_url": "https://example.test/nav",
                 "status": "ok", "parse_status": "parsed", "checked_at": checked, "source_date": "2026-10-23"},
                {"product_id": identifier, "family": "policy_news", "source_url": "https://example.test/stopped",
                 "status": "error", "checked_at": checked, "reason": "合成失败，不自动重试"}],
            "nav_by_product": {identifier: [
                {"product_id": identifier, "date": "2026-10-23", "value": value, "currency": "CNY",
                 "nav_type": "final_unit_nav_candidate", "source": "synthetic_online",
                 "source_url": "https://example.test/nav", "observed_at": checked,
                 "available_at": checked, "share_class": "C"}]},
            "records_by_product": {}, "messages_by_product": {identifier: [
                {"id": "synthetic-message", "title": title, "published_at": "2026-10-23",
                 "content_status": "title_only", "excerpt": "仅用来验证线上事实与原判断分开。",
                 "source_url": "https://example.test/message", "relevance_reason": "披露发行人相关的合成消息"}]},
            "warnings": []}


def test_online_current_is_explicit_cached_for_edits_and_history_never_fetches(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    workspace.online_result = online_context()
    at = run(workspace, overview=True)
    click(at, "查看当前判断")
    assert not at.exception and not workspace.online_calls
    click(at, "重新在线检查")
    assert len(workspace.online_calls) == 1
    assert displayed(at)["amounts"]["current"]["market_value"] == "400"
    assert "合成产业消息甲" not in text(at)
    at.text_input(key="trade_redemption_fee").input("1").run()
    at.slider(key="trade_calendar_days").set_value(7).run()
    assert len(workspace.online_calls) == 1
    workspace.online_result = online_context("5", title="合成产业消息乙")
    click(at, "重新在线检查")
    assert not at.exception and len(workspace.online_calls) == 2
    assert workspace.online_calls[-1][1]["blocked_urls"] == ["https://example.test/stopped"]
    assert displayed(at)["amounts"]["current"]["market_value"] == "500"
    at.session_state["fund_navigation"] = "历史记录"
    at.run()
    assert not at.exception and len(workspace.online_calls) == 2 and workspace.history_reads
    at.run()
    assert len(workspace.online_calls) == 2
    assert all("shares" not in str(call) and "cost" not in str(call) for call in workspace.online_calls)


def test_online_basis_saved_is_original_display_object_and_is_not_refetched_in_history(tmp_path):
    workspace = FakeWorkspace(tmp_path)
    seed(workspace)
    workspace.online_result = online_context()
    at = run(workspace)
    click(at, "重新在线检查")
    workspace.online_result = online_context("5", title="合成产业消息乙")
    click(at, "保存这次判断")
    repository = Portfolio(tmp_path)
    identifier = repository.assessment_history(PRODUCT)[0]["id"]
    original = repository.assessment_detail(identifier)["assessment"]
    assert original["amounts"]["current"]["market_value"] == "400"
    assert original["online_context"]["messages_by_product"][PRODUCT["id"]][0]["title"] == "合成产业消息甲"
    click(at, "重新在线检查")
    assert displayed(at)["online_context"]["messages_by_product"][PRODUCT["id"]][0]["title"] == "合成产业消息乙"
    assert repository.assessment_detail(identifier)["assessment"] == original
    calls = len(workspace.online_calls)
    at.session_state["fund_navigation"] = "历史记录"
    at.run()
    assert not at.exception and len(workspace.online_calls) == calls
    assert "当时保存的操作建议" in text(at)
    assert "合成产业消息甲" not in text(at)
    assert repository.assessment_detail(identifier)["assessment"] == original


def test_unknown_public_code_uses_ram_online_without_a_public_database(tmp_path, monkeypatch):
    from rmbfund.service import Workspace

    calls = []
    subject = {key: value for key, value in PRODUCT.items() if key != "id"}
    subject["name"] = "仅本机的昵称"
    Portfolio(tmp_path).save_fund(subject, shares="100", cost="200")

    def collect(subjects, **kwargs):
        calls.append(deepcopy(subjects))
        return online_context()

    monkeypatch.setattr(Workspace, "current_online", staticmethod(collect))
    at = AppTest.from_function(entry, args=(str(tmp_path),), default_timeout=20).run()
    click(at, "查看当前判断")
    assert not calls
    click(at, "重新在线检查")
    assert not at.exception and len(calls) == 1
    assert "仅本机的昵称" not in str(calls)
    assert displayed(at)["amounts"]["current"]["market_value"] == "400"
    assert "合成产业消息甲" not in text(at)
    assert not (tmp_path / "public.sqlite3").exists()
    click(at, "保存这次判断")
    stored = Portfolio(tmp_path).assessment_history(subject)
    assert len(stored) == 1
    at.session_state["fund_navigation"] = "历史记录"
    at.run()
    assert not at.exception and len(calls) == 1
    assert "当时保存的操作建议" in text(at)
    assert "None笔" not in text(at) and "personal_assessment_v4_fund_buyer" not in text(at)

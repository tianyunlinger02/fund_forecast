"""Permanent, offline checks for the explicit research API/runner boundary."""
from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from rmbfund.providers.research_api import ResearchAPIError
from rmbfund.research_contract import research_output_example
from rmbfund.research_runner import run_research
from rmbfund.research_settings import (
    is_configured,
    read_research_api_settings,
)

PRODUCT = {
    "id": "open_fund:018147:C", "symbol": "018147", "name": "公开测试基金",
    "kind": "open_fund", "share_class": "C", "exchange": "OTC",
}
START = "2026-10-27T10:00:00+08:00"
API_DONE = "2026-10-27T10:00:01+08:00"
END = "2026-10-27T10:00:02+08:00"
DOC = {
    "id": "doc-earnings", "product_id": PRODUCT["id"], "source_url": "https://example.com/report",
    "title": "公开报告", "published_at": "2026-10-26", "publication_precision": "date",
    "observed_at": START, "available_at": START, "content_status": "publisher_body",
    "sections": [{"locator": "section:1", "text": "需求改善，但价格反映程度仍未知。"}],
}


def _payload():
    value = deepcopy(research_output_example(product=PRODUCT, as_of=START))
    value.update(note_id="api-case", research_date="2026-10-27", posture="maintain")
    value["facts"] = [{
        "id": "f-demand", "claim": "需求改善，但价格反映程度未知。", "status": "factual",
        "effective_date": "2026-10-26", "issuer_ids": ["issuer-1"], "channels": ["earnings"],
        "citation_ids": ["c-report"],
    }]
    value["citations"] = [{
        "id": "c-report", "kind": "online_evidence", "evidence_id": DOC["id"],
        "locator": "section:1", "quote": "需求改善", "claim": "公开报告提到需求改善。",
    }]
    value["assessment"] = {
        "summary": "经营事实有支持，但买入价格证据不足。",
        "supporting": [{"reason": "需求事实支持保留观察。", "fact_ids": ["f-demand"]}],
        "counterevidence": [],
        "priced_in": {"status": "unknown", "reason": "没有可核对的价格反映资料。"},
        "missing": ["没有市场一致预期。"],
        "falsifiers": ["后续披露否定需求改善。"],
        "change_conditions": ["取得价格反映证据后重新判断。"], "review_date": "2026-11-27",
    }
    value["action_cases"] = {
        "new_buy": {"action": "wait", "reason": "经营改善不等于现在值得新买。", "fact_ids": ["f-demand"]},
        "held": {"action": "hold", "reason": "现有保留理由仍成立。", "fact_ids": ["f-demand"]},
        "addition": {"action": "wait", "reason": "追加需要价格证据。", "fact_ids": ["f-demand"]},
    }
    value["drivers"] = [{"channel": "earnings", "mechanism": "需求可能影响发行人收入，但当前未覆盖基金传导比例。",
                          "fund_transmission": "本次没有完整持仓覆盖，因此不把单家公司直接当作基金涨幅。",
                          "fact_ids": ["f-demand"]}]
    value["fund_quality"] = {"conclusion": "长期质量暂可观察。", "reason": "有需求事实但资料覆盖有限。",
                              "fact_ids": ["f-demand"]}
    value["entry_case"] = {"conclusion": "当前不优先新买。", "reason": "没有价格反映依据。",
                            "fact_ids": ["f-demand"]}
    value["alternatives"] = []
    value["key_unknowns"] = [{"question": "价格是否已反映需求改善？", "affects": ["new_buy", "addition"],
                               "critical": False}]
    value["change_evaluation"] = {"summary": "首次API研究，没有旧研究可比较。", "fact_ids": ["f-demand"]}
    return value


def FakeResponse(payload, status_code=200, headers=None):
    return httpx.Response(status_code, json=payload, headers=headers or {"request-id": "req-test"})


def _clock():
    return iter((START, API_DONE, END)).__next__


def _settings(protocol="openai_chat_completions"):
    return {"protocol": protocol, "base_url": "https://api.example/v1", "model": "existing-model",
            "timeout": 4, "max_tokens": 3000}


def test_settings_are_nonsecret_and_presence_check_never_calls_network(tmp_path):
    path = tmp_path / "research-api.json"
    path.write_text(json.dumps({"protocol": "openai_chat_completions", "base_url": "https://api.example/v1",
                                "model": "m", "timeout": 5, "max_tokens": 100}), encoding="utf-8")
    settings = read_research_api_settings(path)
    assert "api_key" not in settings
    assert is_configured(settings, api_key="provided-only-for-test", environ={})
    assert not is_configured(settings, environ={})
    with pytest.raises(ValueError):
        read_research_api_settings(tmp_path / "other.json")


def test_runner_sends_only_public_material_and_binds_real_document():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
        body = kwargs["json"]
        encoded = json.dumps(body, ensure_ascii=False)
        assert "secret-key" not in encoded
        assert "shares" not in encoded and "cost_basis" not in encoded and "budget" not in encoded
        assert "action_cases" in encoded and "new_buy" in encoded and "held" in encoded and "addition" in encoded
        return FakeResponse({"model": "existing-model", "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                             "choices": [{"message": {"role": "assistant", "content": json.dumps(_payload())},
                                          "finish_reason": "stop"}]})

    result = run_research(
        product=PRODUCT,
        context={"evidence_by_product": {PRODUCT["id"]: [DOC]}},
        detail={"shares": "99", "cost_basis": "100", "budget": "200"},
        settings=_settings(), api_key="secret-key", environ={}, request=request, clock=_clock(),
    )
    assert result["status"] == "generated"
    assert result["research"]["origin"]["run_id"] == result["run_id"]
    assert result["research"]["origin"]["human_reviewed"] is False
    assert result["research"]["citations"][0]["source_url"] == DOC["source_url"]
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
    assert len(calls) == 1


def test_unconfigured_and_upstream_failure_do_not_replace_research_or_leak_secret():
    no_call = []

    def request(*args, **kwargs):
        no_call.append(True)
        raise AssertionError("must not call without key")

    missing = run_research(product=PRODUCT, context={}, detail={}, settings=_settings(), environ={},
                           request=request, clock=_clock())
    assert missing["status"] == "unconfigured"
    assert not no_call

    def failed(*args, **kwargs):
        return FakeResponse({"error": {"message": "secret-key should not be surfaced"}}, status_code=500)

    failed_result = run_research(product=PRODUCT, context={}, detail={}, settings=_settings(),
                                 api_key="secret-key", environ={}, request=failed, clock=_clock())
    assert failed_result["status"] == "failed"
    assert failed_result["research"] is None
    assert "secret-key" not in json.dumps(failed_result, ensure_ascii=False)


def test_httpx_failure_is_short_and_does_not_include_response_body():
    response = FakeResponse({}, status_code=401)
    with pytest.raises(ResearchAPIError) as exc:
        from rmbfund.providers.research_api import ResearchAPIClient
        ResearchAPIClient(_settings(), api_key="secret-key", environ={}).complete(
            [{"role": "user", "content": "public"}], request=lambda *a, **k: response)
    assert exc.value.code == "authentication_failed"
    assert "secret-key" not in str(exc.value)


@pytest.mark.parametrize("mutation", ["unknown_source", "wrong_quote", "title_only", "invented_number"])
def test_invalid_actual_reference_or_factual_number_is_reported(mutation):
    payload, doc = _payload(), deepcopy(DOC)
    if mutation == "unknown_source":
        payload["citations"][0]["evidence_id"] = "not-in-input"
    elif mutation == "wrong_quote":
        payload["citations"][0]["quote"] = "实际原文并没有这句话"
    elif mutation == "title_only":
        doc["content_status"] = "title_only"
    else:
        payload["facts"][0]["claim"] = "利润增长99%。"
        payload["citations"][0]["claim"] = "利润增长99%。"  # Cannot validate against itself.
    result = run_research(
        product=PRODUCT, context={"evidence_by_product": {PRODUCT["id"]: [doc]}}, detail={},
        settings=_settings(), api_key="test-key", environ={}, clock=_clock(),
        request=lambda *a, **k: FakeResponse({"model": "m", "choices": [
            {"message": {"content": json.dumps(payload)}}]}),
    )
    assert result["status"] == "failed" and result["research"] is None
    assert result["validation"]["status"] == "failed"
    assert result["errors"][0]["code"] == "research_validation_failed"


def test_same_material_preserves_different_model_action_reasons_and_real_metadata():
    outputs, inputs = [], []
    for action, reason in (("hold", "需求支持原保留理由。"), ("reduce", "最强反方削弱原保留理由。")):
        payload = _payload()
        payload["action_cases"]["held"].update(action=action, reason=reason)
        payload["origin"] = {"provider": "fake", "model": "fake", "run_id": "fake", "generated_at": END}
        payload["citations"][0]["source_url"] = "https://invented.example/"

        def request(*args, **kwargs):
            inputs.append(kwargs["json"])
            return FakeResponse({"model": "actual-reported-model", "choices": [
                {"message": {"content": json.dumps(payload)}}]})

        outputs.append(run_research(product=PRODUCT,
                                    context={"evidence_by_product": {PRODUCT["id"]: [DOC]}}, detail={},
                                    settings=_settings(), api_key="test-key", environ={},
                                    clock=_clock(), request=request))
    first_prompt = json.loads(inputs[0]["messages"][1]["content"])
    second_prompt = json.loads(inputs[1]["messages"][1]["content"])
    assert first_prompt["INPUT_PACKAGE"] == second_prompt["INPUT_PACKAGE"]
    assert [row["research"]["action_cases"]["held"]["action"] for row in outputs] == ["hold", "reduce"]
    assert outputs[1]["research"]["action_cases"]["held"]["reason"] == "最强反方削弱原保留理由。"
    assert outputs[0]["research"]["origin"]["model"] == "actual-reported-model"
    assert outputs[0]["research"]["origin"]["generated_at"] == API_DONE
    assert outputs[0]["research"]["citations"][0]["source_url"] == DOC["source_url"]


@pytest.mark.parametrize("claim, accepted", [
    ("同比约12.35%。", True), ("同比约12.36%。", False),
    ("同比12.35%。", False), ("同比12.3456789%。", True),
    ("同比约12.35%，但利润增幅为99%。", False),
])
def test_numeric_display_rounding_is_explicit_and_uses_adopted_precision(claim, accepted):
    doc, payload = deepcopy(DOC), _payload()
    doc["sections"][0]["text"] = "同比增幅为0.123456789。"
    payload["citations"][0]["quote"] = doc["sections"][0]["text"]
    payload["facts"][0]["claim"] = claim
    result = run_research(product=PRODUCT, context={"evidence_by_product": {PRODUCT["id"]: [doc]}},
                          detail={}, settings=_settings(), api_key="test-key", environ={}, clock=_clock(),
                          request=lambda *a, **k: FakeResponse({"model": "m", "choices": [
                              {"message": {"content": json.dumps(payload)}}]}))
    assert (result["status"] == "generated") is accepted


def test_retained_document_keeps_independent_quotes_at_same_and_different_locators():
    from rmbfund.research_runner import _bundle

    original = {**_payload()["citations"][0], **{key: DOC[key] for key in (
        "product_id", "source_url", "title", "published_at", "publication_precision",
        "content_status", "observed_at", "available_at")}}
    second = {**original, "id": "c-risk", "quote": "重要反方尚待核对。"}
    third = {**original, "id": "c-later", "locator": "section:2", "quote": "另一节的独立摘录。"}
    previous = {"payload": {"citations": [original, second, third, original]}}
    package, _, _, _ = _bundle(PRODUCT, {}, {}, None, previous, START)
    assert package["documents"][0]["sections"] == [
        {"locator": row["locator"], "text": row["quote"]} for row in (original, second, third)
    ]
    payload = _payload()
    payload["citations"][0]["quote"] = second["quote"]
    result = run_research(product=PRODUCT, context={}, previous=previous, detail={}, settings=_settings(),
                          api_key="test-key", environ={}, clock=_clock(),
                          request=lambda *a, **k: FakeResponse({"model": "m", "choices": [
                              {"message": {"content": json.dumps(payload)}}]}))
    assert result["status"] == "generated"
    assert result["research"]["citations"][0]["available_at"] == DOC["available_at"]


def test_evidence_requests_are_bounded_and_never_claim_unperformed_acquisition():
    calls, acquisitions = [], []

    def request(*args, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse({"model": "m", "usage": {"total_tokens": 10}, "choices": [{"message": {
            "content": json.dumps({"research": _payload(), "evidence_requests": {
                "evidence_ids": [DOC["id"]], "questions": ["请核对后续披露。"]}})}}]})

    def acquire_more(query):
        acquisitions.append(query)
        return {}  # No acquisition is falsely reported as successful.

    result = run_research(product=PRODUCT,
                          context={"evidence_by_product": {PRODUCT["id"]: [DOC]}}, detail={},
                          settings=_settings(), api_key="test-key", environ={}, clock=lambda: START,
                          request=request, acquire_more=acquire_more)
    assert len(calls) == 3 and len(acquisitions) == 2
    assert result["status"] == "partial" and result["research"] is not None
    assert result["usage"]["total_tokens"] == 30
    assert [row["status"] for row in result["evidence_requests"]] == ["unavailable", "unavailable", "round_limit_reached"]


def test_explicit_anthropic_protocol_uses_messages_shape_without_automatic_switch():
    from rmbfund.providers.research_api import ResearchAPIClient

    def handle(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["system"] == "only public research"
        assert body["messages"] == [{"role": "user", "content": "public input"}]
        return FakeResponse({"model": "configured-anthropic-model", "stop_reason": "end_turn",
                             "content": [{"type": "text", "text": "{}"}],
                             "usage": {"input_tokens": 3, "output_tokens": 2}})

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        response = ResearchAPIClient(_settings("anthropic_messages"), api_key="test-key", environ={}).complete(
            [{"role": "user", "content": "public input"}], system="only public research", request=http.request)
    assert response.text == "{}" and response.usage == {"input_tokens": 3, "output_tokens": 2}


def test_history_rich_public_package_reaches_api_with_current_basis_not_repeated_history():
    from calendar import monthrange
    from datetime import date, timedelta
    from rmbfund.advice_evidence import normalize_record
    from rmbfund.research_runner import _bundle

    records = []

    def record(payload, end, published, issuer):
        return normalize_record({
            "product_id": PRODUCT["id"], "family": "issuer_fundamentals", "source": "synthetic",
            "upstream": "synthetic", "source_url": f"https://example.com/{issuer}/{payload['kind']}/{end}",
            "report_date": end, "published_at": published, "publication_precision": "date",
            "observed_at": START, "available_at": START, "payload": payload,
        })

    for issuer_index in range(4):
        issuer, security = f"issuer-{issuer_index}", f"security-{issuer_index}"
        for index in range(31):  # Original quarterly history, not padded strings or repeated fake rows.
            year, quarter = 2019 + index // 4, index % 4 + 1
            start = date(year, quarter * 3 - 2, 1)
            end = date(year, quarter * 3, monthrange(year, quarter * 3)[1])
            following = end + timedelta(days=1)
            next_quarter = following.month // 3 + 1
            next_end = date(following.year, next_quarter * 3,
                            monthrange(following.year, next_quarter * 3)[1])
            metric = {"name": "eps", "category": "actual", "period": {
                "start": start.isoformat(), "end": end.isoformat(), "basis": "quarter",
                "fiscal_year": year, "fiscal_quarter": quarter},
                "currency": "USD", "unit": "currency_per_share", "scale": "1",
                "accounting_basis": "GAAP", "period_role": "primary", "value": str(index + 1),
                "security_id": security, "eps_basis": "diluted", "share_basis": "ordinary",
                "split_basis": "ordinary-reported", "locator": "table:eps"}
            revenue = {key: value for key, value in metric.items()
                       if key not in {"eps_basis", "share_basis", "split_basis", "security_id"}}
            revenue.update(name="revenue", unit="currency", value=str((index + 1) * 100), locator="table:revenue")
            guidance = {**revenue, "category": "guidance", "period": {
                "start": following.isoformat(), "end": next_end.isoformat(), "basis": "quarter",
                "fiscal_year": following.year, "fiscal_quarter": next_quarter},
                "low": str((index + 2) * 90), "high": str((index + 2) * 110), "locator": "table:guidance"}
            guidance.pop("value")
            records.append(record({
                "kind": "financial_release", "issuer_id": issuer, "issuer_name": issuer,
                "security_id": security, "release_id": f"{issuer}-{index}", "revision_id": "original",
                "revision_kind": "original", "metrics": [metric, revenue, guidance],
            }, end.isoformat(), (end + timedelta(days=10)).isoformat(), issuer))
        records.extend([
            record({"kind": "security_mapping", "issuer_id": issuer, "issuer_name": issuer,
                    "security_id": security, "symbol": f"S{issuer_index}", "exchange": "NYSE",
                    "quote_currency": "USD", "share_basis": "ordinary", "locator": "table:mapping"},
                   "2026-10-26", "2026-10-26", issuer),
            record({"kind": "price_history", "security_id": security, "symbol": f"S{issuer_index}",
                    "exchange": "NYSE", "currency": "USD", "price_basis": {
                        "kind": "unadjusted", "split_basis": "ordinary-reported",
                        "description": "synthetic close", "locator": "table:close"},
                    "closes": [{"date": "2026-10-26", "close": "400"}]},
                   "2026-10-26", "2026-10-26", issuer),
        ])
    detail = {"evidence_records": records}
    context = {"evidence_by_product": {PRODUCT["id"]: [DOC]}}
    package, evidence, _, _ = _bundle(PRODUCT, context, detail, None, None, START)
    assert len(evidence["records"]) == len(records) == 132
    assert len(evidence["calculations"]) == 12  # Latest EPS YoY, actual/prior guidance, TTM/price per security.
    assert package["structured_record_coverage"]["omitted_older_issuer_records"] > 80
    assert len(json.dumps(package, ensure_ascii=False).encode()) < 250_000
    issuer = package["issuer_analysis"]["issuers"][0]
    assert "financial_observations" not in issuer and "quarter_actuals" not in issuer
    assert issuer["eps_yoy"][0]["period"]["end"] == "2026-09-30"
    assert len(issuer["securities"][0]["ttm_diluted_eps"]["quarters"]) == 4
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse({"model": "m", "choices": [{"message": {"content": json.dumps(_payload())}}]})

    result = run_research(product=PRODUCT, context=context, detail=detail, settings=_settings(),
                          api_key="test-key", environ={}, clock=_clock(), request=request)
    assert result["status"] == "generated" and len(calls) == 1


def test_retained_excerpt_reuses_only_its_saved_economic_basis_and_original_clocks():
    from rmbfund.advice_evidence import normalize_record
    from rmbfund.research_runner import _bundle

    old_time = "2026-10-26T10:00:00+08:00"
    record = normalize_record({
        "product_id": PRODUCT["id"], "family": "fees", "source": "synthetic", "upstream": "synthetic",
        "source_url": DOC["source_url"], "report_date": "2026-10-26", "observed_at": old_time,
        "available_at": old_time, "published_at": "2026-10-26", "publication_precision": "date",
        "payload": {"management_rate": "0.01", "custody_rate": "0.001", "sales_service_rate": "0.004",
                    "redemption_bands": [], "holding_day_basis": "unknown", "channel_verified": False},
    })
    retained = {**_payload()["citations"][0], **{key: DOC[key] for key in (
        "product_id", "source_url", "title", "published_at", "publication_precision", "content_status")},
        "observed_at": old_time, "available_at": old_time, "record_basis": [record]}
    previous = {"payload": {"citations": [retained]}}
    package, evidence, _, _ = _bundle(PRODUCT, {}, {}, None, previous, START)
    assert evidence["records"] == [record]
    assert evidence["record_links"] == [{"record_id": record["id"], "evidence_id": DOC["id"],
                                          "locator": "section:1", "quote": "需求改善"}]
    assert package["documents"][0]["content_status"] == "retained_excerpt"
    assert package["documents"][0]["observed_at"] == old_time
    assert package["documents"][0]["available_at"] == old_time
    without_basis = deepcopy(previous)
    without_basis["payload"]["citations"][0].pop("record_basis")
    _, older, _, _ = _bundle(PRODUCT, {}, {}, None, without_basis, START)
    assert older["records"] == [] and older["record_links"] == []

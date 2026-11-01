"""Provider tests are entirely offline by default; no optional AKShare needed."""

import json


import pytest


from rmbfund.domain import Product, timestamp


from rmbfund.providers import akshare as adapter


OBSERVED = "2026-01-12T16:00:00+08:00"


FUND = Product("fund-000051", "000051", "场外基金", "open_fund")


@pytest.fixture(autouse=True)
def _fixed_observation_clock(monkeypatch):
    monkeypatch.setattr(adapter, "now_iso", lambda: OBSERVED)


def _profile_body(rows):
    return 'var fS_code = "000051"; var Data_netWorthTrend = ' + json.dumps(rows) + ";"


def _epoch_ms(day):
    return int(timestamp(day + "T00:00:00+08:00").timestamp() * 1000)


def test_direct_eastmoney_fallback_nav_source_and_response_completion_time(monkeypatch):
    finish = "2026-01-12T16:00:03+08:00"

    def fake(url, timeout):
        assert url == "https://fund.eastmoney.com/pingzhongdata/000051.js"
        assert timeout == 0.5
        monkeypatch.setattr(adapter, "now_iso", lambda: finish)
        return 200, _profile_body([{"x": _epoch_ms("2026-01-09"), "y": 1.234, "unitMoney": ""}])

    monkeypatch.setattr(adapter, "_http_get_text", fake)
    result = adapter.EastmoneyProfileProvider(timeout=0.5, retries=0).fetch(FUND, OBSERVED)
    assert result.status == "partial"
    (fact,) = result.facts
    assert fact.source == "eastmoney_js_json"
    assert fact.observed_at == fact.available_at == finish
    assert fact.published_at is None
    assert fact.event_date == "2026-01-09"
    assert fact.payload["unit_nav"] == "1.234"
    assert result.raw_records[0]["observed_at"] == finish
    assert any("不覆盖分红" in warning for warning in result.warnings)
    json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("<html>captcha</html>", id="html-without-identity"),
        pytest.param(
            "var Data_netWorthTrend = "
            + json.dumps(
                [
                    {"x": _epoch_ms("2026-01-09"), "y": 1.2},
                ]
            )
            + ";",
            id="missing-identity",
        ),
        pytest.param(
            _profile_body([{"x": _epoch_ms("2026-01-09"), "y": 1.2}]).replace(
                '"000051"', '"000001"'
            ),
            id="wrong-identity",
        ),
    ],
)
def test_direct_eastmoney_rejects_missing_or_wrong_product_identity(monkeypatch, body):
    monkeypatch.setattr(adapter, "_http_get_text", lambda *args: (200, body))
    result = adapter.EastmoneyProfileProvider(retries=0).fetch(FUND, OBSERVED)
    assert result.status == "error" and not result.facts
    assert any("fS_code 缺失或与请求产品不符" in warning for warning in result.warnings)

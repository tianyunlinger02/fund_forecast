"""Offline contract tests from the retained Dayfund/Withlees public response shapes."""

import json


import pytest


from rmbfund.providers import intraday


CHECKED = "2026-10-24T13:36:00+08:00"


SNAPSHOT = "2026-10-24T13:31:03+08:00"


DAYFUND_C = "2026-10-22|2.0310|2.0310|-0.1300|-6.40%|0.25%|0.0050|2.0360|2.1610|2026-10-23|09:10:00"


DAYFUND_DOMESTIC = (
    "2026-10-23|1.5837|1.5837|0.0261|1.68%|1.62%|0.0253|1.5829|1.5576|2026-10-23|15:00:00"
)


FIELDS = {
    "status",
    "blocked",
    "source",
    "source_name",
    "symbol",
    "change",
    "estimated_nav",
    "source_time",
    "source_time_kind",
    "source_timezone_known",
    "checked_at",
    "baseline_date",
    "baseline_note",
    "market_session",
    "method",
    "reason",
    "url",
}


def _withlees(**fund_changes):
    return {
        "updatedAt": SNAPSHOT,
        "session": "closed",
        "marketSession": "closed",
        "funds": {
            "539002": {
                "name": "建信新兴市场混合(QDII)A",
                "changePercent": -0.5843,
                "rawPercent": -0.5843,
                "adjustment": 0.0,
                "method": "activeDisclosure",
                **fund_changes,
            },
            "017653": {"changePercent": None, "method": "none", "reason": "no_data"},
        },
    }


def _parse_withlees(data, symbol="539002"):
    return intraday.parse_withlees(json.dumps(data), symbol, checked_at=CHECKED)


@pytest.fixture(autouse=True)
def _offline_clock(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Tests must explicitly supply a mock transport; no network is allowed")

    monkeypatch.setattr(intraday.httpx, "Client", no_network)
    monkeypatch.setattr(intraday, "now_iso", lambda: CHECKED)


@pytest.mark.parametrize(
    "symbol,text,change,nav,source_time",
    [
        ("018147", DAYFUND_C, "0.0025", "2.0360", "2026-10-23 09:10:00"),
        ("000051", DAYFUND_DOMESTIC, "0.0162", "1.5829", "2026-10-23 15:00:00"),
    ],
)
def test_dayfund_observed_samples_and_unknown_baseline(symbol, text, change, nav, source_time):
    result = intraday.parse_dayfund(text, symbol, checked_at=CHECKED)
    assert set(result) == FIELDS
    assert result["status"] == "ok"
    assert result["source"] == "dayfund"
    assert result["source_name"] == "基金速查网"
    assert result["url"] == intraday.SOURCES["dayfund"]["url"]
    assert result["symbol"] == symbol
    assert result["change"] == change
    assert result["estimated_nav"] == nav
    assert result["source_time"] == source_time
    assert result["source_time_kind"] == "estimate"
    assert result["source_timezone_known"] is False
    assert result["checked_at"] == CHECKED
    assert result["baseline_date"] is None
    assert "未明确" in result["baseline_note"]
    assert "未回传基金代码" in result["baseline_note"]
    assert result["market_session"] is None
    assert result["method"] is None
    assert result["reason"] is None
    assert json.loads(json.dumps(result)) == result


def test_withlees_observed_change_and_snapshot_semantics():
    result = _parse_withlees(_withlees())
    assert set(result) == FIELDS
    assert result["status"] == "ok"
    assert result["source"] == "withlees" and result["source_name"] == "持仓记录助手"
    assert result["symbol"] == "539002"
    assert result["change"] == "-0.005843"
    assert result["estimated_nav"] is None
    assert result["source_time"] == SNAPSHOT
    assert result["source_time_kind"] == "snapshot"
    assert result["source_timezone_known"] is True
    assert result["checked_at"] == CHECKED
    assert result["baseline_date"] is None
    assert "未提供" in result["baseline_note"] and "整份快照" in result["baseline_note"]
    assert result["market_session"] == "closed"
    assert result["method"] == "activeDisclosure"
    assert result["reason"] is None
    assert result["url"] == intraday.SOURCES["withlees"]["url"]
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("symbol", ["018147", "000051", "006479", "013308"])
def test_withlees_no_a_to_c_or_similar_fund_substitution(symbol):
    result = _parse_withlees(_withlees(), symbol)
    assert result["status"] == "missing"
    assert result["symbol"] == symbol
    assert result["source"] == "withlees" and result["source_name"] == "持仓记录助手"
    assert result["change"] is None
    assert result["source_time"] == SNAPSHOT
    assert result["source_time_kind"] == "snapshot"
    assert result["reason"]

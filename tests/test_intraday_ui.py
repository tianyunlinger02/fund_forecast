"""Estimate UI stays opt-in, source-time honest, and outside daily/personal data."""

import pytest


from streamlit.testing.v1 import AppTest


from rmbfund.providers.intraday import parse_dayfund, parse_withlees


from rmbfund.ui import intraday


from test_ui import click, text


CHECKED = "2026-10-24T13:36:00+08:00"


DAYFUND = "2026-10-22|2.0310|2.0310|-0.1300|-6.40%|0.25%|0.0050|2.0360|2.1610|2026-10-23|09:10:00"


WITHLEES = """{"updatedAt":"2026-10-24T13:31:03+08:00","marketSession":"closed",
"funds":{"539002":{"changePercent":-0.5843,"method":"activeDisclosure"}}}"""


def sample(source="dayfund", symbol="018147", **changes):
    result = (
        parse_dayfund(DAYFUND, symbol, checked_at=CHECKED)
        if source == "dayfund"
        else parse_withlees(WITHLEES, symbol, checked_at=CHECKED)
    )
    return result | changes


def entry(symbol="018147", mode="public", kind="open_fund"):
    from rmbfund.ui.intraday import render_intraday

    render_intraday(symbol=symbol, mode=mode, kind=kind)


@pytest.fixture
def fetches(monkeypatch):
    requests = []

    def fetch(source, symbol):
        requests.append((source, symbol))
        return sample(source, symbol)

    intraday.cached_estimate.clear()
    monkeypatch.setattr(intraday, "fetch_estimate", fetch)
    yield requests
    intraday.st.cache_data.clear()


def test_passive_view_does_not_fetch_or_display_fake_zero(fetches):
    at = AppTest.from_function(entry).run()
    assert not at.exception and not fetches
    assert "最近估算" in text(at) and not at.metric
    assert not at.toggle(key="fund_intraday_auto").value
    at.run()
    at.selectbox(key="fund_intraday_source").set_value("withlees").run()
    assert not at.exception and not fetches


def test_explicit_refresh_displays_raw_source_time_not_local_check_time(fetches):
    at = AppTest.from_function(entry).run()
    click(at, "刷新估算")
    assert not at.exception and fetches == [("dayfund", "018147")]
    assert next(m.value for m in at.metric if m.label == "来源估算涨跌") == "+0.25%"
    assert "2026-10-23 09:10:00" in text(at) and "2026-10-24 13:36" in text(at)
    assert "时区未标明" in text(at) and "基准日期：来源未说明" in text(at)
    assert "今天涨" not in text(at) and "实时" not in text(at)
    assert len(at.get("plotly_chart")) == 1
    table = at.dataframe[0].value
    assert len(table) == 1 and table.iloc[0, 0] == "2026-10-23 09:10:00"
    click(at, "刷新估算")
    at.run()
    assert not at.exception and fetches == [("dayfund", "018147")]
    assert len(at.dataframe[0].value) == 1

"""Two anonymous third-party estimate snapshots, not verified live quotes or NAV facts."""

from __future__ import annotations

import json
import re
import ssl
from datetime import date, datetime
from decimal import Decimal

import httpx

from ..domain import decimal, now_iso, timestamp

SOURCES = {
    "dayfund": {"name": "基金速查网", "url": "https://fund.dayfund.com.cn/"},
    "withlees": {"name": "持仓记录助手", "url": "https://withlees.cn/h5/"},
}
MAX_RESPONSE_BYTES = 256 * 1024
_FORMAT_REASON = "来源返回的数据格式暂时无法识别。"
_BASELINE_NOTES = {
    "dayfund": (
        "来源未明确涨幅基准净值及日期，时间字段的时区未核实；"
        "仅按请求代码匹配，响应未回传基金代码。"
    ),
    "withlees": (
        "来源未提供涨幅基准净值及日期；时间为整份快照生成时间，"
        "不是单只基金或底层证券的行情时间。"
    ),
}


def _result(source: str, symbol: str, checked_at: str) -> dict:
    info = SOURCES.get(source, {}) if isinstance(source, str) else {}
    return {
        "status": "error",
        "blocked": False,
        "source": source,
        "source_name": info.get("name", "未知来源"),
        "symbol": symbol,
        "change": None,
        "estimated_nav": None,
        "source_time": None,
        "source_time_kind": "estimate" if source == "dayfund" else "snapshot",
        "source_timezone_known": False,
        "checked_at": checked_at,
        "baseline_date": None,
        "baseline_note": _BASELINE_NOTES.get(source, "来源未提供涨幅基准日期。")
        if isinstance(source, str) else "来源未提供涨幅基准日期。",
        "market_session": None,
        "method": None,
        "reason": None,
        "url": info.get("url", ""),
    }


def _valid_input(result: dict) -> bool:
    if not isinstance(result["source"], str) or result["source"] not in SOURCES:
        result.update(status="unsupported", reason="暂不支持这个估值来源。")
        return False
    if not isinstance(result["symbol"], str) or not re.fullmatch(r"[0-9]{6}", result["symbol"]):
        result.update(status="unsupported", reason="仅支持六位数字基金代码。")
        return False
    try:
        if not isinstance(result["checked_at"], str):
            raise ValueError
        timestamp(result["checked_at"])
    except ValueError:
        result.update(checked_at=now_iso(), reason="检查时间必须包含有效时区。")
        return False
    return True


def _text(text: str) -> str:
    if (not isinstance(text, str) or len(text) > MAX_RESPONSE_BYTES
            or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES):
        raise ValueError
    return text.strip()


def _blocked_html(text: str) -> bool:
    lower = text.lower()
    return lower.startswith("<") and any(
        word in lower for word in ("captcha", "验证码", "登录", "login", "log in", "authentic")
    )


def _tls_verification_error(error: BaseException) -> bool:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if (isinstance(error, ssl.SSLCertVerificationError)
                or "CERTIFICATE_VERIFY_FAILED" in str(error).upper()):
            return True
        error = error.__cause__ or error.__context__
    return False


def _percent(value: str) -> Decimal:
    if not value.endswith("%"):
        raise ValueError
    return decimal(value[:-1]) / 100


def parse_dayfund(text: str, symbol: str, *, checked_at: str) -> dict:
    """Parse the observed eleven-field pipe response; it does not echo a fund code."""
    result = _result("dayfund", symbol, checked_at)
    if not _valid_input(result):
        return result
    try:
        raw = _text(text)
        if not raw:
            result.update(status="missing", reason="来源暂未提供这只基金的估值。")
            return result
        if "<" in raw or ">" in raw:
            result["blocked"] = _blocked_html(raw)
            raise ValueError
        fields = [part.strip() for part in raw.split("|")]
        if len(fields) != 11:
            raise ValueError
        if any(value in {"", "--", "-", "null"} for value in fields):
            result.update(status="missing", reason="来源估值字段暂不完整，未按零处理。")
            return result

        # Validate the observed shape, but never derive a baseline from daily NAVs or field 8.
        if date.fromisoformat(fields[0]).isoformat() != fields[0]:
            raise ValueError
        for index in (1, 2, 3, 6, 8):
            decimal(fields[index])
        _percent(fields[4])
        change = _percent(fields[5])
        nav = decimal(fields[7])
        if nav <= 0:
            raise ValueError
        source_time = f"{fields[9]} {fields[10]}"
        parsed_time = datetime.strptime(source_time, "%Y-%m-%d %H:%M:%S")
        if parsed_time.strftime("%Y-%m-%d %H:%M:%S") != source_time:
            raise ValueError
        # No timezone or freshness is inferred from a naive source timestamp.
        result.update(status="ok", change=format(change, "f"),
                      estimated_nav=format(nav, "f"), source_time=source_time)
    except (ValueError, TypeError, OverflowError):
        result["reason"] = _FORMAT_REASON
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError


def _label(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError
    return value or None


def parse_withlees(text: str, symbol: str, *, checked_at: str) -> dict:
    """Read one exact fund key from the shared JSON; updatedAt is a snapshot time."""
    result = _result("withlees", symbol, checked_at)
    if not _valid_input(result):
        return result
    try:
        raw = _text(text)
        result["blocked"] = _blocked_html(raw)
        data = json.loads(raw, parse_float=Decimal, parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
        if not isinstance(data, dict) or not isinstance(data.get("funds"), dict):
            raise ValueError
        source_time = data.get("updatedAt")
        if not isinstance(source_time, str):
            raise ValueError
        generated_at = timestamp(source_time)
        result.update(source_time=source_time, source_timezone_known=True)
        if generated_at > timestamp(checked_at):
            result["reason"] = "来源快照时间晚于本次检查时间，暂不展示估值。"
            return result
        market_session = _label(data.get("marketSession") or data.get("session"))
        result["market_session"] = market_session
        fund = data["funds"].get(symbol)
        if fund is None:
            result.update(status="missing", reason="来源暂未提供这个基金代码的估值。")
            return result
        if not isinstance(fund, dict):
            raise ValueError
        for key in ("symbol", "code", "fundcode", "fundCode"):
            if key in fund and fund[key] != symbol:
                raise ValueError
        method = _label(fund.get("method"))
        if fund.get("changePercent") is None:
            result.update(status="missing", method=method,
                          reason="来源暂未提供这只基金的估值涨幅，未按零处理。")
            return result
        change = decimal(fund["changePercent"]) / 100
        result.update(status="ok", change=format(change, "f"), method=method)
    except (ValueError, TypeError, OverflowError, RecursionError):
        result["reason"] = _FORMAT_REASON
    return result


def fetch_estimate(source: str, symbol: str) -> dict:
    """Make one bounded HTTPS attempt, without credentials, proxies, redirects, or fallback."""
    result = _result(source, symbol, now_iso())
    if not _valid_input(result):
        return result
    url = (
        "https://fund.dayfund.com.cn/ajs/ajaxdata.shtml"
        f"?showtype=getfundvalue&fundcode={symbol}"
        if source == "dayfund" else "https://withlees.cn/api/h5/fund-estimates.json"
    )
    reason = None
    text = ""
    try:
        with httpx.Client(verify=True, trust_env=False, follow_redirects=False,
                          timeout=10.0) as client:
            with client.stream("GET", url) as response:
                if response.status_code in {401, 403, 429}:
                    result["blocked"] = True
                    reason = "来源拒绝访问或要求验证，本次已停止，不会自动重试。"
                elif 300 <= response.status_code < 400:
                    result["blocked"] = True
                    reason = "来源发生跳转，本次未跟随。"
                elif response.status_code != 200:
                    reason = "来源暂时无法读取，本次未重试。"
                else:
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes(chunk_size=16 * 1024):
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            reason = "来源响应超过大小上限，本次已停止。"
                            break
                        chunks.append(chunk)
                    if reason is None:
                        text = b"".join(chunks).decode("utf-8-sig", "strict")
    except httpx.TimeoutException:
        reason = "来源请求超时，本次未重试。"
    except (httpx.HTTPError, ssl.SSLError) as exc:
        result["blocked"] = _tls_verification_error(exc)
        reason = "来源连接失败（可能为网络或证书问题），本次未重试。"
    except UnicodeError:
        reason = _FORMAT_REASON
    result["checked_at"] = now_iso()
    if reason is not None:
        result["reason"] = reason
        return result
    parser = parse_dayfund if source == "dayfund" else parse_withlees
    return parser(text, symbol, checked_at=result["checked_at"])

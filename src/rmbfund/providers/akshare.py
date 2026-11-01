"""Bounded, optional AKShare adapters. No Store, credentials, or synthetic fallback.

AKShare is an API wrapper, not the upstream: every Fact names Eastmoney or Sina.
All historical rows become available at *this fetch*, never at an invented old
publication time. The worker is isolated so an upstream requests.get without a
timeout cannot hang the caller indefinitely.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..domain import SHANGHAI, Fact, Product, ValidationError, decimal, now_iso, timestamp

MAX_SOURCE_ROWS = 50_000
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
EASTMONEY_ETF_URL = "https://quote.eastmoney.com/{market}{symbol}.html"
SINA_ETF_URL = "https://finance.sina.com.cn/fund/quotes/{symbol}/bc.shtml"
EASTMONEY_FUND_URL = "https://fundf10.eastmoney.com/jjjz_{symbol}.html"
EASTMONEY_ACTION_URL = "https://fundf10.eastmoney.com/fhsp_{symbol}.html"
EASTMONEY_PROFILE_JS_URL = "https://fund.eastmoney.com/pingzhongdata/{symbol}.js"
EASTMONEY_JS_SOURCE = "eastmoney_js_json"
MAX_JS_BYTES = 2 * 1024 * 1024
SINA_DAILY_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
)

# Static code only. Neither downloaded JavaScript nor response strings are executed.
# -I excludes cwd and PYTHONPATH; the current interpreter's optional dependency is used.
_WORKER = r'''
import contextlib
import importlib
import json
import sys

request = json.loads(sys.stdin.read())
try:
    with contextlib.redirect_stdout(sys.stderr):
        ak = importlib.import_module("akshare")
        function = getattr(ak, request["function"])
        frame = function(**request["kwargs"])
    if not hasattr(frame, "to_dict") or not hasattr(frame, "notna"):
        raise ValueError("AKShare response is not a DataFrame")
    if len(frame) > 50000:
        raise ValueError("AKShare response exceeds 50000 rows")
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    def encode(value):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError("Unsupported response value: " + type(value).__name__)
    message = json.dumps({"status": "ok", "records": records}, ensure_ascii=False,
                         allow_nan=False, default=encode)
    if len(message.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("AKShare response exceeds 16 MiB")
except ModuleNotFoundError as exc:
    status = "missing_dependency" if exc.name == "akshare" else "error"
    message = json.dumps({"status": status, "error": str(exc)[:1000]})
except Exception as exc:
    message = json.dumps({"status": "error", "error": type(exc).__name__ + ": " + str(exc)[:1000]})
sys.stdout.write(message)
'''


@dataclass
class ProviderResult:
    provider: str
    product_id: str
    facts: list[Fact] = field(default_factory=list)
    raw_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "error"
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible dataclasses, including actual upstream metadata."""
        return asdict(self)


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: str = "error") -> None:
        super().__init__(message)
        self.status = status


def _http_get_text(url: str, timeout: float) -> tuple[int, str]:
    """Bounded HTTP fallback; unlike requests' default, timeout is mandatory."""
    try:
        import httpx
    except ImportError:
        request = Request(url, headers={"User-Agent": "rmbfund-eastmoney-js/0.1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_JS_BYTES:
                        raise ProviderError("Eastmoney JS 响应超过 2 MiB")
                    chunks.append(chunk)
                return int(response.status), b"".join(chunks).decode("utf-8", "replace")
        except HTTPError as exc:
            raise ProviderError(f"Eastmoney JS HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Eastmoney JS 网络失败: {exc}") from exc
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "rmbfund-eastmoney-js/0.1"}) as response:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_JS_BYTES:
                    raise ProviderError("Eastmoney JS 响应超过 2 MiB")
                chunks.append(chunk)
            return response.status_code, b"".join(chunks).decode("utf-8", "replace")
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Eastmoney JS 网络失败: {type(exc).__name__}: {exc}") from exc


def _parse_profile_networth(text: str, expected_symbol: str | None = None) -> list[dict[str, Any]]:
    """Parse exactly one documented JSON assignment; never execute page JavaScript."""
    if expected_symbol is not None:
        identity = re.search(r'\bfS_code\s*=\s*("[0-9]{6}")\s*;', text)
        if identity is None or json.loads(identity.group(1)) != expected_symbol:
            raise ProviderError("Eastmoney profile fS_code 缺失或与请求产品不符")
    match = re.search(r"\bData_netWorthTrend\s*=\s*(\[[\s\S]*?\])\s*;", text)
    if not match:
        raise ProviderError("未找到 Data_netWorthTrend JSON 数组")
    try:
        value = json.loads(match.group(1), parse_constant=lambda token: _reject_constant(token))
    except (ValueError, TypeError) as exc:
        raise ProviderError("Data_netWorthTrend 不是合法有限 JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ProviderError("Data_netWorthTrend 必须是对象列表")
    if any("x" not in item or "y" not in item for item in value):
        raise ProviderError("Data_netWorthTrend 缺少 x/y")
    return value


def _js_millis_date(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError("Eastmoney JS x 必须是有限毫秒时间戳")
    if value != int(value) or not 631_123_200_000 <= value <= 4_102_444_800_000:
        raise ValidationError("Eastmoney JS x 超出 1990..2100 毫秒时间戳范围（不能混用秒）")
    return datetime.fromtimestamp(value / 1000, tz=SHANGHAI).date().isoformat()


def _observation(value: str | None) -> str:
    observed = timestamp(value or now_iso())
    if observed > timestamp(now_iso()) + timedelta(minutes=5):
        raise ValidationError("本次观察时间不得在未来（允许 5 分钟时钟偏差）")
    return observed.isoformat()


def _date(value: Any, field_name: str = "date") -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} 必须是日期，收到 {value!r}")
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ValidationError(f"{field_name} 非法日期: {text}") from exc
    # pandas Timestamp serializes dates with midnight; never truncate arbitrary text.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]00:00:00(?:[+]00:00)?", text):
        return _date(text[:10], field_name)
    raise ValidationError(f"{field_name} 必须为 YYYY-MM-DD: {text!r}")


def _number(value: Any, field_name: str, *, positive: bool = False) -> str:
    if value is None or isinstance(value, (dict, list, tuple, bool)):
        raise ValidationError(f"{field_name} 缺失或不是数字")
    number = decimal(value)
    # Bound magnitude and scale *before* fixed-point formatting (e.g. 1e999999).
    if number.copy_abs() > Decimal("1e18") or number.as_tuple().exponent < -18 or len(number.as_tuple().digits) > 40:
        raise ValidationError(f"{field_name} 数值范围/精度过大")
    if positive and number <= 0:
        raise ValidationError(f"{field_name} 必须大于零")
    return format(number, "f")


def _dividend_per_unit(value: Any) -> str | None:
    """Recognize the documented per-unit RMB phrase, not arbitrary numbers in prose."""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*每份派现金\s*(\d+(?:\.\d+)?)\s*元\s*", value)
        if match:
            return _number(match.group(1), "每份分红")
        if not re.fullmatch(r"\s*[+]?(?:\d+(?:\.\d*)?|\.\d+)\s*", value):
            return None
    return _number(value, "每份分红")


def _split_ratio(value: Any) -> str | None:
    if value in (None, "", "---", "--", "-", "暂无数据"):
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*1\s*:\s*([+]?(?:\d+(?:\.\d*)?|\.\d+))\s*", value)
        if match:
            return _number(match.group(1), "拆分折算比例", positive=True)
    return _number(value, "拆分折算比例", positive=True)


def _run_akshare(function: str, kwargs: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    """Use a killable process, not a thread whose requests call keeps running."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _WORKER],
            input=json.dumps({"function": function, "kwargs": kwargs}),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"{function}: 超过 {timeout:g}s，已终止子进程", "timeout") from exc
    except OSError as exc:
        raise ProviderError(f"无法启动 AKShare 隔离子进程: {exc}") from exc
    if completed.returncode:
        raise ProviderError(f"{function}: 子进程退出 {completed.returncode}: {completed.stderr[:500]}")
    if len(completed.stdout.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ProviderError("AKShare response exceeds 16 MiB")
    try:
        response = json.loads(completed.stdout, parse_constant=lambda x: _reject_constant(x))
    except (ValueError, TypeError) as exc:
        raise ProviderError(f"{function}: 非法 JSON 响应") from exc
    if not isinstance(response, dict):
        raise ProviderError(f"{function}: 非法响应结构")
    if response.get("status") != "ok":
        raise ProviderError(str(response.get("error", "AKShare response error")),
                            str(response.get("status", "error")))
    records = response.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ProviderError(f"{function}: records 必须是对象列表")
    if len(records) > MAX_SOURCE_ROWS:
        raise ProviderError("AKShare response exceeds row limit")
    return records


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


class EastmoneyProfileProvider:
    """Direct, safe Eastmoney profile fallback for final unit-NAV candidates.

    It is intentionally separate from AKShare so callers can record a source
    switch. It reads only ``Data_netWorthTrend`` JSON and never executes JS.
    It does not claim complete coverage or provide dividends, splits, fees, or
    subscription/redemption status.
    """

    def __init__(self, timeout: float = 15, retries: int = 1, min_interval: float = 1):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValidationError("timeout 必须在 (0, 300] 秒内")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValidationError("retries 必须是 0..3 的整数")
        if not math.isfinite(min_interval) or not 0 <= min_interval <= 60:
            raise ValidationError("min_interval 必须在 [0, 60] 秒内")
        self.timeout = float(timeout)
        self.retries = retries
        self.min_interval = float(min_interval)
        self._last_call: float | None = None

    def fetch(self, product: Product, observed_at: str, start_date: str | None = None,
              end_date: str | None = None) -> ProviderResult:
        observed_start = _observation(observed_at)
        end = _date(end_date or timestamp(observed_start).date().isoformat(), "end_date")
        start = _date(start_date or "1990-01-01", "start_date")
        if start > end or end > timestamp(observed_start).date().isoformat():
            raise ValidationError("日期范围倒置或结束日晚于本次观察日期")
        result = ProviderResult(provider="eastmoney_js", product_id=product.id,
                                source_url=EASTMONEY_PROFILE_JS_URL.format(symbol=product.symbol))
        if product.kind != "open_fund":
            result.status = "unsupported"
            result.warnings.append("Eastmoney profile JS fallback 只适用于场外基金单位净值")
            return result
        if product.is_demo:
            result.status = "unsupported"
            result.warnings.append("演示产品不得调用真实数据源")
            return result
        if not re.fullmatch(r"\d{6}", product.symbol):
            raise ValidationError("Eastmoney profile symbol 必须是 6 位数字代码")
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            if self._last_call is not None:
                time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_call)))
            self._last_call = time.monotonic()
            try:
                status_code, text = _http_get_text(result.source_url, self.timeout)
                if status_code != 200:
                    raise ProviderError(f"Eastmoney JS HTTP {status_code}")
                records = _parse_profile_networth(text, expected_symbol=product.symbol)
                if not records:
                    raise ProviderError("Data_netWorthTrend 为空")
                if len(records) > MAX_SOURCE_ROWS:
                    raise ProviderError("Data_netWorthTrend 超过行数上限")
                # Observe at response completion, never at a caller-supplied old
                # start time. Publication time is still unknown.
                actual_observed = max(timestamp(observed_start), timestamp(now_iso())).isoformat()
                result.raw_records.extend({"source": EASTMONEY_JS_SOURCE,
                                           "function": "Data_netWorthTrend",
                                           "source_url": result.source_url,
                                           "observed_at": actual_observed,
                                           "attempt": attempt + 1, "record": row}
                                          for row in records)
                by_day: dict[str, Fact] = {}
                for row in records:
                    event_date = _js_millis_date(row["x"])
                    if event_date > timestamp(actual_observed).date().isoformat():
                        raise ValidationError(f"响应包含未来日期: {event_date}")
                    payload = {"unit_nav": _number(row["y"], "Data_netWorthTrend.y", positive=True),
                               "currency": "CNY", "nav_type": "final_unit_nav_candidate",
                               "source_fields": sorted(row)}
                    if event_date in by_day and by_day[event_date].payload != payload:
                        raise ValidationError(f"Data_netWorthTrend 同日冲突: {event_date}")
                    by_day[event_date] = Fact(product.id, "nav", event_date, payload,
                                             EASTMONEY_JS_SOURCE, actual_observed,
                                             source_url=result.source_url, available_at=actual_observed)
                result.facts = [by_day[day] for day in sorted(by_day) if start <= day <= end]
                if not result.facts:
                    raise ValidationError("请求区间没有可验证的单位净值行")
                result.status = "partial"  # NAV-only candidate, never full entitlement coverage
                result.warnings.extend(errors)
                result.warnings.extend([
                    "来源切换为 Eastmoney JS JSON；只覆盖 Data_netWorthTrend 候选单位净值，不覆盖分红、拆分、费率或申赎状态。",
                    "页面字段/发布时间/历史完整性未经独立验证，不应据此宣称完整净值覆盖。",
                ])
                return result
            except (ProviderError, ValidationError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"尝试 {attempt + 1}: {type(exc).__name__}: {exc}")
        result.status = "error"
        result.facts = []
        result.warnings.extend(errors)
        return result


class SinaDailyProvider:
    """Explicit bounded-window Sina JSON fallback, ETF only, never a second source.

    This endpoint differs from AKShare fund_etf_hist_sina's historical download,
    but both are Sina. It is not independent corroboration. The latest ``datalen``
    daily bars are requested; earlier requested history is never claimed covered.
    """

    def __init__(self, timeout: float = 15, retries: int = 1, min_interval: float = 1,
                 datalen: int = 20):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValidationError("timeout 必须在 (0, 300] 秒内")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValidationError("retries 必须是 0..3 的整数")
        if not math.isfinite(min_interval) or not 0 <= min_interval <= 60:
            raise ValidationError("min_interval 必须在 [0, 60] 秒内")
        if isinstance(datalen, bool) or not isinstance(datalen, int) or not 1 <= datalen <= 1023:
            raise ValidationError("datalen 必须是 1..1023 的整数")
        self.timeout = float(timeout)
        self.retries = retries
        self.min_interval = float(min_interval)
        self.datalen = datalen
        self._last_call: float | None = None

    def fetch(self, product: Product, observed_at: str, start_date: str | None = None,
              end_date: str | None = None) -> ProviderResult:
        observed_start = _observation(observed_at)
        end = _date(end_date or timestamp(observed_start).date().isoformat(), "end_date")
        start = _date(start_date or "1990-01-01", "start_date")
        if start > end or end > timestamp(observed_start).date().isoformat():
            raise ValidationError("日期范围倒置或结束日晚于本次观察日期")
        result = ProviderResult(provider="sina_daily", product_id=product.id)
        if product.kind != "etf" or product.is_demo:
            result.status = "unsupported"
            result.warnings.append("新浪公开日K JSON fallback 仅支持明确分类的非演示 ETF")
            return result
        if not re.fullmatch(r"\d{6}", product.symbol):
            raise ValidationError("ETF symbol 必须是 6 位数字代码")
        prefix = {"sse": "sh", "sh": "sh", "szse": "sz", "sz": "sz"}.get(product.exchange.lower())
        if product.exchange and prefix is None:
            raise ValidationError("ETF exchange 只支持 SSE/SH 或 SZSE/SZ")
        prefix = prefix or ("sh" if product.symbol.startswith(("5", "6")) else "sz")
        result.source_url = SINA_DAILY_URL.format(symbol=prefix + product.symbol,
                                                 datalen=self.datalen)
        source = "sina:CN_MarketData.getKLineData"
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            if self._last_call is not None:
                time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_call)))
            self._last_call = time.monotonic()
            try:
                status_code, text = _http_get_text(result.source_url, self.timeout)
                if status_code != 200:
                    raise ProviderError(f"新浪日K HTTP {status_code}")
                rows = json.loads(text, parse_constant=lambda token: _reject_constant(token))
                if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
                    raise ValidationError("新浪日K不是非空对象数组")
                if len(rows) > self.datalen:
                    raise ValidationError(f"新浪日K返回超过请求的{self.datalen}行")
                actual_observed = max(timestamp(observed_start), timestamp(now_iso())).isoformat()
                result.raw_records.extend({"source": source, "function": "CN_MarketData.getKLineData",
                                           "source_url": result.source_url, "observed_at": actual_observed,
                                           "attempt": attempt + 1, "record": row} for row in rows)
                normalized = []
                for row in rows:
                    day = _date(row["day"], "day")
                    if "volume" not in row:
                        raise ValidationError("新浪日K缺少volume字段")
                    # A current-day daily candle can still be incomplete intraday.
                    if day == timestamp(actual_observed).date().isoformat() and timestamp(actual_observed).hour < 16:
                        result.warnings.append(f"忽略盘中尚未确认收盘的当日日K: {day}")
                        continue
                    normalized.append({**row, "date": day})
                facts = AkshareProvider._facts(normalized, product, actual_observed, start, end,
                                                source, result.source_url, sina=True)
                result.facts = facts
                result.status = "partial"
                result.warnings.extend(errors)
                result.warnings.extend([
                    f"来源切换为新浪公开日K JSON，至多最近{self.datalen}条；与 AKShare 的新浪历史接口同上游，不是独立交叉验证。",
                    "本接口未取得adjust参数，按源日K原价保存但不用于复权序列；成交量单位未独立核验，仅保存 source_volume。",
                    "不覆盖分红、拆分、完整历史、费率或申赎状态；缺少官方日历和权益事件须人工补充。",
                ])
                return result
            except (ProviderError, ValidationError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"尝试 {attempt + 1}: {type(exc).__name__}: {exc}")
        result.status = "error"
        result.facts = []
        result.warnings.extend(errors)
        return result


class AkshareProvider:
    def __init__(self, timeout: float = 15, retries: int = 1, min_interval: float = 1):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValidationError("timeout 必须在 (0, 300] 秒内")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValidationError("retries 必须是 0..3 的整数")
        if not math.isfinite(min_interval) or not 0 <= min_interval <= 60:
            raise ValidationError("min_interval 必须在 [0, 60] 秒内")
        self.timeout = float(timeout)
        self.retries = retries
        self.min_interval = float(min_interval)
        self._last_call: float | None = None

    def _request(self, function: str, kwargs: dict[str, Any], source: str, url: str,
                 parser: Callable[[list[dict[str, Any]]], list[Fact]],
                 result: ProviderResult) -> list[Fact]:
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            if self._last_call is not None:
                time.sleep(max(0.0, self.min_interval - (time.monotonic() - self._last_call)))
            self._last_call = time.monotonic()
            try:
                records = _run_akshare(function, kwargs, self.timeout)
                if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                    raise ValidationError("响应不是对象列表")
                if len(records) > MAX_SOURCE_ROWS:
                    raise ValidationError("响应行数超过上限")
                result.raw_records.extend({"source": source, "function": function,
                                           "source_url": url, "attempt": attempt + 1,
                                           "record": row} for row in records)
                if not records:
                    raise ValidationError("响应为空，不能证明历史完整或不存在权益事件")
                facts = parser(records)
                if errors:
                    result.warnings.append(f"{source}/{function}: 重试后成功；" + "; ".join(errors))
                return facts
            except (ProviderError, ValidationError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"尝试 {attempt + 1}: {type(exc).__name__}: {exc}")
                if isinstance(exc, ProviderError) and exc.status == "missing_dependency":
                    raise
        raise ProviderError(f"{source}/{function} 失败（{self.retries + 1} 次）: " + "; ".join(errors))

    def fetch(self, product: Product, observed_at: str, start_date: str | None = None,
              end_date: str | None = None) -> ProviderResult:
        observed = _observation(observed_at)
        end = _date(end_date or timestamp(observed).date().isoformat(), "end_date")
        start = _date(start_date or "1990-01-01", "start_date")
        if start > end or end > timestamp(observed).date().isoformat():
            raise ValidationError("日期范围倒置或结束日晚于本次观察日期")
        symbol = product.symbol
        if not re.fullmatch(r"\d{6}", symbol):
            raise ValidationError("AKShare 产品 symbol 必须是 6 位数字代码")
        result = ProviderResult(provider="akshare", product_id=product.id)
        if product.kind == "monitor_only":
            result.status = "unsupported"
            result.warnings.append("monitor_only 不自动归类为 ETF 或场外基金，请使用有依据的 CSV/公告")
            return result
        if product.is_demo:
            result.status = "unsupported"
            result.warnings.append("演示产品不得调用真实数据源；演示数据必须单独显式导入")
            return result
        if product.kind == "etf":
            result = self._fetch_etf(product, observed, start, end, result)
        else:
            result = self._fetch_fund(product, observed, start, end, result)
        # Historical rows are not knowable before this bounded fetch completes.
        actual_observed = max(timestamp(observed), timestamp(now_iso())).isoformat()
        result.facts = [replace(fact, observed_at=actual_observed, available_at=actual_observed)
                        for fact in result.facts]
        for record in result.raw_records:
            record["observed_at"] = actual_observed
        return result

    @staticmethod
    def _facts(rows: list[dict[str, Any]], product: Product, observed: str, start: str,
               end: str, source: str, url: str, *, sina: bool = False,
               nav: bool = False) -> list[Fact]:
        facts: dict[str, Fact] = {}
        for row in rows:
            event_date = _date(row["净值日期" if nav else "date" if sina else "日期"])
            if event_date > timestamp(observed).date().isoformat():
                raise ValidationError(f"响应包含未来日期: {event_date}")
            if nav:
                payload = {"unit_nav": _number(row["单位净值"], "单位净值", positive=True),
                           "currency": "CNY", "nav_type": "final_unit_nav"}
            else:
                columns = {"open": "open" if sina else "开盘",
                           "high": "high" if sina else "最高",
                           "low": "low" if sina else "最低",
                           "close": "close" if sina else "收盘"}
                payload = {key: _number(row[column], column, positive=True)
                           for key, column in columns.items()}
                payload.update({"adjustment": "raw", "currency": "CNY"})
                volume_column = "成交量" if not sina else "volume"
                if volume_column in row:
                    volume = decimal(_number(row[volume_column], volume_column))
                    if volume < 0 or volume != volume.to_integral_value():
                        raise ValidationError("上游成交量必须是非负整数")
                    # EM's current ETF history documentation does not state its
                    # volume unit; Sina's 手 label conflicts with sample scale.
                    # Preserve a validated raw value, never guess a share multiplier.
                    payload.update({"source_volume": str(volume), "source_volume_unit": "unverified"})
            fact = Fact(product_id=product.id, kind="nav" if nav else "bar", event_date=event_date,
                        payload=payload, source=source, observed_at=observed,
                        available_at=observed, source_url=url)
            if event_date in facts and facts[event_date].payload != payload:
                raise ValidationError(f"同日出现冲突数据: {event_date}")
            facts[event_date] = fact
        selected = [facts[day] for day in sorted(facts) if start <= day <= end]
        if not selected:
            raise ValidationError("所请求区间没有数据；不以区间外数据冒充覆盖")
        return selected

    def _fetch_etf(self, product: Product, observed: str, start: str, end: str,
                   result: ProviderResult) -> ProviderResult:
        symbol = product.symbol
        exchange = product.exchange.lower()
        prefix = {"sse": "sh", "sh": "sh", "szse": "sz", "sz": "sz"}.get(exchange)
        if exchange and prefix is None:
            raise ValidationError("ETF exchange 只支持 SSE/SH 或 SZSE/SZ")
        prefix = prefix or ("sh" if symbol.startswith(("5", "6")) else "sz")
        em_url = EASTMONEY_ETF_URL.format(market=prefix, symbol=symbol)
        sina_symbol = prefix + symbol
        sina_url = SINA_ETF_URL.format(symbol=sina_symbol)
        endpoints = [
            ("fund_etf_hist_em", {"symbol": symbol, "period": "daily", "adjust": "",
                                  "start_date": start.replace("-", ""),
                                  "end_date": end.replace("-", "")},
             "eastmoney:fund_etf_hist_em", em_url, False),
            ("fund_etf_hist_sina", {"symbol": sina_symbol},
             "sina:fund_etf_hist_sina", sina_url, True),
        ]
        for function, kwargs, source, url, sina in endpoints:
            result.source_url = url
            try:
                result.facts = self._request(function, kwargs, source, url,
                    lambda rows: self._facts(rows, product, observed, start, end, source, url,
                                             sina=sina), result)
                result.source_url = url
                if sina:
                    result.warnings.append("已显式备用为新浪，不与东财拼接")
                result.warnings.append("成交量手/份单位未完成独立核验，原值保留 source_volume，未写入标准 volume")
                result.status = "partial" if result.warnings else "ok"
                return result
            except ProviderError as exc:
                result.warnings.append(str(exc))
                if exc.status == "missing_dependency":
                    result.status = "missing_dependency"
                    return result
        result.status = "error"
        result.source_url = em_url
        return result

    def _fetch_fund(self, product: Product, observed: str, start: str, end: str,
                    result: ProviderResult) -> ProviderResult:
        nav_url = EASTMONEY_FUND_URL.format(symbol=product.symbol)
        action_url = EASTMONEY_ACTION_URL.format(symbol=product.symbol)
        source = "eastmoney:fund_open_fund_info_em"
        function = "fund_open_fund_info_em"
        result.source_url = nav_url
        try:
            result.facts.extend(self._request(function,
                {"symbol": product.symbol, "indicator": "单位净值走势"}, source, nav_url,
                lambda rows: self._facts(rows, product, observed, start, end, source, nav_url,
                                         nav=True), result))
        except ProviderError as exc:
            result.warnings.append(str(exc))
            if exc.status == "missing_dependency":
                result.status = "missing_dependency"
                return result
        for indicator in ("分红送配详情", "拆分详情"):
            try:
                result.facts.extend(self._request(function,
                    {"symbol": product.symbol, "indicator": indicator}, source, action_url,
                    lambda rows, indicator=indicator: self._actions(
                        rows, product, observed, start, end, source, action_url, indicator, result),
                    result))
            except ProviderError as exc:
                result.warnings.append(f"{indicator}: {exc}")
        has_nav = any(f.kind == "nav" for f in result.facts)
        result.status = ("partial" if result.warnings else "ok") if has_nav else "error"
        result.warnings.append("本源不证明历史申赎开放状态、精确历史费率或公告发布时间；不得据此补造")
        return result

    @staticmethod
    def _actions(rows: list[dict[str, Any]], product: Product, observed: str, start: str,
                 end: str, source: str, url: str, indicator: str,
                 result: ProviderResult) -> list[Fact]:
        """Actions without all entitlement dates are announcements, never precise cash flows."""
        facts: list[Fact] = []
        for row in rows:
            is_dividend = indicator == "分红送配详情"
            date_keys = ("权益登记日", "除息日", "派息日") if is_dividend else (
                "权益登记日", "除权日", "除息日", "份额到账日", "拆分折算日")
            recognized = date_keys + (("分红发放日", "红利发放日", "公告日期", "分红日期")
                                      if is_dividend else ("拆分日期", "公告日期"))
            parsed_dates = {key: _date(row[key], key) for key in recognized
                            if row.get(key) not in (None, "", "---", "--", "-", "暂无数据")}
            if not parsed_dates:
                raise ValidationError(f"{indicator} 缺少可解析事件日期，无法定位公告")
            if any(day > timestamp(observed).date().isoformat() for day in parsed_dates.values()):
                raise ValidationError(f"{indicator} 含未来日期，不可作为已发生权益入账")
            event_date = parsed_dates.get("除息日") or parsed_dates.get("除权日") or parsed_dates.get("拆分折算日") or min(parsed_dates.values())
            if not start <= event_date <= end:
                continue
            if is_dividend:
                pay = parsed_dates.get("派息日") or parsed_dates.get("分红发放日") or parsed_dates.get("红利发放日")
                # Typical AKShare rows contain only 年份/权益登记日/除息日/每份分红/分红发放日.
                # Never infer the missing record date from the ex date or vice versa.
                raw_cash = row.get("每份分红")
                cash = _dividend_per_unit(raw_cash) if raw_cash not in (None, "") else None
                if parsed_dates.get("权益登记日") and parsed_dates.get("除息日") and pay and cash is not None:
                    payload = {"record_date": parsed_dates["权益登记日"], "ex_date": parsed_dates["除息日"],
                               "pay_date": pay, "cash_per_unit": cash,
                               "currency": "CNY"}
                    if decimal(payload["cash_per_unit"]) < 0:
                        raise ValidationError("每份分红不得为负")
                    facts.append(Fact(product.id, "dividend", payload["ex_date"], payload,
                                      source, observed, source_url=url, available_at=observed))
                    continue
            else:
                ratio = _split_ratio(row.get("拆分折算比例"))
                record = parsed_dates.get("权益登记日")
                ex = parsed_dates.get("除权日") or parsed_dates.get("除息日")
                pay = parsed_dates.get("份额到账日")
                if record and ex and pay and ratio is not None:
                    if not record <= ex <= pay:
                        raise ValidationError("拆分权益日期顺序错误")
                    facts.append(Fact(product.id, "split", ex,
                                      {"record_date": record, "ex_date": ex, "pay_date": pay,
                                       "ratio": ratio}, source, observed, source_url=url,
                                      available_at=observed))
                    continue
            # A split's single 折算日/ratio is not enough to establish who was entitled.
            # The upstream usually omits record/ex/settlement dates. Preserve as evidence.
            message = f"{indicator} {event_date}: 缺少完整权益登记/除权息/到账日期或明确每份单位；仅公告待人工核验"
            result.warnings.append(message)
            facts.append(Fact(product.id, "announcement", event_date,
                              {"title": indicator, "requires_manual_verification": True,
                               "reason": message, "raw_record": row, "verified": False},
                              source, observed, source_url=url, available_at=observed,
                              quality="unverified"))
        return facts

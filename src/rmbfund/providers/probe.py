"""Low-frequency checks of free public sources; no probe output becomes a Fact."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

FUND_CODE = "000051"
MAX_BODY_BYTES = 2 * 1024 * 1024
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _http_get(url: str, timeout: float) -> tuple[int, str, str]:
    """Return status, content-type, bounded text. Every client has a timeout."""
    try:
        import httpx  # optional at runtime; pyproject includes it for the app
    except ImportError:
        request = Request(url, headers={"User-Agent": "rmbfund-source-probe/0.1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read(MAX_BODY_BYTES + 1)
                if len(body) > MAX_BODY_BYTES:
                    raise ValueError("response exceeds 2 MiB")
                return int(response.status), response.headers.get("content-type", ""), body.decode("utf-8", "replace")
        except HTTPError as exc:
            return exc.code, exc.headers.get("content-type", ""), ""
        except URLError as exc:
            raise TimeoutError(str(exc)) from exc
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "rmbfund-source-probe/0.1"}) as response:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise ValueError("response exceeds 2 MiB")
            chunks.append(chunk)
        body = b"".join(chunks)
        return response.status_code, response.headers.get("content-type", ""), body.decode("utf-8", "replace")


def _parse_fundgz(text: str) -> dict[str, Any]:
    match = re.fullmatch(r"\s*jsonpgz\s*\(\s*(\{.*\})\s*\)\s*;?\s*", text, re.DOTALL)
    if not match:
        raise ValueError("not the expected jsonpgz object")
    value = json.loads(match.group(1), parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON value {token}")))
    if not isinstance(value, dict) or not value.get("fundcode"):
        raise ValueError("jsonpgz object lacks fundcode")
    return value


def _parse_networth(text: str) -> list[dict[str, Any]]:
    # Share the safe JSON parser with the direct provider, not a JS engine.
    from .akshare import ProviderError, _parse_profile_networth
    try:
        return _parse_profile_networth(text)
    except ProviderError as exc:
        raise ValueError(str(exc)) from exc


def _candidate(code: str) -> list[dict[str, str]]:
    return [
        {
            "name": f"1234567_fundgz_{code}",
            "url": f"https://fundgz.1234567.com.cn/js/{code}.js",
            "parser": "fundgz",
            "license": "公开网页；仅盘中估值，不是最终净值，禁止写入 paper NAV；使用须遵守上游条款。",
            "coverage_expected": "盘中估值对象（可能随时失效），不覆盖最终净值/公告",
        },
        {
            "name": f"eastmoney_profile_{code}",
            "url": f"https://fund.eastmoney.com/pingzhongdata/{code}.js",
            "parser": "networth",
            "license": "公开网页 JavaScript 数据；仅安全读取已知 JSON 数组，不执行脚本；使用须遵守上游条款。",
            "coverage_expected": "Data_netWorthTrend 历史净值趋势候选，字段语义仍需人工核验",
        },
    ]


class _AnnouncementLinks(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._href:
            return
        title = " ".join("".join(self._text).split())
        target = urljoin(self.base_url, self._href)
        # Navigation tabs labelled 公告 are not a useful announcement list.
        looks_like_document = bool(re.search(r"\.pdf(?:$|[?#])|20\d{2}[-/]?\d{2}[-/]?\d{2}", target))
        if len(title) >= 10 and "基金" in title and looks_like_document and urlparse(target).scheme == "https":
            self.links.append({"title": title[:300], "url": target})
        self._href = ""
        self._text = []


def _official_candidates() -> list[dict[str, str]]:
    return [{"name": name, "url": url, "parser": "announcement_list",
             "license": "官方公开信息入口；需遵守网站条款/robots/限流，不宣称再分发许可，不绕过登录、验证码或网络限制。",
             "coverage_expected": "可直接读取、有标题和文档地址的基金公告列表（导航/SPA壳不算）"}
            for name, url in [
                ("csrc_fund_disclosure", "https://eid.csrc.gov.cn/fund/web/disclose/index.html"),
                ("sse_fund_announcements", "https://www.sse.com.cn/disclosure/fund/announcement/"),
                ("cninfo_fund_disclosure", "https://www.cninfo.com.cn/new/disclosure/fund"),
            ]]


def _one(candidate: dict[str, str], timeout: float, raw_dir: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "name": candidate["name"], "url": candidate["url"], "license": candidate["license"],
        "coverage": {"expected": candidate["coverage_expected"]}, "errors": [],
        "status": "error", "latency_ms": None,
    }
    try:
        status_code, content_type, text = _http_get(candidate["url"], timeout)
        report["http_status"] = status_code
        report["content_type"] = content_type
        if raw_dir is not None:
            # Each external payload gets its own fresh, empty directory. It stays
            # data-only; no interpreter is ever launched from this directory.
            raw_dir.mkdir(parents=False, exist_ok=False)
            raw_path = raw_dir / "response.txt"
            raw_path.write_text(text, encoding="utf-8")
            report["raw_artifact"] = str(raw_path)
        if status_code != 200:
            report["errors"].append(f"HTTP {status_code}")
        elif candidate["parser"] == "fundgz":
            value = _parse_fundgz(text)
            report["status"] = "ok_estimate_only"
            report["coverage"].update({"observed_fields": sorted(value), "fund_code": value.get("fundcode"),
                                        "data_time": value.get("gztime")})
            report["warnings"] = ["此接口是盘中估值，不能作为最终净值或 paper NAV。"]
        elif candidate["parser"] == "networth":
            values = _parse_networth(text)
            if not values:
                raise ValueError("Data_netWorthTrend is empty; no useful NAV coverage")
            report["status"] = "ok_limited"
            report["coverage"].update({"records": len(values),
                                        "first_x": values[0].get("x") if values else None,
                                        "last_x": values[-1].get("x") if values else None,
                                        "known_fields": sorted(values[0]) if values else []})
            report["warnings"] = ["仅验证已知 JSON 数组可安全解析，未验证该页面全部字段、发布时间或许可。"]
        elif candidate["parser"] == "announcement_list":
            parser = _AnnouncementLinks(candidate["url"])
            parser.feed(text)
            report["coverage"].update({"fund_document_links": len(parser.links),
                                        "sample": parser.links[:3]})
            if parser.links:
                report["status"] = "ok_limited"
                report["warnings"] = ["仅证明入口返回有用文档链接，尚未验证公告正文、产品归属或完整覆盖。"]
            else:
                report["status"] = "reachable_no_useful_list"
                report["errors"].append("HTTP 200 但未取得可直接读取的基金公告列表；可能为 JS/登录/导航壳，不算可用来源")
    except Exception as exc:  # probes report outages; they do not make network errors fatal
        report["errors"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
    report["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
    if report["status"] == "error" and not report["errors"]:
        report["errors"].append("响应未达到可报告覆盖标准")
    return report


def run_probes(output_dir: Path, timeout: float = 12, product_limit: int = 2) -> dict[str, Any]:
    """Check one fund (two HTTP endpoints), plus three official public entrances.

    ``product_limit=0`` skips network calls. A positive limit allows the one
    explicitly supported fund; it is an upper bound, not permission to scan the
    whole market. Requests are sequential and at least one second apart.
    ``output_dir`` should be ``Path('runs/artifacts/source-probes')``; no system
    temporary directory is used.
    """
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be in (0, 120]")
    if isinstance(product_limit, bool) or not isinstance(product_limit, int) or not 0 <= product_limit <= 20:
        raise ValueError("product_limit must be an integer in [0, 20]")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    observed_at = datetime.now(SHANGHAI).isoformat()
    run_key = observed_at.replace("+", "_").replace(":", "-")
    run_dir = directory / f"probe-{run_key}"
    run_dir.mkdir(exist_ok=False)
    candidates = _candidate(FUND_CODE) + _official_candidates() if product_limit else []
    probes: list[dict[str, Any]] = []
    for candidate in candidates:
        if probes:
            time.sleep(1)
        probes.append(_one(candidate, float(timeout), run_dir / candidate["name"]))
    successes = sum(item["status"] in {"ok_estimate_only", "ok_limited"} for item in probes)
    if not probes:
        overall = "skipped"
    elif not successes:
        overall = "error"
    elif successes < len(probes):
        overall = "partial"
    else:
        overall = "ok"
    report: dict[str, Any] = {
        "schema_version": 1,
        "observed_at": observed_at,
        "status": overall,
        "timeout_seconds": timeout,
        "product_limit": product_limit,
        "probes": probes,
        "notes": [
            "探针只报告免费公开来源的可达性与覆盖，不导入 Fact，不把盘中估值当最终净值。",
            "HTTP 200 不等于获得有用列表；解析失败、字段不足和许可不确定性均如实报告。",
            "AKShare 文档地址: https://akshare.akfamily.xyz/data/fund/fund_public.html；文档存在不证明运行环境可用。",
        ],
    }
    filename_time = report["observed_at"].replace("+", "_").replace(":", "-")
    report_file = directory / f"probe-report-{filename_time}.json"
    latest_file = directory / "latest.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_file.write_text(encoded, encoding="utf-8")
    latest_file.write_text(encoded, encoding="utf-8")
    report["artifact"] = str(report_file)
    return report

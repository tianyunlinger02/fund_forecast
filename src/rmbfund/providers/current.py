"""Current public fund information, parsed directly from anonymous responses in RAM.

The changing lists below were qualified against live responses, not fixed old
article URLs. No Store, response archive, download directory or authored research
revision is involved. Publisher text remains unreviewed investment evidence.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

from ..domain import ValidationError, digest, now_iso, timestamp
from .documents import _HTML, _decode, read_public_document, records_from_document
from .followed import Fetch, FetchResponse, _nav, _url, fetch_public

MAX_BYTES = 4 * 1024 * 1024
NAV_URL = "https://fund.eastmoney.com/pingzhongdata/{symbol}.js"
DISCLOSURE_URL = "https://finance.sina.com.cn/fund/quotes/{symbol}/bc.shtml"
NVIDIA_RSS = "https://nvidianews.nvidia.com/releases.xml"
MOTIR_INDEX = "https://english.motir.go.kr/eng/"
PARSERS = frozenset({
    "eastmoney_nav_js_v1", "sina_fund_announcements_v1", "nvidia_rss_v1",
    "motir_press_index_v1", "public_document_v1",
})
_IDENTITY_KEYS = ("product_id", "symbol", "name", "share_class", "currency", "public")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _subjects(public_subjects: Sequence[Mapping]) -> list[dict]:
    identities = {}
    for subject in public_subjects:
        identifier = subject.get("product_id", subject.get("id"))
        symbol = subject.get("symbol")
        if (not isinstance(identifier, str) or not identifier
                or not isinstance(symbol, str) or not re.fullmatch(r"\d{6}", symbol)
                or subject.get("public", True) is not True):
            raise ValidationError("current queries require a public identity and six-digit fund code")
        # Only these public fields survive; shares, costs and budgets never reach a source plan.
        item = {key: subject.get(key, "") for key in _IDENTITY_KEYS}
        item.update(product_id=identifier, public=True, currency=subject.get("currency", "CNY"))
        if identifier in identities and identities[identifier] != item:
            raise ValidationError("conflicting public identities for one product")
        identities[identifier] = item
    return list(identities.values())


def default_current_source_plan(public_subjects: Sequence[Mapping]) -> list[dict]:
    """Rolling NAV and fund-specific announcements, plus known 018147/539002 context."""
    nav_jobs, disclosure_jobs, context_jobs = [], [], []
    for index, subject in enumerate(_subjects(public_subjects)):
        common = {"product_id": subject["product_id"]}
        nav_jobs.append(dict(
            common, job_id=f"nav-{index}", family="nav",
            source_url=NAV_URL.format(symbol=subject["symbol"]),
            parser="eastmoney_nav_js_v1", config={"lookback_days": 400},
        ))
        disclosure_jobs.append(dict(
            common, job_id=f"disclosure-{index}", family="disclosure",
            source_url=DISCLOSURE_URL.format(symbol=subject["symbol"]),
            parser="sina_fund_announcements_v1", config={"max_items": 12, "article_limit": 3},
        ))
        if subject["symbol"] in {"018147", "539002"}:
            context_jobs.extend([
                dict(common, job_id=f"nvidia-{index}", family="policy_news",
                     source_url=NVIDIA_RSS, parser="nvidia_rss_v1",
                     config={"max_items": 5, "article_limit": 1,
                             "issuer_ids": ["issuer:nvidia"], "channels": ["earnings", "valuation"],
                             "relevance_reason": "此前披露涉及英伟达；公司算力需求消息可能影响盈利预期，"
                                                 "不证明基金今天仍持有或消息已转化为收入。"}),
                dict(common, job_id=f"motir-{index}", family="policy_news",
                     source_url=MOTIR_INDEX, parser="motir_press_index_v1",
                     config={"max_items": 4, "article_limit": 1,
                             "issuer_ids": [], "channels": ["policy", "earnings"],
                             "relevance_reason": "基金此前的新兴市场及科技产业投向与韩国贸易、产业政策有关；"
                                                 "这里只提供相关环境，未量化对基金收益的影响。"}),
            ])
    return nav_jobs + disclosure_jobs + context_jobs


def merge_current_source_plan(public_subjects: Sequence[Mapping],
                              extra_plan: Sequence[Mapping] | None = None) -> list[dict]:
    """Defaults plus explicitly configured duties, without changing collect's opt-out.

    For each product, NAV and the announcement index are distinct duties. Context
    parsers are also distinct (an issuer feed does not replace policy news).
    A configured same-parser duty replaces its default; new duties append. Jobs
    for identities outside this evaluation are ignored. Passing [] here means
    defaults only, whereas collect_current(source_plan=[]) still requests nothing.
    """
    subjects = _subjects(public_subjects)
    identifiers = {row["product_id"] for row in subjects}
    extra = [dict(row) for row in extra_plan or [] if row.get("product_id") in identifiers]

    def duty(job):
        parser = job.get("parser")
        role = "nav" if job.get("family") == "nav" else parser
        return job["product_id"], job.get("family"), role

    replaced = {duty(row) for row in extra}
    jobs = [row for row in default_current_source_plan(subjects) if duty(row) not in replaced] + extra
    seen, result = set(), []
    for row in jobs:
        marker = (row["product_id"], row.get("family"), row.get("parser"), row.get("source_url"))
        if marker not in seen:
            result.append(row)
            seen.add(marker)
    return result


def _text_html(value: str) -> str:
    return " ".join(_HTML(value).tokens)


def _public_link(base: str, href: str) -> str:
    parts = urlsplit(urljoin(base, href))
    # Lists advertise some public Sina links as HTTP; request the qualified HTTPS route once.
    url = urlunsplit(("https" if parts.scheme == "http" else parts.scheme,
                      parts.netloc, parts.path, parts.query, ""))
    return _url(url)


def _publication(value: str | None, observed: str) -> tuple[str | None, str]:
    if not value:
        return None, "unknown"
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = date.fromisoformat(value)
        if day > timestamp(observed).date():
            raise ValidationError("source publication date is later than this observation")
        return day.isoformat(), "date"
    try:
        published = timestamp(value)
    except (ValueError, TypeError):
        published = timestamp(parsedate_to_datetime(value))
    if published > timestamp(observed):
        raise ValidationError("source publication timestamp is later than this observation")
    return published.isoformat(), "timestamp"


def _message(job: Mapping, *, url: str, title: str, published: str | None,
             excerpt: str = "", content_status: str = "title_only", observed: str) -> dict:
    publication, precision = _publication(published, observed)
    config = job.get("config", {})
    return {
        "id": digest({"source_url": url, "title": title, "published_at": publication}),
        "title": title[:500], "source_url": url, "index_url": job["source_url"],
        "published_at": publication, "publication_precision": precision,
        "excerpt": excerpt[:2400], "content_status": content_status,
        "issuer_ids": list(config.get("issuer_ids", [])),
        "channels": list(config.get("channels", ["fund_disclosure"])),
        "relevance_reason": config.get("relevance_reason", "该基金公开公告列表中的新条目。"),
        "event_status": "unresolved", "interpretation_status": "unreviewed",
        "observed_at": observed, "available_at": observed,
    }


def _announcements(response: FetchResponse, job: Mapping, observed: str) -> list[dict]:
    page = _HTML(_decode(response))
    symbol = job["subject"]["symbol"]
    if not any(symbol in value for value in page.tokens):
        raise ValidationError("fund announcement page does not identify the requested code")
    messages, seen = [], set()
    for index, link in enumerate(page.links):
        if not re.search(r"/vAkFundInfo_JJGKXX/q/\d+\.phtml", link["href"]):
            continue
        url = _public_link(job["source_url"], link["href"])
        if url in seen:
            continue
        seen.add(url)
        title = link["title"] or link["text"]
        next_start = page.links[index + 1]["start"] if index + 1 < len(page.links) else len(page.tokens)
        nearby = " ".join(page.tokens[link["start"]:min(next_start, link["end"] + 8)])
        match = _DATE.search(nearby)
        messages.append(_message(job, url=url, title=title,
                                 published=match.group() if match else None, observed=observed))
    return messages


def _rss(response: FetchResponse, job: Mapping, observed: str) -> list[dict]:
    root = ET.fromstring(response.body)
    messages = []
    for item in root.findall(".//item"):
        fields = {child.tag.split("}")[-1]: "".join(child.itertext()) for child in item}
        if not fields.get("title") or not fields.get("link"):
            continue
        description = _text_html(fields.get("description", ""))
        messages.append(_message(
            job, url=_public_link(job["source_url"], fields["link"]), title=fields["title"],
            published=fields.get("pubDate"), excerpt=description,
            content_status="publisher_summary" if description else "title_only", observed=observed,
        ))
    return messages


def _motir(response: FetchResponse, job: Mapping, observed: str) -> list[dict]:
    page = _HTML(_decode(response))
    messages = []
    for link in page.links:
        if not re.search(r"/article/EATCLdfa319ada/\d+/view", link["href"]):
            continue
        match = _DATE.search(link["text"])
        if not match:
            continue  # The homepage also has undated 'Read more' carousel links.
        title = link["text"][:match.start()].strip()
        messages.append(_message(
            job, url=_public_link(job["source_url"], link["href"]), title=title,
            published=match.group(), observed=observed,
        ))
    return messages


def _article_order(messages: list[dict], job: Mapping) -> list[dict]:
    if job["parser"] != "sina_fund_announcements_v1":
        return messages

    def priority(message):
        title = message["title"]
        if "季度报告" in title and "提示性公告" not in title:
            return 0
        if "基金产品资料概要" in title and "C类" in title:
            return 1
        if re.search(r"暂停|恢复|限额|申购|赎回", title):
            return 2
        if "中期报告" in title and "提示性公告" not in title:
            return 3
        return 4

    return sorted(messages, key=priority)


def collect_current(
    public_subjects: Sequence[Mapping], *, source_plan: Sequence[Mapping] | None = None,
    fetch: Fetch = fetch_public, clock: Callable[[], str] = now_iso,
    timeout: float = 10, request_budget: int = 20, blocked_urls: Sequence[str] = (),
) -> dict:
    """One evaluation's public facts and located RAM text, never saved here.

    All index/NAV jobs run before bounded linked-article reads. The same URL is
    requested once, including shared issuer feeds across related funds. Failures
    are returned independently; no retry, redirect or mirror restoration occurs.
    ``blocked_urls`` carries session/prior stops into explicit refreshes.
    """
    if not isinstance(request_budget, int) or isinstance(request_budget, bool) or request_budget < 0:
        raise ValidationError("request_budget must be a non-negative integer")
    subjects = _subjects(public_subjects)
    identities = {row["product_id"]: row for row in subjects}
    plan = default_current_source_plan(subjects) if source_plan is None else list(source_plan)
    started = timestamp(clock()).isoformat()
    result = {
        "schema_version": 1, "kind": "current_online_context", "status": "unavailable",
        "started_at": started, "checked_at": started, "evaluated_at": started,
        "requests_made": 0, "public_subjects": subjects, "source_checks": [],
        "nav_by_product": {key: [] for key in identities},
        "records_by_product": {key: [] for key in identities},
        "record_evidence_by_product": {key: {} for key in identities},
        "evidence_by_product": {key: [] for key in identities},
        "messages_by_product": {key: [] for key in identities}, "warnings": [],
        "stopped_urls": list(dict.fromkeys(blocked_urls)),
    }
    stopped = set(blocked_urls)
    stopped_hosts = {urlsplit(url).hostname for url in stopped}
    cache: dict[str, tuple[FetchResponse | None, dict]] = {}
    deferred = []

    def request(job: Mapping) -> tuple[FetchResponse | None, dict]:
        url = job["source_url"]
        check = {key: job[key] for key in ("job_id", "product_id", "family", "source_url")}
        check.update(status="error", parse_status="not_attempted", checked_at=timestamp(clock()).isoformat(),
                     observed_at=None, source_date=None, reason=None, request_made=False, http_status=None)
        result["source_checks"].append(check)
        if url in cache:
            response, original = cache[url]
            check.update({key: original[key] for key in ("status", "observed_at", "http_status", "reason")})
            check["reused_response"] = True
            return response, check
        try:
            _url(url)
        except ValidationError as exc:
            check.update(status="skipped_blocked", reason=str(exc))
            return None, check
        host = urlsplit(url).hostname
        if url in stopped or host in stopped_hosts:
            check.update(status="skipped_blocked" if url in stopped else "stopped",
                         reason="本会话或已有记录已停止该来源；未重新请求。")
            return None, check
        if result["requests_made"] >= request_budget:
            check.update(status="budget_exhausted", reason="本次在线请求额度已用完。")
            return None, check
        result["requests_made"] += 1
        check["request_made"] = True
        response = None
        try:
            response = fetch(url, timeout, MAX_BYTES)
            check.update(checked_at=timestamp(clock()).isoformat(), http_status=response.status_code)
            if response.status_code != 200:
                raise ValidationError(f"HTTP {response.status_code}; 未重试或跟随跳转。")
            if len(response.body) > MAX_BYTES:
                raise ValidationError("source response exceeds 4 MiB")
            check.update(status="ok", observed_at=check["checked_at"])
        except Exception as exc:
            check.update(status="error", checked_at=timestamp(clock()).isoformat(),
                         reason=f"{type(exc).__name__}: {exc}")
            response = None
            stopped.add(url)
            stopped_hosts.add(host)
        cache[url] = response, check
        return response, check

    def read_body(job, message, response, check):
        document, hrefs = read_public_document(
            response, product_id=job["product_id"], source_url=job["source_url"],
            title=message["title"], published_at=message["published_at"],
            publication_precision=message["publication_precision"],
            observed_at=check["observed_at"], fund=job["family"] == "disclosure",
        )
        available = timestamp(clock()).isoformat()
        document["available_at"] = available
        content_status = document["content_status"]
        if document["sections"]:
            identifier = job["product_id"]
            result["evidence_by_product"][identifier].append(document)
            excerpt = " ".join(" ".join(row["text"].split()) for row in document["sections"])
            message.update(excerpt=excerpt[:2400], content_status="online_excerpt",
                           evidence_id=document["id"], content_observed_at=check["observed_at"],
                           available_at=available)
            try:
                records, citations = records_from_document(document, job["subject"])
                completed = timestamp(clock()).isoformat()
                for record in records:
                    record["available_at"] = completed
                result["records_by_product"][identifier].extend(records)
                result["record_evidence_by_product"][identifier].update(citations)
                check.update(record_count=len(records), record_parse_status="parsed" if records else "not_supported")
            except ValidationError as exc:
                # Readable text is still useful when its table cannot be normalized.
                check.update(record_count=0, record_parse_status="unparsed", record_reason=str(exc))
        elif message["content_status"] == "title_only":
            message["content_status"] = content_status
        message["body_status"] = content_status
        check.update(parse_status=content_status, source_date=message["published_at"],
                     evidence_id=document["id"] if document["sections"] else None,
                     reason=None if document["sections"] else
                     "已读取页面，但未取得可解析的公告/正文；未推断持仓或交易条件。")
        attachments = []
        for href in hrefs:
            try:
                attachments.append(_public_link(job["source_url"], href))
            except ValidationError:
                continue  # Advertised but not an eligible public, unstopped HTTPS route.
        if attachments:
            message["attachment_urls"] = attachments
        return attachments if job["family"] == "disclosure" or not document["sections"] else []

    for raw_job in plan:
        job = dict(raw_job)
        if job.get("product_id") not in identities:
            raise ValidationError("every current source job must name a passed public identity")
        job.setdefault("config", {})
        if job.get("parser") not in PARSERS:
            result["source_checks"].append({
                **{key: job.get(key) for key in ("job_id", "product_id", "family", "source_url")},
                "status": "unsupported", "parse_status": "not_attempted",
                "checked_at": timestamp(clock()).isoformat(), "observed_at": None,
                "source_date": None, "reason": "当前内存解析器不支持该来源配置。", "request_made": False,
                "http_status": None,
            })
            continue
        job["subject"] = identities[job["product_id"]]
        response, check = request(job)
        if response is None:
            continue
        try:
            observed = check["observed_at"]
            config = job["config"]
            if job["parser"] == "eastmoney_nav_js_v1":
                days = int(config.get("lookback_days", 400))
                if not 1 <= days <= 3650:
                    raise ValidationError("current NAV lookback_days must be between 1 and 3650")
                today = timestamp(observed).date()
                window = {"start": (today - timedelta(days=days)).isoformat(), "end": today.isoformat()}
                rows = _nav(response.body, job["subject"], window, observed)
                available = timestamp(clock()).isoformat()
                projected = [dict(row, nav_type="final_unit_nav_candidate", source="eastmoney_js_json",
                                  source_url=job["source_url"], observed_at=observed, available_at=available)
                             for row in rows]
                result["nav_by_product"][job["product_id"]].extend(projected)
                check.update(parse_status="parsed", source_date=rows[-1]["date"],
                             record_count=len(rows), requested_window=window)
            elif job["parser"] == "public_document_v1":
                published, _ = _publication(config.get("published_at"), observed)
                message = _message(job, url=job["source_url"], title=config.get("title", "公开资料"),
                                   published=published, observed=observed)
                result["messages_by_product"][job["product_id"]].append(message)
                attachments = read_body(job, message, response, check)
                for attachment in attachments[:2]:
                    deferred.append((dict(job, source_url=attachment, parser="public_document_v1"),
                                     message, True))
            else:
                parser = {"sina_fund_announcements_v1": _announcements, "nvidia_rss_v1": _rss,
                          "motir_press_index_v1": _motir}[job["parser"]]
                messages = parser(response, job, observed)
                maximum = max(1, min(int(config.get("max_items", 5)), 20))
                messages = list({row["source_url"]: row for row in messages}.values())
                messages = _article_order(messages, job)[:maximum]
                if not messages:
                    raise ValidationError("no dated/current items parsed from the changing list")
                available = timestamp(clock()).isoformat()
                for row in messages:
                    row["available_at"] = available
                result["messages_by_product"][job["product_id"]].extend(messages)
                check.update(parse_status="publisher_summary" if job["parser"] == "nvidia_rss_v1" else "title_only",
                             source_date=max((row["published_at"] for row in messages if row["published_at"]),
                                             default=None), message_count=len(messages))
                for row in messages[:max(0, min(int(config.get("article_limit", 1)), 3))]:
                    deferred.append((job, row, False))
        except Exception as exc:
            check.update(status="error", parse_status="failed", reason=f"{type(exc).__name__}: {exc}")
            stopped.add(job["source_url"])
            stopped_hosts.add(urlsplit(job["source_url"]).hostname)

    attachments = []
    for queue in (deferred, attachments):
        for parent, message, is_attachment in queue:
            url = parent["source_url"] if is_attachment else message["source_url"]
            job = dict(parent, job_id=parent["job_id"] + ":article:" + message["id"][:12],
                       source_url=url)
            response, check = request(job)
            if response is None:
                message["attachment_status" if is_attachment else "body_status"] = check["status"]
                continue
            try:
                links = read_body(job, message, response, check)
                if is_attachment:
                    message["attachment_status"] = check["parse_status"]
                else:
                    for link in links[:2]:
                        if link != job["source_url"]:
                            attachments.append((dict(job, source_url=link), message, True))
            except Exception as exc:
                check.update(status="error", parse_status="failed", reason=f"{type(exc).__name__}: {exc}")
                message["attachment_status" if is_attachment else "body_status"] = "error"
                stopped.add(job["source_url"])

    for identifier in identities:
        rows = result["nav_by_product"][identifier]
        # Keep each source observation: date-only deduplication would erase a
        # disagreement before the current view or local calculation can see it.
        result["nav_by_product"][identifier] = sorted(
            rows, key=lambda row: (row["date"], row["source_url"]),
        )
        messages = result["messages_by_product"][identifier]
        result["messages_by_product"][identifier] = list({row["id"]: row for row in messages}.values())
        for key in ("records_by_product", "evidence_by_product"):
            result[key][identifier] = list({row["id"]: row for row in result[key][identifier]}.values())
    checks = result["source_checks"]
    usable = any(result["nav_by_product"].values()) or any(result["messages_by_product"].values())
    incomplete = any(check["status"] != "ok" or check["parse_status"] in {
        "title_only", "unparsed_document", "unparsed_pdf", "failed",
    } for check in checks)
    result["status"] = ("partial" if incomplete else "ready") if usable else "unavailable"
    result["stopped_urls"] = sorted(stopped)
    result["checked_at"] = result["evaluated_at"] = timestamp(clock()).isoformat()
    if incomplete:
        result["warnings"].append("部分在线来源未读到正文或未取得数据；公告标题不等于已更新持仓、费用或申赎条件。")
    if not plan:
        result["warnings"].append("本次未配置当前在线来源。")
    return result

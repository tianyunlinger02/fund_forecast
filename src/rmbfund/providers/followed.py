"""Explicit, finite refresh jobs for *public research samples*, not user holdings.

No Store, account, crawler, implicit source discovery, retry, or fallback is used.
Each response gets a new directory and a receipt. Capturing a page/PDF is NOT
structured disclosure success. Reviewed extracts bind literal UTF-8 positions to
an exact source hash; this module deliberately has no general PDF/holdings parser.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

from ..domain import ValidationError, now_iso, timestamp
from .akshare import _js_millis_date, _number, _parse_profile_networth

FAMILIES = frozenset({"nav", "disclosure", "policy_news"})
PARSERS = frozenset({"eastmoney_nav_js_v1", "reviewed_extract_v1", "capture_only_v1",
                     "reviewed_pdf_extract_v1"})
MAX_REQUESTS = 20
MAX_BYTES = 8 * 1024 * 1024
MAX_PLAN_BYTES = 512 * 1024
MAX_ROWS = 20_000
# Research stop targets are never recovered through this layer. Other stopped
# exact routes can be supplied in blocked_urls; redirects are never followed.
STOP_HOSTS = frozenset({
    "ccbfund.cn", "fundf10.eastmoney.com", "akshare.akfamily.xyz", "msci.com",
    "cfainstitute.org", "nber.org", "openapi.twse.com.tw", "csrc.gov.cn",
    "sse.com.cn", "fred.stlouisfed.org", "ecb.europa.eu",
})
_SECRET_KEYS = frozenset({"token", "access_token", "auth", "authorization", "cookie",
                          "password", "passwd", "session", "sessionid", "signature",
                          "api_key", "apikey", "account", "credential"})
_SUBJECT_KEYS = frozenset({"product_id", "symbol", "name", "share_class", "currency", "public"})
_JOB_KEYS = frozenset({"job_id", "product_id", "family", "source_url", "parser", "config"})


@dataclass(frozen=True)
class FetchResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


Fetch = Callable[[str, float, int], FetchResponse]


def fetch_public(url: str, timeout: float, max_bytes: int) -> FetchResponse:
    """Exactly one anonymous GET, verified TLS, no redirects, auth, or env proxy."""
    with httpx.stream(
        "GET", url, timeout=timeout, follow_redirects=False, trust_env=False,
        headers={"User-Agent": "rmbfund-followed-public/0.1", "Accept-Encoding": "identity"},
    ) as response:
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise ValidationError("response exceeds the explicit byte budget")
        return FetchResponse(response.status_code, bytes(body), dict(response.headers))


def _json(value: Any, maximum: int = MAX_PLAN_BYTES) -> bytes:
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         indent=2).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError("public plan must be finite JSON data") from exc
    if len(raw) > maximum:
        raise ValidationError("public plan/record exceeds its byte budget")
    return raw


def _text(value: Any, label: str, maximum: int = 200) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(char) < 32 for char in value)):
        raise ValidationError(f"invalid {label}")
    return value


def _day(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValidationError("date must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValidationError("invalid calendar date") from exc


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError("expected a SHA-256 digest")
    return value


def _url(value: Any) -> str:
    url = _text(value, "public source URL", 4000)
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        invalid = (parts.scheme != "https" or parts.port not in (None, 443)
                   or parts.username is not None or parts.password is not None
                   or parts.fragment or not host or "." not in host
                   or host.endswith((".local", ".localhost", ".internal", ".test", ".invalid")))
    except ValueError as exc:
        raise ValidationError("invalid public HTTPS URL") from exc
    try:
        ipaddress.ip_address(host)
        invalid = True  # Public identities use explicit publisher DNS names, not IPs.
    except ValueError:
        pass
    if (invalid or any(key.lower() in _SECRET_KEYS for key, _ in parse_qsl(parts.query))
            or any(host == stop or host.endswith("." + stop) for stop in STOP_HOSTS)):
        raise ValidationError("source must be public, credential-free, and not a stopped origin")
    return url


def _validate(
    subjects: Sequence[Mapping], families: Sequence[str], source_plan: Sequence[Mapping],
    request_budget: int, previous_hashes: Mapping[str, str], blocked_urls: Sequence[str],
) -> tuple[dict[str, dict], list[str], list[dict], set[str]]:
    if (not isinstance(request_budget, int) or isinstance(request_budget, bool)
            or not 0 <= request_budget <= MAX_REQUESTS):
        raise ValidationError("request_budget must be an integer from 0 to 20")
    if (not isinstance(subjects, (list, tuple)) or not 1 <= len(subjects) <= 32
            or not isinstance(families, (list, tuple)) or not families
            or any(not isinstance(item, str) or item not in FAMILIES for item in families)
            or len(set(families)) != len(families)):
        raise ValidationError("pass 1..32 public identities and explicit unique families")
    identities = {}
    for subject in subjects:
        if (not isinstance(subject, Mapping) or set(subject) != _SUBJECT_KEYS
                or subject.get("public") is not True):
            raise ValidationError("only the explicit public identity fields are accepted")
        item = dict(subject)
        identifier = _text(item["product_id"], "product_id", 80)
        if identifier in identities:
            raise ValidationError("duplicate public identity")
        if not isinstance(item["symbol"], str) or not re.fullmatch(r"\d{6}", item["symbol"]):
            raise ValidationError("public fund symbols must have six digits")
        _text(item["name"], "public fund name")
        if item["share_class"] not in ("A", "C", "") or item["currency"] != "CNY":
            raise ValidationError("only explicit CNY A/C or unclassified fund identities supported")
        identities[identifier] = item
    if not isinstance(source_plan, (list, tuple)) or len(source_plan) > 96:
        raise ValidationError("source_plan must contain at most 96 explicit jobs")
    _json(source_plan)
    jobs, seen = [], set()
    for job in source_plan:
        if not isinstance(job, Mapping) or set(job) != _JOB_KEYS:
            raise ValidationError("job fields must be job_id/product_id/family/source_url/parser/config")
        item = dict(job)
        identifier = _text(item["job_id"], "job_id", 80)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier) or identifier in seen:
            raise ValidationError("job_id must be a unique safe directory name")
        seen.add(identifier)
        if item["product_id"] not in identities or item["family"] not in FAMILIES:
            raise ValidationError("every job must name a passed public identity and supported family")
        _url(item["source_url"])
        _text(item["parser"], "parser")
        if not isinstance(item["config"], Mapping):
            raise ValidationError("parser config must be explicit JSON data")
        config = item["config"]
        if item["parser"] == "eastmoney_nav_js_v1":
            if item["family"] != "nav" or set(config) != {"start", "end"}:
                raise ValidationError("NAV jobs need the NAV family and explicit start/end only")
            if _day(config["start"]) > _day(config["end"]):
                raise ValidationError("NAV window is reversed")
        elif item["parser"] == "capture_only_v1" and config:
            raise ValidationError("capture-only jobs take no hidden config/input")
        elif item["parser"] in {"reviewed_extract_v1", "reviewed_pdf_extract_v1"}:
            required = {"reviewed", "reviewed_at", "expected_sha256", "extracts"}
            if item["parser"] == "reviewed_pdf_extract_v1":
                required |= {"review_text", "review_text_sha256"}
            if item["family"] == "nav" or set(config) != required:
                raise ValidationError("reviewed excerpt jobs need explicit disclosure/news definitions")
            if config["reviewed"] is not True:
                raise ValidationError("reviewed excerpt jobs require reviewed=True")
            timestamp(config["reviewed_at"])
            _hash(config["expected_sha256"])
        pending = [config]
        while pending:
            node = pending.pop()
            if isinstance(node, Mapping):
                if any(str(key).lower() in _SECRET_KEYS for key in node):
                    raise ValidationError("public source plans must not contain private/credential fields")
                pending.extend(node.values())
            elif isinstance(node, (list, tuple)):
                pending.extend(node)
        jobs.append(item)
    if (not isinstance(previous_hashes, Mapping) or set(previous_hashes) - seen
            or not isinstance(blocked_urls, (list, tuple)) or len(blocked_urls) > 256):
        raise ValidationError("previous hashes must be keyed by planned job_id")
    for value in previous_hashes.values():
        _hash(value)
    stopped = {_text(value, "stopped URL", 4000) for value in blocked_urls}
    return identities, list(families), jobs, stopped


def _nav(raw: bytes, subject: dict, config: Mapping, completed: str) -> list[dict]:
    if set(config) != {"start", "end"}:
        raise ValidationError("NAV config needs only an explicit start/end window")
    start, end = _day(config["start"]), _day(config["end"])
    if not start <= end <= timestamp(completed).date().isoformat():
        raise ValidationError("NAV window is reversed or later than the real capture clock")
    if len(raw) > 2 * 1024 * 1024:
        raise ValidationError("NAV JS exceeds 2 MiB")
    text = raw.decode("utf-8-sig", "strict")
    if (len(re.findall(r"\bfS_code\s*=", text)) != 1
            or len(re.findall(r"\bData_netWorthTrend\s*=", text)) != 1):
        raise ValidationError("NAV identity/assignment is missing or ambiguous")
    rows = _parse_profile_networth(text, expected_symbol=subject["symbol"])
    if len(rows) > MAX_ROWS:
        raise ValidationError("NAV source exceeds 20000 rows")
    result, seen = [], set()
    for index, row in enumerate(rows):
        day = _js_millis_date(row["x"])
        if day in seen or day > timestamp(completed).date().isoformat():
            raise ValidationError("NAV source has duplicate or future dates")
        seen.add(day)
        value = _number(row["y"], "published unit NAV", positive=True)
        if start <= day <= end:
            record = {
                "date": day, "value": value, "currency": subject["currency"],
                "share_class": subject["share_class"], "basis": "unit_nav",
                "published_at": None, "publication_precision": "unknown",
                "locator": f"assignment:Data_netWorthTrend;index:{index};fields:x,y",
            }
            if "unitMoney" in row:
                record["unitMoney"] = row["unitMoney"]
            result.append(record)
    if not result:
        raise ValidationError("no published NAV observations in the explicit window")
    return sorted(result, key=lambda row: row["date"])


def _reviewed(
    raw: bytes, config: Mapping, completed: str, *, pdf_review: bool = False,
) -> list[dict]:
    required = {"reviewed", "reviewed_at", "expected_sha256", "extracts"}
    if pdf_review:
        required |= {"review_text", "review_text_sha256"}
    if set(config) != required or config.get("reviewed") is not True:
        raise ValidationError("reviewed extracts need an explicit attestation and pinned source hash")
    if _hash(config["expected_sha256"]) != hashlib.sha256(raw).hexdigest():
        raise ValidationError("source changed; pinned reviewed extracts no longer apply")
    if timestamp(config["reviewed_at"]) > timestamp(completed):
        raise ValidationError("reviewed_at cannot be later than the real completion clock")
    if pdf_review:
        # This is an explicit human/maintainer attestation, exactly like the
        # existing reviewed-document importer, NOT a claim of PDF text parsing.
        if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-2048:]:
            raise ValidationError("reviewed PDF requires an intact retained PDF")
        text = config["review_text"]
        if (not isinstance(text, str) or not 1 <= len(text.encode("utf-8")) <= 64 * 1024
                or _hash(config["review_text_sha256"]) != hashlib.sha256(text.encode()).hexdigest()):
            raise ValidationError("reviewed PDF text must be an explicit bounded hash-pinned attestation")
    else:
        if raw.startswith(b"%PDF-"):
            raise ValidationError("PDF needs an explicit reviewed_pdf_extract_v1 attestation")
        text = raw.decode("utf-8-sig", "strict")
    extracts = config["extracts"]
    if not isinstance(extracts, list) or not 1 <= len(extracts) <= 32:
        raise ValidationError("reviewed extracts must contain 1..32 literal excerpts")
    result = []
    for item in extracts:
        if (not isinstance(item, Mapping) or set(item) != {
            "start", "end", "quote", "locator", "report_date", "published_at", "payload",
        }):
            raise ValidationError("reviewed extract fields are incomplete or unsupported")
        low, high = item["start"], item["end"]
        if (not isinstance(low, int) or isinstance(low, bool) or not isinstance(high, int)
                or isinstance(high, bool) or not 0 <= low < high <= len(text)
                or high - low > 6000 or text[low:high] != item["quote"]):
            raise ValidationError("literal UTF-8 decoded character positions/quote do not match")
        _text(item["locator"], "extract locator", 1000)
        report = _day(item["report_date"])
        published = item["published_at"]
        if (report > timestamp(completed).date().isoformat() or
                published is not None and _day(published) > timestamp(completed).date().isoformat()):
            raise ValidationError("extract report/publication date is in the future")
        if not isinstance(item["payload"], Mapping):
            raise ValidationError("reviewed payload must be an explicit object")
        _json(item["payload"], 64 * 1024)
        result.append(dict(item, publication_precision="date" if published else "unknown",
                           validation="hash_pinned_literal_excerpt_not_automatic_semantic_parsing"))
    return result


def _review_import_record(record: Mapping) -> dict:
    item = dict(record)
    payload = record.get("payload")
    if (isinstance(payload, Mapping) and set(payload) == {"family", "data", "evidence_refs"}
            and isinstance(payload["family"], str) and isinstance(payload["data"], Mapping)):
        item["evidence_import"] = {
            "family": payload["family"], "report_date": record["report_date"],
            "published_at": record["published_at"],
            "publication_precision": record["publication_precision"],
            "payload": dict(payload["data"]), "locator": record["locator"],
        }
    return item


def review_followed_capture(
    subject: Mapping, capture: Mapping, raw_bytes: bytes, *, parser: str,
    config: Mapping, clock: Callable[[], str] = now_iso,
) -> dict:
    """Review already-captured public bytes, with zero requests and no clock renewal.

    PDF extracted text is explicitly attested by the reviewer, not derived or
    authenticated by this adapter. A changed raw hash fails closed. The original
    observed_at remains untouched; availability is the actual review completion.
    This function returns inert records and never imports them into a Store.
    """
    if parser not in {"reviewed_extract_v1", "reviewed_pdf_extract_v1"}:
        raise ValidationError("retained capture review needs an explicit reviewed adapter")
    if not isinstance(capture, Mapping) or capture.get("status") not in {
        "captured_unparsed", "ok",
    } or capture.get("http_status") != 200:
        raise ValidationError("only successfully retained public HTTP 200 captures can be reviewed")
    job = {"job_id": capture.get("job_id"), "product_id": capture.get("product_id"),
           "family": capture.get("family"), "source_url": capture.get("source_url"),
           "parser": parser, "config": config}
    _validate([subject], [capture.get("family")], [job], 0, {}, ())
    if (not isinstance(raw_bytes, bytes) or not raw_bytes or len(raw_bytes) > MAX_BYTES
            or hashlib.sha256(raw_bytes).hexdigest() != _hash(capture.get("raw_sha256"))):
        raise ValidationError("retained bytes do not match the original capture hash")
    observed = timestamp(capture.get("observed_at")).isoformat()
    started = timestamp(clock()).isoformat()
    if timestamp(started) < timestamp(observed):
        raise ValidationError("review clock precedes the original public capture")
    records = _reviewed(raw_bytes, config, started, pdf_review=parser == "reviewed_pdf_extract_v1")
    completed = timestamp(clock()).isoformat()
    if timestamp(completed) < timestamp(started):
        raise ValidationError("review completion clock precedes review start")
    provenance = {"product_id": subject["product_id"], "job_id": capture["job_id"],
                  "source_url": capture["source_url"], "raw_sha256": capture["raw_sha256"],
                  "observed_at": observed, "available_at": completed, "reviewed_at": completed}
    reviewed_records = [_review_import_record({**record, **provenance}) for record in records]
    return {"status": "ok", "parse_status": "reviewed", "family": capture["family"],
            **provenance, "request_count": 0, "raw_path": capture.get("raw_path"),
            "retrieval": "retained_public_capture_review_not_live_fetch",
            "original_receipt_path": capture.get("receipt_path"), "records": reviewed_records,
            "warnings": ["Explicit reviewer attestation, not publisher authentication or automatic semantic parsing."]}


def refresh_followed(
    subjects: Sequence[Mapping], families: Sequence[str], source_plan: Sequence[Mapping],
    request_budget: int, output_dir: str | Path, *, fetch: Fetch = fetch_public,
    clock: Callable[[], str] = now_iso, previous_hashes: Mapping[str, str] | None = None,
    blocked_urls: Sequence[str] = (), timeout: float = 20, max_bytes: int = MAX_BYTES,
) -> dict:
    """Run only passed public jobs; return independent per-job and family outcomes.

    ``fetch(url, timeout, max_bytes) -> FetchResponse`` is injectable. ``clock``
    must return real timezone-aware timestamps in production; historical report
    dates are never used as collection/availability clocks. A fresh output_dir is
    required. Unsupported adapters and unconfigured families cost no request.
    Failed targets stop without retry; denials/network failures stop that host
    for remaining jobs. Independent hosts/families continue within the budget.
    """
    previous = dict(previous_hashes or {})
    identities, selected, jobs, stopped_urls = _validate(
        subjects, families, source_plan, request_budget, previous, blocked_urls,
    )
    if (not isinstance(timeout, (float, int)) or isinstance(timeout, bool) or not 0 < timeout <= 60
            or not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or not 1 <= max_bytes <= MAX_BYTES):
        raise ValidationError("timeout/response budget exceeds the supported bounds")
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=False)
    started = timestamp(clock()).isoformat()
    (root / "manifest.json").write_bytes(_json({
        "subjects": list(identities.values()), "families": selected, "source_plan": jobs,
        "request_budget": request_budget, "previous_hashes": previous,
        "blocked_urls": sorted(stopped_urls), "started_at": started,
        "purpose": "public_research_samples_not_personal_holdings",
    }))
    outcomes, requests, stopped_hosts = [], 0, set()
    for job in jobs:
        if job["family"] not in selected:
            continue
        item = {key: job[key] for key in ("job_id", "product_id", "family", "source_url", "parser")}
        item.update(records=[], status="unsupported", parse_status="not_run", request_count=0,
                    previous_sha256=previous.get(job["job_id"]), raw_sha256=None, changed=None)
        directory = root / job["job_id"]
        directory.mkdir(exist_ok=False)
        host = urlsplit(job["source_url"]).hostname
        receipt = {**item, "attempt": 0, "retry_count": 0, "credential_use": False,
                   "redirects_followed": False, "started_at": timestamp(clock()).isoformat()}
        body = None
        if job["parser"] not in PARSERS:
            item["reason"] = "unsupported adapter; no request made"
        elif job["source_url"] in stopped_urls or host in stopped_hosts:
            item.update(status="stopped", reason="target/host stopped; no retry or recovery")
        elif requests >= request_budget:
            item.update(status="budget_exhausted", reason="explicit request budget exhausted")
        else:
            requests += 1
            item["request_count"] = 1
            receipt["attempt"] = 1
            stopped_urls.add(job["source_url"])  # never request the same target twice in this run
            try:
                response = fetch(job["source_url"], float(timeout), max_bytes)
                completed = timestamp(clock()).isoformat()
                if timestamp(completed) < timestamp(receipt["started_at"]):
                    raise ValidationError("capture completion clock precedes request start")
                if (not isinstance(response, FetchResponse) or not isinstance(response.body, bytes)
                        or len(response.body) > max_bytes or not isinstance(response.status_code, int)
                        or isinstance(response.status_code, bool) or not 100 <= response.status_code <= 599):
                    raise ValidationError("injected fetch returned an invalid or oversized response")
                body = response.body
                (directory / "response.bin").write_bytes(body)
                sha = hashlib.sha256(body).hexdigest()
                item.update(raw_sha256=sha, changed=None if item["previous_sha256"] is None
                            else sha != item["previous_sha256"], observed_at=completed,
                            raw_path=str(directory / "response.bin"), http_status=response.status_code)
                receipt.update(response_completed_at=completed, http_status=response.status_code,
                               bytes=len(body), raw_sha256=sha,
                               content_type=response.headers.get("content-type"),
                               last_modified=response.headers.get("last-modified"),
                               location=response.headers.get("location"))
                if response.status_code != 200:
                    item.update(status="error", reason=f"HTTP {response.status_code}; no retry")
                    if response.status_code in (401, 403, 429):
                        stopped_hosts.add(host)
                elif job["parser"] == "capture_only_v1":
                    item.update(status="captured_unparsed", parse_status="not_supported",
                                reason="raw capture only; not structured NAV/holdings/news success")
                else:
                    if job["parser"] == "eastmoney_nav_js_v1":
                        if job["family"] != "nav":
                            raise ValidationError("NAV parser cannot establish another family")
                        records = _nav(body, identities[job["product_id"]], job["config"], completed)
                        parse_status = "parsed"
                    else:
                        records = _reviewed(
                            body, job["config"], completed,
                            pdf_review=job["parser"] == "reviewed_pdf_extract_v1",
                        )
                        parse_status = "reviewed"
                    available = timestamp(clock()).isoformat()
                    if timestamp(available) < timestamp(completed):
                        raise ValidationError("parse completion clock precedes response completion")
                    provenance = {"source_url": job["source_url"], "raw_sha256": sha,
                                  "observed_at": completed, "available_at": available,
                                  "product_id": job["product_id"], "job_id": job["job_id"]}
                    item.update(status="ok", parse_status=parse_status,
                                records=[_review_import_record({**record, **provenance})
                                         if parse_status == "reviewed" else {**record, **provenance}
                                         for record in records],
                                available_at=available)
            except Exception as exc:
                item.update(status="error", parse_status="error" if body is not None else "not_run",
                            reason=f"{type(exc).__name__}: {str(exc)[:1000]}")
                if body is None:
                    stopped_hosts.add(host)
        receipt.update(status=item["status"], parse_status=item["parse_status"],
                       request_count=item["request_count"], raw_sha256=item["raw_sha256"],
                       changed=item["changed"], raw_path=item.get("raw_path"),
                       record_count=len(item["records"]),
                       reason=item.get("reason"), completed_at=timestamp(clock()).isoformat())
        item["receipt_path"] = str(directory / "receipt.json")
        (directory / "receipt.json").write_bytes(_json(receipt))
        (directory / "outcome.json").write_bytes(_json(item, 16 * 1024 * 1024))
        outcomes.append(item)
    for identifier in identities:
        for family in selected:
            if not any(item["product_id"] == identifier and item["family"] == family for item in outcomes):
                outcomes.append({"product_id": identifier, "family": family,
                                 "status": "not_configured", "request_count": 0, "records": []})
    finished = timestamp(clock()).isoformat()
    if timestamp(finished) < timestamp(started):
        raise ValidationError("run completion clock precedes start")
    coverage = []
    for identifier in identities:
        for family in selected:
            group = [item for item in outcomes if item["product_id"] == identifier
                     and item["family"] == family]
            statuses = [item["status"] for item in group]
            successful = sum(status == "ok" for status in statuses)
            status = ("ok" if successful == len(group) else "partial" if successful
                      else "captured_unparsed" if "captured_unparsed" in statuses
                      else statuses[0] if len(set(statuses)) == 1 else "incomplete")
            coverage.append({"product_id": identifier, "family": family, "status": status,
                             "structured_jobs": successful, "job_outcomes": statuses,
                             "record_count": sum(len(item["records"]) for item in group),
                             "requests_made": sum(item["request_count"] for item in group)})
    result = {"purpose": "public_research_samples_not_personal_holdings", "started_at": started,
              "completed_at": finished, "request_budget": request_budget, "requests_made": requests,
              "implicit_retries": 0, "implicit_fallbacks": 0, "outcomes": outcomes,
              "family_outcomes": coverage}
    (root / "result.json").write_bytes(_json(result, 16 * 1024 * 1024))
    return result

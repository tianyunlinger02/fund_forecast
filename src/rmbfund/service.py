"""Shared application services for the CLI and the local UI."""

from __future__ import annotations

import json
import re
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from .domain import SHANGHAI, Product, Profile, ValidationError, canonical_json, decimal, digest, now_iso, timestamp
from .reports import csv_text
from .storage import Store

LINKS = [
    {"name": "证监会公告", "url": "https://www.csrc.gov.cn/csrc/c101954/common_list.shtml",
     "description": "免费官方入口；列表尚不代表自动完整覆盖"},
    {"name": "上交所基金公告", "url": "https://www.sse.com.cn/disclosure/fund/announcement/index.shtml",
     "description": "按代码核验上交所产品公告"},
    {"name": "巨潮基金/全文公告", "url": "https://www.cninfo.com.cn/new/fulltextSearch",
     "description": "公开公告检索入口，不含可能收费的批量API"},
    {"name": "上交所休市安排", "url": "https://www.sse.com.cn/disclosure/dealinstruc/closed/",
     "description": "人工核验目标年度交易日历，不能以工作日替代"},
    {"name": "中国人民银行", "url": "https://www.pbc.gov.cn/",
     "description": "货币政策原文；不自动解释为买卖信号"},
    {"name": "国家统计局", "url": "https://www.stats.gov.cn/sj/",
     "description": "宏观统计原始发布，注意发布日与统计期不同"},
]


def _safe_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "demo"}:
        raise ValidationError("来源链接只接受 http/https，演示可用 demo://")
    return url


def _snapshot_nav_rows(
    payload: Any, snapshot: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """Recognize only documented JSON NAV shapes, never execute raw source text."""
    if not isinstance(payload, list) or len(payload) > 50_000:
        raise ValidationError("净值快照必须是最多50000行的JSON对象列表")
    for wrapper in payload:
        if not isinstance(wrapper, dict):
            continue
        record = wrapper.get("record", wrapper)
        if not isinstance(record, dict):
            continue
        source = str(wrapper.get("source", snapshot["source"]))
        source_url = str(wrapper.get("source_url", snapshot["source_url"]))
        function = str(wrapper.get("function", ""))
        try:
            if function == "Data_netWorthTrend" or ("x" in record and "y" in record):
                value = record.get("x")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not 631_123_200_000 <= value <= 4_102_444_800_000 or value != int(value):
                    continue
                day = datetime.fromtimestamp(value / 1000, tz=SHANGHAI).date().isoformat()
                nav = decimal(record["y"])
            elif "净值日期" in record and "单位净值" in record:
                day = date.fromisoformat(str(record["净值日期"])[:10]).isoformat()
                nav = decimal(record["单位净值"])
            elif "unit_nav" in record and ("event_date" in record or "date" in record):
                day = date.fromisoformat(str(record.get("event_date", record.get("date")))).isoformat()
                nav = decimal(record["unit_nav"])
            elif (isinstance(record.get("payload"), dict) and "unit_nav" in record["payload"]
                  and "event_date" in record):
                day = date.fromisoformat(str(record["event_date"])).isoformat()
                nav = decimal(record["payload"]["unit_nav"])
            else:
                continue
            if nav <= 0:
                continue
            raw_time = timestamp(str(wrapper.get("observed_at") or snapshot["observed_at"]))
            if raw_time > timestamp(str(snapshot["observed_at"])):
                continue  # internally inconsistent/future raw evidence is not usable
            captured = raw_time
        except (ValueError, TypeError, KeyError, OverflowError, OSError, ValidationError):
            continue
        marker = record.get("unitMoney")
        marker_text = canonical_json(marker)
        yield {"event_date": day, "unit_nav": format(nav, "f"), "source": source,
               "source_url": source_url, "function": function,
               "observed_at": captured.isoformat(), "snapshot_id": snapshot["id"],
               "unit_money_present": "unitMoney" in record,
               "unitMoney": marker if len(marker_text) <= 2048 else {"content_hash": digest(marker)},
               "nonempty": "unitMoney" in record and marker not in (None, ""),
               "record_hash": digest(record)}


class Workspace:
    def __init__(
        self, root: str | Path = "data", mode: str = "demo", *, read_only: bool = False,
        initialize: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.mode = mode
        self.store = Store(self.root / f"{mode}.sqlite3", mode, read_only=read_only,
                           initialize=initialize)
        from .paper.engine import PaperEngine
        self.paper = PaperEngine(self.store)

    def initialize_demo(self) -> dict[str, Any]:
        from .demo import initialize
        return initialize(self.store)

    def add_product(
        self, symbol: str, name: str, kind: str, exchange: str = "",
        category: str = "未分类", share_class: str = "",
    ) -> str:
        symbol = symbol.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,30}", symbol):
            raise ValidationError("代码仅接受字母、数字、点、短横线和下划线")
        product_id = f"{kind}:{symbol}:{share_class.strip()}"
        self.store.add_product(Product(
            product_id, symbol, name.strip(), kind, self.mode == "demo",
            exchange.strip(), category.strip(), share_class.strip(),
        ))
        return product_id

    def products(self) -> list[dict[str, Any]]:
        """Public identities only; no monitoring, account or ownership projection."""
        return [asdict(product) for product in self.store.list_products()]

    def resolve_fund(
        self, subject: Mapping[str, Any], *, register: bool = False,
    ) -> dict[str, Any] | None:
        """Resolve a public fund identity; registration is an explicit separate action."""
        if not isinstance(subject, Mapping) or subject.get("kind", "open_fund") != "open_fund":
            raise ValidationError("基金身份只接受场外基金的公开字段")
        product = self._resolve_public_product(subject, register=register)
        return asdict(product) if product else None

    def save_profile(self, data: dict[str, Any]) -> str:
        data = dict(data)
        # Public imports cannot backdate what the system knew.
        if self.mode == "public":
            data["observed_at"] = now_iso()
        _safe_url(data.get("source_url", ""))
        profile = Profile(**data)
        self.store.add_profile(profile)
        return profile.id

    def overview(self, as_of: str) -> dict[str, Any]:
        point = timestamp(as_of)
        rows, alerts = [], []
        for product in self.store.list_products():
            fact_kind = "bar" if product.kind == "etf" else "nav"
            fact = self.store.latest_fact(product.id, fact_kind, as_of)
            status = "未获取"
            if fact:
                age = (point.date() - timestamp(fact.observed_at).date()).days
                economic_age = (point.date() - date.fromisoformat(fact.event_date)).days
                status = "已观察" if age <= 4 and economic_age <= 7 else "过期"
            profiles = self.store.profiles(product.id, as_of)
            verified = bool(profiles and profiles[0].verified)
            rows.append({
                **asdict(product), "latest_date": fact.event_date if fact else None,
                "latest_value": (fact.payload.get("close", fact.payload.get("unit_nav"))
                                 if fact else None),
                "observed_at": fact.observed_at if fact else None,
                "source": fact.source if fact else "无", "status": status,
                "paper_ready": "已核验档案，仍需交易日历/状态/权益检查" if verified else "仅监控：规则未核验",
            })
            if not fact or status == "过期":
                alerts.append({"severity": "warning", "title": f"{product.name} 数据{status}",
                               "detail": "无法据此生成可靠模拟成交；未知不等于零风险。",
                               "product_id": product.id, "source_url": ""})
            if not verified:
                alerts.append({"severity": "info", "title": f"{product.name} 尚无已核验规则档案",
                               "detail": "可观察；不能默认零费率、正常申赎或即时到账。",
                               "product_id": product.id, "source_url": ""})
            state = self.store.latest_fact(product.id, "status", as_of)
            if state and any(state.payload.get(key) == "closed" for key in ("subscription", "redemption")):
                alerts.append({"severity": "warning", "title": f"{product.name} 申赎限制",
                               "detail": canonical_json(state.payload), "product_id": product.id,
                               "source_url": state.source_url})
        for item in self.announcements(as_of):
            title = item["title"]
            if any(term in title for term in ("暂停", "限购", "经理", "分红", "拆分", "费率")):
                alerts.append({"severity": "info", "title": title,
                               "detail": "关键词命中，需要阅读原文；不是自动买卖建议。",
                               "product_id": item["product_id"], "source_url": item["url"]})
        alerts.append({"severity": "info", "title": "未设置个人配置目标",
                       "detail": "首版不预设风险偏好，不以基金数量等同于分散；公开持仓有滞后。",
                       "product_id": "", "source_url": ""})
        return {"products": rows, "alerts": alerts, "sources": self.store.snapshots(),
                "links": LINKS, "mode": self.mode, "as_of": point.isoformat()}

    def prices(self, product_id: str, as_of: str) -> list[dict[str, Any]]:
        product = self.store.get_product(product_id)
        if not product:
            raise ValidationError("产品不存在")
        kind = "bar" if product.kind == "etf" else "nav"
        facts = self.store.facts(product_id, kind, as_of,
                                 end=timestamp(as_of).date().isoformat())
        # Most recently observed revision of each date. Revisions remain in the repository.
        versions = {fact.event_date: fact for fact in facts}
        return [{"date": day, "value": fact.payload["close" if kind == "bar" else "unit_nav"],
                 "basis": "不复权市场价格（非总回报）" if kind == "bar" else "单位净值（非红利再投收益）",
                 "source": fact.source, "observed_at": fact.observed_at}
                for day, fact in sorted(versions.items())]

    def fund_detail(
        self, product_id: str, as_of: str | None = None, *, adopted_research: Mapping | None = None,
    ) -> dict[str, Any]:
        """Actually-available NAV and public disclosure, without refresh, fit or writes."""
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        product = self.store.get_product(product_id)
        result = {
            "status": "missing", "product": asdict(product) if product else None,
            "history": [], "latest": None, "previous_change": None,
            "last_checked_at": None, "last_successful_check_at": None,
            "latest_attempt_status": None, "reasons": [], "portfolio": None,
            "background_records": [], "investment_profile": None, "issuer_analysis": None,
            "research_thesis": None, "execution_conditions": None,
        }
        if point > clock:
            result.update(status="error", reasons=["不能提前查看实际时钟之后才可用的资料。"])
            return result
        if product is None:
            result["reasons"] = ["尚未登记这只基金；浏览不会自动创建公开产品。"]
            return result
        if product.is_demo or product.kind != "open_fund":
            result.update(status="unsupported", reasons=["此详情仅展示公开场外基金的单位净值。"])
            return result
        try:
            data = self._forecast_data(product, point.isoformat())
            nav = data["nav"]
            attempt, success = nav["latest_snapshot"], nav["snapshot"]
            result.update(
                last_checked_at=attempt["observed_at"] if attempt else None,
                last_successful_check_at=success["observed_at"] if success else None,
                latest_attempt_status=attempt["status"] if attempt else None,
            )
            reasons = list(data["reasons"])
            units = data["source_units"]
            if (len({currency for currency, _ in units}) > 1
                    or any(nav_type not in {"unit_nav", "final_unit_nav", "final_unit_nav_candidate"}
                           for _, nav_type in units)) and not reasons:
                reasons.append("有效来源的净值币种或口径存在冲突，不能选其中一个用于估值。")
            if reasons:
                # _forecast_data retains conflicting source facts for reproducibility.
                # Its selected row is not a verified value for charts or valuation.
                result.update(status="error", reasons=reasons)
            elif data["series"]:
                result["history"] = [{
                    "date": row["date"], "value": row["nav"],
                    "currency": row["provenance"]["currency"],
                    "basis": "单位净值（非红利再投收益）",
                    "source": row["provenance"]["source"],
                    "observed_at": row["provenance"]["observed_at"],
                } for row in data["series"]]
                latest = data["series"][-1]
                result["latest"] = {
                    "value": latest["nav"], "date": latest["date"],
                    **{key: latest["provenance"][key] for key in (
                        "currency", "observed_at", "available_at", "published_at",
                        "nav_type", "source", "source_url",
                    )},
                }
                if len(data["series"]) > 1:
                    result["previous_change"] = float(
                        decimal(latest["nav"]) / decimal(data["series"][-2]["nav"]) - 1
                    )
                result.update(status="ready", reasons=["显示本地已取得的单位净值；净值日期不是今天的实时估值。"])
                if latest["provenance"]["nav_type"] == "final_unit_nav_candidate":
                    result["reasons"].append("来源提供的是公布净值候选，尚未核实为官方净值。")
            else:
                result["reasons"] = ["本地暂无实际可用的单位净值；显式更新后才会请求数据。"]
            if attempt and attempt["status"] not in {"ok", "partial", "cached"}:
                result["status"] = "error"
                result["reasons"].append("最近一次数据检查失败；已有净值保留原日期，没有冒充本次更新成功。")
        except (ValidationError, OSError, RuntimeError):
            result.update(status="error", reasons=["本地净值资料无法读取；没有改写或重新计算来源记录。"])
        try:
            from .investment_profile import build_investment_profile, prefer_disclosure
            from .research_thesis import with_research_basis
            records, result["research_thesis"] = self._research_context(product.id, point.isoformat())
            records = with_research_basis(records, adopted_research,
                                          product_id=product.id, as_of=point.isoformat())
            result["investment_profile"] = build_investment_profile(records, as_of=point.isoformat())
            from .issuer_fundamentals import build_issuer_analysis
            result["issuer_analysis"] = build_issuer_analysis(
                records, profile=result["investment_profile"], as_of=point.isoformat(),
            )
            latest_records = {}
            for record in records:
                family = record["family"]
                day = point.date().isoformat()
                if (family not in {"mandate", "management", "portfolio", "fees", "operations"}
                        or (record["effective_from"] and record["effective_from"] > day)
                        or (record["effective_to"] and record["effective_to"] < day)):
                    continue
                if prefer_disclosure(record, latest_records.get(family)):
                    latest_records[family] = record
            result["background_records"] = [latest_records[key] for key in sorted(latest_records)]
            if operation := latest_records.get("operations"):
                result["execution_conditions"] = {
                    "subscription": operation["payload"]["subscription_state"],
                    "redemption": operation["payload"]["redemption_state"],
                    "report_date": operation["report_date"], "source_url": operation["source_url"],
                    "channel_verified": operation["payload"]["channel_verified"],
                    "reasons": ["公开披露不是支付宝当前页面确认；申购确认与赎回到账日期不同。"],
                }
            if record := latest_records.get("portfolio"):
                result["portfolio"] = {
                    **record["payload"], "report_date": record["report_date"],
                    "source_url": record["source_url"], "source": record["source"],
                    "observed_at": record["observed_at"], "available_at": record["available_at"],
                }
        except (ValidationError, OSError, RuntimeError):
            result["status"] = "error"
            result["reasons"].append("本地基金背景资料无法读取；没有把缺失披露补成完整资料。")
        return result

    def issuer_fundamentals(self, product_id: str, as_of: str | None = None) -> dict[str, Any]:
        """Read dated company financials and prices; never acquire or change a forecast."""
        from .investment_profile import build_investment_profile
        from .issuer_fundamentals import build_issuer_analysis

        point = timestamp(as_of or now_iso())
        if point > timestamp(now_iso()):
            raise ValidationError("不能提前查看实际时钟之后才可用的公司资料")
        product = self.store.get_product(product_id)
        if product is None or product.is_demo or product.kind != "open_fund":
            raise ValidationError("公司基本面分析需要已登记的公开场外基金")
        records, _ = self._research_context(product.id, point.isoformat())
        profile = build_investment_profile(records, as_of=point.isoformat())
        return build_issuer_analysis(records, profile=profile, as_of=point.isoformat())

    def _research_context(self, product_id: str, as_of: str) -> tuple[list, dict]:
        from .buyer_research import followed_review, followed_state
        from .research_thesis import build_research_thesis, with_research_basis
        records = self.store.advice_evidence(product_id, as_of)
        thesis = build_research_thesis(records, product_id=product_id, as_of=as_of)
        records = with_research_basis(records, thesis, product_id=product_id, as_of=as_of)
        return records, followed_review(thesis, followed_state(self.root), product_id=product_id,
                                       as_of=as_of, records=records)

    def research_thesis(self, product_id: str, as_of: str | None = None) -> dict:
        point = timestamp(as_of or now_iso())
        if point > timestamp(now_iso()):
            raise ValidationError("不能提前查看未来研究")
        return self._research_context(product_id, point.isoformat())[1]

    def import_research_note(self, product_id: str, path: Path) -> dict:
        from .buyer_research import read_public_json
        from .providers.advice_evidence import import_local_research_note
        product = self.store.get_product(product_id)
        if self.mode != "public" or product is None or product.is_demo:
            raise ValidationError("本地研究导入需要已登记的公开基金")
        return import_local_research_note(self.store, product, read_public_json(path), clock=now_iso)

    def import_session_research(self, product_id: str, path: Path, *,
                                evidence_path: Path | None = None) -> dict:
        """Explicitly append this session's research, without a model API call."""
        from .buyer_research import read_public_json
        from .providers.advice_evidence import append_generated_research
        from .research_contract import normalize_generated_research

        if self.mode != "public" or self.store.read_only:
            raise ValidationError("会话研究追加需要可写public工作区")
        product = self.store.get_product(product_id)
        if product is None or product.is_demo or product.kind != "open_fund":
            raise ValidationError("会话研究需要已登记的公开场外基金")
        payload = read_public_json(path)
        if (payload.get("schema_version") != 2
                or (payload.get("origin") or {}).get("provider") != "interactive_assistant"):
            raise ValidationError("此入口只追加明确标记为interactive_assistant的会话研究")
        evidence = read_public_json(evidence_path) if evidence_path is not None else None
        research = normalize_generated_research(payload, evidence=evidence, as_of=now_iso())
        receipt = append_generated_research(self.store, product, research, clock=now_iso)
        return {"receipt": receipt, "research_thesis": self.research_thesis(product_id),
                "research_api_called": False}

    def compare_funds(self, product_id: str, *, peer_set: Mapping | None = None,
                      as_of: str | None = None) -> dict:
        from .buyer_research import comparison_fund, load_buyer_config
        from .fund_comparison import compare_funds
        point = timestamp(as_of or now_iso())
        if point > timestamp(now_iso()):
            raise ValidationError("不能提前查看未来比较")
        config = load_buyer_config(self.root, as_of=point.isoformat())
        peers = peer_set or config.get("peer_sets", {}).get(product_id)
        if not peers:
            return {"status": "missing", "comparisons": [], "reasons": ["没有经过核对的替代候选集合"]}
        ids = [product_id, *(row["product_id"] for row in peers["peers"])]
        funds = {}
        target = self.fund_detail(product_id, point.isoformat())
        for identifier in ids:
            detail = target if identifier == product_id else self.fund_detail(
                identifier, point.isoformat(), adopted_research=target.get("research_thesis"),
            )
            if detail.get("product"):
                funds[identifier] = comparison_fund(detail, config.get("fund_metadata", {}).get(identifier))
        return compare_funds(peers, funds, as_of=point.isoformat())

    @staticmethod
    def current_online(public_subjects, *, source_plan=None, fetch=None, clock=now_iso,
                       request_budget: int = 20, blocked_urls=()) -> dict:
        """Fetch and parse in RAM; no Store, registration or archive is required."""
        from .buyer_research import current_public_identity
        from .providers.current import collect_current
        subjects = {row["id"]: row for row in map(current_public_identity, public_subjects)}
        public = [{"product_id": row["id"], "symbol": row["symbol"], "name": row["symbol"],
                   "share_class": row["share_class"], "currency": "CNY", "public": True}
                  for row in subjects.values()]
        options = {"source_plan": source_plan, "clock": clock, "request_budget": request_budget,
                   "blocked_urls": blocked_urls}
        if fetch is not None:
            options["fetch"] = fetch
        result = collect_current(public, **options)
        result["public_subjects"] = list(subjects.values())
        return result

    def current_subjects(self, subject: Mapping, *, additional=()) -> dict:
        """Resolve the public query scope without sending private holdings."""
        from .buyer_research import current_public_identity, followed_state, load_buyer_config
        from .fund_comparison import PeerSet
        if self.mode != "public":
            raise ValidationError("当前在线资料只用于public模式")
        config = load_buyer_config(self.root)
        target = current_public_identity(subject)
        peer_set = config.get("peer_sets", {}).get(target["id"])
        subjects = [target, *(current_public_identity(row) for row in additional)]
        if peer_set:
            parsed = PeerSet.from_mapping(peer_set)
            subjects.extend(current_public_identity(row.peer_id) for row in parsed.peers)
        subjects = {row["id"]: row for row in subjects}
        plan = config.get("current_source_plan")
        if plan is not None:
            from .providers.current import merge_current_source_plan
            plan = merge_current_source_plan(list(subjects.values()),
                [row for row in plan if row["product_id"] in subjects])
        return {"public_subjects": list(subjects.values()), "source_plan": plan,
                "peer_set": peer_set, "config": config,
                "blocked_urls": followed_state(self.root).get("blocked_urls", [])}

    def current_evaluation(self, product_id: str, context: Mapping, *,
                           public_subjects=(), peer_set: Mapping | None = None) -> dict:
        """Apply one acquisition to valuation, comparison and evidence, in memory."""
        from .buyer_research import compare_current_details, current_public_identity, load_buyer_config
        from .current_online import project_current_detail, snapshot_online_context
        from .research_thesis import with_research_basis
        point = context.get("evaluated_at") or context["checked_at"]
        config = load_buyer_config(self.root, as_of=point)
        peers = peer_set or config.get("peer_sets", {}).get(product_id)
        subjects = {row["id"]: row for row in map(current_public_identity,
                    public_subjects or context.get("public_subjects", []))}
        subjects.setdefault(product_id, current_public_identity(product_id))
        details = {}
        compact = snapshot_online_context(context)
        target = self.fund_detail(product_id, point)
        for identifier, identity in subjects.items():
            detail = target if identifier == product_id else self.fund_detail(
                identifier, point, adopted_research=target.get("research_thesis"),
            )
            if not detail.get("product"):
                detail["product"] = identity
            detail["evidence_records"] = with_research_basis(
                self.store.advice_evidence(identifier, point), target.get("research_thesis"),
                product_id=identifier, as_of=point,
            )
            projected = project_current_detail(detail, context, identifier)
            projected.pop("evidence_records", None)
            details[identifier] = projected
            selected = projected["online_context"]
            for key in ("review_signals_by_product", "execution_conditions_by_product", "projections_by_product"):
                compact.setdefault(key, {}).update(selected.get(key, {}))
            compact["records_by_product"][identifier] = selected["records_by_product"].get(identifier, [])
        comparison = compare_current_details(product_id, details, context, peer_set=peers,
                                              metadata=config.get("fund_metadata"))
        return {"details": details, "fund_comparison": comparison, "online_context": compact,
                "evaluated_at": point}

    def update_research(self, subject: Mapping, *, settings: Mapping, api_key: str | None = None,
                        context: Mapping | None = None, public_subjects=(), peer_set=None,
                        on_progress=None, request=None, acquire_more=None) -> dict:
        """One explicit public-data/API research task; never fit or issue a forecast.

        The completed note and its adopted excerpts use the existing evidence
        tables. Source documents stay in RAM; private holdings never enter here.
        """
        from copy import deepcopy
        from .buyer_research import current_public_identity
        from .providers.advice_evidence import append_generated_research
        from .research_runner import run_research
        from .research_settings import is_configured

        if self.mode != "public" or self.store.read_only:
            raise ValidationError("更新研究需要可写public工作区")
        identity = current_public_identity(subject)
        previous = self.research_thesis(identity["id"])
        if not is_configured(settings, api_key=api_key):
            return {"status": "unconfigured", "research_thesis": previous,
                    "research_run": {"status": "unconfigured", "research": None,
                                     "errors": ["请先配置已有服务的API地址、模型与Key。"]}}

        def progress(text):
            if on_progress is not None:
                on_progress(text)

        # Register only the public product on this explicit action, not on browse.
        resolved = self.resolve_fund(identity, register=True)
        identifier = resolved["id"]
        scope = self.current_subjects(resolved, additional=public_subjects)
        peers = peer_set or scope.get("peer_set")
        subjects = scope["public_subjects"]
        if context is None:
            progress("取材：读取公开净值、基金披露与相关消息…")
            context = self.current_online(subjects, source_plan=scope.get("source_plan"),
                                          blocked_urls=scope.get("blocked_urls", []))
        else:
            context = deepcopy(dict(context))
        evaluation = self.current_evaluation(identifier, context, public_subjects=subjects, peer_set=peers)
        detail = dict(evaluation["details"][identifier])
        point = evaluation["evaluated_at"]
        detail["evidence_records"] = self.store.advice_evidence(identifier, point)
        detail["evidence_links"] = [link for record in detail["evidence_records"]
            for link in self.store.advice_evidence_links(identifier, record["id"], point)]
        detail["peer_details"] = {}
        peer_ids = {row.get("product_id") for row in (peers or {}).get("peers", [])}
        for peer_id in peer_ids:
            peer_detail = evaluation["details"].get(peer_id)
            if peer_detail is None:
                continue
            public_detail = {key: peer_detail.get(key) for key in
                ("product", "investment_profile", "issuer_analysis", "background_records")}
            public_detail["evidence_records"] = self.store.advice_evidence(peer_id, point)
            public_detail["evidence_links"] = [link for record in public_detail["evidence_records"]
                for link in self.store.advice_evidence_links(peer_id, record["id"], point)]
            detail["peer_details"][peer_id] = public_detail
        # The registered public name, never the user's private display label.
        detail["product"] = resolved
        previous = detail.get("research_thesis") or self.research_thesis(identifier)

        def acquire_public_material(request_info: Mapping) -> dict:
            """Read only articles/attachments already discovered by this query.

            The model supplies evidence/message IDs, never URLs.  This keeps the
            acquisition seam bounded to links the publisher actually exposed;
            failed hosts, search results and arbitrary model URLs are not retried.
            """
            wanted = {value for value in request_info.get("evidence_ids", [])
                      if isinstance(value, str)}
            stopped = set(context.get("stopped_urls") or scope.get("blocked_urls", []))
            attempted = {row.get("source_url") for row in context.get("source_checks", [])
                         if isinstance(row, Mapping) and (
                             row.get("request_made") is True
                             or ("request_made" not in row and row.get("status") in {"ok", "error", "failed"})
                         )} | stopped
            jobs = []
            seen = set()
            for product_id, messages in (context.get("messages_by_product") or {}).items():
                for message in messages or []:
                    message_ids = {message.get("id"), message.get("evidence_id")} - {None}
                    if wanted and not wanted.intersection(message_ids):
                        continue
                    if not wanted and message.get("content_status") not in {
                        "title_only", "unparsed_document", "unparsed_pdf"
                    }:
                        continue
                    discovered = list(message.get("attachment_urls") or [])
                    if message.get("source_url"):
                        discovered.append(message["source_url"])
                    for source_url in discovered:
                        if (not isinstance(source_url, str) or source_url in attempted
                                or source_url in seen):
                            continue
                        seen.add(source_url)
                        jobs.append({
                            "product_id": product_id,
                            "job_id": f"research-attachment-{len(jobs)}",
                            "family": "disclosure",
                            "source_url": source_url,
                            "parser": "public_document_v1",
                            "config": {
                                "title": message.get("title") or "公开基金附件",
                                "published_at": message.get("published_at"),
                            },
                        })
            if not jobs:
                return {"status": "unavailable", "evidence_by_product": {},
                        "records_by_product": {}, "warnings": [
                            "本次补证请求没有对应的已发现公开附件；未按模型文字猜测URL。"]}
            more = self.current_online(scope["public_subjects"], source_plan=jobs[:2],
                                      request_budget=2, blocked_urls=sorted(stopped))

            def merge_list(key: str, identity: str) -> None:
                target = context.setdefault(key, {})
                for product_id, rows in (more.get(key) or {}).items():
                    output = target.setdefault(product_id, [])
                    known = {row.get(identity) for row in output if isinstance(row, Mapping)}
                    output.extend(row for row in rows if row.get(identity) not in known)

            for key in ("messages_by_product", "records_by_product", "evidence_by_product"):
                merge_list(key, "id")
            record_links = context.setdefault("record_evidence_by_product", {})
            for product_id, rows in (more.get("record_evidence_by_product") or {}).items():
                destination = record_links.setdefault(product_id, {})
                for record_id, links in rows.items():
                    existing = destination.setdefault(record_id, [])
                    known = {(link.get("evidence_id"), link.get("locator"), link.get("quote"))
                             for link in existing}
                    existing.extend(link for link in links
                                   if (link.get("evidence_id"), link.get("locator"), link.get("quote"))
                                   not in known)
            checks = context.setdefault("source_checks", [])
            known_checks = {(row.get("source_url"), row.get("job_id")) for row in checks
                            if isinstance(row, Mapping)}
            checks.extend(row for row in more.get("source_checks", [])
                          if (row.get("source_url"), row.get("job_id")) not in known_checks)
            context["requests_made"] = context.get("requests_made", 0) + more.get("requests_made", 0)
            context["stopped_urls"] = sorted(stopped | set(more.get("stopped_urls") or []))
            context.setdefault("warnings", []).extend(more.get("warnings") or [])
            return more

        progress("分析：区分新买、保留与追加，核对支持和反方依据…")
        run = run_research(product=resolved, context=context, detail=detail,
            comparison=evaluation.get("fund_comparison"), previous=previous,
            settings=settings, api_key=api_key, request=request,
            on_progress=on_progress, acquire_more=acquire_more or acquire_public_material)
        result = {"status": run["status"], "research_run": run, "context": context,
                  "evaluation": evaluation, "research_thesis": previous}
        if run.get("research") is None or run["status"] not in {"generated", "partial"}:
            return result
        progress("保存：记录本次研究和实际采用的引用摘录…")
        if not self.store.has_evidence_schema():
            self.store.initialize_evidence_schema()
        product = self.store.get_product(identifier)
        try:
            receipt = append_generated_research(self.store, product, run["research"], clock=now_iso)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            result.update(status="failed", errors=[str(exc)])
            return result
        # Generation completed after collection. Re-read the new note as of now,
        # without renewing the source checked_at, observation or publication clocks.
        context["evaluated_at"] = now_iso()
        evaluation = self.current_evaluation(identifier, context, public_subjects=subjects, peer_set=peers)
        result.update(receipt=receipt, evaluation=evaluation, context=context,
                      research_thesis=evaluation["details"][identifier].get("research_thesis"))
        progress("完成：新研究已出具；普通浏览和费用编辑不会再次调用API。")
        return result

    def import_nav_capture(self, product_id: str, path: Path, *, source_url: str,
                           start: str, end: str) -> dict:
        """Explicitly re-parse retained public NAV JS; import time is not live check time."""
        from .domain import Fact
        from .providers.followed import _nav, _url
        product = self.store.get_product(product_id)
        if self.mode != "public" or product is None or product.is_demo or product.kind != "open_fund":
            raise ValidationError("净值原件导入需要已登记公开基金")
        if path.is_symlink() or path.name.startswith(".env") or path.stat().st_size > 2 * 1024 * 1024:
            raise ValidationError("仅接受有界公开净值原件")
        _url(source_url)
        observed = now_iso()
        records = _nav(path.read_bytes(), {"symbol": product.symbol, "share_class": product.share_class,
                                           "currency": "CNY"}, {"start": start, "end": end}, observed)
        raw = [{"record": {"date": row["date"], "unit_nav": row["value"],
                           **({"unitMoney": row["unitMoney"]} if "unitMoney" in row else {})},
                "source": "retained_public_nav_import", "source_url": source_url, "observed_at": observed}
               for row in records]
        with self.store.transaction():
            snapshot = self.store.record_snapshot("retained_public_nav_import", product_id, raw,
                                                  observed, source_url, "ok", ["本地原件导入，不是本次联网检查"])
            count = sum(self.store.add_fact(Fact(product_id, "nav", row["date"],
                {"unit_nav": row["value"], "currency": "CNY", "nav_type": "final_unit_nav_candidate"},
                "retained_public_nav_import", observed, source_url)) for row in records)
        return {"status": "imported", "inserted": count, "snapshot_id": snapshot,
                "observed_at": observed, "network": False, "live_source_check": False}

    def refresh_followed(self, public_subjects: list[Mapping], *, families: list[str],
                         source_plan: list[Mapping] | None = None, request_budget: int = 12,
                         output_dir: Path | None = None) -> dict:
        """Explicit public-only acquisition; no personal amounts cross this seam."""
        from .buyer_research import load_buyer_config
        from .providers.followed import refresh_followed
        if self.mode != "public" or self.store.read_only:
            raise ValidationError("资料更新需要可写public工作区")
        subjects = []
        for identity in public_subjects:
            product = self._resolve_public_product(identity, register=False)
            if product is None:
                raise ValidationError("先登记公开身份，再明确更新资料")
            subjects.append({"product_id": product.id, "symbol": product.symbol,
                             "name": product.name, "share_class": product.share_class,
                             "currency": "CNY", "public": True})
        config = load_buyer_config(self.root)
        ids = {row["product_id"] for row in subjects}
        jobs = [row for row in (source_plan if source_plan is not None else config.get("source_plan", []))
                if row["product_id"] in ids]
        output = output_dir or Path("runs/artifacts") / ("followed-" + uuid.uuid4().hex[:12])
        from .buyer_research import followed_state
        state = followed_state(self.root)
        previous = dict(state.get("hashes", {}))
        blocked = set(state.get("blocked_urls", []))
        stopped_hosts = {urlparse(url).hostname for url in blocked}
        blocked.update(job["source_url"] for job in jobs if urlparse(job["source_url"]).hostname in stopped_hosts)
        # The adapter sees selected jobs only; the durable state retains other funds.
        result = refresh_followed(subjects, families, jobs, request_budget, output,
            previous_hashes={job["job_id"]: previous[job["job_id"]] for job in jobs if job["job_id"] in previous},
            blocked_urls=sorted(blocked))
        return self._store_followed_result(result, output, state)

    def review_retained_sources(self, product_ids: list[str], captures: Path, output: Path) -> dict:
        """Explicitly review saved successful captures; no request or source clock renewal."""
        from .buyer_research import followed_state, load_buyer_config, read_public_json
        from .providers.followed import review_followed_capture
        if self.mode != "public" or self.store.read_only:
            raise ValidationError("留存公开资料复核需要可写public工作区")
        state = followed_state(self.root)
        jobs = [job for job in load_buyer_config(self.root).get("source_plan", [])
                if job["product_id"] in product_ids and job["family"] != "nav"]
        output.mkdir(parents=True, exist_ok=False)
        outcomes = []
        for job in jobs:
            product = self.store.get_product(job["product_id"])
            if product is None:
                raise ValidationError("复核产品尚未登记")
            receipt_path = captures / job["job_id"] / "receipt.json"
            if not receipt_path.exists():
                outcomes.append({**{key: job[key] for key in ("product_id", "family", "job_id", "source_url")},
                                 "status": "not_captured", "records": []})
                continue
            capture = read_public_json(receipt_path)
            if capture["source_url"] != job["source_url"] or capture["family"] != job["family"]:
                raise ValidationError("留存收据的来源或资料家族与当前复核计划不符")
            raw_path = receipt_path.parent / "response.bin"
            if raw_path.is_symlink() or raw_path.stat().st_size > 8 * 1024 * 1024:
                raise ValidationError("留存公开原件超过复核范围")
            identity = {"product_id": product.id, "symbol": product.symbol, "name": product.name,
                        "share_class": product.share_class, "currency": "CNY", "public": True}
            # Old public receipts can lack the share suffix; bind explicitly without editing them.
            if capture["product_id"] not in {product.id, "open_fund:" + product.symbol}:
                raise ValidationError("留存原件没有匹配的公开产品身份")
            bound = {**capture, "product_id": product.id, "receipt_path": str(receipt_path)}
            if "observed_at" not in bound:
                bound["observed_at"] = capture["response_completed_at"]
            reviewed = review_followed_capture(identity, bound, raw_path.read_bytes(),
                                               parser=job["parser"], config=job["config"])
            old_hash = state.get("hashes", {}).get(job["job_id"])
            reviewed.update(job_id=job["job_id"], source_url=job["source_url"], raw_path=str(raw_path),
                raw_sha256=capture["raw_sha256"], http_status=200,
                changed=capture["raw_sha256"] != old_hash if old_hash else None)
            outcomes.append(reviewed)
        result = {"completed_at": now_iso(), "requests_made": 0, "outcomes": outcomes,
                  "retrieval": "retained_public_capture_review_not_live_fetch"}
        return self._store_followed_result(result, output, state)

    def _store_followed_result(self, result: dict, output: Path, state: Mapping) -> dict:
        from .domain import Fact
        previous = dict(state.get("hashes", {}))
        blocked = set(state.get("blocked_urls", []))
        inserted, evidence_added = 0, 0
        updates = list(state.get("updates", []))
        for outcome in result["outcomes"]:
            url = outcome.get("source_url")
            if outcome["status"] == "error" and url:
                blocked.add(url)
            if outcome.get("raw_sha256") and outcome.get("http_status") == 200:
                previous[outcome["job_id"]] = outcome["raw_sha256"]
            records = outcome.get("records", [])
            if outcome["family"] != "nav" and outcome.get("job_id"):
                prior = next((row for row in reversed(updates) if row["job_id"] == outcome["job_id"]), {})
                changed_at = prior.get("unreviewed_change_at")
                if outcome.get("changed") and outcome.get("http_status") == 200:
                    changed_at = result["completed_at"]
                updates.append({key: outcome.get(key) for key in (
                    "job_id", "product_id", "family", "source_url", "status", "changed", "raw_sha256",
                    "observed_at", "available_at", "receipt_path", "reason",
                )} | {"completed_at": result["completed_at"], "unreviewed_change_at": changed_at,
                     "reviewed_facts": records if outcome["family"] == "policy_news" and outcome["status"] == "ok" else []})
            if outcome["status"] != "ok":
                continue
            if outcome["family"] == "disclosure":
                from .providers.advice_evidence import import_reviewed_document
                structured = [row["evidence_import"] for row in records if row.get("evidence_import")]
                if not structured:
                    outcome["import_status"] = "no_structured_evidence"
                    continue
                try:
                    raw_path = Path(outcome.get("raw_path") or Path(outcome["receipt_path"]).parent / "response.bin")
                    if raw_path.is_symlink() or raw_path.stat().st_size > 8 * 1024 * 1024:
                        raise ValidationError("公开原件超过导入范围")
                    import hashlib
                    raw_bytes = raw_path.read_bytes()
                    if hashlib.sha256(raw_bytes).hexdigest() != outcome["raw_sha256"]:
                        raise ValidationError("公开原件与采集收据哈希不匹配")
                    imported = import_reviewed_document(self.store, self.store.get_product(outcome["product_id"]),
                        raw_bytes=raw_bytes, source_url=url, records=structured, reviewed=True)
                    evidence_added += imported["records_added"]
                    outcome["import_status"] = imported["status"]
                    outcome["record_ids"] = imported["record_ids"]
                except (ValidationError, OSError, KeyError) as exc:
                    outcome.update(import_status="unavailable", import_reason=str(exc))
            elif outcome["family"] == "nav":
                raw_records = [{"source": "followed_public_nav", "source_url": row["source_url"],
                    "observed_at": row["observed_at"], "record": {"date": row["date"], "unit_nav": row["value"],
                    **({"unitMoney": row["unitMoney"]} if "unitMoney" in row else {})}} for row in records]
                with self.store.transaction():
                    self.store.record_snapshot("followed_public_nav", outcome["product_id"], raw_records,
                        outcome["observed_at"], url, "ok", [])
                    for row in records:
                        inserted += self.store.add_fact(Fact(outcome["product_id"], "nav", row["date"],
                            {"unit_nav": row["value"], "currency": row["currency"],
                             "nav_type": "final_unit_nav_candidate"}, "followed_public_nav",
                            row["observed_at"], row["source_url"], available_at=row["available_at"]))
        log = self.root / "followed-state.json"
        log.write_text(json.dumps({"hashes": previous, "blocked_urls": sorted(blocked), "updates": updates,
                                   "last_output": str(output), "completed_at": result["completed_at"]},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        result.update(nav_inserted=inserted, evidence_added=evidence_added, output=str(output), research_recomputed=False,
            status="ready" if all(row["status"] == "ok" and row.get("import_status") not in {
                "unavailable", "no_structured_evidence"} for row in result["outcomes"]) else "partial")
        (output / "integrated-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def forecast_models(self, product_id: str | None = None) -> list[dict[str, Any]]:
        from .forecast_models import model_catalog
        if product_id is None:
            return model_catalog()
        product = self.store.get_product(product_id)
        return model_catalog(product.kind if product and not product.is_demo else "unsupported",
                             symbol=product.symbol if product else None)

    def forecast_history(
        self, product_id: str, as_of: str | None = None, *, model_id: str | None = None,
        horizon: int | None = None, limit: int = 30, offset: int = 0,
        role: str = "anchor", entry_lag: int | None = None,
    ) -> list[dict[str, Any]]:
        """Saved heads only; missing outcomes stay unknown, even after maturity."""
        from .forecast_models import model_name
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        if point > clock:
            raise ValidationError("不能用未来查询时间提前查看预测记录或结果")
        return [{**row, "model_name": model_name(row["model_id"])}
                for row in self.store.forecast_history_heads(
                    product_id, point.isoformat(), model_id=model_id, horizon=horizon,
                    role=role, entry_lag=entry_lag, limit=limit, offset=offset,
                )]

    def forecast_summary(
        self, product_id: str, as_of: str | None = None, *, horizon: int = 20,
        role: str = "primary", model_id: str | None = None,
    ) -> dict[str, Any]:
        """Summarize actual saved issues; this read never creates an outcome."""
        from .forecast_outcomes import summarize_outcomes
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        if point > clock:
            raise ValidationError("不能用未来查询时间提前查看预测成绩")
        rows = self.store.forecast_evidence_heads(
            product_id, point.isoformat(), horizon=horizon, model_id=model_id,
        )
        series = []
        product = self.store.get_product(product_id)
        if rows and product is not None and not product.is_demo and product.kind == "open_fund":
            data = self._forecast_data(product, point.isoformat())
            if not data["reasons"]:
                # Dates and availability locate unrecorded windows; their return is
                # not computed here and does not become a saved result by browsing.
                series = [{"date": row["date"], "invalid_return": row.get("invalid_return", False),
                           "provenance": {"available_at": row["provenance"]["available_at"]}}
                          for row in data["series"]]
        summary = summarize_outcomes(rows, series=series, horizon=horizon,
                                     role=role, model_id=model_id)
        return {**summary, "status": "ready" if rows else "missing",
                "product_id": product_id, "evaluated_at": point.isoformat()}

    def forecast_history_detail(
        self, product_id: str, record_id: str, as_of: str | None = None,
    ) -> dict[str, Any]:
        """Original saved prediction and associated outcomes, never its large input series."""
        from .forecast_models import model_name
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        result = {"status": "missing", "record_id": record_id, "product_id": product_id,
                  "generated_at": None, "model_id": None, "model_hash": None,
                  "model_name": None, "prediction": None, "outcomes": []}
        if point > clock:
            result.update(status="error", reasons=["不能用未来查询时间提前查看预测记录或结果。"])
            return result
        record = self.store.forecast_record_by_id(product_id, record_id, as_of=point.isoformat())
        if record is not None:
            result.update(record, status="ready", model_name=model_name(record["model_id"]),
                          outcomes=self.store.forecast_record_outcomes(product_id, record_id, point.isoformat()))
        return result

    def corporate_events(self, product_id: str, as_of: str) -> list[dict[str, Any]]:
        rows = []
        for kind in ("dividend", "split"):
            versions = {}
            for fact in self.store.facts(product_id, kind, as_of):
                versions[fact.event_date] = fact
            for fact in versions.values():
                rows.append({"kind": kind, "event_date": fact.event_date,
                             "record_date": fact.payload.get("record_date"),
                             "ex_date": fact.payload.get("ex_date", fact.event_date),
                             "pay_date": fact.payload.get("pay_date"),
                             "cash_per_unit": fact.payload.get("cash_per_unit"),
                             "ratio": fact.payload.get("ratio"), "source": fact.source,
                             "observed_at": fact.observed_at})
        return sorted(rows, key=lambda row: row["event_date"])

    def announcements(self, as_of: str) -> list[dict[str, Any]]:
        rows = []
        for fact in self.store.facts(kind="announcement", as_of=as_of):
            payload = fact.payload
            rows.append({"id": fact.id, "product_id": fact.product_id,
                         "title": str(payload.get("title", "未命名公告")),
                         "url": payload.get("url", fact.source_url),
                         "event_date": fact.event_date, "observed_at": fact.observed_at,
                         "source": fact.source})
        return list(reversed(rows))

    def save_note(
        self, product_id: str | None, at: str, decision: str, reason: str,
        source_url: str = "",
    ) -> str:
        at = timestamp(at).isoformat()
        if not reason.strip() or len(reason) > 20000:
            raise ValidationError("请记录理由（1–20000字符），也可记录为什么不操作")
        if product_id and not self.store.get_product(product_id):
            raise ValidationError("日志对应产品不存在")
        _safe_url(source_url)
        if self.mode == "public" and timestamp(at) > timestamp(now_iso()) + timedelta(minutes=1):
            raise ValidationError("真实观察日志不能写入未来时间")
        key = uuid.uuid4().hex
        evidence = [fact.id for fact in self.store.facts(product_id, as_of=at)]
        with self.store.transaction():
            self.store.conn.execute(
                "INSERT INTO journal(id,product_id,at,decision,reason,source_url,evidence,recorded_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (key, product_id, at, decision, reason.strip(), source_url,
                 canonical_json(evidence), now_iso()),
            )
        return key

    def notes(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.conn.execute("SELECT * FROM journal ORDER BY at DESC")]

    def refresh(self, product_id: str) -> dict[str, Any]:
        if self.mode != "public":
            raise ValidationError("演示模式不联网；切换 public 数据库后显式刷新")
        product = self.store.get_product(product_id)
        if not product:
            raise ValidationError("观察产品不存在")
        point = timestamp(now_iso())
        for snapshot in self.store.product_snapshots(product_id, point.isoformat(), limit=1):
            age = (point - timestamp(snapshot["observed_at"])).total_seconds()
            if 0 <= age < 300 and snapshot["status"] in {"ok", "partial"}:
                return {"status": "cached", "inserted": 0,
                        "warnings": ["复用5分钟内本地快照，避免重复请求免费源；不是新抓取。"],
                        "snapshot_id": snapshot["id"], "provider": snapshot["source"]}
            break
        from .providers.akshare import AkshareProvider, EastmoneyProfileProvider, SinaDailyProvider
        if product.kind == "monitor_only":
            raise ValidationError("该产品类型仅手动观察，尚无匹配的自动数据适配器")
        providers = ([EastmoneyProfileProvider(retries=0)] if product.kind == "open_fund"
                     else [SinaDailyProvider(), AkshareProvider()])
        warnings, snapshots, attempts = [], [], []
        inserted = 0
        for provider in providers:
            result = provider.fetch(product, now_iso())
            observed = now_iso()
            with self.store.transaction():
                snapshots.append(self.store.record_snapshot(
                    result.provider, product_id, result.raw_records, observed,
                    result.source_url, result.status, result.warnings,
                ))
                inserted += sum(self.store.add_fact(fact) for fact in result.facts)
            attempts.append({"provider": result.provider, "status": result.status})
            warnings.extend(result.warnings)
            if any(fact.kind in {"nav", "bar"} for fact in result.facts):
                break
        return {"status": result.status, "inserted": inserted, "warnings": warnings,
                "snapshot_id": snapshots[-1], "snapshot_ids": snapshots,
                "provider": result.provider, "attempts": attempts}

    # ---- explicit statistical forecasts ---------------------------------

    @staticmethod
    def _subject_match(product: Product, subject: Mapping[str, Any]) -> bool:
        return (not product.is_demo
                and product.symbol == subject["symbol"] and product.kind == subject["kind"]
                and all(not subject.get(key) or getattr(product, key) == subject[key]
                        for key in ("share_class", "exchange")))

    def _resolve_public_product(
        self, subject: Mapping[str, Any], *, register: bool,
    ) -> Product | None:
        allowed = {"symbol", "kind", "share_class", "exchange", "id", "name", "category", "is_demo"}
        if (not isinstance(subject, Mapping) or "symbol" not in subject
                or subject.keys() - allowed):
            raise ValidationError("subject只接受产品身份字段，不接受个人份额、成本或回答")
        if not isinstance(register, bool):
            raise ValidationError("register必须是布尔值")
        if subject.get("is_demo", False) is not False:
            raise ValidationError("公开基金不能登记演示产品")
        if any(not isinstance(value, str) for key, value in subject.items() if key != "is_demo"):
            raise ValidationError("产品公开身份字段必须是文本，代码须保留前导0")
        identity = {key: value.strip() for key, value in subject.items() if key != "is_demo"}
        identity.setdefault("kind", "open_fund")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,30}", identity["symbol"]):
            raise ValidationError("代码仅接受字母、数字、点、短横线和下划线")
        if register and (self.mode != "public" or self.store.read_only):
            raise ValidationError("公开产品登记需要可写public数据库")
        # Empty class/exchange means unknown, not a new competing identity.  An
        # explicit nonempty class/exchange must match; several candidates need a choice.
        with self.store.transaction() if register else nullcontext():
            matches = [product for product in self.store.list_products()
                       if self._subject_match(product, identity)]
            if len(matches) > 1:
                raise ValidationError("基金代码对应多个公开身份，请明确份额类别和交易场所后再选择")
            if matches:
                return matches[0]
            if not register:
                return None
            product_id = self.add_product(
                identity["symbol"], identity.get("name") or identity["symbol"], identity["kind"],
                exchange=identity.get("exchange", ""), category=identity.get("category", "未分类"),
                share_class=identity.get("share_class", ""),
            )
            return self.store.get_product(product_id)

    @staticmethod
    def _fact_provenance(fact: Any) -> dict[str, Any]:
        return {
            "fact_id": fact.id, "event_date": fact.event_date,
            "unit_nav": str(fact.payload.get("unit_nav")),
            "currency": fact.payload.get("currency", "CNY"),
            "nav_type": fact.payload.get("nav_type", "unit_nav"),
            "source": fact.source, "source_url": fact.source_url,
            "observed_at": fact.observed_at, "available_at": fact.available_at,
            "published_at": fact.published_at, "quality": fact.quality,
        }

    @staticmethod
    def _raw_matches_fact(row: Mapping[str, Any], fact: Any, *, match_value: bool = True) -> bool:
        if (row.get("event_date") != fact.event_date or row.get("source") != fact.source
                or row.get("source_url") != fact.source_url):
            return False
        return not match_value or decimal(row["unit_nav"]) == decimal(fact.payload["unit_nav"])

    def _forecast_snapshot(
        self, product_id: str, as_of: str, facts: list[Any],
    ) -> tuple[dict | None, list[dict], list[str]]:
        """Preserve known raw events across full/partial captures, as actually known.

        The newest anchor-matching capture is display metadata, not a replacement
        for earlier event evidence. A partial re-fetch cannot erase a known split.
        """
        warnings: list[str] = []
        if not facts:
            return None, [], warnings
        effective = {fact.event_date: fact for fact in facts}
        selected_snapshot = None
        known_markers: list[dict] = []
        latest_rows: dict[tuple, tuple[datetime, list[dict]]] = {}

        def available_captures():
            offset = 0
            while True:
                page = self.store.product_snapshots(product_id, as_of, limit=128, offset=offset, usable_only=True)
                yield from page
                if len(page) < 128:
                    return
                offset += len(page)

        # Page all actual-available captures: a known event must not disappear
        # merely because 128 later partial captures pushed it out of a window.
        for candidate in available_captures():
            try:
                rows = [row for row in _snapshot_nav_rows(self.store.snapshot_payload(candidate["id"]), candidate)
                        if timestamp(row["observed_at"]) <= timestamp(as_of)
                        and row["event_date"] <= timestamp(as_of).date().isoformat()]
            except ValidationError as exc:
                warnings.append(f"原始净值快照读取失败：{exc}")
                continue
            if selected_snapshot is None and any(self._raw_matches_fact(row, facts[-1]) for row in rows):
                keys = ("id", "source", "source_url", "observed_at", "status", "content_hash", "warnings")
                selected_snapshot = {key: candidate.get(key) for key in keys}
                try:
                    raw_warnings = json.loads(candidate.get("warnings") or "[]")
                    if isinstance(raw_warnings, list):
                        warnings.extend(str(item)[:2000] for item in raw_warnings[:32])
                except (ValueError, TypeError):
                    warnings.append("快照警告字段损坏；没有改写原始记录")
            for row in rows:
                fact = effective.get(row["event_date"])
                if fact is None or not self._raw_matches_fact(row, fact, match_value=False):
                    continue
                if self._raw_matches_fact(row, fact) and row["unit_money_present"]:
                    known_markers.append(row)
                # Old raw NAV revisions are not simultaneous conflicts with a
                # later fact revision. Only its actual later/equal raw version is.
                captured = timestamp(row["observed_at"])
                if captured < timestamp(fact.observed_at):
                    continue
                key = (row["event_date"], row["source"], row["source_url"])
                prior = latest_rows.get(key)
                if prior is None or captured > prior[0]:
                    latest_rows[key] = (captured, [row])
                elif captured == prior[0]:
                    prior[1].append(row)
        if selected_snapshot is None:
            warnings.append("缺少与最新净值日期、值、来源和URL匹配的原始快照；保留事实来源，不推定官方完整覆盖")
        merged = [row for _, rows in latest_rows.values() for row in rows]
        merged.extend(known_markers)
        return selected_snapshot, merged, warnings

    def _forecast_data(self, product: Product, as_of: str) -> dict[str, Any]:
        """Whole actually-available NAV history, with revisions and event masks."""
        point = timestamp(as_of)
        all_facts = self.store.facts(product.id, "nav", as_of=as_of, end=point.date().isoformat())
        versions: dict[tuple[str, str, str], tuple[tuple, list[Any]]] = {}
        for fact in all_facts:
            identity = (fact.event_date, fact.source, fact.source_url)
            version = (timestamp(fact.available_at), timestamp(fact.observed_at))
            previous = versions.get(identity)
            if previous is None or version > previous[0]:
                versions[identity] = (version, [fact])
            elif version == previous[0]:
                previous[1].append(fact)
        by_day: dict[str, list[Any]] = {}
        for _, source_facts in versions.values():
            for fact in source_facts:
                by_day.setdefault(fact.event_date, []).append(fact)
        conflicts = sorted(day for day, items in by_day.items()
                           if len({decimal(item.payload["unit_nav"]) for item in items}) > 1)
        facts = [max(by_day[day], key=lambda item:
                     (timestamp(item.available_at), timestamp(item.observed_at), item.id))
                 for day in sorted(by_day)]
        snapshot, raw, warnings = self._forecast_snapshot(product.id, as_of, facts)
        latest_snapshots = self.store.product_snapshots(product.id, as_of, limit=1)
        latest_snapshot = latest_snapshots[0] if latest_snapshots else None
        markers = []
        raw_conflicts = []
        for fact in facts:
            source_rows = [row for row in raw if self._raw_matches_fact(row, fact, match_value=False)]
            values = {decimal(row["unit_nav"]) for row in source_rows}
            if values and (len(values) > 1 or decimal(fact.payload["unit_nav"]) not in values):
                raw_conflicts.append(fact.event_date)
            matching_rows = [row for row in source_rows if self._raw_matches_fact(row, fact)]
            has_marker_field = any(row["unit_money_present"] for row in matching_rows)
            for row in matching_rows:
                if row["unit_money_present"] or not has_marker_field:
                    markers.append({key: row[key] for key in
                                    ("event_date", "source", "source_url", "unit_money_present", "unitMoney", "nonempty")})
        markers = [item for _, item in sorted({canonical_json(item): item for item in markers}.items())]
        events = []
        event_days = {item["event_date"] for item in markers if item["nonempty"]}
        if facts:
            for kind in ("dividend", "split", "announcement"):
                event_versions = {}
                for fact in self.store.facts(product.id, kind, as_of=as_of):
                    identity = (fact.event_date, fact.source, fact.source_url)
                    previous = event_versions.get(identity)
                    if previous is None or (timestamp(fact.available_at), timestamp(fact.observed_at), fact.id) > (
                            timestamp(previous.available_at), timestamp(previous.observed_at), previous.id):
                        event_versions[identity] = fact
                for fact in event_versions.values():
                    payload = dict(fact.payload)
                    event_day = str(payload.get("ex_date", fact.event_date))
                    if not facts[0].event_date <= event_day <= facts[-1].event_date:
                        continue
                    events.append({"fact_id": fact.id, "kind": kind, "event_date": fact.event_date,
                                   "effective_date": event_day, "payload": payload,
                                   "source": fact.source, "source_url": fact.source_url,
                                   "observed_at": fact.observed_at, "available_at": fact.available_at})
                    if kind in {"dividend", "split"}:
                        event_days.add(event_day)
        series = []
        previous_day = "0001-01-01"
        for fact in facts:
            provenance = self._fact_provenance(fact)
            provenance["source_markers"] = [item for item in markers if item["event_date"] == fact.event_date]
            provenance["corporate_events"] = [item for item in events
                if previous_day < item["effective_date"] <= fact.event_date]
            row = {"date": fact.event_date, "nav": str(fact.payload["unit_nav"]),
                   "provenance": provenance}
            if any(previous_day < event_day <= fact.event_date for event_day in event_days):
                row["invalid_return"] = True
            series.append(row)
            previous_day = fact.event_date
        warnings.append("单位净值不是红利再投资总回报；仅屏蔽实际已知权益事件，缺少权益完整覆盖仍须另行核验。")
        if event_days:
            warnings.append("已知分红/拆分/非空unitMoney所跨收益被排除训练和评分；没有删除日期或擅自复权。")
        basis = digest(series)
        nav = {
            "snapshot": snapshot, "latest_snapshot": latest_snapshot,
            "observation_count": len(series), "first_date": series[0]["date"] if series else None,
            "latest_date": series[-1]["date"] if series else None,
            "latest_nav": series[-1]["nav"] if series else None,
            "latest_available_at": facts[-1].available_at if facts else None,
            "economic_nav_age_days": (point.date() - date.fromisoformat(series[-1]["date"])).days if series else None,
            "nav_conflicts": conflicts, "raw_nav_conflicts": raw_conflicts,
            "unit_money": {"records": len(markers), "nonempty": len([row for row in markers if row["nonempty"]]),
                           "markers": markers, "coverage": "source_fields_only_not_official_complete"},
            "corporate_events": events, "basis_hash": basis, "warnings": warnings,
        }
        reasons = []
        if conflicts:
            reasons.append("有效净值来源之间存在数值冲突，未静默选取一个来源拟合。")
        if raw_conflicts:
            reasons.append("原始快照与有效净值存在数值冲突，需要核验版本。")
        # Equal numbers do not make contradictory source currencies compatible.
        source_units = {(fact.payload.get("currency", "CNY"), fact.payload.get("nav_type", "unit_nav"))
                        for items in by_day.values() for fact in items}
        currencies = {currency for currency, _ in source_units}
        nav_types = {nav_type for _, nav_type in source_units}
        # The direct free source labels the same unit_nav field as a final-NAV
        # *candidate*. Preserve that provenance qualification, not a false type conflict.
        if len(currencies) > 1 or nav_types - {"unit_nav", "final_unit_nav", "final_unit_nav_candidate"}:
            reasons.append("净值币种或净值口径不一致，不能拼接成同一条单位净值收益序列。")
        return {"series": series, "basis_hash": basis, "nav": nav, "reasons": reasons,
                "source_units": source_units,
                "anchor_currency": facts[-1].payload.get("currency") if facts else None}

    def _entry_profile(self, product: Product, as_of: str) -> tuple[Profile | None, time]:
        """The same actually available verified cutoff selection for both contracts."""
        for candidate in self.store.profiles(product.id, as_of):
            if not candidate.verified:
                continue
            raw_cutoff = candidate.rules.get("cutoff")
            try:
                parsed = time.fromisoformat(raw_cutoff)
                if len(raw_cutoff) != 5 or parsed.tzinfo is not None:
                    continue
            except (ValueError, TypeError):
                continue
            return candidate, parsed
        return None, time(15)

    def _entry_assumptions(self, product: Product, as_of: str, anchor_date: str) -> dict:
        point = timestamp(as_of)
        anchor = date.fromisoformat(anchor_date)
        profile, cutoff = self._entry_profile(product, as_of)
        calendar_id = profile.rules.get("calendar_id") if profile else None
        calendar = {}
        if calendar_id:
            for fact in self.store.facts(calendar_id, "calendar", as_of=as_of,
                                         start=anchor.isoformat(),
                                         end=(point.date() + timedelta(days=14)).isoformat()):
                calendar[fact.event_date] = fact
        used_proxy = False

        def is_open(day: date) -> bool:
            nonlocal used_proxy
            fact = calendar.get(day.isoformat())
            if fact is not None:
                return fact.payload["is_open"]
            used_proxy = True
            return day.weekday() < 5

        application = point.date()
        if not (is_open(application) and point.time().replace(tzinfo=None) < cutoff):
            application += timedelta(days=1)
            while not is_open(application):
                application += timedelta(days=1)
        lag = 0
        day = anchor + timedelta(days=1)
        while day <= application:
            lag += int(is_open(day))
            day += timedelta(days=1)
        return {
            "anchor_date": anchor_date, "decision_at": point.isoformat(),
            "application_date": application.isoformat(), "baseline_entry_lag": lag,
            "entry_lags": [lag, lag + 1, lag + 2],
            "cutoff": cutoff.isoformat(timespec="minutes"),
            "cutoff_basis": "verified_profile" if profile else "15:00_research_proxy",
            "profile_id": profile.id if profile else None,
            "profile_source_url": profile.source_url if profile else None,
            "calendar_id": calendar_id,
            "calendar_basis": "weekday_research_proxy" if used_proxy else "observed_profile_calendar",
            "label": "按实际可用档案截止时点；缺失日历用工作日研究代理，不是已核验交易日历",
            "warning": "从最新净值锚点到预计申购计价日另计滞后，并压力测试再延后1/2期；不是今天按旧净值买入。",
        }

    def _calendar_trade_plan(
        self, product: Product, as_of: str, anchor_date: str | None, calendar_days: int,
    ) -> tuple[dict, dict]:
        """Read local calendar versions as of NOW, including future confirmation days."""
        from .trade_calendar import build_trade_plan

        profile, cutoff = self._entry_profile(product, as_of)
        calendar_id = profile.rules.get("calendar_id") if profile else None
        assumptions = {
            "cutoff": cutoff.isoformat(timespec="minutes"),
            "cutoff_basis": "verified_profile" if profile else "15:00_scenario",
            "profile_id": profile.id if profile else None,
            "profile_source_url": profile.source_url if profile else None,
            "calendar_id": calendar_id,
        }
        for candidate in self.store.profiles(product.id, as_of):
            confirm = candidate.rules.get("confirm_days")
            if (candidate.verified and isinstance(confirm, int)
                    and not isinstance(confirm, bool) and confirm >= 0):
                assumptions.update(
                    confirmation_days=confirm, confirmation_basis="verified_profile",
                    confirmation_label=f"适用档案：申请后{confirm}个业务日确认",
                    confirmation_profile_id=candidate.id,
                    confirmation_profile_source_url=candidate.source_url,
                )
                break
        calendar = {}
        if calendar_id:
            # Do not cap economic dates at today (these are preannounced calendar
            # records), nor at an old fixed 14-day window. The pure helper checks
            # all needed dates through the deadline and the new-lot confirmation.
            start = min(anchor_date or timestamp(as_of).date().isoformat(),
                        timestamp(as_of).date().isoformat())
            versions = {}
            for fact in self.store.facts(calendar_id, "calendar", as_of=as_of, start=start):
                key = fact.event_date, fact.source, fact.source_url
                prior = versions.get(key)
                if prior is None or (timestamp(fact.available_at), timestamp(fact.observed_at), fact.id) > (
                        timestamp(prior.available_at), timestamp(prior.observed_at), prior.id):
                    versions[key] = fact
            by_day = {}
            for fact in versions.values():
                by_day.setdefault(fact.event_date, set()).add(fact.payload["is_open"])
            # Contradictory calendars are not silently selected. They count as
            # missing applicable coverage and visibly invoke the weekday scenario.
            calendar = {day: next(iter(values)) for day, values in by_day.items() if len(values) == 1}
        plan = build_trade_plan(
            decision_at=as_of, anchor_date=anchor_date, calendar_days=calendar_days,
            entry_assumptions=assumptions, calendar=calendar,
        )
        lag = plan["entry_lag"]
        assumptions.update(
            anchor_date=anchor_date, decision_at=plan["decision_at"],
            application_date=plan["application_date"], baseline_entry_lag=lag,
            entry_lags=[lag, lag + 1, lag + 2] if lag is not None else [],
            horizon=plan["horizon"], target_unit="nav_observations",
            calendar_basis=plan["calendar_basis"], calendar_label=plan["calendar_label"],
            confirmation_days=plan["confirmation_days"], confirmation_basis=plan["confirmation_basis"],
            confirmation_label=plan["confirmation_label"],
            label="条件买卖日程：每个假定业务日1笔净值，不是已核验未来净值日历",
            warning="锚点至申请日的L及申请日至计划卖出的H均为日程假设；另压力测试再延后1/2笔。",
        )
        return plan, assumptions

    @staticmethod
    def _forecast_model(model_id: str) -> str:
        from .forecast_models import INVESTMENT_MODEL_ID, NAV_MODEL_ID
        if model_id in {"nav_ridge_empirical_v1", NAV_MODEL_ID}:
            return NAV_MODEL_ID
        if model_id != INVESTMENT_MODEL_ID:
            raise ValidationError("没有实现这个预测模型")
        return model_id

    def _investment_inputs(self, product: Product, as_of: str) -> dict:
        from .investment_forecast import profile_mapping_input_hash
        from .investment_profile import build_investment_profile

        if product.symbol != "018147":
            raise ValidationError("这只基金还没有核对投资因素映射，不会套用018147的因素。")
        capture = self.store.factor_capture(product.id, as_of=as_of)
        if capture is None or capture.get("status") != "ready":
            raise ValidationError("尚无成功保存的市场、行业、汇率和利率数据；请明确更新投资因素。")
        profile = build_investment_profile(self.store.advice_evidence(product.id, as_of), as_of=as_of)
        try:
            mapping_hash = profile_mapping_input_hash(profile, product_id=product.id, generated_at=as_of)
        except ValidationError as exc:
            raise ValidationError("已保存的合同与持仓披露还不足以核对市场和半导体参照；需要补齐本基金有日期的投向资料，单独更新价格不能补齐。") from exc
        return {"bundle": capture["bundle"], "profile": profile,
                "factor_input_hash": capture["input_hash"], "profile_mapping_hash": mapping_hash,
                "capture_id": capture["id"], "observed_at": capture["observed_at"]}

    def import_investment_factors(self, product_id: str, path: str | Path) -> dict:
        """Explicit public bundle import; no network, fit or historical clock rewriting."""
        from .investment_factors import load_factor_bundle
        if self.mode != "public" or self.store.read_only:
            raise ValidationError("投资因素只导入可写公开库")
        product = self._forecast_product(product_id)
        if product.symbol != "018147" or product.kind != "open_fund":
            raise ValidationError("目前只核对了018147的投资因素映射")
        bundle = load_factor_bundle(path)
        point = now_iso()
        if any(timestamp(row["available_at"]) > timestamp(point)
               for source in bundle["series"].values() for row in source["rows"]):
            raise ValidationError("不能导入实际时钟之后才取得的投资因素")
        value = self.store.add_factor_capture(product.id, {"status": "ready", "bundle": bundle,
            "warnings": bundle.get("warnings", []), "captures": bundle.get("captures", [])}, observed_at=point)
        return {key: value[key] for key in ("id", "product_id", "status", "observed_at", "input_hash", "warnings")}

    def refresh_investment_factors(self, product_id: str) -> dict:
        """Only an explicit update acquires the qualified public factor sources."""
        import base64
        from .providers.factors import fetch_factor_bundle

        if self.mode != "public" or self.store.read_only:
            raise ValidationError("投资因素采集只支持可写公开库")
        product = self._forecast_product(product_id)
        if product.symbol != "018147" or product.kind != "open_fund":
            raise ValidationError("目前只核对了018147的投资因素映射")
        fetched = fetch_factor_bundle(product_id=product.id)
        point = now_iso()
        capture_dir = self.root / "factor-sources" / uuid.uuid4().hex
        raw_records = fetched.get("raw_records", [])
        try:
            if raw_records:
                capture_dir.mkdir(parents=True, exist_ok=False)
                for index, raw in enumerate(raw_records):
                    target = capture_dir / str(index)
                    target.mkdir()
                    content = base64.b64decode(raw["body_base64"], validate=True)
                    (target / "response.bin").write_bytes(content)
                    (target / "receipt.json").write_text(canonical_json({key: value for key, value in raw.items()
                        if key != "body_base64"}), encoding="utf-8")
                    for capture in fetched.get("captures", []):
                        if capture.get("content_hash") == raw.get("content_hash") and capture.get("url") == raw.get("url"):
                            capture["retained_path"] = str(target.relative_to(self.root))
        except OSError as exc:
            fetched.update(status="error", bundle=None,
                           warnings=[*fetched.get("warnings", []), f"原始因素响应未完整保存：{exc}"])
        stored = self.store.add_factor_capture(product.id, fetched, observed_at=point)
        return {key: stored[key] for key in ("id", "product_id", "status", "observed_at", "input_hash", "warnings")}

    def _forecast_request(
        self, product: Product, as_of: str, data: dict, *, model_hash: str | None = None,
        model_id: str = "nav_ridge_empirical_v2", investment: dict | None = None,
    ) -> tuple[str, dict]:
        from .forecast import RUNTIME_MODEL_HASH
        assumptions = self._entry_assumptions(product, as_of, data["series"][-1]["date"])
        identity = {"product_id": product.id, "model_hash": model_hash or RUNTIME_MODEL_HASH,
                    "basis_hash": data["basis_hash"], "entry_lags": assumptions["entry_lags"],
                    "profile_id": assumptions["profile_id"], "cutoff": assumptions["cutoff"],
                    "cutoff_basis": assumptions["cutoff_basis"], "calendar_basis": assumptions["calendar_basis"]}
        if model_id == "exposure_factor_ridge_empirical_v1":
            from .investment_factors import ALIGNMENT
            from .investment_forecast import FEATURE_DEFINITION_HASH, MODEL_HASH
            investment = investment or self._investment_inputs(product, as_of)
            identity.update(model_hash=MODEL_HASH, factor_input_hash=investment["factor_input_hash"],
                            profile_mapping_hash=investment["profile_mapping_hash"],
                            feature_definition_hash=FEATURE_DEFINITION_HASH, alignment=ALIGNMENT)
        return digest(identity), assumptions

    @staticmethod
    def _forecast_wrapper(product: Product | None, as_of: str) -> dict:
        return {"status": "missing", "product": asdict(product) if product else None,
                "evaluated_at": timestamp(as_of).isoformat(), "prediction": None,
                "reasons": [], "evidence": {"records": [], "nav": {"snapshot": None}}}

    def forecast(
        self, product_id: str, as_of: str | None = None, *,
        model_id: str = "nav_ridge_empirical_v2",
    ) -> dict[str, Any]:
        """Read the selected model only; never fit, acquire, initialize or write."""
        model_id = self._forecast_model(model_id)
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        product = self.store.get_product(product_id)
        result = self._forecast_wrapper(product, min(point, clock).isoformat())
        result["selected_model_id"] = model_id
        if point > clock:
            result.update(status="error", reasons=["不能用未来查询时间提前使用尚未实际可用的资料。"])
            return result
        if product is None:
            result["reasons"] = ["尚未登记公开观察产品；显式更新才登记或计算。"]
            return result
        if product.is_demo or product.kind != "open_fund":
            result.update(status="unsupported", reasons=["当前预测支持公开场外基金单位净值；ETF/演示产品不套用旧规则。"])
            return result
        if model_id == "exposure_factor_ridge_empirical_v1" and product.symbol != "018147":
            result.update(status="unsupported", reasons=["这只基金还没有核对投资因素映射，不套用其他基金的因素。"])
            return result
        try:
            from .forecast import MIN_OBSERVATIONS, MODEL_HASH, RUNTIME_MODEL_HASH
            data = self._forecast_data(product, point.isoformat())
            result["evidence"] = {"records": self.store.advice_evidence(product.id, point.isoformat()),
                                  "nav": data["nav"]}
            record = self.store.forecast_record(product.id, as_of=point.isoformat(), model_id=model_id)
            if record:
                result["prediction"] = record["prediction"]
            if data["reasons"]:
                result.update(status="error", reasons=data["reasons"])
                return result
            if len(data["series"]) < MIN_OBSERVATIONS:
                result.update(status="insufficient", reasons=[f"实际可用净值{len(data['series'])}期，至少需要{MIN_OBSERVATIONS}期才能估计最短预测；没有补值。"])
                return result
            supported_hashes = {MODEL_HASH, RUNTIME_MODEL_HASH}
            if model_id == "exposure_factor_ridge_empirical_v1":
                from .investment_forecast import MODEL_HASH as INVESTMENT_HASH
                supported_hashes = {INVESTMENT_HASH}
            if record and record["model_hash"] not in supported_hashes:
                result.update(status="stale", reasons=["这条记录使用已存档的模型版本；原预测可在历史中查看，显式更新才生成当前版本。"])
                return result
            investment = None
            if model_id == "exposure_factor_ridge_empirical_v1":
                try:
                    investment = self._investment_inputs(product, point.isoformat())
                except ValidationError as exc:
                    result.update(status="insufficient", input_status="missing_investment_inputs", reasons=[str(exc)])
                    return result
            request, assumptions = self._forecast_request(
                product, point.isoformat(), data,
                model_hash=record["model_hash"] if record else None,
                model_id=model_id, investment=investment,
            )
            result["evidence"]["nav"]["entry_assumptions"] = assumptions
            matching = self.store.forecast_record(product.id, as_of=point.isoformat(), request_hash=request)
            if matching is not None:
                record = matching
                result["prediction"] = record["prediction"]
            if record is None:
                result["reasons"] = ["本地有净值但尚无已保存预测；请显式更新，浏览不会自动拟合。"]
            elif record["request_hash"] != request:
                result.update(status="stale", reasons=["净值/权益/模型或申购滞后依据已改变；下方仅是原发布时间的历史预测，显式更新才生成新预测。"])
            else:
                result.update(status="ready", reasons=["显示已保存预测；生成时间和净值所属日期见下方。"])
            latest = data["nav"].get("latest_snapshot")
            if (latest and latest["status"] not in {"ok", "partial", "cached"}
                    and (record is None or timestamp(latest["observed_at"]) >= timestamp(record["generated_at"]))):
                result.update(status="error", refresh={"status": latest["status"],
                              "snapshot_id": latest["id"], "provider": latest["source"],
                              "observed_at": latest["observed_at"], "warnings": latest["warnings"]})
                result["reasons"].append("最近一次显式源采集失败；旧预测保持原发布时间，只供历史查看。")
            return result
        except (ValidationError, OSError, RuntimeError) as exc:
            result.update(status="error", reasons=["本地预测资料无法读取；没有重新拟合或覆盖原记录。"])
            result["evidence"]["diagnostic"] = {"error_type": type(exc).__name__, "detail": str(exc)}
            return result

    def _timing_data(self, product: Product, as_of: str, context: Mapping | None) -> dict:
        """Merge full local provenance with one RAW acquisition, never a display history."""
        from copy import deepcopy

        data = deepcopy(self._forecast_data(product, as_of))
        if context is None:
            return data
        if (not isinstance(context, Mapping) or context.get("kind") != "current_online_context"
                or context.get("schema_version") != 1 or "nav_by_product" not in context
                or "nav_observation_counts" in context or "projections_by_product" in context):
            raise ValidationError("择时计算需要本批次完整RAW在线资料，不能使用压缩快照或详情history。")
        point = timestamp(as_of)
        started = timestamp(context.get("started_at"))
        completed = timestamp(context.get("evaluated_at") or context.get("checked_at"))
        checked = timestamp(context.get("checked_at"))
        if not started <= checked <= completed <= point:
            raise ValidationError("在线批次时钟不一致或晚于实际计算时间。")
        for identity in context.get("public_subjects", []):
            if identity.get("id", identity.get("product_id")) != product.id:
                continue
            if any(identity[key] != getattr(product, key) for key in
                   ("symbol", "share_class", "kind", "exchange", "is_demo") if key in identity):
                raise ValidationError("在线公开身份与所选基金代码、份额或产品类型不匹配。")
        by_product = context["nav_by_product"]
        if not isinstance(by_product, Mapping):
            raise ValidationError("RAW在线净值必须按公开产品标识分组。")
        incoming = by_product.get(product.id, [])
        if not isinstance(incoming, list):
            raise ValidationError("RAW在线净值必须是完整观察列表。")
        accepted, rejected = [], []
        for raw in incoming:
            try:
                if not isinstance(raw, Mapping):
                    raise ValidationError("在线净值观察必须是对象")
                if raw.get("product_id", product.id) != product.id:
                    raise ValidationError("在线净值所属产品不匹配")
                for key in ("symbol", "share_class"):
                    if key in raw and raw[key] != getattr(product, key):
                        raise ValidationError("在线净值代码或份额类别不匹配")
                day = date.fromisoformat(raw["date"]).isoformat()
                if day != raw["date"] or day > point.date().isoformat() or decimal(raw["value"]) <= 0:
                    raise ValidationError("在线净值日期或数值不可用")
                if not re.fullmatch(r"[A-Z]{3}", str(raw.get("currency", ""))):
                    raise ValidationError("在线净值缺少明确币种")
                if raw.get("nav_type") not in {"unit_nav", "final_unit_nav", "final_unit_nav_candidate"}:
                    raise ValidationError("在线资料不是单位净值，不能混用估算或累计净值")
                observed, available = timestamp(raw["observed_at"]), timestamp(raw["available_at"])
                if not (started <= observed <= available <= completed) or day > observed.date().isoformat():
                    raise ValidationError("在线净值不属于本批次实际取得的可用观察")
                if raw.get("published_at"):
                    if raw.get("publication_precision") == "date":
                        if date.fromisoformat(raw["published_at"]) > available.date():
                            raise ValidationError("在线净值披露日期晚于实际可用时间")
                    elif timestamp(raw["published_at"]) > available:
                        raise ValidationError("在线净值披露时间晚于实际可用时间")
                if any(not isinstance(raw.get(key), str) or not raw[key].strip()
                       for key in ("source", "source_url")):
                    raise ValidationError("在线净值缺少实际来源")
                for flag in ("invalid_return", "label_valid"):
                    if flag in raw and not isinstance(raw[flag], bool):
                        raise ValidationError("在线净值事件标记必须是布尔值")
                accepted.append(deepcopy(dict(raw)))
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                rejected.append(str(exc))
        data["reasons"].extend("未采用在线净值：" + reason for reason in dict.fromkeys(rejected))
        data["nav"]["online_context"] = {
            "started_at": context["started_at"], "checked_at": context["checked_at"],
            "evaluated_at": context.get("evaluated_at"), "accepted_observations": len(accepted),
            "rejected_reasons": list(dict.fromkeys(rejected)),
            "source_checks": deepcopy([row for row in context.get("source_checks", [])
                                      if row.get("product_id") == product.id and row.get("family") == "nav"]),
        }
        # Equal values may corroborate a fact; they never overwrite its earlier
        # actual available_at. Different values remain a conflict, even if newer.
        by_day = {}
        for row in accepted:
            by_day.setdefault(row["date"], []).append(row)
        local = {row["date"]: row for row in data["series"]}
        conflicts = []
        for day, rows in by_day.items():
            values = {decimal(row["value"]) for row in rows}
            if day in local:
                values.add(decimal(local[day]["nav"]))
            if len(values) > 1:
                conflicts.append(day)
        if conflicts:
            data["reasons"].append("在线与已有/同批次同日单位净值存在数值冲突，未静默拼接或替换："
                                   + "、".join(sorted(conflicts)) + "。")
        data["nav"]["nav_conflicts"] = sorted(set(data["nav"]["nav_conflicts"]) | set(conflicts))
        units = data["source_units"] | {(row["currency"], row["nav_type"]) for row in accepted}
        if len({currency for currency, _ in units}) > 1:
            data["reasons"].append("在线和已有净值币种不一致，不能拼接成同一条收益序列。")
        data["source_units"] = units
        if data["reasons"]:
            return data

        local_anchor_date = data["series"][-1]["date"] if data["series"] else None
        old_events = data["nav"]["corporate_events"]
        markers = data["nav"]["unit_money"]["markers"]
        old_event_days = {row["effective_date"] for row in old_events if row["kind"] in {"dividend", "split"}}
        old_event_days.update(row["event_date"] for row in markers if row["nonempty"])
        # Reassign a known event to the next real observation when online history
        # fills a gap; do not leave the same incoming move masked twice.
        explicit_invalid = set()
        previous = "0001-01-01"
        for row in data["series"]:
            if row.get("invalid_return") and not any(previous < day <= row["date"] for day in old_event_days):
                explicit_invalid.add(row["date"])
            row.pop("invalid_return", None)
            previous = row["date"]
        for day, rows in by_day.items():
            if day not in local:
                first = min(rows, key=lambda row: (timestamp(row["available_at"]), timestamp(row["observed_at"]),
                                                  row["source"], row["source_url"]))
                local[day] = {"date": day, "nav": str(first["value"]), "provenance": {
                    **{key: deepcopy(first[key]) for key in (
                        "currency", "nav_type", "source", "source_url", "observed_at", "available_at",
                        "published_at", "publication_precision", "locator", "basis",
                    ) if key in first},
                    "event_date": day, "unit_nav": str(first["value"]),
                    "source_markers": [], "corporate_events": [],
                }}
            local[day]["provenance"]["online_observations"] = rows
            for row in rows:
                if row.get("invalid_return"):
                    explicit_invalid.add(day)
                if row.get("label_valid") is False:
                    local[day]["label_valid"] = False
                if "unitMoney" in row:
                    markers.append({"event_date": day, "source": row["source"], "source_url": row["source_url"],
                                    "unit_money_present": True, "unitMoney": deepcopy(row["unitMoney"]),
                                    "nonempty": row["unitMoney"] not in (None, "")})
        series = [local[day] for day in sorted(local)]
        events = old_events
        if series and (not data["series"] or series[0]["date"] != data["series"][0]["date"]
                       or series[-1]["date"] != data["series"][-1]["date"]):
            versions = {}
            for kind in ("dividend", "split", "announcement"):
                for fact in self.store.facts(product.id, kind, as_of=as_of):
                    identity = kind, fact.event_date, fact.source, fact.source_url
                    prior = versions.get(identity)
                    if prior is None or (timestamp(fact.available_at), timestamp(fact.observed_at), fact.id) > (
                            timestamp(prior.available_at), timestamp(prior.observed_at), prior.id):
                        versions[identity] = fact
            events = [{"fact_id": fact.id, "kind": fact.kind, "event_date": fact.event_date,
                       "effective_date": str(fact.payload.get("ex_date", fact.event_date)),
                       "payload": dict(fact.payload), "source": fact.source, "source_url": fact.source_url,
                       "observed_at": fact.observed_at, "available_at": fact.available_at}
                      for fact in versions.values()
                      if series[0]["date"] <= str(fact.payload.get("ex_date", fact.event_date)) <= series[-1]["date"]]
        markers = [item for _, item in sorted({canonical_json(row): row for row in markers}.items())]
        event_days = {row["event_date"] for row in markers if row["nonempty"]}
        event_days.update(row["effective_date"] for row in events if row["kind"] in {"dividend", "split"})
        previous = "0001-01-01"
        for row in series:
            row["provenance"]["source_markers"] = [item for item in markers if item["event_date"] == row["date"]]
            row["provenance"]["corporate_events"] = [item for item in events
                                                       if previous < item["effective_date"] <= row["date"]]
            if row["date"] in explicit_invalid or any(previous < day <= row["date"] for day in event_days):
                row["invalid_return"] = True
            previous = row["date"]
        data.update(series=series, basis_hash=digest(series))
        if series:
            latest = series[-1]
            if latest["date"] != local_anchor_date:
                data["anchor_currency"] = latest["provenance"].get("currency")
            data["nav"].update(observation_count=len(series), first_date=series[0]["date"],
                               latest_date=latest["date"], latest_nav=latest["nav"],
                               latest_available_at=latest["provenance"]["available_at"],
                               economic_nav_age_days=(point.date() - date.fromisoformat(latest["date"])).days)
        data["nav"].update(basis_hash=data["basis_hash"], corporate_events=events,
                           unit_money={"records": len(markers), "nonempty": sum(row["nonempty"] for row in markers),
                                       "markers": markers, "coverage": "source_fields_only_not_official_complete"})
        return data

    def timing_inputs(
        self, product_id: str, *, context: Mapping | None = None,
        model_id: str = "nav_ridge_empirical_v2", calendar_days: int | None = None,
        decision_at: str | None = None,
    ) -> dict[str, Any]:
        """Read-only inputs; an explicit captured clock keeps lookup and fit on one plan."""
        model_id = self._forecast_model(model_id)
        point = timestamp(decision_at if decision_at is not None else now_iso()).isoformat()
        product = self.store.get_product(product_id)
        if product is None and isinstance(context, Mapping):
            from .buyer_research import current_public_identity
            identity = next((row for row in context.get("public_subjects", [])
                             if row.get("id", row.get("product_id")) == product_id), None)
            if identity:
                parsed = current_public_identity({**identity, "id": product_id})
                product = Product(**{key: parsed[key] for key in Product.__dataclass_fields__ if key in parsed})
        result = self._forecast_wrapper(product, point)
        result.update(selected_model_id=model_id, series=[], entry_assumptions=None,
                      cache_key=digest({"product_id": product_id, "model_id": model_id}),
                      calculation={"kind": "local_hypothetical", "persisted": False, "actual_issue": False,
                                   "historical_availability_basis": "one_observation_lag_reconstruction",
                                   "fact_timestamps_rewritten": False})
        if calendar_days is not None:
            from .forecast import LOCAL_TIMING_CONTRACT
            from .trade_calendar import build_trade_plan

            result["trade_plan"] = build_trade_plan(
                decision_at=point, anchor_date=None, calendar_days=calendar_days,
            )
            result["calculation"].update(local_contract=LOCAL_TIMING_CONTRACT,
                                         target_unit="nav_observations")
        if product is None:
            result["reasons"] = ["尚无该公开基金的本地身份或本批次完整RAW在线资料；没有自动登记。"]
            return result
        if self.mode != "public" or product.is_demo or product.kind != "open_fund":
            result.update(status="unsupported", reasons=["择时试算只使用公开场外基金单位净值。"])
            return result
        investment_model = model_id == "exposure_factor_ridge_empirical_v1"
        if investment_model and product.symbol != "018147":
            result.update(status="unsupported", reasons=["所选投资模型未核对这只基金的因素映射，不会替换成净值模型。"])
            return result
        try:
            data = self._timing_data(product, point, context)
            result.update(series=data["series"], evidence={"records": [], "nav": data["nav"]})
            if calendar_days is not None:
                plan, assumptions = self._calendar_trade_plan(
                    product, point, data["series"][-1]["date"] if data["series"] else None,
                    calendar_days,
                )
                result.update(trade_plan=plan, entry_assumptions=assumptions)
                result["evidence"]["nav"]["entry_assumptions"] = assumptions
            result["cache_key"] = digest({"product_id": product.id, "model_id": model_id,
                                         "basis_hash": data["basis_hash"], "nav": data["nav"],
                                         "reasons": data["reasons"]})
            if data["reasons"]:
                result.update(status="error", reasons=data["reasons"])
                return result
            if not data["series"]:
                result.update(status="insufficient", reasons=["暂无实际可用的单位净值序列；没有补值或联网更新。"])
                return result
            if calendar_days is not None and plan["status"] != "ready":
                result.update(status=plan["status"], reasons=plan["reasons"])
                return result
            investment = None
            if investment_model:
                try:
                    investment = self._investment_inputs(product, point)
                except ValidationError as exc:
                    result.update(status="insufficient", input_status="missing_investment_inputs", reasons=[str(exc)])
                    return result
            if calendar_days is None:
                request, assumptions = self._forecast_request(
                    product, point, data, model_id=model_id, investment=investment,
                )
            else:
                from .forecast import LOCAL_RUNTIME_MODEL_HASH, LOCAL_TIMING_CONTRACT

                contract_hash = LOCAL_RUNTIME_MODEL_HASH
                if investment_model:
                    from .investment_forecast import LOCAL_MODEL_HASH
                    contract_hash = LOCAL_MODEL_HASH
                identity = {
                    "product_id": product.id, "model_id": model_id,
                    "local_contract": LOCAL_TIMING_CONTRACT, "local_contract_hash": contract_hash,
                    "basis_hash": data["basis_hash"], "horizon": plan["horizon"],
                    "entry_lags": assumptions["entry_lags"],
                    "anchor_currency": data["anchor_currency"],
                }
                if investment is not None:
                    identity.update(factor_input_hash=investment["factor_input_hash"],
                                    profile_mapping_hash=investment["profile_mapping_hash"])
                request = digest(identity)
                result["local_request_identity"] = identity
                result["calculation"].update(
                    local_contract_hash=contract_hash, horizon=plan["horizon"],
                    entry_lags=assumptions["entry_lags"],
                )
            result.update(status="ready", entry_assumptions=assumptions, cache_key=request)
            result["calculation"].update(cache_key=request, basis_hash=data["basis_hash"],
                                         anchor_currency_explicit=data["anchor_currency"] is not None)
            result["evidence"]["nav"]["entry_assumptions"] = assumptions
            if investment is not None:
                result["investment_inputs"] = investment
            result["reasons"] = ["使用实际已取得的完整净值与原来源时钟；显式计算仅在内存拟合，不发行或保存预测。"]
        except (ValidationError, OSError, RuntimeError) as exc:
            result.update(status="error", reasons=[str(exc)])
        return result

    def calculate_timing(
        self, product_id: str, *, context: Mapping | None = None,
        model_id: str = "nav_ridge_empirical_v2", include_history: bool = False,
        include_curve: bool = False, calendar_days: int | None = None,
        decision_at: str | None = None,
    ) -> dict[str, Any]:
        """Explicit RAM-only hypothetical forecast; never update, issue, score or write."""
        if not isinstance(include_history, bool):
            raise ValidationError("include_history必须是布尔值")
        if not isinstance(include_curve, bool):
            raise ValidationError("include_curve必须是布尔值")
        captured_at = decision_at if decision_at is not None else now_iso()
        result = self.timing_inputs(
            product_id, context=context, model_id=model_id, calendar_days=calendar_days,
            decision_at=captured_at,
        )
        return self._calculate_prepared_timing(
            result, include_history=include_history, include_curve=include_curve,
        )

    @staticmethod
    def _calculate_prepared_timing(
        result: dict, *, include_history: bool = False, include_curve: bool = False,
    ) -> dict:
        """Fit only the captured inputs, sharing the original timing engine."""
        if not isinstance(include_history, bool) or not isinstance(include_curve, bool):
            raise ValidationError("timing calculation flags must be boolean")
        from .forecast import (
            fit_local_timing_forecast, fit_runtime_forecast, local_curve_horizons,
            reconstruct_investment_history,
        )

        product_id = result["product"]["id"] if result.get("product") else None
        calendar_days = (result.get("trade_plan") or {}).get("calendar_days")
        if include_history:
            result["historical_forecasts"] = []
        investment = result.pop("investment_inputs", None)
        if result["status"] != "ready":
            return result
        arguments = {"product_id": product_id, "generated_at": result["evaluated_at"],
                     "entry_lags": tuple(result["entry_assumptions"]["entry_lags"])}
        try:
            if calendar_days is not None:
                plan_horizon = result["trade_plan"]["horizon"]
                requested_horizons = (local_curve_horizons(plan_horizon)
                                      if include_curve else (plan_horizon,))
                arguments["horizons"] = requested_horizons
                result["calculation"].update(
                    include_curve=include_curve,
                    timestamp_kind="calculated",
                    requested_horizons=list(requested_horizons),
                )
                if include_curve:
                    identity = dict(result.get("local_request_identity") or {})
                    identity.update(include_curve=True, requested_horizons=list(requested_horizons))
                    result["local_request_identity"] = identity
                    result["cache_key"] = digest(identity)
                if investment is None:
                    prediction = fit_local_timing_forecast(
                        result["series"], include_history=include_history, **arguments,
                    )
                else:
                    from .investment_forecast import fit_local_investment_forecast
                    prediction = fit_local_investment_forecast(
                        result["series"], factors=investment["bundle"], profile=investment["profile"],
                        include_history=include_history, **arguments,
                    )
                if include_history:
                    result["historical_forecasts"] = prediction.pop("historical_forecasts")
            elif investment is None:
                prediction = fit_runtime_forecast(result["series"], include_history=include_history, **arguments)
                if include_history:
                    result["historical_forecasts"] = prediction.pop("historical_forecasts")
            else:
                from .investment_forecast import fit_investment_forecast
                from .investment_factors import normalize_factor_bundle
                try:
                    prediction = fit_investment_forecast(result["series"], factors=investment["bundle"],
                                                         profile=investment["profile"], **arguments)
                except ValidationError as exc:
                    if include_history and str(exc).startswith("insufficient_history:"):
                        result["historical_forecasts"] = reconstruct_investment_history(
                            result["series"], factors=normalize_factor_bundle(investment["bundle"], as_of=result["evaluated_at"]),
                            entry_lags=arguments["entry_lags"])
                    raise
                if include_history:
                    result["historical_forecasts"] = reconstruct_investment_history(
                        result["series"], factors=prediction["factor_inputs"], entry_lags=arguments["entry_lags"])
            prediction.update(entry_assumptions=result["entry_assumptions"],
                              basis_hash=result["calculation"]["basis_hash"],
                              calculation=dict(result["calculation"]))
            if not result["calculation"]["anchor_currency_explicit"]:
                prediction["anchor"].pop("currency", None)
                prediction["anchor"].pop("currency_source", None)
            result["prediction"] = prediction
            if not prediction["forecasts"] and not prediction["execution_forecasts"]:
                result.update(status="insufficient", reasons=["尚无足够有效的已成熟标签；保留完整历史不可用状态，没有猜测预测。"])
        except (ValidationError, OSError, RuntimeError) as exc:
            result.update(status="insufficient" if str(exc).startswith("insufficient_history:") else "error",
                          reasons=[str(exc)])
        return result

    def stored_nav_series(
        self, product_id: str, *, as_of: str | None = None,
    ) -> dict[str, Any]:
        """Read existing public NAVs for a saved simulation; no acquisition or fit.

        Preserve the normal NAV provenance, availability times and invalid-event
        masks. This does not rebuild a saved choice, assess today's action, or
        write a historical score; the pure replay caller owns those semantics.
        """
        clock = timestamp(now_iso())
        point = timestamp(as_of) if as_of is not None else clock
        product = self.store.get_product(product_id)
        result = self._forecast_wrapper(product, point.isoformat())
        result.update(kind="stored_nav_series", series=[])
        if point > clock:
            result.update(status="error", reasons=["不能提前查看实际时钟之后才可用的净值资料。"])
            return result
        if product is None:
            result["reasons"] = ["本地没有该公开基金身份；只读复盘不会自动登记或取材。"]
            return result
        if self.mode != "public" or product.is_demo or product.kind != "open_fund":
            result.update(status="unsupported", reasons=["只读模拟复盘只使用本地公开场外基金单位净值。"])
            return result
        try:
            data = self._forecast_data(product, point.isoformat())
            result.update(series=data["series"], basis_hash=data["basis_hash"],
                          evidence={"records": [], "nav": data["nav"]})
            if data["reasons"]:
                result.update(status="error", reasons=data["reasons"])
            elif not data["series"]:
                result.update(status="insufficient", reasons=["本地尚无实际可用净值；没有联网或补造数据。"])
            else:
                result.update(status="ready", reasons=["仅使用截止查询时刻已保存且实际可用的净值与权益标记。"])
        except (ValidationError, OSError, RuntimeError) as exc:
            result.update(status="error", reasons=[str(exc)])
        return result

    def sale_timing_inputs(
        self, product_id: str, *, calendar_days: int, old_confirmation_date=None,
        fee_record: Mapping | None = None, current_redemption_override=None,
        future_old_redemption_override=None, base_amount=10000,
        context: Mapping | None = None, model_id: str = "nav_ridge_empirical_v2",
        decision_at: str | None = None,
    ) -> dict[str, Any]:
        """Read one input/calendar snapshot; edits only prepare exact sale requests."""
        from copy import deepcopy

        inputs = self.timing_inputs(
            product_id, context=context, model_id=model_id, calendar_days=calendar_days,
            decision_at=decision_at,
        )
        inputs["sale_options"] = {
            "fee_record": deepcopy(fee_record), "old_confirmation_date": old_confirmation_date,
            "current_redemption_override": current_redemption_override,
            "future_old_redemption_override": future_old_redemption_override,
            "base_amount": base_amount,
        }
        return self.rebuild_sale_timing(inputs)

    @staticmethod
    def rebuild_sale_timing(
        inputs: Mapping, *, cached_predictions: Mapping | None = None,
    ) -> dict[str, Any]:
        """Pure lookup/recalculation; cache values may be timing reports or predictions.

        ``sale_requests`` carry the SAME per-H numeric keys as calculate_timing.
        Fees, confirmation hypotheses, projected dates and ticking seconds do not
        change a key unless they change the actual model inputs or exact H/L.
        Only predictions under those explicit keys are considered. This function
        never fits, reads a clock, accesses a store or mutates the input/cache.
        """
        from .timing_scenario import build_sale_date_comparison

        result = dict(inputs)
        options = dict(inputs["sale_options"])
        plan = inputs["trade_plan"]
        comparison = build_sale_date_comparison(plan=plan, **options)
        requests = {}
        identity = inputs.get("local_request_identity")
        if inputs.get("status") == "ready" and identity is not None:
            for row in comparison["rows"]:
                horizon, lag = row["horizon"], row["entry_lag"]
                if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
                        or isinstance(lag, bool) or not isinstance(lag, int) or lag < 0):
                    continue
                key = digest({**identity, "horizon": horizon})
                request = requests.setdefault(key, {
                    "cache_key": key, "horizon": horizon, "entry_lag": lag,
                    "entry_lags": list(identity["entry_lags"]), "candidate_ids": [], "sale_dates": [],
                })
                request["candidate_ids"].extend(row["candidate_ids"])
                request["sale_dates"].append(row["sale_date"])
        cached = cached_predictions or {}
        predictions, reports = {}, {}
        for key, request in requests.items():
            entry = cached.get(key)
            is_report = isinstance(entry, Mapping) and "prediction" in entry
            raw = entry.get("prediction") if is_report else entry
            # A cached failed/unavailable attempt is a result too. An ordinary
            # edit must not repeatedly refit the same unqualified exact H.
            request["cached"] = is_report or isinstance(raw, Mapping)
            if is_report:
                reports[key] = entry
            if isinstance(raw, Mapping):
                predictions[key] = raw
        comparison = build_sale_date_comparison(tuple(predictions.values()), plan=plan, **options)
        for row in comparison["rows"]:
            request = next((item for item in requests.values()
                            if item["horizon"] == row["horizon"] and item["entry_lag"] == row["entry_lag"]), None)
            row["cache_key"] = request["cache_key"] if request is not None else None
            row["forecast_cached"] = request["cached"] if request is not None else False
            report = reports.get(row["cache_key"])
            if report is not None and row["primary_head"] is None:
                row["reasons"] = list(dict.fromkeys([*row["reasons"], *(report.get("reasons") or [])]))
        result.update(sale_comparison=comparison, sale_requests=list(requests.values()),
                      sale_predictions=predictions, sale_reports=reports)
        return result

    def calculate_sale_timing(
        self, inputs: Mapping, *, cached_predictions: Mapping | None = None,
    ) -> dict[str, Any]:
        """Explicit RAM-only sale comparison using the already captured inputs.

        Fit each missing positive H once with the existing exact-H fitter; never
        reread a calendar/NAV, issue forecasts, write, acquire research or mix
        candidate samples. Returned sale_reports can populate the UI's ordinary
        timing cache. Unavailable exact heads remain unavailable, not backfilled.
        """
        prepared = self.rebuild_sale_timing(inputs, cached_predictions=cached_predictions)
        cached = dict(cached_predictions or {})
        fitted = []
        if prepared.get("status") == "ready":
            for request in prepared["sale_requests"]:
                key = request["cache_key"]
                if request["cached"]:
                    continue
                report = {name: value for name, value in inputs.items() if not name.startswith("sale_")}
                horizon, day = request["horizon"], request["sale_dates"][0]
                original_plan = inputs["trade_plan"]
                confirmation = original_plan.get("new_confirmation_date")
                report.update(
                    prediction=None, cache_key=key,
                    trade_plan={**original_plan, "sale_date": day, "horizon": horizon,
                                "new_shares_sellable": bool(confirmation and
                                    original_plan["application_date"] <= confirmation <= day)},
                    entry_assumptions={**inputs["entry_assumptions"], "horizon": horizon},
                    calculation={**inputs["calculation"], "horizon": horizon, "cache_key": key},
                    local_request_identity={**inputs["local_request_identity"], "horizon": horizon},
                )
                cached[key] = self._calculate_prepared_timing(report)
                fitted.append(key)
        result = self.rebuild_sale_timing(inputs, cached_predictions=cached)
        result["sale_calculation"] = {
            "kind": "local_hypothetical", "persisted": False, "actual_issue": False,
            "decision_at": inputs.get("evaluated_at"), "fitted_cache_keys": fitted,
            "fit_count": len(fitted), "reused_count": sum(row["cached"] for row in prepared["sale_requests"]),
        }
        return result

    def _forecast_product(self, subject: str | Mapping[str, Any]) -> Product:
        if isinstance(subject, str):
            product = self.store.get_product(subject)
            if product is None:
                raise ValidationError("观察产品不存在；新产品请提供公开身份描述")
            return product
        return self._resolve_public_product(subject, register=True)

    def update_forecast(
        self, subject: str | Mapping[str, Any], *, refresh: bool = True,
        model_id: str = "nav_ridge_empirical_v2",
    ) -> dict[str, Any]:
        """Explicit refresh-once/local fit; unchanged requests reuse immutable records."""
        if self.mode != "public" or self.store.read_only:
            raise ValidationError("预测更新仅支持可写public公开库；不会使用演示资料")
        if not isinstance(refresh, bool):
            raise ValidationError("refresh必须是布尔值")
        model_id = self._forecast_model(model_id)
        product = self._forecast_product(subject)
        investment_model = model_id == "exposure_factor_ridge_empirical_v1"
        if product.is_demo or product.kind != "open_fund" or (investment_model and product.symbol != "018147"):
            return self.forecast(product.id, model_id=model_id)
        refreshed = None
        factor_refresh = None
        if refresh:
            try:
                refreshed = self.refresh(product.id)
            except (ValidationError, OSError, RuntimeError) as exc:
                refreshed = {"status": "error", "inserted": 0, "warnings": [str(exc)]}
            if refreshed.get("status") not in {"ok", "partial", "cached"}:
                result = self.forecast(product.id, model_id=model_id)
                result.update(status="error", refresh={"nav": refreshed})
                result["reasons"].append("本次显式净值更新失败，没有把旧预测伪装成本次成功；可查看原始源错误。")
                return result
            if investment_model:
                try:
                    factor_refresh = self.refresh_investment_factors(product.id)
                except (ValidationError, OSError, RuntimeError) as exc:
                    factor_refresh = {"status": "error", "warnings": [str(exc)]}
                if factor_refresh.get("status") not in {"ready", "ok", "partial", "cached"}:
                    result = self.forecast(product.id, model_id=model_id)
                    result.update(status="error", refresh={"status": "error", "nav": refreshed, "factors": factor_refresh})
                    result["reasons"].append("投资因素更新失败，没有用旧因素冒充新输入；原预测仍可在历史查看。")
                    return result
                refreshed = {"status": "ok", "nav": refreshed, "factors": factor_refresh}
        evaluated_at = now_iso()
        result = self.forecast(product.id, evaluated_at, model_id=model_id)
        if refreshed is not None:
            result["refresh"] = refreshed
        newest = self.store.forecast_record(product.id, as_of="9999-12-31T00:00:00+08:00", model_id=model_id)
        if newest and timestamp(newest["generated_at"]) > timestamp(evaluated_at):
            result.update(status="error", reasons=["实际时钟早于已保存预测；没有补造过去发布时间或提前展示未来记录。"])
            return result
        if result["status"] in {"unsupported", "insufficient"}:
            return result
        if result["status"] == "error" and not (
                not refresh and result.get("refresh") and result["refresh"].get("status") == "error"):
            return result
        try:
            from .forecast import fit_runtime_forecast
            data = self._forecast_data(product, evaluated_at)
            investment = self._investment_inputs(product, evaluated_at) if investment_model else None
            # Existing issues can mature even when today's new fit cannot complete.
            self._score_forecasts(product.id, data["series"], now_iso())
            request, assumptions = self._forecast_request(product, evaluated_at, data,
                                                          model_id=model_id, investment=investment)
            existing = self.store.forecast_record(product.id, as_of=evaluated_at, request_hash=request)
            issued_at = None
            while existing is None:
                started_at = now_iso()
                investment = self._investment_inputs(product, started_at) if investment_model else None
                request, assumptions = self._forecast_request(product, started_at, data,
                                                              model_id=model_id, investment=investment)
                existing = self.store.forecast_record(product.id, as_of=started_at, request_hash=request)
                if existing is not None:
                    break
                arguments = {"product_id": product.id, "generated_at": started_at,
                             "entry_lags": tuple(assumptions["entry_lags"])}
                if investment_model:
                    from .investment_forecast import fit_investment_forecast
                    prediction = fit_investment_forecast(data["series"], factors=investment["bundle"],
                                                         profile=investment["profile"], **arguments)
                else:
                    prediction = fit_runtime_forecast(data["series"], **arguments)
                completed_at = now_iso()
                completed_investment = self._investment_inputs(product, completed_at) if investment_model else None
                completed_request, completed_assumptions = self._forecast_request(product, completed_at, data,
                    model_id=model_id, investment=completed_investment)
                if completed_request != request:
                    # Crossing a cutoff or disclosure date can change the request.
                    # Reuse or fit the new request; never issue the expired one.
                    continue
                issued_at, assumptions = completed_at, completed_assumptions
                prediction["generated_at"] = issued_at
                if data["anchor_currency"] is None:
                    # Legacy provenance defaults are not a source declaration
                    # of the currency for a new personal money scenario.
                    prediction["anchor"].pop("currency", None)
                    prediction["anchor"].pop("currency_source", None)
                record_id = digest({"product_id": product.id, "request_hash": request})
                prediction.update(entry_assumptions=assumptions, record_id=record_id,
                                  basis_hash=data["basis_hash"])
                existing = self.store.add_forecast_record({
                    "id": record_id, "product_id": product.id, "request_hash": request,
                    "generated_at": prediction["generated_at"], "model_id": prediction["model_id"],
                    "model_hash": prediction["model_hash"], "input_hash": prediction["input_hash"],
                    "basis_hash": data["basis_hash"], "prediction": prediction,
                    "series": data["series"], "entry_assumptions": assumptions,
                })
            scored_at = now_iso()
            result = self.forecast(product.id, scored_at, model_id=model_id)
            result["prediction"] = existing["prediction"]
            result["reused"] = existing["generated_at"] != issued_at
            if refreshed is not None:
                result["refresh"] = refreshed
            return result
        except (ValidationError, OSError, RuntimeError) as exc:
            result.update(status="insufficient" if str(exc).startswith("insufficient_history") else "error")
            result["reasons"] = [str(exc)]
            return result

    def _score_forecasts(self, product_id: str, series: list[dict], scored_at: str) -> None:
        """Append matured outcomes on explicit update, never on a display read."""
        positions = {row["date"]: index for index, row in enumerate(series)}
        completed = {(row["forecast_id"], row["horizon"], row["entry_lag"])
                     for row in self.store.forecast_outcomes(product_id, scored_at)}
        for record in self.store.forecast_records(product_id, scored_at):
            prediction = record["prediction"]
            anchor = positions.get(prediction["anchor"]["date"])
            if anchor is None:
                continue
            issued = timestamp(record["generated_at"])
            for head in [*prediction.get("forecasts", []), *prediction.get("execution_forecasts", [])]:
                horizon, lag = head["horizon"], head["entry_lag"]
                key = (record["id"], horizon, lag)
                entry_index, target_index = anchor + lag, anchor + lag + horizon
                if key in completed or target_index >= len(series):
                    continue
                if any(row.get("invalid_return") for row in series[entry_index + 1:target_index + 1]):
                    continue
                entry, target = series[entry_index], series[target_index]
                actual = float(decimal(target["nav"]) / decimal(entry["nav"]) - 1)
                # NAV event dates have day granularity: a same-day entry may have
                # begun before issuance, even when its raw observation arrived later.
                # Only a strictly later economic entry date is proven forward.
                eligible = (date.fromisoformat(entry["date"]) > issued.date()
                            and date.fromisoformat(target["date"]) >= issued.date()
                            and timestamp(target["provenance"]["available_at"]) > issued)
                probability = head["up_probability"]
                self.store.add_forecast_outcome({
                    "id": digest({"forecast_id": record["id"], "horizon": horizon, "entry_lag": lag}),
                    "forecast_id": record["id"], "product_id": product_id,
                    "horizon": horizon, "entry_lag": lag, "scored_at": scored_at,
                    "issued_at": record["generated_at"], "kind": "forward" if eligible else "nowcast",
                    "eligible_forward": eligible, "entry": entry, "target": target,
                    "actual_change": actual, "up_probability": probability,
                    "brier_score": (probability - int(actual > 0)) ** 2,
                    "interval_covered": head["lower"] <= actual <= head["upper"],
                    "input_hash": record["input_hash"],
                })
                completed.add(key)

    def forecast_research(self, product_id: str, output_dir: str | Path) -> dict:
        """Explicit offline evaluation, frozen JSON inputs/report, no DB writes."""
        from .forecast import MODEL_HASH, fit_forecast, forecast_input_hash
        from .forecast_research import study_actions
        product = self.store.get_product(product_id)
        if product is None or product.is_demo or product.kind != "open_fund":
            raise ValidationError("离线预测研究需要已登记的公开场外基金")
        output = Path(output_dir).resolve()
        if output.exists():
            raise ValidationError("研究输出目录已存在；不会覆盖冻结输入或报告")
        generated_at = now_iso()
        data = self._forecast_data(product, generated_at)
        if data["reasons"]:
            raise ValidationError("；".join(data["reasons"]))
        if not data["series"]:
            raise ValidationError("缺少本地实际可用净值")
        _, assumptions = self._forecast_request(product, generated_at, data)
        report = fit_forecast(data["series"], product_id=product.id, generated_at=generated_at,
                              entry_lags=tuple(sorted({*assumptions["entry_lags"], 2, 3, 4})),
                              include_evaluation=True)
        report["entry_assumptions"] = assumptions
        inputs = {"schema": "rmbfund.forecast.inputs.v1", "product": asdict(product),
                  "generated_at": generated_at, "model_hash": MODEL_HASH,
                  "input_hash": forecast_input_hash(data["series"]),
                  "series": data["series"], "entry_assumptions": assumptions,
                  "evidence": {"records": self.store.advice_evidence(product.id, generated_at), "nav": data["nav"]}}
        artifacts = {"inputs.json": inputs, "report.json": report, "metrics.json": report["validation"],
                     "actions.json": study_actions(data["series"], report["issued_forecasts"])}
        encoded = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                   for name, value in artifacts.items()}
        output.mkdir(parents=True, exist_ok=False)
        for name, payload in encoded.items():
            (output / name).write_text(payload, encoding="utf-8")
        return {"status": "ready", "output": str(output), "generated_at": generated_at,
                "input_hash": inputs["input_hash"], "model_hash": MODEL_HASH,
                "files": {name: str(output / name) for name in artifacts},
                "validation": report["validation"], "network": False, "database_writes": False,
                "warning": "历史冻结评估不是实际前向业绩；没有账户、交易或伪造历史发布时间。"}

    def export(self, output_dir: str | Path, as_of: str) -> dict[str, str]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        overview = self.overview(as_of)
        files = {}
        for name, rows in (("products", overview["products"]), ("alerts", overview["alerts"]),
                           ("sources", overview["sources"]), ("journal", self.notes())):
            path = directory / f"{name}.csv"
            path.write_text(csv_text(rows), encoding="utf-8")
            files[name] = str(path)
        manifest = {"mode": self.mode, "as_of": as_of, "exported_at": now_iso(),
                    "fact_ids": [fact.id for fact in self.store.facts(as_of=as_of)],
                    "warning": "描述性数据与纸面模拟，不是投资建议。未知字段不等于不存在风险。"}
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        files["manifest"] = str(path)
        for account in self.paper.accounts():
            account_id = account["id"]
            for name, rows in (("orders", self.paper.orders(account_id)),
                               ("ledger", self.paper.ledger(account_id))):
                path = directory / f"{account_id}-{name}.csv"
                path.write_text(csv_text(rows), encoding="utf-8")
                files[f"{account_id}-{name}"] = str(path)
        return files

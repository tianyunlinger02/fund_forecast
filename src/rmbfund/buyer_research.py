"""Public research configuration and pure projection into fund comparisons."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re

from .domain import ValidationError, now_iso, timestamp

_EMPTY_CONFIG = {"peer_sets": {}, "fund_metadata": {}, "source_plan": []}


def read_public_json(path: Path, *, maximum: int = 2 * 1024 * 1024):
    if path.is_symlink() or path.name.startswith(".env") or path.stat().st_size > maximum:
        raise ValidationError("只接受有界普通公开JSON文件")
    return json.loads(path.read_text(encoding="utf-8"))


def _config(value) -> dict:
    if not isinstance(value, dict) or value.keys() - (_EMPTY_CONFIG.keys() | {"current_source_plan"}):
        raise ValidationError("买卖研究配置仅接受公开候选集合、身份来源和采集计划")
    if not isinstance(value.get("peer_sets", {}), Mapping):
        raise ValidationError("peer_sets必须是按目标产品标识索引的对象")
    if not isinstance(value.get("fund_metadata", {}), Mapping):
        raise ValidationError("fund_metadata必须是按产品标识索引的对象")
    if not isinstance(value.get("source_plan", []), list):
        raise ValidationError("source_plan必须是列表")
    for product_id, peer_set in value.get("peer_sets", {}).items():
        if not isinstance(product_id, str) or not isinstance(peer_set, Mapping):
            raise ValidationError("peer_sets中的每个候选集合必须是对象")
        from .fund_comparison import PeerSet
        parsed = PeerSet.from_mapping(peer_set)
        if parsed.subject_id != product_id:
            raise ValidationError("候选集合目标与配置索引不符")
    for metadata in value.get("fund_metadata", {}).values():
        if not isinstance(metadata, Mapping):
            raise ValidationError("每个基金的公开身份来源必须是对象")
    for field in ("source_plan", "current_source_plan"):
        if field in value and not isinstance(value[field], list):
            raise ValidationError(f"{field}必须是列表")
        for job in value.get(field, []):
            if not isinstance(job, Mapping) or not isinstance(job.get("product_id"), str):
                raise ValidationError(f"{field}中的每项必须有文本product_id")
    return value


def load_buyer_config(root: Path, *, as_of: str | None = None) -> dict:
    versions = root / "buyer-config-versions"
    if versions.exists():
        point = timestamp(as_of or now_iso())
        visible = [read_public_json(path) for path in versions.glob("*.json")]
        visible = [row for row in visible if timestamp(row["available_at"]) <= point]
        if not visible:
            return deepcopy(_EMPTY_CONFIG)
        return _config(max(visible, key=lambda row: timestamp(row["available_at"]))["config"])
    path = root / "buyer-config.json"
    if not path.exists():
        return deepcopy(_EMPTY_CONFIG)
    # Unversioned legacy configuration cannot establish historical availability.
    if as_of is not None:
        return deepcopy(_EMPTY_CONFIG)
    return _config(read_public_json(path))


def install_buyer_config(root: Path, config: Mapping, *, replace: bool = False) -> dict:
    import uuid
    value = _config(dict(config))
    target = root / "buyer-config.json"
    if target.exists():
        read_public_json(target)
        if not replace:
            raise ValidationError("配置已存在；需明确--replace，旧版本仍保留")
    versions = root / "buyer-config-versions"
    versions.mkdir(parents=True, exist_ok=True)
    available = now_iso()
    version = versions / (uuid.uuid4().hex + ".json")
    version.write_text(json.dumps({"available_at": available, "config": value},
                                 ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"status": "installed", "path": str(target), "version": str(version),
            "available_at": available, "network": False, "personal_data": False}


def followed_state(root: Path) -> dict:
    path = root / "followed-state.json"
    return read_public_json(path) if path.exists() else {"hashes": {}, "blocked_urls": [], "updates": []}


def followed_review(thesis: Mapping, state: Mapping, *, product_id: str, as_of: str,
                    records=()) -> dict:
    """A later unchanged fetch cannot erase an unreviewed change to a cited source."""
    result = deepcopy(dict(thesis))
    point = timestamp(as_of)
    payload = result.get("payload") or {}
    citations = payload.get("citations") or []
    from .research_thesis import research_explained_record_ids
    record_index = {row["id"]: row for row in records}
    explained = research_explained_record_ids(payload)
    if payload.get("schema_version") == 2:
        fact_ids = set(payload.get("change_evaluation", {}).get("fact_ids", []))
        citation_ids = {identifier for row in payload.get("facts", []) if row["id"] in fact_ids
                        for identifier in row["citation_ids"]}
        citations = [row for row in citations if row["id"] in citation_ids]
    else:
        explained = {row["record_id"] for row in citations if row.get("kind") == "archived_evidence"}
    latest = {}
    for update in state.get("updates", []):
        if (update["product_id"] == product_id and update["family"] != "nav"
                and timestamp(update["completed_at"]) <= point):
            latest[update["job_id"]] = update
    pending = []
    for update in latest.values():
        changed_at = update.get("unreviewed_change_at")
        if not changed_at:
            continue
        acknowledged = any(
            (citation.get("publisher_url") or citation.get("source_url")) == update["source_url"]
            and (citation.get("checked_at") or citation.get("available_at"))
            and timestamp(citation.get("checked_at") or citation["available_at"]) >= timestamp(changed_at)
            for citation in citations
        ) or any(
            identifier in record_index
            and record_index[identifier]["source_url"] == update["source_url"]
            and timestamp(record_index[identifier]["available_at"]) >= timestamp(changed_at)
            for identifier in explained
        )
        # A new authored review must actually postdate the changed source.
        authored_at = (result.get("source") or {}).get("available_at")
        if not acknowledged or not authored_at or timestamp(authored_at) < timestamp(changed_at):
            pending.append("相关来源有未复核变化：" + update["source_url"])
    result["needed_review"] = list(dict.fromkeys([*(result.get("needed_review") or []), *pending]))
    result["source_updates"] = list(latest.values())
    return result


def current_public_identity(subject: Mapping | str) -> dict:
    """A code/class identity for a current query, independent of registration."""
    if isinstance(subject, str):
        parts = subject.split(":")
        if len(parts) != 3 or parts[0] != "open_fund":
            raise ValidationError("当前查询产品标识须为open_fund:代码:份额类别")
        subject = {"kind": parts[0], "symbol": parts[1], "share_class": parts[2]}
    symbol = str(subject.get("symbol") or "").strip()
    share = str(subject.get("share_class") or "").strip()
    if (subject.get("kind", "open_fund") != "open_fund" or subject.get("is_demo", False)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,30}", symbol)):
        raise ValidationError("当前在线查询需要公开场外基金代码，不能使用演示产品")
    identifier = f"open_fund:{symbol}:{share}"
    if subject.get("id", subject.get("product_id", identifier)) != identifier:
        raise ValidationError("基金标识与代码或份额类别不符")
    return {"id": identifier, "symbol": symbol, "name": symbol, "kind": "open_fund",
            "share_class": share, "exchange": str(subject.get("exchange") or ""), "is_demo": False}


def compare_current_details(product_id: str, details: Mapping, context: Mapping, *,
                            peer_set: Mapping | None, metadata: Mapping | None = None) -> dict:
    """Compare this retrieval's NAV only, never a mixed online/local window."""
    from .fund_comparison import PeerSet, compare_funds
    if not peer_set:
        return {"status": "missing", "comparisons": [], "basis": "current_online",
                "reasons": ["没有经过核对的替代候选集合"]}
    peers = PeerSet.from_mapping(peer_set).to_dict()
    historical_interval = {key: peers[key] for key in ("start_date", "end_date")}
    peers.update(start_date=None, end_date=None)
    funds = {}
    sources = {}
    for identifier, detail in details.items():
        current = deepcopy(dict(detail))
        current["history"] = list(detail.get("current_nav_history") or [])
        current["latest"] = max(current["history"], key=lambda row: row["date"], default=None)
        if current.get("product"):
            funds[identifier] = comparison_fund(current, (metadata or {}).get(identifier))
        sources[identifier] = {"nav_origin": "current_online" if current["history"] else "missing",
                               "observations": len(current["history"]),
                               "source_checks": [deepcopy(row) for row in context.get("source_checks", [])
                                                 if row.get("product_id") == identifier and row.get("family") == "nav"]}
    result = compare_funds(peers, funds, as_of=context["evaluated_at"])
    result.update(basis="current_online", checked_at=context.get("checked_at"),
                  source_context=sources,
                  window_policy="仅用本次在线取得且核对通过的共同单位净值日期；不混入旧本地序列。")
    if any(row.get("status") != "available" for row in result.get("comparisons", [])):
        local_funds = {key: comparison_fund(value, (metadata or {}).get(key))
                       for key, value in details.items() if value.get("product")}
        historical_peers = dict(peers, **historical_interval)
        result["historical_reference"] = {
            "label": "带原日期的历史参考，不是本次完整在线比较",
            "comparison": compare_funds(historical_peers, local_funds, as_of=context["evaluated_at"]),
        }
    return result


def comparison_fund(detail: Mapping, metadata: Mapping | None = None) -> dict:
    """Do not invent a currency, underlying identity or security mapping."""
    metadata = metadata or {}
    public = detail.get("product") or {}
    product = {key: public[key] for key in ("id", "symbol", "name", "kind", "share_class", "exchange") if key in public}
    for key in ("currency", "underlying_fund_id", "mandate_id", "evidence_refs"):
        if key in metadata:
            product[key] = deepcopy(metadata[key])
    history = detail.get("history") or []
    currencies = {row.get("currency") for row in history}
    currency = next(iter(currencies)) if len(currencies) == 1 else None
    latest = detail.get("latest") or {}
    refs = [latest["source_url"]] if latest.get("source_url") else []
    result = {"product": product}
    if history and refs and currency:
        result["series"] = {"status": "available", "product_id": product.get("id"),
            "share_class": product.get("share_class"), "currency": currency,
            "basis": "unit_nav", "evidence_refs": refs,
            "observations": [{"date": row["date"], "value": row["value"]} for row in history]}
    else:
        result["series"] = {"status": "missing", "reason": "缺少同币种且带来源的实际单位净值"}
    for record in detail.get("background_records") or []:
        if record["family"] != "fees":
            continue
        items = []
        for key, title in (("management_rate", "管理费"), ("custody_rate", "托管费"), ("sales_service_rate", "销售服务费")):
            if record["payload"].get(key) is not None:
                items.append({"name": title, "rate": record["payload"][key], "basis": "annual_nav"})
        payload = record["payload"]
        if items or payload.get("subscription_rate") is not None or payload.get("redemption_bands"):
            result["fees"] = {"status": "available", "as_of": record["report_date"],
                "evidence_refs": [record["id"], record["source_url"]], "items": items,
                **{key: deepcopy(payload[key]) for key in
                   ("subscription_rate", "redemption_bands", "holding_day_basis", "channel_verified")
                   if key in payload},
                **{key: record[key] for key in ("observed_at", "available_at")}}
    # Existing profile issuer IDs cannot establish security-level identity.
    result["exposure"] = {"status": "missing", "reason": "现有披露为发行人级映射，未据名称猜测相同证券"}
    return result

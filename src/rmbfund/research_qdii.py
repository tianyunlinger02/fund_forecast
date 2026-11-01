"""An illustrative RMB US-equity QDII sample, never an Alipay account connector."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .domain import Product, now_iso
from .providers.akshare import EastmoneyProfileProvider, _js_millis_date
from .research_sources import _fetch, _parse_action_tables, _TextParser, _write_json

SAMPLE = Product("research:270042:RMB-A", "270042", "广发纳斯达克100ETF联接人民币(QDII)A · 研究样本非实际持仓", "open_fund",
                 category="美股方向QDII研究样本", share_class="人民币A")


def acquire_qdii_sample(output_dir: Path, *, start: str, end: str) -> dict[str, Any]:
    root = output_dir / "qdii-sources"
    root.mkdir()
    evidence: list[dict[str, Any]] = []
    warnings = [
        "270042 仅为同类型公开样本，不代表用户持仓，不保证支付宝当前可买，不作为推荐。",
        "QDII 使用人民币份额单位净值；不重复叠加汇率收益或扣净值内管理/托管费用。",
        "QDII 历史开放/限额/两地假期、实际净值披露时间及支付宝费率未核验；全部开放是假设，不是实际可执行证明。",
        "QDII 假设净值延迟2个观测日可知、申请后3个观测日确认、赎回后7个观测日到账；不代表实际产品规则。",
    ]
    directory = root / "eastmoney-nav-270042"
    directory.mkdir()
    result = EastmoneyProfileProvider(timeout=20, retries=1, min_interval=1).fetch(SAMPLE, now_iso(), end_date=end)
    body = _write_json(directory / "provider_result.json", result.to_dict())
    evidence.append({"kind": "nav_provider_result", "source_url": result.source_url,
                     "observed_at": now_iso(), "status": result.status,
                     "path": str((directory / "provider_result.json").resolve()),
                     "content_hash": hashlib.sha256(body).hexdigest(), "fact_count": len(result.facts)})
    warnings.extend(result.warnings)
    recent = [item["record"] for item in result.raw_records if "record" in item
              and start <= _js_millis_date(item["record"]["x"]) <= end]
    check = {"recent_rows": len(recent), "all_have_unitMoney": bool(recent) and all("unitMoney" in row for row in recent),
             "nonempty": sum(row.get("unitMoney") not in (None, "") for row in recent)}
    url = "https://fundf10.eastmoney.com/fhsp_270042.html"
    raw, item, directory = _fetch(root, "eastmoney-actions-270042", url, "raw.html",
                                  kind="actions_html", source="eastmoney:fhsp_html",
                                  evidence=evidence, warnings=warnings)
    actions, coverage = [], False
    if raw is not None:
        try:
            actions, details = _parse_action_tables(raw, symbol="270042", source_url=url,
                                                    observed_at=item["observed_at"], start=start, end=end)
            coverage = check["all_have_unitMoney"] and check["nonempty"] == 0
            item.update(details, status="parsed_recent_window", coverage_start=start, coverage_end=end)
            warnings.append("QDII 源表2014/2016登记日在除息日之后；原值保留HTML，近两年以外不修复或参与现金流。")
        except (ValueError, TypeError, KeyError) as exc:
            item.update(status="parse_error", error=str(exc))
            warnings.append(f"QDII 权益表未完整解析：{exc}；只允许price_only模型。")
        _write_json(directory / "metadata.json", item)
    raw, fee, directory = _fetch(root, "eastmoney-fees-270042", "https://fundf10.eastmoney.com/jjfl_270042.html", "raw.html",
                                kind="secondary_fee_status_html", source="eastmoney:fee_html",
                                evidence=evidence, warnings=warnings)
    if raw is not None:
        parser = _TextParser()
        parser.feed(raw.decode("utf-8", "strict"))
        text = " ".join(parser.parts)
        markers = {marker: marker in text for marker in ("1.30%", "0.13%", "1.50%", "0.50%", "0.30%", "暂停申购")}
        fee.update(markers=markers, historical_applicability_verified=False, alipay_rates_verified=False)
        if markers["暂停申购"]:
            warnings.append("本次东方财富页面显示270042暂停申购；不能把历史全开放情景理解为当前可买。")
        _write_json(directory / "metadata.json", fee)
    _write_json(root / "manifest.json", {"product": SAMPLE.__dict__, "evidence": evidence,
                 "unit_money_check": check, "action_coverage": coverage, "warnings": warnings})
    return {"product": SAMPLE, "facts": result.facts, "actions": actions,
            "action_coverage": coverage, "evidence": evidence, "warnings": warnings,
            "unit_money_check": check}

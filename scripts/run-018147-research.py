"""Run one predeclared, capped 018147 C-class historical comparison.

Run from project code with python -I; downloaded data stays in its own fresh
folders. Reproduce offline via rmbfund research --frozen <out>/inputs.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rmbfund.domain import Product, ValidationError, now_iso
from rmbfund.providers.akshare import EastmoneyProfileProvider, _js_millis_date
from rmbfund.research_replay import Assumptions, price_inputs, public_snapshot, run_research_replay
from rmbfund.research_sources import _fetch, _parse_action_tables, _TextParser, _write_json

START = "2024-10-21"
END = "2026-10-16"
PRODUCT = Product("research:018147:C", "018147", "建信新兴市场混合(QDII)C", "open_fund",
                  category="新兴市场偏股混合QDII", share_class="C")
MODEL = Assumptions(initial_cash="100", fund_subscription_rate="0", fund_daily_purchase_cap="100", fund_purchase_min="10",
                    fund_redemption_tiers=[{"days_lt": 7, "rate": "0.015"},
                                           {"days_lt": 30, "rate": "0.005"},
                                           {"days_lt": None, "rate": "0"}],
                    nav_lag_days=2, confirm_days=2, redemption_cash_days=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="018147 C类：现金/持有/MA20，单日申购不超过100元")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/public.sqlite3"))
    args = parser.parse_args()
    root = args.out.resolve()
    if root.exists():
        raise ValidationError("输出目录已存在，拒绝覆盖既有实验")
    before = public_snapshot(args.db)
    root.mkdir(parents=True)
    evidence, warnings = [], [
        "真实历史数据上的条件研究回放，不是支付宝实际持仓回测、不是真实成交或前向实绩。",
        "018147是新兴市场偏股混合QDII C类，不是纳斯达克/标普500指数基金。",
        "固定窗口2024-10-21至2026-10-16、MA20及60个暖机观测，不调参、不按收益选择窗口。",
        "初始100元、单次申购至少10元且每日最多100元；这些是参考当前二手页的固定情景，不证明历史或支付宝实际限额。",
        "每轮有仓不补仓；盈利卖出后再买也最多100元，超出部分留现金，不假装收益可无限按本金缩放。",
        "只在已有NAV观测日决策；NAV假设延迟2个观测日09:00可知，09:05信号、14:00未知价申请。",
        "申请后2个观测日确认；赎回7个观测日到账为主情景、10个观测日为压力情景，均为假设而非实际保证，观测日不是已核验业务日历。",
        "历史假期、暂停、限额变更与实际披露时刻未完整核验；不能证明每笔历史申请都可受理。",
        "申购0；赎回小于7自然日1.5%、7至未满30日0.5%、满30日0；当前费用表非历史支付宝逐日核验。",
        "持有期从假设确认日计算，不保证与销售渠道持有期算法完全相同。",
        "NAV已含管理/托管/销售服务费与资产汇率折算，不重复扣费或叠加汇率收益。",
        "日级资产按经济日NAV回看估值，决策不使用申请日未知NAV；在途与应收不能用于新订单。",
        "现金利息0，期末不强制赎回，不含未来退出费用；基金份额向下截断至两位，舍入损耗单列。",
        "公开二手权益表解析成功不等于管理人全部公告完整核验，若权益不齐只给price_only。",
    ]
    navdir = root / "nav-source"
    navdir.mkdir()
    result = EastmoneyProfileProvider(timeout=20, retries=1, min_interval=1).fetch(PRODUCT, now_iso(), end_date=END)
    raw = _write_json(navdir / "provider_result.json", result.to_dict())
    evidence.append({"source": result.provider, "source_url": result.source_url, "status": result.status,
                     "observed_at": now_iso(), "path": str(navdir / "provider_result.json"),
                     "content_hash": hashlib.sha256(raw).hexdigest(), "facts": len(result.facts)})
    if not result.facts:
        raise ValidationError("未取得真实净值，保留失败证据，不生成模拟收益")
    rows = price_inputs(PRODUCT, result.facts, start=START, end=END, warmup=MODEL.warmup,
                        nav_lag_days=MODEL.nav_lag_days)
    action_url = "https://fundf10.eastmoney.com/fhsp_018147.html"
    body, item, directory = _fetch(root, "actions", action_url, "raw.html", kind="actions",
                                  source="eastmoney:fhsp_html", evidence=evidence, warnings=warnings)
    actions, coverage = [], False
    observed = [entry["record"] for entry in result.raw_records if "record" in entry
                and START <= _js_millis_date(entry["record"]["x"]) <= END]
    marks = {"records": len(observed), "nonempty_unitMoney": sum(row.get("unitMoney") not in (None, "") for row in observed)}
    if body is not None:
        try:
            actions, detail = _parse_action_tables(body, symbol="018147", source_url=action_url,
                                                  observed_at=item["observed_at"], start=START, end=END)
            coverage = bool(observed) and all("unitMoney" in row for row in observed) and marks["nonempty_unitMoney"] == 0
            item.update(detail, status="parsed", coverage=coverage)
        except (ValueError, KeyError, TypeError) as exc:
            item.update(status="parse_error", error=str(exc))
            warnings.append(f"权益未完整解析：{exc}；不提供完整条件收益")
        _write_json(directory / "metadata.json", item)
    fee_url = "https://fundf10.eastmoney.com/jjfl_018147.html"
    body, item, directory = _fetch(root, "fees", fee_url, "raw.html", kind="secondary_fee_status",
                                  source="eastmoney:fee_html", evidence=evidence, warnings=warnings)
    if body is not None:
        text = _TextParser()
        text.feed(body.decode("utf-8", "strict"))
        cleaned = " ".join(" ".join(text.parts).split())
        found = {marker: marker in cleaned for marker in ("0.00%", "1.50%", "0.50%", "小于30天", "大于等于30天", "100.00元", "10.00元", "T+2")}
        item.update(markers=found, alipay_verified=False, historical_rule_versions_verified=False)
        _write_json(directory / "metadata.json", item)
        (directory / "readable.txt").write_text(cleaned, encoding="utf-8")
        if not all(found.values()):
            warnings.append("当前二手费率/限额页未匹配全部预期标记，费用仍仅为明确命名情景，须人工复核")
    inputs = {"mode": "research_replay", "warning": warnings[0], "start": START, "end": END,
              "assumptions": asdict(MODEL), "warnings": warnings + result.warnings,
              "source_evidence": evidence, "source_public_hash": before["hash"], "unit_money_check": marks,
              "products": [{"product": asdict(PRODUCT), "prices": rows, "actions": actions,
                            "action_coverage": coverage, "blocked": None,
                            "scenarios": [
                                {"id": "C_zero_entry_cap100_min10_lag2_confirm2_cash7", "assumptions": {}},
                                {"id": "C_zero_entry_cap100_min10_lag2_confirm2_cash10_stress",
                                 "assumptions": {"redemption_cash_days": 10}},
                            ]}]}
    _write_json(root / "inputs.json", inputs)
    report = run_research_replay(args.db, root / "results", frozen=root / "inputs.json")
    after = public_snapshot(args.db)
    if before["hash"] != after["hash"]:
        raise ValidationError("源库在运行期间变化，不能宣称只读一致")
    _write_json(root / "public-unchanged.json", {"before": before["hash"], "after": after["hash"], "unchanged": True})
    print(json.dumps({"out": str(root), "start": START, "end": END, "raw_nav_facts": len(result.facts),
                      "window_samples": len(rows) - MODEL.warmup, "action_coverage": coverage,
                      "assumptions": asdict(MODEL), "results": [{"strategy": r["strategy"], **r["metrics"]}
                                                                   for r in report["results"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

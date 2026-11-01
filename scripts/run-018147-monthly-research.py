"""Offline, fixed monthly-momentum comparison on two retained 018147 windows.

Run from project code with python -I. This never downloads or executes source data.
Every experiment uses a new output directory; retained evidence is read-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from rmbfund.domain import Fact, Product, ValidationError, decimal, digest, now_iso
from rmbfund.providers.akshare import _js_millis_date
from rmbfund.research_replay import (
    MONTHLY_DEFINITION,
    MONTHLY_STRATEGY,
    STRATEGIES,
    evaluate_frozen,
    price_inputs,
    public_snapshot,
)
from rmbfund.research_sources import _parse_action_tables

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / "runs/artifacts/research-018147-20261020T082509"
SEEDS = {
    "earlier": PROJECT / "runs/artifacts/research-018147-earlier-20261020/results-weekday",
    "later": ORIGINAL / "final-results",
}
WINDOWS = {"earlier": ("2023-07-03", "2024-06-28"), "later": ("2024-10-21", "2026-10-16")}
INDEX_FIELDS = {"confirmation_index", "cash_due_index", "due_index"}


def read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValidationError(f"Missing or oversized retained JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_indices(value: Any, offset: int) -> Any:
    if isinstance(value, dict):
        return {key: item - offset if key in INDEX_FIELDS and isinstance(item, int) and not isinstance(item, bool)
                else normalize_indices(item, offset) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_indices(item, offset) for item in value]
    return value


def command_log(directory: Path, name: str, arguments: list[str]) -> None:
    with (directory / f"{name}.stdout.txt").open("x", encoding="utf-8") as stdout:
        with (directory / f"{name}.stderr.txt").open("x", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [sys.executable, "-I", *arguments], cwd=PROJECT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, timeout=180, check=False,
            )
    write_json(directory / f"{name}.command.json", {
        "argv": [sys.executable, "-I", *arguments], "cwd": str(PROJECT),
        "returncode": completed.returncode,
    })
    if completed.returncode:
        raise ValidationError(f"{name} failed with exit {completed.returncode}; retained logs: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description="018147离线月度动量：两个原窗口、两到账情景、不调参")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=PROJECT / "data/public.sqlite3")
    args = parser.parse_args()
    root = args.out.resolve()
    if root.exists():
        raise ValidationError("输出目录已存在，拒绝覆盖任何旧试验")
    before = public_snapshot(args.db)
    seeds = {name: read_json(directory / "inputs.json") for name, directory in SEEDS.items()}
    originals = {name: read_json(directory / "report.json") for name, directory in SEEDS.items()}
    for name, seed in seeds.items():
        if (seed["start"], seed["end"]) != WINDOWS[name] or seed["source_public_hash"] != before["hash"]:
            raise ValidationError("原窗口或public来源指纹与固定声明不符")
        if digest(evaluate_frozen(seed)) != digest(originals[name]):
            raise ValidationError("旧冻结输入经济结果变化，停止而不执行新策略")
        for evidence in seed["source_evidence"]:
            if sha256(Path(evidence["path"])) != evidence["content_hash"]:
                raise ValidationError("来源证据字节变化，拒绝继续")
    if seeds["earlier"]["assumptions"] != seeds["later"]["assumptions"]:
        raise ValidationError("两原窗口的资金/执行假设不一致")
    if seeds["earlier"]["products"][0]["scenarios"] != seeds["later"]["products"][0]["scenarios"]:
        raise ValidationError("两原窗口到账情景不一致")
    assumptions = seeds["later"]["assumptions"]
    if any(assumptions[key] != value for key, value in {
        "initial_cash": "100", "fund_subscription_rate": "0", "fund_purchase_min": "10",
        "fund_daily_purchase_cap": "100", "nav_lag_days": 2, "confirm_days": 2,
        "redemption_cash_days": 7, "warmup": 60, "ma_period": 20,
    }.items()):
        raise ValidationError("原参数不符合本轮固定100元、lag2、min10/cap100的声明")
    if assumptions["fund_redemption_tiers"] != [
        {"days_lt": 7, "rate": "0.015"}, {"days_lt": 30, "rate": "0.005"}, {"days_lt": None, "rate": "0"},
    ]:
        raise ValidationError("原赎回费分档与固定声明不符")
    scenarios = seeds["later"]["products"][0]["scenarios"]
    if [scenario["assumptions"] for scenario in scenarios] != [{}, {"redemption_cash_days": 10}]:
        raise ValidationError("必须保留原7/10观测日到账两情景，不添加其它参数变更")
    nav_evidence = next(e for e in seeds["later"]["source_evidence"] if e["path"].endswith("nav-source/provider_result.json"))
    nav_path = Path(nav_evidence["path"])
    nav = read_json(nav_path)
    facts = [Fact(**item) for item in nav["facts"] if date.fromisoformat(item["event_date"]).weekday() < 5]
    weekends = sorted({item["event_date"] for item in nav["facts"] if date.fromisoformat(item["event_date"]).weekday() >= 5})
    strategy_list = [*STRATEGIES, MONTHLY_STRATEGY]
    root.mkdir(parents=True)
    write_json(root / "predeclared.json", {
        "created_at": now_iso(), "windows": WINDOWS, "strategies": strategy_list,
        "strategy_definitions": {MONTHLY_STRATEGY: MONTHLY_DEFINITION},
        "required_strategy_warmup": 62, "assumptions": assumptions,
        "scenarios": seeds["later"]["products"][0]["scenarios"],
        "source_nav_sha256": sha256(nav_path),
        "original_inputs": {name: {"path": str(SEEDS[name] / "inputs.json"), "hash": digest(seed)} for name, seed in seeds.items()},
        "declaration_written_before_this_run_strategy_execution": True,
        "warning": "New hypothesis on already-seen histories, not a sealed out-of-sample or forward result; no parameter search.",
    })
    prepared = {}
    for name, seed in seeds.items():
        directory = root / name
        directory.mkdir()
        inputs = copy.deepcopy(seed)
        item = inputs["products"][0]
        product = Product(**item["product"])
        if len(inputs["products"]) != 1 or product.symbol != "018147" or product.kind != "open_fund":
            raise ValidationError("本实验只接受原018147场外份额")
        rows = price_inputs(product, facts, start=seed["start"], end=seed["end"], warmup=62, nav_lag_days=2)
        if rows[2:] != item["prices"]:
            raise ValidationError("增加2条暖机后，原输入事实/时序发生变化")
        if any(date.fromisoformat(row["event_date"]).weekday() >= 5 for row in rows):
            raise ValidationError("派生观测日序包含周末")
        action_evidence = next(e for e in inputs["source_evidence"] if e.get("kind") == "actions" and e.get("status") == "parsed")
        actions, details = _parse_action_tables(
            Path(action_evidence["path"]).read_bytes(), symbol="018147",
            source_url=action_evidence["source_url"], observed_at=action_evidence["observed_at"],
            start=rows[0]["event_date"], end=seed["end"],
        )
        marks = {}
        for raw in nav["raw_records"]:
            if raw.get("function") == "Data_netWorthTrend" and isinstance(raw.get("record"), dict):
                record = raw["record"]
                day = _js_millis_date(record["x"])
                if rows[0]["event_date"] <= day <= seed["end"]:
                    if day in marks:
                        raise ValidationError("原始净值标记重复，不能默认为覆盖完整")
                    marks[day] = record
        complete = all(
            row["event_date"] in marks
            and "unitMoney" in marks[row["event_date"]]
            and marks[row["event_date"]]["unitMoney"] in (None, "")
            and decimal(marks[row["event_date"]]["y"]) == decimal(row["payload"]["unit_nav"])
            for row in rows
        )
        if not complete or actions:
            raise ValidationError("扩展暖机/收益区间权益无法按本固定新策略核对，保留证据但停止")
        item.update(prices=rows, actions=actions, action_coverage=True)
        inputs.update(
            strategies=strategy_list, strategy_definitions={MONTHLY_STRATEGY: copy.deepcopy(MONTHLY_DEFINITION)},
            required_strategy_warmup=62,
            input_preparation={
                "original_input_path": str(SEEDS[name] / "inputs.json"), "original_input_hash": digest(seed),
                "original_report_hash": digest(originals[name]), "source_nav_path": str(nav_path),
                "source_nav_sha256": sha256(nav_path), "original_rows_preserved": True,
                "original_warmup": 60, "prepared_warmup": 62,
                "source_weekend_dates_not_used": weekends, "calendar": "weekday_NAV_observations_not_official_calendar",
                "extended_action_check": {"from": rows[0]["event_date"], "through": seed["end"],
                                          "unitMoney_rows_checked": len(rows), "empty_unitMoney_and_matching_NAV": complete,
                                          "parsed_actions": actions, "table_details": details,
                                          "official_verification": False},
            },
            warnings=[
                "真实净值与固定假设的历史回放，不是支付宝真实成交或前向实绩，也非严格事先封存样本外。",
                "仅新增monthly_mom60_hold30，60期单位净值动量、月度检查、确认日起至少30自然日持有，不调参不选优。",
                "月初不足30日或未确认/未到账则等下个月重判；不在月中补单、不保证及时止损。",
                "原MA20/现金/持有不变；收益窗口不变，额外2条暖机只供60期动量与lag2使用。",
                "初始100元、不追加；每笔申购10至100元、有仓不补仓；两种到账情景均报告。",
                "NAV假设滞后2个观测日可知、申请后2个观测日确认、7/10个观测日到账均为固定假设，不是支付宝已核规则。",
                "仅使用周一至周五已有净值；没有核验全部境内外假期、历史开放/限额、费用生效和实际披露/到账。",
                "按原来源重新检查62条暖机及窗口内unitMoney和权益HTML；二手源无事件不等于管理人完整核验。",
                "NAV已含运作费与人民币折算；期末不清仓、不含未来退出费，应收并非立即可取现金。",
                "两段市场累计上涨，不代表完整牛熊周期；两窗口时长不同，不直接比较累计收益高低。",
                *nav.get("warnings", []),
            ],
        )
        inputs["unit_money_check"] = {"records": len(rows), "nonempty_unitMoney": 0, "includes_warmup": True}
        write_json(directory / "inputs.json", inputs)
        write_json(directory / "source-window-check.json", inputs["input_preparation"])
        prepared[name] = inputs
    if prepared["earlier"]["end"] >= prepared["later"]["products"][0]["prices"][0]["event_date"]:
        raise ValidationError("两窗口或暖机重叠")
    comparison = []
    for name, inputs in prepared.items():
        directory = root / name
        command_log(directory, "replay", ["-m", "rmbfund.cli", "research", "--frozen", str(directory / "inputs.json"), "--out", str(directory / "results")])
        command_log(directory, "reproduce", ["-m", "rmbfund.cli", "research", "--frozen", str(directory / "results/inputs.json"), "--out", str(directory / "reproduced")])
        report = read_json(directory / "results/report.json")
        if digest(report) != digest(read_json(directory / "reproduced/report.json")):
            raise ValidationError("新输入CLI冻结复现经济结果不一致")
        baseline_audit = []
        for original in originals[name]["results"]:
            fresh = next(result for result in report["results"] if result["strategy"] == original["strategy"] and result["scenario"] == original["scenario"])
            if original != normalize_indices(fresh, 2):
                raise ValidationError(f"{name}/{original['scenario']}/{original['strategy']}原基准经济路径变化")
            baseline_audit.append({"scenario": original["scenario"], "strategy": original["strategy"], "all_dated_economics_unchanged": True})
        write_json(directory / "baseline-equivalence.json", {
            "legacy_unexpanded_report_hash_unchanged": True, "baseline_paths": baseline_audit,
            "normalized_index_fields": sorted(INDEX_FIELDS), "added_warmup_index_offset": 2,
        })
        command_log(directory, "verify", [str(PROJECT / "scripts/verify-research.py"), str(directory / "results"), "--public-db", str(args.db.resolve()), "--out", str(directory / "results/verification.json")])
        for result in report["results"]:
            attempts = Counter(order["signal_at"][:10] for order in result["orders"])
            if max(attempts.values(), default=0) > 1:
                raise ValidationError("同一日出现多次交易尝试")
            comparison.append({"window": name, "start": inputs["start"], "end": inputs["end"],
                               "strategy": result["strategy"], "scenario": result["scenario"],
                               "metrics": result["metrics"]})
    after = public_snapshot(args.db)
    if before["hash"] != after["hash"]:
        raise ValidationError("public源库变化，不得宣称只读一致")
    write_json(root / "public-unchanged.json", {"before": before["hash"], "after": after["hash"], "unchanged": True})
    write_json(root / "comparison.json", comparison)
    print(json.dumps({"out": str(root), "paths": len(comparison), "public_unchanged": True,
                      "results": [{"window": row["window"], "strategy": row["strategy"], "scenario": row["scenario"],
                                   "return": row["metrics"]["conditional_net_return"], "drawdown": row["metrics"]["conditional_max_drawdown"],
                                   "assets": row["metrics"]["end_assets"], "fees": row["metrics"]["fees"], "fills": row["metrics"]["filled_orders"]}
                                  for row in comparison]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

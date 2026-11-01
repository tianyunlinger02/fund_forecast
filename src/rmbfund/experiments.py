"""Isolated, deterministic plumbing checks, not an optimizer or a profitability claim."""

from __future__ import annotations

import json
from pathlib import Path

from .demo import DAYS, END_AT, INITIAL_AT
from .performance import account_history, describe
from .service import Workspace


def run_demo_comparison(root: Path, output_dir: Path) -> dict:
    workspace = Workspace(root, "demo")
    try:
        workspace.initialize_demo()
        strategies = {"cash": "保持现金", "hold": "相同初始建仓后不操作", "review": "预设两次减仓的流程对照"}
        # Both instruments have published the previous week's values on these demo clocks.
        # Do not force a daily comparison using an ETF close and an unpublished fund NAV.
        clocks = [INITIAL_AT] + [f"2026-01-{day}T22:00:00+08:00" for day in ("10", "17", "24", "31")]
        accounts = {item["id"]: item for item in workspace.paper.accounts()}
        for key, label in strategies.items():
            account = "compare-" + key
            if account in accounts:
                if accounts[account]["last_advanced_at"] != END_AT:
                    raise ValueError("比较运行未完成，请使用新的root，不覆盖既有账本")
                continue
            workspace.paper.create_account(label, "100000", INITIAL_AT, account_id=account)
            if key != "cash":
                workspace.paper.submit(account, "DEMO-ETF", "buy", INITIAL_AT,
                                       units="1000", budget="5000", idempotency_key="initial-etf")
                workspace.paper.submit(account, "DEMO-FUND", "buy", INITIAL_AT,
                                       amount="10000", idempotency_key="initial-fund")
            settlement_days = sorted(set(DAYS[1:]) | {clock[:10] for clock in clocks[1:]})
            for day in settlement_days:
                at = f"{day}T22:00:00+08:00"
                workspace.paper.advance(account, at)
                if key == "review" and day == "2026-01-14":
                    workspace.paper.submit(account, "DEMO-FUND", "sell", at,
                                           units="1000", idempotency_key="reduce-fund")
                if key == "review" and day == "2026-01-27":
                    workspace.paper.submit(account, "DEMO-ETF", "sell", at,
                                           units="500", idempotency_key="reduce-etf")
        result = {"mode": "fictional_demo", "purpose": "验证同资金、同结算与费用口径的对照工具",
                  "warning": "所有价格和交易日均为虚构。按两类数据均已披露的周末时点采样，回撤不是每日回撤；收益高低不说明任何方法有效。",
                  "strategies": {}}
        for key, label in strategies.items():
            history = account_history(workspace.paper, "compare-" + key, clocks)
            result["strategies"][key] = {"name": label, "metrics": describe(history, "100000"),
                                         "history": history}
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "demo-comparison.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        workspace.export(output_dir, END_AT)
        return {"path": str(path), "strategies": {
            key: item["metrics"] for key, item in result["strategies"].items()},
                "warning": result["warning"]}
    finally:
        workspace.store.close()

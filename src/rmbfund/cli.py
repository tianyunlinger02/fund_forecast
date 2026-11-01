"""Command-line access to the same services used by the local interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .domain import ValidationError, now_iso


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基金日级研究台（免费源、仅本地纸面模拟）")
    parser.add_argument("--root", default=os.environ.get("RMBFUND_ROOT", "data"))
    parser.add_argument("--mode", choices=["demo", "public"],
                        default=os.environ.get("RMBFUND_MODE", "demo"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-demo", help="初始化独立虚构样例，不联网")
    commands.add_parser("demo-check", help="固定时钟执行完整离线纸面演示")
    compare = commands.add_parser("demo-compare", help="隔离比较现金/持有/预设减仓，仅验证对照流程")
    compare.add_argument("--out", type=Path, default=Path("runs/artifacts/demo-comparison"))
    research = commands.add_parser("research", help="真实历史数据的独立假设回放，不写真实纸面账本")
    research.add_argument("--db", type=Path, help="只读 public SQLite 输入；默认 --root/public.sqlite3")
    research.add_argument("--out", type=Path, help="新的实验目录，已存在则拒绝覆盖")
    research.add_argument("--start", default="2024-10-21")
    research.add_argument("--end", default="2026-10-19")
    research.add_argument("--frozen", type=Path, help="离线重放之前的 inputs.json，不联网、不打开源库")
    research.add_argument("--include-us-example", action="store_true",
                          help="加入270042人民币QDII同类型样本和费用敏感性；非实际持仓/支付宝可买证明")
    view = commands.add_parser("overview", help="查看观察名单与数据质量")
    view.add_argument("--at", default=None)
    add = commands.add_parser("add", help="添加观察产品，不自动启用模拟")
    add.add_argument("symbol")
    add.add_argument("name")
    add.add_argument("--kind", choices=["etf", "open_fund", "monitor_only"], required=True)
    add.add_argument("--exchange", default="")
    add.add_argument("--category", default="未分类")
    add.add_argument("--share-class", default="")
    refresh = commands.add_parser("refresh", help="显式获取一个真实观察标的的免费数据")
    refresh.add_argument("product_id")
    forecast = commands.add_parser("forecast", help="只读本地已保存预测，不联网、不拟合、不初始化表")
    forecast.add_argument("product_id")
    forecast.add_argument("--local", action="store_true", help="明确只读本地（此命令始终不联网）")
    forecast.add_argument("--at", default=None, help="历史可用时间；不能提前读取未来资料")
    model_ids = ["nav_ridge_empirical_v2", "exposure_factor_ridge_empirical_v1"]
    forecast.add_argument("--model", choices=model_ids, default=model_ids[0], help="明确读取哪套已保存模型")
    update = commands.add_parser("forecast-update", help="显式更新一次净值并拟合；同依据复用预测")
    update.add_argument("product_id")
    update.add_argument("--local", action="store_true", help="不联网，仅用本地实际可用净值和所选模型输入计算并保存")
    update.add_argument("--model", choices=model_ids, default=model_ids[0], help="明确更新哪套模型；试验模型不自动晋级")
    factors = commands.add_parser("factor-import", help="显式导入已核对的公开投资因素JSON，不联网、不拟合")
    factors.add_argument("product_id")
    factors.add_argument("path", type=Path)
    factor_refresh = commands.add_parser("factor-refresh", help="显式获取已核对的免费投资因素，首次源失败即停止")
    factor_refresh.add_argument("product_id")
    fundamentals = commands.add_parser("issuer-analysis", help="只读已复核的公司盈利、原指引与匹配证券估值，不联网、不预测")
    fundamentals.add_argument("product_id")
    fundamentals.add_argument("--at", default=None)
    fundamentals.add_argument("--issuer", help="同时计算哪家公司的估值条件；不填写则只看实际资料")
    fundamentals.add_argument("--multiple", help="明确假设的终值市盈率，不自动填合理估值")
    fundamentals.add_argument("--earnings-change", help="明确假设的盈利变化比例，0.1表示10%")
    train = commands.add_parser("forecast-research", help="离线冻结完整输入、预测报告和时序评估，不写数据库")
    train.add_argument("product_id")
    train.add_argument("--local", action="store_true", help="仅本地（研究命令始终不联网）")
    train.add_argument("--output", "--out", type=Path, help="新输出目录，已存在则拒绝覆盖")
    train.add_argument("--frozen", type=Path, help="重放先前inputs.json，不打开业务数据库")
    compare_forecasts = commands.add_parser("forecast-compare", help="离线比较冻结净值上的固定候选，不写预测库")
    compare_forecasts.add_argument("product_id")
    compare_forecasts.add_argument("--frozen", type=Path, required=True, help="主样本已冻结inputs.json")
    compare_forecasts.add_argument("--aux-frozen", type=Path, action="append", default=[], help="辅助公开样本冻结输入，可重复")
    compare_forecasts.add_argument("--output", "--out", type=Path, required=True, help="新的比较输出目录，拒绝覆盖")
    compare_forecasts.add_argument("--include-tree", action="store_true", help="显式运行可选标准浅树，不自动安装依赖")
    compare_forecasts.add_argument("--protocol", type=Path, help="严格复用已冻结的同输入/候选/版本协议")
    compare_forecasts.add_argument("--vix-frozen", type=Path, help="显式加入已验收CBOE日收盘JSON，仍非PIT")
    commands.add_parser("fund-review-init", help="显式备份并初始化公开原始资料表，不创建旧规则检查表、不联网")
    evidence = commands.add_parser("fund-review-import", help="复核导入公开基金文件，不导入账户或持仓")
    evidence.add_argument("product_id")
    evidence.add_argument("path", type=Path, help="公开HTML/JSON/PDF原件，最多8MiB")
    evidence.add_argument("--source-url", required=True, help="文件原公开URL，不是搜索摘要")
    evidence.add_argument("--records", type=Path, required=True, help="带family/report_date/payload/locator的JSON列表")
    evidence.add_argument("--reviewed", action="store_true", help="明确确认已核对原件、产品、期间、单位与页表定位")
    note_import = commands.add_parser("research-note-import", help="导入本地作者复核研究，非发布者原文、不改预测")
    note_import.add_argument("product_id")
    note_import.add_argument("path", type=Path)
    session_import = commands.add_parser("research-session-import", help="显式追加会话助手研究，不调用模型API")
    session_import.add_argument("product_id")
    session_import.add_argument("path", type=Path)
    session_import.add_argument("--evidence", type=Path, help="本次实际采用的证据和计算包；已绑定研究可省略")
    note_view = commands.add_parser("research-note", help="只读截至指定时点的基金研究")
    note_view.add_argument("product_id")
    note_view.add_argument("--at", default=None)
    research_update = commands.add_parser("research-update", help="显式在线取材并调用已有模型API出具研究，不拟合预测")
    research_update.add_argument("product_id")
    research_update.add_argument("--api-config", type=Path, help="明确的research-api.json；Key仅用RMBFUND_RESEARCH_API_KEY")
    buyer_config = commands.add_parser("buyer-config-import", help="明确安装公开候选集合与定向来源配置，不联网")
    buyer_config.add_argument("path", type=Path)
    buyer_config.add_argument("--replace", action="store_true")
    peer_compare = commands.add_parser("fund-compare", help="只读真实基金候选比较，不拟合、不排名")
    peer_compare.add_argument("product_id")
    peer_compare.add_argument("--peer-set", type=Path)
    peer_compare.add_argument("--at", default=None)
    current = commands.add_parser("current-online", help="直接在线读取当前公开基金资料，不写库、不归档")
    current.add_argument("product_ids", nargs="+", help="open_fund:代码:份额类别；未知类别留空")
    current.add_argument("--source-plan", type=Path, help="可选当前来源计划JSON；不填使用代码内已核对来源")
    current.add_argument("--request-budget", type=int, default=20)
    current.add_argument("--blocked-url", action="append", default=[], help="本次会话停止的来源URL，可重复")
    followed = commands.add_parser("followed-refresh", help="显式定向更新公开资料，独立家族收据，不读取私人持仓")
    followed.add_argument("product_ids", nargs="+")
    followed.add_argument("--families", nargs="+", choices=["nav", "disclosure", "policy_news"], default=["nav", "disclosure", "policy_news"])
    followed.add_argument("--request-budget", type=int, default=12)
    followed.add_argument("--output", type=Path, required=True)
    retained = commands.add_parser("followed-review-retained", help="复核成功留存原件并导入资料，零新请求，不冒充本次抓取")
    retained.add_argument("product_ids", nargs="+")
    retained.add_argument("--captures", type=Path, required=True)
    retained.add_argument("--output", type=Path, required=True)
    issuer_research = commands.add_parser("issuer-forecast-research", help="离线资格核对并比较冻结18维候选，不发行或晋级")
    for name in ("nav", "market", "issuer", "prices", "protocol"):
        issuer_research.add_argument("--" + name, type=Path, required=True)
    issuer_research.add_argument("--output", type=Path, required=True)
    profile = commands.add_parser("profile", help="从JSON导入带来源/版本的交易规则档案")
    profile.add_argument("path", type=Path)
    nav_capture = commands.add_parser("nav-capture-import", help="显式解析留存公开净值原件，不联网，保留本次导入时钟")
    nav_capture.add_argument("product_id")
    nav_capture.add_argument("path", type=Path)
    nav_capture.add_argument("--source-url", required=True)
    nav_capture.add_argument("--start", required=True)
    nav_capture.add_argument("--end", required=True)
    imp = commands.add_parser("import-csv", help="导入数据，实际观察时间为本次导入时间")
    imp.add_argument("product_id")
    imp.add_argument("kind", choices=["bar", "nav", "calendar", "announcement", "status", "dividend", "split"])
    imp.add_argument("path", type=Path)
    probes = commands.add_parser("probe", help="低频试用免费源，保留成功及失败证据")
    probes.add_argument("--out", type=Path, default=Path("runs/artifacts/source-probes"))
    probes.add_argument("--timeout", type=int, default=12)
    account = commands.add_parser("account", help="创建独立虚拟账户")
    account.add_argument("name")
    account.add_argument("--cash", default="100000")
    account.add_argument("--at", default=None)
    order = commands.add_parser("order", help="显式确认本地虚拟订单，绝不发送实盘")
    order.add_argument("account_id")
    order.add_argument("product_id")
    order.add_argument("side", choices=["buy", "sell"])
    order.add_argument("--at", required=True)
    order.add_argument("--amount")
    order.add_argument("--units")
    order.add_argument("--budget")
    order.add_argument("--key", help="重复执行同一意图时使用同一幂等键")
    advance = commands.add_parser("advance", help="推进虚拟结算时钟（不可回拨）")
    advance.add_argument("account_id")
    advance.add_argument("--at", required=True)
    summary = commands.add_parser("summary", help="查询纸面账户余额、持仓与不确定项")
    summary.add_argument("account_id")
    summary.add_argument("--at", default=None)
    note = commands.add_parser("note", help="记录操作或不操作的理由")
    note.add_argument("reason")
    note.add_argument("--product-id")
    note.add_argument("--decision", default="不操作")
    note.add_argument("--url", default="")
    note.add_argument("--at", default=None)
    export = commands.add_parser("export", help="导出带快照清单的CSV证据")
    export.add_argument("--out", type=Path, default=Path("runs/artifacts/export"))
    export.add_argument("--at", default=None)
    ui = commands.add_parser("ui", help="启动仅本机访问的中文界面")
    ui.add_argument("--port", type=int, default=8501)
    return parser


def run_demo_check(workspace, output_dir: Path) -> dict:
    from .demo import DAYS, INITIAL_AT
    result = workspace.initialize_demo()
    existing = next((item for item in workspace.paper.accounts() if item["id"] == "demo-e2e"), None)
    if existing:
        if existing["last_advanced_at"] < result["end_at"]:
            raise ValidationError("上次验收未完成，请使用新的 --root 保留旧证据后重试")
        report = {"kind": "fictional_offline_acceptance", "reused": True,
                  "summary": workspace.paper.summary("demo-e2e", result["end_at"]),
                  "files": workspace.export(output_dir, result["end_at"]),
                  "warning": "复用既有离线验收账本，不重复下单或追加决策。"}
        return report
    account = workspace.paper.create_account("离线端到端验收", "100000", INITIAL_AT,
                                             account_id="demo-e2e")
    for product_id, kwargs in (
        ("DEMO-ETF", {"units": "1000", "budget": "5000"}),
        ("DEMO-FUND", {"amount": "10000"}),
    ):
        workspace.paper.submit(account, product_id, "buy", INITIAL_AT,
                               idempotency_key=f"e2e-buy-{product_id}", **kwargs)
    workspace.save_note("DEMO-FUND", INITIAL_AT, "模拟买入",
                        "虚构样例：检验未知价申购、费用与份额确认，不是收益预测。")
    for day in DAYS[1:]:
        at = f"{day}T22:00:00+08:00"
        workspace.paper.advance(account, at)
        if day == "2026-01-14":
            workspace.paper.submit(account, "DEMO-FUND", "sell", at,
                                   units="1000", idempotency_key="e2e-fund-sell")
        if day == "2026-01-27":
            workspace.paper.submit(account, "DEMO-ETF", "sell", at,
                                   units="500", idempotency_key="e2e-etf-sell")
    workspace.paper.advance(account, result["end_at"])
    summary = workspace.paper.summary(account, result["end_at"])
    files = workspace.export(output_dir, result["end_at"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "fictional_offline_acceptance", "summary": summary, "files": files,
              "warning": "仅证明演示流程运行，不证明策略有效或真实产品能精确成交。"}
    (output_dir / "demo-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ui":
            env = {**os.environ, "RMBFUND_ROOT": str(Path(args.root).resolve()),
                   "RMBFUND_MODE": args.mode, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
            app = Path(__file__).parent / "ui" / "app.py"
            # Streamlit watches PYTHONPATH, not editable-install sys.path entries.
            source_root = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                source_root, env.get("PYTHONPATH"),
            )))
            return subprocess.call([
                sys.executable, "-m", "streamlit", "run", str(app),
                "--server.address", "127.0.0.1", "--server.port", str(args.port),
                "--server.headless", "true", "--browser.gatherUsageStats", "false",
            ], env=env)
        if args.command == "current-online":
            from .buyer_research import current_public_identity, read_public_json
            from .current_online import snapshot_online_context
            from .service import Workspace
            if args.mode != "public":
                raise ValidationError("当前在线查询须明确--mode public；演示模式不联网")
            subjects = [current_public_identity(value) for value in args.product_ids]
            plan = read_public_json(args.source_plan) if args.source_plan else None
            context = Workspace.current_online(subjects, source_plan=plan,
                request_budget=args.request_budget, blocked_urls=args.blocked_url)
            _print(snapshot_online_context(context) | {"stopped_urls": context.get("stopped_urls", []),
                                                      "database_writes": False, "raw_archives_written": 0})
            return 0
        if args.command == "probe":
            from .providers.probe import run_probes
            _print(run_probes(args.out, timeout=args.timeout))
            return 0
        if args.command == "demo-compare":
            if args.mode != "demo":
                raise ValidationError("虚构方法对照只能在demo模式运行")
            from .experiments import run_demo_comparison
            _print(run_demo_comparison(Path(args.root) / "comparison", args.out))
            return 0
        if args.command == "research":
            from .research_replay import run_research_replay
            output = args.out or Path("runs/artifacts") / f"research-{now_iso().replace(':', '').replace('+', '-')}"
            database = args.db or (Path(args.root) / "public.sqlite3")
            result = run_research_replay(database, output, start=args.start, end=args.end,
                                         frozen=args.frozen, include_us_example=args.include_us_example)
            _print({"mode": result["mode"], "warning": result["warning"],
                    "start": result["start"], "end": result["end"], "output": str(output),
                    "input_hash": result["input_hash"],
                    "results": [{"symbol": item["product"]["symbol"], "scenario": item["scenario"],
                                 "strategy": item["strategy"], **item["metrics"]} for item in result["results"]],
                    "comparisons": result["comparisons"], "forward_ready": result["forward_ready"]})
            return 0
        if args.command == "forecast-research" and args.frozen:
            from .forecast import MODEL_HASH, fit_forecast, forecast_input_hash
            from .forecast_research import study_actions
            if args.frozen.is_symlink() or args.frozen.stat().st_size > 16 * 1024 * 1024:
                raise ValidationError("冻结输入必须是有界普通JSON文件")
            frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
            if frozen.get("schema") != "rmbfund.forecast.inputs.v1":
                raise ValidationError("未知冻结输入格式")
            if frozen["product"]["id"] != args.product_id:
                raise ValidationError("冻结输入产品不匹配")
            if frozen["model_hash"] != MODEL_HASH or frozen["input_hash"] != forecast_input_hash(frozen["series"]):
                raise ValidationError("冻结输入或模型哈希不匹配，不静默改写")
            output = args.output or Path("runs/artifacts") / f"forecast-research-{now_iso().replace(':', '').replace('+', '-')}"
            if output.exists():
                raise ValidationError("研究输出目录已存在；不会覆盖")
            report = fit_forecast(
                frozen["series"], product_id=args.product_id, generated_at=now_iso(),
                entry_lags=tuple(sorted({*frozen["entry_assumptions"]["entry_lags"], 2, 3, 4})),
                include_evaluation=True,
            )
            report["entry_assumptions"] = frozen["entry_assumptions"]
            report["frozen_input_generated_at"] = frozen["generated_at"]
            files = {"inputs.json": frozen, "report.json": report, "metrics.json": report["validation"],
                     "actions.json": study_actions(frozen["series"], report["issued_forecasts"])}
            encoded = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                       for name, value in files.items()}
            output.mkdir(parents=True, exist_ok=False)
            for name, payload in encoded.items():
                (output / name).write_text(payload, encoding="utf-8")
            _print({"status": "ready", "output": str(output.resolve()), "network": False,
                    "database_writes": False, "input_hash": report["input_hash"],
                    "generated_at": report["generated_at"], "validation": report["validation"]})
            return 0
        if args.command == "forecast-compare":
            from .forecast_comparison import run_forecast_comparison
            _print(run_forecast_comparison(
                args.frozen, args.aux_frozen, args.output, args.include_tree,
                product_id=args.product_id, protocol_path=args.protocol, vix_frozen=args.vix_frozen,
            ))
            return 0
        if args.command == "issuer-forecast-research":
            from .buyer_research import read_public_json
            from .issuer_fundamental_forecast import qualify_then_run18
            inputs = [read_public_json(getattr(args, key), maximum=32 * 1024 * 1024)
                      for key in ("nav", "market", "issuer", "prices", "protocol")]
            result = qualify_then_run18(*inputs)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            _print({"status": result["status"], "fitted": result["fitted"], "output": str(args.output),
                    "runtime_promoted": False, "database_writes": False})
            return 0
        from .service import Workspace
        read_only = args.command in {"forecast", "forecast-research", "issuer-analysis", "research-note", "fund-compare"}
        if args.command == "issuer-analysis" and not (Path(args.root) / f"{args.mode}.sqlite3").is_file():
            _print({"status": "missing", "issuers": [], "as_of": now_iso(),
                    "reasons": ["本地数据库不存在；没有创建或联网补资料。"]})
            return 0
        if args.command == "forecast" and not (Path(args.root) / f"{args.mode}.sqlite3").is_file():
            _print({"status": "missing", "product": None, "evaluated_at": now_iso(),
                    "prediction": None, "reasons": ["本地数据库不存在；只读查看没有创建数据库。"],
                    "evidence": {"records": [], "nav": {"snapshot": None}}})
            return 0
        workspace = Workspace(args.root, args.mode, read_only=read_only)
        from .demo import INITIAL_AT
        at = getattr(args, "at", None) or (INITIAL_AT if args.mode == "demo" else now_iso())
        try:
            if args.command == "init-demo":
                _print(workspace.initialize_demo())
            elif args.command == "demo-check":
                _print(run_demo_check(workspace, Path("runs/artifacts/demo-check")))
            elif args.command == "overview":
                _print(workspace.overview(at))
            elif args.command == "add":
                _print({"product_id": workspace.add_product(
                    args.symbol, args.name, args.kind, args.exchange, args.category, args.share_class
                )})
            elif args.command == "refresh":
                _print(workspace.refresh(args.product_id))
            elif args.command == "forecast":
                _print(workspace.forecast(args.product_id, args.at, model_id=args.model))
            elif args.command == "issuer-analysis":
                analysis = workspace.issuer_fundamentals(args.product_id, args.at)
                if any(value is not None for value in (args.issuer, args.multiple, args.earnings_change)):
                    if args.issuer is None or args.multiple is None:
                        raise ValidationError("条件计算须同时明确--issuer和--multiple；盈利变化可留空")
                    from .issuer_fundamentals import issuer_valuation_scenario
                    _print({"analysis": analysis, "scenario": issuer_valuation_scenario(
                        analysis, issuer_id=args.issuer, terminal_multiple=args.multiple,
                        earnings_change=args.earnings_change, label="CLI明确填写的盈利与估值条件",
                    )})
                else:
                    _print(analysis)
            elif args.command == "factor-import":
                _print(workspace.import_investment_factors(args.product_id, args.path))
            elif args.command == "factor-refresh":
                result = workspace.refresh_investment_factors(args.product_id)
                _print(result)
                if result["status"] == "error":
                    return 2
            elif args.command == "forecast-update":
                result = workspace.update_forecast(args.product_id, refresh=not args.local, model_id=args.model)
                _print(result)
                if result["status"] == "error":
                    return 2
            elif args.command == "forecast-research":
                output = args.output or Path("runs/artifacts") / f"forecast-research-{now_iso().replace(':', '').replace('+', '-')}"
                _print(workspace.forecast_research(args.product_id, output))
            elif args.command == "fund-review-init":
                if args.mode != "public":
                    raise ValidationError("公开原始资料只在公开模式初始化")
                workspace.store.initialize_evidence_schema()
                _print({"status": "initialized", "network": False, "trades": False})
            elif args.command == "fund-review-import":
                if args.mode != "public" or not args.reviewed:
                    raise ValidationError("仅公开资料导入，必须显式--reviewed确认人工复核")
                product = workspace.store.get_product(args.product_id)
                if product is None or product.is_demo or product.kind != "open_fund":
                    raise ValidationError("资料导入需要已登记的公开场外基金；不读取账户或持仓")
                if not workspace.store.has_evidence_schema():
                    raise ValidationError("请先显式执行fund-review-init；读取和导入不隐式迁移")
                if any(path.name.startswith(".env") for path in (args.path, args.records)):
                    raise ValidationError("不读取环境变量或凭据文件，只接受公开资料原件")
                if args.path.is_symlink() or args.records.is_symlink():
                    raise ValidationError("公开原件与结构化记录不能是符号链接")
                with args.path.open("rb") as handle:
                    raw = handle.read(8 * 1024 * 1024 + 1)
                with args.records.open("rb") as handle:
                    structured = handle.read(256 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024 or len(structured) > 256 * 1024:
                    raise ValidationError("公开原件或结构化记录超过大小上限")
                records = json.loads(structured)
                if isinstance(records, dict) and set(records) == {"records"}:
                    records = records["records"]
                if not isinstance(records, list):
                    raise ValidationError("结构化记录必须为JSON列表或仅含records的对象")
                from .providers.advice_evidence import import_reviewed_document
                _print(import_reviewed_document(
                    workspace.store, product, raw_bytes=raw, source_url=args.source_url,
                    records=records, reviewed=True, clock=now_iso,
                ))
            elif args.command == "research-note-import":
                _print(workspace.import_research_note(args.product_id, args.path))
            elif args.command == "research-session-import":
                _print(workspace.import_session_research(args.product_id, args.path,
                                                       evidence_path=args.evidence))
            elif args.command == "research-note":
                _print(workspace.research_thesis(args.product_id, args.at))
            elif args.command == "research-update":
                from .buyer_research import current_public_identity
                from .research_settings import read_research_api_settings
                settings = read_research_api_settings(args.api_config or Path(args.root) / "research-api.json")
                result = workspace.update_research(current_public_identity(args.product_id), settings=settings)
                # Full RAM source texts belong only to the running task.
                _print({key: result[key] for key in ("status", "research_run", "research_thesis", "receipt", "errors") if key in result})
                if result["status"] not in {"generated", "partial"}:
                    return 2
            elif args.command == "fund-compare":
                from .buyer_research import read_public_json
                peers = read_public_json(args.peer_set) if args.peer_set else None
                _print(workspace.compare_funds(args.product_id, peer_set=peers, as_of=args.at))
            elif args.command == "buyer-config-import":
                from .buyer_research import install_buyer_config, read_public_json
                _print(install_buyer_config(Path(args.root), read_public_json(args.path), replace=args.replace))
            elif args.command == "followed-refresh":
                products = []
                for identifier in args.product_ids:
                    product = workspace.store.get_product(identifier)
                    if product is None:
                        raise ValidationError("公开产品尚未登记")
                    products.append(asdict(product))
                _print(workspace.refresh_followed(products, families=args.families,
                    request_budget=args.request_budget, output_dir=args.output))
            elif args.command == "followed-review-retained":
                _print(workspace.review_retained_sources(args.product_ids, args.captures, args.output))
            elif args.command == "profile":
                if args.path.stat().st_size > 1_000_000:
                    raise ValidationError("档案文件过大")
                _print({"profile_id": workspace.save_profile(json.loads(args.path.read_text()))})
            elif args.command == "nav-capture-import":
                _print(workspace.import_nav_capture(args.product_id, args.path,
                    source_url=args.source_url, start=args.start, end=args.end))
            elif args.command == "import-csv":
                from .providers.csv import load_csv
                product = workspace.store.get_product(args.product_id)
                if not product:
                    raise ValidationError("产品不存在")
                facts = load_csv(args.path, product, now_iso(), args.kind)
                with workspace.store.transaction():
                    snapshot = workspace.store.record_snapshot(
                        "csv", product.id, [asdict(fact) for fact in facts], now_iso(),
                        source_url=str(args.path.resolve()),
                    )
                    count = sum(workspace.store.add_fact(fact) for fact in facts)
                _print({"inserted": count, "snapshot": snapshot})
            elif args.command == "account":
                _print({"account_id": workspace.paper.create_account(args.name, args.cash, at)})
            elif args.command == "order":
                _print(workspace.paper.submit(
                    args.account_id, args.product_id, args.side, args.at,
                    amount=args.amount, units=args.units, budget=args.budget,
                    idempotency_key=args.key,
                ))
            elif args.command == "advance":
                _print(workspace.paper.advance(args.account_id, args.at))
            elif args.command == "summary":
                _print(workspace.paper.summary(args.account_id, at))
            elif args.command == "note":
                _print({"id": workspace.save_note(args.product_id, at, args.decision,
                                                  args.reason, args.url)})
            elif args.command == "export":
                _print(workspace.export(args.out, at))
            return 0
        finally:
            workspace.store.close()
    except (ValidationError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

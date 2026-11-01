"""Read-only access to a pinned local study, with an explicit offline replay action.

Catalog paths, never paths embedded in retained evidence, authorize file reads.
Browsing validates retained JSON; it does not calculate investment performance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from .domain import Fact, Product, ValidationError, canonical_json, digest, now_iso, timestamp
from .research_replay import (
    MONTHLY_STRATEGY,
    STRATEGIES,
    Assumptions,
    run_research_replay,
    selected_strategies,
)

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 96 * 1024 * 1024
MAX_PATHS = 16
MAX_ROWS = 20_000
CATALOG_BYTES = 512 * 1024
STRATEGY_IDS = (*STRATEGIES, MONTHLY_STRATEGY)
FACT_KEYS = {item.name for item in fields(Fact)}
PRODUCT_KEYS = {item.name for item in fields(Product)}
AMOUNT_KEYS = (
    "model_assets", "cash_available", "cash_reserved", "cash_receivable", "units",
    "available_units", "unconfirmed_units", "price", "market_value", "fees",
    "slippage_cost", "rounding_loss", "cash_dividends",
)
METRIC_AMOUNTS = ("end_assets", "fees", "slippage_cost", "rounding_loss", "cash_dividends")
RETURN_KEYS = (
    "conditional_net_return", "conditional_max_drawdown", "price_only_change",
    "price_only_drawdown", "verified_total_return", "net_return",
)


def presentation_decimal(value: Any) -> Decimal:
    """Parse bounded display values, including the replay's 28-digit fractions.

    This deliberately does not relax the execution domain's monetary contract.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValidationError("研究展示值必须是有限十进制数")
    raw = str(value)
    if not raw or len(raw) > 128:
        raise ValidationError("研究展示数值文本过长")
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError("无效研究展示数值") from exc
    if (not number.is_finite() or abs(number.as_tuple().exponent) > 60
            or number.copy_abs() > Decimal("1e24")):
        raise ValidationError("研究展示数值或精度超出范围")
    return number


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _day(value: Any) -> str:
    _require(isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)),
             "研究日期必须是 YYYY-MM-DD")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValidationError("无效研究日期") from exc


def _hash(value: Any) -> str:
    _require(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
             "研究哈希必须是 SHA-256")
    return value


def _text(value: Any, *, maximum: int = 2000) -> str:
    _require(isinstance(value, str) and bool(value) and len(value) <= maximum,
             "研究文本字段缺失或过长")
    return value


def _strings(value: Any) -> list[str]:
    _require(isinstance(value, list) and len(value) <= 1000, "研究警告列表无效")
    for item in value:
        _text(item, maximum=16_384)
    return value


def _rows(value: Any, *, minimum: int = 0, maximum: int = MAX_ROWS) -> list[dict]:
    _require(isinstance(value, list) and minimum <= len(value) <= maximum
             and all(isinstance(item, dict) for item in value), "研究记录数量或类型无效")
    return value


def _relative(value: Any, *, artifacts: bool = True) -> str:
    _require(isinstance(value, str) and len(value) <= 1024 and "\\" not in value
             and "\x00" not in value, "不安全的研究相对路径")
    parts = value.split("/")
    _require(not Path(value).is_absolute() and all(part not in {"", ".", ".."} for part in parts),
             "研究路径不得为绝对路径或包含目录穿越")
    if artifacts:
        _require(parts[:2] == ["runs", "artifacts"] and len(parts) >= 4,
                 "研究证据必须位于项目 runs/artifacts 下")
    return value


def _directory_chain(path: Path, *, create: bool = False) -> None:
    """Reject symlinks even in an explicitly configured root's ancestors."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise ValidationError("研究目录不存在") from None
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            info = current.lstat()
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 "研究路径组件必须是非符号链接目录")


def _read_bytes(root: Path, relative: str, *, limit: int = MAX_FILE_BYTES) -> bytes:
    """Open every component without following links; bound the actual read too."""
    _relative(relative, artifacts=False)
    _directory_chain(root)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            next_directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory,
            )
            os.close(directory)
            directory = next_directory
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            _require(stat.S_ISREG(info.st_mode), "研究证据必须为普通文件")
            _require(info.st_size <= limit, "研究文件超过读取上限")
            body = stream.read(limit + 1)
            _require(len(body) <= limit, "研究文件实际读取超过上限")
            return body
    except OSError as exc:
        raise ValidationError(f"无法安全读取研究文件: {relative}") from exc
    finally:
        os.close(directory)


def _json(body: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            _require(key not in result, "研究 JSON 不允许重复键")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValidationError(f"研究 JSON 不允许非有限常量: {value}")

    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValidationError(f"无效研究 JSON: {exc}") from exc
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(count <= 1_000_000 and depth <= 40, "研究 JSON 结构过大或过深")
        if isinstance(item, dict):
            _require(len(item) <= 1000, "研究 JSON 对象字段过多")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            _require(len(item) <= 50_000, "研究 JSON 列表过长")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, float):
            _require(math.isfinite(item), "研究 JSON 不允许非有限数")
        elif isinstance(item, str):
            _require(len(item) <= 128 * 1024, "研究 JSON 文本过长")
        elif isinstance(item, int) and not isinstance(item, bool):
            _require(len(str(item)) <= 128, "研究 JSON 整数过长")
    return value


class _Reads:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bodies: dict[str, bytes] = {}
        self.total = 0

    def read(self, relative: str) -> bytes:
        if relative not in self.bodies:
            _require(len(self.bodies) < MAX_PATHS, "研究读取路径过多")
            body = _read_bytes(self.root, relative,
                               limit=min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - self.total))
            self.total += len(body)
            self.bodies[relative] = body
        return self.bodies[relative]

    def pinned(self, record: dict, *, parse: bool = True) -> Any:
        body = self.read(record["path"])
        _require(hashlib.sha256(body).hexdigest() == record["sha256"], "研究文件字节哈希不匹配")
        if not parse:
            return None
        value = _json(body)
        _require(digest(value) == record["canonical_hash"], "研究文件规范哈希不匹配")
        _require(isinstance(value, dict), "研究 JSON 顶层必须为对象")
        return value


def _pin(record: Any, *, canonical: bool = True) -> None:
    _require(isinstance(record, dict), "研究目录缺少文件声明")
    _relative(record.get("path"))
    _hash(record.get("sha256"))
    if canonical:
        _hash(record.get("canonical_hash"))


def _product(value: Any, expected: dict | None = None) -> None:
    _require(isinstance(value, dict) and set(value) == PRODUCT_KEYS, "研究产品完整身份字段缺失")
    _require(all(isinstance(value[key], str) for key in PRODUCT_KEYS - {"is_demo"}),
             "研究产品身份字段类型无效")
    Product(**value)
    _require(value["id"] == "research:018147:C" and value["symbol"] == "018147"
             and value["kind"] == "open_fund" and value["is_demo"] is False
             and value["share_class"] == "C", "研究仅接受非演示的 018147 场外 C 份额")
    if expected is not None:
        _require(canonical_json(value) == canonical_json(expected), "研究产品完整身份不匹配")


def _fact(value: dict, product: dict) -> Fact:
    _require(set(value) == FACT_KEYS, "净值事实字段不完整")
    _require(value.get("product_id") == product["id"] and value.get("kind") == "nav",
             "净值事实产品或基准不匹配")
    _day(value.get("event_date"))
    _text(value.get("source"))
    _text(value.get("source_url"))
    _hash(value.get("id"))
    try:
        return Fact(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"净值事实无效: {exc}") from exc


def _evidence_match(value: Any, pin: dict, root: Path) -> bool:
    # Embedded absolute pointers are provenance text only, never read authority.
    return value in (pin["path"], str(root / pin["path"]))


def _validate_metrics(result: dict, days: list[str], *, covered: bool) -> None:
    metrics = result.get("metrics")
    _require(isinstance(metrics, dict), "研究结果缺少指标")
    complete = all(row["model_complete"] for row in result["daily"])
    _require(metrics.get("model_complete") is complete
             and metrics.get("verified_data_complete") is False, "研究指标完整性声明无效")
    _require(all(key in metrics for key in (*RETURN_KEYS, *METRIC_AMOUNTS)), "研究指标字段缺失")
    _require(metrics["verified_total_return"] is None and metrics["net_return"] is None,
             "研究不得宣称已核验投资收益")
    _require(type(metrics.get("samples")) is int and metrics["samples"] == len(days)
             and metrics.get("actual_start") == days[0] and metrics.get("actual_end") == days[-1],
             "研究指标样本或日期与每日路径不一致")
    _require(metrics.get("ending_state") == result["daily"][-1], "研究期末状态不匹配")
    for key in ("filled_orders", "rejected_orders"):
        _require(type(metrics.get(key)) is int and 0 <= metrics[key] <= MAX_ROWS,
                 "研究委托计数无效")
    fraction = presentation_decimal(metrics.get("exposure_fraction"))
    _require(0 <= fraction <= 1, "研究持仓比例无效")
    if not complete:
        _require(all(metrics[key] is None for key in (*RETURN_KEYS, *METRIC_AMOUNTS)),
                 "不完整路径的资产、费用和收益指标必须留空")
        return
    last = result["daily"][-1]
    for key in METRIC_AMOUNTS:
        number = presentation_decimal(metrics[key])
        _require(number >= 0 and number == presentation_decimal(
            last["model_assets" if key == "end_assets" else key]), "研究期末金额指标不匹配")
    applicable = ("conditional_net_return", "conditional_max_drawdown") if covered else (
        "price_only_change", "price_only_drawdown")
    absent = ("price_only_change", "price_only_drawdown") if covered else (
        "conditional_net_return", "conditional_max_drawdown")
    _require(all(metrics[key] is None for key in absent), "权益覆盖不足时不得展示条件总收益")
    for key in applicable:
        _require(metrics[key] is not None, "完整模型缺少相应条件或价格指标")
        presentation_decimal(metrics[key])
    expected_basis = "cash_distribution_total_return_under_assumptions" if covered else "price_only"
    _require(metrics.get("valuation_basis") == expected_basis, "研究估值基准不匹配")


def _validate_bundle(inputs: dict, report: dict, manifest: dict, window: dict, product: dict,
                     facts: dict[str, dict], pins: list[dict], root: Path) -> None:
    for document in (inputs, report, manifest):
        _require(document.get("mode") == "research_replay", "仅接受冻结 research_replay 研究")
        _require(document.get("start") == window["start"] and document.get("end") == window["end"],
                 "研究窗口日期不匹配")
    _require(digest(inputs) == manifest.get("input_hash") == report.get("input_hash"),
             "研究输入哈希链不匹配")
    _require(digest(report) == manifest.get("economic_result_hash"), "研究完整报告哈希链不匹配")
    _require(report.get("forward_ready") is False, "历史研究不得声明前向交易就绪")
    _hash(inputs.get("source_public_hash"))
    _require(manifest.get("source_public_hash") == inputs["source_public_hash"],
             "研究原始 public 来源指纹声明不匹配")
    # This fingerprint is historical provenance; do not compare it with today's database.
    evidence = _rows(inputs.get("source_evidence"), minimum=1, maximum=MAX_PATHS)
    _require(canonical_json(manifest.get("sources")) == canonical_json(evidence),
             "研究来源清单不匹配")
    _require(len(evidence) == len(pins), "研究来源清单与目录不匹配")
    seen = set()
    for item in evidence:
        matches = [pin for pin in pins if _evidence_match(item.get("path"), pin, root)]
        _require(len(matches) == 1, "研究来源指针不在显式目录中")
        pin = matches[0]
        _require(pin["path"] not in seen and item.get("content_hash") == pin["sha256"],
                 "研究来源哈希声明不匹配或重复")
        seen.add(pin["path"])
    for key in ("strategies", "strategy_definitions", "required_strategy_warmup"):
        _require(key in inputs and canonical_json(inputs[key]) == canonical_json(report.get(key))
                 == canonical_json(manifest.get(key)), "研究固定策略声明不匹配")
    _require(selected_strategies(inputs) == STRATEGY_IDS
             and type(inputs["required_strategy_warmup"]) is int
             and inputs["required_strategy_warmup"] == 62, "研究必须含三个对照及固定月度策略和62条暖机")
    try:
        assumptions = asdict(Assumptions(**inputs["assumptions"]))
    except (TypeError, KeyError) as exc:
        raise ValidationError("研究基础假设无效") from exc
    _require(canonical_json(assumptions) == canonical_json(inputs["assumptions"])
             == canonical_json(report.get("assumptions")) == canonical_json(manifest.get("assumptions")),
             "研究基础假设不一致")
    products = _rows(inputs.get("products"), minimum=1, maximum=1)
    item = products[0]
    _product(item.get("product"), product)
    _require(item.get("blocked") is None and type(item.get("action_coverage")) is bool,
             "冻结研究产品已阻塞或权益覆盖声明无效")
    _rows(item.get("actions"))
    scenarios = _rows(item.get("scenarios"), minimum=2, maximum=2)
    scenario_models = {}
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        _require(isinstance(scenario_id, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,60}", scenario_id))
                 and scenario_id not in scenario_models and isinstance(scenario.get("assumptions"), dict),
                 "研究情景标识或假设无效")
        try:
            scenario_models[scenario_id] = asdict(Assumptions(**{**assumptions, **scenario["assumptions"]}))
        except TypeError as exc:
            raise ValidationError("研究情景假设无效") from exc
    prices = _rows(item.get("prices"), minimum=63)
    price_days = []
    warmup_count = 0
    for index, row in enumerate(prices):
        raw = {key: row[key] for key in FACT_KEYS if key in row}
        fact = _fact(raw, product)
        _require(raw == facts.get(fact.id), "冻结净值事实与保留来源不匹配")
        day = row["event_date"]
        _require(not price_days or price_days[-1] < day, "冻结净值日期必须严格递增且唯一")
        _require(day <= window["end"] and date.fromisoformat(day).weekday() < 5,
                 "冻结研究观测日不符合已声明周内窗口")
        _require(type(row.get("warmup")) is bool and row["warmup"] == (day < window["start"]),
                 "冻结暖机标记不匹配")
        warmup_count += row["warmup"]
        _require(row.get("price_basis") == "unit_nav" and row.get("parent_fact_id") == fact.id
                 and row.get("input_hash") == digest(raw), "冻结净值基准或父事实哈希不匹配")
        known_day = prices[index + 2]["event_date"] if index + 2 < len(prices) else None
        _require(row.get("assumed_known_at") == (f"{known_day}T09:00:00+08:00" if known_day else None),
                 "冻结净值假设可知时序不匹配")
        price_days.append(day)
    _require(warmup_count == 62 and all(model["nav_lag_days"] == 2 for model in scenario_models.values()),
             "冻结研究必须保留62条暖机和两观测日净值滞后")
    days = price_days[warmup_count:]
    _require(days[0] == window["start"] and days[-1] == window["end"], "冻结窗口端点无观测")
    results = _rows(report.get("results"), minimum=8, maximum=8)
    result_keys = set()
    for result in results:
        _product(result.get("product"), product)
        key = (result.get("scenario"), result.get("strategy"))
        _require(key[0] in scenario_models and key[1] in STRATEGY_IDS and key not in result_keys,
                 "研究结果矩阵缺失、重复或含未知策略情景")
        result_keys.add(key)
        _require(canonical_json(result.get("assumptions")) == canonical_json(scenario_models[key[0]]),
                 "研究结果假设与合并情景不匹配")
        daily = _rows(result.get("daily"), minimum=1)
        _require([row.get("date") for row in daily] == days, "研究每日路径日期、顺序或端点不匹配")
        for row in daily:
            _require(row.get("product_id") == product["id"] and row.get("strategy") == key[1]
                     and type(row.get("model_complete")) is bool, "研究每日路径身份或完整性无效")
            for name in AMOUNT_KEYS:
                _require(name in row, "研究每日路径缺少资产或费用字段")
                if row[name] is None and not row["model_complete"]:
                    continue
                amount = presentation_decimal(row[name])
                _require(amount >= 0 and (name != "price" or amount > 0), "研究资产、份额或费用无效")
            if row["model_complete"]:
                with localcontext() as context:
                    context.prec = 160
                    parts = sum((presentation_decimal(row[name]) for name in (
                        "cash_available", "cash_reserved", "cash_receivable", "market_value")), Decimal(0))
                    _require(parts == presentation_decimal(row["model_assets"]), "研究资产组成不一致")
        for name in ("orders", "signals", "events"):
            _rows(result.get(name))
        _validate_metrics(result, days, covered=item["action_coverage"])
    _require(result_keys == {(scenario, strategy) for scenario in scenario_models for strategy in STRATEGY_IDS},
             "研究结果必须完整覆盖两情景乘四策略")
    _strings(inputs.get("warnings", []))
    _strings(report.get("gaps", []))
    _strings(report.get("forward_blockers", []))


class ResearchLibrary:
    """A small trusted catalog, not a product store or an arbitrary file browser."""

    def __init__(self, project_root: Path | None = None, catalog_path: Path | None = None,
                 *, output_root: Path | None = None) -> None:
        self.project_root = Path(os.path.abspath(project_root or Path(__file__).parents[2]))
        catalog = Path(catalog_path) if catalog_path is not None else Path("configs/research-catalog.json")
        self.catalog_path = catalog if catalog.is_absolute() else self.project_root / catalog
        output = Path(output_root) if output_root is not None else Path("runs/artifacts")
        self.output_root = Path(os.path.abspath(output if output.is_absolute() else self.project_root / output))

    def _catalog(self) -> list[dict]:
        try:
            relative = self.catalog_path.relative_to(self.project_root).as_posix()
        except ValueError as exc:
            raise ValidationError("研究目录必须位于项目 configs 内") from exc
        _require(relative.startswith("configs/"), "研究目录必须位于项目 configs 内")
        catalog = _json(_read_bytes(self.project_root, relative, limit=CATALOG_BYTES))
        _require(isinstance(catalog, dict) and type(catalog.get("version")) is int
                 and catalog["version"] == 1, "不支持的研究目录版本")
        studies = _rows(catalog.get("studies"), minimum=1, maximum=8)
        identifiers = set()
        for study in studies:
            identifier = _text(study.get("id"), maximum=80)
            _require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", identifier)) and identifier not in identifiers,
                     "研究标识无效或重复")
            identifiers.add(identifier)
            _text(study.get("title"), maximum=200)
            _product(study.get("product"))
            _pin(study.get("source"))
            evidence = _rows(study.get("evidence", []), maximum=MAX_PATHS - 7)
            for record in evidence:
                _pin(record, canonical=False)
            windows = _rows(study.get("windows"), minimum=2, maximum=2)
            _require([window.get("id") for window in windows] == ["earlier", "later"],
                     "研究必须声明 earlier 和 later 两个固定窗口")
            paths = [study["source"]["path"], *(item["path"] for item in evidence)]
            for window in windows:
                _text(window.get("label"), maximum=200)
                _require(_day(window.get("start")) <= _day(window.get("end")) < now_iso()[:10],
                         "研究窗口必须已经结束")
                for name in ("inputs", "report", "manifest"):
                    _pin(window.get(name))
                    _require(Path(window[name]["path"]).name == f"{name}.json", "研究仅接受冻结 JSON 叶文件")
                    paths.append(window[name]["path"])
                parents = {Path(window[name]["path"]).parent for name in ("inputs", "report", "manifest")}
                _require(len(parents) == 1 and next(iter(parents)).name == "results",
                         "研究窗口叶文件必须来自同一规范 results 目录，不接受 reproduced")
            _require(windows[0]["end"] < windows[1]["start"], "研究窗口不得重叠")
            _require(len(paths) == len(set(paths)) <= MAX_PATHS, "研究目录路径重复或过多")
        return studies

    def list_studies(self) -> list[dict]:
        """Cheap metadata only: no artifact, database, network, or replay access."""
        return [{key: deepcopy(study[key]) for key in ("id", "title", "product")}
                for study in self._catalog()]

    def _load(self, study_id: str) -> tuple[dict, dict, _Reads]:
        studies = self._catalog()
        study = next((item for item in studies if item["id"] == study_id), None)
        _require(study is not None, "研究标识不在可信目录中")
        reads = _Reads(self.project_root)
        source = reads.pinned(study["source"])
        _require(source.get("product_id") == study["product"]["id"], "原始净值来源产品身份不匹配")
        _text(source.get("status"), maximum=80)
        _text(source.get("source_url"))
        _text(source.get("provider"), maximum=200)
        raw_facts = _rows(source.get("facts"), minimum=1)
        facts, by_date = {}, {}
        observed = []
        for raw in raw_facts:
            fact = _fact(raw, study["product"])
            _require(fact.id not in facts, "原始净值事实标识重复")
            facts[fact.id] = asdict(fact)
            observed.append((timestamp(raw["observed_at"]), raw["observed_at"]))
            previous = by_date.get(fact.event_date)
            if previous is None or (timestamp(raw["observed_at"]), fact.id) > (
                    timestamp(previous["observed_at"]), previous["id"]):
                by_date[fact.event_date] = raw
        source_warnings = _strings(source.get("warnings", []))
        for pin in study.get("evidence", []):
            reads.pinned(pin, parse=False)
        pins = [study["source"], *study.get("evidence", [])]
        windows, warnings = [], list(source_warnings)
        for window in study["windows"]:
            inputs, report, manifest = (reads.pinned(window[name]) for name in ("inputs", "report", "manifest"))
            _validate_bundle(inputs, report, manifest, window, study["product"], facts, pins, self.project_root)
            windows.append({"id": window["id"], "label": window["label"], "start": window["start"],
                            "end": window["end"], "input_hash": manifest["input_hash"],
                            "economic_result_hash": manifest["economic_result_hash"],
                            "inputs": inputs, "report": report})
            warnings.extend(inputs.get("warnings", []))
            warnings.extend(report.get("gaps", []))
            warnings.extend(report.get("forward_blockers", []))
        nav_rows = [{"date": day, "value": raw["payload"]["unit_nav"], "basis": "unit_nav",
                     "source": raw["source"], "source_url": raw["source_url"],
                     "observed_at": raw["observed_at"],
                     "available_at": raw.get("available_at") or facts[raw["id"]]["available_at"]}
                    for day, raw in sorted(by_date.items())]
        public = {"id": study["id"], "title": study["title"], "product": deepcopy(study["product"]),
                  "source": {"status": source["status"], "facts": raw_facts, "url": source["source_url"],
                             "latest_date": nav_rows[-1]["date"], "observed_at": max(observed)[1],
                             "sourcehash": study["source"]["sha256"]},
                  "nav_rows": nav_rows, "windows": windows, "warnings": list(dict.fromkeys(warnings))}
        return public, study, reads

    def load_study(self, study_id: str) -> dict:
        """Validate and expose retained observations and reports without evaluating them."""
        try:
            return self._load(study_id)[0]
        except (KeyError, TypeError, AttributeError, IndexError, OSError, RecursionError) as exc:
            raise ValidationError(f"研究保留数据结构无效: {exc}") from exc

    def reproduce(self, study_id: str, request_id: str | None = None) -> dict:
        """Explicitly replay both frozen windows into a fresh, never-overwritten root.

        A request ID identifies an immutable receipt, including failed receipts.
        Existing incomplete directories are an error, not an invitation to resume.
        """
        operation_id = str(uuid.uuid4()) if request_id is None else str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"rmbfund-research-request:{request_id}"))
        output = self.output_root / f"ui-replay-{operation_id}"
        status: dict = {"ok": False, "operation_id": operation_id,
                        "output_dir": str(output), "windows": []}
        try:
            if request_id is not None:
                _require(isinstance(request_id, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id)),
                         "复现请求标识无效")
            public, study, reads = self._load(study_id)  # Always revalidate before touching output.
            parent = {"study_id": study_id, "source": study["source"],
                      "evidence": study.get("evidence", []), "windows": study["windows"]}
            parent_hash = digest(parent)
            for pin in [study["source"], *study.get("evidence", []), *(
                    window[name] for window in study["windows"] for name in ("inputs", "report", "manifest"))]:
                retained_root = self.project_root.joinpath(*Path(pin["path"]).parts[:3])
                _require(not self.output_root.is_relative_to(retained_root),
                         "复现输出不得写入已有研究证据目录")
            _directory_chain(self.output_root, create=True)
            try:
                output.mkdir(mode=0o700)
            except FileExistsError:
                _directory_chain(output)
                receipt = _json(_read_bytes(output, "receipt.json", limit=256 * 1024))
                _require(isinstance(receipt, dict) and receipt.get("version") == 1
                         and receipt.get("completed") is True and receipt.get("study_id") == study_id
                         and receipt.get("request_id") == request_id
                         and receipt.get("parent_hash") == digest(receipt.get("parent")) == parent_hash,
                         "复现请求目录冲突或未完成；不覆盖、不自动继续")
                saved = receipt.get("status")
                _require(isinstance(saved, dict) and saved.get("operation_id") == operation_id
                         and saved.get("output_dir") == str(output) and type(saved.get("ok")) is bool,
                         "复现回执身份无效")
                entries = _rows(saved.get("windows"), minimum=2, maximum=2)
                _require([item.get("id") for item in entries] == ["earlier", "later"]
                         and all(type(item.get("ok")) is bool and type(item.get("hash_matched")) is bool
                                 and item["ok"] == item["hash_matched"]
                                 and item.get("output_dir") == str(output / item["id"]) for item in entries)
                         and saved["ok"] is all(item["ok"] for item in entries),
                         "复现回执窗口无效")
                return saved
            frozen_root = output / "_frozen"
            frozen_root.mkdir(mode=0o700)
            generated = _Reads(output)
            for window in study["windows"]:
                entry = {"id": window["id"], "ok": False, "hash_matched": False,
                         "output_dir": str(output / window["id"])}
                status["windows"].append(entry)
                try:
                    frozen = frozen_root / f"{window['id']}-inputs.json"
                    with frozen.open("xb") as stream:
                        stream.write(reads.bodies[window["inputs"]["path"]])
                    returned = run_research_replay(
                        self.project_root / "data/public.sqlite3", Path(entry["output_dir"]),
                        start=window["start"], end=window["end"], frozen=frozen,
                    )
                    saved_inputs, saved_report, saved_manifest = (
                        _json(generated.read(f"{window['id']}/{name}.json"))
                        for name in ("inputs", "report", "manifest"))
                    _require(all(isinstance(doc, dict) for doc in (saved_inputs, saved_report, saved_manifest)),
                             "复现输出 JSON 必须为对象")
                    expected = next(item for item in public["windows"] if item["id"] == window["id"])
                    _require(digest(returned) == digest(saved_report) == expected["economic_result_hash"]
                             and digest(saved_inputs) == expected["input_hash"]
                             and saved_manifest.get("input_hash") == expected["input_hash"]
                             and saved_manifest.get("economic_result_hash") == expected["economic_result_hash"],
                             "复现经济结果或输入哈希不匹配；保留输出供检查")
                    entry.update(ok=True, hash_matched=True)
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            status["ok"] = all(item["ok"] for item in status["windows"])
            if not status["ok"]:
                status["error"] = "部分窗口复现失败；已保留所有部分输出，不覆盖或自动重试。"
            receipt = {"version": 1, "completed": True, "study_id": study_id,
                       "request_id": request_id, "created_at": now_iso(), "parent_hash": parent_hash,
                       "parent": parent, "status": status}
            with (output / "receipt.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")) + "\n")
            return status
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            return status

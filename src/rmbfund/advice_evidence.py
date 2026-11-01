"""Strict, independently versioned evidence contracts for experimental fund reviews.

Records describe economic content. Captures describe an actual retrieval/review.
Neither a repeat download nor a later capture changes a record ID.
Unknown disclosure coverage is retained, never redistributed to make a total of 100%.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .domain import ValidationError, decimal, digest, timestamp

SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "1"
MAX_RECORD_BYTES = 256 * 1024
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_WEB_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_RECORDS = 512
MAX_EVIDENCE_LINKS = 4096
MAX_EVIDENCE_READ_BYTES = 8 * 1024 * 1024
FAMILIES = frozenset({
    "mandate", "portfolio", "management", "scale", "fees", "operations",
    "events", "valuation", "macro", "issuer_fundamentals", "research_thesis",
})

_LOCAL_RESEARCH_URI = re.compile(r"rmb-research://(?:local|api|assistant)/[0-9a-f]{64}")
_RESEARCH_FACT_STATUSES = frozenset({"factual", "plan", "effective", "unresolved"})
_RESEARCH_CHANNELS = frozenset({
    "earnings", "valuation", "fx", "rates", "liquidity", "operations", "mandate", "other",
})
_RESEARCH_CITATION_KINDS = frozenset({
    "archived_evidence", "linked_not_archived", "reviewed_excerpt",
})


def _text(value: Any, label: str, *, maximum: int = 1000, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValidationError(f"{label}必须是有界文本")
    if any(ord(char) < 32 for char in value):
        raise ValidationError(f"{label}不能包含控制字符")
    return value


def _object(value: Any, label: str, *, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label}必须是对象")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValidationError(f"{label}字段无效；缺少{sorted(missing)}，未知{sorted(unknown)}")
    return dict(value)


def _list(value: Any, label: str, maximum: int = 512) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValidationError(f"{label}必须是有界列表")
    return value


def _enum(value: Any, choices: set[str] | frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f"{label}取值不支持")
    return value


def _bool(value: Any, label: str, *, nullable: bool = False) -> bool | None:
    if isinstance(value, bool) or (nullable and value is None):
        return value
    raise ValidationError(f"{label}必须是明确布尔值")


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValidationError(f"{label}必须是YYYY-MM-DD日期")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{label}不是有效日期") from exc
    return value


def _number(value: Any, label: str, *, minimum: str = "0", maximum: str | None = None) -> str:
    result = decimal(value)
    if result < Decimal(minimum) or (maximum is not None and result > Decimal(maximum)):
        raise ValidationError(f"{label}超出支持范围")
    if not result:
        return "0"
    text = format(result, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _ratio(value: Any, label: str) -> str:
    return _number(value, label, maximum="1")


def _currency(value: Any, label: str = "币种") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value):
        raise ValidationError(f"{label}必须是三位大写币种代码")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int = 1200) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValidationError(f"{label}必须是范围内的整数")
    return value


def _total(values: list[str], label: str) -> None:
    if sum((Decimal(value) for value in values), Decimal(0)) > 1:
        raise ValidationError(f"{label}同一分母下不能超过100%；不自动归一化")


def normalize_source_url(value: Any) -> str:
    """Validate a public locator without making a request or resolving DNS."""
    value = _text(value, "公开来源URL", maximum=4096)
    if any(char.isspace() for char in value) or "\\" in value:
        raise ValidationError("公开来源URL不能包含空白或反斜杠")
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise ValidationError("公开来源URL无效") from exc
    if parts.scheme not in {"https", "http"} or not host or parts.username or parts.password:
        raise ValidationError("只接受无凭据的公开HTTP(S)来源URL")
    host = host.lower().rstrip(".")
    if parts.fragment or host.endswith((".localhost", ".local", ".internal")) or host == "localhost":
        raise ValidationError("来源URL不能指向本地地址或用片段代替文档定位")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if len(labels) < 2 or labels[-1].isdigit() or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels):
            raise ValidationError("来源URL主机名无效；不接受模糊数字地址")
    else:
        if not address.is_global:
            raise ValidationError("来源URL不能指向私有或保留地址")
    if port is not None and not 1 <= port <= 65535:
        raise ValidationError("来源URL端口无效")
    authority = host.lower()
    if ":" in authority:
        authority = f"[{authority}]"
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parts.scheme, authority, parts.path or "/", parts.query, ""))


def bounded_json(value: Any, *, max_bytes: int = MAX_RECORD_BYTES) -> str:
    """Finite JSON only; reject objects that JSON's default=str would conceal."""
    nodes = 0

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal nodes
        nodes += 1
        if depth > 24 or nodes > 50000:
            raise ValidationError("JSON嵌套或元素数超过上限")
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, Decimal):
            decimal(item)
            return str(item)
        if isinstance(item, float):
            decimal(item)
            return item
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValidationError("JSON对象键必须是文本")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        raise ValidationError("记录必须是JSON兼容对象，不能包含任意Python对象")

    try:
        payload = json.dumps(visit(value), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        size = len(payload.encode("utf-8"))
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValidationError("记录不是有效的有限JSON") from exc
    if size > max_bytes:
        raise ValidationError("记录超过有界字节上限")
    return payload


def _mandate(value: Any) -> dict:
    result = _object(value, "投资范围", required={
        "symbol", "share_class", "name", "fund_id", "base_currency", "asset_class",
        "principal_protected", "sector_mandate", "single_country_mandate", "benchmark",
    }, optional={"recommended_horizon_months"})
    for key in ("symbol", "name", "fund_id"):
        result[key] = _text(result[key], key, maximum=200)
    result["share_class"] = _text(result["share_class"], "份额类别", maximum=80, empty=True)
    result["base_currency"] = _currency(result["base_currency"])
    result["asset_class"] = _enum(result["asset_class"], {"equity", "bond", "mixed", "other"}, "资产类型")
    result["principal_protected"] = _bool(result["principal_protected"], "保本属性", nullable=True)
    for key in ("sector_mandate", "single_country_mandate"):
        result[key] = _bool(result[key], key)
    benchmark = _object(result["benchmark"], "基准", required={
        "id", "name", "equity_weight", "region", "currency",
    })
    for key in ("id", "name", "region"):
        benchmark[key] = _text(benchmark[key], f"基准{key}", maximum=500, empty=True)
    benchmark["equity_weight"] = None if benchmark["equity_weight"] is None else _ratio(benchmark["equity_weight"], "基准权益比例")
    benchmark["currency"] = None if benchmark["currency"] is None else _currency(benchmark["currency"])
    result["benchmark"] = benchmark
    if "recommended_horizon_months" in result:
        result["recommended_horizon_months"] = None if result["recommended_horizon_months"] is None else _integer(result["recommended_horizon_months"], "建议持有月数", minimum=1)
    return result


def _portfolio(value: Any) -> dict:
    result = _object(value, "组合", required={
        "denominator", "assets", "holdings_scope", "holdings", "countries", "sectors", "currencies",
    }, optional={"duration_years", "rate_reference"})
    result["denominator"] = _enum(result["denominator"], {"nav", "equity_assets", "total_assets"}, "组合分母")
    assets = _object(result["assets"], "资产配置", required=set(), optional={"equity", "bond", "cash", "other"})
    if not assets:
        raise ValidationError("组合必须提供实际披露的资产配置；缺失不是零")
    result["assets"] = {key: _ratio(item, f"资产配置{key}") for key, item in assets.items()}
    _total(list(result["assets"].values()), "资产配置")
    result["holdings_scope"] = _enum(result["holdings_scope"], {"top10", "full", "partial"}, "持仓覆盖")
    holdings = []
    seen = set()
    for row in _list(result["holdings"], "披露持仓", 512):
        holding = _object(row, "持仓项", required={"issuer_id", "name", "kind", "weight"}, optional={"country", "sector"})
        for key in ("issuer_id", "name"):
            holding[key] = _text(holding[key], key, maximum=300)
        holding["kind"] = _enum(holding["kind"], {"corporate_equity", "bond", "fund", "cash", "other"}, "持仓类型")
        holding["weight"] = _ratio(holding["weight"], "持仓权重")
        for key in ("country", "sector"):
            if key in holding:
                holding[key] = _text(holding[key], key, maximum=200)
        marker = bounded_json(holding)
        if marker in seen:
            raise ValidationError("披露持仓出现完全重复行")
        seen.add(marker)
        holdings.append(holding)
    if result["holdings_scope"] == "top10" and len(holdings) > 10:
        raise ValidationError("前十大持仓不能包含超过十行")
    _total([row["weight"] for row in holdings], "披露持仓")
    result["holdings"] = holdings
    for key in ("countries", "sectors", "currencies"):
        rows = []
        names = set()
        is_currency = key == "currencies"
        name_key, weight_key = ("currency", "unhedged_net_weight") if is_currency else ("name", "weight")
        for row in _list(result[key], key, 256):
            item = _object(row, key, required={name_key, weight_key})
            item[name_key] = _currency(item[name_key]) if is_currency else _text(item[name_key], key, maximum=200)
            if item[name_key] in names:
                raise ValidationError(f"{key}包含重复分类；必须先核实并合并同口径数据")
            names.add(item[name_key])
            item[weight_key] = _ratio(item[weight_key], weight_key)
            rows.append(item)
        _total([row[weight_key] for row in rows], key)
        result[key] = sorted(rows, key=lambda row: row[name_key])
    result["holdings"] = sorted(holdings, key=lambda row: (row["issuer_id"], row["kind"], row["name"], row["weight"]))
    if "duration_years" in result:
        result["duration_years"] = None if result["duration_years"] is None else _number(result["duration_years"], "久期", maximum="100")
    if "rate_reference" in result:
        reference = _object(result["rate_reference"], "利率适配依据", required={"series_id", "currency", "tenor_years"})
        reference["series_id"] = _text(reference["series_id"], "利率序列标识", maximum=200)
        reference["currency"] = _currency(reference["currency"])
        reference["tenor_years"] = _number(reference["tenor_years"], "利率适配期限", maximum="100")
        if Decimal(reference["tenor_years"]) <= 0:
            raise ValidationError("利率适配期限必须为正；不能从久期猜测期限")
        if reference["series_id"] == "DGS10" and (reference["currency"], reference["tenor_years"]) != ("USD", "10"):
            raise ValidationError("DGS10利率适配必须明确USD十年期限")
        result["rate_reference"] = reference
    return result


def _management(value: Any) -> dict:
    result = _object(value, "管理人", required={"managers"})
    managers = []
    for row in _list(result["managers"], "现任经理", 32):
        manager = _object(row, "经理", required={"name", "start_date"})
        manager["name"] = _text(manager["name"], "经理姓名", maximum=200)
        manager["start_date"] = _day(manager["start_date"], "任职日期")
        managers.append(manager)
    if len({row["name"] for row in managers}) != len(managers):
        raise ValidationError("现任经理姓名重复")
    result["managers"] = sorted(managers, key=lambda row: (row["name"], row["start_date"]))
    return result


def _scale(value: Any) -> dict:
    result = _object(value, "基金规模", required={"total_assets_cny", "combined_share_classes"})
    result["total_assets_cny"] = _number(result["total_assets_cny"], "人民币合并资产")
    result["combined_share_classes"] = _bool(result["combined_share_classes"], "是否合并各份额")
    return result


def _fees(value: Any) -> dict:
    result = _object(value, "费用", required={
        "management_rate", "custody_rate", "sales_service_rate", "redemption_bands",
        "holding_day_basis", "channel_verified",
    }, optional={"subscription_rate"})
    for key in ("management_rate", "custody_rate", "sales_service_rate", "subscription_rate"):
        if key in result:
            result[key] = None if result[key] is None else _ratio(result[key], key)
    result["holding_day_basis"] = _enum(result["holding_day_basis"], {"confirmation_calendar_days", "unknown"}, "持有日口径")
    result["channel_verified"] = _bool(result["channel_verified"], "渠道核实")
    bands = []
    for row in _list(result["redemption_bands"], "赎回费分段", 128):
        band = _object(row, "赎回费分段", required={"min_days", "max_days", "rate"})
        band["min_days"] = _integer(band["min_days"], "最低持有日", maximum=100000)
        band["max_days"] = None if band["max_days"] is None else _integer(band["max_days"], "最高持有日", minimum=band["min_days"], maximum=100000)
        band["rate"] = _ratio(band["rate"], "赎回费率")
        bands.append(band)
    bands.sort(key=lambda row: row["min_days"])
    for previous, current in zip(bands, bands[1:]):
        if previous["max_days"] is None or current["min_days"] <= previous["max_days"]:
            raise ValidationError("赎回费分段交叠；max_days采用含端点口径")
    result["redemption_bands"] = bands
    return result


def _operations(value: Any) -> dict:
    result = _object(value, "申赎状态", required={"subscription_state", "redemption_state", "channel_verified"}, optional={"limit_amount"})
    for key in ("subscription_state", "redemption_state"):
        result[key] = _enum(result[key], {"open", "closed", "limited", "unknown"}, key)
    result["channel_verified"] = _bool(result["channel_verified"], "渠道核实")
    if "limit_amount" in result:
        result["limit_amount"] = None if result["limit_amount"] is None else _number(result["limit_amount"], "限额")
    return result


def _events(value: Any) -> dict:
    result = _object(value, "正式事件", required={"events"})
    events = []
    for row in _list(result["events"], "正式事件", 256):
        item = _object(row, "事件", required={"event_id", "type", "effective_date", "official", "reassessed", "description"})
        item["event_id"] = _text(item["event_id"], "事件标识", maximum=200)
        item["type"] = _enum(item["type"], {"termination", "mandate_change", "other"}, "事件类别")
        item["effective_date"] = _day(item["effective_date"], "事件生效日")
        item["official"] = _bool(item["official"], "正式公告")
        item["reassessed"] = _bool(item["reassessed"], "已重新评估")
        item["description"] = _text(item["description"], "事件说明", maximum=8000)
        events.append(item)
    if len({row["event_id"] for row in events}) != len(events):
        raise ValidationError("事件标识重复")
    result["events"] = sorted(events, key=lambda row: row["event_id"])
    return result


def _valuation(value: Any) -> dict:
    result = _object(value, "基准估值", required={
        "benchmark_id", "region", "pe_ttm", "pb", "pe_definition", "pb_definition", "compatible_scope",
    })
    for key in ("benchmark_id", "region", "pb_definition"):
        result[key] = _text(result[key], key, maximum=300)
    result["pe_definition"] = _enum(result["pe_definition"], {"trailing", "forward", "unknown"}, "市盈率口径")
    for key in ("pe_ttm", "pb"):
        # Nonpositive metrics cannot activate a valuation guard; retain them as
        # unavailable rather than converting a negative P/E into cheapness.
        number = None if result[key] is None else decimal(result[key])
        result[key] = None if number is None or number <= 0 else _number(number, key)
    result["compatible_scope"] = _bool(result["compatible_scope"], "投资范围匹配")
    return result


def _macro(value: Any) -> dict:
    result = _object(value, "市场序列", required={"kind", "series_id", "currency", "unit", "observations"}, optional={"tenor_years"})
    result["kind"] = _enum(result["kind"], {"yield", "fx"}, "市场序列类型")
    result["series_id"] = _text(result["series_id"], "市场序列标识", maximum=200)
    result["currency"] = _currency(result["currency"])
    result["unit"] = _enum(result["unit"], {"percent", "CNY_per_currency"}, "市场序列单位")
    if (result["kind"] == "yield") != (result["unit"] == "percent"):
        raise ValidationError("收益率必须用percent；汇率必须用CNY_per_currency")
    if "tenor_years" in result:
        result["tenor_years"] = None if result["tenor_years"] is None else _number(result["tenor_years"], "收益率期限", maximum="100")
    if result["kind"] == "fx" and result.get("tenor_years") is not None:
        raise ValidationError("汇率序列不能伪造债券期限")
    if result["series_id"] == "DGS10" and not (result["kind"] == "yield" and result["currency"] == "USD" and result.get("tenor_years") == "10"):
        raise ValidationError("DGS10身份必须为USD十年期percent收益率")
    if result["series_id"] == "DEXCHUS" and not (result["kind"] == "fx" and result["currency"] == "USD"):
        raise ValidationError("DEXCHUS身份必须为CNY/USD汇率")
    observations = []
    for row in _list(result["observations"], "市场观察", 2048):
        item = _object(row, "市场观察", required={"date", "value"})
        item["date"] = _day(item["date"], "市场观察日")
        item["value"] = _number(item["value"], "市场观察值", minimum="-100" if result["kind"] == "yield" else "0")
        if result["kind"] == "fx" and Decimal(item["value"]) <= 0:
            raise ValidationError("人民币每外币单位价格必须为正")
        observations.append(item)
    if not observations or len({row["date"] for row in observations}) != len(observations):
        raise ValidationError("市场序列必须有非重复的实际观察")
    result["observations"] = sorted(observations, key=lambda row: row["date"])
    return result


def _signed_number(value: Any, label: str) -> str:
    """Issuer losses/cash flows remain signed; legacy nonnegative fields do not change."""
    result = decimal(value)
    if not result:
        return "0"
    text = format(result, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _issuer_period(value: Any, *, guidance: bool = False) -> dict:
    result = _object(value, "财务期间", required={
        "start", "end", "basis", "fiscal_year", "fiscal_quarter",
    })
    result["start"] = _day(result["start"], "财务期间start")
    if result["end"] is None:
        if not guidance:
            raise ValidationError("实际财务期间必须有明确结束日期")
    else:
        result["end"] = _day(result["end"], "财务期间end")
        if result["end"] < result["start"]:
            raise ValidationError("财务期间起止倒置")
    result["basis"] = _enum(result["basis"], {"quarter", "month", "year", "ytd", "instant"}, "财务期间口径")
    if result["basis"] == "instant":
        if result["end"] != result["start"]:
            raise ValidationError("时点余额须用instant且start=end，不能冒充一段现金流")
    elif result["end"] is not None:
        span = (date.fromisoformat(result["end"]) - date.fromisoformat(result["start"])).days + 1
        bounds = {"quarter": (70, 105), "month": (27, 35), "year": (300, 400), "ytd": (1, 400)}
        low, high = bounds[result["basis"]]
        if not low <= span <= high:
            raise ValidationError("财务期间长度与明确声明的季度/月度/年度/累计口径不匹配")
    result["fiscal_year"] = _integer(result["fiscal_year"], "财年", minimum=1900, maximum=2200)
    quarter = result["fiscal_quarter"]
    if result["basis"] in {"quarter", "ytd"} or (result["basis"] == "instant" and quarter is not None):
        result["fiscal_quarter"] = _integer(quarter, "财季", minimum=1, maximum=4)
    elif quarter is not None:
        raise ValidationError("月度/年度期间不能冒充单一财季")
    return result


def _issuer_metric(value: Any, release_security_id: str | None) -> dict:
    result = _object(value, "发行人财务指标", required={
        "name", "category", "period", "currency", "unit", "scale", "accounting_basis",
        "period_role", "locator",
    }, optional={
        "value", "low", "high", "security_id", "eps_basis", "share_basis", "split_basis",
        "accounting_basis_detail", "comparison_basis", "cash_capex", "debt_definition",
        "cash_flow_definition", "debt_maturity_scope",
    })
    debts = {"total_debt", "short_term_debt", "long_term_debt"}
    balances = {"cash_and_cash_equivalents", "accounts_receivable", "inventory", *debts}
    result["name"] = _enum(result["name"], {
        "eps", "revenue", "gross_margin", "capex", "operating_cash_flow", "free_cash_flow",
        "net_income", *balances,
    }, "财务指标名称")
    result["category"] = _enum(result["category"], {"actual", "guidance"}, "实际/指引类别")
    result["period"] = _issuer_period(result["period"], guidance=result["category"] == "guidance")
    if (result["name"] in balances) != (result["period"]["basis"] == "instant"):
        raise ValidationError("现金/应收/存货/有息债务须为instant余额；利润和现金流须为flow期间")
    if "cash_capex" in result:
        if result["name"] != "capex":
            raise ValidationError("只有capex可以声明现金资本开支定义及符号方向")
        cash_capex = _object(result["cash_capex"], "现金资本开支口径", required={
            "definition", "sign_convention",
        })
        cash_capex["definition"] = _text(cash_capex["definition"], "原文现金资本开支定义", maximum=2000)
        cash_capex["sign_convention"] = _enum(cash_capex["sign_convention"], {
            "outflow_positive", "outflow_negative",
        }, "现金资本开支符号方向")
        result["cash_capex"] = cash_capex
    if "cash_flow_definition" in result:
        if result["name"] != "free_cash_flow":
            raise ValidationError("只有披露FCF可以声明与原文经营现金流及现金资本开支的定义桥接")
        definition = _object(result["cash_flow_definition"], "披露FCF定义桥接", required={
            "definition", "ocf_accounting_basis", "cash_capex_definition",
        }, optional={"ocf_accounting_basis_detail"})
        for key in ("definition", "cash_capex_definition", "ocf_accounting_basis_detail"):
            if key in definition:
                definition[key] = _text(definition[key], f"FCF定义{key}", maximum=2000)
        definition["ocf_accounting_basis"] = _enum(definition["ocf_accounting_basis"], {
            "GAAP", "IFRS", "non_GAAP",
        }, "FCF原文组成项OCF会计口径")
        if definition["ocf_accounting_basis"] == "non_GAAP" and not definition.get("ocf_accounting_basis_detail"):
            raise ValidationError("FCF桥接的非GAAP经营现金流须明确具体调整口径")
        result["cash_flow_definition"] = definition
    if result["name"] in debts:
        result["debt_definition"] = _text(result.get("debt_definition"), "有息债务覆盖定义", maximum=2000)
    elif "debt_definition" in result:
        raise ValidationError("只有有息债务可以声明债务覆盖定义，不能用总负债代替")
    if result["name"] in {"short_term_debt", "long_term_debt"}:
        result["debt_maturity_scope"] = _enum(result.get("debt_maturity_scope"), {
            "current", "noncurrent", "current_and_noncurrent", "unknown",
        }, "有息债务分项流动/非流动覆盖")
    elif "debt_maturity_scope" in result:
        raise ValidationError("只有债务分项可以声明流动/非流动覆盖")
    if "comparison_basis" in result:
        result["comparison_basis"] = _text(result["comparison_basis"], "可比指标原文定义", maximum=1000)
    result["period_role"] = _enum(result["period_role"], {"primary", "comparative"}, "本期/比较期")
    result["accounting_basis"] = _enum(result["accounting_basis"], {
        "GAAP", "IFRS", "non_GAAP", "unknown",
    }, "会计口径")
    if "accounting_basis_detail" in result:
        result["accounting_basis_detail"] = _text(result["accounting_basis_detail"], "会计口径定义", maximum=2000)
    if result["accounting_basis"] == "non_GAAP" and not result.get("accounting_basis_detail"):
        raise ValidationError("非GAAP指标必须保留具体调整口径定义，不能把不同调整口径视为相同")
    result["locator"] = _text(result["locator"], "指标原文定位", maximum=1000)
    expected_unit = ("currency_per_share" if result["name"] == "eps" else
                     "fraction" if result["name"] == "gross_margin" else "currency")
    result["unit"] = _enum(result["unit"], {expected_unit}, "财务数值单位")
    result["scale"] = _number(result["scale"], "原文数值单位倍率")
    if Decimal(result["scale"]) <= 0:
        raise ValidationError("财务单位倍率必须明确且为正")
    if expected_unit == "fraction":
        if result["currency"] is not None or result["scale"] != "1":
            raise ValidationError("毛利率采用fraction、无币种且倍率为1；百分数须先按原文单位转换")
    else:
        result["currency"] = _currency(result["currency"])
    if result["category"] == "actual":
        if "value" not in result or {"low", "high"} & result.keys():
            raise ValidationError("实际财务指标必须是单一带符号数值，不能混用指引区间")
        result["value"] = _signed_number(result["value"], "实际财务值")
    else:
        if not {"low", "high"} <= result.keys() or "value" in result:
            raise ValidationError("财务指引须明确上下限；单点指引用相同上下限，不猜测区间")
        for key in ("low", "high"):
            result[key] = _signed_number(result[key], f"指引{key}")
        if Decimal(result["low"]) > Decimal(result["high"]):
            raise ValidationError("财务指引上下限倒置")
    if "security_id" in result:
        result["security_id"] = _text(result["security_id"], "证券标识", maximum=300)
    if result["name"] == "eps":
        if not {"eps_basis", "share_basis", "split_basis"} <= result.keys():
            raise ValidationError("EPS须明确基本/稀释、普通股/ADS及可比拆股口径")
        result["security_id"] = result.get("security_id", release_security_id)
        if result["security_id"] is None:
            raise ValidationError("EPS必须明确对应的证券，不能从公司名称推断普通股/ADS")
        result["eps_basis"] = _enum(result["eps_basis"], {"basic", "diluted"}, "EPS股数口径")
        result["share_basis"] = _enum(result["share_basis"], {"ordinary", "ADS"}, "EPS证券口径")
        result["split_basis"] = _text(result["split_basis"], "EPS可比拆股口径", maximum=500)
    elif {"eps_basis", "share_basis", "split_basis"} & result.keys():
        raise ValidationError("只有EPS指标可以声明每股/ADS口径")
    return result


def _issuer_fundamentals(value: Any) -> dict:
    """Three small source-located payloads, not a general financial ETL schema.

    Release/revision IDs identify the imported document version; they do not
    authenticate publisher authorship. Raw evidence stays in existing captures.
    """
    if not isinstance(value, Mapping):
        raise ValidationError("发行人基本面必须是对象")
    kind = _enum(value.get("kind"), {"financial_release", "security_mapping", "price_history"}, "发行人资料类型")
    if kind == "financial_release":
        result = _object(value, "财务公告", required={
            "kind", "issuer_id", "issuer_name", "release_id", "revision_id", "revision_kind", "metrics",
        }, optional={"security_id", "revises_revision_id"})
        for key in ("issuer_id", "issuer_name", "release_id", "revision_id", "security_id", "revises_revision_id"):
            if key in result:
                result[key] = _text(result[key], key, maximum=300)
        result["revision_kind"] = _enum(result["revision_kind"], {"original", "revision"}, "公告版本类型")
        if result["revision_kind"] == "revision":
            if not result.get("revises_revision_id") or result["revises_revision_id"] == result["revision_id"]:
                raise ValidationError("修订公告必须明确所修订的不同版本标识")
        elif "revises_revision_id" in result:
            raise ValidationError("原始公告不能同时声明修订前版本")
        metrics = [_issuer_metric(row, result.get("security_id"))
                   for row in _list(result["metrics"], "公告指标", 256)]
        if not metrics or len({bounded_json(row) for row in metrics}) != len(metrics):
            raise ValidationError("财务公告必须有非重复的实际指标或指引")
        result["metrics"] = sorted(metrics, key=bounded_json)
        return result
    if kind == "security_mapping":
        result = _object(value, "证券对应关系", required={
            "kind", "issuer_id", "issuer_name", "security_id", "symbol", "exchange",
            "quote_currency", "share_basis", "locator",
        }, optional={"isin", "ads_ratio"})
        for key in ("issuer_id", "issuer_name", "security_id", "symbol", "exchange", "locator", "isin"):
            if key in result:
                result[key] = _text(result[key], key, maximum=1000 if key == "locator" else 300)
        if "isin" in result and not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", result["isin"]):
            raise ValidationError("ISIN须明确为十二位证券标识")
        result["quote_currency"] = _currency(result["quote_currency"])
        result["share_basis"] = _enum(result["share_basis"], {"ordinary", "ADS"}, "证券每股口径")
        if "ads_ratio" in result:
            if result["share_basis"] != "ADS":
                raise ValidationError("只有ADS映射可以附带原文明示的ADS比例")
            ratio = _object(result["ads_ratio"], "ADS比例", required={"ordinary_shares", "ads"})
            for key in ratio:
                ratio[key] = _number(ratio[key], "ADS比例")
                if Decimal(ratio[key]) <= 0:
                    raise ValidationError("ADS比例必须为正；不从名称或币种推断")
            result["ads_ratio"] = ratio
        return result
    result = _object(value, "证券收盘序列", required={
        "kind", "security_id", "symbol", "currency", "exchange", "price_basis", "closes",
    })
    for key in ("security_id", "symbol", "exchange"):
        result[key] = _text(result[key], key, maximum=300)
    result["currency"] = _currency(result["currency"])
    basis = _object(result["price_basis"], "价格调整口径", required={
        "kind", "split_basis", "description", "locator",
    })
    basis["kind"] = _enum(basis["kind"], {"unadjusted", "split_adjusted", "source_adjusted", "unknown"}, "价格调整类型")
    for key in ("description", "locator"):
        basis[key] = _text(basis[key], f"价格口径{key}", maximum=2000 if key == "description" else 1000)
    if basis["split_basis"] is not None:
        basis["split_basis"] = _text(basis["split_basis"], "价格可比拆股口径", maximum=500)
    if basis["kind"] in {"unadjusted", "split_adjusted"} and basis["split_basis"] is None:
        raise ValidationError("可比收盘价须有原文支持的每股/拆股口径键；不是证明全历史没有拆股")
    result["price_basis"] = basis
    closes = []
    for row in _list(result["closes"], "实际收盘价", 4096):
        item = _object(row, "收盘价", required={"date", "close"})
        item["date"] = _day(item["date"], "收盘日")
        item["close"] = _number(item["close"], "收盘价")
        if Decimal(item["close"]) <= 0:
            raise ValidationError("实际收盘价必须为正")
        closes.append(item)
    if not closes or len({row["date"] for row in closes}) != len(closes):
        raise ValidationError("收盘序列必须包含非重复的实际日期")
    result["closes"] = sorted(closes, key=lambda row: row["date"])
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError(f"{label}必须是64位小写SHA-256")
    return value


def _unique_texts(value: Any, label: str, *, maximum: int = 32, text_maximum: int = 200) -> list[str]:
    result = [_text(item, label, maximum=text_maximum) for item in _list(value, label, maximum)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{label}不能重复")
    return result


def _claim_hash(result: dict, text_key: str, hash_key: str) -> None:
    actual = hashlib.sha256(result[text_key].encode("utf-8")).hexdigest()
    if hash_key in result and _sha256(result[hash_key], hash_key) != actual:
        raise ValidationError(f"{hash_key}与保留的文本不一致；文本哈希不是来源方认证")
    result[hash_key] = actual


def _research_citation(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise ValidationError("研究引用必须是对象")
    kind = _enum(value.get("kind"), _RESEARCH_CITATION_KINDS, "研究引用类型")
    common = {"id", "kind", "locator", "claim"}
    if kind == "archived_evidence":
        result = _object(value, "存档引用", required=common | {"record_id", "capture_id"},
                         optional={"claim_sha256"})
        for key in ("record_id", "capture_id"):
            result[key] = _sha256(result[key], key)
    else:
        required = common | {"publisher_url", "title", "publication_precision", "published_at", "checked_at"}
        optional = {"claim_sha256"}
        if kind == "reviewed_excerpt":
            required |= {"excerpt"}
            optional |= {"excerpt_sha256"}
        result = _object(value, "未存档公开引用", required=required, optional=optional)
        result["publisher_url"] = normalize_source_url(result["publisher_url"])
        result["title"] = _text(result["title"], "引用标题", maximum=500)
        checked = timestamp(result["checked_at"])
        result["checked_at"] = checked.isoformat()
        precision = _enum(result["publication_precision"], {"unknown", "date", "timestamp"}, "引用披露时间精度")
        published = result["published_at"]
        if precision == "unknown":
            if published is not None:
                raise ValidationError("引用披露时间未知时不能附带伪精确日期")
        elif precision == "date":
            published = _day(published, "引用披露日期")
            if published > checked.date().isoformat():
                raise ValidationError("引用核对日期不能早于披露日期")
        else:
            published = timestamp(published).isoformat()
            if timestamp(published) > checked:
                raise ValidationError("引用核对时间不能早于精确披露时间")
        result["published_at"] = published
        if kind == "reviewed_excerpt":
            result["excerpt"] = _text(result["excerpt"], "人工核对摘录", maximum=2000)
            _claim_hash(result, "excerpt", "excerpt_sha256")
    result["id"] = _text(result["id"], "引用标识", maximum=80)
    result["locator"] = _text(result["locator"], "引用定位", maximum=1000)
    result["claim"] = _text(result["claim"], "引用支持的短主张", maximum=1000)
    _claim_hash(result, "claim", "claim_sha256")
    return result


def normalize_research_thesis(value: Any) -> dict:
    """Bounded authored reasoning, never a personal portfolio or model score.

    An external URL is not an archived original; an excerpt hash covers only the
    entered excerpt. Record/capture references are structurally checked here and
    resolved against existing, as-of evidence by the local import adapter.
    """
    if isinstance(value, Mapping) and value.get("schema_version") == 2:
        from .research_contract import normalize_generated_research

        return normalize_generated_research(value)
    bounded_json(value)
    result = _object(value, "研究论题", required={
        "product", "note_id", "author", "reviewer", "research_date", "horizon", "posture",
        "facts", "citations", "assessment", "revision",
    })
    product = _object(result["product"], "研究产品身份", required={
        "id", "symbol", "name", "kind", "share_class", "exchange",
    })
    product["id"] = _text(product["id"], "研究产品id", maximum=80)
    for key in ("symbol", "name"):
        product[key] = _text(product[key], f"研究产品{key}", maximum=200)
    product["kind"] = _enum(product["kind"], {"etf", "open_fund", "monitor_only"}, "研究产品类型")
    for key in ("share_class", "exchange"):
        product[key] = _text(product[key], f"研究产品{key}", maximum=80, empty=True)
    result["product"] = product
    for key in ("note_id", "author", "reviewer"):
        result[key] = _text(result[key], key, maximum=200)
    result["research_date"] = _day(result["research_date"], "研究日期")
    horizon = _object(result["horizon"], "研究期限", required={"start", "end"})
    for key in horizon:
        horizon[key] = _day(horizon[key], f"研究期限{key}")
    if not horizon["start"] <= result["research_date"] <= horizon["end"]:
        raise ValidationError("研究日期必须位于明确的研究期限内")
    if (date.fromisoformat(horizon["end"]) - date.fromisoformat(horizon["start"])).days > 3660:
        raise ValidationError("研究期限不能超过十年")
    result["horizon"] = horizon
    result["posture"] = _enum(result["posture"], {"accumulate", "maintain", "trim", "exit", "unresolved"}, "研究姿态")
    citations = [_research_citation(row) for row in _list(result["citations"], "研究引用", 32)]
    citation_ids = {row["id"] for row in citations}
    if len(citation_ids) != len(citations):
        raise ValidationError("研究引用标识不能重复")
    result["citations"] = sorted(citations, key=lambda row: row["id"])
    facts = []
    for row in _list(result["facts"], "研究事实", 64):
        fact = _object(row, "研究事实", required={
            "id", "claim", "status", "effective_date", "issuer_ids", "channels", "citation_ids",
        })
        fact["id"] = _text(fact["id"], "研究事实标识", maximum=80)
        fact["claim"] = _text(fact["claim"], "研究事实短主张", maximum=2000)
        fact["status"] = _enum(fact["status"], _RESEARCH_FACT_STATUSES, "事实/计划/生效状态")
        if fact["effective_date"] is not None:
            fact["effective_date"] = _day(fact["effective_date"], "事实生效日期")
            if fact["status"] in {"factual", "effective"} and fact["effective_date"] > result["research_date"]:
                raise ValidationError("未来计划不能冒充研究日已经发生或生效的事实")
        elif fact["status"] == "effective":
            raise ValidationError("已生效事实必须明确实际生效日期")
        fact["issuer_ids"] = sorted(_unique_texts(fact["issuer_ids"], "受影响发行人", maximum=32, text_maximum=300))
        channels = _unique_texts(fact["channels"], "传导渠道", maximum=8)
        if not channels:
            raise ValidationError("研究事实须明确至少一个传导渠道")
        fact["channels"] = sorted(_enum(channel, _RESEARCH_CHANNELS, "传导渠道") for channel in channels)
        references = _unique_texts(fact["citation_ids"], "事实引用", maximum=32, text_maximum=80)
        if not references or not set(references) <= citation_ids:
            raise ValidationError("研究事实必须引用本论题实际存在的资料标识")
        fact["citation_ids"] = sorted(references)
        referenced = [citation for citation in citations if citation["id"] in references]
        if fact["status"] != "unresolved" and all(citation["kind"] == "linked_not_archived" for citation in referenced):
            raise ValidationError("只有URL链接不能确证事实或计划状态；须保留unresolved或核对摘录/存档")
        facts.append(fact)
    fact_ids = {row["id"] for row in facts}
    if len(fact_ids) != len(facts):
        raise ValidationError("研究事实标识不能重复")
    result["facts"] = sorted(facts, key=lambda row: row["id"])
    assessment = _object(result["assessment"], "作者研究判断", required={
        "summary", "supporting", "counterevidence", "priced_in", "missing", "falsifiers",
        "change_conditions", "review_date",
    })
    assessment["summary"] = _text(assessment["summary"], "研究结论", maximum=4000)
    for key in ("supporting", "counterevidence"):
        reasons = []
        for row in _list(assessment[key], key, 32):
            reason = _object(row, "论据", required={"reason", "fact_ids"})
            reason["reason"] = _text(reason["reason"], "作者论据", maximum=2000)
            references = _unique_texts(reason["fact_ids"], "论据事实引用", maximum=64, text_maximum=80)
            if not references or not set(references) <= fact_ids:
                raise ValidationError("作者论据必须引用本论题实际存在的事实标识")
            reason["fact_ids"] = sorted(references)
            reasons.append(reason)
        assessment[key] = reasons
    priced_in = _object(assessment["priced_in"], "价格反映程度", required={"status", "reason"})
    priced_in["status"] = _enum(priced_in["status"], {"unknown", "contested"}, "价格是否已反映")
    priced_in["reason"] = _text(priced_in["reason"], "价格反映研究依据", maximum=2000)
    assessment["priced_in"] = priced_in
    for key in ("missing", "falsifiers", "change_conditions"):
        assessment[key] = _unique_texts(assessment[key], key, maximum=32, text_maximum=2000)
    if not assessment["falsifiers"] or not assessment["change_conditions"]:
        raise ValidationError("研究判断必须明确可推翻判断及改变姿态的条件")
    assessment["review_date"] = _day(assessment["review_date"], "下次研究复核日")
    if not result["research_date"] <= assessment["review_date"] <= horizon["end"]:
        raise ValidationError("复核日必须位于研究日至研究期限结束之间")
    supported = {fact_id for reason in assessment["supporting"] for fact_id in reason["fact_ids"]}
    if result["posture"] != "unresolved" and not any(
        fact["id"] in supported and fact["status"] != "unresolved" for fact in facts
    ):
        raise ValidationError("明确研究姿态必须有带引用且不是未确证状态的事实与作者支持论据")
    result["assessment"] = assessment
    revision = _object(result["revision"], "研究修订", required={"id", "supersedes"})
    revision["id"] = _text(revision["id"], "研究修订标识", maximum=200)
    if revision["supersedes"] is not None:
        revision["supersedes"] = _sha256(revision["supersedes"], "被替代研究记录")
    result["revision"] = revision
    bounded_json(result)
    return result


def research_thesis_content(payload: Mapping) -> dict:
    """Canonical authored content; citation check clocks never renew a thesis."""
    value = normalize_research_thesis(payload)
    value["citations"] = [
        {key: item for key, item in citation.items() if key != "checked_at"}
        for citation in value["citations"]
    ]
    return value


def research_thesis_source(payload: Mapping) -> str:
    if payload.get("schema_version") != 2:
        return "local_authored_research"
    if payload["origin"]["provider"] == "interactive_assistant":
        return "assistant_generated_research"
    return "api_generated_research"


def research_thesis_source_uri(product_id: str, payload: Mapping) -> str:
    """Generate an inert local content locator, not a publisher URL."""
    value = research_thesis_content(payload)
    if product_id != value["product"]["id"]:
        raise ValidationError("研究论题产品身份与记录不匹配")
    product = value.pop("product")
    origin = {"assistant_generated_research": "assistant", "api_generated_research": "api",
              "local_authored_research": "local"}[research_thesis_source(value)]
    return f"rmb-research://{origin}/" + digest({"product": product, "content": value})


def _record_source_url(family: str, value: Any) -> str:
    if family != "research_thesis":
        return normalize_source_url(value)
    if not isinstance(value, str) or _LOCAL_RESEARCH_URI.fullmatch(value) is None:
        raise ValidationError("研究来源须为rmb-research://local/、api/或assistant/加64位内容标识")
    return value


def _research_raw_note(raw: bytes) -> dict:
    def unique(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValidationError("研究原始JSON包含重复键")
            result[key] = item
        return result

    def finite(value: str) -> None:
        raise ValidationError(f"研究原始JSON包含非有限值: {value}")

    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_RECORD_BYTES:
        raise ValidationError("研究原始笔记必须为不超过256KiB的非空JSON字节")
    try:
        value = json.loads(raw.decode("utf-8-sig", "strict"), object_pairs_hook=unique,
                           parse_constant=finite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("研究原始笔记不是有效的有限UTF-8 JSON") from exc
    return normalize_research_thesis(value)


_VALIDATORS = {
    "mandate": _mandate, "portfolio": _portfolio, "management": _management,
    "scale": _scale, "fees": _fees, "operations": _operations, "events": _events,
    "valuation": _valuation, "macro": _macro, "issuer_fundamentals": _issuer_fundamentals,
    "research_thesis": normalize_research_thesis,
}


def normalize_record(value: Mapping) -> dict:
    """Return a detached normalized record, rejecting ambiguous/unknown inputs.

    This is a validation function, not authentication of a public document. The
    import adapter must assign actual retrieval/review times; storage also checks
    the linked capture so a first insertion cannot predate its real retrieval.
    """
    bounded_json(value)
    result = _object(value, "资料记录", required={
        "product_id", "family", "source", "upstream", "source_url", "report_date",
        "observed_at", "available_at", "payload",
    }, optional={
        "id", "schema_version", "normalization_version", "period_start", "effective_from",
        "effective_to", "published_at", "publication_precision",
    })
    version = result.get("schema_version", SCHEMA_VERSION)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValidationError("不支持的资料schema_version")
    if result.get("normalization_version", NORMALIZATION_VERSION) != NORMALIZATION_VERSION:
        raise ValidationError("不支持的资料规范化版本")
    result["schema_version"] = SCHEMA_VERSION
    result["normalization_version"] = NORMALIZATION_VERSION
    for key in ("product_id", "source", "upstream"):
        result[key] = _text(result[key], key, maximum=200)
    result["family"] = _enum(result["family"], FAMILIES, "资料家族")
    result["source_url"] = _record_source_url(result["family"], result["source_url"])
    result["report_date"] = _day(result["report_date"], "报告/观察日")
    observed, available = timestamp(result["observed_at"]), timestamp(result["available_at"])
    if available < observed or result["report_date"] > observed.date().isoformat():
        raise ValidationError("可用时间不能早于取得时间；报告期不能晚于实际取得日")
    result["observed_at"], result["available_at"] = observed.isoformat(), available.isoformat()
    for key in ("period_start", "effective_from", "effective_to"):
        result[key] = None if result.get(key) is None else _day(result[key], key)
    if result["period_start"] and result["period_start"] > result["report_date"]:
        raise ValidationError("报告区间起止倒置")
    if result["effective_to"] and (not result["effective_from"] or result["effective_to"] < result["effective_from"]):
        raise ValidationError("有效区间必须完整且顺序正确")
    precision = _enum(result.get("publication_precision", "unknown"), {"unknown", "date", "timestamp"}, "披露时间精度")
    published = result.get("published_at")
    if precision == "unknown":
        if published is not None:
            raise ValidationError("披露时间未知时不能附带伪精确日期")
    elif precision == "date":
        published = _day(published, "披露日期")
        if published > available.date().isoformat():
            raise ValidationError("资料可用日期不能早于披露日期")
    else:
        published = timestamp(published).isoformat()
        if timestamp(published) > available:
            raise ValidationError("资料可用时间不能早于精确披露时间")
    result["published_at"], result["publication_precision"] = published, precision
    result["payload"] = _VALIDATORS[result["family"]](result["payload"])
    if result["family"] == "management" and any(row["start_date"] > result["report_date"] for row in result["payload"]["managers"]):
        raise ValidationError("现任经理的实际上任日不能晚于报告日")
    if result["family"] == "macro" and result["payload"]["observations"][-1]["date"] != result["report_date"]:
        raise ValidationError("市场序列报告日必须是最后实际观察日")
    if result["family"] == "issuer_fundamentals":
        payload = result["payload"]
        if payload["kind"] == "financial_release":
            for metric in payload["metrics"]:
                if metric["category"] == "actual" and (
                    metric["period"]["end"] > result["report_date"]
                    or (published is not None and metric["period"]["end"] > str(published)[:10])
                ):
                    raise ValidationError("实际财务期间不能晚于报告日或原始公开披露日；未来目标须作为指引")
        elif payload["kind"] == "price_history" and payload["closes"][-1]["date"] != result["report_date"]:
            raise ValidationError("证券价格报告日必须为最后实际收盘日")
    if result["family"] == "research_thesis":
        payload = result["payload"]
        source = research_thesis_source(payload)
        if (result["source"] != source or result["upstream"] != source
                or result["source_url"] != research_thesis_source_uri(result["product_id"], payload)):
            raise ValidationError("研究论题来源须区分本地作者与API生成，且URI须匹配规范化产品与内容")
        if (payload.get("schema_version") == 2
                and timestamp(payload["origin"]["generated_at"]) > observed):
            raise ValidationError("API研究记录不能早于实际生成时间")
        if (result["report_date"] != payload["research_date"] or precision != "unknown"
                or any(result[key] is not None for key in ("period_start", "effective_from", "effective_to"))):
            raise ValidationError("研究记录使用研究日期，不能伪称公共披露或正式文件生效期间")
        for citation in payload["citations"]:
            if "checked_at" in citation and timestamp(citation["checked_at"]) > available:
                raise ValidationError("引用核对时间不能晚于实际研究可用时间")
    identity = {key: item for key, item in result.items() if key not in {"id", "observed_at", "available_at"}}
    if result["family"] == "research_thesis":
        identity["payload"] = research_thesis_content(result["payload"])
    record_id = digest(identity)
    if result.get("id") and result["id"] != record_id:
        raise ValidationError("资料ID与规范化经济内容不一致")
    result["id"] = record_id
    bounded_json(result)
    return result


def normalize_capture(value: Mapping) -> dict:
    """Validate bytes and capture provenance; raw hashes are always computed here."""
    result = _object(value, "采集记录", required={
        "product_id", "family", "source", "upstream", "source_url", "observed_at", "status",
        "content_type", "raw_bytes", "parser_version", "warnings",
    }, optional={"metadata"})
    for key in ("product_id", "source", "upstream", "parser_version"):
        result[key] = _text(result[key], key, maximum=200)
    result["family"] = _enum(result["family"], FAMILIES, "资料家族")
    result["source_url"] = _record_source_url(result["family"], result["source_url"])
    result["observed_at"] = timestamp(result["observed_at"]).isoformat()
    result["status"] = _enum(result["status"], {"ok", "reviewed", "error"}, "采集状态")
    result["content_type"] = _text(result["content_type"], "内容类型", maximum=200).lower()
    if not re.fullmatch(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+(?:\s*;[^\r\n]*)?", result["content_type"]):
        raise ValidationError("采集内容类型无效")
    raw = result["raw_bytes"]
    if not isinstance(raw, bytes):
        raise ValidationError("采集原始内容必须是bytes；不能执行或隐式转码")
    web_type = any(part in result["content_type"].split(";", 1)[0] for part in ("html", "json", "javascript", "xml", "text/"))
    if len(raw) > (MAX_WEB_BYTES if web_type else MAX_CAPTURE_BYTES):
        raise ValidationError("原始资料超过HTML/JSON 2MiB或公开文档8MiB上限")
    if result["status"] != "error" and not raw:
        raise ValidationError("空文档不能成为成功采集证据")
    result["warnings"] = [_text(item, "采集警告", maximum=2000) for item in _list(result["warnings"], "采集警告", 64)]
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValidationError("采集元数据必须是对象")
    result["metadata"] = json.loads(bounded_json(metadata, max_bytes=64 * 1024))
    reviewed = result["metadata"].get("reviewed_at")
    if reviewed is not None:
        reviewed = timestamp(reviewed)
        if reviewed < timestamp(result["observed_at"]):
            raise ValidationError("复核完成不能早于取得文档")
        result["metadata"]["reviewed_at"] = reviewed.isoformat()
    result["available_at"] = (reviewed or timestamp(result["observed_at"])).isoformat()
    if result["family"] == "research_thesis":
        note = _research_raw_note(raw)
        generated = note.get("schema_version") == 2
        source = research_thesis_source(note)
        if (result["source"] != source or result["upstream"] != source
                or result["source_url"] != research_thesis_source_uri(result["product_id"], note)
                or result["content_type"].split(";", 1)[0] != "application/json"):
            raise ValidationError("研究采集必须保留匹配生成来源、产品与内容URI的JSON研究")
        if generated:
            from .research_contract import GENERATED_VALIDATION

            metadata = result["metadata"]
            completed = timestamp(metadata.get("validation_completed_at"))
            if (result["status"] != "ok" or reviewed is not None
                    or metadata.get("ai_generated") is not True
                    or metadata.get("human_reviewed") is not False
                    or metadata.get("reviewed") is not None
                    or metadata.get("validation") != GENERATED_VALIDATION
                    or timestamp(note["origin"]["generated_at"]) > timestamp(result["observed_at"])
                    or completed < timestamp(result["observed_at"])):
                raise ValidationError("API研究须记录实际生成及程序引用核对时间，不能标为人工reviewed")
            metadata["validation_completed_at"] = completed.isoformat()
            result["available_at"] = completed.isoformat()
        else:
            if (result["status"] != "reviewed" or reviewed is None
                    or result["metadata"].get("reviewed") is not True
                    or note["research_date"] > timestamp(result["observed_at"]).date().isoformat()):
                raise ValidationError("本地研究必须记录实际取得与复核时间，不能补造未来研究或公共抓取")
            if any("checked_at" in citation and timestamp(citation["checked_at"]) > reviewed
                   for citation in note["citations"]):
                raise ValidationError("研究引用核对不能晚于本次复核完成")
    result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    identity = {key: item for key, item in result.items() if key != "raw_bytes"}
    result["id"] = digest(identity)
    bounded_json({key: item for key, item in result.items() if key != "raw_bytes"})
    return result

"""Dated public-fund context and caller-supplied economic what-if calculations.

No requests, storage, clocks, fitting, scores, forecast probabilities or advice.
Weights stay on their disclosed denominator; an unlisted exposure is not zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from typing import Any

from .advice_evidence import normalize_record
from .domain import ValidationError, decimal, timestamp

SCHEMA_VERSION = 1
__all__ = [
    "build_investment_profile", "disclosure_shock_scenario",
    "earnings_valuation_fx_scenario",
]
_FAMILIES = ("mandate", "portfolio", "management", "valuation")
_DENOMINATORS = {
    "nav": "基金净资产", "equity_assets": "股票资产", "total_assets": "基金总资产",
}
_DATES = (
    "report_date", "period_start", "published_at", "publication_precision",
    "observed_at", "available_at", "effective_from", "effective_to",
)


def _text(value: Decimal) -> str:
    if not value:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _pct(value: Any) -> str:
    return f"{decimal(value) * 100:.2f}%"


def _sum(values) -> Decimal:
    return sum((decimal(value) for value in values), Decimal(0))


def _section(record: Mapping, day: date) -> dict:
    return {
        "record_id": record["id"], "source": record["source"],
        "upstream": record["upstream"], "source_url": record["source_url"],
        **{key: record.get(key) for key in _DATES},
        "report_age_calendar_days": (day - date.fromisoformat(record["report_date"])).days,
        "payload": deepcopy(record["payload"]),
    }


def prefer_disclosure(candidate: Mapping, previous: Mapping | None) -> bool:
    """Prefer a later report, but don't discard richer facts from the same document.

    A fresh partial parse of an unchanged report isn't a new portfolio. When two
    parses conflict rather than contain each other, keep the dated selection and
    let the research layer expose the conflict.
    """
    if previous is None:
        return True
    def order(row):
        return row["report_date"], timestamp(row["available_at"]), row["id"]
    if (candidate["family"] != "portfolio" or previous["family"] != "portfolio"
            or candidate["report_date"] != previous["report_date"]
            or candidate["source_url"] != previous["source_url"]
            or candidate["payload"]["denominator"] != previous["payload"]["denominator"]):
        return order(candidate) > order(previous)

    def covers(left, right):
        for key in ("assets",):
            if any(left[key].get(name) != value for name, value in right[key].items()):
                return False
        for key in ("holdings", "sectors", "countries", "currencies"):
            names = {row.get("issuer_id") or row.get("name") or row.get("currency"): row["weight"]
                     for row in left[key]}
            if any(names.get(row.get("issuer_id") or row.get("name") or row.get("currency")) != row["weight"]
                   for row in right[key]):
                return False
        return True

    before, after = previous["payload"], candidate["payload"]
    old_covers_new, new_covers_old = covers(before, after), covers(after, before)
    if old_covers_new != new_covers_old:
        return new_covers_old
    return order(candidate) > order(previous)


def build_investment_profile(records: Sequence[Mapping], *, as_of: str) -> dict:
    """Build one product's detached JSON profile from advice_evidence records.

    ``as_of`` is an explicit timezone-aware timestamp, not today's implicit clock.
    Store-added ``checked_at`` is accepted but never refreshes a report date.
    Latest eligible report per family wins, then availability and content ID.
    This validates records; it does not authenticate their public sources.
    """
    point = timestamp(as_of)
    day = point.date()
    latest: dict[str, dict] = {}
    product_ids = set()
    for row in records:
        record = normalize_record({key: value for key, value in row.items()
                                   if key != "checked_at"})
        product_ids.add(record["product_id"])
        if len(product_ids) > 1:
            raise ValidationError("投资画像一次只能包含同一只基金的资料")
        if (
            timestamp(record["available_at"]) > point
            or record["report_date"] > day.isoformat()
            or (record["effective_from"] and record["effective_from"] > day.isoformat())
            or (record["effective_to"] and record["effective_to"] < day.isoformat())
            or record["family"] not in _FAMILIES
        ):
            continue
        family = record["family"]
        if prefer_disclosure(record, latest.get(family)):
            latest[family] = record
    result = {
        "schema_version": SCHEMA_VERSION, "as_of": point.isoformat(),
        "product_id": next(iter(product_ids), None),
        "kind": "dated_public_investment_profile", "status": "available" if latest else "missing",
        **{family: _section(latest[family], day) if family in latest else None
           for family in _FAMILIES},
        "missing_families": [family for family in _FAMILIES if family not in latest],
        "factor_mapping": [],
        "limitations": [
            "这是公开基金资料画像，不是你的持仓，也不是买卖建议或未来收益预测。",
            "报告日说明当时的情况；披露日、取得日和可用时间分别保留，不能当作今天的实时持仓。",
            "行业、国家、前十大持仓是同一组合的不同切面，不能把它们的权重或情景影响重复相加。",
        ],
    }
    mappings = result["factor_mapping"]

    def mapping(key: str, label: str, section: str, reason: str, missing: str) -> None:
        mappings.append({
            "factor": key, "label": label, "record_id": result[section]["record_id"],
            "reason": reason, "not_established": missing,
        })

    if section := result["mandate"]:
        payload = section["payload"]
        section["explanation"] = (
            f"合同定位是{ {'equity': '股票型', 'bond': '债券型', 'mixed': '混合型', 'other': '其他类型'}[payload['asset_class']] }；"
            "比较基准是业绩对照，不是基金已经按该比例买入的证明。"
        )
        mapping("mandate", "合同允许投什么，与现在实际买了什么分开看", "mandate",
                "用合同范围与基准核对研究对象是否匹配；实际风险比例仍看组合披露。",
                "不能从基准权益比例推算实际仓位、币种或行业。")
    if section := result["portfolio"]:
        payload = section["payload"]
        denominator = payload["denominator"]
        listed = _sum(row["weight"] for row in payload["holdings"])
        section["denominator_label"] = _DENOMINATORS[denominator]
        section["coverage"] = {
            "holdings_scope": payload["holdings_scope"], "holding_count": len(payload["holdings"]),
            "listed_holdings_weight": _text(listed),
            "outside_list_weight": _text(1 - listed),
            "disclosed_equity_weight": payload["assets"].get("equity"),
            "explanation": (
                f"持仓列表合计占{_DENOMINATORS[denominator]}的{_pct(listed)}；"
                + ("这里只列前十大，不把剩余部分补成相同持仓。" if payload["holdings_scope"] == "top10"
                   else "列表之外的资产、负债或未披露风险不自动补齐。")
            ),
        }
        if "equity" in payload["assets"]:
            mapping("equity_price", "股票整体涨跌会影响多少", "portfolio",
                    f"{section['report_date']}披露股票占{_DENOMINATORS[denominator]}的"
                    f"{_pct(payload['assets']['equity'])}；价格冲击只能按这个历史比例计算。",
                    "未知今天的股票仓位；衍生品名义敞口不在现有组合权重合同中，不能当作零。")
        if payload["sectors"]:
            mapping("sector_price", "哪些行业的涨跌更牵动这只基金", "portfolio",
                    "；".join(f"{row['name']}占{_DENOMINATORS[denominator]}的{_pct(row['weight'])}"
                             for row in payload["sectors"]) + "。这些是报告时点的实际披露比例。",
                    "行业权重不是今天的权重，也不能证明未来行业表现。")
        if payload["countries"]:
            mapping("market_location", "披露的国家或证券市场分布", "portfolio",
                    "保留来源的国家分类；先核对它指交易市场、注册地还是收入来源，再选市场参照。",
                    "国家分类本身不能确认公司业务所在地、交易币种或未对冲汇率敞口。")
        mapping("fx", "外币折成人民币，可能放大或抵消股票涨跌", "portfolio",
                "只使用明确披露的各币种未对冲净权重。" if payload["currencies"] else
                "这份组合记录没有明确的各币种未对冲净权重，因此不能算基金汇率影响。",
                "不把海外股票、美国上市或人民币份额自动等同于100%美元敞口。")
        bond = payload["assets"].get("bond")
        if bond is not None:
            mapping("rates", "利率与这只基金的关系", "portfolio",
                    "报告时点债券仓位为零，不能套用债券久期公式解释其股票涨跌。"
                    if decimal(bond) == 0 else
                    "债券价格对利率的反应还取决于债券比例、币种、期限和久期，不能只看美国十年期利率。",
                    "利率仍可能间接影响股票估值；这里没有估计这种关系的方向或大小。")
    if section := result["management"]:
        section["tenures_at_report"] = [
            {"name": manager["name"], "days": (
                date.fromisoformat(section["report_date"]) - date.fromisoformat(manager["start_date"])
            ).days} for manager in section["payload"]["managers"]
        ]
        section["explanation"] = "任职日期说明管理经历，不证明投资能力；任职时长只算到资料报告日。"
        mapping("management", "谁在管理、从什么时候开始", "management",
                "经理名单与实际上任日可帮助检查策略变化和历史业绩是否属于同一团队。",
                "经理人数、任期长短不换算成收益分数；报告之后是否变更需新公告。")
    if section := result["valuation"]:
        payload = section["payload"]
        mandate = result["mandate"]
        benchmark = mandate["payload"]["benchmark"] if mandate else {}
        matched = bool(payload["compatible_scope"] and benchmark.get("id")
                       and payload["benchmark_id"] == benchmark["id"]
                       and payload["region"] == benchmark["region"])
        section["mandate_benchmark_match"] = matched
        section["explanation"] = (
            "这是可明确对应合同基准的估值资料，不是当前实际持仓的逐股估值。" if matched else
            "尚不能把这条估值明确对应到合同基准，不据此判断本基金贵或便宜。"
        )
        mapping("valuation", "盈利不变时，市场愿意给多少倍价格", "valuation",
                "市盈率是股价相对盈利的倍数；盈利和倍数可以同时变化，需要区分历史与预期口径。",
                "单个估值数值不证明高低，不生成估值分位、目标价或未来盈利。")
    return result


def _exposures(profile: Mapping, scope: str) -> tuple[Mapping, dict[str, Decimal]]:
    section = profile.get("portfolio")
    if not section:
        raise ValidationError("缺少可用的组合披露，不能计算权重情景")
    payload = section["payload"]
    if scope == "equity":
        rows = {"equity": decimal(payload["assets"]["equity"])} if "equity" in payload["assets"] else {}
    elif scope == "sectors":
        rows = {row["name"]: decimal(row["weight"]) for row in payload["sectors"]}
    elif scope == "holdings":
        rows = {}
        for row in payload["holdings"]:
            if row["kind"] == "corporate_equity":
                rows[row["issuer_id"]] = rows.get(row["issuer_id"], Decimal(0)) + decimal(row["weight"])
    elif scope == "currencies":
        rows = {row["currency"]: decimal(row["unhedged_net_weight"]) for row in payload["currencies"]}
    else:
        raise ValidationError("情景scope只支持equity、sectors、holdings或currencies，不能混合重叠切面")
    return section, rows


def _change(value: Any) -> Decimal:
    result = decimal(value)
    if result < -1:
        raise ValidationError("价格、盈利、估值倍数或汇率变化不能低于-100%")
    return result


def disclosure_shock_scenario(
    profile: Mapping, *, scope: str, shocks: Mapping[str, Any], label: str,
) -> dict:
    """Apply explicit proportional shocks to one non-overlapping disclosure view.

    scope='equity': {'equity': change}; 'sectors': {exact sector name: change};
    'holdings': {issuer_id: change}, corporate equities only; 'currencies':
    {ISO currency: change in base-currency price per foreign-currency unit}.
    Changes are fractions, e.g. a caller's -0.10 means a conditional -10%.
    No default shocks and no reweighting. Different calls must not be summed.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValidationError("必须明确填写情景假设说明，不自动生成涨跌情景")
    if not isinstance(shocks, Mapping) or not shocks:
        raise ValidationError("必须明确提供至少一个变化假设")
    section, exposures = _exposures(profile, scope)
    if shocks.keys() - exposures.keys():
        raise ValidationError("情景只能覆盖同一份披露中实际提供的权重，不能猜测缺失持仓或币种")
    contributions = []
    with localcontext() as context:
        context.prec = 128
        for key in sorted(shocks):
            change, weight = _change(shocks[key]), exposures[key]
            contributions.append({
                "exposure": key, "weight": _text(weight), "assumed_change": _text(change),
                "contribution": _text(weight * change),
            })
        total = _sum(row["contribution"] for row in contributions)
        covered = _sum(row["weight"] for row in contributions)
    denominator = section["payload"]["denominator"]
    return {
        "schema_version": SCHEMA_VERSION, "kind": "frozen_disclosure_conditional_shock",
        "label": label, "assumption_source": "caller_supplied_not_a_forecast",
        "as_of": profile["as_of"], "product_id": profile["product_id"],
        "record_id": section["record_id"], **{key: section.get(key) for key in _DATES},
        "scope": scope, "denominator": denominator,
        "denominator_label": _DENOMINATORS[denominator],
        "covered_weight": _text(covered), "contributions": contributions,
        "covered_contribution": _text(total),
        "nav_return_contribution": _text(total) if denominator == "nav" else None,
        "whole_fund_return": None, "additive_with_other_scenarios": False,
        "explanation": (
            f"若{section['report_date']}披露的权重保持不变，且所填变化发生，"
            f"这些已覆盖部分对{_DENOMINATORS[denominator]}的影响为{_pct(total)}。"
            "这是按原分母计算的涨跌贡献，不是全基金收益率，也不是发生概率。"
        ),
        "limitations": [
            "未覆盖部分没有计算，不代表价格不变；没有把已覆盖权重放大到100%。",
            "固定报告持仓，不模拟调仓、分红、费用、期货期权及其他未披露风险。",
            "除明确填写的冲击外不叠加其他变化；价格冲击不另加汇率，不推断货币敞口。",
            "不同切面的情景可能覆盖同一批股票，不能与行业、前十大或整体股票情景相加。",
        ],
    }


def earnings_valuation_fx_scenario(
    profile: Mapping, *, scope: str, members: Sequence[str], earnings_change: Any,
    valuation_change: Any, label: str, fx_change: Any = None,
    fx_covered_weight: Any = None, fx_currency: str | None = None,
) -> dict:
    """Conditional price = EPS * valuation multiple, optionally explicit FX.

    ``members`` selects one disclosure view, never a mixture of sector/top10.
    EPS and comparable valuation-multiple changes are caller assumptions, not
    measured earnings or predictions. A positive comparable EPS base, unchanged
    share count and the same multiple definition are conditions of the identity.

    Optional FX requires ALL of fx_change, fx_covered_weight, fx_currency.
    fx_covered_weight is a caller-specified subset of covered_weight ON THE SAME
    DENOMINATOR, not inferred from equity/country weights or normalized to 100%.
    Formula: w*((1+earnings)*(1+multiple)-1)
             + fx_weight*(1+earnings)*(1+multiple)*fx_change.
    The explicit FX subset is conditional, not a claimed disclosed joint mapping.
    """
    if scope == "currencies":
        raise ValidationError("盈利与估值情景只能选股票整体、行业或已披露股票，不适用于现金币种")
    if isinstance(members, (str, bytes)) or not members or len(set(members)) != len(members):
        raise ValidationError("必须明确选择不重复的已披露股票范围")
    earnings, multiple = _change(earnings_change), _change(valuation_change)
    with localcontext() as context:
        context.prec = 128
        factor = (1 + earnings) * (1 + multiple)
        result = disclosure_shock_scenario(
            profile, scope=scope, shocks={key: factor - 1 for key in members}, label=label,
        )
        result["kind"] = "frozen_disclosure_earnings_valuation_fx_condition"
        result["conditions"] = {
            "earnings_per_share_change": _text(earnings),
            "valuation_multiple_change": _text(multiple),
            "local_price_change": _text(factor - 1),
            "identity": "(1 + EPS change) * (1 + multiple change) - 1",
            "required_basis": "正的可比每股盈利起点、股数不变、前后估值倍数口径一致",
        }
        result["local_price_contribution"] = result["covered_contribution"]
        result["contribution_basis"] = "local_price_only; optional FX contribution is separate"
        result["limitations"][2] = (
            "只计算明确填写的盈利、估值倍数与汇率条件；不把它们解释成已经实现的事实。"
        )
        result["fx_condition"] = None
        supplied = (fx_change is not None, fx_covered_weight is not None, fx_currency is not None)
        if any(supplied) and not all(supplied):
            raise ValidationError("汇率情景必须同时明确变化、同分母覆盖权重和外币币种；不默认100%美元")
        if all(supplied):
            if (not isinstance(fx_currency, str) or len(fx_currency) != 3
                    or not fx_currency.isascii() or not fx_currency.isalpha()
                    or not fx_currency.isupper()):
                raise ValidationError("汇率情景币种须为三位大写代码")
            mandate = profile.get("mandate")
            base_currency = mandate["payload"]["base_currency"] if mandate else None
            if base_currency is None or fx_currency == base_currency:
                raise ValidationError("汇率情景须有明确基金记账币种，并选择不同的外币")
            fx, weight = _change(fx_change), decimal(fx_covered_weight)
            if weight < 0 or weight > decimal(result["covered_weight"]):
                raise ValidationError("汇率覆盖权重必须是所选股票范围内、同分母的明确子集")
            fx_contribution = weight * factor * fx
            total = decimal(result["local_price_contribution"]) + fx_contribution
            result["fx_condition"] = {
                "currency": fx_currency, "base_currency": base_currency,
                "quote": f"{base_currency}_per_{fx_currency}",
                "assumed_change": _text(fx), "covered_weight": _text(weight),
                "weight_basis": "caller_supplied_conditional_subset_not_disclosed_joint_exposure",
                "contribution_including_price_fx_cross_term": _text(fx_contribution),
                "covered_asset_return": _text(factor * (1 + fx) - 1),
            }
            result["covered_contribution"] = _text(total)
            result["nav_return_contribution"] = _text(total) if result["denominator"] == "nav" else None
            result["explanation"] = (
                f"在所填盈利、估值倍数和汇率条件都成立时，按{result['report_date']}"
                f"冻结权重计算的已覆盖贡献为{_pct(total)}（分母：{result['denominator_label']}）。"
                "汇率只作用于你明确指定的同分母子集，含股价与汇率同时变化的乘积项。"
            )
        result["limitations"].append(
            "盈利与估值变化均是外部填写的条件；未核实持仓盈利、估值起点或逐股币种对应关系。"
            "没有填写的汇率部分未计算，不等于已对冲或没有汇率风险。"
        )
    return result

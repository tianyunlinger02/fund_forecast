"""Pure saved-choice simulation replay; neither research reconstruction nor scoring.

Only the original assessment, an optional private annotation and already stored
NAVs enter here. No current view selects an old action, and no annotation is
converted into a claim about an actual trade or personal profit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from .domain import ValidationError, decimal, timestamp
from .forecast_outcomes import RAW_NAV_TARGET, _series_dates, _target_key
from .timing_scenario import (
    _ACTIONS,
    _base,
    _calendar_conditions,
    _calendar_fees,
    _calendar_value,
    _day,
    _number,
    _realized_rows,
    _series,
)

_BRANCH_ACTIONS = {
    "flat": {"buy", "wait", "no_action"},
    "held": {"hold", "add", "reduce50", "exit", "reduce", "redeem", "no_action"},
}
_RAW_BASIS = "raw unit NAV; no dividend reinvestment or split adjustment"
_NAV_TYPES = {"unit_nav", "final_unit_nav", "final_unit_nav_candidate"}


class _ReplayError(Exception):
    def __init__(self, code: str, message: str, status: str = "invalid") -> None:
        self.code, self.message, self.status = code, message, status


def _fail(code: str, message: str, status: str = "invalid") -> None:
    raise _ReplayError(code, message, status)


def freeze_simulation_reference(
    trade_scenario: Mapping, *, branches: Mapping | None = None,
) -> dict:
    """Freeze chosen/simple-reference pairs while the assessment is being built.

    Callers with an economic brief supply its actual preferred actions. ``None``
    and an unspecified-ratio ``reduce`` are deliberately retained, not replaced
    with a model action or a guessed half-sale. With no override, only the
    scenario's already chosen actions are copied, against cash/holding references.
    This helper does not inspect a brief, choose by outcome, or fit anything.
    """
    if not isinstance(trade_scenario, Mapping) or trade_scenario.get("kind") != "calendar_cash":
        raise ValidationError("原模拟参照需要已展示的自然日净额情景")
    if branches is None:
        original = (trade_scenario.get("decision") or {}).get("branches") or {}
        branches = {
            branch: {"action": (original.get(branch) or {}).get("action"),
                     "reference_action": "wait" if branch == "flat" else "hold"}
            for branch in _BRANCH_ACTIONS
        }
        if branches["held"]["action"] == "sell":
            branches["held"]["action"] = "exit"
        for row in branches.values():
            if row["action"] == "unavailable":
                row["action"] = None
    if not isinstance(branches, Mapping) or branches.keys() - _BRANCH_ACTIONS.keys():
        raise ValidationError("原模拟参照只接受新买与已有份额两个分支")
    frozen = {}
    for branch, allowed in _BRANCH_ACTIONS.items():
        pair = branches.get(branch)
        if not isinstance(pair, Mapping):
            raise ValidationError("每个原模拟分支都需要保留动作和简单参照；无首选时动作留空")
        action, reference = pair.get("action"), pair.get("reference_action")
        if action is not None and (not isinstance(action, str) or action not in allowed):
            raise ValidationError("原模拟动作与所属分支不匹配")
        references = {"wait"} if branch == "flat" else {"hold", "exit"}
        if not isinstance(reference, str) or reference not in references:
            raise ValidationError("原模拟简单参照须与该分支同资本：现金、持有或退出")
        frozen[branch] = {"action": action, "reference_action": reference}
    return {"branches": frozen}


def _action(action: str | None, branch: str) -> str | None:
    if action == "no_action":
        return "wait" if branch == "flat" else "hold"
    return "exit" if action == "redeem" else action


def _note_selection(note: Mapping, saved: Mapping, product_id: str) -> dict:
    if not saved.get("id") or note.get("assessment_id") != saved["id"]:
        _fail("note_assessment_mismatch", "笔记不属于这份原判断，未跨判断复用动作。")
    if note.get("product_id") != product_id:
        _fail("note_product_mismatch", "笔记所属基金与原判断不一致。")
    kind, branch, action = note.get("kind"), note.get("branch"), note.get("action")
    if not isinstance(kind, str) or kind not in {"plan", "simulated_choice", "user_report", "observation"}:
        _fail("invalid_note", "笔记类别不是计划、模拟采用、用户自述或观察。")
    if not isinstance(branch, str) or branch not in _BRANCH_ACTIONS:
        _fail("invalid_note", "笔记未指明原判断的新买或已有份额分支。")
    if not isinstance(action, str) or action not in _BRANCH_ACTIONS[branch]:
        _fail("invalid_note", "笔记动作与原分支不匹配。")
    occurred, recorded = _day(note.get("occurred_on")), timestamp(note.get("recorded_at"))
    if saved.get("saved_at") is not None and recorded < timestamp(saved["saved_at"]):
        _fail("invalid_note_clock", "笔记记录时钟早于原判断保存时钟；没有用发生日期回填记录时间。")
    context = ("prospective" if occurred > recorded.date().isoformat() else
               "historical" if occurred < recorded.date().isoformat() else "same_day")
    return {"source": kind, "recorded_at": recorded.isoformat(), "occurred_on": occurred,
            "choice_context": context, "branch": branch, "action": action}


def _selection_timing(selection: dict, entry: str | None, target: str | None) -> dict:
    result = {**selection, "selection_timing": "unknown"}
    if selection.get("recorded_at") is None:
        return result
    recorded = timestamp(selection["recorded_at"]).date().isoformat()
    if target is not None and recorded >= target:
        result["selection_timing"] = "post_hoc"
    elif entry is not None and recorded < entry:
        result["selection_timing"] = "prospective"
    elif target is not None:
        result["selection_timing"] = "pre_target"
    return result


def _units(row: Mapping, currency: str) -> None:
    provenance = row.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        _fail("invalid_basis", "净值来源依据不是已存的对象。")
    currencies = [item["currency"] for item in (row, provenance) if item.get("currency") is not None]
    if not currencies:
        _fail("missing_currency", "实际净值缺少明确币种，未假定人民币或自动换汇。", "unavailable")
    if any(value != currency for value in currencies):
        _fail("currency_mismatch", "原净值锚点与实际窗口币种不一致，未混算或换汇。")
    types = [item["nav_type"] for item in (row, provenance) if item.get("nav_type") is not None]
    if not types:
        _fail("missing_nav_basis", "实际净值没有保存单位净值口径，未冒用累计或复权净值。", "unavailable")
    if any(not isinstance(value, str) or value not in _NAV_TYPES for value in types):
        _fail("nav_basis_mismatch", "实际窗口不是原模拟的未复权单位净值口径。")
    for item in (row, provenance):
        if item.get("nav_basis") not in (None, _RAW_BASIS):
            _fail("nav_basis_mismatch", "实际窗口的净值依据与原未复权单位净值目标不一致。")


def _validate_scope(assessment: Mapping, scenario: Mapping, forecast: Mapping) -> str:
    subject = assessment.get("subject") or {}
    product_id = subject.get("id")
    if not isinstance(product_id, str) or not product_id:
        _fail("missing_product", "原判断没有明确基金身份，未跨产品拼接结果。", "unavailable")
    if forecast.get("product_id") not in (None, product_id):
        _fail("product_mismatch", "原预测与原判断基金身份不一致。")
    scenario_subject = scenario.get("subject") or {}
    if scenario_subject.get("id") not in (None, product_id):
        _fail("product_mismatch", "原金额情景与原判断基金身份不一致。")
    for key in ("kind", "symbol", "share_class", "exchange", "is_demo"):
        if key in scenario_subject and key in subject and scenario_subject[key] != subject[key]:
            _fail("product_mismatch", "原金额情景与原判断的代码、份额或数据模式不一致。")
    return product_id


def _comparison(scenario: Mapping, branch: str, pair: Mapping, actual: float,
                base: float, fees: Mapping, conditions: Mapping, selection: dict) -> dict:
    action, reference = _action(pair["action"], branch), pair["reference_action"]
    result = {"branch": branch, "action": pair["action"], "simulated_action": action,
              "reference_action": reference, "status": "unavailable", "reasons": [],
              "chosen": None, "reference": None, "realized_difference": None,
              "selection": deepcopy(selection), **selection}
    if action is None or action == "reduce":
        result["reasons"] = ["原判断没有唯一首选。" if action is None else
                             "原判断只有减仓方向，未保存减仓比例；没有擅自改为减半。"]
        return result
    rows = {row["action"]: row for row in scenario.get("rows", ())}
    labels = {row[0]: row[1] for row in _ACTIONS}
    initial = base if branch == "flat" else 2 * base
    seeds = []
    for current in dict.fromkeys((action, reference)):
        original = rows.get(current) or {}
        if original.get("initial_value") is not None and _number(original["initial_value"], "原本金") != initial:
            _fail("capital_mismatch", "原动作行与原标准本金B的同资本口径不一致。")
        seeds.append({"action": current, "label": original.get("label") or labels[current],
                      "initial_value": initial, "comparison_action": reference,
                      "comparison_label": labels[reference]})

    def value_for(row):
        return (_calendar_value(row["action"], actual, base, fees)
                if not conditions[row["action"]] else None)

    reference_value = _calendar_value(reference, actual, base, fees) if not conditions[reference] else None

    def advantage_for(row):
        value = value_for(row)
        return value - reference_value if value is not None and reference_value is not None else None

    realized = {row["action"]: row for row in _realized_rows(seeds, actual, value_for, advantage_for)}
    result.update(chosen=realized[action], reference=realized[reference])
    chosen_value = result["chosen"]["actual_value"]
    if chosen_value is None or reference_value is None:
        result["reasons"] = list(dict.fromkeys([
            *conditions[action], *conditions[reference],
            "原动作或参照所需的原批次费用、确认或申请条件不足，未按零费用或今天条件补算。",
        ]))
    else:
        result.update(status="mature", realized_difference=chosen_value - reference_value)
    return result


def replay_saved_assessment(
    saved: Mapping, *, series: Sequence[Mapping], note: Mapping | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Realize original calendar actions with exact original H/L and frozen fees.

    ``saved`` is an assessment_detail record (preferred, with actual saved_at) or
    its unchanged assessment payload. ``series`` is the already stored NAV
    sequence, retaining units, availability and event flags. ``as_of`` optionally
    narrows visibility, without deleting unavailable rows and shifting indices.
    No clock, store, network, fit, issuance or official outcome scorer is used.
    """
    result = {"kind": "saved_calendar_simulation", "status": "unavailable", "reasons": [],
              "reason_codes": [], "comparisons": [], "selection_timing": "unknown",
              "choice_context": None, "actual_nav_change": None,
              "original_basis_hash": None, "basis_relationship": "unverified_anchor",
              "provenance_limitations": [],
              "end_value_basis": "cash_and_redemption_receivable", "personal_actual_pnl": None,
              "assumptions": [
                  "这是原保存判断或明确模拟选择的复盘，不是历史重建、真实发行成绩、实际成交或个人盈亏。",
                  "沿用原B、各批次费用和精确L/H；按真实净值笔数定位，不用今天的研究、费表或日历重选。",
                  "核对原锚点并计算其后的净值窗口；不重建原模型的全部历史输入，也不要求前史来源元数据完全相同。",
                  "期末金额含原情景的赎回应收款，不承诺到账；未重复扣除净值内费用或假定现金收益。",
                  "是否事前记录按真实保存/记录时钟判断，发生日期不能回填事前选择；目标日当日不宣称事前。",
                  "笔记发生日期只描述计划或自述，不会把原模拟进出日期改成笔记日期。",
              ]}
    try:
        if not isinstance(saved, Mapping):
            _fail("invalid_snapshot", "需要原保存判断对象。")
        assessment = saved.get("assessment", saved)
        if not isinstance(assessment, Mapping):
            _fail("invalid_snapshot", "原保存判断内容不是对象。")
        if type(assessment.get("schema_version")) is not int or assessment["schema_version"] not in (6, 7):
            _fail("unsupported_schema", "这份旧判断没有可复盘的自然日净额合同；未补今天的日期或首选。", "unavailable")
        scenario, forecast = assessment.get("trade_scenario"), assessment.get("forecast")
        if not isinstance(scenario, Mapping) or scenario.get("kind") != "calendar_cash" or not isinstance(forecast, Mapping):
            _fail("missing_calendar_scenario", "原判断缺少完整自然日净额情景或原预测依据。", "unavailable")
        product_id = _validate_scope(assessment, scenario, forecast)
        result.update(assessment_id=saved.get("id"), product_id=product_id,
                      original_basis_hash=deepcopy(forecast.get("basis_hash")))
        if not forecast.get("basis_hash"):
            result["provenance_limitations"].append("原判断未保留历史输入依据标识；仅核对原锚点及其后实际窗口。")
        if note is not None:
            if not isinstance(note, Mapping):
                _fail("invalid_note", "需要原判断关联的已记录笔记对象。")
            selection = _note_selection(note, saved, product_id)
            result.update(note=deepcopy(dict(note)), choice_context=selection["choice_context"])
            if note["kind"] in {"user_report", "observation"}:
                result.update(status="note_only", reasons=[
                    "用户自述或观察只保留为笔记；没有独立成交依据，未套原计划冒充个人实际盈亏。"],
                    reason_codes=["annotation_not_trade_basis"], selection=selection)
                return result
        else:
            captured = saved.get("saved_at")
            selection = {"source": "saved_preference", "recorded_at": (
                timestamp(captured).isoformat() if captured is not None else None), "choice_context": None}
        reference = assessment.get("simulation_reference")
        if reference is None and isinstance(assessment.get("decision_brief"), Mapping):
            reference = assessment["decision_brief"].get("simulation_reference")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("branches"), Mapping):
            _fail("missing_reference", "原判断没有保存所选动作和同资本简单参照，未从今天观点补选。", "unavailable")
        pairs = freeze_simulation_reference(scenario, branches=reference["branches"])["branches"]
        if note is not None:
            branch = note["branch"]
            pairs = {branch: {**pairs[branch], "action": note["action"]}}
        plan = scenario.get("plan")
        if not isinstance(plan, Mapping):
            _fail("missing_plan", "原自然日计划没有保存，未使用今天日程。", "unavailable")
        horizon, lag = scenario.get("horizon"), scenario.get("entry_lag")
        if type(horizon) is not int or horizon <= 0 or type(lag) is not int or lag < 0:
            _fail("invalid_window", "原窗口没有正数H或非负L，未借用其他期限。", "unavailable")
        decision = scenario.get("decision") or {}
        head = decision.get("primary_head") or {}
        for source in (plan, assessment, head):
            for key, expected in (("horizon", horizon), ("entry_lag", lag)):
                if key in source and (type(source[key]) is not int or source[key] != expected):
                    _fail("window_mismatch", "原判断、金额情景、原预测头和日程的精确H/L不一致。")
        for key, expected in (("primary_horizon", horizon), ("primary_entry_lag", lag)):
            if key in decision and (type(decision[key]) is not int or decision[key] != expected):
                _fail("window_mismatch", "原金额情景的主预测坐标与保存窗口不一致。")
        for source in (plan, scenario, forecast):
            if source.get("target_unit") not in (None, "nav_observations"):
                _fail("target_unit_mismatch", "原H/L不是净值观察笔数，不能改按今天自然日解释。")
        if scenario.get("end_value_basis") != "cash_and_redemption_receivable":
            _fail("end_basis_mismatch", "原情景不是计划赎回净额及应收款口径。")
        definition = forecast.get("target_definition")
        if definition is None:
            _fail("missing_target", "原判断未保存收益目标定义，未假定单位净值或总回报。", "unavailable")
        if _target_key(definition) != RAW_NAV_TARGET:
            _fail("target_mismatch", "原收益目标不是未复权单位净值的精确H/L简单收益。")
        head_definition = head.get("target_definition_id", head.get("target_definition"))
        if head_definition is not None and _target_key(head_definition) != RAW_NAV_TARGET:
            _fail("target_mismatch", "原执行头与原预测保存的收益目标不一致。")
        if isinstance(definition, Mapping) and (
                definition.get("nav_basis", _RAW_BASIS) != _RAW_BASIS
                or definition.get("target_kind", "raw_nav_simple_return") != "raw_nav_simple_return"
                or definition.get("not_total_return", True) is not True):
            _fail("target_basis_mismatch", "原目标的单位净值、复权或总回报依据存在矛盾。")
        anchor = forecast.get("anchor")
        if not isinstance(anchor, Mapping):
            _fail("missing_anchor", "原净值锚点缺失，未取今天最新净值替代。", "unavailable")
        anchor_day, anchor_nav = _day(anchor.get("date")), decimal(anchor.get("nav"))
        currency = anchor.get("currency")
        if (not isinstance(currency, str) or len(currency) != 3 or not currency.isascii()
                or not currency.isupper() or not currency.isalpha()):
            _fail("missing_anchor_currency", "原锚点没有明确币种，未假定人民币。", "unavailable")
        own_anchor = head.get("anchor")
        if own_anchor is not None:
            if (not isinstance(own_anchor, Mapping) or own_anchor.get("date") != anchor_day
                    or decimal(own_anchor.get("nav")) != anchor_nav):
                _fail("anchor_mismatch", "原执行头与原预测锚点不一致，未选取其中一个替代。")
            if own_anchor.get("currency") not in (None, currency):
                _fail("currency_mismatch", "原执行头与原预测锚点币种不一致。")
        if head.get("currency") not in (None, currency):
            _fail("currency_mismatch", "原执行头币种与原预测锚点不一致。")
        if plan.get("anchor_date") not in (None, anchor_day):
            _fail("anchor_mismatch", "原计划与原预测锚点日期不一致。")
        if (scenario.get("currency") not in (None, currency)
                or (assessment.get("inputs") or {}).get("anchor_currency") not in (None, currency)):
            _fail("currency_mismatch", "原模拟本金或输入币种与原预测锚点不一致。")
        if not isinstance(series, Sequence) or isinstance(series, (str, bytes)) or any(not isinstance(row, Mapping) for row in series):
            _fail("invalid_series", "复盘需要已有、按日期排序且保留依据的完整净值序列。")
        navs = _series(series)
        dates, positions = _series_dates(series)
        origin = positions.get(anchor_day)
        if origin is None:
            _fail("missing_anchor", "已有序列找不到原净值锚点；未平移原点。")
        if anchor_nav <= 0 or decimal(series[origin]["nav"]) != anchor_nav:
            _fail("anchor_mismatch", "已有序列的原锚点净值已不同，未覆盖原值。")
        _units(series[origin], currency)
        result["basis_relationship"] = "original_anchor_verified; historical_forecast_inputs_not_reconstructed"
        start, end = origin + lag, origin + lag + horizon
        entry_day = navs[start]["date"] if start < len(navs) else None
        target_day = navs[end]["date"] if end < len(navs) else None
        selection = _selection_timing(selection, entry_day, target_day)
        base, fees = _base(scenario.get("base_amount")), _calendar_fees(scenario.get("fees"))
        result.update(horizon=horizon, entry_lag=lag, base_amount=base, currency=currency,
                      fees=deepcopy(scenario.get("fees")), selection=selection,
                      selection_timing=selection["selection_timing"],
                      window={"anchor": deepcopy(anchor), "entry_index": start, "target_index": end,
                              "entry_date": entry_day, "target_date": target_day,
                              "planned_application_date": plan.get("application_date"),
                              "planned_sale_date": plan.get("sale_date"),
                              "target_definition": deepcopy(definition)})
        for row in series[origin:min(end + 1, len(series))]:
            if row.get("product_id", product_id) != product_id:
                _fail("product_mismatch", "已有净值窗口混入其他基金，未跨产品计算。")
            _units(row, currency)
        cutoff = timestamp(as_of) if as_of is not None else None
        if end >= len(navs) or any(available is None or (cutoff is not None and (
                available > cutoff or day > cutoff.date().isoformat()))
                for day, available in dates[origin:min(end + 1, len(dates))]):
            _fail("immature_window", "原精确L/H窗口尚未有足够实际可用净值；未填补、跳过日期或伪造未来结果。", "immature")
        if any(row.get("invalid_return", False) or row.get("label_valid") is False
               for row in navs[start + 1:end + 1]):
            _fail("invalid_event", "原实际收益窗口跨已标记的分红、拆分或无效收益事件；未擅自复权。")
        actual = navs[end]["nav"] / navs[start]["nav"] - 1
        if not isfinite(actual):
            _fail("invalid_return", "原实际窗口收益超出可表示范围。")
        conditions, _ = _calendar_conditions(plan, scenario.get("fees"))
        comparisons = [_comparison(scenario, branch, pair, actual, base, fees, conditions, selection)
                       for branch, pair in pairs.items()]
        result.update(status="mature" if any(row["status"] == "mature" for row in comparisons) else "unavailable",
                      comparisons=comparisons, actual_nav_change=actual,
                      has_unavailable_comparisons=any(row["status"] != "mature" for row in comparisons))
        result["window"].update(entry=deepcopy(series[start]), target=deepcopy(series[end]))
        if result["status"] == "unavailable":
            result["reasons"] = list(dict.fromkeys(reason for row in comparisons for reason in row["reasons"]))
            result["reason_codes"] = ["unavailable_action_or_fees"]
    except _ReplayError as exc:
        result.update(status=exc.status, reasons=[exc.message], reason_codes=[exc.code], comparisons=[])
    except (ValidationError, TypeError, KeyError, ValueError, OverflowError, AttributeError) as exc:
        result.update(status="invalid", reasons=["原判断或已有净值依据无效：" + str(exc)],
                      reason_codes=["invalid_input"], comparisons=[])
    return result

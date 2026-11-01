"""Resolve one authored operation recommendation for the displayed cash scenario.

The saved research remains available as evidence, not as a competing headline.
Amounts are compared within each capital group and never rank all actions.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from importlib.resources import files
import json

from .domain import ValidationError, timestamp
from .forecast_models import model_family

POLICY_VERSION = "fund_buyer_economic_advice_v3"
LEGACY_POLICY_VERSION = "fund_buyer_advice_v2"
_ACTION_LABELS = {
    "buy": "新买一笔", "wait": "先留现金", "hold": "继续持有，不追加",
    "add": "保留已有基金，再追加一笔", "reduce": "减少这只基金，比例另定",
    "redeem": "退出这只基金", "reduce50": "卖出一半（金额示例）", "exit": "现在退出",
}
_CASE_BRANCHES = {"new_buy": "flat", "held": "held", "addition": "addition"}
_ACTION_ROWS = {"redeem": "exit", "reduce": "reduce50"}


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def load_decision_rationale(product_id: str | None, research_record_id: str | None) -> dict | None:
    """Read packaged public author content; never update the original research."""
    if not product_id or not research_record_id:
        return None
    resources = json.loads(files("rmbfund").joinpath("resources/decision_rationales.json").read_text(encoding="utf-8"))
    matches = [row for row in resources if row.get("product_id") == product_id
               and row.get("research_record_id") == research_record_id]
    if len(matches) > 1:
        raise ValidationError("同一研究有多份未区分的动作补充说明")
    return deepcopy(matches[0]) if matches else None


def normalize_decision_rationale(value: Mapping, *, research_thesis: Mapping,
                                 product_id: str | None) -> dict:
    """Validate the small author contract, not the truth of an investment view."""
    thesis, payload = research_thesis, _mapping(research_thesis.get("payload"))
    if (value.get("schema_version") != 1 or value.get("product_id") != product_id
            or not thesis.get("record_id") or value.get("research_record_id") != thesis["record_id"]
            or _mapping(payload.get("product")).get("id") != product_id):
        raise ValidationError("动作补充说明与当前基金或研究版本不匹配")
    if not value.get("id") or not value.get("origin"):
        raise ValidationError("动作补充说明缺少作者来源")
    generated = timestamp(value.get("generated_at"))
    original = _mapping(payload.get("origin")).get("generated_at")
    if original and generated < timestamp(original):
        raise ValidationError("动作补充说明不能早于它引用的研究")
    facts = {row["id"] for row in payload.get("facts", [])}
    references = value.get("fact_refs") or []
    if not references or not set(references) <= facts:
        raise ValidationError("动作补充说明引用了原研究不存在的事实")
    if _mapping(value.get("scope")).get("window_semantics") != "review_not_forced_exit":
        raise ValidationError("操作窗口不能被补充说明改成强制持有或清仓日期")

    def reason(row):
        if not isinstance(row, Mapping) or not row.get("reason") or not row.get("fact_ids"):
            raise ValidationError("作者取舍须有理由与事实引用")
        if not set(row["fact_ids"]) <= set(references):
            raise ValidationError("作者取舍使用了未声明的事实")

    cases = _mapping(value.get("cases"))
    if set(cases) != set(_CASE_BRANCHES):
        raise ValidationError("动作补充说明须分别回答新买、持有与追加")
    for key, case in cases.items():
        if case.get("action") != _mapping(_mapping(payload.get("action_cases")).get(key)).get("action"):
            raise ValidationError("补充说明不能冒用旧研究身份改变原动作")
        reason(case)
        for row in case.get("why_not", []):
            reason(row)
            if row.get("action") not in _ACTION_LABELS:
                raise ValidationError("未知的备选动作")
        resolutions = _mapping(case.get("short_window_resolution"))
        if set(resolutions) != {"aligned", "opposed", "unavailable"}:
            raise ValidationError("须说明短窗口同向、反向及缺预测时的取舍")
        for row in resolutions.values():
            reason(row)
            advice = row.get("operation_advice")
            if advice is not None:
                allowed = {"buy", "wait"} if key == "new_buy" else {"hold", "add", "reduce", "redeem"}
                if (not isinstance(advice, Mapping) or advice.get("action") not in allowed
                        or not advice.get("reason") or not advice.get("next_step")):
                    raise ValidationError("操作建议须给出适用动作、主要原因和下一步")
                if case.get("action") in {"reduce", "redeem"} and advice["action"] != case["action"]:
                    raise ValidationError("短期操作说明不能覆盖已成立的减仓或退出理由")
        for row in case.get("checks", []):
            if not row.get("condition") or not row.get("effect") or not row.get("fact_ids"):
                raise ValidationError("改判条件须包含观察内容、影响与依据")
            if not set(row["fact_ids"]) <= set(references):
                raise ValidationError("改判条件引用不匹配")
    return deepcopy(dict(value))


def _number(row: Mapping, key="expected_value") -> Decimal | None:
    value = row.get(key)
    return Decimal(str(value)) if value is not None else None


def _conflict(action, rows, group) -> str:
    """Use each decision's actual alternative, not another row's advantage."""
    if action == "reduce":
        return "unavailable"  # A direction without an amount is not a 50% trade.
    selected = rows.get(_ACTION_ROWS.get(action, action), {})
    reference = rows.get("wait" if group == "flat" else "hold" if action == "add" else "exit", {})
    if action in {"wait", "redeem"}:
        reference = rows.get("buy" if group == "flat" else "hold", {})
    left, right = _number(selected), _number(reference)
    if left is None or right is None:
        return "unavailable"
    opposed = left <= right if action in {"buy", "add"} else left < right
    return "opposed" if opposed else "aligned"


def _execution(action, conditions) -> dict:
    operation = "subscription" if action in {"buy", "add"} else "redemption" if action in {"reduce", "redeem"} else None
    state = conditions.get(operation, "unknown") if operation else "not_required"
    return {"status": "blocked" if state == "closed" else state, "operation": operation,
            "reasons": list(conditions.get("reasons") or [])}


def _branch_action(branch) -> str | None:
    if branch.get("needed_review"):
        return None
    action = branch.get("action")
    return branch.get("economic_action", branch.get("research_action")) if action == "blocked" else action


def _held_choice(held, addition):
    if held == "hold":
        return "add" if addition == "add" else "hold" if addition == "wait" else None
    return None if held == "review" else held


def build_legacy_decision_brief(*, product_id: str | None, research_thesis: Mapping | None,
                                investor_decision: Mapping, trade_scenario: Mapping,
                                model_id: str | None = None, rationale: Mapping | None = None) -> dict:
    """Build the retired v2 policy only for explicit compatibility fixtures.

    Current assessments use build_decision_brief; saved v2 objects replay as-is.
    """
    thesis, investor = _mapping(research_thesis), investor_decision
    payload = _mapping(thesis.get("payload"))
    rows = {row["action"]: row for row in trade_scenario.get("rows", [])}
    branches = _mapping(investor.get("branches"))
    supplement, supplement_status = None, "missing"
    if rationale is not None:
        try:
            supplement = normalize_decision_rationale(rationale, research_thesis=thesis, product_id=product_id)
            scope = supplement["scope"]
            if model_id and model_family(model_id) not in scope.get("model_families", []):
                supplement, supplement_status = None, "model_not_covered"
            else:
                decision_at = _mapping(trade_scenario.get("plan")).get("decision_at")
                if decision_at and timestamp(supplement["generated_at"]) > timestamp(decision_at):
                    supplement, supplement_status = None, "not_yet_available"
                else:
                    supplement_status = "matched"
        except (ValidationError, ValueError, TypeError, KeyError):
            supplement, supplement_status = None, "not_matched"
    independent = investor.get("action_basis") == "independent_cases"
    flat = _branch_action(_mapping(branches.get("flat"))) if independent else None
    held = _branch_action(_mapping(branches.get("held"))) if independent else None
    addition = _branch_action(_mapping(branches.get("addition"))) if independent else None
    actions = {"flat": flat if flat in {"buy", "wait"} else None, "held": _held_choice(held, addition)}
    research_actions = {key: _mapping(branches.get(key)).get("research_action") if independent else None
                        for key in ("flat", "held", "addition")}
    research_actions["held"] = _held_choice(research_actions["held"], research_actions["addition"])
    conditions = _mapping(investor.get("execution_conditions"))
    # build_investor_decision exposes the exact current execution conditions below.
    conditions = conditions or _mapping(trade_scenario.get("execution_conditions"))
    result_cases = {}
    for group, action in actions.items():
        source = "new_buy" if group == "flat" else "addition" if action == "add" else "held"
        branch_keys = ["flat"] if group == "flat" else ["held", "addition"] if held == "hold" else ["held"]
        relevant = [_mapping(branches.get(key)) for key in branch_keys]
        reasons = list(dict.fromkeys(reason for branch in relevant for reason in branch.get("reasons", [])))
        fact_ids = sorted({identifier for branch in relevant for identifier in branch.get("fact_ids", [])})
        case = _mapping(_mapping((supplement or {}).get("cases")).get(source))
        if case.get("action") != action:
            case = {}
        fact_ids = sorted(set(fact_ids) | set(case.get("fact_ids") or []))
        constrained = any(branch.get("constraint_reasons") and _branch_action(branch) not in
                          {None, branch.get("research_action")} for branch in relevant)
        if constrained:
            case = {}
            reasons = list(dict.fromkeys([*(reason for branch in relevant for reason in branch.get("constraint_reasons", [])), *reasons]))
        conflict_status = _conflict(action, rows, group) if action else "unavailable"
        resolution = _mapping(_mapping(case.get("short_window_resolution")).get(conflict_status))
        if group == "held" and held == "hold" and not constrained:
            holding_case = _mapping(_mapping((supplement or {}).get("cases")).get("held"))
            holding_status = _conflict("hold", rows, "held")
            holding_resolution = _mapping(_mapping(holding_case.get("short_window_resolution")).get(holding_status))
            if _mapping(holding_resolution.get("operation_advice")).get("action") in {"reduce", "redeem"}:
                source, case, conflict_status, resolution = "held", holding_case, holding_status, holding_resolution
                fact_ids = sorted(set(fact_ids) | set(case.get("fact_ids") or []))
        review = list(dict.fromkeys([*investor.get("needed_review", []),
                                    *(item for branch in relevant for item in branch.get("needed_review", []))]))
        decision_status = "constrained" if constrained and action else "authored" if action else "unresolved"
        preference = action
        if action and conflict_status == "opposed" and not constrained:
            if resolution:
                decision_status = "authored_conflict_resolved"
            else:
                preference, decision_status = None, "unresolved_conflict"
        elif action and conflict_status == "unavailable" and not constrained:
            decision_status = "research_only"
        advice = _mapping(resolution.get("operation_advice")) if not constrained else {}
        advice_reason = advice.get("reason") or case.get("reason") or next(iter(reasons), None)
        next_step = advice.get("next_step")
        if advice:
            preference, decision_status = advice["action"], "authored_operation"
        if "research_horizon_elapsed" in review:
            preference, decision_status = None, "needs_review"
            advice_reason = "这份研究已过适用期限，先不据此调整；需要更新经营与价格判断。"
            next_step = "更新这只基金的研究后，再决定买入或卖出。"
        elif review and preference:
            decision_status = "provisional"
        economic_action = preference
        execution = _execution(preference or action, conditions)
        if execution["status"] == "blocked":
            preference = "wait" if group == "flat" else "hold"
            decision_status = "execution_blocked"
            advice_reason = ("当前暂停申购，这次先不新买。" if group == "flat" else
                             "当前暂停申购，保留已有份额，这次不追加。" if execution["operation"] == "subscription" else
                             "当前暂停赎回，暂时保留已有份额，不把退出建议当成已能成交。")
            next_step = "公开申赎状态恢复后，再按届时净值与费用检查这次操作。"
        if preference is None:
            if decision_status != "needs_review":
                advice_reason = "当前依据还不足以支持一次新的买卖，先不改变资金安排。"
            next_step = next_step or "先补齐这次费用与经营判断，再决定是否交易。"
        if not next_step:
            checks = case.get("checks") or []
            next_step = checks[0]["condition"] if checks else "下一次净值或经营资料更新后，再检查这次建议。"
        comparison_action = ("wait" if preference == "buy" else "buy" if group == "flat" else
                             "hold" if preference in {"add", "redeem"} else
                             "add" if preference == "hold" and research_actions[group] == "add" else "exit")
        chosen = rows.get(_ACTION_ROWS.get(preference, preference), {}) if preference != "reduce" else {}
        alternatives = []
        authored = list(case.get("why_not") or [])
        if group == "held" and source == "addition":
            authored += list(_mapping(_mapping((supplement or {}).get("cases")).get("held")).get("why_not") or [])
        candidates = ("buy", "wait") if group == "flat" else ("hold", "add", "reduce50", "exit")
        for candidate in candidates:
            if candidate == _ACTION_ROWS.get(preference, preference):
                continue
            explanation = next((row for row in authored if _ACTION_ROWS.get(row.get("action"), row.get("action")) == candidate), {})
            if preference != action and candidate == _ACTION_ROWS.get(action, action):
                explanation = {"reason": advice_reason, "fact_ids": fact_ids}
            row = rows.get(candidate, {})
            delta = (_number(row) - _number(chosen)) if _number(row) is not None and _number(chosen) is not None else None
            alternatives.append({
                "action": candidate, "label": _ACTION_LABELS[candidate],
                "reason": explanation.get("reason"), "fact_ids": list(explanation.get("fact_ids") or []),
                "expected_difference_vs_preferred": float(delta) if delta is not None else None,
                "expected_value": row.get("expected_value"),
            })
        comparisons = [rows.get(key, {}) for key in candidates]
        comparison_status = "ready" if all(row.get("status") == "ready" for row in comparisons) else (
            "partial" if any(row.get("status") == "ready" for row in comparisons) else "unavailable")
        checks = deepcopy(case.get("checks") or [])
        if not checks:
            checks = [{"condition": item, "effect": "重新核对本次买卖取舍", "fact_ids": [], "status": "watch"}
                      for item in investor.get("change_conditions", [])]
        for check in checks:
            check.setdefault("status", "watch")
        label = _ACTION_LABELS.get(preference, "暂不操作，先核对买卖依据")
        if decision_status == "needs_review":
            label = "暂不操作，先更新研究"
        result_cases[group] = {
            "preferred_action": preference, "research_action": research_actions[group], "economic_action": economic_action,
            "label": label, "advice_reason": advice_reason, "next_step": next_step,
            "comparison_action": comparison_action,
            "reasons": list(dict.fromkeys([*([advice_reason] if advice_reason else []),
                                           *([case["reason"]] if case.get("reason") else []), *reasons])),
            "fact_ids": fact_ids, "why_not": alternatives,
            "counterevidence": deepcopy(investor.get("counterevidence") or []),
            "checks": checks, "needed_review": review,
            "conflict": {"status": conflict_status, "resolution": resolution.get("reason"),
                         "fact_ids": list(resolution.get("fact_ids") or [])},
            "execution_status": execution,
            "comparison_status": comparison_status, "decision_status": decision_status,
            "expected_value": chosen.get("expected_value"), "lower_value": chosen.get("lower_value"),
            "upper_value": chosen.get("upper_value"),
            "initial_value": chosen.get("initial_value"),
            "reduction_ratio": None,
        }
    return {
        "schema_version": 2, "policy_version": LEGACY_POLICY_VERSION, "product_id": product_id,
        "research_record_id": thesis.get("record_id"), "research_date": payload.get("research_date"),
        "research_summary": investor.get("research_summary"), "origin": deepcopy(payload.get("origin")),
        "rationale": supplement, "rationale_status": supplement_status, "cases": result_cases,
        "plan_semantics": "天数是本次考虑操作和复核的窗口；计划日卖出只用于同口径算账，不代表必须到期清仓或近期要用钱。",
        "amount_basis": "同组、同资本、同一准确预测头；差额是备选减首选，不是动作全排名或首选正确概率。",
    }


def build_decision_brief(*, product_id: str | None, research_thesis: Mapping | None,
                         investor_decision: Mapping, trade_scenario: Mapping,
                         model_id: str | None = None, rationale: Mapping | None = None) -> dict:
    """Project the adopted economic choice; an unverified mean is not an order.

    The model comparison remains available for research and costs. It cannot
    select a different action through the legacy aligned/opposed resource.
    """
    thesis, investor = _mapping(research_thesis), investor_decision
    payload = _mapping(thesis.get("payload"))
    authored = _mapping(payload.get("action_cases"))
    independent = investor.get("action_basis") == "independent_cases"
    branches = _mapping(investor.get("branches"))
    flat = _branch_action(_mapping(branches.get("flat"))) if independent else None
    held = _branch_action(_mapping(branches.get("held"))) if independent else None
    addition = _branch_action(_mapping(branches.get("addition"))) if independent else None
    actions = {"flat": flat if flat in {"buy", "wait"} else None,
               "held": _held_choice(held, addition)}
    conditions = _mapping(investor.get("execution_conditions")) or _mapping(
        trade_scenario.get("execution_conditions"))
    rows = {row["action"]: row for row in trade_scenario.get("rows", [])}
    result_cases = {}
    for group, action in actions.items():
        source = "new_buy" if group == "flat" else "addition" if action == "add" else "held"
        case = _mapping(authored.get(source))
        branch_keys = ["flat"] if group == "flat" else ["held", "addition"] if held == "hold" else ["held"]
        relevant = [_mapping(branches.get(key)) for key in branch_keys]
        research_action = (_mapping(branches.get("flat")).get("research_action") if group == "flat"
                           else _held_choice(_mapping(branches.get("held")).get("research_action"),
                                             _mapping(branches.get("addition")).get("research_action")))
        constraints = list(dict.fromkeys(reason for branch in relevant
                                         if _branch_action(branch) not in {None, branch.get("research_action")}
                                         for reason in branch.get("constraint_reasons", []) if reason))
        constrained = any(branch.get("constraint_reasons") and _branch_action(branch) not in
                          {None, branch.get("research_action")} for branch in relevant)
        reasons = list(dict.fromkeys(reason for branch in relevant for reason in branch.get("reasons", [])))
        fact_ids = sorted({identifier for branch in relevant for identifier in branch.get("fact_ids", [])})
        preference = action
        review = list(dict.fromkeys([*investor.get("needed_review", []),
                                    *(item for branch in relevant for item in branch.get("needed_review", []))]))
        reason = (constraints[0] if constrained and constraints else
                  case.get("summary") or case.get("reason") or next(iter(reasons), None))
        next_step = case.get("next_step")
        status = "constrained" if constrained and action else "economic_research" if action else "unresolved"
        if "research_horizon_elapsed" in review:
            preference, status = None, "needs_review"
            reason = "这份研究已过适用期限，先更新经营与价格判断，再调整投入。"
            next_step = "取得新的经营与价格依据后，再考虑买卖。"
        elif review and preference:
            status = "provisional"
        economic_action = preference
        execution = _execution(preference, conditions)
        if execution["status"] == "blocked":
            preference, status = ("wait" if group == "flat" else "hold"), "execution_blocked"
            reason = ("当前暂停申购，这次先不新买。" if group == "flat" else
                      "当前暂停申购，保留已有份额，这次不追加。" if execution["operation"] == "subscription" else
                      "当前暂停赎回，暂时保留已有份额。")
            next_step = "申赎恢复后，再按届时经营、价格与费用核对这次操作。"
        if preference is None:
            reason = reason if status == "needs_review" else "当前尚未形成有依据的买卖判断，这次先不改变资金安排。"
            next_step = next_step or "先补齐决定买卖的经营与价格依据。"
        changes = investor.get("change_conditions") or []
        next_step = next_step or next(iter(changes), "下次经营或价格资料更新后，重新核对本次判断。")
        comparison = _mapping(case.get("comparison"))
        comparison_action = comparison.get("action") or (
            "wait" if preference == "buy" else "buy" if group == "flat" else
            "hold" if preference in {"add", "redeem"} else "exit")
        chosen = rows.get(_ACTION_ROWS.get(preference, preference), {}) if preference != "reduce" else {}
        alternatives = []
        for candidate in (("buy", "wait") if group == "flat" else ("hold", "add", "reduce50", "exit")):
            if candidate == _ACTION_ROWS.get(preference, preference):
                continue
            row = rows.get(candidate, {})
            delta = _number(row) - _number(chosen) if _number(row) is not None and _number(chosen) is not None else None
            is_reference = candidate == _ACTION_ROWS.get(comparison_action, comparison_action)
            alternatives.append({"action": candidate, "label": _ACTION_LABELS[candidate],
                "reason": comparison.get("reason") if is_reference else None,
                "fact_ids": list(comparison.get("fact_ids") or []) if is_reference else [],
                "expected_value": row.get("expected_value"),
                "expected_difference_vs_preferred": float(delta) if delta is not None else None})
        candidates = [rows.get(key, {}) for key in (("buy", "wait") if group == "flat" else ("hold", "add", "reduce50", "exit"))]
        comparison_status = ("ready" if all(row.get("status") == "ready" for row in candidates) else
                             "partial" if any(row.get("status") == "ready" for row in candidates) else "unavailable")
        result_cases[group] = {
            "preferred_action": preference, "research_action": research_action,
            "economic_action": economic_action, "label": _ACTION_LABELS.get(preference, "暂不操作，先核对买卖依据"),
            "advice_reason": reason, "next_step": next_step, "fact_ids": fact_ids,
            "comparison_action": comparison_action, "why_not": alternatives,
            "reasons": list(dict.fromkeys([*([reason] if reason else []), *reasons])),
            "counterevidence": deepcopy(investor.get("counterevidence") or []),
            "checks": [{"condition": item, "effect": "重新核对本次经济判断", "status": "watch", "fact_ids": []}
                       for item in changes],
            "needed_review": review, "decision_status": status, "execution_status": execution,
            "comparison_status": comparison_status,
            "conflict": {"status": _conflict(action, rows, group) if action else "unavailable",
                         "role": "diagnostic_only", "resolution": None, "fact_ids": []},
            "decision_basis": "economic_research_and_explicit_constraints",
            "forecast_evidence": {"role": "scenario_reference_only", "timing_advantage_established": False},
            "expected_value": chosen.get("expected_value"), "lower_value": chosen.get("lower_value"),
            "upper_value": chosen.get("upper_value"), "initial_value": chosen.get("initial_value"),
            "reduction_ratio": None,
        }
    return {
        "schema_version": 3, "policy_version": POLICY_VERSION, "product_id": product_id,
        "research_record_id": thesis.get("record_id"), "research_date": payload.get("research_date"),
        "research_summary": investor.get("research_summary"), "origin": deepcopy(payload.get("origin")),
        "data_date": _mapping(trade_scenario.get("plan")).get("anchor_date"),
        "cases": result_cases, "rationale": None, "rationale_status": "not_used_by_economic_policy",
        "plan_semantics": "天数用于考虑操作和复核，不代表必须到期清仓。",
        "amount_basis": "金额仅供后台同资本情景核对，不据未经验证的均值差独立改变操作。",
    }

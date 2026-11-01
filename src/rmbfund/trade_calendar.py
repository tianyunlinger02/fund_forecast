"""Pure, conditional application schedule; never a future NAV calendar or forecast."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, time, timedelta
from typing import Any

from .domain import ValidationError, timestamp
from .paper.fund import application_date


_NEW_SHARES_UNCONFIRMED = "新份额在计划卖出日前尚未假定确认，新买/追加后卖出不可比较；已有份额比较仍保留。"


class _MissingCalendar(Exception):
    def __init__(self, day: date) -> None:
        self.day = day


def _calendar_days(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 90:
        raise ValidationError("calendar_days必须是1至90的自然日整数")
    return value


def adjust_new_confirmation_date(plan: Mapping, confirmation_date: str) -> dict[str, Any]:
    """Update only the confirmation assumptions affected by a manual date edit."""
    result = deepcopy(dict(plan))
    selected = date.fromisoformat(confirmation_date).isoformat()
    if selected == result["new_confirmation_date"]:
        return result
    result["default_new_confirmation_date"] = result["new_confirmation_date"]
    previous_label = result.get("confirmation_label")
    result["confirmation_basis"] = "manual_date_scenario"
    result["confirmation_label"] = "手动指定的模拟确认日（非实际确认记录）"
    result["new_confirmation_date"] = selected
    result["new_shares_sellable"] = bool(result.get("status") == "ready" and result.get("sale_date")
        and result["application_date"] <= selected <= result["sale_date"])
    result["assumptions"] = [result["confirmation_label"] if item == previous_label else item
                             for item in result.get("assumptions", [])]
    result["reasons"] = [item for item in result.get("reasons", []) if item != _NEW_SHARES_UNCONFIRMED]
    if result.get("status") == "ready":
        if selected < result["application_date"]:
            result["reasons"].append("新份额模拟确认日早于本次申购申请日，不能继承旧份额费龄。")
        elif selected > result["sale_date"]:
            result["reasons"].append(_NEW_SHARES_UNCONFIRMED)
    return result


def project_sale_date(plan: Mapping, *, not_before: str) -> dict[str, Any]:
    """Project FORWARD within the original plan, retaining its exact A-to-D H/L.

    The saved assumed-open dates come from the same all-calendar or all-weekday
    schedule as A and S. No new clock, calendar lookup or backward roll is used.
    A legacy plan without those dates cannot supply an exact intermediate H.
    """
    requested = date.fromisoformat(not_before).isoformat()
    application, last = plan.get("application_date"), plan.get("sale_date")
    result = {"requested_date": requested, "sale_date": None, "horizon": None,
              "entry_lag": plan.get("entry_lag"), "calendar_basis": plan.get("calendar_basis"),
              "status": "outside_window", "reasons": []}
    if application is None or last is None or last < application or requested > last:
        result["reasons"] = ["降费日或本次申请日之后，在原计划卖出日之前没有候选开放日。"]
        return result
    if "assumed_open_dates" not in plan:
        return {**result, "status": "missing_calendar_projection",
                "reasons": ["原计划未保留同口径假定开放日，不能另猜中间卖出日或净值笔数。"]}
    opens = sorted(set(plan["assumed_open_dates"]))
    selected = next((day for day in opens if max(application, requested) <= day <= last), None)
    if selected is None:
        result["reasons"] = ["降费日之后至原计划卖出日没有同口径假定开放日；不向前退回未降费日期。"]
        return result
    return {**result, "status": "ready", "sale_date": selected,
            "horizon": sum(application < day <= selected for day in opens)}


def build_trade_plan(
    *, decision_at: str, anchor_date: str | None, calendar_days: int,
    entry_assumptions: Mapping | None = None, calendar: Mapping | None = None,
    confirmation_days: int | None = None,
) -> dict[str, Any]:
    """Map an explicit day window to an assumed, exact NAV-observation H/L.

    ``calendar`` maps ISO dates to booleans or mappings with ``is_open`` (also
    accepted under ``payload``). Its caller owns profile applicability, revision
    selection and the actual availability cutoff. Any missing relevant date
    switches the WHOLE schedule to a visibly named Monday-Friday scenario.
    Even a complete business calendar does not establish future NAV observations.
    This helper performs no I/O, reads no clock and needs no NAV to show dates.
    """
    days = _calendar_days(calendar_days)
    point = timestamp(decision_at)
    today, deadline = point.date(), point.date() + timedelta(days=days)
    assumptions = dict(entry_assumptions or {})
    cutoff = assumptions.get("cutoff", "15:00")
    try:
        parsed_cutoff = time.fromisoformat(cutoff)
        if parsed_cutoff.tzinfo is not None:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValidationError("申请截止时点必须是不含时区的有效时间") from exc
    cutoff = parsed_cutoff.isoformat(timespec="minutes")
    anchor = date.fromisoformat(anchor_date) if anchor_date is not None else None
    if anchor is not None and anchor > today:
        raise ValidationError("净值锚点不能晚于本次测算日")
    if confirmation_days is not None:
        confirm = confirmation_days
        confirmation_basis = "manual_workday_scenario"
        confirmation_label = "手动情景：申请后按假定业务日确认"
    else:
        confirm = assumptions.get("confirmation_days", 2)
        confirmation_basis = assumptions.get("confirmation_basis", "illustrative_2_workdays")
        confirmation_label = assumptions.get("confirmation_label", "示例：2个工作日确认（非已核实规则，可调整）")
    if isinstance(confirm, bool) or not isinstance(confirm, int) or confirm < 0:
        raise ValidationError("confirmation_days必须是非负业务日整数")

    opens = {}
    for day, item in (calendar or {}).items():
        if isinstance(item, bool):
            opens[day] = item
        elif isinstance(item, Mapping):
            payload = item.get("payload", item)
            if isinstance(payload, Mapping) and isinstance(payload.get("is_open"), bool):
                opens[day] = payload["is_open"]
    needed, assumed_open = set(), set()

    def schedule(use_calendar: bool) -> tuple[date, date | None, date, int | None, int]:
        needed.clear()
        assumed_open.clear()

        def is_open(day: date) -> bool:
            key = day.isoformat()
            needed.add(key)
            if use_calendar and key not in opens:
                raise _MissingCalendar(day)
            opened = opens[key] if use_calendar else day.weekday() < 5
            if opened:
                assumed_open.add(key)
            return opened

        # Require the entire relevant window, not just the days encountered by
        # forward/backward searches. Weekend closure records matter as well.
        start = min(today, anchor + timedelta(days=1)) if anchor is not None else today
        day = start
        while day <= deadline:
            is_open(day)
            day += timedelta(days=1)
        application = application_date(point, {"cutoff": cutoff}, is_open)
        sale = deadline
        while sale >= today and not is_open(sale):
            sale -= timedelta(days=1)
        if sale < today:
            sale = None  # Never invent a sale in the past or roll past the deadline.
        confirmation = application
        remaining = confirm
        for _ in range(370):
            if remaining == 0:
                break
            confirmation += timedelta(days=1)
            remaining -= int(is_open(confirmation))
        if remaining:
            raise ValidationError("一年内没有足够的假定业务日完成份额确认")

        def count_after(start: date, end: date) -> int:
            result, day = 0, start + timedelta(days=1)
            while day <= end:
                result += int(is_open(day))
                day += timedelta(days=1)
            return result

        lag = count_after(anchor, application) if anchor is not None else None
        horizon = count_after(application, sale) if sale is not None else 0
        return application, sale, confirmation, lag, horizon

    missing = None
    try:
        scheduled = schedule(True)
        calendar_basis = "available_profile_business_calendar"
        calendar_label = "本地已取得且覆盖完整的适用业务日历（未来净值笔数仍为假设）"
    except _MissingCalendar as exc:
        missing = exc.day.isoformat()
        scheduled = schedule(False)
        calendar_basis = "weekday_scenario"
        calendar_label = "周一至周五模拟日程（未核验业务日历，不含节假日调整）"
    application, sale, confirmation, lag, horizon = scheduled
    reasons = []
    if missing is not None:
        reasons.append(f"本地业务日历在{missing}缺少适用覆盖，整份计划采用周一至周五模拟日程。")
    feasible = application <= deadline and sale is not None and horizon > 0
    if not feasible:
        reasons.append("该窗口过短：截止日前无法在申请后取得正数净值区间；没有强制改成1笔或顺延截止日。")
    new_sellable = bool(feasible and sale is not None and confirmation <= sale)
    if feasible and not new_sellable:
        reasons.append(_NEW_SHARES_UNCONFIRMED)
    if anchor is None:
        reasons.append("尚无实际净值锚点，只展示条件日程；不能据此生成净值预测。")
    return {
        "kind": "calendar_trade_plan", "status": "ready" if feasible else "too_short",
        "decision_at": point.isoformat(), "decision_date": today.isoformat(),
        "calendar_days": days, "deadline": deadline.isoformat(), "anchor_date": anchor_date,
        "application_date": application.isoformat(), "sale_date": sale.isoformat() if sale else None,
        "entry_lag": lag, "horizon": horizon, "target_unit": "nav_observations",
        "new_confirmation_date": confirmation.isoformat(), "confirmation_days": confirm,
        "confirmation_basis": confirmation_basis, "confirmation_label": confirmation_label,
        "new_shares_sellable": new_sellable,
        "cutoff": cutoff, "cutoff_basis": assumptions.get("cutoff_basis", "15:00_scenario"),
        "profile_id": assumptions.get("profile_id"),
        "profile_source_url": assumptions.get("profile_source_url"),
        "calendar_id": assumptions.get("calendar_id"),
        "calendar_basis": calendar_basis, "calendar_label": calendar_label,
        "calendar_coverage_start": min(needed) if needed else None,
        "calendar_coverage_end": max(needed) if needed else None,
        "assumed_open_dates": sorted(assumed_open),
        "first_missing_calendar_date": missing,
        "assumptions": [
            "截止日按本次实际测算日加所选自然日计算，不从滞后的净值日期起算。",
            "申请日使用截止时点规则；计划卖出为截止日前最后假定开放日，不自动寻找最佳买卖日。",
            "每个假定业务日对应1笔未来净值，仅为日程假设，不是已知或已核验的未来净值记录数。",
            "L按锚点后至申请日的假定业务日计，H按申请日后至计划卖出日计；概率目标仍为确切H笔净值。",
            confirmation_label,
            "旧份额与新份额确认起点独立；计划卖出净额包含赎回应收款，不承诺截止日资金到账。",
        ],
        "reasons": reasons,
    }

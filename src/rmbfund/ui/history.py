"""Read saved forecast heads and personal snapshots; never recompute history."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import streamlit as st

from rmbfund.assessment_changes import extract_assessment_changes
from rmbfund.forecast_models import model_catalog, model_family, model_name
from rmbfund.personal_assessment import explain_forecast_head
from rmbfund.ui.formatting import lines, mapping, markdown_text, number, percent, when
from rmbfund.ui.investment import render_saved_investment_analysis


HORIZONS = {1: "之后1笔净值", 5: "之后5笔净值", 20: "之后20笔净值（通常约一个月）"}
ROLES = {"primary": "主行动情景", "anchor": "公布净值起点", "stress": "额外延迟情景"}


def _write(value: Any) -> None:
    st.markdown(markdown_text(value))


def _short(value: Any) -> str:
    return str(value)[:8] if value else "未记录"


def _count(value: Any) -> str:
    return str(int(value)) if number(value) is not None else "未提供"


def _lag(row: Mapping) -> int | None:
    return row.get("entry_lag", mapping(row.get("head")).get("entry_lag"))


def _roles_match(item: Mapping, role: str) -> bool:
    roles = item.get("roles")
    if isinstance(roles, (list, tuple)):
        return role in roles
    return item.get("role") in (None, role)


def outcome_label(outcome: Any) -> str:
    if not isinstance(outcome, Mapping):
        return "尚无已记录结果"
    if outcome.get("eligible_forward") is True:
        return "完整未来区间"
    if outcome.get("eligible_forward") is False:
        return "公布滞后覆盖的区间"
    return "结果类别未注明"


def _score(value: Any, key: str, *, delta: bool = False) -> str:
    parsed = number(value)
    if parsed is None:
        return "暂无结果"
    if key == "brier_score":
        return f"{parsed:+.3f}" if delta else f"{parsed:.3f}"
    if key in {"mae", "interval_width"} or delta:
        return f"{parsed * 100:+.2f}个百分点" if delta else f"{parsed * 100:.2f}个百分点"
    return percent(parsed)


def _paired_table(paired: Mapping, *, model_label: str = "所选预测方法") -> None:
    count = number(paired.get("count"))
    if paired.get("comparison_state") == "not_pooled":
        st.caption(f"已保存{_count(count)}个同发行基准配对；不同算法、延迟或预测目标分开比较，不合成平均成绩。")
        return
    if not count:
        st.caption("还没有同时封存模型与经验基准、且已有真实未来结果的配对样本。旧记录没有基准就不补算。")
        return
    st.caption(f"只比较同一批{int(count)}个经济窗口；没有当时封存基准的记录不在这个对照里。")
    metrics = {"brier_score": "上涨概率误差（越低越好）", "mae": "平均涨跌误差（越低越好）",
               "interval_coverage": "80%范围实际覆盖率", "interval_width": "预测范围平均宽度"}
    rows = []
    for name, label in (("model", model_label), ("baseline", "当时的经验基准"), ("delta", "模型减去基准")):
        values = mapping(paired.get(name))
        rows.append({"比较方法": label, **{label: _score(values.get(key), key, delta=name == "delta")
                                          for key, label in metrics.items()}})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("经验基准是当时已知的同期限历史涨跌分布，不是另一套可选买卖模型。概率误差不是亏损比例；"
               "涨跌误差表示预测与实际相差多少。范围更宽通常更容易覆盖实际结果，不能只看覆盖率。")
    identical = paired.get("identical_to_baseline_count")
    if identical is not None:
        st.caption(f"其中{_count(identical)}个窗口的模型预测与经验基准完全相同，不能当成模型新增的判断能力。")


def _summary_scope(value: Mapping, role: str, *, label: str = "") -> None:
    counts = mapping(value.get("counts"))
    metrics = mapping(value.get("forward_metrics"))
    # These are economic-window counts, not the number of raw saved outcomes.
    forward = metrics.get("count", counts.get("forward_windows"))
    title = label or ("主行动情景的真实成绩" if role == "primary" else f"{ROLES[role]}的单独结果")
    st.markdown(f"**{title}**")
    _write(f"已有{_count(forward)}个不重复的行情区间，记录了预测真正发出后的完整未来结果；"
           f"另有{_count(counts.get('unrecorded'))}条发行记录尚未记录结果。")
    st.caption("这只能帮助核对净值判断，不证明买卖会盈利。尚未记录不一定尚未到期；浏览不会补算结果。")
    if role != "primary":
        st.caption("这组不是主行动判断的成绩，不与主行动情景混算，也不拿历史回算补齐。")
    if number(forward) and value.get("metrics_available") is not False:
        columns = st.columns(3)
        for column, key, title in zip(columns, ("direction_accuracy", "mae", "interval_coverage"),
                                      ("方向判断正确比例", "平均涨跌误差", "80%范围实际覆盖率")):
            column.metric(title, _score(metrics.get(key), key))
    elif number(forward):
        st.info("已有完整未来结果，但这组没有可合并的准确性指标；请分别查看各预测方法或情景，没有用0%代替未知。")
    else:
        st.info("还没有可用于这组真实成绩的完整未来结果，暂不能判断准不准；没有用0%代替未知。")
    with st.expander(f"统计口径与经验基准对照{(' · ' + label) if label else ''}"):
        labels = {
            "issued": "实际发行记录", "recorded_forward": "已记录完整未来结果的发行记录（未去重）",
            "recorded_nowcast": "已有公布滞后区间结果的发行记录", "unrecorded": "尚无已记录结果的发行记录",
            "unknown_window": "实际起止日期尚不能确定的记录", "unresolved_windows": "最早发行起止日期待核对的行情区间",
            "unique_windows": "可确定的不重复行情区间",
            "nonoverlapping_windows": "互不重叠的行情区间", "forward_windows": "有完整未来结果的不重复区间",
            "forward_nonoverlapping_windows": "有完整未来结果且互不重叠的区间", "paired": "可与经验基准配对的区间",
        }
        st.dataframe([{"记录口径": text, "数量": _count(counts.get(key))} for key, text in labels.items()],
                     hide_index=True, width="stretch")
        if number(counts.get("unresolved_windows")):
            st.caption("最早发行的起止日期待核对，没有改用较晚版本，暂未计入成绩。")
        st.caption("每种方法在同一行情区间采用最早的发行，不因缺少结果或基准改用较晚版本。"
                   "相邻区间可能重叠；不重叠也不等于互相独立。"
                   "这里汇总所选方法、基金、期限与情景的全部相关历史，不只是下面这一页。")
        _paired_table(mapping(value.get("paired")),
                      model_label=value.get("model_name") or model_name(value.get("model_id")) or "所选预测方法")


def _model_summary(value: Mapping, role: str, *, model_label: str = "") -> None:
    groups = value.get("groups") or []
    if groups:
        st.caption("不同延迟或不同预测目标分别核对，不合成一个成绩。")
        for index, group in enumerate(groups):
            label = (f"额外延迟{group['entry_lag']}笔净值" if role == "stress" and group.get("entry_lag") is not None
                     else f"{ROLES[role]} · 第{index + 1}组预测目标")
            _summary_scope(group, role, label=f"{model_label} · {label}" if model_label else label)
    elif role == "stress":
        st.info("还没有按额外延迟分别汇总的真实结果；不会合并不同延迟，也不作为主行动成绩。")
    else:
        title = f"{model_label} · {ROLES[role]}的真实成绩" if model_label else ""
        _summary_scope(value, role, label=title)


def _render_summary(callback: Callable | None, *, horizon: int, role: str, model_id: str | None,
                    entry_lag: int | None) -> None:
    if callback is None:
        st.caption("尚未提供全部历史的真实结果汇总；不会用当前页面的记录冒充总成绩。")
        return
    try:
        value = mapping(callback(horizon=horizon, role=role, model_id=model_id))
    except Exception:
        st.info("暂时读不到全部历史的真实结果汇总；不会用当前页面或回测代替。")
        return
    if not value or value.get("status") == "error":
        st.info("这组真实结果汇总暂不可用，没有用0%代替未知。")
        return
    if entry_lag is not None and role != "anchor":
        st.caption("指定延迟只筛选下方记录；成绩仍覆盖所选情景的全部相关历史。")
    by_model = value.get("by_model") or []
    if value.get("metrics_scope") == "models_only" or len(by_model) > 1:
        st.caption("不同预测方法分别核对，不把准确率和误差混成一个总成绩。"
                   "同一行情可以分别检验两种方法，不代表有两段独立行情。")
        for group in by_model:
            label = group.get("model_name") or model_name(group.get("model_id")) or "已保存预测方法"
            _model_summary(group, role, model_label=label)
    else:
        label = value.get("model_name") or model_name(model_id) or ""
        _model_summary(value, role, model_label=label)


def _saved_head(prediction: Mapping, row: Mapping, *, horizon: int, role: str) -> Mapping:
    lag = _lag(row)
    if row.get("horizon", mapping(row.get("head")).get("horizon")) != horizon:
        return {}
    if lag is None or (role == "anchor" and lag != 0) or not _roles_match(row, role):
        return {}
    assumptions = mapping(prediction.get("entry_assumptions"))
    primary = assumptions.get("baseline_entry_lag")
    if primary is not None and ((role == "primary" and lag != primary) or (role == "stress" and lag == primary)):
        return {}
    collection = "forecasts" if role == "anchor" else "execution_forecasts"
    return next((head for head in prediction.get(collection, [])
                 if head.get("horizon") == horizon and head.get("entry_lag") == lag
                 and _roles_match(head, role)), {})


def _render_reasons(prediction: Mapping, head: Mapping, role: str) -> None:
    st.markdown("**当时的理由**")
    for reason in lines(head.get("reasons")):
        _write(reason)
    explanation = explain_forecast_head(prediction, head, horizon=head["horizon"],
                                        entry_lag=head["entry_lag"], role=role)
    for reason in lines(explanation.get("reasons")):
        _write(reason)
    st.caption("以上只解释原记录保存的信息；没有使用今天的资料补写，也没有重新拟合。")


def _public_detail(callback: Callable, row: Mapping, *, horizon: int, role: str) -> None:
    try:
        saved = mapping(callback(row["record_id"]))
    except Exception:
        st.error("这条记录暂时打不开；没有使用今天的数据重新计算。")
        return
    if saved.get("status") != "ready":
        st.info("这条已保存预测暂时无法读取。")
        return
    prediction = mapping(saved.get("prediction"))
    head = _saved_head(prediction, row, horizon=horizon, role=role)
    if not head:
        st.caption("这次记录没有所选范围、延迟和情景的原预测。旧输出可能没有短期限的行动预测，不借其他期限或起点预测替代。")
        return
    lag = head["entry_lag"]
    with st.container(border=True, key="history_detail"):
        st.markdown("**当时的原预测**")
        _write(f"发出于 {when(saved.get('generated_at') or prediction.get('generated_at'))} · "
               f"{saved.get('model_name') or row.get('model_name') or '已保存预测方法'} · "
               f"保存版本 {_short(saved.get('model_hash') or prediction.get('model_hash') or row.get('model_hash'))}")
        anchor = mapping(prediction.get("anchor"))
        st.caption(markdown_text(f"原净值起点 {anchor.get('date', '未记录')} · 单位净值 {anchor.get('nav', '未记录')}"))
        st.caption(f"从公布净值起点往后看{horizon}笔净值。" if role == "anchor" else
                   f"当时假设延迟{lag}笔净值成交，再看之后{horizon}笔净值；不是确定的成交日期。")
        columns = st.columns(3)
        columns[0].metric("当时上涨概率", percent(head.get("up_probability")))
        columns[1].metric("当时预期涨跌", percent(head.get("expected_change"), signed=True))
        columns[2].metric("当时80%预测范围", f"{percent(head.get('lower'), signed=True)} ～ {percent(head.get('upper'), signed=True)}")
        _render_reasons(prediction, head, role)
        outcome = next((item for item in saved.get("outcomes", [])
                        if item.get("horizon") == horizon and item.get("entry_lag") == lag
                        and _roles_match(item, role)), None)
        if outcome is None:
            candidate = mapping(row.get("outcome"))
            if candidate.get("horizon") == horizon and candidate.get("entry_lag") == lag and _roles_match(candidate, role):
                outcome = candidate
        if outcome:
            _write(f"已记录结果：{outcome_label(outcome)} · 实际涨跌 {percent(outcome.get('actual_change'), signed=True)}")
            st.caption(markdown_text(f"实际起点 {mapping(outcome.get('entry')).get('date', '未记录')} → "
                                     f"目标日期 {mapping(outcome.get('target')).get('date', '未记录')} · "
                                     f"结果记录于 {when(outcome.get('scored_at'))}"))
        else:
            st.caption("尚无这个预测头的已记录结果，不代表一定尚未到期。")
        st.caption("原始基金预测没有使用你今天的份额或费用；当时的个人买卖判断请看下方保存快照。")


def _filters(models: list[dict] | None) -> tuple[int, str, str | None, int | None]:
    catalog = {}
    source = models if models is not None else model_catalog()
    for raw in source:
        item = mapping(raw)
        model_id = item.get("id") or item.get("model_id")
        if not model_id:
            continue
        label = item.get("label") or item.get("name") or model_name(model_id) or "已保存预测方法"
        if item.get("available") is False:
            label = f"{label}（当前基金暂不可生成）"
        catalog[model_id] = label
    saved = st.session_state.setdefault("history_filters", {"horizon": 20, "model": None, "role": "primary", "lag": None})
    saved_model = saved.get("model")
    if saved_model not in catalog and saved_model is not None:
        saved_model = next((item for item in catalog
                            if model_family(item) == model_family(saved_model)), None)
    for name, value in (("horizon", saved["horizon"]), ("role", saved["role"]),
                        ("model", saved_model), ("entry_lag", saved["lag"])):
        if f"history_{name}" not in st.session_state:
            st.session_state[f"history_{name}"] = value
    if st.session_state["history_model"] not in (None, *catalog):
        st.session_state["history_model"] = saved_model
    columns = st.columns(2)
    with columns[0]:
        horizon = st.selectbox("预测范围", [20, 5, 1], format_func=HORIZONS.get, key="history_horizon")
        model_id = st.selectbox("保存记录范围", [None, *catalog], key="history_model",
                                format_func=lambda key: "所有已保存模型" if key is None else catalog[key],
                                placeholder="所有已保存模型")
    with columns[1]:
        role = st.selectbox("预测起点与情景", list(ROLES), format_func=ROLES.get, key="history_role")
        if role != saved["role"]:
            st.session_state["history_entry_lag"] = 0 if role == "anchor" else None
        lag = st.number_input("只看指定延迟（净值笔数，可留空）", min_value=0, step=1, value=None,
                              key="history_entry_lag", disabled=role == "anchor", placeholder="留空按原发行情景查看")
    lag = 0 if role == "anchor" else lag
    saved.update(horizon=horizon, model=model_id, role=role, lag=lag)
    st.caption("模型名称来自当前目录；选择一个方法时，同一算法家族的历史输出版本会一起核对。")
    return horizon, role, model_id, lag


def _pagination(page: int, more: bool, *, prefix: str) -> None:
    columns = st.columns([1, 2, 1])
    with columns[0]:
        if st.button("上一页", disabled=page == 0, key=f"{prefix}_previous"):
            st.session_state[f"{prefix}_page"] = page - 1
            st.rerun()
    columns[1].caption(f"第 {page + 1} 页 · 浏览不会改变原记录")
    with columns[2]:
        if st.button("下一页", disabled=not more, key=f"{prefix}_next"):
            st.session_state[f"{prefix}_page"] = page + 1
            st.rerun()


def _public_history(product_id: str | None, read: Callable | None, detail: Callable | None,
                    models: list[dict] | None, summary: Callable | None) -> None:
    st.subheader("原始基金预测")
    st.caption("这里只展示实际保存过的预测，不混入历史回算。结果为空不一定尚未到期，也可能还未记录结果。")
    if product_id is None or read is None:
        st.info("这只基金还没有保存过公开预测。明确更新预测后，记录会留在这里。")
        return
    horizon, role, model_id, lag = _filters(models)
    scope = (product_id, horizon, role, model_id, lag)
    if st.session_state.get("history_scope") != scope:
        st.session_state["history_page"] = 0
        st.session_state["history_scope"] = scope
    page = st.session_state.get("history_page", 0)
    _render_summary(summary, horizon=horizon, role=role, model_id=model_id, entry_lag=lag)
    try:
        rows = read(horizon=horizon, role=role, model_id=model_id, entry_lag=lag, limit=21, offset=page * 20)
    except Exception:
        st.error("暂时读不到历史预测。原记录没有重新计算或覆盖，请稍后再试。")
        return
    if not rows:
        st.info("这个范围还没有已保存的预测。旧输出可能缺少短期限行动预测，不用其他期限或起点补齐。")
        return
    shown = rows[:20]
    table = []
    for item in shown:
        head, outcome = mapping(item.get("head")), mapping(item.get("outcome"))
        if head.get("horizon") != horizon or not _roles_match(item, role) or (lag is not None and _lag(item) != lag):
            head, outcome = {}, {}
        table.append({"预测时间": when(item.get("generated_at")), "情景": ROLES[role],
                      "延迟笔数": str(_lag(item)) if _lag(item) is not None else "未记录",
                      "模型": item.get("model_name") or "已保存预测方法", "版本": _short(item.get("model_hash")),
                      "上涨概率": percent(head.get("up_probability")),
                      "预期涨跌": percent(head.get("expected_change"), signed=True),
                      "80%预测范围": f"{percent(head.get('lower'), signed=True)} ～ {percent(head.get('upper'), signed=True)}",
                      "实际涨跌": percent(outcome.get("actual_change"), signed=True) if outcome else "—",
                      "结果": outcome_label(outcome if outcome else None)})
    st.dataframe(table, hide_index=True, width="stretch")
    _pagination(page, len(rows) > 20, prefix="history")
    if detail is None:
        return
    # Several stress heads can belong to the same record.  Never key this only
    # by record_id, which silently discards one of the original delays.
    records = {(row["record_id"], horizon, _lag(row), role): row for row in shown}
    if st.session_state.get("history_selected_head") not in records:
        st.session_state["history_selected_head"] = next(iter(records))
    choices = list(records)
    selected = st.selectbox("查看某一次预测", choices, key="history_selected_head",
                            format_func=lambda key: f"{when(records[key].get('generated_at'))} · 延迟{key[2]}笔 · 第{choices.index(key) + 1}条")
    _public_detail(detail, records[selected], horizon=horizon, role=role)


def _money(value: Any, currency: Any) -> str:
    if value is None:
        return "未提供"
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return "未提供"
    if not amount.is_finite():
        return "未提供"
    return f"{amount:,.2f} {currency or '（币种未记录）'}"


def _is_standard_trade(payload: Mapping) -> bool:
    return payload.get("schema_version") == 5 and payload.get("mode") == "standard_trade"


def _is_calendar_trade(payload: Mapping) -> bool:
    return payload.get("schema_version") == 6 and payload.get("mode") == "calendar_trade"


def _is_buyer_decision(payload: Mapping) -> bool:
    return payload.get("schema_version") == 7 and payload.get("mode") == "buyer_decision"


def _has_buyer_projection(payload: Mapping) -> bool:
    return (_is_buyer_decision(payload)
            and (mapping(payload.get("decision_brief")).get("schema_version") == 3
                 or payload.get("policy_version") == "fund_buyer_economic_advice_v3"))


def _snapshot_label(payload: Mapping) -> str:
    if _has_buyer_projection(payload):
        view = mapping(payload.get("buyer_view"))
        branch = payload.get("selected_branch")
        context = {"flat": "还没买", "held": "已经持有"}.get(branch, "情境未记录")
        return f"{context} · {view.get('label') or '未保存展示建议'}"
    if _is_buyer_decision(payload):
        brief = mapping(payload.get("decision_brief"))
        cases = mapping(brief.get("cases"))
        plan = mapping(mapping(payload.get("trade_scenario")).get("plan"))
        days = plan.get("calendar_days")
        window = f"未来{days}天" if days is not None else "天数未记录"
        decisions = "；".join(
            f"{name}：{mapping(cases.get(key)).get('label') or '未保存首选'}"
            for key, name in (("flat", "新买"), ("held", "已有基金与备用现金")))
        return f"基金买卖首选 · {window} · {decisions}" if brief else "未保存基金买卖首选摘要"
    if _is_calendar_trade(payload):
        scenario = mapping(payload.get("trade_scenario"))
        if not scenario:
            return "未保存自然日交易情景"
        plan = mapping(scenario.get("plan"))
        days = plan.get("calendar_days")
        window = f"未来{days}天" if days is not None else "天数未记录"
        branches = mapping(mapping(scenario.get("decision")).get("branches"))
        labels = {"flat": "新买", "held": "持有", "addition": "追加"}
        decisions = "；".join(
            f"{labels.get(key, key)}：{mapping(branch).get('label') or '未保存判断标签'}"
            for key, branch in branches.items())
        return f"自然日计划 · {window} · 截止{plan.get('deadline') or '未记录'} · {decisions or '未保存判断分支'}"
    if _is_standard_trade(payload):
        scenario = mapping(payload.get("trade_scenario"))
        if not scenario:
            return "未保存标准交易情景"
        branches = mapping(mapping(scenario.get("decision")).get("branches"))
        return "标准交易 · " + "；".join(
            f"{name}：{mapping(branches.get(key)).get('label') or '未保存判断标签'}"
            for key, name in (("flat", "新买"), ("held", "持有")))
    decision = payload.get("investor_decision") if payload.get("schema_version", 0) >= 4 else payload.get("decision")
    branches = mapping(mapping(decision).get("branches"))
    state = payload.get("holding_status")
    if state in {"held", "flat"}:
        return str(mapping(branches.get(state)).get("label") or "未保存判断标签")
    return "；".join(f"{name}：{mapping(branches.get(key)).get('label', '未记录')}"
                    for key, name in (("held", "若已持有"), ("flat", "若准备买入")))


def _amount_scenario(label: str, value: Mapping, *, new_buy: bool = False) -> None:
    if not value:
        return
    st.markdown(f"**{label}**")
    currency = value.get("currency")
    if value.get("basis_value") is not None:
        _write(f"{'当时拟投入金额' if new_buy else '当时的净值起点金额'}：{_money(value.get('basis_value'), currency)}")
    if not new_buy:
        st.caption(markdown_text(f"原基准净值日期 {value.get('basis_date') or '未记录'} · 原单位净值 {value.get('basis_nav') or '未记录'}；假设份额不变。"))
    rows = []
    for key, name in (("expected", "预期情景"), ("lower", "预测范围下界"), ("upper", "预测范围上界")):
        point = mapping(value.get(key))
        if point:
            rows.append({"情景": name, "账面金额": _money(point.get("value"), currency),
                         "相对原起点变动": _money(point.get("change"), currency),
                         "相对当时成本盈亏": _money(point.get("profit"), currency)})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    for reason in lines(value.get("reasons")):
        st.caption(markdown_text(reason))
    st.caption("这是原快照的账面情景，不是实际到账金额，也不是从今天起的保证收益。")


def _trade_decision_changes(current: Mapping, prior: Mapping, *, label: str) -> bool:
    if current == prior:
        return False
    changed_branch = False
    for key, name in (("flat", "新买"), ("held", "持有"), ("addition", "追加")):
        before = mapping(mapping(prior.get("branches")).get(key))
        after = mapping(mapping(current.get("branches")).get(key))
        if before == after:
            continue
        changed_branch = True
        if before.get("action") != after.get("action") or before.get("label") != after.get("label"):
            _write(f"{label} · {name}：{before.get('label') or '未记录'} → {after.get('label') or '未记录'}。")
        else:
            _write(f"{label} · {name}标签未变，保存的理由或比较数值有变化。")
        if before.get("action") != after.get("action") and before.get("label") == after.get("label"):
            _write(f"原保存动作：{before.get('action') or '未记录'} → {after.get('action') or '未记录'}。")
        for reason in [*lines(after.get("reasons")), *lines(after.get("constraint_reasons"))]:
            if reason not in [*lines(before.get("reasons")), *lines(before.get("constraint_reasons"))]:
                _write("本次保存的理由：" + reason)
        if after.get("research_reason") and after.get("research_reason") != before.get("research_reason"):
            _write("本次保存的研究理由：" + after["research_reason"])
    if not changed_branch:
        _write(f"{label}的原预测数值、费用比较或延迟情景有变化；没有因此生成新的买卖理由。")
    return True


def _standard_trade_changes(current: Mapping, prior: Mapping) -> None:
    current_trade, prior_trade = _is_standard_trade(current), _is_standard_trade(prior)
    scenario = mapping(current.get("trade_scenario")) if current_trade else {}
    old_scenario = mapping(prior.get("trade_scenario")) if prior_trade else {}
    changed = False
    if not current_trade or not prior_trade:
        changed = True
        _write(f"保存口径：schema{prior.get('schema_version')} → schema{current.get('schema_version')}。")
        st.caption("旧版个人判断没有保存标准交易情景，不能比较其假设买入日、周期、标准费用或交易动作；"
                   "缺失不代表默认日期、20笔净值、零费用或保持现金。")
    elif not scenario or not old_scenario:
        changed = True
        st.caption("至少一份快照没有保存标准交易情景，不能重建缺失的假设或动作。")
    else:
        for key, label, suffix in (("horizon", "观察期限", "笔净值"),
                                   ("entry_lag", "假定执行延迟", "笔净值"),
                                   ("base_amount", "标准单笔示例本金", "金额单位")):
            if old_scenario.get(key) != scenario.get(key):
                changed = True
                before, after = old_scenario.get(key), scenario.get(key)
                _write(f"{label}：{str(before) + suffix if before is not None else '未记录'} → "
                       f"{str(after) + suffix if after is not None else '未记录'}。")
        old_purchase, purchase = mapping(old_scenario.get("purchase")), mapping(scenario.get("purchase"))
        for key, label in (("date", "假设买入净值日"), ("latest_date", "当时最新已知净值日")):
            if old_purchase.get(key) != purchase.get(key):
                changed = True
                _write(f"{label}：{old_purchase.get(key) or '未记录'} → {purchase.get(key) or '未记录'}。")
        if old_purchase != purchase:
            changed = True
            st.caption("买入以来已发生区间的原保存值有变化；这是过去净值变化，不拼接成未来回报。")
        old_fees, fees = mapping(old_scenario.get("fees")), mapping(scenario.get("fees"))
        for key, label in (("subscription_fee", "标准假设申购费"), ("redemption_fee", "标准假设赎回费")):
            if old_fees.get(key) != fees.get(key):
                changed = True
                _write(f"{label}：{percent(old_fees.get(key), digits=2)} → {percent(fees.get(key), digits=2)}。")
        if old_fees != fees:
            changed = True
            st.caption("标准费用的费率、来源或持有日依据有变化；各次原依据保留在详情，不据此推定历史渠道费率。")
        changed = _trade_decision_changes(mapping(scenario.get("decision")),
                                          mapping(old_scenario.get("decision")), label="标准交易判断") or changed
    investor, old_investor = mapping(current.get("investor_decision")), mapping(prior.get("investor_decision"))
    if investor and old_investor:
        changed = _trade_decision_changes(investor, old_investor, label="基金研究判断") or changed
    thesis, old_thesis = mapping(current.get("research_thesis")), mapping(prior.get("research_thesis"))
    if thesis != old_thesis:
        changed = True
        new_payload, old_payload = mapping(thesis.get("payload")), mapping(old_thesis.get("payload"))
        _write(f"采用的研究版本或保存内容有变化：研究日 {old_payload.get('research_date') or '未记录'} → "
               f"{new_payload.get('research_date') or '未记录'}；记录 {old_thesis.get('record_id') or '未记录'} → "
               f"{thesis.get('record_id') or '未记录'}。")
        explanation = mapping(new_payload.get("change_evaluation")).get("summary")
        if (explanation and old_thesis.get("record_id") is not None
                and mapping(new_payload.get("revision")).get("supersedes") == old_thesis["record_id"]):
            _write("本次研究对上一版的解释：" + explanation)
        else:
            st.caption("两次快照没有可对应的显式接续说明，不把版本不同自行解释成原理由已失效。")
    if current.get("online_context") != prior.get("online_context"):
        changed = True
        _write("两次保存采用的在线依据有变化；只回放各次已保存内容，不联网判断消息是否已经失效。")
    if current_trade and prior_trade and current.get("personal_conversion") != prior.get("personal_conversion"):
        changed = True
        _write("可选个人金额换算有变化；它与标准单笔本金和市场预测分开保存。")
    if not changed:
        st.caption("这两次保存的标准交易假设、判断、研究版本与在线依据没有上述可见变化；没有生成新的变更理由。")


def render_assessment_changes(changes: list[dict]) -> None:
    """Render already extracted changes; this does not read or compare snapshots."""
    for change in changes:
        if change["level"] == "caption":
            st.caption(markdown_text(change["text"]))
        else:
            _write(change["text"])


def _calendar_trade_changes(current: Mapping, prior: Mapping) -> None:
    changes = extract_assessment_changes(current, prior)
    render_assessment_changes(changes)
    if not changes:
        st.caption("这两次保存的自然日计划、分批费用、净额判断、研究版本与在线依据没有上述可见变化；没有生成新的变更理由。")


def _buyer_decision_changes(current: Mapping, prior: Mapping) -> None:
    changes = extract_assessment_changes(current, prior)
    render_assessment_changes(changes)
    if not changes:
        st.caption("这两次保存的首选、备选取舍、观察条件、自然日假设与依据没有上述可见变化；没有生成新的变更理由。")


def _snapshot_changes(envelope: Mapping, previous: Mapping | None) -> None:
    st.markdown("**较上次保存，变了什么**")
    if previous is None:
        st.caption("没有更早的保存判断可对照；不拿今天的资料补造上次结论。")
        return
    current = mapping(envelope.get("assessment"))
    prior = mapping(previous.get("assessment"))
    if not current or not prior:
        st.caption("两次保存的完整内容尚不可读，不能重建比较。")
        return
    if _has_buyer_projection(current):
        from rmbfund.ui.trade import _render_current_changes

        _render_current_changes(current, previous, current.get("selected_branch"))
        return
    st.caption(f"只比较保存于 {when(envelope.get('saved_at'))} 和 {when(previous.get('saved_at'))} 的两份原快照；不联网、不重算。")
    if _is_buyer_decision(current) or _is_buyer_decision(prior):
        _buyer_decision_changes(current, prior)
        return
    if _is_calendar_trade(current) or _is_calendar_trade(prior):
        _calendar_trade_changes(current, prior)
        return
    if _is_standard_trade(current) or _is_standard_trade(prior):
        _standard_trade_changes(current, prior)
        return
    changed = False
    current_decision = mapping(current.get("investor_decision") or current.get("decision"))
    prior_decision = mapping(prior.get("investor_decision") or prior.get("decision"))
    for key, label in (("flat", "新买"), ("held", "持有"), ("addition", "追加")):
        before = mapping(mapping(prior_decision.get("branches")).get(key))
        after = mapping(mapping(current_decision.get("branches")).get(key))
        if before.get("action") != after.get("action") or before.get("label") != after.get("label"):
            changed = True
            _write(f"{label}：{before.get('label') or '未记录'} → {after.get('label') or '未记录'}。")
            for reason in lines(after.get("reasons")):
                _write("本次保存的理由：" + reason)
        elif before.get("reasons") != after.get("reasons"):
            changed = True
            _write(f"{label}标签未变，保存的理由有变化。")
            for reason in lines(after.get("reasons")):
                if reason not in lines(before.get("reasons")):
                    _write("本次新增理由：" + reason)
    thesis = mapping(current.get("research_thesis"))
    old_thesis = mapping(prior.get("research_thesis"))
    if thesis.get("record_id") != old_thesis.get("record_id"):
        changed = True
        new_payload, old_payload = mapping(thesis.get("payload")), mapping(old_thesis.get("payload"))
        _write(f"采用的研究版本已变化：研究日 {old_payload.get('research_date') or '未记录'} → {new_payload.get('research_date') or '未记录'}。")
        explanation = mapping(new_payload.get("change_evaluation")).get("summary")
        if explanation and mapping(new_payload.get("revision")).get("supersedes") == old_thesis.get("record_id"):
            _write("本次研究对上一版的解释：" + explanation)
        else:
            st.caption("两次快照没有可对应的显式接续说明，不把版本不同自行解释成原理由已失效。")
    before_inputs, after_inputs = mapping(prior.get("inputs")), mapping(current.get("inputs"))
    for key, label in (("subscription_fee", "填写的申购费"), ("redemption_fee", "填写的赎回费")):
        if before_inputs.get(key) != after_inputs.get(key):
            changed = True
            _write(f"{label}：{percent(before_inputs.get(key), digits=2)} → {percent(after_inputs.get(key), digits=2)}。")
    for key, label in (("holding", "份额或成本"), ("budget", "拟投入金额")):
        if before_inputs.get(key) != after_inputs.get(key):
            changed = True
            _write(f"{label}填写有变化；这改变个人金额解释，不代表基金本身发生了同样变化。")
    old_conditions = mapping(mapping(prior.get("sizing_plan")).get("user_conditions"))
    conditions = mapping(mapping(current.get("sizing_plan")).get("user_conditions"))
    if old_conditions != conditions:
        changed = True
        _write("保存的资金用途或金额条件有变化，具体值以各次原快照为准。")
    before_value = mapping(mapping(prior.get("amounts")).get("current"))
    after_value = mapping(mapping(current.get("amounts")).get("current"))
    if before_value != after_value:
        changed = True
        _write(f"保存时估值：{before_value.get('nav_date') or '日期未记'} 的 {_money(before_value.get('market_value'), before_value.get('currency'))} → "
               f"{after_value.get('nav_date') or '日期未记'} 的 {_money(after_value.get('market_value'), after_value.get('currency'))}。不是今天重算的盈亏。")
    if not changed:
        st.caption("这两次保存的研究版本、判断分支、费用与资金条件没有上述可见变化；没有生成新的变更理由。")


def _standard_trade_detail(envelope: Mapping, payload: Mapping) -> None:
    from rmbfund.ui.buyer import render_saved_buyer, render_sizing
    from rmbfund.ui.trade import render_connection, render_reconstructed, render_scenario

    scenario = mapping(payload.get("trade_scenario"))
    forecast, calculation = mapping(payload.get("forecast")), mapping(payload.get("calculation"))
    with st.container(border=True, key="history_assessment_detail"):
        st.markdown("**当时保存的标准交易假设与判断**")
        _write(f"保存于 {when(envelope.get('saved_at'))}；不是买卖成交或公开预测发行时间。")
        if calculation.get("kind") == "local_hypothetical":
            st.caption("来源：当时的本地假设测算，未实际发行；保存判断不把它变成真正发行后的预测成绩。")
        elif calculation.get("actual_issue") is True:
            st.caption("来源：原快照明确记录采用实际发行预测；发行依据与本次保存时间分开，未重新发行或核对结果。")
        else:
            st.caption("原快照未明确记录实际发行来源；不因为保存了预测标识或计算时间就认定已实际发行。")
        st.caption(markdown_text(f"原计算时间 {when(calculation.get('generated_at') or forecast.get('generated_at'))} · "
                                 f"原净值起点 {mapping(forecast.get('anchor')).get('date') or '未记录'} · "
                                 f"原方法 {forecast.get('model_name') or forecast.get('model_id') or '未记录'} · "
                                 f"版本 {_short(forecast.get('model_hash'))}"))
        st.caption("以下为统一示例本金的动作比较，不把保存时的个人持仓当作标准交易，也不按今天的费用或净值重算。")
        if scenario:
            purchase, fees = mapping(scenario.get("purchase")), mapping(scenario.get("fees"))
            basis = mapping(fees.get("basis"))
            st.caption(markdown_text(f"原假设买入净值日 {purchase.get('date') or '未记录'} · "
                                     f"原买入净值 {purchase.get('nav') if purchase.get('nav') is not None else '未记录'} · "
                                     f"当时最新已知净值日 {purchase.get('latest_date') or '未记录'} · "
                                     f"原观察期限 {scenario.get('horizon') if scenario.get('horizon') is not None else '未记录'}笔净值 · "
                                     f"原执行延迟 {scenario.get('entry_lag') if scenario.get('entry_lag') is not None else '未记录'}笔净值"))
            _write(f"原标准费用假设：申购费 {percent(fees.get('subscription_fee'), digits=2)} · "
                   f"赎回费 {percent(fees.get('redemption_fee'), digits=2)}。")
            for key, title in (("subscription", "申购费来源"), ("redemption", "赎回费来源")):
                source = mapping(basis.get(key))
                if source.get("label"):
                    _write(f"{title}：{source['label']}")
            if basis:
                st.caption(markdown_text(f"原费用估值日 {basis.get('valuation_date') or '未记录'} · "
                                         f"原模拟持有天数 {basis.get('holding_days') if basis.get('holding_days') is not None else '未记录'} · "
                                         f"{basis.get('holding_day_label') or '持有日口径未记录'}"))
            render_scenario(scenario)
            with st.expander("当时的完整交易假设、费用与计算来源"):
                st.json({"trade_scenario": dict(scenario), "calculation": dict(calculation),
                         "forecast": dict(forecast)}, expanded=False)
        else:
            st.info("这份快照没有保存标准交易情景，不从兼容旧版的个人判断字段重建。")
        reconstructed = payload.get("reconstructed_history")
        if isinstance(reconstructed, Mapping):
            st.caption("以下历史回算只回放保存时的对象；不是当年真实发行，不使用今天新增净值补齐。")
            render_reconstructed(reconstructed)
        else:
            st.caption("当时未保存历史回算结果；浏览历史不会补跑。")
        st.markdown("**当时保存的基金研究与在线依据**")
        render_connection(mapping(payload.get("trade_research_connection")))
        render_saved_buyer(payload)
        render_saved_investment_analysis(mapping(payload.get("investment_analysis")))
        personal = mapping(payload.get("personal_conversion"))
        if personal:
            with st.expander("当时可选的个人金额换算"):
                st.caption("这是当时单独保存的个人输入与账面换算，不改变上方标准交易；没有用今天的份额、成本或费用重算。")
                inputs, amounts = mapping(personal.get("inputs")), mapping(personal.get("amounts"))
                current = mapping(amounts.get("current"))
                if current:
                    _write(f"原公布净值对应市值 {_money(current.get('market_value'), current.get('currency'))} · "
                           f"原账面盈亏 {_money(current.get('profit'), current.get('currency'))} · "
                           f"净值日 {current.get('nav_date') or '未记录'}。")
                _amount_scenario("当时个人持仓金额情景", mapping(amounts.get("anchor")))
                _amount_scenario("当时个人新买金额情景", mapping(amounts.get("new_buy")), new_buy=True)
                render_sizing(mapping(personal.get("sizing_plan")))
                st.json({"inputs": dict(inputs), "amounts": dict(amounts)}, expanded=False)
        st.caption("以上只呈现原快照；没有重新拟合、联网、发行、记录结果或提交交易。")


def _calendar_trade_detail(envelope: Mapping, payload: Mapping, *, buyer_decision: bool = False) -> None:
    from rmbfund.ui.buyer import render_saved_buyer, render_sizing
    from rmbfund.ui.trade import render_connection, render_reconstructed, render_scenario

    scenario = mapping(payload.get("trade_scenario"))
    forecast, calculation = mapping(payload.get("forecast")), mapping(payload.get("calculation"))
    with st.container(border=True, key="history_assessment_detail"):
        st.markdown("**当时保存的自然日计划与卖出净额判断**")
        _write(f"保存于 {when(envelope.get('saved_at'))}；不是买卖成交或公开预测发行时间。")
        if calculation.get("kind") == "local_hypothetical":
            st.caption("来源：当时的本地假设测算，未实际发行；保存判断不把它变成真正发行后的预测成绩。")
        elif calculation.get("actual_issue") is True:
            st.caption("来源：原快照明确记录采用实际发行预测；发行依据与本次保存时间分开，未重新发行或核对结果。")
        else:
            st.caption("原快照未明确记录实际发行来源；不因为保存了预测标识或计算时间就认定已实际发行。")
        st.caption(markdown_text(f"原计算时间 {when(calculation.get('generated_at') or forecast.get('generated_at'))} · "
                                 f"原净值起点 {mapping(forecast.get('anchor')).get('date') or '未记录'} · "
                                 f"原方法 {forecast.get('model_name') or forecast.get('model_id') or '未记录'} · "
                                 f"版本 {_short(forecast.get('model_hash'))}"))
        if scenario:
            plan, fees = mapping(scenario.get("plan")), mapping(scenario.get("fees"))
            days = plan.get("calendar_days")
            _write(f"原自然日窗口 {str(days) + '天' if days is not None else '未记录'} · "
                   f"原截止日 {plan.get('deadline') or '未记录'}。")
            st.caption(markdown_text(f"原测算时刻 {when(plan.get('decision_at'))} · "
                                     f"原测算日期 {plan.get('decision_date') or '未记录'} · "
                                     f"原本次申请日 {plan.get('application_date') or '未记录'} · "
                                     f"原计划卖出申请日 {plan.get('sale_date') or '未记录'}"))
            st.caption(markdown_text(f"原日程预计 {plan.get('horizon') if plan.get('horizon') is not None else '未记录'}笔净值 · "
                                     f"原日程净值延迟 {plan.get('entry_lag') if plan.get('entry_lag') is not None else '未记录'}笔 · "
                                     f"原新份额确认日 {plan.get('new_confirmation_date') or '未记录'} · "
                                     f"原日程依据 {plan.get('calendar_basis') or '未记录'}"))
            st.caption("自然日窗口经原保存日程映射为净值笔数；概率只针对对应净值笔数，不是已经验证的精确自然日择时概率。"
                       "计划卖出后的净额包含赎回应收款，不表示截止日资金已经到账，也不是窗口内最佳买卖日。")
            for reason in [*lines(plan.get("assumptions")), *lines(plan.get("reasons"))]:
                _write(reason)
            purchase = mapping(scenario.get("purchase"))
            st.caption(markdown_text(f"原假设买入净值日 {purchase.get('date') or '未记录'} · "
                                     f"原买入净值 {purchase.get('nav') if purchase.get('nav') is not None else '未记录'} · "
                                     f"当时最新已知净值日 {purchase.get('latest_date') or '未记录'}；历史净值日不冒充真实确认日。"))
            _write(f"原分批费用假设：申购费 {percent(fees.get('subscription_fee'), digits=2)} · "
                   f"旧份额当前赎回费 {percent(fees.get('current_redemption_fee'), digits=2)} · "
                   f"旧份额计划卖出赎回费 {percent(fees.get('future_old_redemption_fee'), digits=2)} · "
                   f"新份额计划卖出赎回费 {percent(fees.get('future_new_redemption_fee'), digits=2)}。")
            st.caption(markdown_text(f"原旧份额模拟确认日起点 {fees.get('old_confirmation_date') or '未记录'} · "
                                     f"原新份额模拟确认日起点 {fees.get('new_confirmation_date') or '未记录'}。"))
            contexts = mapping(fees.get("contexts"))
            for key, title in (("current", "旧份额当前费用依据"),
                               ("future_old", "旧份额计划卖出费用依据"),
                               ("future_new", "新份额计划卖出费用依据")):
                context = mapping(contexts.get(key))
                if not context:
                    st.caption(f"{title}未保存；不补默认费率、确认日或持有天数。")
                    continue
                source = mapping(context.get("redemption"))
                _write(f"{title}：{source.get('label') or '来源标签未记录'}")
                if key == "current" and mapping(context.get("subscription")).get("label"):
                    _write("申购费来源：" + mapping(context.get("subscription"))["label"])
                st.caption(markdown_text(f"原费用估值日 {context.get('valuation_date') or '未记录'} · "
                                         f"原模拟持有天数 {context.get('holding_days') if context.get('holding_days') is not None else '未记录'} · "
                                         f"{context.get('holding_day_label') or '持有日口径未记录'}"))
                for reason in [*lines(context.get("assumptions")), *lines(context.get("reasons"))]:
                    st.caption(markdown_text(reason))
            for reason in [*lines(fees.get("assumptions")), *lines(fees.get("reasons"))]:
                st.caption(markdown_text(reason))
            st.caption(markdown_text(f"原金额比较口径 {scenario.get('end_value_basis') or '未记录'}；以下直接回放保存的动作与数值。"))
            render_scenario(scenario)
        else:
            st.info("这份快照没有保存自然日交易情景，不从兼容旧版字段或今天日期重建。")
        reconstructed = payload.get("reconstructed_history")
        if isinstance(reconstructed, Mapping):
            st.caption("以下历史回算只回放保存时的对象；检验对应净值笔数，不是严格自然日择时策略。"
                       "采用保存时统一费用情景，不冒充当年真实批次费龄或费表，也不使用今天新增净值补齐。")
            render_reconstructed(reconstructed)
        else:
            st.caption("当时未保存历史回算结果；浏览历史不会补跑。")
        st.markdown("**当时保存的基金研究与在线依据**")
        render_connection(mapping(payload.get("trade_research_connection")))
        if buyer_decision:
            from rmbfund.ui.buyer import render_comparison, render_online, render_portfolio, render_research

            render_research(payload)
            render_comparison(mapping(payload.get("fund_comparison")))
            render_online(mapping(payload.get("online_context")), saved=True,
                          product_id=mapping(payload.get("subject")).get("id"))
            render_portfolio(mapping(payload.get("portfolio_context")))
            render_sizing(mapping(payload.get("sizing_plan")))
        else:
            render_saved_buyer(payload)
        render_saved_investment_analysis(mapping(payload.get("investment_analysis")))
        personal = mapping(payload.get("personal_conversion"))
        if personal:
            with st.expander("当时可选的个人金额换算"):
                st.caption("个人输入、计划净额情景与金额解释均按原快照回放；没有用今天的份额、费用或日期重算。")
                amounts = mapping(personal.get("amounts"))
                current = mapping(amounts.get("current"))
                if current:
                    _write(f"原公布净值对应市值 {_money(current.get('market_value'), current.get('currency'))} · "
                           f"原账面盈亏 {_money(current.get('profit'), current.get('currency'))} · "
                           f"净值日 {current.get('nav_date') or '未记录'}。")
                personal_scenario = personal.get("trade_scenario")
                if isinstance(personal_scenario, Mapping):
                    render_scenario(personal_scenario)
                render_sizing(mapping(personal.get("sizing_plan")))
                st.json(personal, expanded=False)
        with st.expander("当时的完整计划、费用、准确预测头与保存快照"):
            st.json(payload, expanded=False)
        st.caption("以上只呈现原快照；没有读取今天时钟来改日期，也没有重新拟合、联网、发行、记录结果或提交交易。")


def _buyer_decision_detail(envelope: Mapping, payload: Mapping) -> None:
    from rmbfund.ui.buyer import render_decision_brief
    from rmbfund.ui.trade import render_sale_comparison

    st.markdown("**当时保存的基金买卖首选**")
    brief = payload.get("decision_brief")
    if isinstance(brief, Mapping) and brief:
        render_decision_brief(brief, replay=True)
    else:
        st.info("这份快照未保存基金买卖首选摘要；不从旧交易分支、作者说明或今天资料重建。")
    comparison = payload.get("sale_comparison")
    if isinstance(comparison, Mapping):
        render_sale_comparison(comparison)
    _calendar_trade_detail(envelope, payload, buyer_decision=True)


def _projected_buyer_detail(envelope: Mapping, payload: Mapping, *, theme: str) -> None:
    """Replay the actual saved presentation, not a new projection of old evidence."""
    from rmbfund.ui.buyer_view import render_buyer_view, render_forecast_chart

    st.markdown("**当时保存的操作建议**")
    st.caption(f"保存于 {when(envelope.get('saved_at'))}；以下为当时展示的建议与图。")
    branch = payload.get("selected_branch")
    context = {"flat": "还没买", "held": "已经持有"}.get(branch)
    if context:
        st.caption("当时选择：" + context)
    view = payload.get("buyer_view")
    if isinstance(view, Mapping) and view:
        render_buyer_view(view, key="history_buyer_view")
    else:
        st.info("这份快照没有保存当时展示的建议，不从其他分支或今天资料重建。")
    chart = payload.get("forecast_chart")
    if isinstance(chart, Mapping) and chart:
        render_forecast_chart(chart, key="history_forecast_chart", theme=theme)
    else:
        st.info("这份快照没有保存图表，不用今天净值补画。")
    st.caption("只回放保存内容；不重新计算，也不联网更新。保存建议不代表已经成交。")


def _snapshot_detail(envelope: Mapping, *, theme: str = "light") -> None:
    payload = mapping(envelope.get("assessment"))
    if not payload:
        st.info("这条记录没有可读取的个人判断快照，不用今天的输入重建。")
        return
    if _has_buyer_projection(payload):
        _projected_buyer_detail(envelope, payload, theme=theme)
        return
    if _is_buyer_decision(payload):
        _buyer_decision_detail(envelope, payload)
        return
    if _is_calendar_trade(payload):
        _calendar_trade_detail(envelope, payload)
        return
    if _is_standard_trade(payload):
        _standard_trade_detail(envelope, payload)
        return
    forecast, decision = mapping(payload.get("forecast")), mapping(payload.get("decision"))
    with st.container(border=True, key="history_assessment_detail"):
        st.markdown("**当时保存的判断**")
        _write(f"保存于 {when(envelope.get('saved_at'))}")
        if forecast:
            st.caption(markdown_text(f"原预测发出于 {when(forecast.get('generated_at'))} · "
                                     f"看之后{payload.get('horizon') or '未记录'}笔净值 · "
                                     f"原主要成交延迟 {payload.get('entry_lag') if payload.get('entry_lag') is not None else '未记录'}笔净值"))
        else:
            st.caption("当时没有可用的短期预测；以下回放原研究、组合和金额条件，没有补算概率。")
        state = {"held": "已记录持有", "flat": "明确记录未持有", "unknown": "持有情况未知"}.get(payload.get("holding_status"), "未记录")
        st.caption(f"保存时的持有情况：{state}。以下保留当时对应的判断分支。")
        if payload.get("schema_version", 0) >= 4:
            from rmbfund.ui.buyer import render_saved_buyer
            render_saved_buyer(payload)
            st.markdown("**当时的短期费用比较（不是完整基金建议）**")
        branches = mapping(decision.get("branches"))
        for key, name in (("held", "若已持有"), ("flat", "若准备买入")):
            branch = mapping(branches.get(key))
            if branch and payload.get("holding_status") in (key, "unknown", None):
                _write(f"{name}：{branch.get('label', '未保存判断标签')}")
                for reason in lines(branch.get("reasons")):
                    _write(reason)
        sensitive = decision.get("delay_sensitive")
        if sensitive is True or (isinstance(sensitive, Mapping) and any(value is True for value in sensitive.values())):
            st.warning("当时的判断会随成交延迟改变，不能只看主要情景。下面保留了当时的各个延迟比较。")
        inputs = mapping(payload.get("inputs"))
        holding = mapping(inputs.get("holding"))
        current = mapping(mapping(payload.get("amounts")).get("current"))
        if current:
            _write(f"当时已公布净值对应的市值：{_money(current.get('market_value'), current.get('currency'))}；"
                   f"账面盈亏：{_money(current.get('profit'), current.get('currency'))}")
            st.caption(markdown_text(f"这项估值使用当时公布净值日期 {current.get('nav_date') or '未记录'}，不是预测情景的成交价。"))
        amounts = mapping(payload.get("amounts"))
        _amount_scenario("保存时的持仓金额情景", mapping(amounts.get("anchor")))
        _amount_scenario("保存时的新买金额情景", mapping(amounts.get("new_buy")), new_buy=True)
        render_saved_investment_analysis(mapping(payload.get("investment_analysis")))
        with st.expander("当时的输入、延迟比较与预测依据"):
            budget = mapping(inputs.get("budget"))
            st.dataframe([
                {"当时输入": "份额", "保存值": str(holding['shares']) if holding.get('shares') is not None else "未填写"},
                {"当时输入": "持仓总成本", "保存值": _money(holding.get("cost"), holding.get("currency"))},
                {"当时输入": "申购费率", "保存值": percent(inputs.get("subscription_fee"), digits=2)},
                {"当时输入": "赎回费率", "保存值": percent(inputs.get("redemption_fee"), digits=2)},
                {"当时输入": "拟投入金额", "保存值": _money(budget.get("amount"), budget.get("currency"))},
            ], hide_index=True, width="stretch")
            scenarios = decision.get("scenarios") or []
            if scenarios:
                st.dataframe([{"延迟笔数": row.get("entry_lag"), "期限笔数": row.get("horizon"),
                               "新买相对保留现金": percent(row.get("buy_advantage_vs_cash"), signed=True, digits=2),
                               "持有相对现在退出": percent(row.get("hold_advantage_vs_exit"), signed=True, digits=2)}
                              for row in scenarios], hide_index=True, width="stretch")
            for key, label in (("execution", "原行动预测依据"), ("anchor", "原净值起点预测依据")):
                view = mapping(payload.get(key))
                head = mapping(view.get("head"))
                if head:
                    st.markdown(f"**{label}**")
                    _write(f"上涨概率 {percent(head.get('up_probability'))} · 预期涨跌 {percent(head.get('expected_change'), signed=True)}")
                for reason in lines(view.get("reasons")):
                    _write(reason)
            evidence = mapping(payload.get("evidence_summary"))
            if evidence:
                st.markdown("**保存时已知的真实结果概况**")
                counts = mapping(evidence.get("counts"))
                metrics = mapping(evidence.get("forward_metrics"))
                forward = metrics.get("count", counts.get("forward_windows"))
                if forward is None:
                    st.caption("当时没有可读取的真实结果汇总；不会拿今天新增的结果补写。")
                else:
                    scope = "个方法与行情区间组合" if evidence.get("metrics_scope") == "models_only" else "个不重复区间"
                    st.caption(f"当时已有{_count(forward)}{scope}的完整未来结果，"
                               f"{_count(counts.get('unrecorded'))}条发行记录尚未记录结果；这是保存时的证据，不是今天的成绩。")
                _paired_table(mapping(evidence.get("paired")),
                              model_label=model_name(forecast.get("model_id")) or "当时的预测方法")
        st.caption("以上只呈现原快照。今天修改份额、成本或费用不会改写旧内容；保存判断不是交易，也不是一次公开预测发行。")


def _assessments(assessments: Callable | None, detail: Callable | None, subject: Mapping | None, *,
                 action_notes: Callable | None = None, save_action_note: Callable | None = None,
                 replay_assessment: Callable | None = None, theme: str = "light") -> None:
    st.divider()
    st.subheader("我保存的判断")
    st.caption("回看当时保存的建议和依据；不按今天的资料改写旧结论。")
    if assessments is None or subject is None:
        st.info("这只基金还没有可查看的保存判断。保存判断不代表已买入或已卖出。")
        return
    page = st.session_state.get("history_assessment_page", 0)
    try:
        rows = assessments(subject, limit=21, offset=page * 20)
    except Exception:
        st.error("暂时读不到已保存的个人判断；没有重建或覆盖快照。")
        return
    if not rows:
        st.info("这只基金还没有保存过个人判断。不会根据当前份额或费用猜测当时的建议。")
        return
    shown = rows[:20]
    time_column = "原计算或发行时间" if any(
        _is_buyer_decision(mapping(row.get("assessment"))) or _is_calendar_trade(mapping(row.get("assessment")))
        or _is_standard_trade(mapping(row.get("assessment"))) for row in shown) else "原预测发出时间"
    table = []
    for row in shown:
        payload = mapping(row.get("assessment"))
        forecast, calculation = mapping(payload.get("forecast")), mapping(payload.get("calculation"))
        timestamp = when(forecast.get("generated_at"))
        if _is_buyer_decision(payload) or _is_calendar_trade(payload) or _is_standard_trade(payload):
            timestamp = when(calculation.get("generated_at") or forecast.get("generated_at"))
            source = ("本地测算，未实际发行" if calculation.get("kind") == "local_hypothetical" else
                      "实际发行" if calculation.get("actual_issue") is True else "发行来源未注明")
            timestamp = f"{source} · {timestamp}"
        table.append({"保存时间": when(row.get("saved_at")), "当时判断": _snapshot_label(payload), time_column: timestamp})
    st.dataframe(table, hide_index=True, width="stretch")
    _pagination(page, len(rows) > 20, prefix="history_assessment")
    records = {row["id"]: row for row in shown}
    if st.session_state.get("history_assessment_selected") not in records:
        st.session_state["history_assessment_selected"] = next(iter(records))
    choices = list(records)
    selected = st.selectbox("查看某次保存的判断", choices, key="history_assessment_selected",
                            format_func=lambda key: f"{when(records[key].get('saved_at'))} · 第{choices.index(key) + 1}条")
    try:
        saved = detail(selected) if detail else records[selected]
    except Exception:
        st.error("这条保存判断暂时打不开；没有用今天的输入重建。")
        return
    if saved is None:
        st.info("这条保存判断暂时不存在，没有用今天的输入重建。")
        return
    index = next(index for index, row in enumerate(rows) if row["id"] == selected)
    previous = rows[index + 1] if index + 1 < len(rows) else None
    if previous is not None and detail is not None:
        try:
            previous = detail(previous["id"])
        except Exception:
            previous = None
    _snapshot_changes(saved, previous)
    _snapshot_detail(saved, theme=theme)
    if (_is_buyer_decision(mapping(saved.get("assessment")))
            and any(callback is not None for callback in (action_notes, save_action_note, replay_assessment))):
        from rmbfund.ui.assessment_notes import render_assessment_notes

        render_assessment_notes(saved, read=action_notes, save=save_action_note, replay=replay_assessment)


def render_history(
    product_id: str | None, *, read: Callable[..., list[dict]] | None,
    detail: Callable[[str], dict] | None = None, models: list[dict] | None = None,
    summary: Callable[..., Mapping] | None = None, assessments: Callable | None = None,
    assessment_detail: Callable | None = None, subject: Mapping | None = None,
    action_notes: Callable | None = None, save_action_note: Callable | None = None,
    replay_assessment: Callable | None = None,
    theme: str = "light",
) -> None:
    """Callbacks read public history and optional subject-scoped private snapshots."""
    scope = (product_id, tuple(str(mapping(subject).get(key, "")) for key in ("id", "symbol", "kind", "share_class", "exchange")))
    if st.session_state.get("history_product") != scope:
        for key in list(st.session_state):
            if key.startswith("history_"):
                st.session_state.pop(key, None)
        st.session_state["history_product"] = scope
    _assessments(assessments, assessment_detail, subject, action_notes=action_notes,
                 save_action_note=save_action_note, replay_assessment=replay_assessment, theme=theme)
    with st.expander("量化预测记录（展开查看原始预测与实际结果）", expanded=False):
        st.caption("先看你明确保存的个人判断；下面量化记录只回放当时封存的预测，不用今天资料重建旧结论。")
        _public_history(product_id, read, detail, models, summary)

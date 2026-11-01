"""Optional private annotations and explicit stored-NAV simulation review."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import streamlit as st

from rmbfund.domain import now_iso
from rmbfund.ui.formatting import mapping, markdown_text, percent, when

_KINDS = {"simulated_choice": "模拟采用", "plan": "行动计划", "user_report": "我自述的实际行动", "observation": "继续观察"}
_ACTIONS = {"buy": "新买一笔", "wait": "先留现金", "hold": "持有不追加", "add": "保留并追加一笔",
            "reduce50": "卖一半（模拟金额示例）", "exit": "退出", "reduce": "减少，比例未定",
            "redeem": "赎回退出", "no_action": "暂不行动"}


def _write(value):
    st.markdown(markdown_text(value))


def _money(value):
    return f"{value:,.2f}" if value is not None else "未算出"


def render_replay_result(result: Mapping) -> None:
    status = result.get("status")
    labels = {"mature": "原模拟窗口已有后续净值，可以核对", "immature": "原模拟窗口尚未走完",
              "invalid": "这批净值不能与原模拟直接核对", "unavailable": "还缺原模拟所需的依据",
              "note_only": "保留这条笔记，不计算个人真实盈亏"}
    st.markdown(f"**{labels.get(status, '暂不能核对')}**")
    for reason in result.get("reasons") or []:
        _write(reason)
    if status != "mature":
        return
    window = mapping(result.get("window"))
    st.caption(f"原准确净值窗口 {window.get('entry_date')} → {window.get('target_date')}；"
               f"实际单位净值变化 {percent(result.get('actual_nav_change'), signed=True)}。不是个人账户收益。")
    timing = result.get("selection_timing")
    _write({"prospective": "这项选择记录在原模拟入场日之前。", "pre_target": "这项选择记录在原窗口走完之前，不是入场前选择。",
            "post_hoc": "这项选择记录时原窗口已经走完，只能作为事后模拟，不能当事前决策成绩。"}.get(
                timing, "记录时序未完整注明，不把这项结果当作事前成绩。"))
    for row in result.get("comparisons") or []:
        title = "新买一笔" if row.get("branch") == "flat" else "已有基金＋备用现金"
        _write(f"{title}：{_ACTIONS.get(row.get('action'), '原先没有唯一动作')}")
        if row.get("status") != "mature":
            for reason in row.get("reasons") or []:
                _write(reason)
            continue
        chosen, reference = mapping(row.get("chosen")), mapping(row.get("reference"))
        _write(f"按原本金与费用，模拟卖出净额 {_money(chosen.get('actual_value'))} 元（含应收款），"
               f"相对本金模拟盈亏 {_money(chosen.get('realized_payoff'))} 元。")
        _write(f"原简单参照“{_ACTIONS.get(row.get('reference_action'), '原参照')}”净额 {_money(reference.get('actual_value'))} 元；"
               f"本选择相对它多/少 {_money(row.get('realized_difference'))} 元。")
    st.caption("只用原判断、原费用和已有净值算这一次固定本金模拟；不重跑模型、不重新选动作、不写入真正发行成绩。")


def render_assessment_notes(saved: Mapping, *, read=None, save=None, replay=None) -> None:
    """Reading opens no private table; writes and result computation are explicit."""
    identifier = saved.get("id")
    if not identifier or (read is None and save is None and replay is None):
        return
    with st.expander("这次判断，我准备怎么做；后来模拟结果怎样"):
        st.caption("这是可选的本机笔记。模拟采用、计划与自述分开；不改变原判断、持仓或提交订单。")
        notes = []
        if read is not None:
            try:
                notes = read(identifier)
            except Exception:
                st.caption("暂时读不到这次笔记，原判断不受影响。")
        if save is not None:
            branch = st.selectbox("记录哪种情况", ["flat", "held"],
                format_func=lambda key: "准备新买" if key == "flat" else "已经持有", key=f"note_branch_{identifier}")
            actions = ["buy", "wait", "no_action"] if branch == "flat" else ["hold", "add", "reduce50", "exit", "reduce", "no_action"]
            with st.form(f"assessment_note_{identifier}"):
                kind = st.selectbox("记录类型", list(_KINDS), format_func=_KINDS.get, key=f"note_kind_{identifier}")
                action = st.selectbox("选择的做法", actions, format_func=_ACTIONS.get, key=f"note_action_{branch}_{identifier}")
                occurred = st.date_input("计划或自述发生日期", value=date.fromisoformat(now_iso()[:10]), key=f"note_date_{identifier}")
                reason = st.text_area("为什么这样做／接下来观察什么（可留空）", max_chars=1000, key=f"note_reason_{identifier}")
                submitted = st.form_submit_button("保存这条笔记")
            if submitted:
                try:
                    receipt = save(identifier, kind=kind, branch=branch, action=action,
                                   occurred_on=occurred.isoformat(), rationale=reason)
                except Exception:
                    st.error("笔记未保存成功；原判断和持仓没有改变。")
                else:
                    st.success(f"笔记已保存 · 实际记录于 {when(receipt.get('recorded_at'))}；发生日期不改变这个记录时间。")
                    notes = read(identifier) if read is not None else [receipt]
        selected_note = None
        if notes:
            for note in notes[:5]:
                _write(f"{_KINDS.get(note.get('kind'), '笔记')}：{_ACTIONS.get(note.get('action'), '未注明')} · "
                       f"发生日期 {note.get('occurred_on')} · 记录于 {when(note.get('recorded_at'))}")
                if note.get("rationale"):
                    _write(note["rationale"])
            choices = {note["id"]: note for note in notes}
            chosen = st.selectbox("核对原首选，或一条已记录的模拟选择", ["original", *choices],
                format_func=lambda key: "原判断保存时的首选与简单参照" if key == "original" else
                f"{_KINDS.get(choices[key].get('kind'), '笔记')} · {_ACTIONS.get(choices[key].get('action'))} · {when(choices[key].get('recorded_at'))}",
                key=f"note_replay_choice_{identifier}")
            selected_note = choices.get(chosen)
        if replay is not None and st.button("用已有净值查看这次模拟结果", key=f"note_replay_{identifier}"):
            try:
                result = replay(saved, note=selected_note)
            except Exception:
                st.error("这次模拟核对未完成；未更新数据、改写判断或记录正式成绩。")
            else:
                render_replay_result(mapping(result))

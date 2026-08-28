"""解释生成。

解释内容完全基于：求解结果、规则校验结果与题目员工数据。
面向门店店长输出通俗易懂的排班逻辑说明；繁琐的逐条规则核对
单独放入「规则校验明细」，供需要时查看。大模型只做语言润色，
不新增任何规则或事实。
"""

from __future__ import annotations

from typing import Dict, List

from .nlu import DEFAULT_BASE_URL, DEFAULT_MODEL, call_llm_text
from .types import DAYS, SHIFTS, Employee, RuleCheck, Schedule


def _emp_label(emp_map: Dict[str, Employee], eid: str) -> str:
    emp = emp_map.get(eid)
    return f"{eid}（{emp.position}）" if emp else f"{eid}（数据外）"


def _slot_line(schedule: Schedule, emp_map: Dict[str, Employee], day: str, shift: str) -> str:
    ids = schedule.get(day, shift)
    if not ids:
        return f"{day}{shift}：未安排人员"
    labels = "、".join(_emp_label(emp_map, eid) for eid in ids)
    return f"{day}{shift}（{len(ids)}人）：{labels}"


def _build_overview(schedule: Schedule, employees: List[Employee], checks: List[RuleCheck]) -> str:
    emp_map = {e.emp_id: e for e in employees}
    total_shifts = sum(len(ids) for ids in schedule.slots.values())
    involved = sorted({eid for ids in schedule.slots.values() for eid in ids if eid in emp_map})

    pref_hits = pref_total = 0
    for (_, shift_idx), ids in schedule.slots.items():
        shift = SHIFTS[shift_idx]
        for eid in ids:
            emp = emp_map.get(eid)
            if not emp:
                continue
            pref_total += 1
            if emp.preference == shift:
                pref_hits += 1
    pref_rate = round(pref_hits / pref_total * 100) if pref_total else 0

    weekend_part = sorted(
        e.emp_id
        for e in employees
        if set(e.available_days) <= {"周六", "周日"}
        and any(e.emp_id in schedule.get(d, s) for d in ("周六", "周日") for s in SHIFTS)
    )

    lines = [
        "这次排班的思路是：先保证每个班次都满足硬性要求——有店长值守的人、有会做饮品的人、有能收银的人（R-01～R-03），"
        "周一到周五每班至少 4 人、周六日每班至少 6 人（R-04）；同时只让员工在自己可工作的日期上班，请假日绝不安排（R-08），"
        "技能只认员工档案里登记过的（R-09）。",
        "在这些硬性要求都满足后，再避免疲劳：每人每周最多 5 个班、不连续工作超过 5 天、头天晚班后不排次日早班（R-05～R-07）。"
        "最后，尽量照顾员工偏好的班次，让每个人的工作量更均衡。",
        f"具体到这份排班：本周共安排 {total_shifts} 个班次，由 {len(involved)} 名员工承担；"
        f"约 {pref_rate}% 的班次按员工偏好的早班/晚班安排。",
    ]
    if weekend_part:
        lines.append(f"周末客流量大，周六日每班 6 人，兼职员工（{'、'.join(weekend_part)}）主要补充在周末。")
    return "\n".join(lines)


_LOGIC_STEPS: List[Dict[str, str]] = [
    {
        "icon": "🛡️",
        "title": "满足硬性要求",
        "text": "每个班次保证店长值守、饮品制作、收银人手（R-01～R-03），周一到周五每班至少 4 人、"
        "周六日每班至少 6 人（R-04）；只排可工作日期，请假日不排（R-08），技能以员工档案为准（R-09）。",
    },
    {
        "icon": "🧘",
        "title": "避免疲劳",
        "text": "每人每周最多 5 个班（R-05）、连续工作不超过 5 天（R-06）、头天晚班后不排次日早班（R-07）。",
    },
    {
        "icon": "⚖️",
        "title": "照顾偏好与均衡",
        "text": "硬性要求满足后，优先安排员工偏好的早班/晚班，并尽量让每人的工作量更均衡。",
    },
]


def _build_facts(schedule: Schedule, employees: List[Employee]) -> List[str]:
    """本次排班的几个关键数据标签。"""
    emp_map = {e.emp_id: e for e in employees}
    total_shifts = sum(len(ids) for ids in schedule.slots.values())
    involved = sorted({eid for ids in schedule.slots.values() for eid in ids if eid in emp_map})

    pref_hits = pref_total = 0
    for (_, shift_idx), ids in schedule.slots.items():
        shift = SHIFTS[shift_idx]
        for eid in ids:
            emp = emp_map.get(eid)
            if not emp:
                continue
            pref_total += 1
            if emp.preference == shift:
                pref_hits += 1
    pref_rate = round(pref_hits / pref_total * 100) if pref_total else 0

    weekend_part = sorted(
        e.emp_id
        for e in employees
        if set(e.available_days) <= {"周六", "周日"}
        and any(e.emp_id in schedule.get(d, s) for d in ("周六", "周日") for s in SHIFTS)
    )

    facts = [f"总班次 {total_shifts}", f"参与员工 {len(involved)} 名", f"偏好满足约 {pref_rate}%"]
    if weekend_part:
        facts.append(f"周末兼职补充 {'、'.join(weekend_part)}")
    return facts


def _rules_summary(checks: List[RuleCheck]) -> str:
    violated = [c for c in checks if c.status == "违反"]
    unknown = [c for c in checks if c.status == "无法判断"]
    if not violated and not unknown:
        return f"规则检查结果：全部 {len(checks)} 项通过（R-01～R-09、SC-01、REQ-01），这份排班可以直接使用。"
    parts = []
    if violated:
        parts.append(f"有 {len(violated)} 项违反：{'、'.join(c.rule_id for c in violated)}")
    if unknown:
        parts.append(f"有 {len(unknown)} 项因信息不足无法判断：{'、'.join(c.rule_id for c in unknown)}")
    return "规则检查结果：" + "；".join(parts) + "。建议按下方明细调整，或转人工复核。"


def build_schedule_explanation(
    schedule: Schedule,
    employees: List[Employee],
    checks: List[RuleCheck],
) -> Dict[str, object]:
    """生成通俗解释，返回多个小节文本。"""
    emp_map = {e.emp_id: e for e in employees}

    daily_lines: List[str] = []
    for day in DAYS:
        daily_lines.append("；".join(_slot_line(schedule, emp_map, day, s) for s in SHIFTS))

    rule_detail_lines: List[str] = []
    for c in checks:
        if c.status == "通过":
            rule_detail_lines.append(f"{c.rule_id} 通过：{c.attribution}")
        elif c.status == "违反":
            rule_detail_lines.append(f"{c.rule_id} 违反（风险{c.risk_level}）：{c.attribution}；{c.suggestion}")
        else:
            rule_detail_lines.append(f"{c.rule_id} 无法判断：{c.attribution}；{c.suggestion}")

    why_lines = _why_not_scheduled(schedule, employees)

    return {
        "overview": _build_overview(schedule, employees, checks),
        "logic_steps": _LOGIC_STEPS,
        "facts": _build_facts(schedule, employees),
        "special_notes": _build_special_notes(schedule, employees),
        "daily": "\n".join(daily_lines),
        "rules_summary": _rules_summary(checks),
        "rules_detail": "\n".join(rule_detail_lines),
        "why_not": "\n".join(why_lines) if why_lines else "所有员工均已按规则与排班需要参与排班。",
    }


def _build_special_notes(schedule: Schedule, employees: List[Employee]) -> List[Dict[str, str]]:
    """特殊情况说明：只列请假与可工作日受限的员工，结构化输出。"""
    notes: List[Dict[str, str]] = []

    leave = [e for e in employees if e.leave_days]
    if leave:
        parts = "、".join(f"{e.emp_id}（{'、'.join(e.leave_days)}）" for e in sorted(leave, key=lambda x: x.emp_id))
        notes.append(
            {
                "icon": "📅",
                "title": "请假员工",
                "text": (
                    f"本周有 {len(leave)} 名员工在员工数据中登记了请假"
                    "（来自题目员工列表「请假」列，R-08 规定请假日期不得排班）："
                    f"{parts}，已按要求未安排班次。"
                ),
            }
        )

    restricted = [e for e in employees if len(e.available_days) <= 4]
    if restricted:
        parts = "、".join(
            f"{e.emp_id}（仅{'、'.join(e.available_days)}）" for e in sorted(restricted, key=lambda x: x.emp_id)
        )
        notes.append(
            {
                "icon": "🗓️",
                "title": "可工作日较少",
                "text": f"另有 {len(restricted)} 名员工可工作日较少：{parts}，已尽量安排在可工作日期内。",
            }
        )

    if not notes:
        notes.append({"icon": "✅", "title": "无特殊限制", "text": "所有员工均可正常参与排班，无特殊限制。"})
    return notes


def _why_not_scheduled(schedule: Schedule, employees: List[Employee]) -> List[str]:
    """精简版文本（兼容旧调用方）。"""
    return [n["text"] for n in _build_special_notes(schedule, employees)]


def polish_with_llm(
    text: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """用大模型把通俗解释润色为通顺中文；失败时原样返回。"""
    try:
        prompt = (
            "你是排班助手。请把下面基于规则的排班解释整理成面向门店店长的通顺中文说明。\n"
            "要求：语言通俗易懂，不得新增规则、员工、数据或结论；必须保留 R-01～R-09 等规则编号；"
            "不得删除“无法判断/建议人工复核”等提示；直接输出润色后的文字。\n\n"
            + text
        )
        return call_llm_text(prompt, api_key, base_url, model).strip()
    except Exception:
        return text

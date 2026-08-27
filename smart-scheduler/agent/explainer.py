"""解释生成。

解释内容完全基于：求解结果、规则校验结果与题目员工数据；
大模型只做语言润色，不新增任何规则或事实。
"""

from __future__ import annotations

from typing import Dict, List

from .nlu import DEFAULT_BASE_URL, DEFAULT_MODEL, call_llm_text
from .types import DAYS, SHIFT_TIMES, SHIFTS, Employee, RuleCheck, Schedule


def _emp_label(emp_map: Dict[str, Employee], eid: str) -> str:
    emp = emp_map.get(eid)
    return f"{eid}（{emp.position}）" if emp else f"{eid}（数据外）"


def _slot_summary(schedule: Schedule, emp_map: Dict[str, Employee], day: str, shift: str) -> str:
    ids = schedule.get(day, shift)
    if not ids:
        return f"{day}{shift}：未安排人员"
    labels = "、".join(_emp_label(emp_map, eid) for eid in ids)
    manager = sum(1 for eid in ids if eid in emp_map and "店长值守" in emp_map[eid].skills)
    drink = sum(1 for eid in ids if eid in emp_map and "饮品制作" in emp_map[eid].skills)
    cashier = sum(1 for eid in ids if eid in emp_map and "收银" in emp_map[eid].skills)
    return (
        f"{day}{shift}（{SHIFT_TIMES[shift]}）：{labels}；共 {len(ids)} 人，"
        f"其中店长值守 {manager} 人、饮品制作 {drink} 人、收银 {cashier} 人"
    )


def build_schedule_explanation(
    schedule: Schedule,
    employees: List[Employee],
    checks: List[RuleCheck],
) -> Dict[str, str]:
    """生成结构化解释，返回多个小节文本。"""
    emp_map = {e.emp_id: e for e in employees}

    daily_lines: List[str] = []
    for day in DAYS:
        daily_lines.append("；".join(_slot_summary(schedule, emp_map, day, s) for s in SHIFTS))

    rule_lines: List[str] = []
    for c in checks:
        if c.status == "通过":
            rule_lines.append(f"{c.rule_id} 通过：{c.attribution}")
        elif c.status == "违反":
            rule_lines.append(f"{c.rule_id} 违反（风险{c.risk_level}）：{c.attribution}；{c.suggestion}")
        else:
            rule_lines.append(f"{c.rule_id} 无法判断：{c.attribution}；{c.suggestion}")

    why_lines = _why_not_scheduled(schedule, employees)

    return {
        "daily": "\n".join(daily_lines),
        "rules": "\n".join(rule_lines),
        "why_not": "\n".join(why_lines) if why_lines else "所有员工均已按规则与排班需要参与排班。",
    }


def _why_not_scheduled(schedule: Schedule, employees: List[Employee]) -> List[str]:
    assigned_counts: Dict[str, int] = {}
    for emp_ids in schedule.slots.values():
        for eid in emp_ids:
            assigned_counts[eid] = assigned_counts.get(eid, 0) + 1

    lines: List[str] = []
    for emp in employees:
        count = assigned_counts.get(emp.emp_id, 0)
        if count >= 5:
            continue
        reasons: List[str] = []
        if emp.leave_days:
            reasons.append(f"在{'、'.join(emp.leave_days)}请假（R-08），该日不可排班")
        if len(emp.available_days) < 7:
            reasons.append(f"仅可工作{'、'.join(emp.available_days)}（R-08）")
        if not reasons:
            reasons.append("未被排满 5 班，剩余班次系综合技能覆盖与工作量均衡后的取舍（软约束）")
        lines.append(f"{emp.emp_id}（{emp.position}）本周排 {count} 班：" + "；".join(reasons))
    return lines


def polish_with_llm(
    text: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """用大模型把结构化解释润色为通顺中文；失败时原样返回。"""
    try:
        prompt = (
            "你是排班助手。请把下面基于规则的排班解释润色为通顺、简洁、面向门店店长的中文说明。\n"
            "要求：不得新增规则、员工、数据或结论；必须保留 R-01~R-09 等规则编号；"
            "不得删除“无法判断/建议人工复核”等提示；直接输出润色后的文字。\n\n"
            + text
        )
        return call_llm_text(prompt, api_key, base_url, model).strip()
    except Exception:
        return text

"""内置题目提供的规则与员工数据（来自 todo.docx）。

规则 R-01~R-09 与 20 名员工（技能/可工作日期/请假/偏好）全部来自作业文档，
Agent 的排班与校验判断只基于这些数据。
"""

from __future__ import annotations

from typing import Dict, List

from .types import (
    DAYS,
    SKILL_CASHIER,
    SKILL_DRINK,
    SKILL_INVENTORY,
    SKILL_MANAGER,
    Employee,
    Rule,
)

RULES: List[Rule] = [
    Rule("R-01", "每个班至少 1 名具备店长值守资格的员工"),
    Rule("R-02", "每个班至少 2 名具备饮品制作技能的员工"),
    Rule("R-03", "每个班至少 1 名具备收银技能的员工"),
    Rule("R-04", "周一至周五每班至少 4 人；周六、周日每班至少 6 人"),
    Rule("R-05", "每人每周最多 40 小时，即最多 5 个班"),
    Rule("R-06", "不得连续工作超过 5 天"),
    Rule("R-07", "上一天晚班后不得安排次日早班"),
    Rule("R-08", "请假和不可工作日期绝对不得排班"),
    Rule("R-09", "技能必须来自员工数据，Agent 不得自行补技能"),
]

_WEEK = DAYS


def _all_week() -> List[str]:
    return list(_WEEK)


def _range_days(start_idx: int, end_idx: int) -> List[str]:
    return list(_WEEK[start_idx : end_idx + 1])


_EMPLOYEES: List[Employee] = [
    Employee("E01", "店长", [SKILL_MANAGER, SKILL_DRINK, SKILL_CASHIER, SKILL_INVENTORY], _all_week(), ["周三"], "早班"),
    Employee("E02", "副店长", [SKILL_MANAGER, SKILL_DRINK, SKILL_CASHIER], _all_week(), [], "晚班"),
    Employee("E03", "值班主管", [SKILL_MANAGER, SKILL_DRINK, SKILL_CASHIER], _range_days(0, 4), [], None),
    Employee("E04", "值班主管", [SKILL_MANAGER, SKILL_DRINK, SKILL_INVENTORY], _range_days(2, 6), [], "早班"),
    Employee("E05", "高级店员", [SKILL_MANAGER, SKILL_DRINK, SKILL_CASHIER], _range_days(4, 6), [], "晚班"),
    Employee("E06", "店员", [SKILL_DRINK, SKILL_CASHIER], _all_week(), ["周二"], "早班"),
    Employee("E07", "店员", [SKILL_DRINK, SKILL_CASHIER], _all_week(), [], "晚班"),
    Employee("E08", "店员", [SKILL_DRINK, SKILL_CASHIER, SKILL_INVENTORY], _range_days(0, 5), [], None),
    Employee("E09", "店员", [SKILL_DRINK], ["周一", "周三", "周五", "周六", "周日"], [], "早班"),
    Employee("E10", "店员", [SKILL_DRINK, SKILL_CASHIER], _range_days(1, 6), [], None),
    Employee("E11", "店员", [SKILL_CASHIER, SKILL_INVENTORY], _all_week(), [], "晚班"),
    Employee("E12", "店员", [SKILL_DRINK, SKILL_CASHIER], _range_days(0, 4), [], "早班"),
    Employee("E13", "兼职", [SKILL_DRINK], ["周六", "周日"], [], "早班"),
    Employee("E14", "兼职", [SKILL_DRINK, SKILL_CASHIER], ["周六", "周日"], [], "晚班"),
    Employee("E15", "兼职", [SKILL_CASHIER], _range_days(4, 6), [], None),
    Employee("E16", "兼职", [SKILL_DRINK], ["周三", "周四", "周六", "周日"], [], None),
    Employee("E17", "店员", [SKILL_DRINK, SKILL_CASHIER, SKILL_INVENTORY], _all_week(), ["周一"], None),
    Employee("E18", "店员", [SKILL_DRINK, SKILL_CASHIER], _range_days(0, 4), [], "晚班"),
    Employee("E19", "兼职", [SKILL_DRINK, SKILL_CASHIER], ["周六", "周日"], [], None),
    Employee("E20", "店员", [SKILL_DRINK, SKILL_INVENTORY], _range_days(1, 6), ["周四"], "早班"),
]


def load_employees() -> List[Employee]:
    """返回员工数据副本，避免调用方意外修改内置数据。"""
    return [
        Employee(e.emp_id, e.position, list(e.skills), list(e.available_days), list(e.leave_days), e.preference)
        for e in _EMPLOYEES
    ]


def load_rules() -> List[Rule]:
    return [Rule(r.rule_id, r.description) for r in RULES]


def employee_map() -> Dict[str, Employee]:
    return {e.emp_id: e for e in _EMPLOYEES}


def employees_summary_text() -> str:
    """供大模型意图解析使用的员工数据摘要（只包含题目数据）。"""
    lines = ["员工数据："]
    for e in _EMPLOYEES:
        leave = "、".join(e.leave_days) if e.leave_days else "无"
        pref = e.preference or "无"
        lines.append(
            f"- {e.emp_id}（{e.position}）：技能[{('、'.join(e.skills))}]；"
            f"可工作[{('、'.join(e.available_days))}]；请假[{leave}]；偏好[{pref}]"
        )
    return "\n".join(lines)


def rules_summary_text() -> str:
    """供大模型意图解析使用的规则摘要。"""
    return "\n".join(f"- {r.rule_id}：{r.description}" for r in RULES)

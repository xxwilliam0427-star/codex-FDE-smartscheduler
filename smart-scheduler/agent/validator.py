"""规则校验器。

逐条实现题目规则 R-01~R-09，并输出每条规则的：状态（通过/违反/无法判断）、
风险等级、涉及员工、整改建议与归因说明。

“无法判断”用于：员工 ID 不在数据中、班次信息缺失、或数据不足以完成判断，
此时输出建议人工复核，绝不臆测。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .types import (
    DAYS,
    SHIFT_INDEX,
    SHIFTS,
    SKILL_CASHIER,
    SKILL_DRINK,
    SKILL_MANAGER,
    Employee,
    Rule,
    RuleCheck,
    Schedule,
)

ALL_SLOTS: List[Tuple[int, int]] = [(d, s) for d in range(7) for s in range(2)]


def _slot_label(day_idx: int, shift_idx: int) -> str:
    return f"{DAYS[day_idx]}{SHIFTS[shift_idx]}"


def _skill_count(schedule: Schedule, emp_map: Dict[str, Employee], slot, skill: str) -> int:
    return sum(1 for eid in schedule.slots.get(slot, []) if eid in emp_map and skill in emp_map[eid].skills)


def _check_slot_skills(
    rule_id: str,
    description: str,
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    skill: str,
    required: int,
    risk: str,
) -> RuleCheck:
    """R-01/R-02/R-03 通用检查：每个班次至少 N 名具备某技能的员工。"""
    involved: List[str] = []
    details: List[str] = []
    unknown_seen: List[str] = []
    missing_slots: List[str] = []

    for slot in ALL_SLOTS:
        label = _slot_label(*slot)
        if slot not in schedule.slots:
            missing_slots.append(label)
            continue
        emp_ids = schedule.slots[slot]
        slot_unknown = [eid for eid in emp_ids if eid not in emp_map]
        if slot_unknown:
            unknown_seen.extend(slot_unknown)
        count = _skill_count(schedule, emp_map, slot, skill)
        if not slot_unknown and count < required:
            involved.append(label)
            details.append(f"{label}仅有 {count} 名具备[{skill}]技能的员工，需至少 {required} 名")

    has_violation = bool(involved)
    has_unknown = bool(unknown_seen)
    has_missing = bool(missing_slots)

    if has_violation:
        status, risk_level = "违反", risk
        attribution = f"发现 {len(involved)} 个班次不满足[{skill}]覆盖要求"
        suggestion = f"为上述班次增派具备[{skill}]技能的员工，或调整现有排班"
    elif has_unknown or has_missing:
        status, risk_level = "无法判断", "-"
        why = []
        if unknown_seen:
            why.append(f"含数据外员工 {'、'.join(sorted(set(unknown_seen)))}，无法核实其技能")
        if missing_slots:
            why.append(f"未提供班次 {'、'.join(missing_slots[:8])} 的人员信息")
        attribution = "；".join(why)
        suggestion = "补充缺失信息后重新检查，或转人工复核"
    else:
        status, risk_level = "通过", "-"
        attribution = "所有已提供班次均满足要求"
        suggestion = ""

    return RuleCheck(
        rule_id=rule_id,
        description=description,
        status=status,
        risk_level=risk_level,
        involved=sorted(set(involved)),
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _check_headcount(
    schedule: Schedule,
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-04：周一至周五每班至少 4 人；周六、周日每班至少 6 人。"""
    involved: List[str] = []
    details: List[str] = []
    missing_slots: List[str] = []

    for slot in ALL_SLOTS:
        label = _slot_label(*slot)
        if slot not in schedule.slots:
            missing_slots.append(label)
            continue
        day_idx, _ = slot
        required = 4 if day_idx < 5 else 6
        count = len(schedule.slots[slot])
        if count < required:
            involved.append(label)
            details.append(f"{label}仅 {count} 人，需至少 {required} 人")

    if involved:
        status, risk = "违反", "中"
        attribution = f"发现 {len(involved)} 个班次人数不足"
        suggestion = "为人数不足的班次增派可工作员工"
    elif missing_slots:
        status, risk = "无法判断", "-"
        attribution = f"未提供班次 {'、'.join(missing_slots[:8])} 的人员信息，无法核算人数"
        suggestion = "补充缺失信息后重新检查，或转人工复核"
    else:
        status, risk = "通过", "-"
        attribution = "所有已提供班次人数均达到最低要求"
        suggestion = ""

    return RuleCheck(
        rule_id="R-04",
        description=rules_by_id["R-04"].description,
        status=status,
        risk_level=risk,
        involved=sorted(set(involved)),
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _check_weekly_load(
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-05：每人每周最多 5 个班（40 小时）。"""
    counts: Counter[str] = Counter()
    for emp_ids in schedule.slots.values():
        for eid in emp_ids:
            if eid in emp_map:
                counts[eid] += 1

    violated = {eid: n for eid, n in counts.items() if n > 5}
    missing = [s for s in ALL_SLOTS if s not in schedule.slots]

    if violated:
        status, risk = "违反", "中"
        involved = sorted(violated)
        details = [f"{eid} 本周已排 {n} 个班，超过 5 班上限" for eid, n in sorted(violated.items())]
        attribution = "部分员工周班次数超过 40 小时（5 班）上限"
        suggestion = "将超限员工的部分班次移交给其他员工"
    elif missing:
        status, risk = "无法判断", "-"
        involved = []
        details = [f"未提供班次 {'、'.join([_slot_label(*s) for s in missing[:8]])} 的人员信息"]
        attribution = "排班信息不完整，无法核算员工周总班次数"
        suggestion = "补充完整排班后重新检查，或转人工复核"
    else:
        status, risk = "通过", "-"
        involved = []
        details = [f"所有员工周班次数均 ≤ 5（最高 {max(counts.values()) if counts else 0} 班）"]
        attribution = "所有员工均未超过每周 40 小时上限"
        suggestion = ""

    return RuleCheck(
        rule_id="R-05",
        description=rules_by_id["R-05"].description,
        status=status,
        risk_level=risk,
        involved=involved,
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _max_consecutive_days(work_days: set) -> int:
    best = cur = 0
    for d in range(7):
        if d in work_days:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _check_consecutive_days(
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-06：不得连续工作超过 5 天。"""
    work_days: Dict[str, set] = defaultdict(set)
    for (day_idx, _), emp_ids in schedule.slots.items():
        for eid in emp_ids:
            if eid in emp_map:
                work_days[eid].add(day_idx)

    violated = {eid: _max_consecutive_days(days) for eid, days in work_days.items() if _max_consecutive_days(days) > 5}
    missing = [s for s in ALL_SLOTS if s not in schedule.slots]

    if violated:
        status, risk = "违反", "中"
        involved = sorted(violated)
        details = [f"{eid} 连续工作 {n} 天，超过 5 天上限" for eid, n in sorted(violated.items())]
        attribution = "部分员工存在超过 5 天的连续工作区间"
        suggestion = "在连续工作 5 天后为相关员工安排休息日"
    elif missing:
        status, risk = "无法判断", "-"
        involved = []
        details = [f"未提供班次 {'、'.join([_slot_label(*s) for s in missing[:8]])} 的人员信息"]
        attribution = "排班信息不完整，无法核算连续工作天数"
        suggestion = "补充完整排班后重新检查，或转人工复核"
    else:
        status, risk = "通过", "-"
        involved = []
        details = [f"所有员工最长连续工作均 ≤ 5 天"]
        attribution = "不存在超过 5 天的连续工作区间"
        suggestion = ""

    return RuleCheck(
        rule_id="R-06",
        description=rules_by_id["R-06"].description,
        status=status,
        risk_level=risk,
        involved=involved,
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _check_rest_transition(
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-07：上一天晚班后不得安排次日早班。"""
    late_idx = SHIFT_INDEX["晚班"]
    early_idx = SHIFT_INDEX["早班"]
    violations: List[str] = []
    details: List[str] = []

    for d in range(6):
        late_slot = (d, late_idx)
        early_slot = (d + 1, early_idx)
        if late_slot not in schedule.slots or early_slot not in schedule.slots:
            continue
        late_emps = set(schedule.slots[late_slot]) & set(emp_map)
        early_emps = set(schedule.slots[early_slot]) & set(emp_map)
        conflict = sorted(late_emps & early_emps)
        for eid in conflict:
            violations.append(eid)
            details.append(f"{eid} 在{DAYS[d]}晚班后又安排到{DAYS[d + 1]}早班")

    missing = [s for s in ALL_SLOTS if s not in schedule.slots]

    if violations:
        status, risk = "违反", "中"
        attribution = f"{len(set(violations))} 名员工存在晚班后次日早班的情况"
        suggestion = "调整相关员工次日班次，避免晚班后连早班"
    elif missing:
        status, risk = "无法判断", "-"
        attribution = f"未提供班次 {'、'.join([_slot_label(*s) for s in missing[:8]])} 的人员信息，无法完整核对晚班→早班衔接"
        suggestion = "补充完整排班后重新检查，或转人工复核"
    else:
        status, risk = "通过", "-"
        attribution = "不存在晚班后次日早班的情况"
        suggestion = ""

    return RuleCheck(
        rule_id="R-07",
        description=rules_by_id["R-07"].description,
        status=status,
        risk_level=risk,
        involved=sorted(set(violations)),
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _check_availability(
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-08：请假和不可工作日期绝对不得排班。"""
    violations: List[str] = []
    details: List[str] = []
    unknown_seen: List[str] = []

    for (day_idx, _), emp_ids in schedule.slots.items():
        day = DAYS[day_idx]
        for eid in emp_ids:
            if eid not in emp_map:
                unknown_seen.append(eid)
                continue
            emp = emp_map[eid]
            if day in emp.leave_days:
                violations.append(eid)
                details.append(f"{eid} 在{day}请假（R-08），不得排班")
            elif day not in emp.available_days:
                violations.append(eid)
                details.append(f"{eid} 在{day}不可工作（不在可工作日期内），不得排班")

    if violations:
        status, risk = "违反", "高"
        attribution = "发现请假日/不可工作日的排班"
        suggestion = "立即移除相关员工当天班次，并替换为可工作员工"
    elif unknown_seen:
        status, risk = "无法判断", "-"
        attribution = f"含数据外员工 {'、'.join(sorted(set(unknown_seen)))}，无法核实其请假与可工作日期"
        suggestion = "补充员工数据后重新检查，或转人工复核"
    else:
        status, risk = "通过", "-"
        attribution = "所有排班均在员工可工作日期内，且未占用请假日期"
        suggestion = ""

    return RuleCheck(
        rule_id="R-08",
        description=rules_by_id["R-08"].description,
        status=status,
        risk_level=risk,
        involved=sorted(set(violations)),
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def _check_skills_from_data(
    schedule: Schedule,
    emp_map: Dict[str, Employee],
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """R-09：技能必须来自员工数据，Agent 不得自行补技能。"""
    unknown = sorted({eid for emp_ids in schedule.slots.values() for eid in emp_ids if eid not in emp_map})
    if unknown:
        return RuleCheck(
            rule_id="R-09",
            description=rules_by_id["R-09"].description,
            status="违反",
            risk_level="高",
            involved=unknown,
            details=[f"排班中出现数据外员工 {'、'.join(unknown)}，无法核实其技能，视为 Agent 擅自补入人员"],
            suggestion="移除数据外员工，或先在员工数据中补充其技能后再排班",
            attribution="排班使用了不在题目员工数据中的员工，技能无法溯源",
        )
    return RuleCheck(
        rule_id="R-09",
        description=rules_by_id["R-09"].description,
        status="通过",
        risk_level="-",
        attribution="所有排班员工均来自题目员工数据，技能判断完全基于员工技能字段",
    )


def _check_one_shift_per_day(
    schedule: Schedule,
    rules_by_id: Dict[str, Rule],
) -> RuleCheck:
    """场景约束 SC-01：每人每天最多一个班。"""
    per_day: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    violations: List[str] = []
    details: List[str] = []

    for (day_idx, _), emp_ids in schedule.slots.items():
        for eid in emp_ids:
            per_day[(DAYS[day_idx], eid)][0] += 1

    for (day, eid), cnt in per_day.items():
        if cnt[0] > 1:
            violations.append(eid)
            details.append(f"{eid} 在{day}被安排了 {cnt[0]} 个班次")

    if violations:
        status, risk = "违反", "高"
        attribution = "存在同一员工同一天被安排多个班次"
        suggestion = "每人每天最多一个班，请移除同日重复班次"
    elif not schedule.slots:
        status, risk = "无法判断", "-"
        attribution = "未提供任何班次信息"
        suggestion = "补充排班信息后重新检查"
    else:
        status, risk = "通过", "-"
        attribution = "每人每天均不超过一个班"
        suggestion = ""

    return RuleCheck(
        rule_id="SC-01",
        description="每人每天最多一个班（排班场景约束）",
        status=status,
        risk_level=risk,
        involved=sorted(set(violations)),
        details=details,
        suggestion=suggestion,
        attribution=attribution,
    )


def validate_schedule(schedule: Schedule, employees: List[Employee], rules: List[Rule]) -> List[RuleCheck]:
    """对排班逐条执行 R-01~R-09 与场景约束 SC-01 的校验。"""
    emp_map = {e.emp_id: e for e in employees}
    rules_by_id = {r.rule_id: r for r in rules}

    if not schedule.slots:
        return [
            RuleCheck(
                rule_id="INPUT",
                description="排班输入检查",
                status="无法判断",
                risk_level="-",
                details=["未识别到任何班次信息"],
                suggestion="请按「周一早班：E01,E02,E03,E04」的格式提供排班",
                attribution="输入中没有任何可解析的班次内容",
            )
        ]

    checks = [
        _check_slot_skills("R-01", rules_by_id["R-01"].description, schedule, emp_map, SKILL_MANAGER, 1, "高"),
        _check_slot_skills("R-02", rules_by_id["R-02"].description, schedule, emp_map, SKILL_DRINK, 2, "高"),
        _check_slot_skills("R-03", rules_by_id["R-03"].description, schedule, emp_map, SKILL_CASHIER, 1, "高"),
        _check_headcount(schedule, rules_by_id),
        _check_weekly_load(schedule, emp_map, rules_by_id),
        _check_consecutive_days(schedule, emp_map, rules_by_id),
        _check_rest_transition(schedule, emp_map, rules_by_id),
        _check_availability(schedule, emp_map, rules_by_id),
        _check_skills_from_data(schedule, emp_map, rules_by_id),
        _check_one_shift_per_day(schedule, rules_by_id),
    ]
    return checks


def all_pass(checks: List[RuleCheck]) -> bool:
    return all(c.status == "通过" for c in checks)

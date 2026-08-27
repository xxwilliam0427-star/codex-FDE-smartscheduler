"""回溯式 CSP 排班求解器。

确定性算法：对 14 个班次槽位（7 天 × 早/晚班）做约束搜索。
- 硬约束：R-01~R-09（技能覆盖、人数、周上限、连续天数、晚班→早班衔接、
  请假/不可用日期、技能来自数据）以及“每人每天最多一个班”。
- 软约束：员工班次偏好、工作量均衡（排班时优先选择偏好匹配且当前班次较少的员工）。
- 找不到完全合规方案时，输出“尽力方案 + 违规清单 + 建议人工复核”，绝不编造。
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from .validator import all_pass, validate_schedule

MAX_SHIFTS_PER_EMPLOYEE = 5
MAX_CONSECUTIVE_DAYS = 5
NODE_LIMIT = 400_000
TIME_LIMIT_SECONDS = 25.0


def default_min_counts() -> Dict[Tuple[int, int], int]:
    """R-04 默认最低人数：周一至周五 4 人，周六、周日 6 人。"""
    return {(d, s): (4 if d < 5 else 6) for d in range(7) for s in range(2)}


@dataclass
class SolveResult:
    schedule: Schedule
    feasible: bool
    message: str
    checks: List[RuleCheck]


class _SearchExhausted(Exception):
    pass


def _slot_label(day_idx: int, shift_idx: int) -> str:
    return f"{DAYS[day_idx]}{SHIFTS[shift_idx]}"


def _check_requested_headcount(
    schedule: Schedule,
    min_counts: Dict[Tuple[int, int], int],
) -> RuleCheck:
    """REQ-01：满足用户指定（或 R-04 默认）的班次最低人数。"""
    short: List[Tuple[Tuple[int, int], int, int]] = []
    for slot, required in min_counts.items():
        delivered = len(schedule.slots.get(slot, []))
        if delivered < required:
            short.append((slot, required, delivered))
    if not short:
        return RuleCheck(
            rule_id="REQ-01",
            description="满足用户指定/规则默认的班次最低人数",
            status="通过",
            risk_level="-",
            attribution="所有班次人数均达到要求",
        )
    details = [
        f"{_slot_label(d, s)}仅 {delivered} 人，要求 {required} 人"
        for (d, s), required, delivered in short
    ]
    involved = [_slot_label(d, s) for (d, s), _, _ in short]
    return RuleCheck(
        rule_id="REQ-01",
        description="满足用户指定/规则默认的班次最低人数",
        status="违反",
        risk_level="中",
        involved=involved,
        details=details,
        suggestion="人数不足的班次可用的可工作员工不足，请调整人数要求或补充可用员工后人工复核",
        attribution="、".join(details),
    )


def _max_run(work_days: set) -> int:
    best = cur = 0
    for d in range(7):
        if d in work_days:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


class _Solver:
    def __init__(
        self,
        employees: List[Employee],
        min_counts: Dict[Tuple[int, int], int],
        exclude: List[str],
        node_limit: int = NODE_LIMIT,
        time_limit: float = TIME_LIMIT_SECONDS,
    ):
        self.emp_map = {e.emp_id: e for e in employees if e.emp_id not in exclude}
        self.min_counts = min_counts
        self.node_limit = node_limit
        self.deadline = time.monotonic() + time_limit
        self.nodes = 0

        self.assignments: Dict[Tuple[int, int], List[str]] = {}
        self.shift_count: Counter = Counter()
        self.work_days: Dict[str, set] = {eid: set() for eid in self.emp_map}

        self._precompute_candidates()

    def _precompute_candidates(self) -> None:
        self.raw_candidates: Dict[Tuple[int, int], List[str]] = {}
        for d in range(7):
            day = DAYS[d]
            for s in range(2):
                shift = SHIFTS[s]
                cands = [
                    e.emp_id
                    for e in self.emp_map.values()
                    if day in e.available_days and day not in e.leave_days
                ]
                cands.sort(key=lambda eid: self._quality_key(eid, shift, 0))
                self.raw_candidates[(d, s)] = cands

    def _quality_key(self, eid: str, shift: str, shift_count: int) -> Tuple[int, int, int, str]:
        pref = self.emp_map[eid].preference
        if pref == shift:
            rank = 0
        elif pref is None:
            rank = 1
        else:
            rank = 2
        return (rank, shift_count, len(self.work_days[eid]), eid)

    def _eligible(self, eid: str, day_idx: int, shift_idx: int) -> bool:
        """员工是否可被安排到该槽位（含全局约束，不含同槽位技能组合）。"""
        emp = self.emp_map[eid]
        day = DAYS[day_idx]
        if day not in emp.available_days or day in emp.leave_days:
            return False
        if self.shift_count[eid] >= MAX_SHIFTS_PER_EMPLOYEE:
            return False
        # 每人每天最多一个班
        if day_idx in self.work_days[eid]:
            return False
        # R-07：晚班后不得次日早班（双向检查）
        late = SHIFT_INDEX["晚班"]
        early = SHIFT_INDEX["早班"]
        if shift_idx == early and (day_idx - 1, late) in self.assignments and eid in self.assignments[(day_idx - 1, late)]:
            return False
        if shift_idx == late and (day_idx + 1, early) in self.assignments and eid in self.assignments[(day_idx + 1, early)]:
            return False
        # R-06：连续工作不超过 5 天
        new_days = self.work_days[eid] | {day_idx}
        if _max_run(new_days) > MAX_CONSECUTIVE_DAYS:
            return False
        return True

    def _slot_skills_ok(self, slot: Tuple[int, int], combo: List[str]) -> bool:
        """R-01/R-02/R-03：组合是否覆盖店长值守、饮品制作、收银要求。"""
        manager = drink = cashier = 0
        for eid in combo:
            skills = self.emp_map[eid].skills
            manager += 1 if SKILL_MANAGER in skills else 0
            drink += 1 if SKILL_DRINK in skills else 0
            cashier += 1 if SKILL_CASHIER in skills else 0
        return manager >= 1 and drink >= 2 and cashier >= 1

    def _try_combo(self, slot: Tuple[int, int], combo: List[str]) -> bool:
        day_idx, _ = slot
        for eid in combo:
            if not self._eligible(eid, day_idx, slot[1]):
                return False
        if not self._slot_skills_ok(slot, combo):
            return False
        return True

    def _candidates_for(self, slot: Tuple[int, int]) -> List[str]:
        day_idx, shift_idx = slot
        shift = SHIFTS[shift_idx]
        cands = [eid for eid in self.raw_candidates[slot] if self._eligible(eid, day_idx, shift_idx)]
        cands.sort(key=lambda eid: self._quality_key(eid, shift, self.shift_count[eid]))
        return cands

    def _forward_check(self, remaining: List[Tuple[int, int]]) -> bool:
        for slot in remaining:
            if len(self._candidates_for(slot)) < self.min_counts[slot]:
                return False
        return True

    def _assign(self, slot: Tuple[int, int], combo: List[str]) -> None:
        day_idx, shift_idx = slot
        self.assignments[slot] = combo
        for eid in combo:
            self.shift_count[eid] += 1
            self.work_days[eid].add(day_idx)

    def _unassign(self, slot: Tuple[int, int], combo: List[str]) -> None:
        day_idx, _ = slot
        del self.assignments[slot]
        for eid in combo:
            self.shift_count[eid] -= 1
            self.work_days[eid].discard(day_idx)

    def _combos_for(self, slot: Tuple[int, int]) -> List[List[str]]:
        """按质量排序生成候选组合：先试最少人数，再逐步加人。"""
        cands = self._candidates_for(slot)
        k_min = self.min_counts[slot]
        k_max = min(len(cands), k_min + 2)
        combos: List[List[str]] = []
        for k in range(max(k_min, 0), k_max + 1):
            for combo in itertools.combinations(cands, k):
                combos.append(list(combo))
        return combos

    def _search(self, remaining: List[Tuple[int, int]]) -> bool:
        self.nodes += 1
        if self.nodes > self.node_limit or time.monotonic() > self.deadline:
            raise _SearchExhausted()

        if not remaining:
            return True

        # MRV：优先安排候选员工最少的槽位
        slot = min(remaining, key=lambda s: len(self._candidates_for(s)))
        rest = [s for s in remaining if s != slot]

        for combo in self._combos_for(slot):
            if not self._try_combo(slot, combo):
                continue
            self._assign(slot, combo)
            if self._forward_check(rest):
                if self._search(rest):
                    return True
            self._unassign(slot, combo)
        return False

    def solve(self) -> Optional[Schedule]:
        slots = [(d, s) for d in range(7) for s in range(2)]
        try:
            if self._search(slots):
                return Schedule(slots={k: list(v) for k, v in self.assignments.items()})
        except _SearchExhausted:
            return None
        return None

    def greedy_best_effort(self) -> Schedule:
        """尽力方案：逐槽位挑选满足技能覆盖的组合，不保证全局规则全部满足。"""
        result = Schedule()
        order = sorted(
            [(d, s) for d in range(7) for s in range(2)],
            key=lambda slot: (
                slot[0] < 5,                       # 周末优先
                slot[1] != SHIFT_INDEX["晚班"],    # 晚班优先
                -self.min_counts[slot],
                len(self.raw_candidates[slot]),
            ),
        )
        for slot in order:
            cands = self._candidates_for(slot)
            if not cands:
                result.slots[slot] = []
                continue
            k_min = min(self.min_counts[slot], len(cands))
            k_max = min(len(cands), k_min + 2)
            chosen = None
            for k in range(max(k_min, 1), k_max + 1):
                for combo in itertools.combinations(cands, k):
                    if self._slot_skills_ok(slot, list(combo)):
                        chosen = list(combo)
                        break
                if chosen:
                    break
            if chosen is None:
                chosen = list(cands[: max(k_min, 1)])
            self._assign(slot, chosen)
            result.slots[slot] = chosen
        return result


def solve_schedule(
    employees: List[Employee],
    rules: List[Rule],
    min_counts: Optional[Dict[Tuple[int, int], int]] = None,
    exclude: Optional[List[str]] = None,
) -> SolveResult:
    """生成排班。返回排班、是否完全合规、消息与规则校验结果。"""
    min_counts = dict(default_min_counts()) if min_counts is None else min_counts
    exclude = exclude or []

    solver = _Solver(employees, min_counts, exclude)
    schedule = solver.solve()

    if schedule is None:
        # 搜索失败/超时时可能残留部分赋值状态，用全新求解器生成尽力方案
        greedy = _Solver(employees, min_counts, exclude)
        schedule = greedy.greedy_best_effort()
        checks = validate_schedule(schedule, employees, rules)
        checks.append(_check_requested_headcount(schedule, min_counts))
        return SolveResult(
            schedule=schedule,
            feasible=False,
            message="未能在时限内找到满足全部人数要求的排班，已给出尽力方案（含违规标注），建议人工复核。",
            checks=checks,
        )

    checks = validate_schedule(schedule, employees, rules)
    checks.append(_check_requested_headcount(schedule, min_counts))
    feasible = all_pass(checks)
    message = "已生成排班，并通过全部规则与人数要求。" if feasible else "已生成排班，但存在规则冲突或人数缺口，建议人工复核。"
    return SolveResult(schedule=schedule, feasible=feasible, message=message, checks=checks)

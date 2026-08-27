"""排班求解器测试：默认数据可行性与兜底行为。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data_loader import load_employees, load_rules
from agent.scheduler import default_min_counts, solve_schedule
from agent.validator import all_pass


EMPLOYEES = load_employees()
RULES = load_rules()


def test_default_scenario_is_feasible():
    result = solve_schedule(EMPLOYEES, RULES)
    assert result.feasible
    assert all_pass(result.checks)
    assert next(c for c in result.checks if c.rule_id == "REQ-01").status == "通过"
    assert len(result.schedule.slots) == 14
    for (day_idx, _), ids in result.schedule.slots.items():
        assert len(ids) >= (4 if day_idx < 5 else 6)


def test_custom_min_counts_feasible():
    counts = default_min_counts()
    for d in range(5):
        for s in range(2):
            counts[(d, s)] = 5
    result = solve_schedule(EMPLOYEES, RULES, min_counts=counts)
    assert result.feasible, result.message


def test_exclude_employee_still_feasible():
    result = solve_schedule(EMPLOYEES, RULES, exclude=["E01"])
    assert result.feasible, result.message


def test_infeasible_request_returns_best_effort():
    counts = default_min_counts()
    for s in range(2):
        counts[(5, s)] = 9  # 周六单日两班 18 人次，超出周末可用人次
        counts[(6, s)] = 9
    result = solve_schedule(EMPLOYEES, RULES, min_counts=counts)
    assert not result.feasible
    assert result.schedule is not None
    assert any(c.rule_id == "REQ-01" and c.status == "违反" for c in result.checks)
    assert "人工复核" in result.message

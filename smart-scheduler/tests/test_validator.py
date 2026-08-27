"""规则校验器单元测试：R-01~R-09 与场景约束 SC-01。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data_loader import load_employees, load_rules
from agent.scheduler import solve_schedule
from agent.types import Schedule
from agent.validator import validate_schedule


EMPLOYEES = load_employees()
RULES = load_rules()


def make_schedule(**slots) -> Schedule:
    s = Schedule()
    for key, ids in slots.items():
        day, shift = key.split("_")
        s.set(day, shift, ids)
    return s


def valid_full_schedule() -> Schedule:
    """用默认场景求解出的完全合规排班作为基准。"""
    result = solve_schedule(EMPLOYEES, RULES)
    assert result.feasible, result.message
    return result.schedule


def status_of(checks, rule_id: str) -> str:
    return next(c.status for c in checks if c.rule_id == rule_id)


def test_all_rules_pass_on_generated_schedule():
    checks = validate_schedule(valid_full_schedule(), EMPLOYEES, RULES)
    for c in checks:
        assert c.status == "通过", c.rule_id


def test_r01_fail_without_manager():
    s = valid_full_schedule()
    s.set("周一", "早班", ["E06", "E07", "E08", "E10"])  # 均无店长值守
    checks = validate_schedule(s, EMPLOYEES, RULES)
    assert status_of(checks, "R-01") == "违反"
    assert "周一早班" in next(c for c in checks if c.rule_id == "R-01").involved


def test_r02_fail_with_too_few_drink_skill():
    s = valid_full_schedule()
    s.set("周一", "早班", ["E01", "E11", "E15"])  # 仅 E01 有饮品制作
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "R-02") == "违反"


def test_r03_fail_without_cashier():
    s = valid_full_schedule()
    s.set("周一", "早班", ["E04", "E09", "E13", "E16"])  # 均无收银
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "R-03") == "违反"


def test_r04_fail_weekday_and_weekend_headcount():
    s = valid_full_schedule()
    s.set("周一", "早班", ["E01", "E02", "E03"])       # 周一 3 人 < 4
    s.set("周六", "早班", ["E01", "E02", "E03", "E04", "E05"])  # 周六 5 人 < 6
    checks = validate_schedule(s, EMPLOYEES, RULES)
    assert status_of(checks, "R-04") == "违反"
    involved = next(c for c in checks if c.rule_id == "R-04").involved
    assert "周一早班" in involved and "周六早班" in involved


def test_r05_fail_six_shifts():
    # E01 周一/二/四/五/六/日早班（避开周三请假）共 6 班
    s = make_schedule(
        周一_早班=["E01", "E02", "E03", "E06"],
        周二_早班=["E01", "E02", "E03", "E06"],
        周四_早班=["E01", "E02", "E03", "E06"],
        周五_早班=["E01", "E02", "E03", "E06"],
        周六_早班=["E01", "E02", "E03", "E06", "E07", "E08"],
        周日_早班=["E01", "E02", "E03", "E06", "E07", "E08"],
    )
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "R-05") == "违反"


def test_r06_fail_six_consecutive_days():
    # E02 全周可用，周一至周六早班 = 连续 6 天
    s = make_schedule(
        周一_早班=["E02", "E01", "E03", "E06"],
        周二_早班=["E02", "E01", "E03", "E06"],
        周三_早班=["E02", "E01", "E03", "E06"],
        周四_早班=["E02", "E01", "E03", "E06"],
        周五_早班=["E02", "E01", "E03", "E06"],
        周六_早班=["E02", "E01", "E03", "E06", "E07", "E08"],
    )
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "R-06") == "违反"


def test_r07_fail_late_then_early():
    s = valid_full_schedule()
    s.set("周一", "晚班", ["E02", "E07", "E11", "E18"])
    s.set("周二", "早班", ["E02", "E01", "E03", "E06"])  # E02 晚班后次日早班
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "R-07") == "违反"


def test_r08_fail_leave_day_and_unavailable_day():
    s = valid_full_schedule()
    s.set("周三", "早班", ["E01", "E06", "E07", "E08"])  # E01 周三请假
    s.set("周六", "早班", ["E03", "E06", "E07", "E08", "E09", "E10"])  # E03 周六不可工作
    checks = validate_schedule(s, EMPLOYEES, RULES)
    assert status_of(checks, "R-08") == "违反"
    involved = next(c for c in checks if c.rule_id == "R-08").involved
    assert "E01" in involved and "E03" in involved


def test_r09_fail_unknown_employee():
    s = valid_full_schedule()
    s.set("周一", "早班", ["E01", "E02", "E03", "E99"])
    checks = validate_schedule(s, EMPLOYEES, RULES)
    assert status_of(checks, "R-09") == "违反"
    # 数据外员工导致技能类规则对该班次无法判断
    assert status_of(checks, "R-01") == "无法判断"
    assert status_of(checks, "R-02") == "无法判断"
    assert status_of(checks, "R-03") == "无法判断"


def test_sc01_fail_same_day_two_shifts():
    s = valid_full_schedule()
    s.set("周一", "晚班", ["E01", "E07", "E11", "E18"])  # E01 周一早班已排
    assert status_of(validate_schedule(s, EMPLOYEES, RULES), "SC-01") == "违反"


def test_partial_schedule_cannot_judge():
    s = make_schedule(周一_早班=["E01", "E02", "E03", "E06"], 周一_晚班=["E02", "E07", "E11", "E18"])
    checks = validate_schedule(s, EMPLOYEES, RULES)
    assert status_of(checks, "R-04") == "无法判断"
    assert status_of(checks, "R-05") == "无法判断"
    assert status_of(checks, "R-06") == "无法判断"
    assert status_of(checks, "R-07") == "无法判断"


def test_empty_schedule_cannot_judge():
    checks = validate_schedule(Schedule(), EMPLOYEES, RULES)
    assert len(checks) == 1
    assert checks[0].rule_id == "INPUT"
    assert checks[0].status == "无法判断"

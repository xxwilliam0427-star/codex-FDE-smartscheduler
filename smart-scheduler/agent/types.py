"""核心数据模型。

所有类型均与题目文档（todo.docx）中的排班场景、规则和员工数据对应，
保证判断依据可溯源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 一周七天（下标 0-6 对应周一至周日）
DAYS: List[str] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
DAY_INDEX: Dict[str, int] = {d: i for i, d in enumerate(DAYS)}

# 班次（下标 0-1 对应早班、晚班）
SHIFTS: List[str] = ["早班", "晚班"]
SHIFT_INDEX: Dict[str, int] = {s: i for i, s in enumerate(SHIFTS)}
SHIFT_TIMES: Dict[str, str] = {"早班": "09:00-17:00", "晚班": "13:00-21:00"}

# 技能常量（来自题目员工数据）
SKILL_MANAGER = "店长值守"
SKILL_DRINK = "饮品制作"
SKILL_CASHIER = "收银"
SKILL_INVENTORY = "库存管理"

# 班次槽位 key：(day_idx, shift_idx)
SlotKey = Tuple[int, int]


@dataclass
class Employee:
    emp_id: str
    position: str
    skills: List[str]
    available_days: List[str]  # 可工作日期（周几）
    leave_days: List[str]      # 请假日期（周几）
    preference: Optional[str]  # "早班" / "晚班" / None


@dataclass
class Rule:
    rule_id: str
    description: str


@dataclass
class Schedule:
    """排班结果：槽位 -> 员工 ID 列表。缺失槽位视为未提供信息。"""

    slots: Dict[SlotKey, List[str]] = field(default_factory=dict)

    def get(self, day: str, shift: str) -> List[str]:
        return list(self.slots.get((DAY_INDEX[day], SHIFT_INDEX[shift]), []))

    def set(self, day: str, shift: str, emp_ids: List[str]) -> None:
        self.slots[(DAY_INDEX[day], SHIFT_INDEX[shift])] = list(emp_ids)

    def all_slots_filled(self) -> bool:
        return len(self.slots) == len(DAYS) * len(SHIFTS)


@dataclass
class RuleCheck:
    """单条规则的校验结果，含归因说明。"""

    rule_id: str
    description: str
    status: str            # 通过 / 违反 / 无法判断
    risk_level: str        # 高 / 中 / 低 / -
    involved: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    suggestion: str = ""
    attribution: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "status": self.status,
            "risk_level": self.risk_level,
            "involved": self.involved,
            "details": self.details,
            "suggestion": self.suggestion,
            "attribution": self.attribution,
        }

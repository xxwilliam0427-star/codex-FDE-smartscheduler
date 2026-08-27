"""端到端自测脚本：本地模式（无大模型）跑通默认场景，输出完整样例。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data_loader import employees_summary_text, load_employees, load_rules, rules_summary_text
from agent.explainer import build_schedule_explanation
from agent.nlu import parse_intent
from agent.scheduler import default_min_counts, solve_schedule
from agent.types import DAYS, DAY_INDEX, SHIFT_INDEX, SHIFTS


def main() -> None:
    employees = load_employees()
    rules = load_rules()
    user_input = "帮我安排周一到周日的排班，各班按规则最低人数安排"

    intent, mode, notes = parse_intent(
        user_input,
        employees_summary_text(),
        rules_summary_text(),
        api_key=None,
        use_llm=False,
    )

    counts = default_min_counts()
    for (day, shift), n in intent.min_counts.items():
        counts[(DAY_INDEX[day], SHIFT_INDEX[shift])] = n

    result = solve_schedule(employees, rules, min_counts=counts, exclude=intent.exclude)
    explanation = build_schedule_explanation(result.schedule, employees, result.checks)

    print("=" * 60)
    print("自测样例（本地模式，未使用大模型）")
    print("=" * 60)
    print(f"用户输入：{user_input}")
    print(f"解析模式：{mode}")
    for n in notes:
        print(f"提示：{n}")
    print(f"解析结果：action={intent.action}，日期={intent.days}，排除={intent.exclude}")
    print(f"排班结果：{'完全合规' if result.feasible else '未完全合规'}")
    print("-" * 60)

    emp_map = {e.emp_id: e for e in employees}
    for day in DAYS:
        for shift in SHIFTS:
            ids = result.schedule.get(day, shift)
            labels = "、".join(f"{eid}（{emp_map[eid].position}）" for eid in ids)
            print(f"{day}{shift}（{len(ids)}人）：{labels}")
    print("-" * 60)

    print("规则合规报告：")
    for c in result.checks:
        mark = "✅" if c.status == "通过" else ("❌" if c.status == "违反" else "⚠️")
        print(f"  {mark} {c.rule_id} {c.status}（风险{c.risk_level}）：{c.attribution}")
    print("-" * 60)

    print("为什么这样安排（节选）：")
    for line in explanation["daily"].split("\n")[:3]:
        print(f"  {line}")
    print("  ……")


if __name__ == "__main__":
    main()

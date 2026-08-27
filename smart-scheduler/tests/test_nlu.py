"""自然语言解析测试：本地模板与降级路径。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.nlu as nlu
from agent.data_loader import employees_summary_text, rules_summary_text


def test_local_parse_saturday_late_six():
    intent = nlu.parse_local("周六晚班安排6人")
    assert intent.action == "generate"
    assert intent.days == ["周六"]
    assert intent.min_counts[("周六", "晚班")] == 6
    assert intent.min_counts[("周六", "早班")] == 6


def test_local_parse_weekday_range_and_early_count():
    intent = nlu.parse_local("周一到周五早班4人")
    assert intent.days == ["周一", "周二", "周三", "周四", "周五"]
    assert all(intent.min_counts[(d, "早班")] == 4 for d in intent.days)
    assert all(intent.min_counts[(d, "晚班")] == 4 for d in intent.days)


def test_local_parse_default_days():
    intent = nlu.parse_local("帮我排个班")
    assert intent.action == "generate"
    assert intent.days == nlu.DAYS
    assert intent.notes  # 有“未识别日期”提示


def test_local_parse_check_mode():
    text = "帮我检查一下这个排班\n周一早班：E01,E02,E03,E04\n周一晚班：E02,E07,E11,E18"
    intent = nlu.parse_local(text)
    assert intent.action == "check"
    assert intent.schedule_text == text


def test_local_parse_exclude():
    intent = nlu.parse_local("周一到周日排班，不要安排E03")
    assert intent.exclude == ["E03"]


def test_parse_schedule_text():
    schedule, notes = nlu.parse_schedule_text("周一早班：E01,E02,E03,E04\n周六晚班：E14,E05,E08,E10,E15,E16")
    assert schedule.get("周一", "早班") == ["E01", "E02", "E03", "E04"]
    assert schedule.get("周六", "晚班") == ["E14", "E05", "E08", "E10", "E15", "E16"]
    assert notes == []


def test_parse_schedule_text_unknown_token():
    schedule, notes = nlu.parse_schedule_text("周一早班：E01,E99")
    assert schedule.get("周一", "早班") == ["E01", "E99"]
    assert notes == []


def test_parse_schedule_text_no_lines():
    schedule, notes = nlu.parse_schedule_text("随便写点什么")
    assert not schedule.slots
    assert notes


def test_parse_intent_local_mode_without_key(monkeypatch):
    intent, mode, notes = nlu.parse_intent(
        "周六晚班6人",
        employees_summary_text(),
        rules_summary_text(),
        api_key=None,
        use_llm=True,
    )
    assert mode == "local"
    assert intent.min_counts[("周六", "晚班")] == 6


def test_parse_intent_llm_mode(monkeypatch):
    def fake_call_llm_json(prompt, api_key, base_url, model):
        return {
            "action": "generate",
            "days": ["周六"],
            "min_counts": {"周六": {"早班": 5, "晚班": 6}},
            "exclude": ["E03"],
            "schedule_text": None,
        }

    monkeypatch.setattr(nlu, "call_llm_json", fake_call_llm_json)
    intent, mode, notes = nlu.parse_intent(
        "周六早班5人晚班6人，不安排E03",
        employees_summary_text(),
        rules_summary_text(),
        api_key="sk-test",
        use_llm=True,
    )
    assert mode == "llm"
    assert intent.min_counts[("周六", "早班")] == 5
    assert intent.min_counts[("周六", "晚班")] == 6
    assert intent.exclude == ["E03"]


def test_parse_intent_llm_fallback_on_bad_json(monkeypatch):
    def bad_call_llm_json(*args, **kwargs):
        raise ValueError("bad json")

    monkeypatch.setattr(nlu, "call_llm_json", bad_call_llm_json)
    intent, mode, notes = nlu.parse_intent(
        "周六晚班6人",
        employees_summary_text(),
        rules_summary_text(),
        api_key="sk-test",
        use_llm=True,
    )
    assert mode == "local"
    assert notes
    assert intent.min_counts[("周六", "晚班")] == 6

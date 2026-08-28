"""自然语言意图解析。

大模型（DeepSeek，OpenAI 兼容接口）只负责把用户的话翻译成结构化参数，
不参与任何排班决策；API 不可用或返回非法内容时自动降级为本地模板解析。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import DAY_INDEX, DAYS, SHIFTS, Schedule

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class Intent:
    action: str                               # generate / check
    days: List[str]                           # 涉及日期（周几）
    min_counts: Dict[Tuple[str, str], int]    # (日期, 班次) -> 最低人数
    exclude: List[str]                        # 明确不安排的员工
    leave_updates: List[Dict[str, object]]    # 请假更新：[{"emp": "E03", "days": [...]}]
    schedule_text: Optional[str]              # check 模式下用户提供的排班文本
    raw: str
    notes: List[str] = field(default_factory=list)


def _default_min_count(day: str) -> int:
    """R-04 默认最低人数。"""
    return 6 if day in ("周六", "周日") else 4


def _norm_day(ch: str) -> str:
    return "日" if ch == "天" else ch


def _day_key(ch: str) -> str:
    """把「一/二/…/天」转换成「周一/周二/…/周日」。"""
    return "周" + _norm_day(ch)


_GEN_KEYWORDS = ["安排", "排一下", "生成", "帮我排", "排出"]
_CHECK_KEYWORDS = ["检查", "校验", "查一下", "是否合规", "合规吗", "合不合规", "冲突"]

_DAY_RANGE_RE = re.compile(r"周([一二三四五六日天])(?:到|至|－|—)周([一二三四五六日天])")
_DAY_MENTION_RE = re.compile(r"周([一二三四五六日天])")
_EARLY_COUNT_RE = re.compile(r"(?:早班|上午班|白班)\s*[:：为是]?\s*(\d+)\s*人")
_LATE_COUNT_RE = re.compile(r"(?:晚班|下午班|夜班)\s*[:：为是]?\s*(\d+)\s*人")
_BOTH_COUNT_RE = re.compile(r"(?:每班|每个班|各班)\s*[:：为是]?\s*(\d+)\s*人")
_EXCLUDE_RE = re.compile(
    r"(?:不要安排|别安排|不安排|不要排|不排|排除|别排|避开|去掉)\s*(E\d+)|(E\d+)\s*(?:不要安排|别安排|不安排|不要排|不排|排除|别排)"
)
_SCHEDULE_LINE_RE = re.compile(
    r"周([一二三四五六日天])\s*(早班|晚班)\s*(?:(\d+)\s*人)?\s*[:：]?\s*([A-Za-z0-9，,、\s]*[Ee]\d+[A-Za-z0-9，,、\s]*)"
)

_LEAVE_DAY_RE = re.compile(
    r"([Ee]\d+)\s*周([一二三四五六日天])\s*(?:请假|来不了|不能来|不上班|休息|有事)|"
    r"周([一二三四五六日天])\s*([Ee]\d+)\s*(?:请假|来不了|不能来|不上班|休息|有事)"
)
_LEAVE_WEEK_RE = re.compile(
    r"([Ee]\d+)\s*(?:这周|本周|下周)?\s*(?:请假|来不了|不能来|不上班|休息|有事)|"
    r"(?:请假|来不了|不能来|不上班|休息|有事)\s*(?:的是|的)?\s*([Ee]\d+)"
)

_MEANING_RE = re.compile(
    r"排班|安排|排一下|生成|帮我排|排出|检查|校验|查一下|合规|冲突|"
    r"周[一二三四五六日天]|\d+\s*人|[Ee]\d+|不要安排|不安排|排除|不排|别排|"
    r"班次|早班|晚班|员工|请假|来不了|不能来|不上班|休息|有事|本周|这周|下周"
)


def looks_meaningful(text: str) -> bool:
    """判断输入是否包含可识别的排班需求信号，避免无意义输入被默认排班。"""
    return bool(_MEANING_RE.search(text))


def _normalize_day(value: object) -> Optional[str]:
    s = str(value).strip()
    s = s.replace("星期", "周")
    if s.startswith("周") and len(s) == 2:
        return _day_key(s[1])
    if s in DAY_INDEX:
        return s
    if len(s) == 1 and s in "一二三四五六日天":
        return _day_key(s)
    return None


def _extract_days(text: str) -> Optional[List[str]]:
    m = _DAY_RANGE_RE.search(text)
    if m:
        a = DAY_INDEX[_day_key(m.group(1))]
        b = DAY_INDEX[_day_key(m.group(2))]
        if a <= b:
            return DAYS[a : b + 1]
    found: List[str] = []
    for ch in _DAY_MENTION_RE.findall(text):
        day = _day_key(ch)
        if day not in found:
            found.append(day)
    return found or None


def _looks_like_schedule(text: str) -> bool:
    return bool(_SCHEDULE_LINE_RE.search(text))


def _detect_action(text: str) -> str:
    has_gen = any(k in text for k in _GEN_KEYWORDS)
    has_check = any(k in text for k in _CHECK_KEYWORDS)
    if has_gen:
        return "generate"  # 生成后会自动做规则检查
    if _looks_like_schedule(text) or has_check:
        return "check"
    return "generate"


def _extract_leave_updates(text: str) -> List[Dict[str, object]]:
    """提取「E03 请假 / E03 这周来不了 / E03周三请假」这类信息。"""
    updates: Dict[str, List[str]] = {}
    for m in _LEAVE_DAY_RE.finditer(text):
        eid = (m.group(1) or m.group(4) or "").upper()
        day = _day_key(m.group(2) or m.group(3))
        if re.fullmatch(r"E\d+", eid):
            updates.setdefault(eid, []).append(day)
    for m in _LEAVE_WEEK_RE.finditer(text):
        eid = (m.group(1) or m.group(2) or "").upper()
        if not re.fullmatch(r"E\d+", eid):
            continue
        if eid not in updates:  # 已按具体日期登记的不再按整周处理
            updates[eid] = list(DAYS)
    return [{"emp": eid, "days": days} for eid, days in sorted(updates.items())]


def parse_schedule_text(text: str) -> Tuple[Schedule, List[str]]:
    """把「周一早班：E01,E02,E03,E04」这类文本解析为排班对象。

    返回 (排班, 注意事项)；无法识别的行会写入注意事项。
    """
    schedule = Schedule()
    notes: List[str] = []
    lines = re.split(r"[\n；;]+", text)
    found_any = False
    for line in lines:
        m = _SCHEDULE_LINE_RE.search(line)
        if not m:
            continue
        found_any = True
        day = _day_key(m.group(1))
        shift = m.group(2)
        body = m.group(4) or ""
        ids = [x.upper() for x in re.findall(r"[Ee]\d+", body)]
        ids = list(dict.fromkeys(ids))  # 去重保序
        if not ids:
            notes.append(f"「{day}{shift}」未提供员工 ID，无法判断")
        schedule.set(day, shift, ids)
    if not found_any:
        notes.append("未识别到「周X早班/晚班：E01,E02,…」格式的排班行")
    return schedule, notes


def parse_local(text: str) -> Intent:
    """本地模板解析：不依赖大模型，覆盖常用表述。"""
    notes: List[str] = []
    if not looks_meaningful(text):
        return Intent(
            action="clarify",
            days=[],
            min_counts={},
            exclude=[],
            leave_updates=[],
            schedule_text=None,
            raw=text,
            notes=["无法识别为排班需求，请补充排班相关描述"],
        )
    action = _detect_action(text)
    days = _extract_days(text)
    if days is None:
        days = list(DAYS)
        notes.append("未识别到具体日期，默认按周一至周日排班")

    min_counts: Dict[Tuple[str, str], int] = {}
    for day in days:
        for shift in SHIFTS:
            min_counts[(day, shift)] = _default_min_count(day)

    m_early = _EARLY_COUNT_RE.search(text)
    m_late = _LATE_COUNT_RE.search(text)
    m_both = _BOTH_COUNT_RE.search(text)
    if m_both:
        for day in days:
            for shift in SHIFTS:
                min_counts[(day, shift)] = int(m_both.group(1))
    if m_early:
        for day in days:
            min_counts[(day, "早班")] = int(m_early.group(1))
    if m_late:
        for day in days:
            min_counts[(day, "晚班")] = int(m_late.group(1))

    exclude: List[str] = []
    for m in _EXCLUDE_RE.finditer(text):
        eid = (m.group(1) or m.group(2) or "").upper()
        if eid and eid not in exclude:
            exclude.append(eid)

    leave_updates = _extract_leave_updates(text)

    schedule_text = text if action == "check" else None
    if action == "check" and not _looks_like_schedule(text):
        notes.append("检查模式未识别到排班行，请按「周一早班：E01,E02,E03,E04」格式提供")

    return Intent(
        action=action,
        days=days,
        min_counts=min_counts,
        exclude=exclude,
        leave_updates=leave_updates,
        schedule_text=schedule_text,
        raw=text,
        notes=notes,
    )


def _build_llm_prompt(text: str, employees_summary: str, rules_summary: str) -> str:
    schema = (
        '{"action": "generate 或 check", "days": ["周一", "周二"], '
        '"min_counts": {"周一": {"早班": 4, "晚班": 4}}, '
        '"exclude": ["E03"], '
        '"leave_updates": [{"emp": "E03", "days": ["周一", "周二"]}], '
        '"schedule_text": "仅 check 模式下填用户提供的排班原文，否则为 null"}'
    )
    return (
        "你是一个排班需求解析器。用户会输入一句自然语言，你需要提取结构化参数，"
        "只输出 JSON，不要输出任何其他文字或 Markdown。\n\n"
        + employees_summary
        + "\n\n"
        + rules_summary
        + "\n\n输出 JSON 字段说明：\n"
        + "- action：用户要求“安排/生成排班”则为 \"generate\"；要求“检查/校验已有排班”则为 \"check\"。\n"
        + "- days：涉及的星期列表（周一至周日），未提及则列出全部 7 天；用户说“周一到周五”就列出周一至周五。\n"
        + "- min_counts：仅当用户明确要求某班次人数时填写（例如“早班4人”），其余用默认值"
        + "（周一至周五每班 4 人，周六日每班 6 人）。键为日期，值为 {\"早班\": 人数, \"晚班\": 人数}。\n"
        + "- exclude：用户明确说“不安排/排除”的员工 ID 列表，如 [\"E03\"]；没有则为 []。\n"
        + "- leave_updates：用户提到某员工请假/来不了/休息/不上班时填写，days 为该员工请假的星期；"
        + "未写具体日期或说“这周/本周”时列出全部 7 天；没有则为 []。\n"
        + "- schedule_text：action 为 \"check\" 时，原样保留用户粘贴的排班文本；否则为 null。\n\n"
        + "输出格式示例：\n"
        + schema
        + "\n\n用户输入：\n"
        + text
        + "\n\n只输出 JSON。"
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("响应中未找到 JSON")
    return json.loads(text[start : end + 1])


def call_llm_json(
    prompt: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> dict:
    """调用 OpenAI 兼容的 chat/completions 接口并解析 JSON 响应。"""
    content = call_llm_text(prompt, api_key, base_url, model, timeout)
    return _extract_json(content)


def call_llm_text(
    prompt: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> str:
    """调用 OpenAI 兼容的 chat/completions 接口，返回纯文本。"""
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你是排班助手，请按要求输出内容。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _intent_from_llm(data: dict, raw: str) -> Intent:
    action = data.get("action")
    if action not in ("generate", "check"):
        raise ValueError(f"非法 action: {action!r}")

    days: List[str] = []
    for d in data.get("days", []) or []:
        day = _normalize_day(d)
        if day and day not in days:
            days.append(day)
    if not days:
        days = list(DAYS)

    min_counts: Dict[Tuple[str, str], int] = {}
    for day in days:
        for shift in SHIFTS:
            min_counts[(day, shift)] = _default_min_count(day)
    raw_counts = data.get("min_counts") or {}
    for day_key, shift_map in raw_counts.items():
        day = _normalize_day(day_key)
        if not day:
            continue
        for shift_key, val in (shift_map or {}).items():
            shift = str(shift_key).strip()
            if shift in ("早班", "上午班", "白班"):
                shift = "早班"
            elif shift in ("晚班", "下午班", "夜班"):
                shift = "晚班"
            else:
                continue
            try:
                min_counts[(day, shift)] = max(0, int(val))
            except (TypeError, ValueError):
                continue

    exclude: List[str] = []
    for eid in data.get("exclude", []) or []:
        eid = str(eid).strip().upper()
        if re.fullmatch(r"E\d+", eid) and eid not in exclude:
            exclude.append(eid)

    leave_updates: List[Dict[str, object]] = []
    for upd in data.get("leave_updates", []) or []:
        emp = str(upd.get("emp") or "").strip().upper()
        if not re.fullmatch(r"E\d+", emp):
            continue
        days_raw = upd.get("days")
        if not days_raw:
            days = list(DAYS)
        else:
            days = []
            for d in days_raw:
                nd = _normalize_day(d)
                if nd and nd not in days:
                    days.append(nd)
        leave_updates.append({"emp": emp, "days": days})

    schedule_text = None
    if action == "check" and data.get("schedule_text"):
        schedule_text = str(data["schedule_text"]).strip() or None

    return Intent(
        action=action,
        days=days,
        min_counts=min_counts,
        exclude=exclude,
        leave_updates=leave_updates,
        schedule_text=schedule_text,
        raw=raw,
    )


def parse_intent(
    text: str,
    employees_summary: str,
    rules_summary: str,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
) -> Tuple[Intent, str, List[str]]:
    """解析用户输入。返回 (意图, 解析模式["llm"/"local"], 注意事项)。"""
    if not looks_meaningful(text):
        intent = parse_local(text)
        return intent, "local", list(intent.notes)
    if use_llm and api_key:
        try:
            data = call_llm_json(_build_llm_prompt(text, employees_summary, rules_summary), api_key, base_url, model)
            intent = _intent_from_llm(data, text)
            return intent, "llm", []
        except Exception as exc:  # 网络、鉴权、非法 JSON 均降级
            notes = [f"大模型解析失败（{type(exc).__name__}），已自动切换本地解析模式"]
            intent = parse_local(text)
            return intent, "local", notes + intent.notes
    intent = parse_local(text)
    return intent, "local", list(intent.notes)

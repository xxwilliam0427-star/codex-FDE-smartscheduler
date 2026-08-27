"""智能排班助手 —— 对话框式（类豆包 / DeepSeek）Web 应用入口。

启动方式：streamlit run app.py
手机风格对话界面：底部输入框 + 「生成排班 / 检查排班」功能开关，
右上角可随时查看题目规则与员工数据。
排班与规则判断始终由规则引擎完成；大模型（可选）只负责理解自然语言与润色解释。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.data_loader import (
    employees_summary_text,
    load_employees,
    load_rules,
    rules_summary_text,
)
from agent.explainer import build_schedule_explanation
from agent.nlu import DEFAULT_BASE_URL, DEFAULT_MODEL, parse_intent, parse_schedule_text
from agent.scheduler import default_min_counts, solve_schedule
from agent.types import DAY_INDEX, DAYS, SHIFT_INDEX, SHIFTS, SHIFT_TIMES, Schedule
from agent.validator import validate_schedule

load_dotenv()

st.set_page_config(
    page_title="智能排班助手",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

EMPLOYEES = load_employees()
RULES = load_rules()
EMP_MAP = {e.emp_id: e for e in EMPLOYEES}

# 判断输入是否像「周X早班/晚班：E01,…」这样的排班文本
_SCHEDULE_LIKE_RE = re.compile(
    r"周[一二三四五六日天]\s*(?:早班|晚班)\s*[:：]?\s*(?:\d+\s*人)?\s*[:：]?\s*E\d+"
)

_POS_CLASS = {
    "店长": "emp-store",
    "副店长": "emp-vice",
    "值班主管": "emp-super",
    "高级店员": "emp-senior",
    "店员": "emp-clerk",
    "兼职": "emp-part",
}

_EXAMPLES = [
    (
        "📅 生成一周排班",
        "帮我安排周一到周日的排班，各班按规则最低人数安排",
        "生成排班",
    ),
    (
        "🎯 指定人数并排除员工",
        "帮我安排周一到周日的排班，周六晚班安排 6 人，不要安排 E03",
        "生成排班",
    ),
    (
        "🔍 检查已有排班",
        "周一早班：E01,E06,E09,E12\n周一晚班：E02,E07,E11,E18",
        "检查排班",
    ),
]

_HERO = """
<div class="hero">
  <div class="hero-logo">💬</div>
  <h1>智能排班助手</h1>
  <p>像聊天一样排班：一句自然语言即可生成排班，<br>也可以粘贴已有排班进行规则检查。</p>
  <p class="hero-sub">判断依据仅来自题目规则 R-01~R-09 与员工数据</p>
  <div class="hint">👇 试试下面的示例</div>
</div>
"""

_CSS = """
<style>
:root {
  --accent: #4f6ef7;
  --accent2: #7c5cf0;
  --bg: #eef1f8;
  --line: #eceef5;
  --text: #232837;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { display: none; }
#MainMenu, footer { visibility: hidden; }

/* 手机式固定布局：顶栏固定、消息区内部滚动、输入区吸底 */
html, body {
  height: 100% !important;
  font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", "Source Sans", sans-serif;
}
[data-testid="stApp"] { overflow: hidden !important; }
[data-testid="stAppViewContainer"] { overflow: hidden !important; }
[data-testid="stAppViewContainer"] > div { height: 100dvh !important; overflow: hidden !important; }
[data-testid="stMain"] {
  height: 100dvh !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
.block-container {
  max-width: 430px;
  height: 100dvh !important;
  min-height: 0 !important;
  padding: 0.75rem 0.8rem 0.35rem;
  background: #ffffff;
  box-shadow: 0 0 60px rgba(30, 55, 120, 0.18), 0 0 0 1px rgba(30, 55, 120, 0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.block-container > [data-testid="stVerticalBlock"] {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
div[data-testid="stLayoutWrapper"]:has(> .st-key-messages_area) {
  flex: 1 1 0% !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column;
  overflow: hidden;
}
div[data-testid="stLayoutWrapper"]:has(> .st-key-input_area) {
  flex: 0 0 auto !important;
}
/* 窄屏下列默认 min-width 为整行宽度，这里恢复按比例排列 */
[data-testid="stColumn"] { min-width: 0 !important; }
.st-key-messages_area { flex: 1; min-height: 0; overflow-y: auto; scrollbar-width: thin; padding: 4px 2px 8px; }
.st-key-input_area { flex: none; }

/* 顶栏 */
.app-title { font-size: 17px; font-weight: 800; color: #1f2430; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.pill.on { background: #dcfce7; color: #15803d; }
.pill.off { background: #f1f5f9; color: #64748b; }

/* 按钮通用 */
div[data-testid="stButton"] > button {
  border-radius: 12px; border: 1px solid #e6e8f0; background: #fff; color: #4b5563;
  font-weight: 500; min-height: 36px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  transition: all 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color: #c7d2fe; color: var(--accent); background: #f5f7ff; transform: translateY(-1px);
}

/* 聊天气泡 */
[data-testid="stChatMessage"] {
  padding: 4px 0 16px; align-items: flex-start;
  background: transparent !important;
}
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] + [data-testid="stElementContainer"] {
  margin-top: 8px;
}
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] [data-testid="stExpander"] {
  margin-top: 8px;
  border: none !important;
  background: transparent !important;
}

/* 思考气泡 */
.thinking-bubble {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 14px; color: #6b7280; padding: 6px 4px;
}
.tdot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  animation: dotPulse 1.2s infinite ease-in-out;
}
.tdot:nth-child(2) { animation-delay: 0.15s; }
.tdot:nth-child(3) { animation-delay: 0.3s; }
@keyframes dotPulse {
  0%, 100% { opacity: 0.3; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(-3px); }
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"] > div:first-child {
  width: 30px; height: 30px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) > div:first-child {
  background: #eef0f6 !important;
}
[data-testid="stChatMessageContent"] {
  width: fit-content !important; max-width: 84% !important; height: auto !important;
  border-radius: 16px !important; padding: 10px 13px !important;
  font-size: 14px; line-height: 1.65; word-break: break-word;
}
[data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"],
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"] { height: auto !important; }
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
  background: var(--accent) !important;
  color: #fff !important; border-radius: 16px 16px 4px 16px !important;
  box-shadow: 0 4px 14px rgba(79, 110, 247, 0.25);
}
[data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {
  background: #fff !important; border: 1px solid #eef0f6 !important;
  border-radius: 16px 16px 16px 4px !important; color: #232837;
  box-shadow: 0 2px 8px rgba(30, 55, 120, 0.05);
}
[data-testid="stChatMessageContent"] p {
  margin: 0 0 4px;
  font-size: 14px !important;
  line-height: 1.65 !important;
}
[data-testid="stChatMessageContent"] .stMarkdown > div {
  align-items: flex-start !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] strong,
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] p,
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] div { color: #fff; }

/* 输入区：圆角卡片 */
.st-key-input_area {
  background: #ffffff;
  border: 1px solid var(--line); border-radius: 20px;
  padding: 10px 12px 12px;
  box-shadow: 0 -2px 18px rgba(26, 48, 110, 0.05), 0 6px 24px rgba(26, 48, 110, 0.10);
  margin-top: 6px;
}
.st-key-input_area textarea {
  border: none !important; background: transparent !important; box-shadow: none !important;
  resize: none; font-size: 15px; line-height: 1.55;
}
[data-testid="stTextArea"] { background: transparent; }
[data-testid="stTextArea"] label { display: none !important; }

/* 隐藏功能开关的“功能”标签 */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display: none !important; }

/* 功能开关（生成 / 检查）——胶囊式单选 */
[data-testid="stRadio"] {
  background: #eef0f6; border-radius: 999px; padding: 3px;
  height: 38px; display: flex; align-items: center;
}
[data-testid="stRadio"] [role="radiogroup"] {
  gap: 3px; width: 100%; min-height: 0 !important; padding: 0 !important;
}
[data-testid="stRadio"] label {
  border-radius: 999px; font-size: 13px; font-weight: 500;
  padding: 0 12px !important; height: 32px; background: transparent; color: #64748b;
  margin: 0; flex: 1; justify-content: center; min-height: 0 !important;
  display: flex; align-items: center;
  transition: all 0.15s ease;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: #eaf0ff !important;
  color: var(--accent) !important;
  font-weight: 700;
  box-shadow: inset 0 0 0 1px rgba(79, 110, 247, 0.25), 0 2px 8px rgba(79, 110, 247, 0.15);
}
[data-testid="stRadio"] input[type="radio"] { display: none; }
[data-testid="stRadio"] label > div:first-child { display: none; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { margin: 0; }

/* 发送按钮 */
div[data-testid="stElementContainer"].st-key-send_btn button {
  background: var(--accent);
  color: #fff; border: none; border-radius: 999px; font-weight: 700; min-height: 38px;
  box-shadow: 0 4px 14px rgba(79, 110, 247, 0.30);
}
div[data-testid="stElementContainer"].st-key-send_btn button:hover {
  filter: brightness(1.06); color: #fff; transform: none;
}

/* 排班表卡片 */
.sched-card {
  border: 1px solid #e8eaf2; border-radius: 14px; overflow: hidden;
  margin: 6px 0 2px; background: #fff; box-shadow: 0 2px 12px rgba(26, 48, 110, 0.07);
}
.sched-head {
  display: flex; justify-content: space-between; align-items: center;
  background: var(--accent);
  color: #fff; padding: 8px 12px; font-weight: 800; font-size: 14px;
}
.sched-badge {
  background: rgba(255, 255, 255, 0.22); border-radius: 999px;
  padding: 2px 9px; font-size: 11px; font-weight: 600;
}
table.sched-table { width: 100%; border-collapse: collapse; }
.sched-table th {
  font-size: 12px; color: #6b7280; padding: 8px 6px; text-align: left;
  background: #fafbfe; border-bottom: 1px solid #eef0f6; font-weight: 600;
}
.sched-table td { padding: 7px 6px; border-bottom: 1px solid #f2f3f8; vertical-align: top; }
.sched-table tr.weekend td { background: #fffaf0; }
.sched-table tr:last-child td { border-bottom: none; }
.day { font-weight: 800; color: #334155; font-size: 13px; white-space: nowrap; }
tr.weekend .day { color: #d97706; }
.shift-tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.shift-early { background: #fff3e0; color: #d97706; }
.shift-late { background: #eef2ff; color: #4f46e5; }
.emp { display: inline-block; margin: 0 3px 3px 0; padding: 2px 7px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.emp-store { background: #fee2e2; color: #b91c1c; }
.emp-vice { background: #fce7f3; color: #be185d; }
.emp-super { background: #ede9fe; color: #6d28d9; }
.emp-senior { background: #dbeafe; color: #1d4ed8; }
.emp-clerk { background: #f1f5f9; color: #334155; }
.emp-part { background: #ccfbf1; color: #0f766e; }
.emp-unknown { background: #fef3c7; color: #b45309; }
.meta { margin-top: 3px; font-size: 11px; color: #8a93a6; }
.empty { color: #c4c9d4; font-size: 12px; }
.sched-foot {
  padding: 6px 12px; font-size: 11px; color: #9aa3b2;
  background: #fafbfe; border-top: 1px solid #eef0f6;
}
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 3px 0 8px; }

/* 逐日说明卡片（左右滑动） */
.dc-scroll {
  display: flex; gap: 10px; overflow-x: auto;
  padding: 4px 2px 8px; scroll-snap-type: x mandatory; scrollbar-width: thin;
}
.dc-card {
  flex: 0 0 76%; scroll-snap-align: center;
  border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px;
  background: #fff; box-shadow: 0 2px 10px rgba(30, 55, 120, 0.05);
}
.dc-card.weekend { background: #fffaf0; border-color: #f3e2bd; }
.dc-day { font-weight: 800; font-size: 15px; color: var(--text); margin-bottom: 7px; }
.dc-shift + .dc-shift { margin-top: 8px; border-top: 1px dashed var(--line); padding-top: 8px; }
.dc-shift-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.dc-shift-name { font-size: 12px; font-weight: 700; color: #6b7280; }
.dc-count {
  font-size: 11px; font-weight: 700; color: var(--accent);
  background: #eef2ff; border-radius: 999px; padding: 1px 8px;
}
.dc-meta { font-size: 11px; color: #8a93a6; margin-top: 2px; }

/* 规则合规卡片 */
.rule-card { border: 1px solid #e8eaf2; border-radius: 12px; overflow: hidden; margin: 6px 0 2px; }
.rule-item { display: flex; gap: 8px; padding: 8px 10px; border-bottom: 1px solid #f2f3f8; font-size: 13px; }
.rule-item:last-child { border-bottom: none; }
.rule-item.fail { background: #fff8f8; }
.rule-item.unknown { background: #fffcf5; }
.rule-icon { font-size: 14px; line-height: 1.4; }
.rule-body { flex: 1; min-width: 0; }
.rule-title { color: #334155; line-height: 1.5; }
.rule-title b { color: #1f2430; }
.rule-status { float: right; margin-left: 6px; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.rule-status.pass { background: #dcfce7; color: #15803d; }
.rule-status.fail { background: #fee2e2; color: #b91c1c; }
.rule-status.unknown { background: #fef3c7; color: #b45309; }
.rule-detail { font-size: 12px; color: #6b7280; margin-top: 3px; line-height: 1.55; }

/* 欢迎页 */
.hero { text-align: center; padding: 30px 8px 12px; }
.hero-logo {
  width: 64px; height: 64px; margin: 0 auto 14px;
  border-radius: 20px;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 30px;
  box-shadow: 0 8px 20px rgba(79, 110, 247, 0.25);
}
.hero h1 { font-size: 21px; color: var(--text); margin: 0 0 8px; letter-spacing: 0.2px; }
.hero p { color: #8a93a6; font-size: 13px; line-height: 1.7; margin: 0 0 4px; }
.hero .hero-sub { color: #b3bac7; font-size: 12px; margin-top: 8px; }
.hero .hint { color: #b3bac7; font-size: 12px; margin-top: 20px; }

/* 示例按钮：浅色圆角卡片 */
div[data-testid="stElementContainer"][class*="st-key-example_"] button {
  background: #f7f8fd;
  border: 1px solid #e8ebf8;
  border-radius: 14px;
  padding: 10px 12px;
  min-height: 44px;
  text-align: left;
}
div[data-testid="stElementContainer"][class*="st-key-example_"] button:hover {
  background: #eef2ff;
  border-color: #c7d2fe;
}

/* 下载按钮 */
div[data-testid="stDownloadButton"] > button {
  border-radius: 10px; border: 1px solid #d8defc; background: #f5f7ff;
  color: var(--accent); font-weight: 600; font-size: 13px;
}

/* 数据对话框：豆包风格底部抽屉 */
[data-testid="stDialog"] {
  background: rgba(15, 23, 42, 0.45) !important;
  backdrop-filter: blur(2px);
}
[data-testid="stDialog"] > div {
  position: fixed !important;
  bottom: 0 !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: min(430px, 100vw) !important;
  max-width: 100vw !important;
  max-height: 82dvh !important;
  margin: 0 !important;
  overflow: hidden !important;
  border-radius: 24px 24px 0 0 !important;
  border: none !important;
  border-top: 1px solid var(--line) !important;
  box-shadow: 0 -12px 40px rgba(16, 24, 40, 0.18) !important;
}
[data-testid="stDialog"] > div::before {
  content: "";
  display: block;
  width: 36px;
  height: 4px;
  border-radius: 999px;
  background: #dfe3ee;
  margin: 10px auto 2px;
  flex-shrink: 0;
}
[data-testid="stDialog"] section {
  display: flex !important;
  flex-direction: column;
  min-height: 0;
  overflow: hidden !important;
}
[data-testid="stDialog"] section > div:last-child {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow-y: auto !important;
}
[data-testid="stDialog"] h2 {
  font-size: 16px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  padding: 18px 20px 10px !important;
  margin: 0 !important;
}
[data-testid="stDialog"] button[aria-label="Close"] {
  width: 30px !important;
  height: 30px !important;
  border-radius: 50% !important;
  background: #f1f3f9 !important;
  border: none !important;
  color: #6b7280 !important;
  display: flex !important;
  align-items: center;
  justify-content: center;
}
[data-testid="stDialog"] button[aria-label="Close"]:hover {
  background: #e8ebf5 !important;
  color: var(--accent) !important;
}
[data-testid="stDialog"] section > button[aria-label="Close"] {
  top: 16px !important;
  right: 16px !important;
}
[data-testid="stDialog"] [data-testid="stDataFrame"] { width: 100% !important; }

/* 头部图标按钮统一样式 */
div[data-testid="stElementContainer"].st-key-new_chat button,
div[data-testid="stElementContainer"].st-key-data_btn button {
  width: 40px !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 !important;
  border-radius: 14px !important;
  border: 1px solid var(--line) !important;
  background: #fff !important;
  color: #4b5563 !important;
  display: flex !important;
  align-items: center;
  justify-content: center;
  box-shadow: none !important;
}
div[data-testid="stElementContainer"].st-key-new_chat button:hover,
div[data-testid="stElementContainer"].st-key-data_btn button:hover {
  background: #f5f7ff !important;
  border-color: #c7d2fe !important;
  color: var(--accent) !important;
}
</style>
"""

if "messages" not in st.session_state:
    st.session_state.messages = []


def _emp_chip(eid: str) -> str:
    emp = EMP_MAP.get(eid)
    if emp is None:
        return f'<span class="emp emp-unknown" title="{eid} 数据外">{eid}</span>'
    cls = _POS_CLASS.get(emp.position, "emp-unknown")
    return f'<span class="emp {cls}" title="{eid} · {emp.position}">{eid}</span>'


def _skill_counts(emp_ids: List[str]) -> Tuple[int, int, int, int]:
    manager = drink = cashier = 0
    for eid in emp_ids:
        emp = EMP_MAP.get(eid)
        if emp is None:
            continue
        manager += 1 if "店长值守" in emp.skills else 0
        drink += 1 if "饮品制作" in emp.skills else 0
        cashier += 1 if "收银" in emp.skills else 0
    return len(emp_ids), manager, drink, cashier


def _build_schedule_html(schedule: Schedule, title: str = "本周排班表") -> str:
    head = f"<span>📅 {title}</span><span class=\"sched-badge\">7 天 × 2 班</span>"
    thead = (
        "<tr><th>日期</th>"
        f'<th><span class="shift-tag shift-early">早班 {SHIFT_TIMES["早班"]}</span></th>'
        f'<th><span class="shift-tag shift-late">晚班 {SHIFT_TIMES["晚班"]}</span></th></tr>'
    )
    rows: List[str] = []
    for day in DAYS:
        cells: List[str] = []
        for shift in SHIFTS:
            ids = schedule.get(day, shift)
            if not ids:
                cells.append('<span class="empty">—</span>')
                continue
            chips = "".join(_emp_chip(eid) for eid in ids)
            n, manager, drink, cashier = _skill_counts(ids)
            cells.append(
                f'<div class="emps">{chips}</div>'
                f'<div class="meta">{n} 人 · 👔店长 {manager} · 🥤饮品 {drink} · 💰收银 {cashier}</div>'
            )
        cls = ' class="weekend"' if day in ("周六", "周日") else ""
        rows.append(f"<tr{cls}><td class=\"day\">{day}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
    legend = "".join(
        f'<span class="legend"><i class="dot {cls}"></i>{pos}</span>' for pos, cls in _POS_CLASS.items()
    )
    return (
        '<div class="sched-card">'
        f'<div class="sched-head">{head}</div>'
        f'<table class="sched-table"><thead>{thead}</thead><tbody>{"".join(rows)}</tbody></table>'
        f'<div class="sched-foot">{legend}</div></div>'
    )


def _build_day_cards_html(schedule: Schedule) -> str:
    """把每周排班做成一天一张卡片，可左右滑动查看。"""
    cards: List[str] = []
    for day in DAYS:
        shifts_html: List[str] = []
        for shift in SHIFTS:
            ids = schedule.get(day, shift)
            if not ids:
                shifts_html.append(
                    f'<div class="dc-shift"><div class="dc-shift-head">'
                    f'<span class="dc-shift-name">{shift} {SHIFT_TIMES[shift]}</span>'
                    f'<span class="dc-count">未安排</span></div></div>'
                )
                continue
            chips = "".join(_emp_chip(eid) for eid in ids)
            n, manager, drink, cashier = _skill_counts(ids)
            shifts_html.append(
                f'<div class="dc-shift">'
                f'<div class="dc-shift-head">'
                f'<span class="dc-shift-name">{shift} {SHIFT_TIMES[shift]}</span>'
                f'<span class="dc-count">{n} 人</span>'
                f'</div>'
                f'<div class="dc-emps">{chips}</div>'
                f'<div class="dc-meta">店长 {manager} · 饮品 {drink} · 收银 {cashier}</div>'
                f'</div>'
            )
        cls = " weekend" if day in ("周六", "周日") else ""
        cards.append(f'<div class="dc-card{cls}"><div class="dc-day">📅 {day}</div>{"".join(shifts_html)}</div>')
    return f'<div class="dc-scroll">{"".join(cards)}</div>'


def _build_rule_html(checks) -> str:
    items: List[str] = []
    for c in checks:
        if c.status == "通过":
            cls, icon = "pass", "✅"
        elif c.status == "违反":
            cls, icon = "fail", "❌"
        else:
            cls, icon = "unknown", "⚠️"
        detail = ""
        if c.status != "通过":
            parts = []
            if c.attribution:
                parts.append(c.attribution)
            if c.involved:
                parts.append(f"涉及：{'、'.join(c.involved)}")
            if c.suggestion:
                parts.append(f"建议：{c.suggestion}")
            if parts:
                detail = f'<div class="rule-detail">{"；".join(parts)}</div>'
        items.append(
            f'<div class="rule-item {cls}"><span class="rule-icon">{icon}</span>'
            f'<div class="rule-body"><div class="rule-title"><b>{c.rule_id}</b> {c.description}'
            f'<span class="rule-status {cls}">{c.status}</span></div>{detail}</div></div>'
        )
    return f'<div class="rule-card">{"".join(items)}</div>'


def _schedule_csv(schedule: Schedule) -> bytes:
    rows: List[Dict[str, object]] = []
    for day in DAYS:
        for shift in SHIFTS:
            ids = schedule.get(day, shift)
            rows.append(
                {
                    "日期": day,
                    "班次": shift,
                    "时间段": SHIFT_TIMES[shift],
                    "人数": len(ids),
                    "员工": "、".join(ids),
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def _build_explanation_sections(schedule: Schedule, checks, explanation: Dict[str, str]) -> Dict[str, str]:
    """把解释内容整理成易读的卡片与列表。"""
    daily = _build_day_cards_html(schedule)

    rule_lines: List[str] = []
    for c in checks:
        icon = "✅" if c.status == "通过" else ("❌" if c.status == "违反" else "⚠️")
        rule_lines.append(f"- {icon} **{c.rule_id}**：{c.attribution}")

    why_lines = [f"- {ln}" for ln in explanation["why_not"].split("\n") if ln.strip()]

    return {
        "daily": daily,
        "rules": "**规则依据**\n\n" + "\n".join(rule_lines),
        "why_not": "**员工情况说明**\n\n" + "\n".join(why_lines) if why_lines else "**员工情况说明**\n\n所有员工均已按规则参与排班。",
    }


def _rules_reply_content() -> str:
    lines = [f"- **{r.rule_id}**　{r.description}" for r in RULES]
    return (
        "**📋 排班规则（来自题目文档）**\n\n"
        + "\n".join(lines)
        + "\n\n> 所有判断均以这些规则为准，Agent 不会自行新增或修改规则。"
    )


def _employees_reply_content() -> str:
    header = "| 员工 | 岗位 | 可工作 | 请假 | 偏好 |\n|---|---|---|---|---|"
    rows = []
    for e in EMPLOYEES:
        avail = "全周" if len(e.available_days) == 7 else "、".join(e.available_days)
        leave = "、".join(e.leave_days) if e.leave_days else "无"
        pref = e.preference or "—"
        rows.append(f"| {e.emp_id} | {e.position} | {avail} | {leave} | {pref} |")
    return (
        "**👥 员工数据（20 人）**\n\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n\n> 数据来自题目 todo.docx，技能以员工数据为准。"
    )


def _help_reply_content() -> str:
    return (
        "**💡 使用方法**\n\n"
        "- 在下方输入框直接输入需求，例如：`帮我安排周一到周日的排班，各班按规则最低人数安排`\n"
        "- 输入框下方的功能开关可切换「生成排班」或「检查排班」\n"
        "- 「检查排班」可直接粘贴已有排班，每行一个班次：`周一早班：E01,E02,E03,E04`\n"
        "- 支持指定人数与排除员工：`周六晚班安排 6 人，不要安排 E03`\n"
        "- 右上角「📊 数据」可随时查看规则与员工数据\n\n"
        "所有排班与规则判断均由本地规则引擎完成，依据仅来自题目规则 R-01~R-09 与员工数据。"
    )


def _detect_special_query(text: str) -> Optional[dict]:
    if any(k in text for k in ("怎么用", "如何使用", "使用说明", "帮助")):
        return {"role": "assistant", "kind": "text", "content": _help_reply_content()}
    if "规则" in text and "员工" in text:
        return {
            "role": "assistant",
            "kind": "text",
            "content": _rules_reply_content() + "\n\n---\n\n" + _employees_reply_content(),
        }
    if any(k in text for k in ("有哪些规则", "规则有哪些", "规则是什么", "查看规则", "规则列表", "看看规则")):
        return {"role": "assistant", "kind": "text", "content": _rules_reply_content()}
    if any(k in text for k in ("员工名单", "员工列表", "查看员工", "员工数据", "有哪些员工", "都有哪些员工")):
        return {"role": "assistant", "kind": "text", "content": _employees_reply_content()}
    if text.strip() in ("数据", "查看数据"):
        return {
            "role": "assistant",
            "kind": "text",
            "content": _rules_reply_content() + "\n\n---\n\n" + _employees_reply_content(),
        }
    return None


def _generate_reply(text: str) -> dict:
    intent, parse_mode, notes = parse_intent(
        text,
        employees_summary_text(),
        rules_summary_text(),
        os.getenv("DEEPSEEK_API_KEY", "") or None,
        os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        use_llm=True,
    )
    notes = list(notes)
    notes.insert(0, "✅ 已用大模型解析需求（仅理解意图，排班与规则判断由规则引擎完成）" if parse_mode == "llm" else "ℹ️ 本地解析模式（未使用大模型），排班与规则判断由规则引擎完成")

    counts = default_min_counts()
    for (day, shift), n in intent.min_counts.items():
        counts[(DAY_INDEX[day], SHIFT_INDEX[shift])] = n

    result = solve_schedule(EMPLOYEES, RULES, min_counts=counts, exclude=intent.exclude)

    summary = ("✅ " if result.feasible else "⚠️ ") + result.message
    scope = "、".join(intent.days)
    exclude = "、".join(intent.exclude) if intent.exclude else "无"
    summary += f"\n\n📅 排班范围：{scope}　｜　🚫 排除员工：{exclude}"

    explanation = build_schedule_explanation(result.schedule, EMPLOYEES, result.checks)
    sections = _build_explanation_sections(result.schedule, result.checks, explanation)

    return {
        "role": "assistant",
        "kind": "generate",
        "summary": summary,
        "notes": notes,
        "table": _build_schedule_html(result.schedule),
        "checks": _build_rule_html(result.checks),
        "daily": sections["daily"],
        "rules": sections["rules"],
        "why_not": sections["why_not"],
        "csv": _schedule_csv(result.schedule),
    }


def _check_reply(text: str, auto_switched: bool) -> dict:
    schedule, notes = parse_schedule_text(text)
    notes = list(notes)
    if auto_switched:
        notes.insert(0, "检测到排班文本，已自动切换为「检查排班」")

    checks = validate_schedule(schedule, EMPLOYEES, RULES)
    bad = [c for c in checks if c.status == "违反"]
    unknown = [c for c in checks if c.status == "无法判断"]
    if bad:
        summary = f"❌ 发现 {len(bad)} 条规则冲突，请按整改建议调整。"
    elif unknown:
        summary = f"⚠️ 部分规则因信息不足无法判断（{len(unknown)} 条），建议人工复核。"
    else:
        summary = "✅ 全部规则检查通过。"
    if not schedule.slots:
        summary += "\n\n未识别到「周X早班/晚班：员工ID」格式的排班行，请按格式重新粘贴。"

    return {
        "role": "assistant",
        "kind": "check",
        "summary": summary,
        "notes": notes,
        "table": _build_schedule_html(schedule, "已解析的排班") if schedule.slots else None,
        "checks": _build_rule_html(checks),
    }


def _process_reply(text: str, toggle: str) -> dict:
    """根据输入与功能开关生成助手回复（不直接操作消息列表）。"""
    text = text.strip()
    special = _detect_special_query(text)
    if special:
        return special

    auto_switched = False
    if toggle == "生成排班" and _SCHEDULE_LIKE_RE.search(text):
        toggle = "检查排班"
        auto_switched = True

    if toggle == "检查排班":
        return _check_reply(text, auto_switched)
    return _generate_reply(text)


def _render_generate(msg: dict, idx: int) -> None:
    st.markdown(msg["summary"])
    for n in msg.get("notes", []):
        st.caption(n)
    st.markdown(msg["table"], unsafe_allow_html=True)
    st.download_button(
        "⬇️ 下载排班表 CSV",
        data=msg["csv"],
        file_name="排班表.csv",
        mime="text/csv",
        key=f"csv_{idx}",
        use_container_width=True,
    )
    st.markdown("**📋 规则合规报告**")
    st.markdown(msg["checks"], unsafe_allow_html=True)
    with st.expander("📝 为什么这样安排"):
        st.markdown(msg["daily"], unsafe_allow_html=True)
        st.markdown(msg["rules"])
        st.markdown(msg["why_not"])


def _render_check(msg: dict, idx: int) -> None:
    st.markdown(msg["summary"])
    for n in msg.get("notes", []):
        st.caption(n)
    st.markdown("**📋 规则合规报告**")
    st.markdown(msg["checks"], unsafe_allow_html=True)
    if msg.get("table"):
        st.markdown(msg["table"], unsafe_allow_html=True)


def _render_messages() -> None:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🙂"):
                st.markdown(msg["content"].replace("\n", "  \n"))
        else:
            with st.chat_message("assistant", avatar="📅"):
                kind = msg.get("kind", "text")
                if kind == "thinking":
                    st.markdown(
                        '<div class="thinking-bubble">'
                        '<span class="tdot"></span><span class="tdot"></span><span class="tdot"></span>'
                        "正在思考</div>",
                        unsafe_allow_html=True,
                    )
                elif kind == "text":
                    st.markdown(msg["content"])
                elif kind == "generate":
                    _render_generate(msg, idx)
                else:
                    _render_check(msg, idx)


def _render_welcome() -> Optional[Tuple[str, str]]:
    st.markdown(_HERO, unsafe_allow_html=True)
    for i, (label, text, toggle) in enumerate(_EXAMPLES, start=1):
        if st.button(label, key=f"example_{i}", use_container_width=True):
            return text, toggle
    return None


def _render_data_panel() -> None:
    st.markdown("**📋 排班规则 R-01~R-09**")
    for r in RULES:
        st.markdown(f"- **{r.rule_id}**　{r.description}")
    st.divider()
    st.markdown("**👥 员工（20 人）**")
    rows = [
        {
            "员工": e.emp_id,
            "岗位": e.position,
            "可工作": "全周" if len(e.available_days) == 7 else "、".join(e.available_days),
            "请假": "、".join(e.leave_days) if e.leave_days else "无",
            "偏好": e.preference or "—",
        }
        for e in EMPLOYEES
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", height=300, hide_index=True)
    st.caption("数据来自题目文档 todo.docx，技能以员工数据为准。")


@st.dialog("📊 数据", width="small")
def _show_data_dialog() -> None:
    _render_data_panel()


def _render_header() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    c1, c2, c3 = st.columns([6, 1, 1], vertical_alignment="center")
    with c1:
        pill = '<span class="pill on">✨ DeepSeek 在线</span>' if api_key else '<span class="pill off">🔌 本地解析</span>'
        st.markdown(
            f'<div class="app-title">💬 智能排班助手</div><div style="margin-top:3px">{pill}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("", key="new_chat", help="开启新对话", icon=":material/chat_bubble:", use_container_width=True):
            st.session_state.messages = []
    with c3:
        if st.button("", key="data_btn", help="数据与规则", icon=":material/analytics:", use_container_width=True):
            _show_data_dialog()


def _render_input_bar() -> Tuple[bool, str, str]:
    # 提交后延迟清空输入框（组件实例化前重置，避免直接修改已实例化组件的状态）
    if st.session_state.pop("clear_input", False):
        st.session_state.pop("chat_text", None)

    current_mode = st.session_state.get("chat_mode", st.session_state.get("last_mode", "生成排班"))
    placeholder = (
        "例如：帮我安排周一到周日的排班，各班按规则最低人数安排"
        if current_mode == "生成排班"
        else "粘贴排班文本，每行一个班次：\n周一早班：E01,E02,E03,E04\n周一晚班：E02,E07,E11,E18"
    )
    text = st.text_area(
        "输入",
        key="chat_text",
        height=64,
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        mode = st.radio(
            "功能",
            options=["生成排班", "检查排班"],
            format_func=lambda x: {"生成排班": "📝 生成排班", "检查排班": "🔍 检查排班"}[x],
            index=0 if st.session_state.get("last_mode", "生成排班") == "生成排班" else 1,
            key="chat_mode",
            horizontal=True,
            label_visibility="hidden",
        )
    with c2:
        sent = st.button("发送", key="send_btn", icon=":material/send:", use_container_width=True)
    if mode == "检查排班":
        st.caption("支持多行粘贴，每行格式：周X早班/晚班：员工ID（如：周一早班：E01,E02,E03,E04）")
    return sent, text, mode


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    _render_header()

    messages_area = st.container(key="messages_area")
    input_area = st.container(key="input_area")

    with input_area:
        sent, text, mode = _render_input_bar()

    with messages_area:
        if sent:
            st.session_state["last_mode"] = mode
            st.session_state.messages.append({"role": "user", "content": text.strip()})
            st.session_state.messages.append({"role": "assistant", "kind": "thinking"})
            _render_messages()
            reply = _process_reply(text, mode)
            st.session_state.messages[-1] = reply
            st.session_state["clear_input"] = True
            st.rerun()
        elif st.session_state.messages:
            _render_messages()
        else:
            clicked = _render_welcome()
            if clicked:
                text, mode = clicked
                st.session_state["last_mode"] = mode
                st.session_state.messages.append({"role": "user", "content": text.strip()})
                st.session_state.messages.append({"role": "assistant", "kind": "thinking"})
                _render_messages()
                reply = _process_reply(text, mode)
                st.session_state.messages[-1] = reply
                st.rerun()


if __name__ == "__main__":
    main()

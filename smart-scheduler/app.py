"""智能排班助手 —— 对话框式（类豆包 / DeepSeek）Web 应用入口。

启动方式：streamlit run app.py
手机风格对话界面：底部输入框 + 「生成排班 / 检查排班」功能开关，
右上角可随时查看题目规则与员工数据。
排班与规则判断始终由规则引擎完成；大模型（可选）只负责理解自然语言与润色解释。
"""

from __future__ import annotations

import copy
import os
import re
import time
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
from agent.nlu import DEFAULT_BASE_URL, DEFAULT_MODEL, looks_meaningful, parse_intent, parse_schedule_text
from agent.scheduler import default_min_counts, solve_schedule
from agent.types import DAY_INDEX, DAYS, SHIFT_INDEX, SHIFTS, SHIFT_TIMES, Schedule
from agent.validator import validate_schedule

load_dotenv()

st.set_page_config(
    page_title="智能排班助手",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
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
  <p class="hero-sub">判断依据仅来自题目规则 R-01～R-09 与员工数据</p>
  <div class="hint">👇 试试下面的示例</div>
</div>
"""

_CSS = """
<style>
/* ============================================================
   设计系统 Design Tokens
   ============================================================ */
:root {
  /* 品牌色（2 主色 + 1 强调色） */
  --primary: #4d6bfe;
  --primary-600: #3f5be8;
  --primary-700: #334bd4;
  --primary-soft: #eef2ff;
  --accent: #8b5cf6;
  --accent-soft: #f3eefe;

  /* 中性灰阶 */
  --gray-100: #f8f9fa;
  --gray-200: #e9ecef;
  --gray-300: #ced4da;
  --gray-400: #adb5bd;
  --gray-500: #6c757d;
  --gray-900: #212529;

  --text: #212529;
  --text-2: #6c757d;
  --text-3: #adb5bd;
  --line: rgba(0, 0, 0, 0.08);
  --line-strong: rgba(0, 0, 0, 0.14);
  --surface: rgba(255, 255, 255, 0.82);

  /* 状态色板（浅底 + 深字） */
  --success: #1a9e66; --success-bg: #e6f7ef; --success-line: rgba(26, 158, 102, 0.28);
  --warning: #b45309; --warning-bg: #fff6e5; --warning-line: rgba(180, 83, 9, 0.28);
  --danger:  #dc3545; --danger-bg:  #fdeeee; --danger-line:  rgba(220, 53, 69, 0.28);
  --info:    #0d6efd; --info-bg:    #e7f0ff; --info-line:    rgba(13, 110, 253, 0.28);

  /* 圆角体系：小元素 8 / 卡片 12 / 大容器 16 / 圆形 50% */
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-full: 50%;

  /* 阴影层级 */
  --shadow-sm: 0 1px 2px rgba(33, 37, 41, 0.06);
  --shadow-md: 0 4px 12px rgba(33, 37, 41, 0.08);
  --shadow-lg: 0 12px 32px rgba(33, 37, 41, 0.12);

  /* 8px 网格 */
  --sp-1: 8px;
  --sp-2: 16px;
  --sp-3: 24px;
  --sp-4: 32px;
  --sp-6: 48px;

  /* 动效 */
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
  --dur: 0.25s;
}

/* ============================================================
   全局基础
   ============================================================ */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 480px at 85% -10%, rgba(139, 92, 246, 0.14), transparent 60%),
    radial-gradient(700px 420px at -10% 110%, rgba(77, 107, 254, 0.12), transparent 55%),
    linear-gradient(135deg, #f8f9fa 0%, #eef1f8 100%);
}
html, body {
  height: 100% !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  color: var(--text);
}
[data-testid="stHeader"] { display: none; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stApp"] { overflow: hidden !important; }
[data-testid="stAppViewContainer"] { overflow: hidden !important; }
[data-testid="stAppViewContainer"] > div { height: 100dvh !important; overflow: hidden !important; }
[data-testid="stMain"] { height: 100dvh !important; min-height: 0 !important; overflow: hidden !important; }

/* 主容器：毛玻璃卡片 */
.block-container {
  max-width: 460px;
  height: 100dvh !important;
  min-height: 0 !important;
  padding: var(--sp-1) 12px 8px;
  background: var(--surface);
  -webkit-backdrop-filter: blur(20px);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.65);
  box-shadow: var(--shadow-lg);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.block-container > [data-testid="stVerticalBlock"] {
  flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden;
}
div[data-testid="stLayoutWrapper"]:has(> .st-key-messages_area) {
  flex: 1 1 0% !important; min-height: 0 !important;
  display: flex !important; flex-direction: column; overflow: hidden;
}
div[data-testid="stLayoutWrapper"]:has(> .st-key-input_area) { flex: 0 0 auto !important; }
[data-testid="stColumn"] { min-width: 0 !important; }

/* 宽屏适配：主容器悬浮居中 */
@media (min-width: 768px) {
  .block-container { max-width: 720px; }
}
@media (min-width: 1024px) {
  .block-container {
    margin: var(--sp-3) auto;
    height: calc(100dvh - 48px) !important;
    border-radius: var(--r-lg);
    border: 1px solid rgba(255, 255, 255, 0.70);
  }
}

/* ============================================================
   布局：消息区 / 输入区 / 滚动条
   ============================================================ */
.st-key-messages_area {
  flex: 1; min-height: 0; overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(77, 107, 254, 0.40) transparent;
  padding: 4px 2px var(--sp-2);
}
.st-key-messages_area::-webkit-scrollbar { width: 6px; }
.st-key-messages_area::-webkit-scrollbar-track { background: transparent; }
.st-key-messages_area::-webkit-scrollbar-thumb {
  background: rgba(77, 107, 254, 0.35);
  border-radius: 6px;
}
.st-key-input_area { flex: none; }

/* ============================================================
   顶栏
   ============================================================ */
.app-title { font-size: 16px; font-weight: 700; letter-spacing: 0.1px; color: var(--text); }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: var(--r-sm);
  font-size: 12px; font-weight: 600; line-height: 1.5;
}
.pill::before { content: ""; width: 6px; height: 6px; border-radius: var(--r-full); background: currentColor; }
.pill.on { background: var(--success-bg); color: var(--success); }
.pill.off { background: var(--gray-200); color: var(--text-2); }

/* ============================================================
   按钮体系（圆角 8px / 高度 40px）
   ============================================================ */
div[data-testid="stButton"] > button {
  border-radius: var(--r-sm);
  border: 1px solid var(--line-strong);
  background: #ffffff;
  color: var(--text-2);
  font-size: 14px;
  font-weight: 500;
  min-height: 40px;
  box-shadow: var(--shadow-sm);
  transition: all var(--dur) var(--ease);
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--primary);
  color: var(--primary);
  background: var(--primary-soft);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); box-shadow: var(--shadow-sm); }

/* ============================================================
   对话气泡（15px 正文 / 不对称圆角 / 14px 18px 内边距）
   ============================================================ */
[data-testid="stChatMessage"] {
  padding: 4px 0 14px;
  align-items: flex-start;
  background: transparent !important;
  animation: msgIn var(--dur) var(--ease);
}
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] + [data-testid="stElementContainer"] { margin-top: 8px; }
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] [data-testid="stExpander"] { margin-top: 4px; border: none !important; background: transparent !important; }

.thinking-bubble {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 14px; color: var(--text-2); padding: 6px 4px;
}
.tdot {
  width: 7px; height: 7px; border-radius: var(--r-full);
  background: linear-gradient(180deg, var(--primary), var(--accent));
  animation: dotPulse 1.2s infinite ease-in-out;
}
.tdot:nth-child(2) { animation-delay: 0.15s; }
.tdot:nth-child(3) { animation-delay: 0.3s; }
@keyframes dotPulse {
  0%, 100% { opacity: 0.3; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(-3px); }
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(8px) scale(0.995); }
  to { opacity: 1; transform: none; }
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"] > div:first-child {
  width: 30px; height: 30px; border-radius: var(--r-full);
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0;
  box-shadow: var(--shadow-sm);
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) > div:first-child {
  background: var(--gray-200) !important;
}
[data-testid="stChatMessageContent"] {
  width: fit-content !important;
  max-width: 84% !important;
  height: auto !important;
  border-radius: var(--r-lg) var(--r-lg) var(--r-lg) var(--r-sm) !important;
  padding: 14px 18px !important;
  font-size: 15px;
  line-height: 1.6;
  word-break: break-word;
}
[data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"],
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"] { height: auto !important; }
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  color: #fff !important;
  border-radius: var(--r-lg) var(--r-lg) var(--r-sm) var(--r-lg) !important;
  box-shadow: 0 6px 18px rgba(77, 107, 254, 0.28);
}
[data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {
  background: rgba(255, 255, 255, 0.88) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-lg) var(--r-lg) var(--r-lg) 4px !important;
  color: var(--text);
  box-shadow: var(--shadow-sm);
}
[data-testid="stChatMessageContent"] p {
  margin: 0 0 4px;
  font-size: 15px !important;
  line-height: 1.6 !important;
}
[data-testid="stChatMessageContent"] .stMarkdown > div { align-items: flex-start !important; }
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] strong,
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] p,
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] div { color: #fff; }

/* ============================================================
   输入区（透明底 / 独立输入框 / 主色聚焦光环）
   ============================================================ */
.st-key-input_area {
  background: transparent;
  border: none;
  box-shadow: none;
  padding: 2px 0 6px;
  margin-top: 2px;
  gap: 6px !important;
}
.st-key-input_area [data-testid="stChatInput"] {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  box-shadow: var(--shadow-sm);
  padding: 2px;
  transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}
.st-key-input_area [data-testid="stChatInput"]:focus-within {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(77, 107, 254, 0.22);
}
.st-key-input_area textarea {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  resize: none;
  font-size: 14px;
  line-height: 1.45;
  padding: 8px 12px;
}
.st-key-input_area textarea::placeholder { color: var(--gray-400); }
.st-key-input_area [data-testid="stCaptionContainer"] p {
  margin: 2px 4px 0 !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
.st-key-input_area [data-testid="stCaptionContainer"] {
  margin-bottom: 0 !important;
}
div[data-testid="stChatInputSubmitButton"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  border: none !important;
  color: #fff !important;
  box-shadow: 0 4px 14px rgba(77, 107, 254, 0.35);
  transition: all var(--dur) var(--ease);
}
div[data-testid="stChatInputSubmitButton"]:hover {
  filter: brightness(1.05);
  box-shadow: 0 6px 18px rgba(77, 107, 254, 0.40);
}
[data-testid="stExpanderDetails"] { padding-bottom: 4px !important; }
[data-testid="stTextArea"] { background: transparent; }
[data-testid="stTextArea"] label { display: none !important; }

/* ============================================================
   功能开关（分段控件）
   ============================================================ */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display: none !important; }
[data-testid="stRadio"] {
  background: var(--gray-100);
  border-radius: var(--r-sm);
  padding: 3px;
  height: 34px;
  display: flex;
  align-items: center;
  border: 1px solid var(--line);
}
[data-testid="stRadio"] [role="radiogroup"] { gap: 4px; width: 100%; min-height: 0 !important; padding: 0 !important; }
[data-testid="stRadio"] label {
  border-radius: var(--r-sm);
  font-size: 13px; font-weight: 500;
  padding: 0 10px !important; height: 26px;
  background: transparent; color: var(--text-2);
  margin: 0; flex: 1; justify-content: center; min-height: 0 !important;
  display: flex; align-items: center;
  transition: all var(--dur) var(--ease);
}
[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  color: #fff !important;
  font-weight: 600;
  box-shadow: 0 4px 12px rgba(77, 107, 254, 0.30);
}
[data-testid="stRadio"] input[type="radio"] { display: none; }
[data-testid="stRadio"] label > div:first-child { display: none; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { margin: 0; }

/* ============================================================
   排班表卡片（圆角 12 / 阴影 md / 渐变头部）
   ============================================================ */
.sched-card {
  position: relative;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  overflow: hidden;
  margin: 6px 0 2px;
  background: #ffffff;
  box-shadow: var(--shadow-md);
}
.sched-head {
  display: flex; justify-content: space-between; align-items: center;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: #fff;
  padding: 12px var(--sp-2);
  font-weight: 700; font-size: 16px; letter-spacing: 0.1px;
}
.sched-badge {
  background: rgba(255, 255, 255, 0.20);
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: var(--r-sm);
  padding: 2px 10px;
  font-size: 12px; font-weight: 600;
}
table.sched-table { width: 100%; border-collapse: collapse; }
.sched-table th {
  font-size: 12px; color: var(--text-2);
  padding: 10px 8px; text-align: left;
  background: var(--gray-100);
  border-bottom: 1px solid var(--line);
  font-weight: 600;
}
.sched-table td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
  transition: background var(--dur) var(--ease);
}
.sched-table tbody tr:hover td { background: var(--primary-soft); }
.sched-table tr.weekend td { background: var(--warning-bg); }
.sched-table tr.weekend:hover td { background: #fff3d6; }
.sched-table tr:last-child td { border-bottom: none; }
.day { font-weight: 700; color: var(--gray-900); font-size: 12px; white-space: nowrap; }
tr.weekend .day { color: var(--warning); }
.shift-tag {
  display: inline-block; padding: 2px 9px;
  border-radius: var(--r-sm);
  font-size: 12px; font-weight: 600;
}
.shift-early { background: #fff3e0; color: var(--warning); }
.shift-late { background: var(--primary-soft); color: var(--primary-700); }
.emp {
  display: inline-block; margin: 0 3px 3px 0; padding: 2px 8px;
  border-radius: var(--r-sm);
  font-size: 12px; font-weight: 600;
  border: 1px solid transparent;
}
.emp-store { background: #fee2e2; color: #b91c1c; }
.emp-vice { background: #fce7f3; color: #be185d; }
.emp-super { background: #ede9fe; color: #6d28d9; }
.emp-senior { background: #dbeafe; color: #1d4ed8; }
.emp-clerk { background: var(--gray-200); color: var(--gray-900); }
.emp-part { background: #ccfbf1; color: #0f766e; }
.emp-unknown { background: var(--warning-bg); color: var(--warning); }
.meta { margin-top: 4px; font-size: 12px; color: var(--text-2); }
.empty { color: var(--text-3); font-size: 12px; }
.sched-foot {
  padding: 8px var(--sp-2);
  font-size: 12px; color: var(--text-2);
  background: var(--gray-100);
  border-top: 1px solid var(--line);
}
.dot { display: inline-block; width: 8px; height: 8px; border-radius: var(--r-full); margin: 0 3px 0 8px; }

/* ============================================================
   规则合规卡片
   ============================================================ */
.rule-card {
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  overflow: hidden;
  margin: 6px 0 2px;
  box-shadow: var(--shadow-sm);
}
.rule-item {
  display: flex; gap: 10px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  font-size: 14px;
  border-left: 3px solid transparent;
  transition: background var(--dur) var(--ease);
}
.rule-item:last-child { border-bottom: none; }
.rule-item.pass { background: #ffffff; }
.rule-item.fail { background: var(--danger-bg); border-left-color: var(--danger); }
.rule-item.unknown { background: var(--warning-bg); border-left-color: var(--warning); }
.rule-item:hover { background: var(--gray-100); }
.rule-icon { font-size: 16px; line-height: 1.4; }
.rule-body { flex: 1; min-width: 0; }
.rule-title { color: var(--gray-900); line-height: 1.5; }
.rule-title b { color: var(--text); font-weight: 700; }
.rule-status {
  float: right; margin-left: 6px;
  padding: 1px 9px;
  border-radius: var(--r-sm);
  font-size: 12px; font-weight: 600;
}
.rule-status.pass { background: var(--success-bg); color: var(--success); }
.rule-status.fail { background: var(--danger-bg); color: var(--danger); }
.rule-status.unknown { background: var(--warning-bg); color: var(--warning); }
.rule-detail { font-size: 12px; color: var(--text-2); margin-top: 3px; line-height: 1.55; }

/* ============================================================
   欢迎页
   ============================================================ */
.hero { text-align: center; padding: var(--sp-6) var(--sp-1) var(--sp-2); }
.hero-logo {
  position: relative;
  width: 72px; height: 72px; margin: 0 auto var(--sp-2);
  border-radius: var(--r-lg);
  background: linear-gradient(135deg, var(--primary), var(--accent));
  display: flex; align-items: center; justify-content: center;
  font-size: 32px;
  box-shadow: 0 12px 30px rgba(77, 107, 254, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.hero-logo::after {
  content: ""; position: absolute; inset: -10px;
  border-radius: var(--r-lg);
  background: linear-gradient(135deg, rgba(77, 107, 254, 0.18), rgba(139, 92, 246, 0.18));
  filter: blur(16px); z-index: -1;
}
.hero h1 {
  font-size: 24px; font-weight: 700;
  margin: 0 0 var(--sp-2); letter-spacing: 0.3px;
  background: linear-gradient(135deg, var(--primary-700), var(--accent));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--text-2); font-size: 14px; line-height: 1.75; margin: 0 0 4px; }
.hero .hero-sub { color: var(--text-3); font-size: 12px; margin-top: var(--sp-1); }
.hero .hint { color: var(--text-3); font-size: 12px; margin-top: var(--sp-3); }

/* ============================================================
   示例按钮（lg 48px）
   ============================================================ */
div[data-testid="stElementContainer"][class*="st-key-example_"] button {
  position: relative;
  background: var(--gray-100);
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 12px 36px 12px 16px;
  min-height: 48px;
  text-align: left;
  font-size: 14px;
  color: var(--text);
  box-shadow: var(--shadow-sm);
  transition: all var(--dur) var(--ease);
}
div[data-testid="stElementContainer"][class*="st-key-example_"] button:hover {
  background: var(--primary-soft);
  border-color: var(--primary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
div[data-testid="stElementContainer"][class*="st-key-example_"] button::after {
  content: "›"; position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
  color: var(--text-3); font-size: 20px; font-weight: 600;
  transition: transform var(--dur) var(--ease), color var(--dur) var(--ease);
}
div[data-testid="stElementContainer"][class*="st-key-example_"] button:hover::after {
  transform: translate(2px, -50%); color: var(--primary);
}

/* ============================================================
   下载按钮（线框变体）
   ============================================================ */
div[data-testid="stDownloadButton"] > button {
  border-radius: var(--r-sm);
  border: 1px solid var(--primary);
  background: #ffffff;
  color: var(--primary);
  font-size: 14px; font-weight: 600;
  min-height: 40px;
  transition: all var(--dur) var(--ease);
}
div[data-testid="stDownloadButton"] > button:hover {
  background: var(--primary-soft);
  border-color: var(--primary-700);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

/* ============================================================
   数据对话框（模态 16px）
   ============================================================ */
[data-testid="stDialog"] {
  background: rgba(33, 37, 41, 0.45) !important;
  -webkit-backdrop-filter: blur(4px);
  backdrop-filter: blur(4px);
}
[data-testid="stDialog"] > div {
  position: fixed !important;
  bottom: 0 !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: min(480px, 100vw) !important;
  max-width: 100vw !important;
  max-height: 82dvh !important;
  margin: 0 !important;
  overflow: hidden !important;
  border-radius: var(--r-lg) var(--r-lg) 0 0 !important;
  border: none !important;
  border-top: 1px solid var(--line) !important;
  box-shadow: var(--shadow-lg) !important;
  background: #ffffff;
}
[data-testid="stDialog"] > div::before {
  content: ""; display: block;
  width: 38px; height: 5px; border-radius: var(--r-sm);
  background: var(--gray-300);
  margin: 11px auto 3px; flex-shrink: 0;
}
[data-testid="stDialog"] section {
  display: flex !important; flex-direction: column;
  min-height: 0; overflow: hidden !important;
}
[data-testid="stDialog"] section > div:last-child {
  flex: 1 1 auto !important; min-height: 0 !important; overflow-y: auto !important;
}
[data-testid="stDialog"] h2 {
  font-size: 16px !important; font-weight: 700 !important;
  color: var(--text) !important;
  padding: var(--sp-3) var(--sp-3) var(--sp-1) !important;
  margin: 0 !important;
}
[data-testid="stDialog"] button[aria-label="Close"] {
  width: 30px !important; height: 30px !important;
  border-radius: var(--r-full) !important;
  background: var(--gray-100) !important;
  border: none !important;
  color: var(--text-2) !important;
  display: flex !important; align-items: center; justify-content: center;
  transition: all var(--dur) var(--ease);
}
[data-testid="stDialog"] button[aria-label="Close"]:hover {
  background: var(--primary-soft) !important;
  color: var(--primary) !important;
}
[data-testid="stDialog"] section > button[aria-label="Close"] { top: 16px !important; right: 16px !important; }
[data-testid="stDialog"] [data-testid="stDataFrame"] { width: 100% !important; }

/* ============================================================
   头部图标按钮（40px 圆形按钮组）
   ============================================================ */
div[data-testid="stElementContainer"].st-key-new_chat button,
div[data-testid="stElementContainer"].st-key-data_btn button {
  width: 40px !important; height: 40px !important; min-height: 40px !important;
  padding: 0 !important;
  border-radius: var(--r-sm) !important;
  border: 1px solid var(--line) !important;
  background: rgba(255, 255, 255, 0.80) !important;
  color: var(--text-2) !important;
  display: flex !important; align-items: center; justify-content: center;
  box-shadow: var(--shadow-sm) !important;
  transition: all var(--dur) var(--ease) !important;
}
div[data-testid="stElementContainer"].st-key-new_chat button:hover,
div[data-testid="stElementContainer"].st-key-data_btn button:hover {
  background: var(--primary-soft) !important;
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  transform: translateY(-1px);
}

/* ============================================================
   为什么这样安排：排班逻辑
   ============================================================ */
.why-title {
  font-weight: 600; color: var(--text);
  font-size: 16px; margin: 4px 0 var(--sp-1);
}
.why-hint { font-weight: 400; color: var(--text-3); font-size: 12px; margin-left: var(--sp-1); }
.why-steps {
  display: flex; flex-direction: row; gap: var(--sp-1);
  overflow-x: auto; padding: 2px 2px var(--sp-1); margin-bottom: var(--sp-1);
  scroll-snap-type: x mandatory; -webkit-overflow-scrolling: touch;
}
.why-steps::-webkit-scrollbar { height: 0; }
.why-step {
  flex: 0 0 44%; min-width: 150px; scroll-snap-align: start;
  background: linear-gradient(180deg, #ffffff, var(--gray-100));
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: var(--sp-2);
  box-shadow: var(--shadow-sm);
  transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}
.why-step:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.why-step .ws-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 36px; height: 36px;
  border-radius: var(--r-sm);
  background: linear-gradient(135deg, var(--primary-soft), var(--accent-soft));
  font-size: 20px; margin-bottom: var(--sp-1);
}
.why-step .ws-title { font-weight: 600; color: var(--text); font-size: 14px; margin-bottom: 4px; }
.why-step .ws-text { color: var(--text-2); font-size: 12px; line-height: 1.6; }
.why-facts { display: flex; flex-wrap: wrap; gap: var(--sp-1); margin-bottom: var(--sp-1); }
.why-fact {
  background: var(--primary-soft);
  border: 1px solid var(--info-line);
  color: var(--primary-700);
  border-radius: var(--r-sm);
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 600;
}
.why-banner {
  border-radius: var(--r-md);
  padding: 12px var(--sp-2);
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 2px;
  font-weight: 500;
}
.why-banner.ok { background: var(--success-bg); border: 1px solid var(--success-line); color: var(--success); }
.why-banner.warn { background: var(--warning-bg); border: 1px solid var(--warning-line); color: var(--warning); }
.why-banner.err { background: var(--danger-bg); border: 1px solid var(--danger-line); color: var(--danger); }

/* ============================================================
   面板内部间距 / 折叠面板
   ============================================================ */
[data-testid="stChatMessageContent"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] { gap: 4px !important; }
[data-testid="stChatMessageContent"] [data-testid="stExpander"] [data-testid="stExpander"] { margin-top: 2px !important; }
[data-testid="stExpander"] summary {
  padding: 6px 0 !important;
  font-weight: 600; color: var(--text); font-size: 14px;
}

/* ============================================================
   页签
   ============================================================ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 6px; margin-bottom: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: var(--r-sm);
  padding: 6px 14px;
  font-size: 14px; font-weight: 600;
  color: var(--text-2); background: var(--gray-100);
  transition: all var(--dur) var(--ease);
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { background: var(--primary-soft); color: var(--primary); }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: #fff;
  box-shadow: 0 4px 12px rgba(77, 107, 254, 0.30);
}

/* ============================================================
   特殊情况说明
   ============================================================ */
.special-card { display: flex; flex-direction: column; gap: var(--sp-1); }
.special-row {
  display: flex; gap: 12px; align-items: flex-start;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 12px var(--sp-2);
  box-shadow: var(--shadow-sm);
}
.special-row .sp-icon {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border-radius: var(--r-sm);
  background: linear-gradient(135deg, var(--primary-soft), var(--accent-soft));
  font-size: 16px; flex-shrink: 0;
}
.special-row .sp-title { font-weight: 600; color: var(--text); font-size: 14px; margin-bottom: 3px; }
.special-row .sp-text { color: var(--text-2); font-size: 12px; line-height: 1.6; }

/* ============================================================
   侧边栏（毛玻璃 260px）
   ============================================================ */
[data-testid="stSidebar"] {
  width: 260px !important;
  background: rgba(255, 255, 255, 0.55) !important;
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
  border-right: 1px solid rgba(0, 0, 0, 0.08);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: var(--sp-3) var(--sp-2);
}
.sb-brand { display: flex; align-items: center; gap: 12px; margin-bottom: var(--sp-2); }
.sb-logo {
  width: 40px; height: 40px;
  border-radius: var(--r-md);
  background: linear-gradient(135deg, var(--primary), var(--accent));
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; color: #fff;
  box-shadow: 0 6px 16px rgba(77, 107, 254, 0.30);
}
.sb-name { font-size: 16px; font-weight: 700; color: var(--text); line-height: 1.4; }
.sb-sub { font-size: 12px; color: var(--text-3); }
.sb-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px;
  border-radius: var(--r-sm);
  font-size: 12px; font-weight: 600;
}
.sb-pill.on { background: var(--success-bg); color: var(--success); }
.sb-pill.off { background: var(--gray-200); color: var(--text-2); }
[data-testid="stSidebar"] hr {
  border-color: var(--line);
  margin: var(--sp-2) 0;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  width: 100%;
  justify-content: flex-start;
  text-align: left;
  border-radius: var(--r-sm);
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.70);
  color: var(--text-2);
  font-size: 14px;
  font-weight: 500;
  min-height: 40px;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  background: var(--primary-soft);
  border-color: var(--primary);
  color: var(--primary);
  transform: none;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--text-3);
  font-size: 12px;
}
</style>
"""



if "messages" not in st.session_state:
    st.session_state.messages = []
if "employees" not in st.session_state:
    st.session_state.employees = copy.deepcopy(EMPLOYEES)


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


def _build_logic_html(explanation: Dict[str, object]) -> str:
    """把排班逻辑步骤与关键数据渲染成卡片。"""
    steps = explanation["logic_steps"]
    step_html = "".join(
        f'<div class="why-step"><div class="ws-icon">{s["icon"]}</div>'
        f'<div class="ws-body"><div class="ws-title">{s["title"]}</div>'
        f'<div class="ws-text">{s["text"]}</div></div></div>'
        for s in steps
    )
    facts = "".join(f'<span class="why-fact">{f}</span>' for f in explanation["facts"])
    return (
        f'<div class="why-title">🧭 排班逻辑<span class="why-hint">左右滑动查看</span></div>'
        f'<div class="why-steps">{step_html}</div>'
        f'<div class="why-facts">{facts}</div>'
    )


def _build_rule_banner_html(checks, rules_summary: str) -> str:
    """规则检查结果横幅。"""
    if any(c.status == "违反" for c in checks):
        cls, icon = "err", "❌"
    elif any(c.status == "无法判断" for c in checks):
        cls, icon = "warn", "⚠️"
    else:
        cls, icon = "ok", "✅"
    return f'<div class="why-banner {cls}">{icon} {rules_summary}</div>'


def _build_special_html(explanation: Dict[str, object]) -> str:
    """特殊情况说明：结构化卡片。"""
    rows = "".join(
        f'<div class="special-row"><div class="sp-icon">{n["icon"]}</div>'
        f'<div class="sp-body"><div class="sp-title">{n["title"]}</div>'
        f'<div class="sp-text">{n["text"]}</div></div></div>'
        for n in explanation["special_notes"]
    )
    return f'<div class="special-card">{rows}</div>'


def _build_explanation_sections(schedule: Schedule, checks, explanation: Dict[str, object]) -> Dict[str, str]:
    """把解释内容整理成易读的卡片。"""
    return {
        "logic_html": _build_logic_html(explanation),
        "rule_banner": _build_rule_banner_html(checks, explanation["rules_summary"]),
        "special_html": _build_special_html(explanation),
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
        "所有排班与规则判断均由本地规则引擎完成，依据仅来自题目规则 R-01～R-09 与员工数据。"
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

    # 应用请假/不可用更新：更新会话内的员工数据，后续排班持续生效
    employees = st.session_state.employees
    if intent.leave_updates:
        employees = copy.deepcopy(employees)
        emp_by_id = {e.emp_id: e for e in employees}
        for upd in intent.leave_updates:
            emp = emp_by_id.get(upd["emp"])
            if emp is None:
                notes.append(f"员工 {upd['emp']} 不在数据中，无法更新请假信息")
                continue
            days = upd["days"] or list(DAYS)
            emp.available_days = [d for d in emp.available_days if d not in days]
            emp.leave_days = sorted(set(emp.leave_days) | set(days))
            notes.append(f"已更新员工数据：{upd['emp']} 请假（{'、'.join(days)}），相关日期不再排班")
        st.session_state.employees = employees

    counts = default_min_counts()
    for (day, shift), n in intent.min_counts.items():
        counts[(DAY_INDEX[day], SHIFT_INDEX[shift])] = n

    result = solve_schedule(employees, RULES, min_counts=counts, exclude=intent.exclude)

    summary = ("✅ " if result.feasible else "⚠️ ") + result.message
    scope = "、".join(intent.days)
    exclude = "、".join(intent.exclude) if intent.exclude else "无"
    summary += f"\n\n📅 排班范围：{scope}　｜　🚫 排除员工：{exclude}"

    explanation = build_schedule_explanation(result.schedule, employees, result.checks)
    sections = _build_explanation_sections(result.schedule, result.checks, explanation)

    return {
        "role": "assistant",
        "kind": "generate",
        "summary": summary,
        "notes": notes,
        "table": _build_schedule_html(result.schedule),
        "checks": _build_rule_html(result.checks),
        "logic_html": sections["logic_html"],
        "rule_banner": sections["rule_banner"],
        "special_html": sections["special_html"],
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
    if not looks_meaningful(text):
        return {
            "role": "assistant",
            "kind": "text",
            "content": (
                "抱歉，我没理解你的需求。请用一句话描述，例如：\n\n"
                "- 生成排班：`帮我安排周一到周日的排班，各班按规则最低人数安排`\n"
                "- 检查排班：先切换到「检查排班」，再粘贴 `周一早班：E01,E02,E03,E04`"
            ),
        }

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
    with st.expander("📝 为什么这样安排"):
        st.markdown(msg["logic_html"], unsafe_allow_html=True)
        st.markdown(msg["rule_banner"], unsafe_allow_html=True)
        tab_detail, tab_special = st.tabs(["📋 规则校验明细", "ℹ️ 特殊情况说明"])
        with tab_detail:
            st.markdown(msg["checks"], unsafe_allow_html=True)
        with tab_special:
            st.markdown(msg["special_html"], unsafe_allow_html=True)


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
    st.markdown("**📋 排班规则 R-01～R-09**")
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
        if st.button("", key="new_chat", icon=":material/chat_bubble:", use_container_width=True):
            st.session_state.messages = []
            st.session_state.employees = copy.deepcopy(EMPLOYEES)
    with c3:
        if st.button("", key="data_btn", icon=":material/analytics:", use_container_width=True):
            _show_data_dialog()


def _render_sidebar() -> None:
    """企业级侧边栏：品牌、状态与快捷入口。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    pill_class = "on" if api_key else "off"
    pill_text = "✨ DeepSeek 在线" if api_key else "🔌 本地解析"
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand">'
            '<div class="sb-logo">📅</div>'
            '<div><div class="sb-name">智能排班助手</div>'
            '<div class="sb-sub">Smart Scheduler</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="sb-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("📊 查看规则与员工数据", key="sb_data", use_container_width=True):
            _show_data_dialog()
        if st.button("🧹 新建对话", key="sb_new", use_container_width=True):
            st.session_state.messages = []
            st.session_state.employees = copy.deepcopy(EMPLOYEES)
            st.rerun()
        st.caption("判断依据仅来自题目规则 R-01～R-09 与员工数据。")


def _render_input_bar() -> Tuple[bool, str, str]:
    current_mode = st.session_state.get("chat_mode", st.session_state.get("last_mode", "生成排班"))
    mode = st.radio(
        "功能",
        options=["生成排班", "检查排班"],
        format_func=lambda x: {"生成排班": "📝 生成排班", "检查排班": "🔍 检查排班"}[x],
        index=0 if st.session_state.get("last_mode", "生成排班") == "生成排班" else 1,
        key="chat_mode",
        horizontal=True,
        label_visibility="hidden",
    )
    if mode == "检查排班":
        st.caption("支持多行粘贴，每行一个班次")
    placeholder = (
        "例如：帮我安排周一到周日的排班"
        if mode == "生成排班"
        else "粘贴排班文本，如：周一早班：E01"
    )
    prompt = st.chat_input(placeholder, key="chat_prompt")
    if prompt and prompt.strip():
        return True, prompt, mode
    return False, "", mode


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    _render_sidebar()
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

    # 消息区自动滚动到底部（高度为 0 的透明组件，仅执行滚动脚本）
    st.iframe(
        "<script>"
        "(function(){"
        "  var el = window.parent.document.querySelector('.st-key-messages_area');"
        "  if (el) { el.scrollTo({top: el.scrollHeight, behavior: 'smooth'}); }"
        "})();"
        f"//{time.time()}"
        "</script>",
        height=1,
    )


if __name__ == "__main__":
    main()

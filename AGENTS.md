# AGENTS.md — 项目说明

> 本文件供后续开发/协作的 Agent 快速了解项目。内容由通读代码、README、测试与题目文档（`todo.docx`）整理而来。

## 一、项目概述

这是一个**连锁门店智能排班助手**（Agent Demo），源文件位于 `smart-scheduler/` 目录：

- 用户像聊天一样输入一句自然语言（如「帮我安排周一到周日的排班」），即可生成一周排班；
- 支持**生成排班**、**检查已有排班**、**解释排班原因**三类核心能力；
- 排班与规则判断**完全由本地规则引擎**（确定性算法）完成，大模型（可选）只负责理解自然语言与润色解释，不会自行补技能或编造判断依据；
- 判断依据仅来自题目文档 `todo.docx`：9 条规则（R-01~R-09）与 20 名员工数据。

界面为 Streamlit 实现的**手机风格对话框**（类豆包 / DeepSeek）。

## 二、技术栈与运行环境

| 项目 | 说明 |
| --- | --- |
| 语言 | Python 3.12（代码注释与界面文案全部为中文） |
| Web 框架 | Streamlit（>=1.40），单文件应用 `app.py` |
| 依赖 | `streamlit`、`requests`、`python-dotenv`、`pandas`、`pytest`（见 `smart-scheduler/requirements.txt`） |
| 大模型接口 | 可选；DeepSeek 的 OpenAI 兼容 `chat/completions` 接口，`temperature=0` |
| 测试 | pytest + Streamlit AppTest 冒烟测试 |
| 操作系统 | Windows（本机 shell 为 PowerShell） |

## 三、目录结构

```text
codex-fde-bytedance/
├── AGENTS.md                 # 本文件
├── todo.docx                 # 题目材料（排班场景、R-01~R-09、20 名员工数据）
└── smart-scheduler/          # 项目主体
    ├── app.py                # Streamlit 入口：手机风格对话 UI（含大量自定义 CSS）
    ├── README.md             # 项目使用文档（含自测样例）
    ├── requirements.txt      # Python 依赖
    ├── run_demo.bat          # Windows 一键启动脚本
    ├── .env.example          # DeepSeek API 配置模板
    ├── .streamlit/config.toml
    ├── agent/                # 核心引擎包（__version__ = "1.0.0"）
    │   ├── types.py          # 数据模型：Employee / Rule / Schedule / RuleCheck
    │   ├── data_loader.py    # 内置规则与 20 名员工数据（数据唯一来源）
    │   ├── nlu.py            # 自然语言意图解析（LLM + 本地模板降级）
    │   ├── scheduler.py      # 回溯式 CSP 排班求解器
    │   ├── validator.py      # R-01~R-09 + SC-01 规则校验与归因
    │   └── explainer.py      # 排班解释生成（可选 LLM 润色）
    ├── tests/                # 单元测试（test_nlu / test_scheduler / test_validator）
    └── scripts/
        ├── demo_self_test.py # 端到端自测（本地模式，输出完整样例）
        └── smoke_app.py      # Streamlit AppTest 网页冒烟测试
```

## 四、核心架构与数据流

```
用户输入
  → nlu.parse_intent()        意图解析（LLM 或本地模板，返回 Intent）
  → scheduler.solve_schedule() 回溯式约束搜索，生成 7 天 × 2 班排班
  → validator.validate_schedule() 逐条校验 R-01~R-09 与 SC-01
  → explainer.build_schedule_explanation() 生成通俗解释
  → app.py 渲染（排班卡片 / 规则报告 / 解释 / CSV 下载）
```

关键职责划分：

- **nlu 只做翻译**：把自然语言转成结构化参数（`Intent`），不做任何排班决策；API 失败时自动降级为本地模板解析。
- **scheduler 只做求解**：确定性回溯搜索，硬约束不满足时返回「尽力方案 + 违规清单」，绝不编造。
- **validator 只做核验**：每条规则输出状态、风险等级、涉及员工、整改建议与归因。
- **explainer 只做解释**：内容完全基于求解结果、校验结果与员工数据；LLM 仅做语言润色。

## 五、业务规则与数据（核心领域知识）

### 排班场景

- 排班周期：**周一至周日**，每天两个班次：早班 09:00–17:00、晚班 13:00–21:00；
- 每人每天最多一个班（场景约束 SC-01）；
- 求解器始终生成完整的 14 个班次槽位（`(day_idx, shift_idx)` 元组为键）。

### 规则 R-01~R-09

| 编号 | 规则内容 | 风险等级（违反时） |
| --- | --- | --- |
| R-01 | 每班至少 1 名具备「店长值守」技能的员工 | 高 |
| R-02 | 每班至少 2 名具备「饮品制作」技能的员工 | 高 |
| R-03 | 每班至少 1 名具备「收银」技能的员工 | 高 |
| R-04 | 周一至周五每班至少 4 人；周六、周日每班至少 6 人 | 中 |
| R-05 | 每人每周最多 40 小时，即最多 5 个班 | 中 |
| R-06 | 不得连续工作超过 5 天 | 中 |
| R-07 | 上一天晚班后不得安排次日早班 | 中 |
| R-08 | 请假和不可工作日期绝对不得排班 | 高 |
| R-09 | 技能必须来自员工数据，不得自行补技能（数据外员工直接判违规） | 高 |

另有两条非题目规则：

- **SC-01**：每人每天最多一个班（场景约束）；
- **REQ-01**：满足用户指定/规则默认的班次最低人数（由求解器附加检查）。

### 员工数据要点（20 人，E01~E20）

- 岗位：店长、副店长、值班主管、高级店员、店员、兼职；
- 技能字段：`店长值守`、`饮品制作`、`收银`、`库存管理`；
- 员工有 `可工作日期`、`请假日期`、`班次偏好`（早班/晚班/无）三个约束字段；
- 关键特殊员工：
  - E01（店长）：全周可用，**周三请假**，偏好早班；
  - E02（副店长）：全周可用，偏好晚班；
  - E03/E04：值班主管，可工作日期分别只有周一至周五、周三至周日；
  - E13/E14/E19：兼职，基本只在周末可工作；
  - E09：仅周一、周三、周五、周六、周日可工作（只掌握饮品制作）；
  - E17：全周可用但**周一请假**；E20：周二至周日可用但**周四请假**。
- 数据集中在 `agent/data_loader.py`，任何排班判断都只读这里，不要改动数据与规则（除非题目变更）。

## 六、模块要点（改代码前必读）

### agent/types.py

- 常量：`DAYS`（周一~周日）、`SHIFTS`（早/晚班）、`SHIFT_TIMES`、四个技能常量、`SlotKey = (day_idx, shift_idx)`；
- `Schedule`：`slots: Dict[SlotKey, List[str]]`，缺失槽位视为「未提供信息」；
- `RuleCheck`：状态为 `通过` / `违反` / `无法判断`，含 `risk_level`、`involved`、`details`、`suggestion`、`attribution`。

### agent/data_loader.py

- `load_employees()` / `load_rules()` 返回**副本**，防止调用方误改内置数据；
- `employees_summary_text()` / `rules_summary_text()` 用于拼大模型提示词。

### agent/nlu.py

- `Intent`：`action`（generate/check）、`days`、`min_counts`、`exclude`、`schedule_text`；
- `parse_intent()`：有 API Key 时先走 LLM，任何异常（网络/鉴权/非法 JSON）都降级到 `parse_local()`，并在 `notes` 中提示；
- 本地模板：正则识别日期范围（周一到周五等）、人数要求（早班4人/每班5人等）、排除员工（不要安排E03等）；
- 默认人数：周一至周五每班 4 人，周六、周日每班 6 人（R-04）；
- 排班文本解析格式：`周X早班/晚班：E01,E02,…`，按 `；`/换行分句；
- `call_llm_text` / `call_llm_json`：OpenAI 兼容 `chat/completions`，默认 `https://api.deepseek.com`，`deepseek-chat` 模型。

### agent/scheduler.py

- 回溯式 CSP 求解器：14 个槽位逐个搜索；
  - **硬约束**：R-01~R-09 + 每人每天最多一个班；
  - **软约束**：偏好匹配优先、工作量均衡（按偏好等级→班次数→已工作天数排序）；
  - 优化手段：MRV（候选最少的槽位优先）+ 前向检查；
  - 限制：`NODE_LIMIT = 400_000`，`TIME_LIMIT_SECONDS = 25.0`；
- 求解失败/超时时用 `greedy_best_effort()` 生成尽力方案，并附加 REQ-01 人数检查，提示人工复核；
- `solve_schedule()` 返回 `SolveResult`（schedule、feasible、message、checks）。

### agent/validator.py

- `validate_schedule()` 返回 10 项检查（R-01~R-09 + SC-01），空排班时返回单条 `INPUT` 检查；
- 「无法判断」语义：信息不足、班次缺失、含数据外员工时输出，绝不臆测；
- `all_pass()`：所有检查状态均为「通过」才算合规。

### agent/explainer.py

- `build_schedule_explanation()` 返回结构化的解释小节（overview、logic_steps、facts、special_notes、daily、rules_summary、rules_detail、why_not）；
- `polish_with_llm()`：可选润色，失败时原样返回；约束是不得新增规则/事实，必须保留规则编号与「无法判断/人工复核」提示。

### app.py（UI）

- 手机风格布局：固定顶栏、内部滚动消息区、吸底输入区，全部由内嵌 CSS 实现；
- 底部「生成排班 / 检查排班」开关；在生成模式下粘贴排班文本会自动切到检查模式；
- 欢迎页 3 个示例按钮；右上角可查看规则与员工数据（底部抽屉对话框）；
- 特殊提问（规则、员工、使用说明、数据）走 `_detect_special_query()` 直接返回固定文案；
- 排班结果：卡片式排班表（周末高亮、员工按岗位着色、技能覆盖统计）+ CSV 下载（UTF-8 BOM，Excel 友好）；
- 注意：求解器始终生成完整 7 天排班；`intent.days` 只影响汇总文案中显示的排班范围，不会限制求解器只排部分日期。

## 七、常用命令

在 `smart-scheduler/` 目录下执行：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用（浏览器打开 http://localhost:8501）
python -m streamlit run app.py

# 运行单元测试
python -m pytest tests/ -v

# 端到端自测（本地模式，打印完整样例）
python scripts/demo_self_test.py

# 网页冒烟测试（Streamlit AppTest）
python scripts/smoke_app.py
```

Windows 下也可双击 `run_demo.bat` 一键启动（优先使用 Codex 工作区运行时自带的 Python，找不到再回退系统 Python）。

## 八、配置说明

- 复制 `.env.example` 为 `.env` 并填写：
  - `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com`）、`DEEPSEEK_MODEL`（默认 `deepseek-chat`）；
- 未配置 Key 时自动使用**本地解析模式**，排班与规则检查功能完整可用（界面顶部显示「本地解析」徽标）；
- `.env` 已被 `.gitignore` 忽略，**切勿提交**真实密钥。

## 九、工作区现状与注意事项

- 仓库目前只有一个初始提交 `f97594f`（feat: initial version of smart-scheduler），分支为 `main`；
- **工作区有未提交的改动**：`smart-scheduler/README.md`、`agent/explainer.py`、`app.py`、`scripts/smoke_app.py` 相对 HEAD 有修改，改动属于用户，请勿覆盖或擅自提交；
- 本地存在 `.env` 与 `.pytest_cache`（均被忽略，不提交）；
- 项目代码注释、界面文案、测试断言全部为中文，保持这一惯例；
- 数据与规则以 `data_loader.py` 为准，不要在 LLM 提示词之外引入「常识」判断；
- 修改排班逻辑后，务必运行 `pytest` 与 `demo_self_test.py` 验证默认场景仍可产出完全合规的 7 天排班。

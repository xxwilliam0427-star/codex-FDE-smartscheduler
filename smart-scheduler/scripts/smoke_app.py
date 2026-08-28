"""网页应用冒烟测试：用 Streamlit AppTest 模拟真实操作。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def main() -> None:
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    # 生成排班：点击欢迎页示例「生成一周排班」
    gen_btn = [b for b in at.button if "生成一周排班" in b.label]
    assert gen_btn, [b.label for b in at.button]
    gen_btn[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("已生成排班" in m.value for m in at.markdown), [m.value for m in at.markdown]
    assert any("为什么这样安排" in e.label for e in at.expander), [e.label for e in at.expander]
    assert any("排班逻辑" in m.value for m in at.markdown), [m.value for m in at.markdown]
    assert any("满足硬性要求" in m.value for m in at.markdown), [m.value for m in at.markdown]

    # 无意义输入应提示无法理解，而不是默认生成排班
    at.chat_input[0].set_value("1")
    at.run()
    assert not at.exception, at.exception
    assert any("没理解" in m.value for m in at.markdown), [m.value for m in at.markdown]

    # 请假语义：应识别并应用“E03 本周请假”，重新生成排班
    at.chat_input[0].set_value("搞错了，e03这周请假来不了")
    at.run()
    assert not at.exception, at.exception
    assert any("已更新员工数据" in c.value for c in at.caption), [c.value for c in at.caption]
    assert any("已生成排班" in m.value for m in at.markdown), [m.value for m in at.markdown]

    # 多轮对话：示例发送后，再输入一条并发送（chat_input 提交后自动清空）
    assert at.chat_input, "应有聊天输入框"
    at.chat_input[0].set_value("周六晚班安排6人")
    at.run()
    assert not at.exception, at.exception
    assert any("已生成排班" in m.value for m in at.markdown), [m.value for m in at.markdown]

    # 检查排班：切换功能开关并粘贴排班文本
    at.radio[0].set_value("检查排班")
    at.run()
    at.chat_input[0].set_value("周一早班：E01,E06,E09,E12\n周一晚班：E02,E07,E11,E18")
    at.run()
    assert not at.exception, at.exception
    assert any("无法判断" in m.value for m in at.markdown), [m.value for m in at.markdown]

    print("SMOKE OK")


if __name__ == "__main__":
    main()

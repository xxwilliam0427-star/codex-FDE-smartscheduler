"""网页应用冒烟测试：用 Streamlit AppTest 模拟对话式操作。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def main() -> None:
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    # 生成排班：输入一句自然语言并发送
    at.text_area[0].set_value("帮我安排周一到周日的排班，各班按规则最低人数安排")
    at.run()
    at.button(key="send_btn").click()
    at.run()
    assert not at.exception, at.exception
    assert any("已生成排班" in m.value for m in at.markdown), [m.value[:80] for m in at.markdown]

    # 检查排班：切换到「检查排班」开关并粘贴排班文本
    at.text_area[0].set_value(
        "周一早班：E01,E06,E09,E12\n"
        "周一晚班：E02,E07,E11,E18\n"
        "周六早班：E01,E04,E06,E09,E13,E20"
    )
    at.run()
    at.radio[0].set_value("检查排班")
    at.run()
    at.button(key="send_btn").click()
    at.run()
    assert not at.exception, at.exception
    assert any("无法判断" in m.value for m in at.markdown), [m.value[:80] for m in at.markdown]

    print("SMOKE OK")


if __name__ == "__main__":
    main()

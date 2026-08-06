"""``GeminiComputerTool`` — predefined computer-use functions dispatched to RFB.

No live VNC: a recording subclass captures the primitive calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from hud.agents.gemini.tools.computer import GEMINI_COMPUTER_SPEC, GeminiComputerTool
from hud.agents.tools.base import tool_ok

if TYPE_CHECKING:
    from collections.abc import Iterable


class RecordingGemini(GeminiComputerTool):
    client: Any

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.client = SimpleNamespace(width=400, height=300)
        self.excluded_predefined_functions = []

    @override
    async def screenshot(self) -> Any:
        self.calls.append(("screenshot",))
        return tool_ok("shot")

    @override
    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        button: Any = "left",
        hold_keys: Iterable[str] | None = None,
        count: int = 1,
        interval_ms: int = 0,
    ) -> None:
        self.calls.append(("click", x, y))

    @override
    async def move(self, x: Any, y: Any) -> None:
        self.calls.append(("move", x, y))

    @override
    async def type_text(self, text: Any) -> None:
        self.calls.append(("type", text))

    @override
    async def press_keys(self, keys: Any, **kw: Any) -> None:
        self.calls.append(("keys", tuple(keys)))

    @override
    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        scroll_x: int = 0,
        scroll_y: int = 0,
        hold_keys: Iterable[str] | None = None,
    ) -> None:
        kw = {"scroll_x": scroll_x, "scroll_y": scroll_y, "hold_keys": hold_keys}
        self.calls.append(("scroll", x, y, kw))

    @override
    async def drag(self, path: Any, **kw: Any) -> None:
        self.calls.append(("drag", tuple(path)))

    @override
    async def wait(self, duration_ms: int) -> None:
        self.calls.append(("wait", duration_ms))


def test_default_spec() -> None:
    assert GeminiComputerTool.default_spec("any") is GEMINI_COMPUTER_SPEC


async def test_click_at() -> None:
    tool = RecordingGemini()
    await tool.execute({"action": "click_at", "x": 10, "y": 20})
    assert ("click", 10, 20) in tool.calls
    assert tool.calls[-1] == ("screenshot",)


async def test_type_text_at_without_clear() -> None:
    tool = RecordingGemini()
    await tool.execute(
        {"action": "type_text_at", "x": 5, "y": 6, "text": "hi", "clear_before_typing": False}
    )
    assert ("move", 5, 6) in tool.calls
    assert ("type", "hi") in tool.calls


async def test_scroll_at_down() -> None:
    tool = RecordingGemini()
    await tool.execute({"action": "scroll_at", "x": 0, "y": 0, "direction": "down", "magnitude": 3})
    scrolls = [c for c in tool.calls if c[0] == "scroll"]
    assert scrolls and scrolls[0][3]["scroll_y"] == 3


async def test_key_combination() -> None:
    tool = RecordingGemini()
    await tool.execute({"action": "key_combination", "keys": "ctrl+c"})
    assert ("keys", ("Control_L", "c")) in tool.calls


async def test_key_combination_requires_string() -> None:
    tool = RecordingGemini()
    assert (await tool.execute({"action": "key_combination", "keys": 123})).isError


async def test_wait_and_drag() -> None:
    tool = RecordingGemini()
    await tool.execute({"action": "wait_5_seconds"})
    await tool.execute(
        {"action": "drag_and_drop", "x": 50, "y": 50, "destination_x": 100, "destination_y": 100}
    )
    assert ("wait", 5000) in tool.calls
    assert any(c[0] == "drag" for c in tool.calls)


async def test_missing_action_errors() -> None:
    tool = RecordingGemini()
    assert (await tool.execute({})).isError


async def test_unknown_action_errors() -> None:
    tool = RecordingGemini()
    assert (await tool.execute({"action": "fly_to_moon"})).isError

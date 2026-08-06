"""``OpenAIComputerTool`` — key mapping + computer-call dispatch to RFB primitives.

No live VNC: a recording subclass captures the primitive calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from hud.agents.openai.tools.computer import (
    OpenAIComputerTool,
    _hold_keys,
    _map_key,
)
from hud.agents.tools.base import result_text, tool_ok

if TYPE_CHECKING:
    from collections.abc import Iterable


class RecordingOpenAI(OpenAIComputerTool):
    client: Any

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.client = SimpleNamespace(width=200, height=100)

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
        kw = {"button": button, "hold_keys": hold_keys}
        if count != 1:
            kw["count"] = count
        if interval_ms:
            kw["interval_ms"] = interval_ms
        self.calls.append(("click", x, y, kw))

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


def test_key_mapping() -> None:
    assert _map_key("ctrl") == "Control_L"
    assert _map_key("x") == "x"
    assert _hold_keys(["ctrl", "c"]) == ["Control_L", "c"]
    assert _hold_keys("notalist") is None


def test_to_params() -> None:
    assert RecordingOpenAI().to_params() == {"type": "computer"}


async def test_click_returns_screenshot() -> None:
    tool = RecordingOpenAI()
    result = await tool.execute({"type": "click", "x": 1, "y": 2, "button": "left"})
    assert ("click", 1, 2, {"button": "left", "hold_keys": None}) in tool.calls
    assert not result.isError


async def test_type_and_keypress() -> None:
    tool = RecordingOpenAI()
    await tool.execute({"type": "type", "text": "hi"})
    await tool.execute({"type": "keypress", "keys": ["ctrl", "c"]})
    assert ("type", "hi") in tool.calls
    assert ("keys", ("Control_L", "c")) in tool.calls


async def test_drag_and_wait() -> None:
    tool = RecordingOpenAI()
    await tool.execute({"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 5, "y": 5}]})
    await tool.execute({"type": "wait", "ms": 500})
    assert ("drag", ((0, 0), (5, 5))) in tool.calls
    assert ("wait", 500) in tool.calls


async def test_response_action_returns_text() -> None:
    tool = RecordingOpenAI()
    result = await tool.execute({"type": "response", "text": "all done"})
    assert result_text(result) == "all done"


async def test_actions_list_runs_each() -> None:
    tool = RecordingOpenAI()
    await tool.execute(
        {"actions": [{"type": "move", "x": 3, "y": 4}, {"type": "type", "text": "a"}]}
    )
    assert ("move", 3, 4) in tool.calls
    assert ("type", "a") in tool.calls


async def test_empty_actions_errors() -> None:
    tool = RecordingOpenAI()
    assert (await tool.execute({"actions": []})).isError


async def test_invalid_type_errors() -> None:
    tool = RecordingOpenAI()
    assert (await tool.execute({"type": "frobnicate"})).isError
    assert (await tool.execute({})).isError

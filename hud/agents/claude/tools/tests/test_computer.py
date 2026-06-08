"""``ClaudeComputerTool`` — key translation, per-model spec gating, and the
computer-use action dispatch (translation to RFB primitives), without a live VNC.
"""
# pyright: reportPrivateUsage=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from hud.agents.claude.tools.computer import (
    CLAUDE_COMPUTER_SPECS,
    ClaudeComputerTool,
    _hold_keys,
    _split_keys,
)
from hud.agents.tools.base import result_text, tool_ok


class RecordingComputer(ClaudeComputerTool):
    """Bypasses RFBTool init; records the primitive calls dispatch makes."""

    client: Any

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.client = SimpleNamespace(width=200, height=100)

    async def screenshot(self) -> Any:
        self.calls.append(("screenshot",))
        return tool_ok("shot")

    async def click(self, x: Any, y: Any, **kw: Any) -> None:
        self.calls.append(("click", x, y, kw))

    async def move(self, x: Any, y: Any) -> None:
        self.calls.append(("move", x, y))

    async def mouse_down(self, button: Any) -> None:
        self.calls.append(("down", button))

    async def mouse_up(self, button: Any) -> None:
        self.calls.append(("up", button))

    async def type_text(self, text: Any) -> None:
        self.calls.append(("type", text))

    async def press_keys(self, keys: Any, **kw: Any) -> None:
        self.calls.append(("keys", tuple(keys), kw))

    async def hold_key(self, key: Any, **kw: Any) -> None:
        self.calls.append(("hold", key, kw))

    async def scroll(self, x: Any, y: Any, **kw: Any) -> None:
        self.calls.append(("scroll", x, y, kw))

    async def drag(self, path: Any, **kw: Any) -> None:
        self.calls.append(("drag", tuple(path), kw))

    async def wait(self, ms: Any) -> None:
        self.calls.append(("wait", ms))


# ─── key translation helpers ──────────────────────────────────────────


def test_split_and_hold_keys() -> None:
    assert _split_keys("ctrl+c") == ["Control_L", "c"]
    assert _split_keys("Return") == ["Return"]
    assert _split_keys(None) == []
    assert _split_keys("") == []
    assert _hold_keys(None) is None
    assert _hold_keys("alt") == ["Alt_L"]


# ─── spec gating + params ─────────────────────────────────────────────


def test_default_spec_per_model() -> None:
    spec_45 = ClaudeComputerTool.default_spec("claude-sonnet-4-5-20250101")
    assert spec_45 is not None
    assert spec_45.api_type == "computer_20250124"
    # Unknown model falls back to the latest spec.
    spec_unknown = ClaudeComputerTool.default_spec("totally-unknown")
    assert spec_unknown is not None
    assert spec_unknown.api_type == "computer_20251124"


def test_to_params_reflects_spec_version() -> None:
    tool = RecordingComputer()
    tool.spec = CLAUDE_COMPUTER_SPECS[0]
    assert tool.to_params()["type"] == "computer_20251124"
    tool.spec = CLAUDE_COMPUTER_SPECS[1]
    assert tool.to_params()["type"] == "computer_20250124"


# ─── action dispatch ──────────────────────────────────────────────────


async def test_left_click_then_screenshot() -> None:
    tool = RecordingComputer()
    await tool.execute({"action": "left_click", "coordinate": [10, 20], "text": "ctrl"})
    assert tool.calls[0] == ("click", 10, 20, {"hold_keys": ["Control_L"]})
    assert tool.calls[-1] == ("screenshot",)


async def test_type_action() -> None:
    tool = RecordingComputer()
    await tool.execute({"action": "type", "text": "hello"})
    assert ("type", "hello") in tool.calls


async def test_key_action_translates_chord() -> None:
    tool = RecordingComputer()
    await tool.execute({"action": "key", "text": "ctrl+c"})
    assert any(c[0] == "keys" and c[1] == ("Control_L", "c") for c in tool.calls)


async def test_mouse_move_and_down() -> None:
    tool = RecordingComputer()
    await tool.execute({"action": "mouse_move", "coordinate": [5, 6]})
    await tool.execute({"action": "left_mouse_down"})
    assert ("move", 5, 6) in tool.calls
    assert ("down", "left") in tool.calls


async def test_screenshot_only() -> None:
    tool = RecordingComputer()
    await tool.execute({"action": "screenshot"})
    assert tool.calls == [("screenshot",)]


async def test_key_without_text_errors() -> None:
    tool = RecordingComputer()
    result = await tool.execute({"action": "key"})
    assert result.isError


async def test_unsupported_action_errors() -> None:
    tool = RecordingComputer()
    result = await tool.execute({"action": "frobnicate"})
    assert result.isError
    assert "unsupported" in result_text(result).lower()

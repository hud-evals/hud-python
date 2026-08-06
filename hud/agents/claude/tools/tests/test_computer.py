"""``ClaudeComputerTool`` — key translation, per-model spec gating, and the
computer-use action dispatch (translation to RFB primitives), without a live VNC.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import mcp.types as mcp_types
from PIL import Image
from typing_extensions import override

from hud.agents.claude.tools.computer import (
    CLAUDE_COMPUTER_SPECS,
    ClaudeComputerTool,
    _crop_png,
    _hold_keys,
    _split_keys,
    _translate_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
from hud.agents.tools.base import result_text, tool_ok


class RecordingComputer(ClaudeComputerTool):
    """Bypasses RFBTool init; records the primitive calls dispatch makes."""

    client: Any

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.screenshot_mime_type = "image/webp"
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
        kw: dict[str, Any] = {}
        if button != "left":
            kw["button"] = button
        if hold_keys is not None:
            kw["hold_keys"] = hold_keys
        if count != 1:
            kw["count"] = count
        if interval_ms:
            kw["interval_ms"] = interval_ms
        self.calls.append(("click", x, y, kw))

    @override
    async def move(self, x: Any, y: Any) -> None:
        self.calls.append(("move", x, y))

    @override
    async def mouse_down(self, button: Any = "left") -> None:
        self.calls.append(("down", button))

    @override
    async def mouse_up(self, button: Any = "left") -> None:
        self.calls.append(("up", button))

    @override
    async def type_text(self, text: Any) -> None:
        self.calls.append(("type", text))

    @override
    async def press_keys(self, keys: Any, **kw: Any) -> None:
        self.calls.append(("keys", tuple(keys), kw))

    @override
    async def hold_key(self, key: Any, **kw: Any) -> None:
        self.calls.append(("hold", key, kw))

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
        self.calls.append(("drag", tuple(path), kw))

    @override
    async def wait(self, duration_ms: int) -> None:
        self.calls.append(("wait", duration_ms))


# ─── key translation helpers ──────────────────────────────────────────


def test_translate_key_maps_anthropic_to_x11() -> None:
    assert _translate_key("Return") == "Return"
    assert _translate_key("ctrl") == "Control_L"
    assert _translate_key("ctrl+c") == "Control_L+c"


def test_split_and_hold_keys() -> None:
    assert _split_keys("ctrl+c") == ["Control_L", "c"]
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


def test_crop_reports_encoded_mime_type() -> None:
    source = BytesIO()
    Image.effect_noise((128, 128), 100).convert("RGB").save(source, format="PNG")

    cropped, mime_type = _crop_png(source.getvalue(), (0, 0, 128, 128), "image/webp")

    assert mime_type == "image/webp"
    assert cropped.startswith(b"RIFF")


async def test_zoom_reports_encoded_mime_type() -> None:
    tool = RecordingComputer()
    tool.client.screenshot_png = AsyncMock(return_value=(b"png", "image/png"))

    with patch(
        "hud.agents.claude.tools.computer._crop_png",
        return_value=(b"webp", "image/webp"),
    ) as crop:
        result = await tool._zoom({"region": [0, 0, 10, 10]})

    tool.client.screenshot_png.assert_awaited_once_with("image/png")
    crop.assert_called_once_with(b"png", (0, 0, 10, 10), "image/webp")
    image = next(block for block in result.content if isinstance(block, mcp_types.ImageContent))
    assert image.mimeType == "image/webp"

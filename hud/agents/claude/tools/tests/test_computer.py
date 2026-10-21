"""``ClaudeComputerTool`` — key translation, per-model spec gating, and the
computer-use action dispatch (translation to RFB primitives), without a live VNC.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, call, patch

import mcp.types as mcp_types
import pytest
from PIL import Image

from hud.agents.claude.tools.computer import (
    CLAUDE_COMPUTER_SPECS,
    GENERIC_COMPUTER_SPEC,
    ClaudeComputerTool,
    _crop_png,
    _hold_keys,
    _split_keys,
    _translate_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
from hud.agents.tools.base import result_text, tool_ok
from hud.capabilities.rfb import WebPScreenshotEncoding


class RecordingComputer(ClaudeComputerTool):
    """Bypasses RFBTool init; records the primitive calls dispatch makes."""

    client: Any

    def __init__(self) -> None:
        self.spec = CLAUDE_COMPUTER_SPECS[0]
        self.calls: list[tuple[Any, ...]] = []
        self.screenshot_encoding = WebPScreenshotEncoding(quality=42)
        self.client = SimpleNamespace(width=200, height=100)

    async def screenshot(self) -> Any:
        self.calls.append(("screenshot",))
        return tool_ok("shot")

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

    async def move(self, x: Any, y: Any) -> None:
        self.calls.append(("move", x, y))

    async def mouse_down(self, button: Any = "left") -> None:
        self.calls.append(("down", button))

    async def mouse_up(self, button: Any = "left") -> None:
        self.calls.append(("up", button))

    async def type_text(self, text: Any) -> None:
        self.calls.append(("type", text))

    async def press_keys(self, keys: Any, **kw: Any) -> None:
        self.calls.append(("keys", tuple(keys), kw))

    async def hold_key(self, key: Any, **kw: Any) -> None:
        self.calls.append(("hold", key, kw))

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

    async def drag(self, path: Any, **kw: Any) -> None:
        self.calls.append(("drag", tuple(path), kw))


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
    # Unknown Claude model falls back to the latest spec.
    spec_unknown = ClaudeComputerTool.default_spec("claude-totally-unknown")
    assert spec_unknown is not None
    assert spec_unknown.api_type == "computer_20251124"
    # Non-Anthropic models get the generic input_schema spec.
    assert ClaudeComputerTool.default_spec("qwen/qwen3.8-max") is GENERIC_COMPUTER_SPEC


def test_to_params_reflects_spec_version() -> None:
    tool = RecordingComputer()
    tool.spec = CLAUDE_COMPUTER_SPECS[0]
    assert tool.to_params().get("type") == "computer_20251124"
    tool.spec = CLAUDE_COMPUTER_SPECS[1]
    assert tool.to_params().get("type") == "computer_20250124"


def test_to_params_generic_spec_uses_input_schema() -> None:
    tool = RecordingComputer()
    tool.spec = GENERIC_COMPUTER_SPEC
    params: Any = tool.to_params()
    assert "type" not in params
    assert params["name"] == "computer"
    assert "200x100" in params["description"]
    assert params["input_schema"]["required"] == ["action"]
    assert "left_click" in params["input_schema"]["properties"]["action"]["enum"]


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

    cropped, mime_type = _crop_png(
        source.getvalue(),
        (0, 0, 128, 128),
        WebPScreenshotEncoding(quality=42),
    )

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
    crop.assert_called_once_with(b"png", (0, 0, 10, 10), tool.screenshot_encoding)
    image = next(block for block in result.content if isinstance(block, mcp_types.ImageContent))
    assert image.mimeType == "image/webp"


async def test_key_repeat_executes_every_press() -> None:
    tool = RecordingComputer()
    spec = ClaudeComputerTool.default_spec("claude-sonnet-5-5")
    assert spec is not None
    tool.spec = spec
    result = await tool.execute({"action": "key", "text": "Tab", "repeat": 100})
    assert not result.isError
    assert len(tool.calls) == 100
    assert all(item == ("keys", ("Tab",), {}) for item in tool.calls)


@pytest.mark.parametrize("repeat", [0, 101, True, "2", 2.5])
async def test_key_repeat_rejects_out_of_contract_values(repeat: Any) -> None:
    tool = RecordingComputer()
    result = await tool.execute({"action": "key", "text": "Tab", "repeat": repeat})
    assert result.isError
    assert tool.calls == []


@pytest.mark.parametrize("duration", [-1, 301, float("inf"), "3", None, True])
async def test_toolset_wait_rejects_invalid_duration(duration: Any) -> None:
    tool = RecordingComputer()
    spec = ClaudeComputerTool.default_spec("claude-sonnet-5-5")
    assert spec is not None
    tool.spec = spec
    result = await tool.execute({"action": "wait", "duration": duration})
    assert result.isError
    assert tool.calls == []


async def test_toolset_resizes_screenshots_and_remaps_pointer_and_zoom() -> None:
    import base64
    import math

    from hud.capabilities import RFBClient

    source = BytesIO()
    Image.new("RGB", (3840, 2160), "red").save(source, format="PNG")
    client = Mock(spec=RFBClient)
    client.width, client.height = 3840, 2160
    client.conn = SimpleNamespace(mouse=Mock(x=1920, y=1080))
    client.screenshot_png = AsyncMock(return_value=(source.getvalue(), "image/png"))
    client.drain = AsyncMock()
    spec = ClaudeComputerTool.default_spec("claude-sonnet-5-5")
    assert spec is not None
    tool = ClaudeComputerTool(spec=spec, client=client)

    screenshot = await tool.execute({"action": "screenshot"})
    assert not screenshot.isError
    image = screenshot.content[0]
    assert isinstance(image, mcp_types.ImageContent)
    with Image.open(BytesIO(base64.b64decode(image.data))) as frame:
        width, height = frame.size
        assert max(width, height) <= 2000
        assert math.ceil(width / 28) * math.ceil(height / 28) <= 4784
    result = await tool.execute({"action": "left_click", "coordinate": [width // 2, height // 2]})
    assert result_text(result) == "OK"
    client.conn.mouse.move.assert_called_once_with(
        (width // 2) * 3840 // width, (height // 2) * 2160 // height
    )
    cursor = await tool.execute({"action": "cursor_position"})
    assert result_text(cursor) == f"({1920 * width // 3840}, {1080 * height // 2160})"
    zoom = await tool.execute({"action": "zoom", "region": [0, 0, width, height]})
    assert not zoom.isError
    zoom_image = zoom.content[0]
    assert isinstance(zoom_image, mcp_types.ImageContent)
    with Image.open(BytesIO(base64.b64decode(zoom_image.data))) as frame:
        assert frame.size == (width, height)
        assert frame.getpixel((0, 0)) == (255, 0, 0)


async def test_toolset_hold_key_preserves_chord_and_flushes_while_held() -> None:
    from hud.capabilities import RFBClient

    client = Mock(spec=RFBClient)
    client.width, client.height = 1280, 720
    keyboard = Mock()
    keyboard.hold.return_value.__enter__ = Mock()
    keyboard.hold.return_value.__exit__ = Mock(return_value=False)
    client.conn = SimpleNamespace(keyboard=keyboard)
    client.drain = AsyncMock()
    spec = ClaudeComputerTool.default_spec("claude-sonnet-5-5")
    assert spec is not None
    tool = ClaudeComputerTool(spec=spec, client=client)
    with patch("hud.agents.tools.rfb.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await tool.execute({"action": "hold_key", "text": "ctrl+a", "duration": 0.01})
    assert not result.isError
    keyboard.hold.assert_called_once_with("Control_L", "a")
    sleep.assert_awaited_once_with(0.01)
    assert client.drain.await_count == 2


@pytest.mark.parametrize(("direction", "button"), [("left", 5), ("right", 6)])
async def test_toolset_horizontal_scroll_sends_rfb_wheel_events(
    direction: str, button: int
) -> None:
    from hud.capabilities import RFBClient

    client = Mock(spec=RFBClient)
    client.width, client.height = 1280, 720
    client.conn = SimpleNamespace(mouse=Mock())
    client.drain = AsyncMock()
    spec = ClaudeComputerTool.default_spec("claude-sonnet-5-5")
    assert spec is not None
    tool = ClaudeComputerTool(spec=spec, client=client)
    result = await tool.execute(
        {"action": "scroll", "scroll_direction": direction, "scroll_amount": 2}
    )
    assert not result.isError
    assert client.conn.mouse.click.call_args_list == [call(button), call(button)]
    client.drain.assert_awaited_once_with()

"""ClaudeComputerTool: Claude's native ``computer_use`` schema, driven over RFB/VNC.

Translates Claude's computer-use action vocabulary into ``RFBTool`` primitive
calls. The same RFBTool helpers will back the future Gemini/OpenAI computer
tools — only the LLM-facing schema differs.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from hud.agents.tools.base import tool_err, tool_ok
from hud.agents.tools.rfb import RFBTool, keysym
from hud.capabilities.rfb import ScreenshotEncoding, WebPScreenshotEncoding
from hud.types import ImageContent, ToolResult

from .base import ClaudeToolSpec, is_anthropic_model

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaComputerToolset20260801Param,
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
        BetaToolUnionParam,
    )

    from hud.agents.tools.rfb import Button

logger = logging.getLogger(__name__)


def _split_keys(text: str | None) -> list[str]:
    if not text:
        return []
    return [keysym(part.strip()) for part in text.split("+") if part.strip()]


def _hold_keys(text: str | None) -> list[str] | None:
    keys = _split_keys(text)
    return keys or None


# ─── Claude tool specs (per-model gating) ───────────────────────────

# Current Claude models take the computer toolset; only the older models listed in
# ``CLAUDE_COMPUTER_SPECS`` take a single computer tool.
COMPUTER_TOOLSET_SPEC = ClaudeToolSpec(api_type="computer_toolset_20260801", api_name="computer")

CLAUDE_COMPUTER_SPECS: tuple[ClaudeToolSpec, ...] = (
    ClaudeToolSpec(
        api_type="computer_20251124",
        api_name="computer",
        beta="computer-use-2025-11-24",
        supported_models=(
            "*claude-opus-4-5*",
            "*claude-opus-4-6*",
            "*claude-sonnet-4-6*",
            "*claude-opus-4-7*",
            "*claude-opus-4-8*",
        ),
    ),
    ClaudeToolSpec(
        api_type="computer_20250124",
        api_name="computer",
        beta="computer-use-2025-01-24",
        # Claude Opus 4 and Sonnet 4 go by bare, ``-0`` alias, dated, and Vertex ``@`` ids.
        supported_models=(
            "*claude-sonnet-4-5*",
            "*claude-haiku-4-5*",
            "*claude-opus-4-1*",
            "*claude-opus-4",
            "*claude-opus-4-0*",
            "*claude-opus-4-2025*",
            "*claude-opus-4@*",
            "*claude-sonnet-4",
            "*claude-sonnet-4-0*",
            "*claude-sonnet-4-2025*",
            "*claude-sonnet-4@*",
        ),
    ),
)

GENERIC_COMPUTER_SPEC = ClaudeToolSpec(
    api_type="function",
    api_name="computer",
)

_COMPUTER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "screenshot",
                "zoom",
                "left_click",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "mouse_move",
                "left_mouse_down",
                "left_mouse_up",
                "type",
                "key",
                "hold_key",
                "scroll",
                "left_click_drag",
                "wait",
                "cursor_position",
            ],
            "description": "The action to perform on the screen.",
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] pixel coordinate for click/move/scroll, or drag end point.",
        },
        "start_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] drag start point for `left_click_drag`.",
        },
        "text": {
            "type": "string",
            "description": (
                "Text to type for `type`; key chord (xdotool syntax, e.g. `ctrl+s`) for "
                "`key`/`hold_key`; optional modifier keys held during clicks and scrolls."
            ),
        },
        "duration": {
            "type": "number",
            "description": "Seconds to hold for `hold_key` or pause for `wait`.",
        },
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
            "description": "Direction for `scroll`.",
        },
        "scroll_amount": {
            "type": "integer",
            "description": "Number of scroll clicks for `scroll`.",
        },
        "repeat": {
            "type": "integer",
            "description": "Times to repeat the key chord for `key`.",
        },
        "region": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x0, y0, x1, y1] region to magnify for `zoom`.",
        },
    },
    "required": ["action"],
}


class ClaudeComputerTool(RFBTool):
    """Claude's native ``computer_use`` schema, executed over an RFB capability."""

    name = "computer"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        if not is_anthropic_model(model):
            return GENERIC_COMPUTER_SPEC
        for candidate in CLAUDE_COMPUTER_SPECS:
            if candidate.supports_model(model):
                return candidate
        return COMPUTER_TOOLSET_SPEC

    @property
    def is_toolset(self) -> bool:
        """Whether the model calls this tool as the computer toolset, one tool per action."""
        return self.spec.api_type == COMPUTER_TOOLSET_SPEC.api_type

    def to_params(
        self,
    ) -> (
        BetaComputerToolset20260801Param
        | BetaToolComputerUse20250124Param
        | BetaToolComputerUse20251124Param
        | BetaToolUnionParam
    ):
        if self.is_toolset:
            # The model calls each member as its own tool and sizes clicks from screenshots.
            return cast("BetaComputerToolset20260801Param", {"type": "computer_toolset_20260801"})
        if self.spec.api_type == "function":
            return cast(
                "BetaToolUnionParam",
                {
                    "name": self.name,
                    "description": (
                        "Control the remote computer's screen, mouse, and keyboard. "
                        f"The display is {self.display_width}x{self.display_height} pixels. "
                        "Most actions return a screenshot of the result."
                    ),
                    "input_schema": _COMPUTER_INPUT_SCHEMA,
                },
            )
        if self.spec.api_type == "computer_20251124":
            return cast(
                "BetaToolComputerUse20251124Param",
                {
                    "type": "computer_20251124",
                    "name": self.name,
                    "display_width_px": self.display_width,
                    "display_height_px": self.display_height,
                    "display_number": 1,
                    "enable_zoom": True,
                },
            )
        return cast(
            "BetaToolComputerUse20250124Param",
            {
                "type": "computer_20250124",
                "name": self.name,
                "display_width_px": self.display_width,
                "display_height_px": self.display_height,
                "display_number": 1,
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("ClaudeComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    # ─── action dispatch ──────────────────────────────────────────────

    async def _dispatch(self, action: str | None, arguments: dict[str, Any]) -> ToolResult:
        match action:
            case "screenshot":
                return await self.screenshot()

            case "zoom":
                return await self._zoom(arguments)

            case "left_click" | "click":
                x, y = _xy(arguments.get("coordinate"))
                await self.click(x, y, hold_keys=_hold_keys(arguments.get("text")))

            case "right_click":
                x, y = _xy(arguments.get("coordinate"))
                await self.click(x, y, button="right", hold_keys=_hold_keys(arguments.get("text")))

            case "middle_click":
                x, y = _xy(arguments.get("coordinate"))
                await self.click(
                    x,
                    y,
                    button="middle",
                    hold_keys=_hold_keys(arguments.get("text")),
                )

            case "double_click":
                x, y = _xy(arguments.get("coordinate"))
                await self.click(
                    x,
                    y,
                    count=2,
                    interval_ms=100,
                    hold_keys=_hold_keys(arguments.get("text")),
                )

            case "triple_click":
                x, y = _xy(arguments.get("coordinate"))
                await self.click(
                    x,
                    y,
                    count=3,
                    interval_ms=100,
                    hold_keys=_hold_keys(arguments.get("text")),
                )

            case "mouse_move" | "move":
                x, y = _xy(arguments.get("coordinate"))
                await self.move(_required(x, "coordinate.x"), _required(y, "coordinate.y"))

            case "left_mouse_down":
                await self.mouse_down("left")

            case "left_mouse_up":
                await self.mouse_up("left")

            case "type":
                text = arguments.get("text")
                if not isinstance(text, str):
                    return tool_err("`text` is required for type")
                await self.type_text(text)

            case "key":
                text = arguments.get("text")
                # xdotool syntax: space-separated sequential presses, + binds a chord
                chords = (
                    [chord for part in text.split() if (chord := _split_keys(part))]
                    if isinstance(text, str)
                    else []
                )
                if not chords:
                    return tool_err("`text` (key chord) is required for key")
                repeat = arguments.get("repeat")
                count = repeat if isinstance(repeat, int) and repeat > 0 else 1
                for _ in range(min(count, 100)):
                    for chord in chords:
                        await self.press_keys(chord)

            case "hold_key":
                keys = _split_keys(arguments.get("text"))
                if not keys:
                    return tool_err("`text` is required for hold_key")
                duration = _ms_from_seconds(arguments.get("duration"))
                await self.hold_key(keys[0], duration_ms=duration)

            case "scroll":
                x, y = _xy(arguments.get("coordinate"))
                sx, sy = _scroll(arguments)
                await self.scroll(
                    x,
                    y,
                    scroll_x=sx,
                    scroll_y=sy,
                    hold_keys=_hold_keys(arguments.get("text")),
                )

            case "left_click_drag" | "drag":
                path = _drag_path(arguments)
                button: Button = "left"
                await self.drag(path, button=button, hold_keys=_hold_keys(arguments.get("text")))

            case "wait":
                duration = _ms_from_seconds(arguments.get("duration"))
                await self.wait(duration)

            case "cursor_position":
                mouse = self.client.conn.mouse
                return tool_ok(f"({mouse.x}, {mouse.y})")

            case _:
                return tool_err(f"unsupported computer action: {action!r}")

        # Most actions return the post-action screenshot so the model can verify.
        return await self.screenshot()

    # ─── zoom ────────────────────────────────────────────────────────

    async def _zoom(self, arguments: dict[str, Any]) -> ToolResult:
        region = arguments.get("region")
        if not isinstance(region, (list, tuple)):
            return tool_err("region must be [x0, y0, x1, y1]")
        region_seq = cast("list[Any]", region)
        if len(region_seq) != 4:
            return tool_err("region must be [x0, y0, x1, y1]")
        try:
            x0, y0, x1, y1 = (int(v) for v in region_seq)
        except (TypeError, ValueError):
            return tool_err("region must contain 4 integers")
        screenshot, _ = await self.client.screenshot_png("image/png")
        cropped, mime_type = _crop_png(screenshot, (x0, y0, x1, y1), self.screenshot_encoding)
        return ToolResult(
            content=[
                ImageContent(
                    type="image",
                    mimeType=mime_type,
                    data=base64.b64encode(cropped).decode("ascii"),
                )
            ],
        )


# ─── helpers ─────────────────────────────────────────────────────────


def _xy(coordinate: Any) -> tuple[int | None, int | None]:
    if not isinstance(coordinate, (list, tuple)):
        return None, None
    seq = cast("list[Any]", coordinate)
    if len(seq) < 2:
        return None, None
    try:
        return int(seq[0]), int(seq[1])
    except (TypeError, ValueError):
        return None, None


def _required(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _ms_from_seconds(duration: Any) -> int:
    try:
        return int(float(duration or 0) * 1000)
    except (TypeError, ValueError):
        return 0


def _scroll(arguments: dict[str, Any]) -> tuple[int, int]:
    amount = arguments.get("scroll_amount")
    amount = amount if isinstance(amount, int) and amount >= 0 else 0
    match arguments.get("scroll_direction"):
        case "down":
            return 0, amount
        case "up":
            return 0, -amount
        case "right":
            return amount, 0
        case "left":
            return -amount, 0
        case _:
            return 0, 0


def _drag_path(arguments: dict[str, Any]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for key in ("start_coordinate", "coordinate"):
        raw = arguments.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        seq = cast("list[Any]", raw)
        if len(seq) >= 2:
            path.append((int(seq[0]), int(seq[1])))
    if len(path) < 2:
        raise ValueError("drag requires start_coordinate and coordinate")
    return path


def _crop_png(
    png: bytes,
    region: tuple[int, int, int, int],
    encoding: ScreenshotEncoding,
) -> tuple[bytes, str]:
    from PIL import Image

    image = Image.open(BytesIO(png))
    cropped = image.crop(region)
    buf = BytesIO()
    if isinstance(encoding, WebPScreenshotEncoding):
        cropped.save(buf, format="WEBP", quality=encoding.quality)
    else:
        cropped.save(buf, format="PNG")
    return buf.getvalue(), encoding.mime_type

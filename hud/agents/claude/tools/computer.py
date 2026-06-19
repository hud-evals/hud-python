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

import mcp.types as mcp_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err, tool_ok
from hud.types import MCPToolResult

from .base import ClaudeToolSpec

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
    )

    from hud.agents.tools.rfb import Button

logger = logging.getLogger(__name__)


# ─── Anthropic → X11 keysym translation ─────────────────────────────
#
# Claude emits keys in the xdotool / Anthropic vocabulary (``Return``,
# ``Page_Down``, ``Control_L``, ``cmd``, etc.). asyncvnc's keysymdef table
# accepts X11 names directly and already aliases common short forms (``Cmd``,
# ``Alt``, ``Ctrl``, ``Super``, ``Shift``, ``Backspace``, ``Del``, ``Esc``).
# This map covers the residual Anthropic-specific spellings.

_ANTHROPIC_TO_X11: dict[str, str] = {
    "alt": "Alt_L",
    "ctrl": "Control_L",
    "shift": "Shift_L",
    "meta": "Super_L",
    "super": "Super_L",
    "win": "Super_L",
    "cmd": "Super_L",
    "command": "Super_L",
    "option": "Alt_L",
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "del": "Delete",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "arrowup": "Up",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "space": "space",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "printscreen": "Print",
}


def _translate_key(token: str) -> str:
    if "+" in token:
        return "+".join(_translate_key(part) for part in token.split("+"))
    return _ANTHROPIC_TO_X11.get(token.lower(), token)


def _split_keys(text: str | None) -> list[str]:
    if not text:
        return []
    return [_translate_key(part.strip()) for part in text.split("+") if part.strip()]


def _hold_keys(text: str | None) -> list[str] | None:
    keys = _split_keys(text)
    return keys or None


# ─── Claude tool specs (per-model gating) ───────────────────────────


CLAUDE_COMPUTER_SPECS: tuple[ClaudeToolSpec, ...] = (
    ClaudeToolSpec(
        api_type="computer_20251124",
        api_name="computer",
        beta="computer-use-2025-11-24",
        supported_models=(
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
        supported_models=(
            "*claude-sonnet-4-5*",
            "*claude-haiku-4-5*",
        ),
    ),
)

# Fallback for unknown models — use the latest version.
_DEFAULT_COMPUTER_SPEC = CLAUDE_COMPUTER_SPECS[0]


class ClaudeComputerTool(RFBTool):
    """Claude's native ``computer_use`` schema, executed over an RFB capability."""

    name = "computer"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        for candidate in CLAUDE_COMPUTER_SPECS:
            if candidate.supports_model(model):
                return candidate
        return _DEFAULT_COMPUTER_SPEC

    def to_params(self) -> BetaToolComputerUse20250124Param | BetaToolComputerUse20251124Param:
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

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("ClaudeComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    # ─── action dispatch ──────────────────────────────────────────────

    async def _dispatch(self, action: str | None, arguments: dict[str, Any]) -> MCPToolResult:
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
                keys = _split_keys(arguments.get("text"))
                if not keys:
                    return tool_err("`text` (key chord) is required for key")
                repeat = arguments.get("repeat")
                count = repeat if isinstance(repeat, int) and repeat > 0 else 1
                await self.press_keys(keys, count=min(count, 100))

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

    async def _zoom(self, arguments: dict[str, Any]) -> MCPToolResult:
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
        png = await self.client.screenshot_png()
        cropped = _crop_png(png, (x0, y0, x1, y1))
        return MCPToolResult(
            content=[
                mcp_types.ImageContent(
                    type="image",
                    mimeType="image/png",
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


def _crop_png(png: bytes, region: tuple[int, int, int, int]) -> bytes:
    from PIL import Image

    image = Image.open(BytesIO(png))
    cropped = image.crop(region)
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


__all__ = ["CLAUDE_COMPUTER_SPECS", "ClaudeComputerTool"]

"""Gemini Computer Use tool — backed by RFBClient."""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING, Any, TypeVar

from google.genai import types as genai_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err

from .base import GeminiToolSpec

if TYPE_CHECKING:
    from hud.types import MCPToolResult

logger = logging.getLogger(__name__)

GEMINI_DRAG_INSET = 25
IS_MAC = platform.system().lower() == "darwin"

_T = TypeVar("_T")


def _mac_else(mac: _T, other: _T) -> _T:
    """Pick the macOS variant when on macOS, else the non-macOS variant."""
    return mac if IS_MAC else other


_GEMINI_KEY_ALIASES: dict[str, str] = {
    "ctrl": "Control_L",
    "control": "Control_L",
    "alt": "Alt_L",
    "option": "Alt_L",
    "shift": "Shift_L",
    "meta": _mac_else("Super_L", "Control_L"),
    "super": "Super_L",
    "win": "Super_L",
    "cmd": "Super_L",
    "command": "Super_L",
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "del": "Delete",
    "delete": "Delete",
    "backspace": "BackSpace",
    "tab": "Tab",
    "space": "space",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "arrowup": "Up",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "capslock": "Caps_Lock",
    "printscreen": "Print",
}
_SCROLL_DIRECTIONS: dict[str, tuple[int, int]] = {
    "down": (0, 1),
    "up": (0, -1),
    "right": (1, 0),
    "left": (-1, 0),
}

PREDEFINED_COMPUTER_USE_FUNCTIONS = (
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
)

GEMINI_COMPUTER_SPEC = GeminiToolSpec(
    api_type="computer_use",
    api_name="gemini_computer",
)


class GeminiComputerTool(RFBTool):
    """Translate Gemini predefined computer functions into RFBTool primitives."""

    name = "computer_use"

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec | None:
        del model
        return GEMINI_COMPUTER_SPEC

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            computer_use=genai_types.ComputerUse(
                environment=genai_types.Environment.ENVIRONMENT_BROWSER,
            ),
        )

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return tool_err("action is required")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("GeminiComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    async def _dispatch(self, action: str, args: dict[str, Any]) -> MCPToolResult:
        match action:
            case "open_web_browser":
                return await self.screenshot()
            case "click_at":
                await self.click(args.get("x"), args.get("y"))
            case "hover_at":
                x, y = args.get("x"), args.get("y")
                if x is not None and y is not None:
                    await self.move(int(x), int(y))
            case "type_text_at":
                x, y = args.get("x"), args.get("y")
                if x is not None and y is not None:
                    await self.move(int(x), int(y))
                    await self.click(int(x), int(y))
                if args.get("clear_before_typing", True):
                    await self.press_keys(_mac_else(["Super_L", "a"], ["Control_L", "a"]))
                    await self.press_keys([_mac_else("BackSpace", "Delete")])
                text = args.get("text")
                if isinstance(text, str) and text:
                    await self.type_text(text)
                if args.get("press_enter"):
                    await self.press_keys(["Return"])
            case "scroll_document":
                await self._scroll(args, at_pointer=False)
            case "scroll_at":
                await self._scroll(args, at_pointer=True)
            case "wait_5_seconds":
                await self.wait(5000)
            case "go_back":
                await self.press_keys(_mac_else(["Super_L", "bracketleft"], ["Alt_L", "Left"]))
            case "go_forward":
                await self.press_keys(_mac_else(["Super_L", "bracketright"], ["Alt_L", "Right"]))
            case "search":
                await self._navigate_to(str(args.get("url") or "https://www.google.com"))
            case "navigate":
                await self._navigate_to(str(args.get("url") or ""))
            case "key_combination":
                keys_str = args.get("keys")
                if not isinstance(keys_str, str):
                    return tool_err("keys must be a '+'-separated string")
                await self.press_keys(_normalize_chord(keys_str))
            case "drag_and_drop":
                path = [
                    (self._clamp_drag_coord(args.get("x")), self._clamp_drag_coord(args.get("y"))),
                    (
                        self._clamp_drag_coord(args.get("destination_x")),
                        self._clamp_drag_coord(args.get("destination_y")),
                    ),
                ]
                await self.drag(path)
            case _:
                return tool_err(f"Unknown Gemini computer action: {action}")
        return await self.screenshot()

    async def _scroll(self, args: dict[str, Any], *, at_pointer: bool) -> None:
        sx, sy = _scroll_delta(args.get("direction"), int(args.get("magnitude") or 3))
        x = args.get("x") if at_pointer else None
        y = args.get("y") if at_pointer else None
        await self.scroll(
            int(x) if x is not None else None,
            int(y) if y is not None else None,
            scroll_x=sx,
            scroll_y=sy,
        )

    async def _navigate_to(self, target: str) -> None:
        keys = _mac_else(["Super_L", "l"], ["Control_L", "l"])
        await self.press_keys(keys)
        await self.type_text(target)
        await self.press_keys(["Return"])

    def _clamp_drag_coord(self, value: Any) -> int:
        if not isinstance(value, int | float):
            return 0
        max_coord = max(self.display_width, self.display_height)
        return min(max(int(value), GEMINI_DRAG_INSET), max_coord - GEMINI_DRAG_INSET)


def _scroll_delta(direction: Any, magnitude: int) -> tuple[int, int]:
    x, y = _SCROLL_DIRECTIONS.get(str(direction), (0, 0))
    return x * magnitude, y * magnitude


def _normalize_chord(text: str) -> list[str]:
    return [
        _GEMINI_KEY_ALIASES.get(part.strip().lower(), part.strip().lower())
        for part in text.split("+")
        if part.strip()
    ]


__all__ = ["GEMINI_COMPUTER_SPEC", "PREDEFINED_COMPUTER_USE_FUNCTIONS", "GeminiComputerTool"]

"""Gemini Computer Use tool — backed by RFBClient."""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING, Any

from google.genai import types as genai_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err

from .base import GeminiToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.agents.tools.base import AgentToolSpec
    from hud.capabilities import RFBClient
    from hud.capabilities.rfb import ScreenshotEncoding
    from hud.types import MCPToolResult

logger = logging.getLogger(__name__)

GEMINI_DRAG_INSET = 25
#: Gemini places every coordinate on a 1000x1000 grid regardless of the screen size.
GEMINI_COORDINATE_SCALE = 1000
IS_MAC = platform.system().lower() == "darwin"

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

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        client: RFBClient,
        screenshot_encoding: ScreenshotEncoding,
        excluded_predefined_functions: Sequence[str] = (),
    ) -> None:
        super().__init__(spec=spec, client=client, screenshot_encoding=screenshot_encoding)
        self.excluded_predefined_functions = list(excluded_predefined_functions)

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_COMPUTER_SPEC

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            computer_use=genai_types.ComputerUse(
                environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                excluded_predefined_functions=self.excluded_predefined_functions,
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
        if action == "open_web_browser":
            return await self.screenshot()

        if action == "click_at":
            await self.click(*self._point(args))
            return await self.screenshot()

        if action == "hover_at":
            x, y = self._point(args)
            if x is not None and y is not None:
                await self.move(x, y)
            return await self.screenshot()

        if action == "type_text_at":
            x, y = self._point(args)
            if x is not None and y is not None:
                await self.move(x, y)
                await self.click(x, y)
            if args.get("clear_before_typing", True):
                select_all = ["Super_L", "a"] if IS_MAC else ["Control_L", "a"]
                delete_key = "BackSpace" if IS_MAC else "Delete"
                await self.press_keys(select_all)
                await self.press_keys([delete_key])
            text = args.get("text")
            if isinstance(text, str) and text:
                await self.type_text(text)
            if args.get("press_enter"):
                await self.press_keys(["Return"])
            return await self.screenshot()

        if action in ("scroll_document", "scroll_at"):
            direction = args.get("direction")
            magnitude = int(args.get("magnitude") or 3)
            sx, sy = 0, 0
            if direction == "down":
                sy = magnitude
            elif direction == "up":
                sy = -magnitude
            elif direction == "right":
                sx = magnitude
            elif direction == "left":
                sx = -magnitude
            x, y = self._point(args) if action == "scroll_at" else (None, None)
            await self.scroll(x, y, scroll_x=sx, scroll_y=sy)
            return await self.screenshot()

        if action == "wait_5_seconds":
            await self.wait(5000)
            return await self.screenshot()

        if action == "go_back":
            keys = ["Super_L", "bracketleft"] if IS_MAC else ["Alt_L", "Left"]
            await self.press_keys(keys)
            return await self.screenshot()

        if action == "go_forward":
            keys = ["Super_L", "bracketright"] if IS_MAC else ["Alt_L", "Right"]
            await self.press_keys(keys)
            return await self.screenshot()

        if action == "search":
            target = args.get("url") or "https://www.google.com"
            keys = ["Super_L", "l"] if IS_MAC else ["Control_L", "l"]
            await self.press_keys(keys)
            await self.type_text(str(target))
            await self.press_keys(["Return"])
            return await self.screenshot()

        if action == "navigate":
            keys = ["Super_L", "l"] if IS_MAC else ["Control_L", "l"]
            await self.press_keys(keys)
            url = args.get("url") or ""
            await self.type_text(str(url))
            await self.press_keys(["Return"])
            return await self.screenshot()

        if action == "key_combination":
            keys_str = args.get("keys")
            if not isinstance(keys_str, str):
                return tool_err("keys must be a '+'-separated string")
            aliases: dict[str, str] = {
                "control": "Control_L",
                "ctrl": "Control_L",
                "cmd": "Super_L",
                "command": "Super_L",
                "meta": "Super_L" if IS_MAC else "Control_L",
                "alt": "Alt_L",
                "shift": "Shift_L",
                "return": "Return",
                "enter": "Return",
            }
            normalized = [
                aliases.get(k, k) for part in keys_str.split("+") if (k := part.strip().lower())
            ]
            await self.press_keys(normalized)
            return await self.screenshot()

        if action == "drag_and_drop":

            def inset(value: Any, size: int) -> int:
                pixel = self._pixel(value if isinstance(value, int | float) else 0, size)
                return min(max(pixel, GEMINI_DRAG_INSET), size - GEMINI_DRAG_INSET)

            width, height = self.display_width, self.display_height
            path = [
                (inset(args.get("x"), width), inset(args.get("y"), height)),
                (inset(args.get("destination_x"), width), inset(args.get("destination_y"), height)),
            ]
            await self.drag(path)
            return await self.screenshot()

        return tool_err(f"Unknown Gemini computer action: {action}")

    @staticmethod
    def _pixel(value: float, size: int) -> int:
        """Map a coordinate on Gemini's normalized 0-999 grid onto a ``size``-pixel axis."""
        return int(value / GEMINI_COORDINATE_SCALE * size)

    def _point(self, args: dict[str, Any]) -> tuple[int | None, int | None]:
        x, y = args.get("x"), args.get("y")
        if x is None or y is None:
            return None, None
        return self._pixel(x, self.display_width), self._pixel(y, self.display_height)


__all__ = ["GEMINI_COMPUTER_SPEC", "PREDEFINED_COMPUTER_USE_FUNCTIONS", "GeminiComputerTool"]

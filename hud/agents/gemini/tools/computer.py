"""Gemini Computer Use tool — backed by RFBClient."""

from __future__ import annotations

import logging
import platform
from typing import Any, cast

from google.genai import types as genai_types

from hud.agents.tools import RFBTool
from hud.types import MCPToolResult

from .base import GeminiToolSpec

logger = logging.getLogger(__name__)

SUPPORTED_GEMINI_COMPUTER_USE_MODELS = (
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-flash-preview",
)

GEMINI_DRAG_INSET = 25
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
    supported_models=SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
)


class GeminiComputerTool(RFBTool):
    """Translate Gemini predefined computer functions into RFBTool primitives."""

    name = "computer_use"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.excluded_predefined_functions: list[str] = []

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec | None:
        return GEMINI_COMPUTER_SPEC if GEMINI_COMPUTER_SPEC.supports_model(model) else None

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
            return _err("action is required")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("GeminiComputerTool action %s failed", action)
            return _err(f"computer action {action!r} failed: {exc}")

    async def _dispatch(self, action: str, args: dict[str, Any]) -> MCPToolResult:
        if action == "open_web_browser":
            return await self.screenshot()

        if action == "click_at":
            await self.click(args.get("x"), args.get("y"))
            return await self.screenshot()

        if action == "hover_at":
            x, y = args.get("x"), args.get("y")
            if x is not None and y is not None:
                await self.move(int(x), int(y))
            return await self.screenshot()

        if action == "type_text_at":
            x, y = args.get("x"), args.get("y")
            if x is not None and y is not None:
                await self.move(int(x), int(y))
                await self.click(int(x), int(y))
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
            x = args.get("x") if action == "scroll_at" else None
            y = args.get("y") if action == "scroll_at" else None
            await self.scroll(
                int(x) if x is not None else None,
                int(y) if y is not None else None,
                scroll_x=sx, scroll_y=sy,
            )
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
                return _err("keys must be a '+'-separated string")
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
            max_coord = max(self.display_width, self.display_height)

            def clamp(v: Any) -> int:
                if not isinstance(v, int | float):
                    return 0
                return min(max(int(v), GEMINI_DRAG_INSET), max_coord - GEMINI_DRAG_INSET)

            path = [
                (clamp(args.get("x")), clamp(args.get("y"))),
                (clamp(args.get("destination_x")), clamp(args.get("destination_y"))),
            ]
            await self.drag(path)
            return await self.screenshot()

        return _err(f"Unknown Gemini computer action: {action}")


def _err(text: str) -> MCPToolResult:
    import mcp.types as mcp_types
    return MCPToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        isError=True,
    )


__all__ = ["GEMINI_COMPUTER_SPEC", "GeminiComputerTool", "PREDEFINED_COMPUTER_USE_FUNCTIONS"]

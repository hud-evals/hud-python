"""OpenAI computer tool — backed by RFBClient."""

from __future__ import annotations

import logging
from typing import Any, cast

import mcp.types as mcp_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err
from hud.types import MCPToolResult

from .base import OpenAIToolSpec

logger = logging.getLogger(__name__)

OPENAI_COMPUTER_SPEC = OpenAIToolSpec(
    api_type="computer",
    api_name="computer",
)

OPENAI_KEY_ALIASES: dict[str, str] = {
    "return": "Return",
    "escape": "Escape",
    "arrowup": "Up",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "backspace": "BackSpace",
    "delete": "Delete",
    "tab": "Tab",
    "space": "space",
    "control": "Control_L",
    "ctrl": "Control_L",
    "alt": "Alt_L",
    "shift": "Shift_L",
    "meta": "Super_L",
    "cmd": "Super_L",
    "command": "Super_L",
    "super": "Super_L",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "enter": "Return",
}

_SCREENSHOT_ACTIONS = {
    "screenshot",
    "click",
    "double_click",
    "scroll",
    "type",
    "move",
    "keypress",
    "drag",
    "wait",
}


class OpenAIComputerTool(RFBTool):
    """Translate OpenAI native computer calls into RFBTool primitives."""

    name = "computer"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        del model
        return OPENAI_COMPUTER_SPEC

    def to_params(self) -> Any:
        return {"type": "computer"}

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        actions = arguments.get("actions")
        if isinstance(actions, list):
            action_list = cast("list[Any]", actions)
            if not action_list:
                return tool_err("actions list is empty")
            result = MCPToolResult(content=[], isError=False)
            for index, raw_action in enumerate(action_list):
                if not isinstance(raw_action, dict):
                    return tool_err("actions must be objects")
                action = cast("dict[str, Any]", raw_action)
                result = await self._execute_one(
                    action,
                    ensure_screenshot=index == len(action_list) - 1,
                )
                if result.isError:
                    return result
            return result
        return await self._execute_one(arguments, ensure_screenshot=True)

    async def _execute_one(
        self,
        arguments: dict[str, Any],
        *,
        ensure_screenshot: bool,
    ) -> MCPToolResult:
        action_type = arguments.get("type")
        if not isinstance(action_type, str):
            return tool_err("type is required")

        if action_type == "response":
            text = arguments.get("text")
            if not isinstance(text, str):
                return tool_err("text is required for response")
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=text)],
            )

        try:
            await self._dispatch(action_type, arguments)
        except Exception as exc:
            logger.exception("OpenAIComputerTool action %s failed", action_type)
            return tool_err(f"computer action {action_type!r} failed: {exc}")

        needs_screenshot = (
            ensure_screenshot and action_type in _SCREENSHOT_ACTIONS and action_type != "screenshot"
        )
        if action_type == "screenshot" or needs_screenshot:
            return await self.screenshot()
        return MCPToolResult(content=[], isError=False)

    async def _dispatch(self, action_type: str, args: dict[str, Any]) -> None:
        if action_type == "screenshot":
            return

        if action_type == "click":
            button_raw = args.get("button")
            if button_raw == "wheel":
                button = "middle"
            elif isinstance(button_raw, str):
                button = button_raw  # type: ignore[assignment]
            else:
                button = "left"
            hold = _hold_keys(args.get("keys"))
            await self.click(
                args.get("x"),
                args.get("y"),
                button=button,  # type: ignore[arg-type]
                hold_keys=hold,
            )

        elif action_type == "double_click":
            hold = _hold_keys(args.get("keys"))
            await self.click(
                args.get("x"),
                args.get("y"),
                count=2,
                interval_ms=100,
                hold_keys=hold,
            )

        elif action_type == "scroll":
            hold = _hold_keys(args.get("keys"))
            sx = int(args.get("scroll_x") or 0)
            sy = int(args.get("scroll_y") or 0)
            await self.scroll(
                args.get("x"),
                args.get("y"),
                scroll_x=sx,
                scroll_y=sy,
                hold_keys=hold,
            )

        elif action_type == "type":
            text = args.get("text")
            if isinstance(text, str):
                await self.type_text(text)

        elif action_type == "wait":
            ms = int(args.get("ms") or 1000)
            await self.wait(ms)

        elif action_type == "move":
            x, y = args.get("x"), args.get("y")
            if x is not None and y is not None:
                await self.move(int(x), int(y))

        elif action_type == "keypress":
            keys = args.get("keys")
            if isinstance(keys, list):
                mapped = [_map_key(str(k)) for k in cast("list[Any]", keys)]
                await self.press_keys(mapped)

        elif action_type == "drag":
            path_raw = args.get("path") or []
            if not isinstance(path_raw, list) or len(path_raw) < 2:
                raise ValueError("drag requires a path with at least 2 points")
            path = [
                (int(p.get("x", 0)), int(p.get("y", 0)))
                for p in cast("list[dict[str, Any]]", path_raw)
            ]
            hold = _hold_keys(args.get("keys"))
            await self.drag(path, hold_keys=hold)

        elif action_type == "custom":
            raise ValueError(f"Custom action not supported: {args.get('action')}")

        else:
            raise ValueError(f"Invalid action type: {action_type}")


def _map_key(key: str) -> str:
    return OPENAI_KEY_ALIASES.get(key.lower(), key)


def _hold_keys(keys: Any) -> list[str] | None:
    if not isinstance(keys, list):
        return None
    return [_map_key(str(key)) for key in cast("list[Any]", keys)]


__all__ = ["OPENAI_COMPUTER_SPEC", "OpenAIComputerTool"]

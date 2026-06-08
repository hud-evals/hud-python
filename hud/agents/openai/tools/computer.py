"""OpenAI computer tool — backed by RFBClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err
from hud.types import MCPToolResult

from .base import OpenAIToolSpec

if TYPE_CHECKING:
    from hud.agents.tools.rfb import Button

logger = logging.getLogger(__name__)

OPENAI_COMPUTER_SPEC = OpenAIToolSpec(
    api_type="computer",
    api_name="computer",
)
_OPENAI_BUTTONS: dict[str, Button] = {"wheel": "middle", "middle": "middle", "right": "right"}
_OPENAI_KEY_ALIASES: dict[str, str] = {
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

        error_text: str | None = None
        try:
            await self._dispatch(action_type, arguments)
        except Exception as exc:
            logger.exception("OpenAIComputerTool action %s failed", action_type)
            error_text = f"computer action {action_type!r} failed: {exc}"

        # The Responses API answers every computer_call with a computer_call_output
        # screenshot, so for the final action of the call always return the resulting
        # screen (even on error) rather than empty content that would drop the turn.
        if ensure_screenshot:
            shot = await self.screenshot()
            if error_text is None:
                return shot
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=error_text), *shot.content],
                isError=True,
            )
        if error_text is not None:
            return tool_err(error_text)
        return MCPToolResult(content=[], isError=False)

    async def _dispatch(self, action_type: str, args: dict[str, Any]) -> None:
        match action_type:
            case "screenshot":
                return
            case "click":
                await self.click(
                    args.get("x"),
                    args.get("y"),
                    button=_OPENAI_BUTTONS.get(str(args.get("button")), "left"),
                    hold_keys=_hold_keys(args.get("keys")),
                )
            case "double_click":
                await self.click(
                    args.get("x"),
                    args.get("y"),
                    count=2,
                    interval_ms=100,
                    hold_keys=_hold_keys(args.get("keys")),
                )
            case "scroll":
                await self.scroll(
                    args.get("x"),
                    args.get("y"),
                    scroll_x=int(args.get("scroll_x") or 0),
                    scroll_y=int(args.get("scroll_y") or 0),
                    hold_keys=_hold_keys(args.get("keys")),
                )
            case "type":
                text = args.get("text")
                if isinstance(text, str):
                    await self.type_text(text)
            case "wait":
                await self.wait(int(args.get("ms") or 1000))
            case "move":
                x, y = args.get("x"), args.get("y")
                if x is not None and y is not None:
                    await self.move(int(x), int(y))
            case "keypress":
                mapped = _map_keys(args.get("keys"))
                if mapped:
                    await self.press_keys(mapped)
            case "drag":
                path_raw = args.get("path")
                if not isinstance(path_raw, list) or len(cast("list[Any]", path_raw)) < 2:
                    raise ValueError("drag requires a path with at least 2 points")
                path = [
                    (int(p.get("x", 0)), int(p.get("y", 0)))
                    for p in cast("list[dict[str, Any]]", path_raw)
                ]
                await self.drag(path, hold_keys=_hold_keys(args.get("keys")))
            case "custom":
                raise ValueError(f"Custom action not supported: {args.get('action')}")
            case _:
                raise ValueError(f"Invalid action type: {action_type}")


def _map_keys(keys: Any) -> list[str]:
    if not isinstance(keys, list):
        return []
    return [_OPENAI_KEY_ALIASES.get(str(key).lower(), str(key)) for key in cast("list[Any]", keys)]


def _hold_keys(keys: Any) -> list[str] | None:
    return _map_keys(keys) or None


__all__ = ["OPENAI_COMPUTER_SPEC", "OpenAIComputerTool"]

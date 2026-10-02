"""OpenAI computer tool — backed by RFBClient."""

from __future__ import annotations

import logging
from typing import Any, cast

from hud.agents.tools import RFBTool
from hud.agents.tools.base import tool_err
from hud.types import ImageContent, TextContent, ToolResult

from .base import OpenAIToolSpec

logger = logging.getLogger(__name__)

OPENAI_COMPUTER_SPEC = OpenAIToolSpec(
    api_type="computer",
    api_name="computer",
)


def last_image_data(result: ToolResult) -> str | None:
    """Base64 data of the most recent screenshot block in a tool result."""
    image = last_image_content(result)
    return image.data if image is not None else None


def last_image_content(result: ToolResult) -> ImageContent | None:
    """Most recent screenshot block in a tool result."""
    for block in reversed(result.content):
        if isinstance(block, ImageContent):
            return block
    return None


OPENAI_KEY_ALIASES: dict[str, str] = {
    "return": "Return",
    "escape": "Escape",
    "esc": "Escape",
    "arrowup": "Up",
    "up": "Up",
    "arrowdown": "Down",
    "down": "Down",
    "arrowleft": "Left",
    "left": "Left",
    "arrowright": "Right",
    "right": "Right",
    "backspace": "BackSpace",
    "delete": "Delete",
    "del": "Delete",
    "tab": "Tab",
    "space": "space",
    "control": "Control_L",
    "ctrl": "Control_L",
    "alt": "Alt_L",
    "option": "Alt_L",
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
    def default_spec(cls, model: str) -> OpenAIToolSpec:
        del model
        return OPENAI_COMPUTER_SPEC

    def to_params(self) -> Any:
        return {"type": "computer"}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        actions = arguments.get("actions")
        if "actions" in arguments:
            if not isinstance(actions, list):
                return await self._error_result("actions must be a list")
            action_list = cast("list[Any]", actions)
            if not action_list:
                return await self._error_result("actions list is empty")
            result = ToolResult(content=[], isError=False)
            for index, raw_action in enumerate(action_list):
                if not isinstance(raw_action, dict):
                    return await self._error_result("actions must be objects")
                action = cast("dict[str, Any]", raw_action)
                result = await self._execute_one(
                    action,
                    ensure_screenshot=index == len(action_list) - 1,
                )
                if result.isError:
                    return result
            return result
        return await self._execute_one(arguments, ensure_screenshot=True)

    async def _error_result(self, message: str) -> ToolResult:
        result = tool_err(message)
        try:
            result.content.extend((await self.screenshot()).content)
        except TimeoutError as exc:
            exc.add_note(message)
            raise
        except Exception as exc:
            return tool_err(f"{message}\nScreenshot capture also failed: {exc}")
        return result

    async def _execute_one(
        self,
        arguments: dict[str, Any],
        *,
        ensure_screenshot: bool,
    ) -> ToolResult:
        action_type = arguments.get("type")
        if not isinstance(action_type, str):
            return await self._error_result("type is required")

        if action_type == "response":
            text = arguments.get("text")
            if not isinstance(text, str):
                return await self._error_result("text is required for response")
            result = ToolResult(
                content=[TextContent(type="text", text=text)],
            )
            if ensure_screenshot:
                result.content.extend((await self.screenshot()).content)
            return result

        try:
            await self._dispatch(action_type, arguments)
        except TimeoutError:
            raise
        except Exception as exc:
            logger.exception("OpenAIComputerTool action %s failed", action_type)
            return await self._error_result(f"computer action {action_type!r} failed: {exc}")

        if action_type == "screenshot" or ensure_screenshot:
            return await self.screenshot()
        return ToolResult(content=[], isError=False)

    async def _dispatch(self, action_type: str, args: dict[str, Any]) -> None:
        if action_type == "screenshot":
            return

        if action_type == "click":
            button = args.get("button", "left")
            if button is None:
                button = "left"
            elif button == "wheel":
                button = "middle"
            if button not in ("left", "middle", "right"):
                raise ValueError(f"Unsupported mouse button: {button!r}")
            hold = _hold_keys(args.get("keys"))
            await self.click(
                args.get("x"),
                args.get("y"),
                button=button,
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
            if not isinstance(text, str):
                raise ValueError("text is required for type")
            await self.type_text(text)

        elif action_type == "wait":
            duration = args.get("ms")
            ms = 1000 if duration is None else int(duration)
            if ms < 0:
                raise ValueError("wait duration must be non-negative")
            await self.wait(ms)

        elif action_type == "move":
            x, y = args.get("x"), args.get("y")
            if x is None or y is None:
                raise ValueError("x and y are required for move")
            await self.move(int(x), int(y))

        elif action_type == "keypress":
            keys = _hold_keys(args.get("keys"))
            if not keys:
                raise ValueError("keypress requires at least one key")
            await self.press_keys(keys)

        elif action_type == "drag":
            path_raw = args.get("path")
            if not isinstance(path_raw, list):
                raise ValueError("drag requires a path with at least 2 points")
            points = cast("list[dict[str, Any]]", path_raw)
            if len(points) < 2:
                raise ValueError("drag requires a path with at least 2 points")
            path = [(int(p["x"]), int(p["y"])) for p in points]
            hold = _hold_keys(args.get("keys"))
            await self.drag(path, hold_keys=hold)

        elif action_type == "custom":
            raise ValueError(f"Custom action not supported: {args.get('action')}")

        else:
            raise ValueError(f"Invalid action type: {action_type}")


def _map_key(key: str) -> str:
    return OPENAI_KEY_ALIASES.get(key.lower(), key)


def _hold_keys(keys: Any) -> list[str] | None:
    if keys is None:
        return None
    if not isinstance(keys, list) or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("keys must be a list of non-empty strings")
    return [_map_key(key) for key in cast("list[str]", keys)]


__all__ = ["OPENAI_COMPUTER_SPEC", "OpenAIComputerTool"]

"""Agent-side OpenAI native computer tool backed by an environment computer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mcp.types import ImageContent, TextContent

from hud.types import MCPToolResult

from .base import CallTool, OpenAITool, OpenAIToolSpec, call_tool

if TYPE_CHECKING:
    from openai.types.responses import ComputerToolParam
else:
    ComputerToolParam = Any

OPENAI_COMPUTER_SPEC = OpenAIToolSpec(
    api_type="computer",
    api_name="computer",
    supported_models=(
        "gpt-5.4",
        "gpt-5.4-*",
        "gpt-5.5",
        "gpt-5.5-*",
    ),
)

OPENAI_KEY_ALIASES = {
    "return": "enter",
    "escape": "escape",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "backspace": "backspace",
    "delete": "delete",
    "tab": "tab",
    "space": "space",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "meta": "win",
    "cmd": "cmd",
    "command": "cmd",
    "super": "win",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "home": "home",
    "end": "end",
    "insert": "insert",
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


class OpenAIComputerTool(OpenAITool):
    """Translate OpenAI native computer calls into generic environment calls."""

    name = "computer"
    capability = "computer"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        if OPENAI_COMPUTER_SPEC.supports_model(model):
            return OPENAI_COMPUTER_SPEC
        return None

    def __init__(
        self,
        *,
        env_tool_name: str,
        spec: OpenAIToolSpec,
    ) -> None:
        del spec
        super().__init__(env_tool_name=env_tool_name, spec=OPENAI_COMPUTER_SPEC)

    def to_params(self) -> ComputerToolParam:
        return cast("ComputerToolParam", {"type": "computer"})

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        actions = arguments.get("actions")
        if isinstance(actions, list):
            if not actions:
                return _error_result("actions list is empty")
            result = MCPToolResult(content=[], isError=False)
            for index, action in enumerate(actions):
                if not isinstance(action, dict):
                    return _error_result("actions must be objects")
                result = await self._execute_one(
                    caller,
                    action,
                    ensure_screenshot=index == len(actions) - 1,
                )
                if result.isError:
                    return result
            return result

        return await self._execute_one(caller, arguments, ensure_screenshot=True)

    async def _execute_one(
        self,
        caller: CallTool,
        arguments: dict[str, Any],
        *,
        ensure_screenshot: bool,
    ) -> MCPToolResult:
        action_type = arguments.get("type")
        if not isinstance(action_type, str):
            return _error_result("type is required")

        if action_type == "response":
            text = arguments.get("text")
            if not isinstance(text, str):
                return _error_result("text is required for response")
            return MCPToolResult(content=[TextContent(type="text", text=text)], isError=False)

        env_arguments = self._env_arguments(arguments)
        result = await call_tool(caller, self.env_tool_name, env_arguments)
        if (
            ensure_screenshot
            and action_type in _SCREENSHOT_ACTIONS
            and action_type != "screenshot"
            and not _has_image(result)
            and not result.isError
        ):
            screenshot = await call_tool(caller, self.env_tool_name, {"action": "screenshot"})
            if not screenshot.isError and screenshot.content:
                result = MCPToolResult(
                    content=[*result.content, *screenshot.content],
                    isError=result.isError,
                )
        return result

    def _env_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action_type = arguments.get("type")

        if action_type == "screenshot":
            return {"action": "screenshot"}
        if action_type == "click":
            return {
                "action": "click",
                "x": arguments.get("x"),
                "y": arguments.get("y"),
                "button": _map_button(arguments.get("button")),
                "hold_keys": _hold_keys(arguments.get("keys")),
            }
        if action_type == "double_click":
            return {
                "action": "click",
                "x": arguments.get("x"),
                "y": arguments.get("y"),
                "button": "left",
                "pattern": [100],
                "hold_keys": _hold_keys(arguments.get("keys")),
            }
        if action_type == "scroll":
            return {
                "action": "scroll",
                "x": arguments.get("x"),
                "y": arguments.get("y"),
                "scroll_x": arguments.get("scroll_x") or 0,
                "scroll_y": arguments.get("scroll_y") or 0,
                "hold_keys": _hold_keys(arguments.get("keys")),
            }
        if action_type == "type":
            return {
                "action": "write",
                "text": arguments.get("text"),
                "enter_after": False,
            }
        if action_type == "wait":
            return {"action": "wait", "time": arguments.get("ms") or 1000}
        if action_type == "move":
            return {"action": "move", "x": arguments.get("x"), "y": arguments.get("y")}
        if action_type == "keypress":
            keys = arguments.get("keys")
            if not isinstance(keys, list):
                keys = []
            return {"action": "press", "keys": [_map_key(str(key)) for key in keys]}
        if action_type == "drag":
            return {
                "action": "drag",
                "path": arguments.get("path") or [],
                "hold_keys": _hold_keys(arguments.get("keys")),
            }
        if action_type == "custom":
            custom = arguments.get("action")
            raise ValueError(f"Custom action not supported: {custom}")
        raise ValueError(f"Invalid action type: {action_type}")


def _map_key(key: str) -> str:
    return OPENAI_KEY_ALIASES.get(key.lower(), key.lower())


def _hold_keys(keys: Any) -> list[str] | None:
    if not isinstance(keys, list):
        return None
    return [_map_key(str(key)) for key in keys]


def _map_button(button: Any) -> str:
    if button == "wheel":
        return "middle"
    return button if isinstance(button, str) else "left"


def _has_image(result: MCPToolResult) -> bool:
    return any(isinstance(block, ImageContent) for block in result.content)


def _error_result(message: str) -> MCPToolResult:
    return MCPToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


__all__ = ["OPENAI_COMPUTER_SPEC", "OpenAIComputerTool"]

"""Agent-side OpenAI native computer tool backed by an environment computer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mcp.types import TextContent
from openai.types.responses.response_input_param import ComputerCallOutput

from hud.agents.tools.computer import (
    computer_error_result,
    execute_computer_calls,
    last_image_data,
)
from hud.types import MCPToolCall, MCPToolResult

from .base import OpenAITool, OpenAIToolSpec

if TYPE_CHECKING:
    from openai.types.responses import (
        ComputerToolParam,
        ResponseComputerToolCallOutputScreenshotParam,
        ResponseInputItemParam,
    )
    from openai.types.responses.response_input_param import (
        ComputerCallOutputAcknowledgedSafetyCheck,
    )

    from hud.agents.tools.base import CallTool
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

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> ResponseInputItemParam:
        screenshot = last_image_data(result)
        if not screenshot:
            raise ValueError(
                "Computer tool result missing screenshot. "
                "The tool must always return a screenshot for computer_call_output."
            )

        output = ComputerCallOutput(
            type="computer_call_output",
            call_id=call.id,
            output=cast(
                "ResponseComputerToolCallOutputScreenshotParam",
                {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot}",
                    "detail": "original",
                },
            ),
        )

        checks = (call.model_extra or {}).get("pending_safety_checks")
        if isinstance(checks, list):
            acknowledged: list[ComputerCallOutputAcknowledgedSafetyCheck] = []
            for raw_check in cast("list[Any]", checks):
                check: Any = raw_check
                if hasattr(check, "model_dump"):
                    acknowledged.append(
                        cast("ComputerCallOutputAcknowledgedSafetyCheck", check.model_dump())
                    )
                elif isinstance(check, dict):
                    acknowledged.append(cast("ComputerCallOutputAcknowledgedSafetyCheck", check))
            if acknowledged:
                output["acknowledged_safety_checks"] = acknowledged
        return cast("ResponseInputItemParam", output)

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        actions = arguments.get("actions")
        if isinstance(actions, list):
            action_list = cast("list[Any]", actions)
            if not action_list:
                return computer_error_result("actions list is empty")
            result = MCPToolResult(content=[], isError=False)
            for index, raw_action in enumerate(action_list):
                action = cast("dict[str, Any]", raw_action)
                if not isinstance(raw_action, dict):
                    return computer_error_result("actions must be objects")
                result = await self._execute_one(
                    call_tool,
                    action,
                    ensure_screenshot=index == len(action_list) - 1,
                )
                if result.isError:
                    return result
            return result

        return await self._execute_one(call_tool, arguments, ensure_screenshot=True)

    async def _execute_one(
        self,
        call_tool: CallTool,
        arguments: dict[str, Any],
        *,
        ensure_screenshot: bool,
    ) -> MCPToolResult:
        action_type = arguments.get("type")
        if not isinstance(action_type, str):
            return computer_error_result("type is required")

        if action_type == "response":
            text = arguments.get("text")
            if not isinstance(text, str):
                return computer_error_result("text is required for response")
            return MCPToolResult(content=[TextContent(type="text", text=text)], isError=False)

        env_arguments = self._env_arguments(arguments)
        return await execute_computer_calls(
            call_tool,
            env_tool_name=self.env_tool_name,
            calls=[env_arguments],
            ensure_screenshot=(
                ensure_screenshot
                and action_type in _SCREENSHOT_ACTIONS
                and action_type != "screenshot"
            ),
        )

    def _env_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action_type = arguments.get("type")

        if action_type == "screenshot":
            return {"action": "screenshot"}
        if action_type == "click":
            button = arguments.get("button")
            if button == "wheel":
                button_name = "middle"
            elif isinstance(button, str):
                button_name = button
            else:
                button_name = "left"
            return {
                "action": "click",
                "x": arguments.get("x"),
                "y": arguments.get("y"),
                "button": button_name,
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
            return {
                "action": "press",
                "keys": [_map_key(str(key)) for key in cast("list[Any]", keys)],
            }
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
    return [_map_key(str(key)) for key in cast("list[Any]", keys)]

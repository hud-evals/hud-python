"""Agent-side Gemini Computer Use tool."""

from __future__ import annotations

import base64
import platform
from typing import TYPE_CHECKING, Any, cast

from google.genai import types as genai_types
from mcp.types import ImageContent, TextContent

from hud.agents.tools import AgentTool
from hud.agents.tools.computer import computer_error_result, execute_computer_calls
from hud.types import MCPToolCall, MCPToolResult

from .base import GeminiToolSpec

if TYPE_CHECKING:
    from hud.agents.tools.base import CallTool

SUPPORTED_GEMINI_COMPUTER_USE_MODELS = (
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-flash-preview",
)

GEMINI_COORDINATE_SPACE = 1000
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
GEMINI_URL_PREFIX = "__URL__:"
GEMINI_SAFETY_BLOCKED_PREFIX = "__GEMINI_SAFETY_BLOCKED__:"

GEMINI_COMPUTER_SPEC = GeminiToolSpec(
    api_type="computer_use",
    api_name="gemini_computer",
    supported_models=SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
)


class GeminiComputerTool(AgentTool[genai_types.Tool, genai_types.Content]):
    """Translate Gemini Computer Use calls into generic environment computer calls."""

    name = "computer_use"
    capability = "computer"

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec | None:
        if GEMINI_COMPUTER_SPEC.supports_model(model):
            return GEMINI_COMPUTER_SPEC
        return None

    def __init__(
        self,
        *,
        env_tool_name: str,
        spec: GeminiToolSpec,
        excluded_predefined_functions: list[str] | None = None,
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.excluded_predefined_functions = excluded_predefined_functions or []

    def with_excluded_predefined_functions(
        self, excluded_predefined_functions: list[str]
    ) -> GeminiComputerTool:
        return GeminiComputerTool(
            env_tool_name=self.env_tool_name,
            spec=self.spec,
            excluded_predefined_functions=excluded_predefined_functions,
        )

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            computer_use=genai_types.ComputerUse(
                environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                excluded_predefined_functions=self.excluded_predefined_functions,
            )
        )

    def tool_call(self, function_name: str, raw_args: dict[str, Any]) -> MCPToolCall:
        return MCPToolCall(
            name=self.name,
            arguments={"action": function_name, **raw_args},
            provider_name=function_name,
        )

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> genai_types.Content:
        text = next(
            (
                content.text
                for content in result.content
                if isinstance(content, TextContent)
                and not content.text.startswith(GEMINI_URL_PREFIX)
            ),
            None,
        )
        response: dict[str, Any] = (
            {"error": text or "Tool execution failed"} if result.isError else {"success": True}
        )
        if text is not None and not result.isError:
            response["output"] = text

        url = None
        parts: list[genai_types.FunctionResponsePart] = []
        for content in result.content:
            match content:
                case ImageContent(data=data, mimeType=mime_type):
                    parts.append(
                        genai_types.FunctionResponsePart(
                            inline_data=genai_types.FunctionResponseBlob(
                                mime_type=mime_type or "image/png",
                                data=base64.b64decode(data),
                            )
                        )
                    )
                case TextContent(text=text) if text.startswith(GEMINI_URL_PREFIX):
                    url = text.removeprefix(GEMINI_URL_PREFIX)
                case TextContent(text=text) if text.startswith(GEMINI_SAFETY_BLOCKED_PREFIX):
                    response.pop("success", None)
                    response["blocked"] = True
                    response["reason"] = text.removeprefix(GEMINI_SAFETY_BLOCKED_PREFIX)
                case _:
                    continue

        response["url"] = url or "about:blank"
        safety_decision = call.arguments.get("safety_decision") if call.arguments else None
        if safety_decision and not result.isError and not response.get("blocked"):
            response["safety_acknowledgement"] = True

        return genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=call.provider_name or call.name,
                        response=response,
                        parts=parts or None,
                    )
                )
            ],
        )

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return computer_error_result("action is required")
        safety_decision = arguments.get("safety_decision")
        if (
            isinstance(safety_decision, dict)
            and cast("dict[str, Any]", safety_decision).get("decision") == "require_confirmation"
        ):
            return MCPToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"{GEMINI_SAFETY_BLOCKED_PREFIX}"
                            "Gemini Computer Use action requires user confirmation before "
                            "execution."
                        ),
                    )
                ],
                isError=False,
            )

        return await execute_computer_calls(
            call_tool,
            env_tool_name=self.env_tool_name,
            calls=self._computer_actions(action, arguments),
            ensure_screenshot=action != "open_web_browser",
        )

    def _computer_actions(self, action: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        if action == "open_web_browser":
            return [{"action": "screenshot"}]
        if action == "click_at":
            return [{"action": "click", "x": arguments.get("x"), "y": arguments.get("y")}]
        if action == "hover_at":
            return [{"action": "move", "x": arguments.get("x"), "y": arguments.get("y")}]
        if action == "type_text_at":
            calls: list[dict[str, Any]] = []
            if arguments.get("x") is not None and arguments.get("y") is not None:
                calls.extend(
                    [
                        {"action": "move", "x": arguments.get("x"), "y": arguments.get("y")},
                        {"action": "click", "x": arguments.get("x"), "y": arguments.get("y")},
                    ]
                )
            if arguments.get("clear_before_typing", True):
                calls.extend(
                    [
                        {"action": "press", "keys": ["cmd", "a"] if IS_MAC else ["ctrl", "a"]},
                        {"action": "press", "keys": ["backspace" if IS_MAC else "delete"]},
                    ]
                )
            calls.append(
                {
                    "action": "write",
                    "text": arguments.get("text"),
                    "enter_after": bool(arguments.get("press_enter")),
                }
            )
            return calls
        if action in ("scroll_document", "scroll_at"):
            direction = arguments.get("direction")
            magnitude = arguments.get("magnitude") or 800
            if direction == "down":
                call = {"action": "scroll", "scroll_x": None, "scroll_y": magnitude}
            elif direction == "up":
                call = {"action": "scroll", "scroll_x": None, "scroll_y": -magnitude}
            elif direction == "right":
                call = {"action": "scroll", "scroll_x": magnitude, "scroll_y": None}
            elif direction == "left":
                call = {"action": "scroll", "scroll_x": -magnitude, "scroll_y": None}
            else:
                raise ValueError("direction must be one of up, down, left, right")
            if action == "scroll_at":
                call.update({"x": arguments.get("x"), "y": arguments.get("y")})
            return [call]
        if action == "wait_5_seconds":
            return [{"action": "wait", "time": 5000}]
        if action == "go_back":
            return [{"action": "press", "keys": ["cmd", "["] if IS_MAC else ["alt", "left"]}]
        if action == "go_forward":
            return [{"action": "press", "keys": ["cmd", "]"] if IS_MAC else ["alt", "right"]}]
        if action == "search":
            target = arguments.get("url") or "https://www.google.com"
            return [
                {"action": "press", "keys": ["cmd", "l"] if IS_MAC else ["ctrl", "l"]},
                {"action": "write", "text": target, "enter_after": True},
            ]
        if action == "navigate":
            return [
                {"action": "press", "keys": ["cmd", "l"] if IS_MAC else ["ctrl", "l"]},
                {"action": "write", "text": arguments.get("url"), "enter_after": True},
            ]
        if action == "key_combination":
            keys = arguments.get("keys")
            if not isinstance(keys, str):
                raise ValueError("keys must be a '+'-separated string")
            aliases = {
                "control": "ctrl",
                "cmd": "cmd",
                "command": "cmd",
                "meta": "cmd" if IS_MAC else "ctrl",
                "return": "enter",
            }
            normalized_keys = [
                aliases.get(key, key) for part in keys.split("+") if (key := part.strip().lower())
            ]
            return [{"action": "press", "keys": normalized_keys}]
        if action == "drag_and_drop":
            max_drag_coordinate = max(
                GEMINI_COORDINATE_SPACE - GEMINI_DRAG_INSET,
                GEMINI_DRAG_INSET,
            )

            def drag_coordinate(value: Any) -> Any:
                if not isinstance(value, int | float) or not 0 <= value <= GEMINI_COORDINATE_SPACE:
                    return value
                return min(max(int(value), GEMINI_DRAG_INSET), max_drag_coordinate)

            return [
                {
                    "action": "drag",
                    "path": [
                        {
                            "x": drag_coordinate(arguments.get("x")),
                            "y": drag_coordinate(arguments.get("y")),
                        },
                        {
                            "x": drag_coordinate(arguments.get("destination_x")),
                            "y": drag_coordinate(arguments.get("destination_y")),
                        },
                    ],
                }
            ]
        raise ValueError(f"Unknown Gemini computer action: {action}")

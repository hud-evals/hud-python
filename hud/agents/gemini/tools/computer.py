"""Agent-side Gemini Computer Use tool."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

from google.genai import types as genai_types
from mcp.types import ImageContent, TextContent

from hud.types import MCPToolResult

from .base import CallTool, GeminiTool, GeminiToolSpec, call_tool

if TYPE_CHECKING:
    from hud.agents.tools import EnvironmentCapability

SUPPORTED_GEMINI_COMPUTER_USE_MODELS = (
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-flash-preview",
)

GEMINI_COORDINATE_SPACE = 1000
GEMINI_DRAG_INSET = 25

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


def normalize_gemini_computer_use_args(action: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini Computer Use function-call args to agent-tool args."""
    normalized_args: dict[str, Any] = {"action": action}

    coord = raw_args.get("coordinate") or raw_args.get("coordinates")
    if isinstance(coord, list | tuple) and len(coord) >= 2:
        try:
            normalized_args["x"] = int(coord[0])
            normalized_args["y"] = int(coord[1])
        except (TypeError, ValueError):
            pass

    dest = (
        raw_args.get("destination")
        or raw_args.get("destination_coordinate")
        or raw_args.get("destinationCoordinate")
    )
    if isinstance(dest, list | tuple) and len(dest) >= 2:
        try:
            normalized_args["destination_x"] = int(dest[0])
            normalized_args["destination_y"] = int(dest[1])
        except (TypeError, ValueError):
            pass

    for key in (
        "text",
        "press_enter",
        "clear_before_typing",
        "safety_decision",
        "direction",
        "magnitude",
        "url",
        "keys",
        "x",
        "y",
        "destination_x",
        "destination_y",
    ):
        if key in raw_args:
            normalized_args[key] = raw_args[key]

    return normalized_args


class GeminiComputerTool(GeminiTool):
    """Translate Gemini Computer Use calls into generic environment computer calls."""

    name = "computer_use"
    capability = "computer"

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec | None:
        if GEMINI_COMPUTER_SPEC.supports_model(model):
            return GEMINI_COMPUTER_SPEC
        return None

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        spec: GeminiToolSpec,
        model: str,
    ) -> GeminiComputerTool:
        del model
        return cls(env_tool_name=capability.tool_name, spec=spec)

    def __init__(self, *, env_tool_name: str, spec: GeminiToolSpec) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.excluded_predefined_functions: list[str] = []

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            computer_use=genai_types.ComputerUse(
                environment=genai_types.Environment.ENVIRONMENT_BROWSER,
                excluded_predefined_functions=self.excluded_predefined_functions,
            )
        )

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return _error_result("action is required")
        if _requires_confirmation(arguments.get("safety_decision")):
            return _blocked_result(
                "Gemini Computer Use action requires user confirmation before execution."
            )

        result = MCPToolResult(content=[], isError=False)
        for call in self._env_calls(action, arguments):
            result = await call_tool(caller, self.env_tool_name, call)
            if result.isError:
                return result

        if action != "open_web_browser" and not _has_image(result):
            screenshot = await call_tool(caller, self.env_tool_name, {"action": "screenshot"})
            if not screenshot.isError and screenshot.content:
                result = MCPToolResult(
                    content=[*result.content, *screenshot.content],
                    isError=result.isError,
                )
        return result

    def _env_calls(self, action: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
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
                calls.extend(_clear_text_calls())
            calls.append(
                {
                    "action": "write",
                    "text": arguments.get("text"),
                    "enter_after": bool(arguments.get("press_enter")),
                }
            )
            return calls
        if action in ("scroll_document", "scroll_at"):
            call = _scroll_call(arguments)
            if action == "scroll_at":
                call.update({"x": arguments.get("x"), "y": arguments.get("y")})
            return [call]
        if action == "wait_5_seconds":
            return [{"action": "wait", "time": 5000}]
        if action == "go_back":
            return [{"action": "press", "keys": ["cmd", "["] if _is_mac() else ["alt", "left"]}]
        if action == "go_forward":
            return [{"action": "press", "keys": ["cmd", "]"] if _is_mac() else ["alt", "right"]}]
        if action == "search":
            target = arguments.get("url") or "https://www.google.com"
            return [*_address_bar_calls(), {"action": "write", "text": target, "enter_after": True}]
        if action == "navigate":
            return [
                *_address_bar_calls(),
                {"action": "write", "text": arguments.get("url"), "enter_after": True},
            ]
        if action == "key_combination":
            return [{"action": "press", "keys": _normalize_key_combination(arguments.get("keys"))}]
        if action == "drag_and_drop":
            return [
                {
                    "action": "drag",
                    "path": [
                        {
                            "x": _inset_drag_coordinate(arguments.get("x")),
                            "y": _inset_drag_coordinate(arguments.get("y")),
                        },
                        {
                            "x": _inset_drag_coordinate(arguments.get("destination_x")),
                            "y": _inset_drag_coordinate(arguments.get("destination_y")),
                        },
                    ],
                }
            ]
        raise ValueError(f"Unknown Gemini computer action: {action}")


def _scroll_call(arguments: dict[str, Any]) -> dict[str, Any]:
    direction = arguments.get("direction")
    magnitude = arguments.get("magnitude") or 800
    if direction == "down":
        return {"action": "scroll", "scroll_x": None, "scroll_y": magnitude}
    if direction == "up":
        return {"action": "scroll", "scroll_x": None, "scroll_y": -magnitude}
    if direction == "right":
        return {"action": "scroll", "scroll_x": magnitude, "scroll_y": None}
    if direction == "left":
        return {"action": "scroll", "scroll_x": -magnitude, "scroll_y": None}
    raise ValueError("direction must be one of up, down, left, right")


def _inset_drag_coordinate(value: Any) -> Any:
    """Keep Gemini normalized drag endpoints away from display edges."""
    if not isinstance(value, int | float) or not 0 <= value <= GEMINI_COORDINATE_SPACE:
        return value
    max_value = max(GEMINI_COORDINATE_SPACE - GEMINI_DRAG_INSET, GEMINI_DRAG_INSET)
    return min(max(int(value), GEMINI_DRAG_INSET), max_value)


def _clear_text_calls() -> list[dict[str, Any]]:
    is_mac = _is_mac()
    return [
        {"action": "press", "keys": ["cmd", "a"] if is_mac else ["ctrl", "a"]},
        {"action": "press", "keys": ["backspace" if is_mac else "delete"]},
    ]


def _normalize_key_combination(keys: Any) -> list[str] | Any:
    if isinstance(keys, str):
        return [_normalize_key(key) for key in keys.split("+") if key.strip()]
    if isinstance(keys, list):
        return [_normalize_key(key) if isinstance(key, str) else key for key in keys]
    return keys


def _normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    aliases = {
        "control": "ctrl",
        "cmd": "cmd",
        "command": "cmd",
        "meta": "cmd" if _is_mac() else "ctrl",
        "return": "enter",
    }
    return aliases.get(normalized, normalized)


def _requires_confirmation(safety_decision: Any) -> bool:
    if not isinstance(safety_decision, dict):
        return False
    return safety_decision.get("decision") == "require_confirmation"


def _address_bar_calls() -> list[dict[str, Any]]:
    return [{"action": "press", "keys": ["cmd", "l"] if _is_mac() else ["ctrl", "l"]}]


def _is_mac() -> bool:
    return platform.system().lower() == "darwin"


def _has_image(result: MCPToolResult) -> bool:
    return any(isinstance(block, ImageContent) for block in result.content)


def _error_result(message: str) -> MCPToolResult:
    return MCPToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _blocked_result(message: str) -> MCPToolResult:
    return MCPToolResult(
        content=[TextContent(type="text", text=f"__GEMINI_SAFETY_BLOCKED__:{message}")],
        isError=False,
    )


__all__ = [
    "GEMINI_COMPUTER_SPEC",
    "GEMINI_COORDINATE_SPACE",
    "GEMINI_DRAG_INSET",
    "PREDEFINED_COMPUTER_USE_FUNCTIONS",
    "SUPPORTED_GEMINI_COMPUTER_USE_MODELS",
    "GeminiComputerTool",
    "normalize_gemini_computer_use_args",
]

"""Agent-side Qwen computer tool for OpenAI-compatible chat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from hud.agents.tools import AgentToolSpec
from hud.agents.tools.computer import (
    computer_error_result,
    computer_tool_info,
    execute_computer_calls,
)

from .base import OpenAICompatibleTool
from .settings import openai_compatible_tool_settings

if TYPE_CHECKING:
    from openai.types.shared_params.function_parameters import FunctionParameters

    from hud.agents.tools import EnvironmentCapability
    from hud.agents.tools.base import CallTool
    from hud.types import MCPToolResult

QWEN_COMPUTER_SPEC = AgentToolSpec(
    api_type="computer_use",
    api_name="computer_use",
    supported_models=("qwen*",),
)


class QwenComputerUseToolParam(TypedDict):
    """Qwen's OpenAI-compatible computer_use extension."""

    type: Literal["computer_use"]
    name: str
    display_width_px: int
    display_height_px: int
    description: str
    parameters: FunctionParameters


class QwenComputerTool(OpenAICompatibleTool):
    """Translate Qwen computer_use calls into generic environment computer calls."""

    name = "computer_use"
    capability = "computer"

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        if QWEN_COMPUTER_SPEC.supports_model(model):
            return QWEN_COMPUTER_SPEC
        return None

    def __init__(
        self,
        *,
        env_tool_name: str,
        spec: AgentToolSpec,
        display_width: int,
        display_height: int,
        description: str,
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.display_width = display_width
        self.display_height = display_height
        self.description = description

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        model: str,
    ) -> QwenComputerTool | None:
        spec = cls.default_spec(model)
        if spec is None:
            return None

        computer_info = computer_tool_info(
            capability.tool,
            default_width=openai_compatible_tool_settings.QWEN_COMPUTER_WIDTH,
            default_height=openai_compatible_tool_settings.QWEN_COMPUTER_HEIGHT,
        )
        return cls(
            env_tool_name=capability.tool_name,
            spec=spec,
            display_width=computer_info.display_width,
            display_height=computer_info.display_height,
            description=_qwen_description(
                computer_info.display_width, computer_info.display_height
            ),
        )

    def to_params(self) -> QwenComputerUseToolParam:
        tool: QwenComputerUseToolParam = {
            "type": "computer_use",
            "name": self.name,
            "display_width_px": self.display_width,
            "display_height_px": self.display_height,
            "description": self.description,
            "parameters": QWEN_COMPUTER_PARAMETERS,
        }
        return tool

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return computer_error_result("action is required")
        if action == "terminate":
            return computer_error_result("terminate action is not supported for computer control.")
        if action == "answer":
            return computer_error_result("answer action is not supported for computer control.")

        return await execute_computer_calls(
            call_tool,
            env_tool_name=self.env_tool_name,
            calls=self._env_calls(action, arguments),
            ensure_screenshot=action not in {"screenshot", "wait"},
        )

    def _env_calls(self, action: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        coordinate = _parse_qwen_coordinate(arguments.get("coordinate"))
        if action == "screenshot":
            return [{"action": "screenshot"}]
        if action in {"left_click", "right_click", "middle_click"}:
            x, y = _required_coordinate(coordinate, action)
            button = {"left_click": "left", "right_click": "right", "middle_click": "middle"}[
                action
            ]
            return [{"action": "click", "x": x, "y": y, "button": button}]
        if action == "double_click":
            x, y = _required_coordinate(coordinate, action)
            return [{"action": "click", "x": x, "y": y, "pattern": [100]}]
        if action == "triple_click":
            x, y = _required_coordinate(coordinate, action)
            return [{"action": "click", "x": x, "y": y, "pattern": [100, 100]}]
        if action == "mouse_move":
            x, y = _required_coordinate(coordinate, action)
            return [{"action": "move", "x": x, "y": y}]
        if action == "type":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("text is required for type")
            return [{"action": "write", "text": text}]
        if action == "key":
            keys = arguments.get("keys")
            if not isinstance(keys, list):
                raise ValueError("keys is required for key")
            return [{"action": "press", "keys": keys}]
        if action in {"scroll", "hscroll"}:
            pixels = arguments.get("pixels")
            if not isinstance(pixels, int | float):
                raise ValueError("pixels is required for scroll")
            call: dict[str, Any] = {"action": "scroll"}
            if coordinate is not None:
                call.update({"x": coordinate[0], "y": coordinate[1]})
            if action == "scroll":
                call["scroll_y"] = -int(pixels)
            else:
                call["scroll_x"] = int(pixels)
            return [call]
        if action == "left_click_drag":
            x, y = _required_coordinate(coordinate, action)
            return [
                {"action": "mouse_down", "button": "left"},
                {"action": "move", "x": x, "y": y},
                {"action": "mouse_up", "button": "left"},
            ]
        if action == "wait":
            time = arguments.get("time")
            if not isinstance(time, int | float):
                raise ValueError("time is required for wait")
            if time < 0:
                raise ValueError("time must be non-negative")
            return [{"action": "wait", "time": int(time * 1000)}]
        raise ValueError(f"Invalid action: {action}")


QWEN_COMPUTER_PARAMETERS: FunctionParameters = {
    "properties": {
        "action": {
            "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs
key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a vertical scroll.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
""".strip(),
            "enum": [
                "key",
                "type",
                "mouse_move",
                "left_click",
                "left_click_drag",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "scroll",
                "hscroll",
                "wait",
            ],
            "type": "string",
        },
        "keys": {"description": "Required only by `action=key`.", "type": "array"},
        "text": {
            "description": "Required only by `action=type`.",
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y) pixel coordinate to interact with.",
            "type": "array",
        },
        "pixels": {
            "description": "Scroll amount. Positive vertical values scroll up.",
            "type": "number",
        },
        "time": {
            "description": "Seconds to wait. Required only by `action=wait`.",
            "type": "number",
        },
    },
    "required": ["action"],
    "type": "object",
}


def _qwen_description(width: int, height: int) -> str:
    return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or
applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to
wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Whenever you intend to move the cursor to click on an element like an icon, you
should consult a screenshot to determine the coordinates of the element before
moving the cursor.
* Make sure to click buttons, links, and icons with the cursor tip in the center.
""".strip()


def _parse_qwen_coordinate(coordinate: Any) -> tuple[int, int] | None:
    if not isinstance(coordinate, list | tuple):
        return None
    coord = cast("list[Any] | tuple[Any, ...]", coordinate)
    if len(coord) < 2:
        return None
    try:
        return int(coord[0]), int(coord[1])
    except (TypeError, ValueError):
        return None


def _required_coordinate(coordinate: tuple[int, int] | None, action: str) -> tuple[int, int]:
    if coordinate is None:
        raise ValueError(f"coordinate is required for {action}")
    return coordinate

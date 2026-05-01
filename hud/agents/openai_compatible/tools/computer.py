"""Agent-side OpenAI-compatible computer tools."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

from mcp.types import ImageContent, TextContent

from hud.agents.tools import AgentTool, AgentToolSpec, CallTool, call_tool
from hud.tools.computer import computer_settings
from hud.types import MCPToolResult

from .types import OpenAICompatibleToolParam, QwenComputerUseToolParam

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam
    from openai.types.shared_params.function_parameters import FunctionParameters

    from hud.agents.tools import EnvironmentCapability

logger = logging.getLogger(__name__)

GLM_COORDINATE_SPACE = 999

GLMAction = Literal[
    "left_click",
    "click",
    "right_click",
    "middle_click",
    "hover",
    "left_double_click",
    "left_drag",
    "key",
    "type",
    "scroll",
    "screenshot",
    "WAIT",
    "DONE",
    "FAIL",
]

VALID_GLM_ACTIONS: set[str] = set(get_args(GLMAction))

GLM_COMPUTER_SPEC = AgentToolSpec(
    api_type="function",
    api_name="computer",
    supported_models=("glm-*",),
)

QWEN_COMPUTER_SPEC = AgentToolSpec(
    api_type="computer_use",
    api_name="computer_use",
    supported_models=("qwen*",),
)

GLM_SYSTEM_INSTRUCTIONS = (
    "You are a GUI Agent. Your task is to respond accurately to user requests by using "
    "tools or performing GUI operations until the task is fulfilled. Coordinates are in "
    "thousandths (0-999). Complete tasks autonomously without asking for confirmation. "
    "If a task cannot be completed, use FAIL()."
)

GLM_COMPUTER_DESCRIPTION = """\
Use this tool to interact with the computer via GLM's PC action space.
* Coordinates use a 0-999 normalized scale (thousandths of screen dimensions).
* Always use valid JSON for function arguments. Do NOT use XML tags.
  Correct: {"action": "left_click", "start_box": "[500, 300]"}
  Wrong: {"action": "left_click<arg_key>start_box</arg_key>..."}
* Available actions:
  - left_click/right_click/middle_click(start_box='[x,y]')
  - hover(start_box='[x,y]'), left_double_click(start_box='[x,y]')
  - left_drag(start_box='[x,y]', end_box='[x,y]')
  - key(keys='ctrl+c'), type(content='text')
  - scroll(start_box='[x,y]', direction='up|down', step=5)
  - screenshot(), WAIT(), DONE(), FAIL()
* If a task cannot be completed, use FAIL.\
""".strip()

GLM_COMPUTER_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": (
                "REQUIRED. Action to perform: left_click, right_click, middle_click, "
                "hover, left_double_click, left_drag, key, type, scroll, screenshot, "
                "WAIT, DONE, FAIL"
            ),
            "enum": sorted(VALID_GLM_ACTIONS),
        },
        "start_box": {
            "description": (
                "Position as '[x,y]' string or [x,y] array, coordinates 0-999 normalized"
            ),
        },
        "end_box": {
            "description": "End position for drag as '[x,y]' string or [x,y] array",
        },
        "content": {"type": "string", "description": "Text content to type"},
        "keys": {"description": "Key(s) to press, e.g. 'enter', 'ctrl+c', 'alt+tab'"},
        "direction": {"type": "string", "description": "Scroll direction: 'up' or 'down'"},
        "step": {"type": "integer", "description": "Scroll steps", "default": 5},
        "element_info": {"type": "string", "description": "Optional UI element description"},
    },
    "required": ["action"],
}


class GLMComputerTool(AgentTool[OpenAICompatibleToolParam]):
    """Translate GLM native GUI calls into generic environment computer calls."""

    name = "computer"
    capability = "computer"
    ignored_api_types: ClassVar[frozenset[str]] = frozenset({"gui_agent_glm45v"})

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        if GLM_COMPUTER_SPEC.supports_model(model):
            return GLM_COMPUTER_SPEC
        return None

    def __init__(
        self,
        *,
        env_tool_name: str,
        spec: AgentToolSpec,
        display_width: int,
        display_height: int,
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.display_width = display_width
        self.display_height = display_height

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        spec: AgentToolSpec,
        model: str,
    ) -> GLMComputerTool:
        del model
        width, height = _resolution_from_capability(
            capability,
            default_width=computer_settings.GLM_COMPUTER_WIDTH,
            default_height=computer_settings.GLM_COMPUTER_HEIGHT,
        )
        return cls(
            env_tool_name=capability.tool_name,
            spec=spec,
            display_width=width,
            display_height=height,
        )

    def to_params(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"{GLM_COMPUTER_DESCRIPTION}\n* The screen's resolution is "
                    f"{self.display_width}x{self.display_height}."
                ),
                "parameters": GLM_COMPUTER_PARAMETERS,
            },
        }

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        arguments = _fix_glm_xml_args(arguments)
        action = arguments.get("action")
        if not isinstance(action, str):
            return _error_result("'action' is required")
        if action == "DONE":
            return _error_result("DONE action is not supported for computer control.")
        if action == "FAIL":
            return _error_result("FAIL action is not supported for computer control.")

        result = MCPToolResult(content=[], isError=False)
        for call in self._env_calls(action, arguments):
            result = await call_tool(caller, self.env_tool_name, call)
            if result.isError:
                return result

        if action not in {"screenshot", "WAIT"} and not _has_image(result):
            screenshot = await call_tool(caller, self.env_tool_name, {"action": "screenshot"})
            if not screenshot.isError and screenshot.content:
                result = MCPToolResult(
                    content=[*result.content, *screenshot.content],
                    isError=result.isError,
                )
        return result

    def _env_calls(self, action: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        start = _parse_glm_box(arguments.get("start_box"))
        end = _parse_glm_box(arguments.get("end_box"))

        if action == "screenshot":
            return [{"action": "screenshot"}]
        if action == "WAIT":
            return [{"action": "wait", "time": 5000}]
        if action in ("left_click", "click", "right_click", "middle_click"):
            x, y = self._point(start, f"start_box required for {action}")
            button = {
                "left_click": "left",
                "click": "left",
                "right_click": "right",
                "middle_click": "middle",
            }[action]
            return [{"action": "click", "x": x, "y": y, "button": button}]
        if action == "hover":
            x, y = self._point(start, "start_box required for hover")
            return [{"action": "move", "x": x, "y": y}]
        if action == "left_double_click":
            x, y = self._point(start, "start_box required for left_double_click")
            return [{"action": "click", "x": x, "y": y, "button": "left", "pattern": [100]}]
        if action == "left_drag":
            start_x, start_y = self._point(start, "start_box required for left_drag")
            end_x, end_y = self._point(end, "end_box required for left_drag")
            return [
                {
                    "action": "drag",
                    "path": [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}],
                }
            ]
        if action == "key":
            keys = _parse_glm_keys(arguments.get("keys"))
            if not keys:
                raise ValueError("keys required for key action")
            return [{"action": "press", "keys": keys}]
        if action == "type":
            content = arguments.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError("content required for type")
            return [{"action": "write", "text": content, "enter_after": False}]
        if action == "scroll":
            direction = arguments.get("direction")
            if direction not in {"up", "down"}:
                raise ValueError("direction must be 'up' or 'down'")
            point = start or (GLM_COORDINATE_SPACE // 2, GLM_COORDINATE_SPACE // 2)
            x, y = self._scale_normalized_point(point)
            step = arguments.get("step") or 5
            scroll_y = int(step) * 100 if direction == "down" else -int(step) * 100
            return [{"action": "scroll", "x": x, "y": y, "scroll_y": scroll_y}]
        raise ValueError(f"Unknown action: {action}")

    def _point(self, point: tuple[int, int] | None, message: str) -> tuple[int, int]:
        if point is None:
            raise ValueError(message)
        return self._scale_normalized_point(point)

    def _scale_normalized_point(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        scaled_x = round(x / GLM_COORDINATE_SPACE * (self.display_width - 1))
        scaled_y = round(y / GLM_COORDINATE_SPACE * (self.display_height - 1))
        return scaled_x, scaled_y


class QwenComputerTool(AgentTool[OpenAICompatibleToolParam]):
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
        spec: AgentToolSpec,
        model: str,
    ) -> QwenComputerTool:
        del model
        width, height = _resolution_from_capability(
            capability,
            default_width=computer_settings.QWEN_COMPUTER_WIDTH,
            default_height=computer_settings.QWEN_COMPUTER_HEIGHT,
        )
        return cls(
            env_tool_name=capability.tool_name,
            spec=spec,
            display_width=width,
            display_height=height,
            description=_qwen_description(width, height),
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

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return _error_result("action is required")
        if action == "terminate":
            return _error_result("terminate action is not supported for computer control.")
        if action == "answer":
            return _error_result("answer action is not supported for computer control.")

        result = MCPToolResult(content=[], isError=False)
        for call in self._env_calls(action, arguments):
            result = await call_tool(caller, self.env_tool_name, call)
            if result.isError:
                return result

        if action not in {"screenshot", "wait"} and not _has_image(result):
            screenshot = await call_tool(caller, self.env_tool_name, {"action": "screenshot"})
            if not screenshot.isError and screenshot.content:
                result = MCPToolResult(
                    content=[*result.content, *screenshot.content],
                    isError=result.isError,
                )
        return result

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
            return [{"action": "drag", "path": [{"x": x, "y": y}]}]
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
* `terminate`: Terminate the current task and report its completion status (not supported).
* `answer`: Answer a question (not supported).
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
                "terminate",
                "answer",
            ],
            "type": "string",
        },
        "keys": {"description": "Required only by `action=key`.", "type": "array"},
        "text": {
            "description": "Required only by `action=type` and `action=answer`.",
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
        "status": {
            "description": "The status of the task. Required only by `action=terminate`.",
            "type": "string",
            "enum": ["success", "failure"],
        },
    },
    "required": ["action"],
    "type": "object",
}


def _resolution_from_capability(
    capability: EnvironmentCapability,
    *,
    default_width: int,
    default_height: int,
) -> tuple[int, int]:
    metadata_resolution = capability.metadata.get("resolution", {})
    if not isinstance(metadata_resolution, dict):
        metadata_resolution = {}
    tool_resolution = (capability.tool.meta or {}).get("resolution", {})
    if not isinstance(tool_resolution, dict):
        tool_resolution = {}
    width = int(metadata_resolution.get("width") or tool_resolution.get("width") or default_width)
    height = int(
        metadata_resolution.get("height") or tool_resolution.get("height") or default_height
    )
    return width, height


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


def _parse_glm_box(box: Any) -> tuple[int, int] | None:
    if box is None:
        return None
    if isinstance(box, str):
        match = re.match(r"\[?\s*(\d+)\s*,\s*(\d+)\s*\]?", box.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
    if isinstance(box, list):
        if len(box) == 1 and isinstance(box[0], list):
            box = box[0]
        if len(box) >= 2:
            try:
                return int(box[0]), int(box[1])
            except (TypeError, ValueError):
                return None
    return None


def _parse_glm_keys(keys: Any) -> list[str]:
    if not keys:
        return []
    if isinstance(keys, list):
        return [str(key).strip().lower() for key in keys]
    return [key.strip().lower() for key in str(keys).split("+") if key.strip()]


def _fix_glm_xml_args(args: dict[str, Any]) -> dict[str, Any]:
    fixed: dict[str, Any] = {}
    for key, value in args.items():
        if not isinstance(value, str) or not re.search(r"</?arg_", value):
            fixed[key] = value
            continue

        main_value = re.split(r"</?arg_", value, maxsplit=1)[0].strip()
        if main_value:
            fixed[key] = main_value

        matches = re.findall(r"<arg_key>(\w+)</arg_key>\s*<arg_value>([^\"<]+)", value)
        for arg_name, arg_val in matches:
            if arg_name and arg_val:
                fixed[arg_name.strip()] = arg_val.strip()

        if not main_value and not matches:
            fixed[key] = value
        logger.warning("Fixed GLM XML args: %s -> %s", args, fixed)
    return fixed


def _parse_qwen_coordinate(coordinate: Any) -> tuple[int, int] | None:
    if isinstance(coordinate, list | tuple) and len(coordinate) >= 2:
        try:
            return int(coordinate[0]), int(coordinate[1])
        except (TypeError, ValueError):
            return None
    return None


def _required_coordinate(coordinate: tuple[int, int] | None, action: str) -> tuple[int, int]:
    if coordinate is None:
        raise ValueError(f"coordinate is required for {action}")
    return coordinate


def _has_image(result: MCPToolResult) -> bool:
    return any(isinstance(block, ImageContent) for block in result.content)


def _error_result(message: str) -> MCPToolResult:
    return MCPToolResult(content=[TextContent(type="text", text=message)], isError=True)


__all__ = [
    "GLM_COMPUTER_SPEC",
    "GLM_COORDINATE_SPACE",
    "QWEN_COMPUTER_SPEC",
    "VALID_GLM_ACTIONS",
    "GLMComputerTool",
    "QwenComputerTool",
    "_fix_glm_xml_args",
    "_parse_glm_box",
]

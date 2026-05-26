"""Agent-side GLM computer tool for OpenAI-compatible chat models."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from hud.agents.tools import AgentToolSpec
from hud.agents.tools.computer import (
    computer_error_result,
    computer_tool_info,
    execute_computer_calls,
)

from .base import OpenAICompatibleTool
from .settings import openai_compatible_tool_settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam
    from openai.types.shared_params.function_parameters import FunctionParameters

    from hud.agents.tools import EnvironmentCapability
    from hud.agents.tools.base import CallTool
    from hud.types import MCPToolResult

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
]

VALID_GLM_ACTIONS: set[str] = set(get_args(GLMAction))

GLM_COMPUTER_SPEC = AgentToolSpec(
    api_type="function",
    api_name="computer",
    supported_models=("glm-*",),
)

GLM_SYSTEM_INSTRUCTIONS = (
    "You are a GUI Agent. Your task is to respond accurately to user requests by using "
    "tools or performing GUI operations until the task is fulfilled. Coordinates are in "
    "thousandths (0-999). Complete tasks autonomously without asking for confirmation. "
    "If a task cannot be completed, explain the failure in your final response."
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
  - screenshot(), WAIT()
* If a task cannot be completed, explain the failure in your final response.\
""".strip()

GLM_COMPUTER_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": (
                "REQUIRED. Action to perform: left_click, right_click, middle_click, "
                "hover, left_double_click, left_drag, key, type, scroll, screenshot, "
                "WAIT"
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


class GLMComputerTool(OpenAICompatibleTool):
    """Translate GLM native GUI calls into generic environment computer calls."""

    name = "computer"
    capability = "computer"

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
        coordinate_space: int | None,
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.display_width = display_width
        self.display_height = display_height
        self.coordinate_space = coordinate_space

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        model: str,
    ) -> GLMComputerTool | None:
        spec = cls.default_spec(model)
        if spec is None:
            return None

        computer_info = computer_tool_info(
            capability.tool,
            default_width=openai_compatible_tool_settings.GLM_COMPUTER_WIDTH,
            default_height=openai_compatible_tool_settings.GLM_COMPUTER_HEIGHT,
        )
        return cls(
            env_tool_name=capability.tool_name,
            spec=spec,
            display_width=computer_info.display_width,
            display_height=computer_info.display_height,
            coordinate_space=computer_info.coordinate_space,
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        arguments = _normalize_glm_args(arguments)
        action = arguments.get("action")
        if not isinstance(action, str):
            return computer_error_result("'action' is required")

        return await execute_computer_calls(
            call_tool,
            env_tool_name=self.env_tool_name,
            calls=self._env_calls(action, arguments),
            ensure_screenshot=action not in {"screenshot", "WAIT"},
        )

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
            raw_keys = arguments.get("keys")
            if isinstance(raw_keys, list):
                keys = [str(key).strip().lower() for key in cast("list[Any]", raw_keys)]
            else:
                keys = [
                    key.strip().lower() for key in str(raw_keys or "").split("+") if key.strip()
                ]
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
        if self.coordinate_space == GLM_COORDINATE_SPACE:
            return point
        x, y = point
        scaled_x = round(x / GLM_COORDINATE_SPACE * (self.display_width - 1))
        scaled_y = round(y / GLM_COORDINATE_SPACE * (self.display_height - 1))
        return scaled_x, scaled_y


def _parse_glm_box(box: Any) -> tuple[int, int] | None:
    if box is None:
        return None
    if isinstance(box, str):
        match = re.match(r"\[?\s*(\d+)\s*,\s*(\d+)\s*\]?", box.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
    if isinstance(box, list):
        nested = cast("list[Any]", box)
        if len(nested) == 1 and isinstance(nested[0], list):
            nested = cast("list[Any]", nested[0])
        if len(nested) >= 2:
            try:
                return int(nested[0]), int(nested[1])
            except (TypeError, ValueError):
                return None
    return None


def _normalize_glm_args(args: dict[str, Any]) -> dict[str, Any]:
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

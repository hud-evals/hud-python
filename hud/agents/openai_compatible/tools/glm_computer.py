"""GLM computer tool — backed by RFBClient."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, cast, get_args

from hud.agents.tools import RFBTool
from hud.agents.tools.base import AgentToolSpec, tool_err
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

GLM_COMPUTER_PARAMETERS: dict[str, Any] = {
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


class GLMComputerTool(RFBTool):
    """Translate GLM computer calls into RFBTool primitives with normalized coordinates."""

    name = "computer"

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        return GLM_COMPUTER_SPEC if GLM_COMPUTER_SPEC.supports_model(model) else None

    def to_params(self) -> dict[str, Any]:
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

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        arguments = _normalize_glm_args(arguments)
        action = arguments.get("action")
        if not isinstance(action, str):
            return tool_err("'action' is required")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("GLMComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    async def _dispatch(self, action: str, args: dict[str, Any]) -> MCPToolResult:
        start = _parse_glm_box(args.get("start_box"))
        end = _parse_glm_box(args.get("end_box"))

        if action == "screenshot":
            return await self.screenshot()

        if action == "WAIT":
            await self.wait(5000)
            return await self.screenshot()

        if action in ("left_click", "click", "right_click", "middle_click"):
            x, y = self._point(start, f"start_box required for {action}")
            button = {
                "left_click": "left",
                "click": "left",
                "right_click": "right",
                "middle_click": "middle",
            }[action]
            await self.click(x, y, button=button)  # type: ignore[arg-type]
            return await self.screenshot()

        if action == "hover":
            x, y = self._point(start, "start_box required for hover")
            await self.move(x, y)
            return await self.screenshot()

        if action == "left_double_click":
            x, y = self._point(start, "start_box required for left_double_click")
            await self.click(x, y, count=2, interval_ms=100)
            return await self.screenshot()

        if action == "left_drag":
            sx, sy = self._point(start, "start_box required for left_drag")
            ex, ey = self._point(end, "end_box required for left_drag")
            await self.drag([(sx, sy), (ex, ey)])
            return await self.screenshot()

        if action == "key":
            raw_keys = args.get("keys")
            if isinstance(raw_keys, list):
                keys = [str(k).strip().lower() for k in cast("list[Any]", raw_keys)]
            else:
                keys = [k.strip().lower() for k in str(raw_keys or "").split("+") if k.strip()]
            if not keys:
                return tool_err("keys required for key action")
            await self.press_keys(keys)
            return await self.screenshot()

        if action == "type":
            content = args.get("content")
            if not isinstance(content, str) or not content:
                return tool_err("content required for type")
            await self.type_text(content)
            return await self.screenshot()

        if action == "scroll":
            direction = args.get("direction")
            if direction not in ("up", "down"):
                return tool_err("direction must be 'up' or 'down'")
            point = start or (GLM_COORDINATE_SPACE // 2, GLM_COORDINATE_SPACE // 2)
            x, y = self._scale_normalized_point(point)
            step = int(args.get("step") or 5)
            sy = step if direction == "down" else -step
            await self.scroll(x, y, scroll_y=sy)
            return await self.screenshot()

        return tool_err(f"Unknown action: {action}")

    def _point(self, point: tuple[int, int] | None, message: str) -> tuple[int, int]:
        if point is None:
            raise ValueError(message)
        return self._scale_normalized_point(point)

    def _scale_normalized_point(self, point: tuple[int, int]) -> tuple[int, int]:
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


__all__ = ["GLM_SYSTEM_INSTRUCTIONS", "VALID_GLM_ACTIONS", "GLMComputerTool"]

"""Qwen computer tool — backed by RFBClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from hud.agents.tools import RFBTool
from hud.agents.tools.base import AgentToolSpec, tool_err
from hud.types import MCPToolResult

if TYPE_CHECKING:
    from openai.types.shared_params.function_parameters import FunctionParameters

logger = logging.getLogger(__name__)

QWEN_COMPUTER_SPEC = AgentToolSpec(
    api_type="computer_use",
    api_name="computer_use",
    supported_models=("qwen*",),
)


class QwenComputerUseToolParam(TypedDict):
    type: Literal["computer_use"]
    name: str
    display_width_px: int
    display_height_px: int
    description: str
    parameters: FunctionParameters


class QwenComputerTool(RFBTool):
    """Translate Qwen computer_use calls into RFBTool primitives."""

    name = "computer_use"

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        return QWEN_COMPUTER_SPEC if QWEN_COMPUTER_SPEC.supports_model(model) else None

    def to_params(self) -> QwenComputerUseToolParam:
        return {
            "type": "computer_use",
            "name": self.name,
            "display_width_px": self.display_width,
            "display_height_px": self.display_height,
            "description": _qwen_description(self.display_width, self.display_height),
            "parameters": QWEN_COMPUTER_PARAMETERS,
        }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        action = arguments.get("action")
        if not isinstance(action, str):
            return tool_err("action is required")
        if action in ("terminate", "answer"):
            return tool_err(f"{action} action is not supported for computer control.")
        try:
            return await self._dispatch(action, arguments)
        except Exception as exc:
            logger.exception("QwenComputerTool action %s failed", action)
            return tool_err(f"computer action {action!r} failed: {exc}")

    async def _dispatch(self, action: str, args: dict[str, Any]) -> MCPToolResult:
        coordinate = _parse_coordinate(args.get("coordinate"))

        if action == "screenshot":
            return await self.screenshot()

        if action in ("left_click", "right_click", "middle_click"):
            x, y = _require_coord(coordinate, action)
            button = {"left_click": "left", "right_click": "right", "middle_click": "middle"}[
                action
            ]
            await self.click(x, y, button=button)  # type: ignore[arg-type]
            return await self.screenshot()

        if action == "double_click":
            x, y = _require_coord(coordinate, action)
            await self.click(x, y, count=2, interval_ms=100)
            return await self.screenshot()

        if action == "triple_click":
            x, y = _require_coord(coordinate, action)
            await self.click(x, y, count=3, interval_ms=100)
            return await self.screenshot()

        if action == "mouse_move":
            x, y = _require_coord(coordinate, action)
            await self.move(x, y)
            return await self.screenshot()

        if action == "type":
            text = args.get("text")
            if not isinstance(text, str):
                return tool_err("text is required for type")
            await self.type_text(text)
            return await self.screenshot()

        if action == "key":
            keys = args.get("keys")
            if not isinstance(keys, list):
                return tool_err("keys is required for key")
            await self.press_keys(cast("list[str]", keys))
            return await self.screenshot()

        if action in ("scroll", "hscroll"):
            pixels = args.get("pixels")
            if not isinstance(pixels, int | float):
                return tool_err("pixels is required for scroll")
            sx = int(pixels) if action == "hscroll" else 0
            sy = -int(pixels) if action == "scroll" else 0
            cx = coordinate[0] if coordinate else None
            cy = coordinate[1] if coordinate else None
            await self.scroll(cx, cy, scroll_x=sx, scroll_y=sy)
            return await self.screenshot()

        if action == "left_click_drag":
            x, y = _require_coord(coordinate, action)
            mouse = self.client.conn.mouse
            start = (mouse.x, mouse.y)
            await self.drag([start, (x, y)])
            return await self.screenshot()

        if action == "wait":
            time_val = args.get("time")
            if not isinstance(time_val, int | float) or time_val < 0:
                return tool_err("time must be a non-negative number")
            await self.wait(int(time_val * 1000))
            return await self.screenshot()

        return tool_err(f"Unknown action: {action}")


QWEN_COMPUTER_PARAMETERS: dict[str, Any] = {
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


def _parse_coordinate(coordinate: Any) -> tuple[int, int] | None:
    if not isinstance(coordinate, list | tuple):
        return None
    coord = cast("list[Any] | tuple[Any, ...]", coordinate)
    if len(coord) < 2:
        return None
    try:
        return int(coord[0]), int(coord[1])
    except (TypeError, ValueError):
        return None


def _require_coord(coordinate: tuple[int, int] | None, action: str) -> tuple[int, int]:
    if coordinate is None:
        raise ValueError(f"coordinate is required for {action}")
    return coordinate


__all__ = ["QwenComputerTool", "QwenComputerUseToolParam"]

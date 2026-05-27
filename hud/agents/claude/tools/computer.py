"""Agent-side Claude native computer tool.

The environment exposes a generic computer capability. Claude-specific native
tool formatting and argument translation live here, on the agent side.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from mcp.types import ImageContent

from hud.agents.tools.computer import (
    computer_error_result,
    computer_tool_info,
    execute_computer_calls,
    first_image_data,
)
from hud.types import MCPToolResult

from .base import ClaudeTool, ClaudeToolSpec
from .settings import claude_tool_settings

if TYPE_CHECKING:
    import mcp.types as types
    from anthropic.types.beta import (
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
    )

    from hud.agents.tools.base import CallTool

logger = logging.getLogger(__name__)

ANTHROPIC_TO_CLA_KEYS = {
    "Return": "enter",
    "Escape": "escape",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Backspace": "backspace",
    "Delete": "delete",
    "Tab": "tab",
    "Space": "space",
    "Control": "ctrl",
    "Alt": "alt",
    "Shift": "shift",
    "Meta": "win",
    "Command": "cmd",
    "Super": "win",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
    "Insert": "insert",
    "F1": "f1",
    "F2": "f2",
    "F3": "f3",
    "F4": "f4",
    "F5": "f5",
    "F6": "f6",
    "F7": "f7",
    "F8": "f8",
    "F9": "f9",
    "F10": "f10",
    "F11": "f11",
    "F12": "f12",
}

CLAUDE_COMPUTER_SPECS: tuple[ClaudeToolSpec, ...] = (
    ClaudeToolSpec(
        api_type="computer_20251124",
        api_name="computer",
        beta="computer-use-2025-11-24",
        supported_models=(
            "*claude-opus-4-6*",
            "*claude-sonnet-4-6*",
            "*claude-opus-4-7*",
        ),
    ),
    ClaudeToolSpec(
        api_type="computer_20250124",
        api_name="computer",
        beta="computer-use-2025-01-24",
        supported_models=(
            "*claude-sonnet-4-5*",
            "*claude-haiku-4-5*",
        ),
    ),
)


class ClaudeComputerTool(ClaudeTool):
    """Translate Claude native computer calls into environment computer calls."""

    name = "computer"
    capability = "computer"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        for candidate in CLAUDE_COMPUTER_SPECS:
            if candidate.supports_model(model):
                return candidate
        return None

    def __init__(
        self,
        *,
        env_tool_name: str,
        spec: ClaudeToolSpec,
        display_width: int,
        display_height: int,
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)
        self.display_width = display_width
        self.display_height = display_height

    @classmethod
    def from_native_tool(
        cls,
        tool: types.Tool,
        model: str,
    ) -> ClaudeComputerTool | None:
        spec = cls.default_spec(model)
        if spec is None:
            return None

        computer_info = computer_tool_info(
            tool,
            default_width=claude_tool_settings.COMPUTER_WIDTH,
            default_height=claude_tool_settings.COMPUTER_HEIGHT,
        )

        return cls(
            env_tool_name=tool.name,
            spec=spec,
            display_width=computer_info.display_width,
            display_height=computer_info.display_height,
        )

    def to_params(
        self,
    ) -> BetaToolComputerUse20250124Param | BetaToolComputerUse20251124Param:
        if self.spec.api_type == "computer_20251124":
            return cast(
                "BetaToolComputerUse20251124Param",
                {
                    "type": "computer_20251124",
                    "name": self.name,
                    "display_width_px": self.display_width,
                    "display_height_px": self.display_height,
                    "display_number": 1,
                    "enable_zoom": True,
                },
            )
        return cast(
            "BetaToolComputerUse20250124Param",
            {
                "type": "computer_20250124",
                "name": self.name,
                "display_width_px": self.display_width,
                "display_height_px": self.display_height,
                "display_number": 1,
            },
        )

    async def execute(
        self,
        call_tool: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        action = arguments.get("action")

        if action == "zoom":
            return await self._zoom(call_tool, arguments)

        return await execute_computer_calls(
            call_tool,
            env_tool_name=self.env_tool_name,
            calls=self._env_calls(arguments),
            ensure_screenshot=False,
        )

    def _env_calls(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        action = arguments.get("action")
        coordinate = arguments.get("coordinate")
        text = arguments.get("text")

        def xy() -> tuple[int | None, int | None]:
            if isinstance(coordinate, list):
                coords = cast("list[Any]", coordinate)
                if len(coords) >= 2:
                    return int(coords[0]), int(coords[1])
            return None, None

        if action == "screenshot":
            return [{"action": "screenshot"}]
        if action in ("left_click", "click"):
            x, y = xy()
            return [{"action": "click", "x": x, "y": y, "hold_keys": self._hold_keys(text)}]
        if action == "double_click":
            x, y = xy()
            return [
                {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "pattern": [100],
                    "hold_keys": self._hold_keys(text),
                }
            ]
        if action == "triple_click":
            x, y = xy()
            return [
                {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "pattern": [100, 100],
                    "hold_keys": self._hold_keys(text),
                }
            ]
        if action == "right_click":
            x, y = xy()
            return [
                {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "button": "right",
                    "hold_keys": self._hold_keys(text),
                }
            ]
        if action == "middle_click":
            x, y = xy()
            return [
                {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "button": "middle",
                    "hold_keys": self._hold_keys(text),
                }
            ]
        if action in ("mouse_move", "move"):
            x, y = xy()
            return [{"action": "move", "x": x, "y": y}]
        if action == "type":
            return [{"action": "write", "text": text}]
        if action == "key":
            keys = self._keys(text)
            repeat = arguments.get("repeat")
            repeat = repeat if isinstance(repeat, int) and repeat > 0 else 1
            return [{"action": "press", "keys": keys} for _ in range(min(repeat, 100))]
        if action == "scroll":
            x, y = xy()
            scroll_x, scroll_y = self._scroll(arguments)
            return [
                {
                    "action": "scroll",
                    "x": x,
                    "y": y,
                    "scroll_x": scroll_x,
                    "scroll_y": scroll_y,
                    "hold_keys": self._hold_keys(text),
                }
            ]
        if action in ("left_click_drag", "drag"):
            start = arguments.get("start_coordinate")
            path: list[dict[str, Any]] = []
            if isinstance(start, list):
                start_coords = cast("list[Any]", start)
                if len(start_coords) >= 2:
                    path.append({"x": start_coords[0], "y": start_coords[1]})
            if isinstance(coordinate, list):
                end_coords = cast("list[Any]", coordinate)
                if len(end_coords) >= 2:
                    if not path:
                        return [
                            {"action": "mouse_down", "button": "left"},
                            {"action": "move", "x": end_coords[0], "y": end_coords[1]},
                            {"action": "mouse_up", "button": "left"},
                        ]
                    path.append({"x": end_coords[0], "y": end_coords[1]})
            return [{"action": "drag", "path": path, "hold_keys": self._hold_keys(text)}]
        if action == "wait":
            duration = arguments.get("duration") or 0
            return [{"action": "wait", "time": int(float(duration) * 1000)}]
        if action == "hold_key":
            keys = self._keys(text)
            return [
                {
                    "action": "hold_key",
                    "text": keys[0] if keys else text,
                    "duration": arguments.get("duration"),
                }
            ]
        if action == "left_mouse_down":
            return [{"action": "mouse_down", "button": "left"}]
        if action == "left_mouse_up":
            return [{"action": "mouse_up", "button": "left"}]
        if action == "cursor_position":
            return [{"action": "position"}]
        return [dict(arguments)]

    async def _zoom(
        self,
        call_tool: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        region = arguments.get("region")
        region_value = cast("list[Any] | tuple[Any, ...]", region)
        if not isinstance(region, (list, tuple)) or len(region_value) != 4:
            return computer_error_result("region must be [x0, y0, x1, y1]")

        screenshot = await super().execute(call_tool, {"action": "screenshot"})
        if screenshot.isError:
            return screenshot
        image_data = first_image_data(screenshot)
        if image_data is None:
            return computer_error_result("screenshot returned no image")

        try:
            x0, y0, x1, y1 = (int(v) for v in region_value)
            image = ImageContent(
                type="image",
                mimeType="image/png",
                data=_crop_png(image_data, (x0, y0, x1, y1)),
            )
            return MCPToolResult(content=[image], isError=False)
        except Exception as exc:
            logger.warning("Claude computer zoom failed: %s", exc)
            return computer_error_result(str(exc))

    @staticmethod
    def _keys(text: str | None) -> list[str]:
        if not text:
            return []
        mapped = _map_key(text)
        return [k.strip() for k in mapped.split("+")] if "+" in mapped else [mapped]

    @staticmethod
    def _hold_keys(text: str | None) -> list[str] | None:
        keys = ClaudeComputerTool._keys(text)
        return keys or None

    @staticmethod
    def _scroll(arguments: dict[str, Any]) -> tuple[int | None, int | None]:
        amount = arguments.get("scroll_amount")
        amount = amount if isinstance(amount, int) and amount >= 0 else 0
        pixels = amount * 100
        match arguments.get("scroll_direction"):
            case "down":
                return None, pixels
            case "up":
                return None, -pixels
            case "right":
                return pixels, None
            case "left":
                return -pixels, None
            case _:
                return None, None


def _map_key(key: str) -> str:
    if "+" in key:
        return "+".join(_map_key(part) for part in key.split("+"))
    return ANTHROPIC_TO_CLA_KEYS.get(key, ANTHROPIC_TO_CLA_KEYS.get(key.capitalize(), key.lower()))


def _crop_png(image_data: str, region: tuple[int, int, int, int]) -> str:
    from PIL import Image  # type: ignore[import-not-found]

    image = Image.open(BytesIO(base64.b64decode(image_data)))
    crop = image.crop(region)
    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

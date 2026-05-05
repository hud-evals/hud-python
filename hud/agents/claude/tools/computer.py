"""Agent-side Claude native computer tool.

The environment exposes a generic computer capability. Claude-specific native
tool formatting and argument translation live here, on the agent side.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp.types import ImageContent, TextContent

from hud.types import MCPToolResult

from .base import CallTool, ClaudeTool, ClaudeToolSpec, call_tool
from .settings import claude_tool_settings

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
    )

    from hud.agents.tools import EnvironmentCapability

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

_AUTO_SCREENSHOT_OFF_SPECS = {"computer_20251124"}


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
        model: str,
        display_width: int,
        display_height: int,
        schema: Literal["hud", "anthropic"],
    ) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=self._resolve_spec(spec, model))
        self.display_width = display_width
        self.display_height = display_height
        self.schema = schema

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        spec: ClaudeToolSpec,
        model: str,
    ) -> ClaudeComputerTool:
        tool = capability.tool
        props = tool.inputSchema.get("properties", {}) if isinstance(tool.inputSchema, dict) else {}
        schema: Literal["hud", "anthropic"] = (
            "anthropic" if {"coordinate", "scroll_direction"} & set(props) else "hud"
        )

        metadata_resolution = capability.metadata.get("resolution", {})
        if not isinstance(metadata_resolution, dict):
            metadata_resolution = {}
        resolution = (tool.meta or {}).get("resolution", {}) if tool.meta else {}
        display_width = int(
            metadata_resolution.get("width")
            or resolution.get("width")
            or claude_tool_settings.COMPUTER_WIDTH
        )
        display_height = int(
            metadata_resolution.get("height")
            or resolution.get("height")
            or claude_tool_settings.COMPUTER_HEIGHT
        )

        return cls(
            env_tool_name=capability.tool_name,
            spec=spec,
            model=model,
            display_width=display_width,
            display_height=display_height,
            schema=schema,
        )

    @staticmethod
    def _resolve_spec(spec: ClaudeToolSpec, model: str) -> ClaudeToolSpec:
        if spec.api_type and spec.api_type.startswith("computer_"):
            return spec
        for candidate in CLAUDE_COMPUTER_SPECS:
            if candidate.supports_model(model):
                return candidate
        return spec

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
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        if self.schema == "anthropic":
            return await self._call_env(caller, self._as_anthropic_arguments(arguments))
        return await self._call_env_tool(caller, arguments)

    async def _call_env(
        self,
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        return await call_tool(caller, self.env_tool_name, arguments)

    async def _call_env_tool(
        self,
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        action = arguments.get("action")

        if action == "zoom":
            return await self._zoom(caller, arguments)

        calls = self._env_calls(arguments)
        result = MCPToolResult(content=[], isError=False)
        for call in calls:
            result = await self._call_env(caller, call)
            if result.isError:
                return result
        return result

    def _as_anthropic_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if (
            self.spec.api_type in _AUTO_SCREENSHOT_OFF_SPECS
            and args.get("action") != "screenshot"
            and "take_screenshot_on_click" not in args
        ):
            args["take_screenshot_on_click"] = False
        return args

    def _env_calls(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        action = arguments.get("action")
        coordinate = arguments.get("coordinate")
        text = arguments.get("text")

        def xy() -> tuple[int | None, int | None]:
            if isinstance(coordinate, list) and len(coordinate) >= 2:
                return coordinate[0], coordinate[1]
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
            path = []
            if isinstance(start, list) and len(start) >= 2:
                path.append({"x": start[0], "y": start[1]})
            if isinstance(coordinate, list) and len(coordinate) >= 2:
                if not path:
                    return [
                        {"action": "mouse_down", "button": "left"},
                        {"action": "move", "x": coordinate[0], "y": coordinate[1]},
                        {"action": "mouse_up", "button": "left"},
                    ]
                path.append({"x": coordinate[0], "y": coordinate[1]})
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
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        region = arguments.get("region")
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return MCPToolResult(
                content=[TextContent(type="text", text="region must be [x0, y0, x1, y1]")],
                isError=True,
            )

        screenshot = await self._call_env(caller, {"action": "screenshot"})
        if screenshot.isError:
            return screenshot
        image_data = _first_image(screenshot)
        if image_data is None:
            return MCPToolResult(
                content=[TextContent(type="text", text="screenshot returned no image")],
                isError=True,
            )

        try:
            x0, y0, x1, y1 = (int(v) for v in region)
            image = ImageContent(
                type="image",
                mimeType="image/png",
                data=_crop_png(image_data, (x0, y0, x1, y1)),
            )
            return MCPToolResult(content=[image], isError=False)
        except Exception as exc:
            logger.warning("Claude computer zoom failed: %s", exc)
            return MCPToolResult(content=[TextContent(type="text", text=str(exc))], isError=True)

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


def _first_image(result: MCPToolResult) -> str | None:
    for block in result.content or []:
        if isinstance(block, ImageContent):
            return block.data
    return None


def _crop_png(image_data: str, region: tuple[int, int, int, int]) -> str:
    from PIL import Image  # type: ignore[import-not-found]

    image = Image.open(BytesIO(base64.b64decode(image_data)))
    crop = image.crop(region)
    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = ["CLAUDE_COMPUTER_SPECS", "ClaudeComputerTool"]

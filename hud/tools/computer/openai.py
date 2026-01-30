# flake8: noqa: B008
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ContentBlock, TextContent
from pydantic import Field

from hud.tools.computer.settings import computer_settings
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, Coordinate
from hud.types import AgentType

from .hud import HudComputerTool

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


# Map OpenAI key names to CLA standard keys
OPENAI_TO_CLA_KEYS = {
    # Common variations
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


class OpenAIComputerTool(HudComputerTool):
    """
    OpenAI Computer Use tool for interacting with the computer.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.OPENAI: NativeToolSpec(
            api_type="computer_use_preview",
            api_name="computer",
            role="computer",  # Mutually exclusive with other computer tools when native
        ),
        AgentType.OPERATOR: NativeToolSpec(
            api_type="computer_use_preview",
            api_name="computer",
            role="computer",  # Mutually exclusive with other computer tools when native
        ),
    }

    def __init__(
        self,
        # Define within environment based on platform
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        # Overrides for what dimensions the agent thinks it operates in
        width: int = computer_settings.OPENAI_COMPUTER_WIDTH,
        height: int = computer_settings.OPENAI_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.OPENAI_RESCALE_IMAGES,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with OpenAI's default dimensions.

        Args:
            width: Width for agent coordinate system (default: 1024)
            height: Height for agent coordinate system (default: 768)
            rescale_images: If True, rescale screenshots. If False, only rescale action coordinates
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        # Create instance-level native_specs with display dimensions
        instance_native_specs = {
            AgentType.OPENAI: NativeToolSpec(
                api_type="computer_use_preview",
                api_name="computer",
                role="computer",
                extra={
                    "display_width": width,
                    "display_height": height,
                },
            ),
            AgentType.OPERATOR: NativeToolSpec(
                api_type="computer_use_preview",
                api_name="computer",
                role="computer",
                extra={
                    "display_width": width,
                    "display_height": height,
                },
            ),
        }

        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            name=name or "openai_computer",
            title=title or "OpenAI Computer Tool",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            native_specs=instance_native_specs,
            **kwargs,
        )

    def _map_openai_key_to_cla(self, key: str) -> str:
        """Map OpenAI key name to CLA standard key."""
        # OpenAI uses lowercase key names
        return OPENAI_TO_CLA_KEYS.get(key.lower(), key.lower())

    async def __call__(  # type: ignore[override]
        self,
        type: Literal[
            "screenshot",
            "click",
            "double_click",
            "scroll",
            "type",
            "wait",
            "move",
            "keypress",
            "drag",
            "response",
            "custom",
        ] = Field(..., description="The action type to perform"),
        # Coordinate parameters
        x: float | int | None = Field(
            None, description="X coordinate for click/move/scroll actions"
        ),
        y: float | int | None = Field(
            None, description="Y coordinate for click/move/scroll actions"
        ),
        coordinate: list[float | int] | int | None = Field(
            None, description="Coordinate as [x, y] for click/move/scroll actions"
        ),
        # Button parameter
        button: Literal["left", "right", "middle", "back", "forward"] | None = Field(
            None, description="Mouse button for click actions (left, right, middle, wheel)"
        ),
        # Text parameter
        text: str | None = Field(None, description="Text to type or response text"),
        # Scroll parameters
        scroll_x: int | None = Field(None, description="Horizontal scroll amount"),
        scroll_y: int | None = Field(None, description="Vertical scroll amount"),
        # Wait parameter
        ms: int | None = Field(None, description="Time to wait in milliseconds"),
        # Key press parameter
        keys: list[str] | str | int | None = Field(None, description="Keys to press"),
        # Drag parameter
        path: list[Coordinate] | list[list[float | int]] | int | None = Field(
            None,
            description="Path for drag actions as list of {x, y} dicts or [[x, y], ...]",
        ),
        # Custom action parameter
        action: str | None = Field(None, description="Custom action name"),
    ) -> list[ContentBlock]:
        """
        Handle OpenAI Computer Use API calls.

        This converts OpenAI's action format (based on OperatorAdapter) to HudComputerTool's format.

        Returns:
            List of MCP content blocks
        """
        logger.info("OpenAIComputerTool received type: %s", type)

        # Normalize coordinate inputs (some models emit [x, y] instead of x/y fields)
        if (
            isinstance(coordinate, (list, tuple))
            and (x is None and y is None)
            and len(coordinate) >= 2
        ):
            x, y = coordinate[0], coordinate[1]

        # Helper to coerce numeric inputs to ints for executors
        def _to_int(value: float | int | None) -> int | None:
            if value is None:
                return None
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return None

        x_int = _to_int(x)
        y_int = _to_int(y)

        # Normalize keys to list[str]
        if isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, int):
            keys = [str(keys)]

        # Process based on action type
        if type == "screenshot":
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                result = ContentResult(base64_image=screenshot_base64)
            else:
                result = ContentResult(error="Failed to take screenshot")

        elif type == "click":
            if x_int is not None and y_int is not None:
                # Cast button to proper literal type
                button_literal = cast(
                    "Literal['left', 'right', 'middle', 'back', 'forward']", button or "left"
                )
                logger.info(
                    "🎯 Click: Agent coords (%d, %d) in %dx%d space, scale factors: %.3f, %.3f",
                    x_int,
                    y_int,
                    self.width,
                    self.height,
                    self.scale_x,
                    self.scale_y,
                )
                scaled_x, scaled_y = self._scale_coordinates(x_int, y_int)
                logger.info("   → Screen coords (%d, %d) in %dx%d display", scaled_x, scaled_y, self.environment_width, self.environment_height)
                result = await self.executor.click(x=scaled_x, y=scaled_y, button=button_literal)
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for click")
                )

        elif type == "double_click":
            if x_int is not None and y_int is not None:
                # Use pattern for double-click
                scaled_x, scaled_y = self._scale_coordinates(x_int, y_int)
                result = await self.executor.click(
                    x=scaled_x, y=scaled_y, button="left", pattern=[100]
                )
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="x and y coordinates required for double_click"
                    )
                )

        elif type == "scroll":
            if x_int is None or y_int is None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="x and y coordinates required for scroll"
                    )
                )

            # scroll_x and scroll_y default to 0 if not provided
            scaled_x, scaled_y = self._scale_coordinates(x_int, y_int)
            result = await self.executor.scroll(
                x=scaled_x, y=scaled_y, scroll_x=scroll_x or 0, scroll_y=scroll_y or 0
            )

        elif type == "type":
            if text is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for type"))
            result = await self.executor.write(text=text, enter_after=False)

        elif type == "wait":
            wait_time = ms or 1000  # Default to 1 second
            result = await self.executor.wait(time=wait_time)

        elif type == "move":
            if x_int is not None and y_int is not None:
                scaled_x, scaled_y = self._scale_coordinates(x_int, y_int)
                result = await self.executor.move(x=scaled_x, y=scaled_y)
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="x and y coordinates required for move")
                )

        elif type == "keypress":
            if keys is None or len(keys) == 0:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="keys is required for keypress")
                )

            # Map OpenAI keys to CLA standard
            cla_keys = []
            for key in keys:
                cla_key = self._map_openai_key_to_cla(key)
                cla_keys.append(cla_key)

            result = await self.executor.press(keys=cla_keys)

        elif type == "drag":
            if path is None or len(path) < 2:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="path with at least 2 points required for drag"
                    )
                )

            # Convert path from list of Coordinate objects or [[x, y], ...] to tuples
            drag_path: list[tuple[int, int]] = []
            if path and isinstance(path, list) and isinstance(path[0], Coordinate):
                drag_path = [(point.x, point.y) for point in cast("list[Coordinate]", path)]
            elif path and isinstance(path, list) and isinstance(path[0], dict):
                for point in cast("list[dict[str, float | int]]", path):
                    px = _to_int(point.get("x"))
                    py = _to_int(point.get("y"))
                    if px is None or py is None:
                        continue
                    drag_path.append((px, py))
            elif path and isinstance(path, list):
                for point in cast("list[list[float | int]]", path):
                    if len(point) < 2:
                        continue
                    px = _to_int(point[0])
                    py = _to_int(point[1])
                    if px is None or py is None:
                        continue
                    drag_path.append((px, py))

            scaled_path = self._scale_path(drag_path)
            result = await self.executor.drag(path=scaled_path)

        elif type == "response":
            if text is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="text is required for response")
                )
            # Response returns content blocks directly
            return [TextContent(text=text, type="text")]

        elif type == "custom":
            # For custom actions, we just return an error since HudComputerTool doesn't support them
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Custom action not supported: {action}")
            )

        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action type: {type}"))

        # Rescale screenshot in result if the action itself returned one
        # This happens for actions like "screenshot" that produce an image
        if isinstance(result, ContentResult) and result.base64_image and self.rescale_images:
            rescaled_image = await self._rescale_screenshot(result.base64_image)
            result.base64_image = rescaled_image

        # Auto-capture screenshot for actions that need visual confirmation
        # but didn't return one themselves (e.g., click, type)
        screenshot_actions = {
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

        # IMPORTANT: The 'not result.base64_image' condition ensures we don't
        # double-rescale. If the action returned a screenshot (rescaled above),
        # we skip this auto-screenshot section.
        if (
            type in screenshot_actions
            and type != "screenshot"
            and isinstance(result, ContentResult)
            and not result.base64_image  # Only if no screenshot from action itself
        ):
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale the new screenshot if requested
                screenshot_base64 = await self._rescale_screenshot(screenshot_base64)
                result = ContentResult(
                    output=result.output, error=result.error, base64_image=screenshot_base64
                )

        # Convert to content blocks
        return result.to_content_blocks()

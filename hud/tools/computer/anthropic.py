# flake8: noqa: B008
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ContentBlock
from pydantic import Field

from hud.tools.types import ContentResult

from .hud import HudComputerTool
from .settings import computer_settings

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolComputerUse20250124Param

    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

# Map Anthropic key names to CLA standard keys
ANTHROPIC_TO_CLA_KEYS = {
    # Common variations
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
    "Meta": "win",  # Windows key
    "Command": "cmd",  # macOS
    "Super": "win",  # Linux
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


class AnthropicComputerTool(HudComputerTool):
    """
    Anthropic Computer Use tool for interacting with the computer.
    """

    name: str = "computer"
    api_type: str = "computer_20250124"

    def __init__(
        self,
        # Define within environment based on platform
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        # Overrides for what dimensions the agent thinks it operates in
        width: int = computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        height: int = computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.ANTHROPIC_RESCALE_IMAGES,
        # What the agent sees as the tool's name, title, and description
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with Anthropic's default dimensions.

        Args:
            width: Target width for rescaling (None = use environment width)
            height: Target height for rescaling (None = use environment height)
            rescale_images: If True, rescale screenshots. If False, only rescale action coordinates
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            name=name or "anthropic_computer",
            title=title or "Anthropic Computer Tool",
            description=description or "Control computer with mouse, keyboard, and screenshot",
            **kwargs,
        )

    def to_params(self) -> BetaToolComputerUse20250124Param:
        """Convert to Anthropic tool parameters."""
        return cast(
            "BetaToolComputerUse20250124Param",
            {
                "type": self.api_type,
                "name": self.name,
                "display_width_px": self.width,
                "display_height_px": self.height,
            },
        )

    def _map_anthropic_key_to_cla(self, key: str) -> str:
        """Map Anthropic key name to CLA standard key."""
        # Handle key combinations like "ctrl+a"
        if "+" in key:
            parts = key.split("+")
            mapped_parts = []
            for part in parts:
                # Try exact match first, then case-insensitive
                mapped = ANTHROPIC_TO_CLA_KEYS.get(
                    part, ANTHROPIC_TO_CLA_KEYS.get(part.capitalize(), part.lower())
                )
                mapped_parts.append(mapped)
            return "+".join(mapped_parts)
        else:
            # Single key - try exact match first, then case-insensitive
            return ANTHROPIC_TO_CLA_KEYS.get(
                key, ANTHROPIC_TO_CLA_KEYS.get(key.capitalize(), key.lower())
            )

    async def __call__(
        self,
        action: str = Field(..., description="The action to perform on the computer"),
        coordinate: list[int] | tuple[int, int] | None = Field(
            None, description="The coordinate to interact with on the computer [x, y]"
        ),
        text: str | None = Field(
            None, description="The text to type on the computer or key to press"
        ),
        start_coordinate: list[int] | tuple[int, int] | None = Field(
            None, description="The starting coordinate for drag actions [x, y]"
        ),
        scroll_direction: str | None = Field(
            None, description="The direction to scroll (up, down, left, right)"
        ),
        scroll_amount: int | None = Field(None, description="The amount to scroll"),
        duration: float | None = Field(None, description="The duration of the action in seconds"),
        take_screenshot_on_click: bool = Field(
            True, description="Whether to take a screenshot after clicking"
        ),
    ) -> list[ContentBlock]:
        """
        Handle Anthropic Computer Use API calls.

        This converts Anthropic's action format to HudComputerTool's format.

        Returns:
            List of MCP content blocks
        """
        logger.info("AnthropicComputerTool received action: %s", action)

        # Convert lists to tuples if needed
        coord_tuple = None
        if coordinate:
            coord_tuple = tuple(coordinate) if isinstance(coordinate, list) else coordinate

        start_coord_tuple = None
        if start_coordinate:
            start_coord_tuple = (
                tuple(start_coordinate) if isinstance(start_coordinate, list) else start_coordinate
            )

        # Map Anthropic actions to HudComputerTool actions
        if action == "screenshot":
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                result = ContentResult(base64_image=screenshot_base64)
            else:
                result = ContentResult(error="Failed to take screenshot")

        elif action == "left_click" or action == "click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                logger.info("Scaled coordinates: %s, %s", scaled_x, scaled_y)
                result = await self.executor.click(x=scaled_x, y=scaled_y)
            else:
                result = await self.executor.click()

        elif action == "double_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for double-click
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100])
            else:
                result = await self.executor.click(pattern=[100])

        elif action == "triple_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for triple-click
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100, 100])
            else:
                result = await self.executor.click(pattern=[100, 100])

        elif action == "right_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="right")
            else:
                result = await self.executor.click(button="right")

        elif action == "middle_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="middle")
            else:
                result = await self.executor.click(button="middle")

        elif action == "mouse_move" or action == "move":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.move(x=scaled_x, y=scaled_y)
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="coordinate is required for mouse_move")
                )

        elif action == "type":
            if text:
                result = await self.executor.write(text=text)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for type"))

        elif action == "key":
            if text:
                # Anthropic sends single key or combo like "ctrl+a"
                # Map to CLA standard key format
                mapped_key = self._map_anthropic_key_to_cla(text)

                # Split key combination into list of keys
                if "+" in mapped_key:
                    keys_list = [k.strip() for k in mapped_key.split("+")]
                else:
                    keys_list = [mapped_key]

                result = await self.executor.press(keys=keys_list)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required for key"))

        elif action == "scroll":
            # Original implementation validates scroll_direction and scroll_amount
            if scroll_direction not in ["up", "down", "left", "right"]:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="scroll_direction must be 'up', 'down', 'left', or 'right'",
                    )
                )

            if scroll_amount is None or scroll_amount < 0:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="scroll_amount must be a non-negative int"
                    )
                )

            # Convert scroll amount from "clicks" to pixels
            # Anthropic's scroll_amount represents wheel clicks, not pixels
            # Standard conversion: 1 wheel click ≈ 100 pixels (3 lines of text)
            PIXELS_PER_WHEEL_CLICK = 100
            pixel_amount = scroll_amount * PIXELS_PER_WHEEL_CLICK

            # Convert direction to scroll amounts
            scroll_x = None
            scroll_y = None
            if scroll_direction == "down":
                scroll_y = pixel_amount
            elif scroll_direction == "up":
                scroll_y = -pixel_amount
            elif scroll_direction == "right":
                scroll_x = pixel_amount
            elif scroll_direction == "left":
                scroll_x = -pixel_amount

            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.scroll(
                    x=scaled_x, y=scaled_y, scroll_x=scroll_x, scroll_y=scroll_y
                )
            else:
                result = await self.executor.scroll(scroll_x=scroll_x, scroll_y=scroll_y)

        elif action == "left_click_drag" or action == "drag":
            # Anthropic sends drag with start and end coordinates
            if coord_tuple and len(coord_tuple) >= 2:
                if start_coord_tuple and len(start_coord_tuple) >= 2:
                    # Full drag path
                    path = [
                        (start_coord_tuple[0], start_coord_tuple[1]),
                        (coord_tuple[0], coord_tuple[1]),
                    ]
                    scaled_path = self._scale_path(path)
                    result = await self.executor.drag(path=scaled_path)
                else:
                    # Just end coordinate, drag from current position
                    # Original spec allows this
                    current_pos = [(0, 0), (coord_tuple[0], coord_tuple[1])]  # Simplified
                    scaled_path = self._scale_path(current_pos)
                    result = await self.executor.drag(path=scaled_path)
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for left_click_drag"
                    )
                )

        elif action == "wait":
            # Original spec expects duration in seconds
            if duration is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration is required for wait")
                )
            if duration < 0:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration must be non-negative")
                )
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))

            # Convert seconds to milliseconds for HudComputerTool
            result = await self.executor.wait(time=int(duration * 1000))

        elif action == "hold_key":
            # Original spec has hold_key action
            if text is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="text is required for hold_key")
                )
            if duration is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration is required for hold_key")
                )
            if duration < 0:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="duration must be non-negative")
                )
            if duration > 100:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="duration is too long"))

            # Hold key action
            result = await self.executor.hold_key(key=text, duration=duration)

        elif action == "left_mouse_down":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="coordinate is not accepted for left_mouse_down",
                    )
                )
            # Use generic mouse_down method
            result = await self.executor.mouse_down(button="left")

        elif action == "left_mouse_up":
            # These don't accept coordinates in original spec
            if coord_tuple is not None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is not accepted for left_mouse_up"
                    )
                )
            # Use generic mouse_up method
            result = await self.executor.mouse_up(button="left")

        elif action == "cursor_position":
            result = await self.executor.position()

        else:
            # Unknown action
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action: {action}"))

        # Rescale screenshot in result if present
        if isinstance(result, ContentResult) and result.base64_image and self.rescale_images:
            rescaled_image = await self._rescale_screenshot(result.base64_image)
            result.base64_image = rescaled_image

        # Handle screenshot for actions that need it
        screenshot_actions = {
            "screenshot",
            "left_click",
            "click",
            "double_click",
            "triple_click",
            "right_click",
            "middle_click",
            "mouse_move",
            "move",
            "type",
            "key",
            "scroll",
            "left_click_drag",
            "drag",
            "wait",
            "hold_key",
            "left_mouse_down",
            "left_mouse_up",
        }

        if (
            action in screenshot_actions
            and action != "screenshot"
            and take_screenshot_on_click
            and isinstance(result, ContentResult)
            and not result.base64_image
        ):
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                screenshot_base64 = await self._rescale_screenshot(screenshot_base64)
                result = ContentResult(
                    output=result.output, error=result.error, base64_image=screenshot_base64
                )

        # Convert to content blocks
        return result.to_content_blocks()

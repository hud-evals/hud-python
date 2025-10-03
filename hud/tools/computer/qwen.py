# flake8: noqa: B008
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ContentBlock
from pydantic import Field

from hud.tools.types import ContentResult

from .hud import HudComputerTool
from .settings import computer_settings

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


class QwenComputerTool(HudComputerTool):
    """
    Qwen Computer Use tool for interacting with the computer.
    """

    name: str = "computer_use"
    api_type: str = "computer_use"

    def __init__(
        self,
        # Define within environment based on platform
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        # Overrides for what dimensions the agent thinks it operates in
        width: int = computer_settings.QWEN_COMPUTER_WIDTH,
        height: int = computer_settings.QWEN_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.QWEN_RESCALE_IMAGES,
        # What the agent sees as the tool's name, title, and description
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with Qwen's default dimensions.

        Args:
            width: Target width for rescaling (None = use environment width)
            height: Target height for rescaling (None = use environment height)
            rescale_images: If True, rescale screenshots. If False, only rescale action coordinates
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        # Store dimensions for description
        self.display_width_px = width
        self.display_height_px = height

        # Build custom description with resolution info
        custom_description = (
            description
            or f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or
applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to
wait and take successive screenshots to see the results of your actions. E.g. if you
click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {width}x{height}.
* Whenever you intend to move the cursor to click on an element like an icon, you
should consult a screenshot to determine the coordinates of the element before
moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after
waiting, try adjusting your cursor position so that the tip of the cursor visually
falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the
center of the element. Don't click boxes on their edges.
""".strip()
        )

        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            name=name or "qwen_computer",
            title=title or "Qwen Computer Tool",
            description=custom_description,
            **kwargs,
        )

    def to_params(self) -> dict:
        """Convert to Qwen tool parameters."""
        return {
            "type": self.api_type,
            "name": self.name,
            "display_width_px": self.display_width_px,
            "display_height_px": self.display_height_px,
            "description": self.description,
            "parameters": {
                "properties": {
                    "action": {
                        "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs
key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the
screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate
on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel
coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel
coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel
coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel
coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel
coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status
(NOT SUPPORTED).
* `answer`: Answer a question (NOT SUPPORTED).
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
                    "keys": {
                        "description": "Required only by `action=key`.",
                        "type": "array",
                    },
                    "text": {
                        "description": "Required only by `action=type` and `action=answer`.",
                        "type": "string",
                    },
                    "coordinate": {
                        "description": (
                            "(x, y): The x (pixels from the left edge) and y "
                            "(pixels from the top edge) coordinates to move the mouse to."
                        ),
                        "type": "array",
                    },
                    "pixels": {
                        "description": (
                            "The amount of scrolling to perform. Positive values scroll up, "
                            "negative values scroll down. Required only by `action=scroll` "
                            "and `action=hscroll`."
                        ),
                        "type": "number",
                    },
                    "time": {
                        "description": "The seconds to wait. Required only by `action=wait`.",
                        "type": "number",
                    },
                    "status": {
                        "description": (
                            "The status of the task. Required only by `action=terminate`."
                        ),
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
        }

    async def __call__(
        self,
        action: str = Field(..., description="The action to perform on the computer"),
        keys: list[str] | None = Field(None, description="Keys for key action"),
        text: str | None = Field(None, description="Text to type"),
        coordinate: list[int] | tuple[int, int] | None = Field(
            None, description="The coordinate to interact with on the computer [x, y]"
        ),
        pixels: int | None = Field(None, description="Pixels to scroll"),
        time: float | None = Field(None, description="Time to wait in seconds"),
        status: str | None = Field(None, description="Status for terminate action"),
    ) -> list[ContentBlock]:
        """
        Handle Qwen Computer Use API calls.

        This converts Qwen's action format to HudComputerTool's format.

        Returns:
            List of MCP content blocks
        """
        logger.info("QwenComputerTool received action: %s", action)

        # Handle non-computer actions that should raise errors
        if action == "terminate":
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=(
                        "terminate action is not supported for computer control. This is a no-op."
                    ),
                )
            )

        if action == "answer":
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="answer action is not supported for computer control. This is a no-op.",
                )
            )

        # Convert lists to tuples if needed
        coord_tuple = None
        if coordinate:
            coord_tuple = tuple(coordinate) if isinstance(coordinate, list) else coordinate

        # Map Qwen actions to HudComputerTool actions
        if action == "left_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                logger.info("Scaled coordinates: %s, %s", scaled_x, scaled_y)
                result = await self.executor.click(x=scaled_x, y=scaled_y)
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="coordinate is required for left_click")
                )

        elif action == "double_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for double-click
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100])
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for double_click"
                    )
                )

        elif action == "triple_click":
            if coord_tuple and len(coord_tuple) >= 2:
                # Use pattern for triple-click (simulated as double-click)
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                # Note: triple-click simulated as double-click as per requirement
                result = await self.executor.click(x=scaled_x, y=scaled_y, pattern=[100])
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for triple_click"
                    )
                )

        elif action == "right_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="right")
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="coordinate is required for right_click")
                )

        elif action == "middle_click":
            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.click(x=scaled_x, y=scaled_y, button="middle")
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for middle_click"
                    )
                )

        elif action == "mouse_move":
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
            if keys:
                # Qwen sends an array of keys to press
                result = await self.executor.press(keys=keys)
            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="keys is required for key"))

        elif action == "scroll":
            if pixels is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="pixels is required for scroll")
                )

            # Qwen's pixels: positive scrolls up, negative scrolls down
            # HUD's scroll_y: positive scrolls down, negative scrolls up
            # So we need to negate the value
            scroll_y = -pixels

            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.scroll(x=scaled_x, y=scaled_y, scroll_y=scroll_y)
            else:
                result = await self.executor.scroll(scroll_y=scroll_y)

        elif action == "hscroll":
            if pixels is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="pixels is required for hscroll")
                )

            # For horizontal scroll, positive values scroll right, negative scroll left
            scroll_x = pixels

            if coord_tuple and len(coord_tuple) >= 2:
                scaled_x, scaled_y = self._scale_coordinates(coord_tuple[0], coord_tuple[1])
                result = await self.executor.scroll(x=scaled_x, y=scaled_y, scroll_x=scroll_x)
            else:
                result = await self.executor.scroll(scroll_x=scroll_x)

        elif action == "left_click_drag":
            if coord_tuple and len(coord_tuple) >= 2:
                # For drag, we need a path. Qwen provides the end coordinate.
                # We'll get the current position and drag from there to the target
                current_pos = await self.executor.position()
                if isinstance(current_pos, ContentResult) and current_pos.output:
                    # Parse the position from the output
                    match = re.search(r"x=(\d+), y=(\d+)", current_pos.output)
                    if match:
                        # Current position is in screen coordinates
                        screen_start_x, screen_start_y = int(match.group(1)), int(match.group(2))
                        # End position is in agent coordinates, needs scaling
                        scaled_end_x, scaled_end_y = self._scale_coordinates(
                            coord_tuple[0], coord_tuple[1]
                        )
                        # Create path in screen coordinates
                        path = [(screen_start_x, screen_start_y), (scaled_end_x, scaled_end_y)]
                        # Path is already in screen coordinates, no need to scale again
                        result = await self.executor.drag(path=path)
                    else:
                        raise McpError(
                            ErrorData(
                                code=INTERNAL_ERROR, message="Failed to parse current position"
                            )
                        )
                else:
                    raise McpError(
                        ErrorData(code=INTERNAL_ERROR, message="Failed to get current position")
                    )
            else:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="coordinate is required for left_click_drag"
                    )
                )

        elif action == "wait":
            if time is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="time is required for wait"))
            if time < 0:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="time must be non-negative"))

            # Convert seconds to milliseconds for HudComputerTool
            result = await self.executor.wait(time=int(time * 1000))

        else:
            # Unknown action
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Invalid action: {action}"))

        # Rescale screenshot in result if present
        if isinstance(result, ContentResult) and result.base64_image and self.rescale_images:
            rescaled_image = await self._rescale_screenshot(result.base64_image)
            result.base64_image = rescaled_image

        # Auto-add screenshot for interactive actions
        interactive_actions = {
            "left_click",
            "double_click",
            "triple_click",
            "right_click",
            "middle_click",
            "mouse_move",
            "type",
            "key",
            "scroll",
            "hscroll",
            "left_click_drag",
        }

        if (
            action in interactive_actions
            and isinstance(result, ContentResult)
            and not result.base64_image
        ):
            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:
                # Rescale screenshot if requested
                screenshot_base64 = await self._rescale_screenshot(screenshot_base64)
                result = ContentResult(
                    # note: we suppress the output since it's not useful
                    output="",
                    error=result.error,
                    base64_image=screenshot_base64,
                )

        # Convert to content blocks
        return result.to_content_blocks()

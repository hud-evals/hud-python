"""GLM computer tool for interacting with the computer.

GLM 4.6V uses PC action space with (0-999, 0-999) coordinate space.
Coordinates are automatically rescaled to actual screen dimensions.

Native PC actions:
- left_click, right_click, middle_click(start_box='[x,y]')
- hover(start_box='[x,y]')
- left_double_click(start_box='[x,y]')
- left_drag(start_box='[x,y]', end_box='[x,y]')
- key(keys='')
- type(content='')
- scroll(start_box='[x,y]', direction='', step=5)
- WAIT(), DONE(), FAIL()
- screenshot()
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock
from pydantic import Field

from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult
from hud.types import AgentType

from .hud import HudComputerTool
from .settings import computer_settings

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

# GLM uses normalized 0-999 coordinate space
GLM_COORDINATE_SPACE = 999

# All supported GLM PC actions with their call signatures:
# - left_click(start_box='[x,y]', element_info='')
# - right_click(start_box='[x,y]', element_info='')
# - middle_click(start_box='[x,y]', element_info='')
# - hover(start_box='[x,y]', element_info='')
# - left_double_click(start_box='[x,y]', element_info='')
# - left_drag(start_box='[x,y]', end_box='[x,y]', element_info='')
# - key(keys='ctrl+c')
# - type(content='text')
# - scroll(start_box='[x,y]', direction='up|down', step=5)
# - screenshot()
# - WAIT()
# - DONE()
# - FAIL()
GLMAction = Literal[
    "left_click",  # start_box='[x,y]'
    "click",  # alias for left_click
    "right_click",  # start_box='[x,y]'
    "middle_click",  # start_box='[x,y]'
    "hover",  # start_box='[x,y]'
    "left_double_click",  # start_box='[x,y]'
    "left_drag",  # start_box='[x,y]', end_box='[x,y]'
    "key",  # keys='ctrl+c'
    "type",  # content='text'
    "scroll",  # start_box='[x,y]', direction='up|down', step=5
    "screenshot",  # no params
    "WAIT",  # no params
    "DONE",  # no params - task completed
    "FAIL",  # no params - task failed
]

# Field definitions matching GLM's PC action space
ACTION_FIELD = Field(
    ...,
    description=(
        "Action to perform: "
        "left_click/right_click/middle_click/hover/left_double_click(start_box='[x,y]'), "
        "left_drag(start_box, end_box), "
        "key(keys='ctrl+c'), "
        "type(content='text'), "
        "scroll(start_box, direction, step), "
        "screenshot(), WAIT(), DONE(), FAIL()"
    ),
)
START_BOX_FIELD = Field(
    None,
    description="Position as '[x,y]' string or [x,y] array, coordinates 0-999 normalized",
)
END_BOX_FIELD = Field(
    None,
    description="End position for drag as '[x,y]' string or [x,y] array, coordinates 0-999",
)
CONTENT_FIELD = Field(None, description="Text content to type (for 'type' action)")
KEYS_FIELD = Field(None, description="Key(s) to press, e.g. 'enter', 'ctrl+c', 'alt+tab'")
DIRECTION_FIELD = Field(None, description="Scroll direction: 'up' or 'down'")
STEP_FIELD = Field(5, description="Scroll steps (default 5)")
ELEMENT_INFO_FIELD = Field(None, description="Optional description of the UI element")


class GLMComputerTool(HudComputerTool):
    """
    GLM Computer Tool for GLM-4.6V models.

    Uses GLM's native PC action space with normalized coordinates (0-999)
    that are automatically rescaled to actual screen dimensions.

    Supports actions: left_click, right_click, middle_click, hover,
    left_double_click, left_drag, key, type, scroll, WAIT, DONE, FAIL
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GLM_CUA: NativeToolSpec(
            api_type="function",
            api_name="glm_computer",
            role="computer",
        ),
    }

    def __init__(
        self,
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        width: int = computer_settings.GLM_COMPUTER_WIDTH,
        height: int = computer_settings.GLM_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.GLM_RESCALE_IMAGES,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GLM Computer Tool with coordinate scaling."""
        instance_native_specs = {
            AgentType.GLM_CUA: NativeToolSpec(
                api_type="function",
                api_name="glm_computer",
                role="computer",
                extra={
                    "display_width": width,
                    "display_height": height,
                    "coordinate_space": GLM_COORDINATE_SPACE,
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
            name=name or "glm_computer",
            title=title or "GLM Computer Tool",
            description=description
            or "Control computer with GLM PC action space. Coordinates use 0-999 scale.",
            native_specs=instance_native_specs,
            **kwargs,
        )

    def _parse_box(self, box: Any) -> tuple[int, int] | None:
        """Parse start_box/end_box to (x, y) tuple.

        Handles:
        - '[x,y]' string format
        - [x, y] list format
        - [[x, y]] nested list (bounding box format)
        """
        if box is None:
            return None

        # Handle string format: '[513,438]'
        if isinstance(box, str):
            box = box.strip()
            match = re.match(r"\[?\s*(\d+)\s*,\s*(\d+)\s*\]?", box)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return None

        # Handle list format: [513, 438] or [[513, 438]]
        if isinstance(box, list):
            # Unwrap nested list: [[x, y]] → [x, y]
            if len(box) == 1 and isinstance(box[0], list):
                box = box[0]
            if len(box) >= 2:
                try:
                    return (int(box[0]), int(box[1]))
                except (TypeError, ValueError):
                    return None

        return None

    def _scale_coord(self, coord: int, is_x: bool = True) -> int:
        """Scale coordinate from GLM's 0-999 space to actual screen pixels."""
        if is_x:
            return int(coord * self.environment_width / GLM_COORDINATE_SPACE)
        else:
            return int(coord * self.environment_height / GLM_COORDINATE_SPACE)

    def _parse_keys(self, keys: str | list[str] | None) -> list[str]:
        """Parse key input to list of keys."""
        if not keys:
            return []
        if isinstance(keys, list):
            return [k.strip().lower() for k in keys]
        # Handle 'ctrl+c' format
        return [k.strip().lower() for k in keys.split("+")]

    async def __call__(
        self,
        action: GLMAction = ACTION_FIELD,
        start_box: str | list | None = START_BOX_FIELD,
        end_box: str | list | None = END_BOX_FIELD,
        content: str | None = CONTENT_FIELD,
        keys: str | list[str] | None = KEYS_FIELD,
        direction: Literal["up", "down"] | None = DIRECTION_FIELD,
        step: int = STEP_FIELD,
        element_info: str | None = ELEMENT_INFO_FIELD,
    ) -> list[ContentBlock]:
        """Execute a GLM PC action.

        GLM PC Action Space:
        - left_click(start_box='[x,y]', element_info=''): Left mouse click
        - right_click(start_box='[x,y]', element_info=''): Right mouse click
        - middle_click(start_box='[x,y]', element_info=''): Middle mouse click
        - hover(start_box='[x,y]', element_info=''): Move mouse without clicking
        - left_double_click(start_box='[x,y]', element_info=''): Double left click
        - left_drag(start_box='[x,y]', end_box='[x,y]', element_info=''): Drag
        - key(keys=''): Press key(s), e.g. 'ctrl+c', 'alt+tab'
        - type(content=''): Type text content
        - scroll(start_box='[x,y]', direction='', step=5, element_info=''): Scroll
        - WAIT(): Wait 5 seconds
        - DONE(): Task completed successfully
        - FAIL(): Task cannot be completed

        Coordinates are 0-999 normalized, automatically scaled to screen pixels.
        """
        logger.info("GLMComputerTool action: %s (start_box=%s)", action, start_box)

        # Parse boxes to coordinates
        start_coords = self._parse_box(start_box)
        end_coords = self._parse_box(end_box)

        # Scale coordinates
        screen_x: int | None = None
        screen_y: int | None = None
        screen_end_x: int | None = None
        screen_end_y: int | None = None

        if start_coords:
            screen_x = self._scale_coord(start_coords[0], is_x=True)
            screen_y = self._scale_coord(start_coords[1], is_x=False)
            logger.debug(
                "Scaled start: [%s,%s] -> (%s,%s)",
                start_coords[0],
                start_coords[1],
                screen_x,
                screen_y,
            )

        if end_coords:
            screen_end_x = self._scale_coord(end_coords[0], is_x=True)
            screen_end_y = self._scale_coord(end_coords[1], is_x=False)

        result: ContentResult | None = None

        # Click actions
        if action in ("left_click", "click"):
            if screen_x is None or screen_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for left_click")
                )
            result = await self.executor.click(x=screen_x, y=screen_y, button="left")

        elif action == "right_click":
            if screen_x is None or screen_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for right_click")
                )
            result = await self.executor.click(x=screen_x, y=screen_y, button="right")

        elif action == "middle_click":
            if screen_x is None or screen_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for middle_click")
                )
            result = await self.executor.click(x=screen_x, y=screen_y, button="middle")

        elif action == "hover":
            if screen_x is None or screen_y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x, y required for hover"))
            result = await self.executor.move(x=screen_x, y=screen_y)

        elif action == "left_double_click":
            if screen_x is None or screen_y is None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS, message="start_box required for left_double_click"
                    )
                )
            result = await self.executor.click(x=screen_x, y=screen_y, button="left", pattern=[100])

        elif action == "left_drag":
            if screen_x is None or screen_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for left_drag")
                )
            if screen_end_x is None or screen_end_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="end_box required for left_drag")
                )
            result = await self.executor.drag(
                path=[(screen_x, screen_y), (screen_end_x, screen_end_y)]
            )

        # Keyboard actions
        elif action == "key":
            key_list = self._parse_keys(keys)
            if not key_list:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="keys required for key action")
                )
            result = await self.executor.press(keys=key_list)

        elif action == "type":
            if not content:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="content required for type"))
            result = await self.executor.write(text=content, enter_after=False)

        # Scroll action
        elif action == "scroll":
            if not direction:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="direction required for scroll")
                )
            # If no start_box, scroll at center of screen
            if screen_x is None:
                screen_x = self.environment_width // 2
            if screen_y is None:
                screen_y = self.environment_height // 2
            # Convert step count to pixels (each step ~100 pixels)
            scroll_y = step * 100 if direction == "down" else -step * 100
            result = await self.executor.scroll(x=screen_x, y=screen_y, scroll_y=scroll_y)

        # Screenshot action
        elif action == "screenshot":
            screenshot = await self.executor.screenshot()
            if screenshot:
                if self.rescale_images:
                    screenshot = await self._rescale_screenshot(screenshot)
                result = ContentResult(base64_image=screenshot)
            else:
                result = ContentResult(error="Failed to take screenshot")
            return result.to_content_blocks()

        # Control actions
        elif action == "WAIT":
            result = await self.executor.wait(time=5000)

        elif action == "DONE":
            screenshot = await self.executor.screenshot()
            if screenshot and self.rescale_images:
                screenshot = await self._rescale_screenshot(screenshot)
            return ContentResult(
                output="Task completed successfully", base64_image=screenshot
            ).to_content_blocks()

        elif action == "FAIL":
            screenshot = await self.executor.screenshot()
            if screenshot and self.rescale_images:
                screenshot = await self._rescale_screenshot(screenshot)
            return ContentResult(
                error="Task failed or is infeasible", base64_image=screenshot
            ).to_content_blocks()

        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

        # Auto-screenshot for interactive actions
        interactive_actions = {
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
        }
        if action in interactive_actions:
            if result is None or (isinstance(result, ContentResult) and not result.base64_image):
                screenshot = await self.executor.screenshot()
                if screenshot:
                    if self.rescale_images:
                        screenshot = await self._rescale_screenshot(screenshot)
                    if result is None:
                        result = ContentResult(base64_image=screenshot)
                    else:
                        result = ContentResult(
                            output=result.output, error=result.error, base64_image=screenshot
                        )

        if result is None:
            result = ContentResult(output="Action completed")

        return result.to_content_blocks()

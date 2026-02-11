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

Works directly with OpenAIChatAgent:

    from hud.agents import OpenAIChatAgent
    from hud.tools.computer.glm import GLM_CUA_INSTRUCTIONS

    agent = OpenAIChatAgent.create(
        model="glm-4.5v",
        system_prompt=GLM_CUA_INSTRUCTIONS,
    )
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock
from pydantic import Field

from hud.tools.types import ContentResult

from .hud import HudComputerTool
from .settings import computer_settings

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor
    from hud.tools.native_types import NativeToolSpecs

logger = logging.getLogger(__name__)

# GLM uses normalized 0-999 coordinate space
GLM_COORDINATE_SPACE = 999

# GLM CUA system instructions
GLM_CUA_INSTRUCTIONS = """\
You are a GUI Agent, and your primary task is to respond accurately to user \
requests or questions. In addition to directly answering the user's queries, \
you can also use tools or perform GUI operations directly until you fulfill \
the user's request or provide a correct answer. You should carefully read and \
understand the images and questions provided by the user, and engage in \
thinking and reflection when appropriate. The coordinates involved are all \
represented in thousandths (0-999).

# Task Platform
Ubuntu

# Tool Call Format
IMPORTANT: Always use valid JSON for function arguments. Do NOT use XML tags.
Correct: {"action": "left_click", "start_box": "[500, 300]"}
Wrong: {"action": "left_click<arg_key>start_box</arg_key><arg_value>[500, 300]"}

# Some Additional Notes
- Complete tasks autonomously without asking for confirmation.
- If a task cannot be completed, use FAIL().
- Coordinates use 0-999 scale (thousandths of screen dimensions).\
""".strip()

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
    "DONE",  # no params - task completed (no-op)
    "FAIL",  # no params - task failed (no-op)
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

    Handles all model-specific quirks (XML arg fixing, DONE/FAIL no-ops)
    so it works directly with OpenAIChatAgent -- no custom agent class needed.

    Usage:
        from hud.agents import OpenAIChatAgent
        from hud.tools.computer.glm import GLM_CUA_INSTRUCTIONS

        agent = OpenAIChatAgent.create(
            model="glm-4.5v",
            system_prompt=GLM_CUA_INSTRUCTIONS,
        )
    """

    native_specs: ClassVar[NativeToolSpecs] = {}

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
            # Unwrap nested list: [[x, y]] -> [x, y]
            if len(box) == 1 and isinstance(box[0], list):
                box = box[0]
            if len(box) >= 2:
                try:
                    return (int(box[0]), int(box[1]))
                except (TypeError, ValueError):
                    return None

        return None

    def _scale_coord(self, coord: int, is_x: bool = True) -> int:
        """Scale coordinate from GLM's 0-999 space to actual screen pixels.

        Maps [0, 999] -> [0, dimension-1] so the max coordinate lands on the
        last valid pixel index rather than going out of bounds.
        """
        if is_x:
            return int(coord * (self.environment_width - 1) / GLM_COORDINATE_SPACE)
        else:
            return int(coord * (self.environment_height - 1) / GLM_COORDINATE_SPACE)

    def _parse_keys(self, keys: str | list[str] | None) -> list[str]:
        """Parse key input to list of keys."""
        if not keys:
            return []
        if isinstance(keys, list):
            return [k.strip().lower() for k in keys]
        # Handle 'ctrl+c' format
        return [k.strip().lower() for k in keys.split("+")]

    @staticmethod
    def _fix_xml_args(args: dict[str, Any]) -> dict[str, Any]:
        """Fix XML-style arguments that GLM models sometimes output.

        Handles cases like:
        {"action": "left_click\\n<arg_key>start_box</arg_key>\\n<arg_value>[114, 167]"}

        Converts to:
        {"action": "left_click", "start_box": "[114, 167]"}
        """
        fixed: dict[str, Any] = {}

        for key, value in args.items():
            if not isinstance(value, str):
                fixed[key] = value
                continue

            # No XML tags -- pass through
            if "<arg_key>" not in value:
                fixed[key] = value
                continue

            # Extract the plain-text value before the first XML tag
            # Example: "left_click\n<arg_key>..." -> "left_click"
            main_value = value.split("<arg_key>")[0].strip()
            if main_value:
                fixed[key] = main_value

            # Parse XML-style key-value pairs
            pattern = r"<arg_key>(\w+)</arg_key>\s*<arg_value>([^\"<]+)"
            matches = re.findall(pattern, value)

            for arg_name, arg_val in matches:
                arg_name = arg_name.strip()
                arg_val = arg_val.strip()
                if arg_name and arg_val:
                    fixed[arg_name] = arg_val

            # Preserve original key if no plain text prefix and no XML matches
            if not main_value and not matches:
                fixed[key] = value

            logger.warning("Fixed XML args: %s -> %s", args, fixed)

        return fixed

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

        Handles all GLM model quirks:
        - Fixes XML-style arguments that GLM sometimes outputs
        - Treats DONE/FAIL as no-ops (raises McpError)
        - Parses start_box/end_box in multiple formats
        - Scales 0-999 normalized coordinates to screen pixels

        GLM PC Action Space:
        - left_click(start_box='[x,y]'): Left mouse click
        - right_click(start_box='[x,y]'): Right mouse click
        - middle_click(start_box='[x,y]'): Middle mouse click
        - hover(start_box='[x,y]'): Move mouse without clicking
        - left_double_click(start_box='[x,y]'): Double left click
        - left_drag(start_box='[x,y]', end_box='[x,y]'): Drag
        - key(keys=''): Press key(s), e.g. 'ctrl+c', 'alt+tab'
        - type(content=''): Type text content
        - scroll(start_box='[x,y]', direction='', step=5): Scroll
        - screenshot(): Take screenshot
        - WAIT(): Wait 5 seconds
        - DONE(): Task completed (no-op)
        - FAIL(): Task failed (no-op)

        Coordinates are 0-999 normalized, automatically scaled to screen pixels.
        """
        # --- Fix XML-mangled arguments ---
        # GLM sometimes embeds XML in string args:
        #   "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"
        # Only the action string can contain embedded XML key-value pairs.
        if isinstance(action, str) and "<arg_key>" in action:
            fixed = self._fix_xml_args({"action": action})
            action = fixed.pop("action", action)  # type: ignore[assignment]
            # Apply any extracted parameters (start_box, content, etc.)
            start_box = fixed.pop("start_box", start_box)
            end_box = fixed.pop("end_box", end_box)
            content = fixed.pop("content", content)
            keys = fixed.pop("keys", keys)
            direction = fixed.pop("direction", direction)  # type: ignore[assignment]

        # --- Handle DONE/FAIL as no-ops (like Qwen's terminate/answer) ---
        if action == "DONE":
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="DONE action is not supported for computer control. This is a no-op.",
                )
            )

        if action == "FAIL":
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="FAIL action is not supported for computer control. This is a no-op.",
                )
            )

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
        if action in interactive_actions and (
            result is None or (isinstance(result, ContentResult) and not result.base64_image)
        ):
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

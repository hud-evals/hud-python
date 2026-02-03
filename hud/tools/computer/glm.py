# flake8: noqa: B008
"""GLM computer tool for interacting with the computer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ContentBlock, TextContent
from pydantic import Field

from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult
from hud.types import AgentType

from .hud import HudComputerTool
from .settings import computer_settings

if TYPE_CHECKING:
    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


class GLMComputerTool(HudComputerTool):
    """
    GLM computer tool for z-ai/glm4.5v.

    Uses normalized coordinates (0-999) and start_box='[x,y]' format for desktop actions.
    """

    name: str = "glm_computer"
    api_type: str = "glm4_5v_computer"
    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.OPENAI_COMPATIBLE: NativeToolSpec(
            api_type="function",
            api_name="glm_computer",
            role="computer",
        ),
        AgentType.GLM_CUA: NativeToolSpec(
            api_type="glm_computer",
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
        name: str | None = "glm_computer",
        title: str | None = "GLM Computer Tool",
        description: str | None = "Control computer with mouse, keyboard, and screenshots",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            name=name,
            title=title,
            description=description,
            **kwargs,
        )

    def to_params(self) -> dict:
        """Convert to GLM tool parameters."""
        return {
            "type": self.api_type,
            "name": self.name,
            "display_width_px": self.width,
            "display_height_px": self.height,
        }

    def _parse_box(self, box: str | list[int] | None) -> tuple[int | None, int | None]:
        """Parse start_box/end_box from '[x,y]' or [x,y] to screen pixels."""
        if box is None:
            return None, None

        if isinstance(box, str):
            cleaned = box.strip("[]")
            parts = cleaned.split(",")
            if len(parts) < 2:
                return None, None
            try:
                norm_x, norm_y = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None, None
        elif isinstance(box, list) and len(box) >= 2:
            norm_x, norm_y = int(box[0]), int(box[1])
        else:
            return None, None

        # Convert normalized (0-999) to screen coordinates
        screen_x = int(norm_x * self.environment_width / 999)
        screen_y = int(norm_y * self.environment_height / 999)
        return screen_x, screen_y

    def _parse_keys(self, keys: str) -> list[str]:
        """Parse 'ctrl+c' format to ['ctrl', 'c']."""
        if not keys:
            return []
        return [k.strip().lower() for k in keys.split("+")]

    def _parse_function_style_action(
        self, action: str
    ) -> tuple[str, dict[str, Any]]:
        """Parse function-style action like 'left_click(start_box='[513,438]')'.
        
        Returns (action_name, extracted_args).
        """
        import re
        
        # Check if action contains function call syntax
        func_match = re.match(r"(\w+)\((.*)\)", action)
        if not func_match:
            return action, {}
        
        action_name = func_match.group(1)
        args_str = func_match.group(2)
        
        extracted: dict[str, Any] = {}
        
        # Extract quoted arguments: key='value' or key="value"
        quoted_pattern = r"(\w+)=['\"]([^'\"]*)['\"]"
        for match in re.finditer(quoted_pattern, args_str):
            extracted[match.group(1)] = match.group(2)
        
        # Extract unquoted numeric arguments: key=123
        unquoted_pattern = r"(\w+)=(\d+)(?!['\"])"
        for match in re.finditer(unquoted_pattern, args_str):
            if match.group(1) not in extracted:
                extracted[match.group(1)] = int(match.group(2))
        
        return action_name, extracted

    async def __call__(
        self,
        action: str = Field(..., description="The action to perform"),
        start_box: str | list[int] | None = Field(None, description="Coordinates [x,y] (0-999)"),
        end_box: str | list[int] | None = Field(None, description="End coordinates for drag"),
        element_info: str | None = Field(None, description="UI element description"),
        keys: str | None = Field(None, description="Key combination (e.g., 'ctrl+c')"),
        content: str | None = Field(None, description="Text to type"),
        direction: Literal["up", "down"] | None = Field(None, description="Scroll direction"),
        step: int = Field(5, description="Scroll steps"),
    ) -> list[ContentBlock]:
        """Execute a GLM desktop computer action."""
        # Handle GLM 4.6V function-style action: "left_click(start_box='[513,438]')"
        if "(" in action:
            parsed_action, extracted_args = self._parse_function_style_action(action)
            action = parsed_action
            # Override None arguments with extracted values
            if start_box is None and "start_box" in extracted_args:
                start_box = extracted_args["start_box"]
            if end_box is None and "end_box" in extracted_args:
                end_box = extracted_args["end_box"]
            if keys is None and "keys" in extracted_args:
                keys = extracted_args["keys"]
            if content is None and "content" in extracted_args:
                content = extracted_args["content"]
            if direction is None and "direction" in extracted_args:
                direction = extracted_args["direction"]
            if "step" in extracted_args:
                step = extracted_args["step"]
        
        logger.info("GLMComputerTool action: %s", action)

        result: ContentResult | None = None

        # Click actions
        if action in ("left_click", "right_click", "middle_click"):
            x, y = self._parse_box(start_box)
            if x is None or y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message=f"start_box required for {action}")
                )
            button = cast("Literal['left', 'right', 'middle']", action.replace("_click", ""))
            result = await self.executor.click(x=x, y=y, button=button)

        elif action == "left_double_click":
            x, y = self._parse_box(start_box)
            if x is None or y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for double_click")
                )
            result = await self.executor.click(x=x, y=y, button="left", pattern=[100])

        elif action == "hover":
            x, y = self._parse_box(start_box)
            if x is None or y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for hover")
                )
            result = await self.executor.move(x=x, y=y)

        elif action == "left_drag":
            start_x, start_y = self._parse_box(start_box)
            end_x, end_y = self._parse_box(end_box)
            if start_x is None or start_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for left_drag")
                )
            if end_x is None or end_y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="end_box required for left_drag")
                )
            result = await self.executor.drag(path=[(start_x, start_y), (end_x, end_y)])

        # Keyboard actions
        elif action == "key":
            if not keys:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="keys required for key action")
                )
            result = await self.executor.press(keys=self._parse_keys(keys))

        elif action == "type":
            if content is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="content required for type action")
                )
            result = await self.executor.write(text=content, enter_after=False)

        # Scroll action
        elif action == "scroll":
            x, y = self._parse_box(start_box)
            if x is None or y is None:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="start_box required for scroll")
                )
            if not direction:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message="direction required for scroll")
                )
            scroll_y = step * 100 if direction == "down" else -step * 100
            result = await self.executor.scroll(x=x, y=y, scroll_y=scroll_y)

        # Control actions
        elif action == "WAIT":
            result = await self.executor.wait(time=5000)

        elif action == "DONE":
            return [TextContent(text="Task completed successfully.", type="text")]

        elif action == "FAIL":
            return [TextContent(text="Task cannot be completed.", type="text")]

        elif action == "screenshot":
            screenshot = await self.executor.screenshot()
            if screenshot:
                result = ContentResult(base64_image=screenshot)
            else:
                result = ContentResult(error="Failed to take screenshot")

        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unknown action: {action}"))

        # Rescale screenshot if present
        if isinstance(result, ContentResult) and result.base64_image and self.rescale_images:
            result.base64_image = await self._rescale_screenshot(result.base64_image)

        # Auto-screenshot for interactive actions
        interactive = {
            "left_click", "right_click", "middle_click", "left_double_click",
            "hover", "left_drag", "key", "type", "scroll",
        }
        if action in interactive and isinstance(result, ContentResult) and not result.base64_image:
            screenshot = await self.executor.screenshot()
            if screenshot:
                screenshot = await self._rescale_screenshot(screenshot)
                result = ContentResult(output="", error=result.error, base64_image=screenshot)

        if result is None:
            result = ContentResult(output="Action completed")

        return result.to_content_blocks()

"""OpenAI with memory/history tracking for remote browser environment."""
import os, base64
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field
from hud.tools import OpenAIComputerTool
from mcp.types import ContentBlock, ImageContent, TextContent
from hud.tools.computer.settings import computer_settings
from _collections_abc import Callable, Awaitable
from hud.tools.executors import BaseExecutor

logger = logging.getLogger(__name__)

class OpenAIComputerToolWithRecord(OpenAIComputerTool):
    """OpenAI Computer Use tool

    Args:
        OpenAIComputerTool (_type_): _description_
    """

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
            width: Target width for rescaling (default: 1024 for OpenAI)
            height: Target height for rescaling (default: 768 for OpenAI)
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
            name=name or "openai_computer",
            title=title or "OpenAI Computer Tool",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            **kwargs,
        )
        self.add_callback("on_screenshot_action", self._on_screenshot_action)
        self.add_callback("on_recorded_action", self._on_recorded_action)
        
    async def _on_screenshot_action(self, **kwargs) -> None:
        """Callback function to take and save screenshots to /screenshot directory"""
        try:
            # Check if executor is available and properly initialized
            if not hasattr(self, 'executor') or self.executor is None:
                logger.debug("Executor not yet initialized, skipping screenshot")
                return

            # Additional check for executor readiness
            if not hasattr(self.executor, 'screenshot'):
                logger.debug("Executor screenshot method not available, skipping screenshot")
                return

            screenshot_base64 = await self.executor.screenshot()
            if screenshot_base64:


                # Create screenshot directory if it doesn't exist
                screenshot_dir = "/screenshot"
                os.makedirs(screenshot_dir, exist_ok=True)

                # Generate timestamp-based filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(screenshot_dir, filename)

                # Decode base64 and save to file
                image_data = base64.b64decode(screenshot_base64)
                with open(filepath, 'wb') as f:
                    f.write(image_data)

                logger.info(f"Saved screenshot to {filepath}")

        except Exception as e:
            logger.debug(f"Screenshot callback failed (this is normal during initialization): {e}")
            # Don't log as error since this is expected during initialization

    async def _on_recorded_action(self, type=None, x=None, y=None, text=None,
                                  path=None, scroll_x=None, scroll_y=None, **_):
        """Record action in unified representation format

        Creates unified action representations like:
        - <coordinate=[123, 456]> -> CLICK
        - <coordinate=[123, 456]> -> TYPE hello@example.com
        - <start=[100, 200], end=[300, 400]> -> DRAG
        """
        if not type:
            return

        try:
            # Create unified action representation
            action_repr = self._to_action_repr(
                type, x, y, text, path, scroll_x, scroll_y
            )

            # Dump to file
            action_history_dir = "/action_history"
            os.makedirs(action_history_dir, exist_ok=True)
            action_file = os.path.join(action_history_dir, "action_history.txt")

            with open(action_file, "a", encoding="utf-8") as f:
                f.write(f"{action_repr}\n")

            logger.info(f"Recorded action: {action_repr}")

        except Exception as e:
            logger.warning(f"Failed to record action: {e}")

    def _to_action_repr(self, type, x=None, y=None, text=None,
                        path=None, scroll_x=None, scroll_y=None):
        """Create unified action representation following AgentRewardBench format

        Format examples:
        - <coordinate=[123, 456]> -> CLICK
        - <coordinate=[123, 456]> -> TYPE hello@example.com
        - <start=[100, 200], end=[300, 400]> -> DRAG
        - <coordinate=[123, 456], direction=up, amount=3> -> SCROLL
        """

        # Normalize action names to uppercase
        action_name = type.upper().replace("_", "")
        if action_name == "DOUBLECLICK":
            action_name = "DOUBLECLICK"
        elif action_name == "KEYPRESS":
            action_name = "KEY"

        # Build element attributes part
        attributes = []

        if x is not None and y is not None:
            attributes.append(f"coordinate=[{x}, {y}]")

        if path and action_name == "DRAG":
            if len(path) >= 2:
                start = path[0]
                end = path[-1]
                attributes.append(f"start=[{start['x']}, {start['y']}]")
                attributes.append(f"end=[{end['x']}, {end['y']}]")

        if scroll_x is not None or scroll_y is not None:
            if scroll_y and scroll_y > 0:
                attributes.append("direction=down")
                attributes.append(f"amount={abs(scroll_y)}")
            elif scroll_y and scroll_y < 0:
                attributes.append("direction=up")
                attributes.append(f"amount={abs(scroll_y)}")
            elif scroll_x and scroll_x > 0:
                attributes.append("direction=right")
                attributes.append(f"amount={abs(scroll_x)}")
            elif scroll_x and scroll_x < 0:
                attributes.append("direction=left")
                attributes.append(f"amount={abs(scroll_x)}")

        # Create element part
        element_part = f"<{', '.join(attributes)}>" if attributes else "<>"

        # Create action part
        if text and action_name in ["TYPE", "KEY"]:
            action_part = f"{action_name} {text}"
        else:
            action_part = action_name

        return f"{element_part} -> {action_part}"

    async def __call__(
        self,
        type: str = Field(..., description="The action type to perform"),
        # Coordinate parameters
        x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
        y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
        # Button parameter
        button: str | None = Field(
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
        keys: list[str] | None = Field(None, description="Keys to press"),
        # Drag parameter
        path: list[dict[str, int]] | None = Field(
            None, description="Path for drag actions as list of {x, y} dicts"
        ),
        # Custom action parameter
        action: str | None = Field(None, description="Custom action name"),
    ) -> list[ContentBlock]:
        """Overriding OpenAIComputerTool.__call__()"""
        result = await super().__call__(
            type=type, x=x, y=y, button=button, 
            text=text, scroll_x=scroll_x, scroll_y=scroll_y,
            ms=ms, keys=keys, path=path, action=action
        )
        screenshot_action_type = {
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
        logger.info(f"debug action type: {type}")
        if type in screenshot_action_type:
            if hasattr(self, '_trigger_callbacks'):
                await self._trigger_callbacks("on_screenshot_action")
            else:
                logger.warning("_trigger_callbacks method not available")

        recorded_actions = {
            "click",
            "double_click",
            "type",
            "keypress",
            "scroll",
            "drag",
        }
        if type in recorded_actions:
            logger.info("debug record actions")
            await self._trigger_callbacks(
                "on_recorded_action",
                type=type,
                x=x,
                y=y,
                text=text,
                path=path,
                scroll_x=scroll_x,
                scroll_y=scroll_y
            )

        return result

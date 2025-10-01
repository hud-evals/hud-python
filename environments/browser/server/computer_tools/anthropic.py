"""PlaywrightTool with memory/history tracking for remote browser environment."""

import logging, os, base64
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import Field
from hud.tools import AnthropicComputerTool
from mcp.types import ContentBlock, ImageContent, TextContent
from hud.tools.computer.settings import computer_settings
from _collections_abc import Callable, Awaitable
from hud.tools.executors import BaseExecutor

logger = logging.getLogger(__name__)

class AnthropicComputerToolWithRecord(AnthropicComputerTool):
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
    ):
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
        self.add_callback("on_screenshot_action", self._on_screenshot_action)
        self.add_callback("on_recorded_action", self._on_recorded_action)
        
    async def _on_screenshot_action(self, **_) -> None:
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

    async def _on_recorded_action(self, action=None, coordinate=None, text=None,
                                  start_coordinate=None, scroll_direction=None,
                                  scroll_amount=None, **_):
        """Record action in unified representation format

        Creates unified action representations like:
        - <coordinate=[123, 456]> -> CLICK
        - <coordinate=[123, 456]> -> TYPE hello@example.com
        - <start=[100, 200], end=[300, 400]> -> DRAG
        """
        if not action:
            return

        try:
            # Create unified action representation
            action_repr = self._to_action_repr(
                action, coordinate, text, start_coordinate, scroll_direction, scroll_amount
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
        
        result = await super().__call__(action=action, coordinate=coordinate, text=text, 
                                  start_coordinate=start_coordinate, scroll_direction=scroll_direction,
                                  scroll_amount=scroll_amount, duration=duration, 
                                  take_screenshot_on_click=take_screenshot_on_click
                                )
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
        ):
            await self._trigger_callbacks("on_screenshot_action")
        recorded_actions = {
            "left_click",
            "click",
            "double_click",
            "triple_click",
            "right_click",
            "middle_click",
            "type",
            "key",
            "scroll",
            "left_click_drag",
            "drag",
        }
        if (action in recorded_actions):
            await self._trigger_callbacks("on_recorded_action",
                                         action=action,
                                         coordinate=coordinate,
                                         text=text,
                                         start_coordinate=start_coordinate,
                                         scroll_direction=scroll_direction,
                                         scroll_amount=scroll_amount)
        return result


    def _to_action_repr(self, action, coordinate=None, text=None,
                        start_coordinate=None, scroll_direction=None,
                        scroll_amount=None):
        """Create unified action representation following AgentRewardBench format

        Format examples:
        - <coordinate=[123, 456]> -> CLICK
        - <coordinate=[123, 456]> -> TYPE hello@example.com
        - <start=[100, 200], end=[300, 400]> -> DRAG
        - <coordinate=[123, 456], direction=up, amount=3> -> SCROLL
        """

        # Normalize action names to uppercase
        action_name = action.upper().replace("LEFT_", "").replace("_", "")
        if action_name == "LEFTCLICK":
            action_name = "CLICK"
        elif action_name == "LEFTCLICKDRAG":
            action_name = "DRAG"

        # Build element attributes part
        attributes = []

        if coordinate:
            attributes.append(f"coordinate={coordinate}")

        if start_coordinate and action_name == "DRAG":
            attributes.append(f"start={start_coordinate}")
            if coordinate:
                attributes.append(f"end={coordinate}")

        if scroll_direction:
            attributes.append(f"direction={scroll_direction}")

        if scroll_amount:
            attributes.append(f"amount={scroll_amount}")

        # Create element part
        element_part = f"<{', '.join(attributes)}>" if attributes else "<>"

        # Create action part
        if text and action_name in ["TYPE", "KEY"]:
            action_part = f"{action_name} {text}"
        else:
            action_part = action_name

        return f"{element_part} -> {action_part}"

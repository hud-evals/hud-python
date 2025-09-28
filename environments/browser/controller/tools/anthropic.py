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
            description=description ,
            **kwargs,
        )
        self.screenshot_history = []
        self._callbacks: dict[
            str,
            list[Callable[..., Awaitable[Any]]],
        ] = {}  # DELETE after hud-python version bump
        
        # Try to add callback if available
        if hasattr(self, 'add_callback'):
            logger.info("Callback system available, adding screenshot callback")
            self.add_callback("on_screenshot_action", self._on_screenshot_action)
        else:
            logger.error("Callback system not available - missing add_callback method")
        
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
        return result

    # Delete After hud-python version bump
    def add_callback(self, event_type: str, callback: Callable[..., Awaitable[Any]]):
        """Register a callback function for specific event
        
        Args:
            event_type: (Required) Specific event name to trigger callback
                        e.g. "after_click", "before_navigate"
            callback: (Required) Async function to call. Must be defined by `async def f(...)`
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable[..., Awaitable[Any]]):
        """Remove a registered callback
        Args:
            event_type: (Required) Specific event name to trigger callback
                        e.g. "after_click", "before_navigate"
            callback: (Required) Function to remove from callback list.
        """
        if (event_type in self._callbacks) and (callback in self._callbacks[event_type]):
            self._callbacks[event_type].remove(callback)
    
    async def _trigger_callbacks(self, event_type: str, **kwargs):
        """Trigger all registered callback functions of an event type"""
        callback_list = self._callbacks.get(event_type, [])
        for callback in callback_list:
            try:
                await callback(**kwargs)
            except Exception as e:
                logger.warning(f"Callback failed for {event_type}: {e}")

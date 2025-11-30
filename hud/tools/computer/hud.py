# flake8: noqa: B008
from __future__ import annotations

import logging
import platform
from typing import Literal

from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS, ContentBlock, TextContent
from pydantic import Field

from hud.tools.base import BaseTool
from hud.tools.executors.base import BaseExecutor
from hud.tools.executors.pyautogui import PyAutoGUIExecutor
from hud.tools.executors.xdo import XDOExecutor
from hud.tools.types import ContentResult, ToolError

from .settings import computer_settings

logger = logging.getLogger(__name__)


class HudComputerTool(BaseTool):
    """
    A tool that allows the agent to control the computer.
    """

    def __init__(
        self,
        # Define within environment based on platform
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        # Overrides for what dimensions the agent thinks it operates in
        # Define per subclass (e.g., Anthropic, OpenAI)
        width: int | None = computer_settings.HUD_COMPUTER_WIDTH,
        height: int | None = computer_settings.HUD_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.HUD_RESCALE_IMAGES,
        # What the agent sees as the tool's name, title, and description
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Initialize the HUD computer tool.

        Args:
            executor: Executor to use for the tool
            platform_type: Which executor to use if executor not provided:
                - "auto": Automatically detect based on platform
                - "xdo": Use XDOExecutor (Linux/X11 only)
                - "pyautogui": Use PyAutoGUIExecutor (cross-platform)
            display_num: X display number
            width: Target width for rescaling (None = use environment width)
            height: Target height for rescaling (None = use environment height)
            rescale_images: If True, rescale screenshots. If False, only rescale action coordinates
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        # This is the width and height the agent thinks it operates in
        # By default, use subclass's width and height
        # If specifically set to None, use environment width and height
        self.width = width or computer_settings.DISPLAY_WIDTH
        self.height = height or computer_settings.DISPLAY_HEIGHT

        # Build metadata with resolution info
        meta = {
            "resolution": {
                "width": self.width,
                "height": self.height,
            }
        }

        # Initialize base tool with executor as env
        super().__init__(
            env=executor,
            name=name or "computer",
            title=title or "Computer Control",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            meta=meta,
        )

        # This is the static width and height of the environment screen
        # And the width and height of the screenshots taken by the tool
        self.environment_width = computer_settings.DISPLAY_WIDTH
        self.environment_height = computer_settings.DISPLAY_HEIGHT

        # Some APIs rescale screenshots automatically to the agent's width and height, some don't
        # Defined per subclass (e.g., Anthropic, OpenAI)
        # In case you need your agent to receive pre-formatted screenshots, set env variable True
        self.rescale_images = rescale_images

        logger.debug(
            "Agent Screen Width: %s, Agent Screen Height: %s",
            self.width,
            self.height,
            "Environment Screen Width: %s, Environment Screen Height: %s",
            self.environment_width,
            self.environment_height,
        )

        # Calculate scaling factors from base screen size to target size
        self.scale_x = self.width / self.environment_width
        self.scale_y = self.height / self.environment_height

        # Check if we need to scale
        self.needs_scaling = min(self.scale_x, self.scale_y) != 1.0

        # Use environment settings for display number
        self.display_num = display_num or computer_settings.DISPLAY_NUM

        logger.debug("Display number: %s", self.display_num)

        # If no executor provided, create one based on platform
        if self.env is None:
            self._choose_executor(platform_type, self.display_num)

    @property
    def executor(self) -> BaseExecutor:
        """Get the executor (alias for context)."""
        return self.env

    @executor.setter
    def executor(self, value: BaseExecutor) -> None:
        """Set the executor (alias for context)."""
        self.env = value

    def _choose_executor(
        self,
        platform_type: Literal["auto", "xdo", "pyautogui"],
        display_num: int | None,
    ) -> None:
        """Choose executor based on platform_type."""
        # Choose executor based on platform_type
        if platform_type == "auto":
            # Auto-detect based on platform
            system = platform.system().lower()
            if system == "linux":
                # Try XDO first on Linux
                if XDOExecutor.is_available():
                    self.executor = XDOExecutor(display_num=display_num)
                    logger.info("Using XDOExecutor")
                elif PyAutoGUIExecutor.is_available():
                    self.executor = PyAutoGUIExecutor(display_num=display_num)
                    logger.info("Using PyAutoGUIExecutor")
                else:
                    self.executor = BaseExecutor(display_num=display_num)
                    logger.info("No display available, using BaseExecutor (simulation mode)")
            else:
                # Windows/macOS - try PyAutoGUI
                if PyAutoGUIExecutor.is_available():
                    self.executor = PyAutoGUIExecutor(display_num=display_num)
                    logger.info("Using PyAutoGUIExecutor")
                else:
                    self.executor = BaseExecutor(display_num=display_num)
                    logger.info("PyAutoGUI not available, using BaseExecutor (simulation mode)")

        elif platform_type == "xdo":
            if XDOExecutor.is_available():
                self.executor = XDOExecutor(display_num=display_num)
                logger.info("Using XDOExecutor")
            else:
                self.executor = BaseExecutor(display_num=display_num)
                logger.warning("XDO not available, using BaseExecutor (simulation mode)")

        elif platform_type == "pyautogui":
            if PyAutoGUIExecutor.is_available():
                self.executor = PyAutoGUIExecutor(display_num=display_num)
                logger.info("Using PyAutoGUIExecutor")
            else:
                self.executor = BaseExecutor(display_num=display_num)
                logger.warning("PyAutoGUI not available, using BaseExecutor (simulation mode)")
        else:
            raise ValueError(f"Invalid platform_type: {platform_type}")

    def _scale_coordinates(self, x: int | None, y: int | None) -> tuple[int | None, int | None]:
        """Scale coordinates from target space to screen space."""
        if x is not None and self.scale_x != 1.0:
            x = int(x / self.scale_x)
        if y is not None and self.scale_y != 1.0:
            y = int(y / self.scale_y)

        return x, y

    def _scale_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Scale a path from target space to screen space."""
        scaled_path = []
        for x, y in path:
            scaled_x, scaled_y = self._scale_coordinates(x, y)
            if scaled_x is not None and scaled_y is not None:
                scaled_path.append((scaled_x, scaled_y))

        return scaled_path

    async def _rescale_screenshot(self, screenshot_base64: str) -> str:
        """Rescale a screenshot if rescale_images is True."""
        if not self.rescale_images or not self.needs_scaling:
            return screenshot_base64

        try:
            import base64
            from io import BytesIO

            from PIL import Image  # type: ignore[import-not-found]

            # Decode base64 to image
            image_data = base64.b64decode(screenshot_base64)
            image = Image.open(BytesIO(image_data))

            logger.info(
                "Resizing screenshot from %s x %s to %s x %s",
                image.width,
                image.height,
                self.width,
                self.height,
            )

            # Resize to exact target dimensions
            resized = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

            # Convert back to base64
            buffer = BytesIO()
            resized.save(buffer, format="PNG")
            resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return resized_base64
        except Exception as e:
            logger.warning("Failed to rescale screenshot: %s", e)
            return screenshot_base64

    async def __call__(
        self,
        action: Literal[
            "click",
            "press",
            "keydown",
            "keyup",
            "write",
            "scroll",
            "move",
            "wait",
            "drag",
            "response",
            "screenshot",
            "position",
            "hold_key",
            "mouse_down",
            "mouse_up",
        ] = Field(..., description="The action name (click, press, write, move, etc.)"),
        # Click parameters
        x: int | None = Field(None, description="X coordinate for click/move/scroll actions"),
        y: int | None = Field(None, description="Y coordinate for click/move/scroll actions"),
        button: Literal["left", "right", "middle", "back", "forward"] | None = Field(
            None, description="Mouse button for click actions"
        ),
        pattern: list[int] | None = Field(
            None, description="Click pattern for multi-clicks (e.g., [100] for double-click)"
        ),
        # Key/Type parameters
        text: str | None = Field(None, description="Text for write/response actions"),
        keys: list[str] | None = Field(None, description="Keys for press/keydown/keyup actions"),
        enter_after: bool | None = Field(None, description="Whether to press Enter after typing"),
        # Scroll parameters
        scroll_x: int | None = Field(
            None, description="Horizontal scroll amount (positive = right)"
        ),
        scroll_y: int | None = Field(None, description="Vertical scroll amount (positive = down)"),
        # Move parameters
        offset_x: int | None = Field(None, description="X offset for relative move"),
        offset_y: int | None = Field(None, description="Y offset for relative move"),
        # Drag parameters
        path: list[tuple[int, int]] | None = Field(
            None, description="Path for drag actions as list of (x, y) coordinates"
        ),
        # Wait parameter
        time: int | None = Field(None, description="Time in milliseconds for wait action"),
        # General parameters
        hold_keys: list[str] | None = Field(None, description="Keys to hold during action"),
        # hold_key specific
        duration: float | None = Field(None, description="Duration in seconds for hold_key action"),
    ) -> list[ContentBlock]:
        """
        Execute a computer control action by name.

        Returns:
            List of MCP content blocks
        """
        logger.info("HudComputerTool executing action: %s", action)

        try:
            # Delegate to executor based on action
            if action == "click":
                # Scale coordinates from client space to screen space
                scaled_x, scaled_y = self._scale_coordinates(x, y)
                result = await self.executor.click(
                    x=scaled_x,
                    y=scaled_y,
                    button=button or "left",
                    pattern=pattern,
                    hold_keys=hold_keys,
                )

            elif action == "press":
                if keys is None:
                    raise ToolError("keys parameter is required for press")
                result = await self.executor.press(keys=keys)

            elif action == "keydown":
                if keys is None:
                    raise ToolError("keys parameter is required for keydown")
                result = await self.executor.keydown(keys=keys)

            elif action == "keyup":
                if keys is None:
                    raise ToolError("keys parameter is required for keyup")
                result = await self.executor.keyup(keys=keys)

            elif action == "write":
                if text is None:
                    raise ToolError("text parameter is required for write")
                result = await self.executor.write(text=text, enter_after=enter_after or False)

            elif action == "scroll":
                # Scale coordinates from client space to screen space
                scaled_x, scaled_y = self._scale_coordinates(x, y)
                result = await self.executor.scroll(
                    x=scaled_x,
                    y=scaled_y,
                    scroll_x=scroll_x,
                    scroll_y=scroll_y,
                    hold_keys=hold_keys,
                )

            elif action == "move":
                # Scale coordinates from client space to screen space
                scaled_x, scaled_y = self._scale_coordinates(x, y)
                scaled_offset_x, scaled_offset_y = self._scale_coordinates(offset_x, offset_y)
                result = await self.executor.move(
                    x=scaled_x, y=scaled_y, offset_x=scaled_offset_x, offset_y=scaled_offset_y
                )

            elif action == "wait":
                if time is None:
                    raise ToolError("time parameter is required for wait")
                result = await self.executor.wait(time=time)

            elif action == "drag":
                if path is None:
                    raise ToolError("path parameter is required for drag")
                # Scale path from client space to screen space
                scaled_path = self._scale_path(path)
                result = await self.executor.drag(
                    path=scaled_path, pattern=pattern, hold_keys=hold_keys
                )

            elif action == "response":
                if text is None:
                    raise ToolError("text parameter is required for response")
                return [TextContent(text=text, type="text")]

            elif action == "screenshot":
                screenshot = await self.executor.screenshot()
                if screenshot:
                    # Rescale screenshot if requested
                    screenshot = await self._rescale_screenshot(screenshot)
                    result = ContentResult(base64_image=screenshot)
                else:
                    result = ContentResult(error="Failed to take screenshot")

            elif action == "position":
                result = await self.executor.position()

            elif action == "hold_key":
                if text is None:
                    raise ToolError("text parameter is required for hold_key")
                if duration is None:
                    raise ToolError("duration parameter is required for hold_key")
                result = await self.executor.hold_key(key=text, duration=duration)

            elif action == "mouse_down":
                result = await self.executor.mouse_down(button=button or "left")

            elif action == "mouse_up":
                result = await self.executor.mouse_up(button=button or "left")

            else:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

            # Rescale screenshot in result if present
            if isinstance(result, ContentResult) and result.base64_image and self.rescale_images:
                rescaled_image = await self._rescale_screenshot(result.base64_image)
                result.base64_image = rescaled_image

            # Convert result to content blocks
            return result.to_content_blocks()

        except TypeError as e:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Invalid parameters for {action}: {e!s}")
            ) from e

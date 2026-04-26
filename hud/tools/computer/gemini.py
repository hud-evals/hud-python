from __future__ import annotations

import logging
import platform
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
    from collections.abc import Mapping

    from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

SUPPORTED_GEMINI_COMPUTER_USE_MODELS = (
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-3-flash-preview",
)

PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]


def normalize_gemini_computer_use_args(action: str, raw_args: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Gemini Computer Use function-call args to GeminiComputerTool kwargs."""
    normalized_args: dict[str, Any] = {"action": action}

    coord = raw_args.get("coordinate") or raw_args.get("coordinates")
    if isinstance(coord, list | tuple) and len(coord) >= 2:
        try:
            normalized_args["x"] = int(coord[0])
            normalized_args["y"] = int(coord[1])
        except (TypeError, ValueError):
            pass

    dest = (
        raw_args.get("destination")
        or raw_args.get("destination_coordinate")
        or raw_args.get("destinationCoordinate")
    )
    if isinstance(dest, list | tuple) and len(dest) >= 2:
        try:
            normalized_args["destination_x"] = int(dest[0])
            normalized_args["destination_y"] = int(dest[1])
        except (TypeError, ValueError):
            pass

    for key in (
        "text",
        "press_enter",
        "clear_before_typing",
        "safety_decision",
        "direction",
        "magnitude",
        "url",
        "keys",
        "x",
        "y",
        "destination_x",
        "destination_y",
    ):
        if key in raw_args:
            normalized_args[key] = raw_args[key]

    return normalized_args


ACTION_FIELD = Field(..., description="Gemini Computer Use action to perform")
X_FIELD = Field(None, description="X coordinate (pixels in agent space)")
Y_FIELD = Field(None, description="Y coordinate (pixels in agent space)")
TEXT_FIELD = Field(None, description="Text to type")
PRESS_ENTER_FIELD = Field(None, description="Whether to press Enter after typing (type_text_at)")
CLEAR_BEFORE_TYPING_FIELD = Field(
    None, description="Whether to select-all before typing (type_text_at)"
)
DIRECTION_FIELD = Field(None, description="Scroll direction for scroll_document/scroll_at")
MAGNITUDE_FIELD = Field(None, description="Scroll magnitude (pixels in agent space)")
URL_FIELD = Field(None, description="Target URL for navigate")
KEYS_FIELD = Field(None, description="Keys for key_combination")
DESTINATION_X_FIELD = Field(None, description="Destination X for drag_and_drop (agent space)")
DESTINATION_Y_FIELD = Field(None, description="Destination Y for drag_and_drop (agent space)")
TAKE_SCREENSHOT_ON_CLICK_FIELD = Field(
    True, description="Whether to include a screenshot for interactive actions"
)


class GeminiComputerTool(HudComputerTool):
    """
    Gemini Computer Use tool for interacting with a computer via MCP.

    Maps Gemini's predefined function names (open_web_browser, click_at, hover_at,
    type_text_at, scroll_document, scroll_at, wait_5_seconds, go_back, go_forward,
    search, navigate, key_combination, drag_and_drop) to executor actions.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(
            api_type="computer_use",
            api_name="gemini_computer",
            role="computer",  # Mutually exclusive with other computer tools when native
            supported_models=SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
        ),
        AgentType.GEMINI_CUA: NativeToolSpec(
            api_type="computer_use",
            api_name="gemini_computer",
            role="computer",  # Mutually exclusive with other computer tools when native
            supported_models=SUPPORTED_GEMINI_COMPUTER_USE_MODELS,
        ),
    }

    def __init__(
        self,
        # Define within environment based on platform
        executor: BaseExecutor | None = None,
        platform_type: Literal["auto", "xdo", "pyautogui"] = "auto",
        display_num: int | None = None,
        # Overrides for what dimensions the agent thinks it operates in
        width: int = computer_settings.GEMINI_COMPUTER_WIDTH,
        height: int = computer_settings.GEMINI_COMPUTER_HEIGHT,
        rescale_images: bool = computer_settings.GEMINI_RESCALE_IMAGES,
        # What the agent sees as the tool's name, title, and description
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with Gemini's default dimensions.

        Args:
            width: Width for agent coordinate system (default: 1440)
            height: Height for agent coordinate system (default: 900)
        """
        super().__init__(
            executor=executor,
            platform_type=platform_type,
            display_num=display_num,
            width=width,
            height=height,
            rescale_images=rescale_images,
            coordinate_space=1000,
            name=name or "gemini_computer",
            title=title or "Gemini Computer Tool",
            description=description or "Control computer with mouse, keyboard, and screenshots",
            **kwargs,
        )

    async def __call__(
        self,
        action: str = ACTION_FIELD,
        # Common coordinates
        x: int | None = X_FIELD,
        y: int | None = Y_FIELD,
        # Text input
        text: str | None = TEXT_FIELD,
        press_enter: bool | None = PRESS_ENTER_FIELD,
        clear_before_typing: bool | None = CLEAR_BEFORE_TYPING_FIELD,
        # Scroll parameters
        direction: Literal["up", "down", "left", "right"] | None = DIRECTION_FIELD,
        magnitude: int | None = MAGNITUDE_FIELD,
        # Navigation
        url: str | None = URL_FIELD,
        # Key combos
        keys: list[str] | str | None = KEYS_FIELD,
        # Drag parameters
        destination_x: int | None = DESTINATION_X_FIELD,
        destination_y: int | None = DESTINATION_Y_FIELD,
        # Behavior
        take_screenshot_on_click: bool = TAKE_SCREENSHOT_ON_CLICK_FIELD,
    ) -> list[ContentBlock]:
        """
        Handle Gemini Computer Use API calls by mapping to executor actions.

        Returns:
            List of MCP content blocks
        """
        logger.info("GeminiComputerTool received action: %s", action)

        # Helper to finalize ContentResult: rescale if requested and ensure URL metadata
        async def _finalize(
            result: ContentResult, requested_url: str | None = None
        ) -> list[ContentBlock]:
            if result.base64_image and self.rescale_images:
                try:
                    result.base64_image = await self._rescale_screenshot(result.base64_image)
                except Exception as e:
                    logger.warning("Failed to rescale screenshot: %s", e)
            # Always include URL metadata if provided; otherwise default to about:blank
            result.url = requested_url or result.url or "about:blank"
            return result.to_content_blocks()

        # Map actions
        if action == "open_web_browser":
            screenshot = await self.executor.screenshot()
            if screenshot:
                result = ContentResult(base64_image=screenshot, url="about:blank")
            else:
                result = ContentResult(error="Failed to take screenshot", url="about:blank")
            return await _finalize(result)

        elif action == "click_at":
            if x is None or y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y are required"))
            sx, sy = self._scale_coordinates(x, y)
            result = await self.executor.click(x=sx, y=sy)
            return await _finalize(result)

        elif action == "hover_at":
            if x is None or y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y are required"))
            sx, sy = self._scale_coordinates(x, y)
            result = await self.executor.move(x=sx, y=sy)
            return await _finalize(result)

        elif action == "type_text_at":
            if x is None or y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y are required"))
            if text is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="text is required"))

            sx, sy = self._scale_coordinates(x, y)

            # Focus the field
            await self.executor.move(x=sx, y=sy, take_screenshot=False)
            await self.executor.click(x=sx, y=sy, take_screenshot=False)

            # Clear existing text if requested
            if clear_before_typing is None or clear_before_typing:
                is_mac = platform.system().lower() == "darwin"
                combo = ["cmd", "a"] if is_mac else ["ctrl", "a"]
                await self.executor.press(keys=combo, take_screenshot=False)
                delete_key = "backspace" if is_mac else "delete"
                await self.executor.press(keys=[delete_key], take_screenshot=False)

            # Type (optionally press enter after)
            result = await self.executor.write(text=text, enter_after=bool(press_enter))
            return await _finalize(result)

        elif action == "scroll_document":
            if direction is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="direction is required"))
            # Default magnitude similar to reference implementation
            mag = magnitude if magnitude is not None else 800
            # Convert to environment units while preserving sign
            if direction in ("down", "up"):
                distance = self._scale_distance(mag, "y")
                if distance is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="Unable to determine scroll magnitude"
                        )
                    )
                scroll_y = distance if direction == "down" else -distance
                scroll_x = None
            elif direction in ("right", "left"):
                distance = self._scale_distance(mag, "x")
                if distance is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="Unable to determine scroll magnitude"
                        )
                    )
                scroll_x = distance if direction == "right" else -distance
                scroll_y = None
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message=f"Invalid direction: {direction}")
                )
            result = await self.executor.scroll(scroll_x=scroll_x, scroll_y=scroll_y)
            return await _finalize(result)

        elif action == "scroll_at":
            if direction is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="direction is required"))
            if x is None or y is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="x and y are required"))
            mag = magnitude if magnitude is not None else 800
            sx, sy = self._scale_coordinates(x, y)
            if direction in ("down", "up"):
                distance = self._scale_distance(mag, "y")
                if distance is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="Unable to determine scroll magnitude"
                        )
                    )
                scroll_y = distance if direction == "down" else -distance
                scroll_x = None
            elif direction in ("right", "left"):
                distance = self._scale_distance(mag, "x")
                if distance is None:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS, message="Unable to determine scroll magnitude"
                        )
                    )
                scroll_x = distance if direction == "right" else -distance
                scroll_y = None
            else:
                raise McpError(
                    ErrorData(code=INVALID_PARAMS, message=f"Invalid direction: {direction}")
                )
            result = await self.executor.scroll(x=sx, y=sy, scroll_x=scroll_x, scroll_y=scroll_y)
            return await _finalize(result)

        elif action == "wait_5_seconds":
            result = await self.executor.wait(time=5000)
            return await _finalize(result)

        elif action == "go_back":
            is_mac = platform.system().lower() == "darwin"
            combo = ["cmd", "["] if is_mac else ["alt", "left"]
            result = await self.executor.press(keys=combo)
            return await _finalize(result)

        elif action == "go_forward":
            is_mac = platform.system().lower() == "darwin"
            combo = ["cmd", "]"] if is_mac else ["alt", "right"]
            result = await self.executor.press(keys=combo)
            return await _finalize(result)

        elif action == "search":
            # Best-effort navigate to a default search page
            target = url or "https://www.google.com"
            is_mac = platform.system().lower() == "darwin"
            await self.executor.press(
                keys=["cmd", "l"] if is_mac else ["ctrl", "l"], take_screenshot=False
            )
            result = await self.executor.write(text=target, enter_after=True)
            return await _finalize(result, requested_url=target)

        elif action == "navigate":
            if not url:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="url is required"))
            is_mac = platform.system().lower() == "darwin"
            await self.executor.press(
                keys=["cmd", "l"] if is_mac else ["ctrl", "l"], take_screenshot=False
            )
            result = await self.executor.write(text=url, enter_after=True)
            return await _finalize(result, requested_url=url)

        elif action == "key_combination":
            if keys is None:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="keys is required"))
            if isinstance(keys, str):
                # Accept formats like "ctrl+c" or "ctrl+shift+t"
                key_list = [k.strip() for k in keys.split("+") if k.strip()]
            else:
                key_list = keys
            result = await self.executor.press(keys=key_list)
            return await _finalize(result)

        elif action == "drag_and_drop":
            if x is None or y is None or destination_x is None or destination_y is None:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="x, y, destination_x, and destination_y are required",
                    )
                )
            path = self._scale_path([(x, y), (destination_x, destination_y)])
            result = await self.executor.drag(path=path)
            return await _finalize(result)

        else:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown action: {action}"))

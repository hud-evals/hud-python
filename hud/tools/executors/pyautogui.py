from __future__ import annotations

import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import Any, Literal

from hud.tools.types import ContentResult

from .base import BaseExecutor

logger = logging.getLogger(__name__)

# Lazy loading for pyautogui
_pyautogui = None
_pyautogui_available = None


def _get_pyautogui() -> Any | None:
    """Lazily import and return pyautogui module."""
    global _pyautogui, _pyautogui_available

    if _pyautogui_available is False:
        return None

    if _pyautogui is None:
        # Set display if not already set
        if "DISPLAY" not in os.environ:
            try:
                from hud.tools.computer import computer_settings

                os.environ["DISPLAY"] = f":{computer_settings.DISPLAY_NUM}"
            except (ImportError, AttributeError):
                os.environ["DISPLAY"] = ":0"

        try:
            import pyautogui  # type: ignore[import-not-found]

            _pyautogui = pyautogui
            _pyautogui_available = True

            # Configure PyAutoGUI settings
            _pyautogui.FAILSAFE = False  # Disable fail-safe feature
            _pyautogui.PAUSE = 0.1  # Small pause between actions
        except ImportError:
            _pyautogui_available = False
            logger.warning("PyAutoGUI is not available")
            return None
        except Exception as e:
            _pyautogui_available = False
            logger.warning("Failed to initialize PyAutoGUI: %s", e)
            return None

    return _pyautogui


# Map CLA standard keys to PyAutoGUI keys (only where they differ)
CLA_TO_PYAUTOGUI = {
    # Most keys are the same in PyAutoGUI, only map the differences
    "escape": "esc",
    "enter": "return",
    "pageup": "pgup",
    "pagedown": "pgdn",
    "printscreen": "prtscr",
    "prtsc": "prtscr",
    "super": "win",
    "command": "cmd",
}


class PyAutoGUIExecutor(BaseExecutor):
    """
    Cross-platform executor using PyAutoGUI.
    Works on Windows, macOS, and Linux.

    This executor should only be instantiated when PyAutoGUI is available and functional.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the executor.

        Args:
            display_num: X display number (used only on Linux, ignored on Windows/macOS)
        """
        super().__init__(display_num)
        self._pyautogui = None
        logger.info("PyAutoGUIExecutor initialized")

    @property
    def pyautogui(self) -> Any:
        """Get the pyautogui module, importing it lazily if needed."""
        if self._pyautogui is None:
            self._pyautogui = _get_pyautogui()
            if self._pyautogui is None:
                raise RuntimeError("PyAutoGUI is not available")
        return self._pyautogui

    def _map_key(self, key: str) -> str:
        """Map CLA standard key to PyAutoGUI key."""
        return CLA_TO_PYAUTOGUI.get(key.lower(), key.lower())

    def _map_keys(self, keys: list[str]) -> list[str]:
        """Map CLA standard keys to PyAutoGUI keys."""
        mapped_keys = []
        for key in keys:
            # Handle key combinations like "ctrl+a"
            if "+" in key:
                parts = key.split("+")
                mapped_parts = [self._map_key(part) for part in parts]
                mapped_keys.append("+".join(mapped_parts))
            else:
                mapped_keys.append(self._map_key(key))
        return mapped_keys

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if PyAutoGUI is available and functional.

        Returns:
            True if PyAutoGUI is available and functional, False otherwise
        """
        pyautogui = _get_pyautogui()
        if not pyautogui:
            return False

        try:
            # Try to get screen size as a simple test
            pyautogui.size()
            return True
        except Exception:
            return False

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
        try:
            # Take screenshot using PyAutoGUI
            screenshot = self.pyautogui.screenshot()

            # Convert to base64
            buffer = BytesIO()
            screenshot.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            return base64.b64encode(image_data).decode()
        except Exception as e:
            logger.error("Failed to take screenshot: %s", e)
            return None

    # ===== Helper Methods =====

    def _hold_keys_context(self, keys: list[str] | None) -> None:
        """
        Press and hold keys.

        Args:
            keys: List of keys to hold
        """
        if keys:
            for key in keys:
                self.pyautogui.keyDown(key)

    def _release_keys(self, keys: list[str] | None) -> None:
        """Release held keys."""
        if keys:
            for key in reversed(keys):  # Release in reverse order
                self.pyautogui.keyUp(key)

    # ===== CLA Action Implementations =====

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Click at specified coordinates or current position."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                # Handle multi-clicks based on pattern
                if pattern:
                    clicks = len(pattern) + 1
                    interval = pattern[0] / 1000.0 if pattern else 0.1  # Convert ms to seconds

                    if x is not None and y is not None:
                        self.pyautogui.click(
                            x=x, y=y, clicks=clicks, interval=interval, button=button_name
                        )
                    else:
                        self.pyautogui.click(clicks=clicks, interval=interval, button=button_name)
                else:
                    # Single click
                    if x is not None and y is not None:
                        self.pyautogui.click(x=x, y=y, button=button_name)
                    else:
                        self.pyautogui.click(button=button_name)
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            result = ContentResult(
                output=f"Clicked {button} button at ({x}, {y})" if x else f"Clicked {button} button"
            )

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def write(
        self, text: str, enter_after: bool = False, delay: int = 12, take_screenshot: bool = True
    ) -> ContentResult:
        """Type text with specified delay between keystrokes."""
        try:
            # Convert delay from milliseconds to seconds for PyAutoGUI
            interval = delay / 1000.0
            self.pyautogui.typewrite(text, interval=interval)

            if enter_after:
                self.pyautogui.press("enter")

            result = ContentResult(
                output=f"Typed: '{text}'" + (" and pressed Enter" if enter_after else "")
            )

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ContentResult:
        """Press a key or key combination."""
        try:
            # Handle key combinations (e.g., "ctrl+c")
            if "+" in key_sequence:
                keys = key_sequence.split("+")
                self.pyautogui.hotkey(*keys)
                result = ContentResult(output=f"Pressed hotkey: {key_sequence}")
            else:
                # Map common key names from xdotool to PyAutoGUI
                key = key_sequence.lower()
                self.pyautogui.press(CLA_TO_PYAUTOGUI.get(key, key))
                result = ContentResult(output=f"Pressed key: {key_sequence}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def press(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Press a key combination (hotkey)."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)

            # Handle single key or combination
            if len(mapped_keys) == 1 and "+" not in mapped_keys[0]:
                self.pyautogui.press(mapped_keys[0])
                result = ContentResult(output=f"Pressed key: {keys[0]}")
            else:
                # For combinations, use hotkey
                hotkey_parts = []
                for key in mapped_keys:
                    if "+" in key:
                        hotkey_parts.extend(key.split("+"))
                    else:
                        hotkey_parts.append(key)
                self.pyautogui.hotkey(*hotkey_parts)
                result = ContentResult(output=f"Pressed hotkey: {'+'.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def keydown(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Press and hold keys."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)
            for key in mapped_keys:
                self.pyautogui.keyDown(key)

            result = ContentResult(output=f"Keys down: {', '.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def keyup(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Release held keys."""
        try:
            # Map CLA keys to PyAutoGUI keys
            mapped_keys = self._map_keys(keys)
            for key in reversed(mapped_keys):  # Release in reverse order
                self.pyautogui.keyUp(key)

            result = ContentResult(output=f"Keys up: {', '.join(keys)}")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Scroll at specified position."""
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.pyautogui.moveTo(x, y)

            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                msg_parts = []

                # Perform vertical scroll
                if scroll_y and scroll_y != 0:
                    # PyAutoGUI: positive = up, negative = down (opposite of our convention)
                    self.pyautogui.scroll(-scroll_y)
                    msg_parts.append(f"vertically by {scroll_y}")

                # Perform horizontal scroll (if supported)
                if scroll_x and scroll_x != 0:
                    # PyAutoGUI horizontal scroll might not work on all platforms
                    try:
                        self.pyautogui.hscroll(scroll_x)
                        msg_parts.append(f"horizontally by {scroll_x}")
                    except AttributeError:
                        # hscroll not available
                        msg_parts.append(f"horizontally by {scroll_x} (not supported)")

                if not msg_parts:
                    return ContentResult(output="No scroll amount specified")

                msg = "Scrolled " + " and ".join(msg_parts)
                if x is not None and y is not None:
                    msg += f" at ({x}, {y})"
                if hold_keys:
                    msg += f" while holding {hold_keys}"
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            result = ContentResult(output=msg)

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Move mouse cursor."""
        try:
            if x is not None and y is not None:
                # Absolute move
                self.pyautogui.moveTo(x, y, duration=0.1)
                result = ContentResult(output=f"Moved mouse to ({x}, {y})")
            elif offset_x is not None or offset_y is not None:
                # Relative move
                offset_x = offset_x or 0
                offset_y = offset_y or 0
                self.pyautogui.moveRel(xOffset=offset_x, yOffset=offset_y, duration=0.1)
                result = ContentResult(output=f"Moved mouse by offset ({offset_x}, {offset_y})")
            else:
                return ContentResult(output="No move coordinates specified")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def drag(
        self,
        path: list[tuple[int, int]],
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Drag along a path."""
        if len(path) < 2:
            return ContentResult(error="Drag path must have at least 2 points")

        try:
            # Hold keys if specified
            self._hold_keys_context(hold_keys)

            try:
                # Move to start
                start_x, start_y = path[0]
                self.pyautogui.moveTo(start_x, start_y)

                # Handle multi-point drag
                if len(path) == 2:
                    # Simple drag
                    end_x, end_y = path[1]
                    self.pyautogui.dragTo(end_x, end_y, duration=0.5, button="left")
                    result = ContentResult(
                        output=f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                    )
                else:
                    # Multi-point drag
                    self.pyautogui.mouseDown(button="left")
                    for i, (x, y) in enumerate(path[1:], 1):
                        duration = 0.1
                        if pattern and i - 1 < len(pattern):
                            duration = pattern[i - 1] / 1000.0  # Convert ms to seconds
                        self.pyautogui.moveTo(x, y, duration=duration)
                    self.pyautogui.mouseUp(button="left")

                    result = ContentResult(output=f"Dragged along {len(path)} points")

                if hold_keys:
                    result = ContentResult(output=f"{result.output} while holding {hold_keys}")
            finally:
                # Release held keys
                self._release_keys(hold_keys)

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def mouse_down(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Press and hold a mouse button."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            self.pyautogui.mouseDown(button=button_name)
            result = ContentResult(output=f"Mouse down: {button} button")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def mouse_up(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Release a mouse button."""
        try:
            # Map button names (PyAutoGUI doesn't support back/forward)
            button_map = {
                "left": "left",
                "right": "right",
                "middle": "middle",
                "back": "left",
                "forward": "right",
            }  # Fallback for unsupported
            button_name = button_map.get(button, "left")

            self.pyautogui.mouseUp(button=button_name)
            result = ContentResult(output=f"Mouse up: {button} button")

            if take_screenshot:
                await asyncio.sleep(self._screenshot_delay)
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def hold_key(
        self, key: str, duration: float, take_screenshot: bool = True
    ) -> ContentResult:
        """Hold a key for a specified duration."""
        try:
            # Map CLA key to PyAutoGUI key
            mapped_key = self._map_key(key)
            self.pyautogui.keyDown(mapped_key)
            await asyncio.sleep(duration)
            self.pyautogui.keyUp(mapped_key)

            result = ContentResult(output=f"Held key '{key}' for {duration} seconds")

            if take_screenshot:
                screenshot = await self.screenshot()
                if screenshot:
                    result = ContentResult(
                        output=result.output, error=result.error, base64_image=screenshot
                    )

            return result
        except Exception as e:
            return ContentResult(error=str(e))

    async def position(self) -> ContentResult:
        """Get current cursor position."""
        try:
            x, y = self.pyautogui.position()
            return ContentResult(output=f"Mouse position: ({x}, {y})")
        except Exception as e:
            return ContentResult(error=str(e))

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
from pathlib import Path
from tempfile import gettempdir
from typing import Literal
from uuid import uuid4

from hud.tools.types import ContentResult
from hud.tools.utils import run

from .base import BaseExecutor

OUTPUT_DIR = os.environ.get("SCREENSHOT_DIR")
logger = logging.getLogger(__name__)

# Map CLA standard keys to X11/XDO key names
CLA_TO_XDO = {
    "enter": "Return",
    "tab": "Tab",
    "space": "space",
    "backspace": "BackSpace",
    "delete": "Delete",
    "escape": "Escape",
    "esc": "Escape",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "shift": "Shift_L",
    "shiftleft": "Shift_L",
    "shiftright": "Shift_R",
    "ctrl": "Control_L",
    "ctrlleft": "Control_L",
    "ctrlright": "Control_R",
    "alt": "Alt_L",
    "altleft": "Alt_L",
    "altright": "Alt_R",
    "win": "Super_L",
    "winleft": "Super_L",
    "winright": "Super_R",
    "cmd": "Control_L",  # Map cmd to ctrl for Linux
    "command": "Control_L",
    "super": "Super_L",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "home": "Home",
    "end": "End",
    "insert": "Insert",
    "pause": "Pause",
    "capslock": "Caps_Lock",
    "numlock": "Num_Lock",
    "scrolllock": "Scroll_Lock",
    "printscreen": "Print",
    "prtsc": "Print",
    # Function keys
    **{f"f{i}": f"F{i}" for i in range(1, 25)},
}


class XDOExecutor(BaseExecutor):
    """
    Low-level executor for xdotool commands.
    Handles display management and screenshot capture on Linux/X11 systems.

    This executor should only be instantiated when X11 display is available.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """Initialize with optional display number."""
        super().__init__(display_num)

        if display_num is not None:
            self._display_prefix = f"DISPLAY=:{display_num} "
        else:
            self._display_prefix = ""

        self.xdotool = f"{self._display_prefix}xdotool"
        logger.info("XDOExecutor initialized")

    def _map_key(self, key: str) -> str:
        """Map CLA standard key to XDO key."""
        return CLA_TO_XDO.get(key.lower(), key)

    def _map_keys(self, keys: list[str]) -> list[str]:
        """Map CLA standard keys to XDO keys."""
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
        Check if xdotool and X11 display are available.

        Returns:
            True if xdotool can be used, False otherwise
        """
        display = os.environ.get("DISPLAY")
        if not display:
            return False

        # Try a simple xdotool command to test availability
        try:
            import subprocess

            # Try without display prefix if DISPLAY is already set
            result = subprocess.run(
                ["xdotool", "getdisplaygeometry"],  # noqa: S607
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    async def execute(self, command: str, take_screenshot: bool = True) -> ContentResult:
        """
        Execute an xdotool command.

        Args:
            command: The xdotool command (without xdotool prefix)
            take_screenshot: Whether to capture a screenshot after execution

        Returns:
            ContentResult with output, error, and optional screenshot
        """
        full_command = f"{self.xdotool} {command}"

        # Execute command
        returncode, stdout, stderr = await run(full_command)

        # Prepare result
        result = ContentResult(
            output=stdout if stdout else None, error=stderr if stderr or returncode != 0 else None
        )

        # Take screenshot if requested
        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            screenshot = await self.screenshot()
            if screenshot:
                result = ContentResult(
                    output=result.output, error=result.error, base64_image=screenshot
                )

        return result

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
        # Real screenshot using scrot
        if OUTPUT_DIR:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = output_dir / f"screenshot_{uuid4().hex}.png"
        else:
            # Generate a unique path in system temp dir without opening a file
            screenshot_path = Path(gettempdir()) / f"screenshot_{uuid4().hex}.png"

        screenshot_cmd = f"{self._display_prefix}scrot -p {screenshot_path}"

        returncode, _, _stderr = await run(screenshot_cmd)

        if returncode == 0 and screenshot_path.exists():
            try:
                image_data = screenshot_path.read_bytes()
                # Remove the file unless user requested persistence via env var
                if not OUTPUT_DIR:
                    screenshot_path.unlink(missing_ok=True)
                return base64.b64encode(image_data).decode()
            except Exception:
                return None

        return None

    # ===== Helper Methods =====

    async def _hold_keys_context(self, keys: list[str] | None) -> None:
        """
        Press and hold keys, to be used with try/finally.

        Args:
            keys: List of keys to hold

        Example:
            await self._hold_keys_context(['ctrl'])
            try:
                # Do action with ctrl held
            finally:
                await self._release_keys(['ctrl'])
        """
        if keys:
            for key in keys:
                escaped_key = shlex.quote(key)
                await self.execute(f"keydown {escaped_key}", take_screenshot=False)

    async def _release_keys(self, keys: list[str] | None) -> None:
        """Release held keys."""
        if keys:
            for key in reversed(keys):  # Release in reverse order
                escaped_key = shlex.quote(key)
                await self.execute(f"keyup {escaped_key}", take_screenshot=False)

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
        # Map button names to xdotool button numbers
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)

        # Hold keys if specified
        await self._hold_keys_context(hold_keys)

        try:
            # Handle multi-clicks based on pattern
            if pattern:
                click_count = len(pattern) + 1
                delay = pattern[0] if pattern else 10  # Use first delay for all clicks

                if x is not None and y is not None:
                    cmd = f"mousemove {x} {y} click --repeat {click_count} --delay {delay} {button_num}"  # noqa: E501
                else:
                    cmd = f"click --repeat {click_count} --delay {delay} {button_num}"
            else:
                # Single click
                if x is not None and y is not None:
                    cmd = f"mousemove {x} {y} click {button_num}"
                else:
                    cmd = f"click {button_num}"

            result = await self.execute(cmd, take_screenshot=take_screenshot)
        finally:
            # Release held keys
            await self._release_keys(hold_keys)

        return result

    async def write(
        self, text: str, enter_after: bool = False, delay: int = 12, take_screenshot: bool = True
    ) -> ContentResult:
        """Type text with specified delay between keystrokes."""
        # Escape text for shell
        escaped_text = shlex.quote(text)
        cmd = f"type --delay {delay} -- {escaped_text}"
        result = await self.execute(cmd, take_screenshot=False)

        if enter_after:
            enter_result = await self.key("Return", take_screenshot=False)
            # Combine outputs
            combined_output = (result.output or "") + "\n" + (enter_result.output or "")
            combined_error = None
            if result.error or enter_result.error:
                combined_error = (result.error or "") + "\n" + (enter_result.error or "")
            result = ContentResult(output=combined_output.strip(), error=combined_error)

        if take_screenshot:
            screenshot = await self.screenshot()
            if screenshot:
                result = ContentResult(
                    output=result.output, error=result.error, base64_image=screenshot
                )

        return result

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ContentResult:
        """Press a key or key combination."""
        return await self.execute(f"key -- {key_sequence}", take_screenshot=take_screenshot)

    async def press(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Press a key combination (hotkey)."""
        # Map CLA keys to XDO keys
        mapped_keys = self._map_keys(keys)
        # Convert list of keys to xdotool format
        key_combo = "+".join(mapped_keys)
        return await self.key(key_combo, take_screenshot=take_screenshot)

    async def keydown(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Press and hold keys."""
        # Map CLA keys to XDO keys
        mapped_keys = self._map_keys(keys)
        last_result = None
        for key in mapped_keys:
            escaped_key = shlex.quote(key)
            last_result = await self.execute(f"keydown {escaped_key}", take_screenshot=False)

        if take_screenshot and last_result:
            screenshot = await self.screenshot()
            if screenshot:
                last_result = ContentResult(
                    output=last_result.output, error=last_result.error, base64_image=screenshot
                )

        return last_result or ContentResult()

    async def keyup(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """Release held keys."""
        # Map CLA keys to XDO keys
        mapped_keys = self._map_keys(keys)
        last_result = None
        for key in mapped_keys:
            escaped_key = shlex.quote(key)
            last_result = await self.execute(f"keyup {escaped_key}", take_screenshot=False)

        if take_screenshot and last_result:
            screenshot = await self.screenshot()
            if screenshot:
                last_result = ContentResult(
                    output=last_result.output, error=last_result.error, base64_image=screenshot
                )

        return last_result or ContentResult()

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
        # Convert scroll amounts to xdotool format
        scroll_button_map = {"up": 4, "down": 5, "left": 6, "right": 7}

        # Convert pixels to wheel clicks
        # Standard conversion: 1 wheel click ≈ 100 pixels
        PIXELS_PER_WHEEL_CLICK = 100

        # Hold keys if specified
        await self._hold_keys_context(hold_keys)

        try:
            # Handle vertical scroll
            if scroll_y and scroll_y != 0:
                direction = "down" if scroll_y > 0 else "up"
                # Convert pixels to clicks
                clicks = max(1, abs(scroll_y) // PIXELS_PER_WHEEL_CLICK)
                button = scroll_button_map.get(direction, 5)

                if x is not None and y is not None:
                    cmd = f"mousemove {x} {y} click --repeat {clicks} {button}"
                else:
                    cmd = f"click --repeat {clicks} {button}"

                result = await self.execute(cmd, take_screenshot=take_screenshot)

            # Handle horizontal scroll
            elif scroll_x and scroll_x != 0:
                direction = "right" if scroll_x > 0 else "left"
                # Convert pixels to clicks
                clicks = max(1, abs(scroll_x) // PIXELS_PER_WHEEL_CLICK)
                button = scroll_button_map.get(direction, 7)

                if x is not None and y is not None:
                    cmd = f"mousemove {x} {y} click --repeat {clicks} {button}"
                else:
                    cmd = f"click --repeat {clicks} {button}"

                result = await self.execute(cmd, take_screenshot=take_screenshot)

            else:
                result = ContentResult(output="No scroll amount specified")
        finally:
            # Release held keys
            await self._release_keys(hold_keys)

        return result

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Move mouse cursor."""
        if x is not None and y is not None:
            # Absolute move
            return await self.execute(f"mousemove {x} {y}", take_screenshot=take_screenshot)
        elif offset_x is not None or offset_y is not None:
            # Relative move
            offset_x = offset_x or 0
            offset_y = offset_y or 0
            return await self.execute(
                f"mousemove_relative -- {offset_x} {offset_y}", take_screenshot=take_screenshot
            )
        else:
            return ContentResult(output="No move coordinates specified")

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

        # Hold keys if specified
        await self._hold_keys_context(hold_keys)

        try:
            # Start drag
            start_x, start_y = path[0]
            await self.execute(f"mousemove {start_x} {start_y}", take_screenshot=False)
            await self.execute("mousedown 1", take_screenshot=False)

            # Move through intermediate points
            for i, (x, y) in enumerate(path[1:], 1):
                # Apply delay if pattern is specified
                if pattern and i - 1 < len(pattern):
                    await asyncio.sleep(pattern[i - 1] / 1000.0)  # Convert ms to seconds

                await self.execute(f"mousemove {x} {y}", take_screenshot=False)

            # End drag
            await self.execute("mouseup 1", take_screenshot=False)

            # Take final screenshot if requested
            if take_screenshot:
                screenshot = await self.screenshot()
                result = ContentResult(
                    output=f"Dragged along {len(path)} points", base64_image=screenshot
                )
            else:
                result = ContentResult(output=f"Dragged along {len(path)} points")

        finally:
            # Release held keys
            await self._release_keys(hold_keys)

        return result

    async def mouse_down(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Press and hold a mouse button."""
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)
        return await self.execute(f"mousedown {button_num}", take_screenshot=take_screenshot)

    async def mouse_up(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """Release a mouse button."""
        button_map = {"left": 1, "right": 3, "middle": 2, "back": 8, "forward": 9}
        button_num = button_map.get(button, 1)
        return await self.execute(f"mouseup {button_num}", take_screenshot=take_screenshot)

    async def hold_key(
        self, key: str, duration: float, take_screenshot: bool = True
    ) -> ContentResult:
        """Hold a key for a specified duration."""
        # Map CLA key to XDO key
        mapped_key = self._map_key(key)
        escaped_key = shlex.quote(mapped_key)

        # Press the key
        await self.execute(f"keydown {escaped_key}", take_screenshot=False)

        # Wait
        await asyncio.sleep(duration)

        # Release the key
        result = await self.execute(f"keyup {escaped_key}", take_screenshot=False)

        if take_screenshot:
            screenshot = await self.screenshot()
            if screenshot:
                result = ContentResult(
                    output=result.output, error=result.error, base64_image=screenshot
                )

        return result

    async def position(self) -> ContentResult:
        """Get current cursor position."""
        return await self.execute("getmouselocation", take_screenshot=False)

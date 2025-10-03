from __future__ import annotations

import asyncio
import logging
from typing import Literal, TypeAlias

from hud.tools.types import ContentResult

logger = logging.getLogger(__name__)


class BaseExecutor:
    """
    Base executor that provides simulation implementations for all CLA (Common Language Actions).

    This class:
    1. Defines all action methods that HudComputer expects
    2. Provides simulation implementations for environments without display
    3. Serves as the base class for platform-specific executors (XDO, PyAutoGUI)

    When used directly, it simulates all actions. Subclasses provide real implementations.
    """

    def __init__(self, display_num: int | None = None) -> None:
        """
        Initialize the base executor.

        Args:
            display_num: X display number (for Linux/X11 systems)
        """
        if display_num is None:
            from hud.tools.computer.settings import computer_settings

            self.display_num = computer_settings.DISPLAY_NUM
        else:
            self.display_num = display_num
        self._screenshot_delay = 0.5
        logger.info("BaseExecutor initialized")

    # ===== Core CLA Actions =====

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Click at specified coordinates.

        Args:
            x, y: Coordinates to click at (None = current position)
            button: Mouse button to use
            pattern: List of delays for multi-clicks (e.g., [100] for double-click)
            hold_keys: Keys to hold during click
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Click at ({x}, {y}) with {button} button"
        if pattern:
            msg += f" (multi-click pattern: {pattern})"
        if hold_keys:
            msg += f" while holding {hold_keys}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def write(
        self, text: str, enter_after: bool = False, delay: int = 12, take_screenshot: bool = True
    ) -> ContentResult:
        """
        Type text using keyboard.

        Args:
            text: Text to type
            enter_after: Whether to press Enter after typing
            delay: Delay between keystrokes in milliseconds
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Type '{text}'"
        if enter_after:
            msg += " followed by Enter"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def press(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """
        Press a key combination (hotkey).

        Args:
            keys: List of keys to press together (e.g., ["ctrl", "c"])
            take_screenshot: Whether to capture screenshot after action
        """
        key_combo = "+".join(keys)
        msg = f"[SIMULATED] Press key combination: {key_combo}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def key(self, key_sequence: str, take_screenshot: bool = True) -> ContentResult:
        """
        Press a single key or key combination.

        Args:
            key_sequence: Key or combination like "Return" or "ctrl+a"
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Press key: {key_sequence}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def keydown(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """
        Press and hold keys.

        Args:
            keys: Keys to press and hold
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Key down: {', '.join(keys)}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def keyup(self, keys: list[str], take_screenshot: bool = True) -> ContentResult:
        """
        Release held keys.

        Args:
            keys: Keys to release
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Key up: {', '.join(keys)}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: int | None = None,
        scroll_y: int | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Scroll at specified position.

        Args:
            x, y: Position to scroll at (None = current position)
            scroll_x: Horizontal scroll amount (positive = right)
            scroll_y: Vertical scroll amount (positive = down)
            hold_keys: Keys to hold during scroll
            take_screenshot: Whether to capture screenshot after action
        """
        msg = "[SIMULATED] Scroll"
        if x is not None and y is not None:
            msg += f" at ({x}, {y})"
        if scroll_x:
            msg += f" horizontally by {scroll_x}"
        if scroll_y:
            msg += f" vertically by {scroll_y}"
        if hold_keys:
            msg += f" while holding {hold_keys}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def move(
        self,
        x: int | None = None,
        y: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Move mouse cursor.

        Args:
            x, y: Absolute coordinates to move to
            offset_x, offset_y: Relative offset from current position
            take_screenshot: Whether to capture screenshot after action
        """
        if x is not None and y is not None:
            msg = f"[SIMULATED] Move mouse to ({x}, {y})"
        elif offset_x is not None or offset_y is not None:
            msg = f"[SIMULATED] Move mouse by offset ({offset_x or 0}, {offset_y or 0})"
        else:
            msg = "[SIMULATED] Move mouse (no coordinates specified)"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def drag(
        self,
        path: list[tuple[int, int]],
        pattern: list[int] | None = None,
        hold_keys: list[str] | None = None,
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Drag along a path.

        Args:
            path: List of (x, y) coordinates defining the drag path
            pattern: Delays between path points in milliseconds
            hold_keys: Keys to hold during drag
            take_screenshot: Whether to capture screenshot after action
        """
        if len(path) < 2:
            return ContentResult(error="Drag path must have at least 2 points")

        start = path[0]
        end = path[-1]
        msg = f"[SIMULATED] Drag from {start} to {end}"
        if len(path) > 2:
            msg += f" via {len(path) - 2} intermediate points"
        if hold_keys:
            msg += f" while holding {hold_keys}"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def mouse_down(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Press and hold a mouse button.

        Args:
            button: Mouse button to press
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Mouse down: {button} button"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def mouse_up(
        self,
        button: Literal["left", "right", "middle", "back", "forward"] = "left",
        take_screenshot: bool = True,
    ) -> ContentResult:
        """
        Release a mouse button.

        Args:
            button: Mouse button to release
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Mouse up: {button} button"

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    async def hold_key(
        self, key: str, duration: float, take_screenshot: bool = True
    ) -> ContentResult:
        """
        Hold a key for a specified duration.

        Args:
            key: The key to hold
            duration: Duration in seconds
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Hold key '{key}' for {duration} seconds"
        await asyncio.sleep(duration)  # Simulate the wait

        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    # ===== Utility Actions =====

    async def wait(self, time: int, take_screenshot: bool = True) -> ContentResult:
        """
        Wait for specified time.

        Args:
            time: Time to wait in milliseconds
        """
        duration_seconds = time / 1000.0
        await asyncio.sleep(duration_seconds)
        # take screenshot
        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=f"Waited {time}ms", base64_image=screenshot)

    async def screenshot(self) -> str | None:
        """
        Take a screenshot and return base64 encoded image.

        Returns:
            Base64 encoded PNG image or None if failed
        """
        logger.info("[SIMULATION] Taking screenshot")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501

    async def position(self) -> ContentResult:
        """
        Get current cursor position.

        Returns:
            ToolResult with position information
        """
        return ContentResult(output="[SIMULATED] Mouse position: (0, 0)")

    # ===== Legacy/Compatibility Methods =====

    async def execute(self, command: str, take_screenshot: bool = True) -> ContentResult:
        """
        Execute a raw command (for backwards compatibility).

        Args:
            command: Command to execute
            take_screenshot: Whether to capture screenshot after action
        """
        msg = f"[SIMULATED] Execute: {command}"
        screenshot = await self.screenshot() if take_screenshot else None
        return ContentResult(output=msg, base64_image=screenshot)

    # Compatibility aliases
    async def type_text(
        self, text: str, delay: int = 12, take_screenshot: bool = True
    ) -> ContentResult:
        """Alias for type() to maintain compatibility."""
        return await self.write(
            text, enter_after=False, delay=delay, take_screenshot=take_screenshot
        )

    async def mouse_move(self, x: int, y: int, take_screenshot: bool = True) -> ContentResult:
        """Alias for move() to maintain compatibility."""
        return await self.move(x=x, y=y, take_screenshot=take_screenshot)


CLAKey: TypeAlias = Literal[
    # Control keys
    "backspace",
    "tab",
    "enter",
    "shift",
    "shiftleft",
    "shiftright",
    "ctrl",
    "ctrlleft",
    "ctrlright",
    "alt",
    "altleft",
    "altright",
    "pause",
    "capslock",
    "esc",
    "escape",
    "space",
    "pageup",
    "pagedown",
    "end",
    "home",
    "left",
    "up",
    "right",
    "down",
    "select",
    "print",
    "execute",
    "printscreen",
    "prtsc",
    "insert",
    "delete",
    "help",
    "sleep",
    # Special keys
    "numlock",
    "scrolllock",
    "clear",
    "separator",
    "modechange",
    "apps",
    "browserback",
    "browserfavorites",
    "browserforward",
    "browserhome",
    "browserrefresh",
    "browsersearch",
    "browserstop",
    "launchapp1",
    "launchapp2",
    "launchmail",
    "launchmediaselect",
    "playpause",
    "start",
    "stop",
    "prevtrack",
    "nexttrack",
    "volumemute",
    "volumeup",
    "volumedown",
    "zoom",
    # Modifier keys
    "win",
    "winleft",
    "winright",
    "command",
    "option",
    "optionleft",
    "optionright",
    "fn",
    # Numpad keys
    "num0",
    "num1",
    "num2",
    "num3",
    "num4",
    "num5",
    "num6",
    "num7",
    "num8",
    "num9",
    "multiply",
    "add",
    "subtract",
    "decimal",
    "divide",
    # Function keys
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    # Language-specific keys
    "hanguel",
    "hangul",
    "hanja",
    "kana",
    "kanji",
    "junja",
    "convert",
    "nonconvert",
    "yen",
    # Characters
    "\t",
    "\n",
    "\r",
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
]

CLAButton: TypeAlias = Literal["left", "right", "middle", "back", "forward"]

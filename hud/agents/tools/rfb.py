"""RFBTool: capability base for tools driven by an ``RFBClient``.

Provides primitive HID + framebuffer verbs (``screenshot``, ``move``, ``click``,
``type_text``, ``press_keys``, ``scroll``, ``drag``, ``wait``) on top of
``RFBClient``. Provider tools (``ClaudeComputerTool``, ``GeminiComputerTool``,
``OpenAIComputerTool``) extend this with the LLM-facing action schema and
translate the LLM's call into these primitives.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, Self

from hud.agents.tools.base import AgentTool, AgentToolSpec
from hud.capabilities import RFBClient
from hud.capabilities.rfb import PngScreenshotEncoding, ScreenshotEncoding
from hud.types import ImageContent, ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from hud.agents.types import ToolAgentConfig


#: RFB pointer button index.
Button = Literal["left", "middle", "right"]
_BUTTON_INDEX: dict[Button, int] = {"left": 0, "middle": 1, "right": 2}
_DEFAULT_SCREENSHOT_ENCODING = PngScreenshotEncoding()

#: Models name keys the way xdotool or a browser does (``ctrl``, ``Enter``, ``ArrowUp``,
#: ``cmd``); the display takes X11 keysyms. Names not listed, such as letters, ``F5`` or
#: ``Page_Down``, are keysyms already. Environments run Linux, so ``meta`` and ``cmd``
#: are the Super key.
_KEYSYMS: dict[str, str] = {
    "alt": "Alt_L",
    "option": "Alt_L",
    "ctrl": "Control_L",
    "control": "Control_L",
    "shift": "Shift_L",
    "meta": "Super_L",
    "super": "Super_L",
    "win": "Super_L",
    "cmd": "Super_L",
    "command": "Super_L",
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "space": "space",
    "backspace": "BackSpace",
    "del": "Delete",
    "delete": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "arrowup": "Up",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "capslock": "Caps_Lock",
    "printscreen": "Print",
}


def keysym(name: str) -> str:
    """The X11 keysym for a key as a model names it."""
    return _KEYSYMS.get(name.lower(), name)


class RFBTool(AgentTool[RFBClient]):
    """Capability base: tool driven by an ``RFBClient`` (VNC/RFB)."""

    client_type = RFBClient

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        client: RFBClient,
        screenshot_encoding: ScreenshotEncoding = _DEFAULT_SCREENSHOT_ENCODING,
    ) -> None:
        super().__init__(spec=spec, client=client)
        self.screenshot_encoding = screenshot_encoding

    @classmethod
    def from_config(
        cls, *, spec: AgentToolSpec, client: RFBClient, config: ToolAgentConfig
    ) -> Self:
        return cls(spec=spec, client=client, screenshot_encoding=config.screenshot_encoding)

    # ─── geometry ────────────────────────────────────────────────────

    @property
    def display_width(self) -> int:
        return self.client.width

    @property
    def display_height(self) -> int:
        return self.client.height

    # ─── framebuffer ─────────────────────────────────────────────────

    async def screenshot(self) -> ToolResult:
        """Capture a screenshot and return it as a single ``ImageContent`` block."""
        screenshot, mime_type = await self.client.screenshot_png(self.screenshot_encoding)
        return ToolResult(
            content=[
                ImageContent(
                    type="image",
                    mimeType=mime_type,
                    data=base64.b64encode(screenshot).decode("ascii"),
                )
            ],
        )

    # ─── pointer ─────────────────────────────────────────────────────

    async def move(self, x: int, y: int) -> None:
        self.client.move(int(x), int(y))
        await self.client.drain()

    async def click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        button: Button = "left",
        hold_keys: Iterable[str] | None = None,
        count: int = 1,
        interval_ms: int = 0,
    ) -> None:
        """Move (if x/y given), then click ``count`` times with optional modifier hold."""
        if x is not None and y is not None:
            self.client.move(int(x), int(y))
        index = _BUTTON_INDEX[button]
        async with self._with_keys(hold_keys):
            for i in range(max(1, count)):
                if i and interval_ms:
                    await asyncio.sleep(interval_ms / 1000)
                self.client.click(index)
        await self.client.drain()

    async def mouse_down(self, button: Button = "left") -> None:
        """Press ``button`` without releasing (for cross-turn drag sequences)."""
        self.client.button_down(_BUTTON_INDEX[button])
        await self.client.drain()

    async def mouse_up(self, button: Button = "left") -> None:
        """Release ``button`` (paired with a prior ``mouse_down``)."""
        self.client.button_up(_BUTTON_INDEX[button])
        await self.client.drain()

    async def scroll(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        scroll_x: int = 0,
        scroll_y: int = 0,
        hold_keys: Iterable[str] | None = None,
    ) -> None:
        """Scroll at (x, y). ``scroll_y > 0`` scrolls down, ``< 0`` scrolls up.

        ``scroll_x`` / ``scroll_y`` are in *clicks* (VNC has no pixel scroll).
        """
        if x is not None and y is not None:
            self.client.move(int(x), int(y))
        async with self._with_keys(hold_keys):
            self.client.scroll(scroll_y)
        await self.client.drain()

    async def drag(
        self,
        path: list[tuple[int, int]],
        *,
        button: Button = "left",
        hold_keys: Iterable[str] | None = None,
    ) -> None:
        """Press ``button`` at path[0], move through every subsequent point, then release.

        Interpolated and paced: a press/move/release burst within one frame
        reads as a click to pages that sample pointer state per frame.
        """
        if len(path) < 2:
            raise ValueError("drag requires at least 2 points")
        index = _BUTTON_INDEX[button]
        async with self._with_keys(hold_keys):
            x, y = int(path[0][0]), int(path[0][1])
            self.client.move(x, y)
            await self.client.drain()
            await asyncio.sleep(0.2)
            with self.client.holding_button(index):
                await self.client.drain()
                # The page must see the press for a frame before movement
                # starts; environments render at low frame rates.
                await asyncio.sleep(0.2)
                for px, py in path[1:]:
                    nx, ny = int(px), int(py)
                    steps = max(1, min(24, max(abs(nx - x), abs(ny - y)) // 16))
                    for i in range(1, steps + 1):
                        self.client.move(x + (nx - x) * i // steps, y + (ny - y) * i // steps)
                        await self.client.drain()
                        await asyncio.sleep(0.025)
                    x, y = nx, ny
                await asyncio.sleep(0.2)
        await self.client.drain()

    # ─── keyboard ────────────────────────────────────────────────────

    async def type_text(self, text: str) -> None:
        """Type a literal string, one key at a time."""
        self.client.type_text(text)
        await self.client.drain()

    async def press_keys(self, keys: Iterable[str], *, count: int = 1) -> None:
        """Press a chord of keys (e.g. ``['Control_L', 'c']``) ``count`` times."""
        key_list = list(keys)
        for _ in range(max(1, count)):
            self.client.press_keys(*key_list)
        await self.client.drain()

    async def key_down(self, key: str) -> None:
        """Press ``key`` without releasing it (paired with a later :meth:`key_up`)."""
        self.client.key_down(key)
        await self.client.drain()

    async def key_up(self, key: str) -> None:
        """Release ``key`` (paired with a prior :meth:`key_down`)."""
        self.client.key_up(key)
        await self.client.drain()

    async def hold_key(self, key: str, *, duration_ms: int) -> None:
        """Hold a single key for ``duration_ms`` then release."""
        with self.client.holding_keys(key):
            await asyncio.sleep(duration_ms / 1000)
        await self.client.drain()

    # ─── timing ──────────────────────────────────────────────────────

    @staticmethod
    async def wait(duration_ms: int) -> None:
        await asyncio.sleep(duration_ms / 1000)

    # ─── internal ────────────────────────────────────────────────────

    @asynccontextmanager
    async def _with_keys(self, keys: Iterable[str] | None) -> AsyncIterator[None]:
        if not keys:
            yield
            return
        with self.client.holding_keys(*keys):
            yield


__all__ = ["Button", "RFBTool", "keysym"]

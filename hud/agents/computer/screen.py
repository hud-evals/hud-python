"""``Screen``: pointer, keys and screenshots on a remote display over RFB."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from PIL import Image

from hud.capabilities.rfb import RFBClient, ScreenshotEncoding, WebPScreenshotEncoding
from hud.types import ImageContent, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterable


Button = Literal["left", "middle", "right"]
Point = tuple[int, int]
_BUTTON_INDEX: dict[Button, int] = {"left": 0, "middle": 1, "right": 2}

# Models name keys the way xdotool or a browser does (``ctrl``, ``Enter``, ``ArrowUp``,
# ``cmd``); the display takes X11 keysyms. Names not listed, such as letters, ``F5`` or
# ``Page_Down``, are keysyms already. Environments run Linux, so ``meta`` and ``cmd``
# are the Super key.
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


class Screen:
    """A remote display driven over an ``RFBClient``."""

    def __init__(self, client: RFBClient, screenshot_encoding: ScreenshotEncoding) -> None:
        self.client = client
        self.screenshot_encoding = screenshot_encoding

    async def screenshot(self, region: tuple[int, int, int, int] | None = None) -> ToolResult:
        """A screenshot as one ``ImageContent`` block, cropped to ``region`` when given."""
        if region is None:
            data, mime_type = await self.client.screenshot(self.screenshot_encoding)
        else:
            png, _ = await self.client.screenshot("image/png")
            cropped = Image.open(BytesIO(png)).crop(region)
            buffer = BytesIO()
            if isinstance(self.screenshot_encoding, WebPScreenshotEncoding):
                cropped.save(buffer, format="WEBP", quality=self.screenshot_encoding.quality)
            else:
                cropped.save(buffer, format="PNG")
            data, mime_type = buffer.getvalue(), self.screenshot_encoding.mime_type
        return ToolResult(
            content=[
                ImageContent(
                    type="image", mimeType=mime_type, data=base64.b64encode(data).decode("ascii")
                )
            ],
        )

    async def move(self, x: int, y: int) -> None:
        self.client.move(x, y)
        await self.client.drain()

    async def click(
        self,
        at: Point | None = None,
        *,
        button: Button = "left",
        count: int = 1,
        interval: float = 0,
        hold_keys: Iterable[str] = (),
    ) -> None:
        """Click ``count`` times, ``interval`` seconds apart, at ``at`` or where the pointer is."""
        if at is not None:
            self.client.move(*at)
        with self.client.holding_keys(*hold_keys):
            for i in range(count):
                if i:
                    await asyncio.sleep(interval)
                self.client.click(_BUTTON_INDEX[button])
        await self.client.drain()

    async def mouse_down(self, button: Button = "left") -> None:
        """Press ``button`` without releasing (for cross-turn drag sequences)."""
        self.client.button_down(_BUTTON_INDEX[button])
        await self.client.drain()

    async def mouse_up(self, button: Button = "left") -> None:
        self.client.button_up(_BUTTON_INDEX[button])
        await self.client.drain()

    async def scroll(
        self,
        at: Point | None = None,
        *,
        scroll_x: int = 0,
        scroll_y: int = 0,
        hold_keys: Iterable[str] = (),
    ) -> None:
        """Scroll at ``at``; positive ``scroll_x`` and ``scroll_y`` go right and down."""
        if at is not None:
            self.client.move(*at)
        with self.client.holding_keys(*hold_keys):
            self.client.scroll(scroll_y)
            self.client.scroll(scroll_x, horizontal=True)
        await self.client.drain()

    async def drag(
        self,
        path: list[Point],
        *,
        button: Button = "left",
        hold_keys: Iterable[str] = (),
    ) -> None:
        """Press ``button`` at ``path[0]``, move through each later point, then release.

        Paced, because pages render at low frame rates and sample pointer state per frame: they
        must see the press before movement starts, and read a one-frame burst as a click.
        """
        index = _BUTTON_INDEX[button]
        with self.client.holding_keys(*hold_keys):
            x, y = path[0]
            self.client.move(x, y)
            await self.client.drain()
            await asyncio.sleep(0.2)
            with self.client.holding_button(index):
                await self.client.drain()
                await asyncio.sleep(0.2)
                for nx, ny in path[1:]:
                    steps = max(1, min(24, max(abs(nx - x), abs(ny - y)) // 16))
                    for i in range(1, steps + 1):
                        self.client.move(x + (nx - x) * i // steps, y + (ny - y) * i // steps)
                        await self.client.drain()
                        await asyncio.sleep(0.025)
                    x, y = nx, ny
                await asyncio.sleep(0.2)
        await self.client.drain()

    async def type_text(self, text: str) -> None:
        self.client.type_text(text)
        await self.client.drain()

    async def press_keys(self, keys: Iterable[str]) -> None:
        """Press a chord of keysyms, such as ``['Control_L', 'c']``, and release it."""
        self.client.press_keys(*keys)
        await self.client.drain()

    async def key_down(self, key: str) -> None:
        self.client.key_down(key)
        await self.client.drain()

    async def key_up(self, key: str) -> None:
        self.client.key_up(key)
        await self.client.drain()

    async def hold_key(self, key: str, *, duration: float) -> None:
        """Hold a single key for ``duration`` seconds, then release it."""
        with self.client.holding_keys(key):
            await asyncio.sleep(duration)
        await self.client.drain()

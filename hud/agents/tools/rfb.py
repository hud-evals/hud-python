"""RFBTool: capability base for tools driven by an ``RFBClient``.

Provides primitive HID + framebuffer verbs (``screenshot``, ``move``, ``click``,
``type_text``, ``press_keys``, ``scroll``, ``drag``, ``wait``) on top of
``asyncvnc``. Provider tools (``ClaudeComputerTool``, ``GeminiComputerTool``,
``OpenAIComputerTool``) extend this with the LLM-facing action schema and
translate the LLM's call into these primitives.
"""

from __future__ import annotations

import asyncio
import base64
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

import mcp.types as mcp_types

from hud.agents.tools.base import AgentTool, AgentToolSpec
from hud.capabilities import RFBClient
from hud.capabilities.rfb import PngScreenshotEncoding, ScreenshotEncoding
from hud.types import MCPToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable


#: VNC button index (asyncvnc uses 0 = left, 1 = middle, 2 = right, 3 = wheel-up, 4 = wheel-down).
Button = Literal["left", "middle", "right"]
_BUTTON_INDEX: dict[Button, int] = {"left": 0, "middle": 1, "right": 2}
_DEFAULT_SCREENSHOT_ENCODING = PngScreenshotEncoding()
#: Keysym names for control characters missing from asyncvnc's char keymap.
_KEY_FOR_CHAR: dict[str, str] = {"\n": "Return", "\r": "Return", "\t": "Tab"}
_KEY_CHAR_SPLIT = re.compile(r"([\n\r\t])")


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

    # ─── geometry ────────────────────────────────────────────────────

    @property
    def display_width(self) -> int:
        return self.client.width

    @property
    def display_height(self) -> int:
        return self.client.height

    # ─── framebuffer ─────────────────────────────────────────────────

    async def screenshot(self) -> MCPToolResult:
        """Capture a screenshot and return it as a single ``ImageContent`` block."""
        screenshot, mime_type = await self.client.screenshot_png(self.screenshot_encoding)
        return MCPToolResult(
            content=[
                mcp_types.ImageContent(
                    type="image",
                    mimeType=mime_type,
                    data=base64.b64encode(screenshot).decode("ascii"),
                )
            ],
        )

    # ─── pointer ─────────────────────────────────────────────────────

    async def move(self, x: int, y: int) -> None:
        self.client.conn.mouse.move(int(x), int(y))
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
            self.client.conn.mouse.move(int(x), int(y))
        index = _BUTTON_INDEX[button]
        async with self._with_keys(hold_keys):
            for i in range(max(1, count)):
                if i and interval_ms:
                    await asyncio.sleep(interval_ms / 1000)
                self.client.conn.mouse.click(index)
        await self.client.drain()

    async def mouse_down(self, button: Button = "left") -> None:
        """Press ``button`` without releasing (for cross-turn drag sequences)."""
        mouse = self.client.conn.mouse
        mouse.buttons |= 1 << _BUTTON_INDEX[button]
        await self._send_pointer()

    async def mouse_up(self, button: Button = "left") -> None:
        """Release ``button`` (paired with a prior ``mouse_down``)."""
        mouse = self.client.conn.mouse
        mouse.buttons &= ~(1 << _BUTTON_INDEX[button])
        await self._send_pointer()

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
            self.client.conn.mouse.move(int(x), int(y))
        async with self._with_keys(hold_keys):
            if scroll_y > 0:
                self.client.conn.mouse.scroll_down(scroll_y)
            elif scroll_y < 0:
                self.client.conn.mouse.scroll_up(-scroll_y)
            # asyncvnc has no horizontal scroll; ignore scroll_x silently for now.
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
        mouse = self.client.conn.mouse
        index = _BUTTON_INDEX[button]
        async with self._with_keys(hold_keys):
            x, y = int(path[0][0]), int(path[0][1])
            mouse.move(x, y)
            await self.client.drain()
            await asyncio.sleep(0.2)
            with mouse.hold(index):
                await self.client.drain()
                # The page must see the press for a frame before movement
                # starts; environments render at low frame rates.
                await asyncio.sleep(0.2)
                for px, py in path[1:]:
                    nx, ny = int(px), int(py)
                    steps = max(1, min(24, max(abs(nx - x), abs(ny - y)) // 16))
                    for i in range(1, steps + 1):
                        mouse.move(x + (nx - x) * i // steps, y + (ny - y) * i // steps)
                        await self.client.drain()
                        await asyncio.sleep(0.025)
                    x, y = nx, ny
                await asyncio.sleep(0.2)
        await self.client.drain()

    # ─── keyboard ────────────────────────────────────────────────────

    async def type_text(self, text: str) -> None:
        """Type a literal string, one key at a time."""
        keyboard = self.client.conn.keyboard
        for chunk in _KEY_CHAR_SPLIT.split(text.replace("\r\n", "\n")):
            if not chunk:
                continue
            if chunk in _KEY_FOR_CHAR:
                keyboard.press(_KEY_FOR_CHAR[chunk])
            else:
                keyboard.write(chunk)
        await self.client.drain()

    async def press_keys(self, keys: Iterable[str], *, count: int = 1) -> None:
        """Press a chord of keys (e.g. ``['Control_L', 'c']``) ``count`` times."""
        key_list = list(keys)
        for _ in range(max(1, count)):
            self.client.conn.keyboard.press(*key_list)
        await self.client.drain()

    async def hold_key(self, key: str, *, duration_ms: int) -> None:
        """Hold a single key for ``duration_ms`` then release."""
        with self.client.conn.keyboard.hold(key):
            await asyncio.sleep(duration_ms / 1000)
        await self.client.drain()

    # ─── timing ──────────────────────────────────────────────────────

    @staticmethod
    async def wait(duration_ms: int) -> None:
        await asyncio.sleep(duration_ms / 1000)

    # ─── internal ────────────────────────────────────────────────────

    async def _send_pointer(self) -> None:
        """Emit one RFB ``PointerEvent`` (msg type 5) reflecting current mouse state.

        Written directly to the wire because asyncvnc's ``Mouse`` API only
        exposes whole click/hold semantics, not split press/release — which
        Claude's ``left_mouse_down`` / ``left_mouse_up`` actions need.
        """
        mouse = self.client.conn.mouse
        self.client.conn.writer.write(
            b"\x05"
            + mouse.buttons.to_bytes(1, "big")
            + mouse.x.to_bytes(2, "big")
            + mouse.y.to_bytes(2, "big"),
        )
        await self.client.drain()

    @asynccontextmanager
    async def _with_keys(self, keys: Iterable[str] | None) -> AsyncIterator[None]:
        if not keys:
            yield
            return
        with self.client.conn.keyboard.hold(*keys):
            yield


__all__ = ["Button", "RFBTool"]

"""A minimal RFB (VNC) server: a fixed framebuffer that records input events.

It speaks RFB 3.8 with no authentication and Raw encoding, enough for
``asyncvnc`` and therefore ``hud.capabilities.RFBClient`` and the computer-use
tools built on it.
"""

from __future__ import annotations

import asyncio
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass(frozen=True)
class KeyEvent:
    key: int
    down: bool


@dataclass(frozen=True)
class PointerEvent:
    x: int
    y: int
    buttons: int


@dataclass
class FakeScreen:
    """The server side of a VNC session. ``events`` holds every key and pointer event received."""

    width: int = 64
    height: int = 48
    color: tuple[int, int, int] = (40, 120, 200)
    events: list[KeyEvent | PointerEvent] = field(default_factory=list)
    connections: int = 0
    port: int = 0

    @property
    def url(self) -> str:
        return f"rfb://127.0.0.1:{self.port}"

    def key_events(self) -> list[KeyEvent]:
        return [event for event in self.events if isinstance(event, KeyEvent)]

    def pointer_events(self) -> list[PointerEvent]:
        return [event for event in self.events if isinstance(event, PointerEvent)]

    async def _session(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            writer.write(b"RFB 003.008\n")
            await reader.readexactly(12)
            writer.write(b"\x01\x01")  # one security type: None
            await reader.readexactly(1)
            writer.write(struct.pack(">I", 0))
            await reader.readexactly(1)  # ClientInit
            name = b"hud-fake-screen"
            pixel_format = struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 0, 8, 16)
            writer.write(struct.pack(">HH", self.width, self.height) + pixel_format)
            writer.write(struct.pack(">I", len(name)) + name)
            await writer.drain()
            while True:
                (kind,) = await reader.readexactly(1)
                if kind == 0:  # SetPixelFormat
                    await reader.readexactly(19)
                elif kind == 2:  # SetEncodings
                    _, count = struct.unpack(">BH", await reader.readexactly(3))
                    await reader.readexactly(4 * count)
                elif kind == 3:  # FramebufferUpdateRequest
                    await reader.readexactly(9)
                    pixel = bytes([*self.color, 255])
                    writer.write(struct.pack(">BxH", 0, 1))
                    writer.write(struct.pack(">HHHHi", 0, 0, self.width, self.height, 0))
                    writer.write(pixel * (self.width * self.height))
                    await writer.drain()
                elif kind == 4:  # KeyEvent
                    down, _, key = struct.unpack(">B2sI", await reader.readexactly(7))
                    self.events.append(KeyEvent(key, bool(down)))
                elif kind == 5:  # PointerEvent
                    buttons, x, y = struct.unpack(">BHH", await reader.readexactly(5))
                    self.events.append(PointerEvent(x, y, buttons))
                elif kind == 6:  # ClientCutText
                    _, length = struct.unpack(">3sI", await reader.readexactly(7))
                    await reader.readexactly(length)
                else:
                    raise ValueError(f"unexpected RFB client message {kind}")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


@asynccontextmanager
async def fake_screen(
    width: int = 64, height: int = 48, color: tuple[int, int, int] = (40, 120, 200)
) -> AsyncIterator[FakeScreen]:
    """Serve a :class:`FakeScreen` on loopback for the duration of the block."""
    screen = FakeScreen(width, height, color)
    server = await asyncio.start_server(screen._session, "127.0.0.1", 0)
    screen.port = server.sockets[0].getsockname()[1]
    async with server:
        yield screen

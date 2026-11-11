"""A minimal RFB (VNC) server: a fixed framebuffer that records input events.

It speaks RFB 3.8 with no authentication or a VNC password, and Raw or ZLib
encoding, enough for ``hud.capabilities.RFBClient`` and the computer-use tools
built on it.
"""

from __future__ import annotations

import asyncio
import os
import struct
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass(frozen=True)
class KeyEvent:
    key: int
    down: bool


@dataclass(frozen=True)
class PointerEvent:
    x: int
    y: int
    buttons: int


def _vnc_response(password: str, challenge: bytes) -> bytes:
    """The DES answer RFB's VNC authentication expects: each key byte's bits mirrored."""
    key = password.encode("latin-1")[:8].ljust(8, b"\x00")
    mirrored = bytes(sum(((byte >> bit) & 1) << (7 - bit) for bit in range(8)) for byte in key)
    encryptor = Cipher(TripleDES(mirrored * 3), modes.ECB()).encryptor()
    return encryptor.update(challenge) + encryptor.finalize()


@dataclass
class FakeScreen:
    """The server side of a VNC session. ``events`` holds every key and pointer event received."""

    width: int = 64
    height: int = 48
    color: tuple[int, int, int] = (40, 120, 200)
    events: list[KeyEvent | PointerEvent] = field(default_factory=list)
    connections: int = 0
    port: int = 0
    # Close the connection instead of answering this framebuffer update request
    # (counted across connections, from 1); 0 never drops.
    drop_on_update: int = 0
    # Require this VNC password (security type 2); None offers no authentication.
    password: str | None = None
    updates: int = 0
    sessions: set[asyncio.Task[None]] = field(default_factory=set)

    @property
    def url(self) -> str:
        return f"rfb://127.0.0.1:{self.port}"

    def key_events(self) -> list[KeyEvent]:
        return [event for event in self.events if isinstance(event, KeyEvent)]

    def pointer_events(self) -> list[PointerEvent]:
        return [event for event in self.events if isinstance(event, PointerEvent)]

    async def session(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        session = asyncio.current_task()
        assert session is not None
        self.sessions.add(session)
        try:
            writer.write(b"RFB 003.008\n")
            await reader.readexactly(12)
            writer.write(b"\x01\x02" if self.password else b"\x01\x01")
            await reader.readexactly(1)
            if self.password is not None:
                challenge = os.urandom(16)
                writer.write(challenge)
                if await reader.readexactly(16) != _vnc_response(self.password, challenge):
                    reason = b"wrong password"
                    writer.write(struct.pack(">II", 1, len(reason)) + reason)
                    await writer.drain()
                    return
            writer.write(struct.pack(">I", 0))
            await reader.readexactly(1)  # ClientInit
            name = b"hud-fake-screen"
            pixel_format = struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 0, 8, 16)
            writer.write(struct.pack(">HH", self.width, self.height) + pixel_format)
            writer.write(struct.pack(">I", len(name)) + name)
            await writer.drain()
            deflate = zlib.compressobj()
            zlib_encoding = False
            while True:
                (kind,) = await reader.readexactly(1)
                if kind == 0:  # SetPixelFormat
                    await reader.readexactly(19)
                elif kind == 2:  # SetEncodings
                    _, count = struct.unpack(">BH", await reader.readexactly(3))
                    encodings = struct.unpack(f">{count}i", await reader.readexactly(4 * count))
                    zlib_encoding = 6 in encodings
                elif kind == 3:  # FramebufferUpdateRequest
                    await reader.readexactly(9)
                    self.updates += 1
                    if self.updates == self.drop_on_update:
                        return
                    pixel = bytes([*self.color, 255])
                    pixels = pixel * (self.width * self.height)
                    writer.write(struct.pack(">BxH", 0, 1))
                    if not zlib_encoding:
                        writer.write(struct.pack(">HHHHi", 0, 0, self.width, self.height, 0))
                        writer.write(pixels)
                    else:
                        data = deflate.compress(pixels) + deflate.flush(zlib.Z_SYNC_FLUSH)
                        writer.write(
                            struct.pack(">HHHHiI", 0, 0, self.width, self.height, 6, len(data))
                        )
                        writer.write(data)
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
            self.sessions.discard(session)


@asynccontextmanager
async def fake_screen(
    width: int = 64,
    height: int = 48,
    color: tuple[int, int, int] = (40, 120, 200),
    *,
    drop_on_update: int = 0,
    password: str | None = None,
) -> AsyncGenerator[FakeScreen]:
    """Serve a :class:`FakeScreen` on loopback for the duration of the block."""
    screen = FakeScreen(width, height, color, drop_on_update=drop_on_update, password=password)
    server = await asyncio.start_server(screen.session, "127.0.0.1", 0)
    screen.port = server.sockets[0].getsockname()[1]
    async with server:
        yield screen
        # Read what a client sent before it hung up; 3.11's server exit does not wait.
        await asyncio.gather(*screen.sessions)

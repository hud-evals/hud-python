"""RFBClient: a VNC (RFB 3.8) session with screenshots, pointer and keyboard.

Latency note
------------
This impl is tuned for LLM-driven agents (Claude/Gemini/OpenAI computer
use), where the model thinks for seconds per turn and a ~30-70 ms screenshot
round-trip is irrelevant.

It is **not** sufficient for the FDM-1 style video-model hot path
(https://si.inc/posts/fdm1/) which targets ~11 ms round-trip. Reaching that
requires: a Tight/ZRLE-capable transport (this client speaks only Raw + ZLib),
incremental framebuffer streaming with a background reader task, raw RGBA
frames (no PNG re-encoding), and native input bindings. When that workload
shows up, layer it as an ``RFBStreamingClient`` subclass rather than rewriting
this one.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self
from urllib.parse import urlsplit

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from keysymdef import keysymdef
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import Capability, CapabilityClient

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

LOGGER = logging.getLogger("hud.capabilities.rfb")

ScreenshotMimeType = Literal["image/png", "image/webp"]

# X11 keysyms by name (``Return``, ``Control_L``, ``F5``) and by the character they type.
_KEYSYMS: dict[str, int] = {name: code for name, code, _ in keysymdef}
_KEYSYMS.update((chr(char), code) for _, code, char in keysymdef if char)
_KEYSYMS.update({"\n": _KEYSYMS["Return"], "\r": _KEYSYMS["Return"], "\t": _KEYSYMS["Tab"]})


class PngScreenshotEncoding(BaseModel):
    model_config = ConfigDict(frozen=True)

    mime_type: Literal["image/png"] = "image/png"


class WebPScreenshotEncoding(BaseModel):
    model_config = ConfigDict(frozen=True)

    mime_type: Literal["image/webp"] = "image/webp"
    quality: int = Field(default=85, ge=0, le=100)


ScreenshotEncoding = Annotated[
    PngScreenshotEncoding | WebPScreenshotEncoding,
    Field(discriminator="mime_type"),
]


@dataclass
class _Session:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    width: int
    height: int
    inflate: Callable[[bytes], bytes]


async def _read_text(reader: asyncio.StreamReader) -> str:
    (length,) = struct.unpack(">I", await reader.readexactly(4))
    return (await reader.readexactly(length)).decode("utf-8", errors="replace")


async def _open(capability: Capability) -> _Session:
    """Connect, authenticate and initialise an RFB 3.8 session for ``capability``."""
    parts = urlsplit(capability.url)
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"rfb capability missing host or port: {capability.url!r}")
    password = capability.params.get("password")
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
    try:
        greeting = await reader.readline()
        if not greeting.startswith(b"RFB "):
            raise ConnectionError(f"{capability.url} is not a VNC server (it sent {greeting!r})")
        writer.write(b"RFB 003.008\n")
        (count,) = await reader.readexactly(1)
        offered = set(await reader.readexactly(count))
        if not offered:
            raise ConnectionError(
                f"the VNC server refused the connection: {await _read_text(reader)}"
            )
        if 1 in offered:  # None
            writer.write(b"\x01")
        elif 2 in offered:  # VNC password
            if password is None:
                raise PermissionError("the VNC server requires a password")
            writer.write(b"\x02")
            key = password.encode("latin-1")[:8].ljust(8, b"\x00")
            key = bytes(int(f"{byte:08b}"[::-1], 2) for byte in key)
            encryptor = Cipher(TripleDES(key * 3), modes.ECB()).encryptor()  # noqa: S305
            writer.write(encryptor.update(await reader.readexactly(16)) + encryptor.finalize())
        else:
            raise ConnectionError(
                f"the VNC server offers security types {sorted(offered)}; hud supports "
                "None (1) and VNC password (2)"
            )
        (result,) = struct.unpack(">I", await reader.readexactly(4))
        if result != 0:
            raise PermissionError(f"VNC authentication failed: {await _read_text(reader)}")

        writer.write(b"\x01")
        width, height = struct.unpack(">HH", await reader.readexactly(4))
        await reader.readexactly(16)
        await _read_text(reader)
        # SetPixelFormat: 32 bits per pixel, depth 24, little-endian true colour with red,
        # green and blue in the low three bytes, so each pixel arrives as R, G, B, unused.
        writer.write(
            b"\x00\x00\x00\x00"
            + struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 0, 8, 16)
        )
        # SetEncodings: ZLib (6); Raw is always allowed.
        writer.write(struct.pack(">BxHi", 2, 1, 6))
        await writer.drain()
    except BaseException:
        writer.close()
        raise
    return _Session(reader, writer, width, height, zlib.decompressobj().decompress)


class RFBClient(CapabilityClient):
    """Live VNC/RFB connection: screenshots, and the pointer and keyboard it drives."""

    protocol: ClassVar[str] = "rfb/3.8"

    def __init__(self, capability: Capability, session: _Session) -> None:
        self.capability = capability
        self._session = session
        self._x = 0
        self._y = 0
        self._buttons = 0

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        client = cls(cap, await _open(cap))
        try:
            # Warm up: the first full refresh gives later captures real content.
            await client._capture()
        except BaseException:
            await client.close()
            raise
        return client

    async def _reconnect(self) -> None:
        """Tear down the current VNC session and open a fresh one."""
        LOGGER.info("RFB stream desynced; reconnecting to %s", self.capability.url)
        await self.close()
        self._session = await _open(self.capability)
        await self._capture()

    @property
    def width(self) -> int:
        return self._session.width

    @property
    def height(self) -> int:
        return self._session.height

    @property
    def cursor(self) -> tuple[int, int]:
        """Where the pointer was last moved, in screen pixels."""
        return self._x, self._y

    # ─── framebuffer ─────────────────────────────────────────────────

    async def _capture(self) -> Image.Image:
        """Request the whole framebuffer and read updates until every pixel has arrived."""
        session = self._session
        size = (session.width, session.height)
        session.writer.write(struct.pack(">BBHHHH", 3, 0, 0, 0, *size))
        frame = Image.new("RGB", size)
        covered = Image.new("L", size)
        while covered.getextrema() != (255, 255):
            await self._read_message(frame, covered)
        return frame

    async def _read_message(self, frame: Image.Image, covered: Image.Image) -> None:
        reader = self._session.reader
        (kind,) = await reader.readexactly(1)
        match kind:
            case 0:  # FramebufferUpdate
                _, rectangles = struct.unpack(">BH", await reader.readexactly(3))
                for _ in range(rectangles):
                    x, y, w, h, encoding = struct.unpack(">HHHHi", await reader.readexactly(12))
                    match encoding:
                        case 0:  # Raw
                            pixels = await reader.readexactly(w * h * 4)
                        case 6:  # ZLib
                            (length,) = struct.unpack(">I", await reader.readexactly(4))
                            pixels = self._session.inflate(await reader.readexactly(length))
                        case _:
                            raise ConnectionError(
                                f"the VNC server sent unrequested encoding {encoding}"
                            )
                    frame.paste(Image.frombytes("RGB", (w, h), pixels, "raw", "RGBX"), (x, y))
                    covered.paste(255, (x, y, x + w, y + h))
            case 2:  # Bell
                pass
            case 3:  # ServerCutText
                await reader.readexactly(3)
                (length,) = struct.unpack(">I", await reader.readexactly(4))
                await reader.readexactly(length)
            case _:
                raise ConnectionError(f"the VNC server sent unknown message type {kind}")

    async def screenshot_png(
        self,
        encoding: ScreenshotEncoding | ScreenshotMimeType = "image/png",
    ) -> tuple[bytes, ScreenshotMimeType]:
        """Capture the framebuffer in the requested format; reconnect on desync."""
        if isinstance(encoding, str):
            encoding = (
                WebPScreenshotEncoding() if encoding == "image/webp" else PngScreenshotEncoding()
            )
        for attempt in range(3):
            try:
                image = await self._capture()
                buf = io.BytesIO()
                if isinstance(encoding, WebPScreenshotEncoding):
                    image.save(buf, format="WEBP", quality=encoding.quality)
                else:
                    image.save(buf, format="PNG")
                return buf.getvalue(), encoding.mime_type
            except (ValueError, OSError, EOFError, zlib.error) as exc:
                LOGGER.warning("screenshot failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 2:
                    raise
                await self._reconnect()
        raise RuntimeError("unreachable")

    # ─── pointer ─────────────────────────────────────────────────────

    def _pointer(self) -> None:
        self._session.writer.write(struct.pack(">BBHH", 5, self._buttons, self._x, self._y))

    def move(self, x: int, y: int) -> None:
        """Move the pointer to ``(x, y)``, keeping any held buttons down."""
        self._x, self._y = x, y
        self._pointer()

    def button_down(self, button: int) -> None:
        """Press pointer button ``button`` (0 left, 1 middle, 2 right) and keep it down."""
        self._buttons |= 1 << button
        self._pointer()

    def button_up(self, button: int) -> None:
        """Release pointer button ``button``."""
        self._buttons &= ~(1 << button)
        self._pointer()

    @contextlib.contextmanager
    def holding_button(self, button: int) -> Generator[None]:
        """Hold ``button`` down for the block."""
        self.button_down(button)
        try:
            yield
        finally:
            self.button_up(button)

    def click(self, button: int) -> None:
        """Press and release ``button`` where the pointer is."""
        with self.holding_button(button):
            pass

    def scroll(self, clicks: int) -> None:
        """Turn the wheel (buttons 3 up, 4 down) ``clicks`` notches; positive scrolls down."""
        for _ in range(abs(clicks)):
            self.click(4 if clicks > 0 else 3)

    # ─── keyboard ────────────────────────────────────────────────────

    def _key(self, code: int, *, down: bool) -> None:
        self._session.writer.write(struct.pack(">BBxxI", 4, down, code))

    def key_down(self, key: str) -> None:
        """Press ``key`` (an X11 keysym name or a single character) and keep it down."""
        self._key(_KEYSYMS[key], down=True)

    def key_up(self, key: str) -> None:
        """Release ``key``."""
        self._key(_KEYSYMS[key], down=False)

    @contextlib.contextmanager
    def holding_keys(self, *keys: str) -> Generator[None]:
        """Press ``keys`` in order for the block, then release them in reverse."""
        codes = [_KEYSYMS[key] for key in keys]
        for code in codes:
            self._key(code, down=True)
        try:
            yield
        finally:
            for code in reversed(codes):
                self._key(code, down=False)

    def press_keys(self, *keys: str) -> None:
        """Press a chord of ``keys`` and release it."""
        with self.holding_keys(*keys):
            pass

    def type_text(self, text: str) -> None:
        """Type ``text`` one character at a time; newlines press Return and tabs Tab."""
        codes = [_KEYSYMS[char] for char in text.replace("\r\n", "\n")]
        for code in codes:
            self._key(code, down=True)
            self._key(code, down=False)

    async def drain(self) -> None:
        """Flush queued pointer and keyboard events to the server."""
        await self._session.writer.drain()

    async def close(self) -> None:
        writer = self._session.writer
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()

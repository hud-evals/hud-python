"""RFBClient — asyncvnc connection wrapper.

Thin wrapper exposing the live ``asyncvnc.Client`` plus PNG-encoded
screenshots. Higher-level composites (click, type, drag) live on ``RFBTool``.

Latency note
------------
This impl is tuned for LLM-driven agents (Claude/Gemini/OpenAI computer
use), where the model thinks for seconds per turn and a ~30-70 ms screenshot
round-trip is irrelevant.

It is **not** sufficient for the FDM-1 style video-model hot path
(https://si.inc/posts/fdm1/) which targets ~11 ms round-trip. Reaching that
requires: a Tight/ZRLE-capable transport (asyncvnc speaks only Raw + ZLib),
incremental framebuffer streaming with a background reader task, raw RGBA
frames (no PNG re-encoding), and native input bindings. When that workload
shows up, layer it as an ``RFBStreamingClient`` subclass rather than rewriting
this one.
"""

from __future__ import annotations

import contextlib
import io
import logging
from contextlib import AsyncExitStack
from typing import ClassVar, Self
from urllib.parse import urlsplit

import asyncvnc
from PIL import Image

from .base import Capability, CapabilityClient

LOGGER = logging.getLogger("hud.capabilities.rfb")


class RFBClient(CapabilityClient):
    """Live VNC/RFB connection. Exposes raw ``asyncvnc.Client`` via ``conn``."""

    protocol: ClassVar[str] = "rfb/3.8"

    def __init__(
        self,
        capability: Capability,
        conn: asyncvnc.Client,
        exit_stack: AsyncExitStack,
    ) -> None:
        self.capability = capability
        self._conn = conn
        self._exit_stack = exit_stack
        parts = urlsplit(capability.url)
        assert parts.hostname is not None and parts.port is not None  # connect() validated
        self._host = parts.hostname
        self._port = parts.port
        self._user = capability.params.get("user")
        self._password = capability.params.get("password")

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"rfb capability missing host or port: {cap.url!r}")
        stack = AsyncExitStack()
        conn = await cls._open(
            stack, parts.hostname, parts.port, cap.params.get("user"), cap.params.get("password")
        )
        return cls(cap, conn, stack)

    @staticmethod
    async def _open(
        stack: AsyncExitStack,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
    ) -> asyncvnc.Client:
        conn = await stack.enter_async_context(
            asyncvnc.connect(host=host, port=port, username=user, password=password),
        )
        # Warm up — first screenshot resets the framebuffer and forces a full
        # (non-incremental) refresh so later captures have real content.
        await conn.screenshot()
        return conn

    async def _reconnect(self) -> None:
        """Tear down the current VNC session and open a fresh one."""
        LOGGER.info("RFB stream desynced; reconnecting to %s:%s", self._host, self._port)
        with contextlib.suppress(Exception):
            await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._conn = await self._open(
            self._exit_stack,
            self._host,
            self._port,
            self._user,
            self._password,
        )

    @property
    def conn(self) -> asyncvnc.Client:
        """Raw asyncvnc client — use for direct mouse/keyboard/clipboard access."""
        return self._conn

    @property
    def width(self) -> int:
        return self._conn.video.width

    @property
    def height(self) -> int:
        return self._conn.video.height

    async def screenshot_png(self) -> bytes:
        """Capture the framebuffer as PNG bytes; reconnect+retry on desync."""
        for attempt in range(3):
            try:
                rgba = await self._conn.screenshot()
                image = Image.fromarray(rgba)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                return buf.getvalue()
            except (ValueError, ConnectionError, OSError, EOFError) as exc:
                LOGGER.warning("screenshot failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 2:
                    raise
                await self._reconnect()
        raise RuntimeError("unreachable")

    async def drain(self) -> None:
        """Flush any queued mouse/keyboard writes to the server."""
        await self._conn.drain()

    async def close(self) -> None:
        await self._exit_stack.aclose()


__all__ = ["RFBClient"]

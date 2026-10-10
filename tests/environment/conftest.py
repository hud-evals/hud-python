"""Raw control-channel access for the environment scenarios.

The control channel is newline-delimited JSON-RPC over TCP. :class:`Wire` speaks
it byte for byte, so a scenario sees exactly the frames a peer would.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from dirty_equals import IsStr

# The sandbox area's tests import ``served`` from here; it lives in the harness.
from tests.harness import served

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["FRAME_LIMIT", "SESSION_ID", "Wire", "encode", "served", "wire"]

#: A control session id as the server mints it.
SESSION_ID = IsStr(regex=r"sess-[0-9a-f]{8}")

#: The control channel's frame limit, in bytes, excluding the newline.
FRAME_LIMIT = 16 * 1024 * 1024


def encode(frame: dict[str, Any]) -> bytes:
    """One frame as the control channel carries it."""
    return json.dumps(frame, separators=(",", ":")).encode() + b"\n"


class Wire:
    """One TCP connection to a served control channel."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self._ids = 0

    async def send(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._ids += 1
        frame: dict[str, Any] = {"jsonrpc": "2.0", "id": self._ids, "method": method}
        if params is not None:
            frame["params"] = params
        await self.write(encode(frame))

    async def write(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def read(self) -> dict[str, Any] | None:
        """The next reply frame, or ``None`` when the server hung up."""
        line = await asyncio.wait_for(self.reader.readline(), timeout=30)
        return json.loads(line) if line else None

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.send(method, params)
        reply = await self.read()
        assert reply is not None, f"the server hung up on {method}"
        return reply

    async def hang_up(self) -> None:
        """Close the sending side and wait until the server has ended the connection."""
        self.writer.write_eof()
        while await asyncio.wait_for(self.reader.read(65536), timeout=30):
            pass
        await self.close()

    async def close(self) -> None:
        self.writer.close()
        with contextlib.suppress(ConnectionError):
            await self.writer.wait_closed()


@asynccontextmanager
async def wire(url: str) -> AsyncIterator[Wire]:
    """Open a raw connection to the control channel at ``tcp://host:port``."""
    address = urlsplit(url)
    reader, writer = await asyncio.open_connection(
        address.hostname, address.port, limit=2 * FRAME_LIMIT
    )
    connection = Wire(reader, writer)
    try:
        yield connection
    finally:
        await connection.close()

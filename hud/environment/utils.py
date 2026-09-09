"""Shared env helpers: JSON-RPC framing + byte splicing for the control channel."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

CONTROL_FRAME_LIMIT_BYTES = 16 * 1024 * 1024


class FrameTooLargeError(ValueError):
    """An encoded JSON-RPC frame exceeds its channel's byte limit."""


# ─── JSON-RPC 2.0 framing ───


async def send_frame(
    writer: asyncio.StreamWriter,
    msg: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> None:
    """Write one newline-delimited JSON frame and flush."""
    data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        frame = f"{msg['method']!r} request" if "method" in msg else "response"
        raise FrameTooLargeError(
            f"Control-channel {frame} is {len(data)} bytes; limit is {max_bytes} bytes "
            "(excluding the newline). Reduce inline JSON; pass large task data by file ID "
            "and load it inside the environment."
        )
    writer.write(data + b"\n")
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one frame; None on EOF."""
    line = await reader.readline()
    if not line:
        return None
    return json.loads(line)


def reply(msg_id: int, result: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error(msg_id: int, code: int, message: str) -> dict[str, Any]:
    """JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ─── byte splicing (tunneled capability connections) ───


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    # Resets/aborts are a normal way for tunneled streams to end (an SSH
    # client hanging up, a container dying); they end the pump, not the world.
    with contextlib.suppress(OSError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()


async def splice(
    a: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    b: tuple[asyncio.StreamReader, asyncio.StreamWriter],
) -> None:
    """Pipe two byte streams into each other until both directions hit EOF.

    Closes both writers on the way out — under Python 3.12 an unclosed
    connection parks ``Server.wait_closed()`` forever.
    """

    async def _drain_close() -> None:
        for writer in (a[1], b[1]):
            writer.close()
        for writer in (a[1], b[1]):
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    try:
        await asyncio.gather(_pump(a[0], b[1]), _pump(b[0], a[1]))
    except GeneratorExit:
        # Force-close: the task was abandoned (loop shutdown threw GeneratorExit
        # into us). Awaiting here would raise "coroutine ignored GeneratorExit",
        # so close synchronously and get out.
        for writer in (a[1], b[1]):
            writer.close()
        raise
    except BaseException:
        await _drain_close()
        raise
    else:
        await _drain_close()


__all__ = ["error", "read_frame", "reply", "send_frame", "splice"]

"""Shared env helpers: JSON-RPC framing for the control channel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

# ─── JSON-RPC 2.0 framing ───


async def send_frame(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """Write one newline-delimited JSON frame and flush."""
    writer.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
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


__all__ = ["error", "read_frame", "reply", "send_frame"]

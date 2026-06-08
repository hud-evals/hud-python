"""Newline-delimited JSON-RPC framing for HUD control channels."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger("hud._wire")


class WireProtocolError(ValueError):
    """Malformed JSON-RPC framing on the control channel."""


async def send_frame(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """Write one newline-delimited JSON frame and flush."""
    writer.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one JSON-RPC frame; ``None`` only on EOF."""
    try:
        line = await reader.readline()
    except (ValueError, asyncio.LimitOverrunError):
        raise WireProtocolError("over-long JSON-RPC frame") from None
    if not line:
        return None
    try:
        parsed: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WireProtocolError("malformed JSON-RPC frame") from exc
    if not isinstance(parsed, dict):
        raise WireProtocolError("JSON-RPC frame must be an object")
    return parsed


def reply(msg_id: int, result: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error(msg_id: int, code: int, message: str) -> dict[str, Any]:
    """JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


__all__ = ["WireProtocolError", "error", "read_frame", "reply", "send_frame"]

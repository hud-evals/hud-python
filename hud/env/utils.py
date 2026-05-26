"""Shared helpers: JSON-RPC framing + URL normalization."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

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


# ─── URL helpers ───

#: Matches the scheme prefix of a URL (RFC 3986).
SCHEME_RE: re.Pattern[str] = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*):")


def normalize_url(url: str, *, default_scheme: str, default_port: int | None) -> str:
    """Coerce shorthand ``host[:port]`` into a full ``scheme://host:port[/path]`` URL."""
    s = url if "://" in url else f"{default_scheme}://{url}"
    parts = urlsplit(s)
    if parts.scheme == "":
        raise ValueError(f"invalid URL (no scheme): {url!r}")
    if parts.hostname is None:
        raise ValueError(f"invalid URL (no host): {url!r}")
    if parts.port is None and default_port is not None:
        userinfo = f"{parts.username}@" if parts.username else ""
        path = parts.path
        query = f"?{parts.query}" if parts.query else ""
        fragment = f"#{parts.fragment}" if parts.fragment else ""
        return f"{parts.scheme}://{userinfo}{parts.hostname}:{default_port}{path}{query}{fragment}"
    return s


__all__ = ["SCHEME_RE", "error", "normalize_url", "read_frame", "reply", "send_frame"]

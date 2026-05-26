"""Internal utilities shared across the env package.

Two groups:

* **JSON-RPC 2.0 framing** — `send_frame` / `read_frame` / `reply` / `error`.
  The control channel and any future RPC binding speak the same envelope.
* **URL helpers** — `SCHEME_RE` regex + `normalize_url(...)` for the
  capability factories. Accepts shorthand like ``"127.0.0.1:9090"`` and
  produces a well-formed URL with a scheme + port.

Add more cross-module helpers here as they appear. Per-module private
helpers (SSH key generation, mount-flag table, etc.) stay in their
owning module.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urlsplit


# ─────────────────────────── JSON-RPC 2.0 framing ───────────────────────────


async def send_frame(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """Write a single newline-delimited JSON frame and flush."""
    writer.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one newline-delimited JSON frame; returns None on EOF."""
    line = await reader.readline()
    if not line:
        return None
    return json.loads(line)


def reply(msg_id: int, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error(msg_id: int, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ─────────────────────────── URL helpers ───────────────────────────


#: Matches the scheme portion of a URL per RFC 3986: alpha then alnum/+/-/.
SCHEME_RE: re.Pattern[str] = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*):")


def normalize_url(url: str, *, default_scheme: str, default_port: int | None) -> str:
    """Add ``default_scheme://`` if missing; append ``:default_port`` if missing.

    Accepts shorthand like ``"127.0.0.1:9090"`` or ``"127.0.0.1"`` and
    produces a well-formed URL such as ``"ws://127.0.0.1:9090"``. Raises if
    the URL has no scheme after normalization or no hostname.
    """
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


__all__ = [
    "SCHEME_RE",
    "error",
    "normalize_url",
    "read_frame",
    "reply",
    "send_frame",
]

"""FileTrackingClient — pulls workspace diffs over the ``filetracking/1`` wire.

A tiny framed-JSON request/response client (newline-delimited JSON, one request
in flight at a time). The matching server lives in
:mod:`hud.environment.file_tracking`. Kept dependency-free of the environment
package so importing capabilities never pulls the environment stack.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

from .base import Capability, CapabilityClient


class FileTrackingClient(CapabilityClient):
    """Live ``filetracking/1`` connection: ``diff`` / ``snapshot`` / ``advance``."""

    protocol: ClassVar[str] = "filetracking/1"

    def __init__(
        self,
        capability: Capability,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.capability = capability
        self._reader = reader
        self._writer = writer
        self._id = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"filetracking capability missing host or port: {cap.url!r}")
        reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
        return cls(cap, reader, writer)

    async def diff(self) -> dict[str, Any]:
        """Changes since the previous call (advances the server's baseline)."""
        return await self._call("diff")

    async def snapshot(self) -> dict[str, Any]:
        """The current full manifest: ``{files: [{path, size, content_hash}], ...}``."""
        return await self._call("snapshot")

    async def advance(self) -> None:
        """Re-baseline without producing a diff (skip setup / post-burst churn)."""
        await self._call("advance")

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    async def _call(self, method: str) -> dict[str, Any]:
        async with self._lock:
            self._id += 1
            msg_id = self._id
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": {}},
                separators=(",", ":"),
            )
            self._writer.write(payload.encode("utf-8") + b"\n")
            await self._writer.drain()
            line = await self._reader.readline()
            if not line:
                raise ConnectionError(f"filetracking: connection closed during {method!r}")
            reply: dict[str, Any] = json.loads(line)
            if "error" in reply:
                err = reply["error"]
                raise RuntimeError(f"filetracking {method!r} error: {err.get('message')}")
            result = reply.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"filetracking {method!r}: result was not an object")
            return result


__all__ = ["FileTrackingClient"]

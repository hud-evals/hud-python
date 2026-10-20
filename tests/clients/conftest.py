"""A control-channel peer on the environment's side, answering an SDK client from a script."""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


_DEFAULT_ID = object()


@dataclass(frozen=True)
class Action:
    """What the peer does with the next frame on a connection.

    ``"answer"`` replies with ``result`` or ``error`` under ``reply_id`` (the request's
    id unless given), ``"hang_up"`` closes the connection (before reading when
    ``immediately``), and ``"hold"`` reads on without replying.
    """

    kind: Literal["answer", "hang_up", "hold"]
    result: Any = None
    error: dict[str, Any] | None = None
    reply_id: Any = _DEFAULT_ID
    immediately: bool = False


@dataclass(frozen=True)
class Frame:
    connection: int
    message: dict[str, Any]


@dataclass
class ControlPeer:
    """The peer's record. Use :func:`control_peer`."""

    scripts: Sequence[Sequence[Action]]
    upstreams: dict[str, tuple[str, int]]
    port: int = 0
    #: Control connections accepted, tunnel streams excluded.
    accepted: int = 0
    tunnels: int = 0
    frames: list[Frame] = field(default_factory=list)
    server: asyncio.Server | None = None
    handlers: set[asyncio.Task[None]] = field(default_factory=set)

    @property
    def url(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    def requests(self, method: str | None = None) -> list[dict[str, Any]]:
        """The request frames received, those calling ``method`` when given."""
        return [
            frame.message
            for frame in self.frames
            if method is None or frame.message.get("method") == method
        ]

    def stop_accepting(self) -> None:
        """Close the listening socket; open connections stay up."""
        assert self.server is not None
        self.server.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        try:
            await self._converse(reader, writer)
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            self.handlers.discard(task)
            writer.close()

    async def _converse(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        script = list(self.scripts[min(self.accepted, len(self.scripts) - 1)])
        if script and script[0].kind == "hang_up" and script[0].immediately:
            self.accepted += 1
            return
        first = await _read(reader)
        if first is None:
            self.accepted += 1
            return
        if first.get("method") == "tunnel.open":
            await self._tunnel(first, reader, writer)
            return
        connection = self.accepted
        self.accepted += 1
        message: dict[str, Any] | None = first
        while message is not None:
            self.frames.append(Frame(connection, message))
            action = script.pop(0) if script else Action("hold")
            if action.kind == "hang_up":
                return
            if action.kind == "answer":
                reply_id = message.get("id") if action.reply_id is _DEFAULT_ID else action.reply_id
                body = (
                    {"error": action.error}
                    if action.error is not None
                    else {"result": action.result}
                )
                await _write(writer, {"jsonrpc": "2.0", "id": reply_id, **body})
            message = await _read(reader)

    async def _tunnel(
        self, opened: dict[str, Any], reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.tunnels += 1
        self.frames.append(Frame(-1, opened))
        name = opened.get("params", {}).get("capability")
        upstream = self.upstreams.get(name)
        if upstream is None:
            error = {"code": -32602, "message": f"unknown capability: {name!r}"}
            await _write(writer, {"jsonrpc": "2.0", "id": opened.get("id"), "error": error})
            return
        up_reader, up_writer = await asyncio.open_connection(*upstream)
        await _write(
            writer, {"jsonrpc": "2.0", "id": opened.get("id"), "result": {"capability": name}}
        )
        try:
            await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
        finally:
            up_writer.close()


async def _read(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    line = await reader.readline()
    return json.loads(line) if line else None


async def _write(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(json.dumps(message).encode() + b"\n")
    await writer.drain()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


@asynccontextmanager
async def control_peer(
    *scripts: Sequence[Action], upstreams: dict[str, tuple[str, int]] | None = None
) -> AsyncIterator[ControlPeer]:
    """Serve a :class:`ControlPeer` answering connections from ``scripts`` for the block."""
    peer = ControlPeer(scripts=scripts or ([],), upstreams=upstreams or {})
    server = await asyncio.start_server(peer.handle, "127.0.0.1", 0)
    peer.server = server
    peer.port = server.sockets[0].getsockname()[1]
    try:
        yield peer
    finally:
        server.close()
        for handler in list(peer.handlers):
            handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*peer.handlers, return_exceptions=True)
        await server.wait_closed()

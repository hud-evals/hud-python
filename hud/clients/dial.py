"""Opening a connection to an environment's control channel by its address's scheme.

``tcp://`` (or no scheme) is a plain socket, ``tls://`` the same stream inside TLS,
and ``ws://`` / ``wss://`` a WebSocket whose binary messages carry the stream.
Every scheme yields an asyncio stream pair, so the control protocol above it is
the same everywhere.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import ssl
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosedOK, InvalidStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

LOGGER = logging.getLogger("hud.clients")

Streams = tuple[asyncio.StreamReader, asyncio.StreamWriter]

_CARRIERS: set[asyncio.Task[None]] = set()


def dialer(url: str, params: Mapping[str, Any]) -> Callable[[], Awaitable[Streams]]:
    """The function that opens one connection to ``url``.

    A WebSocket address sends ``params["headers"]`` with its handshake; its
    server answering 5xx is not serving yet, like a refused TCP connect.
    """
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port
    match parts.scheme:
        case "" | "tcp":
            return lambda: asyncio.open_connection(host, port or 0)
        case "tls":
            context = ssl.create_default_context()
            return lambda: asyncio.open_connection(
                host, port or 443, ssl=context, server_hostname=host
            )
        case "ws" | "wss":
            headers = dict(params.get("headers") or {})
            return lambda: _websocket(url, headers)
        case scheme:
            raise NotImplementedError(f"control transport {scheme!r} is not supported")


async def _websocket(url: str, headers: dict[str, str]) -> Streams:
    try:
        websocket = await websockets.connect(url, additional_headers=headers, max_size=None)
    except InvalidStatus as exc:
        if exc.response.status_code >= 500:
            raise ConnectionRefusedError(str(exc)) from exc
        raise
    local, remote = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=local)
    remote_reader, remote_writer = await asyncio.open_connection(sock=remote)
    carrier = asyncio.create_task(_carry(remote_reader, remote_writer, websocket))
    _CARRIERS.add(carrier)
    carrier.add_done_callback(_CARRIERS.discard)
    return reader, writer


async def _carry(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, websocket: Any
) -> None:
    try:
        await _splice_websocket(reader, writer, websocket)
    except Exception as exc:
        LOGGER.warning("control WebSocket ended: %s", exc)
    finally:
        await websocket.close()
        writer.close()


async def _splice_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    websocket: Any,
) -> None:
    """Carry bytes both ways between a stream pair and a WebSocket until one side closes."""

    async def tcp_to_ws() -> None:
        while data := await reader.read(65536):
            await websocket.send(data)

    async def ws_to_tcp() -> None:
        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            writer.write(data)
            await writer.drain()

    tasks = [asyncio.create_task(tcp_to_ws()), asyncio.create_task(ws_to_tcp())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        done_results = await asyncio.gather(*done, return_exceptions=True)
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for result in done_results:
        if isinstance(result, BaseException) and not isinstance(result, ConnectionClosedOK):
            raise result

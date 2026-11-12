"""The WebSocket leg shared by providers that reach a box's control channel over one."""

from __future__ import annotations

import asyncio
from typing import Any

from websockets.exceptions import ConnectionClosedOK


async def splice_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    websocket: Any,
) -> None:
    """Carry bytes both ways between a TCP connection and a WebSocket until one side closes."""

    async def tcp_to_ws() -> None:
        while data := await reader.read(65536):
            await websocket.send(data)

    async def ws_to_tcp() -> None:
        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            writer.write(data)
            await writer.drain()

    tasks = [
        asyncio.create_task(tcp_to_ws()),
        asyncio.create_task(ws_to_tcp()),
    ]
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

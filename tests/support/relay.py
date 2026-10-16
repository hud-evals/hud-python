"""A TCP relay in front of one upstream that can sever, refuse and stall connections.

Clients that reconnect (``SSHClient``, the control-channel tunnels) are driven
through :func:`relay` to fake an unreliable network between them and a real
server: :meth:`FlakyRelay.sever` drops every relayed connection,
:meth:`FlakyRelay.refuse` closes the next connections as soon as they are
accepted, and :meth:`FlakyRelay.stall` holds new connections without
forwarding them until :meth:`FlakyRelay.release`.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class FlakyRelay:
    """The relay's controls. Use :func:`relay`."""

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.port = 0
        #: Connections accepted so far, refused and stalled ones included.
        self.accepted = 0
        self._refusals = 0
        self._released = asyncio.Event()
        self._released.set()
        self._changed = asyncio.Condition()
        self._writers: set[asyncio.StreamWriter] = set()
        self._handlers: set[asyncio.Task[None]] = set()
        self._stalled = 0

    @property
    def open(self) -> int:
        """Client connections the relay currently holds open."""
        return len(self._writers)

    @property
    def stalled(self) -> int:
        """Connections waiting for :meth:`release`."""
        return self._stalled

    def sever(self) -> None:
        """Drop every connection the relay holds, as a network failure would."""
        for writer in list(self._writers):
            writer.transport.abort()

    def refuse(self, count: int) -> None:
        """Close each of the next ``count`` connections as soon as it is accepted."""
        self._refusals = count

    def stall(self) -> None:
        """Hold new connections open without forwarding them until :meth:`release`."""
        self._released.clear()

    def release(self) -> None:
        self._released.set()

    async def wait_until(self, *, accepted: int | None = None, stalled: int | None = None) -> None:
        """Wait until at least ``accepted`` connections were accepted or ``stalled`` wait."""
        async with self._changed:
            await asyncio.wait_for(
                self._changed.wait_for(
                    lambda: (
                        (accepted is None or self.accepted >= accepted)
                        and (stalled is None or self._stalled >= stalled)
                    )
                ),
                timeout=10,
            )

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._handlers.add(task)
        self._writers.add(writer)
        self.accepted += 1
        try:
            if self._refusals:
                self._refusals -= 1
                await self._notify()
                return
            if not self._released.is_set():
                self._stalled += 1
                await self._notify()
                try:
                    await self._released.wait()
                finally:
                    self._stalled -= 1
            else:
                await self._notify()
            up_reader, up_writer = await asyncio.open_connection(*self.upstream)
            try:
                await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
            finally:
                up_writer.transport.abort()
        except OSError:
            pass
        finally:
            self._writers.discard(writer)
            self._handlers.discard(task)
            writer.transport.abort()
            await self._notify()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.transport.abort()


@asynccontextmanager
async def relay(host: str, port: int) -> AsyncIterator[FlakyRelay]:
    """Serve a :class:`FlakyRelay` to ``host:port`` on a loopback port for the block."""
    controls = FlakyRelay((host, port))
    server = await asyncio.start_server(controls._handle, "127.0.0.1", 0)
    controls.port = server.sockets[0].getsockname()[1]
    try:
        yield controls
    finally:
        server.close()
        controls.release()
        controls.sever()
        for handler in list(controls._handlers):
            handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*controls._handlers, return_exceptions=True)
        await server.wait_closed()

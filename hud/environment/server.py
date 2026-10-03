"""The control-channel server for an Environment (``python -m hud.environment.server``)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hud.capabilities import Connection
from hud.utils.process import run_until_terminated

from .egress import WorkspaceRoute
from .session import Session
from .utils import (
    CONTROL_FRAME_LIMIT_BYTES,
    FILE_TRACKING_FRAME_LIMIT_BYTES,
    FrameTooLargeError,
    HudProtocolError,
    encode_frame,
    error,
    read_frame,
    reply,
    send_frame,
    splice,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .env import Environment

LOGGER = logging.getLogger("hud.environment.server")

# Line a serving process prints once its control channel is bound;
# ``SubprocessRuntime`` reads it from the child's stdout.
PORT_ANNOUNCEMENT = "HUD_SERVE_PORT="


class _ControlChannel:
    """The open sessions of one bound server, by id.

    A connection's first ``hello`` opens a session or attaches to an open one by id, and a
    session has at most one connection. When its connection ends, even mid-call, a session
    stays open for the lease that connection's ``hello`` asked for (none by default), so a
    later connection can attach and carry on; a task start or grade in progress runs on
    meanwhile. Then the session closes, cancelling its task, even mid-call, and releasing
    its bindings. ``bye`` closes it at once.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self._open: dict[str, Session] = {}
        self._attached: set[str] = set()
        self._expiries: dict[str, asyncio.Task[None]] = {}
        # The task start or grade each session is running, which can outlive its connection.
        self._running: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def handle(
        self,
        first: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one control connection's requests until it hangs up."""
        connection = _Connection(self, writer)
        try:
            await connection.serve(first, reader)
        finally:
            if connection.session is not None:
                await self._detach(connection.session, connection.lease)

    def _attach(self, requested: str | None) -> Session:
        """The session a connection's first ``hello`` names, or a new one."""
        if requested is None:
            session = Session(self.env)
            self._open[session.session_id] = session
        elif requested in self._attached:
            raise HudProtocolError(-32600, f"session {requested!r} has a live connection")
        elif (session := self._open.get(requested)) is None:
            raise HudProtocolError(-32600, f"unknown session: {requested!r}")
        else:
            self._expiries.pop(requested).cancel()
        self._attached.add(session.session_id)
        return session

    async def _detach(self, session: Session, lease: float) -> None:
        """Close a session its connection left, or keep it open ``lease`` seconds longer."""
        self._attached.discard(session.session_id)
        if lease:
            self._expiries[session.session_id] = asyncio.create_task(self._expire(session, lease))
        else:
            await self._close(session)

    async def _expire(self, session: Session, lease: float) -> None:
        await asyncio.sleep(lease)
        del self._expiries[session.session_id]
        await self._close(session)

    async def _close(self, session: Session) -> None:
        del self._open[session.session_id]
        await self._stop(session)
        await session.close()

    async def _stop(self, session: Session) -> None:
        """Cancel the task start or grade ``session`` is running, if any."""
        if (call := self._running.pop(session.session_id, None)) is not None:
            call.cancel()
            await asyncio.wait((call,))

    async def close(self) -> None:
        """Close every open session (server shutdown)."""
        expiries = list(self._expiries.values())
        self._expiries.clear()
        for expiry in expiries:
            expiry.cancel()
        await asyncio.gather(*expiries, return_exceptions=True)
        for session in list(self._open.values()):
            await self._close(session)


def _log_abandoned_failure(call: asyncio.Task[dict[str, Any]]) -> None:
    if not call.cancelled() and (exc := call.exception()) is not None:
        LOGGER.warning("a task call its connection left failed: %r", exc)


class _Connection:
    """One control connection: the session it speaks for, and the task call it awaits.

    tasks.start and tasks.grade run the task's own code, which can take as long as it
    likes, so the connection awaits them beside its read loop: a cancel or a bye still
    reaches the call, and any other request meanwhile is refused. A call the connection
    leaves stays with its session, which the lease keeps open or closes.
    """

    def __init__(self, channel: _ControlChannel, writer: asyncio.StreamWriter) -> None:
        self.channel = channel
        self.writer = writer
        self.session: Session | None = None
        self.lease = 0.0
        self._work: asyncio.Task[bool] | None = None
        self._work_id: int | str | None = None
        self._call: asyncio.Task[dict[str, Any]] | None = None

    async def serve(self, first: dict[str, Any], reader: asyncio.StreamReader) -> None:
        # The first frame is already read.
        next_frame: asyncio.Future[dict[str, Any] | None] = (
            asyncio.get_running_loop().create_future()
        )
        next_frame.set_result(first)
        try:
            while True:
                waiting: set[asyncio.Future[Any]] = {next_frame}
                if self._work is not None:
                    waiting.add(self._work)
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if self._work is not None and self._work in done:
                    # A failure that escaped the handler ends the connection.
                    finished, self._work = self._work, None
                    finished.result()
                if next_frame not in done:
                    continue
                msg = next_frame.result()
                if msg is None:
                    return
                next_frame = asyncio.ensure_future(
                    read_frame(reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
                )
                if not await self._receive(msg):
                    return
        finally:
            next_frame.cancel()
            if self._work is not None and self._call is not None and not self._call.done():
                # The connection ended mid-call: the lease decides the rest, and a failure
                # has no one to report to but the log.
                self._call.add_done_callback(_log_abandoned_failure)
            await self._stop_work()

    async def _receive(self, msg: dict[str, Any]) -> bool:
        """Act on one frame; False once the peer has said goodbye."""
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method in ("tasks.cancel", "bye"):
            if self._work is not None:
                await self._stop_work()
                if self._work_id is not None:
                    await _send_control_error(
                        self.writer, self._work_id, -32000, f"{method} interrupted the task"
                    )
        elif self._work is not None:
            if msg_id is not None:
                await _send_control_error(
                    self.writer, msg_id, -32600, f"{method}: a task call is still running"
                )
            return True
        elif method in ("tasks.start", "tasks.grade"):
            self._work = asyncio.create_task(self._respond(msg))
            self._work_id = msg_id
            return True
        return await self._respond(msg)

    async def _stop_work(self) -> None:
        """Stop awaiting the call; the call itself stays with the session."""
        if self._work is None:
            return
        stopping, self._work = self._work, None
        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping

    async def _respond(self, msg: dict[str, Any]) -> bool:
        """Answer one request; False once it closed the session with ``bye``."""
        env = self.channel.env
        method = msg.get("method", "")
        params = msg.get("params") or {}
        msg_id = msg.get("id")
        limit = CONTROL_FRAME_LIMIT_BYTES
        try:
            if method == "hello":
                raw_routes = params.get("workspace_routes", [])
                if not isinstance(raw_routes, list):
                    raise HudProtocolError(-32602, "hello: 'workspace_routes' must be a list")
                raw_connections = params.get("connections", [])
                if not isinstance(raw_connections, list):
                    raise HudProtocolError(-32602, "hello: 'connections' must be a list")
                try:
                    routes = [WorkspaceRoute.from_wire(route) for route in raw_routes]
                    connections = [Connection.from_wire(c) for c in raw_connections]
                except ValueError as exc:
                    raise HudProtocolError(-32602, f"hello: {exc}") from exc
                requested = params.get("session_id")
                if requested is not None and not isinstance(requested, str):
                    raise HudProtocolError(-32602, "hello: 'session_id' must be a string")
                requested_lease = params.get("lease", 0)
                if (
                    isinstance(requested_lease, bool)
                    or not isinstance(requested_lease, int | float)
                    or not math.isfinite(requested_lease)
                    or requested_lease < 0
                ):
                    raise HudProtocolError(
                        -32602, "hello: 'lease' must be a non-negative number of seconds"
                    )
                if self.session is None:
                    self.session = self.channel._attach(requested)
                elif requested not in (None, self.session.session_id):
                    raise HudProtocolError(-32600, "this connection already has a session")
                self.lease = float(requested_lease)
                self.session.bind(connections=connections, workspace_routes=routes)
                # env.start() ran before serving, so hook-published capabilities
                # are concrete here.
                result: dict[str, Any] = {
                    "session_id": self.session.session_id,
                    "env": {"name": env.name, "version": env.version},
                    "bindings": [c.to_manifest() for c in env.capabilities],
                }
            elif method == "tasks.list":
                result = {"tasks": [t.manifest_entry() for t in env.tasks.values()]}
            # An environment without a tracked workspace serves no files.* methods.
            elif method.startswith("files.") and (tracker := env.file_tracker) is not None:
                result = await asyncio.to_thread(tracker.call, method.removeprefix("files."))
                limit = FILE_TRACKING_FRAME_LIMIT_BYTES
            elif method not in ("tasks.start", "tasks.grade", "tasks.cancel", "bye"):
                raise HudProtocolError(-32601, f"method not found: {method}")
            elif (session := self.session) is None:
                raise HudProtocolError(-32600, f"{method}: send hello first")
            elif method in ("tasks.start", "tasks.grade"):
                running = self.channel._running.get(session.session_id)
                if running is not None and not running.done():
                    raise HudProtocolError(
                        -32600, f"{method}: the session is still running an earlier call"
                    )
                if method == "tasks.start":
                    task_id, args = params.get("id"), params.get("args") or {}
                    if not isinstance(task_id, str):
                        raise HudProtocolError(-32602, "tasks.start: 'id' must be a string")
                    if not isinstance(args, dict):
                        raise HudProtocolError(-32602, "tasks.start: 'args' must be an object")
                    if task_id not in env.tasks:
                        raise HudProtocolError(-32602, f"unknown task: {task_id!r}")
                    self._call = asyncio.create_task(session.start_task(task_id, args))
                else:
                    if not session.has_task:
                        raise HudProtocolError(-32600, "no task in progress")
                    self._call = asyncio.create_task(session.grade(params))
                self.channel._running[session.session_id] = self._call
                # Shielded: the connection can stop waiting while the call runs on.
                result = await asyncio.shield(self._call)
            elif method == "tasks.cancel":
                await self.channel._stop(session)
                await session.cancel()
                result = {"cancelled": True}
            else:
                await self.channel._close(session)
                self.session = None
                result = {"goodbye": True}
            if msg_id is not None:
                try:
                    await send_frame(self.writer, reply(msg_id, result), max_bytes=limit)
                except FrameTooLargeError:
                    if method == "tasks.start":
                        await session.cancel()  # its prompt never reaches the client
                    raise
        except HudProtocolError as exc:
            if msg_id is not None:
                await _send_control_error(self.writer, msg_id, exc.code, exc.message)
        except Exception as exc:
            LOGGER.exception("error handling %s", method)
            if msg_id is not None:
                await _send_control_error(self.writer, msg_id, -32000, str(exc))
        return method != "bye" or self.session is not None


async def _send_control_error(
    writer: asyncio.StreamWriter, msg_id: int | str | None, code: int, message: str
) -> None:
    try:
        frame = encode_frame(error(msg_id, code, message), max_bytes=CONTROL_FRAME_LIMIT_BYTES)
    except FrameTooLargeError as exc:
        try:
            frame = encode_frame(
                error(msg_id, -32000, str(exc)), max_bytes=CONTROL_FRAME_LIMIT_BYTES
            )
        except FrameTooLargeError:
            frame = encode_frame(error(None, -32000, str(exc)), max_bytes=CONTROL_FRAME_LIMIT_BYTES)
    writer.write(frame)
    await writer.drain()


async def _stream(
    env: Environment,
    msg: dict[str, Any],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """One capability stream: dial the resolved daemon and splice raw bytes."""
    msg_id = msg.get("id")
    try:
        name = (msg.get("params") or {}).get("capability")
        if not isinstance(name, str):
            raise ValueError("tunnel.open: 'capability' must be a string")
        cap = env.capability(name)
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"capability {name!r} has no host:port to tunnel to")
        response = (
            encode_frame(reply(msg_id, {"capability": name}), max_bytes=CONTROL_FRAME_LIMIT_BYTES)
            if msg_id is not None
            else None
        )
        backend = await asyncio.open_connection(parts.hostname, parts.port)
    except Exception as exc:
        LOGGER.warning("refusing capability stream: %s", exc)
        if msg_id is not None:
            code = -32602 if isinstance(exc, ValueError) else -32000
            await _send_control_error(writer, msg_id, code, str(exc))
        return
    try:
        if response is not None:
            writer.write(response)
            await writer.drain()
        await splice((reader, writer), backend)
    finally:
        backend[1].close()
        with contextlib.suppress(Exception):
            await backend[1].wait_closed()


@contextlib.asynccontextmanager
async def bind(
    env: Environment, host: str = "127.0.0.1", port: int = 0
) -> AsyncIterator[asyncio.Server]:
    """Accept control connections for the started ``env`` until the block exits.

    A connection's first frame decides what it is: ``tunnel.open`` opens one raw
    capability stream; anything else begins a JSON-RPC control session. On exit the
    server closes, its connections are cut, and its open sessions close, cancelling
    their tasks.
    """
    channel = _ControlChannel(env)
    # Tracked so shutdown cancels handlers instead of abandoning them mid-splice.
    handlers: set[asyncio.Task[None]] = set()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        task.add_done_callback(handlers.discard)
        try:
            first = await read_frame(reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
            if first is None:
                return
            if first.get("method") == "tunnel.open":
                await _stream(env, first, reader, writer)
            else:
                await channel.handle(first, reader, writer)
        except FrameTooLargeError as exc:
            await _send_control_error(writer, None, -32000, str(exc))
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    server = await asyncio.start_server(accept, host=host, port=port)
    sock = server.sockets[0].getsockname()
    LOGGER.info("env %r bound on %s:%s", env.name, sock[0], sock[1])
    try:
        yield server
    finally:
        server.close()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        await channel.close()
        await server.wait_closed()


async def serve(env: Environment, host: str = "127.0.0.1", port: int = 0) -> None:
    """Start ``env``'s daemons and serve its control channel until cancelled."""
    await env.start()
    try:
        async with bind(env, host, port) as server:
            port_line = f"{PORT_ANNOUNCEMENT}{server.sockets[0].getsockname()[1]}"
            print(port_line, flush=True)  # noqa: T201 - SubprocessRuntime reads this from stdout
            # The bound server is already accepting. Not serve_forever(): cancelling it
            # waits for open connections to close, and only leaving bind() closes them.
            await asyncio.Event().wait()
    finally:
        await env.stop()


def main() -> None:
    from hud.environment import load_environment

    parser = argparse.ArgumentParser(description="Serve a HUD environment from source.")
    parser.add_argument("path", help="A .py file or a directory defining an Environment.")
    parser.add_argument("--env", default=None, help="Environment name when several are defined.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Interface to bind (0.0.0.0 inside containers)."
    )
    parser.add_argument("--port", type=int, default=0, help="Port to bind (0 = ephemeral).")
    args = parser.parse_args()
    run_until_terminated(serve(load_environment(args.path, name=args.env), args.host, args.port))


if __name__ == "__main__":
    main()

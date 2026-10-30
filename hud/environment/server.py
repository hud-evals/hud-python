"""``python -m hud.environment.server`` — the protocol server for an Environment.

The substrate side of the runtime contract: an :class:`Environment` only
declares what exists; this module puts one on the wire. It owns task execution
(:class:`TaskRunner`), per-connection protocol dispatch and serving-time state
(:func:`bind`), and the full serving lifecycle (:func:`serve`) — backing
daemons up, control channel bound (announcing the port on stdout as
``HUD_SERVE_PORT=<port>``), daemons down. Every substrate shape runs it: the
:class:`~hud.eval.runtime.SubprocessRuntime` child process, a container CMD, and
``hud serve``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import math
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from pydantic import BaseModel, TypeAdapter, ValidationError

from hud.capabilities import Connection
from hud.graders.results import EvaluationResult
from hud.utils.process import run_until_terminated

from .egress import WorkspaceRoute
from .env import Answer, current_session_id
from .utils import (
    CONTROL_FRAME_LIMIT_BYTES,
    FrameTooLargeError,
    encode_frame,
    error,
    read_frame,
    reply,
    send_frame,
    splice,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from .env import Environment, _TaskFactory

LOGGER = logging.getLogger("hud.environment.server")

#: Line a serving process prints once its control channel is bound; the
#: ``spawn`` provider reads it from the child's stdout.
PORT_ANNOUNCEMENT = "HUD_SERVE_PORT="


@dataclass
class _ServerState:
    handlers: set[asyncio.Task[None]]
    channel: _ControlChannel


_SERVER_STATES: WeakKeyDictionary[asyncio.Server, _ServerState] = WeakKeyDictionary()


# ─── task execution ──────────────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Recursively convert a prompt payload into JSON-safe primitives.

    The prompt frame may carry rich objects — most importantly a list of
    ``PromptMessage`` (chat-style message prompts) — which must become plain
    dicts/lists before the JSON-RPC framing layer (``json.dumps``) ships them.
    Models dump by alias, so MCP's models keep their wire names (``mimeType``).
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce_args(sig: inspect.Signature, args: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for name, value in args.items():
        param = sig.parameters.get(name)
        annotation = param.annotation if param is not None else inspect.Parameter.empty
        if annotation in (inspect.Parameter.empty, Any):
            coerced[name] = value
            continue
        adapter = TypeAdapter(annotation)
        try:
            coerced[name] = adapter.validate_python(value)
        except ValidationError:
            if not isinstance(value, str):
                coerced[name] = value
                continue
            try:
                coerced[name] = adapter.validate_json(value)
            except ValidationError:
                coerced[name] = value
    return coerced


def _build_answer(return_type: Any, payload: dict[str, Any]) -> Any:
    """Build the value sent into the task gen for evaluation.

    Without a declared ``return_type`` the answer value is forwarded unchanged.
    With one, the agent's answer is parsed into an ``Answer[T]`` — the
    structured-answer contract (parse failures fall back to the raw string
    on ``content`` so the task can grade them).
    """
    if return_type is None:
        return payload.get("answer")

    raw_text = payload.get("answer", "")
    adapter = TypeAdapter(return_type)
    try:
        content = (
            adapter.validate_json(raw_text)
            if isinstance(raw_text, str)
            else adapter.validate_python(raw_text)
        )
    except ValidationError:
        content = raw_text
    return Answer(
        content=content,
        raw=raw_text if isinstance(raw_text, str) else str(raw_text),
    )


def _score_value(result: Any) -> float:
    """Normalize a task's grade yield to a float score, loudly.

    Accepts a number or an object with a numeric ``reward`` attribute (the
    ``hud.graders.EvaluationResult`` shape). Anything else is an authoring bug;
    grading it silently as 0.0 would hide it.
    """
    score = getattr(result, "reward", result)
    if isinstance(score, (int, float)):
        return float(score)
    raise TypeError(
        f"task graded with {type(result).__name__}: yield a number, an object "
        "with a numeric .reward, or a dict containing a numeric 'score'"
    )


class TaskRunner:
    """Holds one task's suspended generator between ``tasks.start`` and ``tasks.grade``."""

    def __init__(self, task: _TaskFactory[Any], args: dict[str, Any] | None = None) -> None:
        self.task = task
        self._args = args or {}
        self._gen: AsyncGenerator[Any, Any] | None = None

        # Fail fast on bad args (TypeError before any side-effects run).
        try:
            task.sig.bind(**self._args)
        except TypeError as exc:
            raise TypeError(
                f"task {task.id!r}: bad args {sorted(self._args)}: {exc}",
            ) from exc

    async def start(self) -> dict[str, Any]:
        self._gen = self.task.func(**_coerce_args(self.task.sig, self._args))
        prompt = await self._gen.__anext__()
        frame = prompt if isinstance(prompt, dict) and "prompt" in prompt else {"prompt": prompt}
        return cast("dict[str, Any]", _jsonable(frame))

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._gen is None:
            raise RuntimeError("task not started")
        try:
            evaluation = await self._gen.asend(_build_answer(self.task.return_type, payload))
        except StopAsyncIteration:
            evaluation = 0.0
        finally:
            await self.cancel()
        if isinstance(evaluation, dict):
            if not isinstance(evaluation.get("score"), (int, float)):
                raise TypeError(
                    f"task {self.task.id!r} graded with a dict missing a numeric "
                    f"'score' (keys: {sorted(evaluation)})"
                )
            frame = cast("dict[str, Any]", _jsonable(evaluation))
        elif isinstance(evaluation, EvaluationResult):
            # Forward the full grade frame so metadata (info/content/done/isError/
            # subscores) survives; the wire names the score "score", the model "reward".
            frame = evaluation.model_dump(mode="json")
            frame["score"] = frame.pop("reward")
        else:
            frame = {"score": _score_value(evaluation)}
        if not math.isfinite(frame["score"]):
            raise ValueError(
                f"task {self.task.id!r} graded with a non-finite score ({frame['score']!r}): "
                "yield a finite number"
            )
        return frame

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None
        # Session ended (cancel / bye / post-grade) — run teardown hooks.
        for hook in self.task.env._on_task_teardown:
            with contextlib.suppress(Exception):
                await hook()


# ─── wire protocol ───────────────────────────────────────────────────────
# The connection grammar (control session vs capability stream) lives on
# :func:`bind` — the accept point. Session dispatch lives on _ControlChannel.


class _NoTaskInProgress(RuntimeError):
    pass


async def _frames(
    first: dict[str, Any],
    reader: asyncio.StreamReader,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ``first`` and then every subsequent frame until the peer hangs up."""
    msg: dict[str, Any] | None = first
    while msg is not None:
        yield msg
        msg = await read_frame(reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)


class _ControlChannel:
    """Serving-time state for one bound control channel.

    Owns what the declaration must not: runtime state — one suspended task per
    session, keyed by session id (scoped to this server; two servers for one
    env never share). A session's ``start`` replaces its own runner, ``grade``
    consumes it, ``cancel`` clears it. A connection drop parks the session's
    runner in place; a later connection grades it by resuming the session
    (``hello`` with its id) or, when exactly one session is parked, by a plain
    ``tasks.grade`` — the split start/grade flow (e.g. ``hud task grade`` or
    harbor's verifier reconnecting to grade) with no parking handoff to manage.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        #: Suspended tasks by session id. An entry whose session has no live
        #: connection is parked, awaiting a resume (or adoption) to grade.
        self._runners: dict[str, TaskRunner] = {}
        self._live: set[str] = set()

    async def start(self, session_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        await self._cancel_runner(session_id)
        runner = TaskRunner(self.env.tasks[task_id], args)
        self._runners[session_id] = runner
        try:
            return await runner.start()
        except BaseException:
            self._runners.pop(session_id, None)
            await runner.cancel()
            raise

    async def grade(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        runner = self._runners.pop(session_id, None)
        claim_sid = session_id
        if runner is None:
            # Blind adopt: free the parked session's robot claim, not this connection's.
            claim_sid, runner = self._adopt_parked()
        token = current_session_id.set(claim_sid)
        try:
            return await runner.grade(payload)
        finally:
            current_session_id.reset(token)
            self._release_if_idle(claim_sid)

    def _adopt_parked(self) -> tuple[str, TaskRunner]:
        """Claim the parked session iff unambiguous — the blind-reconnect grade path."""
        parked = sorted(sid for sid in self._runners if sid not in self._live)
        if not parked:
            raise _NoTaskInProgress("no task in progress")
        if len(parked) > 1:
            raise _NoTaskInProgress(
                f"{len(parked)} parked sessions ({', '.join(parked)}); "
                "resume one by sending hello with its session_id"
            )
        sid = parked[0]
        return sid, self._runners.pop(sid)

    async def _cancel_runner(self, session_id: str) -> None:
        runner = self._runners.pop(session_id, None)
        if runner is None:
            return
        # Bind session id so teardown hooks (robot slot release) free the right claim.
        token = current_session_id.set(session_id)
        try:
            await runner.cancel()
        finally:
            current_session_id.reset(token)

    async def cancel(self, session_id: str) -> None:
        await self._cancel_runner(session_id)
        self._release_if_idle(session_id)

    def _release_if_idle(self, session_id: str) -> None:
        """Release a session's controller connections and routes once nothing can use them.

        A live control connection or a parked task (awaiting resume or grade) keeps
        them; otherwise they are released so a later session inherits neither.
        """
        if session_id not in self._live and session_id not in self._runners:
            self.env.release_session(session_id)

    async def cancel_all(self) -> None:
        """Tear down every suspended/live task (server shutdown)."""
        for session_id in list(self._runners):
            await self.cancel(session_id)

    async def session(
        self,
        first: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """One control session: JSON-RPC dispatch for the connection's lifetime."""
        await _Session(self, writer).run(first, reader)


class _Session:
    """One control connection: the session it speaks for, and the task call it runs.

    tasks.start and tasks.grade run the task's own code, which can take as long as it
    likes; they run beside the read loop so a cancel or a bye still reaches them. A
    hang-up lets a start finish and park for resume, and ends a grade, whose result
    nobody is left to receive.
    """

    def __init__(self, channel: _ControlChannel, writer: asyncio.StreamWriter) -> None:
        self.channel = channel
        self.env = channel.env
        self.writer = writer
        self.id = "sess-" + secrets.token_hex(4)
        channel._live.add(self.id)
        self._token = current_session_id.set(self.id)
        self._work: asyncio.Task[None] | None = None
        self._work_id: int | str | None = None
        self._work_method = ""

    async def run(self, first: dict[str, Any], reader: asyncio.StreamReader) -> None:
        frames = aiter(_frames(first, reader))
        next_frame: asyncio.Future[dict[str, Any]] = asyncio.ensure_future(anext(frames))
        try:
            while True:
                waiting: set[asyncio.Future[Any]] = {next_frame}
                if self._work is not None:
                    waiting.add(self._work)
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if self._work is not None and self._work in done:
                    # A failure that escaped the handler ends the session.
                    finished, self._work = self._work, None
                    finished.result()
                if next_frame not in done:
                    continue
                try:
                    msg = next_frame.result()
                except StopAsyncIteration:
                    if self._work is not None and self._work_method == "tasks.start":
                        starting, self._work = self._work, None
                        await starting
                    return
                next_frame = asyncio.ensure_future(anext(frames))
                if not await self._receive(msg):
                    return
        finally:
            next_frame.cancel()
            await self._stop_work()
            self.channel._live.discard(self.id)
            # Task stays parked for resume/grade; robot slots free on cancel/bye/grade
            # teardown (callers that abort must cancel first — see rollout timeout).
            self.channel._release_if_idle(self.id)
            current_session_id.reset(self._token)

    async def _receive(self, msg: dict[str, Any]) -> bool:
        """Act on one frame; False once the peer has said goodbye."""
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method in ("tasks.cancel", "bye"):
            if self._work is not None:
                await self._stop_work()
                await self._error(self._work_id, -32000, f"{method} interrupted the task")
            await self.channel.cancel(self.id)
            if method == "bye":
                await self._reply(msg_id, {"goodbye": True})
                return False
            await self._reply(msg_id, {"cancelled": True})
        elif self._work is not None:
            await self._error(msg_id, -32600, f"{method}: a task call is still running")
        elif method in ("tasks.start", "tasks.grade"):
            self._work = asyncio.create_task(self._handle(msg))
            self._work_id, self._work_method = msg_id, method
        else:
            await self._handle(msg)
        return True

    async def _stop_work(self) -> None:
        if self._work is None:
            return
        stopping, self._work = self._work, None
        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping

    async def _reply(self, msg_id: int | str | None, result: dict[str, Any]) -> None:
        if msg_id is not None:
            await send_frame(
                self.writer, reply(msg_id, result), max_bytes=CONTROL_FRAME_LIMIT_BYTES
            )

    async def _error(self, msg_id: int | str | None, code: int, message: str) -> None:
        if msg_id is not None:
            await _send_control_error(self.writer, msg_id, code, message)

    async def _handle(self, msg: dict[str, Any]) -> None:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        msg_id = msg.get("id")
        try:
            match method:
                case "hello":
                    await self._hello(msg_id, params)
                case "tasks.list":
                    await self._reply(
                        msg_id, {"tasks": [t.manifest_entry() for t in self.env.tasks.values()]}
                    )
                case "tasks.start":
                    await self._start(msg_id, params)
                case "tasks.grade":
                    try:
                        evaluation = await self.channel.grade(self.id, params)
                    except _NoTaskInProgress as exc:
                        await self._error(msg_id, -32600, str(exc))
                        return
                    await self._reply(msg_id, evaluation)
                case _:
                    await self._error(msg_id, -32601, f"method not found: {method}")
        except Exception as exc:
            LOGGER.exception("error handling %s", method)
            await self._error(msg_id, -32000, str(exc))

    async def _hello(self, msg_id: int | str | None, params: dict[str, Any]) -> None:
        requested = params.get("session_id")
        if requested is not None and requested != self.id:
            if not isinstance(requested, str):
                await self._error(msg_id, -32602, "hello: 'session_id' must be a string")
                return
            if requested in self.channel._live:
                await self._error(msg_id, -32600, f"session {requested!r} has a live connection")
                return
            if requested not in self.channel._runners:
                await self._error(msg_id, -32600, f"unknown session: {requested!r}")
                return
            # Resume: this connection becomes the parked session.
            self.channel._live.discard(self.id)
            self.id = requested
            self.channel._live.add(self.id)
            current_session_id.reset(self._token)
            self._token = current_session_id.set(self.id)
        raw_routes = params.get("workspace_routes", [])
        if not isinstance(raw_routes, list):
            await self._error(msg_id, -32602, "hello: 'workspace_routes' must be a list")
            return
        try:
            workspace_routes = [WorkspaceRoute.from_wire(route) for route in raw_routes]
        except ValueError as exc:
            await self._error(msg_id, -32602, f"hello: {exc}")
            return
        raw_connections = params.get("connections", [])
        if not isinstance(raw_connections, list):
            await self._error(msg_id, -32602, "hello: 'connections' must be a list")
            return
        try:
            connections = [Connection.from_wire(connection) for connection in raw_connections]
        except ValueError as exc:
            await self._error(msg_id, -32602, f"hello: {exc}")
            return
        self.env.bind_connections(connections, session=self.id)
        self.env.bind_workspace_routes(workspace_routes, session=self.id)
        # env.start() ran before serving, so hook-published capabilities (e.g. a
        # workspace's ssh address) are already concrete here.
        bindings = [c.to_manifest() for c in self.env.capabilities]
        await self._reply(
            msg_id,
            {
                "session_id": self.id,
                "env": {"name": self.env.name, "version": self.env.version},
                "bindings": bindings,
            },
        )

    async def _start(self, msg_id: int | str | None, params: dict[str, Any]) -> None:
        task_id = params.get("id")
        if not isinstance(task_id, str):
            await self._error(msg_id, -32602, "tasks.start: 'id' must be a string")
            return
        args = params.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            await self._error(msg_id, -32602, "tasks.start: 'args' must be an object")
            return
        if task_id not in self.env.tasks:
            await self._error(msg_id, -32602, f"unknown task: {task_id!r}")
            return
        prompt = await self.channel.start(self.id, task_id, args)
        try:
            await self._reply(msg_id, prompt)
        except FrameTooLargeError:
            await self.channel.cancel(self.id)
            raise


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
    """One capability stream: dial the resolved daemon and splice raw bytes.

    The client opens one such connection per capability stream, so the
    control port is the only address a substrate ever needs to expose.
    """
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


async def bind(env: Environment, host: str = "127.0.0.1", port: int = 0) -> asyncio.Server:
    """Bind a control-channel server for *env* (not yet serving).

    The accept point owns the transport's connection grammar — TCP has no
    native streams, so the preface (first) frame decides what a connection
    is: a ``tunnel.open`` frame opens one capability stream (a single reply,
    then raw bytes — the CONNECT analog); anything else begins a JSON-RPC
    control session. Session methods are transport-invariant; the preface is
    TCP routing (a WebSocket transport would tunnel via its native upgrade).

    Each bind gets fresh serving state. Callers read the assigned port from
    ``server.sockets[0].getsockname()`` and drive it with
    ``server.serve_forever()``.
    """
    channel = _ControlChannel(env)
    # Live connection handlers, so teardown can cancel them instead of
    # abandoning them to loop shutdown (-> "Task was destroyed but it is
    # pending" + GeneratorExit thrown into mid-splice coroutines).
    active: set[asyncio.Task[None]] = set()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            active.add(task)
            task.add_done_callback(active.discard)
        try:
            first = await read_frame(reader, max_bytes=CONTROL_FRAME_LIMIT_BYTES)
            if first is None:
                return
            if first.get("method") == "tunnel.open":
                await _stream(env, first, reader, writer)
            else:
                await channel.session(first, reader, writer)
        except FrameTooLargeError as exc:
            await _send_control_error(writer, None, -32000, str(exc))
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    server = await asyncio.start_server(accept, host=host, port=port)
    _SERVER_STATES[server] = _ServerState(handlers=active, channel=channel)
    sock = server.sockets[0].getsockname()
    LOGGER.info("env %r bound on %s:%s", env.name, sock[0], sock[1])
    return server


async def _shutdown(server: asyncio.Server) -> None:
    server.close()
    state = _SERVER_STATES.pop(server)
    handlers = list(state.handlers)
    for task in handlers:
        task.cancel()
    await asyncio.gather(*handlers, return_exceptions=True)
    await state.channel.cancel_all()
    await server.wait_closed()


async def serve(env: Environment, host: str = "127.0.0.1", port: int = 0) -> None:
    """Start *env*'s daemons and serve its control channel until cancelled."""
    await env.start()
    server: asyncio.Server | None = None
    try:
        server = await bind(env, host, port)
        port_line = f"{PORT_ANNOUNCEMENT}{server.sockets[0].getsockname()[1]}"
        print(port_line, flush=True)  # noqa: T201 - the spawn provider reads this from stdout
        # The bound server is already accepting. Not serve_forever(): cancelling it
        # waits for open connections to close, and only _shutdown() closes them.
        await asyncio.Event().wait()
    finally:
        if server is not None:
            await _shutdown(server)
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

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
import secrets
import signal
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, TypeAdapter, ValidationError

from hud.graders.results import EvaluationResult

from .env import Answer
from .utils import error, read_frame, reply, send_frame, splice

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from .env import Environment, _TaskFactory

LOGGER = logging.getLogger("hud.environment.server")

#: Line a serving process prints once its control channel is bound; the
#: ``spawn`` provider reads it from the child's stdout.
PORT_ANNOUNCEMENT = "HUD_SERVE_PORT="


# ─── task execution ──────────────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Recursively convert a prompt payload into JSON-safe primitives.

    The prompt frame may carry rich objects — most importantly a list of
    ``PromptMessage`` (chat-style message prompts) — which must become plain
    dicts/lists before the JSON-RPC framing layer (``json.dumps``) ships them.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce_args(sig: inspect.Signature, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce string wire args into the task fn's annotated param types.

    JSON-RPC sends args as JSON scalars/strings; a param annotated with a richer
    type (Pydantic model, list, etc.) is validated via a ``TypeAdapter``. Values
    that already match (or fail to validate) are passed through unchanged.
    """
    coerced: dict[str, Any] = {}
    for name, value in args.items():
        param = sig.parameters.get(name)
        annotation = param.annotation if param is not None else inspect.Parameter.empty
        if annotation in (inspect.Parameter.empty, str, Any) or not isinstance(value, str):
            coerced[name] = value
            continue
        try:
            coerced[name] = TypeAdapter(annotation).validate_json(value)
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
            return cast("dict[str, Any]", _jsonable(evaluation))
        if isinstance(evaluation, EvaluationResult):
            # Forward the full grade frame so metadata (info/content/done/isError/
            # subscores) survives; the wire names the score "score", the model "reward".
            frame = evaluation.model_dump_for_span()
            frame["score"] = frame.pop("reward")
            return frame
        return {"score": _score_value(evaluation)}

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None


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
        msg = await read_frame(reader)


class _ControlChannel:
    """Serving-time state for one bound control channel.

    Owns what the declaration must not: runtime state — at most one suspended
    task at a time, living on the channel itself (scoped to this server; two
    servers for one env never share). ``start`` replaces it, ``grade``
    consumes it, ``cancel`` clears it, and a connection drop leaves it in
    place — which is exactly the split start/grade flow (e.g. harbor's
    verifier reconnecting to grade) with no parking handoff to manage.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self._runner: TaskRunner | None = None

    async def start(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        await self.cancel()
        self._runner = TaskRunner(self.env.tasks[task_id], args)
        return await self._runner.start()

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        runner, self._runner = self._runner, None
        if runner is None:
            raise _NoTaskInProgress("no task in progress")
        return await runner.grade(payload)

    async def cancel(self) -> None:
        if self._runner is not None:
            await self._runner.cancel()
            self._runner = None

    async def session(
        self,
        first: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """One control session: JSON-RPC dispatch for the connection's lifetime."""
        env = self.env
        session_id = "sess-" + secrets.token_hex(4)

        async def reply_to(msg_id: int | None, result: dict[str, Any]) -> None:
            if msg_id is not None:
                await send_frame(writer, reply(msg_id, result))

        async def error_to(msg_id: int | None, code: int, message: str) -> None:
            if msg_id is not None:
                await send_frame(writer, error(msg_id, code, message))

        async for msg in _frames(first, reader):
            method = msg.get("method", "")
            params = msg.get("params") or {}
            msg_id = msg.get("id")

            try:
                if method == "hello":
                    # env.start() ran before serving, so hook-published
                    # capabilities (e.g. a workspace's ssh address) are
                    # already concrete here.
                    bindings = [c.to_manifest() for c in env.capabilities]
                    await reply_to(
                        msg_id,
                        {
                            "session_id": session_id,
                            "env": {"name": env.name, "version": env.version},
                            "bindings": bindings,
                        },
                    )

                elif method == "tasks.list":
                    await reply_to(
                        msg_id,
                        {"tasks": [t.manifest_entry() for t in env.tasks.values()]},
                    )

                elif method == "tasks.start":
                    task_id = params.get("id")
                    if not isinstance(task_id, str):
                        await error_to(msg_id, -32602, "tasks.start: 'id' must be a string")
                        continue
                    args = params.get("args") or {}
                    if not isinstance(args, dict):
                        await error_to(msg_id, -32602, "tasks.start: 'args' must be an object")
                        continue
                    try:
                        prompt = await self.start(task_id, args)
                    except KeyError:
                        await error_to(msg_id, -32602, f"unknown task: {task_id!r}")
                        continue
                    await reply_to(msg_id, prompt)

                elif method == "tasks.grade":
                    try:
                        evaluation = await self.grade(params)
                    except _NoTaskInProgress:
                        await error_to(msg_id, -32600, "no task in progress")
                        continue
                    await reply_to(msg_id, evaluation)

                elif method == "tasks.cancel":
                    await self.cancel()
                    await reply_to(msg_id, {"cancelled": True})

                elif method == "bye":
                    await self.cancel()
                    await reply_to(msg_id, {"goodbye": True})
                    return

                else:
                    await error_to(msg_id, -32601, f"method not found: {method}")

            except Exception as exc:
                LOGGER.exception("error handling %s", method)
                await error_to(msg_id, -32000, str(exc))


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
        backend = await asyncio.open_connection(parts.hostname, parts.port)
    except Exception as exc:
        LOGGER.warning("refusing capability stream: %s", exc)
        if msg_id is not None:
            code = -32602 if isinstance(exc, ValueError) else -32000
            await send_frame(writer, error(msg_id, code, str(exc)))
        return
    if msg_id is not None:
        await send_frame(writer, reply(msg_id, {"capability": name}))
    await splice((reader, writer), backend)


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
            first = await read_frame(reader)
            if first is None:
                return
            if first.get("method") == "tunnel.open":
                await _stream(env, first, reader, writer)
            else:
                await channel.session(first, reader, writer)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    server = await asyncio.start_server(accept, host=host, port=port)
    server._hud_handlers = active  # type: ignore[attr-defined]
    sock = server.sockets[0].getsockname()
    LOGGER.info("env %r bound on %s:%s", env.name, sock[0], sock[1])
    return server


async def serve(env: Environment, host: str = "127.0.0.1", port: int = 0) -> None:
    """Start *env*'s daemons and serve its control channel until cancelled."""
    await env.start()
    server: asyncio.Server | None = None
    try:
        server = await bind(env, host, port)
        port_line = f"{PORT_ANNOUNCEMENT}{server.sockets[0].getsockname()[1]}"
        print(port_line, flush=True)  # noqa: T201 - the spawn provider reads this from stdout
        async with server:
            await server.serve_forever()
    finally:
        if server is not None:
            for task in list(getattr(server, "_hud_handlers", ())):
                task.cancel()
        await env.stop()


async def _serve_until_terminated(env: Environment, host: str, port: int) -> None:
    main_task = asyncio.current_task()
    assert main_task is not None
    # SIGTERM (the spawn provider's teardown) cancels serving so env.stop()
    # runs and backing daemons don't orphan. Not available on Windows loops.
    with contextlib.suppress(NotImplementedError):
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, main_task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await serve(env, host, port)


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
    asyncio.run(
        _serve_until_terminated(load_environment(args.path, name=args.env), args.host, args.port)
    )


if __name__ == "__main__":
    main()

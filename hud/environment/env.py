"""Environment: declarative capabilities + tasks behind the HUD wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import secrets
from typing import TYPE_CHECKING, Any, ParamSpec, cast

from hud._wire import WireProtocolError, error, read_frame, reply, send_frame

from .legacy import LegacyEnvMixin
from .task import Task, TaskRunner

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from hud.capabilities import Capability

LOGGER = logging.getLogger("hud.environment.env")

P = ParamSpec("P")


class Environment(LegacyEnvMixin):
    """Capabilities + tasks dispatched over the HUD wire protocol.

    Also accepts the deprecated v5 env-authoring surface (positional ``name``,
    ``@env.scenario``, ``@env.tool`` / ``env.add_tool``, ``env("scenario")``,
    ``env.run``) via :class:`~hud.environment.legacy.LegacyEnvMixin`, so deployed
    v5 envs keep running. Each legacy entry point warns and adapts to v6.
    """

    def __init__(
        self,
        name: str = "environment",
        *,
        version: str = "0.0.1",
        capabilities: list[Capability] | None = None,
        auth_token: str | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        if legacy_kwargs:
            import warnings

            warnings.warn(
                f"Environment(): ignoring v5 keyword(s) {sorted(legacy_kwargs)} "
                "(no longer part of the v6 Environment surface).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.name = name
        self.version = version
        self.capabilities: list[Capability] = list(capabilities or [])
        # When set, a client must present this token in ``hello`` before it can
        # drive tasks (and before it sees capability bindings / their secrets).
        # Default ``None`` keeps the control channel open (loopback dev).
        self._auth_token = auth_token
        self._tasks: dict[str, Task[Any]] = {}
        # Backing-daemon lifecycle hooks (e.g. a legacy MCP server the adapter
        # stands up). Run once by the substrate (LocalSandbox) around serving.
        self._on_start: list[Callable[[], Awaitable[None]]] = []
        self._on_stop: list[Callable[[], Awaitable[None]]] = []
        self._init_legacy()

    # ─── task registration ───────────────────────────────────────────

    def task(
        self,
        *,
        id: str | None = None,
        description: str = "",
        input: Any = None,
        returns: Any = None,
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], Task[P]]:
        """Register an async-generator task (``id`` defaults to the function name).

        The task yields a prompt, then — once the answer is sent back — a reward.
        Either form works (both normalized to the wire protocol): friendly (``yield
        prompt`` → ``yield reward``) or explicit (``yield {"prompt": ...}`` → ``yield
        {"score": ...}``). ``input``/``returns`` optionally declare the agent's I/O
        types (surfaced in the manifest as JSON schemas). Returns a ``Task`` — call it
        with the task's args to get a runnable :class:`~hud.eval.Variant`.
        """
        from .task import scenario_to_task_fn

        def decorate(func: Callable[P, AsyncGenerator[Any, Any]]) -> Task[P]:
            if not inspect.isasyncgenfunction(func):
                raise TypeError(
                    f"@env.task: {getattr(func, '__qualname__', func)} must be an async "
                    "generator function (`async def ...:` with `yield`)",
                )
            task_id = id or func.__name__
            if task_id in self._tasks:
                raise ValueError(
                    f"task {task_id!r} already registered on env {self.name!r}",
                )
            normalized = cast(
                "Callable[P, AsyncGenerator[dict[str, Any], dict[str, Any]]]",
                scenario_to_task_fn(func),
            )
            task = Task(self, task_id, description, normalized, input=input, returns=returns)
            self._tasks[task_id] = cast("Task[Any]", task)
            return task

        return decorate

    def add_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def _task_runner(self, task_id: str, args: dict[str, Any] | None = None) -> TaskRunner:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id!r}")
        return TaskRunner(task, args)

    def tasks_manifest(self) -> list[dict[str, Any]]:
        return [task.manifest_entry() for task in self._tasks.values()]

    def initialize(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register an initializer, run once before the control channel serves.

        Use it to start a backing daemon — e.g. a :class:`~hud.environment.Workspace`'s
        SSH server — whose capability is declared at construction
        (``Environment(..., capabilities=[ws.capability()])``).
        """
        self._on_start.append(fn)
        return fn

    def shutdown(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register a teardown hook (run in reverse order on stop)."""
        self._on_stop.append(fn)
        return fn

    # ─── serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the env descriptor: identity, capabilities, and task list.

        Task generator *code* is not serializable; ``tasks`` carries id/description
        metadata for discovery.
        """
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": [c.to_manifest() for c in self.capabilities],
            "tasks": self.tasks_manifest(),
        }

    # ─── control-channel server ──────────────────────────────────────────

    async def bind(self, host: str = "127.0.0.1", port: int = 0) -> asyncio.Server:
        """Bind the control-channel socket (not yet serving). Returns the server.

        Callers read the assigned port via ``server.sockets[0].getsockname()`` and
        drive it with ``server.serve_forever()``. Used by ``hud.launch`` to bring
        up a live env on an ephemeral loopback port.
        """
        server = await asyncio.start_server(self._handle_session, host=host, port=port)
        sock = server.sockets[0].getsockname()
        LOGGER.info("env %r bound on %s:%s", self.name, sock[0], sock[1])
        return server

    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Accept HUD control-channel connections; cap daemons must already be running."""
        await self.start()
        server = await self.bind(host, port)
        async with server:
            await server.serve_forever()

    async def start(self) -> None:
        """Bring up any backing capability daemons. Idempotent per registered hook.

        No-op unless something (e.g. the legacy adapter) registered ``_on_start``
        hooks. Run once by the substrate before the control channel serves, so the
        ``hello`` manifest reflects any capabilities the hooks publish.
        """
        for hook in self._on_start:
            await hook()

    async def stop(self) -> None:
        """Tear down backing daemons started by :meth:`start` (best-effort)."""
        for hook in reversed(self._on_stop):
            with contextlib.suppress(Exception):
                await hook()

    # ─── per-connection protocol dispatch (transport-agnostic) ───────────

    async def _handle_session(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session_id = "sess-" + secrets.token_hex(4)
        active_runner: TaskRunner | None = None
        # No configured token => open channel; otherwise hello must authenticate.
        authenticated = self._auth_token is None

        async def reply_to(msg_id: int | None, result: dict[str, Any]) -> None:
            if msg_id is not None:
                await send_frame(writer, reply(msg_id, result))

        async def error_to(msg_id: int | None, code: int, message: str) -> None:
            if msg_id is not None:
                await send_frame(writer, error(msg_id, code, message))

        try:
            while True:
                try:
                    msg = await read_frame(reader)
                except WireProtocolError as exc:
                    LOGGER.warning("closing %s: %s", session_id, exc)
                    return
                if msg is None:
                    return

                method = msg.get("method", "")
                params = msg.get("params") or {}
                msg_id = msg.get("id")

                try:
                    if method == "hello":
                        if self._auth_token is not None and not secrets.compare_digest(
                            str(params.get("token", "")), self._auth_token
                        ):
                            await error_to(msg_id, -32001, "unauthorized")
                            return
                        authenticated = True
                        await reply_to(
                            msg_id,
                            {
                                "session_id": session_id,
                                "env": {"name": self.name, "version": self.version},
                                "bindings": [c.to_manifest() for c in self.capabilities],
                            },
                        )

                    elif not authenticated:
                        await error_to(msg_id, -32001, "unauthorized: authenticate via hello")
                        continue

                    elif method == "tasks.list":
                        await reply_to(msg_id, {"tasks": self.tasks_manifest()})

                    elif method == "tasks.start":
                        task_id = params.get("id")
                        if not isinstance(task_id, str):
                            await error_to(msg_id, -32602, "tasks.start: 'id' must be a string")
                            continue
                        args = params.get("args") or {}
                        if not isinstance(args, dict):
                            await error_to(msg_id, -32602, "tasks.start: 'args' must be an object")
                            continue
                        if active_runner is not None:
                            await active_runner.cancel()
                        try:
                            active_runner = self._task_runner(task_id, args)
                        except KeyError as exc:
                            await error_to(msg_id, -32602, str(exc.args[0]))
                            continue
                        prompt = await active_runner.start()
                        await reply_to(msg_id, prompt)

                    elif method == "tasks.evaluate":
                        if active_runner is None:
                            await error_to(msg_id, -32600, "no task in progress")
                            continue
                        # Take ownership before evaluating so a failing evaluate
                        # doesn't leave the session pinned to a spent runner.
                        runner, active_runner = active_runner, None
                        evaluation = await runner.evaluate(params)
                        await reply_to(msg_id, evaluation)

                    elif method == "tasks.cancel":
                        if active_runner is not None:
                            await active_runner.cancel()
                            active_runner = None
                        await reply_to(msg_id, {"cancelled": True})

                    elif method == "bye":
                        await reply_to(msg_id, {"goodbye": True})
                        return

                    else:
                        await error_to(msg_id, -32601, f"method not found: {method}")

                except Exception as exc:
                    LOGGER.exception("error handling %s", method)
                    await error_to(msg_id, -32000, str(exc))

        finally:
            if active_runner is not None:
                with contextlib.suppress(Exception):
                    await active_runner.cancel()
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

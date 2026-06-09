"""Environment: declarative capabilities + tasks behind the HUD wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import secrets
from typing import TYPE_CHECKING, Any, ParamSpec, cast

from .legacy import LegacyEnvMixin
from .task import TaskRunner, _TaskFactory
from .utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from hud.capabilities import Capability

LOGGER = logging.getLogger("hud.environment.env")

P = ParamSpec("P")


class _NoTaskInProgress(RuntimeError):
    pass


class _TaskSession:
    """Per-control-connection task state.

    A connection owns its active runner while connected. If the connection drops
    after ``tasks.start`` but before ``tasks.grade``, the runner is parked on the
    environment so a later ``tasks.grade`` can resume it. This keeps the
    disconnect/resume rule in one place instead of repeating local-vs-parked
    branches across every protocol method.
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._runner: TaskRunner | None = None

    async def start(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        await self.cancel()
        self._runner = TaskRunner(self._env._task_factory(task_id), args)
        return await self._runner.start()

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        runner = self._runner or self._env._claim_parked_runner()
        if runner is None:
            raise _NoTaskInProgress("no task in progress")
        try:
            return await runner.grade(payload)
        finally:
            if runner is self._runner:
                self._runner = None

    async def cancel(self) -> None:
        if self._runner is not None:
            await self._runner.cancel()
            self._runner = None
        await self._env._cancel_parked_runner()

    async def detach(self) -> None:
        if self._runner is not None:
            await self._env._park_runner(self._runner)
            self._runner = None


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
        self._tasks: dict[str, _TaskFactory[Any]] = {}
        # A disconnected task start can be resumed by a later grade request.
        self._parked_runner: TaskRunner | None = None
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
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], _TaskFactory[P]]:
        """Register an async-generator task (``id`` defaults to the function name).

        The task yields a prompt, then — once the answer is sent back — a reward.
        Either form works (both normalized to the wire protocol): friendly (``yield
        prompt`` → ``yield reward``) or explicit (``yield {"prompt": ...}`` → ``yield
        {"score": ...}``). ``input``/``returns`` optionally declare the agent's I/O
        types (surfaced in the manifest as JSON schemas). The decorated callable
        returns a concrete :class:`~hud.eval.Task` when called with task args.
        """

        def decorate(func: Callable[P, AsyncGenerator[Any, Any]]) -> _TaskFactory[P]:
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
            task = _TaskFactory(
                self,
                task_id,
                description,
                func,
                input=input,
                returns=returns,
            )
            self._tasks[task_id] = cast("_TaskFactory[Any]", task)
            return task

        return decorate

    def add_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def task_entries(self) -> list[dict[str, Any]]:
        """Return manifest entries for registered tasks."""
        return [task.manifest_entry() for task in self._tasks.values()]

    async def task_prompt(self, task_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Materialize a task's first yield without parking a resumable run."""
        runner = TaskRunner(self._task_factory(task_id), args or {})
        try:
            return await runner.start()
        finally:
            await runner.cancel()

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
        metadata for discovery. :meth:`from_dict` restores identity + capabilities
        (runnable task funcs come from the env's source/image when launched).
        """
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": [c.to_manifest() for c in self.capabilities],
            "tasks": self.task_entries(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment:
        """Rebuild an Environment from :meth:`to_dict` output (identity + capabilities).

        Tasks are not reconstructed — their generator code lives in the env's
        source. A deserialized Environment carries identity + capability metadata only.
        """
        from hud.capabilities import Capability

        return cls(
            name=data["name"],
            version=data.get("version", "0.0.1"),
            capabilities=[Capability.from_manifest(c) for c in data.get("capabilities") or []],
        )

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

    def _task_factory(self, task_id: str) -> _TaskFactory[Any]:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id!r}")
        return task

    async def _park_runner(self, runner: TaskRunner) -> None:
        await self._cancel_parked_runner()
        self._parked_runner = runner

    def _claim_parked_runner(self) -> TaskRunner | None:
        runner = self._parked_runner
        self._parked_runner = None
        return runner

    async def _cancel_parked_runner(self) -> None:
        if self._parked_runner is not None:
            await self._parked_runner.cancel()
            self._parked_runner = None

    async def _handle_session(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session_id = "sess-" + secrets.token_hex(4)
        task_session = _TaskSession(self)

        async def reply_to(msg_id: int | None, result: dict[str, Any]) -> None:
            if msg_id is not None:
                await send_frame(writer, reply(msg_id, result))

        async def error_to(msg_id: int | None, code: int, message: str) -> None:
            if msg_id is not None:
                await send_frame(writer, error(msg_id, code, message))

        try:
            while True:
                msg = await read_frame(reader)
                if msg is None:
                    return

                method = msg.get("method", "")
                params = msg.get("params") or {}
                msg_id = msg.get("id")

                try:
                    if method == "hello":
                        await reply_to(
                            msg_id,
                            {
                                "session_id": session_id,
                                "env": {"name": self.name, "version": self.version},
                                "bindings": [c.to_manifest() for c in self.capabilities],
                            },
                        )

                    elif method == "tasks.list":
                        await reply_to(
                            msg_id,
                            {
                                "tasks": self.task_entries(),
                            },
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
                            prompt = await task_session.start(task_id, args)
                        except KeyError:
                            await error_to(msg_id, -32602, f"unknown task: {task_id!r}")
                            continue
                        await reply_to(msg_id, prompt)

                    elif method == "tasks.grade":
                        try:
                            evaluation = await task_session.grade(params)
                        except _NoTaskInProgress:
                            await error_to(msg_id, -32600, "no task in progress")
                            continue
                        await reply_to(msg_id, evaluation)

                    elif method == "tasks.cancel":
                        await task_session.cancel()
                        await reply_to(msg_id, {"cancelled": True})

                    elif method == "bye":
                        await task_session.cancel()
                        await reply_to(msg_id, {"goodbye": True})
                        return

                    else:
                        await error_to(msg_id, -32601, f"method not found: {method}")

                except Exception as exc:
                    LOGGER.exception("error handling %s", method)
                    await error_to(msg_id, -32000, str(exc))

        finally:
            await task_session.detach()
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

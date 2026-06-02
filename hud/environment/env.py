"""Environment: declarative capabilities + tasks behind the HUD wire protocol."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import secrets
from typing import TYPE_CHECKING, Any, ParamSpec, cast

from .task import Task, TaskRunner
from .utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from hud.capabilities import Capability

LOGGER = logging.getLogger("hud.environment.env")

P = ParamSpec("P")


class Environment:
    """Capabilities + tasks dispatched over the HUD wire protocol."""

    def __init__(
        self,
        *,
        name: str,
        version: str = "0.0.1",
        capabilities: list[Capability] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.capabilities: list[Capability] = list(capabilities or [])
        self._tasks: dict[str, Task[Any]] = {}
        # Backing-daemon lifecycle hooks (e.g. a legacy MCP server the adapter
        # stands up). Run once by the substrate (LocalSandbox) around serving.
        self._on_start: list[Callable[[], Awaitable[None]]] = []
        self._on_stop: list[Callable[[], Awaitable[None]]] = []

    # ─── task registration ───────────────────────────────────────────

    def task(
        self,
        *,
        id: str | None = None,
        description: str = "",
        input: Any = None,
        returns: Any = None,
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], Task[P]]:
        """Register an async-generator task. ``id`` defaults to the function name.

        A task yields a prompt for the agent, then — once the answer is sent back —
        yields a reward. The friendly form yields a raw prompt then a float /
        ``EvaluationResult``; the explicit form yields ``{"prompt": ...}`` then
        ``{"score": ...}``. Both are normalized to the wire protocol, so write
        whichever reads better.

        ``input`` declares the type(s) the agent is given (a model or union of
        models; ``None`` = plain text); ``returns`` declares the type the agent
        must produce (``None`` = plain text, else the answer is parsed into
        ``AgentAnswer[returns]``). Both surface in the task manifest (as JSON
        schemas) so an agent can inspect whether the task fits it.

        Returns the :class:`~hud.environment.task.Task` — calling it with the task's
        args yields a runnable :class:`~hud.eval.Variant`.
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

    def scenario(
        self,
        name: str | None = None,
        *,
        description: str = "",
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], Task[P]]:
        """Deprecated alias for :meth:`task`. Prefer ``@env.task``."""
        return self.task(id=name, description=description)

    def add_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def initialize(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register an initializer, run once before the control channel serves.

        Use it to bring up a backing daemon and publish its capability — e.g. start
        a :class:`~hud.environment.Workspace` and ``add_capability`` its SSH endpoint::

            ws = Workspace()

            @env.initialize
            async def _serve_shell() -> None:
                await ws.start()
                env.add_capability(Capability.ssh(
                    url=ws.ssh_url, user=ws.ssh_user,
                    host_pubkey=ws.ssh_host_pubkey, client_key_path=ws.ssh_client_key_path,
                ))
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
            "tasks": [t.manifest_entry() for t in self._tasks.values()],
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

    async def _handle_session(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session_id = "sess-" + secrets.token_hex(4)
        active_runner: TaskRunner | None = None

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
                                "tasks": [t.manifest_entry() for t in self._tasks.values()],
                            },
                        )

                    elif method == "tasks.start":
                        task_id = params.get("id")
                        if not isinstance(task_id, str):
                            await error_to(msg_id, -32602, "tasks.start: 'id' must be a string")
                            continue
                        task = self._tasks.get(task_id)
                        if task is None:
                            await error_to(msg_id, -32602, f"unknown task: {task_id!r}")
                            continue
                        args = params.get("args") or {}
                        if not isinstance(args, dict):
                            await error_to(
                                msg_id, -32602, "tasks.start: 'args' must be an object"
                            )
                            continue
                        if active_runner is not None:
                            await active_runner.cancel()
                        active_runner = TaskRunner(task, args)
                        prompt = await active_runner.start()
                        await reply_to(msg_id, prompt)

                    elif method == "tasks.evaluate":
                        if active_runner is None:
                            await error_to(msg_id, -32600, "no task in progress")
                            continue
                        evaluation = await active_runner.evaluate(params)
                        active_runner = None
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

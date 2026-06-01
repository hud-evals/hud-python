"""Env: declarative capabilities + tasks behind the HUD wire protocol. Single-tenant."""

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
    from collections.abc import AsyncGenerator, Callable

    from hud.capabilities import Capability

LOGGER = logging.getLogger("hud.env.env")

P = ParamSpec("P")


class Env:
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

    # ─── task registration ───────────────────────────────────────────

    def task(
        self,
        *,
        id: str | None = None,
        description: str = "",
    ) -> Callable[[Callable[P, AsyncGenerator[dict[str, Any], dict[str, Any]]]], Task[P]]:
        """Register an async-generator task. ``id`` defaults to fn name.

        Returns the :class:`~hud.env.task.Task` — calling it with the task's args
        yields a runnable :class:`~hud.client.Variant`.
        """

        def decorate(
            func: Callable[P, AsyncGenerator[dict[str, Any], dict[str, Any]]],
        ) -> Task[P]:
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
            task = Task(self, task_id, description, func)
            self._tasks[task_id] = cast("Task[Any]", task)
            return task

        return decorate

    def add_capability(self, cap: Capability) -> None:
        self.capabilities.append(cap)

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
        server = await self.bind(host, port)
        async with server:
            await server.serve_forever()

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

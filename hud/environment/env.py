"""Environment: declarative capabilities + tasks behind the HUD wire protocol.

Pure declaration — what exists (identity, capabilities, registered tasks) and
the daemon hooks a substrate runs around serving. The protocol server that
puts a declaration on the wire lives in :mod:`hud.environment.server`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from hud.capabilities import Capability, Connection

from .egress import ConnectionRelay, Peer, WorkspaceRoute
from .workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
    from pathlib import Path

    from hud.eval import Task as EvalTask

P = ParamSpec("P")
T = TypeVar("T")

#: Control-session id for the running accept/cancel task (robot slot claims key on it).
current_session_id: ContextVar[str | None] = ContextVar("hud_current_session_id", default=None)


class _TaskFunction(Protocol[P]):
    __name__: str
    __doc__: str | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any, Any]: ...


class Answer(BaseModel, Generic[T]):
    """The maybe-parsed answer a ``returns=``-typed task receives for grading.

    When a task specifies ``returns=SomeModel``, the answer received by the
    task's evaluate phase is an ``Answer[SomeModel]``: ``content`` is the agent's
    answer parsed into the declared type (or the original string when parsing
    failed — grade it accordingly), ``raw`` is always the string as submitted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: T = Field(description="The parsed structured answer")
    raw: str = Field(default="", description="Original answer string before parsing")


def _args_json_schema(sig: inspect.Signature) -> dict[str, Any]:
    """JSON Schema for a task function's parameters — the task's args contract.

    Published in the manifest (`tasks.list`) so the platform can validate
    stored task args at sync time and render argument forms. Unannotated params
    accept anything.
    """
    fields: dict[str, Any] = {}
    allow_additional = False
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            allow_additional = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    schema = create_model("TaskArgs", **fields).model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = allow_additional
    return schema


class _TaskFactory(Generic[P]):
    """Registered ``@env.template`` callable that creates concrete public tasks.

    The server side (:class:`~hud.environment.server.TaskRunner`) drives its
    async-generator ``func`` (prompt → score); calling this object with args
    binds a runnable :class:`~hud.eval.Task`::

        task = fix_bug(difficulty=3)  # -> Task
        job = await task.run(agent)
    """

    def __init__(
        self,
        env: Environment,
        id: str,
        description: str,
        func: _TaskFunction[P],
        *,
        input: Any = None,
        returns: Any = None,
    ) -> None:
        self.env = env
        self.id = id
        self.description = description
        self.func: Callable[..., AsyncGenerator[Any, Any]] = func
        #: Type(s) the agent is given as input (a model or union; ``None`` = text).
        self.input_type = input
        #: Type the agent must produce (``None`` = plain text). Drives answer
        #: deserialization into ``Answer[T]``.
        self.return_type = returns
        self.sig = inspect.signature(func, eval_str=True)
        functools.update_wrapper(self, func)

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "args": _args_json_schema(self.sig),
        }
        for key, typ in (("input", self.input_type), ("returns", self.return_type)):
            if typ is not None:
                entry[key] = TypeAdapter(typ).json_schema()
        return entry

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EvalTask:
        # Avoid the environment -> eval import cycle.
        from hud.eval.task import Task

        bound = self.sig.bind(*args, **kwargs)
        task = Task(env=self.env.name, id=self.id, args=dict(bound.arguments))
        task._env = self.env
        return task


class Environment:
    """Capabilities + tasks dispatched over the HUD wire protocol."""

    def __init__(
        self,
        name: str = "environment",
        *,
        version: str = "0.0.1",
        capabilities: Sequence[Capability] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        #: Published capabilities — always concrete wire data. Daemons the env
        #: runs itself publish theirs at serve time (:meth:`add_capability`
        #: from an ``@env.initialize`` hook; :meth:`workspace` wires the
        #: common ssh case).
        self.capabilities: list[Capability] = []
        self._started = False
        self._hooks_done = False  # True only after all @env.initialize hooks have completed
        for entry in capabilities or []:
            self.add_capability(entry)
        #: Registered task templates by id (the ``@env.template`` registry).
        #: Each value mints concrete :class:`~hud.eval.Task` rows when called.
        self.tasks: dict[str, _TaskFactory[Any]] = {}
        # Backing-daemon lifecycle hooks run once by the serving substrate
        # around its lifetime.
        self._on_start: list[Callable[[], Awaitable[None]]] = []
        self._on_stop: list[Callable[[], Awaitable[None]]] = []
        # Per task-session end (cancel / bye / post-grade cleanup).
        self._on_task_teardown: list[Callable[[], Awaitable[None]]] = []
        self._workspaces: dict[str, Workspace] = {}
        self._workspace_routes: dict[WorkspaceRoute, tuple[Workspace, Peer | None]] = {}
        self._connections: dict[str, tuple[Connection, Workspace, Peer, ConnectionRelay]] = {}

    # ─── task registration ───────────────────────────────────────────

    @property
    def templates(self) -> dict[str, _TaskFactory[Any]]:
        """The registered ``@env.template`` factories by id (alias of ``tasks``)."""
        return self.tasks

    def template(
        self,
        *,
        id: str | None = None,
        description: str = "",
        input: Any = None,
        returns: Any = None,
    ) -> Callable[[_TaskFunction[P]], _TaskFactory[P]]:
        """Register a **task template** — an async generator that mints tasks.

        The generator yields a prompt, then — once the answer is sent back — a
        reward. Either form works (both normalized to the wire protocol):
        friendly (``yield prompt`` → ``yield reward``) or explicit (``yield
        {"prompt": ...}`` → ``yield {"score": ...}``). ``input``/``returns``
        optionally declare the agent's I/O types (surfaced in the manifest as
        JSON schemas). The decorated callable is a *template*: calling it with
        args returns a concrete :class:`~hud.eval.Task` row.
        """

        def decorate(func: _TaskFunction[P]) -> _TaskFactory[P]:
            if not inspect.isasyncgenfunction(func):
                raise TypeError(
                    f"@env.template: {getattr(func, '__qualname__', func)} must be an async "
                    "generator function (`async def ...:` with `yield`)",
                )
            task_id = id or func.__name__
            if task_id in self.tasks:
                raise ValueError(
                    f"template {task_id!r} already registered on env {self.name!r}",
                )
            task = _TaskFactory(
                self,
                task_id,
                description,
                func,
                input=input,
                returns=returns,
            )
            self.tasks[task_id] = cast("_TaskFactory[Any]", task)
            return task

        return decorate

    def initialize(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register an initializer, run once before the control channel serves.

        Seed state, or stand up a daemon and publish its address with
        :meth:`add_capability` — that is how capabilities the env runs itself
        come into existence at serve time rather than at import.
        """
        self._on_start.append(fn)
        return fn

    def shutdown(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register a teardown hook (run in reverse order on stop)."""
        self._on_stop.append(fn)
        return fn

    # ─── capabilities ─────────────────────────────────────────────────────

    def add_capability(self, cap: Capability) -> None:
        """Publish concrete wire data, replacing any same-named entry.

        Call at declaration for services that already exist, or from an
        ``@env.initialize`` hook once a daemon the env runs is up. Replacement
        keeps restarts idempotent: a re-run hook overwrites its stale address.
        """
        if not isinstance(cap, Capability):
            raise TypeError(f"add_capability: expected Capability, got {cap!r}")
        if not cap.url:
            raise ValueError(
                f"capability {cap.name!r} has no url; start the service in an "
                "@env.initialize hook and publish its concrete address",
            )
        if self._hooks_done:
            import logging

            logging.getLogger("hud.environment").warning(
                "add_capability(%r) called after @env.initialize hooks have already run — "
                "the capability will not appear in any already-negotiated agent manifest. "
                "Move this call inside an @env.initialize hook.",
                cap.name,
            )
        self.capabilities = [c for c in self.capabilities if c.name != cap.name] + [cap]

    def capability(self, name: str) -> Capability:
        """Look up a published capability by name."""
        cap = next((c for c in self.capabilities if c.name == name), None)
        if cap is None:
            raise KeyError(f"unknown capability: {name!r}")
        return cap

    def workspace(
        self,
        root: Path | str,
        *,
        name: str = "shell",
        track_files: bool | None = None,
        **kwargs: Any,
    ) -> Workspace:
        """Attach a :class:`Workspace` serving ``name`` over ``ssh/2``.

        Registers the start → publish → stop lifecycle on this env's hooks;
        nothing touches the filesystem until the env actually serves. Extra
        kwargs go to :class:`Workspace` (``network=``, ``env=``, ...).

        When ``track_files`` is set (defaulting to ``HUD_FILE_TRACKING_ENABLED``)
        the workspace also publishes an observation-only ``filetracking/1``
        capability the rollout streams setup and agent diffs from.
        """
        if track_files is None:
            from hud.settings import settings

            track_files = settings.file_tracking_enabled
        if name in self._workspaces:
            raise ValueError(f"workspace capability {name!r} is already attached")
        ws = Workspace(root, track_files=track_files, **kwargs)
        self._workspaces[name] = ws

        @self.initialize
        async def _up() -> None:
            await ws.start()
            self.add_capability(ws.capability(name))
            if ws.tracks_files:
                self.add_capability(ws.file_tracking_capability())

        @self.shutdown
        async def _down() -> None:
            await ws.stop()

        return ws

    def gym(self, target: Any, *, name: str = "robot", **kwargs: Any) -> Any:
        """Attach a gym-style sim serving ``name`` over the ``robot`` protocol.

        ``target`` is a factory, gymnasium id (``"CartPole-v1"``), or constructed
        registry env (reduced to its spec). Registers spawn → publish → teardown
        on this env's hooks; nothing runs until serve. Returns a
        :class:`~hud.environment.robot.RobotEndpoint` (``sim.reset`` / ``sim.result``).

        Every capability the bridge declares is published, not just the wire —
        a ``bridge=`` subclass serving its own tools from the sim process shows
        up in the manifest alongside ``name``.
        """
        from hud.environment.robot import RobotEndpoint
        from hud.environment.robot.gym import gym_command

        sim = RobotEndpoint.spawn(gym_command(target, **kwargs)).attach(self)

        @self.initialize
        async def _up() -> None:
            await sim.start()
            for cap in await sim.capabilities(name):
                self.add_capability(cap)

        @self.shutdown
        async def _down() -> None:
            await sim.stop()

        return sim

    # ─── substrate-run daemon lifecycle ──────────────────────────────────

    async def start(self) -> None:
        """Run ``@env.initialize`` hooks. Idempotent until :meth:`stop`.

        Run by the substrate before the control channel serves, so every
        capability — including ones published by hooks — is concrete by the
        time a client says ``hello``.
        """
        if self._started:
            return
        self._started = True
        for hook in self._on_start:
            await hook()
        self._hooks_done = True

    async def stop(self) -> None:
        """Run ``@env.shutdown`` hooks in reverse order (best-effort)."""
        for hook in reversed(self._on_stop):
            with contextlib.suppress(Exception):
                await hook()
        for workspace, peer in reversed(self._workspace_routes.values()):
            if peer is not None:
                workspace.remove_peer(peer)
        self._workspace_routes.clear()
        for connection, workspace, peer, relay in reversed(self._connections.values()):
            workspace.remove_process_connection(connection.name, peer)
            workspace.remove_peer(peer)
            relay.stop()
        self._connections.clear()
        self._started = False
        self._hooks_done = False

    def _workspace_for(self, capability: str) -> Workspace:
        workspace = self._workspaces.get(capability)
        if workspace is None and capability in {"ssh", "ssh/2"}:
            if len(self._workspaces) > 1:
                names = ", ".join(sorted(self._workspaces))
                raise RuntimeError(f"workspace capability {capability!r} is ambiguous: {names}")
            workspace = next(iter(self._workspaces.values()), None)
        if workspace is None:
            raise RuntimeError(f"workspace capability {capability!r} does not exist")
        return workspace

    def bind_connections(self, connections: Sequence[Connection]) -> None:
        """Install controller connections before a workspace starts its sandbox."""
        if not self._started:
            raise RuntimeError("environment must be started before connections are bound")

        bound: list[tuple[Connection, Workspace, Peer, ConnectionRelay]] = []
        try:
            for connection in connections:
                existing = self._connections.get(connection.name)
                if existing is not None:
                    if existing[0] != connection:
                        raise RuntimeError(f"connection {connection.name!r} was already bound")
                    continue
                workspace = self._workspace_for(connection.capability)
                if not workspace.supports_process_connections:
                    raise RuntimeError(
                        f"workspace capability {connection.capability!r} does not support "
                        "process-bound connections"
                    )
                if any(
                    peer.name == connection.host and peer.port == connection.port
                    for peer in workspace.peers
                ):
                    raise RuntimeError(
                        f"connection endpoint {connection.host}:{connection.port} conflicts with "
                        "an authored peer"
                    )
                relay = ConnectionRelay(connection)
                relay.start()
                peer = Peer(
                    connection.host,
                    connection.port,
                    target=("127.0.0.1", relay.port),
                )
                workspace.add_peer(peer, first=True)
                workspace.add_process_connection(connection.name, peer)
                record = (connection, workspace, peer, relay)
                self._connections[connection.name] = record
                bound.append(record)
        except BaseException:
            for connection, workspace, peer, relay in reversed(bound):
                workspace.remove_process_connection(connection.name, peer)
                workspace.remove_peer(peer)
                relay.stop()
                self._connections.pop(connection.name, None)
            raise

    def bind_workspace_routes(self, routes: Sequence[WorkspaceRoute]) -> None:
        """Install controller routes before a workspace starts its sandbox."""
        if not self._started:
            raise RuntimeError("environment must be started before workspace routes are bound")

        planned: list[tuple[WorkspaceRoute, Workspace, Peer | None]] = []
        for route in dict.fromkeys(routes):
            if route in self._workspace_routes:
                continue
            workspace = self._workspace_for(route.capability)
            if not workspace.bwrap_available or not workspace.owns_netns:
                raise RuntimeError(
                    f"workspace route for {route.capability!r} requires an isolated network"
                )
            matching = [
                peer
                for peer in workspace.peers
                if peer.name == route.host and peer.port == route.port
            ]
            if matching:
                if any(peer.address != (route.host, route.port) for peer in matching):
                    raise RuntimeError(
                        f"workspace route {route.host}:{route.port} conflicts with an authored peer"
                    )
                planned.append((route, workspace, None))
                continue
            planned.append(
                (
                    route,
                    workspace,
                    Peer(route.host, route.port, target=(route.host, route.port)),
                )
            )

        bound: list[tuple[WorkspaceRoute, Workspace, Peer | None]] = []
        try:
            for route, workspace, peer in planned:
                if peer is not None:
                    workspace.add_peer(peer, first=True)
                self._workspace_routes[route] = (workspace, peer)
                bound.append((route, workspace, peer))
        except BaseException:
            for route, workspace, peer in reversed(bound):
                if peer is not None:
                    workspace.remove_peer(peer)
                self._workspace_routes.pop(route, None)
            raise

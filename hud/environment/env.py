"""Environment: capabilities and task templates behind the HUD wire protocol.

It declares what exists (identity, capabilities, registered templates) and the
daemon hooks a substrate runs around serving, and holds the per-session
connections and routes the server binds. ``hud.environment.server`` puts it on
the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import types
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from hud.capabilities.base import Capability
from hud.settings import settings

from .egress import ConnectionRelay, Peer, WorkspaceRoute
from .workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
    from pathlib import Path

    from hud.capabilities.connection import Connection
    from hud.eval.task import Task as EvalTask

LOGGER = logging.getLogger("hud.environment")

P = ParamSpec("P")
T = TypeVar("T")

# The control session whose task call is running; session-keyed state (robot slot
# claims, per-session state dirs) keys on it.
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
    """JSON Schema for a task function's parameters, published in the ``tasks.list`` manifest.

    The platform validates stored task args against it and renders argument forms
    from it. Unannotated parameters accept anything.
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


def _parameter_signature(func: types.FunctionType) -> inspect.Signature:
    """``func``'s signature with its parameter annotations evaluated.

    A template's return annotation is never read, so it is left unevaluated and may
    name a type imported only under ``TYPE_CHECKING``.
    """
    params_only = types.FunctionType(
        func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__
    )
    params_only.__kwdefaults__ = func.__kwdefaults__
    params_only.__annotations__ = {
        name: hint for name, hint in func.__annotations__.items() if name != "return"
    }
    return inspect.signature(params_only, eval_str=True)


class TaskTemplate(Generic[P]):
    """A registered ``@env.template``: calling it with args binds a runnable ``Task``.

    ``TaskRunner`` drives its async-generator ``func`` when the env is served.
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
        # Type(s) the agent is given as input (a model or union; ``None`` = text).
        self.input_type = input
        # Type the agent must produce (``None`` = plain text). Drives answer
        # deserialization into ``Answer[T]``.
        self.return_type = returns
        self.sig = _parameter_signature(cast("types.FunctionType", func))
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
        return Task(env=self.env.name, id=self.id, args=dict(bound.arguments), environment=self.env)


@dataclass(frozen=True, slots=True)
class _BoundConnection:
    """A controller connection relayed into one workspace for one control session."""

    connection: Connection
    workspace: Workspace
    peer: Peer
    relay: ConnectionRelay
    session: str

    def release(self) -> None:
        self.workspace.remove_process_connection(self.connection.name, self.peer)
        self.workspace.remove_peer(self.peer)
        self.relay.stop()


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
        # Published capabilities; hooks publish the ones the env runs itself at serve time.
        self.capabilities: list[Capability] = []
        self._started = False
        self._hooks_done = False
        for entry in capabilities or []:
            self.add_capability(entry)
        # Registered task templates by id (the ``@env.template`` registry).
        # Each value mints concrete ``Task`` rows when called.
        self.tasks: dict[str, TaskTemplate[Any]] = {}
        # Backing-daemon lifecycle hooks run once by the serving substrate
        # around its lifetime.
        self._on_start: list[Callable[[], Awaitable[None]]] = []
        self._on_stop: list[Callable[[], Awaitable[None]]] = []
        self._on_task_teardown: list[Callable[[], Awaitable[None]]] = []
        self._workspaces: dict[str, Workspace] = {}
        # Controller routes by the control sessions holding them; ``None`` peer when
        # an authored peer already provides the route.
        self._workspace_routes: dict[WorkspaceRoute, tuple[Workspace, Peer | None, set[str]]] = {}
        self._connections: dict[str, _BoundConnection] = {}
        self._binding = asyncio.Lock()

    @property
    def templates(self) -> dict[str, TaskTemplate[Any]]:
        """The registered ``@env.template`` factories by id (alias of ``tasks``)."""
        return self.tasks

    def template(
        self,
        *,
        id: str | None = None,
        description: str = "",
        input: Any = None,
        returns: Any = None,
    ) -> Callable[[_TaskFunction[P]], TaskTemplate[P]]:
        """Register an async-generator task template; calling it with args returns a ``Task``.

        The generator yields a prompt, then, once the answer is sent back, a reward:
        ``yield prompt`` then ``yield reward``, or ``yield {"prompt": ...}`` then
        ``yield {"score": ...}``. ``input`` and ``returns`` declare the agent's input
        and answer types, published in the manifest as JSON schemas; ``returns`` also
        parses the answer into ``Answer[T]``.
        """

        def decorate(func: _TaskFunction[P]) -> TaskTemplate[P]:
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
            task = TaskTemplate(
                self,
                task_id,
                description,
                func,
                input=input,
                returns=returns,
            )
            self.tasks[task_id] = cast("TaskTemplate[Any]", task)
            return task

        return decorate

    def initialize(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register an initializer, run once before the control channel serves.

        Seed state, or stand up a daemon and publish its address with
        ``add_capability`` — that is how capabilities the env runs itself
        come into existence at serve time rather than at import.
        """
        self._on_start.append(fn)
        return fn

    def shutdown(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register a teardown hook (run in reverse order on stop)."""
        self._on_stop.append(fn)
        return fn

    def task_teardown(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register a hook run each time a task session ends: cancelled, closed or graded."""
        self._on_task_teardown.append(fn)
        return fn

    async def end_task(self) -> None:
        """Run every task-teardown hook; one that fails is logged and the rest still run."""
        for hook in self._on_task_teardown:
            try:
                await hook()
            except Exception:
                LOGGER.exception("task teardown hook %r failed", hook)

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
            LOGGER.warning(
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
        """Attach a ``Workspace`` serving ``name`` over ``ssh/2``.

        Registers the start → publish → stop lifecycle on this env's hooks;
        nothing touches the filesystem until the env actually serves. Extra
        kwargs go to ``Workspace`` (``network=``, ``env=``, ...).

        When ``track_files`` is set (defaulting to ``HUD_FILE_TRACKING_ENABLED``)
        the workspace also publishes an observation-only ``filetracking/1``
        capability the rollout streams setup and agent diffs from.
        """
        if track_files is None:
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
        ``RobotEndpoint`` (``sim.reset`` / ``sim.result``).

        Every capability the bridge declares is published, not just the wire —
        a ``bridge=`` subclass serving its own tools from the sim process shows
        up in the manifest alongside ``name``.
        """
        from hud.environment.robot.endpoint import RobotEndpoint
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

    async def start(self) -> None:
        """Run ``@env.initialize`` hooks. Idempotent until ``stop``.

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
        for workspace, peer, _sessions in reversed(self._workspace_routes.values()):
            if peer is not None:
                workspace.remove_peer(peer)
        self._workspace_routes.clear()
        for bound in reversed(self._connections.values()):
            bound.release()
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

    async def bind_connections(self, connections: Sequence[Connection], *, session: str) -> None:
        """Install one control session's controller connections into their workspaces.

        A connection belongs to the session that supplied it until
        ``release_session``; a resumed session rebinds its own, replacing a
        rotated credential, and no session can take a name another one holds.
        """
        async with self._binding:
            if not self._started:
                raise RuntimeError("environment must be started before connections are bound")
            owned = [
                bound.connection for bound in self._connections.values() if bound.session == session
            ]
            if owned == list(connections):
                return
            for connection in connections:
                holder = self._connections.get(connection.name)
                if holder is not None and holder.session != session:
                    raise RuntimeError(
                        f"connection {connection.name!r} is bound to another control session"
                    )
            for name, held in reversed(list(self._connections.items())):
                if held.session == session:
                    held.release()
                    del self._connections[name]

            bound: list[_BoundConnection] = []
            try:
                for connection in connections:
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
                            f"connection endpoint {connection.host}:{connection.port} "
                            "conflicts with an authored peer"
                        )
                    relay = ConnectionRelay(connection)
                    relay.start()
                    peer = Peer(
                        connection.host,
                        connection.port,
                        target=("127.0.0.1", relay.port),
                    )
                    await workspace.add_peer(peer, first=True)
                    workspace.add_process_connection(connection.name, peer)
                    record = _BoundConnection(connection, workspace, peer, relay, session)
                    self._connections[connection.name] = record
                    bound.append(record)
            except BaseException:
                for record in reversed(bound):
                    record.release()
                    self._connections.pop(record.connection.name, None)
                raise

    def release_session(self, session: str) -> None:
        """Release a control session's relays, and its routes once no other session holds them."""
        for name, bound in reversed(list(self._connections.items())):
            if bound.session == session:
                bound.release()
                del self._connections[name]
        for route, (workspace, peer, sessions) in reversed(list(self._workspace_routes.items())):
            sessions.discard(session)
            if not sessions:
                if peer is not None:
                    workspace.remove_peer(peer)
                del self._workspace_routes[route]

    async def bind_workspace_routes(
        self, routes: Sequence[WorkspaceRoute], *, session: str
    ) -> None:
        """Install a control session's routes into their workspaces."""
        async with self._binding:
            if not self._started:
                raise RuntimeError("environment must be started before workspace routes are bound")

            planned: list[tuple[WorkspaceRoute, Workspace, Peer | None]] = []
            for route in dict.fromkeys(routes):
                if (held := self._workspace_routes.get(route)) is not None:
                    held[2].add(session)
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
                            f"workspace route {route.host}:{route.port} conflicts with "
                            "an authored peer"
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
                        await workspace.add_peer(peer, first=True)
                    self._workspace_routes[route] = (workspace, peer, {session})
                    bound.append((route, workspace, peer))
            except BaseException:
                for route, workspace, peer in reversed(bound):
                    if peer is not None:
                        workspace.remove_peer(peer)
                    self._workspace_routes.pop(route, None)
                raise

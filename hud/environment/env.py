"""Environment: declarative capabilities + tasks behind the HUD wire protocol.

Pure declaration — what exists (identity, capabilities, registered tasks) and
the daemon hooks a substrate runs around serving. The protocol server that
puts a declaration on the wire lives in :mod:`hud.environment.server`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, cast

from pydantic import TypeAdapter

from hud.capabilities import Capability

from .legacy import LegacyEnvMixin
from .workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from hud.eval import Task as EvalTask

P = ParamSpec("P")


class _TaskFactory(Generic[P]):
    """Registered ``@env.task`` callable that creates concrete public tasks.

    The server side (:class:`~hud.environment.server.TaskRunner`) drives its
    async-generator ``func`` (prompt → score); calling this object with args
    binds a runnable :class:`~hud.eval.Task`::

        task = fix_bug(difficulty=3)  # -> Task
        run = await task.run(agent, on=spawn("env.py"))
    """

    def __init__(
        self,
        env: Environment,
        id: str,
        description: str,
        func: Callable[P, AsyncGenerator[Any, Any]],
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
        #: deserialization into ``AgentAnswer[T]``.
        self.return_type = returns
        self.sig = inspect.signature(func)
        functools.update_wrapper(self, func)

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"id": self.id, "description": self.description}
        for key, typ in (("input", self.input_type), ("returns", self.return_type)):
            if typ is not None:
                entry[key] = TypeAdapter(typ).json_schema()
        return entry

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EvalTask:
        from hud.eval.task import Task  # local import: avoid env<->eval cycle

        bound = self.sig.bind(*args, **kwargs)
        return Task(env=self.env, id=self.id, args=dict(bound.arguments))


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
        capabilities: Sequence[Capability] | None = None,
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
        #: Declared capabilities — pure data. Entries with an empty ``url`` are
        #: *backed*: :meth:`resolve_capability` materializes the daemon (e.g. a
        #: managed ``Workspace``) when the env answers ``hello``.
        self.capabilities: list[Capability] = []
        for entry in capabilities or []:
            if not isinstance(entry, Capability):
                raise TypeError(
                    f"Environment(capabilities=...): expected Capability, got {entry!r}",
                )
            self.capabilities.append(entry)
        #: Daemons materialized for backed declarations, keyed by capability name.
        self._backings: dict[str, Workspace] = {}
        self._started = False
        #: Registered task factories by id (the ``@env.task`` registry).
        self.tasks: dict[str, _TaskFactory[Any]] = {}
        # Backing-daemon lifecycle hooks (e.g. a legacy MCP server the adapter
        # stands up). Run once by the serving substrate around its lifetime.
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
            if task_id in self.tasks:
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
            self.tasks[task_id] = cast("_TaskFactory[Any]", task)
            return task

        return decorate

    def initialize(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register an initializer, run once before the control channel serves.

        Use it to start a hand-rolled backing daemon. Daemons that own their
        capability (e.g. a :class:`~hud.environment.Workspace`) don't need a
        hook — declare them directly (``Environment(..., capabilities=[ws])``)
        and the substrate starts them.
        """
        self._on_start.append(fn)
        return fn

    def shutdown(self, fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        """Register a teardown hook (run in reverse order on stop)."""
        self._on_stop.append(fn)
        return fn

    # ─── substrate-run daemon lifecycle ──────────────────────────────────

    async def start(self) -> None:
        """Run ``@env.initialize`` hooks. Idempotent until :meth:`stop`.

        Run by the substrate before the control channel serves. Backed
        capability daemons are *not* started here — they materialize when the
        env answers ``hello`` (:meth:`resolve_capability`).
        """
        if self._started:
            return
        self._started = True
        for hook in self._on_start:
            await hook()

    async def stop(self) -> None:
        """Tear down hooks and any backing daemons that materialized (best-effort)."""
        for hook in reversed(self._on_stop):
            with contextlib.suppress(Exception):
                await hook()
        for backing in reversed(self._backings.values()):
            with contextlib.suppress(Exception):
                await backing.stop()
        self._backings.clear()
        self._started = False

    # ─── capability resolution (drives the ``hello`` manifest) ────────────

    async def resolve_capability(self, name: str) -> Capability:
        """Resolve a declared capability to concrete wire data.

        Concrete declarations (non-empty ``url``) are returned as-is. Backed
        declarations materialize their daemon here — for ``ssh/2``, a managed
        :class:`Workspace` built from the declaration's params — so addresses
        come into existence when the env serves a client, never at
        declaration/import time. Idempotent: one daemon per name.
        """
        entry = next((c for c in self.capabilities if c.name == name), None)
        if entry is None:
            raise KeyError(f"unknown capability: {name!r}")
        if entry.url:
            return entry
        family = entry.protocol.split("/", 1)[0]
        if family != "ssh":
            raise RuntimeError(
                f"capability {name!r} ({entry.protocol}) has no url and no managed "
                "backing; declare it with a concrete url",
            )
        backing = self._backings.get(name)
        if backing is None:
            backing = Workspace(**entry.params)
            self._backings[name] = backing
        await backing.start()
        return backing.capability(name=entry.name)

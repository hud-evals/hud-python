"""Environment: declarative capabilities + tasks behind the HUD wire protocol.

Pure declaration — what exists (identity, capabilities, registered tasks) and
the daemon hooks a substrate runs around serving. The protocol server that
puts a declaration on the wire lives in :mod:`hud.environment.server`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from hud.capabilities import Capability

from .legacy import LegacyEnvMixin
from .workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
    from pathlib import Path

    from hud.eval import Task as EvalTask

P = ParamSpec("P")
T = TypeVar("T")


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
        job = await task.run(agent, runtime=LocalRuntime("env.py"))
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
        #: deserialization into ``Answer[T]``.
        self.return_type = returns
        self.sig = inspect.signature(func)
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
        # The one sanctioned upward import: eval sits above environment and
        # agents and imports both; neither imports eval. Calling a declaration
        # is where env hands the row to eval, and the import stays local to
        # break the load-time cycle. Don't add more edges like this.
        from hud.eval.task import Task

        bound = self.sig.bind(*args, **kwargs)
        task = Task(env=self.env.name, id=self.id, args=dict(bound.arguments))
        # Record where this template was defined so ``task.run()`` can default to
        # serving that source locally (in-process only; never crosses the wire).
        source = inspect.getsourcefile(self.func)
        if source is not None:
            task._source = source
        return task


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
        # Backing-daemon lifecycle hooks (e.g. a legacy MCP server the adapter
        # stands up). Run once by the serving substrate around its lifetime.
        self._on_start: list[Callable[[], Awaitable[None]]] = []
        self._on_stop: list[Callable[[], Awaitable[None]]] = []
        self._init_legacy()

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
    ) -> Callable[[Callable[P, AsyncGenerator[Any, Any]]], _TaskFactory[P]]:
        """Register a **task template** — an async generator that mints tasks.

        The generator yields a prompt, then — once the answer is sent back — a
        reward. Either form works (both normalized to the wire protocol):
        friendly (``yield prompt`` → ``yield reward``) or explicit (``yield
        {"prompt": ...}`` → ``yield {"score": ...}``). ``input``/``returns``
        optionally declare the agent's I/O types (surfaced in the manifest as
        JSON schemas). The decorated callable is a *template*: calling it with
        args returns a concrete :class:`~hud.eval.Task` row.
        """

        def decorate(func: Callable[P, AsyncGenerator[Any, Any]]) -> _TaskFactory[P]:
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
        capability the rollout streams diffs from.
        """
        if track_files is None:
            from hud.settings import settings

            track_files = settings.file_tracking_enabled
        ws = Workspace(root, track_files=track_files, **kwargs)

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
        self._started = False
        self._hooks_done = False

"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, runtime_checkable
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from hud.clients import HudClient, HudProtocolError

from .compose import ComposeProject

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from hud.capabilities import Connection
    from hud.environment.egress import WorkspaceRoute
    from hud.eval.run import EnvironmentSession
    from hud.eval.task import Task

    from .hosted import HostedRuntime

LOGGER = logging.getLogger("hud.runtime")
# How long a connect waits for a freshly provisioned env to serve when nothing sets a limit.
DEFAULT_READY_TIMEOUT = 240.0
_CONNECT_RETRY_SECONDS = 0.5
_HEARTBEAT_INTERVAL_SECONDS = 120.0
_HEARTBEAT_TIMEOUT_SECONDS = 5.0


class RuntimeGPU(BaseModel):
    """Requested GPU resources, provider-neutral where possible."""

    model_config = ConfigDict(extra="forbid")

    type: str | list[str] | None = Field(default=None, min_length=1)
    count: int = Field(default=1, ge=1)

    @property
    def acceptable_types(self) -> list[str]:
        if self.type is None:
            return []
        return [self.type] if isinstance(self.type, str) else self.type


class RuntimeTPU(BaseModel):
    """Requested TPU slice."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    topology: str = Field(pattern=r"^[1-9]\d*(?:x[1-9]\d*)+$")


class RuntimeResources(BaseModel):
    """Provider-neutral runtime placement requests."""

    model_config = ConfigDict(extra="forbid")

    cpu: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpu: RuntimeGPU | None = None
    os: str | None = Field(default=None, min_length=1)
    tpu: RuntimeTPU | None = None

    def _require_support(self, provider: str, supported: set[str]) -> None:
        unsupported = self.model_dump(exclude_none=True).keys() - supported
        unsupported.discard("storage_mb")
        if unsupported:
            fields = ", ".join(f"runtime_config.resources.{name}" for name in sorted(unsupported))
            raise ValueError(f"{provider} does not support {fields}")


class RuntimeLimits(BaseModel):
    """Runtime lifecycle limits in seconds."""

    model_config = ConfigDict(extra="forbid")

    startup_timeout_s: int | None = Field(default=None, gt=0)
    run_timeout_s: int | None = Field(default=None, gt=0)


class RuntimeConfig(BaseModel):
    """Typed task-environment launch requirements.

    ``Task.runtime_config`` is the requested config; ``Runtime.config`` is the one applied.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, min_length=1)
    compose: ComposeProject | None = None
    resources: RuntimeResources | None = None
    limits: RuntimeLimits | None = None

    @field_serializer("resources", "limits", when_used="json")
    def _serialize_options(
        self,
        value: RuntimeResources | RuntimeLimits | None,
        info: SerializationInfo,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.model_dump(
            mode=info.mode,
            exclude_none=True,
            exclude_unset=True,
            context=info.context,
        )

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.image is not None and self.compose is not None:
            raise ValueError("runtime_config accepts either image or compose, not both")
        return self

    def with_overrides(self, override: RuntimeConfig | None) -> RuntimeConfig:
        if override is None:
            return self
        config = self.model_dump()
        changes = override.model_dump(exclude_unset=True)
        if override.image is not None:
            config["compose"] = None
        elif override.compose is not None:
            config["image"] = None
        return RuntimeConfig.model_validate(config | changes)


class Provider(Protocol):
    """Acquires one fresh env substrate for a task and yields its connectable ``Runtime``."""

    def __call__(self, task: Task, /) -> AbstractAsyncContextManager[Runtime]: ...


@runtime_checkable
class _ConfiguredProvider(Protocol):
    runtime_config: RuntimeConfig | None


def validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("runtime session id must be a single path component")


@dataclass(frozen=True)
class Runtime:
    """An acquired substrate: where its environment is, how to open sessions on it, and
    how to move a session's files between substrates.

    ``url`` is the environment's control channel; ``params`` are connection-time settings
    such as ``ready_timeout``. Constructed directly, a runtime is also a provider that
    yields itself and leaves teardown to whoever provisioned the substrate.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    config: RuntimeConfig | None = None

    # Whether ``snapshot_session`` can export a control session for another
    # runtime to restore. Without it, a verifier must run beside the actor.
    transports_sessions: ClassVar[bool] = False

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)

    @asynccontextmanager
    async def connect(
        self,
        *,
        session_id: str | None = None,
        lease: float | None = None,
        workspace_routes: Sequence[WorkspaceRoute] = (),
        connections: Sequence[Connection] = (),
    ) -> AsyncIterator[EnvironmentSession]:
        """Open a session on the environment over its control channel.

        ``session_id`` attaches to that open session instead; ``lease`` keeps the session
        open that many seconds after this connection ends, for a later one to attach.

        Connecting retries until the environment answers ``hello``, for
        ``params["ready_timeout"]`` seconds: a proxied port (``docker -p``, a port-forward)
        can accept before the environment behind it serves. While the session is open, a
        heartbeat probes the connection; a failure cancels the body and raises that error.
        """
        parts = urlsplit(self.url)
        if parts.scheme not in ("", "tcp"):
            raise NotImplementedError(
                f"control transport {parts.scheme!r} not supported yet (only tcp://)",
            )
        host, port = parts.hostname or "127.0.0.1", parts.port or 0
        timeout = self.params.get("ready_timeout", DEFAULT_READY_TIMEOUT)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("runtime.params['ready_timeout'] must be a positive finite number")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            try:
                reader, writer = await asyncio.open_connection(host, port)
            except OSError:
                if loop.time() >= deadline:
                    raise
                await asyncio.sleep(_CONNECT_RETRY_SECONDS)
                continue
            client = HudClient(reader, writer, endpoint=(host, port))
            try:
                await client.hello(
                    session_id,
                    lease=lease,
                    workspace_routes=workspace_routes,
                    connections=connections,
                )
                break
            except asyncio.CancelledError:
                client.abort()
                raise
            except (EOFError, OSError):
                # No live env behind the accepted connection yet: EOF on the reply, or a
                # reset racing the hello write.
                await client.close()
                if loop.time() >= deadline:
                    raise
                await asyncio.sleep(_CONNECT_RETRY_SECONDS)
            except BaseException:
                # A real failure still closes the socket: an unclosed connection parks the
                # env's connection handler.
                await client.close()
                raise

        owner = asyncio.current_task()
        assert owner is not None
        heartbeat_error: Exception | None = None

        async def heartbeat() -> None:
            nonlocal heartbeat_error
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
                try:
                    await client.ping(reply_timeout=_HEARTBEAT_TIMEOUT_SECONDS)
                except HudProtocolError as exc:
                    LOGGER.warning("control heartbeat failed: %s", exc)
                    return
                except Exception as exc:
                    LOGGER.warning("control heartbeat failed: %s", exc)
                    heartbeat_error = exc
                    owner.cancel()
                    return

        heartbeat_task = asyncio.create_task(heartbeat(), name="hud-control-heartbeat")
        try:
            try:
                yield client
            finally:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
                await client.close()
        except asyncio.CancelledError:
            if heartbeat_error is not None:
                raise heartbeat_error from None
            raise

    @asynccontextmanager
    async def snapshot_session(self, session_id: str) -> AsyncIterator[Path | None]:
        """Yield a portable archive of one control session's files when present.

        ``None`` means this runtime cannot export the session.
        """
        validate_session_id(session_id)
        yield None

    async def restore_session(self, session_id: str, source: Path) -> None:
        """Restore a portable archive into one control session."""
        validate_session_id(session_id)


class Shared:
    """Lease provider: at most ``width`` rollouts share each task placement.

    One substrate per environment and runtime config boots on its first lease and
    lives for the ``async with`` scope, or until its run timeout, when the next lease
    boots a fresh one. Excess leases wait for a slot. ``Taskset.run`` opens the scope
    for one call; open it yourself to keep the substrate warm across calls.
    """

    def __init__(self, inner: Provider, *, width: int) -> None:
        if width < 1:
            raise ValueError("Shared width must be >= 1")
        self.inner = inner
        self.width = width
        self._boot = asyncio.Lock()
        self._semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._addresses: dict[tuple[str, str], Runtime] = {}
        self._contexts: dict[tuple[str, str], contextlib.AsyncExitStack] = {}
        self._deadlines: dict[tuple[str, str], float] = {}
        self._opens = 0

    async def __aenter__(self) -> Self:
        self._opens += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._opens -= 1
        if self._opens == 0:
            contexts, self._contexts = self._contexts, {}
            self._addresses.clear()
            self._deadlines.clear()
            self._semaphores.clear()
            stack = contextlib.AsyncExitStack()
            for context in contexts.values():
                stack.push_async_callback(context.aclose)
            await stack.aclose()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if self._opens == 0:
            raise RuntimeError(
                "Shared substrates outlive single rollouts; lease inside the scope "
                "(Taskset.run opens it for you, or wrap calls in `async with Shared(...)`)"
            )
        config = (
            json.dumps(
                task.runtime_config.model_dump(mode="python", exclude_unset=True),
                sort_keys=True,
                default=str,
            )
            if task.runtime_config is not None
            else ""
        )
        key = (task.env, config)
        semaphore = self._semaphores.setdefault(key, asyncio.Semaphore(self.width))
        async with semaphore:
            async with self._boot:
                deadline = self._deadlines.get(key)
                if deadline is not None and deadline <= asyncio.get_running_loop().time():
                    self._addresses.pop(key)
                    self._deadlines.pop(key)
                    await self._contexts.pop(key).aclose()
                if key not in self._addresses:
                    # A failed boot retains nothing: it fails only its own rollout,
                    # and the next lease retries.
                    config = resolve_runtime_config(self.inner, task)
                    run_timeout = (
                        config.limits.run_timeout_s
                        if config is not None and config.limits is not None
                        else None
                    )
                    context = contextlib.AsyncExitStack()
                    address = await context.enter_async_context(self.inner(task))
                    self._contexts[key] = context
                    self._addresses[key] = address
                    if run_timeout is not None:
                        self._deadlines[key] = asyncio.get_running_loop().time() + run_timeout
                addr = self._addresses[key]
            yield addr


def resolve_runtime_config(provider: Provider | HostedRuntime, task: Task) -> RuntimeConfig | None:
    """Return the runtime configuration a placement will apply to a task."""
    if isinstance(provider, Shared):
        return resolve_runtime_config(provider.inner, task)
    if isinstance(provider, Runtime):
        return provider.config if provider.config is not None else task.runtime_config
    if isinstance(provider, _ConfiguredProvider) and provider.runtime_config is not None:
        base = provider.runtime_config
        config = base.with_overrides(task.runtime_config)
        return config if config.model_dump(exclude_none=True) else None
    return task.runtime_config

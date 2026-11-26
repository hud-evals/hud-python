"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from .compose import ComposeProject

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping
    from pathlib import Path

    from hud.eval.task import Task

    from .hosted import HostedRuntime

# How long a connect waits for a freshly provisioned env to serve when nothing sets a limit.
READY_TIMEOUT_SECONDS = 240.0


# Where a session keeps, inside its host, the state it hands to a verifier placed elsewhere.
SESSION_STATE_ROOT = PurePosixPath("/runtime/sessions")


def validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("runtime session id must be a single path component")


def session_dir(session_id: str) -> PurePosixPath:
    """``session_id``'s directory under ``SESSION_STATE_ROOT``."""
    validate_session_id(session_id)
    return SESSION_STATE_ROOT / session_id


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

    def require_support(self, provider: str, supported: set[str]) -> None:
        """Raise naming every resource set here that ``provider`` cannot honor.

        ``supported`` names the fields it can; ``storage_mb`` is never refused.
        """
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
    """Places a task: an async context manager yielding a connectable ``Runtime`` for its scope."""

    def __call__(self, task: Task, /) -> AbstractAsyncContextManager[Runtime]: ...


@runtime_checkable
class _ConfiguredProvider(Protocol):
    def config_for(self, task: Task, /) -> RuntimeConfig | None: ...


@dataclass(frozen=True)
class Runtime:
    """An acquired host: where its environment is, and how to move a session's files
    between hosts.

    ``url`` is the environment's control channel, and opening a session on it waits up to
    ``ready_timeout`` seconds for the environment behind it to answer; a WebSocket ``url``
    sends ``headers`` with its handshake. ``config`` is the runtime configuration its
    provider applied. Constructed directly, a runtime is also a provider that yields itself
    and leaves teardown to whoever provisioned the host.
    """

    url: str
    config: RuntimeConfig | None = None
    ready_timeout: float = field(default=READY_TIMEOUT_SECONDS, kw_only=True)
    headers: Mapping[str, str] = field(default_factory=dict, kw_only=True)

    # Whether ``snapshot_session`` can export a control session for another runtime to
    # restore. Without it, a verifier must run beside the actor.
    transports_sessions: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not 0 < self.ready_timeout < math.inf:
            raise ValueError("Runtime ready_timeout must be a positive, finite number of seconds")

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)

    def config_for(self, task: Task) -> RuntimeConfig | None:
        """The runtime configuration this placement applies to ``task``."""
        return self.config if self.config is not None else task.runtime_config

    @asynccontextmanager
    async def snapshot_session(self, session_id: str) -> AsyncGenerator[Path | None]:
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

    One host per environment and runtime config boots on its first lease and
    lives for the ``async with`` scope, or until its run timeout, when the next lease
    boots a fresh one. Excess leases wait for a slot. ``Taskset.run`` opens the scope
    for one call; open it yourself to keep the host warm across calls.
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
    async def __call__(self, task: Task) -> AsyncGenerator[Runtime]:
        if self._opens == 0:
            raise RuntimeError(
                "Shared hosts outlive single rollouts; lease inside the scope "
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
                    context = contextlib.AsyncExitStack()
                    address = await context.enter_async_context(self.inner(task))
                    self._contexts[key] = context
                    self._addresses[key] = address
                    limits = address.config.limits if address.config is not None else None
                    if limits is not None and limits.run_timeout_s is not None:
                        self._deadlines[key] = (
                            asyncio.get_running_loop().time() + limits.run_timeout_s
                        )
                addr = self._addresses[key]
            yield addr

    def config_for(self, task: Task) -> RuntimeConfig | None:
        """The runtime configuration this placement applies to ``task``."""
        return resolve_runtime_config(self.inner, task)


def resolve_runtime_config(provider: Provider | HostedRuntime, task: Task) -> RuntimeConfig | None:
    """The runtime configuration a placement applies to ``task``; a placement that
    cannot say is taken to apply the task's own."""
    if isinstance(provider, _ConfiguredProvider):
        return provider.config_for(task)
    return task.runtime_config

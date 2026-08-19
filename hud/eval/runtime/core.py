"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

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
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval.task import Task


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

    ``Task.runtime_config`` is requested construction input. ``Runtime.config``
    is the effective config used to construct a runtime.

    A Compose project uses local paths while authored and serialized project
    data in platform records.
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
    """Server placement: called with the task row being placed, acquire one
    fresh env substrate for it and yield its connectable :class:`Runtime`.

    A provider brings up the *server* (the env's control channel) wherever it
    lives — a local subprocess, a container, a cloud sandbox — and the agent
    loop drives it from this process (:func:`hud.eval.run.rollout`). The
    channel is location-transparent, so "co-located" (loopback) and "split"
    (agent here, env elsewhere) are the same code, differing only in the url.
    """

    def __call__(self, task: Task, /) -> AbstractAsyncContextManager[Runtime]: ...


@runtime_checkable
class _ConfiguredProvider(Protocol):
    runtime_config: RuntimeConfig | None


@dataclass(frozen=True)
class RuntimeSession:
    """One control session in a provisioned runtime."""

    session_id: str

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or self.session_id in {".", ".."}
            or "/" in self.session_id
            or "\\" in self.session_id
        ):
            raise ValueError("runtime session id must be a single path component")

    @asynccontextmanager
    async def snapshot(self) -> AsyncIterator[Path | None]:
        """Yield a portable archive of this session's files when present."""
        yield None

    async def restore(self, source: Path) -> None:
        """Restore a portable session archive into this session."""
        return


@dataclass(frozen=True)
class Runtime:
    """The connectable address of a provisioned substrate.

    ``url`` is the control-channel address (``tcp://127.0.0.1:7000`` for a
    local process, ``tcp://sandbox-abc.hud.so:443`` for a hosted box).
    ``params`` carries connection-time data a transport may need (auth token,
    sandbox id). ``config`` is the effective runtime configuration used to
    construct the runtime. Constructed directly, it is also a provider — the
    borrowed, shared case: it yields itself with a no-op lifecycle, since
    whoever provisioned the substrate owns its teardown.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    config: RuntimeConfig | None = None

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)

    def session(self, session_id: str) -> RuntimeSession:
        """Bind a negotiated control session to this runtime."""
        return RuntimeSession(session_id)


class Shared:
    """Lease provider: at most ``width`` rollouts share each task placement.

    Each environment and runtime configuration boots lazily on its first lease
    and lives for the enclosing ``async with`` scope. ``width`` is a substrate's
    capacity (e.g. a vectorized sim's slot count): lease ``width + 1`` waits
    for a slot instead of erroring, so the scheduler needs no pairing —
    ``group`` and ``max_concurrent`` keep their ordinary meanings.

    ``Taskset.run`` scopes a context-manager placement to the call, so
    ``runtime=Shared(DockerRuntime(...), width=8)`` works bare; open the scope
    yourself to keep the substrate warm across several calls::

        async with Shared(DockerRuntime("hud-isaac-env"), width=8) as rt:
            await taskset.run(agent, runtime=rt, group=8)
    """

    def __init__(self, inner: Provider, *, width: int) -> None:
        if width < 1:
            raise ValueError("Shared width must be >= 1")
        self.inner = inner
        self.width = width
        self._boot = asyncio.Lock()
        self._semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._addresses: dict[tuple[str, str], Runtime] = {}
        self._stack: contextlib.AsyncExitStack | None = None
        self._opens = 0

    async def __aenter__(self) -> Self:
        self._opens += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._opens -= 1
        if self._opens == 0 and self._stack is not None:
            stack, self._stack = self._stack, None
            self._addresses.clear()
            self._semaphores.clear()
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
                if key not in self._addresses:
                    # First leaseholder boots. A failed boot fails only its own
                    # rollout (nothing entered the stack); the next lease retries.
                    if self._stack is None:
                        self._stack = contextlib.AsyncExitStack()
                    self._addresses[key] = await self._stack.enter_async_context(self.inner(task))
                addr = self._addresses[key]
            yield addr


def resolve_runtime_config(provider: Provider, task: Task) -> RuntimeConfig | None:
    """Return the runtime configuration a provider will apply to a task."""
    if isinstance(provider, Shared):
        return resolve_runtime_config(provider.inner, task)
    if isinstance(provider, Runtime):
        return provider.config if provider.config is not None else task.runtime_config
    if isinstance(provider, _ConfiguredProvider) and provider.runtime_config is not None:
        base = provider.runtime_config
        config = base.with_overrides(task.runtime_config)
        return config if config.model_dump(exclude_none=True) else None
    return task.runtime_config

"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import re
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Protocol,
    Self,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from .compose import ComposeProject

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
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


_MOUNT_PATH_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
_PREFIX_MAX_LENGTH = 512


class RuntimeDataMode(StrEnum):
    """What happens to writes under a data mount."""

    READONLY = "readonly"
    OVERLAY = "overlay"


def _relative_parts(value: str, field: str) -> list[str]:
    parts = [part for part in value.split("/") if part]
    if any(part in (".", "..") for part in parts):
        raise ValueError(f"{field} must be a relative path without '.' or '..'")
    return parts


class RuntimeDataFile(BaseModel):
    """One file of a mounted folder, shown at ``path`` under the mount."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=_PREFIX_MAX_LENGTH)
    path: str = Field(default="", max_length=255, validate_default=True)

    @field_validator("key")
    @classmethod
    def _relative_key(cls, value: str) -> str:
        parts = _relative_parts(value, "key")
        if not parts:
            raise ValueError("key must name a file")
        return "/".join(parts)

    @model_validator(mode="after")
    def _safe_path(self) -> Self:
        parts = _relative_parts(self.path or self.key, "path")
        if not all(_MOUNT_PATH_SEGMENT.fullmatch(part) for part in parts):
            raise ValueError("path segments may use only letters, digits, '.', '_' and '-'")
        self.path = "/".join(parts)
        return self


class RuntimeDataMount(BaseModel):
    """One folder of the team's shared data, mounted at ``mount_path``.

    With ``files``, only those files of the folder appear under the mount.
    """

    model_config = ConfigDict(extra="forbid")

    prefix: str = Field(default="", max_length=_PREFIX_MAX_LENGTH)
    mount_path: str = Field(max_length=255)
    mode: RuntimeDataMode = RuntimeDataMode.READONLY
    files: list[RuntimeDataFile] | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _separate_files(self) -> Self:
        if self.files is None:
            return self
        if self.mode is not RuntimeDataMode.READONLY:
            raise ValueError("a mount that lists files must be readonly")
        paths = sorted(PurePosixPath(file.path) for file in self.files)
        for outer, inner in itertools.pairwise(paths):
            if inner == outer or outer in inner.parents:
                raise ValueError(f"files {outer} and {inner} overlap")
        return self

    @field_validator("prefix")
    @classmethod
    def _relative_prefix(cls, value: str) -> str:
        prefix = "".join(f"{part}/" for part in _relative_parts(value, "prefix"))
        if len(prefix) > _PREFIX_MAX_LENGTH:
            raise ValueError(f"prefix is longer than {_PREFIX_MAX_LENGTH} characters")
        return prefix

    @field_validator("mount_path")
    @classmethod
    def _absolute_mount_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError("mount_path must be an absolute path other than /")
        if not all(_MOUNT_PATH_SEGMENT.fullmatch(part) for part in path.parts[1:]):
            raise ValueError("mount_path segments may use only letters, digits, '.', '_' and '-'")
        return str(path)


RuntimeDataName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")]


class RuntimeConfig(BaseModel):
    """Typed task-environment launch requirements.

    ``Task.runtime_config`` is the requested config; ``Runtime.config`` is the one applied.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, min_length=1)
    compose: ComposeProject | None = None
    resources: RuntimeResources | None = None
    limits: RuntimeLimits | None = None
    data: dict[RuntimeDataName, RuntimeDataMount] | None = Field(
        default=None,
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def _separate_data_mounts(self) -> Self:
        paths = sorted(PurePosixPath(mount.mount_path) for mount in (self.data or {}).values())
        for outer, inner in itertools.pairwise(paths):
            if inner == outer or outer in inner.parents:
                raise ValueError(f"data mounts {outer} and {inner} overlap")
        return self

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
    runtime_config: RuntimeConfig | None


def validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("runtime session id must be a single path component")


@dataclass(frozen=True)
class Runtime:
    """A provisioned host's control-channel ``url`` plus connection-time ``params``.

    Constructed directly, it is also a provider that yields itself and leaves teardown
    to whoever provisioned the host.
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

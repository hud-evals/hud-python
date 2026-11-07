"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tarfile
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
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from hud.eval.task import Task

# Where an environment keeps each control session's files; the verifier of a
# task reads the actor's session here after the runtime moves it.
SESSION_ROOT = PurePosixPath("/runtime/sessions")


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
    runtime_config: RuntimeConfig | None


def validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("runtime session id must be a single path component")


class RuntimeUnsupportedError(RuntimeError):
    """The runtime cannot perform an operation its contract allows a provider to decline."""

    def __init__(self, runtime: str, operation: str, reason: str) -> None:
        super().__init__(f"{runtime} cannot {operation}: {reason}")
        self.runtime = runtime
        self.operation = operation
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SessionScope:
    """One control session's files under ``SESSION_ROOT``."""

    session_id: str

    def __post_init__(self) -> None:
        validate_session_id(self.session_id)

    @property
    def path(self) -> PurePosixPath:
        return SESSION_ROOT / self.session_id


@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    """One directory of the environment's filesystem."""

    path: PurePosixPath

    def __init__(self, path: str | PurePosixPath) -> None:
        resolved = PurePosixPath(path)
        if not resolved.is_absolute() or resolved == PurePosixPath("/"):
            raise ValueError("workspace scope must be an absolute directory below /")
        object.__setattr__(self, "path", resolved)


@dataclass(frozen=True, slots=True)
class RootScope:
    """The environment's whole filesystem."""


CheckpointScope = SessionScope | WorkspaceScope | RootScope


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A checkpoint the runtime that took it can restore.

    ``ref`` is the provider's handle: for a session or workspace scope a local
    archive of files and directories, for the root scope an image the same
    provider boots from ``runtime_config.image``. It is valid inside the
    ``checkpoint`` scope that yielded it.
    """

    scope: CheckpointScope
    ref: str


@dataclass(frozen=True, slots=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


def validate_archive(archive: Path) -> None:
    """Reject a session or workspace archive holding anything but files and directories."""
    with tarfile.open(archive, "r:gz") as tar:
        if any(not (member.isfile() or member.isdir()) for member in tar.getmembers()):
            raise ValueError("runtime archive contains an unsupported entry")


def extract_archive(archive: Path, destination: Path) -> None:
    validate_archive(archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def check_restorable(checkpoint: Checkpoint, scope: CheckpointScope) -> None:
    """Reject a restore the checkpoint contract rules out."""
    if isinstance(scope, RootScope) or isinstance(checkpoint.scope, RootScope):
        raise ValueError(
            "a root checkpoint is restored by starting a new box from "
            f"runtime_config.image={checkpoint.ref!r}"
        )
    if type(checkpoint.scope) is not type(scope):
        raise ValueError("a checkpoint restores into a scope of its own kind")


# Runs ``exec``'s command inside the box with its pid recorded in the file named by
# ``$0``, so that a call ending early can have the box kill it with ``KILL_TREE``.
# Ending the provider's client, or the provider's own deadline, stops only the
# command's first process; its descendants would run on.
EXEC_LAUNCHER = '"$@" & echo $! > "$0"; wait $!; status=$?; rm -f "$0"; exit $status'
# Kills the process whose pid the file named by ``$0`` holds, descendants first.
KILL_TREE = """
kill_tree() {
    for p in /proc/[0-9]*; do
        if [ "$(sed -n 's/^PPid:[[:space:]]*//p' "$p/status" 2>/dev/null)" = "$1" ]; then
            kill_tree "${p#/proc/}"
        fi
    done
    kill -9 "$1" 2>/dev/null
}
kill_tree "$(cat "$0")"
rm -f "$0"
"""


@dataclass(frozen=True)
class Runtime:
    """A provisioned host's control-channel ``url`` plus connection-time ``params``.

    Constructed directly, it is also a provider that yields itself and leaves teardown
    to whoever provisioned the host.

    A provisioned runtime also owns the box's filesystem operations: ``exec``
    and ``collect`` reach the environment and, for a Compose project, its
    sibling services; ``checkpoint`` and ``restore`` move a session, a
    workspace directory, or the root filesystem. A borrowed runtime declines
    them all with ``RuntimeUnsupportedError``, as does any provider for
    a scope its host cannot capture faithfully.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    config: RuntimeConfig | None = None

    # Whether ``checkpoint`` and ``restore`` move a control session
    # between boxes. Without it, a verifier must run beside the actor.
    transports_sessions: ClassVar[bool] = False

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)

    def _unsupported(self, operation: str, reason: str) -> RuntimeUnsupportedError:
        return RuntimeUnsupportedError(type(self).__name__, operation, reason)

    async def terminated(self) -> str | None:
        """Why the box stopped, or ``None`` while it runs."""
        raise self._unsupported("report termination", "it did not provision the host")

    async def exec(
        self,
        argv: Sequence[str],
        *,
        service: str = "main",
        max_wait: float | None = None,
    ) -> ExecResult:
        """Run ``argv`` inside ``service`` (``main`` is the environment) and return its outcome.

        A command still running after ``max_wait`` seconds is stopped and the
        call raises ``TimeoutError``.
        """
        raise self._unsupported("exec", "it did not provision the host")

    async def collect(
        self,
        paths: Sequence[str | PurePosixPath],
        destination: Path,
        *,
        service: str = "main",
    ) -> list[Path]:
        """Copy each of ``paths`` out of ``service`` into ``destination``; returns the copies.

        Each path lands at ``destination / path.name``. A link to an absolute
        path or outside ``destination`` is rejected (``tarfile.FilterError``).
        """
        raise self._unsupported("collect", "it did not provision the host")

    def checkpoint(self, scope: CheckpointScope) -> AbstractAsyncContextManager[Checkpoint]:
        """Capture ``scope`` for the duration of the returned context.

        A session or workspace checkpoint is an archive of regular files and
        directories (a session that was never written is an empty archive).
        A root checkpoint is an image this provider starts a new box from.
        """
        raise self._unsupported("checkpoint", "it did not provision the host")

    async def restore(self, checkpoint: Checkpoint, scope: CheckpointScope) -> None:
        """Replace ``scope`` in this box with ``checkpoint``; the scopes must be of one kind."""
        raise self._unsupported("restore", "it did not provision the host")


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

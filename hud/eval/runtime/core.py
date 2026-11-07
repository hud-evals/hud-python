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
from websockets.exceptions import ConnectionClosedOK

from .compose import ComposeProject

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from hud.eval.task import Task

#: Where an environment keeps each control session's files; the verifier of a
#: task reads the actor's session here after the runtime moves it.
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
    """One control session's files under :data:`SESSION_ROOT`."""

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


async def splice_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    websocket: Any,
) -> None:
    """Carry bytes both ways between a local TCP connection and a WebSocket until one
    side closes; a transport that tunnels the control channel over WebSockets ends here."""

    async def tcp_to_ws() -> None:
        while data := await reader.read(65536):
            await websocket.send(data)

    async def ws_to_tcp() -> None:
        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            writer.write(data)
            await writer.drain()

    tasks = [
        asyncio.create_task(tcp_to_ws()),
        asyncio.create_task(ws_to_tcp()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        done_results = await asyncio.gather(*done, return_exceptions=True)
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for result in done_results:
        if isinstance(result, BaseException) and not isinstance(result, ConnectionClosedOK):
            raise result


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

    A provisioned runtime also owns the box's filesystem operations: ``exec``,
    ``collect`` and ``put`` reach the environment and, for a Compose project,
    its sibling services; ``checkpoint`` and ``restore`` move a session, a
    workspace directory, or the root filesystem. A borrowed runtime declines
    them all with :class:`RuntimeUnsupportedError`, as does any provider for
    a scope its substrate cannot capture faithfully.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    config: RuntimeConfig | None = None

    #: Whether :meth:`checkpoint` and :meth:`restore` move a control session
    #: between boxes. Without it, a verifier must run beside the actor.
    transports_sessions: ClassVar[bool] = False

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)

    def _unsupported(self, operation: str, reason: str) -> RuntimeUnsupportedError:
        return RuntimeUnsupportedError(type(self).__name__, operation, reason)

    async def terminated(self) -> str | None:
        """Why the box stopped, or ``None`` while it runs or where the runtime cannot see it."""
        return None

    async def exec(
        self,
        argv: Sequence[str],
        *,
        service: str = "main",
    ) -> ExecResult:
        """Run ``argv`` inside ``service`` (``main`` is the environment) and return its outcome.

        Cancelling the call stops the command.
        """
        raise self._unsupported("exec", "it did not provision the substrate")

    async def collect(
        self,
        paths: Sequence[str | PurePosixPath],
        destination: Path,
        *,
        service: str = "main",
    ) -> list[Path]:
        """Copy each of ``paths`` out of ``service`` into ``destination``; returns the copies.

        Each path lands at ``destination / path.name``. A link to an absolute
        path or outside ``destination`` is rejected (:class:`tarfile.FilterError`).
        """
        raise self._unsupported("collect", "it did not provision the substrate")

    async def put(
        self,
        source: Path,
        path: str | PurePosixPath,
        *,
        service: str = "main",
    ) -> None:
        """Copy the local file or directory ``source`` into ``service`` at ``path``,
        replacing whatever is there."""
        raise self._unsupported("put", "it did not provision the substrate")

    def checkpoint(self, scope: CheckpointScope) -> AbstractAsyncContextManager[Checkpoint]:
        """Capture ``scope`` for the duration of the returned context.

        A session or workspace checkpoint is an archive of regular files and
        directories (a session that was never written is an empty archive).
        A root checkpoint is an image this provider starts a new box from.
        """
        raise self._unsupported("checkpoint", "it did not provision the substrate")

    async def restore(self, checkpoint: Checkpoint, scope: CheckpointScope) -> None:
        """Replace ``scope`` in this box with ``checkpoint``; the scopes must be of one kind."""
        raise self._unsupported("restore", "it did not provision the substrate")


class Shared:
    """Lease provider: at most ``width`` rollouts share each task placement.

    Each environment and runtime configuration boots lazily on its first lease.
    It lives for the enclosing ``async with`` scope unless its run timeout
    expires, in which case the next lease boots a fresh substrate. ``width`` is
    a substrate's capacity (e.g. a vectorized sim's slot count): an excess lease
    waits for a slot instead of erroring, so the scheduler needs no pairing —
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
                    # First leaseholder boots. A failed boot fails only its own
                    # rollout (nothing is retained); the next lease retries.
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

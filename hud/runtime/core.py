"""Shared runtime configuration and placement contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)

from hud.clients.client import HudClient
from hud.clients.dial import dialer
from hud.environment.session import validate_session_id
from hud.environment.utils import HudProtocolError
from hud.utils.process import create_process_group_exec

from .compose import ComposeProject

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from hud.environment.egress import WorkspaceRoute
    from hud.eval.run import EnvironmentSession
    from hud.eval.task import Task
    from hud.utils.process import ProcessResult

    from .hosted import HostedRuntime

LOGGER = logging.getLogger("hud.runtime")
# How long a connect waits for a freshly provisioned env to serve when nothing sets a limit.
READY_TIMEOUT_SECONDS = 240.0
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


class RuntimeBuild(BaseModel):
    """An image built from ``context``, with ``dockerfile`` relative to it."""

    model_config = ConfigDict(extra="forbid")

    context: Path
    dockerfile: str | None = Field(default=None, min_length=1)


class EnvRef(BaseModel):
    """The value of ``name`` in this process's environment at placement.

    ``default`` stands in when ``name`` is unset or empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    default: str | None = None


class RuntimeHealthcheck(BaseModel):
    """A shell ``command`` whose success reports the environment ready, and how to poll it."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    interval_s: float = Field(default=30.0, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    start_period_s: float = Field(default=0.0, ge=0)
    start_interval_s: float = Field(default=5.0, gt=0)
    retries: int = Field(default=3, ge=1)


class RuntimeProcess(BaseModel):
    """How the environment's main process runs: the image's own, or a Compose ``main``'s.

    ``command`` replaces the image's command and keeps its entrypoint, and with it the HUD
    control channel an environment image serves: a placement with a ``command`` is a box
    the controller drives through its runtime. ``env`` is set over the image's own. The
    environment is ready once ``healthcheck`` passes.
    """

    model_config = ConfigDict(extra="forbid")

    command: list[str] | None = Field(default=None, min_length=1)
    env: dict[str, str | EnvRef] = Field(default_factory=dict)
    healthcheck: RuntimeHealthcheck | None = None


class RuntimePort(BaseModel):
    """A port a service of the environment serves, published for the controller to reach."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int = Field(ge=1, le=65535)
    service: str = Field(default="main", min_length=1)


def resolve_env(env: Mapping[str, str | EnvRef], available: Mapping[str, str]) -> dict[str, str]:
    """``env`` with each reference read from ``available``."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        if isinstance(value, str):
            resolved[key] = value
            continue
        found = available.get(value.name)
        if found is not None and (found or value.default is None):
            resolved[key] = found
        elif value.default is not None:
            resolved[key] = value.default
        else:
            raise ValueError(f"env {key!r} reads {value.name!r}, which is not set here")
    return resolved


class RuntimeConfig(BaseModel):
    """Typed task-environment launch requirements.

    The environment comes from an ``image``, pulled by reference or built, or from a
    ``compose`` project; ``process`` says how its main process runs, and ``ports`` which
    ports the controller reaches.
    ``Task.runtime_config`` is the requested config; ``Runtime.config`` is the one applied.
    """

    model_config = ConfigDict(extra="forbid")

    image: Annotated[str, Field(min_length=1)] | RuntimeBuild | None = None
    compose: ComposeProject | None = None
    process: RuntimeProcess | None = None
    ports: list[RuntimePort] | None = None
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

    def pulled_image(self, provider: str) -> str | None:
        """The image reference, for a provider that runs only pulled images as they are."""
        if isinstance(self.image, RuntimeBuild):
            raise ValueError(f"{provider} does not support building runtime_config.image")
        if self.process is not None:
            raise ValueError(f"{provider} does not support runtime_config.process")
        if self.ports:
            raise ValueError(f"{provider} does not support runtime_config.ports")
        return self.image


class Provider(Protocol):
    """Places a task: an async context manager yielding a connectable ``Runtime`` for its scope."""

    def __call__(self, task: Task, /) -> AbstractAsyncContextManager[Runtime]: ...


@runtime_checkable
class Box(Protocol):
    """A placed runtime the controller runs commands in and copies files in and out of."""

    def exec_argv(
        self,
        argv: Sequence[str],
        *,
        user: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        tty: bool = False,
    ) -> list[str]:
        """The local command line that runs ``argv`` in the box, with its stdio attached.

        ``env`` is set over the box's own environment. With ``tty`` the command gets a
        terminal, so the command line's own stdio must be one.
        """
        ...

    async def put(self, source: Path, target: str) -> None:
        """Copy ``source``, a file or the contents of a directory, to ``target`` in the box."""
        ...

    async def get(self, source: str, target: Path) -> None:
        """Copy ``source`` from the box to ``target``."""
        ...

    def service(self, name: str) -> Box:
        """The box of ``name``, another service placed with this one."""
        ...

    def address(self, port: int) -> str:
        """``host:port`` where this process reaches ``port``, one ``RuntimeConfig.ports`` named."""
        ...


async def exec_in(
    box: Box,
    argv: Sequence[str],
    *,
    user: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    input: bytes = b"",
    max_wait: float | None = None,
) -> ProcessResult:
    """Run ``argv`` in ``box`` to completion, or for at most ``max_wait`` seconds."""
    process = await create_process_group_exec(
        *box.exec_argv(argv, user=user, cwd=cwd, env=env),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin = process.process.stdin
    assert stdin is not None
    stdin.write(input)
    stdin.close()
    return await process.complete(max_wait=max_wait)


@runtime_checkable
class _ConfiguredProvider(Protocol):
    def config_for(self, task: Task, /) -> RuntimeConfig | None: ...


@dataclass(frozen=True)
class Runtime:
    """An acquired host: where its environment is, how to open sessions on it, and
    how to move a session's files between hosts.

    ``url`` is the environment's control channel, and ``connect`` waits up to
    ``ready_timeout`` seconds for the environment behind it to answer; a WebSocket ``url``
    sends ``headers`` with its handshake. ``config`` is the runtime configuration its
    provider applied. Constructed directly, a runtime is also a provider that yields itself
    and leaves teardown to whoever provisioned the host.
    """

    url: str
    ready_timeout: float = READY_TIMEOUT_SECONDS
    config: RuntimeConfig | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

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
    async def connect(
        self,
        *,
        session_id: str | None = None,
        lease: float | None = None,
        workspace_routes: Sequence[WorkspaceRoute] = (),
    ) -> AsyncGenerator[EnvironmentSession]:
        """Open a session on the environment over its control channel.

        ``session_id`` attaches to that open session instead; ``lease`` keeps the session
        open that many seconds after this connection ends, for a later one to attach.

        Connecting retries until the environment answers ``hello``, for ``ready_timeout``
        seconds: a proxied port (``docker -p``, a port-forward) can accept before the
        environment behind it serves. While the session is open, a heartbeat probes the
        connection; a failure cancels the body and raises that error.
        """
        dial = dialer(self.url, self.headers)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.ready_timeout
        while True:
            try:
                reader, writer = await dial()
            except OSError:
                if loop.time() >= deadline:
                    raise
                await asyncio.sleep(_CONNECT_RETRY_SECONDS)
                continue
            client = HudClient(reader, writer, dial=dial)
            try:
                await client.hello(
                    session_id,
                    lease=lease,
                    workspace_routes=workspace_routes,
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
                # A real failure still closes the socket, or the env's handler, and the
                # session this hello opened, stay open waiting on it.
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

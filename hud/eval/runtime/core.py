"""Runtime configuration, addresses, sharing, and local placement."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections import deque
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hud.utils.process import ProcessGroup, create_process_group_exec

from .compose import ComposeConfig, ComposeProjectRef, ComposeSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from hud.environment.env import Environment
    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")


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

    ``compose`` and ``compose_project`` are authored as local paths; platform
    task records carry them as the serialized compose document and a
    :class:`ComposeProjectRef`. Both forms validate; only the path form is
    runnable by local providers.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, min_length=1)
    compose: Path | ComposeConfig | None = None
    compose_project: Path | ComposeProjectRef | None = None
    compose_service_access: bool | None = None
    resources: RuntimeResources | None = None
    limits: RuntimeLimits | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.image is not None and self.compose is not None:
            raise ValueError("runtime_config accepts either image or compose, not both")
        if self.compose_project is not None and self.compose is None:
            raise ValueError("compose_project requires runtime_config.compose")
        if self.compose_service_access and self.compose is None:
            raise ValueError("compose_service_access requires runtime_config.compose")
        return self

    def with_overrides(self, override: RuntimeConfig | None) -> RuntimeConfig:
        if override is None:
            return self
        config = self.model_dump()
        changes = override.model_dump(exclude_unset=True)
        if override.image is not None:
            config["compose"] = None
            config["compose_project"] = None
            config["compose_service_access"] = None
        elif override.compose is not None:
            config["image"] = None
            config["compose_project"] = None
        return RuntimeConfig.model_validate(config | changes)

    def request_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_unset=True)
        source = self.compose_source()
        if source is not None:
            payload.update(source.request_payload())
        return payload

    def compose_source(self) -> ComposeSource | None:
        """The authored or wire-form Compose source, when configured."""
        if self.compose is None:
            return None
        return ComposeSource(self.compose, self.compose_project)


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


class HandoffEndpoint(Protocol):
    """Provider-owned transfer of the runtime handoff namespace."""

    async def export_to(self, destination: Path) -> None: ...

    async def import_from(self, source: Path) -> None: ...


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
    handoff: HandoffEndpoint | None = field(default=None, repr=False, compare=False)

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)


class Shared:
    """Lease provider: at most ``width`` concurrent rollouts share one substrate.

    The substrate boots lazily on the first lease and lives for the enclosing
    ``async with`` scope — one boot however many rollouts flow through, torn
    down deterministically at scope exit. ``width`` is the substrate's real
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
        self._sem = asyncio.Semaphore(width)
        self._boot = asyncio.Lock()
        self._addr: Runtime | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._opens = 0

    async def __aenter__(self) -> Self:
        self._opens += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._opens -= 1
        if self._opens == 0 and self._stack is not None:
            stack, self._stack, self._addr = self._stack, None, None
            await stack.aclose()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if self._opens == 0:
            raise RuntimeError(
                "Shared substrates outlive single rollouts; lease inside the scope "
                "(Taskset.run opens it for you, or wrap calls in `async with Shared(...)`)"
            )
        async with self._sem:
            async with self._boot:
                if self._addr is None:
                    # First leaseholder boots. A failed boot fails only its own
                    # rollout (nothing entered the stack); the next lease retries.
                    stack = contextlib.AsyncExitStack()
                    self._addr = await stack.enter_async_context(self.inner(task))
                    self._stack = stack
                addr = self._addr
            yield addr


class LocalRuntime:
    """The local provider: serve a fresh env per rollout, in this process.

    *source* points at the env in whatever form you have:

    - a ``.py`` file or directory — imported fresh per acquisition (sibling
      imports resolve); *env* pins one name when several are declared,
      defaulting to the placed task's env
    - a live :class:`~hud.environment.Environment` — shorthand for its
      declaring file; the instance itself is never served
    - a ``(task) -> Environment`` callable — called per acquisition with the
      placed row

    ::

        runtime = LocalRuntime("env.py")
        runtime = LocalRuntime(env)
        runtime = LocalRuntime(lambda task: build_env(task.env))

    ``ready_timeout`` bounds ``@env.initialize`` startup. Freshness covers
    the env's own source; modules it imports are cached as usual and shared
    across rollouts. Hooks share this process's event loop, so blocking env
    code stalls concurrent rollouts — use :class:`SubprocessRuntime` or
    :class:`DockerRuntime` for process isolation, and ``Runtime(url)`` to
    attach to a substrate served elsewhere.
    """

    def __init__(
        self,
        source: str | Path | Environment | Callable[[Task], Environment],
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        from hud.environment.env import Environment as _Environment

        self.ready_timeout = ready_timeout
        # A live instance may have been mutated since its module was imported;
        # verify the fresh copy still declares its templates, so drift fails
        # at acquisition with the cause named instead of "unknown task" later.
        expected_templates: frozenset[str] = frozenset()
        if isinstance(source, _Environment):
            file = _declaring_file(source, env or source.name)
            if file is None:
                raise TypeError(
                    f"LocalRuntime: env {source.name!r} is not rebuilt by importing "
                    "any file this process has loaded (constructed in a function or "
                    "notebook cell, or declared inside a package using relative "
                    "imports); pass its constructor instead: "
                    "LocalRuntime(lambda task: <build the env>)"
                )
            expected_templates = frozenset(source.tasks)
            source, env = file, env or source.name
        self._source_dir: Path | None = None
        if isinstance(source, (str, Path)):
            path, pinned = Path(source).resolve(), env
            self._source_dir = path if path.is_dir() else path.parent
            from hud.environment import load_environment

            def _load(task: Task) -> _Environment:
                loaded = load_environment(path, name=pinned or task.env)
                missing = expected_templates - loaded.tasks.keys()
                if missing:
                    raise ValueError(
                        f"env {loaded.name!r} loaded from {path} lacks template(s) "
                        f"{sorted(missing)} present on the live instance — it was "
                        "modified after import; pass a constructor instead: "
                        "LocalRuntime(lambda task: <build the env>)"
                    )
                return loaded

            self._build: Callable[[Task], _Environment] = _load
        elif callable(source):
            if env is not None:
                raise TypeError("LocalRuntime: env= applies only to source paths")
            self._build = source
        else:
            raise TypeError(
                f"LocalRuntime: expected a source path, a live Environment, or a "
                f"(task) -> Environment constructor; got {source!r}"
            )

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.environment.env import Environment as _Environment

        if task.runtime_config is not None:
            raise ValueError("LocalRuntime does not support task runtime_config")
        # The source dir stays importable for the whole acquisition, not just
        # the initial import, so a template can lazily import a sibling
        # module at run time (as it could under the child-process runtime).
        # Always insert-and-remove one entry: balanced under concurrency.
        if self._source_dir is not None:
            sys.path.insert(0, str(self._source_dir))
        try:
            try:
                env = self._build(task)
            except RuntimeError as e:
                # The source ran an event loop at import — usually an unguarded
                # top-level run call; name the actual mistake.
                if "running event loop" not in str(e):
                    raise
                raise RuntimeError(
                    "the env source ran async code while being imported to place a "
                    'rollout — guard top-level run calls with `if __name__ == "__main__":`'
                ) from e
            if not isinstance(env, _Environment):
                raise TypeError(f"LocalRuntime: constructor returned {env!r}, not an Environment")
            async with _local(env, ready_timeout=self.ready_timeout) as runtime:
                yield runtime
        finally:
            if self._source_dir is not None:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(self._source_dir))


def _live_envs() -> Iterator[tuple[Environment, str]]:
    """Envs declared in loaded, file-backed modules' globals, with their files.

    The in-memory counterpart of scanning ``.py`` sources on disk
    (:func:`~hud.environment.load_environment`): an env found here can be
    served fresh by re-importing its file. Envs in modules without a file
    (a notebook ``__main__``) are not yielded — re-import could not
    reconstruct them.
    """
    from hud.environment.env import Environment as _Environment

    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        module_vars = getattr(module, "__dict__", None)
        if not module_file or not isinstance(module_vars, dict):
            continue
        for value in list(module_vars.values()):
            if isinstance(value, _Environment):
                yield value, module_file


def _declaring_file(env: Environment, name: str) -> Path | None:
    """A file whose fresh import re-declares *env*, else None.

    Candidate files hold the instance in their module globals, but a holder
    may be a re-exporter (``from .env import env`` in a package
    ``__init__``, a tasks file re-exporting its env): validate each by
    loading it fresh — a declarer yields a *new* instance under *name*, a
    re-exporter yields the same live one (or fails to import standalone).
    ``__init__.py`` holders are tried last.
    """
    from hud.environment import load_environment

    candidates = dict.fromkeys(Path(file) for live, file in _live_envs() if live is env)
    for file in sorted(candidates, key=lambda f: f.name == "__init__.py"):
        try:
            probe = load_environment(file, name=name)
        except Exception as e:
            logger.debug("candidate %s does not rebuild env %r: %s", file, name, e)
            continue
        if probe is not env:
            return file
    return None


def _declared_env(name: str) -> Environment | None:
    """The one live env named *name*, else None; two distinct ones raise.

    The same instance re-exported across modules is one match; distinct envs
    claiming one name are ambiguous.
    """
    matches = {id(env): env for env, _ in _live_envs() if env.name == name}
    if len(matches) > 1:
        files = sorted({file for env, file in _live_envs() if env.name == name})
        raise ValueError(
            f"env name {name!r} is declared by multiple live environments "
            f"({', '.join(files)}); pass runtime= explicitly — the exact "
            "instance disambiguates: runtime=LocalRuntime(env)"
        )
    return next(iter(matches.values()), None)


def _declared_names(source: Path) -> set[str]:
    """Env names a ``.py`` source (file or directory) itself declares.

    A fresh execution of the source yields *new* instances for envs it
    declares; an env it merely imports is the already-live one and does not
    count — importing the source again could not rebuild it.
    """
    from hud.environment.env import Environment as _Environment
    from hud.utils.modules import iter_modules

    live = {id(env) for env, _ in _live_envs()}
    return {
        value.name
        for module in iter_modules(source)
        for value in vars(module).values()
        if isinstance(value, _Environment) and id(value) not in live
    }


class SubprocessRuntime:
    """The child-process provider: serve the placed row's env from *path*.

    Each acquisition runs ``python -m hud.environment.server <path> --env
    name`` — the same serving entry point a container CMD runs — on an
    ephemeral loopback port, yields its :class:`Runtime`, and terminates the
    child on exit. *path* is a ``.py`` file or a directory of them. The served
    env is the placed task's ``env`` name (so a mixed-env taskset works
    against one source), unless *env* pins one explicitly; placing a row whose
    env the source does not define fails loudly in the child.

    The child's working directory is the source's directory, so sibling
    imports and relative data paths resolve; ``@env.initialize`` daemons start
    in the child and die with it. Because the source is re-imported in the
    child, a script spawning itself (``SubprocessRuntime(__file__)``) must keep
    top-level run calls under ``if __name__ == "__main__":``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        self.source = Path(path).resolve()
        self.env = env
        self.ready_timeout = ready_timeout

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if task.runtime_config is not None:
            raise ValueError("SubprocessRuntime does not support task runtime_config")
        if not self.source.exists():
            raise FileNotFoundError(f"SubprocessRuntime: source not found: {self.source}")
        cmd = [sys.executable, "-m", "hud.environment.server", str(self.source)]
        cmd += ["--env", self.env or task.env]
        proc = await create_process_group_exec(
            *cmd,
            term_timeout=10.0,
            stdout=asyncio.subprocess.PIPE,
            # Capture stderr (don't inherit it): under concurrent rollouts an
            # inherited fd interleaves every child's output unattributably, so a
            # crash-before-serving leaves no traceable diagnostic. We keep a
            # bounded tail and attach it to the failure below.
            stderr=asyncio.subprocess.PIPE,
            cwd=self.source if self.source.is_dir() else self.source.parent,
        )
        assert proc.stderr is not None
        # Drain stderr into a bounded tail from the start: it never blocks on a
        # full pipe, and the last lines survive if the child dies early.
        stderr_tail: deque[str] = deque(maxlen=50)
        capture = asyncio.create_task(_capture(proc.stderr, stderr_tail))
        try:
            assert proc.stdout is not None
            port = await asyncio.wait_for(_read_port(proc.stdout), self.ready_timeout)
            if port is None:
                raise RuntimeError(await _exit_detail(proc, self.source, capture, stderr_tail))
            drain = asyncio.create_task(_drain(proc.stdout))
            try:
                yield Runtime(f"tcp://127.0.0.1:{port}")
            finally:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
        finally:
            capture.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await capture
            await proc.terminate()


@asynccontextmanager
async def _local(env: Environment, *, ready_timeout: float | None = None) -> AsyncIterator[Runtime]:
    """Substrate-side serving: a live env owned by *this* process, as a runtime.

    One env lifecycle (start → serve → stop) around one bound control
    channel; ``ready_timeout`` bounds ``env.start()`` (initialize
    hooks/daemons). ``LocalRuntime`` enters this per acquisition with the
    fresh env it built; test harnesses enter it directly with a live one.
    """
    from hud.environment.server import _shutdown, bind

    # start() inside the try: a failed or timed-out initialize hook still gets
    # its already-started daemons torn down by stop() (best-effort per hook).
    try:
        started = env.start()
        await (asyncio.wait_for(started, ready_timeout) if ready_timeout is not None else started)
        server = await bind(env, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            yield Runtime(f"tcp://{host}:{port}")
        finally:
            serve_task.cancel()
            await _shutdown(server)
            with contextlib.suppress(asyncio.CancelledError):
                await serve_task
    finally:
        await env.stop()


async def _read_port(stdout: asyncio.StreamReader) -> int | None:
    """Read the child's stdout until it announces its port; ``None`` if stdout
    hits EOF first (the child exited before serving — caller builds the error)."""
    # Imported lazily: a module-level import would pre-load hud.environment.server
    # in every `python -m hud.environment.server` child, tripping runpy's
    # found-in-sys.modules RuntimeWarning on each spawned rollout.
    from hud.environment.server import PORT_ANNOUNCEMENT

    while True:
        line = await stdout.readline()
        if not line:
            return None
        text = line.decode("utf-8", "replace").strip()
        if text.startswith(PORT_ANNOUNCEMENT):
            return int(text.removeprefix(PORT_ANNOUNCEMENT))


async def _exit_detail(
    proc: ProcessGroup,
    source: Path,
    capture: asyncio.Task[None],
    stderr_tail: deque[str],
) -> str:
    """Message for a child that exited before serving, with its captured stderr
    tail. The child is gone, so its stderr is at EOF — let the capture finish so
    the traceback it wrote on the way out is included, not raced past."""
    code = await proc.wait()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.shield(capture), 2.0)
    tail = "\n".join(stderr_tail).strip()
    detail = f":\n{tail}" if tail else " (no stderr captured)"
    return f"spawned env exited with code {code} before serving (source: {source}){detail}"


async def _capture(stream: asyncio.StreamReader, sink: deque[str]) -> None:
    """Drain a child stream into a bounded tail so it never blocks on a full pipe
    and its last lines survive for diagnostics."""
    while line := await stream.readline():
        sink.append(line.decode("utf-8", "replace").rstrip())


async def _drain(stream: asyncio.StreamReader) -> None:
    """Keep consuming the child's stdout so it never blocks on a full pipe."""
    while await stream.read(65536):
        pass

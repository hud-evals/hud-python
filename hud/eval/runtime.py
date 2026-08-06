"""Provider: server placement — where the env's control channel comes from.

A :class:`Provider` brings up the *server* (the env's control channel) for one
rollout and yields its connectable :class:`Runtime`; the agent loop drives it
from this process (:func:`hud.eval.run.rollout`). The channel is location
transparent, so "co-located" (loopback) and "split" (agent here, env
elsewhere) are the same code, differing only in the url.

- :class:`LocalRuntime` — serve a fresh env per rollout, in this process,
  from any pointer to it: a ``.py`` source path, a live module-level
  :class:`Environment` (its declaring file is the recipe), or a
  ``(task) -> Environment`` constructor.
- :class:`SubprocessRuntime` — serve the row's env from a ``.py`` source in a
  child process, when the env should not share the orchestrator's fate.
- :class:`DockerRuntime` — starts an image or Compose environment whose primary
  service serves the channel.
- ``Runtime(url)`` — the ``nullcontext`` of providers: yields itself, a
  *borrowed, shared* substrate provisioned elsewhere (env served anywhere —
  a cloud sandbox, another host — that this process connects to).

The provider contract is structural (anything callable as ``(task) -> async
context manager of Runtime``), so per-task heterogeneity (this row on 1 GPU,
that one on 4, different images) is just a provider that reads the row.

The delegated placement — :class:`HostedRuntime`, running the whole rollout
off-box on a HUD sandbox — also lives here; the scheduler (:meth:`Taskset.run`)
chooses between it and providers. A hosted box's own driver is itself a
``Provider`` (its ``DockerRuntime``) driven by the same ``rollout`` atom —
co-location all the way down.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import os
import shlex
import sys
import tarfile
import tempfile
import uuid
from collections import deque
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hud.telemetry.context import get_current_trace_id
from hud.types import Step
from hud.utils.docker import docker as _docker
from hud.utils.platform import PlatformClient
from hud.utils.process import ProcessGroup, create_process_group_exec

from .run import Grade, Run, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

    from hud.agents.base import Agent
    from hud.environment.env import Environment

    from .task import Task

logger = logging.getLogger("hud.eval.runtime")

_COMPOSE_SERVICE_SOCKET = "/media/hud/docker.sock"


class RuntimeGPU(BaseModel):
    """Requested GPU resources, provider-neutral where possible."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = Field(default=None, min_length=1)
    count: int = Field(default=1, ge=1)


class RuntimeResources(BaseModel):
    """Requested compute resources for a runtime."""

    model_config = ConfigDict(extra="forbid")

    cpu: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    gpu: RuntimeGPU | None = None


class RuntimeLimits(BaseModel):
    """Runtime lifecycle limits in seconds."""

    model_config = ConfigDict(extra="forbid")

    startup_timeout_s: int | None = Field(default=None, gt=0)
    run_timeout_s: int | None = Field(default=None, gt=0)


class RuntimeConfig(BaseModel):
    """Typed task-environment launch requirements.

    ``Task.runtime_config`` is requested construction input. ``Runtime.config``
    is the effective config used to construct a runtime.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(default=None, min_length=1)
    compose: Path | None = None
    compose_service_access: bool | None = None
    resources: RuntimeResources | None = None
    limits: RuntimeLimits | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.image is not None and self.compose is not None:
            raise ValueError("runtime_config accepts either image or compose, not both")
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
            config["compose_service_access"] = None
        elif override.compose is not None:
            config["image"] = None
        return RuntimeConfig.model_validate(config | changes)

    def request_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)


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


def _modal_image_from_uri(modal: Any, image_uri: str) -> Any:
    modal_uri_prefix = "modal://"
    if image_uri.startswith(modal_uri_prefix):
        return modal.Image.from_id(image_uri.removeprefix(modal_uri_prefix))
    return modal.Image.from_registry(image_uri)


#: DockerRuntime always serves HUD environments, so this is part of the
#: provider contract rather than a per-image option. This is intentionally a
#: default-allow compatibility profile: Workspace's bwrap sessions need the
#: namespace and mount syscalls, while unrelated kernel interfaces stay denied.
_DOCKER_SECCOMP_PROFILE = Path(__file__).with_name("docker-seccomp.json")
_DOCKER_SECURITY_ARGS = (
    "--security-opt",
    f"seccomp={_DOCKER_SECCOMP_PROFILE}",
    # Docker exposes system-path masking only as an all-or-nothing option;
    # bwrap replaces the container's proc and dev mounts while building a wall.
    "--security-opt",
    "systempaths=unconfined",
)


def _docker_compose_main(
    published_port: str,
    resources: RuntimeResources | None,
    *,
    seccomp: str | Path = _DOCKER_SECCOMP_PROFILE,
    service_socket: str | None = None,
) -> dict[str, Any]:
    main: dict[str, Any] = {
        "ports": [published_port],
        "security_opt": [f"seccomp={seccomp}", "systempaths=unconfined"],
    }
    if service_socket is not None:
        main["volumes"] = [
            {
                "type": "bind",
                "source": service_socket,
                "target": _COMPOSE_SERVICE_SOCKET,
            }
        ]
    if resources is None:
        return main
    if resources.cpu is not None:
        main["cpus"] = resources.cpu
    if resources.memory_mb is not None:
        main["mem_limit"] = f"{resources.memory_mb}m"
    if resources.gpu is not None:
        main["gpus"] = resources.gpu.count
    return main


def _compose_overrides(directory: Path, main: dict[str, Any]) -> tuple[Path, Path]:
    main = dict(main)
    (published_port,) = main.pop("ports")
    override = directory / "override.json"
    override.write_text(json.dumps({"services": {"main": main}}), encoding="utf-8")
    ports = directory / "ports.yaml"
    ports.write_text(
        f'services:\n  main:\n    ports: !override ["{published_port}"]\n',
        encoding="utf-8",
    )
    return override, ports


@contextlib.contextmanager
def _compose_payload(
    compose: Path,
    main: dict[str, Any],
) -> Iterator[tuple[Path, Path, Path]]:
    with tempfile.TemporaryDirectory(prefix="hud-compose-") as directory:
        root = Path(directory)
        archive = root / "project.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for entry in compose.parent.iterdir():
                tar.add(entry, arcname=entry.name)
        override, ports = _compose_overrides(root, main)
        yield archive, override, ports


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


class DockerRuntime:
    """Start a HUD environment from an image or a Docker Compose file.

    An image is started with ``docker run``. A Compose file is started unchanged
    except for a small provider override that publishes the ``main`` service's
    control-channel port and applies HUD's nested-workspace security profile.
    """

    def __init__(
        self,
        image: str | None = None,
        *,
        port: int = 8765,
        run_args: Sequence[str] = (),
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
    ) -> None:
        self.port = port
        self.run_args = tuple(run_args)
        config = RuntimeConfig(image=image) if image is not None else RuntimeConfig()
        if runtime_config is not None:
            config = config.with_overrides(RuntimeConfig.model_validate(runtime_config))
        self.runtime_config = config if config.model_dump(exclude_none=True) else None

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        if config.limits is not None and config.limits.model_dump(exclude_none=True):
            raise ValueError("DockerRuntime does not support runtime_config limits")
        if config.compose is not None:
            if self.run_args:
                raise ValueError("DockerRuntime run_args apply only to image environments")
            compose = config.compose.resolve()
            resources = config.resources
            if (
                resources is not None
                and resources.gpu is not None
                and resources.gpu.type is not None
            ):
                raise ValueError("DockerRuntime cannot select Compose GPUs by type")
            service_socket = None
            if config.compose_service_access:
                endpoint = os.environ.get("DOCKER_HOST")
                if not endpoint:
                    endpoint, _ = await _docker(
                        "context",
                        "inspect",
                        "--format",
                        "{{.Endpoints.docker.Host}}",
                    )
                    endpoint = endpoint.strip()
                parsed = urlsplit(endpoint)
                if parsed.scheme != "unix" or not parsed.path:
                    raise ValueError(
                        "DockerRuntime Compose service access requires a local Unix "
                        f"Docker endpoint, got {endpoint!r}"
                    )
                service_socket = parsed.path
            main = _docker_compose_main(
                f"127.0.0.1::{self.port}",
                resources,
                service_socket=service_socket,
            )
            project = f"hud-{uuid.uuid4().hex[:12]}"
            with tempfile.TemporaryDirectory(prefix="hud-compose-") as directory:
                override, ports = _compose_overrides(Path(directory), main)
                command = (
                    "compose",
                    "--project-name",
                    project,
                    "--file",
                    str(compose),
                    "--file",
                    str(override),
                    "--file",
                    str(ports),
                )
                try:
                    await _docker(*command, "up", "--detach", "--build", "--remove-orphans")
                    mapping, _ = await _docker(*command, "port", "main", str(self.port))
                    if not mapping.strip():
                        logs_out, logs_err = await _docker(
                            *command, "logs", "--tail", "40", "main", check=False
                        )
                        raise RuntimeError(
                            f"Compose main service exited before serving port {self.port}:\n"
                            f"{(logs_err or logs_out).strip()}"
                        )
                    host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
                    yield Runtime(
                        f"tcp://127.0.0.1:{host_port}",
                        config=config if config.model_dump(exclude_none=True) else None,
                    )
                finally:
                    await _docker(
                        *command,
                        "down",
                        "--volumes",
                        "--remove-orphans",
                        check=False,
                    )
            return
        if config.image is None:
            raise ValueError(
                "DockerRuntime requires runtime_config.image or runtime_config.compose"
            )

        resource_args: list[str] = []
        resources = config.resources
        if resources is not None:
            if resources.cpu is not None:
                cpu = (
                    str(int(resources.cpu))
                    if isinstance(resources.cpu, float) and resources.cpu.is_integer()
                    else str(resources.cpu)
                )
                resource_args.extend(("--cpus", cpu))
            if resources.memory_mb is not None:
                resource_args.extend(("--memory", f"{resources.memory_mb}m"))
            if resources.gpu is not None:
                if resources.gpu.type is not None:
                    raise ValueError("DockerRuntime cannot select GPUs by type")
                resource_args.extend(("--gpus", str(resources.gpu.count)))

        out, _ = await _docker(
            "run",
            "--detach",
            *self.run_args,
            *resource_args,
            *_DOCKER_SECURITY_ARGS,
            "--publish",
            f"127.0.0.1::{self.port}",
            config.image,
        )
        container = out.strip()
        try:
            mapping, _ = await _docker("port", container, str(self.port))
            if not mapping.strip():
                logs_out, logs_err = await _docker("logs", "--tail", "40", container, check=False)
                raise RuntimeError(
                    f"container for image {config.image!r} exited before serving port "
                    f"{self.port}:\n{(logs_err or logs_out).strip()}",
                )
            host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
            yield Runtime(f"tcp://127.0.0.1:{host_port}", config=config)
        finally:
            # check=False: teardown must not shadow the run's own error, and
            # rm -f only fails when the daemon itself is broken.
            await _docker("rm", "--force", container, check=False)


class ModalRuntime:
    """The Modal provider: each acquisition ``Sandbox.create``s a fresh container.

    The cloud :class:`DockerRuntime` — boots a sandbox from a pre-built image,
    exposes the env's control channel as a raw-TCP tunnel (``unencrypted_ports``,
    the only kind :func:`hud.clients.connect` dials), yields its :class:`Runtime`,
    terminates on exit. Acquisitions are independent, so a batch fans out into
    isolated containers (one ``sb-…`` id each).

    The image resolves once (so concurrent rollouts can't race a build): pass a
    published name — ``ModalRuntime("hud-libero-env")``, the preferred durable
    handle — or, as an escape hatch, an ``Image`` to build lazily on first use.
    Requires the ``modal`` extra and a configured token.
    """

    def __init__(
        self,
        image_name: str | None = None,
        *,
        image: Any = None,
        command: Sequence[str] | None = None,
        app_name: str = "hud-envs",
        workdir: str | None = None,
        port: int = 8765,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> None:
        self.image_name = image_name
        self.port = port
        self.env_vars = dict(env_vars or {})
        self.workdir = workdir
        # Default CMD mirrors the scaffolded Dockerfile.hud entrypoint. Leave
        # workdir unset by default so Modal preserves the image WORKDIR.
        self.command = (
            tuple(command)
            if command is not None
            else (
                "hud",
                "serve",
                "env.py",
                "--host",
                "0.0.0.0",  # noqa: S104 - serving inside the sandbox; the tunnel is the only ingress
                "--port",
                str(port),
            )
        )
        self.app_name = app_name
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config
        # Resolved (named) or built-once (from Dockerfile) image, behind a lock so
        # concurrent first acquisitions build/look up exactly once.
        self._image = image
        self._resolved: Any = None
        self._image_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        compose = config.compose.resolve() if config.compose is not None else None
        modal: Any = importlib.import_module("modal")

        app = None
        if compose is not None:
            image = modal.Image.from_registry("docker:28.3.3-dind")
        elif config.image is not None:
            image = _modal_image_from_uri(modal, config.image)
        elif self.image_name is not None:
            image = modal.Image.from_name(self.image_name)
        elif self._image is None:
            raise ValueError(
                "ModalRuntime requires image=, image_name=, runtime_config.image, "
                "or runtime_config.compose"
            )
        else:
            if self._resolved is None:
                async with self._image_lock:
                    if self._resolved is None:
                        app = await modal.App.lookup.aio(
                            self.app_name,
                            create_if_missing=True,
                        )
                        await self._image.build.aio(app=app)
                        self._resolved = self._image
            image = self._resolved
        if self.env_vars:
            image = image.env(self.env_vars)

        if app is None:
            app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)

        sandbox_kwargs: dict[str, Any] = {}
        resources = config.resources
        if resources is not None and resources.cpu is not None:
            sandbox_kwargs["cpu"] = resources.cpu
        if resources is not None and resources.memory_mb is not None:
            sandbox_kwargs["memory"] = resources.memory_mb
        if resources is not None and resources.gpu is not None:
            gpu_type = resources.gpu.type or "any"
            gpu = gpu_type if resources.gpu.count == 1 else f"{gpu_type}:{resources.gpu.count}"
            sandbox_kwargs["gpu"] = gpu

        run_timeout = 3600
        ready_timeout = 600
        if config.limits is not None:
            run_timeout = config.limits.run_timeout_s or run_timeout
            ready_timeout = config.limits.startup_timeout_s or ready_timeout

        sb = await modal.Sandbox.create.aio(
            *(() if compose is not None else self.command),
            app=app,
            image=image,
            workdir=None if compose is not None else self.workdir,
            unencrypted_ports=[self.port],
            readiness_probe=(None if compose is not None else modal.Probe.with_tcp(self.port)),
            # Modal types both timeouts as int seconds; floats raise at proto encode.
            timeout=run_timeout,
            **({"experimental_options": {"vm_runtime": True}} if compose is not None else {}),
            **sandbox_kwargs,
        )
        try:
            if compose is None:
                await sb.wait_until_ready.aio(timeout=ready_timeout)
            else:
                main = _docker_compose_main(
                    f"{self.port}:{self.port}",
                    resources,
                    seccomp="/hud/docker-seccomp.json",
                    service_socket=(
                        "/var/run/docker.sock" if config.compose_service_access else None
                    ),
                )
                with _compose_payload(compose, main) as (
                    archive,
                    override,
                    ports,
                ):
                    await sb.filesystem.copy_from_local.aio(archive, "/hud/project.tar.gz")
                    await sb.filesystem.copy_from_local.aio(override, "/hud/override.json")
                    await sb.filesystem.copy_from_local.aio(ports, "/hud/ports.yaml")
                    await sb.filesystem.copy_from_local.aio(
                        _DOCKER_SECCOMP_PROFILE, "/hud/docker-seccomp.json"
                    )
                command = (
                    "mkdir -p /hud/project && "
                    "tar -xzf /hud/project.tar.gz -C /hud/project && "
                    "until docker info >/dev/null 2>&1; do sleep 1; done && "
                    "docker compose --project-directory /hud/project "
                    f"--file /hud/project/{shlex.quote(compose.name)} "
                    "--file /hud/override.json --file /hud/ports.yaml "
                    "up --detach --build --remove-orphans"
                )
                process = await sb.exec.aio("sh", "-c", command, timeout=ready_timeout)
                if await process.wait.aio() != 0:
                    error = (await process.stderr.read.aio()).strip()
                    raise RuntimeError(f"Modal Compose startup failed: {error}")
            host, port = (await sb.tunnels.aio())[self.port].tcp_socket
            yield Runtime(
                f"tcp://{host}:{port}",
                params={"provider": "modal", "instance_id": sb.object_id},
                config=config if config.model_dump(exclude_none=True) else None,
            )
        finally:
            # check-free teardown: never shadow the run's own error.
            if compose is not None:
                with contextlib.suppress(Exception):
                    process = await sb.exec.aio(
                        "docker",
                        "compose",
                        "--project-directory",
                        "/hud/project",
                        "--file",
                        f"/hud/project/{compose.name}",
                        "--file",
                        "/hud/override.json",
                        "--file",
                        "/hud/ports.yaml",
                        "down",
                        "--volumes",
                        "--remove-orphans",
                        timeout=30,
                    )
                    await process.wait.aio()
            with contextlib.suppress(Exception):
                await sb.terminate.aio()


async def _snapshot_is_current(snapshot: Any, image: Any) -> bool:
    """Whether *snapshot* was built from *image* as it exists right now.

    Daytona records what a snapshot was built from — the registry ref, or for a
    built ``Image`` its Dockerfile text plus the hashes of the context it
    uploads (``build_info``). The hashes are recomputed here with the SDK's own
    hasher so they are comparable to what ``snapshot.create`` would upload.
    ``_context_list`` is the same private attribute ``snapshot.create`` reads.
    """
    if isinstance(image, str):
        return bool(snapshot.image_name == image)
    build = snapshot.build_info
    if build is None:
        return False
    object_storage: Any = importlib.import_module("daytona._async.object_storage")
    AsyncObjectStorage = object_storage.AsyncObjectStorage

    # The hasher is an instance method only for code organization; credentials
    # are needed to upload, not to hash, so skip the credentialed __init__.
    storage = AsyncObjectStorage.__new__(AsyncObjectStorage)
    hashes = [
        await storage._compute_hash_for_path_md5(entry.source_path, entry.archive_path)
        for entry in image._context_list
    ]
    return bool(
        build.dockerfile_content == image.dockerfile()
        and list(build.context_hashes or []) == hashes
    )


class DaytonaRuntime:
    """The Daytona provider: each acquisition creates a fresh sandbox from a snapshot.

    The Daytona runtime boots a sandbox from a pre-built *snapshot*
    (the durable handle, the snapshot equivalent of Modal's image name), starts the
    env's control channel inside it, then reaches it over an SSH local-forward:
    Daytona exposes services only as HTTPS previews, but :func:`hud.clients.connect`
    dials ``tcp://``, so the raw control channel is tunneled over SSH to a local
    port. Yields its :class:`Runtime`, deletes the sandbox on exit.

    Pass a snapshot name — ``DaytonaRuntime("hud-libero-env")`` — optionally with an
    ``image`` (Dockerfile/registry ref) to build that snapshot if it is missing.
    With *image*, an existing snapshot is compared against the image's recorded
    build (Dockerfile plus context hashes) and rebuilt under the same name when
    they differ, so editing the env never silently reuses the snapshot built
    before the edit.
    Resources (cpu/memory/gpu) live on the snapshot, not here. *workdir* defaults to
    ``/app`` (the scaffolded ``Dockerfile.hud`` WORKDIR) since a Daytona session
    starts in ``~``, not the image's WORKDIR; override only for a non-standard layout.
    Requires the ``daytona`` extra and ``DAYTONA_API_KEY``.
    """

    def __init__(
        self,
        snapshot_name: str | None = None,
        *,
        image: Any = None,
        command: str | None = None,
        workdir: str | None = "/app",
        port: int = 8765,
        ssh_host: str = "ssh.app.daytona.io",
        ssh_expires_minutes: int = 24 * 60,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
    ) -> None:
        self.snapshot_name = snapshot_name
        # Default command serves on *port*, so the SSH forward target always
        # matches what's listening; override only for a non-default layout.
        self.command = (
            command or f'PATH="$PWD/.venv/bin:$PATH" hud serve env.py --host 0.0.0.0 --port {port}'
        )
        self.workdir = workdir
        self.port = port
        self.ssh_host = ssh_host
        self.ssh_expires_minutes = ssh_expires_minutes
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config
        # Resolve each snapshot name against the image once; lock so concurrent
        # first acquisitions resolve exactly once.
        self._image = image
        self._resolved: set[str] = set()
        self._snapshot_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        import asyncssh

        daytona_sdk: Any = importlib.import_module("daytona")
        AsyncDaytona = daytona_sdk.AsyncDaytona
        CreateSandboxFromImageParams = daytona_sdk.CreateSandboxFromImageParams
        CreateSandboxFromSnapshotParams = daytona_sdk.CreateSandboxFromSnapshotParams
        CreateSnapshotParams = daytona_sdk.CreateSnapshotParams
        DaytonaNotFoundError = daytona_sdk.DaytonaNotFoundError
        GpuType = daytona_sdk.GpuType
        Image = daytona_sdk.Image
        Resources = daytona_sdk.Resources
        SessionExecuteRequest = daytona_sdk.SessionExecuteRequest

        async with AsyncDaytona() as daytona:
            config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
            if config.compose is not None:
                raise ValueError("DaytonaRuntime does not support runtime_config.compose")
            if config.limits is not None and config.limits.run_timeout_s is not None:
                raise ValueError("DaytonaRuntime does not support runtime_config.run_timeout_s")

            daytona_resources = None
            if config.resources is not None:
                resource_kwargs: dict[str, Any] = {}
                if config.resources.cpu is not None:
                    # Daytona allocates whole cores; truncating resizes silently.
                    if (
                        isinstance(config.resources.cpu, float)
                        and not config.resources.cpu.is_integer()
                    ):
                        raise ValueError(
                            "DaytonaRuntime needs a whole number of CPUs, got "
                            f"{config.resources.cpu}"
                        )
                    resource_kwargs["cpu"] = int(config.resources.cpu)
                if config.resources.memory_mb is not None:
                    resource_kwargs["memory"] = max(
                        1,
                        (config.resources.memory_mb + 1023) // 1024,
                    )
                if config.resources.gpu is not None:
                    resource_kwargs["gpu"] = config.resources.gpu.count
                    if config.resources.gpu.type is not None:
                        resource_kwargs["gpu_type"] = [GpuType(config.resources.gpu.type)]
                if resource_kwargs:
                    daytona_resources = Resources(**resource_kwargs)

            if config.image is not None:
                sandbox_params = CreateSandboxFromImageParams(
                    image=Image.base(config.image),
                    ephemeral=True,
                    auto_stop_interval=0,
                    resources=daytona_resources,
                )
            else:
                snapshot_name = self.snapshot_name
                snapshot_image = self._image
                if snapshot_name is None:
                    raise ValueError(
                        "DaytonaRuntime requires snapshot_name or runtime_config.image"
                    )
                if daytona_resources is not None and snapshot_image is None:
                    raise ValueError(
                        "DaytonaRuntime cannot resize an already-built snapshot: resources "
                        "are fixed when it is built, so pass image= to build one"
                    )
                if snapshot_image is not None:
                    if daytona_resources is not None:
                        # Sizing is baked in at build time, so each sizing is its
                        # own snapshot under a readable suffix (env-4cpu-8gb).
                        sizing = []
                        if daytona_resources.cpu:
                            sizing.append(f"{daytona_resources.cpu}cpu")
                        if daytona_resources.memory:
                            sizing.append(f"{daytona_resources.memory}gb")
                        if daytona_resources.gpu:
                            sizing.append(f"{daytona_resources.gpu}gpu")
                            sizing.extend(
                                (t.value if isinstance(t, GpuType) else t).lower()
                                for t in daytona_resources.gpu_type or []
                            )
                        snapshot_name = "-".join([snapshot_name, *sizing])
                    async with self._snapshot_lock:
                        if snapshot_name not in self._resolved:
                            try:
                                existing = await daytona.snapshot.get(snapshot_name)
                            except DaytonaNotFoundError:
                                existing = None
                            if existing is not None and not await _snapshot_is_current(
                                existing, snapshot_image
                            ):
                                logger.info(
                                    "Daytona snapshot %s is stale; rebuilding", snapshot_name
                                )
                                await daytona.snapshot.delete(existing)
                                # Deletion frees the name asynchronously (~10s
                                # observed); creating before it lands conflicts.
                                async with asyncio.timeout(120):
                                    while True:
                                        try:
                                            await daytona.snapshot.get(snapshot_name)
                                        except DaytonaNotFoundError:
                                            break
                                        await asyncio.sleep(0.5)
                                existing = None
                            if existing is None:
                                logger.info("building Daytona snapshot %s", snapshot_name)
                                await daytona.snapshot.create(
                                    CreateSnapshotParams(
                                        name=snapshot_name,
                                        image=snapshot_image,
                                        resources=daytona_resources,
                                    )
                                )
                            self._resolved.add(snapshot_name)
                sandbox_params = CreateSandboxFromSnapshotParams(
                    snapshot=snapshot_name,
                    ephemeral=True,
                    auto_stop_interval=0,
                )

            create_timeout = 120
            if config.limits is not None and config.limits.startup_timeout_s is not None:
                create_timeout = config.limits.startup_timeout_s
            # ephemeral: these sandboxes are per-rollout and deleted on exit anyway,
            # and some regions only permit ephemeral sandboxes.
            sandbox = await daytona.create(
                sandbox_params,
                timeout=create_timeout,
            )
            session_command: Any = None
            try:
                # Start the env server in a background session (the snapshot's CMD is
                # not the sandbox's main process). connect() retries the handshake,
                # so we don't poll for readiness here.
                session: str = "hud-serve"
                await sandbox.process.create_session(session)
                cmd = f"cd {self.workdir} && {self.command}" if self.workdir else self.command
                session_command = await sandbox.process.execute_session_command(
                    session, SessionExecuteRequest(command=cmd, run_async=True)
                )
                ssh = await sandbox.create_ssh_access(expires_in_minutes=self.ssh_expires_minutes)
                async with asyncssh.connect(
                    self.ssh_host, username=ssh.token, known_hosts=None
                ) as conn:
                    listener = await conn.forward_local_port("127.0.0.1", 0, "127.0.0.1", self.port)
                    try:
                        yield Runtime(
                            f"tcp://127.0.0.1:{listener.get_port()}",
                            params={"provider": "daytona", "instance_id": sandbox.id},
                            config=config if config.model_dump(exclude_none=True) else None,
                        )
                    except (EOFError, OSError) as exc:
                        # Why it died only exists inside the sandbox, and the
                        # sandbox may already be gone.
                        try:
                            logs = await sandbox.process.get_session_command_logs(
                                session, session_command.cmd_id
                            )
                            output = (logs.stderr or logs.output or logs.stdout or "").strip()
                        except Exception as log_exc:
                            exc.add_note(f"env output unavailable: {log_exc}")
                        else:
                            exc.add_note(
                                f"env output in sandbox {sandbox.id}:\n{output}"
                                if output
                                else "env printed nothing"
                            )
                        raise
            finally:
                try:
                    await daytona.delete(sandbox)
                except Exception:
                    # Swallowing this is how a billable sandbox outlives its process.
                    logger.warning(
                        "failed to delete Daytona sandbox %s; it may still be running",
                        sandbox.id,
                        exc_info=True,
                    )


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


#: Platform trace statuses that end a hosted rollout.
_TERMINAL_TRACE_STATUSES = frozenset({"completed", "error", "cancelled"})
_RUNTIME_READY_TIMEOUT = 300.0


class HUDRuntime:
    """HUD tunnel placement: local agent loop against a HUD-hosted environment.

    The SDK creates a runtime session by environment name, exposes the remote
    control channel through a local TCP listener, and lets the normal rollout
    atom drive it from this process.
    """

    def __init__(self, *, run_timeout: float = 3600.0, runtime_url: str | None = None) -> None:
        self.run_timeout = run_timeout
        self.runtime_url = runtime_url
        self._warned_unsupported_config = False

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
    ) -> Run:
        return await rollout(
            task,
            agent,
            runtime=self,
            trace_id=trace_id,
            job_id=job_id,
            group_id=group_id,
            rollout_timeout=self.run_timeout,
        )

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return self._runtime_session(task)

    @asynccontextmanager
    async def _runtime_session(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.settings import settings as sdk_settings

        if task.runtime_config is not None:
            # The lease resolves the env by name: a stamped image is
            # provenance and rides along. Declared cpu/memory are best-effort
            # on the platform's substrate (warned once, not fatal — loaders
            # stamp them on every row), but a GPU or explicit limits change
            # what the task *is*; running without them would grade a
            # different environment than declared.
            resources = task.runtime_config.resources
            if (resources is not None and resources.gpu is not None) or (
                task.runtime_config.limits is not None
                and task.runtime_config.limits.model_dump(exclude_none=True)
            ):
                raise ValueError(
                    "HUDRuntime cannot honor this task's declared GPU/limits on an "
                    "already-deployed env; run it on a placement that provisions them"
                )
            softly_ignored = task.runtime_config.model_dump(
                exclude_none=True, exclude={"image", "compose"}
            )
            if softly_ignored and not self._warned_unsupported_config:
                self._warned_unsupported_config = True
                logger.warning(
                    "HUDRuntime cannot honor task runtime_config %s on an "
                    "already-deployed env; rollouts proceed on the platform's "
                    "defaults",
                    sorted(softly_ignored),
                )
        api_key = sdk_settings.api_key
        if not api_key:
            raise RuntimeError("HUD runtime tunnel requires HUD_API_KEY")
        runtime_url = (self.runtime_url or sdk_settings.hud_runtime_url).rstrip("/")
        session_id = await self._create_runtime_session(runtime_url, api_key, task)
        server: asyncio.Server | None = None
        try:
            server = await asyncio.start_server(
                lambda reader, writer: self._forward_runtime_connection(
                    runtime_url,
                    api_key,
                    session_id,
                    reader,
                    writer,
                ),
                "127.0.0.1",
                0,
            )
            port = server.sockets[0].getsockname()[1]
            yield Runtime(
                f"tcp://127.0.0.1:{port}",
                params={
                    "session_id": session_id,
                    "gateway_url": runtime_url,
                    "ready_timeout": min(self.run_timeout, _RUNTIME_READY_TIMEOUT),
                },
            )
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            await self._delete_runtime_session(runtime_url, api_key, session_id)

    async def _create_runtime_session(self, runtime_url: str, api_key: str, task: Task) -> str:
        payload: dict[str, Any] = {"environment": task.env}
        trace_id = get_current_trace_id()
        if trace_id is not None:
            with contextlib.suppress(ValueError):
                payload["trace_id"] = str(uuid.UUID(trace_id))
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{runtime_url}/runtime/sessions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        session_id = body.get("id")
        if not isinstance(session_id, str):
            raise RuntimeError("Runtime gateway did not return a session id")
        return session_id

    async def _delete_runtime_session(
        self, runtime_url: str, api_key: str, session_id: str
    ) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            with contextlib.suppress(Exception):
                await client.delete(
                    f"{runtime_url}/runtime/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

    async def _forward_runtime_connection(
        self,
        runtime_url: str,
        api_key: str,
        session_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        import websockets

        ws_url = _runtime_tunnel_ws_url(runtime_url, session_id)
        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                max_size=None,
            ) as websocket:
                await _splice_websocket(reader, writer, websocket)
        finally:
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()


class HostedRuntime:
    """HUD-hosted placement: runs the rollout on a leased box and returns its ``Run``.

    The *client-elsewhere* placement. Where a :class:`Provider` yields a channel
    this process drives, ``HostedRuntime`` runs the whole rollout off-box: the
    platform leases an instance, brings the env's container up on it, and runs
    the agent right next to it (the instance-side driver is just
    :func:`hud.eval.run.rollout` over a ``DockerRuntime`` — co-location all the
    way down). This process only submits the rollout and polls the trace to
    completion, folding the result into a :class:`~hud.eval.run.Run`. Because
    the agent runs remotely, its identity travels via :func:`_agent_spec`.

    ``run_timeout`` bounds one rollout end to end, including instance
    provisioning (a cold EC2 boot plus image pull), queueing, and the agent
    run itself. A local cancel (Ctrl-C) requests a platform-side cancel before
    propagating, so abandoned rollouts do not hold instances open.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 5.0,
        run_timeout: float = 3600.0,
    ) -> None:
        self.poll_interval = poll_interval
        self.run_timeout = run_timeout
        self._cancellations: set[asyncio.Task[None]] = set()

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
    ) -> Run:
        """Submit one rollout, await its terminal trace, and fold it into a ``Run``.

        The platform owns the trace lifecycle (the instance-side driver reports
        enter/exit and streams telemetry), so this never double-reports.
        Failures isolating one rollout from its batch (submit rejected, the
        env/model unresolved) surface as :meth:`Run.failed`; a timeout or a
        local cancel propagate, having first asked the platform to release the
        lease.
        """
        trace_id = trace_id or uuid.uuid4().hex
        try:
            if task.verifier is not None:
                raise ValueError(
                    "HostedRuntime does not support verifier tasks until hosted rollouts "
                    "can keep both phases in one runtime scope"
                )
            async with asyncio.timeout(self.run_timeout):
                state = await self._submit_and_await(
                    task, agent, job_id=job_id, group_id=group_id, trace_id=trace_id
                )
        except asyncio.CancelledError:
            self._cancel_later(trace_id)
            raise
        except TimeoutError:
            self._cancel_later(trace_id)
            detail = f"hosted rollout {trace_id} did not finish within {self.run_timeout:g}s"
            logger.warning(detail)
            run = Run.failed(detail)
            run.trace.stop_reason = "timeout"
        except Exception as exc:
            logger.warning("hosted rollout failed to launch: %s", exc)
            run = Run.failed(str(exc))
        else:
            run = self._fold(state, trace_id)
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        return run

    async def _submit_and_await(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None,
        trace_id: str,
    ) -> dict[str, Any]:
        from hud.agents.tool_agent import ToolAgent

        if not isinstance(agent, ToolAgent):
            raise ValueError(
                f"hosted execution requires a gateway agent that can serialize its "
                f"identity (Claude/OpenAI/Gemini/OpenAIChat); got {type(agent).__name__}"
            )
        spec = agent.hosted_spec()
        if task.agent_config:
            spec = {
                **spec,
                "config": {**spec.get("config", {}), **task.agent_config},
            }
        platform = PlatformClient.from_settings()
        if not platform.api_key:
            raise RuntimeError("HUD-hosted execution requires HUD_API_KEY")
        payload: dict[str, Any] = {
            # The SDK's hex ids travel as canonical UUID strings.
            "trace_id": str(uuid.UUID(trace_id)),
            "job_id": str(uuid.UUID(job_id)),
            "env": task.env,
            "task": task.id,
            "slug": task.slug,
            "args": task.args,
            "agent": spec,
        }
        if group_id is not None:
            payload["group_id"] = group_id
        if task.runtime_config is not None:
            runtime_config = task.runtime_config.request_payload()
            if runtime_config:
                payload["runtime_config"] = runtime_config
        await platform.apost("/rollouts/submit", json=payload)
        return await self._await_terminal(platform, payload["trace_id"])

    @staticmethod
    def _fold(state: dict[str, Any], trace_id: str) -> Run:
        """Build the local view of a remotely-executed rollout from its trace state."""
        run = Run(None, "", {})
        # The poll loop only returns terminal states, so the status is one of
        # the trace vocabulary; anything else would be a platform bug.
        status = state.get("status")
        run.trace.status = status if status in ("completed", "error", "cancelled") else "error"
        error = state.get("error")
        if error:
            run.record(Step(source="system", error=str(error)))
        reward = state.get("reward")
        ungraded_failure = run.trace.status in ("error", "cancelled") and reward is None
        grade_error = str(error) if error else None
        if ungraded_failure and grade_error is None:
            grade_error = (
                "rollout was cancelled before grading"
                if run.trace.status == "cancelled"
                else "rollout failed before grading"
            )
        run.grade = Grade(
            reward=float(reward) if reward is not None else 0.0,
            is_error=ungraded_failure,
            content=grade_error,
            raw={"score": float(reward)} if reward is not None else {},
        )
        run._runtime = f"hud://trace/{trace_id}"
        return run

    async def _await_terminal(self, platform: PlatformClient, trace_id: str) -> dict[str, Any]:
        while True:
            state: dict[str, Any] = await platform.aget(f"/trace/{trace_id}")
            if state.get("status") in _TERMINAL_TRACE_STATUSES:
                return state
            await asyncio.sleep(self.poll_interval)

    async def _cancel(self, platform: PlatformClient, trace_id: str) -> None:
        # The platform also bounds instances by max runtime; this just releases
        # the lease promptly. Never shadow the caller's outcome.
        try:
            await platform.apost("/rollouts/cancel", json={"trace_id": trace_id})
        except Exception as exc:
            logger.warning("hosted rollout %s cancel failed: %s", trace_id, exc)

    def _cancel_later(self, trace_id: str) -> None:
        task = asyncio.create_task(
            self._cancel(PlatformClient.from_settings(), str(uuid.UUID(trace_id)))
        )
        self._cancellations.add(task)
        task.add_done_callback(self._cancellations.discard)


def _runtime_tunnel_ws_url(runtime_url: str, session_id: str) -> str:
    parts = urlsplit(runtime_url.rstrip("/"))
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = f"{parts.path.rstrip('/')}/runtime/tunnels/{session_id}"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


async def _splice_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    websocket: Any,
) -> None:
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
        if isinstance(result, BaseException):
            raise result


__all__ = [
    "DaytonaRuntime",
    "DockerRuntime",
    "HUDRuntime",
    "HostedRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "Provider",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "Shared",
    "SubprocessRuntime",
]

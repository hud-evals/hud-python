"""Docker runtime provider."""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import tarfile
import tempfile
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Self

from hud.clients.client import DEFAULT_READY_TIMEOUT
from hud.environment.server import DEFAULT_PORT, PORT_ANNOUNCEMENT
from hud.types import ClassifiedError
from hud.utils.docker import docker as _docker
from hud.utils.process import create_process_group_exec, finish_output, stream_output

from .compose import SECCOMP_PROFILE, ComposeConfig, DockerBindMount
from .core import (
    SESSION_ROOT,
    Checkpoint,
    ExecResult,
    Runtime,
    RuntimeConfig,
    SessionScope,
    extract_archive,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from hud.eval.task import Task

logger = logging.getLogger("hud.runtime")

_SESSION_ALIAS_ROOT = "/media/hud/sessions"

_DOCKER_SECURITY_ARGS = (
    "--security-opt",
    f"seccomp={SECCOMP_PROFILE}",
    # Docker exposes system-path masking only as an all-or-nothing option;
    # bwrap replaces the container's proc and dev mounts while building a wall.
    "--security-opt",
    "systempaths=unconfined",
    "--security-opt",
    "apparmor=unconfined",
)


def _require_free_disk(output: str, storage_mb: int) -> None:
    try:
        available_kib = int(output.strip().splitlines()[-1].split()[-3])
    except (IndexError, ValueError):
        raise RuntimeError("DockerRuntime could not measure the environment's free disk") from None
    available_mb = available_kib // 1024
    if available_mb < storage_mb:
        raise RuntimeError(
            f"DockerRuntime requires {storage_mb} MB of free disk; "
            f"the environment has {available_mb} MB"
        )


class _RunDeadline:
    """Tears the box down when the run limit expires, and remembers that it did."""

    def __init__(self, seconds: float | None, teardown: tuple[str, ...]) -> None:
        self._seconds = seconds
        self._teardown = teardown
        self._expiry: asyncio.Task[None] | None = None
        # Why the box is gone once the limit has expired.
        self.expired: str | None = None

    async def _expire(self) -> None:
        assert self._seconds is not None
        await asyncio.sleep(self._seconds)
        self.expired = f"the box was removed when the run limit of {self._seconds:g}s expired"
        await _docker(*self._teardown, check=False)

    async def __aenter__(self) -> Self:
        if self._seconds is not None:
            self._expiry = asyncio.create_task(self._expire())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            await asyncio.gather(self._expiry, return_exceptions=True)


async def _wait_serving(container: str, name: str, seconds: float) -> None:
    """Return once the env in ``container`` announces its port; raise if it exits first."""
    logs = await create_process_group_exec(
        "docker",
        "logs",
        "--follow",
        container,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert logs.stdout is not None
    tail: deque[str] = deque(maxlen=40)
    try:
        async with asyncio.timeout(seconds):
            while True:
                try:
                    line = await logs.stdout.readline()
                except ValueError:
                    # Longer than the reader's limit, so not the announcement.
                    continue
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text.startswith(PORT_ANNOUNCEMENT):
                    return
                tail.append(text)
            # The log follow ends when the container stops.
            code, _ = await _docker("wait", container)
    except TimeoutError:
        raise ClassifiedError(
            f"{name} did not serve within {seconds:.0f}s",
            code="container_start_failed",
            source="user",
        ) from None
    finally:
        await logs.terminate()
    output = "\n".join(tail).strip() or "(no output)"
    raise ClassifiedError(
        f"{name} exited with code {code.strip()} before serving:\n{output}",
        code="container_start_failed",
        source="user",
    )


@asynccontextmanager
async def _docker_output(*args: str) -> AsyncGenerator[None]:
    process = await create_process_group_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    tasks = (
        asyncio.create_task(stream_output(process.stdout, sys.stdout)),
        asyncio.create_task(stream_output(process.stderr, sys.stderr)),
    )
    try:
        yield
    finally:
        await process.terminate()
        await finish_output(*tasks)


class DockerRuntime:
    """Start a HUD environment from an image or a Docker Compose file.

    An image is started with ``docker run``. A Compose file is normalized (interpolated
    against its project ``.env``) and started with a provider override: the control-channel
    port is published on the service that owns ``main``'s network, and ``main`` gets HUD's
    nested-workspace security profile, session volumes, bind mounts, env vars, resource
    limits and, with ``service_access``, a Docker daemon private to ``main``.
    """

    def __init__(
        self,
        image: str | None = None,
        *,
        port: int = DEFAULT_PORT,
        run_args: Sequence[str] = (),
        bind_mounts: Sequence[DockerBindMount] = (),
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> None:
        self.port = port
        self.run_args = tuple(run_args)
        self.bind_mounts = tuple(bind_mounts)
        self.env_vars = dict(env_vars or {})
        config = RuntimeConfig(image=image) if image is not None else RuntimeConfig()
        if runtime_config is not None:
            config = config.with_overrides(RuntimeConfig.model_validate(runtime_config))
        self.runtime_config = config if config.model_dump(exclude_none=True) else None

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncGenerator[Runtime]:
        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        startup_timeout = config.limits.startup_timeout_s if config.limits is not None else None
        run_timeout = config.limits.run_timeout_s if config.limits is not None else None
        params = {"ready_timeout": startup_timeout} if startup_timeout is not None else {}
        resources = config.resources
        if resources is not None:
            resources.require_support("DockerRuntime", {"cpu", "memory_mb", "storage_mb", "gpu"})
        compose_project = config.compose
        if compose_project is not None:
            compose = compose_project.document
            if not isinstance(compose, Path):
                raise ValueError("DockerRuntime requires compose as a local file path")
            compose = compose.resolve()
            if self.run_args:
                raise ValueError("DockerRuntime run_args apply only to image environments")
            compose_config = ComposeConfig.from_file(compose)
            port_service = compose_config.network_owner("main")
            resources = config.resources
            if (
                resources is not None
                and resources.gpu is not None
                and resources.gpu.type is not None
            ):
                raise ValueError("DockerRuntime cannot select Compose GPUs by type")
            project = f"hud-{uuid.uuid4().hex[:12]}"
            with compose_project.stage(
                f"127.0.0.1::{self.port}",
                port_service=port_service,
                seccomp=SECCOMP_PROFILE,
                docker=bool(compose_project.service_access),
                bind_mounts=self.bind_mounts,
                env_vars=self.env_vars,
                cpu=resources.cpu if resources is not None else None,
                memory_mb=resources.memory_mb if resources is not None else None,
                gpu_count=(
                    resources.gpu.count
                    if resources is not None and resources.gpu is not None
                    else None
                ),
            ) as files:
                command = (
                    "compose",
                    "--project-name",
                    project,
                    "--project-directory",
                    str(files.project_directory),
                    "--file",
                    str(files.compose),
                    "--file",
                    str(files.override),
                    "--file",
                    str(files.ports),
                )
                teardown = (*command, "down", "--volumes", "--remove-orphans")
                loop = asyncio.get_running_loop()
                startup_deadline = (
                    None if startup_timeout is None else loop.time() + startup_timeout
                )
                try:
                    try:
                        await _docker(
                            *command,
                            "up",
                            "--detach",
                            "--build",
                            "--remove-orphans",
                            timeout_s=startup_timeout,
                        )
                        if awaited := compose_config.healthchecked_services():
                            await _docker(
                                *command,
                                "up",
                                "--wait",
                                "--no-deps",
                                "--no-recreate",
                                "--no-build",
                                *awaited,
                                timeout_s=(
                                    None
                                    if startup_deadline is None
                                    else startup_deadline - loop.time()
                                ),
                            )
                        out, _ = await _docker(*command, "ps", "--all", "--quiet", "main")
                    except RuntimeError as exc:
                        raise ClassifiedError(
                            str(exc), code="compose_execution_failed", source="user"
                        ) from exc
                    container = out.strip()
                    async with _docker_output(*command, "logs", "--follow", "--no-color"):
                        await _wait_serving(
                            container,
                            "Compose main service",
                            DEFAULT_READY_TIMEOUT
                            if startup_deadline is None
                            else startup_deadline - loop.time(),
                        )
                        if resources is not None and resources.storage_mb is not None:
                            free_disk, _ = await _docker(
                                *command, "exec", "-T", "main", "df", "-Pk", "/"
                            )
                            _require_free_disk(free_disk, resources.storage_mb)
                        try:
                            mapping, _ = await _docker(
                                *command, "port", port_service, str(self.port)
                            )
                        except RuntimeError as exc:  # the service owning the network is gone
                            raise ClassifiedError(
                                str(exc), code="compose_execution_failed", source="user"
                            ) from exc
                        host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
                        async with _RunDeadline(run_timeout, teardown) as deadline:
                            yield DockerEndpoint(
                                f"tcp://127.0.0.1:{host_port}",
                                params=params,
                                config=config if config.model_dump(exclude_none=True) else None,
                                container=container,
                                project=project,
                                deadline=deadline,
                            )
                finally:
                    await _docker(*teardown, check=False)
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

        env_args: list[str] = []
        for key, value in self.env_vars.items():
            env_args.extend(("--env", f"{key}={value}"))
        mount_args: list[str] = []
        for mount in self.bind_mounts:
            mount_args.extend(("--mount", mount.docker_argument()))
        session_volume = f"hud-runtime-sessions-{uuid.uuid4().hex}"
        container = ""
        try:
            await _docker("volume", "create", session_volume, timeout_s=startup_timeout)
            try:
                out, _ = await _docker(
                    "run",
                    "--detach",
                    *self.run_args,
                    *env_args,
                    *resource_args,
                    *mount_args,
                    *_DOCKER_SECURITY_ARGS,
                    "--mount",
                    f"type=volume,source={session_volume},target={SESSION_ROOT}",
                    "--mount",
                    f"type=volume,source={session_volume},target={_SESSION_ALIAS_ROOT}",
                    "--publish",
                    f"127.0.0.1::{self.port}",
                    config.image,
                    timeout_s=startup_timeout,
                )
            except RuntimeError as exc:
                raise ClassifiedError(
                    str(exc), code="container_start_failed", source="user"
                ) from exc
            container = out.strip()
            teardown = ("rm", "--force", container)
            async with _docker_output("logs", "--follow", container):
                await _wait_serving(
                    container,
                    f"container for image {config.image!r}",
                    DEFAULT_READY_TIMEOUT if startup_timeout is None else startup_timeout,
                )
                if resources is not None and resources.storage_mb is not None:
                    free_disk, _ = await _docker("exec", container, "df", "-Pk", "/")
                    _require_free_disk(free_disk, resources.storage_mb)
                mapping, _ = await _docker("port", container, str(self.port))
                host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
                async with _RunDeadline(run_timeout, teardown) as deadline:
                    yield DockerEndpoint(
                        f"tcp://127.0.0.1:{host_port}",
                        params=params,
                        config=config,
                        container=container,
                        deadline=deadline,
                    )
        finally:
            # check=False: teardown must not shadow the run's own error, and
            # rm -f only fails when the daemon itself is broken.
            if container:
                await _docker("rm", "--force", container, check=False)
            await _docker("volume", "rm", "--force", session_volume, check=False)


_ARCHIVE_SCRIPT = """
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
entries = list(root.rglob("*")) if root.exists() else []
if any(entry.is_symlink() for entry in entries):
    raise ValueError("runtime archive contains a symbolic link")
if any(not (entry.is_file() or entry.is_dir()) for entry in entries):
    raise ValueError("runtime archive contains an unsupported entry")
with tarfile.open(sys.argv[2], "w:gz") as output:
    for entry in entries:
        output.add(entry, arcname=entry.relative_to(root), recursive=False)
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class DockerEndpoint(Runtime):
    """The ``Runtime`` a ``DockerRuntime`` yields: its control channel and ``container`` id."""

    transports_sessions: ClassVar[bool] = True

    container: str
    # The Compose project name when the environment is a project; its other
    # services are the siblings ``exec`` and ``collect`` reach by name.
    project: str | None = None
    # The run limit that removes the box when it expires.
    deadline: _RunDeadline | None = None

    async def terminated(self) -> str | None:
        if self.deadline is not None and self.deadline.expired is not None:
            return self.deadline.expired
        state, _ = await _docker(
            "inspect",
            "--format",
            "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}",
            self.container,
        )
        status, code, oom = state.split()
        if status == "running":
            return None
        return f"container {status} with exit code {code}" + (
            " (killed by the memory limit)" if oom == "true" else ""
        )

    async def _service_container(self, service: str) -> str:
        if service == "main":
            return self.container
        if self.project is None:
            raise ValueError(f"service {service!r}: this environment is a single container")
        listed, _ = await _docker(
            "ps",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={self.project}",
            "--filter",
            f"label=com.docker.compose.service={service}",
        )
        container = listed.strip()
        if not container:
            raise ValueError(f"Compose service {service!r} is not running")
        return container

    async def exec(
        self, argv: Sequence[str], *, service: str = "main", max_wait: float | None = None
    ) -> ExecResult:
        container = await self._service_container(service)
        group = await create_process_group_exec(
            "docker",
            "exec",
            container,
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(max_wait):
                out, err = await group.process.communicate()
        except (asyncio.CancelledError, TimeoutError):
            await group.terminate()
            raise
        assert group.process.returncode is not None
        return ExecResult(
            group.process.returncode,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )

    async def collect(
        self,
        paths: Sequence[str | PurePosixPath],
        destination: Path,
        *,
        service: str = "main",
    ) -> list[Path]:
        container = await self._service_container(service)
        collected: list[Path] = []
        for path in paths:
            source = PurePosixPath(path)
            group = await create_process_group_exec(
                "docker",
                "cp",
                f"{container}:{source}",
                "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                archive, err = await group.process.communicate()
            except asyncio.CancelledError:
                await group.terminate()
                raise
            if group.process.returncode != 0:
                raise RuntimeError(
                    f"collecting {source} from {service!r} failed: "
                    f"{err.decode('utf-8', 'replace').strip()}"
                )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(destination, filter="data")
            collected.append(destination / source.name)
        return collected

    async def put(self, source: Path, path: str | PurePosixPath, *, service: str = "main") -> None:
        container = await self._service_container(service)
        target = PurePosixPath(path)
        await _docker(
            "exec",
            "--user",
            "0",
            container,
            "sh",
            "-c",
            'rm -rf "$1" && mkdir -p "$2"',
            "hud-put",
            str(target),
            str(target.parent),
        )
        await _docker("cp", str(source), f"{container}:{target}")

    @asynccontextmanager
    async def checkpoint(self, scope: SessionScope) -> AsyncGenerator[Checkpoint]:
        with tempfile.TemporaryDirectory(prefix="hud-checkpoint-") as directory:
            destination = Path(directory) / "checkpoint.tar.gz"
            archive = f"/runtime/checkpoint-{uuid.uuid4().hex}.tar.gz"
            try:
                await _docker(
                    "exec",
                    "--user",
                    "0",
                    self.container,
                    "python3",
                    "-c",
                    _ARCHIVE_SCRIPT,
                    str(scope.path),
                    archive,
                )
                await _docker("cp", f"{self.container}:{archive}", str(destination))
            finally:
                await _docker(
                    "exec", "--user", "0", self.container, "rm", "-f", archive, check=False
                )
            yield Checkpoint(scope, str(destination))

    async def restore(self, checkpoint: Checkpoint, scope: SessionScope) -> None:
        with tempfile.TemporaryDirectory(prefix="hud-restore-") as directory:
            root = Path(directory)
            extract_archive(Path(checkpoint.ref), root)
            target = str(scope.path)
            await _docker("exec", "--user", "0", self.container, "rm", "-rf", target)
            await _docker("exec", "--user", "0", self.container, "mkdir", "-p", target)
            await _docker("cp", f"{root}/.", f"{self.container}:{target}")

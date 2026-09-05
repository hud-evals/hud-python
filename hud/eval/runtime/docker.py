"""Docker runtime provider."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hud.utils.docker import docker as _docker
from hud.utils.process import create_process_group_exec, finish_output, stream_output

from .compose import ComposeConfig
from .core import Runtime, RuntimeConfig, validate_session_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

#: DockerRuntime always serves HUD environments, so this is part of the
#: provider contract rather than a per-image option. This is intentionally a
#: default-allow compatibility profile: Workspace's bwrap sessions need the
#: namespace and mount syscalls, while unrelated kernel interfaces stay denied.
_DOCKER_SECCOMP_PROFILE = Path(__file__).parent.parent / "docker-seccomp.json"
_DOCKER_SECURITY_ARGS = (
    "--security-opt",
    f"seccomp={_DOCKER_SECCOMP_PROFILE}",
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


async def _prepare_compose_project(compose: Path, max_wait: float | None) -> bool:
    script = compose.parent / "build.sh"
    if not script.is_file():
        return False
    process = await create_process_group_exec(
        "sh",
        str(script),
        cwd=str(compose.parent),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    result = await process.complete(max_wait=max_wait)
    if result.timed_out:
        raise TimeoutError(f"Compose project build timed out after {max_wait:g} seconds")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise RuntimeError(f"Compose project build failed: {detail}")
    return True


@asynccontextmanager
async def _docker_deadline(
    seconds: float | None,
    teardown: tuple[str, ...],
) -> AsyncIterator[None]:
    if seconds is None:
        yield
        return

    async def expire() -> None:
        await asyncio.sleep(seconds)
        await _docker(*teardown, check=False)

    expiry = asyncio.create_task(expire())
    try:
        yield
    finally:
        expiry.cancel()
        await asyncio.gather(expiry, return_exceptions=True)


@asynccontextmanager
async def _docker_output(*args: str) -> AsyncIterator[None]:
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
        compose_service_socket: str | Path | None = None,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> None:
        self.port = port
        self.run_args = tuple(run_args)
        self.env_vars = dict(env_vars or {})
        self.compose_service_socket = (
            str(Path(compose_service_socket)) if compose_service_socket is not None else None
        )
        config = RuntimeConfig(image=image) if image is not None else RuntimeConfig()
        if runtime_config is not None:
            config = config.with_overrides(RuntimeConfig.model_validate(runtime_config))
        self.runtime_config = config if config.model_dump(exclude_none=True) else None
        self._compose_preparation_locks: dict[Path, asyncio.Lock] = {}

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        if config.mounts:
            raise ValueError("DockerRuntime does not support runtime_config.mounts")
        startup_timeout = config.limits.startup_timeout_s if config.limits is not None else None
        run_timeout = config.limits.run_timeout_s if config.limits is not None else None
        params = {"ready_timeout": startup_timeout} if startup_timeout is not None else {}
        resources = config.resources
        if resources is not None:
            resources._require_support("DockerRuntime", {"cpu", "memory_mb", "storage_mb", "gpu"})
        compose_project = config.compose
        if compose_project is not None:
            compose = compose_project.document
            if not isinstance(compose, Path):
                raise ValueError("DockerRuntime requires compose as a local file path")
            compose = compose.resolve()
            if self.run_args:
                raise ValueError("DockerRuntime run_args apply only to image environments")
            port_service = ComposeConfig.from_file(compose).network_owner("main")
            resources = config.resources
            if (
                resources is not None
                and resources.gpu is not None
                and resources.gpu.type is not None
            ):
                raise ValueError("DockerRuntime cannot select Compose GPUs by type")
            service_socket = None
            if compose_project.service_access:
                service_socket = self.compose_service_socket
                if service_socket is None:
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
                            "DockerRuntime Compose service access through a remote daemon "
                            "requires compose_service_socket"
                        )
                    service_socket = parsed.path
            project = f"hud-{uuid.uuid4().hex[:12]}"
            lock = self._compose_preparation_locks.setdefault(compose, asyncio.Lock())
            async with lock:
                prepared = await _prepare_compose_project(compose, startup_timeout)
            with compose_project.stage(
                f"127.0.0.1::{self.port}",
                port_service=port_service,
                seccomp=_DOCKER_SECCOMP_PROFILE,
                service_socket=service_socket,
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
                try:
                    await _docker(
                        *command,
                        "up",
                        "--detach",
                        "--no-build" if prepared else "--build",
                        "--remove-orphans",
                        deadline=startup_timeout,
                    )
                    if resources is not None and resources.storage_mb is not None:
                        free_disk, _ = await _docker(
                            *command, "exec", "-T", "main", "df", "-Pk", "/"
                        )
                        _require_free_disk(free_disk, resources.storage_mb)
                    mapping, _ = await _docker(*command, "port", port_service, str(self.port))
                    if not mapping.strip():
                        logs_out, logs_err = await _docker(
                            *command, "logs", "--tail", "40", port_service, check=False
                        )
                        raise RuntimeError(
                            f"Compose {port_service} service exited before serving "
                            f"port {self.port}:\n"
                            f"{(logs_err or logs_out).strip()}"
                        )
                    host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
                    container, _ = await _docker(*command, "ps", "--quiet", "main")
                    async with (
                        _docker_output(
                            *command,
                            "logs",
                            "--follow",
                            "--no-color",
                        ),
                        _docker_deadline(run_timeout, teardown),
                    ):
                        yield DockerEndpoint(
                            f"tcp://127.0.0.1:{host_port}",
                            params=params,
                            config=config if config.model_dump(exclude_none=True) else None,
                            container=container.strip(),
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
        out, _ = await _docker(
            "run",
            "--detach",
            *self.run_args,
            *env_args,
            *resource_args,
            *_DOCKER_SECURITY_ARGS,
            "--publish",
            f"127.0.0.1::{self.port}",
            config.image,
            deadline=startup_timeout,
        )
        container = out.strip()
        teardown = ("rm", "--force", container)
        try:
            if resources is not None and resources.storage_mb is not None:
                free_disk, _ = await _docker("exec", container, "df", "-Pk", "/")
                _require_free_disk(free_disk, resources.storage_mb)
            mapping, _ = await _docker("port", container, str(self.port))
            if not mapping.strip():
                logs_out, logs_err = await _docker("logs", "--tail", "40", container, check=False)
                raise RuntimeError(
                    f"container for image {config.image!r} exited before serving port "
                    f"{self.port}:\n{(logs_err or logs_out).strip()}",
                )
            host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
            async with (
                _docker_output(
                    "logs",
                    "--follow",
                    container,
                ),
                _docker_deadline(run_timeout, teardown),
            ):
                yield DockerEndpoint(
                    f"tcp://127.0.0.1:{host_port}",
                    params=params,
                    config=config,
                    container=container,
                )
        finally:
            # check=False: teardown must not shadow the run's own error, and
            # rm -f only fails when the daemon itself is broken.
            await _docker(*teardown, check=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class DockerEndpoint(Runtime):
    container: str

    @asynccontextmanager
    async def snapshot_session(self, session_id: str) -> AsyncIterator[Path | None]:
        validate_session_id(session_id)
        with tempfile.TemporaryDirectory(prefix="hud-session-") as directory:
            destination = Path(directory) / "session.tar.gz"
            root = f"/media/hud/sessions/{session_id}"
            exists, _ = await _docker(
                "exec",
                self.container,
                "sh",
                "-c",
                'if [ -d "$1" ]; then printf 1; fi',
                "hud-session",
                root,
            )
            if not exists:
                yield None
                return
            archive = f"/media/hud/session-export-{uuid.uuid4().hex}.tar.gz"
            script = """
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
entries = list(root.rglob("*"))
if any(entry.is_symlink() for entry in entries):
    raise ValueError("runtime session contains a symbolic link")
if any(not (entry.is_file() or entry.is_dir()) for entry in entries):
    raise ValueError("runtime session contains an unsupported entry")
with tarfile.open(sys.argv[2], "w:gz") as output:
    for entry in entries:
        output.add(entry, arcname=entry.relative_to(root), recursive=False)
"""
            try:
                await _docker(
                    "exec",
                    "--user",
                    "0",
                    self.container,
                    "python3",
                    "-c",
                    script,
                    root,
                    archive,
                )
                await _docker("cp", f"{self.container}:{archive}", str(destination))
            finally:
                await _docker(
                    "exec", "--user", "0", self.container, "rm", "-f", archive, check=False
                )
            yield destination

    async def restore_session(self, session_id: str, source: Path) -> None:
        validate_session_id(session_id)
        target = f"/media/hud/sessions/{session_id}"
        with tempfile.TemporaryDirectory(prefix="hud-session-import-") as directory:
            root = Path(directory)
            _extract_session_archive(source, root)
            await _docker("exec", "--user", "0", self.container, "rm", "-rf", target)
            await _docker("exec", "--user", "0", self.container, "mkdir", "-p", target)
            await _docker(
                "cp",
                f"{root}/.",
                f"{self.container}:{target}",
            )


def _extract_session_archive(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r:gz") as archive:
        if any(not (member.isfile() or member.isdir()) for member in archive.getmembers()):
            raise ValueError("runtime session archive contains an unsupported entry")
        archive.extractall(destination, filter="data")

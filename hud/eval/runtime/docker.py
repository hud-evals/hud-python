"""Docker runtime provider."""

from __future__ import annotations

import asyncio
import logging
import os
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hud.utils.docker import docker as _docker
from hud.utils.process import create_process_group_exec

from .compose import ComposeConfig, ComposeProject
from .core import Runtime, RuntimeConfig

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


async def _prepare_compose_project(compose: Path) -> bool:
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
    result = await process.complete()
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise RuntimeError(f"Compose project build failed: {detail}")
    return True


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
        if config.limits is not None and config.limits.run_timeout_s is not None:
            raise ValueError("DockerRuntime does not support runtime_config.limits.run_timeout_s")
        params = (
            {"ready_timeout": config.limits.startup_timeout_s}
            if config.limits is not None and config.limits.startup_timeout_s is not None
            else {}
        )
        resources = config.resources
        if resources is not None:
            resources._require_support("DockerRuntime", {"cpu", "memory_mb", "storage_mb", "gpu"})
        compose_source = config.compose_source()
        if compose_source is not None:
            if self.run_args:
                raise ValueError("DockerRuntime run_args apply only to image environments")
            compose = compose_source.runnable_path("DockerRuntime")
            port_service = ComposeConfig.from_file(compose).network_owner("main")
            resources = config.resources
            if (
                resources is not None
                and resources.gpu is not None
                and resources.gpu.type is not None
            ):
                raise ValueError("DockerRuntime cannot select Compose GPUs by type")
            service_socket = None
            if config.compose_service_access:
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
            project_files = ComposeProject(compose)
            project = f"hud-{uuid.uuid4().hex[:12]}"
            lock = self._compose_preparation_locks.setdefault(compose, asyncio.Lock())
            async with lock:
                prepared = await _prepare_compose_project(compose)
            with project_files.stage(
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
                    "--file",
                    str(files.compose),
                    "--file",
                    str(files.override),
                    "--file",
                    str(files.ports),
                )
                try:
                    await _docker(
                        *command,
                        "up",
                        "--detach",
                        "--no-build" if prepared else "--build",
                        "--remove-orphans",
                    )
                    if resources is not None and resources.storage_mb is not None:
                        free_disk, _ = await _docker(
                            *command, "exec", "-T", "main", "df", "-Pk", "/"
                        )
                        _require_free_disk(free_disk, resources.storage_mb)
                    mapping, _ = await _docker(*command, "port", port_service, str(self.port))
                    if not mapping.strip():
                        logs_out, logs_err = await _docker(
                            *command, "logs", "--tail", "40", "main", check=False
                        )
                        raise RuntimeError(
                            f"Compose main service exited before serving port {self.port}:\n"
                            f"{(logs_err or logs_out).strip()}"
                        )
                    host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
                    container, _ = await _docker(*command, "ps", "--quiet", "main")
                    handoff = _DockerHandoff(container.strip())
                    await handoff.prepare()
                    yield Runtime(
                        f"tcp://127.0.0.1:{host_port}",
                        params=params,
                        config=config if config.model_dump(exclude_none=True) else None,
                        handoff=handoff,
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
        )
        container = out.strip()
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
            handoff = _DockerHandoff(container)
            await handoff.prepare()
            yield Runtime(
                f"tcp://127.0.0.1:{host_port}",
                params=params,
                config=config,
                handoff=handoff,
            )
        finally:
            # check=False: teardown must not shadow the run's own error, and
            # rm -f only fails when the daemon itself is broken.
            await _docker("rm", "--force", container, check=False)


@dataclass(frozen=True, slots=True)
class _DockerHandoff:
    container: str

    async def prepare(self) -> None:
        await _docker("exec", self.container, "mkdir", "-p", "/media/hud/handoffs")

    async def export_to(self, destination: Path) -> None:
        archive = f"/media/hud/handoff-export-{uuid.uuid4().hex}.tar.gz"
        script = """
import sys
import tarfile
from pathlib import Path

root = Path("/media/hud/handoffs")
entries = list(root.rglob("*"))
if any(entry.is_symlink() for entry in entries):
    raise ValueError("runtime handoff contains a symbolic link")
with tarfile.open(sys.argv[1], "w:gz") as output:
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
                archive,
            )
            await _docker("cp", f"{self.container}:{archive}", str(destination))
        finally:
            await _docker("exec", "--user", "0", self.container, "rm", "-f", archive, check=False)

    async def import_from(self, source: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="hud-handoff-import-") as directory:
            root = Path(directory)
            _extract_handoff_archive(source, root)
            await self.prepare()
            await _docker(
                "cp",
                f"{root}/.",
                f"{self.container}:/media/hud/handoffs",
            )


def _extract_handoff_archive(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r:gz") as archive:
        if any(not (member.isfile() or member.isdir()) for member in archive.getmembers()):
            raise ValueError("runtime handoff archive contains an unsupported entry")
        archive.extractall(destination, filter="data")

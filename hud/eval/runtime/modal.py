"""Modal runtime provider."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from hud.utils.process import finish_output, stream_output

from .compose import ComposeConfig
from .core import Runtime, RuntimeConfig, validate_session_id
from .docker import _DOCKER_SECCOMP_PROFILE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    import modal

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

_MODAL_COMPOSE_CPU = 4.0
_MODAL_COMPOSE_MEMORY_MB = 8192


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalEndpoint(Runtime):
    sandbox: modal.Sandbox
    compose: str | None

    def _container(self) -> str:
        if self.compose is None:
            return ""
        compose = shlex.quote(self.compose)
        project_directory = shlex.quote(str(PurePosixPath(self.compose).parent))
        return (
            f"CONTAINER=$(docker compose --project-directory {project_directory} "
            f"--file {compose} --file /hud/override.json "
            "--file /hud/ports.yaml ps --quiet main); "
        )

    async def _exec(self, command: str) -> str:
        process = await self.sandbox.exec.aio("sh", "-c", command)
        returncode = await process.wait.aio()
        stdout = await process.stdout.read.aio()
        if returncode != 0:
            raise RuntimeError((await process.stderr.read.aio()).strip())
        return stdout

    @asynccontextmanager
    async def snapshot_session(self, session_id: str) -> AsyncIterator[Path | None]:
        validate_session_id(session_id)
        with tempfile.TemporaryDirectory(prefix="hud-session-") as directory:
            destination = Path(directory) / "session.tar.gz"
            session = f"/media/hud/sessions/{session_id}"
            if self.compose is None:
                root = session
                command = ""
                check = f"if [ -d {root} ]; then printf 1; fi"
            else:
                root = "/media/hud/session-export"
                command = (
                    self._container()
                    + f"rm -rf {root} && mkdir -p {root} && "
                    + f'docker cp "$CONTAINER":{session}/. {root} && '
                )
                check = (
                    self._container()
                    + f'docker exec "$CONTAINER" test -d {session} && printf 1 || true'
                )
            if not (await self._exec(check)).strip():
                yield None
                return
            command += (
                f"if find {root} -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; "
                "then echo 'runtime session contains an unsupported entry' >&2; exit 1; fi; "
                f"tar -czf /media/hud/session.tar.gz -C {root} ."
            )
            await self._exec(command)
            await self.sandbox.filesystem.copy_to_local.aio(
                "/media/hud/session.tar.gz", destination
            )
            yield destination

    async def restore_session(self, session_id: str, source: Path) -> None:
        validate_session_id(session_id)
        session = f"/media/hud/sessions/{session_id}"
        await self.sandbox.filesystem.copy_from_local.aio(source, "/media/hud/session.tar.gz")
        if self.compose is None:
            command = (
                f"rm -rf {session} && mkdir -p {session} && "
                f"tar -xzf /media/hud/session.tar.gz -C {session}"
            )
        else:
            command = (
                self._container()
                + "rm -rf /tmp/hud-session && mkdir -p /tmp/hud-session && "
                + "tar -xzf /media/hud/session.tar.gz -C /tmp/hud-session && "
                + f'docker exec "$CONTAINER" sh -c "rm -rf {session} && mkdir -p {session}" && '
                + f'docker cp /tmp/hud-session/. "$CONTAINER":{session}'
            )
        await self._exec(command)


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
        image: modal.Image | None = None,
        command: Sequence[str] | None = None,
        app: modal.App | None = None,
        app_name: str | None = None,
        workdir: str | None = None,
        port: int = 8765,
        runtime_config: RuntimeConfig | dict[str, Any] | None = None,
        env_vars: Mapping[str, str] | None = None,
        secrets: Sequence[modal.Secret] | None = None,
        registry_secret: modal.Secret | None = None,
    ) -> None:
        if app is not None and app_name is not None:
            raise ValueError("ModalRuntime accepts either app or app_name, not both")
        self.image_name = image_name
        self.port = port
        self.env_vars = dict(env_vars or {})
        self.secrets = tuple(secrets or ())
        self.registry_secret = registry_secret
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
        self.app = app
        self.app_name = "hud-envs" if app_name is None else app_name
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config
        # Resolved (named) or built-once (from Dockerfile) image, behind a lock so
        # concurrent first acquisitions build/look up exactly once.
        self._image = image
        self._resolved: modal.Image | None = None
        self._image_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        import modal

        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        resources = config.resources
        if resources is not None:
            resources._require_support("ModalRuntime", {"cpu", "memory_mb", "gpu"})
            if resources.gpu is not None and len(resources.gpu.acceptable_types) > 1:
                raise ValueError("ModalRuntime does not support alternative GPU types")
        project = config.compose
        compose = None
        if project is not None:
            compose = project.document
            if not isinstance(compose, Path):
                raise ValueError("ModalRuntime requires compose as a local file path")
            compose = compose.resolve()
        if compose is not None and resources is not None and resources.gpu is not None:
            raise ValueError(
                "ModalRuntime cannot attach GPUs to services inside Docker-in-Docker; "
                "use a materialized image or omit runtime_config.compose"
            )
        if compose is not None and self.secrets:
            raise ValueError(
                "ModalRuntime secrets require an image runtime; attaching them to "
                "the outer Docker-in-Docker sandbox would not expose them to main"
            )
        port_service = ComposeConfig.from_file(compose).network_owner("main") if compose else "main"
        if compose is not None:
            image = modal.Image.from_registry("docker:28.3.3-dind")
        elif config.image is not None:
            image = (
                modal.Image.from_id(config.image.removeprefix("modal://"))
                if config.image.startswith("modal://")
                else modal.Image.from_registry(
                    config.image, secret=self.registry_secret
                )  # pull auth
            )
        elif self.image_name is not None:
            image = modal.Image.from_name(self.image_name)
        elif self._image is None:
            raise ValueError(
                "ModalRuntime requires image=, image_name=, runtime_config.image, "
                "or runtime_config.compose"
            )
        else:
            image = self._resolved

        app = self.app
        if app is None:
            app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        if image is None:
            assert self._image is not None
            if self._resolved is None:
                async with self._image_lock:
                    if self._resolved is None:
                        await self._image.build.aio(app=app)
                        self._resolved = self._image
            image = self._image

        sandbox_kwargs: dict[str, Any] = {}
        if compose is not None:
            sandbox_kwargs["cpu"] = max(
                resources.cpu if resources is not None and resources.cpu is not None else 0,
                _MODAL_COMPOSE_CPU,
            )
            sandbox_kwargs["memory"] = max(
                (
                    resources.memory_mb
                    if resources is not None and resources.memory_mb is not None
                    else 0
                ),
                _MODAL_COMPOSE_MEMORY_MB,
            )
        else:
            if resources is not None and resources.cpu is not None:
                sandbox_kwargs["cpu"] = resources.cpu
            if resources is not None and resources.memory_mb is not None:
                sandbox_kwargs["memory"] = resources.memory_mb
            if self.env_vars:
                sandbox_kwargs["env"] = self.env_vars
            if self.secrets:
                sandbox_kwargs["secrets"] = self.secrets
        if resources is not None and resources.gpu is not None:
            gpu_types = resources.gpu.acceptable_types
            gpu_type = gpu_types[0] if gpu_types else "any"
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
        output_tasks: tuple[asyncio.Task[None], ...] = ()
        if compose is None:
            output_tasks = (
                asyncio.create_task(stream_output(sb.stdout, sys.stdout)),
                asyncio.create_task(stream_output(sb.stderr, sys.stderr)),
            )
        compose_path: str | None = None
        try:
            if project is None:
                await sb.wait_until_ready.aio(timeout=ready_timeout)
            else:
                assert compose is not None
                project_root = project.root if isinstance(project.root, Path) else compose.parent
                compose_path = str(
                    PurePosixPath("/hud/project")
                    / compose.relative_to(project_root.resolve()).as_posix()
                )
                project_directory = str(PurePosixPath(compose_path).parent)
                with project.stage(
                    f"{self.port}:{self.port}",
                    port_service=port_service,
                    seccomp="/hud/docker-seccomp.json",
                    service_socket=("/var/run/docker.sock" if project.service_access else None),
                    env_vars=self.env_vars,
                    cpu=resources.cpu if resources is not None else None,
                    memory_mb=resources.memory_mb if resources is not None else None,
                    gpu_count=(
                        resources.gpu.count
                        if resources is not None and resources.gpu is not None
                        else None
                    ),
                    archive=True,
                ) as files:
                    assert files.archive is not None
                    await sb.filesystem.copy_from_local.aio(files.archive, "/hud/project.tar.gz")
                    await sb.filesystem.copy_from_local.aio(files.override, "/hud/override.json")
                    await sb.filesystem.copy_from_local.aio(files.ports, "/hud/ports.yaml")
                    await sb.filesystem.copy_from_local.aio(
                        _DOCKER_SECCOMP_PROFILE, "/hud/docker-seccomp.json"
                    )
                command = (
                    "mkdir -p /hud/project && "
                    "tar -xzf /hud/project.tar.gz -C /hud/project && "
                    "until docker info >/dev/null 2>&1; do sleep 1; done && "
                    "BUILD_FLAG=--build && "
                    "if [ -f /hud/project/build.sh ]; then "
                    "sh /hud/project/build.sh && BUILD_FLAG=--no-build; fi && "
                    f"docker compose --project-directory {shlex.quote(project_directory)} "
                    f"--file {shlex.quote(compose_path)} "
                    "--file /hud/override.json --file /hud/ports.yaml "
                    'up --detach "$BUILD_FLAG" --remove-orphans'
                )
                try:
                    async with asyncio.timeout(ready_timeout):
                        process = await sb.exec.aio("sh", "-c", command, timeout=ready_timeout)
                        returncode = await process.wait.aio()
                except TimeoutError:
                    raise TimeoutError(
                        f"Modal Compose startup timed out after {ready_timeout} seconds"
                    ) from None
                if returncode != 0:
                    error = (await process.stderr.read.aio()).strip()
                    raise RuntimeError(f"Modal Compose startup failed: {error}")
                logs = await sb.exec.aio(
                    "docker",
                    "compose",
                    "--project-directory",
                    project_directory,
                    "--file",
                    compose_path,
                    "--file",
                    "/hud/override.json",
                    "--file",
                    "/hud/ports.yaml",
                    "logs",
                    "--follow",
                    "--no-color",
                )
                output_tasks = (
                    asyncio.create_task(stream_output(logs.stdout, sys.stdout)),
                    asyncio.create_task(stream_output(logs.stderr, sys.stderr)),
                )
            host, port = (await sb.tunnels.aio())[self.port].tcp_socket
            yield ModalEndpoint(
                url=f"tcp://{host}:{port}",
                params={
                    "provider": "modal",
                    "instance_id": sb.object_id,
                    **({"ready_timeout": ready_timeout} if compose is not None else {}),
                },
                config=config if config.model_dump(exclude_none=True) else None,
                sandbox=sb,
                compose=compose_path,
            )
        finally:
            # check-free teardown: never shadow the run's own error.
            if compose_path is not None:
                with contextlib.suppress(Exception):
                    process = await sb.exec.aio(
                        "docker",
                        "compose",
                        "--project-directory",
                        str(PurePosixPath(compose_path).parent),
                        "--file",
                        compose_path,
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
            await finish_output(*output_tasks)

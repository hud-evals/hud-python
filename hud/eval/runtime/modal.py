"""Modal runtime provider."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from .compose import ComposeConfig, ComposeProject
from .core import Runtime, RuntimeConfig
from .docker import _DOCKER_SECCOMP_PROFILE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from pathlib import Path

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

T_co = TypeVar("T_co", covariant=True)


class AioMethod(Protocol[T_co]):
    async def aio(self, *args: object, **kwargs: object) -> T_co: ...


class ModalImage(Protocol):
    build: AioMethod[None]

    def env(self, variables: Mapping[str, str]) -> ModalImage: ...


class _ModalImageFactory(Protocol):
    def from_id(self, image_id: str) -> ModalImage: ...

    def from_registry(self, image: str) -> ModalImage: ...

    def from_name(self, name: str) -> ModalImage: ...


class _ModalAppFactory(Protocol):
    lookup: AioMethod[object]


class _ModalStream(Protocol):
    read: AioMethod[str]


class _ModalProcess(Protocol):
    wait: AioMethod[int]
    stderr: _ModalStream


class _ModalFilesystem(Protocol):
    copy_from_local: AioMethod[None]
    copy_to_local: AioMethod[None]


class _ModalTunnel(Protocol):
    tcp_socket: tuple[str, int]


class ModalSandbox(Protocol):
    object_id: str
    wait_until_ready: AioMethod[None]
    filesystem: _ModalFilesystem
    exec: AioMethod[_ModalProcess]
    tunnels: AioMethod[dict[int, _ModalTunnel]]
    terminate: AioMethod[None]


class _ModalSandboxFactory(Protocol):
    create: AioMethod[ModalSandbox]


class _ModalProbeFactory(Protocol):
    def with_tcp(self, port: int) -> object: ...


class ModalModule(Protocol):
    Image: _ModalImageFactory
    App: _ModalAppFactory
    Sandbox: _ModalSandboxFactory
    Probe: _ModalProbeFactory


_MODAL_COMPOSE_CPU = 4.0
_MODAL_COMPOSE_MEMORY_MB = 8192


def _modal_image_from_uri(modal: ModalModule, image_uri: str) -> ModalImage:
    modal_uri_prefix = "modal://"
    if image_uri.startswith(modal_uri_prefix):
        return modal.Image.from_id(image_uri.removeprefix(modal_uri_prefix))
    return modal.Image.from_registry(image_uri)


@dataclass(frozen=True, slots=True)
class _ModalHandoff:
    sandbox: ModalSandbox
    compose: str | None

    def _container(self) -> str:
        if self.compose is None:
            return ""
        compose = shlex.quote(self.compose)
        return (
            "CONTAINER=$(docker compose --project-directory /hud/project "
            f"--file /hud/project/{compose} --file /hud/override.json "
            "--file /hud/ports.yaml ps --quiet main); "
        )

    async def _exec(self, command: str) -> None:
        process = await self.sandbox.exec.aio("sh", "-c", command)
        if await process.wait.aio() != 0:
            raise RuntimeError((await process.stderr.read.aio()).strip())

    async def prepare(self) -> None:
        if self.compose is None:
            await self._exec("mkdir -p /media/hud/handoffs")
        else:
            await self._exec(
                self._container() + 'test -n "$CONTAINER" && docker exec "$CONTAINER" '
                "mkdir -p /media/hud/handoffs"
            )

    async def export_to(self, destination: Path) -> None:
        if self.compose is None:
            root = "/media/hud/handoffs"
            command = ""
        else:
            root = "/media/hud/handoff-export"
            command = (
                self._container()
                + f"rm -rf {root} && mkdir -p {root} && "
                + f'docker cp "$CONTAINER":/media/hud/handoffs/. {root} && '
            )
        command += (
            f"if find {root} -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; "
            "then echo 'runtime handoff contains an unsupported entry' >&2; exit 1; fi; "
            f"tar -czf /media/hud/handoff.tar.gz -C {root} ."
        )
        await self._exec(command)
        await self.sandbox.filesystem.copy_to_local.aio("/media/hud/handoff.tar.gz", destination)

    async def import_from(self, source: Path) -> None:
        await self.sandbox.filesystem.copy_from_local.aio(source, "/media/hud/handoff.tar.gz")
        if self.compose is None:
            command = (
                "mkdir -p /media/hud/handoffs && "
                "tar -xzf /media/hud/handoff.tar.gz -C /media/hud/handoffs"
            )
        else:
            command = (
                self._container()
                + "rm -rf /tmp/hud-handoff && mkdir -p /tmp/hud-handoff && "
                + "tar -xzf /media/hud/handoff.tar.gz -C /tmp/hud-handoff && "
                + 'docker exec "$CONTAINER" mkdir -p /media/hud/handoffs && '
                + 'docker cp /tmp/hud-handoff/. "$CONTAINER":/media/hud/handoffs'
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
        image: ModalImage | None = None,
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
        self._resolved: ModalImage | None = None
        self._image_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        resources = config.resources
        if resources is not None:
            resources._require_support("ModalRuntime", {"cpu", "memory_mb", "gpu"})
        compose_source = config.compose_source()
        compose = (
            compose_source.runnable_path("ModalRuntime") if compose_source is not None else None
        )
        if compose is not None and resources is not None and resources.gpu is not None:
            raise ValueError(
                "ModalRuntime cannot attach GPUs to services inside Docker-in-Docker; "
                "use a materialized image or omit runtime_config.compose"
            )
        port_service = ComposeConfig.from_file(compose).network_owner("main") if compose else "main"
        modal = cast("ModalModule", importlib.import_module("modal"))

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
        if app is None:
            app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)

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
        try:
            if compose is None:
                await sb.wait_until_ready.aio(timeout=ready_timeout)
            else:
                project = ComposeProject(compose)
                with project.stage(
                    f"{self.port}:{self.port}",
                    port_service=port_service,
                    seccomp="/hud/docker-seccomp.json",
                    service_socket=(
                        "/var/run/docker.sock" if config.compose_service_access else None
                    ),
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
                    "docker compose --project-directory /hud/project "
                    f"--file /hud/project/{shlex.quote(compose.name)} "
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
            host, port = (await sb.tunnels.aio())[self.port].tcp_socket
            handoff = _ModalHandoff(sb, compose.name if compose is not None else None)
            await handoff.prepare()
            yield Runtime(
                f"tcp://{host}:{port}",
                params={
                    "provider": "modal",
                    "instance_id": sb.object_id,
                    **({"ready_timeout": ready_timeout} if compose is not None else {}),
                },
                config=config if config.model_dump(exclude_none=True) else None,
                handoff=handoff,
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

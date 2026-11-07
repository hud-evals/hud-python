"""Modal runtime provider."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import shlex
import sys
import tarfile
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import websockets

from hud.utils.process import finish_output, stream_output

from .compose import (
    DOCKER_DAEMON_IMAGE,
    DOCKER_DAEMON_PORT,
    DOCKER_DAEMON_SERVICE,
    DOCKER_HOST,
    ComposeConfig,
    ComposeHealthcheck,
    ComposeService,
)
from .core import (
    SESSION_ROOT,
    Checkpoint,
    CheckpointScope,
    ExecResult,
    RootScope,
    Runtime,
    RuntimeConfig,
    SessionScope,
    splice_websocket,
    validate_archive,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    import modal
    from modal.sandbox import SidecarContainer

    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

_BRIDGE_SOURCE = Path(__file__).with_name("modal_bridge.py").read_text(encoding="utf-8")
#: The sandbox port the connect token routes to; the bridge listens here and
#: splices each WebSocket to the environment's own port on loopback.
_BRIDGE_PORT = 18765
_START_FILE = "/hud/start"
_ARCHIVE = "/tmp/hud-transfer.tar.gz"  # noqa: S108 - inside the sandbox, every image has /tmp
#: Memory the VM runtime reserves for a sidecar that declares no limit, and for
#: the Docker daemon; the sandbox's own request does not cover sidecars there.
_SIDECAR_MEMORY_MIB = 512
_DOCKER_DAEMON_MEMORY_MIB = 1024
_VM_MAIN_MEMORY_MIB = 1024
_RESTART_POLL_SECONDS = 2.0
_CONDITIONS = frozenset({"service_started", "service_healthy", "service_completed_successfully"})
_RESTART_POLICIES = frozenset({"no", "always", "unless-stopped", "on-failure"})

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|us|ns|h|m|s)")
_BYTES = re.compile(r"^(\d+)([kmg]?)b?$", re.IGNORECASE)


def _seconds(text: str | None, default: float) -> float:
    """A Compose duration (``1m30s``, ``500ms``) in seconds."""
    if text is None:
        return default
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    parts = _DURATION.findall(text)
    if not parts or "".join(number + unit for number, unit in parts) != text:
        raise ValueError(f"invalid Compose duration {text!r}")
    return sum(float(number) * units[unit] for number, unit in parts)


def _mebibytes(value: str | int) -> int:
    """A Compose byte value (``512m``, ``2g``, ``1073741824``) in whole MiB, rounded up."""
    if isinstance(value, int):
        return math.ceil(value / 2**20)
    match = _BYTES.match(value.strip())
    if match is None:
        raise ValueError(f"invalid Compose memory value {value!r}")
    scale = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30}[match.group(2).lower()]
    return math.ceil(int(match.group(1)) * scale / 2**20)


@dataclass(frozen=True, slots=True)
class _Service:
    """One runnable Compose service as the sandbox runs it."""

    name: str
    image: str
    argv: tuple[str, ...]
    environment: dict[str, str]
    workdir: str | None
    #: ``depends_on`` as {service: condition}.
    dependencies: dict[str, str]
    healthcheck: ComposeHealthcheck | None
    restart: str
    memory_mib: int | None


@dataclass(frozen=True, slots=True)
class _Plan:
    """A Compose project laid out as one sandbox: ``main`` is its main container, the
    sidecars run in dependency order, and ``docker`` adds the daemon sidecar."""

    main: _Service
    sidecars: tuple[_Service, ...]
    docker: bool


def _healthcheck(service: ComposeService) -> ComposeHealthcheck | None:
    check = service.healthcheck
    if check is None or check.disable is True or check.test in (None, ["NONE"]):
        return None
    return check


def _healthcheck_argv(check: ComposeHealthcheck) -> list[str]:
    assert check.test is not None
    kind, *rest = check.test
    if kind == "CMD":
        return rest
    if kind == "CMD-SHELL":
        return ["sh", "-c", *rest]
    raise ValueError(f"unsupported Compose healthcheck test {check.test!r}")


def _plan(compose: ComposeConfig, *, docker: bool, gpu: bool) -> _Plan:
    """Lay the project out as a sandbox; each refusal names what the sandbox cannot do."""
    runnable = {
        name: service
        for name, service in compose.services.items()
        if not (service.model_extra or {}).get("profiles")
        and (service.model_extra or {}).get("scale") != 0
    }
    if "main" not in runnable:
        raise ValueError("ModalRuntime requires a Compose service named 'main'")
    if docker and gpu:
        raise ValueError(
            "ModalRuntime cannot give a GPU environment a Docker daemon: the daemon needs "
            "the VM runtime, which has no GPUs"
        )
    if DOCKER_DAEMON_SERVICE in runnable:
        raise ValueError(
            f"Compose service name {DOCKER_DAEMON_SERVICE!r} is reserved for the Docker daemon"
        )
    services: dict[str, _Service] = {}
    for name, service in runnable.items():
        extra = service.model_extra or {}
        if service.image is None:
            raise ValueError(
                f"ModalRuntime runs pre-built images only; Compose service {name!r} has no image"
            )
        if service.network_mode is not None:
            raise ValueError(
                f"ModalRuntime cannot apply network_mode to Compose service {name!r}; "
                "services share the sandbox's network and reach each other by name"
            )
        if service.user is not None:
            raise ValueError(
                f"ModalRuntime cannot run Compose service {name!r} as user {service.user!r}"
            )
        deploy = extra.get("deploy") or {}
        devices = deploy.get("resources", {}).get("reservations", {}).get("devices")
        if name != "main" and (devices or extra.get("gpus")):
            raise ValueError(
                f"ModalRuntime cannot attach a GPU to Compose service {name!r}: "
                "only the environment (main) runs on the sandbox's GPU; sidecars are CPU-only"
            )
        raw_dependencies = extra.get("depends_on") or {}
        dependencies: dict[str, str] = (
            {str(dependency): "service_started" for dependency in raw_dependencies}
            if isinstance(raw_dependencies, list)
            else {
                str(dependency): str((spec or {}).get("condition", "service_started"))
                for dependency, spec in raw_dependencies.items()
            }
        )
        for dependency, condition in dependencies.items():
            if dependency not in runnable:
                raise ValueError(
                    f"Compose service {name!r} depends on {dependency!r}, which does not run"
                )
            if condition not in _CONDITIONS:
                raise ValueError(f"unsupported depends_on condition {condition!r}")
            if condition == "service_healthy" and _healthcheck(runnable[dependency]) is None:
                raise ValueError(
                    f"Compose service {name!r} waits for {dependency!r} to be healthy, "
                    "which declares no healthcheck"
                )
        restart = str(extra.get("restart") or "no")
        if restart.partition(":")[0] not in _RESTART_POLICIES:
            raise ValueError(f"unsupported Compose restart policy {restart!r}")
        memory = extra.get("mem_limit") or deploy.get("resources", {}).get("limits", {}).get(
            "memory"
        )
        services[name] = _Service(
            name=name,
            image=service.image,
            argv=(
                (*(service.entrypoint or ()), *service.command)
                if service.command is not None
                else ()
            ),
            environment=dict(service.environment),
            workdir=service.working_dir,
            dependencies=dependencies,
            healthcheck=_healthcheck(service),
            restart=restart,
            memory_mib=_mebibytes(memory) if memory is not None else None,
        )
    graph = {
        name: set(service.dependencies) - {"main"}
        for name, service in services.items()
        if name != "main"
    }
    try:
        ordered = list(TopologicalSorter(graph).static_order())
    except CycleError:
        raise ValueError("Compose depends_on forms a cycle") from None
    return _Plan(
        main=services["main"],
        sidecars=tuple(services[name] for name in ordered),
        docker=docker,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalEndpoint(Runtime):
    transports_sessions: ClassVar[bool] = True

    sandbox: modal.Sandbox
    #: Live sidecar handles by Compose service name; a restart replaces one.
    sidecars: dict[str, SidecarContainer] = field(default_factory=dict)
    #: Whether the sandbox runs on Modal's VM runtime, whose filesystem
    #: snapshots are faithful; the default gVisor runtime's are not.
    vm: bool = False

    def _target(self, service: str) -> modal.Sandbox | SidecarContainer:
        if service == "main":
            return self.sandbox
        try:
            return self.sidecars[service]
        except KeyError:
            raise ValueError(f"Compose service {service!r} is not running") from None

    async def terminated(self) -> str | None:
        code = await self.sandbox.poll.aio()
        return None if code is None else f"sandbox exited with exit code {code}"

    async def exec(self, argv: Sequence[str], *, service: str = "main") -> ExecResult:
        process = await self._target(service).exec.aio(*argv)
        returncode = await process.wait.aio()
        return ExecResult(
            returncode, await process.stdout.read.aio(), await process.stderr.read.aio()
        )

    async def _sh(self, command: str, *, service: str = "main") -> str:
        result = await self.exec(["sh", "-c", command], service=service)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    async def collect(
        self,
        paths: Sequence[str | PurePosixPath],
        destination: Path,
        *,
        service: str = "main",
    ) -> list[Path]:
        target = self._target(service)
        collected: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="hud-collect-") as directory:
            archive = Path(directory) / "collect.tar.gz"
            for path in paths:
                source = PurePosixPath(path)
                parent, name = shlex.quote(str(source.parent)), shlex.quote(source.name)
                await self._sh(f"tar -czf {_ARCHIVE} -C {parent} {name}", service=service)
                await target.filesystem.copy_to_local.aio(_ARCHIVE, archive)
                with tarfile.open(archive) as tar:
                    tar.extractall(destination, filter="data")
                collected.append(destination / source.name)
        return collected

    @asynccontextmanager
    async def checkpoint(self, scope: CheckpointScope) -> AsyncIterator[Checkpoint]:
        if isinstance(scope, RootScope):
            if not self.vm:
                raise self._unsupported(
                    "checkpoint the root filesystem",
                    "Modal's default runtime strips setuid, setgid and sticky bits from "
                    "filesystem snapshots; only the VM runtime, which an environment with "
                    "a Docker daemon runs on, snapshots faithfully",
                )
            image = await self.sandbox.snapshot_filesystem.aio()
            yield Checkpoint(scope, f"modal://{image.object_id}")
            return
        with tempfile.TemporaryDirectory(prefix="hud-checkpoint-") as directory:
            destination = Path(directory) / "checkpoint.tar.gz"
            root = shlex.quote(str(scope.path))
            await self._sh(
                (f"mkdir -p {root}; " if isinstance(scope, SessionScope) else "")
                + f"if [ ! -d {root} ]; then echo '{root} is not a directory' >&2; exit 1; fi; "
                + f"if find {root} -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; "
                + "then echo 'runtime archive contains an unsupported entry' >&2; exit 1; fi; "
                + f"tar -czf {_ARCHIVE} -C {root} ."
            )
            await self.sandbox.filesystem.copy_to_local.aio(_ARCHIVE, destination)
            yield Checkpoint(scope, str(destination))

    async def restore(self, checkpoint: Checkpoint, scope: CheckpointScope) -> None:
        if isinstance(scope, RootScope) or isinstance(checkpoint.scope, RootScope):
            raise self._unsupported(
                "restore a root checkpoint into a running sandbox",
                f"start a new sandbox from runtime_config.image={checkpoint.ref!r}",
            )
        if type(checkpoint.scope) is not type(scope):
            raise ValueError("a checkpoint restores into a scope of its own kind")
        validate_archive(Path(checkpoint.ref))
        target = shlex.quote(str(scope.path))
        await self.sandbox.filesystem.copy_from_local.aio(checkpoint.ref, _ARCHIVE)
        await self._sh(f"rm -rf {target} && mkdir -p {target} && tar -xzf {_ARCHIVE} -C {target}")


class ModalRuntime:
    """The Modal provider: each acquisition ``Sandbox.create``s a fresh sandbox.

    The environment is the sandbox's main container, where a requested GPU
    attaches. A Compose project's other services run as sidecars of the same
    sandbox, reached by service name; the runtime starts them in ``depends_on``
    order, waits on their healthchecks, and restarts them by their ``restart``
    policy. Every service image must be pre-built: a registry reference, or
    ``modal://<image id>``. A project with ``service_access`` gets a Docker
    daemon sidecar, which needs Modal's VM runtime and so excludes a GPU.

    The control channel is reached through a connect-token WebSocket on the
    sandbox: a local listener mints a token for every connection and splices
    it to the environment's port. Sandboxes run in the caller's own region
    (``MODAL_REGION``) unless ``region`` says otherwise, carry ``tags``, and
    treat declared ``cpu`` and ``memory_mb`` as hard limits. Acquisitions are
    independent, so a batch fans out into isolated sandboxes.

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
        sandbox_secrets: Sequence[modal.Secret] | None = None,
        registry_secret: modal.Secret | None = None,
        region: str | None = None,
        tags: Mapping[str, str] | None = None,
        volumes: Mapping[str, modal.Volume] | None = None,
    ) -> None:
        if app is not None and app_name is not None:
            raise ValueError("ModalRuntime accepts either app or app_name, not both")
        self.image_name = image_name
        self.port = port
        self.env_vars = dict(env_vars or {})
        self.sandbox_secrets = tuple(sandbox_secrets or ())
        self.registry_secret = registry_secret
        self.workdir = workdir
        self.region = region if region is not None else os.environ.get("MODAL_REGION")
        self.tags = dict(tags or {})
        self.volumes = dict(volumes or {})
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
                "0.0.0.0",  # noqa: S104 - serving inside the sandbox; the bridge is the only ingress
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
        # A sidecar needs its image's object id, so each reference hydrates once.
        self._built: dict[str, modal.Image] = {}

    def _reference(self, reference: str) -> modal.Image:
        import modal

        if reference.startswith("modal://"):
            return modal.Image.from_id(reference.removeprefix("modal://"))
        return modal.Image.from_registry(reference, secret=self.registry_secret)

    async def _built_image(self, reference: str, app: modal.App) -> modal.Image:
        async with self._image_lock:
            if reference not in self._built:
                image = self._reference(reference)
                await image.build.aio(app=app)
                self._built[reference] = image
            return self._built[reference]

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        import modal

        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        resources = config.resources
        if resources is not None:
            resources._require_support("ModalRuntime", {"cpu", "memory_mb", "gpu"})
            if resources.gpu is not None and len(resources.gpu.acceptable_types) > 1:
                raise ValueError("ModalRuntime does not support alternative GPU types")
        gpu = resources.gpu if resources is not None else None
        plan: _Plan | None = None
        if config.compose is not None:
            document = config.compose.document
            compose = ComposeConfig.from_file(document) if isinstance(document, Path) else document
            plan = _plan(compose, docker=bool(config.compose.service_access), gpu=gpu is not None)
        elif config.image is None and self.image_name is None and self._image is None:
            raise ValueError(
                "ModalRuntime requires image=, image_name=, runtime_config.image, "
                "or runtime_config.compose"
            )

        app = self.app
        if app is None:
            app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        if plan is not None:
            image = self._reference(plan.main.image)
        elif config.image is not None:
            image = self._reference(config.image)
        elif self.image_name is not None:
            image = modal.Image.from_name(self.image_name)
        else:
            assert self._image is not None
            if self._resolved is None:
                async with self._image_lock:
                    if self._resolved is None:
                        await self._image.build.aio(app=app)
                        self._resolved = self._image
            image = self._image
        sidecar_images = {
            service.name: await self._built_image(service.image, app)
            for service in (plan.sidecars if plan is not None else ())
        }
        docker_image = (
            await self._built_image(DOCKER_DAEMON_IMAGE, app)
            if plan is not None and plan.docker
            else None
        )

        vm = plan is not None and plan.docker
        sandbox_kwargs: dict[str, Any] = {}
        if resources is not None and resources.cpu is not None:
            sandbox_kwargs["cpu"] = (resources.cpu, resources.cpu)
        memory = resources.memory_mb if resources is not None else None
        if plan is not None and plan.docker:
            reserve = _DOCKER_DAEMON_MEMORY_MIB + sum(
                service.memory_mib if service.memory_mib is not None else _SIDECAR_MEMORY_MIB
                for service in plan.sidecars
            )
            total = (memory if memory is not None else _VM_MAIN_MEMORY_MIB) + reserve
            sandbox_kwargs["memory"] = (total, total)
            sandbox_kwargs["runtime"] = "vm"
            sandbox_kwargs["experimental_options"] = {"vm_sidecar_memory_reserve_mib": reserve}
        elif memory is not None:
            sandbox_kwargs["memory"] = (memory, memory)
        if gpu is not None:
            gpu_type = gpu.acceptable_types[0] if gpu.acceptable_types else "any"
            sandbox_kwargs["gpu"] = gpu_type if gpu.count == 1 else f"{gpu_type}:{gpu.count}"
        if self.region is not None:
            sandbox_kwargs["region"] = self.region
        if self.tags:
            sandbox_kwargs["tags"] = self.tags
        if self.sandbox_secrets:
            sandbox_kwargs["secrets"] = self.sandbox_secrets
        if self.volumes:
            sandbox_kwargs["volumes"] = self.volumes

        # Without a declared run limit, the sandbox outlives the task's agent budget by the
        # default allowance for startup, grading, and teardown.
        agent_timeout = (task.agent_config or {}).get("timeout_seconds")
        run_timeout = 3600 + (math.ceil(agent_timeout) if agent_timeout is not None else 0)
        ready_timeout = 600
        if config.limits is not None:
            run_timeout = config.limits.run_timeout_s or run_timeout
            ready_timeout = config.limits.startup_timeout_s or ready_timeout

        environment = dict(self.env_vars)
        workdir = self.workdir
        argv: tuple[str, ...] = self.command
        held_back = False
        if plan is not None:
            environment.update(plan.main.environment)
            if plan.docker:
                environment["DOCKER_HOST"] = DOCKER_HOST
            if plan.main.workdir is not None:
                workdir = plan.main.workdir
            if plan.main.argv:
                argv = plan.main.argv
            # The environment starts once what it depends on is up, as under Compose.
            held_back = bool(plan.main.dependencies) or plan.docker
        command = (
            "sh",
            "-c",
            f"mkdir -p {SESSION_ROOT} /media/hud /hud && "
            "rm -rf /media/hud/sessions && "
            f"ln -s {SESSION_ROOT} /media/hud/sessions && "
            'exec "$@"',
            "hud-runtime",
            "python3",
            "-c",
            _BRIDGE_SOURCE,
            "--listen",
            str(_BRIDGE_PORT),
            "--upstream",
            str(self.port),
            *(("--start-file", _START_FILE) if held_back else ()),
            "--",
            *argv,
        )
        sb = await modal.Sandbox.create.aio(
            *command,
            app=app,
            image=image,
            env=environment or None,
            workdir=workdir,
            readiness_probe=modal.Probe.with_tcp(_BRIDGE_PORT),
            # Modal types both timeouts as int seconds; floats raise at proto encode.
            timeout=run_timeout,
            **sandbox_kwargs,
        )
        output_tasks = (
            asyncio.create_task(stream_output(sb.stdout, sys.stdout)),
            asyncio.create_task(stream_output(sb.stderr, sys.stderr)),
        )
        sidecars: dict[str, SidecarContainer] = {}
        watcher: asyncio.Task[None] | None = None
        server: asyncio.Server | None = None
        try:
            try:
                async with asyncio.timeout(ready_timeout):
                    if plan is not None:
                        await _start_sidecars(sb, sidecars, plan, sidecar_images, docker_image, vm)
                        if held_back:
                            process = await sb.exec.aio("touch", _START_FILE)
                            await process.wait.aio()
                    await sb.wait_until_ready.aio(timeout=ready_timeout)
            except TimeoutError:
                raise TimeoutError(
                    f"Modal sandbox startup timed out after {ready_timeout} seconds"
                ) from None
            if plan is not None:
                watcher = asyncio.create_task(
                    _restart_exited(sb, sidecars, plan, sidecar_images, vm)
                )
            server = await asyncio.start_server(
                lambda reader, writer: _forward(sb, reader, writer), "127.0.0.1", 0
            )
            port = server.sockets[0].getsockname()[1]
            yield ModalEndpoint(
                url=f"tcp://127.0.0.1:{port}",
                params={
                    "provider": "modal",
                    "instance_id": sb.object_id,
                    "ready_timeout": ready_timeout,
                },
                config=config if config.model_dump(exclude_none=True) else None,
                sandbox=sb,
                sidecars=sidecars,
                vm=vm,
            )
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            if server is not None:
                server.close()
                await server.wait_closed()
            # check-free teardown: never shadow the run's own error.
            with contextlib.suppress(Exception):
                await sb.terminate.aio()
            await finish_output(*output_tasks)


async def _create_sidecar(
    sb: modal.Sandbox, service: _Service, image: modal.Image, *, vm: bool
) -> SidecarContainer:
    memory = service.memory_mib if service.memory_mib is not None else _SIDECAR_MEMORY_MIB
    return await sb._experimental_sidecars.create.aio(
        *service.argv,
        name=service.name,
        image=image,
        env=service.environment or None,
        workdir=service.workdir,
        experimental_memory_reserve_consume_mib=memory if vm else None,
    )


async def _await_condition(
    sidecars: Mapping[str, SidecarContainer], plan: _Plan, dependency: str, condition: str
) -> None:
    handle = sidecars[dependency]
    service = next(service for service in plan.sidecars if service.name == dependency)
    if condition == "service_healthy":
        assert service.healthcheck is not None
        await _healthy(handle, dependency, service.healthcheck)
    elif condition == "service_completed_successfully":
        await handle.wait.aio(raise_on_termination=False)
        code = await handle.poll.aio()
        if code != 0:
            raise RuntimeError(f"Compose service {dependency!r} exited with code {code}")


async def _start_sidecars(
    sb: modal.Sandbox,
    sidecars: dict[str, SidecarContainer],
    plan: _Plan,
    images: Mapping[str, modal.Image],
    docker_image: modal.Image | None,
    vm: bool,
) -> None:
    """Start the daemon and the sidecars in dependency order, then, like
    ``docker compose up --wait``, wait for every healthchecked service."""
    if docker_image is not None:
        daemon = await sb._experimental_sidecars.create.aio(
            name=DOCKER_DAEMON_SERVICE,
            image=docker_image,
            env={"DOCKER_TLS_CERTDIR": ""},
            experimental_memory_reserve_consume_mib=_DOCKER_DAEMON_MEMORY_MIB,
        )
        sidecars[DOCKER_DAEMON_SERVICE] = daemon
        await _healthy(
            daemon,
            DOCKER_DAEMON_SERVICE,
            ComposeHealthcheck(
                test=["CMD", "docker", "-H", f"tcp://127.0.0.1:{DOCKER_DAEMON_PORT}", "info"],
                interval="2s",
                timeout="5s",
                retries=30,
            ),
        )
    for service in plan.sidecars:
        for dependency, condition in service.dependencies.items():
            if dependency != "main":
                await _await_condition(sidecars, plan, dependency, condition)
        sidecars[service.name] = await _create_sidecar(sb, service, images[service.name], vm=vm)
    for dependency, condition in plan.main.dependencies.items():
        await _await_condition(sidecars, plan, dependency, condition)
    for service in plan.sidecars:
        if service.healthcheck is not None:
            await _healthy(sidecars[service.name], service.name, service.healthcheck)


async def _restart_exited(
    sb: modal.Sandbox,
    sidecars: dict[str, SidecarContainer],
    plan: _Plan,
    images: Mapping[str, modal.Image],
    vm: bool,
) -> None:
    """Apply each sidecar's ``restart`` policy, as the Compose runtime would."""
    failures: dict[str, int] = {}
    while True:
        await asyncio.sleep(_RESTART_POLL_SECONDS)
        for service in plan.sidecars:
            code = await sidecars[service.name].poll.aio()
            if code is None:
                continue
            policy, _, limit = service.restart.partition(":")
            if policy == "no" or (policy == "on-failure" and code == 0):
                continue
            failures[service.name] = failures.get(service.name, 0) + 1
            if policy == "on-failure" and limit and failures[service.name] > int(limit):
                logger.warning(
                    "Compose service %r exited with code %s; its restart limit %s is reached",
                    service.name,
                    code,
                    limit,
                )
                continue
            logger.warning("Compose service %r exited with code %s; restarting", service.name, code)
            sidecars[service.name] = await _create_sidecar(sb, service, images[service.name], vm=vm)


async def _healthy(handle: SidecarContainer, name: str, check: ComposeHealthcheck) -> None:
    """Return once ``check`` passes in the sidecar; raise once it has failed ``retries``
    times in a row after ``start_period``."""
    argv = _healthcheck_argv(check)
    interval = _seconds(check.interval, 30.0)
    timeout = _seconds(check.timeout, 30.0)
    retries = check.retries if check.retries is not None else 3
    start_interval = _seconds(check.start_interval, 5.0)
    loop = asyncio.get_running_loop()
    grace_until = loop.time() + _seconds(check.start_period, 0.0)
    failures = 0
    while True:
        try:
            async with asyncio.timeout(timeout):
                process = await handle.exec.aio(*argv)
                code = await process.wait.aio()
        except TimeoutError:
            code = None
        if code == 0:
            return
        if loop.time() >= grace_until:
            failures += 1
            if failures >= retries:
                raise RuntimeError(
                    f"Compose service {name!r} is unhealthy: "
                    + ("its healthcheck timed out" if code is None else f"exit code {code}")
                )
        await asyncio.sleep(start_interval if loop.time() < grace_until else interval)


async def _forward(
    sb: modal.Sandbox, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Carry one local control connection to the sandbox over a fresh connect token."""
    try:
        credentials = await sb.create_connect_token.aio(port=_BRIDGE_PORT)
        parts = urlsplit(credentials.url)
        url = urlunsplit(
            ("wss", parts.netloc, "/", f"_modal_connect_token={credentials.token}", "")
        )
        async with websockets.connect(url, max_size=None) as websocket:
            await splice_websocket(reader, writer, websocket)
    except Exception as exc:
        # The local client retries until the environment answers; a sandbox
        # that is gone fails the rollout with the sandbox's own reason.
        logger.warning("modal control connection failed: %s", exc)
    finally:
        if not writer.is_closing():
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

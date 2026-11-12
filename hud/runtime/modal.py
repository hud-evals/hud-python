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
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar

from hud.environment.server import DEFAULT_PORT, default_serve_command
from hud.types import ClassifiedError
from hud.utils.process import finish_output, stream_output

from .compose import (
    DOCKER_DAEMON_IMAGE,
    DOCKER_DAEMON_PORT,
    DOCKER_DAEMON_SERVICE,
    DOCKER_HOST,
    ComposeConfig,
    ComposeHealthcheck,
)
from .core import (
    SESSION_ROOT,
    Checkpoint,
    ExecResult,
    Runtime,
    RuntimeConfig,
    SessionScope,
    validate_archive,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    import modal
    from modal.sandbox import SidecarContainer

    from hud.eval.task import Task

logger = logging.getLogger("hud.runtime")

# Created once the services ``main`` depends on are up; ``main`` waits for it.
_START_FILE = "/hud/start"
# Memory the VM runtime reserves for the Docker daemon sidecar; the sandbox's
# own request does not cover sidecars there.
_DOCKER_DAEMON_MEMORY_MIB = 1024

_BYTES = re.compile(r"^(\d+)([kmg]?)b?$", re.IGNORECASE)


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
    # ``depends_on`` as {service: condition}.
    dependencies: dict[str, str]
    healthcheck: ComposeHealthcheck | None
    memory_mib: int | None


@dataclass(frozen=True, slots=True)
class _Plan:
    """A Compose project laid out as one sandbox.

    ``main`` is its main container, the sidecars run beside it, ``order`` starts every
    service after what it depends on, and ``docker`` adds the daemon sidecar.
    """

    main: _Service
    sidecars: tuple[_Service, ...]
    order: tuple[str, ...]
    docker: bool

    def service(self, name: str) -> _Service:
        return self.main if name == "main" else next(s for s in self.sidecars if s.name == name)


def _healthcheck_argv(check: ComposeHealthcheck) -> list[str]:
    assert check.test is not None
    kind, *rest = check.test
    if kind == "CMD":
        return rest
    if kind == "CMD-SHELL":
        return ["sh", "-c", *rest]
    raise ValueError(f"unsupported Compose healthcheck test {check.test!r}")


def _plan(compose: ComposeConfig, *, docker: bool, gpu: bool, memory_mib: int | None) -> _Plan:
    """Lay the project out as a sandbox; each refusal names what the sandbox cannot do."""
    runnable = compose.started_services()
    healthchecks = {
        name: service.healthcheck
        for name, service in runnable.items()
        if service.healthcheck is not None and not service.healthcheck.disabled
    }
    if "main" not in runnable:
        raise ValueError("ModalRuntime requires a Compose service named 'main'")
    if docker and gpu:
        raise ValueError(
            "ModalRuntime cannot give a GPU environment a Docker daemon: the daemon needs "
            "the VM runtime, which has no GPUs"
        )
    if docker and memory_mib is None:
        raise ValueError(
            "ModalRuntime needs resources.memory_mb for an environment with a Docker daemon: "
            "the VM runtime sizes every container of the sandbox"
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
        if service.restart not in (None, "no"):
            logger.warning(
                "ModalRuntime does not restart Compose service %r (restart: %s): a sidecar "
                "that exits stays down",
                name,
                service.restart,
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
        dependencies: dict[str, str] = {
            dependency: spec.condition for dependency, spec in service.depends_on.items()
        }
        for dependency, condition in dependencies.items():
            if dependency not in runnable:
                raise ValueError(
                    f"Compose service {name!r} depends on {dependency!r}, which does not run"
                )
            if condition == "service_healthy" and dependency not in healthchecks:
                raise ValueError(
                    f"Compose service {name!r} waits for {dependency!r} to be healthy, "
                    "which declares no healthcheck"
                )
            if condition == "service_completed_successfully" and dependency == "main":
                raise ValueError(
                    f"Compose service {name!r} waits for 'main' to complete, "
                    "which serves for the whole rollout"
                )
        memory = extra.get("mem_limit") or deploy.get("resources", {}).get("limits", {}).get(
            "memory"
        )
        if docker and name != "main" and memory is None:
            raise ValueError(
                f"ModalRuntime needs mem_limit on Compose service {name!r}: with a Docker "
                "daemon the sandbox runs on the VM runtime, which sizes every container"
            )
        # As under Compose, an entrypoint override drops the image's command.
        argv = (
            (*service.entrypoint, *(service.command or ()))
            if service.entrypoint is not None
            else tuple(service.command or ())
        )
        services[name] = _Service(
            name=name,
            image=service.image,
            argv=argv,
            environment=dict(service.environment),
            workdir=service.working_dir,
            dependencies=dependencies,
            healthcheck=healthchecks.get(name),
            memory_mib=_mebibytes(memory) if memory is not None else None,
        )
    graph = {name: set(service.dependencies) for name, service in services.items()}
    try:
        ordered = tuple(TopologicalSorter(graph).static_order())
    except CycleError:
        raise ValueError("Compose depends_on forms a cycle") from None
    return _Plan(
        main=services["main"],
        sidecars=tuple(services[name] for name in ordered if name != "main"),
        order=ordered,
        docker=docker,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalEndpoint(Runtime):
    """The ``Runtime`` a ``ModalRuntime`` yields: its control channel and Modal ``sandbox``."""

    transports_sessions: ClassVar[bool] = True

    sandbox: modal.Sandbox
    # Sidecar handles by Compose service name.
    sidecars: dict[str, SidecarContainer] = field(default_factory=dict)

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

    async def exec(
        self, argv: Sequence[str], *, service: str = "main", max_wait: float | None = None
    ) -> ExecResult:
        target = self._target(service)
        async with asyncio.timeout(max_wait):
            process = await target.exec.aio(*argv)
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
                staged = _staging_path()
                parent, name = shlex.quote(str(source.parent)), shlex.quote(source.name)
                await self._sh(f"tar -czf {staged} -C {parent} {name}", service=service)
                try:
                    await target.filesystem.copy_to_local.aio(staged, archive)
                finally:
                    await self._sh(f"rm -f {staged}", service=service)
                with tarfile.open(archive) as tar:
                    tar.extractall(destination, filter="data")
                collected.append(destination / source.name)
        return collected

    async def put(self, source: Path, path: str | PurePosixPath, *, service: str = "main") -> None:
        target = PurePosixPath(path)
        staged = _staging_path()
        with tempfile.TemporaryDirectory(prefix="hud-put-") as directory:
            archive = Path(directory) / "put.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source, arcname=target.name)
            await self._target(service).filesystem.copy_from_local.aio(archive, staged)
        parent = shlex.quote(str(target.parent))
        await self._sh(
            f"rm -rf {shlex.quote(str(target))} && mkdir -p {parent} && "
            f"tar -xzf {staged} -C {parent}; status=$?; rm -f {staged}; exit $status",
            service=service,
        )

    @asynccontextmanager
    async def checkpoint(self, scope: SessionScope) -> AsyncGenerator[Checkpoint]:
        staged = _staging_path()
        with tempfile.TemporaryDirectory(prefix="hud-checkpoint-") as directory:
            destination = Path(directory) / "checkpoint.tar.gz"
            root = shlex.quote(str(scope.path))
            await self._sh(
                f"mkdir -p {root}; "
                f"if find {root} -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; "
                "then echo 'runtime archive contains an unsupported entry' >&2; exit 1; fi; "
                f"tar -czf {staged} -C {root} ."
            )
            try:
                await self.sandbox.filesystem.copy_to_local.aio(staged, destination)
            finally:
                await self._sh(f"rm -f {staged}")
            yield Checkpoint(scope, str(destination))

    async def restore(self, checkpoint: Checkpoint, scope: SessionScope) -> None:
        validate_archive(Path(checkpoint.ref))
        staged = _staging_path()
        target = shlex.quote(str(scope.path))
        await self.sandbox.filesystem.copy_from_local.aio(checkpoint.ref, staged)
        await self._sh(
            f"rm -rf {target} && mkdir -p {target} && tar -xzf {staged} -C {target}; "
            f"status=$?; rm -f {staged}; exit $status"
        )


def _staging_path() -> str:
    """A path inside the box for one transfer's archive; every image has ``/tmp``."""
    return f"/tmp/hud-transfer-{uuid.uuid4().hex}.tar.gz"  # noqa: S108 - inside the sandbox


class ModalRuntime:
    """The Modal provider: each acquisition ``Sandbox.create``s a fresh sandbox.

    The environment is the sandbox's main container, where a requested GPU
    attaches. A Compose project's other services run as sidecars of the same
    sandbox, reached by service name; the runtime starts them in ``depends_on``
    order and waits on their healthchecks. A sidecar that exits is not
    restarted. Every service image must be pre-built: a registry reference, or
    ``modal://<image id>``. A project with ``service_access`` gets a Docker
    daemon sidecar, which needs Modal's VM runtime and so excludes a GPU.

    The control channel is the environment's port behind a Modal encrypted
    port: the runtime's address is ``tls://<host>:<port>``, and whoever holds
    that address can connect, so it stays out of logs. Sandboxes run in the caller's own region
    (``MODAL_REGION``) unless ``region`` says otherwise, carry ``tags``, and
    treat declared ``cpu`` and ``memory_mb`` as hard limits. Acquisitions are
    independent, so a batch fans out into isolated sandboxes.

    ``image_name`` names a published image; an ``image`` is built once, on first use.
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
        port: int = DEFAULT_PORT,
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
        # An unset workdir keeps the image WORKDIR.
        self.command = (
            tuple(command) if command is not None else ("sh", "-c", default_serve_command(port))
        )
        self.app = app
        self.app_name = "hud-envs" if app_name is None else app_name
        config = None
        if runtime_config is not None:
            config = RuntimeConfig.model_validate(runtime_config)
        self.runtime_config = config
        # The lock makes concurrent first acquisitions build image exactly once.
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
    async def __call__(self, task: Task) -> AsyncGenerator[Runtime]:
        import modal

        config = (self.runtime_config or RuntimeConfig()).with_overrides(task.runtime_config)
        resources = config.resources
        if resources is not None:
            resources.require_support("ModalRuntime", {"cpu", "memory_mb", "gpu"})
            if resources.gpu is not None and len(resources.gpu.acceptable_types) > 1:
                raise ValueError("ModalRuntime does not support alternative GPU types")
        gpu = resources.gpu if resources is not None else None
        memory = resources.memory_mb if resources is not None else None
        plan: _Plan | None = None
        if config.compose is not None:
            document = config.compose.document
            compose = ComposeConfig.from_file(document) if isinstance(document, Path) else document
            plan = _plan(
                compose,
                docker=bool(config.compose.service_access),
                gpu=gpu is not None,
                memory_mib=memory,
            )
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
        if plan is not None and plan.docker:
            # The plan refused a project that leaves any of these undeclared.
            assert memory is not None
            reserve = _DOCKER_DAEMON_MEMORY_MIB + sum(
                service.memory_mib for service in plan.sidecars if service.memory_mib is not None
            )
            total = memory + reserve
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
        wait_for_start = f"until [ -e {_START_FILE} ]; do sleep 0.2; done && " if held_back else ""
        command = (
            "sh",
            "-c",
            f"mkdir -p {SESSION_ROOT} /media/hud /hud && "
            "rm -rf /media/hud/sessions && "
            f"ln -s {SESSION_ROOT} /media/hud/sessions && "
            f'{wait_for_start}exec "$@"',
            "hud-runtime",
            *argv,
        )
        sb = await modal.Sandbox.create.aio(
            *command,
            app=app,
            image=image,
            env=environment or None,
            workdir=workdir,
            encrypted_ports=[self.port],
            readiness_probe=modal.Probe.with_tcp(self.port),
            # Modal types both timeouts as int seconds; floats raise at proto encode.
            timeout=run_timeout,
            **sandbox_kwargs,
        )
        output_tasks = (
            asyncio.create_task(stream_output(sb.stdout, sys.stdout)),
            asyncio.create_task(stream_output(sb.stderr, sys.stderr)),
        )
        sidecars: dict[str, SidecarContainer] = {}
        try:
            try:
                async with asyncio.timeout(ready_timeout):
                    if plan is not None:
                        try:
                            await _start_services(
                                sb,
                                sidecars,
                                plan,
                                sidecar_images,
                                docker_image,
                                vm,
                                release_main=held_back,
                            )
                        except Exception as exc:
                            raise ClassifiedError(
                                str(exc), code="compose_execution_failed", source="user"
                            ) from exc
                    try:
                        await sb.wait_until_ready.aio(timeout=ready_timeout)
                    except Exception as exc:
                        raise ClassifiedError(
                            f"Modal sandbox did not become ready: {exc}",
                            code="container_start_failed",
                            source="user",
                        ) from exc
            except TimeoutError:
                raise ClassifiedError(
                    f"Modal sandbox startup timed out after {ready_timeout} seconds",
                    code="container_start_failed",
                    source="user",
                ) from None
            host, port = (await sb.tunnels.aio())[self.port].tls_socket
            yield ModalEndpoint(
                url=f"tls://{host}:{port}",
                params={
                    "provider": "modal",
                    "instance_id": sb.object_id,
                    "ready_timeout": ready_timeout,
                },
                config=config if config.model_dump(exclude_none=True) else None,
                sandbox=sb,
                sidecars=sidecars,
            )
        finally:
            # Best-effort teardown: never shadow the run's own error.
            with contextlib.suppress(Exception):
                await sb.terminate.aio()
            await finish_output(*output_tasks)


async def _create_sidecar(
    sb: modal.Sandbox, service: _Service, image: modal.Image, *, vm: bool
) -> SidecarContainer:
    return await sb._experimental_sidecars.create.aio(  # noqa: SLF001 - no public sidecar API
        *service.argv,
        name=service.name,
        image=image,
        env=service.environment or None,
        workdir=service.workdir,
        experimental_memory_reserve_consume_mib=service.memory_mib if vm else None,
    )


async def _start_services(
    sb: modal.Sandbox,
    sidecars: dict[str, SidecarContainer],
    plan: _Plan,
    images: Mapping[str, modal.Image],
    docker_image: modal.Image | None,
    vm: bool,
    *,
    release_main: bool,
) -> None:
    """Start the daemon, then every service once what it depends on is up.

    ``main`` is released at its turn when it was held back. Like ``docker compose up
    --wait``, it then waits for every healthchecked service.
    """
    healthy: set[str] = set()

    def handle(name: str) -> modal.Sandbox | SidecarContainer:
        return sb if name == "main" else sidecars[name]

    async def satisfied(dependency: str, condition: str) -> None:
        if condition == "service_healthy":
            check = plan.service(dependency).healthcheck
            assert check is not None
            await _healthy(handle(dependency), dependency, check)
            healthy.add(dependency)
        elif condition == "service_completed_successfully":
            sidecar = sidecars[dependency]
            await sidecar.wait.aio(raise_on_termination=False)
            code = await sidecar.poll.aio()
            if code != 0:
                raise RuntimeError(f"Compose service {dependency!r} exited with code {code}")

    if docker_image is not None:
        daemon = await sb._experimental_sidecars.create.aio(  # noqa: SLF001 - no public sidecar API
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
    for name in plan.order:
        service = plan.service(name)
        for dependency, condition in service.dependencies.items():
            await satisfied(dependency, condition)
        if name != "main":
            sidecars[name] = await _create_sidecar(sb, service, images[name], vm=vm)
        elif release_main:
            process = await sb.exec.aio("touch", _START_FILE)
            await process.wait.aio()
    for name in plan.order:
        check = plan.service(name).healthcheck
        if check is not None and name not in healthy:
            await _healthy(handle(name), name, check)


async def _healthy(
    handle: modal.Sandbox | SidecarContainer, name: str, check: ComposeHealthcheck
) -> None:
    """Return once ``check`` passes in the container.

    Raises once it has failed ``retries`` times in a row after ``start_period``.
    """
    argv = _healthcheck_argv(check)
    retries = check.retries if check.retries is not None else 3
    loop = asyncio.get_running_loop()
    grace_until = loop.time() + check.start_period_sec
    failures = 0
    while True:
        try:
            async with asyncio.timeout(check.timeout_sec):
                process = await handle.exec.aio(*argv, timeout=math.ceil(check.timeout_sec))
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
        await asyncio.sleep(
            check.start_interval_sec if loop.time() < grace_until else check.interval_sec
        )

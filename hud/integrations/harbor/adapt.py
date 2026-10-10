"""Adapt Harbor task directories into runnable HUD environments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hud.capabilities import Capability
from hud.environment import Mount, Peer
from hud.environment.egress import ANY_HOST, BRIDGE_PORT, VISITOR_PORT
from hud.eval import Task, Taskset
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeLimits, RuntimeResources, RuntimeTPU
from hud.eval.runtime.compose import (
    ComposeConfig,
    ComposeProject,
    ComposeService,
    ComposeUnboundVariableError,
    ComposeUnsupportedError,
)
from hud.utils.naming import normalize_environment_name
from hud.version import __version__

from .build import (
    ImageResolutionError,
    image_environment,
    image_ports,
    require_docker,
    resolve_images,
)
from .config import (
    Artifact,
    Collect,
    ControllerConfig,
    EnvironmentPolicy,
    HealthcheckConfig,
    Network,
    PhasePolicy,
    VerifierImage,
)

LOGGER = logging.getLogger(__name__)
ASSETS = Path(__file__).parent
COMPOSE_FILENAME = "docker-compose.yaml"
CONTROLLER_ROOT = Path("/controller")
CONTROLLER_MODULE = "hud.integrations.harbor.env:env"
MOUNTS_ROOT = Path("/mounts")
TASK_ROOT = Path("/rootfs")
IGNORED = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "venv",
    "*.egg-info",
    ".pytest_cache",
)
MCPTransport = Literal["sse", "streamable-http", "stdio"]
FindingKind = Literal["contract", "invalid"]
NetworkMode = Literal["public", "no-network", "allowlist"]


def harbor_network(mode: NetworkMode, allowed_hosts: list[str]) -> Network:
    if mode == "no-network":
        return Network(enabled=False, allowed_hosts=[])
    return Network(enabled=True, allowed_hosts=allowed_hosts if mode == "allowlist" else [ANY_HOST])


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    transport: MCPTransport
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    docker_image: str | None = None
    os: Literal["linux", "windows"] = "linux"
    cpus: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    build_timeout_sec: float | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] = Field(default_factory=list)
    tpu: RuntimeTPU | None = None
    network_mode: NetworkMode = "public"
    allowed_hosts: list[str] = Field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    healthcheck: HealthcheckConfig | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    @property
    def runtime_resources(self) -> RuntimeResources | None:
        resources = RuntimeResources(
            cpu=self.cpus,
            memory_mb=self.memory_mb,
            storage_mb=self.storage_mb,
            gpu=RuntimeGPU(count=self.gpus, type=self.gpu_types or None) if self.gpus else None,
            os=None if self.os == "linux" else self.os,
            tpu=self.tpu,
        )
        return resources if resources.model_dump(exclude_none=True) else None

    @property
    def runtime_limits(self) -> RuntimeLimits | None:
        if self.build_timeout_sec is None:
            return None
        return RuntimeLimits(startup_timeout_s=math.ceil(self.build_timeout_sec))

    def compose_variables(self, *, main_image: str) -> dict[str, str]:
        """Return the variables Harbor defines for a task's Compose file.

        Harbor's infrastructure variables take precedence over literal task
        ``[environment.env]`` values; variables only a host could supply stay unbound.
        """
        literal_env = {name: value for name, value in self.env.items() if "$" not in value}
        infra = {
            "CONTEXT_DIR": ".",
            "MAIN_IMAGE_NAME": main_image,
            "CPUS": None if self.cpus is None else f"{self.cpus:g}",
            "MEMORY": None if self.memory_mb is None else f"{self.memory_mb}M",
            "PREBUILT_IMAGE_NAME": self.docker_image,
        }
        return literal_env | {name: value for name, value in infra.items() if value is not None}


class Phase(BaseModel):
    model_config = ConfigDict(extra="allow")

    user: str | int | None = None
    network_mode: NetworkMode | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float | None = Field(default=None, gt=0)
    environment: EnvironmentConfig | None = None
    environment_mode: Literal["shared", "separate"] | None = None
    collect: list[Collect] = Field(default_factory=list)

    @property
    def separate(self) -> bool:
        return self.environment_mode == "separate" or self.environment is not None

    def network(self, baseline: EnvironmentConfig) -> Network:
        if self.network_mode is None:
            return harbor_network(baseline.network_mode, baseline.allowed_hosts)
        return harbor_network(self.network_mode, self.allowed_hosts)


class PackageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str | None = None
    task: PackageInfo = Field(default_factory=PackageInfo)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agent: Phase = Field(default_factory=Phase)
    verifier: Phase = Field(default_factory=Phase)
    steps: list[dict[str, Any]] | None = None


class AdaptFinding(BaseModel):
    """One independently detectable reason a Harbor task was not adapted."""

    code: str
    kind: FindingKind
    message: str


class AdaptFailure(BaseModel):
    """All findings for one Harbor task."""

    task: str
    path: Path
    findings: tuple[AdaptFinding, ...]


@dataclass(frozen=True, slots=True)
class AdaptResult:
    """Successful task rows and structured failures from one adaptation."""

    taskset: Taskset
    failures: tuple[AdaptFailure, ...]


@dataclass(frozen=True, slots=True)
class HarborTask:
    path: Path
    config: TaskConfig
    instruction: str
    environment_hash: str
    compose: ComposeConfig
    dockerfile: Path
    base_image: str
    resources: RuntimeResources | None


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(root.rglob("*")):
        relative_path = entry.relative_to(root).as_posix().encode()
        if entry.is_symlink():
            digest.update(relative_path + b"\0symlink\0" + os.readlink(entry).encode())
        elif entry.is_file():
            digest.update(relative_path + b"\0" + entry.read_bytes())
    return digest.hexdigest()[:16]


def _inspect_task(task_dir: Path) -> tuple[HarborTask | None, tuple[AdaptFinding, ...]]:
    findings: list[AdaptFinding] = []

    def add(code: str, message: str) -> None:
        kind: FindingKind = "contract" if ".unsupported." in code else "invalid"
        findings.append(AdaptFinding(code=code, kind=kind, message=message))

    try:
        raw_config = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
    except OSError as error:
        return None, (
            AdaptFinding(code="harbor.invalid.task_config_io", kind="invalid", message=str(error)),
        )
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        return None, (
            AdaptFinding(
                code="harbor.invalid.task_config_toml", kind="invalid", message=str(error)
            ),
        )

    try:
        config = TaskConfig.model_validate(raw_config)
    except ValidationError as error:
        return None, tuple(
            AdaptFinding(
                code="harbor.invalid.task_config",
                kind="invalid",
                message=f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}",
            )
            for detail in error.errors(include_url=False)
        )

    environment = config.environment
    resources = environment.runtime_resources

    if config.steps:
        add("harbor.unsupported.multi_step", "multi-step tasks are not supported")

    server_names = [server.name for server in environment.mcp_servers]
    if len(server_names) != len(set(server_names)):
        add("harbor.invalid.duplicate_mcp_name", "MCP server names must be unique")
    for name in sorted({"shell", "filetracking"} & set(server_names)):
        add(
            "harbor.invalid.reserved_mcp_name",
            f"MCP server name {name!r} is reserved by the workspace",
        )
    for server in environment.mcp_servers:
        if server.transport != "stdio" and server.url is None:
            add(
                "harbor.invalid.mcp_url",
                f"MCP server {server.name!r} requires a URL",
            )

    environment_dir = task_dir / "environment"
    environment_hash = _tree_hash(environment_dir) if environment_dir.exists() else "missing"
    built_base_image = f"hud-harbor-base:{environment_hash}"
    compose_path = environment_dir / COMPOSE_FILENAME
    compose: ComposeConfig | None = ComposeConfig(services={})
    dockerfile = environment_dir / "Dockerfile"
    base_image: str | None = None
    if compose_path.is_file():
        try:
            compose = ComposeConfig.from_file(
                compose_path,
                variables=environment.compose_variables(main_image=built_base_image),
            )
        except ComposeUnboundVariableError as error:
            add("harbor.unsupported.host_compose_variable", str(error))
            compose = None
        except ComposeUnsupportedError as error:
            add("harbor.unsupported.compose_include_extends", str(error))
            compose = None
        except (OSError, ValueError, ValidationError) as error:
            add("harbor.invalid.compose", str(error))
            compose = None

    if compose is not None:
        compose.services.setdefault("main", ComposeService())
        compose.name = None
        try:
            compose.with_project_directory("./environment")
        except ValueError as error:
            add("harbor.invalid.compose_project_path", str(error))
        compose_main = compose.services["main"]
        base_image = environment.docker_image or compose_main.image
        build = compose_main.build
        if build is not None:
            build_config = {"context": build} if isinstance(build, str) else build
            build_context = build_config.get("context", ".")
            build_dockerfile = build_config.get("dockerfile", "Dockerfile")
            if not isinstance(build_context, str) or not isinstance(build_dockerfile, str):
                add(
                    "harbor.invalid.compose_main_build_path",
                    "Compose main build paths must be strings",
                )
            else:
                dockerfile = (environment_dir / build_context / build_dockerfile).resolve()
                try:
                    dockerfile.relative_to(environment_dir.resolve())
                except ValueError:
                    add(
                        "harbor.invalid.compose_main_build_escape",
                        "Compose main build escapes environment",
                    )
        if dockerfile.is_file():
            base_image = built_base_image
        elif build is not None:
            add(
                "harbor.invalid.missing_compose_main_dockerfile",
                "Compose main Dockerfile does not exist",
            )
        elif base_image is None:
            add(
                "harbor.invalid.environment_recipe",
                "main has no environment/Dockerfile, docker_image, or Compose image or build",
            )

        if not config.steps:
            if config.verifier.separate and not (task_dir / "tests" / "Dockerfile").is_file():
                add(
                    "harbor.invalid.missing_verifier_dockerfile",
                    "separate verifier requires tests/Dockerfile",
                )
            elif not (task_dir / "tests").is_dir():
                add(
                    "harbor.invalid.missing_tests",
                    "task requires a tests directory",
                )

        if {"hud-base", "hud-verifier"} & compose.services.keys():
            add(
                "harbor.invalid.reserved_compose_service",
                "Compose service names 'hud-base' and 'hud-verifier' are reserved",
            )
        for service_name, service in compose.services.items():
            if service_name == "main":
                continue
            if service.build is None and service.image is None:
                add(
                    "harbor.invalid.sidecar_recipe",
                    f"Compose service {service_name!r} has neither image nor build",
                )
        for port in sorted(compose_main.tcp_ports & {BRIDGE_PORT, VISITOR_PORT, 8765}):
            add(
                "harbor.invalid.reserved_main_port",
                f"Harbor main service port {port} conflicts with a HUD reserved port",
            )
        if environment.healthcheck is None and compose_main.healthcheck is not None:
            try:
                HealthcheckConfig.from_compose(compose_main.healthcheck)
            except ValueError as error:
                add("harbor.invalid.healthcheck", str(error))

    instruction = task_dir / "instruction.md"
    if not config.steps and not instruction.is_file():
        add(
            "harbor.invalid.missing_instruction",
            f"{task_dir.name} has no instruction.md",
        )
    if findings:
        return None, tuple(findings)

    assert compose is not None
    assert base_image is not None
    return (
        HarborTask(
            path=task_dir,
            config=config,
            instruction=instruction.read_text("utf-8"),
            environment_hash=environment_hash,
            compose=compose,
            dockerfile=dockerfile,
            base_image=base_image,
            resources=resources,
        ),
        (),
    )


def adapt(
    path: str | Path,
    *,
    hud_requirement: str = f"hud=={__version__}",
) -> AdaptResult:
    """Resolve Harbor images and package tasks as conventional Compose projects.

    ``hud_requirement`` installs the controller each image serves; it defaults to
    this ``hud`` release so the controller matches the adapter that configured it.
    """
    root = Path(path).resolve()
    if (root / "task.toml").is_file():
        task_dirs = [root]
        dataset = root.parent
    elif root.is_dir():
        task_dirs = sorted(child for child in root.iterdir() if (child / "task.toml").is_file())
        dataset = root
    else:
        task_dirs = []
        dataset = root
    if not task_dirs:
        raise ValueError(f"no Harbor tasks found in {path}")

    tasks: list[HarborTask] = []
    failures: list[AdaptFailure] = []
    for task_dir in task_dirs:
        task, findings = _inspect_task(task_dir)
        if task is not None:
            tasks.append(task)
        else:
            failures.append(AdaptFailure(task=task_dir.name, path=task_dir, findings=findings))

    grouped: dict[tuple[str, str, str], list[HarborTask]] = {}
    for task in tasks:
        group_config = task.config.model_dump(
            mode="json",
            exclude={
                "task": True,
                "metadata": True,
                "steps": True,
                "artifacts": True,
                "agent": {"timeout_sec"},
                "verifier": {"timeout_sec", "collect"},
            },
        )
        config_json = json.dumps(
            group_config,
            sort_keys=True,
        )
        grouped.setdefault(
            (
                task.environment_hash,
                config_json,
                task.path.name if task.config.verifier.separate else "",
            ),
            [],
        ).append(task)

    if grouped:
        require_docker()
    rows = []
    base_name = normalize_environment_name(dataset.name, default="harbor")
    for group_key, group in sorted(grouped.items()):
        digest = hashlib.sha256("\0".join(group_key).encode()).hexdigest()[:12]
        name = f"{base_name}-{digest}"
        source = group[0]
        environment = source.config.environment
        compose = source.compose.model_copy(deep=True)
        compose_project = compose.with_project_directory("./environment")
        for service_name, service in compose_project.services.items():
            if service_name != "main" and service.build is not None and service.image is None:
                sidecar_tag = hashlib.sha256(
                    f"{source.environment_hash}\0{service_name}".encode()
                ).hexdigest()[:16]
                compose_project.services[service_name] = service.model_copy(
                    update={"image": f"hud-harbor-sidecar:{sidecar_tag}"}
                )
        compose_main = compose.services["main"]
        workspace_mounts: list[Mount] = []
        main = compose_project.services["main"]
        runtime_volumes: list[str | dict[str, Any]] = []
        for index, volume in enumerate(main.volumes):
            if isinstance(volume, str):
                parts = volume.split(":")
                target_index = 0 if len(parts) == 1 else 1
                target = PurePosixPath(parts[target_index])
                if not target.is_absolute():
                    raise ValueError(f"Compose main volume target must be absolute: {volume!r}")
                read_only = len(parts) > 2 and "ro" in parts[2].split(",")
                parts[target_index] = str(MOUNTS_ROOT / str(index))
                runtime_volumes.append(":".join(parts))
            else:
                target_value = volume.get("target")
                target = PurePosixPath(target_value) if isinstance(target_value, str) else None
                if target is None or not target.is_absolute():
                    raise ValueError(f"Compose main volume target must be absolute: {volume!r}")
                read_only = volume.get("read_only") is True
                runtime_volumes.append({**volume, "target": str(MOUNTS_ROOT / str(index))})
            workspace_mounts.append(
                Mount(
                    "ro" if read_only else "rw",
                    src=str(MOUNTS_ROOT / str(index)),
                    dst=str(target),
                )
            )
        compose_project.services["main"] = main.model_copy(update={"volumes": runtime_volumes})
        dockerfile = source.dockerfile
        base_image = source.base_image

        separate = source.config.verifier.separate
        verifier_environment = source.config.verifier.environment or EnvironmentConfig()
        verifier_image = base_image
        if separate:
            verifier_dockerfile = source.path / "tests" / "Dockerfile"
            verifier_image = f"hud-harbor-verifier:{name}-{_tree_hash(verifier_dockerfile.parent)}"

        peers: list[Peer] = []
        peer_services: set[str] = set()
        completed_services: set[str] = set()
        for service in compose.services.values():
            depends_on = (service.model_extra or {}).get("depends_on")
            if not isinstance(depends_on, dict):
                continue
            completed_services.update(
                name
                for name, dependency in depends_on.items()
                if isinstance(name, str)
                and isinstance(dependency, dict)
                and dependency.get("condition") == "service_completed_successfully"
            )
        for service_name, service in compose.services.items():
            if service_name == "main" or service_name in completed_services:
                continue
            if service.tcp_ports:
                peers.extend(
                    Peer(service_name, port, target=(service_name, port))
                    for port in sorted(service.tcp_ports)
                )
            else:
                peer_services.add(service_name)
        try:
            resolved = resolve_images(
                source,
                compose_project,
                verifier_image=verifier_image,
                peer_services=peer_services,
            )
        except ImageResolutionError as error:
            finding = AdaptFinding(code="harbor.invalid.image", kind="invalid", message=str(error))
            failures.extend(
                AdaptFailure(task=task.path.name, path=task.path, findings=(finding,))
                for task in group
            )
            continue
        for service_name, image_config in sorted(resolved.peers.items()):
            service_ports = image_ports(image_config, image=f"Compose service {service_name!r}")
            peers.extend(
                Peer(service_name, port, target=(service_name, port))
                for port in sorted(service_ports)
            )
        context = dataset / ".hud-adapt" / name
        if context.exists():
            shutil.rmtree(context)
        project = context / "compose-project"
        payload = project / "main"
        (payload / "packages").mkdir(parents=True)
        shutil.copy2(ASSETS / "install.sh", payload / "install.sh")
        shutil.copy2(ASSETS / "Dockerfile", payload / "Dockerfile")
        # ``hud deploy`` resolves an environment's identity from a literal
        # Environment(...) in source; the image serves CONTROLLER_MODULE.
        (context / "env.py").write_text(
            f'"""Deploy identity for this project; the image serves {CONTROLLER_MODULE}."""\n\n'
            "from hud import Environment\n\n"
            f"env = Environment({name!r})\n",
            encoding="utf-8",
            newline="\n",
        )

        workdir = environment.workdir or compose_main.working_dir or resolved.main.get("WorkingDir")
        if workdir is not None and not isinstance(workdir, str):
            raise ValueError("OCI image WorkingDir must be a string")
        workdir = workdir or "/"
        image_user = compose_main.user
        if image_user is None:
            image_user = resolved.main.get("User") or None
        if image_user is not None and not isinstance(image_user, (str, int)):
            raise ValueError("OCI image User must be a string")
        entrypoint = compose_main.entrypoint
        if entrypoint is None:
            entrypoint = resolved.main.get("Entrypoint") or []
        if not isinstance(entrypoint, list) or not all(
            isinstance(argument, str) for argument in entrypoint
        ):
            raise ValueError("OCI image Entrypoint must be a list of strings")
        # Reserved ports an image EXPOSEs are not forwarded; _inspect_task rejects
        # explicit Compose declarations of them.
        ports = compose_main.tcp_ports | (
            image_ports(resolved.main, image="main image") - {BRIDGE_PORT, VISITOR_PORT, 8765}
        )
        verifier_image_user = (resolved.verifier.get("User") or None) if separate else image_user
        verifier_workdir = (
            verifier_environment.workdir or resolved.verifier.get("WorkingDir") or "/"
        )
        if not isinstance(verifier_workdir, str):
            raise ValueError("OCI verifier image WorkingDir must be a string")
        healthcheck = environment.healthcheck
        if healthcheck is None and compose_main.healthcheck is not None:
            healthcheck = HealthcheckConfig.from_compose(compose_main.healthcheck)
        agent_phase = source.config.agent
        verifier_phase = source.config.verifier
        controller_config = ControllerConfig(
            name=name,
            mounts=workspace_mounts,
            workdir=workdir,
            image_user=image_user,
            image_env=image_environment(resolved.main),
            entrypoint=entrypoint,
            ports=sorted(ports),
            verifier_root="/verifier" if separate else None,
            verifier_image=VerifierImage(
                workdir=verifier_workdir,
                env=image_environment(resolved.verifier),
            ),
            environment=EnvironmentPolicy(
                env={**compose_main.environment, **environment.env},
                network=harbor_network(environment.network_mode, environment.allowed_hosts),
                healthcheck=healthcheck,
            ),
            agent=PhasePolicy(
                user=image_user if agent_phase.user is None else agent_phase.user,
                network=agent_phase.network(environment),
                env=agent_phase.env,
            ),
            verifier=PhasePolicy(
                user=verifier_image_user if verifier_phase.user is None else verifier_phase.user,
                network=verifier_phase.network(verifier_phase.environment or environment),
                env=(
                    verifier_phase.env
                    if verifier_phase.environment is None
                    else {**verifier_phase.environment.env, **verifier_phase.env}
                ),
            ),
            capabilities=[
                Capability.mcp(
                    name=server.name,
                    url=cast("str", server.url),
                    transport=server.transport,
                ).to_manifest()
                for server in environment.mcp_servers
                if server.transport != "stdio"
            ],
            local_aliases=["main"],
            peers=peers,
        )
        (payload / "config.json").write_text(
            json.dumps(controller_config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        wheel = Path(hud_requirement)
        requirement = hud_requirement
        if wheel.suffix == ".whl" and wheel.is_file():
            shutil.copy2(wheel, payload / "packages" / wheel.name)
            requirement = f"{CONTROLLER_ROOT}/packages/{wheel.name}"

        tag = hashlib.sha256(f"{_tree_hash(payload)}\0{requirement}".encode()).hexdigest()[:16]
        image = f"hud-harbor:{name}-{tag}"
        group_service_access = any(
            item.service != "main"
            for task in group
            for item in (*task.config.verifier.collect, *task.config.artifacts)
        )
        runtime_command = [
            "/controller/venv/bin/hud",
            "serve",
            CONTROLLER_MODULE,
            "--host",
            "0.0.0.0",  # noqa: S104 - container control channel
            "--port",
            "8765",
        ]
        for service_name, service in compose_project.services.items():
            depends_on = (service.model_extra or {}).get("depends_on")
            if service_name == "main" or not isinstance(depends_on, dict):
                continue
            main_dependency = depends_on.get("main")
            if (
                isinstance(main_dependency, dict)
                and main_dependency.get("condition") == "service_healthy"
            ):
                main_dependency["condition"] = "service_started"
        source_environment = source.path / "environment"
        project_environment = project / "environment"
        if source_environment.is_dir():
            shutil.copytree(source_environment, project_environment, symlinks=True)
        else:
            project_environment.mkdir()
        authored_main = compose_project.services["main"]
        base_build = authored_main.build
        if base_build is None and dockerfile.is_file():
            base_build = {"context": "./environment"}
        additional_contexts: dict[str, str] = {}
        if base_build is not None:
            # scale: 0 keeps build-only services in the Compose model so
            # service: additional contexts resolve, without starting them.
            compose_project.services["hud-base"] = ComposeService(
                image=base_image,
                build=base_build,
            ).model_copy(update={"scale": 0})
            additional_contexts["hud-base"] = "service:hud-base"
        if separate:
            shutil.copytree(source.path / "tests", project / "verifier", symlinks=True)
            compose_project.services["hud-verifier"] = ComposeService(
                image=verifier_image,
                build={"context": "./verifier"},
            ).model_copy(update={"scale": 0})
            additional_contexts["hud-verifier"] = "service:hud-verifier"

        wrapper_build: dict[str, Any] = {
            "context": "./main",
            "target": (
                "verifier" if separate else "service-access" if group_service_access else "plain"
            ),
            "args": {
                "BASE_IMAGE": "hud-base" if base_build is not None else base_image,
                "VERIFIER_IMAGE": "hud-verifier" if separate else base_image,
                "HUD_REQUIREMENT": requirement,
            },
        }
        if additional_contexts:
            wrapper_build["additional_contexts"] = additional_contexts
        compose_project.services["main"] = authored_main.model_copy(
            update={
                "image": image,
                "build": wrapper_build,
                "entrypoint": [],
                "command": runtime_command,
                "working_dir": None,
                "user": None,
                "healthcheck": None,
            }
        )

        if not separate:
            tests_root = project / "tests"
            tests_root.mkdir()
            for task in group:
                shutil.copytree(
                    task.path / "tests",
                    tests_root / task.path.name,
                    symlinks=True,
                    ignore=IGNORED,
                )
            main = compose_project.services["main"]
            compose_project.services["main"] = main.model_copy(
                update={"volumes": [*main.volumes, "./tests:/controller/tests:ro"]}
            )
        recipe = compose_project.with_project_directory("./compose-project")
        compose_document = context / "compose.yaml"
        compose_document.write_text(
            json.dumps(
                recipe.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        group_rows = []
        for task in group:
            config = task.config
            task_separate = config.verifier.separate
            task_config = {
                "id": task.path.name,
                "description": config.task.description,
                "verifier_timeout": config.verifier.timeout_sec or 600.0,
                "separate_verifier": task_separate,
                "collect": [hook.model_dump() for hook in config.verifier.collect],
                "artifacts": [
                    artifact.model_dump(
                        exclude_none=True,
                        exclude={"exclude"} if not artifact.exclude else None,
                    )
                    for artifact in config.artifacts
                ],
            }
            verifier_environment = config.verifier.environment
            verifier_resources = (
                verifier_environment.runtime_resources if verifier_environment else None
            )
            verifier_limits = verifier_environment.runtime_limits if verifier_environment else None
            needs_service_access = any(
                item.service != "main" for item in (*config.verifier.collect, *config.artifacts)
            )
            runtime_limits = config.environment.runtime_limits
            verifier_uses_actor = (
                verifier_resources == task.resources and verifier_limits == runtime_limits
            )
            columns = dict(config.metadata)
            if config.task.keywords:
                columns.setdefault("keywords", config.task.keywords)
            row = Task(
                env=name,
                id="run",
                args={
                    "instruction": task.instruction,
                    "task": task_config,
                },
                slug=task.path.name,
                agent_config=(
                    {"timeout_seconds": config.agent.timeout_sec}
                    if config.agent.timeout_sec is not None
                    else None
                ),
                columns=columns or None,
                runtime_config=RuntimeConfig(
                    compose=ComposeProject(
                        document=compose_document,
                        root=context,
                        service_access=(True if needs_service_access else None),
                    ),
                    resources=task.resources,
                    limits=runtime_limits,
                ),
                verifier=(
                    Task(
                        env=name,
                        id="verify",
                        args={"task": task_config},
                        slug=f"{task.path.name}:verify",
                        runtime_config=(
                            RuntimeConfig(
                                compose=ComposeProject(
                                    document=compose_document,
                                    root=context,
                                ),
                                resources=verifier_resources,
                                limits=verifier_limits,
                            )
                            if not verifier_uses_actor
                            and (verifier_resources is not None or verifier_limits is not None)
                            else None
                        ),
                    )
                    if task_separate
                    else None
                ),
            )
            rows.append(row)
            group_rows.append(row)
        Taskset(dataset.name, group_rows).to_file(context / "tasks.json")

    LOGGER.info("adapted %d Harbor project(s)", len({task.env for task in rows}))
    return AdaptResult(
        taskset=Taskset(dataset.name, rows),
        failures=tuple(failures),
    )

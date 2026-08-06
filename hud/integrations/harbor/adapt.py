"""Build Harbor task directories as runnable HUD environments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hud.capabilities import Capability
from hud.environment.egress import BRIDGE_PORT, VISITOR_PORT
from hud.eval import Task, Taskset
from hud.eval.compose import ComposeConfig, ComposeHealthcheck, ComposeService, ImageConfig
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeResources
from hud.utils.docker import docker
from hud.utils.naming import normalize_environment_name

LOGGER = logging.getLogger(__name__)
ASSETS = Path(__file__).parent
HUD_ROOT = Path("/media/hud")
IGNORED = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "venv",
    "*.egg-info",
    ".pytest_cache",
)
NetworkMode = Literal["public", "no-network", "allowlist"]
MCPTransport = Literal["sse", "streamable-http", "stdio"]
COMPOSE_FILENAMES = (
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
)


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=r"^/")
    service: str = Field(default="main", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Any:
        return {"source": value} if isinstance(value, str) else value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        path = PurePosixPath(value)
        if len(path.parts) == 1 or ".." in path.parts:
            raise ValueError("artifact source must name a path beneath /")
        return str(path)


class Collect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = Field(default="main", min_length=1)
    command: str = Field(min_length=1)
    timeout_sec: float = Field(default=600.0, gt=0)


class HealthcheckConfig(BaseModel):
    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3

    @classmethod
    def from_compose(cls, value: ComposeHealthcheck) -> HealthcheckConfig | None:
        if value.disable or value.test in (None, ["NONE"]):
            return None
        test = value.test
        assert test
        if test[0] == "CMD" and len(test) > 1:
            command = shlex.join(str(part) for part in test[1:])
        elif test[0] == "CMD-SHELL" and len(test) == 2:
            command = str(test[1])
        else:
            raise ValueError("Compose main healthcheck test must be CMD or CMD-SHELL")

        def seconds(raw: str | None, default: float) -> float:
            if raw is None:
                return default
            units = {
                "ns": 1e-9,
                "us": 1e-6,
                "µs": 1e-6,
                "ms": 1e-3,
                "s": 1,
                "m": 60,
                "h": 3600,
            }
            parts = re.findall(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)", str(raw))
            if not parts or "".join(number + unit for number, unit in parts) != raw:
                raise ValueError(f"invalid Compose healthcheck duration {raw!r}")
            return sum(float(number) * units[unit] for number, unit in parts)

        return cls(
            command=command,
            interval_sec=seconds(value.interval, 30.0),
            timeout_sec=seconds(value.timeout, 30.0),
            start_period_sec=seconds(value.start_period, 0.0),
            start_interval_sec=seconds(value.start_interval, 5.0),
            retries=value.retries if value.retries is not None else 3,
        )


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    transport: MCPTransport
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    build_timeout_sec: float = Field(default=600.0, gt=0)
    docker_image: str | None = None
    os: str = "linux"
    cpus: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] = Field(default_factory=list)
    tpu: dict[str, Any] | None = None
    network_mode: NetworkMode = "public"
    allowed_hosts: list[str] = Field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    healthcheck: HealthcheckConfig | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    skills_dir: str | None = None


class Phase(BaseModel):
    model_config = ConfigDict(extra="allow")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: NetworkMode | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    environment: EnvironmentConfig | None = None
    environment_mode: Literal["separate"] | None = None
    collect: list[Collect] = Field(default_factory=list)

    @property
    def separate(self) -> bool:
        return self.environment_mode == "separate" or self.environment is not None


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


@dataclass(frozen=True, slots=True)
class HarborTask:
    path: Path
    config: TaskConfig
    environment_hash: str
    compose: ComposeConfig | None


async def adapt(
    path: str | Path,
    *,
    push: str | None = None,
    hud_requirement: str = "hud",
) -> Taskset:
    """Build a runnable HUD image for each distinct Harbor environment."""
    root = await asyncio.to_thread(Path(path).resolve)
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

    tasks = []
    for task_dir in task_dirs:
        try:
            config = TaskConfig.model_validate(
                tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
            )
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
            raise ValueError(
                f"{task_dir.name}/task.toml is not a valid Harbor task: {error}"
            ) from error

        unsupported = []
        if config.environment.os != "linux":
            unsupported.append(f"os={config.environment.os!r}")
        if config.environment.tpu:
            unsupported.append("TPUs")
        if len(config.environment.gpu_types) > 1:
            unsupported.append("multiple GPU types")
        elif config.environment.gpu_types and not config.environment.gpus:
            unsupported.append("GPU types without GPUs")
        if any(server.transport == "stdio" for server in config.environment.mcp_servers):
            unsupported.append("stdio MCP servers")
        if config.environment.skills_dir:
            unsupported.append("skills_dir")
        verifier_environment = config.verifier.environment
        if verifier_environment is not None:
            if verifier_environment.os != "linux":
                unsupported.append(f"verifier os={verifier_environment.os!r}")
            if verifier_environment.tpu:
                unsupported.append("verifier TPUs")
            if len(verifier_environment.gpu_types) > 1:
                unsupported.append("multiple verifier GPU types")
            elif verifier_environment.gpu_types and not verifier_environment.gpus:
                unsupported.append("verifier GPU types without GPUs")
            gpu_types = {
                *config.environment.gpu_types,
                *verifier_environment.gpu_types,
            }
            if len(gpu_types) > 1:
                unsupported.append("different agent and verifier GPU types")
        if config.steps:
            unsupported.append("multi-step tasks")
        if unsupported:
            raise NotImplementedError(
                f"Harbor task {task_dir.name!r} uses unsupported features: "
                + ", ".join(unsupported)
            )

        server_names = [server.name for server in config.environment.mcp_servers]
        if len(server_names) != len(set(server_names)):
            raise ValueError("MCP server names must be unique")
        if reserved := {"shell", "filetracking"} & set(server_names):
            raise ValueError(f"MCP server name {min(reserved)!r} is reserved by the workspace")
        for server in config.environment.mcp_servers:
            if server.url is None:
                raise ValueError(f"MCP server {server.name!r} requires a URL")

        environment_dir = task_dir / "environment"
        authored_compose = next(
            (
                environment_dir / filename
                for filename in COMPOSE_FILENAMES
                if (environment_dir / filename).is_file()
            ),
            None,
        )
        compose = None
        if authored_compose is not None:
            with tempfile.TemporaryDirectory(prefix="hud-harbor-compose-") as directory:
                base_compose = Path(directory) / "base.json"
                base_compose.write_text(json.dumps({"services": {"main": {"image": "hud-main"}}}))
                try:
                    compose = await ComposeConfig.load(
                        base_compose,
                        authored_compose,
                        docker=docker,
                        project_name="hud",
                        project_directory=environment_dir,
                    )
                    compose.services["main"]
                except (ValidationError, KeyError) as error:
                    raise ValueError(
                        f"{task_dir.name} did not resolve to a Compose project"
                    ) from error
            compose.name = None
            if "default" in compose.networks and compose.networks["default"] == {
                "name": "hud_default"
            }:
                compose.networks["default"] = {}

        environment_digest = hashlib.sha256()
        if environment_dir.exists():
            for entry in sorted(environment_dir.rglob("*")):
                relative_path = entry.relative_to(environment_dir).as_posix().encode()
                if entry.is_symlink():
                    environment_digest.update(
                        relative_path + b"\0symlink\0" + os.readlink(entry).encode()
                    )
                elif entry.is_file():
                    environment_digest.update(relative_path + b"\0" + entry.read_bytes())
            environment_hash = environment_digest.hexdigest()[:16]
        else:
            environment_hash = "missing"

        tasks.append(
            HarborTask(
                path=task_dir,
                config=config,
                environment_hash=environment_hash,
                compose=compose,
            )
        )

    grouped: dict[tuple[str, str, str], list[HarborTask]] = {}
    for task in tasks:
        config_json = json.dumps(
            task.config.model_dump(mode="json", exclude={"task", "metadata", "steps"}),
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

    rows = []
    base_name = normalize_environment_name(dataset.name, default="harbor")
    for group_key, group in sorted(grouped.items()):
        environment_hash, config_json, _ = group_key
        digest = hashlib.sha256("\0".join(group_key).encode()).hexdigest()[:12]
        name = f"{base_name}-{digest}"
        source = group[0]
        environment = source.config.environment
        compose = source.compose.model_copy(deep=True) if source.compose is not None else None
        compose_main = compose.services["main"] if compose is not None else ComposeService()
        dockerfile = source.path / "environment" / "Dockerfile"
        compose_image = compose_main.image
        base_image = environment.docker_image or (
            compose_image if compose_image != "hud-main" else None
        )
        build_timeout = max(task.config.environment.build_timeout_sec for task in group)
        if compose_main.build is not None and environment.docker_image is None:
            assert compose is not None
            base_image = f"hud-harbor-base:{source.environment_hash}"
            compose.services["main"] = compose_main.model_copy(update={"image": base_image})
            with tempfile.TemporaryDirectory(prefix="hud-harbor-main-") as directory:
                compose_file = Path(directory) / "compose.json"
                compose_file.write_text(
                    compose.model_dump_json(exclude_none=True),
                    encoding="utf-8",
                )
                await docker(
                    "compose",
                    "--file",
                    str(compose_file),
                    "build",
                    "main",
                    deadline=build_timeout,
                )
        elif base_image and not dockerfile.is_file():
            await docker("pull", base_image)
        elif dockerfile.is_file():
            base_image = f"hud-harbor-base:{source.environment_hash}"
            await docker(
                "build",
                "--tag",
                base_image,
                str(dockerfile.parent),
                deadline=build_timeout,
            )
        else:
            raise FileNotFoundError(f"{source.path.name} has no environment/Dockerfile")

        image_config = await ImageConfig.inspect(base_image, docker)
        separate = source.config.verifier.separate
        verifier_environment = source.config.verifier.environment or EnvironmentConfig()
        verifier_image = base_image
        verifier_config = image_config
        if separate:
            verifier_dockerfile = source.path / "tests" / "Dockerfile"
            if not verifier_dockerfile.is_file():
                raise FileNotFoundError(
                    f"{source.path.name} uses a separate verifier but has no tests/Dockerfile"
                )
            verifier_digest = hashlib.sha256()
            for entry in sorted(verifier_dockerfile.parent.rglob("*")):
                relative_path = entry.relative_to(verifier_dockerfile.parent).as_posix().encode()
                if entry.is_symlink():
                    verifier_digest.update(
                        relative_path + b"\0symlink\0" + os.readlink(entry).encode()
                    )
                elif entry.is_file():
                    verifier_digest.update(relative_path + b"\0" + entry.read_bytes())
            verifier_image = f"hud-harbor-verifier:{name}-{verifier_digest.hexdigest()[:16]}"
            await docker(
                "build",
                "--tag",
                verifier_image,
                str(verifier_dockerfile.parent),
                deadline=verifier_environment.build_timeout_sec,
            )
            verifier_config = await ImageConfig.inspect(verifier_image, docker)

        peers = []
        if compose is not None:
            for service_name, service in compose.services.items():
                if service_name == "main":
                    continue
                ports = {
                    int(value)
                    for exposed in service.expose
                    if (value := str(exposed).partition("/")[0]).isdigit()
                }
                ports.update(
                    published.target for published in service.ports if published.protocol == "tcp"
                )
                source_image = service.image
                if service.build is not None:
                    source_image = f"hud-harbor-sidecar-build:{uuid.uuid4().hex}"
                    compose.services[service_name] = service.model_copy(
                        update={"image": source_image}
                    )
                    with tempfile.TemporaryDirectory(prefix="hud-harbor-sidecar-") as directory:
                        compose_file = Path(directory) / "compose.json"
                        compose_file.write_text(
                            compose.model_dump_json(exclude_none=True),
                            encoding="utf-8",
                        )
                        await docker("compose", "--file", str(compose_file), "build", service_name)
                elif source_image:
                    await docker("pull", source_image)
                else:
                    raise ValueError(
                        f"Compose service {service_name!r} has neither image nor build"
                    )
                sidecar_config = await ImageConfig.inspect(source_image, docker)
                if not ports:
                    for value in sidecar_config.exposed_ports:
                        port, separator, protocol = value.partition("/")
                        if port.isdigit() and separator and protocol == "tcp":
                            ports.add(int(port))
                if len(ports) > 1:
                    raise NotImplementedError(
                        f"Compose service {service_name!r} exposes multiple ports; "
                        "Peer names one endpoint"
                    )
                if not ports:
                    raise ValueError(
                        f"Compose service {service_name!r} declares no TCP port in Compose "
                        "or its image"
                    )
                peers.append({"name": service_name, "port": next(iter(ports))})
                image_id, _ = await docker("image", "inspect", "--format", "{{.Id}}", source_image)
                fingerprint = image_id.strip().removeprefix("sha256:")[:16]
                component = normalize_environment_name(f"{name}-{service_name}", default="sidecar")
                sidecar_image = (
                    f"{push}/{component}:{fingerprint}"
                    if push
                    else f"hud-harbor-sidecar:{component}-{fingerprint}"
                )
                await docker("tag", source_image, sidecar_image)
                if push:
                    await docker("push", sidecar_image)
                compose.services[service_name] = service.with_image(
                    sidecar_image,
                    sidecar_config,
                ).model_copy(update={"build": None})
        context = dataset / ".hud-adapt" / name
        if context.exists():
            shutil.rmtree(context)
        (context / "tasks").mkdir(parents=True)
        (context / "packages").mkdir()
        for asset in ("Dockerfile", "install.sh"):
            shutil.copy2(ASSETS / asset, context / asset)
        # ``hud deploy`` resolves the context's identity from a literal
        # Environment(...) name in source, so the copy carries the group's
        # name as a literal; the value is the same one tasks.json serves.
        served = (ASSETS / "env.py").read_text("utf-8")
        sentinel = 'Environment(CONFIG["name"])'
        if sentinel not in served:
            raise RuntimeError(f"env.py asset no longer constructs {sentinel}")
        (context / "env.py").write_text(
            served.replace(sentinel, f'Environment("{name}")'),
            encoding="utf-8",
            newline="\n",
        )

        workdir = environment.workdir or compose_main.working_dir or image_config.working_dir or "/"
        if Path(workdir).is_relative_to(HUD_ROOT):
            raise ValueError(f"Harbor workdir {workdir!r} is inside reserved path {HUD_ROOT}")
        image_env = {}
        for entry in image_config.environment:
            key, _, value = entry.partition("=")
            image_env[key] = value
        ports = {
            int(port)
            for exposed in (*image_config.exposed_ports, *compose_main.expose)
            if (port := str(exposed).partition("/")[0]).isdigit()
        }
        ports.update(
            published.target for published in compose_main.ports if published.protocol == "tcp"
        )
        if conflict := ports & {BRIDGE_PORT, VISITOR_PORT, 8765}:
            raise ValueError(
                f"Harbor main service port {min(conflict)} conflicts with a HUD reserved port"
            )
        healthcheck = environment.healthcheck
        if healthcheck is None and compose_main.healthcheck is not None:
            healthcheck = HealthcheckConfig.from_compose(compose_main.healthcheck)
        verifier_env = {
            key: value
            for entry in verifier_config.environment
            for key, _, value in (entry.partition("="),)
        }
        verifier_phase = source.config.verifier
        verifier_policy = verifier_phase.model_dump(
            include={"user", "network_mode", "allowed_hosts", "env"}
        )
        if verifier_phase.environment is not None:
            if verifier_phase.network_mode is None:
                verifier_policy["network_mode"] = verifier_environment.network_mode
                verifier_policy["allowed_hosts"] = verifier_environment.allowed_hosts
            verifier_policy["env"] = {
                **verifier_environment.env,
                **verifier_phase.env,
            }
        manifest = {
            "name": name,
            "workdir": workdir,
            "image_user": (
                compose_main.user if compose_main.user is not None else image_config.user or None
            ),
            "image_env": image_env,
            "entrypoint": (
                compose_main.entrypoint
                if compose is not None and compose_main.entrypoint is not None
                else image_config.entrypoint or []
            ),
            "ports": sorted(ports),
            "verifier_root": str(HUD_ROOT / "verifier") if separate else None,
            "verifier_image": {
                "user": verifier_config.user or None,
                "workdir": verifier_environment.workdir or verifier_config.working_dir or "/",
                "env": verifier_env,
            },
            "environment": {
                "env": {
                    **compose_main.environment,
                    **environment.env,
                },
                "network_mode": environment.network_mode,
                "allowed_hosts": environment.allowed_hosts,
                "healthcheck": healthcheck.model_dump() if healthcheck is not None else None,
            },
            "agent": source.config.agent.model_dump(
                include={"user", "network_mode", "allowed_hosts", "env"}
            ),
            "verifier": verifier_policy,
            "capabilities": [
                Capability.mcp(
                    name=server.name,
                    url=cast("str", server.url),
                    transport=cast('Literal["sse", "streamable-http"]', server.transport),
                ).to_manifest()
                for server in environment.mcp_servers
            ],
            "local_aliases": ["main"] if compose is not None else [],
            "peers": peers,
            "tasks": [],
        }
        for task in group:
            if not (task.path / "instruction.md").is_file():
                raise FileNotFoundError(f"{task.path.name} has no instruction.md")
            target = context / "tasks" / task.path.name
            target.mkdir()
            shutil.copy2(task.path / "instruction.md", target / "instruction.md")
            task_separate = task.config.verifier.separate
            if not task_separate:
                shutil.copytree(
                    task.path / "tests",
                    target / "tests",
                    symlinks=True,
                    ignore=IGNORED,
                )
            manifest["tasks"].append(
                {
                    "id": task.path.name,
                    "description": task.config.task.description,
                    "verifier_timeout": task.config.verifier.timeout_sec or 600.0,
                    "separate_verifier": task_separate,
                    "collect": [hook.model_dump() for hook in task.config.verifier.collect],
                    "artifacts": [artifact.model_dump() for artifact in task.config.artifacts],
                }
            )
        (context / "tasks.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        wheel = Path(hud_requirement)
        requirement = hud_requirement
        if wheel.suffix == ".whl" and await asyncio.to_thread(wheel.is_file):
            shutil.copy2(wheel, context / "packages" / wheel.name)
            requirement = f"{HUD_ROOT}/packages/{wheel.name}"

        context_digest = hashlib.sha256()
        for entry in sorted(context.rglob("*")):
            relative_path = entry.relative_to(context).as_posix().encode()
            if entry.is_symlink():
                context_digest.update(relative_path + b"\0symlink\0" + os.readlink(entry).encode())
            elif entry.is_file():
                context_digest.update(relative_path + b"\0" + entry.read_bytes())
        tag = context_digest.hexdigest()[:16]
        image = f"{push}/{name}:{tag}" if push else f"hud-harbor:{name}-{tag}"
        await docker(
            "build",
            "--target",
            "verifier" if separate else "plain",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"VERIFIER_IMAGE={verifier_image}",
            "--build-arg",
            f"HUD_REQUIREMENT={requirement}",
            "--tag",
            image,
            str(context),
        )
        if push:
            await docker("push", image)

        if compose is not None:
            main = compose.services["main"].model_copy(
                update={
                    "build": None,
                    "command": None,
                    "entrypoint": None,
                    "working_dir": None,
                    "user": None,
                    "healthcheck": None,
                }
            )
            compose.services["main"] = main.with_image(
                image,
                await ImageConfig.inspect(image, docker),
            )
            (context / "compose.json").write_text(
                json.dumps(
                    compose.model_dump(mode="json", exclude_none=True),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        for task in group:
            config = task.config
            task_separate = config.verifier.separate
            phase_environment = config.verifier.environment or EnvironmentConfig()
            gpu_count = max(config.environment.gpus or 0, phase_environment.gpus or 0)
            gpu_types = config.environment.gpu_types or phase_environment.gpu_types
            resources = RuntimeResources(
                cpu=max(config.environment.cpus or 0, phase_environment.cpus or 0) or None,
                memory_mb=max(
                    config.environment.memory_mb or 0,
                    phase_environment.memory_mb or 0,
                )
                or None,
                gpu=(
                    RuntimeGPU(
                        count=gpu_count,
                        type=next(iter(filter(None, gpu_types)), None),
                    )
                    if gpu_count
                    else None
                ),
            )
            columns = dict(config.metadata)
            if config.task.keywords:
                columns.setdefault("keywords", config.task.keywords)
            rows.append(
                Task(
                    env=name,
                    id=task.path.name,
                    agent_config=(
                        {"timeout_seconds": config.agent.timeout_sec}
                        if config.agent.timeout_sec is not None
                        else None
                    ),
                    columns=columns or None,
                    runtime_config=RuntimeConfig(
                        image=image if compose is None else None,
                        compose=context / "compose.json" if compose is not None else None,
                        compose_service_access=(
                            True if compose is not None and task_separate else None
                        ),
                        resources=resources if resources.model_dump(exclude_none=True) else None,
                    ),
                    verifier=(
                        Task(env=name, id=f"{task.path.name}:verify") if task_separate else None
                    ),
                )
            )

    LOGGER.info("adapted %d Harbor image(s)", len({task.env for task in rows}))
    return Taskset(dataset.name, rows, origin=f"harbor:{dataset}")

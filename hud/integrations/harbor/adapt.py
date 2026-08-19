"""Adapt Harbor task directories into runnable HUD environments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hud.capabilities import Capability
from hud.environment.egress import BRIDGE_PORT, VISITOR_PORT
from hud.eval import Task, Taskset
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeLimits, RuntimeResources, RuntimeTPU
from hud.eval.runtime.compose import (
    ComposeConfig,
    ComposeHealthcheck,
    ComposeProject,
    ComposeService,
    ComposeUnboundVariableError,
)
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
FindingKind = Literal["contract", "invalid"]
COMPOSE_FILENAME = "docker-compose.yaml"


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=r"^/")
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)
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

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str | None) -> str | None:
        if not value:
            return None
        if "\\" in value:
            raise ValueError("artifact destination must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("artifact destination must be a relative path")
        if value.rstrip("/") == "manifest.json":
            raise ValueError("artifact destination 'manifest.json' is reserved")
        return value


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
    skills_dir: str | None = None


class Phase(BaseModel):
    model_config = ConfigDict(extra="allow")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: NetworkMode | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    environment: EnvironmentConfig | None = None
    environment_mode: Literal["shared", "separate"] | None = None
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
    compose: ComposeConfig | None
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


def _runtime_resources(environment: EnvironmentConfig) -> RuntimeResources | None:
    resources = RuntimeResources(
        cpu=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpu=(
            RuntimeGPU(
                count=environment.gpus,
                type=(
                    environment.gpu_types[0]
                    if len(environment.gpu_types) == 1
                    else environment.gpu_types or None
                ),
            )
            if environment.gpus
            else None
        ),
        os=environment.os if environment.os != "linux" else None,
        tpu=environment.tpu,
    )
    return resources if resources.model_dump(exclude_none=True) else None


def _runtime_limits(environment: EnvironmentConfig) -> RuntimeLimits | None:
    if environment.build_timeout_sec is None:
        return None
    return RuntimeLimits(startup_timeout_s=math.ceil(environment.build_timeout_sec))


def _dockerfile_stages(lines: list[str]) -> list[tuple[int, str | None]]:
    escape = "\\"
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        directive = re.fullmatch(r"#\s*escape\s*=\s*([\\`])", stripped, re.IGNORECASE)
        if directive is not None:
            escape = directive.group(1)
        if not stripped.startswith("#"):
            break

    stages: list[tuple[int, str | None]] = []
    heredoc_pattern = re.compile(
        r"<<(?P<strip>-?)[ \t]*(?P<quote>['\"]?)"
        r"(?P<name>[A-Za-z_][\w.-]*)(?P=quote)"
    )
    index = 0
    pattern = re.compile(
        r"^\s*FROM\s+(?:--platform=\S+\s+)?\S+"
        r"(?:\s+AS\s+(?P<name>[A-Za-z0-9_.-]+))?"
        r"\s*(?:#.*)?$",
        re.IGNORECASE,
    )
    while index < len(lines):
        parts: list[str] = []
        while index < len(lines):
            content = lines[index].rstrip("\r\n")
            stripped = content.rstrip(" \t")
            continued = stripped.endswith(escape)
            parts.append(stripped[:-1] if continued else content)
            index += 1
            if not continued:
                break
        instruction = " ".join(parts)
        if re.match(r"^\s*FROM\b", instruction, re.IGNORECASE):
            match = pattern.fullmatch(instruction)
            if match is None:
                raise ValueError("unsupported FROM instruction")
            stages.append((index - 1, match.group("name")))
        for heredoc in heredoc_pattern.finditer(instruction):
            delimiter = heredoc.group("name")
            strip_tabs = bool(heredoc.group("strip"))
            while index < len(lines):
                terminator = lines[index].rstrip("\r\n")
                index += 1
                if strip_tabs:
                    terminator = terminator.lstrip("\t")
                if terminator == delimiter:
                    break
    return stages


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
    except tomllib.TOMLDecodeError as error:
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
    resources = _runtime_resources(environment)
    if any(server.transport == "stdio" for server in environment.mcp_servers):
        add("harbor.unsupported.mcp_stdio", "stdio MCP servers are not supported")
    if environment.skills_dir:
        add(
            "harbor.unsupported.skills_dir",
            "per-task agent skills are not supported",
        )

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
    compose_path = environment_dir / COMPOSE_FILENAME
    authored_compose = compose_path if compose_path.is_file() else None
    compose = None
    dockerfile = environment_dir / "Dockerfile"
    base_image: str | None = None
    if authored_compose is not None:
        try:
            compose = ComposeConfig.from_file(authored_compose)
        except ComposeUnboundVariableError as error:
            add("harbor.unsupported.host_compose_variable", str(error))
        except (OSError, ValueError, ValidationError) as error:
            add("harbor.invalid.compose", str(error))
        else:
            compose.services.setdefault("main", ComposeService())
            compose.name = None
            try:
                compose.with_project_directory("./environment")
            except ValueError as error:
                add("harbor.invalid.compose_project_path", str(error))

    if authored_compose is None or compose is not None:
        compose_main = compose.services["main"] if compose is not None else ComposeService()
        base_image = environment.docker_image or compose_main.image
        if compose is not None:
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
                base_image = f"hud-harbor-base:{_tree_hash(environment_dir)}"
            elif build is not None:
                add(
                    "harbor.invalid.missing_compose_main_dockerfile",
                    "Compose main Dockerfile does not exist",
                )
            elif base_image is None:
                add(
                    "harbor.invalid.compose_main_recipe",
                    "Compose main has neither image nor build",
                )
        elif dockerfile.is_file():
            base_image = f"hud-harbor-base:{_tree_hash(environment_dir)}"
        elif base_image is None:
            add(
                "harbor.invalid.environment_recipe",
                "task has neither environment/Dockerfile nor docker_image",
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

        if compose is not None:
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
        workdir = environment.workdir or compose_main.working_dir
        if workdir is not None and Path(workdir).is_relative_to(HUD_ROOT):
            add(
                "harbor.invalid.reserved_workdir",
                f"Harbor workdir {workdir!r} is inside reserved path {HUD_ROOT}",
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

        if compose is None and dockerfile.is_file():
            try:
                lines = dockerfile.read_text("utf-8").splitlines(keepends=True)
                stages = _dockerfile_stages(lines)
                if not stages:
                    raise ValueError("environment/Dockerfile has no FROM stage")
            except (OSError, UnicodeError, ValueError) as error:
                add("harbor.invalid.dockerfile", str(error))
            else:
                stage_names = {
                    stage_name.lower() for _, stage_name in stages if stage_name is not None
                }
                reserved_names = {"hud-base", "hud-runtime"}
                if config.verifier.separate:
                    reserved_names.update({"hud-docker-cli", "hud-verifier", "hud-verifier-root"})
                for stage in sorted(reserved_names & stage_names):
                    add(
                        "harbor.invalid.reserved_dockerfile_stage",
                        f"environment/Dockerfile uses reserved stage {stage!r}",
                    )

    instruction = task_dir / "instruction.md"
    if not config.steps and not instruction.is_file():
        add(
            "harbor.invalid.missing_instruction",
            f"{task_dir.name} has no instruction.md",
        )
    if findings:
        return None, tuple(findings)

    assert base_image is not None
    return (
        HarborTask(
            path=task_dir,
            config=config,
            instruction=instruction.read_text("utf-8"),
            environment_hash=_tree_hash(environment_dir) if environment_dir.exists() else "missing",
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
    hud_requirement: str = "hud",
) -> AdaptResult:
    """Package Harbor tasks as buildable Compose projects."""
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

    rows = []
    base_name = normalize_environment_name(dataset.name, default="harbor")
    for group_key, group in sorted(grouped.items()):
        digest = hashlib.sha256("\0".join(group_key).encode()).hexdigest()[:12]
        name = f"{base_name}-{digest}"
        source = group[0]
        environment = source.config.environment
        compose = source.compose.model_copy(deep=True) if source.compose is not None else None
        compose_project = (
            compose.with_project_directory("./environment") if compose is not None else None
        )
        if compose_project is not None:
            for service_name, service in compose_project.services.items():
                if service_name != "main" and service.build is not None and service.image is None:
                    sidecar_tag = hashlib.sha256(
                        f"{source.environment_hash}\0{service_name}".encode()
                    ).hexdigest()[:16]
                    compose_project.services[service_name] = service.model_copy(
                        update={"image": f"hud-harbor-sidecar:{sidecar_tag}"}
                    )
        compose_main = compose.services["main"] if compose is not None else ComposeService()
        dockerfile = source.dockerfile
        base_image = source.base_image

        separate = source.config.verifier.separate
        verifier_environment = source.config.verifier.environment or EnvironmentConfig()
        verifier_image = base_image
        if separate:
            verifier_dockerfile = source.path / "tests" / "Dockerfile"
            verifier_image = f"hud-harbor-verifier:{name}-{_tree_hash(verifier_dockerfile.parent)}"

        peers = []
        healthy_services = []
        peer_image_configs: dict[str, str] = {}
        if compose is not None:
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
                healthcheck = service.healthcheck
                if (
                    healthcheck is not None
                    and healthcheck.disable is not True
                    and healthcheck.test != ["NONE"]
                ):
                    healthy_services.append(service_name)
                if service.tcp_ports:
                    peers.extend(
                        {"name": service_name, "port": port} for port in sorted(service.tcp_ports)
                    )
                else:
                    peer_image_configs[service_name] = f"peer-image-configs/{service_name}.json"
        context = dataset / ".hud-adapt" / name
        if context.exists():
            shutil.rmtree(context)
        project = context / "compose-project"
        payload = project / ("main" if compose is not None else "hud")
        (payload / "packages").mkdir(parents=True)
        if compose is not None:
            (payload / "peer-image-configs").mkdir()
            (payload / "peer-image-configs" / ".keep").touch()
        shutil.copy2(ASSETS / "install.sh", payload / "install.sh")
        if compose is not None:
            shutil.copy2(ASSETS / "Dockerfile", payload / "Dockerfile")
        # ``hud deploy`` resolves the context's identity from a literal
        # Environment(...) name in source, so the copy carries the group's
        # name as a literal; the value is the same one config.json serves.
        served = (ASSETS / "env.py").read_text("utf-8")
        sentinel = 'Environment(CONFIG["name"])'
        if sentinel not in served:
            raise RuntimeError(f"env.py asset no longer constructs {sentinel}")
        served = served.replace(sentinel, f'Environment("{name}")')
        for target in (context / "env.py", payload / "env.py"):
            target.write_text(served, encoding="utf-8", newline="\n")

        workdir = environment.workdir or compose_main.working_dir
        ports = compose_main.tcp_ports
        healthcheck = environment.healthcheck
        if healthcheck is None and compose_main.healthcheck is not None:
            healthcheck = HealthcheckConfig.from_compose(compose_main.healthcheck)
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
            "image_user": compose_main.user,
            "image_env": {},
            "entrypoint": compose_main.entrypoint if compose is not None else None,
            "ports": sorted(ports),
            "verifier_root": str(HUD_ROOT / "verifier") if separate else None,
            "verifier_image": {
                "user": None,
                "workdir": verifier_environment.workdir,
                "env": {},
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
            "local_aliases": ["main"],
            "peers": peers,
            "healthy_services": sorted(healthy_services),
            "peer_image_configs": peer_image_configs,
        }
        (payload / "config.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        wheel = Path(hud_requirement)
        requirement = hud_requirement
        if wheel.suffix == ".whl" and wheel.is_file():
            shutil.copy2(wheel, payload / "packages" / wheel.name)
            requirement = f"{HUD_ROOT}/packages/{wheel.name}"

        tag = _tree_hash(payload)
        image = f"hud-harbor:{name}-{tag}"
        group_service_access = bool(healthy_services) or any(
            item.service != "main"
            for task in group
            for item in (*task.config.verifier.collect, *task.config.artifacts)
        )
        runtime_command = [
            "/media/hud/venv/bin/hud",
            "serve",
            "/media/hud/env.py",
            "--host",
            "0.0.0.0",  # noqa: S104 - container control channel
            "--port",
            "8765",
        ]
        base_target: str | None = None
        base_build: str | dict[str, Any] | None = None
        if compose is None:
            project_environment = project / "environment"
            source_environment = source.path / "environment"
            if source_environment.is_dir():
                shutil.copytree(source_environment, project_environment, symlinks=True)
            else:
                project_environment.mkdir(parents=True)
            dockerfile_source = (
                dockerfile.read_bytes().decode("utf-8")
                if dockerfile.is_file()
                else f"FROM {base_image} AS hud-base\n"
            )

            lines = dockerfile_source.splitlines(keepends=True)
            stages = _dockerfile_stages(lines)
            assert stages
            final_index, base_stage = stages[-1]
            if base_stage is None:
                line = lines[final_index]
                content = line.rstrip("\r\n")
                ending = line[len(content) :]
                suffix = re.search(r"\s*(?:#.*)?$", content)
                assert suffix is not None
                lines[final_index] = (
                    f"{content[: suffix.start()]} AS hud-base{content[suffix.start() :]}{ending}"
                )
                base_stage = "hud-base"
            base_target = base_stage
            combined = "".join(lines)
            if combined and not combined.endswith("\n"):
                combined += "\n"
            verifier_stages = (
                "\nFROM hud-verifier AS hud-verifier-root\n"
                "FROM docker:28.3.3-cli AS hud-docker-cli\n"
                if separate
                else ""
            )
            verifier_copies = (
                "COPY --from=hud-docker-cli /usr/local/bin/docker /media/hud/bin/docker\n"
                "COPY --from=hud-verifier-root / /media/hud/verifier\n"
                if separate
                else ""
            )
            combined += f"""{verifier_stages}
FROM {base_stage} AS hud-runtime

USER root
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /media/hud/bin/uv
COPY --from=hud env.py install.sh config.json image-config.json /media/hud/
COPY --from=hud verifier-image-config.json /media/hud/
COPY --from=hud packages /media/hud/packages
RUN sh /media/hud/install.sh {shlex.quote(requirement)}
{verifier_copies}
ENV HUD_SKIP_VERSION_CHECK=1
EXPOSE 8765
ENTRYPOINT []
CMD ["/media/hud/venv/bin/hud", "serve", "/media/hud/env.py", "--host", "0.0.0.0", "--port", "8765"]
"""
            (project / "Dockerfile").write_bytes(combined.encode("utf-8"))

            main_build: dict[str, Any] = {
                "context": "./environment",
                # dockerfile resolves relative to the context; additional
                # context paths resolve relative to the project directory.
                "dockerfile": "../Dockerfile",
                "additional_contexts": {"hud": "./hud"},
            }
            services: dict[str, ComposeService] = {}
            if separate:
                shutil.copytree(source.path / "tests", project / "verifier", symlinks=True)
                services["hud-verifier"] = ComposeService(
                    image=verifier_image,
                    build={"context": "./verifier"},
                ).model_copy(update={"scale": 0})
                main_build["additional_contexts"]["hud-verifier"] = "service:hud-verifier"
            runtime_main = ComposeService(
                image=image,
                build=main_build,
                entrypoint=[],
                command=runtime_command,
            )
            services["main"] = runtime_main
            compose_project = ComposeConfig(services=services)

        if compose is not None:
            assert compose_project is not None
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
            authored_main = compose_project.services["main"]
            main = authored_main.model_copy(
                update={
                    "build": None,
                    "command": None,
                    "entrypoint": None,
                    "working_dir": None,
                    "user": None,
                    "healthcheck": None,
                }
            )
            runtime_main = main.model_copy(
                update={
                    "image": image,
                    "entrypoint": [],
                    "command": runtime_command,
                }
            )

            source_environment = source.path / "environment"
            project_environment = project / "environment"
            shutil.copytree(source_environment, project_environment, symlinks=True)
            base_build = authored_main.build
            if base_build is None and dockerfile.is_file():
                base_build = {"context": "./environment"}
            if base_build is not None:
                # scale: 0 keeps build-only services in the Compose model so
                # service: additional contexts resolve, without starting them.
                compose_project.services["hud-base"] = ComposeService(
                    image=base_image,
                    build=base_build,
                ).model_copy(update={"scale": 0})

            additional_contexts: dict[str, str] = {}
            if base_build is not None:
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
                    "verifier"
                    if separate
                    else "service-access"
                    if group_service_access
                    else "plain"
                ),
                "args": {
                    "BASE_IMAGE": "hud-base" if base_build is not None else base_image,
                    "VERIFIER_IMAGE": "hud-verifier" if separate else base_image,
                    "HUD_REQUIREMENT": requirement,
                },
            }
            if additional_contexts:
                wrapper_build["additional_contexts"] = additional_contexts
            compose_project.services["main"] = runtime_main.model_copy(
                update={"build": wrapper_build}
            )

        assert compose_project is not None
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
                update={"volumes": [*main.volumes, "./tests:/media/hud/tests:ro"]}
            )
        project_compose = context / "compose-project" / "compose.json"
        project_compose.write_text(
            json.dumps(
                compose_project.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assert base_image is not None
        compose_command = (
            'docker compose --project-directory "$PROJECT" --file "$PROJECT/compose.json"'
        )
        if compose is not None and base_build is not None:
            prepare_base = f"{compose_command} build hud-base"
        elif compose is None and dockerfile.is_file():
            assert base_target is not None
            prepare_base = (
                f"docker build --target {shlex.quote(base_target)} "
                f'--tag {shlex.quote(base_image)} --file "$PROJECT/Dockerfile" '
                '"$PROJECT/environment"'
            )
        else:
            prepare_base = f"docker pull {shlex.quote(base_image)}"
        image_config_path = f'"$PROJECT/{payload.name}/image-config.json"'
        verifier_config_path = f'"$PROJECT/{payload.name}/verifier-image-config.json"'
        peer_config_paths = {
            service: f'"$PROJECT/{payload.name}/{path}"'
            for service, path in peer_image_configs.items()
        }
        prepare_peer_lines = []
        for service, path in peer_config_paths.items():
            peer = compose_project.services[service]
            assert peer.image is not None
            operation = "build" if peer.build is not None else "pull"
            prepare_peer_lines.append(
                f"{compose_command} {operation} {shlex.quote(service)}\n"
                f"inspect_peer {shlex.quote(peer.image)} {shlex.quote(service)} > {path}"
            )
        prepare_peers = "\n".join(prepare_peer_lines)
        prepare_verifier = (
            f"{compose_command} build hud-verifier"
            if separate
            else f"cp {image_config_path} {verifier_config_path}"
        )
        inspect_verifier = (
            f"inspect_image {shlex.quote(str(verifier_image))} > {verifier_config_path}"
            if separate
            else ""
        )
        build_script = f"""#!/bin/sh
set -eu
PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cleanup() {{
  rm -f {image_config_path} {verifier_config_path} {" ".join(peer_config_paths.values())}
}}
trap cleanup EXIT HUP INT TERM
inspect_image() {{
  docker image inspect --format '{{{{json .Config}}}}' "$1"
}}
inspect_peer() {{
  if ! docker image inspect --format \
    '{{{{range $port, $_ := .Config.ExposedPorts}}}}{{{{println $port}}}}{{{{end}}}}' "$1" \
    | grep -Eq '^[0-9]+/tcp$'; then
    echo "Compose service '$2' declares no TCP ports in Compose or its image" >&2
    return 1
  fi
  inspect_image "$1"
}}
{prepare_base}
inspect_image {shlex.quote(base_image)} > {image_config_path}
{prepare_verifier}
{inspect_verifier}
{prepare_peers}
{compose_command} build
if [ "$#" -gt 0 ]; then
  docker tag "$({compose_command} images -q main)" "$1"
fi
"""
        script = project / "build.sh"
        script.write_text(build_script, encoding="utf-8", newline="\n")
        script.chmod(0o755)
        launcher = context / "build.sh"
        launcher.write_text(
            '#!/bin/sh\nset -eu\nexec sh "$(dirname -- "$0")/compose-project/build.sh" "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        launcher.chmod(0o755)
        recipe = compose_project.with_project_directory("./compose-project")
        (context / "compose.yaml").write_text(
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
            verifier_resources = (
                _runtime_resources(config.verifier.environment)
                if config.verifier.environment is not None
                else None
            )
            verifier_limits = (
                _runtime_limits(config.verifier.environment)
                if config.verifier.environment is not None
                else None
            )
            needs_service_access = any(
                item.service != "main" for item in (*config.verifier.collect, *config.artifacts)
            ) or bool(healthy_services)
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
                        document=context / "compose-project" / "compose.json",
                        root=context,
                        service_access=(True if needs_service_access else None),
                    ),
                    resources=task.resources,
                    limits=_runtime_limits(config.environment),
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
                                    document=context / "compose-project" / "compose.json",
                                    root=context,
                                ),
                                resources=verifier_resources,
                                limits=verifier_limits,
                            )
                            if verifier_resources is not None or verifier_limits is not None
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

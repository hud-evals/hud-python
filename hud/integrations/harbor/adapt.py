"""Adapt Harbor task directories into runnable HUD environments."""

from __future__ import annotations

import hashlib
import json
import logging
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
from hud.eval.compose import ComposeConfig, ComposeHealthcheck, ComposeService
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeResources
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
    instruction: str
    environment_hash: str
    compose: ComposeConfig | None


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(root.rglob("*")):
        relative_path = entry.relative_to(root).as_posix().encode()
        if entry.is_symlink():
            digest.update(relative_path + b"\0symlink\0" + os.readlink(entry).encode())
        elif entry.is_file():
            digest.update(relative_path + b"\0" + entry.read_bytes())
    return digest.hexdigest()[:16]


def adapt(
    path: str | Path,
    *,
    hud_requirement: str = "hud",
) -> Taskset:
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
            try:
                compose = ComposeConfig.from_file(authored_compose)
                compose.services["main"]
            except (ValidationError, KeyError) as error:
                raise ValueError(f"{task_dir.name} did not resolve to a Compose project") from error
            compose.name = None

        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            raise FileNotFoundError(f"{task_dir.name} has no instruction.md")

        tasks.append(
            HarborTask(
                path=task_dir,
                config=config,
                instruction=instruction.read_text("utf-8"),
                environment_hash=_tree_hash(environment_dir)
                if environment_dir.exists()
                else "missing",
                compose=compose,
            )
        )

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
        compose_main = compose.services["main"] if compose is not None else ComposeService()
        dockerfile = source.path / "environment" / "Dockerfile"
        base_image = environment.docker_image or compose_main.image
        if compose is not None:
            build = compose_main.build
            if build is not None:
                build_config = {"context": build} if isinstance(build, str) else build
                build_context = build_config.get("context", ".")
                build_dockerfile = build_config.get("dockerfile", "Dockerfile")
                if not isinstance(build_context, str) or not isinstance(build_dockerfile, str):
                    raise ValueError("Compose main build paths must be strings")
                dockerfile = (
                    source.path / "environment" / build_context / build_dockerfile
                ).resolve()
                try:
                    dockerfile.relative_to((source.path / "environment").resolve())
                except ValueError:
                    raise ValueError("Compose main build escapes environment") from None
            if dockerfile.is_file():
                base_image = f"hud-harbor-base:{source.environment_hash}"
            elif build is not None:
                raise FileNotFoundError(
                    f"{source.path.name} Compose main Dockerfile does not exist"
                )
            elif base_image is None:
                raise FileNotFoundError(
                    f"{source.path.name} Compose main has neither image nor build"
                )
        elif dockerfile.is_file():
            base_image = f"hud-harbor-base:{source.environment_hash}"
        elif base_image is None:
            raise FileNotFoundError(
                f"{source.path.name} has neither environment/Dockerfile nor docker_image"
            )

        separate = source.config.verifier.separate
        verifier_environment = source.config.verifier.environment or EnvironmentConfig()
        verifier_image = base_image
        if separate:
            verifier_dockerfile = source.path / "tests" / "Dockerfile"
            if not verifier_dockerfile.is_file():
                raise FileNotFoundError(
                    f"{source.path.name} uses a separate verifier but has no tests/Dockerfile"
                )
            verifier_image = f"hud-harbor-verifier:{name}-{_tree_hash(verifier_dockerfile.parent)}"

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
                if service.build is None and service.image is None:
                    raise ValueError(
                        f"Compose service {service_name!r} has neither image nor build"
                    )
                if len(ports) > 1:
                    raise NotImplementedError(
                        f"Compose service {service_name!r} exposes multiple ports; "
                        "Peer names one endpoint"
                    )
                if not ports:
                    raise ValueError(f"Compose service {service_name!r} declares no TCP port")
                peers.append({"name": service_name, "port": next(iter(ports))})
        context = dataset / ".hud-adapt" / name
        if context.exists():
            shutil.rmtree(context)
        project = context / "compose-project"
        payload = project / ("main" if compose is not None else "hud")
        (payload / "packages").mkdir(parents=True)
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
        if workdir is not None and Path(workdir).is_relative_to(HUD_ROOT):
            raise ValueError(f"Harbor workdir {workdir!r} is inside reserved path {HUD_ROOT}")
        ports = {
            int(port)
            for exposed in compose_main.expose
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
            stages: list[tuple[int, re.Match[str]]] = []
            from_pattern = re.compile(
                r"^(?P<from>\s*FROM\s+(?:--platform=\S+\s+)?\S+)"
                r"(?P<alias>\s+AS\s+(?P<name>[A-Za-z0-9_.-]+))?"
                r"(?P<suffix>\s*(?:#.*)?)$",
                re.IGNORECASE,
            )
            for index, raw_line in enumerate(lines):
                line = raw_line.rstrip("\r\n")
                if not re.match(r"^\s*FROM\b", line, re.IGNORECASE):
                    continue
                match = from_pattern.fullmatch(line)
                if match is None:
                    raise ValueError(
                        f"{source.path.name} environment/Dockerfile has an unsupported "
                        "multi-line FROM instruction"
                    )
                stages.append((index, match))
            if not stages:
                raise ValueError(f"{source.path.name} environment/Dockerfile has no FROM stage")
            stage_names = {
                match.group("name").lower()
                for _, match in stages
                if match.group("name") is not None
            }
            reserved_names = {"hud-base", "hud-runtime"}
            if separate:
                reserved_names.update({"hud-docker-cli", "hud-verifier", "hud-verifier-root"})
            reserved = reserved_names & stage_names if dockerfile.is_file() else set()
            if reserved:
                raise ValueError(
                    f"{source.path.name} environment/Dockerfile uses reserved stage "
                    f"{min(reserved)!r}"
                )
            final_index, final = stages[-1]
            base_stage = final.group("name")
            if base_stage is None:
                ending = lines[final_index][len(lines[final_index].rstrip("\r\n")) :]
                lines[final_index] = (
                    f"{final.group('from')} AS hud-base{final.group('suffix')}{ending}"
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
            if "hud-base" in compose_project.services or "hud-verifier" in compose_project.services:
                raise ValueError("Compose service names 'hud-base' and 'hud-verifier' are reserved")
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
            elif base_image is None:
                raise ValueError("Compose main service requires an image or build")

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
                "target": "verifier" if separate else "plain",
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
  rm -f {image_config_path} {verifier_config_path}
}}
trap cleanup EXIT HUP INT TERM
inspect_image() {{
  docker image inspect --format '{{{{json .Config}}}}' "$1"
}}
{prepare_base}
inspect_image {shlex.quote(base_image)} > {image_config_path}
{prepare_verifier}
{inspect_verifier}
{compose_command} build
if [ "$#" -gt 0 ]; then
  docker tag "$({compose_command} images -q main)" "$1"
fi
"""
        script = project / "build.sh"
        script.write_text(build_script, encoding="utf-8", newline="\n")
        script.chmod(0o755)
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
                "artifacts": [artifact.model_dump() for artifact in config.artifacts],
            }
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
                    compose=context / "compose-project" / "compose.json",
                    compose_project=context,
                    compose_service_access=(True if task_separate else None),
                    resources=resources if resources.model_dump(exclude_none=True) else None,
                ),
                verifier=(
                    Task(
                        env=name,
                        id="verify",
                        args={"task": task_config},
                        slug=f"{task.path.name}:verify",
                    )
                    if task_separate
                    else None
                ),
            )
            rows.append(row)
            group_rows.append(row)
        Taskset(dataset.name, group_rows).to_file(context / "tasks.json")

    LOGGER.info("adapted %d Harbor project(s)", len({task.env for task in rows}))
    return Taskset(dataset.name, rows, origin=f"harbor:{dataset}")

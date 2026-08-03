"""Build Harbor task directories as runnable HUD environments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hud.eval import Task, Taskset
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


class Phase(BaseModel):
    model_config = ConfigDict(extra="allow")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: NetworkMode | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, Any] | None = None
    environment_mode: str | None = None


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
    healthcheck: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None


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
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agent: Phase = Field(default_factory=Phase)
    verifier: Phase = Field(default_factory=Phase)
    steps: list[dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class HarborTask:
    path: Path
    config: TaskConfig
    environment_hash: str
    runtime: dict[str, Any]


async def adapt(
    path: str | Path,
    *,
    push: str | None = None,
    hud_requirement: str = "hud",
) -> Taskset:
    """Build a runnable HUD image for each distinct Harbor environment."""
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
        if config.environment.healthcheck:
            unsupported.append("healthcheck")
        if config.environment.mcp_servers:
            unsupported.append("MCP servers")
        if config.environment.skills_dir:
            unsupported.append("skills_dir")
        if config.verifier.environment_mode == "separate" or config.verifier.environment:
            unsupported.append("a separate verifier environment")
        if config.steps:
            unsupported.append("multi-step tasks")
        if any(
            (task_dir / "environment" / filename).is_file()
            for filename in (
                "compose.yaml",
                "compose.yml",
                "docker-compose.yaml",
                "docker-compose.yml",
            )
        ):
            unsupported.append("Docker Compose")
        if unsupported:
            raise NotImplementedError(
                f"Harbor task {task_dir.name!r} uses unsupported features: "
                + ", ".join(unsupported)
            )

        environment = config.environment
        tasks.append(
            HarborTask(
                path=task_dir,
                config=config,
                environment_hash=_tree_hash(task_dir / "environment"),
                runtime={
                    "image": environment.docker_image,
                    "workdir": environment.workdir,
                    "environment_env": environment.env,
                    "environment_network": environment.network_mode,
                    "environment_hosts": environment.allowed_hosts,
                    "agent": config.agent.model_dump(
                        include={"user", "network_mode", "allowed_hosts", "env"}
                    ),
                    "verifier": config.verifier.model_dump(
                        include={"user", "network_mode", "allowed_hosts", "env"}
                    ),
                },
            )
        )

    grouped: dict[tuple[str, str], list[HarborTask]] = {}
    for task in tasks:
        runtime_json = json.dumps(task.runtime, sort_keys=True)
        grouped.setdefault((task.environment_hash, runtime_json), []).append(task)

    rows = []
    base_name = normalize_environment_name(dataset.name, default="harbor")
    for (environment_hash, runtime_json), group in sorted(grouped.items()):
        digest = hashlib.sha256((environment_hash + "\0" + runtime_json).encode()).hexdigest()[:12]
        name = f"{base_name}-{digest}"
        source = group[0]
        dockerfile = source.path / "environment" / "Dockerfile"
        base_image = source.runtime["image"]
        if base_image and not dockerfile.is_file():
            await docker("pull", base_image)
        elif dockerfile.is_file():
            base_image = f"hud-harbor-base:{source.environment_hash}"
            await docker(
                "build",
                "--tag",
                base_image,
                str(dockerfile.parent),
                deadline=max(task.config.environment.build_timeout_sec for task in group),
            )
        else:
            raise FileNotFoundError(f"{source.path.name} has no environment/Dockerfile")

        output, _ = await docker(
            "image",
            "inspect",
            "--format",
            "{{json .Config}}",
            base_image,
        )
        image_config = json.loads(output)
        context = dataset / ".hud-adapt" / name
        if context.exists():
            shutil.rmtree(context)
        (context / "tasks").mkdir(parents=True)
        (context / "packages").mkdir()
        for asset in ("Dockerfile", "env.py", "install.sh"):
            shutil.copy2(ASSETS / asset, context / asset)

        workdir = source.runtime["workdir"] or image_config.get("WorkingDir") or "/"
        if Path(workdir).is_relative_to(HUD_ROOT):
            raise ValueError(f"Harbor workdir {workdir!r} is inside reserved path {HUD_ROOT}")
        manifest = {
            "name": name,
            "workdir": workdir,
            "image_user": image_config.get("User") or None,
            "environment": {
                "env": source.runtime["environment_env"],
                "network_mode": source.runtime["environment_network"],
                "allowed_hosts": source.runtime["environment_hosts"],
            },
            "agent": source.runtime["agent"],
            "verifier": source.runtime["verifier"],
            "tasks": [],
        }
        for task in group:
            if not (task.path / "instruction.md").is_file():
                raise FileNotFoundError(f"{task.path.name} has no instruction.md")
            target = context / "tasks" / task.path.name
            target.mkdir()
            shutil.copy2(task.path / "instruction.md", target / "instruction.md")
            shutil.copytree(task.path / "tests", target / "tests", symlinks=True, ignore=IGNORED)
            manifest["tasks"].append(
                {
                    "id": task.path.name,
                    "description": task.config.task.description,
                    "verifier_timeout": task.config.verifier.timeout_sec or 600.0,
                }
            )
        (context / "tasks.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        wheel = Path(hud_requirement)
        requirement = hud_requirement
        if wheel.suffix == ".whl" and wheel.is_file():
            shutil.copy2(wheel, context / "packages" / wheel.name)
            requirement = f"{HUD_ROOT}/packages/{wheel.name}"

        tag = _tree_hash(context)
        image = f"{push}/{name}:{tag}" if push else f"hud-harbor:{name}-{tag}"
        await docker(
            "build",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"HUD_REQUIREMENT={requirement}",
            "--tag",
            image,
            str(context),
        )
        if push:
            await docker("push", image)

        for task in group:
            config = task.config
            resources = RuntimeResources(
                cpu=config.environment.cpus,
                memory_mb=config.environment.memory_mb,
                gpu=(
                    RuntimeGPU(
                        count=config.environment.gpus,
                        type=next(iter(filter(None, config.environment.gpu_types)), None),
                    )
                    if config.environment.gpus
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
                        image=image,
                        resources=resources if resources.model_dump(exclude_none=True) else None,
                    ),
                )
            )

    LOGGER.info("adapted %d Harbor image(s)", len({task.env for task in rows}))
    return Taskset(dataset.name, rows, origin=f"harbor:{dataset}")


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return "missing"
    for entry in sorted(path.rglob("*")):
        name = entry.relative_to(path).as_posix().encode()
        if entry.is_symlink():
            digest.update(name + b"\0symlink\0" + os.readlink(entry).encode())
        elif entry.is_file():
            digest.update(name + b"\0" + entry.read_bytes())
    return digest.hexdigest()[:16]

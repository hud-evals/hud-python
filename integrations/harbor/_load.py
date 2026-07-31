"""Harbor task-dir parsing and loading: dirs -> Taskset rows with provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hud.environment.egress import ANY_HOST
from hud.eval import Task, Taskset
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeResources
from hud.utils.naming import normalize_environment_name

LOGGER = logging.getLogger(__name__)

#: Where an adapted image keeps the harness: its venv, the baked tasks, the
#: session keys, the grading material. Nested rather than at the root, and
#: rebuilt out of the agent's namespace by :func:`~harbor.environment`,
#: so what the graded party sees is the empty ``/media`` any container has —
#: not a directory named after the thing evaluating it.
HUD_ROOT = Path("/media/hud")

DEFAULT_VERIFIER_TIMEOUT = 600.0


class _Phase(BaseModel):
    """A phase Harbor runs: ``[agent]`` or ``[verifier]``."""

    model_config = ConfigDict(extra="allow")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, Any] | None = None
    environment_mode: str | None = None


class _EnvironmentSection(BaseModel):
    """``[environment]``: how the task's container is built and run."""

    model_config = ConfigDict(extra="allow")

    build_timeout_sec: float | None = Field(default=None, gt=0)
    docker_image: str | None = None
    os: str | None = None
    cpus: float | None = Field(default=None, ge=0)
    memory_mb: int | None = Field(default=None, ge=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] = Field(default_factory=list)
    tpu: dict[str, Any] | None = None
    network_mode: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    healthcheck: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)


class _Package(BaseModel):
    """``[task]``: what the task calls itself."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """A task's ``task.toml``, as much of it as this integration consumes.

    The schema carries the coercion — a positive-number constraint instead of
    a hand-rolled check, a typed table instead of ``isinstance(x, dict)`` at
    every read — so the rest of the integration reads attributes. Unknown
    keys are kept rather than rejected: Harbor's format grows, and a field
    this integration does not consume is not an error.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str | None = None
    task: _Package = Field(default_factory=_Package)
    metadata: dict[str, Any] = Field(default_factory=dict)
    environment: _EnvironmentSection = Field(default_factory=_EnvironmentSection)
    agent: _Phase = Field(default_factory=_Phase)
    verifier: _Phase = Field(default_factory=_Phase)
    steps: list[dict[str, Any]] | None = None

    @classmethod
    def read(cls, task_dir: Path) -> TaskConfig:
        """Parse *task_dir*'s ``task.toml``.

        A file that will not parse declares nothing — its defaults apply. A
        file that parses but declares something invalid (a GPU count that is
        not a number, a timeout of zero) is an error: running it would grade
        a task under requirements its author did not write.
        """
        try:
            raw = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return cls()
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise ValueError(
                f"{task_dir.name}/task.toml is not a valid Harbor task: {error}"
            ) from error

    def phase(self, role: str) -> _Phase:
        if role == "agent":
            return self.agent
        if role == "verifier":
            return self.verifier
        raise ValueError(f"unknown Harbor phase {role!r}")

    def network(self, role: str) -> bool:
        """Whether *role*'s processes may reach the network at all.

        Harbor declares isolation container-wide or per phase; either severs
        that phase, so the workspace and the verifier cannot disagree about
        what one task declared.
        """
        return "no-network" not in (self.environment.network_mode, self.phase(role).network_mode)

    def allowed_hosts(self, role: str) -> frozenset[str]:
        """Which hosts *role* may reach.

        A phase declaring ``allowlist`` may reach the hosts it names and
        nothing else — including nothing else on the substrate, which is where
        the harness serving it lives. ``public`` names no hosts and so permits
        all of them; ``no-network`` is the empty set, which permits none.
        Every phase gets an answer: a task that says nothing about its network
        is still held to something, rather than sharing the substrate's.
        """
        mode = self.phase(role).network_mode or self.environment.network_mode or "public"
        if mode == "no-network":
            return frozenset()
        if mode != "allowlist":
            # Public still means every host, but reached the same way as any
            # other policy: through the workspace's own way out. Sharing the
            # substrate's network would make "public" mean the substrate's
            # services too — the channel that grades the rollout among them.
            return frozenset({ANY_HOST})
        declared = [*self.environment.allowed_hosts, *self.phase(role).allowed_hosts]
        return frozenset(declared)

    def phase_user(self, role: str) -> str | int | None:
        """The identity *role* runs as, if the task names one for it.

        Per phase, because Harbor's are: a task may hand the agent a
        restricted account and still verify as root. They collapsed into one
        value while the identity was a ``USER`` directive, which an image has
        only one of; applied per phase, each keeps what it declared.
        """
        return self.phase(role).user


def detect(path: str | Path) -> bool:
    """True when *path* is a Harbor task dir or a dataset of them."""
    return bool(task_dirs(path))


def load(path: str | Path) -> Taskset:
    """Load a Harbor task dir (or dataset dir) into a :class:`Taskset`.

    One row per task dir (``id`` = the dir name); rows share one env name per
    distinct environment. Each row carries the task's declared launch
    requirements (cpu/memory/gpu). Adapted images are bound by :func:`adapt`
    before the returned taskset is run::

        taskset = await harbor.adapt(path)
        job = await taskset.run(agent, runtime=DockerRuntime())
    """
    return _load(path)


def _load(path: str | Path, *, images: dict[str, str] | None = None) -> Taskset:
    """Load rows, optionally binding image refs produced by :func:`adapt`."""
    root = Path(path).resolve()
    dataset_name = root.parent.name if is_harbor_task(root) else root.name

    tasks: list[Task] = []
    for env_name, group in _task_groups(root):
        image = (images or {}).get(env_name)
        tasks.extend(
            Task(
                env=env_name,
                id=task.path.name,
                columns=task.columns,
                runtime_config=task.runtime_config(image=image),
            )
            for task in group
        )
    return Taskset(slugify(dataset_name), tasks, origin=f"harbor:{root}")


@dataclass(frozen=True)
class FinalStage:
    """What a Dockerfile's last stage declares.

    Adaptation appends to the final stage, so only that stage's state is
    meaningful: a build stage's ``ENTRYPOINT`` starts nothing in the shipped
    image, and its ``USER`` is not the shipped identity. Each ``FROM`` opens
    a new stage whose inherited state comes from its base image — unknowable
    from the text, so it reads as unset.
    """

    directives: frozenset[str] = frozenset()
    user: str | None = None
    workdir: str | None = None


def dockerfile_instructions(dockerfile_text: str) -> list[tuple[str, str, list[int]]]:
    """Logical Dockerfile instructions: ``(word, operand, line numbers)``.

    A backslash continues an instruction onto the next line, so a physical
    line is not an instruction — treating it as one both misreads operands
    and, when rewriting, leaves continuation lines behind as invalid
    top-level directives. A heredoc body is likewise not instructions: a
    ``RUN <<EOF`` that writes another Dockerfile would otherwise read as
    ``FROM``/``USER`` directives of this one.
    """
    instructions: list[tuple[str, str, list[int]]] = []
    pending: list[str] = []
    numbers: list[int] = []
    heredocs: list[str] = []
    for number, raw in enumerate(dockerfile_text.splitlines()):
        stripped = raw.strip()
        if heredocs:
            numbers.append(number)
            if stripped == heredocs[0]:
                heredocs.pop(0)
            if not heredocs and not pending:
                instructions[-1] = (*instructions[-1][:2], [*instructions[-1][2], *numbers])
                numbers = []
            continue
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        numbers.append(number)
        continued = stripped.endswith("\\")
        pending.append(stripped[:-1] if continued else stripped)
        if continued:
            continue
        joined = " ".join(part for part in pending if part)
        word, _, rest = joined.partition(" ")
        instructions.append((word.upper(), rest.strip(), numbers))
        # ``<<EOF`` / ``<<-"EOF"`` open bodies that end at their delimiter.
        heredocs = [
            match.group("tag").strip("\"'")
            for match in re.finditer(r"<<-?\s*(?P<tag>[\"']?[A-Za-z_][A-Za-z0-9_]*[\"']?)", joined)
        ]
        pending, numbers = [], []
        if heredocs:
            numbers = []
    if pending:  # trailing backslash at EOF
        joined = " ".join(part for part in pending if part)
        word, _, rest = joined.partition(" ")
        instructions.append((word.upper(), rest.strip(), numbers))
    return instructions


def final_stage(dockerfile_text: str) -> FinalStage:
    """Parse *dockerfile_text* into its :class:`FinalStage`."""
    directives: set[str] = set()
    user: str | None = None
    workdir: str | None = None
    for word, operand, _ in dockerfile_instructions(dockerfile_text):
        if word == "FROM":
            directives, user, workdir = set(), None, None
        elif word == "USER":
            user = operand or None
        elif word == "WORKDIR":
            workdir = operand.strip().strip('"') or None
        directives.add(word)
    return FinalStage(
        frozenset(directives),
        None if user in ("root", "0", "root:root", "0:0") else user,
        workdir,
    )


@dataclass(frozen=True, slots=True)
class HarborTask:
    """The parsed, immutable view of one Harbor task directory.

    Loading, grouping, image adaptation, and runtime setup all consume this
    same record. That keeps task.toml and Dockerfile parsing at one boundary
    instead of making each phase rediscover the task from its path.
    """

    path: Path
    config: TaskConfig
    dockerfile: str
    final_stage: FinalStage
    environment_hash: str

    @classmethod
    def read(cls, task_dir: Path) -> HarborTask:
        path = Path(task_dir).resolve()
        dockerfile_path = path / "environment" / "Dockerfile"
        dockerfile = (
            dockerfile_path.read_text("utf-8", errors="replace")
            if dockerfile_path.is_file()
            else ""
        )
        environment = path / "environment"
        return cls(
            path=path,
            config=TaskConfig.read(path),
            dockerfile=dockerfile,
            final_stage=final_stage(dockerfile),
            environment_hash=hash_directory(environment) if environment.exists() else "no-env",
        )

    @property
    def columns(self) -> dict[str, Any] | None:
        fields = dict(self.config.metadata)
        if self.config.task.keywords:
            fields.setdefault("keywords", self.config.task.keywords)
        return fields or None

    def runtime_config(self, *, image: str | None = None) -> RuntimeConfig | None:
        """Map portable launch requirements into the SDK's runtime config."""
        environment = self.config.environment
        resources = RuntimeResources(
            cpu=environment.cpus or None,
            memory_mb=environment.memory_mb or None,
            gpu=RuntimeGPU(
                count=environment.gpus,
                type=next((gpu_type for gpu_type in environment.gpu_types if gpu_type), None),
            )
            if environment.gpus
            else None,
        )
        declared = RuntimeConfig(
            image=image,
            resources=resources if resources.model_dump(exclude_none=True) else None,
        )
        return declared if declared.model_dump(exclude_none=True) else None

    @property
    def workspace_key(self) -> str:
        """Stable serialization of the workspace contract used for grouping."""
        config = self.config
        return json.dumps(
            {
                "network": config.network("agent"),
                "env": dict(config.environment.env),
                "agent_env": dict(config.agent.env),
                "workdir": config.environment.workdir or None,
                "user": (
                    config.agent.user if config.agent.user is not None else config.verifier.user
                ),
                "allowed_hosts": sorted(config.allowed_hosts("agent")),
                "verifier_allowed_hosts": sorted(config.allowed_hosts("verifier")),
                "agent_user": config.phase_user("agent"),
                "verifier_user": config.phase_user("verifier"),
            },
            sort_keys=True,
        )

    def unsupported_features(self) -> list[str]:
        """Declarations this integration cannot reproduce faithfully."""
        config = self.config
        environment, agent, verifier = config.environment, config.agent, config.verifier
        reasons: list[str] = []

        if environment.os not in (None, "linux"):
            reasons.append(f"environment.os={environment.os!r}")
        if environment.tpu:
            reasons.append("environment.tpu (no TPU resource model)")
        if agent.user is not None and verifier.user is not None and agent.user != verifier.user:
            reasons.append("agent.user and verifier.user differ (the image has one USER)")
        workdir = environment.workdir or self.final_stage.workdir
        if workdir and Path(workdir).is_relative_to(HUD_ROOT):
            reasons.append(f"working directory {workdir!r} is inside {HUD_ROOT} (reserved)")
        if environment.docker_image and not (self.path / "environment" / "Dockerfile").is_file():
            reasons.append(
                "prebuilt docker_image environments (adapt builds from environment/Dockerfile)"
            )
        if "ENTRYPOINT" in self.final_stage.directives:
            reasons.append(
                "environment/Dockerfile ENTRYPOINT (adaptation replaces container startup)"
            )
        if any(
            (self.path / "environment" / name).is_file()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        ):
            reasons.append("compose environments (sidecar services would never start)")
        if environment.healthcheck:
            reasons.append("environment.healthcheck (nothing starts the services it would await)")
        if environment.mcp_servers:
            reasons.append("environment.mcp_servers (nothing starts the servers they point at)")
        if verifier.environment_mode or verifier.environment:
            reasons.append("verifier runs in its own environment")
        if config.steps:
            reasons.append("multi-step tasks ([[steps]])")
        return reasons


def _task_groups(root: str | Path) -> list[tuple[str, list[HarborTask]]]:
    resolved = Path(root).resolve()
    dataset_name = resolved.parent.name if is_harbor_task(resolved) else resolved.name
    dirs = task_dirs(resolved)
    if not dirs:
        raise ValueError(f"no Harbor tasks found in {root}")

    groups: dict[tuple[str, str], list[HarborTask]] = {}
    for task in (HarborTask.read(task_dir) for task_dir in dirs):
        groups.setdefault((task.environment_hash, task.workspace_key), []).append(task)
    base_name = slugify(dataset_name)
    return sorted(
        (
            f"{base_name}-{_group_digest(environment_hash, workspace_key)}",
            group,
        )
        for (environment_hash, workspace_key), group in groups.items()
    )


def _group_digest(environment_hash: str, workspace_key: str) -> str:
    """Short stable digest of one group's build and workspace contract."""
    return hashlib.sha256(f"{environment_hash}\0{workspace_key}".encode()).hexdigest()[:12]


# ─── task-dir primitives ────────────────────────────────────────────────


def slugify(name: str) -> str:
    """A valid env name from a dataset dir name — the SDK's own normalizer,
    so a row's env name matches what a deploy of the same context registers."""
    return normalize_environment_name(name, default="harbor")


def is_harbor_task(path: Path) -> bool:
    """A ``task.toml`` plus an instruction: at the root for a single-step
    task, or per ``[[steps]]`` entry for a multi-step one."""
    if not path.is_dir() or not (path / "task.toml").exists():
        return False
    if (path / "instruction.md").is_file():
        return True
    try:  # a multi-step task is only recognizable from its config
        return bool(TaskConfig.read(path).steps)
    except ValueError:
        return False


def task_dirs(path: str | Path) -> list[Path]:
    """The task dirs under *path*: itself when it is one, else its children."""
    root = Path(path)
    if is_harbor_task(root):
        return [root]
    if root.is_dir():
        return sorted(d for d in root.iterdir() if d.is_dir() and is_harbor_task(d))
    return []


def hash_directory(path: Path) -> str:
    """Content-hash a directory for grouping tasks by identical environments."""
    hasher = hashlib.sha256()
    if not path.exists():
        return "empty"
    for entry in sorted(path.rglob("*")):
        name = str(entry.relative_to(path)).encode()
        if entry.is_symlink():
            # The link itself is the content. Reading through it would make
            # the hash depend on host state outside the context — and could
            # walk out of it entirely.
            hasher.update(name)
            hasher.update(b"\0symlink\0")
            hasher.update(os.readlink(entry).encode())
        elif entry.is_file():
            hasher.update(name)
            hasher.update(entry.read_bytes())
    return hasher.hexdigest()[:16]

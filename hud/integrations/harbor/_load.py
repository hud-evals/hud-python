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

from hud.eval import Task, Taskset
from hud.eval.runtime import RuntimeConfig, RuntimeGPU, RuntimeResources
from hud.utils.naming import normalize_environment_name

LOGGER = logging.getLogger(__name__)

DEFAULT_VERIFIER_TIMEOUT = 600.0


class _Phase(BaseModel):
    """A phase Harbor runs: ``[agent]`` or ``[verifier]``."""

    model_config = ConfigDict(extra="allow")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: str | None = None
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
        return self.agent if role == "agent" else self.verifier

    def network(self, role: str) -> bool:
        """Whether *role*'s processes may reach the network.

        Harbor declares isolation container-wide or per phase; either severs
        that phase, so the workspace and the verifier cannot disagree about
        what one task declared.
        """
        return "no-network" not in (self.environment.network_mode, self.phase(role).network_mode)

    @property
    def user(self) -> str | int | None:
        """The identity the task's phases run as, if it names one."""
        return self.agent.user if self.agent.user is not None else self.verifier.user


def detect(path: str | Path) -> bool:
    """True when *path* is a Harbor task dir or a dataset of them."""
    return bool(task_dirs(path))


def load(path: str | Path, *, images: dict[str, str] | None = None) -> Taskset:
    """Load a Harbor task dir (or dataset dir) into a :class:`Taskset`.

    One row per task dir (``id`` = the dir name); rows share one env name per
    distinct ``environment/`` build context (content-hashed), derived from
    the dataset name. Each row carries the task's declared launch
    requirements (:func:`runtime_config`: cpu/memory/gpu and time budgets),
    plus the adapted image ref once
    :func:`~hud.integrations.harbor.adapt` has produced it, so it runs on any
    container placement::

        await harbor.adapt(path)
        job = await harbor.load(path).run(agent, runtime=DockerRuntime())
    """
    root = Path(path).resolve()
    dataset_name = root.parent.name if is_harbor_task(root) else root.name
    if not task_dirs(root):
        raise ValueError(f"no Harbor tasks found in {path}")

    tasks: list[Task] = []
    for env_name, group_dirs in grouped(root):
        image = (images or {}).get(env_name)
        tasks.extend(
            Task(
                env=env_name,
                id=task_dir.name,
                columns=columns(task_dir),
                runtime_config=runtime_config(task_dir, image=image),
            )
            for task_dir in group_dirs
        )
    return Taskset(slugify(dataset_name), tasks, origin=f"harbor:{root}")


def columns(task_dir: Path) -> dict[str, Any] | None:
    """The task's ``[metadata]`` (plus keywords) as filterable columns."""
    config = TaskConfig.read(task_dir)
    fields = dict(config.metadata)
    if config.task.keywords:
        fields.setdefault("keywords", config.task.keywords)
    return fields or None


def workspace_policy(task_dir: Path) -> dict[str, Any]:
    """What the task declares about the workspace its agent works in.

    Grouping keys on this, so tasks that mean different things never share
    an environment. Only settings this integration can honor appear here —
    see :func:`unsupported_features` for the rest.
    """
    config = TaskConfig.read(task_dir)
    return {
        "network": config.network("agent"),
        # Container-wide variables reach every process (baked as image ENV);
        # the agent phase's reach only its sessions, and the verifier's are
        # applied where the verifier runs. Each phase sees what Harbor scoped
        # to it, and all three are in this key so tasks that differ never
        # share an environment.
        "env": dict(config.environment.env),
        "agent_env": dict(config.agent.env),
        "workdir": config.environment.workdir or None,
        "user": config.user,
    }


def runtime_config(task_dir: Path, *, image: str | None = None) -> RuntimeConfig | None:
    """The task's declared launch requirements as HUD's portable config.

    ``storage_mb`` has no portable counterpart and is dropped; time budgets
    bound the *rollout*, not the substrate, so they stay out of here (see
    :func:`agent_timeout`).
    """
    environment = TaskConfig.read(task_dir).environment
    resources = RuntimeResources(
        cpu=environment.cpus or None,
        memory_mb=environment.memory_mb or None,
        gpu=RuntimeGPU(
            count=environment.gpus,
            type=next((t for t in environment.gpu_types if t), None),
        )
        if environment.gpus
        else None,
    )
    declared = RuntimeConfig(
        image=image,
        resources=resources if resources.model_dump(exclude_none=True) else None,
    )
    return declared if declared.model_dump(exclude_none=True) else None


def agent_timeout(task_dir: Path) -> float | None:
    """How long the task allows the agent to work (``[agent] timeout_sec``).

    A rollout budget, not a launch requirement: pass it as ``rollout_timeout``
    when running the row.
    """
    return TaskConfig.read(task_dir).agent.timeout_sec or None


def unsupported_features(task_dir: Path) -> list[str]:
    """Declarations this integration cannot reproduce faithfully.

    A wrong score is worse than a refused task, so each of these names itself
    rather than being silently dropped.
    """
    config = TaskConfig.read(task_dir)
    environment, agent, verifier = config.environment, config.agent, config.verifier
    reasons: list[str] = []

    for role, mode in (
        ("environment", environment.network_mode),
        ("agent", agent.network_mode),
        ("verifier", verifier.network_mode),
    ):
        if mode == "allowlist":
            reasons.append(f"{role}.network_mode='allowlist' (per-host policy is not enforceable)")
    if environment.os not in (None, "linux"):
        reasons.append(f"environment.os={environment.os!r}")
    if environment.tpu:
        reasons.append("environment.tpu (no TPU resource model)")
    if agent.user is not None and verifier.user is not None and agent.user != verifier.user:
        # One image, one USER. Only one phase naming an identity is fine —
        # both phases run as it; two *different* identities are not.
        reasons.append("agent.user and verifier.user differ (the image has one USER)")
    workdir = environment.workdir or _final_stage_workdir(task_dir)
    if workdir and (workdir == "/hud" or workdir.startswith("/hud/")):
        # The adaptation layer owns /hud inside the image and hides it from
        # agent sessions; a task working there would find it empty.
        reasons.append(f"working directory {workdir!r} is inside /hud (reserved by adaptation)")
    if environment.docker_image and not (task_dir / "environment" / "Dockerfile").is_file():
        reasons.append(
            "prebuilt docker_image environments (adapt builds from environment/Dockerfile)"
        )

    # Everything below describes the container's own boot process, which
    # adaptation replaces with the serving CMD: services an ENTRYPOINT would
    # start never start, so healthchecks would await nothing and MCP server
    # URLs would point at nothing.
    dockerfile = task_dir / "environment" / "Dockerfile"
    directives = (
        final_stage(dockerfile.read_text("utf-8", errors="replace")).directives
        if dockerfile.is_file()
        else frozenset()
    )
    if "ENTRYPOINT" in directives:
        reasons.append("environment/Dockerfile ENTRYPOINT (adaptation replaces container startup)")
    if any(
        (task_dir / "environment" / name).is_file()
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


def _final_stage_workdir(task_dir: Path) -> str | None:
    """The ``WORKDIR`` the image's final stage ends in, if it sets one."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    workdir = None
    for word, operand, _ in dockerfile_instructions(
        dockerfile.read_text("utf-8", errors="replace")
    ):
        if word == "FROM":
            workdir = None
        elif word == "WORKDIR":
            workdir = operand.strip().strip('"') or None
    return workdir


def final_stage(dockerfile_text: str) -> FinalStage:
    """Parse *dockerfile_text* into its :class:`FinalStage`."""
    directives: set[str] = set()
    user: str | None = None
    for word, operand, _ in dockerfile_instructions(dockerfile_text):
        if word == "FROM":
            directives, user = set(), None
        elif word == "USER":
            user = operand or None
        directives.add(word)
    return FinalStage(
        frozenset(directives), None if user in ("root", "0", "root:root", "0:0") else user
    )


def grouped(root: str | Path) -> list[tuple[str, list[Path]]]:
    """Task dirs grouped by identical ``environment/`` content, largest first.

    One env name per group (the dataset slug, ``-gN``-suffixed when there are
    several): the join key between :func:`load`'s rows and
    :func:`~hud.integrations.harbor.adapt`'s images.
    """
    resolved = Path(root).resolve()
    dataset_name = resolved.parent.name if is_harbor_task(resolved) else resolved.name
    dirs = task_dirs(resolved)
    if not dirs:
        raise ValueError(f"no Harbor tasks found in {root}")

    groups: dict[tuple[str, str], list[Path]] = {}
    for task_dir in dirs:
        env_dir = task_dir / "environment"
        env_hash = hash_directory(env_dir) if env_dir.exists() else "no-env"
        # Tasks sharing a build context still need separate envs when their
        # declared workspace behaviour differs: one env serves one policy.
        # Invariant: everything environment() consumes per group is either in
        # this key (workspace_policy) or refused (unsupported_features) — a
        # declaration outside both would silently take the first task's value
        # for the whole group.
        policy = json.dumps(workspace_policy(task_dir), sort_keys=True)
        groups.setdefault((env_hash, policy), []).append(task_dir)
    ordered = sorted(groups.values(), key=lambda group: -len(group))
    base_name = slugify(dataset_name)
    if len(ordered) == 1:
        return [(base_name, ordered[0])]
    return [(f"{base_name}-g{idx}", group) for idx, group in enumerate(ordered, start=1)]


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

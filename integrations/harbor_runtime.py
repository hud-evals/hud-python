"""Local Docker-backed runtime for Harbor task directories."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
import uuid
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves this at runtime.
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from hud.environment import Environment
from hud.environment.workspace import Workspace
from integrations.harbor_common import _hash_directory, _slugify, _task_dirs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Mapping

    from hud.eval import Task
    from hud.eval.runtime import Runtime


class _PhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    network_mode: Literal["public", "no-network", "allowlist"] | None = None
    allowed_hosts: list[str] | None = None


class _EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_timeout_sec: float = Field(default=600.0, gt=0)
    docker_image: str | None = Field(default=None, min_length=1)
    os: Literal["linux", "windows"] = "linux"
    cpus: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] | None = None
    tpu: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None
    healthcheck: dict[str, Any] | None = None
    workdir: str | None = None
    network_mode: Literal["public", "no-network", "allowlist"] = "public"
    allowed_hosts: list[str] | None = None
    allow_internet: bool | None = None

    @field_validator("os", mode="before")
    @classmethod
    def _normalize_os(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_sizes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        for legacy, current in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if legacy not in migrated:
                continue
            parsed = _parse_size_mb(migrated.pop(legacy), field=legacy)
            if current in migrated and migrated[current] != parsed:
                raise ValueError(f"conflicting Harbor environment values: {legacy} and {current}")
            migrated.setdefault(current, parsed)
        return migrated


class _VerifierConfig(_PhaseConfig):
    timeout_sec: float | None = Field(default=600.0, gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["shared", "separate"] | None = None
    environment: dict[str, Any] | None = None
    collect: list[dict[str, Any]] = Field(default_factory=list)


class _RuntimeTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.4", min_length=1)
    task: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: _VerifierConfig = Field(default_factory=_VerifierConfig)
    agent: _PhaseConfig = Field(default_factory=_PhaseConfig)
    environment: _EnvironmentConfig = Field(default_factory=_EnvironmentConfig)
    solution: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    multi_step_reward_strategy: str | None = None
    steps: list[dict[str, Any]] | None = None
    artifacts: list[str | dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_version_key(cls, value: Any) -> Any:
        if isinstance(value, dict) and "version" in value:
            value = dict(value)
            value.setdefault("schema_version", value.pop("version"))
            value.pop("version", None)
        return value


def _load_runtime_config(task_dir: Path) -> _RuntimeTaskConfig:
    config_path = task_dir / "task.toml"
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config = _RuntimeTaskConfig.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid Harbor task config {config_path}: {exc}") from exc

    unsupported: list[str] = []
    if config.steps:
        unsupported.append("multi-step tasks")
    if config.multi_step_reward_strategy is not None:
        unsupported.append("multi_step_reward_strategy")
    if config.artifacts:
        unsupported.append("artifact collection")
    if config.environment.os != "linux":
        unsupported.append("Windows containers")
    if config.environment.network_mode != "public" or config.environment.allowed_hosts:
        unsupported.append("restricted environment network policy")
    if (
        config.environment.allow_internet is False
        and "network_mode" not in config.environment.model_fields_set
        and "allowed_hosts" not in config.environment.model_fields_set
    ):
        unsupported.append("environment.allow_internet=false")
    if config.environment.tpu is not None:
        unsupported.append("TPU resources")
    if config.environment.mcp_servers:
        unsupported.append("environment MCP servers")
    if config.environment.skills_dir is not None:
        unsupported.append("environment skills_dir")
    if config.environment.healthcheck is not None:
        unsupported.append("task-config healthcheck")
    if config.environment.gpu_types:
        unsupported.append("GPU type selection")
    if config.environment.gpus:
        unsupported.append("GPU resources")
    if config.environment.workdir is not None:
        workdir = PurePosixPath(config.environment.workdir)
        if not workdir.is_absolute() or ".." in workdir.parts:
            raise ValueError(
                f"invalid Harbor workdir {config.environment.workdir!r}: "
                "expected an absolute container path without '..'"
            )
    for label, phase in (("agent", config.agent), ("verifier", config.verifier)):
        if phase.network_mode not in (None, "public") or phase.allowed_hosts:
            unsupported.append(f"{label} network override")
    if config.verifier.environment_mode == "separate" or config.verifier.environment is not None:
        unsupported.append("separate verifier environment")
    if config.verifier.collect:
        unsupported.append("verifier collect hooks")
    if unsupported:
        rendered = ", ".join(dict.fromkeys(unsupported))
        raise ValueError(
            f"HarborRuntime does not support {rendered} in {config_path}; "
            "run this task through Harbor proper or remove the unsupported declaration"
        )

    config.environment.env = _resolve_env_vars(
        config.environment.env,
        label=f"[environment].env in {config_path}",
    )
    config.verifier.env = _resolve_env_vars(
        config.verifier.env,
        label=f"[verifier].env in {config_path}",
    )
    _validate_env_values(config.environment.env)
    _validate_env_values(config.verifier.env)
    return config


class HarborRuntime:
    """Run Harbor task directories through HUD's local rollout engine.

    The provider builds the Harbor task's ``environment/`` Docker context and
    starts a fresh container for agent work. If the task ships a Compose file,
    its ``main`` service and sidecars use the task-authored startup semantics. HUD's SSH
    capability executes every shell and file operation through ``docker exec``,
    so edits land directly in the real container workdir without shadowing image
    contents with a host bind mount. Hidden tests are mounted from an initially
    empty staging directory and copied into it only after the agent finishes.
    Grading then runs ``tests/test.sh`` in the shared main container, bounded by
    ``[verifier].timeout_sec``, and reads ``reward.json`` or ``reward.txt``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ready_timeout: float = 120.0,
        reward_key: str | None = None,
    ) -> None:
        if reward_key is not None and not reward_key:
            raise ValueError("HarborRuntime reward_key must be non-empty")
        self.root = Path(path).resolve()
        self.ready_timeout = ready_timeout
        self.reward_key = reward_key
        self._task_dirs = {task_dir.name: task_dir for task_dir in _task_dirs(self.root)}
        self._image_names: dict[Path, str] = {}
        self._build_locks: dict[str, asyncio.Lock] = {}
        if not self._task_dirs:
            raise ValueError(f"no Harbor tasks found in {path}")

    @contextlib.asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.eval.runtime import LocalRuntime, Runtime

        task_dir = self._task_dirs.get(task.id)
        if task_dir is None:
            raise KeyError(f"HarborRuntime has no task directory for {task.id!r}")
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        config = _load_runtime_config(task_dir)
        compose_file = _compose_file(env_dir)
        if (
            config.environment.docker_image is None
            and not (env_dir / "Dockerfile").is_file()
            and compose_file is None
        ):
            raise FileNotFoundError(
                f"Harbor task {task.id!r} needs environment/Dockerfile, "
                "environment/docker-compose.yaml, or [environment].docker_image"
            )
        if not (tests_dir / "test.sh").is_file():
            raise FileNotFoundError(f"Harbor task {task.id!r} has no tests/test.sh")

        tmp_path = Path(tempfile.mkdtemp(prefix=f"hud-harbor-{_slugify(task.id)}-"))
        try:
            control_root = tmp_path / "control"
            staged_tests = tmp_path / "tests"
            logs = tmp_path / "logs"
            control_root.mkdir()
            staged_tests.mkdir()
            _prepare_host_logs(logs)
            acquire = self._compose_container(
                task,
                env_dir,
                compose_file,
                staged_tests,
                logs,
                config,
            )
            async with acquire as (container, provider, workdir):
                env = self._environment_for(
                    task,
                    task_dir,
                    control_root,
                    workdir,
                    tests_dir,
                    staged_tests,
                    logs,
                    container,
                    config,
                )
                local = LocalRuntime(lambda _task: env, ready_timeout=self.ready_timeout)
                async with local(task) as runtime:
                    yield Runtime(
                        runtime.url,
                        params={
                            **runtime.params,
                            "provider": provider,
                            "container": container,
                            "ready_timeout": self.ready_timeout,
                        },
                        config=runtime.config,
                        agent_timeout_s=config.agent.timeout_sec,
                    )
        finally:
            await _shielded_cleanup(asyncio.to_thread(_remove_tree, tmp_path))

    @contextlib.asynccontextmanager
    async def _compose_container(
        self,
        task: Task,
        env_dir: Path,
        compose_file: Path | None,
        staged_tests: Path,
        logs: Path,
        config: _RuntimeTaskConfig,
    ) -> AsyncIterator[tuple[str, str, str]]:
        from hud.eval.runtime import _docker

        project = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        control_dir = staged_tests.parent
        base = control_dir / "compose.base.json"
        overlay = control_dir / "compose.hud.json"
        compose_env_file = control_dir / "compose.env"
        compose_env: dict[str, str] = {}
        compose_args: list[str] = []
        container = ""
        try:
            try:
                async with asyncio.timeout(config.environment.build_timeout_sec):
                    main_image = await self._main_image_name(env_dir)
                    compose_env = {
                        **config.environment.env,
                        **_compose_infra_env(
                            env_dir=env_dir,
                            logs=logs,
                            main_image=main_image,
                            docker_image=config.environment.docker_image,
                            environment=config.environment,
                        ),
                    }
                    await asyncio.to_thread(
                        _write_compose_control_files,
                        base=base,
                        overlay=overlay,
                        compose_env_file=compose_env_file,
                        staged_tests=staged_tests,
                        logs=logs,
                        environment=config.environment,
                        compose_env=compose_env,
                    )
                    compose_files = [
                        base,
                        *([compose_file] if compose_file is not None else []),
                        overlay,
                    ]
                    compose_args = [
                        "compose",
                        "--env-file",
                        str(compose_env_file),
                        "--project-directory",
                        str(env_dir),
                        "-p",
                        project,
                    ]
                    for path in compose_files:
                        compose_args.extend(["-f", str(path)])
                    if config.environment.docker_image is None:
                        lock = self._build_locks.setdefault(main_image, asyncio.Lock())
                        async with lock:
                            await _docker(
                                *compose_args,
                                "build",
                                unset_env=tuple(compose_env),
                            )
                    await _docker(
                        *compose_args,
                        "up",
                        "--detach",
                        "--wait",
                        unset_env=tuple(compose_env),
                    )
                    out, _ = await _docker(
                        *compose_args,
                        "ps",
                        "-q",
                        "main",
                        unset_env=tuple(compose_env),
                    )
                    container = out.strip()
                    if not container:
                        raise RuntimeError(
                            f"docker compose project {project} did not create a main service"
                        )
                    workdir = config.environment.workdir or await _container_workdir(container)
                    await _prepare_container_logs(container)
                    if _should_upload_environment_dir(
                        env_dir=env_dir,
                        docker_image=config.environment.docker_image,
                    ):
                        await _docker("cp", f"{env_dir}/.", f"{container}:{workdir}")
            except TimeoutError as exc:
                raise TimeoutError(
                    "Harbor environment build/start timed out after "
                    f"{config.environment.build_timeout_sec:.0f}s"
                ) from exc
            yield container, "harbor-compose", workdir
        finally:
            if compose_args:

                async def cleanup() -> None:
                    if container:
                        with contextlib.suppress(Exception):
                            await _release_log_permissions(container)
                    await _docker(
                        *compose_args,
                        "down",
                        "--volumes",
                        "--remove-orphans",
                        "--rmi",
                        "local",
                        check=False,
                        unset_env=tuple(compose_env),
                    )

                await _shielded_cleanup(cleanup())

    async def _main_image_name(self, env_dir: Path) -> str:
        cached = self._image_names.get(env_dir)
        if cached is not None:
            return cached
        digest = await asyncio.to_thread(_hash_directory, env_dir)
        image = f"hud-harbor:{digest}"
        return self._image_names.setdefault(env_dir, image)

    def _environment_for(
        self,
        task: Task,
        task_dir: Path,
        control_root: Path,
        workdir: str,
        tests_dir: Path,
        staged_tests: Path,
        logs: Path,
        container: str,
        config: _RuntimeTaskConfig,
    ) -> Environment:
        env = Environment(task.env)
        workspace_daemon = _DockerWorkspace(
            control_root,
            container=container,
            guest_path=workdir,
            exec_user=config.agent.user,
        )
        verifier_timeout = config.verifier.timeout_sec or _DEFAULT_VERIFIER_TIMEOUT

        @env.initialize
        async def _up() -> None:
            await workspace_daemon.start()
            env.add_capability(workspace_daemon.capability("shell"))

        @env.shutdown
        async def _down() -> None:
            await workspace_daemon.stop()

        @env.template(id=task.id, description=f"Harbor task {task.id}")
        async def _run_harbor_task() -> AsyncGenerator[Any, Any]:
            answer = yield (task_dir / "instruction.md").read_text(encoding="utf-8")
            yield await self._grade(
                container,
                workdir,
                tests_dir,
                staged_tests,
                logs,
                answer,
                verifier_timeout=verifier_timeout,
                verifier_user=config.verifier.user,
                verifier_env={**config.environment.env, **config.verifier.env},
            )

        return env

    async def _grade(
        self,
        container: str,
        workdir: str,
        tests_dir: Path,
        staged_tests: Path,
        logs: Path,
        answer: Any,
        *,
        verifier_timeout: float,
        verifier_user: str | int | None,
        verifier_env: dict[str, str],
    ) -> dict[str, Any]:
        answer_file = staged_tests.parent / "agent_answer.txt"
        await asyncio.to_thread(
            answer_file.write_text,
            "" if answer is None else str(answer),
            encoding="utf-8",
        )
        await _release_log_permissions(container)
        await asyncio.to_thread(
            _prepare_verifier_staging,
            tests_dir,
            staged_tests,
            logs / "verifier",
        )
        verifier_env_file = staged_tests.parent / "verifier.env"
        _write_docker_env_file(verifier_env_file, verifier_env)
        exec_args = ["exec", "--workdir", workdir]
        if verifier_user is not None:
            exec_args.extend(["--user", str(verifier_user)])
        if verifier_env:
            exec_args.extend(["--env-file", str(verifier_env_file)])
        exec_args.extend([container, "/tests/test.sh"])
        from hud.eval.runtime import _docker

        try:
            out, err = await asyncio.wait_for(
                _docker(*exec_args, check=False), timeout=verifier_timeout
            )
        except TimeoutError:
            return {
                "score": 0.0,
                "isError": True,
                "content": f"Harbor verifier timed out after {verifier_timeout:.0f}s",
                "info": {"verifier_timeout_sec": verifier_timeout},
            }
        await _release_log_permissions(container)
        reward, info = await asyncio.to_thread(
            _read_harbor_reward,
            logs / "verifier",
            reward_key=self.reward_key,
        )
        info.update(
            {
                "stdout": out[-4000:],
                "stderr": err[-4000:],
            }
        )
        if reward is None:
            parse_error = info.get("reward_parse_error")
            return {
                "score": 0.0,
                "isError": True,
                "content": (
                    f"Harbor verifier reward is invalid: {parse_error}"
                    if parse_error
                    else "Harbor verifier did not write reward.json or reward.txt"
                ),
                "info": info,
            }
        return {"score": reward, "info": info}


class _DockerWorkspace(Workspace):
    """Workspace SSH whose commands and file streams run through ``docker exec``."""

    def __init__(
        self,
        *args: Any,
        container: str,
        exec_user: str | int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._container = container
        self._exec_user = exec_user
        self._docker = shutil.which("docker") or "docker"

    def shell_argv(
        self,
        command: str | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        argv = [self._docker, "exec", "-i", "--workdir", cwd or self._guest_path]
        if self._exec_user is not None:
            argv.extend(["--user", str(self._exec_user)])
        for key, value in (env or {}).items():
            argv.extend(["--env", f"{key}={value}"])
        argv.extend([self._container, "bash", "-lc", command or "bash -l"])
        return argv


_DEFAULT_VERIFIER_TIMEOUT = 600.0
_ENV_TEMPLATE_RE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_REWARD_BYTES = 1_000_000
_CLEANUP_TIMEOUT = 60.0


def _parse_size_mb(value: Any, *, field: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"legacy Harbor environment.{field} must be a size string")
    normalized = value.strip().upper()
    multipliers = {"G": 1024.0, "M": 1.0, "K": 1.0 / 1024.0}
    suffix = normalized[-1:] if normalized else ""
    if suffix not in multipliers:
        raise ValueError(f"invalid Harbor environment.{field} size {value!r}")
    try:
        return int(float(normalized[:-1]) * multipliers[suffix])
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid Harbor environment.{field} size {value!r}") from exc


def _resolve_env_vars(values: dict[str, str], *, label: str) -> dict[str, str]:
    """Resolve Harbor's ``${VAR}`` / ``${VAR:-default}`` task env syntax."""

    def resolve(value: str) -> str:
        match = _ENV_TEMPLATE_RE.fullmatch(value)
        if match is None:
            return value
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"{label} references unset environment variable {name!r}")

    return {key: resolve(value) for key, value in values.items()}


async def _container_workdir(container: str) -> str:
    """The running container's configured ``WORKDIR``, or Docker's root default."""
    from hud.eval.runtime import _docker

    out, _ = await _docker("inspect", "--format", "{{.Config.WorkingDir}}", container)
    return out.strip() or "/"


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prepare_host_logs(logs: Path) -> None:
    """Create bind-mounted Harbor log paths writable before container startup."""
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o777)
    for name in ("agent", "verifier", "artifacts"):
        child = logs / name
        child.mkdir(exist_ok=True)
        child.chmod(0o777)


def _prepare_verifier_staging(tests_dir: Path, staged_tests: Path, verifier_logs: Path) -> None:
    """Expose hidden tests only after agent work and discard any pre-seeded reward."""
    _clear_directory(verifier_logs)
    _clear_directory(staged_tests)
    shutil.copytree(tests_dir, staged_tests, dirs_exist_ok=True)
    test_script = staged_tests / "test.sh"
    test_script.chmod(test_script.stat().st_mode | 0o111)


def _read_harbor_reward(
    verifier_logs: Path,
    *,
    reward_key: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    reward_json = verifier_logs / "reward.json"
    try:
        reward_json_text = _read_regular_text(reward_json)
    except ValueError as exc:
        return None, {
            "reward_file": "reward.json",
            "reward_parse_error": str(exc),
        }
    if reward_json_text is not None:
        try:
            data = json.loads(reward_json_text)
        except json.JSONDecodeError as exc:
            return None, {
                "reward_file": "reward.json",
                "reward_parse_error": f"invalid reward.json: {exc}",
            }
        if isinstance(data, dict):
            converted = {str(key): _finite_number(value) for key, value in data.items()}
            invalid_keys = [key for key, value in converted.items() if value is None]
            if invalid_keys:
                return None, {
                    "reward_file": "reward.json",
                    "reward_parse_error": (
                        "reward.json contains non-numeric reward(s): " + ", ".join(invalid_keys)
                    ),
                }
            rewards = {key: value for key, value in converted.items() if value is not None}
            info: dict[str, Any] = {
                "reward_file": "reward.json",
                "harbor_rewards": rewards,
            }
            if reward_key is not None:
                if reward_key not in rewards:
                    info["reward_parse_error"] = (
                        f"configured reward key {reward_key!r} is missing or non-numeric"
                    )
                    return None, info
                info["primary_reward_key"] = reward_key
                return rewards[reward_key], info
            if "reward" in rewards:
                info["primary_reward_key"] = "reward"
                return rewards["reward"], info
            if len(rewards) == 1:
                key, value = next(iter(rewards.items()))
                info["primary_reward_key"] = key
                return value, info
            info["reward_parse_error"] = (
                "reward.json has no unambiguous primary reward; set HarborRuntime(reward_key=...)"
                if rewards
                else "reward.json has no numeric rewards"
            )
            return None, info
        return None, {
            "reward_file": "reward.json",
            "reward_parse_error": "reward.json must be an object of numeric rewards",
        }

    reward_txt = verifier_logs / "reward.txt"
    try:
        reward_text = _read_regular_text(reward_txt)
    except ValueError as exc:
        return None, {
            "reward_file": "reward.txt",
            "reward_parse_error": str(exc),
        }
    if reward_text is not None:
        text = reward_text.strip()
        try:
            reward = float(text)
        except (OverflowError, ValueError):
            reward = math.nan
        if math.isfinite(reward):
            return reward, {"reward_file": "reward.txt"}
        return None, {"reward_file": "reward.txt", "reward_parse_error": text}

    return None, {}


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _read_regular_text(path: Path) -> str | None:
    """Read a small regular file without following an untrusted verifier symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path.name} is not a regular file")
    if metadata.st_size > _MAX_REWARD_BYTES:
        raise ValueError(f"{path.name} exceeds {_MAX_REWARD_BYTES} bytes")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"could not safely open {path.name}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as file:
        opened = os.fstat(file.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{path.name} is not a regular file")
        payload = file.read(_MAX_REWARD_BYTES + 1)
    if len(payload) > _MAX_REWARD_BYTES:
        raise ValueError(f"{path.name} exceeds {_MAX_REWARD_BYTES} bytes")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is not valid UTF-8") from exc


async def _release_log_permissions(container: str) -> None:
    """Let the host user remove verifier logs written by container root."""
    from hud.eval.runtime import _docker

    await _docker(
        "exec",
        "--user",
        "0",
        container,
        "chmod",
        "-R",
        "a+rwX",
        "/logs",
        check=False,
    )


async def _prepare_container_logs(container: str) -> None:
    """Create Harbor log directories writable by non-root agent/verifier users."""
    from hud.eval.runtime import _docker

    await _docker(
        "exec",
        "--user",
        "0",
        container,
        "sh",
        "-c",
        "mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod -R a+rwX /logs",
    )


def _compose_file(env_dir: Path) -> Path | None:
    for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
        path = env_dir / name
        if path.is_file():
            return path
    return None


def _should_upload_environment_dir(*, env_dir: Path, docker_image: str | None) -> bool:
    """Match Harbor's prebuilt-image upload rule for environment-only files."""
    return bool(
        docker_image
        and env_dir.is_dir()
        and not (env_dir / "Dockerfile").exists()
        and _compose_file(env_dir) is None
        and any(env_dir.iterdir())
    )


def _compose_base(*, docker_image: str | None) -> str:
    """Harbor's base ``main`` service: image/build plus its keepalive command."""
    main: dict[str, Any]
    if docker_image is not None:
        main = {"image": "${PREBUILT_IMAGE_NAME}"}
    else:
        main = {
            "build": {"context": "${CONTEXT_DIR}"},
            "image": "${MAIN_IMAGE_NAME}",
            "pull_policy": "build",
        }
    main["command"] = ["sh", "-c", "sleep infinity"]
    return json.dumps({"services": {"main": main}}, indent=2) + "\n"


def _compose_overlay(
    *,
    tests_dir: Path,
    logs: Path,
    environment: _EnvironmentConfig,
) -> str:
    """Final Compose override for staged tests, logs, env, workdir, and limits."""
    main: dict[str, Any] = {
        "volumes": [
            f"{tests_dir}:/tests:ro",
            f"{logs}:/logs",
        ],
    }
    if environment.env:
        # Keep task values literal here. Harbor infrastructure variables win for
        # Compose-file interpolation, but a same-named task variable must still
        # reach the main container with the value the task declared.
        main["environment"] = environment.env
    if environment.cpus is not None:
        main["cpus"] = float(environment.cpus)
    if environment.memory_mb is not None:
        main["mem_limit"] = f"{environment.memory_mb}m"
    return json.dumps({"services": {"main": main}}, indent=2) + "\n"


def _compose_infra_env(
    *,
    env_dir: Path,
    logs: Path,
    main_image: str,
    docker_image: str | None,
    environment: _EnvironmentConfig,
) -> dict[str, str]:
    """Protected Harbor variables used by task-authored Compose templates."""
    values = {
        "CONTEXT_DIR": str(env_dir.resolve()),
        "MAIN_IMAGE_NAME": main_image,
        "HOST_AGENT_LOGS_PATH": str((logs / "agent").resolve()),
        "ENV_AGENT_LOGS_PATH": "/logs/agent",
        "HOST_VERIFIER_LOGS_PATH": str((logs / "verifier").resolve()),
        "ENV_VERIFIER_LOGS_PATH": "/logs/verifier",
        "HOST_ARTIFACTS_LOGS_PATH": str((logs / "artifacts").resolve()),
        "ENV_ARTIFACTS_LOGS_PATH": "/logs/artifacts",
    }
    if docker_image is not None:
        values["PREBUILT_IMAGE_NAME"] = docker_image
    if environment.cpus is not None:
        values["CPUS"] = str(environment.cpus)
    if environment.memory_mb is not None:
        values["MEMORY"] = f"{environment.memory_mb}M"
    return values


def _validate_env_values(values: Mapping[str, str]) -> None:
    for key, value in values.items():
        if _ENV_NAME_RE.fullmatch(key) is None:
            raise ValueError(f"invalid Harbor environment variable name {key!r}")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(
                f"Harbor environment variable {key!r} contains an unsupported newline or NUL"
            )


def _write_compose_env_file(path: Path, values: Mapping[str, str]) -> None:
    """Write a mode-0600 Compose interpolation file with literal values."""
    _validate_env_values(values)
    lines = []
    for key, value in values.items():
        escaped = value.replace("'", "\\'")
        lines.append(f"{key}='{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o600)


def _write_compose_control_files(
    *,
    base: Path,
    overlay: Path,
    compose_env_file: Path,
    staged_tests: Path,
    logs: Path,
    environment: _EnvironmentConfig,
    compose_env: Mapping[str, str],
) -> None:
    """Materialize one acquisition's generated Compose inputs off the event loop."""
    base.write_text(
        _compose_base(docker_image=environment.docker_image),
        encoding="utf-8",
        newline="\n",
    )
    overlay.write_text(
        _compose_overlay(
            tests_dir=staged_tests,
            logs=logs,
            environment=environment,
        ),
        encoding="utf-8",
        newline="\n",
    )
    overlay.chmod(0o600)
    _write_compose_env_file(compose_env_file, compose_env)


def _write_docker_env_file(path: Path, values: Mapping[str, str]) -> None:
    """Write a mode-0600 Docker ``--env-file`` for verifier overrides."""
    _validate_env_values(values)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o600)


async def _shielded_cleanup(
    cleanup: Coroutine[Any, Any, Any], *, timeout_s: float = _CLEANUP_TIMEOUT
) -> None:
    """Finish bounded Docker cleanup even when the rollout is cancelled."""
    cleanup_task = asyncio.create_task(cleanup)

    async def wait() -> None:
        async with asyncio.timeout(timeout_s):
            await asyncio.shield(cleanup_task)

    try:
        await wait()
    except asyncio.CancelledError:
        try:
            await wait()
        except BaseException:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        raise
    except Exception:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)


def _remove_tree(path: Path) -> None:
    """Best-effort removal of one runtime-owned temporary directory."""
    if not path.exists():
        return

    def make_writable(function: Any, target: str, _error: Any) -> None:
        os.chmod(target, 0o700)
        function(target)

    with contextlib.suppress(OSError):
        shutil.rmtree(path, onerror=make_writable)

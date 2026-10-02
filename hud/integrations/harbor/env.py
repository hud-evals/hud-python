from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import math
import os
import re
import shlex
import shutil
import socket
import tempfile
from collections.abc import AsyncGenerator, Iterator  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from hud.capabilities import Capability
from hud.environment import Environment, Mount, Workspace
from hud.environment.egress import ANY_HOST
from hud.environment.env import current_session_id
from hud.graders import EvaluationResult
from hud.integrations.harbor.config import Artifact, ControllerConfig, TaskSpec
from hud.utils.process import ProcessResult, create_process_group_exec

if TYPE_CHECKING:
    from hud.environment.namespace import NamespaceProcess

CONTROLLER_ROOT = Path("/controller")
TESTS = Path("/tests")
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
AGENT_ANSWER = LOGS / "agent_answer.txt"
DOCKER = CONTROLLER_ROOT / "bin" / "docker"
CONFIG = ControllerConfig.model_validate_json((CONTROLLER_ROOT / "config.json").read_text("utf-8"))
TASK_ROOT = Path("/rootfs")
RUNTIME_ROOT = Path("/runtime")
SESSIONS = RUNTIME_ROOT / "sessions"
DOCKER_SOCKET = Path("/var/run/docker.sock")
MANAGED_AGENTS = Path("/usr/local/lib/agents")

ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.group(1), match.group(2)
        runtime_value = os.environ.get(name)
        if runtime_value is not None and (runtime_value or default is None):
            resolved[key] = runtime_value
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(
                f"Harbor env template for {key!r} needs {name!r}; "
                "the runtime environment must provide it"
            )
    return resolved


for policy in (CONFIG.environment, CONFIG.agent):
    policy.env = resolve_env_templates(policy.env)
os.environ.update(CONFIG.environment.env)
TASK_ENV = {**CONFIG.image_env, **CONFIG.environment.env}
WORKDIR = Path(CONFIG.workdir)
if not WORKDIR.is_absolute():
    raise ValueError(f"Harbor workdir must be absolute: {WORKDIR}")
TASK_WORKDIR = TASK_ROOT / WORKDIR.relative_to("/")
GPU_DRIVER_MOUNTS = tuple(
    Mount("ro", src=str(path), dst=str(path))
    for path in sorted(
        (
            {
                path
                for root in (Path("/usr/lib"), Path("/usr/lib64"))
                if root.is_dir()
                for path in root.rglob("*.so*")
                if path.name.startswith(("libcuda.so", "libnvidia-"))
                and (path.is_file() or path.is_symlink())
            }
            | {
                path
                for path in Path("/usr/bin").glob("nvidia-*")
                if path.is_file() or path.is_symlink()
            }
        )
        if Path("/dev/nvidiactl").exists()
        else set()
    )
)


@dataclass(frozen=True, slots=True)
class Account:
    """A Docker ``USER`` spec resolved against an image's ``/etc/passwd`` and ``/etc/group``."""

    #: The uid and gid to drop to; ``None`` keeps root.
    identity: tuple[int, int] | None
    #: ``HOME`` from the passwd entry of a non-root account.
    env: dict[str, str]


def records(path: Path, width: int) -> list[list[str]]:
    lines = path.read_text("utf-8").splitlines() if path.is_file() else []
    return [fields for line in lines if len(fields := line.split(":")) >= width]


def account(user: str | int | None, root: Path) -> Account:
    spec = str(user or "")
    if not spec:
        return Account(None, {})
    user_name, separator, group_name = spec.partition(":")
    passwd = records(root / "etc/passwd", 6)
    if user_name.isdigit():
        uid = int(user_name)
        entry = next((fields for fields in passwd if int(fields[2]) == uid), None)
    else:
        entry = next((fields for fields in passwd if fields[0] == user_name), None)
        if entry is None:
            raise ValueError(f"Harbor user {spec!r} does not exist in this image")
        uid = int(entry[2])
    if not separator:
        # Docker resolves a known account's primary group; a bare numeric uid
        # with no passwd entry keeps the container default group (root).
        gid = int(entry[3]) if entry is not None else 0
    elif group_name.isdigit():
        gid = int(group_name)
    else:
        groups = records(root / "etc/group", 3)
        group = next((int(fields[2]) for fields in groups if fields[0] == group_name), None)
        if group is None:
            raise ValueError(f"Harbor group {group_name!r} does not exist in this image")
        gid = group
    if (uid, gid) == (0, 0):
        return Account(None, {})
    return Account((uid, gid), {"HOME": entry[5]} if entry is not None and entry[5] else {})


agent = CONFIG.agent
image_identity = account(CONFIG.image_user, TASK_ROOT).identity
agent_account = account(agent.user, TASK_ROOT)
agent_hosts = frozenset(agent.network.allowed_hosts)
environment_hosts = frozenset(CONFIG.environment.network.allowed_hosts)
rooted_at_filesystem = len(WORKDIR.parts) == 1
task_mounts = tuple(CONFIG.mounts)
agent_mounts = (
    *task_mounts,
    Mount("tmpfs", dst=str(TESTS)),
    Mount("tmpfs", dst=str(VERIFIER_LOGS)),
    Mount("ro", src="/dev/null", dst=str(AGENT_ANSWER)),
)
verifier_mounts = (
    *task_mounts,
    Mount("rw", src=str(TESTS), dst=str(TESTS)),
    Mount("rw", src=str(LOGS), dst=str(LOGS)),
)

env = Environment(CONFIG.name)
verifier_lock = asyncio.Lock()
for capability in CONFIG.capabilities:
    env.add_capability(Capability.from_manifest(capability))
workspace = env.workspace(
    TASK_WORKDIR,
    guest_path=WORKDIR.as_posix(),
    system_mounts=(
        Mount("rw", src=str(TASK_ROOT), dst="/"),
        Mount("dev", dst="/dev"),
        Mount("proc", dst="/proc"),
        *GPU_DRIVER_MOUNTS,
        Mount("ro", src=str(MANAGED_AGENTS), dst=str(MANAGED_AGENTS), optional=True),
    ),
    mounts=agent_mounts,
    credentials_dir=RUNTIME_ROOT / "session-keys",
    hosts_path=RUNTIME_ROOT / "hosts",
    shell_uid=agent_account.identity[0] if agent_account.identity else None,
    shell_gid=agent_account.identity[1] if agent_account.identity else None,
    hand_over_root=False,
    track_files=False if rooted_at_filesystem else None,
    env={**TASK_ENV, **agent.env, **agent_account.env},
    network=agent.network.enabled,
    allowed_hosts=agent_hosts,
    peers=CONFIG.peers,
    local_aliases=CONFIG.local_aliases,
    ports=CONFIG.ports,
    require_isolation=True,
)


async def start_entrypoint() -> NamespaceProcess | None:
    entrypoint = CONFIG.entrypoint
    if not entrypoint:
        return None
    sandbox = await workspace.sandbox_pid()
    if sandbox is None:
        raise RuntimeError("Harbor entrypoints require an isolated workspace")
    process = await workspace.launch(
        [*entrypoint, "sh", "-c", "sleep infinity"],
        env=TASK_ENV,
        identity=image_identity,
        inherit_workspace_env=False,
        no_new_privs=False,
        persistent=True,
    )
    await asyncio.sleep(0)
    if process.returncode is not None:
        raise RuntimeError(f"Harbor environment entrypoint exited with status {process.returncode}")
    return process


async def wait_until_healthy(entrypoint: NamespaceProcess | None) -> None:
    healthcheck = CONFIG.environment.healthcheck
    if healthcheck is None:
        return
    loop = asyncio.get_running_loop()
    start_period = healthcheck.start_period_sec
    start_period_end = loop.time() + start_period
    delay = healthcheck.start_interval_sec if start_period > 0 else healthcheck.interval_sec
    failures = 0
    while True:
        await asyncio.sleep(delay)
        in_start_period = loop.time() < start_period_end
        if entrypoint is not None and entrypoint.returncode is not None:
            raise RuntimeError(
                f"Harbor environment entrypoint exited with status {entrypoint.returncode}"
            )
        result = await workspace.run(
            ["sh", "-c", healthcheck.command],
            env=TASK_ENV,
            identity=image_identity,
            inherit_workspace_env=False,
            allowed_hosts=None if environment_hosts == agent_hosts else environment_hosts,
            no_new_privs=False,
            max_wait=healthcheck.timeout_sec,
        )
        if result.returncode == 0 and not result.timed_out:
            return

        if in_start_period:
            delay = healthcheck.start_interval_sec
        else:
            failures += 1
            if failures >= healthcheck.retries:
                detail = result.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"Harbor environment healthcheck failed after {failures} attempts"
                    + (f": {detail}" if detail else "")
                )
            delay = healthcheck.interval_sec


async def docker(*args: str, max_wait: float = 60.0, check: bool = True) -> ProcessResult:
    process = await create_process_group_exec(
        str(DOCKER),
        "--host",
        f"unix://{DOCKER_SOCKET}",
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    result = await process.complete(max_wait=max_wait)
    if result.timed_out:
        raise TimeoutError(f"docker {' '.join(args)} timed out after {max_wait:g}s")
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"docker {' '.join(args)} failed: {detail}")
    return result


async def compose_containers() -> dict[str, str]:
    if not await asyncio.to_thread(DOCKER_SOCKET.exists):
        raise RuntimeError("Compose service access is unavailable in this runtime")
    project = await docker(
        "inspect",
        "--format",
        '{{ index .Config.Labels "com.docker.compose.project" }}',
        socket.gethostname(),
    )
    project_name = project.stdout.decode().strip()
    if not project_name:
        raise RuntimeError("the Harbor container has no Compose project label")
    listed = await docker(
        "ps",
        "--filter",
        f"label=com.docker.compose.project={project_name}",
        "--format",
        '{{.ID}} {{.Label "com.docker.compose.service"}}',
    )
    return {
        service: container_id
        for line in listed.stdout.decode().splitlines()
        for container_id, service in (line.split(maxsplit=1),)
    }


def exclude_artifact_paths(root: Path, patterns: list[str]) -> None:
    if not patterns or not root.is_dir() or root.is_symlink():
        return
    for entry in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        relative = entry.relative_to(root).as_posix()
        parts = Path(relative).parts
        candidates = (
            relative,
            f"./{relative}",
            *("/".join(parts[index:]) for index in range(1, len(parts))),
        )
        if not any(
            fnmatch.fnmatchcase(candidate, pattern)
            for candidate in candidates
            for pattern in patterns
        ):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def copy_artifact(source: Path, target: Path, exclude: list[str], *, name: str) -> None:
    if source.is_symlink():
        raise RuntimeError(f"artifact {name} is a symbolic link")
    if source.resolve(strict=False) != source.absolute():
        raise RuntimeError(f"artifact {name} has a symbolic link in its path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
        exclude_artifact_paths(target, exclude)
    elif source.exists() or source.is_symlink():
        shutil.copy2(source, target, follow_symlinks=False)


def task_path(guest: str) -> Path:
    """Return the controller path behind ``guest`` as the agent sees it.

    Authored volumes live under their task mount rather than the authored root.
    """
    path = PurePosixPath("/", guest)
    mounts = [mount for mount in task_mounts if path.is_relative_to(mount.dst)]
    if not mounts:
        return TASK_ROOT / path.relative_to("/")
    mount = max(mounts, key=lambda mount: len(PurePosixPath(mount.dst).parts))
    return Path(mount.src, path.relative_to(mount.dst))


def artifact_path(artifact: Artifact, artifacts: Path) -> Path:
    relative = artifact.destination or artifact.source.lstrip("/").rstrip("/")
    return artifacts / relative


async def collect(task: TaskSpec, artifacts: Path) -> None:
    clear(artifacts)
    services: dict[str, str] = {}

    async def container(service: str) -> str:
        if service == "main":
            return ""
        if service in services:
            return services[service]
        if not services:
            services.update(await compose_containers())
        try:
            return services[service]
        except KeyError as error:
            raise RuntimeError(f"Compose service {service!r} is not running") from error

    for hook in task.collect:
        service = hook.service
        container_id = await container(service)
        if container_id:
            await docker(
                "exec",
                container_id,
                "sh",
                "-c",
                hook.command,
                max_wait=hook.timeout_sec,
            )
        else:
            execution = await workspace.run(
                ["sh", "-c", hook.command],
                mounts=task_mounts,
                env=TASK_ENV,
                identity=image_identity,
                inherit_workspace_env=False,
                allowed_hosts=None,
                no_new_privs=False,
                max_wait=hook.timeout_sec,
            )
            if execution.timed_out:
                raise TimeoutError(
                    f"collect hook on {service!r} timed out after {hook.timeout_sec:g}s"
                )
            if execution.returncode != 0:
                detail = execution.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"collect hook on {service!r} failed: {detail}")

    for artifact in task.artifacts:
        source = artifact.source
        target = artifact_path(artifact, artifacts)
        exclude = artifact.exclude
        service = artifact.service
        container_id = await container(service)
        if container_id:
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = await docker(
                "cp",
                f"{container_id}:{source.rstrip('/') or '/'}",
                str(target),
                max_wait=task.verifier_timeout,
                check=False,
            )
            if copied.returncode != 0:
                continue
            exclude_artifact_paths(target, exclude)
        else:
            copy_artifact(task_path(source), target, exclude, name=source)
        if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
            raise RuntimeError(f"artifact {source} contains a symbolic link")


@env.template(id="run", description="Run a Harbor task")
async def run(instruction: str, task: TaskSpec) -> AsyncGenerator[Any, Any]:
    clear_grading_files()
    AGENT_ANSWER.parent.mkdir(parents=True, exist_ok=True)
    AGENT_ANSWER.touch()
    entrypoint = None
    try:
        entrypoint = await start_entrypoint()
        await wait_until_healthy(entrypoint)
        answer = yield instruction
        if entrypoint is not None and entrypoint.returncode is not None:
            raise RuntimeError(
                f"Harbor environment entrypoint exited with status {entrypoint.returncode}"
            )
        if task.separate_verifier:
            session_id = current_session_id.get()
            if session_id is None:
                raise RuntimeError("Harbor actor is not running in an environment session")
            await workspace.terminate_sessions()
            try:
                await collect(task, SESSIONS / session_id / "artifacts")
            except Exception as error:
                yield EvaluationResult(isError=True, content=str(error))
            else:
                yield EvaluationResult(info={"answer": "" if answer is None else str(answer)})
        else:
            yield await grade(task.id, task.verifier_timeout, answer)
    finally:
        clear_grading_files()
        await workspace.discard_sandbox()
        if entrypoint is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(entrypoint.wait(), 10.0)


if CONFIG.verifier_root is not None:

    @env.template(id="verify", description="Verify a Harbor task")
    async def verify(task: TaskSpec) -> AsyncGenerator[Any, Any]:
        actor = EvaluationResult.model_validate((yield ""))
        if actor.isError:
            raise RuntimeError(actor.content)
        session_id = current_session_id.get()
        if session_id is None:
            raise RuntimeError("Harbor verifier is not running in an environment session")
        session = SESSIONS / session_id
        if not session.is_dir():
            raise ValueError("Harbor actor session files are unavailable in this runtime")
        try:
            yield await grade_separate(task, session / "artifacts", actor.info["answer"])
        finally:
            clear_grading_files()
            shutil.rmtree(session, ignore_errors=True)


def clear(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    if not path.is_dir():
        path.mkdir(parents=True)
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def clear_grading_files() -> None:
    for path in (TESTS, VERIFIER_LOGS):
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(path)
    with contextlib.suppress(FileNotFoundError):
        if AGENT_ANSWER.is_dir() and not AGENT_ANSWER.is_symlink():
            shutil.rmtree(AGENT_ANSWER)
        else:
            AGENT_ANSWER.unlink()


def verifier_command(script: Path, path: str | None = None) -> list[str]:
    target = path or str(script)
    for line in script.read_text("utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            return [*shlex.split(stripped[2:]), target]
        if stripped and not stripped.startswith("#"):
            break
    return ["/bin/sh", target]


async def grade(task_id: str, timeout_sec: float, answer: Any) -> EvaluationResult:
    clear(TESTS)
    shutil.copytree(CONTROLLER_ROOT / "tests" / task_id, TESTS, symlinks=True, dirs_exist_ok=True)
    test_script = TESTS / "test.sh"
    test_script.chmod(test_script.stat().st_mode | 0o111)

    clear(VERIFIER_LOGS)
    AGENT_ANSWER.write_text("" if answer is None else str(answer), encoding="utf-8")

    verifier = CONFIG.verifier
    verifier_account = account(verifier.user, TASK_ROOT)
    if verifier_account.identity:
        for root in (TESTS, VERIFIER_LOGS):
            for path in (root, *root.rglob("*")):
                os.lchown(path, *verifier_account.identity)

    execution = await workspace.run(
        verifier_command(test_script),
        mounts=verifier_mounts,
        env={**TASK_ENV, **resolve_env_templates(verifier.env), **verifier_account.env},
        identity=verifier_account.identity,
        inherit_workspace_env=False,
        allowed_hosts=verifier.network.allowed_hosts,
        no_new_privs=False,
        max_wait=timeout_sec,
        writable_hosts=True,
    )
    return evaluation(execution, timeout_sec)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def copy_path(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target, follow_symlinks=False)
    for path in (source, *source.rglob("*")):
        relative = path.relative_to(source) if path != source else Path()
        metadata = path.lstat()
        os.lchown(target / relative, metadata.st_uid, metadata.st_gid)


@contextlib.contextmanager
def materialized_artifacts(
    task: TaskSpec,
    verifier_root: Path,
    artifacts: Path,
    verifier_identity: tuple[int, int] | None,
) -> Iterator[None]:
    with tempfile.TemporaryDirectory(prefix="verifier-backup-", dir=RUNTIME_ROOT) as directory:
        backup_root = Path(directory)
        replacements: list[tuple[Path, Path | None]] = []
        modes: dict[Path, int] = {}
        entries: dict[Path, set[str]] = {}
        created: list[Path] = []
        try:
            for artifact in task.artifacts:
                staged = artifact_path(artifact, artifacts)
                if not staged.exists() and not staged.is_symlink():
                    continue
                destination = artifact.source.rstrip("/") or "/"
                target = verifier_root / destination.lstrip("/")
                if target == verifier_root:
                    raise ValueError("the verifier root cannot be replaced by an artifact")

                missing: list[Path] = []
                parent = target.parent
                while parent != verifier_root:
                    if not parent.exists():
                        missing.append(parent)
                    parent = parent.parent
                target.parent.mkdir(parents=True, exist_ok=True)
                created.extend(reversed(missing))
                entries.setdefault(target.parent, {path.name for path in target.parent.iterdir()})

                parent = target.parent
                while parent != verifier_root:
                    mode = parent.stat().st_mode & 0o7777
                    modes.setdefault(parent, mode)
                    required = 0o003 if parent == target.parent else 0o001
                    parent.chmod(mode | required)
                    parent = parent.parent

                backup = None
                if target.exists() or target.is_symlink():
                    backup = backup_root / destination.lstrip("/")
                    copy_path(target, backup)
                    remove_path(target)
                replacements.append((target, backup))
                copy_path(staged, target)
                if verifier_identity is not None:
                    for path in (target, *target.rglob("*")):
                        os.lchown(path, *verifier_identity)
            yield
        finally:
            for target, backup in reversed(replacements):
                remove_path(target)
                if backup is not None:
                    copy_path(backup, target)
            for parent, names in entries.items():
                for path in parent.iterdir():
                    if path.name not in names:
                        remove_path(path)
            for path, mode in modes.items():
                path.chmod(mode)
            for path in reversed(created):
                with contextlib.suppress(OSError):
                    path.rmdir()


async def grade_separate(
    task: TaskSpec,
    artifacts: Path,
    answer: str,
) -> EvaluationResult:
    assert CONFIG.verifier_root is not None
    async with verifier_lock:
        verifier_root = Path(CONFIG.verifier_root)
        test_script = verifier_root / "tests/test.sh"
        test_mode = (await asyncio.to_thread(test_script.stat)).st_mode
        await asyncio.to_thread(test_script.chmod, test_mode | 0o111)
        await asyncio.to_thread(clear, VERIFIER_LOGS)
        await asyncio.to_thread(VERIFIER_LOGS.chmod, 0o777)
        await asyncio.to_thread(LOGS.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(AGENT_ANSWER.write_text, answer, encoding="utf-8")

        try:
            verifier = CONFIG.verifier
            public = ANY_HOST in verifier.network.allowed_hosts
            verifier_access = None if public else verifier.network.allowed_hosts
            image = CONFIG.verifier_image
            verifier_account = account(verifier.user, verifier_root)
            verifier_env = {
                **CONFIG.environment.env,
                **image.env,
                **resolve_env_templates(verifier.env),
                **verifier_account.env,
            }
            with materialized_artifacts(task, verifier_root, artifacts, verifier_account.identity):
                isolated = Workspace(
                    verifier_root,
                    guest_path="/",
                    system_mounts=(),
                    mounts=(
                        Mount("dev", dst="/dev"),
                        Mount("proc", dst="/proc"),
                        Mount("rw", src=str(LOGS), dst="/logs"),
                        *GPU_DRIVER_MOUNTS,
                        *(
                            [Mount("ro", src="/etc/resolv.conf", dst="/etc/resolv.conf")]
                            if public
                            else []
                        ),
                    ),
                    env=verifier_env,
                    network=verifier.network.enabled,
                    allowed_hosts=verifier_access,
                    credentials_dir=RUNTIME_ROOT / "verifier-keys",
                    hand_over_root=False,
                    require_isolation=True,
                )
                try:
                    await isolated.start()
                    execution = await isolated.run(
                        verifier_command(test_script, "/tests/test.sh"),
                        env=verifier_env,
                        cwd=image.workdir,
                        identity=verifier_account.identity,
                        inherit_workspace_env=False,
                        allowed_hosts=verifier_access,
                        no_new_privs=False,
                        max_wait=task.verifier_timeout,
                        writable_hosts=True,
                    )
                finally:
                    await isolated.stop()
        finally:
            test_script.chmod(test_mode)
    return evaluation(execution, task.verifier_timeout)


def evaluation(execution: ProcessResult, timeout_sec: float) -> EvaluationResult:
    info: dict[str, Any] = {
        "exit_code": execution.returncode,
        "stdout": execution.stdout.decode("utf-8", "replace")[-4000:],
        "stderr": execution.stderr.decode("utf-8", "replace")[-4000:],
    }
    if execution.timed_out:
        info["verifier_timeout_sec"] = timeout_sec
        return EvaluationResult(
            isError=True,
            content=f"Harbor verifier timed out after {timeout_sec:.0f}s",
            info=info,
        )

    score, reward_info = reward()
    info.update(reward_info)
    if score is None:
        return EvaluationResult(
            isError=True,
            content="Harbor verifier did not write a numeric reward",
            info=info,
        )
    return EvaluationResult(reward=score, info=info)


def reward() -> tuple[float | None, dict[str, Any]]:
    reward_json = VERIFIER_LOGS / "reward.json"
    if reward_json.is_file():
        try:
            data = json.loads(reward_json.read_text("utf-8"))
        except json.JSONDecodeError:
            return None, {"reward_parse_error": "reward.json is not valid JSON"}
        candidates = [data]
        if isinstance(data, dict):
            candidates.extend((data.get("reward"), data.get("score")))
        for value in candidates:
            if isinstance(value, int | float) and not isinstance(value, bool):
                score = float(value)
                if math.isfinite(score):
                    return score, {"reward_file": str(reward_json)}
        return None, {"reward_parse_error": "reward.json has no numeric reward"}

    reward_text = VERIFIER_LOGS / "reward.txt"
    if reward_text.is_file():
        text = reward_text.read_text("utf-8").strip()
        try:
            score = float(text)
        except ValueError:
            score = math.nan
        if math.isfinite(score):
            return score, {"reward_file": str(reward_text)}
        return None, {"reward_parse_error": text}
    return None, {}

"""HUD environment served by every adapted Harbor image."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import grp
import json
import math
import os
import pwd
import shlex
import shutil
import socket
import tempfile
import uuid
from collections.abc import AsyncGenerator, Iterator  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hud.capabilities import Capability
from hud.environment import Answer, Environment, Mount, Peer, Workspace
from hud.environment.egress import ANY_HOST, BRIDGE_PORT, VISITOR_PORT
from hud.graders import EvaluationResult
from hud.utils.process import ProcessResult, create_process_group_exec

if TYPE_CHECKING:
    from hud.environment.namespace import NamespaceProcess

ROOT = Path("/media/hud")
TESTS = Path("/tests")
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
AGENT_ANSWER = LOGS / "agent_answer.txt"
HANDOFFS = ROOT / "handoffs"
HANDOFF_ANSWER = "agent-answer.txt"
HANDOFF_ERROR = "error.txt"
DOCKER_SOCKET = ROOT / "docker.sock"
DOCKER = ROOT / "bin" / "docker"
CONFIG = json.loads((ROOT / "config.json").read_text("utf-8"))


def load_image_config(name: str) -> dict[str, Any]:
    value = json.loads((ROOT / name).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an OCI image config object")
    return value


def image_environment(config: dict[str, Any]) -> dict[str, str]:
    entries = config.get("Env") or []
    if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
        raise ValueError("OCI image Env must be a list of strings")
    return {
        key: value
        for entry in entries
        for key, separator, value in (entry.partition("="),)
        if separator
    }


IMAGE_CONFIG = load_image_config("image-config.json")
VERIFIER_IMAGE_CONFIG = load_image_config("verifier-image-config.json")
CONFIG["image_env"] = image_environment(IMAGE_CONFIG)
if CONFIG["image_user"] is None:
    CONFIG["image_user"] = IMAGE_CONFIG.get("User") or None
if CONFIG["workdir"] is None:
    CONFIG["workdir"] = IMAGE_CONFIG.get("WorkingDir") or "/"
if CONFIG["entrypoint"] is None:
    CONFIG["entrypoint"] = IMAGE_CONFIG.get("Entrypoint") or []
exposed = IMAGE_CONFIG.get("ExposedPorts") or {}
if not isinstance(exposed, dict):
    raise ValueError("OCI image ExposedPorts must be an object")
CONFIG["ports"] = sorted(
    {
        *CONFIG["ports"],
        *(int(port) for value in exposed if (port := str(value).partition("/")[0]).isdigit()),
    }
)
for service, config_path in CONFIG["peer_image_configs"].items():
    peer_config = load_image_config(config_path)
    peer_exposed = peer_config.get("ExposedPorts") or {}
    if not isinstance(peer_exposed, dict):
        raise ValueError(
            f"OCI image ExposedPorts for Compose service {service!r} must be an object"
        )
    peer_ports = sorted(
        int(port)
        for value in peer_exposed
        if (port := str(value).partition("/")[0]).isdigit()
        and str(value).partition("/")[2] in {"", "tcp"}
    )
    if not peer_ports:
        raise ValueError(
            f"Compose service {service!r} declares no TCP ports in Compose or its image"
        )
    CONFIG["peers"].extend({"name": service, "port": port} for port in peer_ports)
if conflict := set(CONFIG["ports"]) & {BRIDGE_PORT, VISITOR_PORT, 8765}:
    raise ValueError(f"Harbor main service port {min(conflict)} conflicts with a HUD reserved port")
verifier_image = CONFIG["verifier_image"]
if verifier_image["user"] is None:
    verifier_image["user"] = VERIFIER_IMAGE_CONFIG.get("User") or None
if verifier_image["workdir"] is None:
    verifier_image["workdir"] = VERIFIER_IMAGE_CONFIG.get("WorkingDir") or "/"
verifier_image["env"] = image_environment(VERIFIER_IMAGE_CONFIG)

os.environ.update(CONFIG["environment"]["env"])
WORKDIR = Path(CONFIG["workdir"])
os.chdir(WORKDIR)


def network(phase: dict[str, Any] | None) -> tuple[bool, frozenset[str]]:
    baseline = CONFIG["environment"]
    mode = phase["network_mode"] if phase is not None else baseline["network_mode"]
    hosts = phase["allowed_hosts"] if phase is not None else baseline["allowed_hosts"]
    if mode is None:
        mode = baseline["network_mode"]
        hosts = baseline["allowed_hosts"]
    if mode == "no-network":
        return False, frozenset()
    if mode == "allowlist":
        return True, frozenset(hosts or [])
    return True, frozenset({ANY_HOST})


def identity(
    phase: dict[str, Any] | None,
    *,
    image_user: str | int | None,
    root: Path | None = None,
) -> tuple[int, int] | None:
    declared = phase["user"] if phase is not None else None
    user = str(declared if declared is not None else image_user or "")
    if not user:
        return None
    user_name, separator, group_name = user.partition(":")
    if root is None:
        account = None
        if user_name.isdigit():
            user_id = int(user_name)
            with contextlib.suppress(KeyError):
                account = pwd.getpwuid(user_id)
        else:
            try:
                account = pwd.getpwnam(user_name)
            except KeyError as error:
                raise ValueError(f"Harbor user {user!r} does not exist in this image") from error
            user_id = account.pw_uid
        primary_group = account.pw_gid if account is not None else 0
    else:
        passwd = root / "etc/passwd"
        accounts = {
            fields[0]: (int(fields[2]), int(fields[3]))
            for line in (passwd.read_text("utf-8").splitlines() if passwd.is_file() else ())
            if len(fields := line.split(":")) >= 4
        }
        if user_name.isdigit():
            user_id = int(user_name)
            primary_group = next(
                (group_id for uid, group_id in accounts.values() if uid == user_id),
                0,
            )
        else:
            try:
                user_id, primary_group = accounts[user_name]
            except KeyError as error:
                raise ValueError(f"Harbor user {user!r} does not exist in this image") from error

    if separator:
        if group_name.isdigit():
            group_id = int(group_name)
        elif root is not None:
            group = root / "etc/group"
            groups = {
                fields[0]: int(fields[2])
                for line in (group.read_text("utf-8").splitlines() if group.is_file() else ())
                if len(fields := line.split(":")) >= 3
            }
            try:
                group_id = groups[group_name]
            except KeyError as error:
                raise ValueError(
                    f"Harbor group {group_name!r} does not exist in this image"
                ) from error
        else:
            try:
                group_id = grp.getgrnam(group_name).gr_gid
            except KeyError as error:
                raise ValueError(
                    f"Harbor group {group_name!r} does not exist in this image"
                ) from error
    else:
        # Docker resolves a known account's primary group; a bare numeric uid
        # with no passwd entry keeps the container default group (root).
        group_id = primary_group
    return None if (user_id, group_id) == (0, 0) else (user_id, group_id)


def home(user_id: int | None, *, root: Path | None = None) -> str | None:
    if user_id is None:
        return None
    if root is not None:
        passwd = root / "etc/passwd"
        return next(
            (
                fields[5]
                for line in (passwd.read_text("utf-8").splitlines() if passwd.is_file() else ())
                if len(fields := line.split(":")) >= 6 and int(fields[2]) == user_id
            ),
            None,
        )
    with contextlib.suppress(KeyError):
        return pwd.getpwuid(user_id).pw_dir
    return None


agent = CONFIG["agent"]
image_identity = identity(None, image_user=CONFIG["image_user"])
agent_identity = identity(agent, image_user=CONFIG["image_user"])
agent_uid = agent_identity[0] if agent_identity is not None else None
agent_network, agent_hosts = network(agent)
environment_hosts = network(None)[1]
rooted_at_filesystem = len(WORKDIR.parts) == 1
harness_parent = ROOT.parent
harness_mounts = (
    Mount("tmpfs", dst=str(harness_parent)),
    *(
        Mount("rw", src=str(path), dst=str(path))
        for path in sorted(harness_parent.iterdir())
        if path != ROOT
    ),
)
agent_mounts = (
    *harness_mounts,
    Mount("tmpfs", dst=str(TESTS)),
    Mount("tmpfs", dst=str(VERIFIER_LOGS)),
    Mount("ro", src="/dev/null", dst=str(AGENT_ANSWER)),
)

env = Environment(CONFIG["name"])
verifier_lock = asyncio.Lock()
for capability in CONFIG["capabilities"]:
    env.add_capability(Capability.from_manifest(capability))
workspace = env.workspace(
    WORKDIR,
    guest_path=WORKDIR.as_posix(),
    system_mounts=(
        Mount("rw", src="/", dst="/"),
        Mount("dev", dst="/dev"),
        Mount("proc", dst="/proc"),
    ),
    mounts=agent_mounts,
    credentials_dir=ROOT / "session-keys",
    hosts_path=ROOT / "hosts",
    shell_uid=agent_uid,
    shell_gid=agent_identity[1] if agent_identity is not None else None,
    hand_over_root=False,
    track_files=False if rooted_at_filesystem else None,
    env={
        **CONFIG["image_env"],
        **CONFIG["environment"]["env"],
        **agent["env"],
        **({"HOME": agent_home} if (agent_home := home(agent_uid)) else {}),
    },
    network=agent_network,
    allowed_hosts=agent_hosts,
    peers=[
        Peer(peer["name"], peer["port"], target=(peer["name"], peer["port"]))
        for peer in CONFIG["peers"]
    ],
    local_aliases=CONFIG["local_aliases"],
    ports=CONFIG["ports"],
    require_isolation=True,
)


async def start_entrypoint() -> NamespaceProcess | None:
    entrypoint = CONFIG["entrypoint"]
    if not entrypoint:
        return None
    sandbox = await workspace.sandbox_pid()
    if sandbox is None:
        raise RuntimeError("Harbor entrypoints require an isolated workspace")
    process = await workspace.launch(
        [*entrypoint, "sh", "-c", "sleep infinity"],
        env=CONFIG["environment"]["env"],
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
    healthcheck = CONFIG["environment"]["healthcheck"]
    if healthcheck is not None:
        loop = asyncio.get_running_loop()
        start_period = healthcheck["start_period_sec"]
        start_period_end = loop.time() + start_period
        delay = (
            healthcheck["start_interval_sec"] if start_period > 0 else healthcheck["interval_sec"]
        )
        failures = 0
        while True:
            await asyncio.sleep(delay)
            in_start_period = loop.time() < start_period_end
            if entrypoint is not None and entrypoint.returncode is not None:
                raise RuntimeError(
                    f"Harbor environment entrypoint exited with status {entrypoint.returncode}"
                )
            result = await workspace.run(
                ["sh", "-c", healthcheck["command"]],
                env=CONFIG["environment"]["env"],
                identity=image_identity,
                inherit_workspace_env=False,
                allowed_hosts=None if environment_hosts == agent_hosts else environment_hosts,
                no_new_privs=False,
                max_wait=healthcheck["timeout_sec"],
                scope="environment",
            )
            if result.returncode == 0 and not result.timed_out:
                break

            if in_start_period:
                delay = healthcheck["start_interval_sec"]
            else:
                failures += 1
                if failures >= healthcheck["retries"]:
                    detail = result.stderr.decode("utf-8", "replace").strip()
                    raise RuntimeError(
                        f"Harbor environment healthcheck failed after {failures} attempts"
                        + (f": {detail}" if detail else "")
                    )
                delay = healthcheck["interval_sec"]

    pending = set(CONFIG["healthy_services"])
    if not pending:
        return
    services = await compose_containers()
    if missing := pending - services.keys():
        raise RuntimeError(f"Compose service {min(missing)!r} is not running")
    while pending:
        for service in sorted(pending):
            result = await docker(
                "inspect",
                "--format",
                "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                services[service],
            )
            state, _, health = result.stdout.decode().strip().partition(" ")
            if health == "healthy":
                pending.remove(service)
            elif health == "unhealthy":
                raise RuntimeError(f"Compose service {service!r} is unhealthy")
            elif state not in {"running", "restarting"}:
                raise RuntimeError(f"Compose service {service!r} is {state}")
            elif not health and state == "running":
                raise RuntimeError(f"Compose service {service!r} has no health status")
        if pending:
            if entrypoint is not None and entrypoint.returncode is not None:
                raise RuntimeError(
                    f"Harbor environment entrypoint exited with status {entrypoint.returncode}"
                )
            await asyncio.sleep(1.0)


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
    if not DOCKER_SOCKET.exists():
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


def copy_artifact(source: Path, target: Path, exclude: list[str]) -> None:
    if source.is_symlink():
        raise RuntimeError(f"artifact {source} is a symbolic link")
    if source.resolve(strict=False) != source.absolute():
        raise RuntimeError(f"artifact {source} has a symbolic link in its path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for root, directories, files in os.walk(source, followlinks=False):
            for name in (*directories, *files):
                entry = Path(root, name)
                if entry.is_symlink():
                    raise RuntimeError(f"artifact {source} contains symbolic link {entry}")
        shutil.copytree(source, target)
        exclude_artifact_paths(target, exclude)
    elif source.exists() or source.is_symlink():
        shutil.copy2(source, target, follow_symlinks=False)


class ActorHandoff(BaseModel):
    handoff: str


def artifact_path(artifact: dict[str, Any], artifacts: Path) -> Path:
    relative = artifact.get("destination") or artifact["source"].lstrip("/").rstrip("/")
    return artifacts / relative


async def collect(task: dict[str, Any], artifacts: Path) -> None:
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

    for hook in task["collect"]:
        service = hook["service"]
        container_id = await container(service)
        if container_id:
            await docker(
                "exec",
                container_id,
                "sh",
                "-c",
                hook["command"],
                max_wait=hook["timeout_sec"],
            )
        else:
            execution = await workspace.run(
                ["sh", "-c", hook["command"]],
                mounts=harness_mounts,
                identity=image_identity,
                inherit_workspace_env=False,
                allowed_hosts=None,
                no_new_privs=False,
                max_wait=hook["timeout_sec"],
                scope="environment",
            )
            if execution.timed_out:
                raise TimeoutError(
                    f"collect hook on {service!r} timed out after {hook['timeout_sec']:g}s"
                )
            if execution.returncode != 0:
                detail = execution.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"collect hook on {service!r} failed: {detail}")

    for artifact in task["artifacts"]:
        source = artifact["source"]
        target = artifact_path(artifact, artifacts)
        exclude = artifact.get("exclude", [])
        service = artifact["service"]
        container_id = await container(service)
        if container_id:
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = await docker(
                "cp",
                f"{container_id}:{source.rstrip('/') or '/'}",
                str(target),
                max_wait=task["verifier_timeout"],
                check=False,
            )
            if copied.returncode != 0:
                continue
            exclude_artifact_paths(target, exclude)
        else:
            copy_artifact(Path(source), target, exclude)
        if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
            raise RuntimeError(f"artifact {source} contains a symbolic link")


@env.template(id="run", description="Run a Harbor task")
async def run(instruction: str, task: dict[str, Any]) -> AsyncGenerator[Any, Any]:
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
        if task["separate_verifier"]:
            token = uuid.uuid4().hex
            handoff = HANDOFFS / token
            artifacts = handoff / "artifacts"
            handoff.mkdir(parents=True)
            (handoff / HANDOFF_ANSWER).write_text(
                "" if answer is None else str(answer),
                encoding="utf-8",
            )
            try:
                await collect(task, artifacts)
            except Exception as error:
                detail = str(error)
                (handoff / HANDOFF_ERROR).write_text(detail, encoding="utf-8")
                result = {
                    "score": 0.0,
                    "handoff": token,
                    "content": detail,
                    "isError": True,
                }
            else:
                result = {"score": 0.0, "handoff": token}
            await workspace.terminate_sessions()
            yield result
        else:
            yield await grade(task["id"], task["verifier_timeout"], answer)
    finally:
        clear_grading_files()
        await workspace.discard_sandbox()
        if entrypoint is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(entrypoint.wait(), 10.0)


if CONFIG["verifier_root"] is not None:

    @env.template(id="verify", description="Verify a Harbor task", returns=ActorHandoff)
    async def verify(task: dict[str, Any]) -> AsyncGenerator[Any, Any]:
        received = yield ""
        if not isinstance(received, Answer) or not isinstance(received.content, ActorHandoff):
            raise ValueError("Harbor verifier requires an actor handoff")
        token = received.content.handoff
        if len(token) != 32 or not token.isalnum():
            raise ValueError("Harbor verifier received an invalid actor handoff")
        handoff = HANDOFFS / token
        if not handoff.is_dir():
            raise ValueError("Harbor actor handoff is unavailable in this runtime")
        try:
            if (error := handoff / HANDOFF_ERROR).is_file():
                raise RuntimeError(error.read_text("utf-8"))
            yield await grade_separate(task, handoff)
        finally:
            clear_grading_files()
            shutil.rmtree(handoff, ignore_errors=True)


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
    shutil.copytree(ROOT / "tests" / task_id, TESTS, symlinks=True, dirs_exist_ok=True)
    test_script = TESTS / "test.sh"
    test_script.chmod(test_script.stat().st_mode | 0o111)

    clear(VERIFIER_LOGS)
    AGENT_ANSWER.write_text("" if answer is None else str(answer), encoding="utf-8")

    verifier = CONFIG["verifier"]
    verifier_identity = identity(verifier, image_user=CONFIG["image_user"])
    verifier_uid = verifier_identity[0] if verifier_identity is not None else None
    verifier_env = {**CONFIG["environment"]["env"], **verifier["env"]}
    if verifier_uid is not None:
        assert verifier_identity is not None
        for root in (TESTS, VERIFIER_LOGS):
            for path in (root, *root.rglob("*")):
                os.lchown(path, *verifier_identity)
        if verifier_home := home(verifier_uid):
            verifier_env["HOME"] = verifier_home

    verifier_hosts = network(verifier)[1]
    execution = await workspace.run(
        verifier_command(test_script),
        mounts=harness_mounts,
        env=verifier_env,
        identity=verifier_identity,
        inherit_workspace_env=False,
        allowed_hosts=verifier_hosts,
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
    task: dict[str, Any],
    verifier_root: Path,
    artifacts: Path,
    verifier_identity: tuple[int, int] | None,
) -> Iterator[list[Mount]]:
    driver_files = (
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
        if Path("/dev/nvidiactl").exists()
        else set()
    )
    mounts = [
        Mount("dev", dst="/dev"),
        Mount("proc", dst="/proc"),
        Mount("rw", src=str(LOGS), dst="/logs"),
        *(Mount("ro", src=str(path), dst=str(path)) for path in sorted(driver_files)),
    ]
    with tempfile.TemporaryDirectory(prefix="verifier-backup-", dir=ROOT) as directory:
        backup_root = Path(directory)
        replacements: list[tuple[Path, Path | None]] = []
        modes: dict[Path, int] = {}
        entries: dict[Path, set[str]] = {}
        created: list[Path] = []
        try:
            for artifact in task["artifacts"]:
                staged = artifact_path(artifact, artifacts)
                if not staged.exists() and not staged.is_symlink():
                    continue
                destination = artifact["source"].rstrip("/") or "/"
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
            yield mounts
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
    task: dict[str, Any],
    handoff: Path,
) -> EvaluationResult:
    async with verifier_lock:
        verifier_root = Path(CONFIG["verifier_root"])
        test_script = verifier_root / "tests/test.sh"
        test_mode = (await asyncio.to_thread(test_script.stat)).st_mode
        await asyncio.to_thread(test_script.chmod, test_mode | 0o111)
        await asyncio.to_thread(clear, VERIFIER_LOGS)
        await asyncio.to_thread(VERIFIER_LOGS.chmod, 0o777)
        await asyncio.to_thread(LOGS.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            AGENT_ANSWER.write_bytes,
            (handoff / HANDOFF_ANSWER).read_bytes(),
        )

        try:
            verifier = CONFIG["verifier"]
            verifier_network, verifier_hosts = network(verifier)
            verifier_mode = verifier["network_mode"] or CONFIG["environment"]["network_mode"]
            verifier_access = None if verifier_mode == "public" else verifier_hosts
            image = CONFIG["verifier_image"]
            verifier_identity = identity(
                verifier,
                image_user=image["user"],
                root=verifier_root,
            )
            with materialized_artifacts(
                task,
                verifier_root,
                handoff / "artifacts",
                verifier_identity,
            ) as mounts:
                verifier_mounts = tuple(mounts)
                if verifier_mode == "public":
                    verifier_mounts += (
                        Mount("ro", src="/etc/resolv.conf", dst="/etc/resolv.conf"),
                    )
                verifier_uid = verifier_identity[0] if verifier_identity is not None else None
                verifier_env = {
                    **CONFIG["environment"]["env"],
                    **image["env"],
                    **verifier["env"],
                }
                if verifier_home := home(verifier_uid, root=verifier_root):
                    verifier_env["HOME"] = verifier_home
                isolated = Workspace(
                    verifier_root,
                    guest_path="/",
                    system_mounts=(),
                    mounts=verifier_mounts,
                    env=verifier_env,
                    network=verifier_network,
                    allowed_hosts=verifier_access,
                    credentials_dir=ROOT / "verifier-keys",
                    hand_over_root=False,
                    require_isolation=True,
                )
                try:
                    await isolated.start()
                    execution = await isolated.run(
                        verifier_command(test_script, "/tests/test.sh"),
                        env=verifier_env,
                        cwd=image["workdir"],
                        identity=verifier_identity,
                        inherit_workspace_env=False,
                        allowed_hosts=verifier_access,
                        no_new_privs=False,
                        max_wait=task["verifier_timeout"],
                        writable_hosts=True,
                    )
                finally:
                    await isolated.stop()
        finally:
            test_script.chmod(test_mode)
    return evaluation(execution, task["verifier_timeout"])


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

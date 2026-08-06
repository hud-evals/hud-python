"""HUD environment served by every adapted Harbor image."""

from __future__ import annotations

import asyncio
import contextlib
import grp
import json
import math
import os
import pwd
import shutil
import socket
import tempfile
from collections.abc import AsyncGenerator, Iterator  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.capabilities import Capability
from hud.environment import Environment, Mount, Peer, Workspace
from hud.environment.egress import ANY_HOST
from hud.graders import EvaluationResult
from hud.utils.process import ProcessResult, create_process_group_exec

if TYPE_CHECKING:
    from hud.environment.namespace import NamespaceProcess

ROOT = Path("/media/hud")
TESTS = Path("/tests")
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
AGENT_ANSWER = LOGS / "agent_answer.txt"
ARTIFACTS = ROOT / "artifacts"
DOCKER_SOCKET = ROOT / "docker.sock"
DOCKER = ROOT / "bin" / "docker"
CONFIG = json.loads((ROOT / "tasks.json").read_text("utf-8"))

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
    if healthcheck is None:
        return

    loop = asyncio.get_running_loop()
    start_period = healthcheck["start_period_sec"]
    start_period_end = loop.time() + start_period
    delay = healthcheck["start_interval_sec"] if start_period > 0 else healthcheck["interval_sec"]
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
            return

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


def copy_artifact(source: Path, target: Path) -> None:
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
    elif source.exists() or source.is_symlink():
        shutil.copy2(source, target, follow_symlinks=False)


async def collect(task: dict[str, Any]) -> None:
    clear(ARTIFACTS)
    services: dict[str, str] = {}

    async def container(service: str) -> str:
        service = "main" if service == "workspace" else service
        if service == "main":
            return ""
        if service in services:
            return services[service]
        if not DOCKER_SOCKET.exists():
            raise RuntimeError(
                f"collecting from Compose service {service!r} requires runtime service access"
            )
        if not services:
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
            for line in listed.stdout.decode().splitlines():
                container_id, service_name = line.split(maxsplit=1)
                services[service_name] = container_id
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
        target = ARTIFACTS / source.lstrip("/").rstrip("/")
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
        else:
            copy_artifact(Path(source), target)
        if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
            raise RuntimeError(f"artifact {source} contains a symbolic link")


def register(task: dict[str, Any]) -> None:
    task_dir = ROOT / "tasks" / task["id"]

    @env.template(
        id=task["id"],
        description=task["description"] or f"Harbor task {task['id']}",
    )
    async def run() -> AsyncGenerator[Any, Any]:
        clear_grading_files()
        AGENT_ANSWER.parent.mkdir(parents=True, exist_ok=True)
        AGENT_ANSWER.touch()
        entrypoint = None
        try:
            entrypoint = await start_entrypoint()
            await wait_until_healthy(entrypoint)
            answer = yield (task_dir / "instruction.md").read_text("utf-8")
            if entrypoint is not None and entrypoint.returncode is not None:
                raise RuntimeError(
                    f"Harbor environment entrypoint exited with status {entrypoint.returncode}"
                )
            if task["separate_verifier"]:
                await workspace.terminate_sessions()
                await collect(task)
                yield 0.0
            else:
                yield await grade(task_dir, task["verifier_timeout"], answer)
        finally:
            clear_grading_files()
            await workspace.discard_sandbox()
            if entrypoint is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(entrypoint.wait(), 10.0)

    if task["separate_verifier"]:

        @env.template(
            id=f"{task['id']}:verify",
            description=f"Verify Harbor task {task['id']}",
        )
        async def verify() -> AsyncGenerator[Any, Any]:
            answer = yield ""
            try:
                yield await grade_separate(task, answer)
            finally:
                clear_grading_files()
                shutil.rmtree(ARTIFACTS, ignore_errors=True)


for task_config in CONFIG["tasks"]:
    register(task_config)


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


async def grade(task_dir: Path, timeout_sec: float, answer: Any) -> EvaluationResult:
    clear(TESTS)
    shutil.copytree(task_dir / "tests", TESTS, symlinks=True, dirs_exist_ok=True)
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
        [str(test_script)],
        mounts=harness_mounts,
        env=verifier_env,
        identity=verifier_identity,
        inherit_workspace_env=False,
        allowed_hosts=verifier_hosts,
        no_new_privs=False,
        max_wait=timeout_sec,
    )
    return evaluation(execution, timeout_sec)


@contextlib.contextmanager
def artifact_mounts(task: dict[str, Any], verifier_root: Path) -> Iterator[list[Mount]]:
    mounts: list[Mount] = [
        Mount("dev", dst="/dev"),
        Mount("proc", dst="/proc"),
    ]
    bindings: list[tuple[Path | None, str]] = [(LOGS, "/logs")]
    for artifact in task["artifacts"]:
        source = artifact["source"].rstrip("/") or "/"
        staged = ARTIFACTS / source.lstrip("/")
        bindings.append((staged if staged.exists() or staged.is_symlink() else None, source))

    with tempfile.TemporaryDirectory(prefix="verifier-backup-", dir=ROOT) as directory:
        backup_root = Path(directory)
        replacements: list[tuple[Path, Path | None]] = []
        try:
            for staged, destination in bindings:
                target = verifier_root / destination.lstrip("/")
                compatible = (
                    staged is not None
                    and target.exists()
                    and not target.is_symlink()
                    and staged.is_dir() == target.is_dir()
                )
                if not compatible and (target.exists() or target.is_symlink()):
                    backup = backup_root / destination.lstrip("/")
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    target.rename(backup)
                    replacements.append((target, backup))
                elif staged is not None and not compatible:
                    replacements.append((target, None))
                if staged is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if staged.is_dir():
                    target.mkdir(exist_ok=True)
                else:
                    target.touch(exist_ok=True)
                mounts.append(Mount("rw", src=str(staged), dst=destination))
            yield mounts
        finally:
            for target, backup in reversed(replacements):
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
                if backup is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup.rename(target)


async def grade_separate(task: dict[str, Any], answer: Any) -> EvaluationResult:
    async with verifier_lock:
        verifier_root = Path(CONFIG["verifier_root"])
        test_script = verifier_root / "tests/test.sh"
        test_mode = (await asyncio.to_thread(test_script.stat)).st_mode
        await asyncio.to_thread(test_script.chmod, test_mode | 0o111)
        await asyncio.to_thread(clear, VERIFIER_LOGS)
        await asyncio.to_thread(VERIFIER_LOGS.chmod, 0o777)
        await asyncio.to_thread(LOGS.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            AGENT_ANSWER.write_text,
            "" if answer is None else str(answer),
            encoding="utf-8",
        )

        try:
            with artifact_mounts(task, verifier_root) as mounts:
                verifier = CONFIG["verifier"]
                verifier_network, verifier_hosts = network(verifier)
                image = CONFIG["verifier_image"]
                verifier_identity = identity(
                    verifier,
                    image_user=image["user"],
                    root=verifier_root,
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
                    mounts=mounts,
                    env=verifier_env,
                    network=verifier_network,
                    allowed_hosts=verifier_hosts,
                    credentials_dir=ROOT / "verifier-keys",
                    hand_over_root=False,
                    require_isolation=True,
                )
                try:
                    await isolated.start()
                    execution = await isolated.run(
                        ["/tests/test.sh"],
                        env=verifier_env,
                        cwd=image["workdir"],
                        identity=verifier_identity,
                        inherit_workspace_env=False,
                        allowed_hosts=verifier_hosts,
                        no_new_privs=False,
                        max_wait=task["verifier_timeout"],
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

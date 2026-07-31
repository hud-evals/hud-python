"""HUD environment served by every adapted Harbor image."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import pwd
import shutil
from collections.abc import AsyncGenerator  # noqa: TC003
from pathlib import Path
from typing import Any

from hud.environment import Environment, Mount
from hud.environment.egress import ANY_HOST
from hud.environment.workspace import install_identity_map
from hud.graders import EvaluationResult
from hud.utils.process import ProcessGroup, create_process_group_exec

ROOT = Path("/media/hud")
TESTS = Path("/tests")
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
CONFIG = json.loads((ROOT / "tasks.json").read_text("utf-8"))

os.environ.update(CONFIG["environment"]["env"])
WORKDIR = Path(CONFIG["workdir"])
os.chdir(WORKDIR)


def network(phase: dict[str, Any]) -> tuple[bool, frozenset[str]]:
    baseline = CONFIG["environment"]
    mode = phase.get("network_mode") or baseline["network_mode"]
    hosts = phase.get("allowed_hosts") if phase.get("network_mode") else baseline["allowed_hosts"]
    if mode == "no-network":
        return False, frozenset()
    if mode == "allowlist":
        return True, frozenset(hosts or [])
    return True, frozenset({ANY_HOST})


def uid(phase: dict[str, Any]) -> int | None:
    declared = phase.get("user")
    user = str(declared if declared is not None else CONFIG.get("image_user") or "")
    if not user or user in {"root", "0", "root:root", "0:0"}:
        return None
    name = user.split(":", 1)[0]
    if name.isdigit():
        return int(name)
    try:
        return pwd.getpwnam(name).pw_uid
    except KeyError as error:
        raise ValueError(f"Harbor user {user!r} does not exist in this image") from error


def home(user_id: int | None) -> str | None:
    if user_id is None:
        return None
    with contextlib.suppress(KeyError):
        return pwd.getpwuid(user_id).pw_dir
    return None


def harness_mounts() -> tuple[Mount, ...]:
    parent = ROOT.parent
    siblings = (
        [
            Mount("rw", src=str(path), dst=str(path))
            for path in sorted(parent.iterdir())
            if path != ROOT
        ]
        if parent.is_dir()
        else []
    )
    return (Mount("tmpfs", dst=str(parent)), *siblings)


agent = CONFIG["agent"]
agent_uid = uid(agent)
agent_network, agent_hosts = network(agent)
rooted_at_filesystem = len(WORKDIR.parts) == 1

env = Environment(CONFIG["name"])
workspace = env.workspace(
    WORKDIR,
    guest_path=WORKDIR.as_posix(),
    system_mounts=(
        Mount("rw", src="/", dst="/"),
        Mount("proc", dst="/proc"),
        Mount("dev", dst="/dev"),
    ),
    mounts=harness_mounts(),
    credentials_dir=ROOT / "session-keys",
    shell_uid=agent_uid,
    hand_over_root=False,
    track_files=False if rooted_at_filesystem else None,
    env={
        **CONFIG["environment"]["env"],
        **agent["env"],
        **({"HOME": agent_home} if (agent_home := home(agent_uid)) else {}),
    },
    network=agent_network,
    allowed_hosts=agent_hosts,
    require_isolation=True,
)


def register(task: dict[str, Any]) -> None:
    task_dir = ROOT / "tasks" / task["id"]

    @env.template(
        id=task["id"],
        description=task["description"] or f"Harbor task {task['id']}",
    )
    async def run() -> AsyncGenerator[Any, Any]:
        clear_grading_files()
        try:
            answer = yield (task_dir / "instruction.md").read_text("utf-8")
            yield await grade(task_dir, task["verifier_timeout"], answer)
        finally:
            clear_grading_files()
            await workspace.discard_sandbox()


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


async def grade(task_dir: Path, timeout_sec: float, answer: Any) -> EvaluationResult:
    clear(TESTS)
    shutil.copytree(task_dir / "tests", TESTS, symlinks=True, dirs_exist_ok=True)
    test_script = TESTS / "test.sh"
    test_script.chmod(test_script.stat().st_mode | 0o111)

    clear(VERIFIER_LOGS)
    write(LOGS / "agent_answer.txt", "" if answer is None else str(answer))

    verifier = CONFIG["verifier"]
    verifier_uid = uid(verifier)
    verifier_env = dict(verifier["env"])
    if verifier_uid is not None:
        if os.geteuid() == 0 and shutil.which("setpriv") is None:
            raise RuntimeError("setpriv is required to run the Harbor verifier as another user")
        for root in (TESTS, VERIFIER_LOGS):
            for path in (root, *root.rglob("*")):
                os.lchown(path, verifier_uid, verifier_uid)
        if verifier_home := home(verifier_uid):
            verifier_env["HOME"] = verifier_home

    verifier_network, verifier_hosts = network(verifier)
    command = [str(test_script)]
    if verifier_network:
        sandbox = await workspace.sandbox_pid()
        if sandbox is None:
            raise RuntimeError("the Harbor verifier requires the workspace sandbox")
        async with workspace.visiting(verifier_hosts) as visitor_env:
            process = await create_process_group_exec(
                *workspace.enter_argv(
                    sandbox,
                    command,
                    env={**verifier_env, **visitor_env},
                    identity=verifier_uid,
                    inherit_workspace_env=False,
                    preserve_credentials=True,
                    no_new_privs=False,
                ),
                cwd=WORKDIR,
                env={**os.environ, **verifier_env, **visitor_env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            execution = await process.complete(max_wait=timeout_sec)
    else:
        process = await isolated(command, verifier_env, verifier_uid)
        execution = await process.complete(max_wait=timeout_sec)

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


async def isolated(
    command: list[str],
    verifier_env: dict[str, str],
    verifier_uid: int | None,
) -> ProcessGroup:
    info_read, info_write = os.pipe()
    block_read, block_write = os.pipe()
    try:
        os.set_inheritable(info_write, True)
        os.set_inheritable(block_read, True)
        if verifier_uid is not None:
            command = [
                shutil.which("setpriv") or "setpriv",
                "--reuid",
                str(verifier_uid),
                "--regid",
                str(verifier_uid),
                "--clear-groups",
                "--",
                *command,
            ]
        process = await create_process_group_exec(
            *workspace.bwrap_argv(
                command,
                cwd=WORKDIR.as_posix(),
                env=verifier_env,
                inherit_workspace_env=False,
                info_fd=info_write,
                userns_block_fd=block_read,
                network=False,
                mount_hosts=False,
                isolate_processes=False,
            ),
            cwd=WORKDIR,
            env={**os.environ, **verifier_env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=(info_write, block_read),
        )
        os.close(info_write)
        os.close(block_read)
        info_write = block_read = -1
        await install_identity_map(info_read, block_write)
        return process
    finally:
        os.close(info_read)
        os.close(block_write)
        for descriptor in (info_write, block_read):
            if descriptor != -1:
                os.close(descriptor)


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


def write(path: Path, text: str) -> None:
    if path.is_symlink():
        path.unlink()
    path.write_text(text, encoding="utf-8")

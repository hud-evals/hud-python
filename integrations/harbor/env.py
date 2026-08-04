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
import subprocess
from collections.abc import AsyncGenerator  # noqa: TC003
from pathlib import Path
from typing import Any

from hud.environment import Environment, Mount
from hud.environment.egress import ANY_HOST
from hud.graders import EvaluationResult

ROOT = Path("/media/hud")
TESTS = Path("/tests")
LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
AGENT_ANSWER = LOGS / "agent_answer.txt"
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


def identity(phase: dict[str, Any]) -> tuple[int, int] | None:
    declared = phase.get("user")
    user = str(declared if declared is not None else CONFIG.get("image_user") or "")
    if not user:
        return None
    user_name, separator, group_name = user.partition(":")
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

    if separator:
        if group_name.isdigit():
            group_id = int(group_name)
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
        group_id = account.pw_gid if account is not None else 0
    return None if (user_id, group_id) == (0, 0) else (user_id, group_id)


def home(user_id: int | None) -> str | None:
    if user_id is None:
        return None
    with contextlib.suppress(KeyError):
        return pwd.getpwuid(user_id).pw_dir
    return None


agent = CONFIG["agent"]
image_identity = identity({})
agent_identity = identity(agent)
agent_uid = agent_identity[0] if agent_identity is not None else None
agent_network, agent_hosts = network(agent)
environment_hosts = network({})[1]
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
workspace = env.workspace(
    WORKDIR,
    guest_path=WORKDIR.as_posix(),
    system_mounts=(
        Mount("rw", src="/", dst="/"),
        Mount("proc", dst="/proc"),
        Mount("dev", dst="/dev"),
    ),
    mounts=agent_mounts,
    credentials_dir=ROOT / "session-keys",
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
    require_isolation=True,
)


async def start_entrypoint() -> asyncio.subprocess.Process | None:
    entrypoint = CONFIG["entrypoint"]
    if not entrypoint:
        return None
    sandbox = await workspace.sandbox_pid()
    if sandbox is None:
        raise RuntimeError("Harbor entrypoints require an isolated workspace")
    process = await asyncio.create_subprocess_exec(
        *workspace.enter_argv(
            sandbox,
            [*entrypoint, "sh", "-c", "sleep infinity"],
            env=CONFIG["environment"]["env"],
            identity=image_identity,
            inherit_workspace_env=False,
            no_new_privs=False,
        ),
        stdin=subprocess.DEVNULL,
        env={},
    )
    await asyncio.sleep(0)
    if process.returncode is not None:
        raise RuntimeError(f"Harbor environment entrypoint exited with status {process.returncode}")
    return process


async def wait_until_healthy(entrypoint: asyncio.subprocess.Process | None) -> None:
    healthcheck = CONFIG["environment"]["healthcheck"]
    if healthcheck is None:
        return

    loop = asyncio.get_running_loop()
    start_period_end = loop.time() + healthcheck["start_period_sec"]
    failures = 0
    while True:
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
            allowed_hosts=environment_hosts,
            no_new_privs=False,
            max_wait=healthcheck["timeout_sec"],
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
        await asyncio.sleep(delay)


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
            yield await grade(task_dir, task["verifier_timeout"], answer)
        finally:
            clear_grading_files()
            await workspace.discard_sandbox()
            if entrypoint is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(entrypoint.wait(), 10.0)


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
    verifier_identity = identity(verifier)
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

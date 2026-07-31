"""Runtime for an adapted Harbor image."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import pwd
import shutil
from collections.abc import AsyncGenerator, Awaitable, Callable  # noqa: TC003
from pathlib import Path
from typing import Any

from hud.environment import Environment, Mount
from hud.environment.workspace import Workspace, install_identity_map
from hud.utils.process import ProcessGroup, create_process_group_exec

from ._adapt import _copy_task_content, _validated_user
from ._load import DEFAULT_VERIFIER_TIMEOUT, HUD_ROOT, HarborTask, TaskConfig, slugify

LOGGER = logging.getLogger(__name__)

LOGS = Path("/logs")
VERIFIER_LOGS = LOGS / "verifier"
TESTS = Path("/tests")


def phase_uid(task: HarborTask | Path, role: str) -> int | None:
    """Resolve the identity for one Harbor phase inside the adapted image."""
    if not isinstance(task, HarborTask):
        task = HarborTask.read(task)
    user = _validated_user(task.config.phase_user(role), _image_user())
    if user is None:
        return None
    name = user.split(":", 1)[0]
    if name.isdigit():
        return int(name)
    try:
        return pwd.getpwnam(name).pw_uid
    except KeyError as error:
        raise ValueError(f"declared user {user!r} does not exist in this image") from error


def phase_home(uid: int | None) -> str | None:
    """Return the home directory for a phase that drops to uid."""
    if uid is None:
        return None
    with contextlib.suppress(KeyError):
        return pwd.getpwuid(uid).pw_dir
    return None


def _image_user() -> str | None:
    """Read the final image USER recorded by the adaptation layer."""
    with contextlib.suppress(OSError):
        return (HUD_ROOT / "image-user").read_text(encoding="utf-8").strip() or None
    return None


def _harness_out_of_view() -> tuple[Mount, ...]:
    """Hide the adapted harness tree from agent sessions."""
    parent = HUD_ROOT.parent
    mounts = [Mount("tmpfs", dst=str(parent))]
    if parent.is_dir():
        mounts += [
            Mount("rw", src=str(sibling), dst=str(sibling))
            for sibling in sorted(parent.iterdir())
            if sibling != HUD_ROOT
        ]
    return tuple(mounts)


def environment(ref: str | Path = HUD_ROOT / "tasks", *, name: str | None = None) -> Environment:
    """Serve the task directories baked into an adapted Harbor image."""
    root = Path(ref)
    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not task_dirs:
        raise ValueError(f"no Harbor tasks under {root}")

    tasks = [HarborTask.read(task_dir) for task_dir in task_dirs]
    source = tasks[0]
    workdir = Path.cwd()
    rooted_at_filesystem = workdir == Path("/")
    if rooted_at_filesystem:
        LOGGER.warning(
            "%s has no WORKDIR and declares no [environment] workdir; serving from / "
            "with file tracking disabled",
            root,
        )

    agent_uid = phase_uid(source, "agent")
    env = Environment(name or slugify(root.name))
    workspace = env.workspace(
        workdir,
        guest_path=workdir.as_posix(),
        system_mounts=(
            Mount("rw", src="/", dst="/"),
            Mount("proc", dst="/proc"),
            Mount("dev", dst="/dev"),
        ),
        mounts=_harness_out_of_view(),
        credentials_dir=HUD_ROOT / "session-keys",
        shell_uid=agent_uid,
        hand_over_root=False,
        track_files=False if rooted_at_filesystem else None,
        env={
            **({"HOME": home} if (home := phase_home(agent_uid)) else {}),
            **source.config.agent.env,
        },
        network=source.config.network("agent"),
        allowed_hosts=source.config.allowed_hosts("agent"),
        require_isolation=True,
    )

    for task in tasks:
        _register(env, task, workdir, workspace)
    return env


def _register(env: Environment, task: HarborTask, workdir: Path, workspace: Workspace) -> None:
    config = task.config

    @env.template(
        id=task.path.name,
        description=config.task.description or f"Harbor task {task.path.name}",
    )
    async def _run_harbor_task() -> AsyncGenerator[Any, Any]:
        _hide_grading_dirs()
        try:
            answer = yield (task.path / "instruction.md").read_text(encoding="utf-8")
            yield await _grade(task, workdir, answer, workspace)
        finally:
            _hide_grading_dirs()
            await workspace.discard_sandbox()


def _sync_tests(task: HarborTask) -> None:
    """Leave /tests holding exactly the task's tests."""
    _reset_dir(TESTS)
    for child in (task.path / "tests").iterdir():
        _copy_task_content(child, TESTS / child.name)


def _hide_grading_dirs() -> None:
    """Remove the verifier's tests and verdict from the agent's view."""
    for path in (TESTS, VERIFIER_LOGS):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError:
            LOGGER.debug("could not remove %s; emptying it instead", path)
            _reset_dir(path)


def _reset_dir(path: Path) -> None:
    """Leave path as an existing, empty directory."""
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


async def _grade(
    task: HarborTask, workdir: Path, answer: Any, workspace: Workspace | None = None
) -> dict[str, Any]:
    _sync_tests(task)
    test_sh = TESTS / "test.sh"
    test_sh.chmod(test_sh.stat().st_mode | 0o111)
    command = [str(test_sh)]
    verifier_command = command

    config = task.config
    verifier_env: dict[str, str] = {}
    verifier_uid = phase_uid(task, "verifier")
    setpriv = shutil.which("setpriv")
    if verifier_uid is not None and setpriv is not None and os.geteuid() == 0:
        _reset_dir(VERIFIER_LOGS)
        for root_path in (TESTS, VERIFIER_LOGS):
            for target in (root_path, *root_path.rglob("*")):
                os.lchown(target, verifier_uid, verifier_uid)
        if home := phase_home(verifier_uid):
            verifier_env["HOME"] = home
        verifier_command = [
            setpriv,
            "--reuid",
            str(verifier_uid),
            "--regid",
            str(verifier_uid),
            "--clear-groups",
            *command,
        ]

    severed = not config.network("verifier")
    verifier_workspace = workspace
    if severed and verifier_workspace is None:
        verifier_workspace = Workspace(
            Path("/"),
            system_mounts=(Mount("proc", dst="/proc"), Mount("dev", dst="/dev")),
            network=False,
            allowed_hosts=(),
            guest_path="/",
            hand_over_root=False,
        )
    if severed and (verifier_workspace is None or not verifier_workspace.bwrap_available):
        raise RuntimeError(
            "the verifier declares no-network but bwrap cannot sandbox here; "
            "refusing to grade with network access the task ruled out"
        )

    spawn = verifier_command

    async def run_tests() -> ProcessGroup:
        environ = {**os.environ, **verifier_env, **config.verifier.env}
        if not severed:
            return await create_process_group_exec(
                *spawn,
                cwd=workdir,
                env=environ,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        assert verifier_workspace is not None
        info_read, info_write = os.pipe()
        block_read, block_write = os.pipe()
        try:
            os.set_inheritable(info_write, True)
            os.set_inheritable(block_read, True)
            group = await create_process_group_exec(
                *verifier_workspace.bwrap_argv(
                    spawn,
                    cwd=workdir.as_posix(),
                    env={**verifier_env, **config.verifier.env},
                    inherit_workspace_env=False,
                    info_fd=info_write,
                    userns_block_fd=block_read,
                    network=False,
                    mount_hosts=False,
                    isolate_processes=False,
                ),
                cwd=workdir,
                env=environ,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(info_write, block_read),
            )
            os.close(info_write)
            os.close(block_read)
            info_write = block_read = -1
            await install_identity_map(info_read, block_write)
            return group
        finally:
            os.close(info_read)
            os.close(block_write)
            for stray in (info_write, block_read):
                if stray != -1:
                    os.close(stray)

    if workspace is None or severed or not workspace.owns_netns:
        return await _grade_with_verifier(config, LOGS, answer, run_tests)
    sandbox = await workspace.sandbox_pid()
    if sandbox is None:
        return await _grade_with_verifier(config, LOGS, answer, run_tests)

    async with workspace.visiting(config.allowed_hosts("verifier")) as visitor_env:
        verifier_env.update(visitor_env)
        spawn = workspace.enter_argv(
            sandbox,
            command,
            env={**verifier_env, **config.verifier.env},
            identity=verifier_uid,
            inherit_workspace_env=False,
            preserve_credentials=True,
            no_new_privs=False,
        )
        return await _grade_with_verifier(config, LOGS, answer, run_tests)


async def _grade_with_verifier(
    config: TaskConfig,
    logs: Path,
    answer: Any,
    run_tests: Callable[[], Awaitable[ProcessGroup]],
) -> dict[str, Any]:
    """Run the verifier and shape its reward into a HUD grade."""
    timeout = config.verifier.timeout_sec or DEFAULT_VERIFIER_TIMEOUT
    _reset_dir(logs / "verifier")
    _write_no_follow(logs / "agent_answer.txt", "" if answer is None else str(answer))
    group = await run_tests()
    proc = group.process
    assert proc.stdout is not None and proc.stderr is not None
    reading = (
        asyncio.create_task(proc.stdout.read()),
        asyncio.create_task(proc.stderr.read()),
    )
    try:
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            timed_out = False
        except TimeoutError:
            timed_out = True
        await group.terminate()
        out_bytes, err_bytes = await asyncio.gather(*reading)
    except BaseException:
        for reader in reading:
            reader.cancel()
        await group.terminate()
        raise
    if timed_out:
        return {
            "score": 0.0,
            "isError": True,
            "content": f"Harbor verifier timed out after {timeout:.0f}s",
            "info": {
                "verifier_timeout_sec": timeout,
                "stdout": out_bytes.decode("utf-8", "replace")[-4000:],
                "stderr": err_bytes.decode("utf-8", "replace")[-4000:],
            },
        }

    reward, info = _read_reward(logs / "verifier")
    info.update(
        {
            "stdout": out_bytes.decode("utf-8", "replace")[-4000:],
            "stderr": err_bytes.decode("utf-8", "replace")[-4000:],
        }
    )
    if reward is None:
        return {
            "score": 0.0,
            "isError": True,
            "content": "Harbor verifier did not write reward.json or reward.txt",
            "info": info,
        }
    return {"score": reward, "info": info}


def _read_reward(verifier_logs: Path) -> tuple[float | None, dict[str, Any]]:
    """Read a finite reward from reward.json or reward.txt."""
    reward_json = verifier_logs / "reward.json"
    if reward_json.is_file():
        try:
            data = json.loads(reward_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, {"parse_error": "reward.json is not valid JSON"}
        if (score := _as_score(data)) is not None:
            return score, {"reward_file": str(reward_json)}
        if isinstance(data, dict):
            for key in ("reward", "score"):
                if (score := _as_score(data.get(key))) is not None:
                    return score, {"reward_file": str(reward_json), "reward_json": data}
        return None, {"reward_file": str(reward_json), "reward_parse_error": "no numeric reward"}

    reward_txt = verifier_logs / "reward.txt"
    if reward_txt.is_file():
        text = reward_txt.read_text(encoding="utf-8").strip()
        try:
            value = float(text)
        except ValueError:
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}
        if not math.isfinite(value):
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}
        return value, {"reward_file": str(reward_txt)}

    return None, {}


def _write_no_follow(path: Path, text: str) -> None:
    """Write text without following a symlink at path."""
    if path.is_symlink():
        path.unlink()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _as_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None

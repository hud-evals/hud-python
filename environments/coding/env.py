"""A repository workspace graded by hidden tests."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from hud.environment import Environment, Workspace
from hud.graders import EvaluationResult
from hud.settings import settings

from coding import repo as repo_lib
from coding.grading import parse_junit, score_tests

logger = logging.getLogger(__name__)

_explicit_repo = os.environ.get("REPO_DIR")
if _explicit_repo:
    REPO_DIR = Path(_explicit_repo)
    VAULT_DIR = Path(os.environ.get("VAULT_DIR", "/hud/vault"))
    LOGS_DIR = Path(os.environ.get("GRADING_LOGS_DIR", "/hud/logs"))
    _LOCAL = False
else:
    _LOCAL_ROOT = Path(tempfile.gettempdir()) / "hud-coding" / str(os.getpid())
    REPO_DIR = _LOCAL_ROOT / "workspace"
    VAULT_DIR = _LOCAL_ROOT / "vault"
    LOGS_DIR = _LOCAL_ROOT / "logs"
    _LOCAL = True

REPO_SOURCE = os.environ.get("REPO_URL") or str(Path(__file__).with_name("flask.bundle"))
TEST_TIMEOUT = float(os.environ.get("GRADING_TIMEOUT", "3600"))
AGENT_UID = 1000
AGENT_HOME = Path("/tmp/agent-home")  # noqa: S108 - container-local
AGENT_ENV = {"HOME": str(AGENT_HOME)}
VENV_ACTIVATE = Path(sys.executable).with_name("activate")
if VENV_ACTIVATE.is_file():
    AGENT_ENV["BASH_ENV"] = str(VENV_ACTIVATE)

env = Environment(name="coding")
_workspace: Workspace | None = None


@env.initialize
async def _initialize() -> None:
    global _workspace
    if _LOCAL and not (REPO_DIR / ".git").exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s", REPO_SOURCE)
        await repo_lib.run("git", "clone", "-q", REPO_SOURCE, str(REPO_DIR), cwd=REPO_DIR.parent)

    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    _workspace = Workspace(
        REPO_DIR,
        guest_path=str(REPO_DIR),
        network=False,
        env=AGENT_ENV,
        track_files=settings.file_tracking_enabled,
        shell_uid=AGENT_UID,
        require_isolation=not is_root,
    )
    await _workspace.start()
    env.add_capability(_workspace.capability("shell"))
    if _workspace.tracks_files:
        env.add_capability(_workspace.file_tracking_capability())


@env.shutdown
async def _shutdown() -> None:
    global _workspace
    if _workspace is not None:
        await _workspace.stop()
        _workspace = None
    if _LOCAL:
        shutil.rmtree(_LOCAL_ROOT, ignore_errors=True)


async def _setup(base_ref: str | None) -> None:
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if is_root:
        await repo_lib.run(
            "git",
            "config",
            "--global",
            "--add",
            "safe.directory",
            str(REPO_DIR),
            cwd=Path("/"),
        )
    if base_ref:
        await repo_lib.git(REPO_DIR, "checkout", "-qf", base_ref)
    await repo_lib.vault_history(REPO_DIR, VAULT_DIR)
    if is_root:
        AGENT_HOME.mkdir(parents=True, exist_ok=True)
        await repo_lib.run(
            "chown",
            "-R",
            f"{AGENT_UID}:{AGENT_UID}",
            str(REPO_DIR),
            str(AGENT_HOME),
            cwd=Path("/"),
        )


async def _run_tests(
    test_script: str,
    test_files: list[str],
    f2p_test_nodeids: list[str] | None,
    p2p_test_nodeids: list[str] | None,
    use_binary_score: bool,
) -> EvaluationResult:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    junit_path = LOGS_DIR / "junit.xml"
    junit_path.unlink(missing_ok=True)
    command = test_script.replace("{test_files}", shlex.join(test_files)).replace(
        "{junit_path}", shlex.quote(str(junit_path))
    )
    exit_code, stdout_bytes, stderr_bytes = await repo_lib.run(
        "bash",
        "-c",
        command,
        cwd=REPO_DIR,
        timeout=TEST_TIMEOUT,
        check=False,
    )
    stdout = stdout_bytes.decode("utf-8", "replace")
    stderr = stderr_bytes.decode("utf-8", "replace")
    (LOGS_DIR / "stdout.log").write_text(stdout, encoding="utf-8")
    (LOGS_DIR / "stderr.log").write_text(stderr, encoding="utf-8")
    info = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
    if not junit_path.is_file():
        return EvaluationResult(
            reward=0.0,
            content=f"test script did not write JUnit XML to {junit_path}",
            info=info,
            isError=True,
        )

    try:
        result = score_tests(
            parse_junit(junit_path),
            f2p_test_nodeids,
            p2p_test_nodeids,
            use_binary_score,
        )
    except (ValueError, OSError) as exc:
        return EvaluationResult(
            reward=0.0,
            content=f"invalid test results: {exc}",
            info=info,
            isError=True,
        )
    result.info.update(info)
    return result


async def _grade(
    test_script: str,
    test_patch: str,
    test_files: list[str],
    f2p_test_nodeids: list[str] | None,
    p2p_test_nodeids: list[str] | None,
    use_binary_score: bool,
) -> EvaluationResult:
    setup_commit = await repo_lib.restore_history(REPO_DIR, VAULT_DIR)
    agent_diff = await repo_lib.capture_agent_diff(REPO_DIR, setup_commit)

    await repo_lib.reset_worktree(REPO_DIR, setup_commit)
    apply_error = await repo_lib.apply_diff(REPO_DIR, agent_diff, LOGS_DIR / "agent.patch")
    if apply_error is not None:
        return EvaluationResult(
            reward=0.0,
            content="changes failed to apply",
            info={"git_apply": apply_error},
        )
    await repo_lib.restore_paths(REPO_DIR, setup_commit, test_files)
    test_apply_error = await repo_lib.apply_diff(REPO_DIR, test_patch, LOGS_DIR / "tests.patch")
    if test_apply_error is not None:
        return EvaluationResult(
            reward=0.0,
            content="hidden tests failed to apply",
            info={"git_apply": test_apply_error},
            isError=True,
        )

    return await _run_tests(
        test_script,
        test_files,
        f2p_test_nodeids,
        p2p_test_nodeids,
        use_binary_score,
    )


@env.template(id="coding-task", description="Modify a repository and grade it with hidden tests.")
async def coding_task(
    description: str,
    test_script: str,
    test_patch: str,
    test_files: list[str],
    base_ref: str | None = None,
    f2p_test_nodeids: list[str] | None = None,
    p2p_test_nodeids: list[str] | None = None,
    use_binary_score: bool = False,
):
    if "{junit_path}" not in test_script:
        raise ValueError("test_script must write JUnit XML using the {junit_path} placeholder")
    if not test_files:
        raise ValueError("test_files must list every path changed by test_patch")

    await _setup(base_ref)
    _ = yield description.strip()
    yield await _grade(
        test_script,
        test_patch,
        test_files,
        f2p_test_nodeids,
        p2p_test_nodeids,
        use_binary_score,
    )

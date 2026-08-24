"""A repository workspace graded by hidden tests."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from hud.environment import Environment, Workspace
from hud.graders import EvaluationResult, combine
from hud.settings import settings
from pydantic import Field

from grader import JUnitGrader

logger = logging.getLogger(__name__)

if repo_dir := os.environ.get("REPO_DIR"):
    REPO_DIR = Path(repo_dir)
    BASELINE_DIR = Path(os.environ.get("BASELINE_DIR", "/hud/baseline"))
    _LOCAL = False
else:
    _LOCAL_ROOT = Path(tempfile.gettempdir()) / "hud-coding" / str(os.getpid())
    REPO_DIR = _LOCAL_ROOT / "workspace"
    BASELINE_DIR = _LOCAL_ROOT / "baseline"
    _LOCAL = True

REPO_SOURCE = os.environ.get("REPO_URL") or str(Path(__file__).with_name("flask.bundle"))
TEST_TIMEOUT = float(os.environ.get("GRADING_TIMEOUT", "3600"))
AGENT_UID = 1000
AGENT_HOME = Path("/tmp/agent-home")  # noqa: S108 - container-local
AGENT_ENV = {"HOME": str(AGENT_HOME)}
VENV_ACTIVATE = Path(sys.executable).with_name("activate")
if VENV_ACTIVATE.is_file():
    os.environ["BASH_ENV"] = AGENT_ENV["BASH_ENV"] = str(VENV_ACTIVATE)

env = Environment(name="coding")
workspace: Workspace | None = None
_GIT = (
    "git",
    "-c",
    f"safe.directory={REPO_DIR}",
    "-c",
    "user.name=hud",
    "-c",
    "user.email=hud@localhost",
)


@env.initialize
async def _initialize() -> None:
    global workspace
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(
        REPO_DIR,
        guest_path=str(REPO_DIR),
        network=False,
        env=AGENT_ENV,
        track_files=settings.file_tracking_enabled,
        shell_uid=AGENT_UID,
        require_isolation=True,
    )
    await workspace.start()
    env.add_capability(workspace.capability("shell"))
    if workspace.tracks_files:
        env.add_capability(workspace.file_tracking_capability())


@env.shutdown
async def _shutdown() -> None:
    global workspace
    if workspace is not None:
        await workspace.stop()
        workspace = None
    if _LOCAL:
        shutil.rmtree(_LOCAL_ROOT, ignore_errors=True)


async def _setup(base_ref: str) -> None:
    shutil.rmtree(BASELINE_DIR, ignore_errors=True)
    logger.info("preparing %s from %s", base_ref, REPO_SOURCE)
    subprocess.run(
        ["git", "clone", "-q", REPO_SOURCE, str(BASELINE_DIR)],
        check=True,
        capture_output=True,
    )
    subprocess.run([*_GIT, "checkout", "-qf", base_ref], cwd=BASELINE_DIR, check=True)
    subprocess.run([*_GIT, "clean", "-fdx"], cwd=BASELINE_DIR, check=True)

    for path in REPO_DIR.iterdir():
        if path.name == ".hud":
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    shutil.copytree(
        BASELINE_DIR,
        REPO_DIR,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    subprocess.run([*_GIT, "init", "-q", "."], cwd=REPO_DIR, check=True)
    (REPO_DIR / ".git" / "info" / "exclude").write_text(".hud/\n", encoding="utf-8")
    subprocess.run([*_GIT, "add", "-A"], cwd=REPO_DIR, check=True)
    subprocess.run([*_GIT, "commit", "-qm", "baseline"], cwd=REPO_DIR, check=True)

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        AGENT_HOME.mkdir(parents=True, exist_ok=True)
        agent_paths = [path for path in REPO_DIR.iterdir() if path.name != ".hud"]
        subprocess.run(
            ["chown", "-R", f"{AGENT_UID}:{AGENT_UID}", *agent_paths, AGENT_HOME],
            check=True,
        )


async def _grade(
    test_command: str,
    test_patch: str,
    test_path: str,
    fail_to_pass: list[str] | None,
    pass_to_pass: list[str] | None,
    binary: bool,
) -> EvaluationResult:
    assert workspace is not None
    await workspace.terminate_sessions()
    agent_git = REPO_DIR / ".git"
    if agent_git.is_symlink() or agent_git.is_file():
        agent_git.unlink()
    elif agent_git.is_dir():
        shutil.rmtree(agent_git)

    source = BASELINE_DIR / test_path
    target = REPO_DIR / test_path
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
    elif source.is_file():
        shutil.copy2(source, target)
    elif source.is_dir():
        shutil.copytree(source, target, symlinks=True)

    subprocess.run([*_GIT, "init", "-q", "."], cwd=REPO_DIR, check=True)
    applied = subprocess.run(
        [*_GIT, "apply", "-"],
        cwd=REPO_DIR,
        input=test_patch,
        text=True,
        capture_output=True,
        check=False,
    )
    if applied.returncode != 0:
        return EvaluationResult(
            content="hidden tests failed to apply",
            info={"git_apply": applied.stderr[-4000:]},
            isError=True,
        )

    grader_command = shlex.join(
        workspace.shell_argv(
            test_command,
            cwd=str(REPO_DIR),
            env={
                "HOME": "/tmp",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
    )
    test_result = await JUnitGrader.grade(
        weight=1.0,
        name="tests",
        command=grader_command,
        cwd=str(REPO_DIR),
        timeout_seconds=TEST_TIMEOUT,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        binary=binary,
    )
    if error := (test_result.info or {}).get("error"):
        return EvaluationResult(content=str(error), info=test_result.info or {}, isError=True)
    return await combine(test_result)


@env.template(id="coding-task", description="Modify a repository and grade it with hidden tests.")
async def coding_task(
    description: Annotated[str, Field(json_schema_extra={"x-hud-hint": "prompt"})],
    test_command: str,
    test_patch: str,
    test_path: str,
    base_ref: str,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    binary: bool = False,
):
    path = Path(test_path)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError("test_path must be a top-level file or directory")

    await _setup(base_ref)
    try:
        yield description.strip()
        result = await _grade(
            test_command,
            test_patch,
            test_path,
            fail_to_pass,
            pass_to_pass,
            binary,
        )
    finally:
        assert workspace is not None
        await workspace.discard_sandbox()
    yield result

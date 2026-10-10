"""Run the environment the way HUD does, against a small fixture repository.

Tests marked ``sandbox`` serve the real workspace, which refuses to run without
isolation. They need Linux, root, bubblewrap, and this project's Python
environment under /usr, the only prefix the sandbox shows the grader (the image
installs it at /usr/local/venv). Run them in the coding container.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hud.environment import Workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SANDBOXED = (
    sys.platform == "linux"
    and os.geteuid() == 0
    and shutil.which("bwrap") is not None
    and Path(sys.prefix).is_relative_to("/usr")
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if SANDBOXED:
        return
    skip = pytest.mark.skip(reason="needs Linux, root, bubblewrap, and the environment under /usr")
    for item in items:
        if "sandbox" in item.keywords:
            item.add_marker(skip)


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={cwd}", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repository whose ``bug_baseline`` branch is broken and ``bug_golden`` fixes it."""
    repo = tmp_path_factory.mktemp("fixture-repo")
    (repo / "widget.py").write_text("BROKEN = True\n")
    for args in (
        ("init", "-q", "-b", "main"),
        ("add", "-A"),
        ("commit", "-qm", "buggy widget"),
        ("branch", "bug_baseline"),
        ("checkout", "-qb", "bug_golden"),
    ):
        git(repo, *args).check_returncode()
    (repo / "widget.py").write_text("BROKEN = False\n")
    git(repo, "commit", "-qam", "reference fix").check_returncode()
    return repo


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture_repo: Path) -> Path:
    """Configure the environment to clone ``fixture_repo``; returns the agent's repository."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "0")
    monkeypatch.setenv("REPO_URL", str(fixture_repo))
    monkeypatch.setenv("REPO_DIR", str(tmp_path / "repo"))
    monkeypatch.setenv("BASELINE_DIR", str(tmp_path / "baseline"))
    return tmp_path / "repo"


@pytest.fixture
async def workspace(tmp_path: Path):
    """An isolated workspace like the environment's, for grading commands directly."""
    root = tmp_path / "workspace"
    served = Workspace(root, guest_path=str(root), network=False, shell_uid=1000, require_isolation=True)
    await served.start()
    yield served
    await served.stop()

"""Hermetic end-to-end validation of the generic flavor (no Docker, no network).

Builds a tiny 3-branch fixture repo (``bug_baseline`` / ``bug_test`` /
``bug_golden``), serves ``env.py`` on a local substrate cloned from it, and
drives the full lifecycle with no agent edits:

- ``validate_mode="golden"``: the reference fix grades to 1.0
- no validate_mode: the untouched baseline grades to 0.0

This is the 3-branch analog of the SWE-bench gold-patch sanity check, and it
exercises the whole vault/diff/hidden-test pipeline through real git.
"""

import os
import subprocess
from pathlib import Path

import pytest
from hud import LocalRuntime, Run, connect

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("fixture-repo")
    (repo / "widget.py").write_text("BROKEN = True\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "buggy widget")
    _git(repo, "branch", "bug_baseline")

    _git(repo, "checkout", "-qb", "bug_test")
    (repo / "test_widget.py").write_text(
        "import unittest\n"
        "import widget\n\n\n"
        "class TestWidget(unittest.TestCase):\n"
        "    def test_not_broken(self):\n"
        "        self.assertFalse(widget.BROKEN)\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "hidden tests")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "checkout", "-qb", "bug_golden")
    (repo / "widget.py").write_text("BROKEN = False\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "reference fix")
    return repo


async def _run_task(fixture_repo: Path, task) -> float:
    # The served subprocess inherits the environment; point it at the fixture.
    os.environ["REPO_URL"] = str(fixture_repo)
    runtime = LocalRuntime(str(PROJECT_ROOT / "env.py"))
    async with runtime(task) as addr, connect(addr) as client:
        async with Run(client, task.id, task.args) as run:
            pass  # no agent work: setup on start, grading on exit
    return run.reward


def _coding_task(validate_mode: str | None):
    from env import coding_task

    return coding_task(
        description="Fix the widget.",
        test_command="python3 -m unittest {test_files}",
        base_ref="origin/bug_baseline",
        test_ref="origin/bug_test",
        test_files=["test_widget.py"],
        golden_ref="origin/bug_golden",
        validate_mode=validate_mode,
    )


def _sdlc_task(validate_mode: str | None):
    from env import sdlc_task

    return sdlc_task(
        description="Issue #1 reports a broken widget. Fix it and open a PR.",
        test_command="python3 -m unittest {test_files}",
        base_ref="origin/bug_baseline",
        test_ref="origin/bug_test",
        test_files=["test_widget.py"],
        golden_ref="origin/bug_golden",
        issues=[{"number": 1, "title": "Widget broken", "body": "BROKEN should be False."}],
        validate_mode=validate_mode,
    )


async def test_golden_ref_scores_one(fixture_repo):
    assert await _run_task(fixture_repo, _coding_task("golden")) == 1.0


async def test_untouched_baseline_scores_zero(fixture_repo):
    assert await _run_task(fixture_repo, _coding_task(None)) == 0.0


async def test_sdlc_golden_pr_scores_one(fixture_repo):
    """Golden validation of the full PR workflow: pushed branch + opened PR."""
    assert await _run_task(fixture_repo, _sdlc_task("golden")) == 1.0


async def test_sdlc_no_pull_request_scores_zero(fixture_repo):
    assert await _run_task(fixture_repo, _sdlc_task(None)) == 0.0

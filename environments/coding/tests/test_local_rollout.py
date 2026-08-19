"""Hermetic end-to-end validation of repository setup and grading."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hud import LocalRuntime, Run, connect
from hud.environment import workspace as workspace_lib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTEST_SCRIPT = f"{shlex.quote(sys.executable)} -m pytest -q {{test_files}} --junitxml={{junit_path}}"
TEST_PATCH = """diff --git a/test_widget.py b/test_widget.py
new file mode 100644
--- /dev/null
+++ b/test_widget.py
@@ -0,0 +1,10 @@
+import unittest
+import widget
+
+
+class TestWidget(unittest.TestCase):
+    def test_not_broken(self):
+        self.assertFalse(widget.BROKEN)
+
+    def test_existing_behavior(self):
+        self.assertTrue(hasattr(widget, 'BROKEN'))
"""

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def isolated_workspace(monkeypatch):
    monkeypatch.setattr(workspace_lib, "usable_bwrap", lambda: "/usr/bin/true")


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("fixture-repo")
    (repo / "widget.py").write_text("BROKEN = True\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "buggy widget")
    _git(repo, "branch", "bug_baseline")

    _git(repo, "checkout", "-qb", "bug_golden")
    (repo / "widget.py").write_text("BROKEN = False\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "reference fix")
    return repo


def _coding_task():
    from env import coding_task

    return coding_task(
        description="Fix the widget.",
        test_script=f'test -n "${{HOME}}" && {PYTEST_SCRIPT}',
        test_patch=TEST_PATCH,
        test_files=["test_widget.py"],
        base_ref="origin/bug_baseline",
        f2p_test_nodeids=["test_widget.TestWidget.test_not_broken"],
        p2p_test_nodeids=["test_widget.TestWidget.test_existing_behavior"],
    )


async def _run_task(fixture_repo: Path, task) -> float:
    os.environ["REPO_URL"] = str(fixture_repo)
    runtime = LocalRuntime(str(PROJECT_ROOT / "env.py"))
    async with runtime(task) as addr, connect(addr) as client:
        async with Run(client, task.id, task.args) as run:
            pass
    return run.reward


async def test_agent_fix_scores_one(fixture_repo, tmp_path, monkeypatch):
    import env as coding_env

    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(coding_env, "LOGS_DIR", tmp_path / "logs")

    task = coding_env.coding_task.func(**_coding_task().args)
    await task.asend(None)
    (repo / "widget.py").write_text("BROKEN = False\n")
    result = await task.asend("done")

    assert result.reward == 1.0


async def test_untouched_baseline_gets_only_regression_credit(fixture_repo, isolated_workspace):
    assert await _run_task(fixture_repo, _coding_task()) == 0.5


async def test_local_runtime_refuses_unisolated_non_root_workspace(
    fixture_repo,
    monkeypatch,
):
    monkeypatch.setattr(workspace_lib, "usable_bwrap", lambda: None)
    monkeypatch.setattr(os, "geteuid", lambda: 501)
    os.environ["REPO_URL"] = str(fixture_repo)

    runtime = LocalRuntime(str(PROJECT_ROOT / "env.py"))
    with pytest.raises(RuntimeError, match="isolation was required"):
        async with runtime(_coding_task()):
            pass


async def test_grading_discards_agent_git_config_and_test_changes(
    fixture_repo,
    tmp_path,
    monkeypatch,
):
    import env as coding_env

    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(coding_env, "LOGS_DIR", tmp_path / "logs")

    task = coding_env.coding_task.func(
        description="Fix the widget.",
        test_script=PYTEST_SCRIPT,
        test_patch=TEST_PATCH,
        test_files=["test_widget.py"],
        base_ref="origin/bug_baseline",
        f2p_test_nodeids=["test_widget.TestWidget.test_not_broken"],
        p2p_test_nodeids=["test_widget.TestWidget.test_existing_behavior"],
    )
    await task.asend(None)

    marker = tmp_path / "agent-filter-ran"
    _git(repo, "config", "filter.agent.clean", f"touch {shlex.quote(str(marker))}; cat")
    (repo / ".gitattributes").write_text("* filter=agent\n")
    (repo / "test_widget.py").write_text("def test_not_broken():\n    assert True\n")

    result = await task.asend("done")

    assert result.reward == 0.5
    assert not marker.exists()


async def test_grading_preserves_prepared_dependencies(tmp_path, monkeypatch):
    import env as coding_env

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".prepared/\n")
    (repo / "widget.py").write_text("BROKEN = True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "buggy widget")
    _git(repo, "branch", "bug_baseline")

    _git(repo, "checkout", "-qb", "bug_golden")
    (repo / "widget.py").write_text("BROKEN = False\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "reference fix")
    _git(repo, "checkout", "-q", "bug_baseline")

    prepared = repo / ".prepared" / "dependency"
    prepared.parent.mkdir()
    prepared.write_text("ready\n")

    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(coding_env, "LOGS_DIR", tmp_path / "logs")

    task = coding_env.coding_task.func(
        description="Fix the widget.",
        test_script=f"test -f .prepared/dependency && {PYTEST_SCRIPT}",
        test_patch=TEST_PATCH,
        test_files=["test_widget.py"],
        base_ref="bug_baseline",
        f2p_test_nodeids=["test_widget.TestWidget.test_not_broken"],
    )
    await task.asend(None)
    (repo / "widget.py").write_text("BROKEN = False\n")
    result = await task.asend("done")

    assert result.reward == 1.0
    assert prepared.read_text() == "ready\n"

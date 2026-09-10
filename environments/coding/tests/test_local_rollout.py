"""Hermetic end-to-end validation of repository setup and grading."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hud import LocalRuntime, Run, connect
from hud.environment import workspace as workspace_lib

import tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTEST_COMMAND = (
    f"{shlex.quote(sys.executable)} -m pytest -q -c /dev/null --rootdir=. test_widget.py --junitxml={{junit_path}}"
)
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
        ["git", "-c", f"safe.directory={cwd}", "-c", "user.name=t", "-c", "user.email=t@t", *args],
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

    _git(repo, "checkout", "-qb", "bug_golden")
    (repo / "widget.py").write_text("BROKEN = False\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "reference fix")
    return repo


def _coding_task():
    from env import coding_task

    return coding_task(
        description="Fix the widget.",
        test_command=f'test -n "${{HOME}}" && {PYTEST_COMMAND}',
        test_patch=TEST_PATCH,
        test_path="test_widget.py",
        base_ref="origin/bug_baseline",
        fail_to_pass=["test_widget.TestWidget.test_not_broken"],
        pass_to_pass=["test_widget.TestWidget.test_existing_behavior"],
    )


async def _run_task(fixture_repo: Path, task, monkeypatch) -> float:
    monkeypatch.setenv("REPO_URL", str(fixture_repo))
    monkeypatch.delenv("REPO_DIR", raising=False)
    monkeypatch.delenv("BASELINE_DIR", raising=False)
    runtime = LocalRuntime(str(PROJECT_ROOT / "env.py"))
    async with runtime(task) as addr, connect(addr) as client:
        async with Run(client, task.id, task.args) as run:
            pass
    assert not run.evaluation.get("isError"), run.evaluation
    return run.reward


async def test_agent_fix_scores_one(fixture_repo, tmp_path, monkeypatch, grading_workspace):
    import env as coding_env

    repo = grading_workspace.root
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "BASELINE_DIR", tmp_path / "baseline")
    monkeypatch.setattr(coding_env, "REPO_SOURCE", str(fixture_repo))

    task = coding_env.coding_task.func(**_coding_task().args)
    await task.asend(None)
    (repo / "fix.py").write_text("BROKEN = False\n")
    (repo / "widget.py").write_text("from fix import BROKEN\n")
    result = await task.asend("done")

    assert result.reward == 1.0, result
    assert (repo / "fix.py").exists()


async def test_missing_junit_report_is_grading_error(
    fixture_repo,
    tmp_path,
    monkeypatch,
    grading_workspace,
):
    import env as coding_env

    repo = grading_workspace.root
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "BASELINE_DIR", tmp_path / "baseline")
    monkeypatch.setattr(coding_env, "REPO_SOURCE", str(fixture_repo))

    task = coding_env.coding_task.func(
        description="Fix the widget.",
        test_command="true {junit_path}",
        test_patch=TEST_PATCH,
        test_path="test_widget.py",
        base_ref="origin/bug_baseline",
    )
    await task.asend(None)
    result = await task.asend("done")

    assert result.reward == 0.0
    assert result.isError is True
    assert result.content == "test command did not write JUnit XML"
    assert result.info["exit_code"] == 0


async def test_untouched_baseline_gets_regression_credit(fixture_repo, isolated_workspace, monkeypatch):
    assert await _run_task(fixture_repo, _coding_task(), monkeypatch) == 0.5


@pytest.mark.parametrize("task", tasks.tasks, ids=lambda task: task.slug)
@pytest.mark.parametrize("fixed", [False, True], ids=["baseline", "reference-fix"])
async def test_bundled_tasks_grade_in_workspace(task, fixed, isolated_workspace, tmp_path, monkeypatch):
    bundle = tmp_path / "flask.bundle"
    bundle.write_bytes((PROJECT_ROOT / "flask.bundle").read_bytes())
    if fixed:
        task = task.model_copy(
            update={"args": {**task.args, "base_ref": f"origin/{task.slug.replace('-', '_')}_golden"}}
        )
    assert await _run_task(bundle, task, monkeypatch) == float(fixed)


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
    grading_workspace,
):
    import env as coding_env

    repo = grading_workspace.root
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "BASELINE_DIR", tmp_path / "baseline")
    monkeypatch.setattr(coding_env, "REPO_SOURCE", str(fixture_repo))

    task = coding_env.coding_task.func(
        description="Fix the widget.",
        test_command=PYTEST_COMMAND,
        test_patch=TEST_PATCH,
        test_path="test_widget.py",
        base_ref="origin/bug_baseline",
        fail_to_pass=["test_widget.TestWidget.test_not_broken"],
        pass_to_pass=["test_widget.TestWidget.test_existing_behavior"],
    )
    await task.asend(None)

    _git(repo, "config", "filter.agent.clean", "cat")
    (repo / ".gitattributes").write_text("* filter=agent\n")
    (repo / "test_widget.py").write_text("def test_not_broken():\n    assert True\n")

    result = await task.asend("done")

    assert result.reward == 0.5, result
    config = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "config", "--get", "filter.agent.clean"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    assert config.returncode == 1


async def test_setup_removes_files_from_the_previous_hidden_patch(
    fixture_repo,
    tmp_path,
    monkeypatch,
):
    import env as coding_env

    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(fixture_repo), str(repo)], check=True)
    monkeypatch.setattr(coding_env, "REPO_DIR", repo)
    monkeypatch.setattr(coding_env, "BASELINE_DIR", tmp_path / "baseline")
    monkeypatch.setattr(coding_env, "REPO_SOURCE", str(fixture_repo))

    await coding_env._setup("origin/bug_baseline")
    subprocess.run(
        ["git", "apply", "-"],
        cwd=repo,
        input=TEST_PATCH,
        text=True,
        check=True,
    )
    assert (repo / "test_widget.py").exists()

    await coding_env._setup("origin/bug_baseline")

    assert not (repo / "test_widget.py").exists()

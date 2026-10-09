"""The coding environment, served the way HUD serves it and driven over its control channel.

Each test starts a fresh environment process, starts a task, edits the agent's
repository the way an agent would, and grades: what a rollout does, without an
agent. The workspace refuses to run unisolated, so every test here is ``sandbox``.
"""

import shlex
import subprocess
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from hud import SubprocessRuntime, Task, connect
from hud.clients import HudProtocolError

import tasks
from env import coding_task

pytestmark = pytest.mark.sandbox

ENV_SOURCE = Path(__file__).resolve().parent.parent / "env.py"
PYTEST = f"{shlex.quote(sys.executable)} -m pytest -q -c /dev/null --rootdir=. test_widget.py --junitxml={{junit_path}}"
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


def widget_task(**overrides: object) -> Task:
    return coding_task(
        **{
            "description": "\nFix the widget.\n",
            "test_command": PYTEST,
            "test_patch": TEST_PATCH,
            "test_path": "test_widget.py",
            "base_ref": "origin/bug_baseline",
            "fail_to_pass": ["test_widget.TestWidget.test_not_broken"],
            "pass_to_pass": ["test_widget.TestWidget.test_existing_behavior"],
            **overrides,
        }
    )


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", *args], cwd=repo, check=False, capture_output=True, text=True
    )


@asynccontextmanager
async def served(row: Task):
    """A fresh environment process serving ``row``, and a client connected to it."""
    async with SubprocessRuntime(ENV_SOURCE)(row) as runtime, connect(runtime) as client:
        yield client


def leave_untouched(repo: Path) -> None:
    del repo


def fix_the_widget(repo: Path) -> None:
    (repo / "fix.py").write_text("BROKEN = False\n")
    (repo / "widget.py").write_text("from fix import BROKEN\n")


def tamper_with_the_tests(repo: Path) -> None:
    """Rewrite the hidden test's path and plant a git filter that would rewrite it on apply."""
    git(repo, "config", "filter.agent.clean", "cat").check_returncode()
    (repo / ".gitattributes").write_text("* filter=agent\n")
    (repo / "test_widget.py").write_text("def test_not_broken():\n    assert True\n")


async def test_the_description_is_the_prompt(repo):
    row = widget_task()
    async with served(row) as client:
        (template,) = await client.list_tasks()
        started = await client.start_task(row.id, row.args)

    assert template["id"] == "coding-task"
    assert template["args"]["properties"]["description"]["x-hud-hint"] == "prompt"
    assert started["prompt"] == "Fix the widget."
    assert (repo / "widget.py").read_text() == "BROKEN = True\n"
    assert not (repo / "test_widget.py").exists()


@pytest.mark.parametrize(
    ("overrides", "edit", "score"),
    [
        pytest.param({}, leave_untouched, 0.5, id="untouched-baseline-keeps-regression-credit"),
        pytest.param({}, fix_the_widget, 1.0, id="agent-fix"),
        pytest.param({"base_ref": "origin/bug_golden"}, leave_untouched, 1.0, id="golden-ref"),
        pytest.param({}, tamper_with_the_tests, 0.5, id="agent-test-changes-are-discarded"),
    ],
)
async def test_hidden_tests_grade_the_agents_repository(
    overrides: dict, edit: Callable[[Path], None], score: float, repo: Path
):
    row = widget_task(**overrides)
    async with served(row) as client:
        await client.start_task(row.id, row.args)
        edit(repo)
        grade = await client.grade({"answer": "done"})

    assert not grade["isError"], grade
    assert grade["score"] == score
    assert git(repo, "config", "--get", "filter.agent.clean").returncode == 1


async def test_a_command_without_a_junit_report_is_a_grading_error(repo):
    row = widget_task(test_command="true {junit_path}")
    async with served(row) as client:
        await client.start_task(row.id, row.args)
        grade = await client.grade({"answer": "done"})

    assert grade["isError"] is True
    assert grade["content"] == "test command did not write JUnit XML"
    assert grade["score"] == 0.0


@pytest.mark.parametrize("test_path", ["tests/test_widget.py", "/test_widget.py", ".."])
async def test_hidden_tests_must_sit_at_the_top_level(test_path: str, repo: Path):
    row = widget_task(test_path=test_path)
    async with served(row) as client:
        with pytest.raises(HudProtocolError) as failure:
            await client.start_task(row.id, row.args)

    assert "test_path must be a top-level file or directory" in failure.value.message
    assert not repo.exists() or list(repo.iterdir()) == []


async def test_a_new_task_starts_from_a_clean_baseline(repo):
    row = widget_task()
    async with served(row) as client:
        await client.start_task(row.id, row.args)
        fix_the_widget(repo)
        await client.grade({"answer": "done"})
        await client.start_task(row.id, row.args)

    assert not (repo / "test_widget.py").exists()
    assert not (repo / "fix.py").exists()
    assert (repo / "widget.py").read_text() == "BROKEN = True\n"


@pytest.mark.parametrize("bundled", tasks.tasks, ids=lambda task: task.slug)
@pytest.mark.parametrize(("ref", "score"), [("baseline", 0.0), ("golden", 1.0)])
async def test_bundled_tasks_score_zero_at_baseline_and_one_with_the_reference_fix(
    bundled: Task, ref: str, score: float, repo: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("REPO_URL")
    base_ref = bundled.args["base_ref"].replace("_baseline", f"_{ref}")
    row = bundled.model_copy(update={"args": {**bundled.args, "base_ref": base_ref}})
    async with served(row) as client:
        await client.start_task(row.id, row.args)
        grade = await client.grade({"answer": "done"})

    assert not grade["isError"], grade
    assert grade["score"] == score

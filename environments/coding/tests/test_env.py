"""In-process wiring smoke tests for the served environment."""

from unittest.mock import AsyncMock

import pytest

import env as coding_env
from env import env


def test_coding_template_registered():
    assert "coding-task" in env.tasks
    assert env.tasks["coding-task"].manifest_entry()["id"] == "coding-task"


@pytest.mark.asyncio
async def test_coding_task_uses_description_as_prompt(monkeypatch):
    monkeypatch.setattr(coding_env, "_setup", AsyncMock())
    workspace = AsyncMock()
    monkeypatch.setattr(coding_env, "workspace", workspace)
    task = coding_env.coding_task.func(
        description="\nFix the bug.\n",
        test_command="pytest -q --junitxml={junit_path}",
        test_patch="diff --git a/test_bug.py b/test_bug.py\n",
        test_path="tests",
        base_ref="main",
    )

    assert await anext(task) == "Fix the bug."
    await task.aclose()
    workspace.discard_sandbox.assert_awaited_once_with()

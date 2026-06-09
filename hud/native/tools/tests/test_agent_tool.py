"""The v6 ``AgentTool``: schema derivation + sub-agent execution over a Task."""

from __future__ import annotations

from typing import Any, cast

import pytest

from hud.environment import Environment
from hud.native.tools.agent import AgentTool


class _FakeAgent:
    """Stand-in agent that fills ``run.trace`` like a real agent would."""

    def __init__(self, **_: Any) -> None: ...

    async def __call__(self, run: Any) -> None:
        run.trace.content = f"answer for {run.prompt}"


def _env_with_task() -> Environment:
    env = Environment("agent-tool-test")

    @env.task()
    async def investigate(issue_id: str, expected_cause: str | None = None):
        yield f"Investigate {issue_id}"
        yield 1.0

    return env


def test_requires_model_or_agent() -> None:
    env = _env_with_task()
    task = env._tasks["investigate"]

    with pytest.raises(ValueError, match="provide either"):
        AgentTool(task)


def test_schema_hides_eval_only_params() -> None:
    env = _env_with_task()
    task = env._tasks["investigate"]

    tool = AgentTool(task, agent=_FakeAgent, name="inv")

    props = tool._param_schema["properties"]
    assert "issue_id" in props  # required, visible
    assert "expected_cause" not in props  # eval-only (None default + None type) is hidden
    assert tool.name == "inv"


async def test_call_runs_subagent_over_task() -> None:
    env = _env_with_task()
    task = env._tasks["investigate"]
    tool = AgentTool(task, agent=_FakeAgent)

    result = await tool(issue_id="BUG-1")

    assert cast("Any", result.content[0]).text == "answer for Investigate BUG-1"

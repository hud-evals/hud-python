"""The CUA environment, served the way HUD serves it and driven over its control channel.

Each test starts a fresh environment process, starts a task with the arguments a
task row carries, and grades an answer: what a rollout does, without an agent.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from hud import SubprocessRuntime, Task, connect
from hud.clients import HudProtocolError

from env import cua_task

ENV_SOURCE = Path(__file__).resolve().parent.parent / "env.py"
PASS = {"name": "ok", "command": "true", "weight": 1.0}


@asynccontextmanager
async def served(row: Task):
    """A fresh environment process serving ``row``, and a client connected to it."""
    async with SubprocessRuntime(ENV_SOURCE)(row) as runtime, connect(runtime) as client:
        yield client


async def test_the_manifest_publishes_the_screen_and_a_prompt_box(desktop):
    row = cua_task(prompt="p", bash_checks=[PASS])
    async with served(row) as client:
        (template,) = await client.list_tasks()
        screens = [binding.url for binding in client.manifest.bindings if binding.name == "screen"]

    assert screens == [f"rfb://127.0.0.1:{desktop}"]
    arguments = template["args"]
    assert arguments["properties"]["prompt"]["x-hud-hint"] == "prompt"
    assert "hud_api_key" in arguments["properties"]
    assert arguments["required"] == ["prompt"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param({}, "at least one grader", id="no-graders"),
        pytest.param(
            {"bash_checks": [{"name": "no", "command": "false", "weight": -1.0}]},
            "nonnegative",
            id="negative-weight",
        ),
        pytest.param({"grading_criteria": ["Anything."]}, "HUD_API_KEY is required", id="judge-without-key"),
        pytest.param(
            {"grading_criteria": ["Anything."], "hud_api_key": "task-key"},
            "HUD_API_KEY is required",
            id="a-task-key-is-not-the-runtime-key",
        ),
    ],
)
async def test_a_misconfigured_task_fails_before_the_agent_sees_a_prompt(arguments, message):
    row = cua_task(prompt="p", **arguments)
    async with served(row) as client:
        with pytest.raises(HudProtocolError) as failure:
            await client.start_task(row.id, row.args)

    assert message in failure.value.message


@pytest.mark.parametrize(
    ("arguments", "score", "weights"),
    [
        pytest.param({"bash_checks": [PASS]}, 1.0, {"ok": 1.0}, id="passing-check"),
        pytest.param({"bash_checks": [{"name": "no", "command": "false"}]}, 0.0, {"no": 1.0}, id="failing-check"),
        pytest.param(
            {
                "bash_checks": [
                    {"name": "alpha", "command": "true", "weight": 0.4},
                    {"name": "beta", "command": "false", "weight": 0.6},
                ]
            },
            0.4,
            {"alpha": 0.4, "beta": 0.6},
            id="weighted-checks",
        ),
        pytest.param(
            {
                "bash_checks": [{"name": "ok", "command": "true", "weight": 0.3}],
                "grading_criteria": ["Says hello."],
            },
            1.0,
            {"ok": 0.5, "llm_judge": 0.5},
            id="checks-and-judge-split-the-score",
        ),
    ],
)
async def test_the_grade_weighs_named_checks_and_the_judge(arguments, score, weights, monkeypatch, judge):
    monkeypatch.setenv("HUD_API_KEY", "runtime-key")
    row = cua_task(prompt="Say hello.", hud_api_key="task-key", **arguments)
    async with served(row) as client:
        started = await client.start_task(row.id, row.args)
        grade = await client.grade({"answer": "hello"})

    assert started["prompt"].endswith("Say hello.")
    assert grade["score"] == pytest.approx(score)
    assert {sub["name"]: sub["weight"] for sub in grade["subscores"]} == pytest.approx(weights)
    assert {authorization for authorization, _ in judge.requests} <= {"Bearer runtime-key"}


async def test_a_desktop_serves_one_task():
    row = cua_task(prompt="first", bash_checks=[PASS])
    async with served(row) as client:
        await client.start_task(row.id, row.args)
        with pytest.raises(HudProtocolError) as failure:
            await client.start_task(row.id, {**row.args, "prompt": "second"})

    assert "one task per substrate" in failure.value.message

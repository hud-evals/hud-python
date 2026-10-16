"""Rollout spans reach the HUD telemetry service, and its failures never reach the run."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, Taskset, rollout
from hud.telemetry import flush
from tests.support import Reply, ScriptedAgent, call, say

if TYPE_CHECKING:
    from hud.eval import Run
    from tests.support import HudEnv, Models, Platform

UPLOAD = "/trace/{id}/telemetry-upload"
ROOT = Path(__file__).parents[2]


def _env() -> Environment:
    env = Environment("upload")

    @env.template()
    async def answer(broken: bool = False):
        reply = yield "Reply with ok."
        if broken:
            raise RuntimeError("grader broke")
        yield 1.0 if reply == "ok" else 0.0

    return env


def _step_rows(spans: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    steps = [span for span in spans if span["attributes"].get("hud.schema") == "hud.step.v1"]
    return [
        (span["name"], span["status_code"], span["attributes"]["hud.payload"]["step_id"])
        for span in sorted(steps, key=lambda span: span["attributes"]["hud.payload"]["step_id"])
    ]


@pytest.mark.parametrize(
    ("broken", "reward", "rows"),
    [
        pytest.param(
            False,
            1.0,
            [
                ("step.task", "OK", 1),
                ("step.user", "OK", 2),
                ("step.agent", "OK", 3),
                ("step.tool", "OK", 4),
                ("step.agent", "OK", 5),
                ("step.task", "OK", 6),
            ],
            id="graded",
        ),
        pytest.param(
            True,
            0.0,
            [
                ("step.task", "OK", 1),
                ("step.user", "OK", 2),
                ("step.agent", "OK", 3),
                ("step.tool", "OK", 4),
                ("step.agent", "OK", 5),
                ("step.system", "ERROR", 6),
            ],
            id="grading-error",
        ),
    ],
)
async def test_every_rollout_step_is_uploaded_to_its_own_trace(
    reporting: Platform,
    models: Models,
    broken: bool,
    reward: float,
    rows: list[tuple[str, str, int]],
) -> None:
    models.script([call("missing_tool", x=1), say("ok")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))
    task = Task(env="upload", id="answer", args={"broken": broken})

    job = await Taskset("t", [task]).run(agent, runtime=LocalRuntime(_env()), group=2)

    assert [run.reward for run in job.runs] == [reward, reward]
    assert flush(timeout=10)
    assert sorted(reporting.traces) == sorted(
        str(run.trace_id).replace("-", "") for run in job.runs
    )
    for trace_id, trace in reporting.traces.items():
        assert {span["attributes"]["hud.task_run_id"].replace("-", "") for span in trace.spans} == {
            trace_id
        }
        assert _step_rows(trace.spans) == rows


async def test_nothing_is_uploaded_without_a_key(reporting: Platform, hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY=None)

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert job.reward == 1.0
    assert reporting.services.requests("telemetry") == []


async def test_an_upload_answered_503_is_retried_until_delivered(reporting: Platform) -> None:
    reporting.services.route(
        "telemetry", "POST", UPLOAD, Reply(status=503), Reply(status=503), Reply(json={})
    )

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert flush(timeout=10)
    first, *retries = reporting.services.requests("telemetry", "POST", UPLOAD)
    assert job.reward == 1.0
    assert [retry.json for retry in retries] == [first.json, first.json]


async def test_a_dead_telemetry_service_leaves_the_run_and_the_job_unaffected(
    reporting: Platform, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_TELEMETRY_URL="http://127.0.0.1:9")

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert job.reward == 1.0
    assert [run.trace.status for run in job.runs] == ["completed"]


async def test_flush_returns_false_at_its_deadline_while_an_upload_hangs(
    reporting: Platform,
) -> None:
    reporting.services.route("telemetry", "POST", UPLOAD, Reply(json={}, delay=2.0))

    run: Run = await rollout(
        Task(env="upload", id="answer"), ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert run.reward == 1.0
    assert flush(timeout=0.2) is False
    assert flush(timeout=30) is True
    assert reporting.services.requests("telemetry", "POST", UPLOAD) != []


def test_spans_still_queued_at_interpreter_exit_are_uploaded(
    reporting: Platform, tmp_path: Path
) -> None:
    script = tmp_path / "rollout_then_exit.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            from hud import Environment
            from hud.eval import LocalRuntime, Task, rollout
            from tests.support import ScriptedAgent

            env = Environment("upload")

            @env.template()
            async def answer():
                reply = yield "Reply with ok."
                yield 1.0 if reply == "ok" else 0.0

            task = Task(env="upload", id="answer")
            run = asyncio.run(rollout(task, ScriptedAgent("ok"), runtime=LocalRuntime(env)))
            print(run.trace_id)
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        cwd=ROOT,
    )

    trace_id = completed.stdout.strip().replace("-", "")
    assert [trace_id for trace_id, trace in reporting.traces.items() if trace.spans] == [trace_id]

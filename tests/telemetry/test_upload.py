"""Rollout spans reach the HUD telemetry service, and its failures never reach the run."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud import Environment, instrument
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, Taskset, rollout
from hud.telemetry import flush
from tests.harness import Reply, ScriptedAgent, call, say

if TYPE_CHECKING:
    from hud.eval import Run
    from tests.harness import FakeServices, HudEnv, Models

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


def _platform(services: FakeServices, *upload_replies: Reply) -> None:
    for path in ("/v2/trace/job/{id}/enter", "/v2/trace/{id}/enter", "/v2/trace/{id}/exit"):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", UPLOAD, *(upload_replies or (Reply(json={}),)))


def _uploaded(services: FakeServices) -> dict[str, list[dict[str, Any]]]:
    """Every uploaded span, grouped by the trace id in the upload path."""
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for request in services.requests("telemetry", "POST", UPLOAD):
        by_trace.setdefault(request.params["id"], []).extend(request.json["telemetry"])
    return by_trace


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
            snapshot(
                [
                    ("step.task", "OK", 1),
                    ("step.user", "OK", 2),
                    ("step.agent", "OK", 3),
                    ("step.tool", "OK", 4),
                    ("step.agent", "OK", 5),
                    ("step.task", "OK", 6),
                ]
            ),
            id="graded",
        ),
        pytest.param(
            True,
            0.0,
            snapshot(
                [
                    ("step.task", "OK", 1),
                    ("step.user", "OK", 2),
                    ("step.agent", "OK", 3),
                    ("step.tool", "OK", 4),
                    ("step.agent", "OK", 5),
                    ("step.system", "ERROR", 6),
                ]
            ),
            id="grading-error",
        ),
    ],
)
async def test_every_rollout_step_is_uploaded_to_its_own_trace(
    services: FakeServices,
    models: Models,
    hud_env: HudEnv,
    broken: bool,
    reward: float,
    rows: list[tuple[str, str, int]],
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services)
    models.script([call("missing_tool", x=1), say("ok")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))
    task = Task(env="upload", id="answer", args={"broken": broken})

    job = await Taskset("t", [task]).run(agent, runtime=LocalRuntime(_env()), group=2)

    assert [run.reward for run in job.runs] == [reward, reward]
    uploaded = _uploaded(services)
    assert sorted(uploaded) == sorted(str(run.trace_id).replace("-", "") for run in job.runs)
    for request in services.requests("telemetry", "POST", UPLOAD):
        assert request.bearer == "k"
        assert set(request.json) == {"telemetry"}
    for trace_id, spans in uploaded.items():
        assert {span["attributes"]["hud.task_run_id"].replace("-", "") for span in spans} == {
            trace_id
        }
        assert _step_rows(spans) == rows


def _instrumented_agent(calls: int, payload_bytes: int) -> ScriptedAgent:
    @instrument(name="probe")
    def probe(index: int) -> str:
        return "x" * payload_bytes

    def answer(prompt: str) -> str:
        del prompt
        for index in range(calls):
            probe(index)
        return "ok"

    return ScriptedAgent(answer)


@pytest.mark.parametrize(
    ("calls", "payload_bytes", "uploads"),
    [
        pytest.param(250, 10, 3, id="100-spans"),
        pytest.param(3, 3 * 1024 * 1024, 2, id="4-mib"),
    ],
)
async def test_a_batch_closes_at_100_spans_or_4_mib(
    services: FakeServices, hud_env: HudEnv, calls: int, payload_bytes: int, uploads: int
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services)

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        _instrumented_agent(calls, payload_bytes), runtime=LocalRuntime(_env())
    )

    batches = [
        upload.json["telemetry"] for upload in services.requests("telemetry", "POST", UPLOAD)
    ]
    assert job.reward == 1.0
    assert sum(span["name"] == "probe" for batch in batches for span in batch) == calls
    assert len(batches) >= uploads
    for batch in batches:
        assert len(batch) <= 100
        assert len(json.dumps(batch[:-1])) < 4 * 1024 * 1024


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param({"HUD_TELEMETRY_ENABLED": "1"}, id="no-api-key"),
        pytest.param({"HUD_API_KEY": "k", "HUD_TELEMETRY_ENABLED": "0"}, id="uploads-disabled"),
    ],
)
async def test_nothing_is_uploaded_without_a_key_or_with_uploads_off(
    services: FakeServices, hud_env: HudEnv, configuration: dict[str, str]
) -> None:
    hud_env.set(**configuration)
    _platform(services)

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert job.reward == 1.0
    assert services.requests("telemetry") == []


async def test_spans_outside_a_run_are_not_uploaded(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services)

    @instrument
    def outside() -> int:
        return 1

    assert outside() == 1
    assert flush(timeout=10)
    assert services.requests("telemetry") == []


async def test_an_upload_answered_503_is_retried_until_delivered(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services, Reply(status=503), Reply(status=503), Reply(json={}))

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    first, *retries = services.requests("telemetry", "POST", UPLOAD)
    assert job.reward == 1.0
    assert [retry.json for retry in retries] == [first.json, first.json]


async def test_a_dead_telemetry_service_leaves_the_run_and_the_job_unaffected(
    services: FakeServices, hud_env: HudEnv
) -> None:
    _platform(services)
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1", HUD_TELEMETRY_URL="http://127.0.0.1:9")

    job = await Taskset("t", [Task(env="upload", id="answer")]).run(
        ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert job.reward == 1.0
    assert [run.trace.status for run in job.runs] == ["completed"]
    assert flush(timeout=10)


async def test_flush_returns_false_at_its_deadline_while_an_upload_hangs(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services, Reply(json={}, delay=2.0))

    run: Run = await rollout(
        Task(env="upload", id="answer"), ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )

    assert run.reward == 1.0
    assert flush(timeout=0.2) is False
    assert flush(timeout=30) is True
    assert services.requests("telemetry", "POST", UPLOAD) != []


def test_spans_still_queued_at_interpreter_exit_are_uploaded(
    services: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    _platform(services)
    script = tmp_path / "rollout_then_exit.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            from hud import Environment
            from hud.eval import LocalRuntime, Task, rollout
            from tests.harness import ScriptedAgent

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
    assert [name for name in _uploaded(services)] == [trace_id]

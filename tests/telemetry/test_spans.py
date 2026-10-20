"""Where a run's spans go: local span files and the HUD telemetry service, including
``@hud.instrument`` spans, and when an upload is flushed."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud import Environment, instrument
from hud.agents.base import Agent
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, Taskset, rollout
from hud.telemetry import flush
from tests.conftest import spans, steps
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn, call
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

ISO = IsStr(regex=r"\d{4}-\d\d-\d\dT.*")
UPLOAD = "/trace/{id}/telemetry-upload"


def spans_env() -> Environment:
    env = Environment("spans")

    @env.template()
    async def answer():
        reply = yield "Reply with ok."
        yield 1.0 if reply == "ok" else 0.0

    return env


TASK = Task(env="spans", id="answer")


@pytest.mark.parametrize(
    ("uploads", "local_dir", "written_to", "uploaded"),
    [
        pytest.param("0", None, "home", False, id="uploads-off"),
        pytest.param("0", "spans", "spans", False, id="uploads-off-with-dir"),
        pytest.param("1", "spans", "spans", True, id="uploads-on-with-dir"),
        pytest.param("1", None, None, True, id="uploads-on"),
    ],
)
async def test_spans_are_written_where_the_configuration_says(
    reporting: FakeServices,
    hud_env: HudEnv,
    tmp_path: Path,
    uploads: str,
    local_dir: str | None,
    written_to: str | None,
    uploaded: bool,
) -> None:
    hud_env.set(
        HUD_TELEMETRY_ENABLED=uploads,
        HUD_TELEMETRY_LOCAL_DIR=str(tmp_path / local_dir) if local_dir else None,
    )
    directories = {"home": hud_env.home / ".hud" / "spans", "spans": tmp_path / "spans"}

    job = await Taskset("t", [TASK]).run(ScriptedAgent("ok"), runtime=LocalRuntime(spans_env()))

    (run,) = job.runs
    written = {name: spans(run.trace_id, path) != [] for name, path in directories.items()}
    assert written == {name: name == written_to for name in directories}
    assert flush(timeout=10)
    assert bool(reporting.requests("telemetry", "POST", UPLOAD)) is uploaded


async def test_step_spans_carry_the_step_payload_and_timing() -> None:
    run = await rollout(TASK, ScriptedAgent("ok"), runtime=LocalRuntime(spans_env()))

    exported = [span for span in spans(run.trace_id) if span["name"].startswith("step.")]
    assert all(span["end_time"] == ISO for span in exported)
    (prompted,) = [step for step in steps(run.trace_id) if step["source"] == "user"]
    assert prompted["messages"][0]["content"]["text"] == "Reply with ok."


class Probe:
    """An agent-side helper whose methods are instrumented."""

    @instrument(name="probe.lookup")
    async def lookup(self, items: list[int], limit: int = 3) -> dict[str, Any]:
        return {"found": items[:limit]}

    @instrument(name="probe.quiet", record_args=False, record_result=False)
    def quiet(self, secret: str) -> str:
        return secret

    @instrument(name="probe.fail")
    def fail(self, reason: str) -> None:
        raise ValueError(reason)

    @instrument(name="probe.opaque")
    def opaque(self, value: object) -> object:
        return value


class Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


class ProbingAgent(Agent):
    """Calls each instrumented helper once, then answers."""

    async def __call__(self, run: Run) -> None:
        probe = Probe()
        await probe.lookup(list(range(20)))
        probe.quiet("hunter2")
        with pytest.raises(ValueError, match="bad input"):
            probe.fail("bad input")
        probe.opaque(Unprintable())
        run.trace.content = "ok"


async def test_instrumented_calls_inside_a_run_record_their_arguments_and_outcome() -> None:
    run = await rollout(TASK, ProbingAgent(), runtime=LocalRuntime(spans_env()))

    assert run.reward == 1.0
    probes = {
        span["name"]: span for span in spans(run.trace_id) if span["name"].startswith("probe.")
    }
    events = {
        name: {event["name"]: event["attributes"] for event in span["events"]}
        for name, span in probes.items()
    }
    assert [
        probes[name]["status_code"] for name in ("probe.lookup", "probe.quiet", "probe.opaque")
    ] == ["OK"] * 3
    assert events["probe.lookup"] == {
        "hud.request": {"hud.payload": {"items": list(range(10)), "limit": 3}},
        "hud.result": {"hud.payload": {"found": [0, 1, 2]}},
    }
    assert events["probe.quiet"] == {}
    assert probes["probe.fail"]["status_code"] == "ERROR"
    assert "bad input" in probes["probe.fail"]["status_message"]
    assert events["probe.fail"]["hud.request"] == {"hud.payload": {"reason": "bad input"}}
    assert "bad input" in events["probe.fail"]["exception"]["exception.message"]
    assert events["probe.opaque"] == {
        "hud.request": {"hud.payload": {"value": "<Unprintable: not serializable>"}},
        "hud.result": {"hud.payload": "<Unprintable: not serializable>"},
    }


async def test_instrumented_calls_outside_a_run_export_nothing(hud_env: HudEnv) -> None:
    probe = Probe()

    assert await probe.lookup([1, 2]) == {"found": [1, 2]}
    assert not (hud_env.home / ".hud" / "spans").exists()


# ─── the HUD telemetry service ─────────────────────────────────────────

ROOT = Path(__file__).parents[2]


def upload_env() -> Environment:
    env = Environment("upload")

    @env.template()
    async def answer(broken: bool = False):
        reply = yield "Reply with ok."
        if broken:
            raise RuntimeError("grader broke")
        yield 1.0 if reply == "ok" else 0.0

    return env


def step_rows(spans: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
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
    reporting: FakeServices,
    models: Models,
    broken: bool,
    reward: float,
    rows: list[tuple[str, str, int]],
) -> None:
    models.script([call("missing_tool", x=1), Turn(text="ok")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))
    task = Task(env="upload", id="answer", args={"broken": broken})

    job = await Taskset("t", [task]).run(agent, runtime=LocalRuntime(upload_env()), group=2)

    assert [run.reward for run in job.runs] == [reward, reward]
    assert flush(timeout=10)
    uploaded: dict[str, list[dict[str, Any]]] = {}
    for request in reporting.requests("telemetry", "POST", UPLOAD):
        uploaded.setdefault(request.path.split("/")[2].replace("-", ""), []).extend(
            request.json["telemetry"]
        )
    assert sorted(uploaded) == sorted(str(run.trace_id).replace("-", "") for run in job.runs)
    for trace_id, sent in uploaded.items():
        assert {span["attributes"]["hud.task_run_id"].replace("-", "") for span in sent} == {
            trace_id
        }
        assert step_rows(sent) == rows


async def test_flush_returns_false_at_its_deadline_while_an_upload_hangs(
    reporting: FakeServices,
) -> None:
    reporting.route("telemetry", "POST", UPLOAD, Reply(json={}, delay=2.0))

    run: Run = await rollout(
        Task(env="upload", id="answer"), ScriptedAgent("ok"), runtime=LocalRuntime(upload_env())
    )

    assert run.reward == 1.0
    assert flush(timeout=0.2) is False
    assert flush(timeout=30) is True
    assert reporting.requests("telemetry", "POST", UPLOAD) != []


def test_spans_still_queued_at_interpreter_exit_are_uploaded(
    reporting: FakeServices, tmp_path: Path
) -> None:
    script = tmp_path / "rollout_then_exit.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            from hud import Environment
            from hud.eval import LocalRuntime, Task, rollout
            from tests.fixtures.agents import ScriptedAgent

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
    uploaded = {r.path.split("/")[2].replace("-", "") for r in reporting.requests("telemetry")}
    assert uploaded == {trace_id}

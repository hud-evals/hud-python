"""Spans a run writes to local files, including ``@hud.instrument`` spans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud import Environment, instrument
from hud.agents.base import Agent
from hud.eval import LocalRuntime, Task, Taskset, rollout
from tests.harness import ScriptedAgent, spans, steps

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Run
    from tests.harness import FakeServices, HudEnv

ISO = IsStr(regex=r"\d{4}-\d\d-\d\dT.*")


def _env() -> Environment:
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
    services: FakeServices,
    hud_env: HudEnv,
    tmp_path: Path,
    uploads: str,
    local_dir: str | None,
    written_to: str | None,
    uploaded: bool,
) -> None:
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})
    hud_env.set(
        HUD_API_KEY="k",
        HUD_TELEMETRY_ENABLED=uploads,
        HUD_TELEMETRY_LOCAL_DIR=str(tmp_path / local_dir) if local_dir else None,
    )
    directories = {"home": hud_env.home / ".hud" / "spans", "spans": tmp_path / "spans"}

    job = await Taskset("t", [TASK]).run(ScriptedAgent("ok"), runtime=LocalRuntime(_env()))

    (run,) = job.runs
    written = {name: spans(run.trace_id, path) != [] for name, path in directories.items()}
    assert written == {name: name == written_to for name in directories}
    assert (services.requests("telemetry") != []) is uploaded


async def test_an_unwritable_span_directory_does_not_fail_the_run(
    hud_env: HudEnv, tmp_path: Path
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    hud_env.set(HUD_TELEMETRY_LOCAL_DIR=str(blocker / "spans"))

    run = await rollout(TASK, ScriptedAgent("ok"), runtime=LocalRuntime(_env()))

    assert run.reward == 1.0
    assert run.trace.status == "completed"


async def test_step_spans_carry_the_step_schema_payload_and_timing() -> None:
    run = await rollout(TASK, ScriptedAgent("ok"), runtime=LocalRuntime(_env()))

    exported = [span for span in spans(run.trace_id) if span["name"].startswith("step.")]
    assert [
        (span["name"], span["attributes"]["hud.schema"], span["status_code"]) for span in exported
    ] == snapshot(
        [
            ("step.task", "hud.step.v1", "OK"),
            ("step.user", "hud.step.v1", "OK"),
            ("step.task", "hud.step.v1", "OK"),
        ]
    )
    assert {span["attributes"]["hud.task_run_id"] for span in exported} == {run.trace_id}
    assert [step["step_id"] for step in steps(run.trace_id)] == [1, 2, 3]
    assert all(span["end_time"] == ISO for span in exported)
    assert steps(run.trace_id)[1]["messages"][0]["content"]["text"] == "Reply with ok."


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
    run = await rollout(TASK, ProbingAgent(), runtime=LocalRuntime(_env()))

    assert run.reward == 1.0
    probes = {
        span["name"]: {
            "status": span["status_code"],
            "message": span["status_message"],
            "events": {event["name"]: event["attributes"] for event in span["events"]},
        }
        for span in spans(run.trace_id)
        if span["name"].startswith("probe.")
    }
    assert probes == snapshot(
        {
            "probe.lookup": {
                "status": "OK",
                "message": None,
                "events": {
                    "hud.request": {
                        "hud.payload": {"items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "limit": 3}
                    },
                    "hud.result": {"hud.payload": {"found": [0, 1, 2]}},
                },
            },
            "probe.quiet": {"status": "OK", "message": None, "events": {}},
            "probe.fail": {
                "status": "ERROR",
                "message": "ValueError: bad input",
                "events": {
                    "hud.request": {"hud.payload": {"reason": "bad input"}},
                    "exception": {"exception.message": "ValueError: bad input"},
                },
            },
            "probe.opaque": {
                "status": "OK",
                "message": None,
                "events": {
                    "hud.request": {"hud.payload": {"value": "<Unprintable: not serializable>"}},
                    "hud.result": {"hud.payload": "<Unprintable: not serializable>"},
                },
            },
        }
    )


async def test_instrumented_calls_outside_a_run_export_nothing(hud_env: HudEnv) -> None:
    probe = Probe()

    assert await probe.lookup([1, 2]) == {"found": [1, 2]}
    assert not (hud_env.home / ".hud" / "spans").exists()

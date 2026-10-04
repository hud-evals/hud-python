"""Where a run's spans go: local span files and the HUD telemetry service, and when an
upload is flushed."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment, Task, Taskset
from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval.run import rollout
from hud.runtime import LocalRuntime
from hud.telemetry import flush
from tests.conftest import spans
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn, call
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from hud import Trace
    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

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

    (trace,) = job.traces
    written = {name: spans(trace.trace_id, path) != [] for name, path in directories.items()}
    assert written == {name: name == written_to for name in directories}
    assert flush(timeout=10)
    assert bool(reporting.requests("telemetry", "POST", UPLOAD)) is uploaded


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


@pytest.mark.parametrize(
    ("broken", "reward"), [(False, 1.0), (True, 0.0)], ids=["graded", "grading-error"]
)
async def test_every_rollout_step_is_uploaded_to_its_own_trace(
    reporting: FakeServices, models: Models, broken: bool, reward: float
) -> None:
    models.script([call("missing_tool", x=1), Turn(text="ok")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="scripted", max_steps=4))
    task = Task(env="upload", id="answer", args={"broken": broken})

    job = await Taskset("t", [task]).run(agent, runtime=LocalRuntime(upload_env()), group=2)

    assert [trace.reward for trace in job.traces] == [reward, reward]
    assert flush(timeout=10)
    uploaded: dict[str, list[dict[str, Any]]] = {}
    for request in reporting.requests("telemetry", "POST", UPLOAD):
        uploaded.setdefault(request.path.split("/")[2].replace("-", ""), []).extend(
            request.json["telemetry"]
        )
    assert sorted(uploaded) == sorted(str(trace.trace_id).replace("-", "") for trace in job.traces)
    for trace_id, sent in uploaded.items():
        assert {span["attributes"]["hud.task_run_id"].replace("-", "") for span in sent} == {
            trace_id
        }
        step_statuses = {
            span["status_code"]
            for span in sent
            if span["attributes"].get("hud.schema") == "hud.step.v1"
        }
        assert ("ERROR" in step_statuses) is broken


async def test_flush_returns_false_at_its_deadline_while_an_upload_hangs(
    reporting: FakeServices,
) -> None:
    reporting.route("telemetry", "POST", UPLOAD, Reply(json={}, delay=2.0))

    trace: Trace = await rollout(
        Task(env="upload", id="answer"), ScriptedAgent("ok"), runtime=LocalRuntime(upload_env())
    )

    assert trace.reward == 1.0
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
            from hud import Task
            from hud.eval.run import rollout
            from hud.runtime import LocalRuntime
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

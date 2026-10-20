"""A place a run reports to that fails never fails the run: the local span directory, the
HUD telemetry service, and the platform's job and trace records. A report retried until
it lands arrives intact; one given up on leaves a warning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.eval import LocalRuntime, Task
from hud.telemetry import flush
from tests.fixtures.agents import ScriptedAgent
from tests.support import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import FakeServices, HudEnv


def checked() -> Environment:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    return env


async def test_a_span_directory_that_cannot_be_created_does_not_fail_the_run(
    reporting: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    (tmp_path / "a-file").write_text("")
    hud_env.set(HUD_TELEMETRY_LOCAL_DIR=str(tmp_path / "a-file" / "spans"))

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )

    assert (job.reward, job.runs[0].trace.status) == (1.0, "completed")


async def test_a_telemetry_service_that_is_down_does_not_fail_the_run(
    reporting: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_TELEMETRY_URL="http://127.0.0.1:9")

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )
    assert flush(timeout=10)

    assert (job.reward, job.runs[0].trace.status) == (1.0, "completed")


@pytest.mark.parametrize(
    ("service", "path", "statuses", "attempts", "gave_up"),
    [
        ("telemetry", "/trace/{id}/telemetry-upload", [503, 503, 200], 3, False),
        ("api", "/v2/trace/job/{id}/enter", [500, 200], 2, False),
        ("api", "/v2/trace/job/{id}/enter", [500], 2, True),
    ],
    ids=[
        "span-uploads-answered-503-until-they-land",
        "a-job-report-rejected-once",
        "a-job-report-rejected-twice",
    ],
)
async def test_a_rejected_report_is_retried_intact_and_never_fails_the_run(
    service: str,
    path: str,
    statuses: list[int],
    attempts: int,
    gave_up: bool,
    reporting: FakeServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reporting.route(service, "POST", path, *(Reply(status=code, json={}) for code in statuses))

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )
    assert flush(timeout=10)

    sent = reporting.requests(service, "POST", path)
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "hud.eval.job" and r.levelno == logging.WARNING
    ]
    assert (job.reward, job.runs[0].trace.status) == (1.0, "completed")
    assert (len(sent), len({str(request.json) for request in sent})) == (attempts, 1)
    assert [w.startswith("platform report /trace/job/") and "500" in w for w in warnings] == (
        [True] if gave_up else []
    )

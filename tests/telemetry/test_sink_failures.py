"""A place a run reports to that fails never fails the run: the local span directory, the
HUD telemetry service, and the platform's job and trace records. A report retried until
it lands arrives intact; one given up on leaves a warning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from hud.eval import LocalRuntime, Task
from hud.telemetry import flush
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices


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
    ("service", "path", "statuses", "gave_up"),
    [
        ("telemetry", "/trace/{id}/telemetry-upload", [503, 503, 200], False),
        ("api", "/v2/trace/job/{id}/enter", [500, 200], False),
        ("api", "/v2/trace/job/{id}/enter", [500], True),
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
    gave_up: bool,
    reporting: FakeServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reporting.route(service, "POST", path, *(Reply(status=code, json={}) for code in statuses))

    job = await Task(env="checked", id="check").run(
        ScriptedAgent("yes"), runtime=LocalRuntime(checked())
    )
    assert flush(timeout=10)

    first, *retries = reporting.requests(service, "POST", path)
    warned = any(r.name == "hud.eval.job" and r.levelno == logging.WARNING for r in caplog.records)
    assert (job.reward, job.runs[0].trace.status) == (1.0, "completed")
    assert retries
    assert retries[-1].json == first.json
    assert warned is gave_up

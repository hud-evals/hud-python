"""A reporting sink that fails never fails the run.

Every place a run reports to is a row: the local span directory, the HUD
telemetry service, and the platform's job and trace records. Each row breaks its
sink, runs a graded taskset, and checks that the run and the job are unaffected
and what the sink still received: a retried report arrives once, a report given
up on leaves a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.eval import LocalRuntime, Task, Taskset
from hud.telemetry import flush
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import lab, solve
from tests.support import Reply

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.support import HudEnv, Platform

UPLOAD = "/trace/{id}/telemetry-upload"
JOB_ENTER = "/v2/trace/job/{id}/enter"


@dataclass(frozen=True)
class Failure:
    """How to break the sink, and what it holds afterwards."""

    variables: Callable[[Path], dict[str, str]] = lambda tmp_path: {}
    service: str | None = None
    path: str = ""
    replies: list[Reply] = field(default_factory=list)
    attempts: int | None = None
    warnings: list[Any] = field(default_factory=list)


FAILURES = {
    "a span directory that cannot be created": Failure(
        lambda tmp_path: {"HUD_TELEMETRY_LOCAL_DIR": str(tmp_path / "a-file" / "spans")}
    ),
    "a telemetry service that is down": Failure(
        lambda tmp_path: {"HUD_TELEMETRY_URL": "http://127.0.0.1:9"}
    ),
    "a telemetry service answering 503 until it delivers": Failure(
        service="telemetry",
        path=UPLOAD,
        replies=[Reply(status=503), Reply(status=503), Reply(json={})],
        attempts=3,
    ),
    "a job report rejected once": Failure(
        service="api", path=JOB_ENTER, replies=[Reply(status=500), Reply(json={})], attempts=2
    ),
    "a job report rejected twice": Failure(
        service="api",
        path=JOB_ENTER,
        replies=[Reply(status=500)],
        attempts=2,
        warnings=[IsStr(regex=r"platform report /trace/job/[0-9a-f]{32}/enter failed: .*500.*")],
    ),
}


@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
async def test_a_failing_sink_never_fails_the_run(
    failure: Failure,
    reporting: Platform,
    hud_env: HudEnv,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "a-file").write_text("")
    hud_env.set(**failure.variables(tmp_path))
    if failure.service is not None:
        reporting.services.route(failure.service, "POST", failure.path, *failure.replies)

    job = await Taskset("t", [Task(env="lab", id="add", args={"a": 2, "b": 3})]).run(
        ScriptedAgent(solve), runtime=LocalRuntime(lab())
    )
    assert flush(timeout=10)

    (run,) = job.runs
    assert (run.reward, run.trace.status, job.reward) == (1.0, "completed", 1.0)
    if failure.service is not None:
        sent = reporting.services.requests(failure.service, "POST", failure.path)
        assert len(sent) == failure.attempts
        assert {str(request.json) for request in sent} == {str(sent[0].json)}
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "hud.eval.job" and record.levelno == logging.WARNING
    ] == failure.warnings

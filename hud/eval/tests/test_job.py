"""``hud.eval.job`` reporting — the trace-exit payload sent to the platform.

No network: the platform client is replaced with a recorder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import job as job_mod
from hud.eval.run import Run

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Recorder:
    """Stand-in platform client that captures the last reported body."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def apost(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, json))
        return {}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    rec = _Recorder()
    monkeypatch.setattr(job_mod.PlatformClient, "from_settings", classmethod(lambda cls: rec))
    yield rec


def _run_with(trace_id: str, *, extra: dict[str, Any]) -> Run:
    run = Run(None, "task", {})
    run.trace.trace_id = trace_id
    run.trace.status = "completed"
    run.trace.extra = extra
    return run


async def test_open_job_reports_submission_lifecycle(recorder: _Recorder) -> None:
    await job_mod.job_enter(
        "job-1",
        name="benchmark",
        group=2,
        taskset_id="taskset-1",
        is_open=True,
    )
    await job_mod.job_exit("job-1")

    assert recorder.calls == [
        (
            "/trace/job/job-1/enter",
            {
                "name": "benchmark",
                "group": 2,
                "taskset_id": "taskset-1",
                "is_open": True,
            },
        ),
        ("/trace/job/job-1/exit", {"failed": False}),
    ]


async def test_trace_enter_reports_task_and_group_identity(recorder: _Recorder) -> None:
    await job_mod.trace_enter(
        "abc",
        job_id="job-1",
        group_id="group-1",
        task_slug="fix-bug-3",
        model="test-model",
    )

    assert recorder.calls == [
        (
            "/trace/abc/enter",
            {
                "job_id": "job-1",
                "group_id": "group-1",
                "task_slug": "fix-bug-3",
                "model": "test-model",
            },
        )
    ]


async def test_job_enter_logs_canonical_web_uuid(
    recorder: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from hud.settings import settings

    compact_id = "03dd2a73d3df4d10a54ae3d87c2d530d"
    canonical_id = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
    monkeypatch.setattr(settings, "hud_web_url", "https://hud.test")
    caplog.set_level(logging.INFO, logger="hud.eval.job")

    await job_mod.job_enter(compact_id, name="test", group=1)

    assert recorder.calls[0][0] == f"/trace/job/{compact_id}/enter"
    assert f"job: https://hud.test/jobs/{canonical_id}" in caplog.text


async def test_trace_exit_propagates_stop_reason(recorder: _Recorder) -> None:
    run = _run_with("abc", extra={})
    run.trace.stop_reason = "max_steps"
    await job_mod.trace_exit(run)

    assert len(recorder.calls) == 1
    path, body = recorder.calls[0]
    assert path == "/trace/abc/exit"
    assert body["stop_reason"] == "max_steps"
    assert "metadata" not in body


async def test_trace_exit_omits_metadata_when_extra_empty(recorder: _Recorder) -> None:
    await job_mod.trace_exit(_run_with("abc", extra={}))

    assert len(recorder.calls) == 1
    _, body = recorder.calls[0]
    assert "metadata" not in body


def test_errored_runs_do_not_deflate_the_job_reward() -> None:
    """Infrastructure failure is never a score: a run that errored (a launch
    failure, or a hosted trace that ended in error) carries no verdict and
    must not drag the job mean down as a silent zero."""
    from hud.eval.job import Job
    from hud.eval.run import Grade

    graded = _run_with("t1", extra={})
    graded.grade = Grade(reward=1.0)
    failed = Run.failed("provisioning never finished")

    job = Job(id="j1", name="test", runs=[graded, failed])

    assert job.reward == 1.0
    assert job.errors == [failed]


def test_job_with_only_errors_reports_zero_reward() -> None:
    from hud.eval.job import Job

    job = Job(id="j2", name="test", runs=[Run.failed("boom")])

    assert job.reward == 0.0
    assert job.errors and job.errors[0].trace.is_error

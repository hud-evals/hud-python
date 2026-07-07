"""``hud.eval.job`` reporting — the trace-exit payload sent to the platform.

No network: the platform client is replaced with a recorder.
"""

from __future__ import annotations

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

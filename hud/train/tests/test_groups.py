"""GRPO group cardinality with timed-out / failed rollouts (HUD-2099)."""

from __future__ import annotations

import uuid

import pytest

from hud.eval.run import Run
from hud.train.client import _check_groups, _to_inputs


def _terminal_run(*, error: str | None = None, stop_reason: str | None = None) -> Run:
    run = Run.failed(error) if error else Run(None, "t", {})
    if error is None:
        run.trace.status = "completed"
    run.trace.trace_id = uuid.uuid4().hex
    if stop_reason is not None:
        run.trace.stop_reason = stop_reason  # type: ignore[assignment]
    return run


def test_check_groups_rejects_incomplete_cardinality() -> None:
    with pytest.raises(ValueError, match="do not divide evenly"):
        _check_groups(7, 8)


def test_timed_out_runs_keep_grpo_group_full() -> None:
    """Keep timed-out members in the batch — dropping them breaks group_size."""
    batch = [
        _terminal_run(error="agent loop timed out", stop_reason="timeout"),
        *(_terminal_run() for _ in range(7)),
    ]
    inputs = _to_inputs(batch)
    assert len(inputs) == 8
    _check_groups(len(inputs), 8)  # must not raise

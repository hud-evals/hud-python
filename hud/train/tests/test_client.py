from __future__ import annotations

import contextlib
import logging

import pytest

from hud.eval.run import Run
from hud.train import TrainingClient


async def test_training_rejects_timed_out_run_without_explicit_reward() -> None:
    run = Run.failed("rollout timed out")
    run.trace.stop_reason = "timeout"

    with pytest.raises(ValueError, match="explicit training reward"):
        await TrainingClient("test-model").forward_backward([run], group_size=1)


async def test_training_rejects_incomplete_run_groups() -> None:
    runs = [Run(None, "", {}) for _ in range(4)]
    for index, run in enumerate(runs):
        run.trace.trace_id = str(index)
        run.group_id = "a" if index < 3 else "b"

    with pytest.raises(ValueError, match="incomplete GRPO groups"):
        await TrainingClient("test-model").forward_backward(runs, group_size=2)


async def test_training_warns_when_no_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Identical rewards zero every GRPO advantage, so nothing accumulates and the
    # next optim_step 409s with a message that blames the wrong call.
    runs = [Run(None, "", {}) for _ in range(4)]
    for index, run in enumerate(runs):
        run.trace.trace_id = str(index)
        run.group_id = "a" if index < 2 else "b"
        run.grade.reward = 1.0

    with caplog.at_level(logging.WARNING, logger="hud.train"), contextlib.suppress(Exception):
        await TrainingClient("test-model").forward_backward(runs, group_size=2)

    assert "no gradient" in caplog.text


async def test_training_is_quiet_when_a_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs = [Run(None, "", {}) for _ in range(4)]
    for index, run in enumerate(runs):
        run.trace.trace_id = str(index)
        run.group_id = "a" if index < 2 else "b"
        run.grade.reward = float(index % 2)

    with caplog.at_level(logging.WARNING, logger="hud.train"), contextlib.suppress(Exception):
        await TrainingClient("test-model").forward_backward(runs, group_size=2)

    assert "no gradient" not in caplog.text

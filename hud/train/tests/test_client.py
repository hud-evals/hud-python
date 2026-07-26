from __future__ import annotations

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

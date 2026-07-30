from __future__ import annotations

import logging
from typing import Any

import pytest

from hud.eval.run import Run
from hud.train import TrainingClient

_MODEL_ID = "00000000-0000-0000-0000-000000000001"


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


def _grpo_runs(groups: list[str], rewards: list[float]) -> list[Run]:
    runs = [Run(None, "", {}) for _ in rewards]
    for index, (run, group_id, reward) in enumerate(zip(runs, groups, rewards, strict=True)):
        run.trace.trace_id = str(index)
        run.group_id = group_id
        run.grade.reward = reward
    return runs


async def _forward_backward(runs: list[Run], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run forward_backward against a stubbed training service (a UUID model id
    resolves without a catalog lookup)."""

    async def fake_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        return {"metrics": {}, "num_datums": len(runs)}

    monkeypatch.setattr("hud.train.base.make_request", fake_request)
    await TrainingClient(_MODEL_ID).forward_backward(runs, group_size=2)


async def test_training_warns_when_no_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Identical rewards within every group zero every GRPO advantage, so the
    # pass accumulates nothing and the later optim_step fails confusingly.
    runs = _grpo_runs(["a", "a", "b", "b"], [1.0, 1.0, 0.0, 0.0])

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await _forward_backward(runs, monkeypatch)

    assert "no gradient" in caplog.text


async def test_training_groups_rewards_by_group_id_not_position(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An interleaved batch is valid (groups are keyed by group_id), and its
    # positional slices have spread even when the real groups have none.
    runs = _grpo_runs(["a", "b", "a", "b"], [1.0, 0.0, 1.0, 0.0])

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await _forward_backward(runs, monkeypatch)

    assert "no gradient" in caplog.text


async def test_training_is_quiet_when_a_group_has_reward_spread(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _grpo_runs(["a", "a", "b", "b"], [0.0, 1.0, 0.0, 1.0])

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await _forward_backward(runs, monkeypatch)

    assert "no gradient" not in caplog.text

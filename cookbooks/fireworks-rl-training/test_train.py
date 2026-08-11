from __future__ import annotations

import pytest
from hud.agents.types import AgentStep, Sample
from hud.eval.run import Run

from train import (
    group_relative_advantages,
    make_training_batch,
    serverless_url,
    within_group_reward_std,
)


def rollout(*, group: str, reward: float) -> Run:
    run = Run(None, "multiply", {})
    run.group_id = group
    run.grade.reward = reward
    run.record(
        AgentStep(
            content="42",
            sample=Sample(
                prompt_token_ids=[1, 2],
                output_token_ids=[3, 4],
                output_logprobs=[-0.2, -0.1],
            ),
        )
    )
    return run


def test_serverless_url_accepts_root_and_complete_endpoint() -> None:
    expected = "https://api.fireworks.ai/training/v1/serverless"
    assert serverless_url("https://api.fireworks.ai") == expected
    assert serverless_url(expected) == expected


def test_within_group_reward_std_averages_per_group_spread() -> None:
    runs = [
        rollout(group="spread", reward=0.0),
        rollout(group="spread", reward=1.0),
        rollout(group="flat", reward=1.0),
        rollout(group="flat", reward=1.0),
    ]
    # sample std of [0, 1] is ~0.707; the flat group contributes 0.
    assert within_group_reward_std(runs) == pytest.approx(0.3536, abs=1e-3)
    assert within_group_reward_std([]) == 0.0


def test_group_relative_advantages_are_centered() -> None:
    advantages = group_relative_advantages([0.0, 1.0])
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < 0 < advantages[1]


def test_training_batch_keeps_only_groups_with_reward_spread() -> None:
    runs = [
        rollout(group="learnable", reward=0.0),
        rollout(group="learnable", reward=1.0),
        rollout(group="flat", reward=0.0),
        rollout(group="flat", reward=0.0),
    ]

    datums, kept_groups = make_training_batch(runs)

    assert kept_groups == 1
    assert len(datums) == 2
    for datum in datums:
        assert datum.model_input.length == 3
        assert len(datum.loss_fn_inputs["target_tokens"].data) == 3
        assert len(datum.loss_fn_inputs["logprobs"].data) == 3
        assert len(datum.loss_fn_inputs["advantages"].data) == 3

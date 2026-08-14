from __future__ import annotations

from argparse import Namespace

import pytest
from hud.agents.types import AgentStep, Sample
from hud.eval import HUDRuntime, LocalRuntime, Taskset
from hud.eval.run import Run

from train import (
    group_relative_advantages,
    make_taskset,
    make_training_batch,
    resolve_rollout_source,
    serverless_url,
    split_taskset,
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


def test_split_taskset_creates_disjoint_deterministic_subsets() -> None:
    source = make_taskset(count=8, seed=0, a=(10, 99), b=(10, 99))

    train, evaluation = split_taskset(source, train_count=5, eval_count=3, seed=7)
    train_again, evaluation_again = split_taskset(source, train_count=5, eval_count=3, seed=7)

    train_slugs = [task.slug for task in train]
    evaluation_slugs = [task.slug for task in evaluation]
    assert train_slugs == [task.slug for task in train_again]
    assert evaluation_slugs == [task.slug for task in evaluation_again]
    assert set(train_slugs).isdisjoint(evaluation_slugs)


def test_make_taskset_rejects_more_tasks_than_unique_operand_pairs() -> None:
    with pytest.raises(ValueError, match="only 4 unique pairs"):
        make_taskset(count=5, seed=0, a=(1, 2), b=(1, 2))


def test_resolve_rollout_source_supports_hosted_and_local_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_taskset(count=4, seed=0, a=(10, 99), b=(10, 99))
    monkeypatch.setattr(Taskset, "from_api", classmethod(lambda cls, name: source))
    monkeypatch.setattr(Taskset, "from_file", classmethod(lambda cls, path: source))
    common = {
        "tasks_per_step": 2,
        "eval_tasks": 2,
        "seed": 0,
    }

    _, _, hosted_runtime = resolve_rollout_source(
        Namespace(taskset="demo", tasks_file=None, env_path=None, **common)
    )
    _, _, local_runtime = resolve_rollout_source(
        Namespace(taskset=None, tasks_file="tasks.py", env_path="env.py", **common)
    )

    assert isinstance(hosted_runtime, HUDRuntime)
    assert isinstance(local_runtime, LocalRuntime)


def test_default_rollout_source_has_disjoint_evaluation_tasks() -> None:
    train, evaluation, runtime = resolve_rollout_source(
        Namespace(
            taskset=None,
            tasks_file=None,
            env_path=None,
            tasks_per_step=5,
            eval_tasks=3,
            seed=0,
            min_a=10,
            max_a=99,
            min_b=10,
            max_b=99,
        )
    )

    assert {task.slug for task in train}.isdisjoint(task.slug for task in evaluation)
    assert isinstance(runtime, LocalRuntime)

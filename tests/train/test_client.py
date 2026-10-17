"""What TrainingClient sends to the HUD training service, and what it refuses to send."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any

import pytest
import torch
from dirty_equals import IsStr

from hud import Environment
from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Run, Task, Taskset
from hud.train import DatumTensors, TrainingClient
from hud.utils.exceptions import HudRequestError
from tests.support import Reply, ScriptedAgent

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from tests.support import FakeServices, Platform, Request

MODEL_ID = "00000000-0000-4000-8000-000000000001"
TRAIN = f"/v1/models/{MODEL_ID}/train"
TRACE_ID = IsStr(regex=r"[0-9a-f]{32}")


def _env() -> Environment:
    env = Environment("train")

    @env.template()
    async def answer(target: str = "ok"):
        reply = yield "Reply with ok."
        yield 1.0 if reply == target else 0.0

    return env


@pytest.fixture
def trainable(platform: Platform) -> Platform:
    """The platform, with ``owner/model`` in its catalog."""
    platform.add_model(id=MODEL_ID, model_name="owner/model")
    return platform


def _sampling_gateway(services: FakeServices) -> None:
    """A trainable model behind the gateway: every reply carries token ids and logprobs."""
    turns = itertools.count()

    def complete(request: Request) -> Reply:
        assert request.json["return_token_ids"] is True
        content = "ok" if next(turns) % 2 == 0 else "no"
        return Reply(
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "owner/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                        "logprobs": {
                            "content": [
                                {"token": content, "logprob": -0.5, "bytes": [], "top_logprobs": []}
                            ]
                        },
                        "prompt_token_ids": [1, 2, 3],
                        "token_ids": [7],
                    }
                ],
            }
        )

    services.route("gateway", "POST", "/chat/completions", handler=complete)


async def _runs(agent: Agent, *, tasks: list[Task] | None = None, group: int = 2) -> list[Run]:
    job = await Taskset("t", tasks or [Task(env="train", id="answer")]).run(
        agent, runtime=LocalRuntime(_env()), group=group, max_concurrent=1
    )
    return job.runs


def _trainable_agent() -> OpenAIChatAgent:
    return OpenAIChatAgent(
        OpenAIChatConfig(
            model="owner/model",
            max_steps=1,
            completion_kwargs={"extra_body": {"return_token_ids": True}},
        )
    )


async def test_a_step_sends_the_sampled_runs_then_one_optimizer_step(
    trainable: Platform,
) -> None:
    _sampling_gateway(trainable.services)
    runs = await _runs(_trainable_agent())

    await TrainingClient("owner/model").step(runs, learning_rate=1e-5, group_size=2)

    training = trainable.training[MODEL_ID]
    (sent,) = training.batches
    sample = {"prompt_token_ids": [1, 2, 3], "output_token_ids": [7], "output_logprobs": [-0.5]}
    assert [(i["samples"], i["reward"], i["trace_id"]) for i in sent["inputs"]] == [
        ([sample], 1.0, TRACE_ID),
        ([sample], 0.0, TRACE_ID),
    ]
    assert [step["learning_rate"] for step in training.steps] == [1e-5]


async def test_runs_without_token_samples_train_by_trace_id(trainable: Platform) -> None:
    runs = await _runs(ScriptedAgent("ok"))

    await TrainingClient("owner/model").forward_backward(runs, group_size=2)

    (batch,) = trainable.training[MODEL_ID].batches
    assert batch["inputs"] == [str(run.trace_id) for run in runs]


async def test_the_model_slug_resolves_once_with_the_clients_own_key(
    trainable: Platform,
) -> None:
    runs = await _runs(ScriptedAgent("ok"))
    trainable.keys = {"team-key"}
    client = TrainingClient("owner/model", api_key="team-key")

    await client.forward_backward(runs, group_size=2)
    await client.optim_step(learning_rate=1e-5)

    training = trainable.training[MODEL_ID]
    assert (len(training.batches), len(training.steps)) == (1, 1)
    assert len(trainable.services.requests("api", "GET", "/v2/models")) == 1


async def test_an_optimizer_step_is_never_retried(trainable: Platform) -> None:
    trainable.services.route("rl", "POST", f"{TRAIN}/optim-step", Reply(status=503))

    with pytest.raises(HudRequestError) as raised:
        await TrainingClient(MODEL_ID).optim_step(learning_rate=1e-5)

    assert raised.value.status_code == 503
    assert len(trainable.services.requests("rl", "POST", f"{TRAIN}/optim-step")) == 1


async def test_a_custom_loss_sends_its_gradients_to_backward(trainable: Platform) -> None:
    runs = await _runs(ScriptedAgent("ok"))
    trainable.services.route(
        "rl",
        "POST",
        f"{TRAIN}/forward",
        json={
            "forward_id": "fwd-1",
            "data": [
                {
                    "reward": reward,
                    "traj_idx": index,
                    "logprobs": [-0.5, -1.0],
                    "sampling_logprobs": [-0.5, -1.0],
                    "mask": [1.0, 1.0],
                }
                for index, reward in enumerate([1.0, 0.5])
            ],
        },
    )

    def loss(
        data: list[DatumTensors], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = sum(-datum.reward * lp.sum() for datum, lp in zip(data, logprobs, strict=True))
        assert isinstance(total, torch.Tensor)
        return total, {"loss": total.item()}

    await TrainingClient(MODEL_ID).forward_backward_custom(runs, loss, group_size=2)

    (forward,) = trainable.services.bodies("rl", "POST", f"{TRAIN}/forward")
    (backward,) = trainable.training[MODEL_ID].batches
    assert forward["inputs"] == [str(run.trace_id) for run in runs]
    assert backward == {
        "forward_id": "fwd-1",
        "weights": [[1.0, 1.0], [0.5, 0.5]],
        "metrics": {"loss": 2.25},
    }


async def _timed_out() -> list[Run]:
    job = await Taskset("t", [Task(env="train", id="answer")]).run(
        ScriptedAgent("ok", delay=5), runtime=LocalRuntime(_env()), rollout_timeout=0.2
    )
    return job.runs


@pytest.mark.parametrize(
    ("runs", "group_size", "message"),
    [
        pytest.param(
            _timed_out, None, "timed-out runs require an explicit training reward", id="timeout"
        ),
        pytest.param(
            lambda: _first(3, _runs(ScriptedAgent("ok"), group=2, tasks=_two_tasks())),
            2,
            "3 trajectories do not divide evenly into groups of 2",
            id="uneven",
        ),
        pytest.param(
            lambda: _split_two_groups(_runs(ScriptedAgent("ok"), group=2, tasks=_three_tasks())),
            2,
            "incomplete GRPO groups",
            id="incomplete-group",
        ),
        pytest.param(
            lambda: _single(Run.failed("never launched")),
            None,
            "run carries neither token-level samples nor a trace_id",
            id="no-trace",
        ),
    ],
)
async def test_an_untrainable_batch_is_refused_before_anything_is_sent(
    trainable: Platform, runs: Any, group_size: int | None, message: str
) -> None:
    batch = await runs()

    with pytest.raises(ValueError, match=message):
        await TrainingClient("owner/model").forward_backward(batch, group_size=group_size)
    assert trainable.services.requests() == []


def _two_tasks() -> list[Task]:
    return [Task(env="train", id="answer", args={"target": target}) for target in ("ok", "no")]


def _three_tasks() -> list[Task]:
    return [Task(env="train", id="answer", args={"target": t}) for t in ("ok", "no", "maybe")]


async def _first(count: int, runs: Any) -> list[Run]:
    return (await runs)[:count]


async def _split_two_groups(runs: Any) -> list[Run]:
    batch = await runs
    return [batch[0], batch[2], batch[4], batch[5]]


async def _single(run: Run) -> list[Run]:
    return [run]


@pytest.mark.parametrize(
    ("answers", "order", "warns"),
    [
        pytest.param(["ok", "ok", "ok", "ok"], [0, 1, 2, 3], True, id="no-spread"),
        pytest.param(["ok", "no", "ok", "no"], [0, 1, 2, 3], False, id="spread"),
        pytest.param(["ok", "ok", "ok", "ok"], [0, 2, 1, 3], True, id="interleaved-groups"),
    ],
)
async def test_a_batch_with_no_reward_spread_warns_that_it_trains_nothing(
    trainable: Platform,
    caplog: pytest.LogCaptureFixture,
    answers: list[str],
    order: list[int],
    warns: bool,
) -> None:
    replies = iter(answers)
    runs = await _runs(ScriptedAgent(lambda _: next(replies)), tasks=_two_tasks())

    with caplog.at_level(logging.WARNING, logger="hud.train.client"):
        await TrainingClient("owner/model").forward_backward(
            [runs[index] for index in order], group_size=2
        )

    assert ("accumulates no gradient" in caplog.text) is warns
    assert len(trainable.training[MODEL_ID].batches) == 1

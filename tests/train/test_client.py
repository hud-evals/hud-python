"""What TrainingClient sends to the HUD training service, and what it refuses to send."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any

import pytest
import torch
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud import Environment
from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Run, Task, Taskset
from hud.train import DatumTensors, TrainingClient
from hud.utils.exceptions import HudRequestError
from tests.harness import Reply, ScriptedAgent

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from tests.harness import FakeServices, HudEnv, Request

MODEL_ID = "00000000-0000-4000-8000-000000000001"
TRAIN = f"/v1/models/{MODEL_ID}/train"
OPTIM = {
    "step": 1,
    "checkpoint_id": "ckpt-1",
    "sampler_path": "s",
    "state_path": "t",
    "model": "owner/model",
}
TRACE_ID = IsStr(regex=r"[0-9a-f]{32}")


def _env() -> Environment:
    env = Environment("train")

    @env.template()
    async def answer(target: str = "ok"):
        reply = yield "Reply with ok."
        yield 1.0 if reply == target else 0.0

    return env


@pytest.fixture
def service(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The model catalog and the training service for ``owner/model``."""
    hud_env.set(HUD_API_KEY="hud-key")
    services.route(
        "api",
        "GET",
        "/v2/models",
        json={"items": [{"id": MODEL_ID, "model_name": "owner/model"}], "total": 1},
    )
    services.route("rl", "POST", f"{TRAIN}/forward-backward", json={"metrics": {}, "num_datums": 2})
    services.route("rl", "POST", f"{TRAIN}/optim-step", json=OPTIM)
    return services


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


async def test_a_step_sends_sampled_runs_inline_then_one_optimizer_step(
    service: FakeServices,
) -> None:
    _sampling_gateway(service)
    runs = await _runs(_trainable_agent())
    client = TrainingClient("owner/model", api_key="team-key")

    result = await client.step(runs, learning_rate=1e-5, group_size=2)

    assert result.checkpoint_id == "ckpt-1"
    assert [run.reward for run in runs] == [1.0, 0.0]
    (forward_backward,) = service.requests("rl", "POST", f"{TRAIN}/forward-backward")
    assert forward_backward.bearer == "team-key"
    assert forward_backward.json == snapshot(
        {
            "inputs": [
                {
                    "samples": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "output_token_ids": [7],
                            "output_logprobs": [-0.5],
                        }
                    ],
                    "reward": 1.0,
                    "trace_id": TRACE_ID,
                },
                {
                    "samples": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "output_token_ids": [7],
                            "output_logprobs": [-0.5],
                        }
                    ],
                    "reward": 0.0,
                    "trace_id": TRACE_ID,
                },
            ],
            "loss_fn": "importance_sampling",
            "group_size": 2,
            "reward_scale": 1.0,
            "num_substeps": 1,
        }
    )
    (optim,) = service.requests("rl", "POST", f"{TRAIN}/optim-step")
    assert optim.json == snapshot(
        {"learning_rate": 1e-05, "beta1": 0.9, "beta2": 0.95, "eps": 1e-08, "weight_decay": 0.0}
    )


async def test_runs_without_token_samples_train_by_trace_id(service: FakeServices) -> None:
    runs = await _runs(ScriptedAgent("ok"))

    await TrainingClient("owner/model").forward_backward(runs, group_size=2)

    (request,) = service.requests("rl", "POST", f"{TRAIN}/forward-backward")
    assert request.json["inputs"] == [str(run.trace_id) for run in runs]
    assert request.bearer == "hud-key"


async def test_the_model_slug_resolves_once_with_the_clients_own_key(
    service: FakeServices,
) -> None:
    runs = await _runs(ScriptedAgent("ok"))
    client = TrainingClient("owner/model", api_key="team-key")

    await client.forward_backward(runs, group_size=2)
    await client.optim_step(learning_rate=1e-5)

    (catalog,) = service.requests("api", "GET", "/v2/models")
    assert catalog.bearer == "team-key"
    assert [request.path for request in service.requests("rl")] == [
        f"{TRAIN}/forward-backward",
        f"{TRAIN}/optim-step",
    ]


@pytest.mark.parametrize("status", [502, 503, 504])
async def test_an_optimizer_step_is_never_retried(service: FakeServices, status: int) -> None:
    service.route("rl", "POST", f"{TRAIN}/optim-step", Reply(status=status), Reply(json=OPTIM))

    with pytest.raises(HudRequestError) as raised:
        await TrainingClient(MODEL_ID).optim_step(learning_rate=1e-5)

    assert raised.value.status_code == status
    assert len(service.requests("rl", "POST", f"{TRAIN}/optim-step")) == 1


async def test_a_custom_loss_sends_its_gradients_to_backward(service: FakeServices) -> None:
    runs = await _runs(ScriptedAgent("ok"))
    service.route(
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
    service.route("rl", "POST", f"{TRAIN}/backward", json={"metrics": {}, "num_datums": 2})

    def loss(
        data: list[DatumTensors], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = sum(-datum.reward * lp.sum() for datum, lp in zip(data, logprobs, strict=True))
        assert isinstance(total, torch.Tensor)
        return total, {"loss": total.item()}

    await TrainingClient(MODEL_ID).forward_backward_custom(runs, loss, group_size=2)

    (forward,) = service.requests("rl", "POST", f"{TRAIN}/forward")
    (backward,) = service.requests("rl", "POST", f"{TRAIN}/backward")
    assert forward.json == {
        "inputs": [str(run.trace_id) for run in runs],
        "group_size": 2,
        "reward_scale": 1.0,
    }
    assert backward.json == {
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
    service: FakeServices, runs: Any, group_size: int | None, message: str
) -> None:
    batch = await runs()

    with pytest.raises(ValueError, match=message):
        await TrainingClient("owner/model").forward_backward(batch, group_size=group_size)
    assert service.requests("rl") == []
    assert service.requests("api", "GET", "/v2/models") == []


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
    service: FakeServices,
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
    assert len(service.requests("rl")) == 1

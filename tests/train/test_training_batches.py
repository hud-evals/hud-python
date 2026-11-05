"""What a training batch sends to the HUD training service, and what it refuses to send."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from dirty_equals import IsStr

from hud import Environment, Taskset, Trace
from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.train import DatumTensors, TrainingClient
from hud.utils.exceptions import HudRequestError
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

MODEL_ID = "00000000-0000-4000-8000-000000000001"
TRAIN = f"/v1/models/{MODEL_ID}/train"
TRACE_ID = IsStr(regex=r"[0-9a-f]{32}")


@pytest.fixture
def trainer(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The training service for ``owner/model``, signed in with a HUD key."""
    hud_env.set(HUD_API_KEY="k")
    catalog = {"items": [{"id": MODEL_ID, "model_name": "owner/model"}], "total": 1}
    services.route("api", "GET", "/v2/models", json=catalog)
    for path in ("forward-backward", "backward"):
        services.route("rl", "POST", f"{TRAIN}/{path}", json={"metrics": {}, "num_datums": 2})
    services.route(
        "rl",
        "POST",
        f"{TRAIN}/optim-step",
        json={
            "step": 1,
            "checkpoint_id": "ckpt",
            "sampler_path": "s",
            "state_path": "t",
            "model": "owner/model",
        },
    )
    return services


env = Environment("train")


@env.template()
async def answer(target: str = "ok"):
    reply = yield "Reply with ok."
    yield 1.0 if reply == target else 0.0


async def answered(agent: Agent, *targets: str) -> list[Trace]:
    """Two traces per target of ``agent`` answering "Reply with ok.", graded 1 on the target."""
    tasks = [answer(target=target) for target in targets or ("ok",)]
    return (await Taskset("t", tasks).run(agent, group=2, max_concurrent=1)).traces


def sampled(text: str) -> Turn:
    """A reply from a trainable model: it carries its token ids and logprobs."""
    logprob = {"token": text, "logprob": -0.5, "bytes": [], "top_logprobs": []}
    return Turn(
        text=text,
        extra={"prompt_token_ids": [1, 2, 3], "token_ids": [7], "logprobs": {"content": [logprob]}},
    )


async def test_a_step_sends_the_sampled_runs_then_one_optimizer_step(
    trainer: FakeServices, models: Models
) -> None:
    models.queue([sampled("ok"), sampled("no")])
    agent = OpenAIChatAgent(
        OpenAIChatConfig(
            model="owner/model",
            max_steps=1,
            completion_kwargs={"extra_body": {"return_token_ids": True}},
        )
    )
    runs = await answered(agent)

    await TrainingClient("owner/model").step(runs, learning_rate=1e-5, group_size=2)

    (sent,) = trainer.bodies("rl", "POST", f"{TRAIN}/forward-backward")
    sample = {"prompt_token_ids": [1, 2, 3], "output_token_ids": [7], "output_logprobs": [-0.5]}
    assert [(i["samples"], i["reward"], i["trace_id"]) for i in sent["inputs"]] == [
        ([sample], 1.0, TRACE_ID),
        ([sample], 0.0, TRACE_ID),
    ]
    assert [s["learning_rate"] for s in trainer.bodies("rl", "POST", f"{TRAIN}/optim-step")] == [
        1e-5
    ]


async def test_runs_without_token_samples_train_by_trace_id(trainer: FakeServices) -> None:
    runs = await answered(ScriptedAgent("ok"))

    await TrainingClient("owner/model").forward_backward(runs, group_size=2)

    (batch,) = trainer.bodies("rl", "POST", f"{TRAIN}/forward-backward")
    assert batch["inputs"] == [str(run.trace_id) for run in runs]


async def test_an_optimizer_step_is_never_retried(trainer: FakeServices) -> None:
    trainer.route("rl", "POST", f"{TRAIN}/optim-step", Reply(status=503))

    with pytest.raises(HudRequestError) as raised:
        await TrainingClient(MODEL_ID).optim_step(learning_rate=1e-5)

    assert raised.value.status_code == 503
    assert len(trainer.requests("rl", "POST", f"{TRAIN}/optim-step")) == 1


async def test_a_custom_loss_sends_its_gradients_to_backward(trainer: FakeServices) -> None:
    runs = await answered(ScriptedAgent("ok"))
    trainer.route(
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

    (forward,) = trainer.bodies("rl", "POST", f"{TRAIN}/forward")
    (backward,) = trainer.bodies("rl", "POST", f"{TRAIN}/backward")
    assert forward["inputs"] == [str(run.trace_id) for run in runs]
    assert backward == {
        "forward_id": "fwd-1",
        "weights": [[1.0, 1.0], [0.5, 0.5]],
        "metrics": {"loss": 2.25},
    }


async def test_timed_out_runs_need_an_explicit_reward_and_nothing_is_sent(
    trainer: FakeServices,
) -> None:
    traces = (await answer().run(ScriptedAgent("ok", delay=5), rollout_timeout=0.2)).traces

    with pytest.raises(ValueError, match="timed-out runs require an explicit training reward"):
        await TrainingClient("owner/model").forward_backward(traces)
    assert trainer.requests("rl") == []


async def test_groups_split_across_tasks_are_refused(trainer: FakeServices) -> None:
    runs = await answered(ScriptedAgent("ok"), "ok", "no", "maybe")

    with pytest.raises(ValueError, match="incomplete GRPO groups"):
        await TrainingClient("owner/model").forward_backward(
            [runs[0], runs[2], runs[4], runs[5]], group_size=2
        )
    assert trainer.requests("rl") == []


async def test_interleaved_groups_are_refused(trainer: FakeServices) -> None:
    ok, ok_again, no, no_again = await answered(ScriptedAgent("ok"), "ok", "no")

    with pytest.raises(ValueError, match="GRPO groups must be contiguous"):
        await TrainingClient("owner/model").forward_backward(
            [ok, no, ok_again, no_again], group_size=2
        )
    assert trainer.requests("rl") == []

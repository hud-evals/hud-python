"""What only ``OpenAIChatAgent`` does over Chat Completions: where its requests go, how it
continues a sampled conversation by token ids, how it replays an incomplete call, and the
training sample it records."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from tests.conftest import steps
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "qwen3.6-plus"
TOKEN_IDS = {"extra_body": {"return_token_ids": True}}


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


async def test_the_models_reasoning_is_replayed_in_the_conversation(
    models: Models, tmp_path: Path
) -> None:
    models.script(
        [replace(call("bash", command="true"), reasoning="check first"), Turn(text="done")]
    )

    task = workspace_task(tmp_path / "ws")
    await task().run(OpenAIChatAgent(OpenAIChatConfig(model=MODEL)))

    replayed = models.requests()[-1].body["messages"][1]
    assert (replayed["role"], replayed["reasoning_content"]) == ("assistant", "check first")


async def test_a_completion_with_token_ids_records_its_training_sample(
    models: Models, tmp_path: Path
) -> None:
    logprobs = [
        {"token": "Bl", "logprob": -0.5, "bytes": None, "top_logprobs": []},
        {"token": "ue.", "logprob": -0.25, "bytes": None, "top_logprobs": []},
    ]
    sample = {"prompt_token_ids": [1, 2], "token_ids": [3, 4], "logprobs": {"content": logprobs}}
    models.script([Turn(text="Blue.", extra=sample)])

    task = workspace_task(tmp_path / "ws")
    (run,) = (
        await task().run(
            OpenAIChatAgent(OpenAIChatConfig(model=MODEL, completion_kwargs=TOKEN_IDS))
        )
    ).runs

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert step["sample"] == {
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "output_logprobs": [-0.5, -0.25],
    }

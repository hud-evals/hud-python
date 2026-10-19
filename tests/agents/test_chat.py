"""What only ``OpenAIChatAgent`` sends and records: its completion options on every Chat
Completions request (token ids, checkpoints), how it replays a cut-off call, and the
training sample it exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, call, say, steps

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Models


def chat(**config: Any) -> OpenAIChatAgent:
    return OpenAIChatAgent(OpenAIChatConfig.model_validate({"model": "qwen3.6-plus", **config}))


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


BASE = {"model": "qwen3.6-plus", "stream": False}
TOKEN_IDS = {"extra_body": {"return_token_ids": True}}
SAMPLED = BASE | {"logprobs": True, "return_token_ids": True}
CALL = call("bash", command="true").tool_calls


@pytest.mark.parametrize(
    ("config", "turns", "options"),
    [
        pytest.param(
            {"completion_kwargs": TOKEN_IDS},
            [
                Turn(tool_calls=CALL, extra={"prompt_token_ids": [1, 2], "token_ids": [3]}),
                Turn(text="done", extra={"prompt_token_ids": [1, 2, 3, 4], "token_ids": [5]}),
            ],
            [SAMPLED, SAMPLED | {"prompt_token_ids": [1, 2, 3], "continuation_from": 2}],
            id="token-ids-continue-the-kv-cache",
        ),
        pytest.param(
            {"completion_kwargs": TOKEN_IDS},
            [
                Turn(
                    tool_calls=CALL,
                    extra={"prompt_chunks": [{"type": "text", "text": "hi"}], "token_ids": [3]},
                ),
                say("done"),
            ],
            [SAMPLED, SAMPLED],
            id="multimodal-prompt-chunks-reset-the-continuation",
        ),
        pytest.param(
            {"checkpoint": "ckpt-7"},
            [say("done")],
            [BASE | {"checkpoint": "ckpt-7"}],
            id="a-checkpoint-routes-in-the-body",
        ),
        pytest.param(
            {
                "completion_kwargs": {
                    "model": "other",
                    "stream": True,
                    "tools": [],
                    "messages": [],
                    "max_tokens": 50,
                }
            },
            [say("done")],
            [BASE | {"max_tokens": 50}],
            id="reserved-completion-kwargs-are-dropped",
        ),
    ],
)
async def test_every_request_carries_the_completion_options_its_configuration_sets(
    config: dict[str, Any],
    turns: list[Turn],
    options: list[dict[str, Any]],
    models: Models,
    tmp_path: Path,
) -> None:
    models.script(turns)

    await run_task(workspace_env(tmp_path / "ws"), chat(**config))

    sent = [
        {key: value for key, value in request.body.items() if key not in {"messages", "tools"}}
        for request in models.requests()
    ]
    assert sent == options


async def test_a_cut_off_call_is_replayed_to_the_model_with_empty_arguments(
    models: Models, tmp_path: Path
) -> None:
    cut_off = Turn(
        native=(
            {
                "id": "call_cut",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ec'},
            },
        )
    )
    models.script([cut_off, say("done")])

    await run_task(workspace_env(tmp_path / "ws"), chat())

    replayed = models.requests()[-1].body["messages"][1]["tool_calls"][0]
    assert replayed["function"]["arguments"] == "{}"


async def test_a_completion_with_token_ids_exports_its_training_sample(
    models: Models, tmp_path: Path
) -> None:
    logprobs = [
        {"token": "Bl", "logprob": -0.5, "bytes": None, "top_logprobs": []},
        {"token": "ue.", "logprob": -0.25, "bytes": None, "top_logprobs": []},
    ]
    models.script(
        [
            Turn(
                text="Blue.",
                extra={
                    "prompt_token_ids": [1, 2],
                    "token_ids": [3, 4],
                    "logprobs": {"content": logprobs},
                },
            )
        ]
    )

    run = await run_task(workspace_env(tmp_path / "ws"), chat(completion_kwargs=TOKEN_IDS))

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert step["sample"] == {
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "output_logprobs": [-0.5, -0.25],
    }

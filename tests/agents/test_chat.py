"""What only ``OpenAIChatAgent`` does over Chat Completions: where its requests go, how it
continues a sampled conversation by token ids, how it replays an incomplete call, and the
training sample it records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import LocalRuntime, Task, rollout
from tests.conftest import steps
from tests.fixtures.environments import workspace_env
from tests.fixtures.models import Turn

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "qwen3.6-plus"
TOKEN_IDS = {"extra_body": {"return_token_ids": True}}


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@pytest.mark.parametrize(
    ("openai_key", "explicit", "route", "credential"),
    [
        (None, True, "provider", "explicit"),
        (None, False, "gateway", "hud-key"),
        ("openai-key", False, "gateway", "hud-key"),
    ],
    ids=["an-explicit-endpoint", "only-a-hud-key", "an-openai-key-is-not-its-own"],
)
async def test_requests_go_to_an_explicit_endpoint_and_otherwise_to_the_gateway(
    openai_key: str | None,
    explicit: bool,
    route: str,
    credential: str,
    models: Models,
    provider: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(OPENAI_API_KEY=openai_key)
    models.script([Turn(text="done")])
    provider.script([Turn(text="done")])
    endpoint = {"api_key": "explicit", "base_url": provider.url} if explicit else {}

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL, **endpoint)),
        runtime=LocalRuntime(env),
    )

    (request,) = {"gateway": models, "provider": provider}[route].requests()
    assert (run.trace.status, request.credential) == ("completed", credential)


@pytest.mark.parametrize(
    ("sampled", "continued"),
    [
        (
            [{"prompt_token_ids": [1, 2], "token_ids": [3]}, {"token_ids": [5]}],
            [{}, {"prompt_token_ids": [1, 2, 3], "continuation_from": 2}],
        ),
        (
            [
                {"prompt_token_ids": [1, 2], "token_ids": [3]},
                {"prompt_chunks": [{"type": "text", "text": "hi"}], "token_ids": [4]},
                {"token_ids": [5]},
            ],
            [{}, {"prompt_token_ids": [1, 2, 3], "continuation_from": 2}, {}],
        ),
    ],
    ids=["text-prompts-continue-the-kv-cache", "a-multimodal-prompt-starts-over"],
)
async def test_token_ids_continue_the_conversation_from_what_was_sampled(
    sampled: list[dict[str, Any]],
    continued: list[dict[str, Any]],
    models: Models,
    tmp_path: Path,
) -> None:
    *calls, last = sampled
    models.script(
        [Turn(shell="true", extra=extra) for extra in calls] + [Turn(text="done", extra=last)]
    )

    env = workspace_env(tmp_path / "ws")
    await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL, completion_kwargs=TOKEN_IDS)),
        runtime=LocalRuntime(env),
    )

    keys = {"prompt_token_ids", "continuation_from"}
    sent = [{k: v for k, v in r.body.items() if k in keys} for r in models.requests()]
    assert sent == continued


async def test_the_models_reasoning_is_replayed_in_the_conversation(
    models: Models, tmp_path: Path
) -> None:
    models.script([Turn(shell="true", reasoning="check first"), Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    replayed = models.requests()[-1].body["messages"][1]
    assert (replayed["role"], replayed["reasoning_content"]) == ("assistant", "check first")


async def test_completion_kwargs_cannot_override_what_the_agent_sends_itself(
    models: Models, tmp_path: Path
) -> None:
    reserved = {"model": "other", "stream": True, "tools": [], "messages": [], "max_tokens": 50}
    models.script([Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL, completion_kwargs=reserved)),
        runtime=LocalRuntime(env),
    )

    (request,) = models.requests()
    assert (request.body["model"], request.body["stream"], request.body["max_tokens"]) == (
        MODEL,
        False,
        50,
    )
    assert request.body["messages"] != [] and request.body["tools"] != []


async def test_an_incomplete_call_is_explained_and_replayed_with_empty_arguments(
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
    models.script([cut_off, Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    last = models.requests()[-1].body
    assert run.trace.status == "completed"
    assert last["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "the arguments for this 'bash' call arrived incomplete" in json.dumps(last)


async def test_a_completion_with_token_ids_records_its_training_sample(
    models: Models, tmp_path: Path
) -> None:
    logprobs = [
        {"token": "Bl", "logprob": -0.5, "bytes": None, "top_logprobs": []},
        {"token": "ue.", "logprob": -0.25, "bytes": None, "top_logprobs": []},
    ]
    sample = {"prompt_token_ids": [1, 2], "token_ids": [3, 4], "logprobs": {"content": logprobs}}
    models.script([Turn(text="Blue.", extra=sample)])

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        OpenAIChatAgent(OpenAIChatConfig(model=MODEL, completion_kwargs=TOKEN_IDS)),
        runtime=LocalRuntime(env),
    )

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert step["sample"] == {
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "output_logprobs": [-0.5, -0.25],
    }

"""OpenAI-compatible chat agent tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion

from hud.agents.base import AgentContext
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.openai_compatible.agent import OpenAIChatAgentState
from hud.agents.openai_compatible.tools import OpenAICompatibleAgentTools
from hud.agents.tests.conftest import (
    RecordingToolEnvironment,
    mcp_tool,
    text_prompt,
    text_result,
)


def _chat_completion(message: dict[str, Any], *, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
        }
    )


def _client(*responses: ChatCompletion) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=list(responses)))
        )
    )


def provider_state(messages: list[Any] | None = None) -> OpenAIChatAgentState:
    return OpenAIChatAgentState.model_construct(
        messages=[] if messages is None else messages,
        tools=OpenAICompatibleAgentTools(),
    )


def _chat_completion_with_token_ids(
    message: dict[str, Any],
    *,
    prompt_token_ids: list[int],
    token_ids: list[int],
) -> ChatCompletion:
    completion = _chat_completion(message)
    choice = completion.choices[0]
    object.__setattr__(choice, "prompt_token_ids", prompt_token_ids)
    object.__setattr__(choice, "token_ids", token_ids)
    return completion


@pytest.mark.asyncio
async def test_openai_compatible_run_executes_model_tool_call_and_returns_final_answer() -> None:
    client = _client(
        _chat_completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"hud"}',
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        ),
        _chat_completion({"role": "assistant", "content": "final answer"}),
    )
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("tool result")},
    )
    agent = OpenAIChatAgent.create(model="test-model", openai_client=client)

    result = await agent.run(
        AgentContext(prompt=[text_prompt("answer with lookup")], tool_client=environment.client)
    )

    assert result.content == "final answer"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "hud"})
    ]
    assert client.chat.completions.create.await_count == 2
    second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool result",
    } in second_messages


@pytest.mark.asyncio
async def test_openai_compatible_auto_respond_followup_does_not_repeat_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def continue_once(content: str | None, *, enabled: bool) -> object:
        assert enabled is True
        if content == "need input":
            return text_prompt("continue")
        return None

    monkeypatch.setattr("hud.agents.base.auto_respond", continue_once)
    client = _client(
        _chat_completion({"role": "assistant", "content": "need input"}),
        _chat_completion({"role": "assistant", "content": "final answer"}),
    )
    agent = OpenAIChatAgent.create(
        model="test-model",
        openai_client=client,
        system_prompt="system rules",
        auto_respond=True,
    )

    result = await agent.run(AgentContext(prompt=[text_prompt("start")]))

    assert result.content == "final answer"
    second_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    system_messages = [message for message in second_messages if message["role"] == "system"]
    assert system_messages == [{"role": "system", "content": "system rules"}]


@pytest.mark.asyncio
async def test_openai_compatible_preserves_reasoning_fields_on_assistant_message() -> None:
    reasoning_details = [{"type": "reasoning.text", "text": "step"}]
    client = _client(
        _chat_completion(
            {
                "role": "assistant",
                "content": "answer",
                "reasoning": "private reasoning",
                "reasoning_details": reasoning_details,
            }
        )
    )
    agent = OpenAIChatAgent.create(model="reasoning-model", openai_client=client)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "question"}]

    result = await agent.get_response(provider_state(cast("list[Any]", messages)))

    assert result.content == "answer"
    assert result.reasoning == "private reasoning"
    assert messages[-1]["reasoning"] == "private reasoning"
    assert messages[-1]["reasoning_details"] == reasoning_details


@pytest.mark.asyncio
async def test_openai_compatible_api_error_returns_error_response() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
        )
    )
    agent = OpenAIChatAgent.create(model="test-model", openai_client=client)

    response = await agent.get_response(
        provider_state(cast("list[Any]", [{"role": "user", "content": "question"}]))
    )

    assert response.done is True
    assert response.isError is True
    assert response.content == "Error getting response boom"


@pytest.mark.asyncio
async def test_openai_compatible_checkpoint_is_sent_in_provider_body() -> None:
    client = _client(_chat_completion({"role": "assistant", "content": "answer"}))
    agent = OpenAIChatAgent.create(
        model="test-model",
        openai_client=client,
        checkpoint="checkpoint-123",
    )

    response = await agent.get_response(
        provider_state(cast("list[Any]", [{"role": "user", "content": "question"}]))
    )

    assert response.content == "answer"
    assert client.chat.completions.create.await_args.kwargs["extra_body"] == {
        "checkpoint": "checkpoint-123"
    }


@pytest.mark.asyncio
async def test_openai_compatible_token_continuation_is_sent_after_first_response() -> None:
    client = _client(
        _chat_completion_with_token_ids(
            {"role": "assistant", "content": "first"},
            prompt_token_ids=[1, 2],
            token_ids=[3],
        ),
        _chat_completion({"role": "assistant", "content": "second"}),
    )
    agent = OpenAIChatAgent.create(
        model="test-model",
        openai_client=client,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )
    messages = cast("Any", [{"role": "user", "content": "question"}])
    state = provider_state(cast("list[Any]", messages))

    first = await agent.get_response(state)
    second = await agent.get_response(state)

    assert first.content == "first"
    assert second.content == "second"
    second_body = client.chat.completions.create.await_args_list[1].kwargs["extra_body"]
    assert second_body == {
        "return_token_ids": True,
        "prompt_token_ids": [1, 2, 3],
        "continuation_from": 2,
    }


@pytest.mark.asyncio
async def test_openai_compatible_run_resets_token_continuation_between_runs() -> None:
    client = _client(
        _chat_completion_with_token_ids(
            {"role": "assistant", "content": "first"},
            prompt_token_ids=[1, 2],
            token_ids=[3],
        ),
        _chat_completion({"role": "assistant", "content": "second"}),
    )
    agent = OpenAIChatAgent.create(
        model="test-model",
        openai_client=client,
        completion_kwargs={"extra_body": {"return_token_ids": True}},
    )

    first = await agent.run(AgentContext(prompt=[text_prompt("first")]))
    second = await agent.run(AgentContext(prompt=[text_prompt("second")]))

    assert first.content == "first"
    assert second.content == "second"
    second_body = client.chat.completions.create.await_args_list[1].kwargs["extra_body"]
    assert second_body == {"return_token_ids": True}

"""Gemini agent tests."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types as genai_types

from hud.agents.base import AgentContext
from hud.agents.gemini import GeminiAgent
from hud.agents.gemini.agent import GeminiAgentState
from hud.agents.gemini.tools import GeminiAgentTools
from hud.agents.tests.conftest import (
    RecordingToolEnvironment,
    mcp_tool,
    text_prompt,
    text_result,
)


def _gemini_response(*parts: genai_types.Part) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=list(parts),
                )
            )
        ]
    )


def _gemini_client(*responses: genai_types.GenerateContentResponse) -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=list(responses))
    return client


def provider_state(messages: list[Any] | None = None) -> GeminiAgentState:
    return GeminiAgentState.model_construct(
        messages=[] if messages is None else messages,
        tools=GeminiAgentTools(),
    )


@pytest.mark.asyncio
async def test_gemini_run_executes_model_tool_call_and_returns_final_answer() -> None:
    client = _gemini_client(
        _gemini_response(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name="lookup",
                    args={"query": "hud"},
                )
            )
        ),
        _gemini_response(genai_types.Part(text="final answer")),
    )
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("tool result")},
    )
    agent = GeminiAgent.create(model_client=client, validate_api_key=False)

    result = await agent.run(
        AgentContext(prompt=[text_prompt("answer with lookup")], tool_client=environment.client)
    )

    assert result.content == "final answer"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "hud"})
    ]
    assert client.aio.models.generate_content.await_count == 2
    second_contents = cast(
        "list[genai_types.Content]",
        client.aio.models.generate_content.await_args_list[1].kwargs["contents"],
    )
    function_response_names: list[str] = []
    for content in second_contents:
        for part in content.parts or []:
            function_response = part.function_response
            if function_response is not None:
                function_response_names.append(function_response.name or "")
    assert "lookup" in function_response_names


@pytest.mark.asyncio
async def test_gemini_no_candidates_is_a_user_visible_error() -> None:
    client = _gemini_client(genai_types.GenerateContentResponse(candidates=[]))
    agent = GeminiAgent.create(model_client=client, validate_api_key=False)

    with pytest.raises(RuntimeError, match="returned no candidates"):
        await agent.get_response(provider_state())


@pytest.mark.asyncio
async def test_gemini_citations_enable_google_search_at_provider_boundary() -> None:
    client = _gemini_client(_gemini_response(genai_types.Part(text="answer")))
    agent = GeminiAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response(provider_state(), citations_enabled=True)

    assert response.content == "answer"
    config = client.aio.models.generate_content.await_args.kwargs["config"]
    assert any(tool.google_search is not None for tool in config.tools)


@pytest.mark.asyncio
async def test_gemini_preserves_thought_parts_as_reasoning() -> None:
    client = _gemini_client(
        _gemini_response(
            genai_types.Part(text="private reasoning", thought=True),
            genai_types.Part(text="answer"),
        )
    )
    agent = GeminiAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response(provider_state())

    assert response.content == "answer"
    assert response.reasoning == "private reasoning"


@pytest.mark.asyncio
async def test_gemini_prunes_older_computer_screenshots_before_request() -> None:
    def computer_response(name: str) -> genai_types.FunctionResponse:
        return genai_types.FunctionResponse(
            name=name,
            response={"success": True},
            parts=[
                genai_types.FunctionResponsePart(
                    inline_data=genai_types.FunctionResponseBlob(
                        mime_type="image/png",
                        data=b"image-bytes",
                    )
                )
            ],
        )

    old_response = computer_response("click_at")
    recent_response = computer_response("navigate")
    messages = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(function_response=old_response)],
        ),
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(function_response=recent_response)],
        ),
    ]
    client = _gemini_client(_gemini_response(genai_types.Part(text="answer")))
    agent = GeminiAgent.create(model_client=client, validate_api_key=False)
    agent.max_recent_turn_with_screenshots = 1

    response = await agent.get_response(provider_state(cast("list[Any]", messages)))

    assert response.content == "answer"
    assert old_response.parts is None
    assert recent_response.parts is not None
    requested_contents = client.aio.models.generate_content.await_args.kwargs["contents"]
    assert requested_contents is messages

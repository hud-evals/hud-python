"""OpenAI Responses agent tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Summary

from hud.agents.base import AgentContext
from hud.agents.openai import OpenAIAgent
from hud.agents.tests.conftest import RecordingToolEnvironment, mcp_tool, text_prompt, text_result


def _message_response(text: str, *, response_id: str = "resp_final") -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        output=[
            ResponseOutputMessage(
                id=f"msg_{response_id}",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
    )


@pytest.mark.asyncio
async def test_openai_run_executes_model_tool_call_and_returns_final_answer() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        id="resp_tool",
                        output=[
                            ResponseFunctionToolCall(
                                id="item_1",
                                type="function_call",
                                call_id="call_1",
                                name="lookup",
                                arguments='{"query":"hud"}',
                            )
                        ],
                    ),
                    _message_response("final answer"),
                ]
            )
        )
    )
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("tool result")},
    )
    agent = OpenAIAgent.create(model_client=client, validate_api_key=False)

    result = await agent.run(
        AgentContext(messages=[text_prompt("answer with lookup")], tool_client=environment.client)
    )

    assert result.content == "final answer"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "hud"})
    ]
    assert client.responses.create.await_count == 2
    second_input = client.responses.create.await_args_list[1].kwargs["input"]
    assert client.responses.create.await_args_list[1].kwargs["previous_response_id"] == "resp_tool"
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call_1"


@pytest.mark.asyncio
async def test_openai_get_response_preserves_reasoning_and_citations() -> None:
    text = ResponseOutputText.model_validate(
        {
            "type": "output_text",
            "text": "Example",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 7,
                }
            ],
        }
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    id="resp",
                    output=[
                        ResponseReasoningItem(
                            id="reason",
                            type="reasoning",
                            summary=[Summary(type="summary_text", text="thought")],
                        ),
                        ResponseOutputMessage(
                            id="msg",
                            type="message",
                            role="assistant",
                            status="completed",
                            content=[text],
                        ),
                    ],
                )
            )
        )
    )
    agent = OpenAIAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response([])

    assert response.content == "Example"
    assert response.reasoning == "thought"
    assert response.citations == [
        {
            "type": "url_citation",
            "text": "Example",
            "source": "https://example.com",
            "title": "Example",
            "start_index": 0,
            "end_index": 7,
        }
    ]


@pytest.mark.asyncio
async def test_openai_citation_mode_requests_provider_source_metadata() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_message_response("answer")))
    )
    agent = OpenAIAgent.create(model_client=client, validate_api_key=False)
    agent.enable_citations = True

    response = await agent.get_response([])

    assert response.content == "answer"
    assert client.responses.create.await_args.kwargs["include"] == [
        "web_search_call.action.sources"
    ]


@pytest.mark.asyncio
async def test_openai_get_response_parses_native_computer_and_shell_calls() -> None:
    def _action(payload: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(to_dict=lambda: payload)

    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    id="resp",
                    output=[
                        SimpleNamespace(
                            type="computer_call",
                            call_id="computer_call_1",
                            actions=[_action({"type": "click", "x": 1, "y": 2})],
                            action=None,
                            pending_safety_checks=[],
                        ),
                        SimpleNamespace(
                            type="shell_call",
                            call_id="shell_call_1",
                            action=_action({"commands": ["pwd"]}),
                        ),
                    ],
                )
            )
        )
    )
    agent = OpenAIAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response([])

    assert response.done is False
    assert [(call.name, call.arguments, call.id) for call in response.tool_calls] == [
        ("computer", {"actions": [{"type": "click", "x": 1, "y": 2}]}, "computer_call_1"),
        ("shell", {"commands": ["pwd"]}, "shell_call_1"),
    ]


@pytest.mark.asyncio
async def test_openai_run_returns_error_trace_for_provider_failure() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("provider down")))
    )
    agent = OpenAIAgent.create(model_client=client, validate_api_key=False)

    result = await agent.run(AgentContext(messages=[text_prompt("hello")]))

    assert result.isError is True
    assert result.content == "provider down"
    assert result.info["error"] == "provider down"

"""Claude agent tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from hud.agents.base import AgentContext
from hud.agents.claude import ClaudeAgent
from hud.agents.tests.conftest import RecordingToolEnvironment, mcp_tool, text_prompt, text_result


class Stream:
    def __init__(self, response: MagicMock) -> None:
        self.response = response

    async def __aenter__(self) -> Stream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        return False

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> None:
        raise StopAsyncIteration

    async def get_final_message(self) -> MagicMock:
        return self.response


class ErrorStream:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> ErrorStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        return False

    def __aiter__(self) -> ErrorStream:
        return self

    async def __anext__(self) -> None:
        raise self.error


def _tool_use(name: str, arguments: dict[str, object]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = "call_1"
    block.name = name
    block.input = arguments
    return block


def _text_block(text: str, *, thinking: bool = False) -> MagicMock:
    block = MagicMock()
    block.type = "thinking" if thinking else "text"
    block.text = text
    block.thinking = text
    block.citations = None
    return block


def _message(*blocks: MagicMock) -> MagicMock:
    response = MagicMock()
    response.content = list(blocks)
    return response


@pytest.mark.asyncio
async def test_claude_run_executes_model_tool_call_and_returns_final_answer() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(
                    side_effect=[
                        Stream(_message(_tool_use("lookup", {"query": "hud"}))),
                        Stream(_message(_text_block("final answer"))),
                    ]
                )
            )
        )
    )
    environment = RecordingToolEnvironment(
        [mcp_tool("lookup")],
        results={"lookup": text_result("tool result")},
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    result = await agent.run(
        AgentContext(messages=[text_prompt("answer with lookup")], tool_client=environment.client)
    )

    assert result.content == "final answer"
    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("lookup", {"query": "hud"})
    ]
    assert client.beta.messages.stream.call_count == 2
    second_messages = client.beta.messages.stream.call_args_list[1].kwargs["messages"]
    assert second_messages[-1]["role"] == "user"
    assert second_messages[-1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_claude_retries_streamed_invalid_tool_json_once() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(
                    side_effect=[
                        ErrorStream(
                            ValueError("Unable to parse tool parameter JSON from model. JSON: {bad")
                        ),
                        Stream(_message(_text_block("ok"))),
                    ]
                )
            )
        )
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )

    assert response.content == "ok"
    assert response.done is True
    assert client.beta.messages.stream.call_count == 2


@pytest.mark.asyncio
async def test_claude_second_invalid_json_retry_adds_guidance_message() -> None:
    invalid_json_error = ValueError("Unable to parse tool parameter JSON from model. JSON: {bad")
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(
                    side_effect=[
                        ErrorStream(invalid_json_error),
                        ErrorStream(invalid_json_error),
                        Stream(_message(_text_block("ok"))),
                    ]
                )
            )
        )
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    response = await agent.get_response(cast("Any", messages))

    assert response.content == "ok"
    assert client.beta.messages.stream.call_count == 3
    retry_messages = client.beta.messages.stream.call_args_list[2].kwargs["messages"]
    retry_text = retry_messages[-1]["content"][0]["text"]
    assert "INVALID_JSON" in retry_text
    assert "Retry the same intended tool call" in retry_text


@pytest.mark.asyncio
async def test_claude_response_preserves_thinking_as_reasoning() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(
                    return_value=Stream(
                        _message(_text_block("answer"), _text_block("plan", thinking=True))
                    )
                )
            )
        )
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )

    assert response.content == "answer"
    assert response.reasoning == "plan"


@pytest.mark.asyncio
async def test_claude_extracts_document_citations_from_text_blocks() -> None:
    citation = MagicMock()
    citation.type = "char_location"
    citation.cited_text = "Revenue"
    citation.document_index = 0
    citation.document_title = "financials.pdf"
    citation.start_char_index = 0
    citation.end_char_index = 7
    text_block = _text_block("Revenue")
    text_block.citations = [citation]
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(stream=MagicMock(return_value=Stream(_message(text_block))))
        )
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    response = await agent.get_response(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )

    assert response.citations == [
        {
            "type": "document_citation",
            "text": "Revenue",
            "source": "0",
            "title": "financials.pdf",
            "start_index": 0,
            "end_index": 7,
        }
    ]


@pytest.mark.asyncio
async def test_claude_native_computer_requests_required_beta_header() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(return_value=Stream(_message(_text_block("answer"))))
            )
        )
    )
    agent = ClaudeAgent.create(
        model="claude-sonnet-4-6",
        model_client=client,
        validate_api_key=False,
    )
    agent.tools.prepare(model=agent.config.model, tools=[mcp_tool("computer")])

    response = await agent.get_response(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )

    assert response.content == "answer"
    kwargs = client.beta.messages.stream.call_args.kwargs
    assert "computer-use-2025-11-24" in kwargs["betas"]
    assert kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

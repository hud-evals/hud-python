"""Claude agent tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import mcp.types as mcp_types
import pytest

from hud.agents.base import AgentContext
from hud.agents.claude import ClaudeAgent
from hud.agents.claude.agent import ClaudeAgentState
from hud.agents.claude.tools import ClaudeAgentTools
from hud.agents.tests.conftest import (
    RecordingToolEnvironment,
    mcp_tool,
    text_prompt,
    text_result,
)


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


def provider_state(messages: list[Any] | None = None) -> ClaudeAgentState:
    return ClaudeAgentState.model_construct(
        messages=[] if messages is None else messages,
        tools=ClaudeAgentTools(),
    )


def _user_state() -> ClaudeAgentState:
    return provider_state([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])


@pytest.mark.asyncio
async def test_claude_formats_pdf_prompt_message() -> None:
    agent = ClaudeAgent.create(model_client=MagicMock(), validate_api_key=False)

    state = await agent.initialize_state(
        [
            mcp_types.PromptMessage(
                role="user",
                content=mcp_types.EmbeddedResource(
                    type="resource",
                    resource=mcp_types.BlobResourceContents.model_validate(
                        {
                            "uri": "file:///tmp/financials.pdf",
                            "mimeType": "application/pdf",
                            "blob": "JVBERi0=",
                        }
                    ),
                ),
            )
        ]
    )

    message = cast("dict[str, Any]", state.messages[0])
    content_blocks = cast("list[dict[str, Any]]", message["content"])
    content = content_blocks[0]
    assert content == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0=",
        },
    }


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
        AgentContext(prompt=[text_prompt("answer with lookup")], tool_client=environment.client)
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

    response = await agent.get_response(_user_state())

    assert response.content == "ok"
    assert response.done is True
    assert client.beta.messages.stream.call_count == 2


@pytest.mark.asyncio
async def test_claude_does_not_retry_unrelated_value_errors() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                stream=MagicMock(side_effect=[ErrorStream(ValueError("provider failed"))])
            )
        )
    )
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    with pytest.raises(ValueError, match="provider failed"):
        await agent.get_response(_user_state())

    assert client.beta.messages.stream.call_count == 1


@pytest.mark.asyncio
async def test_claude_bedrock_does_not_retry_invalid_tool_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BedrockClient:
        def __init__(self) -> None:
            self.beta = SimpleNamespace(
                messages=SimpleNamespace(
                    create=AsyncMock(
                        side_effect=ValueError(
                            "Unable to parse tool parameter JSON from model. JSON: {bad"
                        )
                    )
                )
            )

    client = BedrockClient()
    monkeypatch.setattr("hud.agents.claude.agent.AsyncAnthropicBedrock", BedrockClient)
    agent = ClaudeAgent.create(model_client=client, validate_api_key=False)

    with pytest.raises(ValueError, match="Unable to parse tool parameter JSON"):
        await agent.get_response(_user_state())

    assert client.beta.messages.create.await_count == 1


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

    response = await agent.get_response(provider_state(cast("list[Any]", messages)))

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

    response = await agent.get_response(_user_state())

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

    response = await agent.get_response(_user_state())

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
    state = _user_state()
    state.tools.prepare(
        model=agent.config.model,
        tools=[mcp_tool("computer", meta={"capability": "computer"})],
    )

    response = await agent.get_response(state)

    assert response.content == "answer"
    kwargs = client.beta.messages.stream.call_args.kwargs
    assert "computer-use-2025-11-24" in kwargs["betas"]
    assert kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

"""Provider-hosted tool configuration tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types as genai_types
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from hud.agents.base import AgentContext
from hud.agents.claude import (
    ClaudeAgent,
    ClaudeToolSearchTool,
    ClaudeWebFetchTool,
    ClaudeWebSearchTool,
)
from hud.agents.gemini import GeminiAgent, GeminiCodeExecutionTool, GeminiGoogleSearchTool
from hud.agents.openai import OpenAIAgent, OpenAICodeInterpreterTool, OpenAIToolSearchTool
from hud.agents.tests.conftest import RecordingToolEnvironment, mcp_tool, text_prompt


def _message_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp",
        output=[
            ResponseOutputMessage(
                id="msg",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
    )


class Stream:
    def __init__(self, text: str) -> None:
        block = MagicMock()
        block.type = "text"
        block.text = text
        block.citations = None
        self.response = MagicMock()
        self.response.content = [block]

    async def __aenter__(self) -> Stream:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> None:
        raise StopAsyncIteration

    async def get_final_message(self) -> MagicMock:
        return self.response


def _gemini_response(text: str) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
            )
        ]
    )


def _gemini_client(response: genai_types.GenerateContentResponse) -> MagicMock:
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


def test_openai_hosted_tools_are_model_gated() -> None:
    tool = OpenAICodeInterpreterTool(container={"type": "auto"})

    assert tool.supports_model("gpt-5.4")
    assert not tool.supports_model("gpt-4.1")


@pytest.mark.asyncio
async def test_supported_openai_hosted_tool_is_sent_to_provider() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_message_response("done")))
    )
    agent = OpenAIAgent.create(
        model="gpt-5.4",
        model_client=client,
        validate_api_key=False,
        hosted_tools=[OpenAICodeInterpreterTool(container={"type": "auto"})],
    )

    result = await agent.run(AgentContext(prompt=[text_prompt("use hosted code")]))

    assert result.content == "done"
    tools = client.responses.create.await_args.kwargs["tools"]
    assert any(tool["type"] == "code_interpreter" for tool in tools)


@pytest.mark.asyncio
async def test_unsupported_openai_hosted_tool_is_not_sent_to_provider() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_message_response("done")))
    )
    agent = OpenAIAgent.create(
        model="gpt-4.1",
        model_client=client,
        validate_api_key=False,
        hosted_tools=[OpenAICodeInterpreterTool(container={"type": "auto"})],
    )

    result = await agent.run(AgentContext(prompt=[text_prompt("use hosted code")]))

    assert result.content == "done"
    tools = client.responses.create.await_args.kwargs["tools"]
    assert not isinstance(tools, list)


def test_claude_hosted_domain_filters_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either allowed_domains or blocked_domains"):
        ClaudeWebSearchTool(
            allowed_domains=["example.com"],
            blocked_domains=["blocked.example"],
        ).to_params()

    with pytest.raises(ValueError, match="either allowed_domains or blocked_domains"):
        ClaudeWebFetchTool(
            allowed_domains=["example.com"],
            blocked_domains=["blocked.example"],
        ).to_params()


def test_gemini_google_search_rejects_unsupported_dynamic_threshold() -> None:
    with pytest.raises(ValueError, match="dynamic_threshold"):
        GeminiGoogleSearchTool(dynamic_threshold=0.2).to_params()


@pytest.mark.asyncio
async def test_openai_tool_search_threshold_defers_function_loading() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_message_response("done")))
    )
    agent = OpenAIAgent.create(
        model="gpt-5.4",
        model_client=client,
        validate_api_key=False,
        hosted_tools=[OpenAIToolSearchTool(threshold=1)],
    )
    environment = RecordingToolEnvironment([mcp_tool("first"), mcp_tool("second")])

    result = await agent.run(
        AgentContext(
            prompt=[text_prompt("use tools")],
            tool_client=environment.client,
        )
    )

    assert result.content == "done"
    tools = client.responses.create.await_args.kwargs["tools"]
    function_tools = [tool for tool in tools if tool["type"] == "function"]
    assert len(function_tools) == 2
    assert all(tool["defer_loading"] is True for tool in function_tools)


@pytest.mark.asyncio
async def test_claude_hosted_web_fetch_payload_is_sent_to_provider() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(stream=MagicMock(return_value=Stream("done")))
        )
    )
    agent = ClaudeAgent.create(
        model="claude-sonnet-4-6",
        model_client=client,
        validate_api_key=False,
        hosted_tools=[
            ClaudeWebFetchTool(
                max_uses=2,
                allowed_domains=["example.com"],
                max_content_tokens=500,
                citations_enabled=True,
            )
        ],
    )

    result = await agent.run(AgentContext(prompt=[text_prompt("fetch")]))

    assert result.content == "done"
    tools = client.beta.messages.stream.call_args.kwargs["tools"]
    assert tools == [
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 2,
            "allowed_domains": ["example.com"],
            "max_content_tokens": 500,
            "citations": {"enabled": True},
        }
    ]


@pytest.mark.asyncio
async def test_claude_tool_search_threshold_defers_generic_tools() -> None:
    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(stream=MagicMock(return_value=Stream("done")))
        )
    )
    agent = ClaudeAgent.create(
        model="claude-sonnet-4-6",
        model_client=client,
        validate_api_key=False,
        hosted_tools=[ClaudeToolSearchTool(threshold=1)],
    )

    result = await agent.run(
        AgentContext(
            prompt=[text_prompt("use tools")],
            tool_client=RecordingToolEnvironment([mcp_tool("first"), mcp_tool("second")]).client,
        )
    )

    assert result.content == "done"
    tools = client.beta.messages.stream.call_args.kwargs["tools"]
    generic_tools = [tool for tool in tools if "input_schema" in tool]
    assert len(generic_tools) == 2
    assert all(tool["defer_loading"] is True for tool in generic_tools)
    assert any(tool["type"] == "tool_search_tool_bm25_20251119" for tool in tools)


@pytest.mark.asyncio
async def test_gemini_hosted_code_execution_payload_is_sent_to_provider() -> None:
    client = _gemini_client(_gemini_response("done"))
    agent = GeminiAgent.create(
        model_client=client,
        validate_api_key=False,
        hosted_tools=[GeminiCodeExecutionTool()],
    )

    result = await agent.run(AgentContext(prompt=[text_prompt("run code")]))

    assert result.content == "done"
    config = client.aio.models.generate_content.await_args.kwargs["config"]
    assert len(config.tools) == 1
    assert config.tools[0].code_execution is not None

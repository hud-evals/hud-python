from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

import mcp.types as types

from hud.agents.openrouter import OpenRouterAgent
from hud.settings import settings
from hud.types import MCPToolCall, MCPToolResult


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable HUD telemetry during unit tests."""
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_openrouter_agent_builds_cached_messages() -> None:
    responses_create = AsyncMock(
        return_value=FakeResponse({"output": [{"type": "message", "content": []}], "status": "completed"})
    )
    mock_client = MagicMock()
    mock_client.responses.create = responses_create

    agent = OpenRouterAgent(
        api_key="test-key",
        openai_client=mock_client,
        cache_control={"type": "ephemeral"},
    )
    agent._available_tools = []  # mimic initialized agent

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": "Previous reply"},
    ]

    await agent.get_response(messages)

    await_call = responses_create.await_args
    assert await_call is not None
    kwargs = await_call.kwargs
    assert kwargs["model"] == agent.model_name
    input_payload = kwargs["input"]

    system_block = input_payload[0]["content"][0]
    user_block = input_payload[1]["content"][0]
    assistant_block = input_payload[2]["content"][0]

    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert user_block["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in assistant_block


@pytest.mark.asyncio
async def test_openrouter_agent_parses_tool_calls() -> None:
    responses_create = AsyncMock(
        return_value=FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Calling tool"}],
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "search", "arguments": "{\"query\": \"hud\"}"},
                            }
                        ],
                    }
                ],
                "status": "requires_action",
            }
        )
    )
    mock_client = MagicMock()
    mock_client.responses.create = responses_create

    agent = OpenRouterAgent(api_key="test-key", openai_client=mock_client)
    agent._available_tools = []

    result = await agent.get_response(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
    )

    assert not result.done
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "hud"}


@pytest.mark.asyncio
async def test_openrouter_agent_returns_text_response() -> None:
    responses_create = AsyncMock(
        return_value=FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hi there"}],
                    }
                ],
                "status": "completed",
            }
        )
    )
    mock_client = MagicMock()
    mock_client.responses.create = responses_create

    agent = OpenRouterAgent(api_key="test-key", openai_client=mock_client)
    agent._available_tools = []

    result = await agent.get_response(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
    )

    assert result.done
    assert result.content == "Hi there"
    assert result.tool_calls == []


def test_openrouter_agent_sanitizes_fieldinfo_in_tools() -> None:
    mock_client = MagicMock()
    agent = OpenRouterAgent(api_key="test-key", openai_client=mock_client)

    from pydantic import Field

    tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click an element",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": Field(default="", description="CSS selector"),
                    },
                    "required": ["selector"],
                },
            },
        }
    ]

    converted = agent._convert_tools_for_responses(tools)
    selector_schema = converted[0]["parameters"]["properties"]["selector"]
    assert isinstance(selector_schema, dict)
    assert selector_schema.get("description") == "CSS selector"


def test_openrouter_agent_converts_image_blocks() -> None:
    mock_client = MagicMock()
    agent = OpenRouterAgent(api_key="test-key", openai_client=mock_client)

    content = [
        {
            "type": "image",
            "mimeType": "image/png",
            "data": "dGVzdA==",
            "detail": "high",
        }
    ]

    message_blocks = agent._convert_messages([{"role": "user", "content": content}])
    image_block = message_blocks[0]["content"][0]
    assert image_block["type"] == "input_image"
    assert image_block["image_url"].startswith("data:image/png;base64,")
    assert image_block["detail"] == "high"


@pytest.mark.asyncio
async def test_format_tool_results_produces_function_call_output() -> None:
    mock_client = MagicMock()
    agent = OpenRouterAgent(api_key="test-key", openai_client=mock_client)

    tool_call = MCPToolCall(id="call-1", name="playwright", arguments={})
    tool_result = MCPToolResult(
        content=[
            types.TextContent(type="text", text="navigation complete"),
            types.ImageContent(type="image", data="dGVzdA==", mimeType="image/png"),
        ]
    )

    formatted = await agent.format_tool_results([tool_call], [tool_result])

    assert formatted[0]["type"] == "function_call_output"
    assert formatted[0]["call_id"] == "call-1"
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"][0]["type"] == "input_image"

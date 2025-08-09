"""Tests for Claude MCP Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import BadRequestError
from mcp import types

from hud.agents.claude import (
    ClaudeMCPAgent,
    base64_to_content_block,
    text_to_content_block,
    tool_use_content_block,
)
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from anthropic.types.beta import BetaImageBlockParam, BetaMessageParam, BetaTextBlockParam


class TestClaudeHelperFunctions:
    """Test helper functions for Claude message formatting."""

    def test_base64_to_content_block(self):
        """Test base64 image conversion."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501
        result = base64_to_content_block(base64_data)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == base64_data

    def test_text_to_content_block(self):
        """Test text conversion."""
        text = "Hello, world!"
        result = text_to_content_block(text)

        assert result["type"] == "text"
        assert result["text"] == text

    def test_tool_use_content_block(self):
        """Test tool result content block creation."""
        tool_use_id = "tool_123"
        content: list[BetaTextBlockParam | BetaImageBlockParam] = [
            text_to_content_block("Result text")
        ]

        result = tool_use_content_block(tool_use_id, content)

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == tool_use_id
        assert result["content"] == content  # type: ignore


class TestClaudeMCPAgent:
    """Test ClaudeMCPAgent class."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        mcp_client = MagicMock()
        return mcp_client

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("hud.agents.claude.AsyncAnthropic") as mock:
            client = AsyncMock()
            # Add beta attribute with messages
            client.beta = AsyncMock()
            client.beta.messages = AsyncMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client, mock_anthropic):
        """Test agent initialization."""
        # Test with provided model_client
        mock_model_client = MagicMock()
        agent = ClaudeMCPAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            model="claude-3-opus-20240229",
            max_tokens=1000,
        )

        assert agent.model_name == "claude-3-opus-20240229"
        assert agent.max_tokens == 1000
        assert agent.anthropic_client == mock_model_client

    @pytest.mark.asyncio
    async def test_init_without_model_client(self, mock_mcp_client):
        """Test agent initialization without model client."""
        with patch("hud.settings.settings.anthropic_api_key", "test_key"):
            agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model="claude-3-opus-20240229")

            assert agent.model_name == "claude-3-opus-20240229"
            assert agent.anthropic_client is not None

    @pytest.mark.asyncio
    async def test_create_initial_messages(self, mock_mcp_client):
        """Test creating initial messages."""
        mock_model_client = MagicMock()
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_model_client)

        # Test with text only
        messages = await agent.create_initial_messages("Hello, Claude!")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = list(messages[0]["content"])
        assert content[0]["type"] == "text"  # type: ignore
        assert content[0]["text"] == "Hello, Claude!"  # type: ignore

        # Test with screenshot
        messages = await agent.create_initial_messages("Look at this", screenshot="base64data")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = list(messages[0]["content"])
        assert len(content) == 2
        # Claude puts text first, then image
        assert content[0]["type"] == "text"  # type: ignore
        assert content[0]["text"] == "Look at this"  # type: ignore
        assert content[1]["type"] == "image"  # type: ignore

    @pytest.mark.asyncio
    async def test_format_tool_results_method(self, mock_mcp_client):
        """Test the agent's format_tool_results method."""
        mock_model_client = MagicMock()
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_model_client)

        tool_calls = [
            MCPToolCall(name="test_tool", arguments={}, tool_use_id="id1"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(content=[types.TextContent(type="text", text="Success")], isError=False),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # format_tool_results returns a single user message with tool result content
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # The content is wrapped in a tool result block
        content = list(messages[0]["content"])
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"  # type: ignore
        assert content[0]["tool_use_id"] == "id1"  # type: ignore
        # The actual content is nested inside
        inner_content = list(content[0]["content"])  # type: ignore
        assert inner_content[0]["type"] == "text"  # type: ignore
        assert inner_content[0]["text"] == "Success"  # type: ignore

    @pytest.mark.asyncio
    async def test_get_model_response(self, mock_mcp_client, mock_anthropic):
        """Test getting model response from Claude API."""
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_anthropic)

        # Mock the API response
        mock_response = MagicMock()

        # Create text block
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello!"

        # Create tool use block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "test_tool"
        tool_block.input = {"param": "value"}

        mock_response.content = [text_block, tool_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_anthropic.beta.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            cast("BetaMessageParam", {"role": "user", "content": [{"type": "text", "text": "Hi"}]})
        ]
        response = await agent.get_model_response(messages)

        assert response.content == "Hello!"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert response.tool_calls[0].arguments == {"param": "value"}
        # The test was checking for Claude-specific attributes that aren't part of ModelResponse
        # These would need to be accessed from the original Claude response if needed

        # Verify API was called correctly
        mock_anthropic.beta.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_text_only(self, mock_mcp_client, mock_anthropic):
        """Test getting text-only response."""
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_anthropic)

        mock_response = MagicMock()
        # Create text block
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Just text"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=10)
        mock_anthropic.beta.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            cast("BetaMessageParam", {"role": "user", "content": [{"type": "text", "text": "Hi"}]})
        ]
        response = await agent.get_model_response(messages)

        assert response.content == "Just text"
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_get_model_response_error(self, mock_mcp_client, mock_anthropic):
        """Test handling API errors."""
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_anthropic)

        # Mock API error
        mock_anthropic.beta.messages.create = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid request",
                response=MagicMock(status_code=400),
                body={"error": {"message": "Invalid request"}},
            )
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

        with pytest.raises(BadRequestError):
            await agent.get_model_response(messages)  # type: ignore

    @pytest.mark.asyncio
    async def test_run_with_tools(self, mock_mcp_client, mock_anthropic):
        """Test running agent with tool usage."""
        agent = ClaudeMCPAgent(mcp_client=mock_mcp_client, model_client=mock_anthropic)

        # Mock tool availability
        agent._available_tools = [
            types.Tool(name="calculator", description="Calculator", inputSchema={"type": "object"})
        ]
        agent._tool_map = {
            "calculator": types.Tool(
                name="calculator", description="Calculator", inputSchema={"type": "object"}
            )
        }

        # Mock initial response with tool use
        initial_response = MagicMock()
        # Create tool use block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "calc_123"
        tool_block.name = "calculator"
        tool_block.input = {"operation": "add", "a": 2, "b": 3}
        initial_response.content = [tool_block]
        initial_response.usage = MagicMock(input_tokens=10, output_tokens=15)

        # Mock follow-up response
        final_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "2 + 3 = 5"
        final_response.content = [text_block]
        final_response.usage = MagicMock(input_tokens=20, output_tokens=10)

        mock_anthropic.beta.messages.create = AsyncMock(
            side_effect=[initial_response, final_response]
        )

        # Mock tool execution
        agent.mcp_client.call_tool = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="5")], isError=False
            )
        )

        # Use a string prompt instead of a task
        result = await agent.run("What is 2 + 3?")

        assert result.content == "2 + 3 = 5"
        assert result.done is True

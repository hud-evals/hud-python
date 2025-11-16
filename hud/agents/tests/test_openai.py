"""Tests for OpenAI MCP Agent implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from hud.agents.openai import OperatorAgent
from hud.types import MCPToolCall, MCPToolResult


class TestOperatorAgent:
    """Test OperatorAgent class."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        mcp_client = AsyncMock()
        # Set up the mcp_config attribute as a regular dict, not a coroutine
        mcp_client.mcp_config = {"test_server": {"url": "http://test"}}
        # Mock list_tools to return the required openai_computer tool
        mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(
                    name="openai_computer", description="OpenAI computer use tool", inputSchema={}
                )
            ]
        )
        mcp_client.initialize = AsyncMock()
        return mcp_client

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("hud.agents.openai.AsyncOpenAI") as mock:
            client = AsyncMock()
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client):
        """Test agent initialization."""
        mock_model_client = MagicMock()
        agent = OperatorAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            model="gpt-4",
            validate_api_key=False,  # Skip validation in tests
        )

        assert agent.model_name == "Operator"
        assert agent.model == "gpt-4"
        assert agent.openai_client == mock_model_client

    @pytest.mark.asyncio
    async def test_format_blocks(self, mock_mcp_client):
        """Test formatting content blocks."""
        mock_model_client = MagicMock()
        agent = OperatorAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            validate_api_key=False,  # Skip validation in tests
        )

        # Test with text blocks
        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, GPT!"),
            types.TextContent(type="text", text="Another message"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 2
        assert messages[0] == {"type": "input_text", "text": "Hello, GPT!"}
        assert messages[1] == {"type": "input_text", "text": "Another message"}

        # Test with mixed content
        blocks = [
            types.TextContent(type="text", text="Text content"),
            types.ImageContent(type="image", data="base64data", mimeType="image/png"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 2
        assert messages[0] == {"type": "input_text", "text": "Text content"}
        assert messages[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,base64data",
        }

    @pytest.mark.asyncio
    async def test_format_tool_results(self, mock_mcp_client, mock_openai):
        """Test formatting tool results."""
        agent = OperatorAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,  # Skip validation in tests
        )

        tool_calls = [
            MCPToolCall(name="test_tool", arguments={}, id="call_123"),  # type: ignore
            MCPToolCall(name="screenshot", arguments={}, id="call_456"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(content=[types.TextContent(type="text", text="Success")], isError=False),
            MCPToolResult(
                content=[types.ImageContent(type="image", data="base64data", mimeType="image/png")],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # OpenAI's format_tool_results returns input_image with screenshot
        assert len(messages) == 1
        assert messages[0]["type"] == "input_image"
        assert "image_url" in messages[0]
        assert messages[0]["image_url"] == "data:image/png;base64,base64data"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with errors."""
        agent = OperatorAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,  # Skip validation in tests
        )

        tool_calls = [
            MCPToolCall(name="failing_tool", arguments={}, id="call_error"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Something went wrong")], isError=True
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # Since the result has isError=True and no screenshot, returns empty list
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_get_model_response(self, mock_mcp_client, mock_openai):
        """Test getting model response from OpenAI API."""
        # Disable telemetry for this test to avoid backend configuration issues
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OperatorAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,  # Skip validation in tests
            )

            # Set up available tools so agent doesn't return "No computer use tools available"
            agent._available_tools = [
                types.Tool(name="computer_openai", description="Computer tool", inputSchema={})
            ]

            # Mock OpenAI API response for a successful computer use response
            mock_response = MagicMock()
            mock_response.id = "response_123"
            mock_response.state = "completed"
            # Mock the output message structure
            mock_output_text = MagicMock()
            mock_output_text.type = "output_text"
            mock_output_text.text = "I can see the screen content."

            mock_output_message = MagicMock()
            mock_output_message.type = "message"
            mock_output_message.content = [mock_output_text]

            mock_response.output = [mock_output_message]

            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [{"prompt": "What's on the screen?", "screenshot": None}]
            response = await agent.get_response(messages)

            # The test should verify that the response is processed correctly
            # Since the isinstance checks will fail, content will be empty, but done should be True
            assert response.done is True
            assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_handle_empty_response(self, mock_mcp_client, mock_openai):
        """Test handling empty response from API."""
        agent = OperatorAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,  # Skip validation in tests
        )

        # Set up available tools
        agent._available_tools = [
            types.Tool(name="openai_computer", description="Computer tool", inputSchema={})
        ]

        # Mock empty response
        mock_response = MagicMock()
        mock_response.id = "response_empty"
        mock_response.state = "completed"
        mock_response.output = []  # Empty output

        mock_openai.responses.create = AsyncMock(return_value=mock_response)

        messages = [{"prompt": "Hi", "screenshot": None}]
        response = await agent.get_response(messages)

        assert response.content == ""
        assert response.tool_calls == []

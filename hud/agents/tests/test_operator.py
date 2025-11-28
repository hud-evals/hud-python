"""Tests for OperatorAgent implementation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types
from openai import AsyncOpenAI
from openai.types.responses.response_computer_tool_call import PendingSafetyCheck

from hud.agents.operator import OperatorAgent
from hud.types import MCPToolCall, MCPToolResult


class TestOperatorAgent:
    """Test OperatorAgent class."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        client.responses.create = AsyncMock()
        with patch("hud.agents.openai.AsyncOpenAI", return_value=client):
            yield client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client_openai_computer):
        """Test agent initialization."""
        mock_model_client = AsyncOpenAI(api_key="test")
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client_openai_computer,
            model_client=mock_model_client,
            checkpoint_name="gpt-4",
            validate_api_key=False,  # Skip validation in tests
        )

        assert agent.model_name == "Operator"
        assert agent.config.checkpoint_name == "gpt-4"
        assert agent.openai_client == mock_model_client

    @pytest.mark.asyncio
    async def test_format_blocks(self, mock_mcp_client_openai_computer):
        """Test formatting content blocks."""
        mock_model_client = AsyncOpenAI(api_key="test")
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client_openai_computer,
            model_client=mock_model_client,
            validate_api_key=False,  # Skip validation in tests
        )

        # Test with text blocks
        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, GPT!"),
            types.TextContent(type="text", text="Another message"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["role"] == "user"
        content = cast("list[dict[str, Any]]", msg["content"])
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Hello, GPT!"}
        assert content[1] == {"type": "input_text", "text": "Another message"}

        # Test with mixed content
        blocks = [
            types.TextContent(type="text", text="Text content"),
            types.ImageContent(type="image", data="base64data", mimeType="image/png"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["role"] == "user"
        content = cast("list[dict[str, Any]]", msg["content"])
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Text content"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,base64data",
            "detail": "auto",
        }

    @pytest.mark.asyncio
    async def test_format_tool_results(self, mock_mcp_client_openai_computer, mock_openai):
        """Test formatting tool results."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client_openai_computer,
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

        # Should return both tool results as function_call_output
        assert len(messages) == 2
        # First result is text
        msg0 = cast("dict[str, Any]", messages[0])
        assert msg0["type"] == "function_call_output"
        assert msg0["call_id"] == "call_123"
        output0 = cast("list[dict[str, Any]]", msg0["output"])
        assert output0[0]["type"] == "input_text"
        assert output0[0]["text"] == "Success"
        # Second result is image
        msg1 = cast("dict[str, Any]", messages[1])
        assert msg1["type"] == "function_call_output"
        assert msg1["call_id"] == "call_456"
        output1 = cast("list[dict[str, Any]]", msg1["output"])
        assert output1[0]["type"] == "input_image"
        assert output1[0]["image_url"] == "data:image/png;base64,base64data"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(
        self, mock_mcp_client_openai_computer, mock_openai
    ):
        """Test formatting tool results with errors."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client_openai_computer,
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

        # Error results are returned with error flag and content
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_error"
        output = cast("list[dict[str, Any]]", msg["output"])
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == "[tool_error] true"
        assert output[1]["type"] == "input_text"
        assert output[1]["text"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_get_model_response(self, mock_mcp_client_openai_computer, mock_openai):
        """Test getting model response from OpenAI API."""
        # Disable telemetry for this test to avoid backend configuration issues
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OperatorAgent.create(
                mcp_client=mock_mcp_client_openai_computer,
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
    async def test_handle_empty_response(self, mock_mcp_client_openai_computer, mock_openai):
        """Test handling empty response from API."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client_openai_computer,
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

    @pytest.mark.asyncio
    async def test_pending_safety_checks_initialization(self, mock_mcp_client, mock_openai):
        """Test that OperatorAgent initializes pending_call_id and pending_safety_checks."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Verify initial state
        assert agent.pending_call_id is None
        assert agent.pending_safety_checks == []

        # Set some state
        agent.pending_call_id = "call_id"
        agent.pending_safety_checks = [
            PendingSafetyCheck(id="safety_check_id", code="value", message="message")
        ]

        # Verify state was set
        assert agent.pending_call_id == "call_id"
        assert len(agent.pending_safety_checks) == 1
        assert agent.pending_safety_checks[0].id == "safety_check_id"

    @pytest.mark.asyncio
    async def test_extract_tool_call_computer(self, mock_mcp_client, mock_openai):
        """Test that _extract_tool_call routes computer_call to openai_computer."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Create a mock computer_call item
        mock_item = MagicMock()
        mock_item.type = "computer_call"
        mock_item.call_id = "call_123"
        mock_item.pending_safety_checks = [
            PendingSafetyCheck(id="check_1", code="code", message="msg")
        ]
        mock_item.action.to_dict.return_value = {"type": "screenshot"}

        tool_call = agent._extract_tool_call(mock_item)

        # Should route to openai_computer tool
        assert tool_call is not None
        assert tool_call.name == "openai_computer"
        assert tool_call.id == "call_123"
        assert tool_call.arguments == {"type": "screenshot"}
        # Should update pending_safety_checks
        assert agent.pending_safety_checks == mock_item.pending_safety_checks

    @pytest.mark.asyncio
    async def test_extract_tool_call_delegates_to_super(self, mock_mcp_client, mock_openai):
        """Test that _extract_tool_call delegates non-computer calls to parent."""
        agent = OperatorAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Set up tool name map
        agent._tool_name_map = {"test_tool": "mcp_test_tool"}

        # Create a mock function_call item
        mock_item = MagicMock()
        mock_item.type = "function_call"
        mock_item.call_id = "call_456"
        mock_item.name = "test_tool"
        mock_item.arguments = '{"arg": "value"}'

        tool_call = agent._extract_tool_call(mock_item)

        # Should delegate to parent and map the tool name
        assert tool_call is not None
        assert tool_call.name == "mcp_test_tool"
        assert tool_call.id == "call_456"
        assert tool_call.arguments == {"arg": "value"}

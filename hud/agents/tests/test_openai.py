"""Tests for OpenAI MCP Agent implementation."""

from __future__ import annotations

from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Summary

from hud.agents.openai import OpenAIAgent
from hud.types import MCPToolCall, MCPToolResult


class TestOpenAIAgent:
    """Test OpenAIAgent class."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        mcp_client = AsyncMock()
        mcp_client.mcp_config = {"test_server": {"url": "http://test"}}
        mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(
                    name="test_tool",
                    description="A test tool",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )
        mcp_client.initialize = AsyncMock()
        return mcp_client

    @pytest.fixture
    def mock_openai(self):
        """Create a stub OpenAI client."""
        with patch("hud.agents.openai.AsyncOpenAI") as mock_class:
            client = AsyncOpenAI(api_key="test", base_url="http://localhost")
            client.chat.completions.create = AsyncMock()
            client.responses.create = AsyncMock()
            mock_class.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_mcp_client):
        """Test agent initialization with provided client."""
        mock_model_client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            model="gpt-4o",
            validate_api_key=False,
        )

        assert agent.model_name == "OpenAI"
        assert agent.model == "gpt-4o"
        assert agent.checkpoint_name == "gpt-4o"
        assert agent.openai_client == mock_model_client
        assert agent.max_output_tokens is None
        assert agent.temperature is None

    @pytest.mark.asyncio
    async def test_init_with_parameters(self, mock_mcp_client):
        """Test agent initialization with various parameters."""
        mock_model_client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            model="gpt-4o",
            max_output_tokens=2048,
            temperature=0.7,
            reasoning="auto",
            tool_choice="auto",
            parallel_tool_calls=True,
            validate_api_key=False,
        )

        assert agent.max_output_tokens == 2048
        assert agent.temperature == 0.7
        assert agent.reasoning == "auto"
        assert agent.tool_choice == "auto"
        assert agent.parallel_tool_calls is True

    @pytest.mark.asyncio
    async def test_init_without_client_no_api_key(self, mock_mcp_client):
        """Test agent initialization fails without API key."""
        with patch("hud.agents.openai.settings") as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIAgent(mcp_client=mock_mcp_client)

    @pytest.mark.asyncio
    async def test_format_blocks_text_only(self, mock_mcp_client, mock_openai):
        """Test formatting text content blocks."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, world!"),
            types.TextContent(type="text", text="How are you?"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["role"] == "user"
        content = cast("list[dict[str, Any]]", msg["content"])
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Hello, world!"}
        assert content[1] == {"type": "input_text", "text": "How are you?"}

    @pytest.mark.asyncio
    async def test_format_blocks_with_image(self, mock_mcp_client, mock_openai):
        """Test formatting content blocks with images."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Check this out:"),
            types.ImageContent(type="image", data="base64imagedata", mimeType="image/jpeg"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["role"] == "user"
        content = cast("list[dict[str, Any]]", msg["content"])
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "Check this out:"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,base64imagedata",
        }

    @pytest.mark.asyncio
    async def test_format_blocks_empty(self, mock_mcp_client, mock_openai):
        """Test formatting empty content blocks."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = []

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["role"] == "user"
        content = cast("list[dict[str, Any]]", msg["content"])
        assert len(content) == 1
        assert content[0] == {"type": "input_text", "text": ""}

    @pytest.mark.asyncio
    async def test_format_tool_results_text(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with text content."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="test_tool", arguments={"arg": "value"}, id="call_123"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Tool executed successfully")],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_123"
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_image(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with image content."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="screenshot", arguments={}, id="call_456"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[
                    types.ImageContent(type="image", data="screenshot_data", mimeType="image/png")
                ],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_456"
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_image"
        assert output[0]["image_url"] == "data:image/png;base64,screenshot_data"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with errors."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="failing_tool", arguments={}, id="call_error"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Error: Something went wrong")],
                isError=True,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_error"
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 2
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == "[tool_error] true"
        assert output[1]["type"] == "input_text"
        assert output[1]["text"] == "Error: Something went wrong"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_structured_content(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with structured content."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="data_tool", arguments={}, id="call_789"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[],
                structuredContent={"key": "value", "number": 42},
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_789"
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_text"
        # Structured content is JSON serialized
        import json

        parsed = json.loads(output[0]["text"])
        assert parsed == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_format_tool_results_multiple(self, mock_mcp_client, mock_openai):
        """Test formatting multiple tool results."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="tool1", arguments={}, id="call_1"),  # type: ignore
            MCPToolCall(name="tool2", arguments={}, id="call_2"),  # type: ignore
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Result 1")],
                isError=False,
            ),
            MCPToolResult(
                content=[types.TextContent(type="text", text="Result 2")],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 2
        msg0 = cast("dict[str, Any]", messages[0])
        assert msg0["call_id"] == "call_1"
        msg1 = cast("dict[str, Any]", messages[1])
        assert msg1["call_id"] == "call_2"

    @pytest.mark.asyncio
    async def test_format_tool_results_missing_call_id(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with missing call_id."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="tool_no_id", arguments={}, id=""),  # Empty string instead of None
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Some result")],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # Should skip tools without call_id (empty string is falsy)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_get_response_with_text(self, mock_mcp_client, mock_openai):
        """Test getting model response with text output."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Mock OpenAI API response
            mock_response = MagicMock()
            mock_response.id = "response_123"

            # Create properly typed output text with all required fields
            mock_output_text = ResponseOutputText(
                type="output_text",
                text="This is the response text",
                annotations=[],  # Required field
            )

            # Create properly typed output message with all required fields
            mock_output_message = ResponseOutputMessage(
                type="message",
                id="msg_123",  # Required field
                role="assistant",  # Required field
                status="completed",  # Required field
                content=[mock_output_text],
            )

            mock_response.output = [mock_output_message]

            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            # Test with initial message
            messages = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
            response = await agent.get_response(messages)

            assert response.content == "This is the response text"
            assert response.done is True
            assert response.tool_calls == []
            assert agent.last_response_id == "response_123"

    @pytest.mark.asyncio
    async def test_get_response_with_tool_call(self, mock_mcp_client, mock_openai):
        """Test getting model response with tool call."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Set up tool name map
            agent._tool_name_map = {"test_tool": "test_tool"}

            # Mock OpenAI API response with properly typed function call
            mock_response = MagicMock()
            mock_response.id = "response_456"

            # Create properly typed function call with correct type value
            mock_function_call = ResponseFunctionToolCall(
                type="function_call",  # Correct type value
                call_id="call_123",
                name="test_tool",
                arguments='{"param": "value"}',
            )

            mock_response.output = [mock_function_call]

            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": [{"type": "input_text", "text": "Do something"}]}
            ]
            response = await agent.get_response(messages)

            assert response.done is False
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "test_tool"
            assert response.tool_calls[0].id == "call_123"
            assert response.tool_calls[0].arguments == {"param": "value"}

    @pytest.mark.asyncio
    async def test_get_response_with_reasoning(self, mock_mcp_client, mock_openai):
        """Test getting model response with reasoning."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Mock OpenAI API response with properly typed reasoning
            mock_response = MagicMock()
            mock_response.id = "response_789"

            # Create a properly typed reasoning item with all required fields
            mock_summary = Summary(
                type="summary_text",  # Correct literal type value
                text="Let me think about this...",
            )

            mock_reasoning = ResponseReasoningItem(
                type="reasoning",
                id="reasoning_1",  # Required field
                summary=[mock_summary],  # Required field
                status="completed",  # Required field
            )

            # Create properly typed output message with all required fields
            mock_output_text = ResponseOutputText(
                type="output_text",
                text="Final answer",
                annotations=[],  # Required field
            )
            mock_output_message = ResponseOutputMessage(
                type="message",
                id="msg_789",  # Required field
                role="assistant",  # Required field
                status="completed",  # Required field
                content=[mock_output_text],
            )

            mock_response.output = [mock_reasoning, mock_output_message]

            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": [{"type": "input_text", "text": "Hard question"}]}
            ]
            response = await agent.get_response(messages)

            assert "Thinking: Let me think about this..." in response.content
            assert "Final answer" in response.content

    @pytest.mark.asyncio
    async def test_get_response_empty_messages(self, mock_mcp_client, mock_openai):
        """Test getting model response with empty messages."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Mock empty response
            mock_response = MagicMock()
            mock_response.id = "response_empty"
            mock_response.output = []

            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = []
            response = await agent.get_response(messages)

            assert response.content == ""
            assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_get_response_no_new_messages_with_previous_id(
        self, mock_mcp_client, mock_openai
    ):
        """Test getting model response when no new messages and previous response exists."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            agent.last_response_id = "prev_response"
            agent._message_cursor = 1

            messages = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
            response = await agent.get_response(messages)

            # Should return early without calling API
            assert response.content == ""
            assert response.done is True
            mock_openai.responses.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_request_payload(self, mock_mcp_client, mock_openai):
        """Test building request payload."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            model="gpt-4o",
            max_output_tokens=1024,
            temperature=0.5,
            reasoning="auto",
            tool_choice="auto",
            parallel_tool_calls=True,
            validate_api_key=False,
        )

        agent._openai_tools = [cast("Any", {"type": "function", "name": "test"})]
        agent.system_prompt = "You are a helpful assistant"
        agent.last_response_id = "prev_123"

        new_items = cast(
            "Any", [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
        )
        payload = agent._build_request_payload(new_items)

        assert payload["model"] == "gpt-4o"
        assert payload["input"] == new_items
        assert payload["instructions"] == "You are a helpful assistant"
        assert payload["max_output_tokens"] == 1024
        assert payload["temperature"] == 0.5
        assert payload["reasoning"] == "auto"
        assert payload["tool_choice"] == "auto"
        assert payload["parallel_tool_calls"] is True
        assert payload["tools"] == [{"type": "function", "name": "test"}]
        assert payload["previous_response_id"] == "prev_123"

    @pytest.mark.asyncio
    async def test_build_request_payload_minimal(self, mock_mcp_client, mock_openai):
        """Test building request payload with minimal parameters."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        new_items = cast(
            "Any", [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
        )
        payload = agent._build_request_payload(new_items)

        assert payload["model"] == "gpt-5.1"  # default
        assert payload["input"] == new_items
        assert "max_output_tokens" not in payload
        assert "temperature" not in payload
        assert "reasoning" not in payload
        assert "tools" not in payload
        assert "previous_response_id" not in payload

    @pytest.mark.asyncio
    async def test_reset_response_state(self, mock_mcp_client, mock_openai):
        """Test resetting response state."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Set some state
        agent.last_response_id = "some_id"
        agent.pending_call_id = "call_id"
        agent.pending_safety_checks = [{"check": "value"}]
        agent._message_cursor = 5

        # Reset
        agent._reset_response_state()

        assert agent.last_response_id is None
        assert agent.pending_call_id is None
        assert agent.pending_safety_checks == []
        assert agent._message_cursor == 0

    @pytest.mark.asyncio
    async def test_get_system_messages(self, mock_mcp_client, mock_openai):
        """Test getting system messages."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # OpenAI agent returns empty list (uses instructions field instead)
        messages = await agent.get_system_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_build_openai_tools(self, mock_mcp_client, mock_openai):
        """Test building OpenAI tools from MCP tools."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Mock MCP tools
        mock_tools = [
            types.Tool(
                name="tool1",
                description="First tool",
                inputSchema={
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                    "required": ["arg1"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(
                name="tool2",
                description="Second tool",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ]

        agent._available_tools = mock_tools
        agent._build_openai_tools()

        assert len(agent._openai_tools) == 2
        assert agent._tool_name_map == {"tool1": "tool1", "tool2": "tool2"}

        tool1 = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool1["type"] == "function"
        assert tool1["name"] == "tool1"
        assert tool1["description"] == "First tool"
        assert tool1["strict"] is True

    @pytest.mark.asyncio
    async def test_build_openai_tools_skips_incomplete(self, mock_mcp_client, mock_openai):
        """Test building OpenAI tools skips tools without description or schema."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Create mock tools directly as objects that bypass pydantic validation
        incomplete1 = MagicMock(spec=types.Tool)
        incomplete1.name = "incomplete1"
        incomplete1.description = None
        incomplete1.inputSchema = {"type": "object"}

        incomplete2 = MagicMock(spec=types.Tool)
        incomplete2.name = "incomplete2"
        incomplete2.description = "Has description"
        incomplete2.inputSchema = None

        complete = types.Tool(
            name="complete",
            description="Complete tool",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        )

        agent._available_tools = [incomplete1, incomplete2, complete]
        agent._build_openai_tools()

        # Should only have the complete tool
        assert len(agent._openai_tools) == 1
        tool = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool["name"] == "complete"

    @pytest.mark.asyncio
    async def test_convert_function_tool_call(self, mock_mcp_client, mock_openai):
        """Test converting OpenAI function tool call to MCP format."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent._tool_name_map = {"openai_name": "mcp_name"}

        mock_call = MagicMock()
        mock_call.call_id = "call_123"
        mock_call.name = "openai_name"
        mock_call.arguments = '{"key": "value", "number": 42}'

        result = agent._convert_function_tool_call(mock_call)

        assert result is not None
        assert result.name == "mcp_name"
        assert result.id == "call_123"
        assert result.arguments == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_convert_function_tool_call_invalid_json(self, mock_mcp_client, mock_openai):
        """Test converting function tool call with invalid JSON."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent._tool_name_map = {"tool": "tool"}

        mock_call = MagicMock()
        mock_call.call_id = "call_456"
        mock_call.name = "tool"
        mock_call.arguments = "invalid json {{"

        result = agent._convert_function_tool_call(mock_call)

        assert result is not None
        assert result.name == "tool"
        assert result.id == "call_456"
        # Should wrap invalid JSON in raw_arguments
        assert result.arguments == {"raw_arguments": "invalid json {{"}

    @pytest.mark.asyncio
    async def test_convert_function_tool_call_empty_args(self, mock_mcp_client, mock_openai):
        """Test converting function tool call with empty arguments."""
        agent = OpenAIAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent._tool_name_map = {"tool": "tool"}

        mock_call = MagicMock()
        mock_call.call_id = "call_789"
        mock_call.name = "tool"
        mock_call.arguments = None

        result = agent._convert_function_tool_call(mock_call)

        assert result is not None
        assert result.arguments == {}

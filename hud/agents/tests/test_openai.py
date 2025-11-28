"""Tests for OpenAI MCP Agent implementation."""

from __future__ import annotations

from typing import Any, cast
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
from pydantic import AnyUrl

from hud.agents.openai import OpenAIAgent
from hud.types import MCPToolCall, MCPToolResult


class TestOpenAIAgent:
    """Test OpenAIAgent class."""

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
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            checkpoint_name="gpt-4o",
            validate_api_key=False,
        )

        assert agent.model_name == "OpenAI"
        assert agent.config.checkpoint_name == "gpt-4o"
        assert agent.checkpoint_name == "gpt-4o"
        assert agent.openai_client == mock_model_client
        assert agent.max_output_tokens is None
        assert agent.temperature is None

    @pytest.mark.asyncio
    async def test_init_with_parameters(self, mock_mcp_client):
        """Test agent initialization with various parameters."""
        mock_model_client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_model_client,
            checkpoint_name="gpt-4o",
            max_output_tokens=2048,
            temperature=0.7,
            reasoning={"effort": "high"},
            tool_choice="auto",
            parallel_tool_calls=True,
            validate_api_key=False,
        )

        assert agent.max_output_tokens == 2048
        assert agent.temperature == 0.7
        assert agent.reasoning == {"effort": "high"}
        assert agent.tool_choice == "auto"
        assert agent.parallel_tool_calls is True

    @pytest.mark.asyncio
    async def test_init_without_client_no_api_key(self, mock_mcp_client):
        """Test agent initialization fails without API key."""
        with patch("hud.agents.openai.settings") as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIAgent.create(mcp_client=mock_mcp_client)

    @pytest.mark.asyncio
    async def test_format_blocks_text_only(self, mock_mcp_client, mock_openai):
        """Test formatting text content blocks."""
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
            "detail": "auto",
        }

    @pytest.mark.asyncio
    async def test_format_blocks_empty(self, mock_mcp_client, mock_openai):
        """Test formatting empty content blocks."""
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
        agent = OpenAIAgent.create(
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
            agent = OpenAIAgent.create(
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
            agent = OpenAIAgent.create(
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
            agent = OpenAIAgent.create(
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
            agent = OpenAIAgent.create(
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
            agent = OpenAIAgent.create(
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
    async def test_get_response_passes_correct_payload(self, mock_mcp_client, mock_openai):
        """Test that get_response passes correct parameters to OpenAI API."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                checkpoint_name="gpt-4o",
                max_output_tokens=1024,
                temperature=0.5,
                reasoning={"effort": "high"},
                tool_choice="auto",
                parallel_tool_calls=True,
                validate_api_key=False,
            )

            agent._openai_tools = [cast("Any", {"type": "function", "name": "test"})]
            agent.system_prompt = "You are a helpful assistant"
            agent.last_response_id = "prev_123"

            # Mock the API response
            mock_response = MagicMock()
            mock_response.id = "response_new"
            mock_response.output = []
            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
            await agent.get_response(messages)

            # Verify the API was called with the correct parameters
            mock_openai.responses.create.assert_called_once()
            call_kwargs = mock_openai.responses.create.call_args.kwargs

            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["input"] == messages
            assert call_kwargs["instructions"] == "You are a helpful assistant"
            assert call_kwargs["max_output_tokens"] == 1024
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["reasoning"] == {"effort": "high"}
            assert call_kwargs["tool_choice"] == "auto"
            assert call_kwargs["parallel_tool_calls"] is True
            assert call_kwargs["tools"] == [{"type": "function", "name": "test"}]
            assert call_kwargs["previous_response_id"] == "prev_123"

    @pytest.mark.asyncio
    async def test_get_response_passes_minimal_payload(self, mock_mcp_client, mock_openai):
        """Test that get_response passes minimal parameters when not configured."""
        from openai import Omit

        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Mock the API response
            mock_response = MagicMock()
            mock_response.id = "response_new"
            mock_response.output = []
            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
            await agent.get_response(messages)

            # Verify the API was called with minimal parameters
            mock_openai.responses.create.assert_called_once()
            call_kwargs = mock_openai.responses.create.call_args.kwargs

            assert call_kwargs["model"] == "gpt-5.1"  # default
            assert call_kwargs["input"] == messages
            assert call_kwargs["max_output_tokens"] is None
            assert call_kwargs["temperature"] is None
            # tool_choice should be Omit() when not set
            assert isinstance(call_kwargs["tool_choice"], Omit)
            # tools should be Omit() when empty
            assert isinstance(call_kwargs["tools"], Omit)
            # previous_response_id should be Omit() when not set
            assert isinstance(call_kwargs["previous_response_id"], Omit)

    @pytest.mark.asyncio
    async def test_reset_response_state(self, mock_mcp_client, mock_openai):
        """Test resetting response state."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Set some state
        agent.last_response_id = "some_id"
        agent._message_cursor = 5

        # Reset
        agent._reset_response_state()

        assert agent.last_response_id is None
        assert agent._message_cursor == 0

    @pytest.mark.asyncio
    async def test_get_system_messages(self, mock_mcp_client, mock_openai):
        """Test getting system messages."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # OpenAI agent returns empty list (uses instructions field instead)
        messages = await agent.get_system_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai(self, mock_mcp_client, mock_openai):
        """Test converting MCP tools to OpenAI format."""
        agent = OpenAIAgent.create(
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
        agent._convert_tools_for_openai()

        assert len(agent._openai_tools) == 2
        assert agent._tool_name_map == {"tool1": "tool1", "tool2": "tool2"}

        tool1 = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool1["type"] == "function"
        assert tool1["name"] == "tool1"
        assert tool1["description"] == "First tool"
        assert tool1["strict"] is True

    @pytest.mark.asyncio
    async def test_convert_tools_raises_on_incomplete(self, mock_mcp_client, mock_openai):
        """Test that converting tools raises error for incomplete tool definitions."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Create mock tools directly as objects that bypass pydantic validation
        incomplete1 = MagicMock(spec=types.Tool)
        incomplete1.name = "incomplete1"
        incomplete1.description = None
        incomplete1.inputSchema = {"type": "object"}

        agent._available_tools = [incomplete1]

        # Should raise ValueError for tool without description
        with pytest.raises(ValueError, match="requires both a description and inputSchema"):
            agent._convert_tools_for_openai()

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai_via_initialize(self, mock_mcp_client, mock_openai):
        """Test that initialize properly converts tools."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Mock the list_tools to return our test tools
        mock_mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(
                    name="complete",
                    description="Complete tool",
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
                )
            ]
        )

        await agent.initialize()

        # Should have the complete tool converted
        assert len(agent._openai_tools) == 1
        tool = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool["name"] == "complete"

    @pytest.mark.asyncio
    async def test_get_response_converts_function_tool_call(self, mock_mcp_client, mock_openai):
        """Test that get_response properly converts OpenAI function tool calls to MCP format."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Set up tool name map (simulating tool conversion)
            agent._tool_name_map = {"openai_name": "mcp_name"}

            # Mock OpenAI API response with function call
            mock_response = MagicMock()
            mock_response.id = "response_123"

            mock_function_call = ResponseFunctionToolCall(
                type="function_call",
                call_id="call_123",
                name="openai_name",
                arguments='{"key": "value", "number": 42}',
            )

            mock_response.output = [mock_function_call]
            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": [{"type": "input_text", "text": "Do something"}]}
            ]
            response = await agent.get_response(messages)

            # Verify the tool call was converted correctly
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "mcp_name"
            assert response.tool_calls[0].id == "call_123"
            assert response.tool_calls[0].arguments == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_convert_function_tool_call_invalid_json(self, mock_mcp_client, mock_openai):
        """Test converting function tool call with invalid JSON."""
        _agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

    async def test_get_response_raises_on_invalid_json_arguments(
        self, mock_mcp_client, mock_openai
    ):
        """Test that get_response raises error on invalid JSON in function call arguments.

        With strict mode being mandatory, invalid JSON arguments should never occur
        in practice since schemas are validated. This test verifies that if it does
        happen, we get an appropriate error rather than silently failing.
        """
        import json

        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            agent._tool_name_map = {"tool": "tool"}

            # Mock OpenAI API response with function call that has invalid JSON
            mock_response = MagicMock()
            mock_response.id = "response_456"

            mock_function_call = ResponseFunctionToolCall(
                type="function_call",
                call_id="call_456",
                name="tool",
                arguments="invalid json {{",
            )

            mock_response.output = [mock_function_call]
            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": [{"type": "input_text", "text": "Do something"}]}
            ]

            # With strict mode mandatory, invalid JSON should raise an error
            with pytest.raises(json.JSONDecodeError):
                await agent.get_response(messages)

    @pytest.mark.asyncio
    async def test_get_response_handles_tool_name_mapping(self, mock_mcp_client, mock_openai):
        """Test that get_response correctly maps tool names that aren't in the map."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = OpenAIAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_openai,
                validate_api_key=False,
            )

            # Tool name is NOT in the map, should fall back to the original name
            agent._tool_name_map = {}

            mock_response = MagicMock()
            mock_response.id = "response_789"

            mock_function_call = ResponseFunctionToolCall(
                type="function_call",
                call_id="call_789",
                name="unmapped_tool",
                arguments="{}",
            )

            mock_response.output = [mock_function_call]
            mock_openai.responses.create = AsyncMock(return_value=mock_response)

            messages = [
                {"role": "user", "content": [{"type": "input_text", "text": "Do something"}]}
            ]
            response = await agent.get_response(messages)

            # Should use the original tool name when not in map
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "unmapped_tool"
            assert response.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai_shell_tool(self, mock_mcp_client, mock_openai):
        """Test that shell tool is converted to OpenAI native shell type."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Mock a shell tool
        shell_tool = types.Tool(
            name="shell",
            description="Execute shell commands",
            inputSchema={"type": "object", "properties": {}},
        )

        agent._available_tools = [shell_tool]
        agent._convert_tools_for_openai()

        assert len(agent._openai_tools) == 1
        tool = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool["type"] == "shell"

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai_apply_patch_tool(self, mock_mcp_client, mock_openai):
        """Test that apply_patch tool is converted to OpenAI native apply_patch type."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Mock an apply_patch tool
        apply_patch_tool = types.Tool(
            name="apply_patch",
            description="Apply patches to files",
            inputSchema={"type": "object", "properties": {}},
        )

        agent._available_tools = [apply_patch_tool]
        agent._convert_tools_for_openai()

        assert len(agent._openai_tools) == 1
        tool = cast("dict[str, Any]", agent._openai_tools[0])
        assert tool["type"] == "apply_patch"

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai_strict_schema_failure(
        self, mock_mcp_client, mock_openai
    ):
        """Test that tool conversion raises error when strict schema conversion fails."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Mock a tool with a schema that will fail strict conversion
        # Using a schema without additionalProperties which is required for strict mode
        mock_tool = types.Tool(
            name="non_strict_tool",
            description="A tool with non-strict schema",
            inputSchema={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                # Missing additionalProperties and required - will fail strict conversion
            },
        )

        agent._available_tools = [mock_tool]

        # Mock ensure_strict_json_schema to raise an exception
        with patch("hud.agents.openai.ensure_strict_json_schema") as mock_strict:
            mock_strict.side_effect = ValueError("Schema not strict compatible")
            # Now strict compatibility is mandatory, so this should raise
            with pytest.raises(ValueError, match="Schema not strict compatible"):
                agent._convert_tools_for_openai()

    @pytest.mark.asyncio
    async def test_format_tool_results_with_resource_link(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with ResourceLink content."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="resource_tool", arguments={}, id="call_resource"),
        ]

        # Create a ResourceLink content
        resource_link = types.ResourceLink(
            type="resource_link",
            name="test_resource",
            uri=AnyUrl("file:///test/resource"),
        )

        tool_results = [
            MCPToolResult(
                content=[resource_link],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_file"
        assert output[0]["file_url"] == "file:///test/resource"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_embedded_text_resource(
        self, mock_mcp_client, mock_openai
    ):
        """Test formatting tool results with EmbeddedResource containing text."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="embed_tool", arguments={}, id="call_embed"),
        ]

        # Create an EmbeddedResource with TextResourceContents
        text_resource = types.TextResourceContents(
            uri=AnyUrl("file:///test.txt"),
            mimeType="text/plain",
            text="Embedded text content",
        )
        embedded = types.EmbeddedResource(
            type="resource",
            resource=text_resource,
        )

        tool_results = [
            MCPToolResult(
                content=[embedded],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == "Embedded text content"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_embedded_blob_resource(
        self, mock_mcp_client, mock_openai
    ):
        """Test formatting tool results with EmbeddedResource containing blob."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="blob_tool", arguments={}, id="call_blob"),
        ]

        # Create an EmbeddedResource with BlobResourceContents
        blob_resource = types.BlobResourceContents(
            uri=AnyUrl("file:///test.bin"),
            mimeType="application/octet-stream",
            blob="YmluYXJ5IGRhdGE=",  # base64 encoded "binary data"
        )
        embedded = types.EmbeddedResource(
            type="resource",
            resource=blob_resource,
        )

        tool_results = [
            MCPToolResult(
                content=[embedded],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        output = cast("list[dict[str, Any]]", msg["output"])
        assert len(output) == 1
        assert output[0]["type"] == "input_file"
        assert output[0]["file_data"] == "YmluYXJ5IGRhdGE="

    @pytest.mark.asyncio
    async def test_format_tool_results_empty_content(self, mock_mcp_client, mock_openai):
        """Test formatting tool results with completely empty content."""
        agent = OpenAIAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_openai,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(name="empty_tool", arguments={}, id="call_empty"),
        ]

        tool_results = [
            MCPToolResult(
                content=[],  # Empty content
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        assert len(messages) == 1
        msg = cast("dict[str, Any]", messages[0])
        output = cast("list[dict[str, Any]]", msg["output"])
        # Should have fallback empty text when no content
        assert len(output) == 1
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == ""

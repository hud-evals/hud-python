"""Tests for Claude MCP Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic
from mcp import types

from hud.agents.claude import (
    ClaudeAgent,
    base64_to_content_block,
    text_to_content_block,
    tool_use_content_block,
)
from hud.eval.context import EvalContext
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from anthropic.types.beta import BetaImageBlockParam, BetaTextBlockParam


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing."""

    def __init__(self, tools: list[types.Tool] | None = None) -> None:
        self.prompt = "Test prompt"
        self._tools = tools or []
        self._submitted: str | None = None
        self.reward: float | None = None

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        return MCPToolResult(
            content=[types.TextContent(type="text", text="ok")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


class MockStreamContextManager:
    """Mock for Claude's streaming context manager."""

    def __init__(self, response: MagicMock) -> None:
        self.response = response

    async def __aenter__(self) -> MockStreamContextManager:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> bool:
        return False

    def __aiter__(self) -> MockStreamContextManager:
        return self

    async def __anext__(self) -> None:
        raise StopAsyncIteration

    async def get_final_message(self) -> MagicMock:
        return self.response


class TestClaudeHelperFunctions:
    """Test helper functions for Claude message formatting."""

    def test_base64_to_content_block(self) -> None:
        """Test base64 image conversion."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        result = base64_to_content_block(base64_data)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == base64_data

    def test_text_to_content_block(self) -> None:
        """Test text conversion."""
        text = "Hello, world!"
        result = text_to_content_block(text)

        assert result["type"] == "text"
        assert result["text"] == text

    def test_tool_use_content_block(self) -> None:
        """Test tool result content block creation."""
        tool_use_id = "tool_123"
        content: list[BetaTextBlockParam | BetaImageBlockParam] = [
            text_to_content_block("Result text")
        ]

        result = tool_use_content_block(tool_use_id, content)

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == tool_use_id
        assert result["content"] == content  # type: ignore


class TestClaudeAgent:
    """Test ClaudeAgent class."""

    @pytest.fixture
    def mock_anthropic(self) -> AsyncAnthropic:
        """Create a stub Anthropic client."""
        with patch("hud.agents.claude.AsyncAnthropic") as mock_class, patch(
            "hud.agents.claude.Anthropic"
        ) as mock_sync:
            # Mock the sync client's models.list() for validation
            mock_sync.return_value.models.list.return_value = []

            client = MagicMock(spec=AsyncAnthropic)
            client.api_key = "test-key"
            mock_class.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test agent initialization with provided client."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            checkpoint_name="claude-sonnet-4-20250514",
            validate_api_key=False,
        )

        assert agent.model_name == "Claude"
        assert agent.config.checkpoint_name == "claude-sonnet-4-20250514"
        assert agent.anthropic_client == mock_anthropic

    @pytest.mark.asyncio
    async def test_init_with_parameters(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test agent initialization with various parameters."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            checkpoint_name="claude-sonnet-4-20250514",
            max_tokens=4096,
            validate_api_key=False,
        )

        assert agent.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_format_blocks_text_only(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test formatting text content blocks."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, world!"),
            types.TextContent(type="text", text="How are you?"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_format_blocks_with_image(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test formatting image content blocks."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Look at this:"),
            types.ImageContent(type="image", data="base64data", mimeType="image/png"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_format_tool_results_text(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test formatting tool results with text content."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        tool_calls = [MCPToolCall(id="call_123", name="test_tool", arguments={})]
        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Tool output")],
                isError=False,
            )
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test formatting tool results with error."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        tool_calls = [MCPToolCall(id="call_123", name="test_tool", arguments={})]
        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Error message")],
                isError=True,
            )
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        content = messages[0]["content"]
        # Error content should include "Error:" prefix
        assert any("Error" in str(block) for block in content[0]["content"])

    @pytest.mark.asyncio
    async def test_get_system_messages(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test that system messages return empty (Claude uses system param)."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            system_prompt="You are a helpful assistant.",
            validate_api_key=False,
        )

        messages = await agent.get_system_messages()
        # Claude doesn't use system messages in the message list
        assert messages == []

    @pytest.mark.asyncio
    async def test_convert_tools_for_claude(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test converting MCP tools to Claude format."""
        tools = [
            types.Tool(
                name="my_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Check that tools were converted
        assert len(agent.claude_tools) == 1
        assert agent.claude_tools[0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_computer_tool_detection(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test that computer tools are detected for beta API."""
        tools = [
            types.Tool(
                name="computer",
                description="Control computer",
                inputSchema={"type": "object"},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        assert agent.has_computer_tool is True

    @pytest.mark.asyncio
    async def test_get_response_with_text(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test getting response with text output."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]

        mock_stream = MockStreamContextManager(mock_response)
        mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent.has_computer_tool = False
        agent._initialized = True

        response = await agent.get_response([])
        assert response.content == "Hello!"
        assert response.done is True
        assert len(response.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_get_response_with_tool_call(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test getting response with tool call."""
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "call_123"
        mock_tool_use.name = "my_tool"
        mock_tool_use.input = {"x": "value"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]

        mock_stream = MockStreamContextManager(mock_response)
        mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        agent.claude_tools = []
        agent.tool_mapping = {"my_tool": "my_tool"}
        agent.has_computer_tool = False
        agent._initialized = True

        response = await agent.get_response([])
        assert response.done is False
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "my_tool"
        assert response.tool_calls[0].arguments == {"x": "value"}

"""Tests for Claude MCP Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from mcp import types

from hud.agents.claude import (
    ClaudeAgent,
    base64_to_content_block,
    text_to_content_block,
    tool_use_content_block,
)
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from collections.abc import Generator

    from anthropic.types.beta import BetaImageBlockParam, BetaMessageParam, BetaTextBlockParam


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing."""

    def __init__(self, tools: list[types.Tool] | None = None) -> None:
        # Core attributes
        self.prompt = "Test prompt"
        self._tools = tools or []
        self._submitted: str | None = None
        self.reward: float | None = None

        # Environment attributes
        self._router = ToolRouter()
        self._agent_include: list[str] | None = None
        self._agent_exclude: list[str] | None = None

        # EvalContext attributes
        self._task = None
        self.trace_id = "test-trace-id"
        self.eval_name = "test-eval"
        self.job_id: str | None = None
        self.group_id: str | None = None
        self.index = 0
        self.variants: dict[str, Any] = {}
        self.answer: str | None = None
        self.system_prompt: str | None = None
        self.error: BaseException | None = None
        self.metadata: dict[str, Any] = {}
        self.results: list[Any] = []
        self._is_summary = False

    def as_tools(self) -> list[types.Tool]:
        return self._tools

    @property
    def has_scenario(self) -> bool:
        return False

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
    def mock_anthropic(self) -> Generator[AsyncAnthropic, None, None]:  # type: ignore[misc]
        """Create a stub Anthropic client."""
        with patch("hud.agents.claude.AsyncAnthropic") as mock_class:
            client = MagicMock(spec=AsyncAnthropic)
            client.api_key = "test-key"
            mock_class.return_value = client
            yield client  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test agent initialization with provided client."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            model="claude-sonnet-4-20250514",
            validate_api_key=False,
        )

        assert agent.model_name == "Claude"
        assert agent.config.model == "claude-sonnet-4-20250514"
        assert agent.anthropic_client == mock_anthropic

    @pytest.mark.asyncio
    async def test_init_with_parameters(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test agent initialization with various parameters."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            model="claude-sonnet-4-20250514",
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
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"  # type: ignore[index]
        assert content[0]["text"] == "Hello, world!"  # type: ignore[index]

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
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[1]["type"] == "image"  # type: ignore[index]

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
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"  # type: ignore[index]
        assert content[0]["tool_use_id"] == "call_123"  # type: ignore[index]

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
        assert any("Error" in str(block) for block in content[0]["content"])  # type: ignore[index]

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
    async def test_get_response_with_thinking(self, mock_anthropic: AsyncAnthropic) -> None:
        """Test getting model response with thinking content."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = ClaudeAgent.create(
                model_client=mock_anthropic,
                validate_api_key=False,
            )
            # Set up agent as initialized
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = False
            agent._initialized = True

            mock_response = MagicMock()

            thinking_block = MagicMock()
            thinking_block.type = "thinking"
            thinking_block.thinking = "Let me analyze this problem..."

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Here is the answer"

            mock_response.content = [thinking_block, text_block]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=30)

            mock_stream = MockStreamContextManager(mock_response)
            mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

            messages = [
                cast(
                    "BetaMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": "Hard question"}]},
                )
            ]
            response = await agent.get_response(messages)

            assert response.content == "Here is the answer"
            assert response.reasoning == "Let me analyze this problem..."

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
        assert agent.claude_tools[0]["name"] == "my_tool"  # type: ignore[typeddict-item]

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


class TestClaudeAgentBedrock:
    """Test ClaudeAgent class with Bedrock."""

    @pytest.fixture
    def bedrock_client(self) -> AsyncAnthropicBedrock:
        """Create a real AsyncAnthropicBedrock client and stub networked methods."""
        client = AsyncAnthropicBedrock(
            aws_access_key="AKIATEST",
            aws_secret_key="secret",
            aws_region="us-east-1",
        )
        # Stub the actual Bedrock call so tests are hermetic.
        client.beta.messages.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_init(self, bedrock_client: AsyncAnthropicBedrock) -> None:
        """Test agent initialization."""
        agent = ClaudeAgent.create(
            model_client=bedrock_client,
            model="test-model-arn",
            validate_api_key=False,
        )

        assert agent.model_name == "Claude"
        assert agent.config.model == "test-model-arn"
        assert agent.anthropic_client == bedrock_client

    @pytest.mark.asyncio
    async def test_get_response_bedrock_uses_create_not_stream(
        self, bedrock_client: AsyncAnthropicBedrock
    ) -> None:
        """Bedrock path must call messages.create() (Bedrock doesn't support stream())."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = ClaudeAgent.create(
                model_client=bedrock_client,
                model="test-model-arn",
                validate_api_key=False,
            )

            # Enable computer tool to verify betas list includes computer-use in Bedrock mode.
            agent.has_computer_tool = True

            mock_response = MagicMock()
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Hello from Bedrock"
            mock_response.content = [text_block]

            bedrock_client.beta.messages.create.return_value = mock_response  # type: ignore[union-attr]

            messages = [
                cast(
                    "BetaMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                )
            ]
            response = await agent.get_response(messages)

            assert response.content == "Hello from Bedrock"
            assert response.tool_calls == []

            # Bedrock-specific behavior: uses create() and appends assistant message directly.
            assert not hasattr(bedrock_client.beta.messages, "stream")
            bedrock_client.beta.messages.create.assert_awaited_once()  # type: ignore[union-attr]
            assert len(messages) == 2
            assert messages[-1]["role"] == "assistant"

            # Ensure the Bedrock call shape is stable.
            _, kwargs = bedrock_client.beta.messages.create.call_args  # type: ignore[union-attr]
            assert kwargs["model"] == "test-model-arn"
            assert kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            assert "fine-grained-tool-streaming-2025-05-14" in kwargs["betas"]
            assert "computer-use-2025-01-24" in kwargs["betas"]

    @pytest.mark.asyncio
    async def test_get_response_bedrock_missing_boto3_raises_value_error(
        self, bedrock_client: AsyncAnthropicBedrock
    ) -> None:
        """If boto3 isn't installed, Bedrock client import path should raise a clear ValueError."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = ClaudeAgent.create(
                model_client=bedrock_client,
                model="test-model-arn",
                validate_api_key=False,
            )

            bedrock_client.beta.messages.create.side_effect = ModuleNotFoundError("boto3")  # type: ignore[union-attr]
            messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

            with pytest.raises(ValueError, match=r"boto3 is required for AWS Bedrock"):
                await agent.get_response(messages)  # type: ignore

    def test_init_with_bedrock_client_does_not_require_anthropic_api_key(
        self, bedrock_client: AsyncAnthropicBedrock
    ) -> None:
        """Providing model_client should bypass ANTHROPIC_API_KEY validation."""
        with patch("hud.settings.settings.anthropic_api_key", None):
            agent = ClaudeAgent.create(
                model_client=bedrock_client,
                validate_api_key=False,
            )
            assert agent.anthropic_client == bedrock_client

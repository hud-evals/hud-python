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

    from anthropic.types.beta import BetaMessageParam


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


class MockErrorStreamContextManager:
    """Mock stream context manager that raises a fixed error while streaming."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> MockErrorStreamContextManager:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> bool:
        return False

    def __aiter__(self) -> MockErrorStreamContextManager:
        return self

    async def __anext__(self) -> None:
        raise self.error

    async def get_final_message(self) -> MagicMock:
        raise AssertionError("get_final_message should not be called when stream iteration fails")


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
        content = [text_to_content_block("Result text")]

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

    @pytest.mark.asyncio
    async def test_get_response_retries_once_on_invalid_streamed_tool_json(
        self, mock_anthropic: AsyncAnthropic
    ) -> None:
        """Invalid streamed tool JSON should trigger one retry with wrapped payload."""
        invalid_json_error = ValueError(
            'Unable to parse tool parameter JSON from model. Please retry your request or adjust your '
            'prompt. Error: expected value at line 1 column 10. JSON: {"labels": bug}'
        )
        first_stream = MockErrorStreamContextManager(invalid_json_error)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Recovered")]
        second_stream = MockStreamContextManager(mock_response)

        mock_anthropic.beta.messages.stream = MagicMock(
            side_effect=[first_stream, second_stream]
        )

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent.has_computer_tool = False
        agent._initialized = True

        messages: list[BetaMessageParam] = [
            cast(
                "BetaMessageParam",
                {"role": "user", "content": [{"type": "text", "text": "Create a Linear ticket"}]},
            )
        ]

        response = await agent.get_response(messages)

        assert response.content == "Recovered"
        assert mock_anthropic.beta.messages.stream.call_count == 2
        # Original user message + retry guidance + assistant response
        assert len(messages) == 3
        retry_message = messages[1]
        assert retry_message["role"] == "user"
        retry_content = cast("list[dict[str, Any]]", retry_message["content"])
        assert "INVALID_JSON" in retry_content[0]["text"]

    @pytest.mark.asyncio
    async def test_get_response_does_not_retry_unrelated_value_error(
        self, mock_anthropic: AsyncAnthropic
    ) -> None:
        """Non-tool-json ValueErrors should propagate immediately."""
        unrelated_error = ValueError("stream exploded for unrelated reason")
        mock_anthropic.beta.messages.stream = MagicMock(
            return_value=MockErrorStreamContextManager(unrelated_error)
        )

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        agent.claude_tools = []
        agent.tool_mapping = {}
        agent.has_computer_tool = False
        agent._initialized = True

        with pytest.raises(ValueError, match="unrelated reason"):
            await agent.get_response([])

        assert mock_anthropic.beta.messages.stream.call_count == 1


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
            # In real usage, this beta is added by _convert_tools_for_claude when it detects
            # a computer tool. Here we manually set both flags to simulate that.
            agent.has_computer_tool = True
            agent._required_betas.add("computer-use-2025-01-24")

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


class TestClaudeAgentComputerTool20251124:
    """Test ClaudeAgent with the new computer_20251124 tool type."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        from unittest.mock import MagicMock

        return MagicMock(spec=["messages", "beta"])

    def test_build_native_tool_computer_20251124(self, mock_anthropic: Any) -> None:
        """Test that _build_native_tool handles computer_20251124."""
        from hud.tools.native_types import NativeToolSpec

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            model="claude-opus-4-6-20260101",
            validate_api_key=False,
        )

        spec = NativeToolSpec(
            api_type="computer_20251124",
            api_name="computer",
            beta="computer-use-2025-11-24",
            role="computer",
            extra={"display_width": 1920, "display_height": 1080},
        )

        tool = types.Tool(
            name="computer",
            description="Computer tool",
            inputSchema={},
        )

        result = cast("dict[str, Any]", agent._build_native_tool(tool, spec))
        assert result["type"] == "computer_20251124"
        assert result["name"] == "computer"
        assert result["display_width_px"] == 1920
        assert result["display_height_px"] == 1080
        assert result["enable_zoom"] is True

    def test_get_native_api_name_computer_20251124(self, mock_anthropic: Any) -> None:
        """Test that _get_native_api_name handles computer_20251124."""
        from hud.tools.native_types import NativeToolSpec

        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        spec = NativeToolSpec(api_type="computer_20251124", api_name="computer")
        assert agent._get_native_api_name(spec) == "computer"

    def test_legacy_fallback_opus_46_uses_new_computer(self, mock_anthropic: Any) -> None:
        """Test legacy fallback returns computer_20251124 for Opus 4.6 models."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            model="claude-opus-4-6-20260101",
            validate_api_key=False,
        )
        agent._available_tools = []

        legacy_tool = types.Tool(
            name="anthropic_computer",
            description="Old-style computer tool",
            inputSchema={"type": "object", "properties": {}},
        )

        spec = agent._legacy_native_spec_fallback(legacy_tool)
        assert spec is not None
        assert spec.api_type == "computer_20251124"
        assert spec.beta == "computer-use-2025-11-24"

    def test_legacy_fallback_sonnet4_uses_old_computer(self, mock_anthropic: Any) -> None:
        """Test legacy fallback returns computer_20250124 for non-Opus 4.5/4.6 models."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            model="claude-sonnet-4-20250514",
            validate_api_key=False,
        )
        agent._available_tools = []

        legacy_tool = types.Tool(
            name="anthropic_computer",
            description="Old-style computer tool",
            inputSchema={"type": "object", "properties": {}},
        )

        spec = agent._legacy_native_spec_fallback(legacy_tool)
        assert spec is not None
        assert spec.api_type == "computer_20250124"
        assert spec.beta == "computer-use-2025-01-24"

    def test_no_fine_grained_streaming_beta(self, mock_anthropic: Any) -> None:
        """Test that fine-grained-tool-streaming beta is no longer included."""
        agent = ClaudeAgent.create(
            model_client=mock_anthropic,
            validate_api_key=False,
        )
        assert "fine-grained-tool-streaming-2025-05-14" not in agent._required_betas


class TestClaudeAgentBetaHeader:
    """Test that the Anthropic-Beta header is handled correctly."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        return MagicMock(spec=["messages", "beta"])

    @pytest.mark.asyncio
    async def test_empty_betas_sends_omit_not_empty_list(self, mock_anthropic: Any) -> None:
        """When no tools require a beta, betas should be Omit() not []."""
        from anthropic import Omit

        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = ClaudeAgent.create(
                model_client=mock_anthropic,
                validate_api_key=False,
            )
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = False
            agent._required_betas = set()  # No betas required
            agent._initialized = True

            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Hello")]

            mock_stream = MockStreamContextManager(mock_response)
            mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

            messages = [
                cast(
                    "BetaMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                )
            ]
            await agent.get_response(messages)

            _, kwargs = mock_anthropic.beta.messages.stream.call_args
            assert isinstance(kwargs["betas"], Omit), (
                f"Expected Omit() when no betas required, got {type(kwargs['betas'])}"
            )

    @pytest.mark.asyncio
    async def test_nonempty_betas_sends_list(self, mock_anthropic: Any) -> None:
        """When tools require betas, betas should be a list of strings."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = ClaudeAgent.create(
                model_client=mock_anthropic,
                validate_api_key=False,
            )
            agent.claude_tools = []
            agent.tool_mapping = {}
            agent.has_computer_tool = True
            agent._required_betas = {"computer-use-2025-01-24"}
            agent._initialized = True

            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Hello")]

            mock_stream = MockStreamContextManager(mock_response)
            mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

            messages = [
                cast(
                    "BetaMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                )
            ]
            await agent.get_response(messages)

            _, kwargs = mock_anthropic.beta.messages.stream.call_args
            assert isinstance(kwargs["betas"], list)
            assert "computer-use-2025-01-24" in kwargs["betas"]

    @pytest.mark.asyncio
    async def test_generic_tools_only_no_beta_header(self, mock_anthropic: Any) -> None:
        """Generic function tools should not produce a beta header."""
        with patch("hud.settings.settings.telemetry_enabled", False):
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

            # Generic tools should not add any betas
            assert len(agent._required_betas) == 0

            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Hello")]

            mock_stream = MockStreamContextManager(mock_response)
            mock_anthropic.beta.messages.stream = MagicMock(return_value=mock_stream)

            from anthropic import Omit

            messages = [
                cast(
                    "BetaMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                )
            ]
            await agent.get_response(messages)

            _, kwargs = mock_anthropic.beta.messages.stream.call_args
            assert isinstance(kwargs["betas"], Omit)

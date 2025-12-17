"""Tests for OpenAI MCP Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

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
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from collections.abc import Generator


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


class TestOpenAIAgent:
    """Test OpenAIAgent class."""

    @pytest.fixture
    def mock_openai(self) -> Generator[AsyncOpenAI, None, None]:  # type: ignore[misc]
        """Create a stub OpenAI client."""
        with patch("hud.agents.openai.AsyncOpenAI") as mock_class:
            client = AsyncOpenAI(api_key="test", base_url="http://localhost")
            client.chat.completions.create = AsyncMock()
            client.responses.create = AsyncMock()
            mock_class.return_value = client
            yield client  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_openai: AsyncOpenAI) -> None:
        """Test agent initialization with provided client."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            model="gpt-4o",
            validate_api_key=False,
        )

        assert agent.model_name == "OpenAI"
        assert agent.config.model == "gpt-4o"
        assert agent.model == "gpt-4o"
        assert agent.openai_client == mock_openai
        assert agent.max_output_tokens is None
        assert agent.temperature is None

    @pytest.mark.asyncio
    async def test_init_with_parameters(self, mock_openai: AsyncOpenAI) -> None:
        """Test agent initialization with various parameters."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            model="gpt-4o",
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
    async def test_init_without_client_no_api_key(self) -> None:
        """Test agent initialization fails without API key."""
        with patch("hud.agents.openai.settings") as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIAgent.create()

    @pytest.mark.asyncio
    async def test_format_blocks_text_only(self, mock_openai: AsyncOpenAI) -> None:
        """Test formatting text content blocks."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
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
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][0]["text"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_format_blocks_with_image(self, mock_openai: AsyncOpenAI) -> None:
        """Test formatting image content blocks."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Look at this:"),
            types.ImageContent(type="image", data="base64data", mimeType="image/png"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][1]["type"] == "input_image"
        assert messages[0]["content"][1]["image_url"] == "data:image/png;base64,base64data"  # type: ignore[typeddict-item]

    @pytest.mark.asyncio
    async def test_format_blocks_empty(self, mock_openai: AsyncOpenAI) -> None:
        """Test formatting empty content blocks."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        messages = await agent.format_blocks([])
        assert len(messages) == 1
        # Empty blocks produce a single empty text item
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][0]["text"] == ""

    @pytest.mark.asyncio
    async def test_format_tool_results_text(self, mock_openai: AsyncOpenAI) -> None:
        """Test formatting tool results with text content."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
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
        assert messages[0]["type"] == "function_call_output"
        assert messages[0]["call_id"] == "call_123"
        # Output is a list of content items
        assert len(messages[0]["output"]) == 1
        assert messages[0]["output"][0]["text"] == "Tool output"  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_openai: AsyncOpenAI) -> None:
        """Test formatting tool results with error."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
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
        # Output is a list; first item is error indicator, second is the message
        msg = cast("dict[str, Any]", messages[0])
        output = cast("list[dict[str, Any]]", msg["output"])
        assert any(item.get("text") == "[tool_error] true" for item in output)
        assert any(item.get("text") == "Error message" for item in output)

    @pytest.mark.asyncio
    async def test_get_system_messages(self, mock_openai: AsyncOpenAI) -> None:
        """Test getting system messages - OpenAI uses instructions field instead."""
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            system_prompt="You are a helpful assistant.",
            validate_api_key=False,
        )

        # OpenAI agent returns empty list - system prompt is passed via instructions
        messages = await agent.get_system_messages()
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_convert_tools_for_openai(self, mock_openai: AsyncOpenAI) -> None:
        """Test converting MCP tools to OpenAI format."""
        tools = [
            types.Tool(
                name="my_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        # Initialize with context to trigger tool conversion
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Check that tools were converted
        assert len(agent._openai_tools) >= 1
        # Find our tool
        tool = next((t for t in agent._openai_tools if t.get("name") == "my_tool"), None)
        assert tool is not None
        assert tool["type"] == "function"

    @pytest.mark.asyncio
    async def test_convert_tools_raises_on_incomplete(self, mock_openai: AsyncOpenAI) -> None:
        """Test that tools without description raise error."""
        tools = [
            types.Tool(
                name="incomplete_tool",
                description=None,  # Missing description
                inputSchema={"type": "object"},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent.ctx = ctx
        with pytest.raises(ValueError, match="requires both a description"):
            await agent._initialize_from_ctx(ctx)

    @pytest.mark.asyncio
    async def test_get_response_with_text(self, mock_openai: AsyncOpenAI) -> None:
        """Test getting response with text output."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.output = [
            ResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Hello!", annotations=[])],
            )
        ]
        mock_openai.responses.create = AsyncMock(return_value=mock_response)

        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )
        # Set empty tools to avoid needing initialization
        agent._openai_tools = []
        agent._initialized = True

        response = await agent.get_response([])
        assert response.content == "Hello!"
        assert response.done is True
        assert len(response.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_get_response_with_tool_call(self, mock_openai: AsyncOpenAI) -> None:
        """Test getting response with tool call."""
        mock_response = AsyncMock()
        # Tool calls come as separate output items, not inside message content
        mock_response.output = [
            ResponseFunctionToolCall(
                id="call_123",
                type="function_call",
                call_id="call_123",
                name="my_tool",
                arguments='{"x": "value"}',
            )
        ]
        mock_openai.responses.create = AsyncMock(return_value=mock_response)

        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )
        agent._openai_tools = []
        agent._tool_name_map = {"my_tool": "my_tool"}
        agent._initialized = True

        response = await agent.get_response([])
        assert response.done is False
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "my_tool"
        assert response.tool_calls[0].arguments == {"x": "value"}

    @pytest.mark.asyncio
    async def test_get_response_with_reasoning(self, mock_openai: AsyncOpenAI) -> None:
        """Test getting response with reasoning."""
        mock_response = AsyncMock()
        mock_response.output = [
            ResponseReasoningItem(
                id="reason_123",
                type="reasoning",
                summary=[Summary(type="summary_text", text="Thinking about it...")],
            ),
            ResponseOutputMessage(
                id="msg_123",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Answer!", annotations=[])],
            ),
        ]
        mock_openai.responses.create = AsyncMock(return_value=mock_response)

        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )
        agent._openai_tools = []
        agent._initialized = True

        response = await agent.get_response([])
        # Reasoning is stored separately from content
        assert response.reasoning == "Thinking about it..."
        assert response.content == "Answer!"


class TestOpenAIToolConversion:
    """Tests for tool conversion to OpenAI format."""

    @pytest.fixture
    def mock_openai(self) -> Generator[AsyncOpenAI, None, None]:  # type: ignore[misc]
        """Create a stub OpenAI client."""
        with patch("hud.agents.openai.AsyncOpenAI") as mock_class:
            client = AsyncOpenAI(api_key="test", base_url="http://localhost")
            client.responses.create = AsyncMock()
            mock_class.return_value = client
            yield client  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_shell_tool_conversion(self, mock_openai: AsyncOpenAI) -> None:
        """Test that shell tool is converted to native format."""
        tools = [
            types.Tool(
                name="shell",
                description="Execute shell commands",
                inputSchema={"type": "object"},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Check for native shell tool
        shell_tool = next((t for t in agent._openai_tools if t.get("type") == "shell"), None)
        assert shell_tool is not None

    @pytest.mark.asyncio
    async def test_computer_tool_conversion(self, mock_openai: AsyncOpenAI) -> None:
        """Test that computer tool is converted to function format."""
        tools = [
            types.Tool(
                name="computer",
                description="Control computer",
                inputSchema={"type": "object"},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = OpenAIAgent.create(
            model_client=mock_openai,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Computer tool is converted to a regular function tool
        computer_tool = next(
            (t for t in agent._openai_tools if t.get("name") == "computer"),
            None,
        )
        assert computer_tool is not None
        assert computer_tool.get("type") == "function"

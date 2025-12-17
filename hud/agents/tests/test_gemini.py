"""Tests for Gemini MCP Agent implementation."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google import genai
from google.genai import types as genai_types
from mcp import types

from hud.agents.gemini import GeminiAgent
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import MCPToolCall, MCPToolResult


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


class TestGeminiAgent:
    """Test GeminiAgent base class."""

    @pytest.fixture
    def mock_gemini_client(self) -> MagicMock:
        """Create a stub Gemini client."""
        client = MagicMock(spec=genai.Client)
        client.api_key = "test_key"
        client.models = MagicMock()
        client.models.list = MagicMock(return_value=iter([]))
        client.models.generate_content = MagicMock()
        # Set up async interface (aio.models.generate_content)
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_init(self, mock_gemini_client: MagicMock) -> None:
        """Test agent initialization."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            model="gemini-2.5-flash",
            validate_api_key=False,
        )

        assert agent.model_name == "Gemini"
        assert agent.config.model == "gemini-2.5-flash"
        assert agent.gemini_client == mock_gemini_client

    @pytest.mark.asyncio
    async def test_init_without_model_client(self) -> None:
        """Test agent initialization without model client."""
        with (
            patch("hud.settings.settings.gemini_api_key", "test_key"),
            patch("hud.agents.gemini.genai.Client") as mock_client_class,
        ):
            mock_client = MagicMock()
            mock_client.api_key = "test_key"
            mock_client.models = MagicMock()
            mock_client.models.list = MagicMock(return_value=iter([]))
            mock_client_class.return_value = mock_client

            agent = GeminiAgent.create(
                model="gemini-2.5-flash",
                validate_api_key=False,
            )

            assert agent.gemini_client is not None

    @pytest.mark.asyncio
    async def test_format_blocks_text_only(self, mock_gemini_client: MagicMock) -> None:
        """Test formatting text content blocks."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, world!"),
            types.TextContent(type="text", text="How are you?"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].parts is not None
        assert len(messages[0].parts) == 2

    @pytest.mark.asyncio
    async def test_format_blocks_with_image(self, mock_gemini_client: MagicMock) -> None:
        """Test formatting image content blocks."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Create a tiny valid base64 PNG
        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()

        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Look at this:"),
            types.ImageContent(type="image", data=png_data, mimeType="image/png"),
        ]

        messages = await agent.format_blocks(blocks)
        assert len(messages) == 1
        assert messages[0].parts is not None
        assert len(messages[0].parts) == 2

    @pytest.mark.asyncio
    async def test_format_tool_results(self, mock_gemini_client: MagicMock) -> None:
        """Test formatting tool results."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
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
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_get_system_messages(self, mock_gemini_client: MagicMock) -> None:
        """Test that system messages return empty (Gemini uses system_instruction)."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            system_prompt="You are a helpful assistant.",
            validate_api_key=False,
        )

        messages = await agent.get_system_messages()
        # Gemini doesn't use system messages in the message list
        assert messages == []

    @pytest.mark.asyncio
    async def test_get_response_text_only(self, mock_gemini_client: MagicMock) -> None:
        """Test getting text-only response."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
                model_client=mock_gemini_client,
                validate_api_key=False,
            )
            # Set up agent as initialized (no tools needed for this test)
            agent.gemini_tools = []
            agent._initialized = True

            # Mock the API response with text only
            mock_response = MagicMock()
            mock_candidate = MagicMock()

            text_part = MagicMock()
            text_part.text = "Task completed successfully"
            text_part.function_call = None

            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [text_part]

            mock_response.candidates = [mock_candidate]

            mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            messages = [
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="Status?")])
            ]
            response = await agent.get_response(messages)

            assert response.content == "Task completed successfully"
            assert response.tool_calls == []
            assert response.done is True

    @pytest.mark.asyncio
    async def test_get_response_with_thinking(self, mock_gemini_client: MagicMock) -> None:
        """Test getting response with thinking content."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
                model_client=mock_gemini_client,
                validate_api_key=False,
            )
            # Set up agent as initialized (no tools needed for this test)
            agent.gemini_tools = []
            agent._initialized = True

            mock_response = MagicMock()
            mock_candidate = MagicMock()

            thinking_part = MagicMock()
            thinking_part.text = "Let me reason through this..."
            thinking_part.function_call = None
            thinking_part.thought = True

            text_part = MagicMock()
            text_part.text = "Here is my answer"
            text_part.function_call = None
            text_part.thought = False

            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [thinking_part, text_part]

            mock_response.candidates = [mock_candidate]

            mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            messages = [
                genai_types.Content(
                    role="user", parts=[genai_types.Part.from_text(text="Hard question")]
                )
            ]
            response = await agent.get_response(messages)

            assert response.content == "Here is my answer"
            assert response.reasoning == "Let me reason through this..."

    @pytest.mark.asyncio
    async def test_convert_tools_for_gemini(self, mock_gemini_client: MagicMock) -> None:
        """Test converting MCP tools to Gemini format."""
        tools = [
            types.Tool(
                name="my_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Check that tools were converted
        assert len(agent.gemini_tools) == 1
        # Gemini tools have function_declarations - cast to genai Tool type
        gemini_tool = agent.gemini_tools[0]
        assert isinstance(gemini_tool, genai_types.Tool)
        assert gemini_tool.function_declarations is not None
        assert gemini_tool.function_declarations[0].name == "my_tool"


class TestGeminiToolConversion:
    """Tests for tool conversion to Gemini format."""

    @pytest.fixture
    def mock_gemini_client(self) -> MagicMock:
        """Create a stub Gemini client."""
        client = MagicMock(spec=genai.Client)
        client.api_key = "test_key"
        client.models = MagicMock()
        client.models.list = MagicMock(return_value=iter([]))
        # Set up async interface
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_tool_with_properties(self, mock_gemini_client: MagicMock) -> None:
        """Test tool with input properties."""
        tools = [
            types.Tool(
                name="search",
                description="Search the web",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        assert len(agent.gemini_tools) == 1
        gemini_tool = agent.gemini_tools[0]
        # Gemini tools have function_declarations - cast to genai Tool type
        assert isinstance(gemini_tool, genai_types.Tool)
        assert gemini_tool.function_declarations is not None
        assert gemini_tool.function_declarations[0].name == "search"
        assert gemini_tool.function_declarations[0].parameters_json_schema is not None

    @pytest.mark.asyncio
    async def test_tool_without_schema(self, mock_gemini_client: MagicMock) -> None:
        """Test tool without description raises error."""
        # Create a tool with inputSchema but no description
        tools = [
            types.Tool(
                name="incomplete",
                description=None,
                inputSchema={"type": "object"},
            )
        ]
        ctx = MockEvalContext(tools=tools)
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        agent.ctx = ctx
        with pytest.raises(ValueError, match="requires both a description"):
            await agent._initialize_from_ctx(ctx)

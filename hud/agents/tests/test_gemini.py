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
        self._submitted: str | dict[str, Any] | None = None
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
        self.answer: str | dict[str, Any] | None = None
        self.system_prompt: str | None = None
        self.error: BaseException | None = None
        self.scenario_enable_citations: bool = False
        self.scenario_returns_schema: dict[str, Any] | None = None
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

    async def submit(self, answer: str | dict[str, Any]) -> None:
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
    async def test_get_response_raises_on_no_candidates(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """A no-candidate Gemini response should fail loudly, not submit an empty answer."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
                model_client=mock_gemini_client,
                model="gemini-3-flash-preview",
                validate_api_key=False,
            )
            agent.gemini_tools = []
            agent._initialized = True

            mock_response = MagicMock()
            mock_response.candidates = []
            mock_response.prompt_feedback = "blocked"
            mock_response.usage_metadata = None
            mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            messages = [
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="Status?")])
            ]

            with pytest.raises(RuntimeError, match="returned no candidates"):
                await agent.get_response(messages)

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
    async def test_get_response_passes_thinking_config(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """Gemini 3 thinking options should be passed to GenerateContentConfig."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
                model_client=mock_gemini_client,
                model="gemini-3-flash-preview",
                validate_api_key=False,
                thinking_level="high",
                include_thoughts=True,
            )
            agent.gemini_tools = []
            agent._initialized = True

            mock_response = MagicMock()
            mock_candidate = MagicMock()
            text_part = MagicMock()
            text_part.text = "Answer"
            text_part.function_call = None
            text_part.thought = False
            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [text_part]
            mock_response.candidates = [mock_candidate]

            mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            messages = [
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="Hi")])
            ]
            await agent.get_response(messages)

            config = mock_gemini_client.aio.models.generate_content.call_args.kwargs["config"]
            assert config.thinking_config is not None
            assert config.thinking_config.include_thoughts is True
            assert config.thinking_config.thinking_level.value == "HIGH"

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

    @pytest.mark.asyncio
    async def test_regular_agent_uses_native_computer_use(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """GeminiAgent should register GeminiComputerTool as native Computer Use."""
        computer_tool = types.Tool(
            name="gemini_computer",
            description="Control computer with mouse, keyboard, and screenshots",
            inputSchema={"type": "object", "properties": {}},
        )
        computer_tool.meta = {
            "native_tools": {
                "gemini": {
                    "api_type": "computer_use",
                    "api_name": "gemini_computer",
                    "role": "computer",
                    "supported_models": ["gemini-3-flash-preview"],
                }
            }
        }
        tools = [
            computer_tool,
        ]
        ctx = MockEvalContext(tools=tools)
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            model="gemini-3-flash-preview",
            validate_api_key=False,
            excluded_predefined_functions=["drag_and_drop"],
        )

        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        assert agent._computer_tool_name == "gemini_computer"
        assert len(agent.gemini_tools) == 1
        computer_tool = agent.gemini_tools[0]
        assert isinstance(computer_tool, genai_types.Tool)
        assert computer_tool.computer_use is not None
        assert computer_tool.computer_use.excluded_predefined_functions == ["drag_and_drop"]

    def test_regular_agent_routes_computer_use_function_call(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """Gemini Computer Use calls should route to the MCP computer tool."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )
        agent._computer_tool_name = "gemini_computer"

        function_call = MagicMock()
        function_call.name = "click_at"
        function_call.args = {"x": 500, "y": 250, "safety_decision": {"decision": "allowed"}}
        part = MagicMock(function_call=function_call)

        tool_call = agent._extract_tool_call(part)

        assert tool_call is not None
        assert tool_call.name == "gemini_computer"
        assert tool_call.arguments == {
            "action": "click_at",
            "safety_decision": {"decision": "allowed"},
            "x": 500,
            "y": 250,
        }
        assert getattr(tool_call, "gemini_name") == "click_at"

    @pytest.mark.asyncio
    async def test_regular_agent_formats_computer_use_results(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """GeminiAgent should return URL and screenshot parts for native computer use."""
        agent = GeminiAgent.create(
            model_client=mock_gemini_client,
            validate_api_key=False,
        )
        agent._computer_tool_name = "gemini_computer"
        screenshot = base64.b64encode(b"png bytes").decode()
        tool_calls = [
            MCPToolCall(
                name="gemini_computer",
                arguments={"action": "click_at", "safety_decision": {"decision": "allowed"}},
                gemini_name="click_at",  # type: ignore[arg-type]
            )
        ]
        tool_results = [
            MCPToolResult(
                content=[
                    types.TextContent(type="text", text="__URL__:https://example.com"),
                    types.ImageContent(type="image", data=screenshot, mimeType="image/png"),
                ],
                isError=False,
            )
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        parts = messages[0].parts
        assert parts is not None
        function_response = parts[0].function_response
        assert function_response is not None
        assert function_response.name == "click_at"
        response = function_response.response
        assert response is not None
        assert response["url"] == "https://example.com"
        assert response["safety_acknowledgement"] is True
        assert function_response.parts is not None
        inline_data = function_response.parts[0].inline_data
        assert inline_data is not None
        assert inline_data.mime_type == "image/png"


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


class TestGeminiCitations:
    """Tests for Gemini grounding citation extraction."""

    @pytest.fixture
    def mock_gemini_client(self) -> MagicMock:
        client = MagicMock(spec=genai.Client)
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock()
        return client

    def _make_agent(self, client: MagicMock) -> GeminiAgent:
        agent = GeminiAgent.create(model_client=client, validate_api_key=False)
        agent.gemini_tools = []
        agent._initialized = True
        return agent

    def _text_candidate(self, text: str = "answer") -> MagicMock:
        candidate = MagicMock()
        part = MagicMock()
        part.text = text
        part.function_call = None
        part.thought = False
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        return candidate

    @pytest.mark.asyncio
    async def test_no_grounding_metadata(self, mock_gemini_client: MagicMock) -> None:
        """No citations when groundingMetadata is absent."""
        agent = self._make_agent(mock_gemini_client)
        candidate = self._text_candidate()
        candidate.grounding_metadata = None
        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        result = await agent.get_response([])
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_grounding_chunks_only(self, mock_gemini_client: MagicMock) -> None:
        """Chunks without supports produce citations with source but no anchoring."""
        agent = self._make_agent(mock_gemini_client)
        candidate = self._text_candidate()

        chunk = MagicMock()
        chunk.web = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"

        grounding_meta = MagicMock()
        grounding_meta.grounding_chunks = [chunk]
        grounding_meta.grounding_supports = []
        candidate.grounding_metadata = grounding_meta

        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        result = await agent.get_response([])
        assert len(result.citations) == 1
        assert result.citations[0]["source"] == "https://example.com"
        assert result.citations[0]["title"] == "Example"
        assert result.citations[0]["text"] == ""

    @pytest.mark.asyncio
    async def test_grounding_supports_with_anchoring(self, mock_gemini_client: MagicMock) -> None:
        """Supports produce citations with start_index/end_index from segments."""
        agent = self._make_agent(mock_gemini_client)
        candidate = self._text_candidate("The sky is blue because of Rayleigh scattering.")

        chunk = MagicMock()
        chunk.web = MagicMock()
        chunk.web.uri = "https://physics.org/scattering"
        chunk.web.title = "Scattering"

        support = MagicMock()
        support.segment = MagicMock()
        support.segment.text = "Rayleigh scattering"
        support.segment.start_index = 28
        support.segment.end_index = 47
        support.grounding_chunk_indices = [0]

        grounding_meta = MagicMock()
        grounding_meta.grounding_chunks = [chunk]
        grounding_meta.grounding_supports = [support]
        candidate.grounding_metadata = grounding_meta

        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        result = await agent.get_response([])
        assert len(result.citations) == 1
        cit = result.citations[0]
        assert cit["type"] == "grounding"
        assert cit["text"] == "Rayleigh scattering"
        assert cit["source"] == "https://physics.org/scattering"
        assert cit["start_index"] == 28
        assert cit["end_index"] == 47

    @pytest.mark.asyncio
    async def test_multiple_supports_and_chunks(self, mock_gemini_client: MagicMock) -> None:
        """Multiple supports across multiple chunks produce the right citations."""
        agent = self._make_agent(mock_gemini_client)
        candidate = self._text_candidate()

        chunk_a = MagicMock()
        chunk_a.web = MagicMock()
        chunk_a.web.uri = "https://a.com"
        chunk_a.web.title = "A"

        chunk_b = MagicMock()
        chunk_b.web = MagicMock()
        chunk_b.web.uri = "https://b.com"
        chunk_b.web.title = "B"

        support1 = MagicMock()
        support1.segment = MagicMock()
        support1.segment.text = "fact one"
        support1.segment.start_index = 0
        support1.segment.end_index = 8
        support1.grounding_chunk_indices = [0]

        support2 = MagicMock()
        support2.segment = MagicMock()
        support2.segment.text = "fact two"
        support2.segment.start_index = 10
        support2.segment.end_index = 18
        support2.grounding_chunk_indices = [1]

        grounding_meta = MagicMock()
        grounding_meta.grounding_chunks = [chunk_a, chunk_b]
        grounding_meta.grounding_supports = [support1, support2]
        candidate.grounding_metadata = grounding_meta

        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        result = await agent.get_response([])
        assert len(result.citations) == 2
        assert result.citations[0]["source"] == "https://a.com"
        assert result.citations[0]["text"] == "fact one"
        assert result.citations[1]["source"] == "https://b.com"
        assert result.citations[1]["text"] == "fact two"


class TestGeminiCitationInjection:
    """Test that enable_citations injects google_search when missing."""

    @pytest.fixture
    def mock_gemini_client(self) -> MagicMock:
        client = MagicMock(spec=genai.Client)
        client.aio = MagicMock()
        client.aio.models = MagicMock()
        client.aio.models.generate_content = AsyncMock()
        return client

    def _make_agent(self, client: MagicMock) -> GeminiAgent:
        agent = GeminiAgent.create(model_client=client, validate_api_key=False)
        agent.gemini_tools = []
        agent._gemini_to_mcp_tool_map = {}
        agent._initialized = True
        return agent

    @pytest.mark.asyncio
    async def test_google_search_injected_when_citations_enabled(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """When scenario_enable_citations=True and no google_search tool, inject one."""
        agent = self._make_agent(mock_gemini_client)
        ctx = MockEvalContext()
        ctx.scenario_enable_citations = True
        agent.ctx = ctx

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [MagicMock(function_call=None, thought=False, text="Hi")]
        candidate.grounding_metadata = None
        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        await agent.get_response([])

        call_kwargs = mock_gemini_client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        tools_passed = config.tools
        assert any(
            isinstance(t, genai_types.Tool) and t.google_search is not None for t in tools_passed
        )

    @pytest.mark.asyncio
    async def test_no_duplicate_google_search_when_already_present(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """When google_search tool already exists, don't add a second one."""
        agent = self._make_agent(mock_gemini_client)
        existing_search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
        agent.gemini_tools = [existing_search_tool]
        ctx = MockEvalContext()
        ctx.scenario_enable_citations = True
        agent.ctx = ctx

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [MagicMock(function_call=None, thought=False, text="Hi")]
        candidate.grounding_metadata = None
        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        await agent.get_response([])

        call_kwargs = mock_gemini_client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        tools_passed = config.tools
        search_count = sum(
            1
            for t in tools_passed
            if isinstance(t, genai_types.Tool) and t.google_search is not None
        )
        assert search_count == 1

    @pytest.mark.asyncio
    async def test_no_injection_when_citations_disabled(
        self, mock_gemini_client: MagicMock
    ) -> None:
        """When scenario_enable_citations=False, no google_search is injected."""
        agent = self._make_agent(mock_gemini_client)
        ctx = MockEvalContext()
        ctx.scenario_enable_citations = False
        agent.ctx = ctx

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [MagicMock(function_call=None, thought=False, text="Hi")]
        candidate.grounding_metadata = None
        resp = MagicMock()
        resp.candidates = [candidate]
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=resp)

        await agent.get_response([])

        call_kwargs = mock_gemini_client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        tools_passed = config.tools
        assert not any(
            isinstance(t, genai_types.Tool) and t.google_search is not None for t in tools_passed
        )

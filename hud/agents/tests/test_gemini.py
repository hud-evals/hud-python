"""Tests for Gemini MCP Agent implementation."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types as genai_types
from mcp import types

from hud.agents.gemini import GeminiAgent
from hud.types import MCPToolCall, MCPToolResult


class TestGeminiAgent:
    """Test GeminiAgent class."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        mcp_client = AsyncMock()
        # Set up the mcp_config attribute as a regular dict, not a coroutine
        mcp_client.mcp_config = {"test_server": {"url": "http://test"}}
        # Mock list_tools to return gemini_computer tool
        mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(
                    name="gemini_computer",
                    description="Gemini computer use tool",
                    inputSchema={},
                )
            ]
        )
        mcp_client.initialize = AsyncMock()
        return mcp_client

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock Gemini client."""
        client = MagicMock()
        client.api_key = "test_key"
        # Mock models.list for validation
        client.models = MagicMock()
        client.models.list = MagicMock(return_value=iter([]))
        return client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client, mock_gemini_client):
        """Test agent initialization."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            model="gemini-2.5-computer-use-preview-10-2025",
            validate_api_key=False,  # Skip validation in tests
        )

        assert agent.model_name == "gemini-2.5-computer-use-preview-10-2025"
        assert agent.model == "gemini-2.5-computer-use-preview-10-2025"
        assert agent.gemini_client == mock_gemini_client

    @pytest.mark.asyncio
    async def test_init_without_model_client(self, mock_mcp_client):
        """Test agent initialization without model client."""
        with patch("hud.settings.settings.gemini_api_key", "test_key"):
            with patch("hud.agents.gemini.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.api_key = "test_key"
                mock_client.models = MagicMock()
                mock_client.models.list = MagicMock(return_value=iter([]))
                mock_client_class.return_value = mock_client

                agent = GeminiAgent(
                    mcp_client=mock_mcp_client,
                    model="gemini-2.5-computer-use-preview-10-2025",
                    validate_api_key=False,
                )

                assert agent.model_name == "gemini-2.5-computer-use-preview-10-2025"
                assert agent.gemini_client is not None

    @pytest.mark.asyncio
    async def test_format_blocks(self, mock_mcp_client, mock_gemini_client):
        """Test formatting content blocks into Gemini messages."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Test with text only
        text_blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Hello, Gemini!")
        ]
        messages = await agent.format_blocks(text_blocks)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert len(messages[0].parts) == 1
        assert messages[0].parts[0].text == "Hello, Gemini!"

        # Test with screenshot
        image_blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Look at this"),
            types.ImageContent(
                type="image",
                data=base64.b64encode(b"fakeimage").decode("utf-8"),
                mimeType="image/png",
            ),
        ]
        messages = await agent.format_blocks(image_blocks)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert len(messages[0].parts) == 2
        # First part is text
        assert messages[0].parts[0].text == "Look at this"
        # Second part is image - check that it was created from bytes
        assert messages[0].parts[1].inline_data is not None

    @pytest.mark.asyncio
    async def test_format_tool_results(self, mock_mcp_client, mock_gemini_client):
        """Test the agent's format_tool_results method."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(
                name="gemini_computer",
                arguments={"action": "click_at", "x": 100, "y": 200},
                id="call_1",  # type: ignore
                gemini_name="click_at",  # type: ignore
            ),
        ]

        tool_results = [
            MCPToolResult(
                content=[
                    types.TextContent(type="text", text="Clicked successfully"),
                    types.ImageContent(
                        type="image",
                        data=base64.b64encode(b"screenshot").decode("utf-8"),
                        mimeType="image/png",
                    ),
                ],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # format_tool_results returns a single user message with function responses
        assert len(messages) == 1
        assert messages[0].role == "user"
        # The content contains function response parts
        assert len(messages[0].parts) == 1
        assert messages[0].parts[0].function_response is not None
        assert messages[0].parts[0].function_response.name == "click_at"
        assert messages[0].parts[0].function_response.response.get("success") is True

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_mcp_client, mock_gemini_client):
        """Test formatting tool results with errors."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(
                name="gemini_computer",
                arguments={"action": "invalid"},
                id="call_error",  # type: ignore
                gemini_name="invalid_action",  # type: ignore
            ),
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Action failed: invalid action")],
                isError=True,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # Check that error is in the response
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].parts[0].function_response is not None
        assert "error" in messages[0].parts[0].function_response.response

    @pytest.mark.asyncio
    async def test_get_response(self, mock_mcp_client, mock_gemini_client):
        """Test getting model response from Gemini API."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_gemini_client,
                validate_api_key=False,
            )

            # Set up available tools
            agent._available_tools = [
                types.Tool(
                    name="gemini_computer", description="Computer tool", inputSchema={}
                )
            ]

            # Mock the API response
            mock_response = MagicMock()
            mock_candidate = MagicMock()

            # Create text part
            text_part = MagicMock()
            text_part.text = "I will click at coordinates"
            text_part.function_call = None

            # Create function call part
            function_call_part = MagicMock()
            function_call_part.text = None
            function_call_part.function_call = MagicMock()
            function_call_part.function_call.name = "click_at"
            function_call_part.function_call.args = {"x": 100, "y": 200}

            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [text_part, function_call_part]

            mock_response.candidates = [mock_candidate]

            mock_gemini_client.models = MagicMock()
            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [genai_types.Content(role="user", parts=[genai_types.Part(text="Click")])]
            response = await agent.get_response(messages)

            assert response.content == "I will click at coordinates"
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].arguments == {"action": "click_at", "x": 100, "y": 200}
            assert response.done is False

    @pytest.mark.asyncio
    async def test_get_response_text_only(self, mock_mcp_client, mock_gemini_client):
        """Test getting text-only response."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_gemini_client,
                validate_api_key=False,
            )

            # Mock the API response with text only
            mock_response = MagicMock()
            mock_candidate = MagicMock()

            text_part = MagicMock()
            text_part.text = "Task completed successfully"
            text_part.function_call = None

            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [text_part]

            mock_response.candidates = [mock_candidate]

            mock_gemini_client.models = MagicMock()
            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [
                genai_types.Content(role="user", parts=[genai_types.Part(text="Status?")])
            ]
            response = await agent.get_response(messages)

            assert response.content == "Task completed successfully"
            assert response.tool_calls == []
            assert response.done is True

    @pytest.mark.asyncio
    async def test_convert_tools_for_gemini(self, mock_mcp_client, mock_gemini_client):
        """Test converting MCP tools to Gemini format."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Set up available tools
        agent._available_tools = [
            types.Tool(
                name="gemini_computer",
                description="Computer tool",
                inputSchema={"type": "object"},
            ),
            types.Tool(
                name="calculator",
                description="Calculator tool",
                inputSchema={
                    "type": "object",
                    "properties": {"operation": {"type": "string"}},
                },
            ),
        ]

        gemini_tools = agent._convert_tools_for_gemini()

        # Should have 2 tools: computer_use and calculator
        assert len(gemini_tools) == 2

        # First should be computer use tool
        assert gemini_tools[0].computer_use is not None
        assert gemini_tools[0].computer_use.environment == genai_types.Environment.ENVIRONMENT_BROWSER

        # Second should be calculator as function declaration
        assert gemini_tools[1].function_declarations is not None
        assert len(gemini_tools[1].function_declarations) == 1
        assert gemini_tools[1].function_declarations[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_create_user_message(self, mock_mcp_client, mock_gemini_client):
        """Test creating a user message."""
        agent = GeminiAgent(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        message = await agent.create_user_message("Hello Gemini")

        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello Gemini"

    @pytest.mark.asyncio
    async def test_handle_empty_response(self, mock_mcp_client, mock_gemini_client):
        """Test handling empty response from API."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent(
                mcp_client=mock_mcp_client,
                model_client=mock_gemini_client,
                validate_api_key=False,
            )

            # Mock empty response
            mock_response = MagicMock()
            mock_response.candidates = []

            mock_gemini_client.models = MagicMock()
            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [genai_types.Content(role="user", parts=[genai_types.Part(text="Hi")])]
            response = await agent.get_response(messages)

            assert response.content == ""
            assert response.tool_calls == []
            assert response.done is True

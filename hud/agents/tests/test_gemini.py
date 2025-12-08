"""Tests for Gemini MCP Agent implementation."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from google import genai
from google.genai import types as genai_types
from mcp import types

from hud.agents.gemini import GeminiAgent
from hud.agents.gemini_cua import GeminiCUAAgent
from hud.types import MCPToolCall, MCPToolResult


class TestGeminiAgent:
    """Test GeminiAgent base class."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a stub Gemini client."""
        client = genai.Client(api_key="test_key")
        client.models.list = MagicMock(return_value=iter([]))
        client.models.generate_content = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client, mock_gemini_client):
        """Test agent initialization."""
        agent = GeminiAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            checkpoint_name="gemini-2.5-flash",
            validate_api_key=False,  # Skip validation in tests
        )

        assert agent.model_name == "Gemini"
        assert agent.config.checkpoint_name == "gemini-2.5-flash"
        assert agent.gemini_client == mock_gemini_client

    @pytest.mark.asyncio
    async def test_init_without_model_client(self, mock_mcp_client):
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
                mcp_client=mock_mcp_client,
                checkpoint_name="gemini-2.5-flash",
                validate_api_key=False,
            )

            assert agent.model_name == "Gemini"
            assert agent.gemini_client is not None

    @pytest.mark.asyncio
    async def test_format_blocks(self, mock_mcp_client, mock_gemini_client):
        """Test formatting content blocks into Gemini messages."""
        agent = GeminiAgent.create(
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
        parts = messages[0].parts
        assert parts is not None
        assert len(parts) == 1
        assert parts[0].text == "Hello, Gemini!"

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
        parts = messages[0].parts
        assert parts is not None
        assert len(parts) == 2
        # First part is text
        assert parts[0].text == "Look at this"
        # Second part is image - check that it was created from bytes
        assert parts[1].inline_data is not None

    @pytest.mark.asyncio
    async def test_format_tool_results(self, mock_mcp_client, mock_gemini_client):
        """Test the agent's format_tool_results method for non-computer tools."""
        agent = GeminiAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(
                name="calculator",
                arguments={"operation": "add", "a": 1, "b": 2},
                id="call_1",  # type: ignore
                gemini_name="calculator",  # type: ignore
            ),
        ]

        tool_results = [
            MCPToolResult(
                content=[
                    types.TextContent(type="text", text="Result: 3"),
                ],
                isError=False,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # format_tool_results returns a single user message with function responses
        assert len(messages) == 1
        assert messages[0].role == "user"
        # The content contains function response parts
        parts = messages[0].parts
        assert parts is not None
        assert len(parts) == 1
        function_response = parts[0].function_response
        assert function_response is not None
        assert function_response.name == "calculator"
        response_payload = function_response.response or {}
        assert response_payload.get("success") is True
        assert response_payload.get("output") == "Result: 3"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(self, mock_mcp_client, mock_gemini_client):
        """Test formatting tool results with errors."""
        agent = GeminiAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(
                name="calculator",
                arguments={"operation": "divide", "a": 1, "b": 0},
                id="call_error",  # type: ignore
                gemini_name="calculator",  # type: ignore
            ),
        ]

        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Division by zero error")],
                isError=True,
            ),
        ]

        messages = await agent.format_tool_results(tool_calls, tool_results)

        # Check that error is in the response
        assert len(messages) == 1
        assert messages[0].role == "user"
        parts = messages[0].parts
        assert parts is not None
        function_response = parts[0].function_response
        assert function_response is not None
        response_payload = function_response.response or {}
        assert "error" in response_payload

    @pytest.mark.asyncio
    async def test_get_response_text_only(self, mock_mcp_client, mock_gemini_client):
        """Test getting text-only response."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
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

            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [genai_types.Content(role="user", parts=[genai_types.Part(text="Status?")])]
            response = await agent.get_response(messages)

            assert response.content == "Task completed successfully"
            assert response.tool_calls == []
            assert response.done is True

    @pytest.mark.asyncio
    async def test_convert_tools_for_gemini(self, mock_mcp_client, mock_gemini_client):
        """Test converting MCP tools to Gemini format."""
        agent = GeminiAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Set up available tools (no computer tool for base agent)
        agent._available_tools = [
            types.Tool(
                name="calculator",
                description="Calculator tool",
                inputSchema={
                    "type": "object",
                    "properties": {"operation": {"type": "string"}},
                },
            ),
            types.Tool(
                name="weather",
                description="Weather tool",
                inputSchema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            ),
        ]

        gemini_tools = agent._convert_tools_for_gemini()

        # Should have 2 function declaration tools
        assert len(gemini_tools) == 2

        # Both should be function declarations
        assert gemini_tools[0].function_declarations is not None  # type: ignore[reportAttributeAccessIssue]
        assert len(gemini_tools[0].function_declarations) == 1  # type: ignore[reportAttributeAccessIssue]
        assert gemini_tools[0].function_declarations[0].name == "calculator"  # type: ignore[reportAttributeAccessIssue]

        assert gemini_tools[1].function_declarations is not None  # type: ignore[reportAttributeAccessIssue]
        assert len(gemini_tools[1].function_declarations) == 1  # type: ignore[reportAttributeAccessIssue]
        assert gemini_tools[1].function_declarations[0].name == "weather"  # type: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_create_user_message(self, mock_mcp_client, mock_gemini_client):
        """Test creating a user message."""
        agent = GeminiAgent.create(
            mcp_client=mock_mcp_client,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        message = await agent.create_user_message("Hello Gemini")

        assert message.role == "user"
        parts = message.parts
        assert parts is not None
        assert len(parts) == 1
        assert parts[0].text == "Hello Gemini"

    @pytest.mark.asyncio
    async def test_handle_empty_response(self, mock_mcp_client, mock_gemini_client):
        """Test handling empty response from API."""
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiAgent.create(
                mcp_client=mock_mcp_client,
                model_client=mock_gemini_client,
                validate_api_key=False,
            )

            # Mock empty response
            mock_response = MagicMock()
            mock_response.candidates = []

            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [genai_types.Content(role="user", parts=[genai_types.Part(text="Hi")])]
            response = await agent.get_response(messages)

            assert response.content == ""
            assert response.tool_calls == []
            assert response.done is True


class TestGeminiCUAAgent:
    """Test GeminiCUAAgent computer use agent."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a stub Gemini client."""
        client = genai.Client(api_key="test_key")
        client.models.list = MagicMock(return_value=iter([]))
        client.models.generate_content = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_init(self, mock_mcp_client_gemini_computer, mock_gemini_client):
        """Test agent initialization."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
            model_client=mock_gemini_client,
            checkpoint_name="gemini-2.5-computer-use-preview",
            validate_api_key=False,  # Skip validation in tests
        )

        assert agent.model_name == "GeminiCUA"
        assert agent.config.checkpoint_name == "gemini-2.5-computer-use-preview"
        assert agent.gemini_client == mock_gemini_client

    @pytest.mark.asyncio
    async def test_format_tool_results_with_screenshot(
        self, mock_mcp_client_gemini_computer, mock_gemini_client
    ):
        """Test the agent's format_tool_results method with screenshots."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
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
                    types.TextContent(type="text", text="__URL__:https://example.com"),
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
        parts = messages[0].parts
        assert parts is not None
        assert len(parts) == 1
        function_response = parts[0].function_response
        assert function_response is not None
        assert function_response.name == "click_at"
        response_payload = function_response.response or {}
        assert response_payload.get("success") is True
        assert response_payload.get("url") == "https://example.com"

    @pytest.mark.asyncio
    async def test_format_tool_results_with_error(
        self, mock_mcp_client_gemini_computer, mock_gemini_client
    ):
        """Test formatting tool results with errors."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        tool_calls = [
            MCPToolCall(
                name="gemini_computer",
                arguments={"action": "invalid"},
                id="call_error",  # type: ignore
                gemini_name="click_at",  # type: ignore
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
        parts = messages[0].parts
        assert parts is not None
        function_response = parts[0].function_response
        assert function_response is not None
        response_payload = function_response.response or {}
        assert "error" in response_payload

    @pytest.mark.asyncio
    async def test_get_response(self, mock_mcp_client_gemini_computer, mock_gemini_client):
        """Test getting model response from Gemini API."""
        # Disable telemetry for this test
        with patch("hud.settings.settings.telemetry_enabled", False):
            agent = GeminiCUAAgent.create(
                mcp_client=mock_mcp_client_gemini_computer,
                model_client=mock_gemini_client,
                validate_api_key=False,
            )

            # Set up available tools
            agent._available_tools = [
                types.Tool(name="gemini_computer", description="Computer tool", inputSchema={})
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
            function_call_part.function_call.args = {"coordinate": [100, 200]}

            mock_candidate.content = MagicMock()
            mock_candidate.content.parts = [text_part, function_call_part]

            mock_response.candidates = [mock_candidate]

            mock_gemini_client.models.generate_content = MagicMock(return_value=mock_response)

            messages = [genai_types.Content(role="user", parts=[genai_types.Part(text="Click")])]
            response = await agent.get_response(messages)

            assert response.content == "I will click at coordinates"
            assert len(response.tool_calls) == 1
            # Check normalized arguments
            assert response.tool_calls[0].arguments == {"action": "click_at", "x": 100, "y": 200}
            assert response.done is False

    @pytest.mark.asyncio
    async def test_convert_tools_for_gemini(
        self, mock_mcp_client_gemini_computer, mock_gemini_client
    ):
        """Test converting MCP tools to Gemini format."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
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
        assert gemini_tools[0].computer_use is not None  # type: ignore[reportAttributeAccessIssue]
        assert (
            gemini_tools[0].computer_use.environment == genai_types.Environment.ENVIRONMENT_BROWSER  # type: ignore[reportAttributeAccessIssue]
        )

        # Second should be calculator as function declaration
        assert gemini_tools[1].function_declarations is not None  # type: ignore[reportAttributeAccessIssue]
        assert len(gemini_tools[1].function_declarations) == 1  # type: ignore[reportAttributeAccessIssue]
        assert gemini_tools[1].function_declarations[0].name == "calculator"  # type: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_extract_tool_call_normalizes_coordinates(
        self, mock_mcp_client_gemini_computer, mock_gemini_client
    ):
        """Test that _extract_tool_call normalizes coordinate arrays to x/y."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Set up tool mapping
        agent._gemini_to_mcp_tool_map = {"click_at": "gemini_computer"}

        # Create a mock part with function call
        part = MagicMock()
        part.function_call = MagicMock()
        part.function_call.name = "click_at"
        part.function_call.args = {"coordinate": [150, 250]}

        tool_call = agent._extract_tool_call(part)

        assert tool_call is not None
        assert tool_call.name == "gemini_computer"
        assert tool_call.arguments["action"] == "click_at"  # type: ignore[reportAttributeAccessIssue]
        assert tool_call.arguments["x"] == 150  # type: ignore[reportAttributeAccessIssue]
        assert tool_call.arguments["y"] == 250  # type: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_extract_tool_call_preserves_non_computer_args(
        self, mock_mcp_client_gemini_computer, mock_gemini_client
    ):
        """Test that _extract_tool_call preserves arguments for non-computer tools."""
        agent = GeminiCUAAgent.create(
            mcp_client=mock_mcp_client_gemini_computer,
            model_client=mock_gemini_client,
            validate_api_key=False,
        )

        # Set up tool mapping
        agent._gemini_to_mcp_tool_map = {"calculator": "calculator"}

        # Create a mock part with function call for non-computer tool
        part = MagicMock()
        part.function_call = MagicMock()
        part.function_call.name = "calculator"
        part.function_call.args = {"operation": "add", "a": 1, "b": 2}

        tool_call = agent._extract_tool_call(part)

        assert tool_call is not None
        assert tool_call.name == "calculator"
        # Arguments should be passed as-is, no normalization
        assert tool_call.arguments == {"operation": "add", "a": 1, "b": 2}

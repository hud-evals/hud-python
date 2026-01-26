"""Tests for format integrations - OpenAI, Anthropic, Gemini."""

from __future__ import annotations

from typing import Any

import mcp.types as mcp_types


def create_mock_tool(
    name: str, description: str = "", schema: dict | None = None
) -> mcp_types.Tool:
    """Create a mock MCP tool for testing."""
    return mcp_types.Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


class TestOpenAIMixin:
    """Tests for OpenAI format conversion."""

    def test_as_openai_chat_tools_basic(self) -> None:
        """as_openai_chat_tools converts MCP tools to OpenAI format."""
        from hud.environment.integrations.openai import OpenAIMixin

        class TestEnv(OpenAIMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [
                    create_mock_tool(
                        "navigate",
                        "Navigate to URL",
                        {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"],
                        },
                    ),
                ]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_openai_chat_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "navigate"  # type: ignore[typeddict-item]
        assert tools[0]["function"]["description"] == "Navigate to URL"  # type: ignore[typeddict-item]
        assert "url" in tools[0]["function"]["parameters"]["properties"]  # type: ignore[typeddict-item, operator]

    def test_as_openai_chat_tools_strict_mode(self) -> None:
        """as_openai_chat_tools with strict=True adds strict flag."""
        from hud.environment.integrations.openai import OpenAIMixin

        class TestEnv(OpenAIMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [create_mock_tool("test_tool")]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_openai_chat_tools(strict=True)

        assert tools[0]["function"]["strict"] is True  # type: ignore[typeddict-item]

    def test_as_openai_chat_tools_empty(self) -> None:
        """as_openai_chat_tools returns empty list when no tools."""
        from hud.environment.integrations.openai import OpenAIMixin

        class TestEnv(OpenAIMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return []

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_openai_chat_tools()

        assert tools == []

    def test_as_openai_responses_tools(self) -> None:
        """as_openai_responses_tools converts to Responses API format."""
        from hud.environment.integrations.openai import OpenAIMixin

        class TestEnv(OpenAIMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [create_mock_tool("search", "Search the web")]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_openai_responses_tools()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"


class TestAnthropicMixin:
    """Tests for Anthropic/Claude format conversion."""

    def test_as_claude_tools_basic(self) -> None:
        """as_claude_tools converts MCP tools to Claude format."""
        from hud.environment.integrations.anthropic import AnthropicMixin

        class TestEnv(AnthropicMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [
                    create_mock_tool(
                        "click",
                        "Click element",
                        {
                            "type": "object",
                            "properties": {"selector": {"type": "string"}},
                        },
                    ),
                ]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_claude_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "click"
        assert tools[0]["description"] == "Click element"
        assert "input_schema" in tools[0]
        assert "cache_control" not in tools[0]

    def test_as_claude_tools_with_cache_control(self) -> None:
        """as_claude_tools with cache_control=True adds cache field."""
        from hud.environment.integrations.anthropic import AnthropicMixin

        class TestEnv(AnthropicMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [create_mock_tool("test")]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_claude_tools(cache_control=True)

        assert tools[0]["cache_control"] == {"type": "ephemeral"}

    def test_as_claude_programmatic_tools(self) -> None:
        """as_claude_programmatic_tools includes allowed_callers."""
        from hud.environment.integrations.anthropic import AnthropicMixin

        class TestEnv(AnthropicMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [create_mock_tool("analyze")]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                pass

        env = TestEnv()
        tools = env.as_claude_programmatic_tools()

        assert tools[0]["allowed_callers"] == ["code_execution_20250825"]


class TestGeminiMixin:
    """Tests for Google/Gemini format conversion."""

    def test_as_gemini_tools_basic(self) -> None:
        """as_gemini_tools converts MCP tools to Gemini format."""
        from hud.environment.integrations.gemini import GeminiMixin

        class TestEnv(GeminiMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [
                    create_mock_tool(
                        "search",
                        "Search query",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    ),
                ]

        env = TestEnv()
        tools = env.as_gemini_tools()

        assert len(tools) == 1
        assert "function_declarations" in tools[0]
        declarations = tools[0]["function_declarations"]
        assert len(declarations) == 1
        assert declarations[0]["name"] == "search"
        assert declarations[0]["description"] == "Search query"

    def test_as_gemini_tools_multiple(self) -> None:
        """as_gemini_tools wraps multiple tools in single declaration list."""
        from hud.environment.integrations.gemini import GeminiMixin

        class TestEnv(GeminiMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return [
                    create_mock_tool("tool1"),
                    create_mock_tool("tool2"),
                    create_mock_tool("tool3"),
                ]

        env = TestEnv()
        tools = env.as_gemini_tools()

        assert len(tools) == 1  # Single wrapper object
        assert len(tools[0]["function_declarations"]) == 3

    def test_as_gemini_tool_config_auto(self) -> None:
        """as_gemini_tool_config with AUTO mode."""
        from hud.environment.integrations.gemini import GeminiMixin

        class TestEnv(GeminiMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return []

        env = TestEnv()
        config = env.as_gemini_tool_config(mode="AUTO")

        assert config["function_calling_config"]["mode"] == "AUTO"

    def test_as_gemini_tool_config_any_with_allowed(self) -> None:
        """as_gemini_tool_config with ANY mode and allowed tools."""
        from hud.environment.integrations.gemini import GeminiMixin

        class TestEnv(GeminiMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return []

        env = TestEnv()
        config = env.as_gemini_tool_config(mode="ANY", allowed_tools=["search", "navigate"])

        assert config["function_calling_config"]["mode"] == "ANY"
        assert config["function_calling_config"]["allowed_function_names"] == ["search", "navigate"]

    def test_as_gemini_tool_config_none(self) -> None:
        """as_gemini_tool_config with NONE mode disables tools."""
        from hud.environment.integrations.gemini import GeminiMixin

        class TestEnv(GeminiMixin):
            def as_tools(self) -> list[mcp_types.Tool]:
                return []

        env = TestEnv()
        config = env.as_gemini_tool_config(mode="NONE")

        assert config["function_calling_config"]["mode"] == "NONE"

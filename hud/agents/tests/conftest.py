"""Shared test fixtures for agent tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
from mcp import types

from hud.types import MCPToolCall, MCPToolResult


class MockMCPClient:
    """Mock MCP client that satisfies AgentMCPClient protocol."""

    _initialized: bool = False

    def __init__(
        self,
        tools: list[types.Tool] | None = None,
        call_tool_handler: Callable[[MCPToolCall], MCPToolResult] | None = None,
        initialize_error: Exception | None = None,
    ) -> None:
        self._mcp_config: dict[str, dict[str, Any]] = {"test": {"url": "http://test"}}
        self._tools = tools or []
        self._call_tool_handler = call_tool_handler
        self._initialize_error = initialize_error
        self.call_tool_calls: list[MCPToolCall] = []
        self.shutdown_called = False

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        return self._mcp_config

    @property
    def is_connected(self) -> bool:
        return self._initialized

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        if self._initialize_error:
            raise self._initialize_error
        self._initialized = True

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        self.call_tool_calls.append(tool_call)
        if self._call_tool_handler:
            return self._call_tool_handler(tool_call)
        return MCPToolResult(content=[])

    def get_available_tools(self) -> list[types.Tool]:
        return self._tools

    def get_tool_map(self) -> dict[str, types.Tool]:
        return {t.name: t for t in self._tools}


@pytest.fixture
def mock_mcp_client() -> MockMCPClient:
    """Create a mock MCP client that satisfies the AgentMCPClient protocol."""
    return MockMCPClient()


@pytest.fixture
def mock_mcp_client_with_tools() -> MockMCPClient:
    """Create a mock MCP client with a test tool."""
    return MockMCPClient(
        tools=[
            types.Tool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )


@pytest.fixture
def mock_mcp_client_openai_computer() -> MockMCPClient:
    """Create a mock MCP client with openai_computer tool for Operator tests."""
    return MockMCPClient(
        tools=[
            types.Tool(
                name="openai_computer",
                description="OpenAI computer use tool",
                inputSchema={},
            )
        ]
    )


@pytest.fixture
def mock_mcp_client_gemini_computer() -> MockMCPClient:
    """Create a mock MCP client with gemini_computer tool for Gemini tests."""
    return MockMCPClient(
        tools=[
            types.Tool(
                name="gemini_computer",
                description="Gemini computer use tool",
                inputSchema={},
            )
        ]
    )


@pytest.fixture
def mock_mcp_client_browser_tools() -> MockMCPClient:
    """Create a mock MCP client with browser-like tools for extended tests."""
    return MockMCPClient(
        tools=[
            types.Tool(name="screenshot", description="Take screenshot", inputSchema={}),
            types.Tool(name="click", description="Click at coordinates", inputSchema={}),
            types.Tool(name="type", description="Type text", inputSchema={}),
            types.Tool(name="bad_tool", description="A tool that fails", inputSchema={}),
        ]
    )

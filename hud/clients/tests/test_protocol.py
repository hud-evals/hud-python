"""Tests for the MCP client protocol and implementations."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from hud.clients.base import AgentMCPClient, BaseHUDClient
from hud.clients.fastmcp import FastMCPHUDClient
from hud.clients.mcp_use import MCPUseHUDClient
from hud.types import MCPToolResult


class MockClient(BaseHUDClient):
    """Mock client for testing the base class."""

    def __init__(self, **kwargs):
        super().__init__(mcp_config={"test": {"url": "mock://test"}}, **kwargs)
        self._connected = False
        self._mock_tools = [
            types.Tool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    async def _connect(self) -> None:
        self._connected = True

    async def list_tools(self) -> list[types.Tool]:
        if not self._connected:
            raise RuntimeError("Not connected")
        return self._mock_tools

    async def list_resources(self) -> list[types.Resource]:
        """Minimal list_resources for protocol satisfaction in tests."""
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        if name == "test_tool":
            return MCPToolResult(
                content=[types.TextContent(type="text", text="Success")], isError=False
            )
        raise ValueError(f"Tool {name} not found")

    async def _read_resource_internal(self, uri: str) -> types.ReadResourceResult | None:
        if uri == "telemetry://live":
            from pydantic import AnyUrl

            return types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=AnyUrl(uri),
                        mimeType="application/json",
                        text='{"status": "healthy", "services": {"api": "running"}}',
                    )
                ]
            )
        return None

    async def close(self) -> None:
        """Close the connection."""
        self._connected = False
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class TestProtocol:
    """Test that all clients implement the protocol correctly."""

    def test_mock_client_implements_protocol(self):
        """Test that our mock client implements the protocol."""
        client = MockClient()
        assert isinstance(client, AgentMCPClient)

    def test_fastmcp_client_implements_protocol(self):
        """Test that FastMCPHUDClient implements the protocol."""
        client = FastMCPHUDClient({"test": {"url": "http://localhost"}})
        assert isinstance(client, AgentMCPClient)

    def test_mcp_use_client_implements_protocol(self):
        """Test that MCPUseHUDClient implements the protocol."""
        client = MCPUseHUDClient({"test": {"url": "http://localhost"}})
        assert isinstance(client, AgentMCPClient)

    @pytest.mark.asyncio
    async def test_base_client_initialization(self):
        """Test that base client initialization works correctly."""
        client = MockClient()

        # Not initialized yet
        assert not client._initialized
        assert len(client.get_available_tools()) == 0  # Should be empty before init

        # Initialize
        await client.initialize()

        # Should be initialized with tools discovered
        assert client._initialized
        assert len(client.get_available_tools()) == 1
        assert client.get_available_tools()[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_telemetry_fetching(self):
        """Test that telemetry is fetched during initialization."""
        client = MockClient()

        # No telemetry before initialization
        assert client.get_telemetry_data() == {}

        # Initialize
        await client.initialize()

        # Should have telemetry
        telemetry = client.get_telemetry_data()
        assert telemetry["status"] == "healthy"
        assert telemetry["services"]["api"] == "running"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that clients work as context managers."""
        client = MockClient()

        async with client:
            assert client._initialized
            tools = await client.list_tools()
            assert len(tools) == 1

        # Note: MockClient doesn't implement close() so it stays initialized

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution through the protocol."""
        client = MockClient()

        await client.initialize()

        # Execute a tool
        result = await client.call_tool("test_tool", {"arg": "value"})

        assert isinstance(result, MCPToolResult)
        assert not result.isError
        from mcp.types import TextContent

        assert isinstance(result.content[0], TextContent) and result.content[0].text == "Success"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test error handling for missing tools."""
        client = MockClient()

        await client.initialize()

        # Try to execute non-existent tool
        with pytest.raises(ValueError, match="Tool unknown_tool not found"):
            await client.call_tool("unknown_tool", {})


class TestClientCompatibility:
    """Test that clients are compatible with agents."""

    def test_protocol_satisfied(self):
        """Test that all clients satisfy the protocol."""
        # Test mock client
        mock_client = MockClient()
        assert isinstance(mock_client, AgentMCPClient)
        assert hasattr(mock_client, "initialize")
        assert hasattr(mock_client, "list_tools")
        assert hasattr(mock_client, "call_tool")

        # Test FastMCP client
        fastmcp_client = FastMCPHUDClient({"test": {"url": "http://localhost"}})
        assert isinstance(fastmcp_client, AgentMCPClient)

        # Test MCP-use client
        mcp_use_client = MCPUseHUDClient({"test": {"url": "http://localhost"}})
        assert isinstance(mcp_use_client, AgentMCPClient)

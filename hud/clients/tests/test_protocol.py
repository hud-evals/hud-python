"""Tests for the MCP client protocol and implementations."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from hud.clients.base import AgentMCPClient, BaseHUDClient
from hud.clients.fastmcp import FastMCPHUDClient
from hud.clients.mcp_use import MCPUseHUDClient
from hud.types import MCPToolCall, MCPToolResult


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

    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        self._connected = True

    async def list_tools(self) -> list[types.Tool]:
        if not self._connected:
            raise RuntimeError("Not connected")
        return self._mock_tools

    async def _list_resources_impl(self) -> list[types.Resource]:
        """Minimal resource listing implementation for tests."""
        from pydantic import AnyUrl

        return [
            types.Resource(
                uri=AnyUrl("telemetry://live"), name="telemetry", description="Live telemetry data"
            )
        ]

    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        if tool_call.name == "test_tool":
            return MCPToolResult(
                content=[types.TextContent(type="text", text="Success")], isError=False
            )
        raise ValueError(f"Tool {tool_call.name} not found")

    async def read_resource(self, uri: str) -> types.ReadResourceResult | None:
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

    async def _disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False


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
        # Can't call list_tools before initialization, it would raise an error

        # Initialize
        await client.initialize()

        # Should be initialized with tools discovered
        assert client._initialized
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_telemetry_fetching(self):
        """Test that telemetry is fetched during initialization."""
        client = MockClient()

        # No telemetry before initialization
        assert not hasattr(client, "_telemetry_data") or client._telemetry_data == {}

        # Initialize
        await client.initialize()

        # Should have telemetry
        assert hasattr(client, "_telemetry_data")
        assert client._telemetry_data["status"] == "healthy"
        assert client._telemetry_data["services"]["api"] == "running"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that clients work as context managers."""
        client = MockClient()

        async with client:
            assert client._initialized
            tools = await client.list_tools()
            assert len(tools) == 1

        # Should be closed after exiting context
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution through the protocol."""
        client = MockClient()

        await client.initialize()

        # Execute a tool - test both call signatures
        # Test with MCPToolCall
        tool_call = MCPToolCall(name="test_tool", arguments={"arg": "value"})
        result = await client.call_tool(tool_call)

        assert isinstance(result, MCPToolResult)
        assert not result.isError
        from mcp.types import TextContent

        assert isinstance(result.content[0], TextContent) and result.content[0].text == "Success"

        # Test with name/arguments
        result2 = await client.call_tool(name="test_tool", arguments={"arg": "value"})
        assert isinstance(result2, MCPToolResult)
        assert not result2.isError
        assert isinstance(result2.content[0], TextContent) and result2.content[0].text == "Success"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test error handling for missing tools."""
        client = MockClient()

        await client.initialize()

        # Try to execute non-existent tool
        with pytest.raises(ValueError, match="Tool unknown_tool not found"):
            await client.call_tool(name="unknown_tool", arguments={})


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

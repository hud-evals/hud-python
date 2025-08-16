"""Tests for MCP Client implementation."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types
from pydantic import AnyUrl

from hud.clients.mcp_use import MCPUseHUDClient as MCPClient
from hud.types import MCPToolResult

logger = logging.getLogger(__name__)


class TestMCPClient:
    """Test MCPClient class."""

    @pytest.fixture
    def mock_mcp_use_client(self):
        """Create a mock MCPUseClient (the internal mcp_use client)."""
        # Create a mock instance that will be returned by from_dict
        mock_instance = MagicMock()
        mock_instance.create_session = AsyncMock()
        mock_instance.create_all_sessions = AsyncMock(return_value={})
        mock_instance.close_all_sessions = AsyncMock()
        mock_instance.get_all_active_sessions = MagicMock(return_value={})

        # Patch MCPUseClient class used by MCP-use backend
        with patch("hud.clients.mcp_use.MCPUseClient") as mock_class:
            mock_class.from_dict = MagicMock(return_value=mock_instance)
            yield mock_instance

    @pytest.mark.asyncio
    async def test_init_with_config(self):
        """Test client initialization with config dictionary."""
        mcp_config = {
            "test_server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {"TEST": "true"},
            }
        }

        with patch("hud.clients.mcp_use.MCPUseClient") as mock_use_client:
            mock_use_client.from_dict.return_value = MagicMock()
            client = MCPClient(mcp_config=mcp_config, verbose=True)

            assert client.verbose is True
            # Verify MCPUseClient.from_dict was called with proper config
            mock_use_client.from_dict.assert_called_once_with({"mcpServers": mcp_config})

    @pytest.mark.asyncio
    async def test_connect_single_server(self, mock_mcp_use_client):
        """Test connecting to a single server."""
        config = {"test_server": {"command": "python", "args": ["-m", "test_server"]}}

        # Create the MCPClient - the fixture already patches MCPUseClient
        client = MCPClient(mcp_config=config, verbose=True)

        # Internal client created
        assert hasattr(client, "_mcp_client")

        # Mock session
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        # Mock list_tools response
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                ]
            )

        mock_session.connector.client_session.list_tools = mock_list_tools

        # Mock create_all_sessions to return a dict with our session
        mock_mcp_use_client.create_all_sessions = AsyncMock(
            return_value={"test_server": mock_session}
        )

        # Initialize the client (creates sessions and discovers tools)
        await client.initialize()

        # Verify session was created
        mock_mcp_use_client.create_all_sessions.assert_called_once()

        # Check tools were discovered via public API
        tools = client.get_available_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_connect_multiple_servers(self, mock_mcp_use_client):
        """Test connecting to multiple servers."""
        config = {
            "server1": {"command": "python", "args": ["-m", "server1"]},
            "server2": {"command": "node", "args": ["server2.js"]},
        }

        client = MCPClient(mcp_config=config)

        # Mock sessions
        mock_session1 = MagicMock()
        mock_session1.connector = MagicMock()
        mock_session1.connector.client_session = MagicMock()

        mock_session2 = MagicMock()
        mock_session2.connector = MagicMock()
        mock_session2.connector.client_session = MagicMock()

        # Mock tools for each server
        async def mock_list_tools1():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"})
                ]
            )

        async def mock_list_tools2():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"})
                ]
            )

        mock_session1.connector.client_session.list_tools = mock_list_tools1
        mock_session2.connector.client_session.list_tools = mock_list_tools2

        # Mock create_all_sessions to return both sessions
        mock_mcp_use_client.create_all_sessions = AsyncMock(
            return_value={"server1": mock_session1, "server2": mock_session2}
        )

        await client.initialize()

        # Verify sessions were created
        mock_mcp_use_client.create_all_sessions.assert_called_once()

        # Check tools from both servers
        tools = client.get_available_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_call_tool(self, mock_mcp_use_client):
        """Test calling a tool."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        # Setup mock session
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        # Mock tool
        tool = types.Tool(
            name="calculator", description="Calculator", inputSchema={"type": "object"}
        )

        async def mock_list_tools():
            return types.ListToolsResult(tools=[tool])

        mock_session.connector.client_session.list_tools = mock_list_tools

        # The session returns CallToolResult, but the client should return MCPToolResult
        mock_call_result = types.CallToolResult(
            content=[types.TextContent(type="text", text="Result: 42")], isError=False
        )

        mock_session.connector.client_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})

        await client.initialize()

        # Call the tool
        result = await client.call_tool("calculator", {"operation": "add", "a": 20, "b": 22})

        assert isinstance(result, MCPToolResult)
        assert result.content[0].text == "Result: 42"  # type: ignore
        assert result.isError is False
        mock_session.connector.client_session.call_tool.assert_called_once_with(
            name="calculator", arguments={"operation": "add", "a": 20, "b": 22}
        )

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, mock_mcp_use_client):
        """Test calling a non-existent tool."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})

        await client.initialize()

        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await client.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_get_telemetry_data(self, mock_mcp_use_client):
        """Test getting telemetry data."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        # Mock tools
        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.connector.client_session.list_tools = mock_list_tools

        # Mock telemetry resource
        mock_telemetry = types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=AnyUrl("telemetry://live"),
                    mimeType="application/json",
                    text='{"events": [{"type": "test", "data": "value"}]}',
                )
            ]
        )

        mock_session.connector.client_session.read_resource = AsyncMock(return_value=mock_telemetry)

        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})

        await client.initialize()

        telemetry_data = client.get_telemetry_data()
        # In the new client, telemetry is a flat dict of fields
        assert isinstance(telemetry_data, dict)

    @pytest.mark.asyncio
    async def test_close(self, mock_mcp_use_client):
        """Test closing client connections."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_session = AsyncMock(return_value=mock_session)
        mock_mcp_use_client.close_all_sessions = AsyncMock()

        await client.initialize()
        await client.close()

        mock_mcp_use_client.close_all_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_mcp_use_client):
        """Test using client as context manager."""
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})
        mock_mcp_use_client.close_all_sessions = AsyncMock()

        config = {"test": {"command": "test"}}
        # The fixture already patches MCPUseClient
        async with MCPClient(mcp_config=config) as client:
            assert client._mcp_client is not None
            # Verify that the client uses our mock
            assert client._mcp_client == mock_mcp_use_client

        # Verify cleanup was called
        mock_mcp_use_client.close_all_sessions.assert_called_once()

    def test_get_available_tools(self, mock_mcp_use_client):
        """Test getting available tools."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        # Prefer using public API; simulate via internal _tools if available
        try:
            client._tools = [  # type: ignore[attr-defined]
                types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
            ]
        except Exception:
            logger.warning("Failed to set tools")

        tools = client.get_available_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    def test_get_tool_map(self, mock_mcp_use_client):
        """Test getting tool map."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        # Simulate discovered tools
        client._tools = [  # type: ignore[attr-defined]
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
        ]

        tools = client.get_available_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

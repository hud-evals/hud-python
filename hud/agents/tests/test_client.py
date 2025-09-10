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


@patch("hud.clients.base.setup_hud_telemetry")
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

        # Patch MCPUseClient.from_dict at the module level
        with patch("mcp_use.client.MCPClient.from_dict", return_value=mock_instance):
            yield mock_instance

    @pytest.mark.asyncio
    async def test_connect_single_server(self, mock_telemetry, mock_mcp_use_client):
        """Test connecting to a single server."""
        config = {"test_server": {"command": "python", "args": ["-m", "test_server"]}}

        # Create the MCPClient - the fixture already patches MCPUseClient
        client = MCPClient(mcp_config=config, verbose=True)

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

        # Internal client created
        assert client._client is not None

        # Verify session was created
        mock_mcp_use_client.create_all_sessions.assert_called_once()

        # Check tools were discovered via public API
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_connect_multiple_servers(self, mock_telemetry, mock_mcp_use_client):
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

        # Check tools from both servers - should be prefixed with server names
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"server1_tool1", "server2_tool2"}

    @pytest.mark.asyncio
    async def test_call_tool(self, mock_telemetry, mock_mcp_use_client):
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

        # Set up the mock to return the session both when creating and when getting sessions
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})
        mock_mcp_use_client.get_all_active_sessions = MagicMock(return_value={"test": mock_session})

        await client.initialize()

        # First discover tools by calling list_tools
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

        # Call the tool
        result = await client.call_tool(
            name="calculator", arguments={"operation": "add", "a": 20, "b": 22}
        )

        assert isinstance(result, MCPToolResult)
        assert result.content[0].text == "Result: 42"  # type: ignore
        assert result.isError is False
        mock_session.connector.client_session.call_tool.assert_called_once_with(
            name="calculator", arguments={"operation": "add", "a": 20, "b": 22}
        )

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, mock_telemetry, mock_mcp_use_client):
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

        # Calling a non-existent tool should return an error result
        result = await client.call_tool(name="nonexistent", arguments={})
        assert result.isError is True
        # Check that the error message is in the text content
        text_content = ""
        for content in result.content:
            if isinstance(content, types.TextContent):
                text_content += content.text
        assert "Tool 'nonexistent' not found" in text_content

    @pytest.mark.asyncio
    async def test_get_telemetry_data(self, mock_telemetry, mock_mcp_use_client):
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

        telemetry_data = client._telemetry_data
        # In the new client, telemetry is a flat dict of fields
        assert isinstance(telemetry_data, dict)

    @pytest.mark.asyncio
    async def test_close(self, mock_telemetry, mock_mcp_use_client):
        """Test closing client connections."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})
        mock_mcp_use_client.close_all_sessions = AsyncMock()

        await client.initialize()
        await client.shutdown()

        mock_mcp_use_client.close_all_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_telemetry, mock_mcp_use_client):
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
            assert client._client is not None
            # Verify that the client uses our mock
            assert client._client == mock_mcp_use_client

        # Verify cleanup was called
        mock_mcp_use_client.close_all_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_tools(self, mock_telemetry, mock_mcp_use_client):
        """Test getting available tools."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        # Create tool objects
        tool1 = types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"})
        tool2 = types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"})

        # Setup mock session with tools
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[tool1, tool2])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})

        # Initialize to populate tools
        await client.initialize()

        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_get_tool_map(self, mock_telemetry, mock_mcp_use_client):
        """Test getting tool map."""
        config = {"test": {"command": "test"}}
        client = MCPClient(mcp_config=config)

        # Create tool objects
        tool1 = types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"})
        tool2 = types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"})

        # Setup mock session with tools
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[tool1, tool2])

        mock_session.connector.client_session.list_tools = mock_list_tools
        mock_mcp_use_client.create_all_sessions = AsyncMock(return_value={"test": mock_session})

        # Initialize to populate tools
        await client.initialize()

        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

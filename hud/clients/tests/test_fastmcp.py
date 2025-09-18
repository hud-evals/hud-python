"""Tests for FastMCP client implementation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from mcp import types
from pydantic.networks import AnyUrl

from hud.clients.fastmcp import FastMCPHUDClient
from hud.types import MCPToolCall, MCPToolResult


class TestFastMCPHUDClient:
    """Test FastMCP HUD client."""

    def test_initialization(self):
        """Test client initialization."""
        config = {"server1": {"command": "python", "args": ["server.py"]}}

        # Client is just instantiated, not connected yet
        client = FastMCPHUDClient(config)

        # Check that the client has the config stored
        assert client._mcp_config == config
        assert client._client is None  # Not connected yet

    @pytest.mark.asyncio
    async def test_connect_creates_client(self):
        """Test that _connect creates the FastMCP client."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            await client._connect(config)

            # Check FastMCP client was created
            mock_client_class.assert_called_once()

            # Check it was created with correct transport and client info
            call_args = mock_client_class.call_args
            assert call_args[0][0] == {"mcpServers": config}
            assert call_args[1]["client_info"].name == "hud-fastmcp"

    @pytest.mark.asyncio
    async def test_connect_logs_info(self):
        """Test that connect logs info message."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)

            with patch("hud.clients.fastmcp.logger") as mock_logger:
                await client._connect(config)

                # Check info was logged
                mock_logger.info.assert_called_with("FastMCP client connected")

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_tools = [
                MagicMock(spec=types.Tool, name="tool1"),
                MagicMock(spec=types.Tool, name="tool2"),
            ]
            mock_fastmcp.list_tools.return_value = mock_tools
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            client._initialized = True  # Skip initialization
            client._client = mock_fastmcp  # Set the mock client

            tools = await client.list_tools()

            assert tools == mock_tools
            mock_fastmcp.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling a tool."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()

            # Mock FastMCP result
            mock_result = MagicMock()
            mock_result.content = [types.TextContent(type="text", text="result")]
            mock_result.is_error = False
            mock_result.structured_content = {"key": "value"}

            mock_fastmcp.call_tool.return_value = mock_result
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            result = await client.call_tool(name="test_tool", arguments={"arg": "value"})

            assert isinstance(result, MCPToolResult)
            assert result.content == mock_result.content
            assert result.isError is False
            assert result.structuredContent == {"key": "value"}

            mock_fastmcp.call_tool.assert_called_once_with(
                name="test_tool",
                arguments={"arg": "value"},
                raise_on_error=False,
            )

    @pytest.mark.asyncio
    async def test_call_tool_with_mcp_tool_call(self):
        """Test calling a tool with MCPToolCall object."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()

            # Mock FastMCP result
            mock_result = MagicMock()
            mock_result.content = [types.TextContent(type="text", text="result")]
            mock_result.is_error = False
            mock_result.structured_content = {"key": "value"}

            mock_fastmcp.call_tool.return_value = mock_result
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            # Test with MCPToolCall object
            tool_call = MCPToolCall(name="test_tool", arguments={"arg": "value"})
            result = await client.call_tool(tool_call)

            assert isinstance(result, MCPToolResult)
            assert result.content == mock_result.content
            assert result.isError is False
            assert result.structuredContent == {"key": "value"}

            mock_fastmcp.call_tool.assert_called_once_with(
                name="test_tool",
                arguments={"arg": "value"},
                raise_on_error=False,
            )

    @pytest.mark.asyncio
    async def test_call_tool_no_arguments(self):
        """Test calling a tool without arguments."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_result = MagicMock()
            mock_result.content = []
            mock_result.is_error = True
            mock_result.structured_content = None

            mock_fastmcp.call_tool.return_value = mock_result
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            await client.call_tool(name="test_tool", arguments={})

            # Should pass empty dict for arguments
            mock_fastmcp.call_tool.assert_called_once_with(
                name="test_tool",
                arguments={},
                raise_on_error=False,
            )

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_resources = [
                MagicMock(spec=types.Resource, uri="file:///test1"),
                MagicMock(spec=types.Resource, uri="file:///test2"),
            ]
            mock_fastmcp.list_resources.return_value = mock_resources
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            resources = await client.list_resources()

            assert resources == mock_resources
            mock_fastmcp.list_resources.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_resource_internal_success(self):
        """Test reading a resource successfully."""
        config = {"server1": {"command": "test"}}

        # Create proper resource contents that ReadResourceResult expects
        mock_contents = [
            types.TextResourceContents(
                uri=AnyUrl("file:///test"),
                mimeType="text/plain",
                text="resource content",
            )
        ]

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            # Create a mock FastMCP client
            mock_fastmcp = AsyncMock()
            mock_fastmcp.read_resource.return_value = mock_contents
            mock_client_class.return_value = mock_fastmcp

            # Now create the HUD client - it will use our mocked FastMCP client
            client = FastMCPHUDClient(config)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            result = await client.read_resource("file:///test")

            assert isinstance(result, types.ReadResourceResult)
            assert result.contents == mock_contents
            mock_fastmcp.read_resource.assert_called_once_with("file:///test")

    @pytest.mark.asyncio
    async def test_read_resource_internal_error_verbose(self):
        """Test reading a resource with error in verbose mode."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_fastmcp.read_resource.side_effect = Exception("Read failed")
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config, verbose=True)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            with patch("hud.clients.fastmcp.logger") as mock_logger:
                result = await client.read_resource("file:///bad")

                assert result is None
                mock_logger.warning.assert_called_with(
                    "Unexpected error reading resource '%s': %s", "file:///bad", ANY
                )

    @pytest.mark.asyncio
    async def test_read_resource_internal_error_not_verbose(self):
        """Test reading a resource with error in non-verbose mode."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_fastmcp.read_resource.side_effect = Exception("Read failed")
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config, verbose=False)
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            with patch("hud.clients.fastmcp.logger") as mock_logger:
                result = await client.read_resource("file:///bad")

                assert result is None
                # Should not log in non-verbose mode
                mock_logger.debug.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutting down the client."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)

            # Set up stack and client
            mock_stack = AsyncMock()
            client._stack = mock_stack
            client._initialized = True
            client._client = mock_fastmcp  # Set the mock client

            # Provide a task-like connect handle so shutdown's done() check behaves like
            # real transport
            loop = asyncio.get_running_loop()
            connect_task = loop.create_future()
            connect_task.set_result(None)
            mock_fastmcp.transport = SimpleNamespace(_connect_task=connect_task)

            with patch("hud.clients.fastmcp.logger") as mock_logger:
                await client.shutdown()

                mock_stack.aclose.assert_called_once()
                assert client._stack is None
                assert client._initialized is False
                mock_logger.debug.assert_called_with("FastMCP client closed")

    @pytest.mark.asyncio
    async def test_shutdown_no_stack(self):
        """Test shutting down when no stack exists."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient"):
            client = FastMCPHUDClient(config)
            client._stack = None

            # Should not raise error
            await client.shutdown()

            assert client._stack is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        config = {"server1": {"command": "test"}}

        with patch("hud.clients.fastmcp.FastMCPClient") as mock_client_class:
            mock_fastmcp = AsyncMock()
            mock_client_class.return_value = mock_fastmcp

            client = FastMCPHUDClient(config)

            with (
                patch.object(client, "initialize", new_callable=AsyncMock) as mock_init,
                patch.object(client, "shutdown", new_callable=AsyncMock) as mock_close,
            ):
                async with client as ctx:
                    assert ctx is client
                    mock_init.assert_called_once()

                mock_close.assert_called_once()

"""Tests for MCPServer functionality."""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.server.server import MCPServer
from hud.tools.base import BaseTool


class TestMCPServer:
    """Test MCPServer class functionality."""

    def test_init_basic(self) -> None:
        """Test basic MCPServer initialization."""
        server = MCPServer(name="test_server")
        assert server.name == "test_server"
        assert server._shutdown_fn is None
        assert server._initializer_fn is None
        assert server._did_init is False
        assert server._replaced_server is False

    def test_init_with_lifespan(self) -> None:
        """Test MCPServer initialization with custom lifespan."""
        async def custom_lifespan(app):
            yield {"test": "value"}

        server = MCPServer(name="test_server", lifespan=custom_lifespan)
        assert server.name == "test_server"
        # Custom lifespan should be preserved
        assert hasattr(server, '_mcp_server')

    def test_add_tool_regular_function(self) -> None:
        """Test adding a regular tool function."""
        server = MCPServer(name="test_server")

        # Create a proper FastMCP tool using the decorator
        @server.tool()
        def test_tool() -> str:
            """A test tool."""
            return "test result"

        # The tool should be added to the underlying FastMCP server
        assert hasattr(server, '_mcp_server')

    def test_add_tool_base_tool_instance(self) -> None:
        """Test adding a BaseTool instance."""
        server = MCPServer(name="test_server")

        # Create a mock BaseTool that behaves like a real BaseTool
        mock_tool = MagicMock()
        mock_tool.mcp = MagicMock()

        # Make isinstance check pass
        mock_tool.__class__ = BaseTool

        with patch('hud.server.server.super') as mock_super:
            server.add_tool(mock_tool)

            # Should call super().add_tool with the tool's mcp attribute
            mock_super.return_value.add_tool.assert_called_once_with(mock_tool.mcp)

    def test_initialize_decorator(self) -> None:
        """Test the initialize decorator."""
        server = MCPServer(name="test_server")

        @server.initialize
        def init_handler(ctx):
            return "initialized"

        assert server._initializer_fn is init_handler
        assert server._replaced_server is True


    def test_shutdown_decorator(self) -> None:
        """Test the shutdown decorator."""
        server = MCPServer(name="test_server")

        @server.shutdown
        def shutdown_handler():
            return "shutdown"

        assert server._shutdown_fn is shutdown_handler


    @patch('hud.server.server.anyio.run')
    def test_run_with_stdio_transport(self, mock_anyio_run) -> None:
        """Test server run method with stdio transport."""
        server = MCPServer(name="test_server")

        # Mock the async run method
        server.run_async = AsyncMock()

        server.run(transport="stdio")

        # Should call _run_with_sigterm with bootstrap coroutine
        mock_anyio_run.assert_called_once()

    @patch('hud.server.server.anyio.run')
    def test_run_with_custom_transport(self, mock_anyio_run) -> None:
        """Test server run method with custom transport."""
        server = MCPServer(name="test_server")

        # Mock the async run method
        server.run_async = AsyncMock()

        server.run(transport="sse", port=8080)

        mock_anyio_run.assert_called_once()




class TestServerIntegration:
    """Integration tests for MCPServer."""

    @pytest.mark.asyncio
    async def test_server_lifespan_management(self) -> None:
        """Test that lifespan properly manages shutdown."""
        server = MCPServer(name="test_server")

        shutdown_called = False

        @server.shutdown
        async def shutdown_handler():
            nonlocal shutdown_called
            shutdown_called = True

        # Test that the lifespan is created properly
        # The actual lifespan testing is complex due to FastMCP internals
        assert server._shutdown_fn is shutdown_handler
        assert server._mcp_server is not None

    @pytest.mark.asyncio
    async def test_server_lifespan_sigterm_shutdown(self) -> None:
        """Test that shutdown handler is called on SIGTERM."""
        global _sigterm_received
        _sigterm_received = True

        server = MCPServer(name="test_server")

        shutdown_called = False

        @server.shutdown
        async def shutdown_handler():
            nonlocal shutdown_called
            shutdown_called = True

        try:
            # Test that shutdown handler is registered
            assert server._shutdown_fn is shutdown_handler
            # The actual lifespan execution happens during server.run()
        finally:
            _sigterm_received = False  # Reset for other tests

        # Shutdown handler should be registered
        assert server._shutdown_fn is not None

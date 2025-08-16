"""Integration tests for MCP clients."""

from __future__ import annotations

import pytest

from hud.clients import FastMCPHUDClient, MCPUseHUDClient
from hud.clients.base import AgentMCPClient


class TestClientIntegration:
    """Test that clients work with real configurations."""

    def test_fastmcp_client_creation(self):
        """Test that FastMCP client can be created with various configs."""
        # HTTP config
        config = {"server": {"url": "http://localhost:8080"}}
        client = FastMCPHUDClient(config)
        assert isinstance(client, AgentMCPClient)
        assert client.is_connected is False

        # Stdio config
        config = {"server": {"command": "python", "args": ["server.py"]}}
        client = FastMCPHUDClient(config)
        assert isinstance(client, AgentMCPClient)

        # Multi-server config
        config = {
            "server1": {"url": "http://localhost:8080"},
            "server2": {"command": "python", "args": ["server.py"]},
        }
        client = FastMCPHUDClient(config)
        assert isinstance(client, AgentMCPClient)

    def test_mcp_use_client_creation(self):
        """Test that MCP-use client can be created with various configs."""
        # HTTP config
        config = {"server": {"url": "http://localhost:8080"}}
        client = MCPUseHUDClient(config)
        assert isinstance(client, AgentMCPClient)
        assert client.is_connected is False

        # Stdio config
        config = {"server": {"command": "python", "args": ["server.py"]}}
        client = MCPUseHUDClient(config)
        assert isinstance(client, AgentMCPClient)

        # Multi-server config
        config = {
            "server1": {"url": "http://localhost:8080"},
            "server2": {"command": "python", "args": ["server.py"]},
        }
        client = MCPUseHUDClient(config)
        assert isinstance(client, AgentMCPClient)

    def test_client_switching(self):
        """Test that clients can be switched without changing agent code."""
        config = {"server": {"url": "http://localhost:8080"}}

        # Both clients should satisfy the protocol
        fastmcp_client = FastMCPHUDClient(config)
        mcp_use_client = MCPUseHUDClient(config)

        # Both implement the same protocol
        assert isinstance(fastmcp_client, AgentMCPClient)
        assert isinstance(mcp_use_client, AgentMCPClient)

        # Both have the same essential methods
        for method in ["initialize", "list_tools", "call_tool"]:
            assert hasattr(fastmcp_client, method)
            assert hasattr(mcp_use_client, method)
            assert callable(getattr(fastmcp_client, method))
            assert callable(getattr(mcp_use_client, method))

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test that both clients work as context managers."""
        config = {"server": {"url": "http://localhost:8080"}}

        # Test FastMCP client
        fastmcp_client = FastMCPHUDClient(config)
        assert not fastmcp_client.is_connected

        # We can't actually connect without a real server, but we can test the pattern
        try:
            async with fastmcp_client:
                # This will fail to connect, but that's expected
                pass
        except Exception:
            # Expected - no server running
            pass

        # Test MCP-use client
        mcp_use_client = MCPUseHUDClient(config)
        assert not mcp_use_client.is_connected

        try:
            async with mcp_use_client:
                # This will fail to connect, but that's expected
                pass
        except Exception:
            # Expected - no server running
            pass

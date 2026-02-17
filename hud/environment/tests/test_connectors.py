"""Tests for hud.environment.connectors module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from hud.environment.connection import ConnectionType, Connector


class TestBaseConnectorMixin:
    """Tests for BaseConnectorMixin._add_connection."""

    def test_add_connection_stores_transport_config(self) -> None:
        """_add_connection stores transport, doesn't create client."""
        from hud.environment.connectors.base import BaseConnectorMixin

        class TestEnv(BaseConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        transport = {"server": {"url": "http://example.com"}}

        env._add_connection(
            "test-server",
            transport,
            connection_type=ConnectionType.REMOTE,
            auth="test-token",
            prefix="myprefix",
        )

        assert "test-server" in env._connections
        conn = env._connections["test-server"]
        assert conn._transport == transport
        assert conn._auth == "test-token"
        assert conn.config.prefix == "myprefix"
        assert conn.client is None  # Not created yet

    def test_add_connection_returns_self(self) -> None:
        """_add_connection returns self for chaining."""
        from hud.environment.connectors.base import BaseConnectorMixin

        class TestEnv(BaseConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        result = env._add_connection(
            "test",
            {},
            connection_type=ConnectionType.REMOTE,
        )

        assert result is env


class TestMCPConfigConnectorMixin:
    """Tests for MCPConfigConnectorMixin."""

    def test_connect_mcp_detects_local_connection(self) -> None:
        """connect_mcp detects LOCAL type from command in config."""
        from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

        class TestEnv(MCPConfigConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        config = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            }
        }

        env.connect_mcp(config)

        conn = env._connections["filesystem"]
        assert conn.connection_type == ConnectionType.LOCAL

    def test_connect_mcp_detects_remote_connection(self) -> None:
        """connect_mcp detects REMOTE type from URL in config."""
        from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

        class TestEnv(MCPConfigConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        config = {
            "browser": {
                "url": "https://mcp.hud.ai/browser",
            }
        }

        env.connect_mcp(config)

        conn = env._connections["browser"]
        assert conn.connection_type == ConnectionType.REMOTE

    def test_connect_mcp_uses_alias(self) -> None:
        """connect_mcp uses alias if provided."""
        from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

        class TestEnv(MCPConfigConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        config = {"server": {"url": "http://example.com"}}

        env.connect_mcp(config, alias="my-alias")

        assert "my-alias" in env._connections
        assert "server" not in env._connections

    def test_connect_mcp_config_creates_multiple_connections(self) -> None:
        """connect_mcp_config creates a connection for each server."""
        from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

        class TestEnv(MCPConfigConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

        env = TestEnv()
        mcp_config = {
            "server1": {"url": "http://example1.com"},
            "server2": {"url": "http://example2.com"},
            "server3": {"command": "npx", "args": ["server"]},
        }

        env.connect_mcp_config(mcp_config)

        assert len(env._connections) == 3
        assert "server1" in env._connections
        assert "server2" in env._connections
        assert "server3" in env._connections


class TestRemoteConnectorMixin:
    """Tests for RemoteConnectorMixin."""

    def test_connect_url_creates_remote_connection(self) -> None:
        """connect_url creates REMOTE connection."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        class TestEnv(RemoteConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        env.connect_url("https://mcp.example.com", alias="example")

        assert "example" in env._connections
        conn = env._connections["example"]
        assert conn.connection_type == ConnectionType.REMOTE

    def test_connect_url_extracts_auth_from_headers(self) -> None:
        """connect_url extracts Authorization from headers."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        class TestEnv(RemoteConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        env.connect_url(
            "https://mcp.example.com",
            headers={"Authorization": "Bearer my-token"},
            alias="example",
        )

        conn = env._connections["example"]
        assert conn._auth == "Bearer my-token"

    def test_connect_hub_creates_connection(self) -> None:
        """connect_hub creates connection with correct config."""
        from hud.environment.connectors.remote import RemoteConnectorMixin

        class TestEnv(RemoteConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}
                self._hub_config: dict[str, Any] | None = None

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        from hud.settings import Settings

        env = TestEnv()
        with patch("hud.settings.settings", spec=Settings) as mock_settings:
            mock_settings.hud_mcp_url = "https://mcp.hud.ai"
            mock_settings.client_timeout = 300  # Used in connect_mcp for sse_read_timeout

            env.connect_hub("browser")

        # connect_hub creates a connection named "hud" (from mcp_config key)
        assert "hud" in env._connections
        # Verify hub config is stored for serialization
        assert env._hub_config == {"name": "browser"}

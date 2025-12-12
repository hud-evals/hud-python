"""Tests for local connectors - connect_image, connect_server, connect_fastapi."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from hud.environment.connection import ConnectionType, Connector


class TestConnectImage:
    """Tests for LocalConnectorMixin.connect_image."""

    @patch("hud.cli.utils.docker.create_docker_run_command")
    def test_connect_image_creates_local_connection(self, mock_docker_cmd: MagicMock) -> None:
        """connect_image creates LOCAL connection with docker command."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_docker_cmd.return_value = ["docker", "run", "-i", "--rm", "mcp/fetch"]

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        env.connect_image("mcp/fetch")

        assert "mcp/fetch" in env._connections
        conn = env._connections["mcp/fetch"]
        assert conn.connection_type == ConnectionType.LOCAL
        mock_docker_cmd.assert_called_once()

    @patch("hud.cli.utils.docker.create_docker_run_command")
    def test_connect_image_with_alias(self, mock_docker_cmd: MagicMock) -> None:
        """connect_image uses alias for connection name."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_docker_cmd.return_value = ["docker", "run", "-i", "--rm", "mcp/fetch"]

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        env.connect_image("mcp/fetch", alias="fetcher")

        assert "fetcher" in env._connections
        assert "mcp/fetch" not in env._connections

    @patch("hud.cli.utils.docker.create_docker_run_command")
    def test_connect_image_with_prefix(self, mock_docker_cmd: MagicMock) -> None:
        """connect_image passes prefix to config."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_docker_cmd.return_value = ["docker", "run", "-i", "--rm", "mcp/fetch"]

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        env.connect_image("mcp/fetch", prefix="fetch")

        conn = env._connections["mcp/fetch"]
        assert conn.config.prefix == "fetch"

    @patch("hud.cli.utils.docker.create_docker_run_command")
    def test_connect_image_returns_self(self, mock_docker_cmd: MagicMock) -> None:
        """connect_image returns self for chaining."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_docker_cmd.return_value = ["docker", "run", "-i", "--rm", "mcp/fetch"]

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def mount(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        result = env.connect_image("mcp/fetch")

        assert result is env


class TestConnectServer:
    """Tests for LocalConnectorMixin.connect_server."""

    def test_connect_server_calls_include_router(self) -> None:
        """connect_server calls include_router with server and prefix."""
        from hud.environment.connectors.local import LocalConnectorMixin

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}
                self.routers: list[tuple[Any, str | None]] = []

            def include_router(self, server: Any, *, prefix: str | None = None) -> None:
                self.routers.append((server, prefix))

        env = TestEnv()
        mock_server = MagicMock()
        env.connect_server(mock_server, prefix="tools")

        assert len(env.routers) == 1
        assert env.routers[0] == (mock_server, "tools")

    def test_connect_server_returns_self(self) -> None:
        """connect_server returns self for chaining."""
        from hud.environment.connectors.local import LocalConnectorMixin

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def include_router(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        result = env.connect_server(MagicMock())

        assert result is env


class TestConnectFastAPI:
    """Tests for LocalConnectorMixin.connect_fastapi."""

    @patch("fastmcp.FastMCP")
    def test_connect_fastapi_creates_mcp_server(self, mock_fastmcp: MagicMock) -> None:
        """connect_fastapi converts FastAPI app to MCP server."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_mcp_server = MagicMock()
        mock_fastmcp.from_fastapi.return_value = mock_mcp_server

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}
                self.routers: list[tuple[Any, str | None]] = []

            def include_router(self, server: Any, *, prefix: str | None = None) -> None:
                self.routers.append((server, prefix))

        env = TestEnv()
        mock_app = MagicMock()
        mock_app.title = "My API"
        env.connect_fastapi(mock_app)

        mock_fastmcp.from_fastapi.assert_called_once_with(app=mock_app, name="My API")
        assert len(env.routers) == 1
        assert env.routers[0] == (mock_mcp_server, None)

    @patch("fastmcp.FastMCP")
    def test_connect_fastapi_with_custom_name(self, mock_fastmcp: MagicMock) -> None:
        """connect_fastapi uses custom name if provided."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_fastmcp.from_fastapi.return_value = MagicMock()

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def include_router(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        mock_app = MagicMock()
        mock_app.title = "Original"
        env.connect_fastapi(mock_app, name="custom-api")

        mock_fastmcp.from_fastapi.assert_called_once_with(app=mock_app, name="custom-api")

    @patch("fastmcp.FastMCP")
    def test_connect_fastapi_returns_self(self, mock_fastmcp: MagicMock) -> None:
        """connect_fastapi returns self for chaining."""
        from hud.environment.connectors.local import LocalConnectorMixin

        mock_fastmcp.from_fastapi.return_value = MagicMock()

        class TestEnv(LocalConnectorMixin):
            def __init__(self) -> None:
                self._connections: dict[str, Connector] = {}

            def include_router(self, server: Any, *, prefix: str | None = None) -> None:
                pass

        env = TestEnv()
        result = env.connect_fastapi(MagicMock())

        assert result is env

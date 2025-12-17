"""Tests for hud.environment.connection module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mcp_types
import pytest

from hud.environment.connection import ConnectionConfig, ConnectionType, Connector


class TestConnectionConfig:
    """Tests for ConnectionConfig."""

    def test_default_config(self) -> None:
        """Config with no options set."""
        config = ConnectionConfig()
        assert config.prefix is None
        assert config.include is None
        assert config.exclude is None
        assert config.transform is None

    def test_config_with_options(self) -> None:
        """Config with all options set."""
        transform_fn = lambda t: t  # noqa: E731
        config = ConnectionConfig(
            prefix="test",
            include=["tool1", "tool2"],
            exclude=["tool3"],
            transform=transform_fn,
        )
        assert config.prefix == "test"
        assert config.include == ["tool1", "tool2"]
        assert config.exclude == ["tool3"]
        assert config.transform is transform_fn


class TestConnectionType:
    """Tests for ConnectionType enum."""

    def test_local_type(self) -> None:
        """LOCAL type for stdio/Docker connections."""
        assert ConnectionType.LOCAL.value == "local"

    def test_remote_type(self) -> None:
        """REMOTE type for HTTP connections."""
        assert ConnectionType.REMOTE.value == "remote"


class TestConnector:
    """Tests for Connector class."""

    def test_init_stores_transport_config(self) -> None:
        """__init__ stores transport config, doesn't create client."""
        transport = {"server": {"url": "http://example.com"}}
        config = ConnectionConfig()

        connector = Connector(
            transport=transport,
            config=config,
            name="test",
            connection_type=ConnectionType.REMOTE,
            auth="test-token",
        )

        assert connector._transport == transport
        assert connector._auth == "test-token"
        assert connector.name == "test"
        assert connector.connection_type == ConnectionType.REMOTE
        assert connector.client is None  # Not created yet
        assert connector._tools_cache is None

    def test_is_local_property(self) -> None:
        """is_local returns True for LOCAL connections."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="local-test",
            connection_type=ConnectionType.LOCAL,
        )
        assert connector.is_local is True
        assert connector.is_remote is False

    def test_is_remote_property(self) -> None:
        """is_remote returns True for REMOTE connections."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="remote-test",
            connection_type=ConnectionType.REMOTE,
        )
        assert connector.is_remote is True
        assert connector.is_local is False

    def test_is_connected_false_when_no_client(self) -> None:
        """is_connected returns False when client is None."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )
        assert connector.is_connected is False

    def test_cached_tools_empty_initially(self) -> None:
        """cached_tools returns empty list initially."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )
        assert connector.cached_tools == []

    @pytest.mark.asyncio
    async def test_connect_creates_client(self) -> None:
        """connect() creates FastMCPClient and enters context."""
        transport = {"server": {"url": "http://example.com"}}
        connector = Connector(
            transport=transport,
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
            auth="test-token",
        )

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.is_connected = MagicMock(return_value=True)

        # Patch where it's imported from, not where it's used
        with patch("fastmcp.client.Client", return_value=mock_client) as mock_cls:
            await connector.connect()

            # Client was created with correct args
            mock_cls.assert_called_once_with(transport=transport, auth="test-token")
            # Client context was entered
            mock_client.__aenter__.assert_called_once()
            # Client is now set
            assert connector.client is mock_client

    @pytest.mark.asyncio
    async def test_disconnect_clears_client(self) -> None:
        """disconnect() exits client context and clears state."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.is_connected = MagicMock(return_value=True)
        connector.client = mock_client
        connector._tools_cache = [MagicMock()]

        await connector.disconnect()

        mock_client.__aexit__.assert_called_once_with(None, None, None)
        assert connector.client is None
        assert connector._tools_cache is None

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_not_connected(self) -> None:
        """list_tools() raises RuntimeError when not connected."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            await connector.list_tools()

    @pytest.mark.asyncio
    async def test_list_tools_applies_include_filter(self) -> None:
        """list_tools() filters tools based on include list."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(include=["tool1"]),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                mcp_types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                mcp_types.Tool(name="tool2", description="Tool 2", inputSchema={}),
            ]
        )
        connector.client = mock_client

        tools = await connector.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_list_tools_applies_exclude_filter(self) -> None:
        """list_tools() filters out tools in exclude list."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(exclude=["tool2"]),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                mcp_types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                mcp_types.Tool(name="tool2", description="Tool 2", inputSchema={}),
            ]
        )
        connector.client = mock_client

        tools = await connector.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_list_tools_applies_prefix(self) -> None:
        """list_tools() adds prefix to tool names."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(prefix="myprefix"),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                mcp_types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            ]
        )
        connector.client = mock_client

        tools = await connector.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "myprefix_tool1"

    @pytest.mark.asyncio
    async def test_list_tools_caches_results(self) -> None:
        """list_tools() caches results."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                mcp_types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            ]
        )
        connector.client = mock_client

        tools = await connector.list_tools()

        assert connector._tools_cache == tools
        assert connector.cached_tools == tools

    @pytest.mark.asyncio
    async def test_call_tool_strips_prefix(self) -> None:
        """call_tool() strips prefix before calling."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(prefix="myprefix"),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        mock_result = mcp_types.CallToolResult(content=[], isError=False)
        mock_client = MagicMock()
        mock_client.call_tool_mcp = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        await connector.call_tool("myprefix_tool1", {"arg": "value"})

        # Prefix should be stripped
        mock_client.call_tool_mcp.assert_called_once_with("tool1", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_raises_when_not_connected(self) -> None:
        """call_tool() raises RuntimeError when not connected."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="test",
            connection_type=ConnectionType.REMOTE,
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            await connector.call_tool("tool1", {})

    def test_repr(self) -> None:
        """__repr__ shows useful info."""
        connector = Connector(
            transport={},
            config=ConnectionConfig(),
            name="my-server",
            connection_type=ConnectionType.REMOTE,
        )

        repr_str = repr(connector)
        assert "my-server" in repr_str
        assert "remote" in repr_str
        assert "connected=False" in repr_str

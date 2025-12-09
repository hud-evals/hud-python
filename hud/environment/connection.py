"""Connection management for MCP servers."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.client import Client as FastMCPClient
    from fastmcp.tools.tool import Tool

__all__ = ["ConnectionConfig", "ConnectionType", "Connector"]

logger = logging.getLogger(__name__)


class ConnectionType(str, Enum):
    """Type of connection - determines parallelization capability."""

    LOCAL = "local"  # Stdio/Docker - single instance, not parallelizable
    REMOTE = "remote"  # HTTP/URL - can spawn multiple instances


class ConnectionConfig:
    """Configuration for filtering/transforming tools from a remote connection."""

    def __init__(
        self,
        *,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> None:
        self.prefix = prefix
        self.include = include
        self.exclude = exclude
        self.transform = transform


class Connector:
    """Manages a connection to an MCP server with tool caching.

    Client creation is deferred to connect() so that:
    1. Each parallel trace gets fresh client instances
    2. Connection happens inside trace context (for header injection)
    """

    def __init__(
        self,
        transport: Any,
        config: ConnectionConfig,
        name: str,
        connection_type: ConnectionType,
        *,
        auth: str | None = None,
    ) -> None:
        # Store transport config - client created in connect()
        self._transport = transport
        self._auth = auth
        self.config = config
        self.name = name
        self.connection_type = connection_type
        self.client: FastMCPClient[Any] | None = None
        self._tools_cache: list[mcp_types.Tool] | None = None

    def copy(self) -> Connector:
        """Create a copy of this connector with fresh (unconnected) state.

        The copy shares transport config but has its own client instance,
        allowing parallel execution without conflicts.
        """
        return Connector(
            transport=self._transport,
            config=self.config,
            name=self.name,
            connection_type=self.connection_type,
            auth=self._auth,
        )

    @property
    def is_local(self) -> bool:
        """True if this is a local (non-parallelizable) connection."""
        return self.connection_type == ConnectionType.LOCAL

    @property
    def is_remote(self) -> bool:
        """True if this is a remote (parallelizable) connection."""
        return self.connection_type == ConnectionType.REMOTE

    @property
    def is_connected(self) -> bool:
        return self.client is not None and self.client.is_connected()

    @property
    def cached_tools(self) -> list[mcp_types.Tool]:
        return self._tools_cache or []

    async def connect(self) -> None:
        """Create FastMCP client and connect.

        Client is created here (not in __init__) so that:
        1. Each parallel trace gets fresh client instances
        2. httpx auto-instrumentation can inject trace headers
        """
        from fastmcp.client import Client as FastMCPClient

        # Create fresh client from stored transport config
        self.client = FastMCPClient(transport=self._transport, auth=self._auth)
        await self.client.__aenter__()

    async def disconnect(self) -> None:
        """Disconnect and clear cache."""
        if self.client is not None and self.is_connected:
            await self.client.__aexit__(None, None, None)
        self.client = None
        self._tools_cache = None

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Fetch tools from server, apply filters/transforms/prefix, and cache."""
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        tools = await self.client.list_tools()

        result: list[mcp_types.Tool] = []
        for tool in tools:
            # Apply include/exclude filter
            if self.config.include is not None and tool.name not in self.config.include:
                continue
            if self.config.exclude is not None and tool.name in self.config.exclude:
                continue

            # Apply transform
            if self.config.transform is not None:
                from fastmcp.tools.tool import Tool as FastMCPTool

                fastmcp_tool = FastMCPTool.model_construct(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                )
                transformed = self.config.transform(fastmcp_tool)
                if transformed is None:
                    continue
                tool = mcp_types.Tool(
                    name=transformed.name,
                    description=transformed.description,
                    inputSchema=transformed.parameters,
                )

            # Apply prefix
            name = f"{self.config.prefix}_{tool.name}" if self.config.prefix else tool.name
            result.append(
                mcp_types.Tool(
                    name=name,
                    description=tool.description,
                    inputSchema=tool.inputSchema,
                )
            )

        self._tools_cache = result
        return result

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mcp_types.CallToolResult:
        """Call a tool, stripping prefix if needed."""
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        # Strip prefix when calling remote
        if self.config.prefix and name.startswith(f"{self.config.prefix}_"):
            name = name[len(self.config.prefix) + 1 :]
        return await self.client.call_tool_mcp(name, arguments or {})

    async def list_resources(self) -> list[mcp_types.Resource]:
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        return await self.client.list_resources()

    async def list_prompts(self) -> list[mcp_types.Prompt]:
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        return await self.client.list_prompts()

    async def read_resource(
        self, uri: str
    ) -> list[mcp_types.TextResourceContents | mcp_types.BlobResourceContents]:
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        return await self.client.read_resource(uri)

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mcp_types.GetPromptResult:
        if self.client is None:
            raise RuntimeError("Not connected - call connect() first")
        return await self.client.get_prompt(name, arguments)

    def __repr__(self) -> str:
        t = self.connection_type.value
        return f"Connector({self.name!r}, {t}, connected={self.is_connected})"

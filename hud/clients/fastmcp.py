"""FastMCP-based client implementation."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from fastmcp import Client as FastMCPClient
from mcp import types

from hud.types import MCPToolResult

from .base import BaseHUDClient

if TYPE_CHECKING:
    from pydantic import AnyUrl

logger = logging.getLogger(__name__)


class FastMCPHUDClient(BaseHUDClient):
    """FastMCP-based implementation of HUD MCP client."""

    def __init__(self, mcp_config: dict[str, dict[str, Any]], **kwargs) -> None:
        """
        Initialize FastMCP client.

        Args:
            mcp_config: MCP server configuration dict
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(mcp_config, **kwargs)

        # Convert to FastMCP config format
        config = {"mcpServers": mcp_config}

        # Create FastMCP client
        from mcp.types import Implementation

        client_info = Implementation(name="hud-python", version="3.0.3")

        self._client = FastMCPClient(config, client_info=client_info)
        self._stack: AsyncExitStack | None = None

    async def _connect(self) -> None:
        """Enter FastMCP context to establish connection."""
        if self._stack is None:
            self._stack = AsyncExitStack()
            await self._stack.enter_async_context(self._client)
            logger.info("FastMCP client connected")

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        return await self._client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Execute a tool by name."""
        # FastMCP returns a different result type, convert it
        result = await self._client.call_tool(
            name=name,
            arguments=arguments or {},
            raise_on_error=False,  # Don't raise, return error in result
        )

        # Convert FastMCP result to MCPToolResult
        return MCPToolResult(
            content=result.content,
            isError=result.is_error,
            structuredContent=result.structured_content,
        )

    async def list_resources(self) -> list[types.Resource]:
        """List all available resources."""
        return await self._client.list_resources()

    async def _read_resource_internal(self, uri: str | AnyUrl) -> types.ReadResourceResult | None:
        """Read a resource by URI."""
        try:
            contents = await self._client.read_resource(uri)
            return types.ReadResourceResult(contents=contents)
        except Exception as e:
            if self.verbose:
                logger.debug("Could not read resource '%s': %s", uri, e)
            return None

    async def close(self) -> None:
        """Close the client connection."""
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self._initialized = False
            logger.info("FastMCP client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

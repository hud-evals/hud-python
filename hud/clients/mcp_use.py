"""MCP-use based client implementation (legacy)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp.shared.exceptions import McpError
from mcp_use.client import MCPClient as MCPUseClient
from pydantic import AnyUrl

from hud.types import MCPToolResult

from .base import BaseHUDClient

if TYPE_CHECKING:
    from mcp import types
    from mcp_use.session import MCPSession as MCPUseSession

logger = logging.getLogger(__name__)


class MCPUseHUDClient(BaseHUDClient):
    """MCP-use based implementation of HUD MCP client."""

    def __init__(self, mcp_config: dict[str, dict[str, Any]], **kwargs) -> None:
        """
        Initialize MCP-use client.

        Args:
            mcp_config: MCP server configuration dict
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(mcp_config, **kwargs)

        # Initialize mcp_use client with proper config
        config = {"mcpServers": mcp_config}
        self._mcp_client = MCPUseClient.from_dict(config)

        self._sessions: dict[str, MCPUseSession] = {}
        self._tool_map: dict[str, tuple[str, types.Tool]] = {}

    async def _connect(self) -> None:
        """Create all sessions for MCP-use client."""
        try:
            self._sessions = await self._mcp_client.create_all_sessions()
            logger.info("Created %d MCP sessions", len(self._sessions))

            # Log session details in verbose mode
            if self.verbose and self._sessions:
                for name, session in self._sessions.items():
                    logger.debug("  - %s: %s", name, type(session).__name__)

        except McpError as e:
            # Protocol error - the server is reachable but rejecting our request
            logger.error("MCP protocol error: %s", e)
            logger.error("This typically means:")
            logger.error("- Invalid or missing initialization parameters")
            logger.error("- Incompatible protocol version")
            logger.error("- Server-side configuration issues")
            raise
        except Exception as e:
            # Transport or other errors
            logger.error("Failed to create sessions: %s", e)
            if self.verbose:
                logger.info("Check that the MCP server is running and accessible")
            raise

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools from all sessions."""
        all_tools = []
        self._tool_map = {}

        for server_name, session in self._sessions.items():
            try:
                # Ensure session is initialized
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    await session.initialize()

                if session.connector.client_session is None:
                    logger.warning("Client session not initialized for %s", server_name)
                    continue

                # List tools
                tools_result = await session.connector.client_session.list_tools()

                logger.info(
                    "Discovered %d tools from '%s': %s",
                    len(tools_result.tools),
                    server_name,
                    [tool.name for tool in tools_result.tools],
                )

                # Add to collections
                for tool in tools_result.tools:
                    all_tools.append(tool)
                    self._tool_map[tool.name] = (server_name, tool)

                # Log detailed tool info in verbose mode
                if self.verbose:
                    for tool in tools_result.tools:
                        description = tool.description or ""
                        logger.debug(
                            "  Tool '%s': %s",
                            tool.name,
                            description[:100] + "..." if len(description) > 100 else description,
                        )

            except Exception as e:
                logger.error("Error discovering tools from '%s': %s", server_name, e)
                if self.verbose:
                    logger.exception("Full error details:")

        return all_tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Execute a tool by name."""
        if name not in self._tool_map:
            raise ValueError(f"Tool '{name}' not found")

        server_name, _ = self._tool_map[name]
        session = self._sessions[server_name]

        if self.verbose:
            logger.debug(
                "Calling tool '%s' on server '%s' with arguments: %s",
                name,
                server_name,
                arguments,
            )

        if session.connector.client_session is None:
            raise ValueError(f"Client session not initialized for {server_name}")

        result = await session.connector.client_session.call_tool(
            name=name,
            arguments=arguments or {},
        )

        if self.verbose:
            logger.debug("Tool '%s' result: %s", name, result)

        # MCP-use already returns the correct type, but we need to ensure it's MCPToolResult
        return MCPToolResult(
            content=result.content,
            isError=result.isError,
            structuredContent=result.structuredContent,
        )

    async def list_resources(self) -> list[types.Resource]:
        """List all available resources."""
        for server_name, session in self._sessions.items():
            try:
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    continue
                if session.connector.client_session is None:
                    continue
                # Prefer standard method name if available
                if hasattr(session.connector.client_session, "list_resources"):
                    resources = await session.connector.client_session.list_resources()
                else:
                    # If the client doesn't support resource listing, skip
                    continue
                return resources.resources
            except Exception as e:
                if self.verbose:
                    logger.debug("Could not list resources from server '%s': %s", server_name, e)
                continue
        return []

    async def _read_resource_internal(self, uri: str | AnyUrl) -> types.ReadResourceResult | None:
        """Read a resource by URI from any server that provides it."""
        for server_name, session in self._sessions.items():
            try:
                if not hasattr(session, "connector") or not hasattr(
                    session.connector, "client_session"
                ):
                    continue

                if session.connector.client_session is None:
                    continue

                # Convert str to AnyUrl if needed
                resource_uri = AnyUrl(uri) if isinstance(uri, str) else uri
                # Prefer read_resource; fall back to list_resources if needed
                if hasattr(session.connector.client_session, "read_resource"):
                    result = await session.connector.client_session.read_resource(resource_uri)
                else:
                    # Fallback path for older clients: not supported in strict typing
                    raise AttributeError("read_resource not available")

                if self.verbose:
                    logger.debug(
                        "Successfully read resource '%s' from server '%s'", uri, server_name
                    )

                return result

            except Exception as e:
                if self.verbose:
                    logger.debug(
                        "Could not read resource '%s' from server '%s': %s", uri, server_name, e
                    )
                continue

        return None

    async def close(self) -> None:
        """Close all active sessions."""
        await self._mcp_client.close_all_sessions()
        self._sessions = {}
        self._tool_map = {}
        self._initialized = False
        logger.info("MCP-use client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Legacy compatibility methods (limited; tests should not rely on these)
    def get_sessions(self) -> dict[str, MCPUseSession]:
        """Get active MCP sessions."""
        return self._sessions

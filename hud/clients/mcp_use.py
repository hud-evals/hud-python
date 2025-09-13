"""MCP-use based client implementation (legacy)."""

from __future__ import annotations

import logging
from typing import Any

from mcp import Implementation, types
from mcp.shared.exceptions import McpError
from mcp_use.client import MCPClient as MCPUseClient
from mcp_use.session import MCPSession as MCPUseSession
from pydantic import AnyUrl

from hud.types import MCPToolCall, MCPToolResult
from hud.version import __version__ as hud_version

from .base import BaseHUDClient
from .utils.mcp_use_retry import patch_all_sessions

logger = logging.getLogger(__name__)


class MCPUseHUDClient(BaseHUDClient):
    """MCP-use based implementation of HUD MCP client."""

    client_info = Implementation(
        name="hud-mcp-use", title="hud MCP-use Client", version=hud_version
    )

    def __init__(self, mcp_config: dict[str, dict[str, Any]] | None = None, **kwargs: Any) -> None:
        """
        Initialize MCP-use client.

        Args:
            mcp_config: MCP server configuration dict
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(mcp_config=mcp_config, **kwargs)

        if MCPUseClient is None or MCPUseSession is None:
            raise ImportError(
                "MCP-use dependencies are not available. "
                "Please install the optional agent dependencies: pip install 'hud-python[agent]'"
            )

        self._sessions: dict[str, Any] = {}  # Will be MCPUseSession when available
        self._tool_map: dict[
            str, tuple[str, types.Tool, types.Tool]
        ] = {}  # server_name, original_tool, prefixed_tool
        self._client: Any | None = None  # Will be MCPUseClient when available

    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Create all sessions for MCP-use client."""
        if self._client is not None:
            logger.warning("Client is already connected, cannot connect again")
            return

        config = {"mcpServers": mcp_config}
        if MCPUseClient is None:
            raise ImportError("MCPUseClient is not available")
        self._client = MCPUseClient.from_dict(config)
        try:
            assert self._client is not None  # noqa: S101
            self._sessions = await self._client.create_all_sessions()
            logger.info("Created %d MCP sessions", len(self._sessions))

            # Patch all sessions with retry logic
            patch_all_sessions(self._sessions)
            logger.debug("Applied retry logic to all MCP sessions")

            # Configure validation for all sessions based on client setting
            try:
                for session in self._sessions.values():
                    if (
                        hasattr(session, "connector")
                        and hasattr(session.connector, "client_session")
                        and session.connector.client_session is not None
                    ):
                        session.connector.client_session._validate_structured_outputs = (
                            self._strict_validation
                        )
            except ImportError:
                # ValidationOptions may not be available in some mcp versions
                pass

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

        # Populate tool map during initialization
        await self.list_tools()

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools from all sessions."""
        if self._client is None or not self._sessions:
            raise ValueError("Client is not connected, call initialize() first")

        if self._tool_map:
            return [tool[2] for tool in self._tool_map.values()]

        all_tools = []
        self._tool_map = {}

        # Check if we need to prefix (more than one server)
        use_prefix = len(self._sessions) > 1

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

                # List tools (retry logic is handled at transport level)
                tools_result = await session.connector.client_session.list_tools()

                logger.info(
                    "Discovered %d tools from '%s': %s",
                    len(tools_result.tools),
                    server_name,
                    [tool.name for tool in tools_result.tools],
                )

                # Add to collections with optional prefix
                for tool in tools_result.tools:
                    if use_prefix:
                        # Create a new tool with prefixed name
                        prefixed_name = f"{server_name}_{tool.name}"
                        # Create a new tool instance with prefixed name
                        from mcp import types as mcp_types

                        prefixed_tool = mcp_types.Tool(
                            name=prefixed_name,
                            description=tool.description,
                            inputSchema=tool.inputSchema,
                        )
                        all_tools.append(prefixed_tool)
                        # Map prefixed name to (server_name, original_tool)
                        self._tool_map[prefixed_name] = (server_name, tool, prefixed_tool)
                    else:
                        # Single server - no prefix needed
                        all_tools.append(tool)
                        self._tool_map[tool.name] = (server_name, tool, tool)

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

    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        if self._client is None or not self._initialized:
            raise ValueError("Client is not connected, call initialize() first")

        if tool_call.name not in self._tool_map:
            return MCPToolResult(
                content=[types.TextContent(type="text", text=f"Tool '{tool_call.name}' not found")],
                isError=True,
                structuredContent=None,
            )

        server_name, original_tool, _ = self._tool_map[tool_call.name]
        session = self._sessions[server_name]

        if self.verbose:
            logger.debug(
                "Calling tool '%s' (original: '%s') on server '%s' with arguments: %s",
                tool_call.name,
                original_tool.name,
                server_name,
                tool_call.arguments,
            )

        if session.connector.client_session is None:
            raise ValueError(f"Client session not initialized for {server_name}")

        # Call tool (retry logic is handled at transport level)
        result = await session.connector.client_session.call_tool(
            name=original_tool.name,  # Use original tool name, not prefixed
            arguments=tool_call.arguments or {},
        )

        if self.verbose:
            logger.debug("Tool '%s' result: %s", tool_call.name, result)

        # MCP-use already returns the correct type, but we need to ensure it's MCPToolResult
        return MCPToolResult(
            content=result.content,
            isError=result.isError,
            structuredContent=result.structuredContent,
        )

    async def list_resources(self) -> list[types.Resource]:
        """List all available resources."""
        if self._client is None or not self._sessions:
            raise ValueError("Client is not connected, call initialize() first")

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
                    # List resources (retry logic is handled at transport level)
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

    async def read_resource(self, uri: str | AnyUrl) -> types.ReadResourceResult | None:
        """Read a resource by URI from any server that provides it."""
        if self._client is None or not self._sessions:
            raise ValueError("Client is not connected, call initialize() first")

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
                    # Read resource (retry logic is handled at transport level)
                    result = await session.connector.client_session.read_resource(resource_uri)
                else:
                    # Fallback path for older clients: not supported in strict typing
                    raise AttributeError("read_resource not available")

                if self.verbose:
                    logger.debug(
                        "Successfully read resource '%s' from server '%s'", uri, server_name
                    )

                return result

            except McpError as e:
                # McpError is expected for unsupported resources
                if "telemetry://" in str(uri):
                    logger.debug(
                        "Telemetry resource not supported by server '%s': %s", server_name, e
                    )
                elif self.verbose:
                    logger.debug(
                        "MCP resource error for '%s' from server '%s': %s", uri, server_name, e
                    )
                continue
            except Exception as e:
                # Other errors might be more serious
                if "telemetry://" in str(uri):
                    logger.debug("Failed to fetch telemetry from server '%s': %s", server_name, e)
                else:
                    logger.warning(
                        "Unexpected error reading resource '%s' from server '%s': %s",
                        uri,
                        server_name,
                        e,
                    )
                continue

        return None

    async def _disconnect(self) -> None:
        """Close all active sessions."""
        if self._client is None:
            logger.warning("Client is not connected, cannot close")
            return

        await self._client.close_all_sessions()
        self._sessions = {}
        self._tool_map = {}
        self._initialized = False
        logger.debug("MCP-use client disconnected")

    # Legacy compatibility methods (limited; tests should not rely on these)
    def get_sessions(self) -> dict[str, Any]:
        """Get active MCP sessions."""
        return self._sessions

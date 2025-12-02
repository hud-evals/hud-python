"""MCP-use based client implementation (legacy)."""

from __future__ import annotations

import logging
import traceback
from typing import Any
from urllib.parse import urlparse

from mcp import Implementation, types
from mcp.shared.exceptions import McpError
from mcp_use.client import MCPClient as MCPUseClient
from mcp_use.session import MCPSession as MCPUseSession
from mcp_use.types.http import HttpOptions
from pydantic import AnyUrl

from hud.settings import settings
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole
from hud.version import __version__ as hud_version

from .base import BaseHUDClient
from .utils.retry_transport import create_retry_httpx_client

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger=logger)


class MCPUseHUDClient(BaseHUDClient):
    """MCP-use based implementation of HUD MCP client."""

    client_info = Implementation(
        name="hud-mcp-use", title="hud MCP-use Client", version=hud_version
    )

    def __init__(
        self,
        mcp_config: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
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
        # Transport options for MCP-use (disable_sse_fallback, httpx_client_factory, etc.)
        # Default to retry-enabled HTTPX client if factory not provided
        self._http_options: HttpOptions = HttpOptions(
            httpx_client_factory=create_retry_httpx_client,
            disable_sse_fallback=True,
        )

    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Create all sessions for MCP-use client."""
        if self._client is not None:
            logger.warning("Client is already connected, cannot connect again")
            return

        # If a server target matches HUD's MCP host and no auth is provided,
        # inject the HUD API key as a Bearer token to avoid OAuth browser flow.
        try:
            hud_mcp_host = urlparse(settings.hud_mcp_url).netloc
            if mcp_config and settings.api_key and hud_mcp_host:
                for server_cfg in mcp_config.values():
                    server_url = server_cfg.get("url")
                    if not server_url:
                        continue
                    if urlparse(server_url).netloc == hud_mcp_host and not server_cfg.get("auth"):
                        server_cfg["auth"] = settings.api_key
        except Exception:
            logger.warning("Failed to parse HUD MCP URL")

        config = {"mcpServers": mcp_config}
        if MCPUseClient is None:
            raise ImportError("MCPUseClient is not available")
        self._client = MCPUseClient.from_dict(config, http_options=self._http_options)
        try:
            assert self._client is not None
            self._sessions = await self._client.create_all_sessions()
            session_count = len(self._sessions)
            session_text = "session" if session_count == 1 else "sessions"
            hud_console.info(f"Created {session_count} MCP {session_text}")

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
                    hud_console.debug(f"  - {name}: {type(session).__name__}")

        except McpError as e:
            # Protocol error - the server is reachable but rejecting our request
            hud_console.warning(f"MCP protocol error: {e}")
            hud_console.warning("This typically means:")
            hud_console.warning("- Invalid or missing initialization parameters")
            hud_console.warning("- Incompatible protocol version")
            hud_console.warning("- Server-side configuration issues")
            raise
        except Exception as e:
            # Transport or other errors
            hud_console.error(f"Failed to create sessions: {e}")
            if self.verbose:
                hud_console.info("Check that the MCP server is running and accessible")
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
                    hud_console.warning(f"Client session not initialized for {server_name}")
                    continue

                # List tools (retry logic is handled at transport level)
                tools_result = await session.connector.client_session.list_tools()

                hud_console.info(
                    f"Discovered {len(tools_result.tools)} tools from '{server_name}': {', '.join([tool.name for tool in tools_result.tools])}",  # noqa: E501
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
                        hud_console.debug(
                            f"  Tool '{tool.name}': {description[:100] + '...' if len(description) > 100 else description}",  # noqa: E501
                        )

            except Exception as e:
                hud_console.error(f"Error discovering tools from '{server_name}': {e}")
                if self.verbose:
                    hud_console.error("Full error details:")
                    traceback.print_exc()

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
            hud_console.debug(
                f"Calling tool '{tool_call.name}' (original: '{original_tool.name}') on server '{server_name}' with arguments: {tool_call.arguments}"  # noqa: E501
            )

        if session.connector.client_session is None:
            raise ValueError(f"Client session not initialized for {server_name}")

        # Call tool (retry logic is handled at transport level)
        result = await session.connector.client_session.call_tool(
            name=original_tool.name,  # Use original tool name, not prefixed
            arguments=tool_call.arguments or {},
        )

        if self.verbose:
            hud_console.debug(f"Tool '{tool_call.name}' result: {result}")

        # MCP-use already returns the correct type, but we need to ensure it's MCPToolResult
        return MCPToolResult(
            content=result.content,
            isError=result.isError,
            structuredContent=result.structuredContent,
        )

    async def _list_resources_impl(self) -> list[types.Resource]:
        """Implementation of resource listing for MCP-use client."""
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
                    hud_console.debug(f"Could not list resources from server '{server_name}': {e}")
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
                    hud_console.debug(
                        f"Successfully read resource '{uri}' from server '{server_name}'"
                    )

                return result

            except McpError as e:
                # McpError is expected for unsupported resources
                if "telemetry://" in str(uri):
                    hud_console.debug(
                        f"Telemetry resource not supported by server '{server_name}': {e}"
                    )
                elif self.verbose:
                    hud_console.debug(
                        f"MCP resource error for '{uri}' from server '{server_name}': {e}"
                    )
                continue
            except Exception as e:
                # Other errors might be more serious
                if "telemetry://" in str(uri):
                    hud_console.debug(f"Failed to fetch telemetry from server '{server_name}': {e}")
                else:
                    hud_console.warning(
                        f"Unexpected error reading resource '{uri}' from server '{server_name}': {e}"  # noqa: E501
                    )
                continue

        return None

    async def _disconnect(self) -> None:
        """Close all active sessions."""
        if self._client is None:
            hud_console.warning("Client is not connected, cannot close")
            return

        await self._client.close_all_sessions()
        self._sessions = {}
        self._tool_map = {}
        self._initialized = False
        hud_console.debug("MCP-use client disconnected")

    # Legacy compatibility methods (limited; tests should not rely on these)
    def get_sessions(self) -> dict[str, Any]:
        """Get active MCP sessions."""
        return self._sessions

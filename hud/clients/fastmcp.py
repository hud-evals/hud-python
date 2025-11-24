"""FastMCP-based client implementation."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from fastmcp import Client as FastMCPClient
from mcp import Implementation, types
from mcp.shared.exceptions import McpError

from hud.types import MCPToolCall, MCPToolResult
from hud.version import __version__ as hud_version

from .base import BaseHUDClient
from .utils.retry_transport import create_retry_httpx_client

if TYPE_CHECKING:
    from pydantic import AnyUrl

logger = logging.getLogger(__name__)


class FastMCPHUDClient(BaseHUDClient):
    """FastMCP-based implementation of HUD MCP client."""

    client_info = Implementation(
        name="hud-fastmcp", title="hud FastMCP Client", version=hud_version
    )

    def __init__(self, mcp_config: dict[str, dict[str, Any]] | None = None, **kwargs: Any) -> None:
        """
        Initialize FastMCP client with retry support for HTTP transports.

        Args:
            mcp_config: MCP server configuration dict
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(mcp_config=mcp_config, **kwargs)

        self._stack: AsyncExitStack | None = None
        self._client: FastMCPClient | None = None

    def _create_transport_with_retry(self, mcp_config: dict[str, dict[str, Any]]) -> Any:
        """Create transport with retry support for HTTP-based servers."""
        from fastmcp.client.transports import StreamableHttpTransport

        # If single server with HTTP URL, create transport directly with retry
        if len(mcp_config) == 1:
            _, server_config = next(iter(mcp_config.items()))
            url = server_config.get("url", "")

            if url.startswith("http") and not url.endswith("/sse"):
                headers = server_config.get("headers", {})

                logger.debug("Enabling retry mechanism for HTTP transport to %s", url)
                return StreamableHttpTransport(
                    url=url,
                    headers=headers,
                    httpx_client_factory=create_retry_httpx_client,
                )

        # For all other cases, use standard config (no retry)
        return {"mcpServers": mcp_config}

    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:
        """Enter FastMCP context to establish connection."""
        if self._client is not None:
            logger.warning("Client is already connected, cannot connect again")
            return

        # Create FastMCP client with the custom transport
        timeout = 10 * 60  # 5 minutes
        os.environ["FASTMCP_CLIENT_INIT_TIMEOUT"] = str(timeout)

        # Create custom transport with retry support for HTTP servers
        transport = self._create_transport_with_retry(mcp_config)
        self._client = FastMCPClient(transport, client_info=self.client_info, timeout=timeout)

        if self._stack is None:
            self._stack = AsyncExitStack()
            try:
                await self._stack.enter_async_context(self._client)
            except Exception as e:
                # Check for authentication errors
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    # Check if connecting to HUD API
                    for server_config in mcp_config.values():
                        url = server_config.get("url", "")
                        if "mcp.hud.ai" in url:
                            raise RuntimeError(
                                "Authentication failed for HUD API. "
                                "Please ensure your HUD_API_KEY environment variable is set correctly."  # noqa: E501
                                "You can get an API key at https://hud.ai"
                            ) from e
                    # Generic 401 error
                    raise RuntimeError(
                        "Authentication failed (401 Unauthorized). "
                        "Please check your credentials or API key."
                    ) from e
                raise

            # Configure validation for output schemas based on client setting
            try:
                if (
                    hasattr(self._client, "_session_state")
                    and self._client._session_state.session is not None
                ):
                    self._client._session_state.session._validate_structured_outputs = (
                        self._strict_validation
                    )
            except ImportError:
                pass

            logger.info("FastMCP client connected")

    async def list_tools(self) -> list[types.Tool]:
        """List all available tools."""
        if self._client is None:
            raise ValueError("Client is not connected, call initialize() first")
        return await self._client.list_tools()

    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Execute a tool by name."""
        if self._client is None:
            raise ValueError("Client is not connected, call initialize() first")

        # FastMCP returns a different result type, convert it
        result = await self._client.call_tool(
            name=tool_call.name,
            arguments=tool_call.arguments or {},
            raise_on_error=False,  # Don't raise, return error in result
        )

        # Convert FastMCP result to MCPToolResult
        return MCPToolResult(
            content=result.content,
            isError=result.is_error,
            structuredContent=result.structured_content,
        )

    async def _list_resources_impl(self) -> list[types.Resource]:
        """Implementation of resource listing for FastMCP client."""
        if self._client is None:
            raise ValueError("Client is not connected, call initialize() first")
        return await self._client.list_resources()

    async def read_resource(self, uri: str | AnyUrl) -> types.ReadResourceResult | None:
        """Read a resource by URI."""
        if self._client is None:
            raise ValueError("Client is not connected, call initialize() first")
        try:
            contents = await self._client.read_resource(uri)
            return types.ReadResourceResult(contents=contents)
        except McpError as e:
            if "telemetry://" in str(uri):
                logger.debug("Telemetry resource not supported by server: %s", e)
            elif self.verbose:
                logger.debug("MCP resource error for '%s': %s", uri, e)
            return None
        except Exception as e:
            if "telemetry://" in str(uri):
                logger.debug("Failed to fetch telemetry: %s", e)
            else:
                logger.warning("Unexpected error reading resource '%s': %s", uri, e)
            return None

    async def _disconnect(self) -> None:
        """Close the client connection, ensuring the underlying transport is terminated."""
        if self._client is None:
            logger.warning("Client is not connected, cannot disconnect")
            return

        # First close any active async context stack (this triggers client.__aexit__()).
        if self._stack:
            await self._stack.aclose()
            self._stack = None

        try:
            # Close the FastMCP client - this calls transport.close()
            await self._client.close()

            # CRITICAL: Cancel any lingering transport tasks to ensure subprocess termination
            # FastMCP's StdioTransport creates asyncio tasks that can outlive the client
            # We need to handle nested transport structures (MCPConfigTransport -> StdioTransport)
            transport = getattr(self._client, "transport", None)
            if transport:
                # If it's an MCPConfigTransport with a nested transport
                if hasattr(transport, "transport"):
                    transport = transport.transport

                # Now check if it's a StdioTransport with a _connect_task
                if (
                    hasattr(transport, "_connect_task")
                    and transport._connect_task
                    and not transport._connect_task.done()
                ):
                    logger.debug("Canceling lingering StdioTransport connect task")
                    transport._connect_task.cancel()
                    try:
                        await transport._connect_task
                    except asyncio.CancelledError:
                        logger.debug("Transport task cancelled successfully")
                    except Exception as e:
                        logger.debug("Error canceling transport task: %s", e)

        except Exception as e:
            logger.debug("Error while closing FastMCP client transport: %s", e)

        logger.debug("FastMCP client closed")

    async def __aenter__(self: Any) -> Any:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

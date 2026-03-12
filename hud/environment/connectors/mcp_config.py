"""MCP config connection connectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hud.environment.connectors.base import BaseConnectorMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools import Tool

__all__ = ["MCPConfigConnectorMixin"]


class MCPConfigConnectorMixin(BaseConnectorMixin):
    """Mixin providing mcp_config connection methods."""

    def connect_mcp(
        self,
        config: dict[str, dict[str, Any]],
        *,
        alias: str | None = None,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> Any:
        """Connect using an mcp_config dictionary (single server).

        Auto-detects LOCAL (stdio) vs REMOTE (URL) based on config.

        Example:
            ```python
            env = Environment("my-env")

            # Stdio server
            env.connect_mcp(
                {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    }
                }
            )

            async with env:
                await env.call_tool("read_file", path="/tmp/test.txt")
            ```
        """
        from hud.environment.connection import ConnectionType
        from hud.settings import settings

        name = alias or next(iter(config.keys()), "mcp")
        server_config = next(iter(config.values()), {})

        is_local = "command" in server_config or "args" in server_config
        conn_type = ConnectionType.LOCAL if is_local else ConnectionType.REMOTE

        transport: Any = config
        if not is_local and "url" in server_config:
            request_timeout = 840
            timeout = (
                min(request_timeout, settings.client_timeout)
                if settings.client_timeout > 0
                else min(request_timeout, settings.__class__.model_fields["client_timeout"].default)
            )
            transport = _build_transport(server_config, timeout=timeout)

        return self._add_connection(
            name,
            transport,
            connection_type=conn_type,
            prefix=prefix,
            include=include,
            exclude=exclude,
            transform=transform,
        )

    def connect_mcp_config(
        self,
        mcp_config: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Connect multiple servers from an mcp_config dictionary.

        Example:
            ```python
            env = Environment("my-env")

            # Claude Desktop style config
            env.connect_mcp_config(
                {
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    },
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "..."},
                    },
                }
            )

            async with env:
                await env.call_tool("read_file", path="/tmp/test.txt")
                await env.call_tool("search_repositories", query="mcp")
            ```
        """
        # Store mcp_config for serialization (v4 format)
        # Merge with existing if called multiple times
        if not hasattr(self, "_mcp_config") or self._mcp_config is None:
            self._mcp_config = {}
        self._mcp_config.update(mcp_config)

        for server_name, server_config in mcp_config.items():
            self.connect_mcp({server_name: server_config}, alias=server_name, **kwargs)
        return self


def _build_transport(server_config: dict[str, Any], *, timeout: float | None = None) -> Any:
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    from fastmcp.mcp_config import infer_transport_type_from_url

    url = server_config["url"]
    transport_type = server_config.get("transport") or infer_transport_type_from_url(url)
    transport_timeout = timeout if timeout is not None else server_config.get("sse_read_timeout")
    transport_kwargs = {
        "url": url,
        "headers": server_config.get("headers"),
        "auth": server_config.get("auth"),
        "httpx_client_factory": server_config.get("httpx_client_factory"),
    }

    if transport_type == "sse":
        return SSETransport(
            **transport_kwargs,
            sse_read_timeout=transport_timeout,
        )

    transport = StreamableHttpTransport(**transport_kwargs)
    if transport_timeout is not None:
        # FastMCP 3.x wants streamable HTTP timeouts on the client/session,
        # not on the transport constructor.
        cast("Any", transport)._hud_client_timeout = transport_timeout
    return transport

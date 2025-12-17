"""Remote connection connectors - HUD Hub, URL, OpenAPI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools.tool import Tool

__all__ = ["RemoteConnectorMixin"]

logger = logging.getLogger(__name__)


class RemoteConnectorMixin(MCPConfigConnectorMixin):
    """Mixin providing remote connection methods.

    Note: include_router() is inherited from MCPServer (via FastMCP).
    """

    def connect_hub(
        self,
        slug: str,
        *,
        alias: str | None = None,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> Any:
        """Connect to a HUD Hub environment.

        Creates an MCP connection to the HUD API with the hub slug in headers.

        Example:
            ```python
            env = Environment("my-env")
            env.connect_hub("browser")

            async with env:
                await env.call_tool("navigate", url="https://google.com")
            ```
        """
        from hud.settings import settings

        logger.info("Connecting to hub environment: %s", slug)

        # Store hub config for serialization (v5 format)
        # Note: Only first hub is stored for serialization (task configs use single hub)
        if not hasattr(self, "_hub_config") or self._hub_config is None:
            hub_config: dict[str, Any] = {"name": slug}
            if include:
                hub_config["include"] = include
            if exclude:
                hub_config["exclude"] = exclude
            self._hub_config = hub_config

        # Create mcp_config with standard MCP URL and hub slug in headers
        mcp_config = {
            "hud": {
                "url": settings.hud_mcp_url,
                "headers": {
                    "Authorization": f"Bearer {settings.api_key}",
                    "Environment-Name": slug,
                },
            }
        }

        self.connect_mcp_config(
            mcp_config, prefix=prefix, include=include, exclude=exclude, transform=transform
        )
        logger.info("Hub connected: %s", slug)
        return self

    def connect_url(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        alias: str | None = None,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> Any:
        """Connect to an MCP server via URL.

        Example:
            ```python
            env = Environment("my-env")
            env.connect_url(
                "https://mcp.example.com",
                headers={"Authorization": "Bearer token"},
            )

            async with env:
                await env.call_tool("search", query="hello")
            ```
        """
        from hud.environment.connection import ConnectionType

        auth = headers.get("Authorization") if headers else None
        return self._add_connection(
            alias or url,
            url,
            connection_type=ConnectionType.REMOTE,
            auth=auth,
            prefix=prefix,
            include=include,
            exclude=exclude,
            transform=transform,
        )

    def connect_openapi(
        self,
        openapi_spec: dict[str, Any] | str,
        *,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        name: str | None = None,
        prefix: str | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Mount an OpenAPI specification as an MCP server.

        Converts REST API endpoints to MCP tools. Base URL is auto-inferred
        from the spec URL when possible.

        Example:
            ```python
            env = Environment("my-env")
            env.connect_openapi("https://petstore.swagger.io/v2/swagger.json")

            async with env:
                result = await env.call_tool("getPetById", petId=1)
            ```
        """
        from urllib.parse import urlparse

        import httpx
        from fastmcp import FastMCP

        if isinstance(openapi_spec, str):
            if openapi_spec.startswith(("http://", "https://")):
                if base_url is None:
                    parsed = urlparse(openapi_spec)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"

                resp = httpx.get(openapi_spec, headers=headers)
                resp.raise_for_status()
                openapi_spec = resp.json()
            else:
                import json

                with open(openapi_spec) as f:
                    openapi_spec = json.load(f)

        if base_url is None:
            raise ValueError("base_url is required when openapi_spec is a dict or file")

        client = httpx.AsyncClient(base_url=base_url, headers=headers or {}, timeout=timeout)
        mcp_server = FastMCP.from_openapi(
            openapi_spec=cast("dict[str, Any]", openapi_spec),
            client=client,
            name=name or "openapi",
        )
        self.include_router(mcp_server, prefix=prefix)  # type: ignore
        return self

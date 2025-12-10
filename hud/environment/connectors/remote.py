"""Remote connection connectors - HUD Hub, URL, OpenAPI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools.tool import Tool

    from hud.environment.types import HubConfig

__all__ = ["RemoteConnectorMixin"]

logger = logging.getLogger(__name__)


class RemoteConnectorMixin(MCPConfigConnectorMixin):
    """Mixin providing remote connection methods.
    
    Note: include_router() is inherited from MCPServer (via FastMCP).
    """

    # Store hub configs for trace serialization
    _hub_configs: list[HubConfig]

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

        Fetches mcp_config from api.hud.so immediately and creates connectors.

        Example:
            ```python
            env = Environment("my-env")
            env.connect_hub("hud/browser")

            async with env:
                await env.call_tool("navigate", url="https://google.com")
            ```
        """
        import httpx

        from hud.environment.types import HubConfig
        from hud.settings import settings

        # Store hub config for trace serialization
        hub_config = HubConfig(
            slug=slug,
            alias=alias,
            prefix=prefix,
            include=include,
            exclude=exclude,
        )

        if not hasattr(self, "_hub_configs"):
            self._hub_configs = []
        self._hub_configs.append(hub_config)

        # Fetch mcp_config synchronously
        logger.info("Loading hub environment: %s", slug)

        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"

        with httpx.Client() as client:
            response = client.get(
                f"{settings.hud_api_url}/environments/{slug}/mcp-config",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        mcp_config: dict[str, dict[str, Any]] = data.get("mcp_config", data)
        self.connect_mcp_config(
            mcp_config, prefix=prefix, include=include, exclude=exclude, transform=transform
        )
        logger.info("Hub connected: %s (%d servers)", slug, len(mcp_config))
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
        self.include_router(mcp_server, prefix=prefix) # type: ignore
        return self

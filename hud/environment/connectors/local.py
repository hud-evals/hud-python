"""Local connection connectors - Docker image, FastAPI, MCPServer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools.tool import Tool

__all__ = ["LocalConnectorMixin"]


class LocalConnectorMixin(MCPConfigConnectorMixin):
    """Mixin providing local connection methods.

    Methods:
        connect_image(image) - Run Docker image via stdio
        connect_fastapi(app) - Mount FastAPI app as MCP server
        connect_server(server) - Mount any MCPServer/FastMCP directly

    Inherits connect_mcp() from MCPConfigConnectorMixin.

    Note: include_router() is inherited from MCPServer (via FastMCP).
    """

    def connect_image(
        self,
        image: str,
        *,
        alias: str | None = None,
        docker_args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> Any:
        """Connect to a Docker image via stdio.

        Creates an MCP config that runs: docker run -i --rm {image}
        Environment variables from `.env` files are auto-injected.

        Example:
            ```python
            env = Environment("my-env")
            env.connect_image("mcp/fetch")

            async with env:
                result = await env.call_tool("fetch", url="https://example.com")
            ```
        """
        from hud.cli.utils.docker import create_docker_run_command

        cmd = create_docker_run_command(
            image=image,
            docker_args=docker_args,
            extra_env=env_vars,
            interactive=True,
            remove=True,
        )

        name = alias or image
        mcp_config = {
            name: {
                "command": cmd[0],
                "args": cmd[1:],
            }
        }
        return self.connect_mcp(
            mcp_config,
            alias=name,
            prefix=prefix,
            include=include,
            exclude=exclude,
            transform=transform,
        )

    def connect_fastapi(
        self,
        app: Any,
        *,
        name: str | None = None,
        prefix: str | None = None,
        include_hidden: bool = True,
    ) -> Any:
        """Import a FastAPI application's routes as MCP tools.

        Uses FastMCP's from_fastapi() to convert FastAPI endpoints to MCP tools,
        then imports them synchronously so they're available immediately.

        Args:
            app: FastAPI application instance
            name: Custom name for the server (defaults to app.title)
            prefix: Optional prefix for tool names
            include_hidden: If True (default), includes routes with include_in_schema=False

        Example:
            ```python
            from fastapi import FastAPI

            api = FastAPI()


            @api.get("/users/{user_id}", operation_id="get_user")
            def get_user(user_id: int):
                return {"id": user_id, "name": "Alice"}


            env = Environment("my-env")
            env.connect_fastapi(api)

            async with env:
                result = await env.call_tool("get_user", user_id=1)
            ```

        Tip: Use operation_id in FastAPI decorators for cleaner tool names.
        """
        from fastmcp import FastMCP

        # Temporarily enable hidden routes for OpenAPI generation
        hidden_routes: list[Any] = []
        if include_hidden:
            for route in getattr(app, "routes", []):
                if hasattr(route, "include_in_schema") and not route.include_in_schema:
                    hidden_routes.append(route)
                    route.include_in_schema = True
            # Clear cached openapi schema so it regenerates
            if hasattr(app, "openapi_schema"):
                app.openapi_schema = None

        try:
            server_name = name or getattr(app, "title", None) or "fastapi"
            mcp_server = FastMCP.from_fastapi(app=app, name=server_name)
            # Use include_router for synchronous import (tools available immediately)
            self.include_router(mcp_server, prefix=prefix)  # type: ignore
        finally:
            # Restore original states
            for route in hidden_routes:
                route.include_in_schema = False
            if hidden_routes and hasattr(app, "openapi_schema"):
                app.openapi_schema = None  # Clear cache again

        return self

    def connect_server(
        self,
        server: Any,
        *,
        prefix: str | None = None,
    ) -> Any:
        """Import an MCPServer or FastMCP instance's tools directly.

        Example:
            ```python
            from fastmcp import FastMCP

            tools = FastMCP("tools")


            @tools.tool
            def greet(name: str) -> str:
                return f"Hello, {name}!"


            env = Environment("my-env")
            env.connect_server(tools)

            async with env:
                result = await env.call_tool("greet", name="World")
            ```
        """
        self.include_router(server, prefix=prefix)  # type: ignore
        return self

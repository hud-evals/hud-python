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
    """

    def mount(self, server: Any, *, prefix: str | None = None) -> None:
        """Mount method from MCPServer base class."""
        raise NotImplementedError

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
    ) -> Any:
        """Mount a FastAPI application as an MCP server.

        Uses FastMCP's from_fastapi() to convert FastAPI endpoints to MCP tools.

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

        server_name = name or getattr(app, "title", None) or "fastapi"
        mcp_server = FastMCP.from_fastapi(app=app, name=server_name)
        self.mount(mcp_server, prefix=prefix)
        return self

    def connect_server(
        self,
        server: Any,
        *,
        prefix: str | None = None,
    ) -> Any:
        """Mount an MCPServer or FastMCP instance directly.

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
        self.mount(server, prefix=prefix)
        return self

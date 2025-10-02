"""MCP Router utilities for FastAPI-like composition patterns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hud.server.server import MCPServer

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp import FastMCP
    from fastmcp.tools import Tool

logger = logging.getLogger(__name__)

# MCPRouter is just an alias to FastMCP for FastAPI-like patterns
MCPRouter = MCPServer

# Prefix for internal tool names
_INTERNAL_PREFIX = "int_"


class HiddenRouter(MCPRouter):
    """Wraps a FastMCP router to provide a single dispatcher tool for its sub-tools.

    Instead of exposing all tools at the top level, this creates a single tool
    (named after the router) that dispatches to the router's tools internally.

    Useful for setup/evaluate patterns where you want:
    - A single 'setup' tool that can call setup_basic(), setup_advanced(), etc.
    - A single 'evaluate' tool that can call evaluate_score(), evaluate_complete(), etc.

    Example:
        # Create a router with multiple setup functions
        setup_router = MCPRouter(name="setup")

        @setup_router.tool
        async def reset():
            return "Environment reset"

        @setup_router.tool
        async def seed_data():
            return "Data seeded"

        # Wrap in HiddenRouter
        hidden_setup = HiddenRouter(setup_router)

        # Now you have one 'setup' tool that dispatches to reset/seed_data
        mcp.include_router(hidden_setup)
    """

    def __init__(
        self,
        router: FastMCP,
        *,
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Wrap an existing router with a dispatcher pattern.

        Args:
            router: The FastMCP router to wrap
            title: Optional title for the dispatcher tool (defaults to "{name} Dispatcher")
            description: Optional description for the dispatcher tool
            meta: Optional metadata for the dispatcher tool
        """
        name = router.name or "router"

        # Naming scheme for hidden/internal tools
        self._prefix_fn: Callable[[str], str] = lambda n: f"{_INTERNAL_PREFIX}{n}"

        super().__init__(name=name)

        # Set up dispatcher tool
        dispatcher_title = title or f"{name.title()} Dispatcher"
        dispatcher_desc = description or f"Call internal '{name}' functions"

        # Register dispatcher that routes to hidden tools
        async def _dispatch(
            name: str,
            arguments: dict | str | None = None,
            ctx: Any | None = None,
        ) -> Any:
            """Gateway to hidden tools.

            Args:
                name: Internal function name (without prefix)
                arguments: Arguments to forward to the internal tool (dict or JSON string)
                ctx: Request context injected by FastMCP
            """
            # Handle JSON string inputs
            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            # Call the internal tool
            return await self._tool_manager.call_tool(self._prefix_fn(name), arguments or {})  # type: ignore

        from fastmcp.tools.tool import FunctionTool

        dispatcher_tool = FunctionTool.from_function(
            _dispatch,
            name=name,
            title=dispatcher_title,
            description=dispatcher_desc,
            tags=set(),
            meta=meta,
        )
        self._tool_manager.add_tool(dispatcher_tool)

        # Copy all tools from source router as hidden tools
        for tool in router._tool_manager._tools.values():
            tool._key = self._prefix_fn(tool.name)
            self._tool_manager.add_tool(tool)

        # Expose list of available functions via resource
        async def _functions_catalogue() -> list[str]:
            """List all internal function names without prefix."""
            return [
                key.removeprefix(_INTERNAL_PREFIX)
                for key in self._tool_manager._tools
                if key.startswith(_INTERNAL_PREFIX)
            ]

        from fastmcp.resources import Resource

        catalogue_resource = Resource.from_function(
            _functions_catalogue,
            uri=f"{name}://functions",
            name=f"{name.title()} Functions",
            description=f"List of available {name} functions",
        )
        self._resource_manager.add_resource(catalogue_resource)

    # Override _list_tools to hide internal tools when mounted
    async def _list_tools(self) -> list[Tool]:
        """Override _list_tools to hide internal tools when mounted."""
        return [
            tool
            for key, tool in self._tool_manager._tools.items()
            if not key.startswith(_INTERNAL_PREFIX)
        ]

    def _sync_list_tools(self) -> dict[str, Tool]:
        """Override _list_tools to hide internal tools when mounted."""
        return {
            key: tool
            for key, tool in self._tool_manager._tools.items()
            if not key.startswith(_INTERNAL_PREFIX)
        }


__all__ = ["HiddenRouter", "MCPRouter"]

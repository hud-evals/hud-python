"""HiddenRouter -- wraps a FastMCP router with a dispatcher + hidden tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from hud.server.server import MCPServer

if TYPE_CHECKING:
    from fastmcp.tools import Tool

_INTERNAL_PREFIX = "int_"

__all__ = ["HiddenRouter", "MCPRouter"]


class HiddenRouter(FastMCP):
    """A composition-friendly FastMCP server that hides internal tools behind a dispatcher.

    Internal tools are prefixed and only accessible through the dispatcher tool.
    """

    def __init__(
        self,
        name: str,
        *,
        router: FastMCP | None = None,
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name)

        self._prefix_fn = lambda n: f"{_INTERNAL_PREFIX}{n}"

        dispatcher_title = title or f"{name.title()} Dispatcher"
        dispatcher_desc = description or f"Call internal '{name}' functions"
        hidden_self = self

        async def _dispatch(
            name: str,
            arguments: dict[str, Any] | str | None = None,
            ctx: Any | None = None,
        ) -> Any:
            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            prefixed = hidden_self._prefix_fn(name)
            tool = await hidden_self._local_provider.get_tool(prefixed)
            if tool is None:
                raise ValueError(f"Internal tool '{name}' not found")
            args = arguments if isinstance(arguments, dict) else {}
            return await tool.run(args)

        from fastmcp.tools.function_tool import FunctionTool

        dispatcher_tool = FunctionTool.from_function(
            _dispatch,
            name=name,
            title=dispatcher_title,
            description=dispatcher_desc,
            tags=set(),
            meta=meta,
        )
        self._local_provider.add_tool(dispatcher_tool)

        if router is not None:
            self._copy_tools_from(router)

        async def _functions_catalogue() -> list[str]:
            tools = await hidden_self._local_provider.list_tools()
            return [
                t.name.removeprefix(_INTERNAL_PREFIX)
                for t in tools
                if t.name.startswith(_INTERNAL_PREFIX)
            ]

        from fastmcp.resources import Resource

        catalogue_resource = Resource.from_function(
            _functions_catalogue,
            uri=f"{name}://functions",
            name=f"{name.title()} Functions",
            description=f"List of available {name} functions",
        )
        self._local_provider.add_resource(catalogue_resource)

    def _copy_tools_from(self, router: FastMCP) -> None:
        """Copy tools from a source router as hidden (prefixed) tools."""
        src_components = router._local_provider._components
        for key, comp in src_components.items():
            if not key.startswith("tool:"):
                continue
            prefixed_name = self._prefix_fn(comp.name)
            comp_copy = comp.model_copy(update={"name": prefixed_name})
            comp_copy._key = f"tool:{prefixed_name}@"  # type: ignore[attr-defined]
            self._local_provider.add_tool(comp_copy)  # type: ignore[arg-type]

    async def _list_tools(self, context: Any = None) -> list[Tool]:
        """Hide internal tools -- only show the dispatcher."""
        tools = await self._local_provider.list_tools()
        return [t for t in tools if not t.name.startswith(_INTERNAL_PREFIX)]

    def _sync_list_tools(self) -> dict[str, Any]:
        """Sync version of tool listing without internal tools."""
        components = self._local_provider._components
        return {
            k: v
            for k, v in components.items()
            if k.startswith("tool:") and not v.name.startswith(_INTERNAL_PREFIX)
        }


# MCPRouter is an alias for MCPServer for FastAPI-like patterns
MCPRouter = MCPServer

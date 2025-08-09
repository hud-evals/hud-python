from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from fastmcp import FastMCP

from hud.tools.types import ContentBlock, EvaluationResult, SetupResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools import FunctionTool
    from fastmcp.tools.tool import Tool, ToolResult

# Basic result types for tools
BaseResult = list[ContentBlock] | EvaluationResult | SetupResult


class BaseTool(ABC):
    """
    Base helper class for all MCP tools to constrain their output.

    USAGE:
    All tools should inherit from this class and implement the __call__ method.
    Tools are registered with FastMCP using add_tool.

    FORMAT:
    Tools that return messages should return a list[ContentBlock].
    Tools that return miscallaneous content should return a pydantic model such as EvaluationResult.
    Both of these types of tools are processed via structuredContent.
    Any other type of tool will not be processed well by the client.
    """

    def __init__(
        self,
        env: Any = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            env: Optional, often stateful, context object that the tool operates on. Could be:
                - A game instance (e.g., Chess Board)
                - An executor (e.g., PyAutoGUIExecutor for computer control)
                - A browser/page instance (e.g., Playwright Page)
                - Any stateful resource the tool needs to interact with
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
        """
        self.env = env
        self.name = name or self.__class__.__name__.lower().replace("tool", "")
        self.title = title or self.__class__.__name__.replace("Tool", "").replace("_", " ").title()
        self.description = description or (self.__doc__.strip() if self.__doc__ else None)

        # Expose attributes FastMCP expects when registering an instance directly
        self.__name__ = self.name  # FastMCP uses fn.__name__ if name param omitted
        if self.description:
            self.__doc__ = self.description

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Often uses the context to perform an action.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            List of ContentBlock (TextContent, ImageContent, etc.) with the tool's output
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def register(self, server: FastMCP, **meta: Any) -> BaseTool:
        """Register this tool on a FastMCP server and return self for chaining."""
        server.add_tool(self.mcp, **meta)
        return self

    @property
    def mcp(self) -> FunctionTool:
        """Get this tool as a FastMCP FunctionTool (cached).

        This allows clean registration:
            server.add_tool(my_tool.mcp)
        """
        if not hasattr(self, "_mcp_tool"):
            from fastmcp.tools import FunctionTool

            self._mcp_tool = FunctionTool.from_function(
                self,
                name=self.name,
                title=self.title,
                description=self.description,
            )
        return self._mcp_tool


# Prefix for internal tool names
_INTERNAL_PREFIX = "int_"


class BaseHub(FastMCP):
    """A composition-friendly FastMCP server that holds an internal tool dispatcher."""

    env: Any

    def __init__(
        self,
        name: str,
        *,
        env: Any | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Create a new BaseHub.

        Parameters
        ----------
        name:
            Public name. Also becomes the *dispatcher tool* name.
        env:
            Optional long-lived environment object. Stored on the server
            instance (``layer.env``) and therefore available to every request
            via ``ctx.fastmcp.env``.
        """

        # Naming scheme for hidden objects
        self._prefix_fn: Callable[[str], str] = lambda n: f"{_INTERNAL_PREFIX}{n}"

        super().__init__(name=name)

        if env is not None:
            self.env = env

        dispatcher_title = title or f"{name.title()} Dispatcher"
        dispatcher_desc = description or f"Call internal '{name}' functions"

        # Register dispatcher manually with FunctionTool
        async def _dispatch(  # noqa: ANN202
            function: str,
            args: dict | None = None,
            ctx=None,  # noqa: ANN001
        ):
            """Gateway to hidden tools.

            Parameters
            ----------
            function : str
                Internal function name *without* prefix.
            args : dict | None
                Arguments forwarded to the internal tool.
            ctx : Context
                Injected by FastMCP; can be the custom subclass.
            """

            # Use the tool manager to call internal tools
            return await self._tool_manager.call_tool(self._prefix_fn(function), args or {})

        from fastmcp.tools.tool import FunctionTool

        dispatcher_tool = FunctionTool.from_function(
            _dispatch,
            name=name,
            title=dispatcher_title,
            description=dispatcher_desc,
            tags=set(),
        )
        self._tool_manager.add_tool(dispatcher_tool)

        # Expose list of internal functions via read-only resource
        async def _functions_catalogue() -> list[str]:
            # List all internal function names without prefix
            return [
                key.removeprefix(_INTERNAL_PREFIX)
                for key in self._tool_manager._tools
                if key.startswith(_INTERNAL_PREFIX)
            ]

        from fastmcp.resources import Resource

        catalogue_resource = Resource.from_function(
            _functions_catalogue,
            uri=f"file:///{name}/functions",
            name=f"{name} Functions Catalogue",
            description=f"List of internal functions available in {name}",
            mime_type="application/json",
            tags=set(),
        )
        self._resource_manager.add_resource(catalogue_resource)

    def tool(self, name_or_fn: Any = None, /, **kwargs: Any) -> Callable[..., Any]:
        """Register an *internal* tool (hidden from clients)."""
        # Handle when decorator's partial calls us back with the function
        if callable(name_or_fn):
            # This only happens in phase 2 of decorator application
            # The name was already prefixed in phase 1, just pass through
            result = super().tool(name_or_fn, **kwargs)
            return cast("Callable[..., Any]", result)

        # Handle the name from either positional or keyword argument
        if isinstance(name_or_fn, str):
            # Called as @hub.tool("name")
            name = name_or_fn
        elif name_or_fn is None and "name" in kwargs:
            # Called as @hub.tool(name="name")
            name = kwargs.pop("name")
        else:
            # Called as @hub.tool or @hub.tool()
            name = None

        new_name = self._prefix_fn(name) if name is not None else None
        tags = kwargs.pop("tags", None) or set()

        # Pass through correctly to parent
        if new_name is not None:
            return super().tool(new_name, **kwargs, tags=tags)
        else:
            return super().tool(**kwargs, tags=tags)

    # Override _list_tools to hide internal tools when mounted
    async def _list_tools(self) -> list[Tool]:
        """Override _list_tools to hide internal tools when mounted."""
        return [
            tool
            for key, tool in self._tool_manager._tools.items()
            if not key.startswith(_INTERNAL_PREFIX)
        ]

    resource = FastMCP.resource
    prompt = FastMCP.prompt

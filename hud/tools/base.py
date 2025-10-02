from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from fastmcp import FastMCP

from hud.tools.types import ContentBlock, EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastmcp.tools import FunctionTool
    from fastmcp.tools.tool import Tool, ToolResult

# Basic result types for tools
BaseResult = list[ContentBlock] | EvaluationResult

logger = logging.getLogger(__name__)


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
        meta: dict[str, Any] | None = None,
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
            meta: Metadata to include in MCP tool listing (e.g., resolution info)
        """
        self.env = env
        self.name = name or self.__class__.__name__.lower().replace("tool", "")
        self.title = title or self.__class__.__name__.replace("Tool", "").replace("_", " ").title()
        self.description = description or (self.__doc__.strip() if self.__doc__ else None)
        self.meta = meta
        self._callbacks: dict[
            str,
            list[Callable[..., Awaitable[Any]]],
        ] = {}  # {"event_name": [callback_functions]}

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
                meta=self.meta,
            )
        return self._mcp_tool

    def add_callback(self, event_type: str, callback: Callable[..., Awaitable[Any]]) -> None:
        """Register a callback function for specific event

        Args:
            event_type: (Required) Specific event name to trigger callback
                        e.g. "after_click", "before_navigate"
            callback: (Required) Async function to call. Must be defined by `async def f(...)`
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable[..., Awaitable[Any]]) -> None:
        """Remove a registered callback
        Args:
            event_type: (Required) Specific event name to trigger callback
                        e.g. "after_click", "before_navigate"
            callback: (Required) Function to remove from callback list.
        """
        if (event_type in self._callbacks) and (callback in self._callbacks[event_type]):
            self._callbacks[event_type].remove(callback)

    async def _trigger_callbacks(self, event_type: str, **kwargs: Any) -> None:
        """Trigger all registered callback functions of an event type"""
        callback_list = self._callbacks.get(event_type, [])
        for callback in callback_list:
            try:
                await callback(**kwargs)
            except Exception as e:
                logger.warning("Callback failed for %s: %s", event_type, e)


# Prefix for internal tool names
_INTERNAL_PREFIX = "int_"


class BaseHub(FastMCP):
    """A composition-friendly FastMCP server that holds an internal tool dispatcher.

    Note: BaseHub can be used standalone or to wrap existing routers. For the newer
    FastAPI-like pattern, consider using HiddenRouter from hud.server instead.
    """

    env: Any

    def __init__(
        self,
        name: str,
        *,
        env: Any | None = None,
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, Any] | None = None,
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
        title:
            Optional title for the dispatcher tool.
        description:
            Optional description for the dispatcher tool.
        meta:
            Metadata to include in MCP tool listing.
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
            name: str,
            arguments: dict | str | None = None,
            ctx: Any | None = None,
        ):
            """Gateway to hidden tools.

            Parameters
            ----------
            name : str
                Internal function name *without* prefix.
            arguments : dict | str | None
                Arguments forwarded to the internal tool. Can be dict or JSON string.
            ctx : Context
                Injected by FastMCP; can be the custom subclass.
            """

            # Handle JSON string inputs
            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as empty dict
                    arguments = {}

            # Use the tool manager to call internal tools
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

            # Update dispatcher description after registering tool
            self._update_dispatcher_description()

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

    def _update_dispatcher_description(self) -> None:
        """Update the dispatcher tool's description and schema with available tools."""
        # Get list of internal tools with their details
        internal_tools = []
        for key, tool in self._tool_manager._tools.items():
            if key.startswith(_INTERNAL_PREFIX):
                tool_name = key.removeprefix(_INTERNAL_PREFIX)
                internal_tools.append((tool_name, tool))

        if internal_tools:
            # Update the dispatcher tool's description
            dispatcher_name = self.name
            if dispatcher_name in self._tool_manager._tools:
                dispatcher_tool = self._tool_manager._tools[dispatcher_name]

                # Build detailed description
                desc_lines = [f"Call internal '{self.name}' functions. Available tools:"]
                desc_lines.append("")  # Empty line for readability

                # Build tool schemas for oneOf
                tool_schemas = []

                for tool_name, tool in sorted(internal_tools):
                    # Add tool name and description
                    tool_desc = tool.description or "No description"
                    desc_lines.append(f"• Name: {tool_name} ({tool_desc})")

                    # Build schema for this specific tool call
                    tool_schema = {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "const": tool_name,
                                "description": f"Must be '{tool_name}'",
                            },
                            "arguments": tool.parameters
                            if hasattr(tool, "parameters") and tool.parameters
                            else {"type": "object"},
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    }
                    tool_schemas.append(tool_schema)

                    # Add parameters from the tool's parameters field (JSON schema)
                    if hasattr(tool, "parameters") and tool.parameters:
                        schema = tool.parameters
                        if isinstance(schema, dict) and "properties" in schema:
                            params = []
                            required = schema.get("required", [])
                            for prop_name, prop_info in schema["properties"].items():
                                prop_type = prop_info.get("type", "any")
                                # Check for more detailed type info
                                if "anyOf" in prop_info:
                                    types = [
                                        t.get("type", "unknown")
                                        for t in prop_info["anyOf"]
                                        if isinstance(t, dict)
                                    ]
                                    prop_type = " | ".join(types) if types else "any"

                                param_str = f"{prop_name} ({prop_type})"
                                if prop_name not in required:
                                    param_str += " (optional)"
                                params.append(param_str)

                            if params:
                                desc_lines.append(f"  Arguments: {', '.join(params)}")
                            else:
                                desc_lines.append("  Arguments: none")
                    else:
                        desc_lines.append("  Arguments: none")

                    desc_lines.append("")  # Empty line between tools

                dispatcher_tool.description = "\n".join(desc_lines).strip()

                # Update the input schema to better document available tools
                # Build examples of tool calls
                examples = []
                for tool_name, tool in sorted(internal_tools)[:3]:  # Show first 3 as examples
                    if hasattr(tool, "parameters") and tool.parameters:
                        schema = tool.parameters
                        if isinstance(schema, dict) and "properties" in schema:
                            example_args = {}
                            for prop_name, prop_info in schema["properties"].items():
                                # Generate example value based on type
                                prop_type = prop_info.get("type", "any")
                                if prop_type == "string":
                                    example_args[prop_name] = f"<{prop_name}>"
                                elif prop_type == "integer" or prop_type == "number":
                                    example_args[prop_name] = 0
                                elif prop_type == "boolean":
                                    example_args[prop_name] = True
                                else:
                                    example_args[prop_name] = None
                            examples.append({"name": tool_name, "arguments": example_args})
                    else:
                        examples.append({"name": tool_name, "arguments": {}})

                # Enhanced schema with better documentation
                dispatcher_tool.parameters = {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": f"Name of the internal tool to call. Must be one of: {', '.join(t[0] for t in sorted(internal_tools))}",  # noqa: E501
                            "enum": [t[0] for t in sorted(internal_tools)],
                        },
                        "arguments": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "description": "Arguments object to pass to the internal tool",
                                },
                                {
                                    "type": "string",
                                    "description": "JSON string of arguments to pass to the internal tool",  # noqa: E501
                                },
                            ],
                            "description": "Arguments to pass to the internal tool. Can be an object or JSON string. See description for details on each tool's parameters.",  # noqa: E501
                        },
                    },
                    "required": ["name", "arguments"],
                    "examples": examples if examples else None,
                }

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

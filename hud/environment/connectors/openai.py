"""OpenAI Agents SDK connectors - import tools from OpenAI agents."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["OpenAIConnectorMixin"]


class OpenAIConnectorMixin:
    """Mixin providing OpenAI Agents SDK connector methods."""

    # These are defined on Environment/MCPServer
    _tool_manager: Any

    def connect_function_tools(
        self,
        tools: list[Any],
        *,
        prefix: str | None = None,
    ) -> Any:
        """Import FunctionTools from the OpenAI Agents SDK.

        Wraps each tool so calls go through HUD with telemetry.

        Example:
            ```python
            from agents import function_tool


            @function_tool
            def search(query: str) -> str:
                '''Search for information.'''
                return f"Results for {query}"


            @function_tool
            def calculate(expression: str) -> float:
                '''Evaluate a math expression.'''
                return eval(expression)


            env = Environment("my-env")
            env.connect_function_tools([search, calculate])

            async with env:
                result = await env.call_tool("search", query="MCP protocol")
            ```

        Note:
            Requires `openai-agents`: pip install openai-agents
        """
        try:
            from agents import FunctionTool
        except ImportError as e:
            raise ImportError(
                "openai-agents is required for connect_function_tools. "
                "Install with: pip install openai-agents"
            ) from e

        for tool in tools:
            if isinstance(tool, FunctionTool):
                self._add_openai_function_tool(tool, prefix)

        return self

    def _add_openai_function_tool(self, tool: Any, prefix: str | None) -> None:
        """Convert OpenAI FunctionTool to local MCP tool."""
        name = f"{prefix}_{tool.name}" if prefix else tool.name

        # Get the original invoke function
        original_invoke = tool.on_invoke_tool

        # Create wrapper that calls the original
        async def invoke(**arguments: Any) -> Any:
            # OpenAI's on_invoke_tool expects (ToolContext, str_json_args)
            # We need to create a minimal context
            from agents.tool_context import ToolContext

            ctx = ToolContext(context=None)
            result = await original_invoke(ctx, json.dumps(arguments))
            return result

        # Set function metadata for FastMCP
        invoke.__name__ = name
        invoke.__doc__ = tool.description

        # Register using FastMCP's tool decorator mechanism
        # We access the internal _tool_manager from MCPServer
        from fastmcp.tools import Tool as FastMCPTool

        fastmcp_tool = FastMCPTool.from_function(
            fn=invoke,
            name=name,
            description=tool.description,
        )
        # Override the schema with OpenAI's (more accurate)
        fastmcp_tool.parameters = tool.params_json_schema

        self._tool_manager.add_tool(fastmcp_tool)

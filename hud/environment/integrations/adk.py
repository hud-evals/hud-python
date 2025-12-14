"""Google ADK integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.environment.utils.tool_wrappers import create_async_tool_fn

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["ADKMixin"]


class ADKMixin:
    """Mixin providing Google ADK (Agent Development Kit) integration.

    Integration methods (requires google-adk):
        as_adk_tools() - ADK FunctionTool objects

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    def as_adk_tools(self) -> list[Any]:
        """Convert to Google ADK FunctionTool objects.

        Requires: pip install google-adk

        Returns:
            List of FunctionTool objects for Google ADK agents.

        Example:
            ```python
            from google.adk.agents import Agent
            from google.adk.runners import Runner

            async with env:
                agent = Agent(
                    name="assistant",
                    model="gemini-2.0-flash",
                    instruction="You are a helpful assistant.",
                    tools=env.as_adk_tools(),
                )
                runner = Runner(agent=agent)
                result = await runner.run("Find information about Python")
            ```
        """
        try:
            from google.adk.tools.function_tool import FunctionTool
        except ImportError as e:
            raise ImportError(
                "Google ADK not installed. Install with: pip install google-adk"
            ) from e

        tools = []
        for t in self.as_tools():
            # ADK only needs async function - it wraps it in FunctionTool
            async_fn = create_async_tool_fn(self, t.name, t.description)
            tool = FunctionTool(async_fn)
            tools.append(tool)
        return tools

"""LlamaIndex integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.environment.utils.tool_wrappers import create_tool_fns

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["LlamaIndexMixin"]


class LlamaIndexMixin:
    """Mixin providing LlamaIndex integration.

    Integration methods (requires llama-index-core):
        as_llamaindex_tools() - LlamaIndex FunctionTool objects

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    def as_llamaindex_tools(self) -> list[Any]:
        """Convert to LlamaIndex FunctionTool objects.

        Requires: pip install llama-index-core

        Returns:
            List of FunctionTool objects for LlamaIndex agents.

        Example:
            ```python
            from llama_index.llms.openai import OpenAI
            from llama_index.core.agent import ReActAgent

            llm = OpenAI(model="gpt-4o")
            async with env:
                tools = env.as_llamaindex_tools()
                agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
                response = await agent.achat("Find information about Python")
            ```
        """
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError as e:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index-core"
            ) from e

        tools = []
        for t in self.as_tools():
            sync_fn, async_fn = create_tool_fns(self, t)

            tool = FunctionTool.from_defaults(
                fn=sync_fn,
                async_fn=async_fn,
                name=t.name,
                description=t.description or "",
            )
            tools.append(tool)
        return tools

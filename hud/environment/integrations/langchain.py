"""LangChain integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from hud.environment.utils.schema import schema_to_pydantic

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["LangChainMixin"]


class LangChainMixin:
    """Mixin providing LangChain integration.

    Integration methods (requires langchain-core):
        as_langchain_tools() - LangChain StructuredTool objects

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    def as_langchain_tools(self) -> list[Any]:
        """Convert to LangChain StructuredTool objects.

        Requires: pip install langchain-core

        Returns:
            List of StructuredTool objects for LangChain agents.

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain.agents import create_tool_calling_agent, AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatOpenAI(model="gpt-4o")
            async with env:
                tools = env.as_langchain_tools()

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant."),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}"),
                    ]
                )

                agent = create_tool_calling_agent(llm, tools, prompt)
                executor = AgentExecutor(agent=agent, tools=tools)
                result = await executor.ainvoke({"input": "Navigate to google.com"})
            ```
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain-core"
            ) from e

        tools = []
        for t in self.as_tools():
            tool = _create_structured_tool(self, t, StructuredTool)
            tools.append(tool)
        return tools


def _create_structured_tool(env: LangChainMixin, tool: mcp_types.Tool, StructuredTool: type) -> Any:
    """Create a StructuredTool that calls back to the environment."""
    import asyncio

    schema = tool.inputSchema or {"type": "object", "properties": {}}

    def sync_invoke(**kwargs: Any) -> str:
        """Synchronous wrapper for the tool."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, env.call_tool(tool.name, **kwargs))
                result = future.result()
        else:
            result = loop.run_until_complete(env.call_tool(tool.name, **kwargs))

        if isinstance(result, str):
            return result
        return json.dumps(result) if result else ""

    async def async_invoke(**kwargs: Any) -> str:
        """Async wrapper for the tool."""
        result = await env.call_tool(tool.name, **kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result) if result else ""

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        func=sync_invoke,
        coroutine=async_invoke,
        args_schema=schema_to_pydantic(tool.name, schema),
    )

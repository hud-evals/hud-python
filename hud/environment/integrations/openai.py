"""OpenAI integrations - format conversion and Agents SDK."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from hud.environment.utils.schema import ensure_strict_schema, validate_openai_schema

if TYPE_CHECKING:
    import mcp.types as mcp_types
    from openai.types.chat import ChatCompletionToolUnionParam

__all__ = ["OpenAIMixin"]

logger = logging.getLogger(__name__)


class OpenAIMixin:
    """Mixin providing OpenAI format conversion and Agents SDK integration.

    Format methods (no deps):
        as_openai_chat_tools() - Chat Completions format
        as_openai_responses_tools() - Responses API format

    Integration methods (requires openai-agents):
        as_openai_agent_tools() - Agents SDK FunctionTool objects

    Note: The OpenAI Agents SDK also supports:
        - HostedMCPTool - MCP tools hosted by OpenAI
        - MCPServerStdio/Sse/StreamableHttp - Direct MCP server connections

    For MCP server integration, use as_mcp_server() from the mcp integration.

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    # =========================================================================
    # Format Conversion (no external deps)
    # =========================================================================

    def as_openai_chat_tools(
        self, *, strict: bool = False, validate: bool = True
    ) -> list[ChatCompletionToolUnionParam]:
        """Convert to OpenAI Chat Completions tool format.

        Args:
            strict: Enable strict mode for structured outputs
            validate: Validate schemas and skip incompatible tools with warnings

        Returns:
            List of tool definitions for OpenAI Chat Completions API.

        Example:
            ```python
            from openai import OpenAI

            client = OpenAI()
            async with env:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Navigate to google.com"}],
                    tools=env.as_openai_chat_tools(),
                )
                # Execute tool calls and get results in OpenAI format
                results = await env.call_tools(response.choices[0].message.tool_calls)
                # results are {"role": "tool", "tool_call_id": ..., "content": ...}
            ```
        """
        tools: list[ChatCompletionToolUnionParam] = []
        for t in self.as_tools():
            schema = dict(t.inputSchema) if t.inputSchema else {"type": "object", "properties": {}}

            # Validate schema for OpenAI compatibility
            if validate:
                errors = validate_openai_schema(schema, t.name)
                if errors:
                    for error in errors:
                        logger.warning("Skipping tool: %s", error)
                    continue

            if strict:
                schema = ensure_strict_schema(schema)

            tools.append(
                cast(
                    "ChatCompletionToolUnionParam",
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": schema,
                            **({"strict": True} if strict else {}),
                        },
                    },
                )
            )
        return tools

    def as_openai_responses_tools(self, *, validate: bool = True) -> list[dict[str, Any]]:
        """Convert to OpenAI Responses API tool format.

        Note: Like Chat Completions, you must execute tools yourself.
        OpenAI only auto-executes their built-in tools (code_interpreter, etc).

        Args:
            validate: Validate schemas and skip incompatible tools with warnings

        Returns:
            List of tool definitions for OpenAI Responses API.

        Example:
            ```python
            from openai import OpenAI

            client = OpenAI()
            async with env:
                response = client.responses.create(
                    model="gpt-4o",
                    input="Navigate to google.com",
                    tools=env.as_openai_responses_tools(),
                )
                # Check for function calls in the response
                for item in response.output:
                    if item.type == "function_call":
                        result = await env.call_tool(item.name, **item.arguments)
            ```
        """
        tools = []
        for t in self.as_tools():
            schema = dict(t.inputSchema) if t.inputSchema else {"type": "object", "properties": {}}

            # Validate schema for OpenAI compatibility
            if validate:
                errors = validate_openai_schema(schema, t.name)
                if errors:
                    for error in errors:
                        logger.warning("Skipping tool: %s", error)
                    continue

            tools.append(
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema,
                }
            )
        return tools

    # =========================================================================
    # Agents SDK Integration (requires openai-agents)
    # =========================================================================

    def as_openai_agent_tools(self, *, validate: bool = True) -> list[Any]:
        """Convert to OpenAI Agents SDK FunctionTool objects.

        This creates FunctionTool objects that automatically execute against
        this environment. The Agents SDK Runner handles the tool loop.

        Note: The Agents SDK also supports other tool types:
            - HostedMCPTool: MCP tools hosted by OpenAI
            - MCPServerStdio/Sse/StreamableHttp: Direct MCP server connections

        For direct MCP integration, consider using as_mcp_server().

        Requires: pip install openai-agents

        Args:
            validate: Validate schemas and skip incompatible tools with warnings

        Returns:
            List of FunctionTool objects for OpenAI Agents SDK.

        Example:
            ```python
            from agents import Agent, Runner

            async with env:
                agent = Agent(
                    name="browser-agent",
                    instructions="You browse the web.",
                    tools=env.as_openai_agent_tools(),
                )
                result = await Runner.run(agent, "Go to google.com")
                print(result.final_output)
            ```
        """
        try:
            from agents import FunctionTool
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        tools = []
        for t in self.as_tools():
            schema = dict(t.inputSchema) if t.inputSchema else {"type": "object", "properties": {}}

            # Validate schema for OpenAI compatibility
            if validate:
                errors = validate_openai_schema(schema, t.name)
                if errors:
                    for error in errors:
                        logger.warning("Skipping tool: %s", error)
                    continue

            tool = _create_function_tool(self, t, FunctionTool)
            tools.append(tool)
        return tools


def _create_function_tool(env: OpenAIMixin, tool: mcp_types.Tool, FunctionTool: type) -> Any:
    """Create a FunctionTool that calls back to the environment."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}

    async def async_wrapper(ctx: Any, args_json: str) -> str:
        """Async wrapper for the tool that matches FunctionTool signature."""
        kwargs = json.loads(args_json) if args_json else {}
        result = await env.call_tool(tool.name, **kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result) if result else ""

    return FunctionTool(
        name=tool.name,
        description=tool.description or "",
        params_json_schema=schema,
        on_invoke_tool=async_wrapper,
    )

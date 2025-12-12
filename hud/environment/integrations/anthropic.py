"""Anthropic/Claude integrations - format conversion and tool runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["AnthropicMixin"]


class AnthropicMixin:
    """Mixin providing Anthropic/Claude format conversion and tool runner.

    Format methods (no deps):
        as_claude_tools() - Claude API format
        as_claude_programmatic_tools() - Programmatic tool use format

    Integration methods (requires anthropic):
        as_anthropic_runner() - Tool runner for executing tool_use blocks

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    # =========================================================================
    # Format Conversion (no external deps)
    # =========================================================================

    def as_claude_tools(self, *, cache_control: bool = False) -> list[dict[str, Any]]:
        """Convert to Claude/Anthropic tool format.

        Args:
            cache_control: Add cache_control for prompt caching

        Returns:
            List of tool definitions for Claude API.

        Example:
            ```python
            from anthropic import Anthropic

            client = Anthropic()
            async with env:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Navigate to google.com"}],
                    tools=env.as_claude_tools(),
                )
                # Execute tool calls
                for block in response.content:
                    if block.type == "tool_use":
                        result = await env.call_tool(block)
            ```
        """
        tools = []
        for t in self.as_tools():
            tool: dict[str, Any] = {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
            if cache_control:
                tool["cache_control"] = {"type": "ephemeral"}
            tools.append(tool)
        return tools

    def as_claude_programmatic_tools(self, *, cache_control: bool = False) -> list[dict[str, Any]]:
        """Convert to Claude programmatic tool use format.

        Programmatic tool use allows Claude to execute tools via code execution.

        Example:
            ```python
            from anthropic import Anthropic

            client = Anthropic()
            async with env:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Analyze the data"}],
                    tools=env.as_claude_programmatic_tools(),
                    betas=["code-execution-2025-01-24"],
                )
            ```
        """
        tools = []
        for t in self.as_tools():
            tool: dict[str, Any] = {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                "allowed_callers": ["code_execution_20250825"],
            }
            if cache_control:
                tool["cache_control"] = {"type": "ephemeral"}
            tools.append(tool)
        return tools

    # =========================================================================
    # Tool Runner Integration (requires anthropic)
    # =========================================================================

    def as_anthropic_runner(self) -> EnvToolRunner:
        """Create an Anthropic tool runner for this environment.

        Requires: pip install anthropic

        Returns:
            EnvToolRunner that can process tool_use blocks from Claude.

        Example:
            ```python
            from anthropic import Anthropic

            client = Anthropic()
            async with env:
                runner = env.as_anthropic_runner()

                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Navigate to google.com"}],
                    tools=env.as_claude_tools(),
                )

                # Execute all tool_use blocks
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await runner.run(block)
                        results.append(result)
            ```
        """
        return EnvToolRunner(self)


class EnvToolRunner:
    """Tool runner that executes tools against an Environment."""

    def __init__(self, env: AnthropicMixin) -> None:
        self.env = env
        self._tool_names: set[str] | None = None

    @property
    def tool_names(self) -> set[str]:
        """Get available tool names."""
        if self._tool_names is None:
            self._tool_names = {t.name for t in self.env.as_tools()}
        return self._tool_names

    async def run(self, tool_use_block: Any) -> Any:
        """Execute a tool_use block from Claude.

        Args:
            tool_use_block: A ToolUseBlock from Claude's response.

        Returns:
            Tool result dict (or BetaToolResultBlockParam if anthropic installed).
        """
        name = tool_use_block.name
        tool_use_id = tool_use_block.id
        arguments = tool_use_block.input or {}

        try:
            result = await self.env.call_tool(name, **arguments)
            content = result if isinstance(result, str) else json.dumps(result) if result else ""
            result_dict: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        except Exception as e:
            result_dict = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        # Return typed object if anthropic is available
        try:
            from anthropic.types.beta import BetaToolResultBlockParam

            return BetaToolResultBlockParam(**result_dict)
        except ImportError:
            return result_dict

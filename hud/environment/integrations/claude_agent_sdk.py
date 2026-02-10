"""Claude Agent SDK integration - expose Environment tools as MCP servers for the SDK."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from hud.environment.utils.tool_wrappers import stringify_result

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["ClaudeAgentSDKMixin"]

logger = logging.getLogger(__name__)


class ClaudeAgentSDKMixin:
    """Mixin providing Claude Agent SDK integration.

    Exposes Environment tools as an in-process MCP server that the
    Claude Agent SDK can connect to, allowing the SDK's autonomous
    agent loop to use environment-provided tools.

    Integration methods (requires claude-agent-sdk):
        as_claude_agent_mcp_server() - SDK-compatible MCP server config
        as_claude_agent_options() - Full ClaudeAgentOptions with tools wired up

    Requires: as_tools() -> list[mcp_types.Tool], call_tool(name, args)
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    def as_claude_agent_mcp_server(
        self,
        *,
        server_name: str = "hud_env",
        server_version: str = "1.0.0",
    ) -> Any:
        """Create an in-process MCP server exposing this environment's tools.

        The returned server object can be passed directly to
        ``ClaudeAgentOptions(mcp_servers={"env": server})``.

        Requires: pip install claude-agent-sdk

        Args:
            server_name: Name for the MCP server (used in tool prefixing).
            server_version: Version string for the server.

        Returns:
            An SDK MCP server object suitable for ``ClaudeAgentOptions.mcp_servers``.

        Example:
            ```python
            from claude_agent_sdk import query, ClaudeAgentOptions

            async with env:
                server = env.as_claude_agent_mcp_server()
                options = ClaudeAgentOptions(
                    mcp_servers={"env": server},
                    allowed_tools=[f"mcp__env__{t.name}" for t in env.as_tools()],
                )
                async for msg in query(prompt="Do the task", options=options):
                    print(msg)
            ```
        """
        try:
            from claude_agent_sdk import create_sdk_mcp_server, tool
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
            ) from e

        sdk_tools = []
        for t in self.as_tools():
            sdk_tool = _create_sdk_tool(self, t, tool)
            sdk_tools.append(sdk_tool)

        return create_sdk_mcp_server(
            name=server_name,
            version=server_version,
            tools=sdk_tools,
        )

    def as_claude_agent_options(
        self,
        *,
        server_name: str = "hud_env",
        prompt: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        permission_mode: str = "bypassPermissions",
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        include_builtin_tools: bool = False,
        **extra_options: Any,
    ) -> Any:
        """Create ClaudeAgentOptions pre-configured with this environment's tools.

        Convenience method that builds the full options object. By default,
        only environment tools are allowed (no built-in Read/Write/Bash etc.)
        unless ``include_builtin_tools=True``.

        Requires: pip install claude-agent-sdk

        Args:
            server_name: MCP server name (tools will be ``mcp__<name>__<tool>``).
            prompt: System prompt override.
            model: Model to use (e.g. "sonnet", "opus").
            system_prompt: Custom system prompt.
            max_turns: Maximum conversation turns.
            permission_mode: Permission mode (default "bypassPermissions" for evals).
            allowed_tools: Explicit tool allowlist. If None and not including
                builtins, auto-populated from environment tools.
            disallowed_tools: Tool denylist.
            include_builtin_tools: If True, also allow SDK built-in tools
                (Read, Write, Edit, Bash, etc.).
            **extra_options: Additional kwargs passed to ClaudeAgentOptions.

        Returns:
            Configured ClaudeAgentOptions instance.

        Example:
            ```python
            from claude_agent_sdk import query

            async with env:
                options = env.as_claude_agent_options(model="sonnet")
                async for msg in query(prompt="Do the task", options=options):
                    print(msg)
            ```
        """
        try:
            from claude_agent_sdk import ClaudeAgentOptions
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
            ) from e

        mcp_server = self.as_claude_agent_mcp_server(server_name=server_name)

        # Build allowed_tools list
        if allowed_tools is None and not include_builtin_tools:
            allowed_tools = [f"mcp__{server_name}__{t.name}" for t in self.as_tools()]

        options_kwargs: dict[str, Any] = {
            "mcp_servers": {server_name: mcp_server},
            "permission_mode": permission_mode,
            **extra_options,
        }

        if allowed_tools is not None:
            options_kwargs["allowed_tools"] = allowed_tools
        if disallowed_tools is not None:
            options_kwargs["disallowed_tools"] = disallowed_tools
        if model is not None:
            options_kwargs["model"] = model
        if system_prompt is not None:
            options_kwargs["system_prompt"] = system_prompt
        if max_turns is not None:
            options_kwargs["max_turns"] = max_turns

        return ClaudeAgentOptions(**options_kwargs)


def _create_sdk_tool(env: ClaudeAgentSDKMixin, tool: mcp_types.Tool, tool_decorator: Any) -> Any:
    """Create a claude-agent-sdk @tool-decorated function for an environment tool.

    Args:
        env: The environment mixin instance.
        tool: MCP tool definition.
        tool_decorator: The ``@tool`` decorator from claude_agent_sdk.

    Returns:
        A decorated tool function compatible with ``create_sdk_mcp_server``.
    """
    # Build parameter schema from MCP inputSchema
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    properties = schema.get("properties", {})

    # Build a param_types dict mapping param names to types
    # The SDK @tool decorator expects: @tool(name, description, {param: type})
    param_types: dict[str, type] = {}
    for param_name, param_schema in properties.items():
        param_types[param_name] = _json_type_to_python(param_schema)

    @tool_decorator(tool.name, tool.description or f"Tool: {tool.name}", param_types)
    async def sdk_tool_fn(args: dict[str, Any]) -> dict[str, Any]:
        """Wrapper that delegates to the environment's call_tool."""
        result = await env.call_tool(tool.name, **args)
        text = stringify_result(result)
        return {"content": [{"type": "text", "text": text}]}

    return sdk_tool_fn


def _json_type_to_python(schema: dict[str, Any]) -> type:
    """Map JSON Schema type to Python type for the SDK @tool decorator."""
    json_type = schema.get("type", "string")
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)

"""Shared tool wrapper utilities for agent framework integrations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import mcp.types as mcp_types

__all__ = [
    "create_async_tool_fn",
    "create_sync_tool_fn",
    "create_tool_fns",
    "stringify_result",
]


def stringify_result(result: Any) -> str:
    """Convert a tool result to string format.

    Args:
        result: The tool result (str, dict, or other).

    Returns:
        String representation of the result.
    """
    if isinstance(result, str):
        return result
    return json.dumps(result) if result else ""


def create_async_tool_fn(
    env: Any,
    tool_name: str,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create an async function that calls a tool on the environment.

    Args:
        env: Environment with call_tool method.
        tool_name: Name of the tool to call.
        description: Optional description for the function docstring.

    Returns:
        Async function that calls the tool and returns string result.
    """

    async def async_fn(**kwargs: Any) -> str:
        result = await env.call_tool(tool_name, **kwargs)
        return stringify_result(result)

    async_fn.__name__ = tool_name
    async_fn.__doc__ = description or f"Tool: {tool_name}"
    return async_fn


def create_sync_tool_fn(
    env: Any,
    tool_name: str,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create a sync function that calls a tool on the environment.

    This handles the complexity of running async code from sync context,
    including when already in an async event loop.

    Args:
        env: Environment with call_tool method.
        tool_name: Name of the tool to call.
        description: Optional description for the function docstring.

    Returns:
        Sync function that calls the tool and returns string result.
    """
    import asyncio

    def sync_fn(**kwargs: Any) -> str:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, env.call_tool(tool_name, **kwargs))
                result = future.result()
        else:
            result = loop.run_until_complete(env.call_tool(tool_name, **kwargs))

        return stringify_result(result)

    sync_fn.__name__ = tool_name
    sync_fn.__doc__ = description or f"Tool: {tool_name}"
    return sync_fn


def create_tool_fns(
    env: Any,
    tool: mcp_types.Tool,
) -> tuple[Callable[..., str], Callable[..., Any]]:
    """Create both sync and async functions for a tool.

    Args:
        env: Environment with call_tool method.
        tool: MCP tool definition.

    Returns:
        Tuple of (sync_fn, async_fn).
    """
    sync_fn = create_sync_tool_fn(env, tool.name, tool.description)
    async_fn = create_async_tool_fn(env, tool.name, tool.description)
    return sync_fn, async_fn

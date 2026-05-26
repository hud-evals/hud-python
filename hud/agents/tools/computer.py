"""Shared helpers for agent-side computer tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mcp.types import ImageContent, TextContent

from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from mcp import types as mcp_types

CallTool = Callable[[MCPToolCall], Awaitable[MCPToolResult]]


@dataclass(frozen=True)
class ComputerToolInfo:
    """Computer MCP tool metadata needed by provider adapters."""

    display_width: int
    display_height: int
    coordinate_space: int | None


def computer_tool_info(
    tool: mcp_types.Tool,
    *,
    default_width: int,
    default_height: int,
) -> ComputerToolInfo:
    """Resolve the computer contract advertised by the MCP tool."""
    meta = cast("Mapping[str, object]", tool.meta or {})
    resolution = meta.get("resolution")
    display_width = default_width
    display_height = default_height

    if isinstance(resolution, Mapping):
        resolution = cast("Mapping[str, object]", resolution)
        width = resolution.get("width")
        height = resolution.get("height")
        if type(width) is int:
            display_width = width
        if type(height) is int:
            display_height = height

    coordinate_space_raw = meta.get("coordinate_space")
    coordinate_space = coordinate_space_raw if type(coordinate_space_raw) is int else None

    return ComputerToolInfo(
        display_width=display_width,
        display_height=display_height,
        coordinate_space=coordinate_space,
    )


def computer_error_result(message: str) -> MCPToolResult:
    return MCPToolResult(content=[TextContent(type="text", text=message)], isError=True)


def result_has_image(result: MCPToolResult) -> bool:
    return any(isinstance(block, ImageContent) for block in result.content)


def first_image_data(result: MCPToolResult) -> str | None:
    for block in result.content:
        if isinstance(block, ImageContent):
            return block.data
    return None


def last_image_data(result: MCPToolResult) -> str | None:
    for block in reversed(result.content):
        if isinstance(block, ImageContent):
            return block.data
    return None


async def execute_computer_calls(
    call_tool: CallTool,
    *,
    env_tool_name: str,
    calls: list[dict[str, Any]],
    ensure_screenshot: bool,
) -> MCPToolResult:
    result = MCPToolResult(content=[], isError=False)
    for arguments in calls:
        result = await call_tool(MCPToolCall(name=env_tool_name, arguments=arguments))
        if result.isError:
            return result

    if ensure_screenshot and not result_has_image(result):
        screenshot = await call_tool(
            MCPToolCall(name=env_tool_name, arguments={"action": "screenshot"})
        )
        if not screenshot.isError and screenshot.content:
            return MCPToolResult(
                content=[*result.content, *screenshot.content],
                isError=result.isError,
            )

    return result

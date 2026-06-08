"""MCP server that exposes computer-use over VNC.

Single tool ``computer`` backed by ``ClaudeComputerTool`` / ``RFBTool``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import fastmcp
import mcp.types as mcp_types

from hud.agents.claude.tools.computer import ClaudeComputerTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.capabilities.rfb import RFBClient

logger = logging.getLogger(__name__)

#: ``computer`` params whose string value may be a JSON array (e.g. ``[x, y]``).
_JSON_FIELDS = frozenset({"coordinate", "start_coordinate", "region"})


def _maybe_json(value: str) -> Any:
    """Parse a JSON-array-ish argument, falling back to the raw string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def create_computer_mcp(rfb: RFBClient) -> fastmcp.FastMCP:
    """Build a FastMCP server with one ``computer`` tool backed by ``rfb``."""

    mcp = fastmcp.FastMCP("computer-use")
    spec = ClaudeComputerTool.default_spec("claude-sonnet-4-6")
    if spec is None:
        raise RuntimeError("Claude computer MCP requires a Claude computer spec")
    computer_tool = ClaudeComputerTool(spec=spec, client=rfb)

    async def computer(
        action: str,
        coordinate: str | None = None,
        text: str | None = None,
        scroll_direction: str | None = None,
        scroll_amount: int | None = None,
        start_coordinate: str | None = None,
        duration: float | None = None,
        repeat: int | None = None,
        region: str | None = None,
    ) -> list[Any]:
        """Control a remote screen — screenshot, click, type, key, scroll, move, drag, wait, zoom.

        Actions: screenshot, left_click, right_click, middle_click, double_click,
        triple_click, mouse_move, move, type, key, scroll, left_click_drag, drag,
        wait, hold_key, cursor_position, zoom, left_mouse_down, left_mouse_up.

        Returns the resulting screenshot image so you can see the screen state.
        """
        arguments: dict[str, Any] = {"action": action}
        optional: dict[str, str | int | float | None] = {
            "coordinate": coordinate,
            "text": text,
            "scroll_direction": scroll_direction,
            "scroll_amount": scroll_amount,
            "start_coordinate": start_coordinate,
            "duration": duration,
            "repeat": repeat,
            "region": region,
        }
        for key, value in optional.items():
            if value is None:
                continue
            if key in _JSON_FIELDS and isinstance(value, str):
                arguments[key] = _maybe_json(value)
            else:
                arguments[key] = value

        result = await computer_tool.execute(arguments)

        # Return content blocks directly so the CLI/model sees real images.
        blocks: list[Any] = []
        for block in result.content:
            if isinstance(block, mcp_types.ImageContent):
                blocks.append(
                    mcp_types.ImageContent(
                        type="image", data=block.data, mimeType=block.mimeType,
                    ),
                )
            elif isinstance(block, mcp_types.TextContent):
                blocks.append(mcp_types.TextContent(type="text", text=block.text))
        if not blocks:
            blocks.append(mcp_types.TextContent(type="text", text="ok"))
        if result.isError:
            blocks.insert(0, mcp_types.TextContent(type="text", text="ERROR"))
        return blocks

    mcp.tool()(computer)
    return mcp


@asynccontextmanager
async def computer_mcp_server(
    rfb: RFBClient,
    host: str = "127.0.0.1",
) -> AsyncIterator[int]:
    """Run the computer-use MCP server for the lifetime of the context.

    Binds an ephemeral port, waits until uvicorn reports it has started (surfacing
    any startup error instead of a fixed sleep), yields the bound port, then shuts
    the server down on exit. Scoping the server to one rollout avoids the leaked
    background task and TOCTOU port grab of the previous implementation.
    """
    import uvicorn

    mcp = create_computer_mcp(rfb)
    app = mcp.http_app(path="/mcp")
    config = uvicorn.Config(app, host=host, port=0, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                task.result()  # re-raise the startup failure
                raise RuntimeError("computer-use MCP server exited before startup")
            await asyncio.sleep(0.02)
        port = int(server.servers[0].sockets[0].getsockname()[1])
        logger.info("computer-use MCP server on %s:%d", host, port)
        yield port
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await task


__all__ = ["computer_mcp_server", "create_computer_mcp"]

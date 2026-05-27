"""MCP server that exposes computer-use over VNC.

Single tool ``computer`` backed by ``ClaudeComputerTool`` / ``RFBTool``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import fastmcp

from hud.capabilities.rfb import RFBClient

logger = logging.getLogger(__name__)


def create_computer_mcp(rfb: RFBClient) -> fastmcp.FastMCP:
    """Build a FastMCP server with one ``computer`` tool backed by ``rfb``."""

    mcp = fastmcp.FastMCP("computer-use")

    @mcp.tool()
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
    ) -> str:
        """Control a remote screen — screenshot, click, type, key, scroll, move, drag, wait, zoom.

        Actions: screenshot, left_click, right_click, middle_click, double_click,
        triple_click, mouse_move, move, type, key, scroll, left_click_drag, drag,
        wait, hold_key, cursor_position, zoom, left_mouse_down, left_mouse_up.
        """
        from hud.agents.claude.tools.computer import ClaudeComputerTool
        from hud.agents.tools.base import AgentToolSpec

        arguments: dict[str, Any] = {"action": action}
        if coordinate is not None:
            try:
                arguments["coordinate"] = json.loads(coordinate)
            except json.JSONDecodeError:
                arguments["coordinate"] = coordinate
        if text is not None:
            arguments["text"] = text
        if scroll_direction is not None:
            arguments["scroll_direction"] = scroll_direction
        if scroll_amount is not None:
            arguments["scroll_amount"] = scroll_amount
        if start_coordinate is not None:
            try:
                arguments["start_coordinate"] = json.loads(start_coordinate)
            except json.JSONDecodeError:
                arguments["start_coordinate"] = start_coordinate
        if duration is not None:
            arguments["duration"] = duration
        if repeat is not None:
            arguments["repeat"] = repeat
        if region is not None:
            try:
                arguments["region"] = json.loads(region)
            except json.JSONDecodeError:
                arguments["region"] = region

        spec = AgentToolSpec(api_type="computer", api_name="computer")
        tool = ClaudeComputerTool(spec=spec, client=rfb)
        result = await tool.execute(arguments)

        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[screenshot:{len(block.data)}b]")
        text_out = "".join(parts) if parts else "ok"
        return f"ERROR: {text_out}" if result.isError else text_out

    return mcp


async def serve_computer_mcp(
    rfb: RFBClient,
    host: str = "127.0.0.1",
    port: int = 0,
) -> int:
    """Start the computer-use MCP server in the background, return the port."""
    if port == 0:
        srv = await asyncio.get_event_loop().create_server(lambda: asyncio.Protocol(), host, 0)
        port = srv.sockets[0].getsockname()[1]
        srv.close()

    mcp = create_computer_mcp(rfb)
    asyncio.create_task(_run(mcp, host, port))
    await asyncio.sleep(0.5)
    logger.info("computer-use MCP server on %s:%d", host, port)
    return port


async def _run(mcp: fastmcp.FastMCP, host: str, port: int) -> None:
    try:
        await mcp.run_http_async(host=host, port=port)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("computer-use MCP server crashed")


__all__ = ["create_computer_mcp", "serve_computer_mcp"]

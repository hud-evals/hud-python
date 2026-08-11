"""MCP server that exposes computer-use over VNC.

Single tool ``computer`` backed by ``ClaudeComputerTool`` / ``RFBTool``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import fastmcp
from pydantic import TypeAdapter

from hud.capabilities import Capability
from hud.capabilities.rfb import RFBClient, ScreenshotEncoding, WebPScreenshotEncoding

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFAULT_SCREENSHOT_ENCODING = WebPScreenshotEncoding()
RFB_CAPABILITY_ENV = "HUD_RFB_CAPABILITY"
SCREENSHOT_ENCODING_ENV = "HUD_SCREENSHOT_ENCODING"


def create_computer_mcp(
    rfb: RFBClient,
    screenshot_encoding: ScreenshotEncoding = _DEFAULT_SCREENSHOT_ENCODING,
) -> fastmcp.FastMCP:
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
    ) -> list[Any]:
        """Control a remote screen — screenshot, click, type, key, scroll, move, drag, wait, zoom.

        Actions: screenshot, left_click, right_click, middle_click, double_click,
        triple_click, mouse_move, move, type, key, scroll, left_click_drag, drag,
        wait, hold_key, cursor_position, zoom, left_mouse_down, left_mouse_up.

        Returns the resulting screenshot image so you can see the screen state.
        """
        import mcp.types as mcp_types

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
        tool = ClaudeComputerTool(
            spec=spec,
            client=rfb,
            screenshot_encoding=screenshot_encoding,
        )
        result = await tool.execute(arguments)

        # Return content blocks directly so the CLI/model sees real images.
        blocks: list[Any] = []
        for block in result.content:
            if isinstance(block, mcp_types.ImageContent):
                blocks.append(
                    mcp_types.ImageContent(
                        type="image",
                        data=block.data,
                        mimeType=block.mimeType,
                    ),
                )
            elif isinstance(block, mcp_types.TextContent):
                blocks.append(mcp_types.TextContent(type="text", text=block.text))
        if not blocks:
            blocks.append(mcp_types.TextContent(type="text", text="ok"))
        if result.isError:
            blocks.insert(0, mcp_types.TextContent(type="text", text="ERROR"))
        return blocks

    return mcp


def _required_env(environ: Mapping[str, str], name: str) -> str:
    try:
        return environ[name]
    except KeyError as exc:
        raise RuntimeError(f"missing required environment variable {name}") from exc


async def run_computer_mcp(environ: Mapping[str, str] = os.environ) -> None:
    """Run computer-use over stdio for the lifetime of the invoking Claude CLI."""
    raw_manifest = json.loads(_required_env(environ, RFB_CAPABILITY_ENV))
    if not isinstance(raw_manifest, dict):
        raise ValueError(f"{RFB_CAPABILITY_ENV} must contain a JSON object")
    capability = Capability.from_manifest(raw_manifest)
    if capability.protocol.split("/", 1)[0] != "rfb":
        raise ValueError(f"{RFB_CAPABILITY_ENV} must describe an RFB capability")
    screenshot_encoding = TypeAdapter(ScreenshotEncoding).validate_json(
        _required_env(environ, SCREENSHOT_ENCODING_ENV)
    )

    rfb = await RFBClient.connect(capability)
    try:
        await create_computer_mcp(rfb, screenshot_encoding).run_async(
            transport="stdio",
            show_banner=False,
        )
    finally:
        await rfb.close()


if __name__ == "__main__":
    asyncio.run(run_computer_mcp())


__all__ = [
    "RFB_CAPABILITY_ENV",
    "SCREENSHOT_ENCODING_ENV",
    "create_computer_mcp",
    "run_computer_mcp",
]

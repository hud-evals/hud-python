"""Tests for include_router prefix handling.

Regression: _sync_import_router must update .name of tools, resources, and
prompts to match the prefixed dict key.  Otherwise MCP wire serialisation
(which uses tool.name / tool.key) disagrees with the internal lookup key,
and clients get "unknown tool" errors.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import suppress

import pytest
from fastmcp import Client as MCPClient
from fastmcp import FastMCP

from hud.server import MCPServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_router() -> FastMCP:
    router = FastMCP("helper")

    @router.tool()
    def greet(name: str) -> str:
        return f"hello {name}"

    @router.resource("res://info")
    def info() -> str:
        return "some info"

    @router.prompt()
    def ask(topic: str) -> str:
        return f"Tell me about {topic}"

    return router


def test_prefixed_names_match_dict_keys() -> None:
    """After include_router(prefix=...), .name must equal the dict key
    for tools, resources, and prompts."""
    mcp = MCPServer(name="PrefixSync")
    mcp.include_router(_make_router(), prefix="ns")

    tool = mcp._tool_manager._tools["ns_greet"]
    assert tool.name == "ns_greet"
    assert tool.key == "ns_greet"

    resource = mcp._resource_manager._resources["ns_res://info"]
    assert resource.name == "ns_res://info"

    prompt = mcp._prompt_manager._prompts["ns_ask"]
    assert prompt.name == "ns_ask"


@pytest.mark.asyncio
async def test_mcp_client_can_list_and_call_prefixed_tool() -> None:
    """End-to-end: a real MCP client must see the prefixed name and call it."""
    port = _free_port()

    mcp = MCPServer(name="E2EPrefix")
    mcp.include_router(_make_router(), prefix="ns")

    task = asyncio.create_task(
        mcp.run_async(
            transport="http",
            host="127.0.0.1",
            port=port,
            path="/mcp",
            log_level="ERROR",
            show_banner=False,
        )
    )
    await asyncio.sleep(0.1)

    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        async with MCPClient({"mcpServers": cfg}) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            assert "ns_greet" in tool_names, f"expected ns_greet in {tool_names}"

            result = await client.call_tool("ns_greet", {"name": "world"})
            from mcp.types import TextContent

            first = result.content[0]
            assert isinstance(first, TextContent)
            assert "hello world" in first.text
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

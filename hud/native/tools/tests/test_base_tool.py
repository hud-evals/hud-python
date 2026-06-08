"""``BaseTool`` — name derivation, cached ``.mcp``, and register."""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from hud.native.tools.base import BaseTool


class EchoTool(BaseTool):
    async def __call__(self, value: str = "x") -> list[TextContent]:
        return [TextContent(type="text", text=value)]


def _result_text(result: Any) -> str:
    blocks = getattr(result, "content", result)
    return "\n".join(getattr(b, "text", "") for b in blocks)


def test_name_and_title_autoderive_from_class() -> None:
    tool = EchoTool()
    assert tool.name == "echo"
    assert tool.title == "Echo"


def test_mcp_property_is_cached() -> None:
    tool = EchoTool()
    assert tool.mcp is tool.mcp


async def test_mcp_runs_tool_call() -> None:
    tool = EchoTool()

    result = await tool.mcp.run({"value": "hi"})

    assert "hi" in _result_text(result)


async def test_register_adds_tool_to_server() -> None:
    from hud.server import MCPServer

    server = MCPServer("s")
    EchoTool(name="ping").register(server)

    assert "ping" in {tool.name for tool in await server.list_tools()}

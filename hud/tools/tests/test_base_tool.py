"""``BaseTool`` — name derivation, cached ``.mcp``, before/after callbacks, register."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import TextContent

from hud.tools.base import BaseTool


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


async def test_before_callback_rewrites_kwargs_and_after_observes_result() -> None:
    tool = EchoTool()
    seen: list[Any] = []

    @tool.before
    async def upcase(value: str = "", **_: Any) -> dict[str, Any]:
        return {"value": value.upper()}

    @tool.after
    async def record(result: Any = None, **_: Any) -> None:
        seen.append(result)

    result = await tool.mcp.run({"value": "hi"})

    assert "HI" in _result_text(result)  # before-callback rewrote the args
    assert seen  # after-callback ran


async def test_before_callback_can_block_execution() -> None:
    tool = EchoTool()

    @tool.before
    async def guard(**_: Any) -> dict[str, Any]:
        raise ValueError("blocked")

    with pytest.raises(Exception, match="blocked"):
        await tool.mcp.run({"value": "x"})


async def test_register_adds_tool_to_server() -> None:
    from hud.server import MCPServer

    server = MCPServer("s")
    EchoTool(name="ping").register(server)

    assert "ping" in {tool.name for tool in await server.list_tools()}

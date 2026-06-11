from __future__ import annotations

import sys
import types
from typing import Any, cast

from hud.server import MCPServer


def test_add_tool_accepts_base_tool(monkeypatch):
    """If obj is BaseTool, its `.mcp` gets passed through to FastMCP.add_tool."""
    # Stub hud.tools.base.BaseTool and capture FastMCP.add_tool calls
    mod = types.ModuleType("hud.tools.base")

    class FakeBaseTool:
        """Stub type checked by isinstance() inside add_tool."""

    # Tell the type checker we're mutating a dynamic module
    mod_any = cast("Any", mod)
    mod_any.BaseTool = FakeBaseTool
    monkeypatch.setitem(sys.modules, "hud.tools.base", mod)

    calls: dict[str, object | None] = {"obj": None, "kwargs": None}

    def fake_super_add(self, obj: object, **kwargs: object) -> None:  # keep runtime the same
        calls["obj"] = obj
        calls["kwargs"] = kwargs

    monkeypatch.setattr("hud.server.server.FastMCP.add_tool", fake_super_add, raising=True)

    mcp = MCPServer(name="AddTool")
    sentinel = object()

    class MyTool(FakeBaseTool):
        def __init__(self) -> None:
            self.mcp = sentinel

    mcp.add_tool(MyTool(), extra="yes")
    assert calls["obj"] is sentinel
    assert isinstance(calls["kwargs"], dict)
    assert calls["kwargs"]["extra"] == "yes"


def test_add_tool_plain_falls_back_to_super(monkeypatch):
    """Non-BaseTool objects are passed unchanged to FastMCP.add_tool."""
    calls = []

    def fake_super_add(self, obj, **kwargs):
        calls.append((obj, kwargs))

    monkeypatch.setattr("hud.server.server.FastMCP.add_tool", fake_super_add, raising=True)

    mcp = MCPServer(name="AddToolPlain")

    async def fn():  # pragma: no cover - never awaited by FastMCP here
        return "ok"

    mcp.add_tool(fn, desc="x")
    assert calls and calls[0][0] is fn
    assert calls[0][1]["desc"] == "x"

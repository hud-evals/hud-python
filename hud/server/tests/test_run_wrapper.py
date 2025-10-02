from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from hud.server import MCPServer

if TYPE_CHECKING:
    import pytest


def test_run_uses_sigterm_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCPServer.run should delegate through _run_with_sigterm (don't actually start a server)."""
    called = {"hit": False, "args": None, "kwargs": None}

    def fake_wrapper(coro_fn, *args, **kwargs):
        called["hit"] = True
        called["args"] = args
        called["kwargs"] = kwargs
        # Do not execute the bootstrap coroutine; this is unit wiring only.

    monkeypatch.setattr("hud.server.server._run_with_sigterm", fake_wrapper)

    mcp = MCPServer(name="RunWrapper")
    # Should immediately return after calling our fake wrapper
    mcp.run(transport="http", host="127.0.0.1", port=9999, path="/mcp", show_banner=False)

    assert called["hit"] is True


def test_run_defaults_to_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    """transport=None in .run should resolve to 'stdio' and forward to run_async."""
    seen = {}

    async def fake_run_async(self, *, transport, show_banner, **kwargs):
        seen["transport"] = transport
        seen["show_banner"] = show_banner
        seen["kwargs"] = kwargs

    # Replace bound method on the instance class
    monkeypatch.setattr(MCPServer, "run_async", fake_run_async, raising=False)

    # Execute the bootstrap coroutine immediately (no real server)
    def fake_wrapper(coro_fn, *args, **kwargs):
        asyncio.run(coro_fn())

    monkeypatch.setattr("hud.server.server._run_with_sigterm", fake_wrapper)

    mcp = MCPServer(name="RunDefaultTransport")
    mcp.run(transport=None, show_banner=False)

    assert seen["transport"] == "stdio"
    assert seen["show_banner"] is False

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

import anyio
import pytest

from hud.server import MCPServer
from hud.server import server as server_mod


def test__run_with_sigterm_executes_coro_when_handler_disabled(monkeypatch: pytest.MonkeyPatch):
    """With FASTMCP_DISABLE_SIGTERM_HANDLER=1, _run_with_sigterm should just run the task."""
    monkeypatch.setenv("FASTMCP_DISABLE_SIGTERM_HANDLER", "1")

    hit = {"v": False}

    async def work(arg, *, kw=None):
        assert arg == 123 and kw == "ok"
        hit["v"] = True

    # Wrapper to exercise kwargs since TaskGroup.start_soon only accepts positional args
    async def wrapper(arg):
        await work(arg, kw="ok")

    # Should return cleanly and mark hit
    server_mod._run_with_sigterm(wrapper, 123)
    assert hit["v"] is True


@asynccontextmanager
async def _fake_stdio_server():
    """Stand-in for stdio_server that avoids reading real stdin."""
    send_in, recv_in = anyio.create_memory_object_stream(100)
    send_out, recv_out = anyio.create_memory_object_stream(100)
    try:
        yield recv_in, send_out
    finally:
        for s in (send_in, recv_in, send_out, recv_out):
            close = getattr(s, "close", None) or getattr(s, "aclose", None)
            try:
                if close is not None:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass


@pytest.fixture
def patch_stdio(monkeypatch: pytest.MonkeyPatch):
    """Patch stdio server to avoid stdin issues during tests."""
    monkeypatch.setenv("FASTMCP_DISABLE_BANNER", "1")
    monkeypatch.setattr("mcp.server.stdio.stdio_server", _fake_stdio_server)
    monkeypatch.setattr("fastmcp.server.server.stdio_server", _fake_stdio_server)


@pytest.mark.asyncio
async def test_shutdown_handler_exception_does_not_crash_and_resets_flag(patch_stdio):
    """If @shutdown raises, run_async must swallow it and still reset the SIGTERM flag."""
    mcp = MCPServer(name="ShutdownRaises")

    @mcp.shutdown
    async def _boom() -> None:
        raise RuntimeError("kaboom")

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        server_mod._sigterm_received = True  # trigger shutdown path
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    # No exception propagated; flag must be reset
    assert not getattr(server_mod, "_sigterm_received")

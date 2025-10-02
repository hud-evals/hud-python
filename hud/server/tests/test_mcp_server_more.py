from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager, suppress

import anyio
import pytest

from hud.clients import MCPClient
from hud.server import MCPServer
from hud.server import server as server_mod  # to toggle _sigterm_received


def _free_port() -> int:
    """Get a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def _fake_stdio_server():
    """
    Stand-in for mcp.server.stdio.stdio_server that avoids reading real stdin.

    It yields a pair of in-memory streams (receive, send) so the low-level server
    can start and idle without touching sys.stdin/sys.stdout.
    """
    # Server reads from recv_in and writes to send_out
    send_in, recv_in = anyio.create_memory_object_stream(100)  # stdin → server
    send_out, recv_out = anyio.create_memory_object_stream(100)  # server → stdout
    try:
        yield recv_in, send_out
    finally:
        # Best effort close; methods exist across anyio versions
        for s in (send_in, recv_in, send_out, recv_out):
            close = getattr(s, "close", None) or getattr(s, "aclose", None)
            try:
                if close is not None:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _patch_stdio(monkeypatch: pytest.MonkeyPatch):
    """Patch stdio server for all tests to avoid stdin reading issues."""
    monkeypatch.setenv("FASTMCP_DISABLE_BANNER", "1")
    # Patch both the source and the bound symbol FastMCP uses
    monkeypatch.setattr("mcp.server.stdio.stdio_server", _fake_stdio_server)
    monkeypatch.setattr("fastmcp.server.server.stdio_server", _fake_stdio_server)


@pytest.mark.asyncio
async def test_stdio_shutdown_handler_on_sigterm_flag() -> None:
    """@mcp.shutdown runs on stdio transport when the SIGTERM flag is set."""
    mcp = MCPServer(name="StdIOShutdown")
    calls = {"n": 0}

    @mcp.shutdown
    async def _on_shutdown() -> None:
        calls["n"] += 1

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        # Simulate SIGTERM path
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert calls["n"] == 1
    assert not getattr(server_mod, "_sigterm_received")


@pytest.mark.asyncio
async def test_stdio_shutdown_handler_not_called_without_sigterm() -> None:
    """@mcp.shutdown must NOT run on stdio cancel when no SIGTERM flag."""
    mcp = MCPServer(name="StdIONoSigterm")
    called = {"n": 0}

    @mcp.shutdown
    async def _on_shutdown() -> None:
        called["n"] += 1

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        # no SIGTERM flag
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert called["n"] == 0


@pytest.mark.asyncio
async def test_last_initialize_handler_wins_and_ctx_shape_exists() -> None:
    """If multiple @initialize decorators are applied, only the last one should execute.
    Also sanity-check that ctx has the expected core attributes in a version-tolerant way.
    """
    port = _free_port()

    mcp = MCPServer(name="InitOverride")
    seen = {"a": False, "b": False, "has_session": False, "has_request": False}

    @mcp.initialize
    async def _init_a(ctx) -> None:  # type: ignore[override]
        # This one should get overridden and never run
        seen["a"] = True

    @mcp.initialize
    async def _init_b(ctx) -> None:  # type: ignore[override]
        # This is the one that should actually run
        seen["b"] = True
        seen["has_session"] = hasattr(ctx, "session") and ctx.session is not None
        seen["has_request"] = hasattr(ctx, "request") and ctx.request is not None

    # A simple echo tool so we can verify the server works post-init
    @mcp.tool()
    async def echo(text: str = "ok") -> str:  # type: ignore[override]
        return f"echo:{text}"

    # Start HTTP transport (quickest way to use a real client)
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
    await asyncio.sleep(0.05)

    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()

        # Call a tool to ensure init didn't break anything
        res = await c.call_tool(name="echo", arguments={"text": "ping"})
        text = getattr(res, "content", None)
        if isinstance(text, list) and text and hasattr(text[0], "text"):
            text = text[0].text
        assert text == "echo:ping"

        await c.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    # Only the last initializer should have run
    assert seen["a"] is False
    assert seen["b"] is True
    # And the ctx had the key attributes (shape may vary by lib version; just presence)
    assert seen["has_session"] is True
    assert seen["has_request"] is True


@pytest.mark.asyncio
async def test_stdio_shutdown_handler_runs_once_when_both_paths_fire() -> None:
    """Even on stdio, when SIGTERM is set, ensure shutdown runs exactly once."""
    mcp = MCPServer(name="StdIOOnce")
    calls = {"n": 0}

    @mcp.shutdown
    async def _on_shutdown() -> None:
        calls["n"] += 1

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        # Make both the lifespan.finally and run_async.finally want to execute
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert calls["n"] == 1
    # Reset global flag always
    server_mod._sigterm_received = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_async_defaults_to_stdio_and_uses_patched_stdio(monkeypatch: pytest.MonkeyPatch):
    """transport=None should default to stdio and use our patched stdio server."""
    entered = {"v": False}

    @asynccontextmanager
    async def tracking_stdio():
        entered["v"] = True
        async with _fake_stdio_server() as streams:
            yield streams

    # Override the autouse fixture for this test to track entry
    monkeypatch.setattr("fastmcp.server.server.stdio_server", tracking_stdio)

    mcp = MCPServer(name="DefaultStdio")
    task = asyncio.create_task(mcp.run_async(transport=None, show_banner=False))
    try:
        await asyncio.sleep(0.05)
        assert entered["v"] is True, "Expected stdio transport to be used by default"
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task


@pytest.mark.asyncio
async def test_custom_lifespan_relies_on_run_async_fallback_for_sigterm() -> None:
    """When a custom lifespan is supplied, run_async's finally path must still call
    @shutdown on SIGTERM."""

    @asynccontextmanager
    async def custom_lifespan(_):
        # No shutdown call here on purpose
        yield {}

    mcp = MCPServer(name="CustomLS", lifespan=custom_lifespan)
    calls = {"n": 0}

    @mcp.shutdown
    async def _s() -> None:
        calls["n"] += 1

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        # Ensure finalizer believes SIGTERM happened
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert calls["n"] == 1
    assert not getattr(server_mod, "_sigterm_received")

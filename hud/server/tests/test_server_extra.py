# filename: hud/server/tests/test_server_extra.py
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager, suppress

import anyio
import pytest

from hud.server import MCPServer
from hud.server import server as server_mod


@asynccontextmanager
async def _fake_stdio_server():
    """
    Stand-in for stdio_server that avoids reading real stdin.

    It yields a pair of in-memory streams (receive, send) so the low-level server
    can start and idle without touching sys.stdin/sys.stdout.
    """
    send_in, recv_in = anyio.create_memory_object_stream(100)
    send_out, recv_out = anyio.create_memory_object_stream(100)
    try:
        yield recv_in, send_out
    finally:
        # best-effort close across anyio versions
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
async def test_sigterm_flag_remains_true_without_shutdown_handler(patch_stdio):
    """
    When no @mcp.shutdown is registered, neither the lifespan.finally nor run_async.finally
    should reset the global SIGTERM flag. This exercises the 'no handler' branches.
    """
    mcp = MCPServer(name="NoShutdownHandler")

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        # Simulate SIGTERM path
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    # Flag must remain set since no shutdown handler was installed
    assert getattr(server_mod, "_sigterm_received") is True

    # Always reset for other tests
    server_mod._sigterm_received = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_last_shutdown_handler_wins(patch_stdio):
    """
    If multiple @mcp.shutdown decorators are applied, the last one should be the one that runs.
    """
    mcp = MCPServer(name="ShutdownOverride")
    calls: list[str] = []

    @mcp.shutdown
    async def _first() -> None:
        calls.append("first")

    @mcp.shutdown
    async def _second() -> None:
        calls.append("second")

    task = asyncio.create_task(mcp.run_async(transport="stdio", show_banner=False))
    try:
        await asyncio.sleep(0.05)
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert calls == ["second"], "Only the last registered shutdown handler should run"
    server_mod._sigterm_received = False  # type: ignore[attr-defined]


@pytest.mark.skipif(sys.platform == "win32", reason="asyncio.add_signal_handler is Unix-only")
def test__run_with_sigterm_registers_handlers_when_enabled(monkeypatch: pytest.MonkeyPatch):
    """
    Verify that _run_with_sigterm attempts to register SIGTERM/SIGINT handlers
    when the env var does NOT disable the handler. We stub AnyIO's TaskGroup so
    the watcher doesn't block and the test returns immediately.
    """
    # Ensure handler is enabled
    monkeypatch.delenv("FASTMCP_DISABLE_SIGTERM_HANDLER", raising=False)

    # Record what the server tries to register
    added_signals: list[int] = []

    import asyncio as _asyncio

    orig_get_running_loop = _asyncio.get_running_loop

    def proxy_get_running_loop():
        real = orig_get_running_loop()

        class _LoopProxy:
            __slots__ = ("_inner",)

            def __init__(self, inner):
                self._inner = inner

            def add_signal_handler(self, signum, callback, *args):
                added_signals.append(signum)  # don't actually install
                # no-op: skip calling inner.add_signal_handler to avoid OS constraints

            def __getattr__(self, name):
                # delegate everything else (create_task, call_soon, etc.)
                return getattr(self._inner, name)

        return _LoopProxy(real)

    # Patch globally so both the test and hud.server.server see the proxy
    monkeypatch.setattr(_asyncio, "get_running_loop", proxy_get_running_loop)

    # Dummy TaskGroup that runs the work but skips _watch
    class _DummyTG:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def start_soon(self, fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "_watch":
                return
            _asyncio.get_running_loop().create_task(fn(*args, **kwargs))

    monkeypatch.setattr("anyio.create_task_group", lambda: _DummyTG())

    # Simple coroutine that should run to completion
    hit = {"v": False}

    async def work():
        hit["v"] = True

    server_mod._run_with_sigterm(work)
    assert hit["v"] is True

    import signal as _signal

    assert _signal.SIGTERM in added_signals
    assert _signal.SIGINT in added_signals

from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from typing import Any, cast

import pytest

from hud.clients import MCPClient
from hud.server import MCPServer
from hud.server import server as server_mod  # for toggling _sigterm_received
from hud.server.low_level import LowLevelServerWithInit


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_http_server(mcp: MCPServer, port: int) -> asyncio.Task:
    # run the server in the background; cancel to stop
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
    # brief yield so uvicorn can boot
    await asyncio.sleep(0.05)
    return task


def _first_text(result) -> str | None:
    # Result.content is usually a list of TextContent
    c = getattr(result, "content", None)
    if isinstance(c, list) and c and hasattr(c[0], "text"):
        return c[0].text
    if isinstance(c, str):
        return c
    return None


@pytest.mark.asyncio
async def test_low_level_injection_happens_when_initialize_used() -> None:
    mcp = MCPServer(name="InitInject")
    assert not isinstance(mcp._mcp_server, LowLevelServerWithInit)

    @mcp.initialize
    async def _init(_ctx) -> None:
        return None

    assert isinstance(mcp._mcp_server, LowLevelServerWithInit)


@pytest.mark.asyncio
async def test_initialize_runs_once_and_tools_work() -> None:
    port = _free_port()

    mcp = MCPServer(name="ServerInitOnce")
    state = {"init_calls": 0, "initialized": False}

    @mcp.initialize
    async def _init(_ctx) -> None:
        # this would corrupt stdout if not redirected; we rely on stderr redirection
        print("hello from init")  # noqa: T201
        state["init_calls"] += 1
        state["initialized"] = True

    @mcp.tool()
    async def initialized() -> bool:  # type: ignore[override]
        return state["initialized"]

    @mcp.tool()
    async def echo(text: str = "ok") -> str:  # type: ignore[override]
        return f"echo:{text}"

    server_task = await _start_http_server(mcp, port)

    async def connect_and_check() -> None:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        client = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await client.initialize()
        tools = await client.list_tools()
        names = sorted(t.name for t in tools)
        assert {"echo", "initialized"} <= set(names)
        res = await client.call_tool(name="initialized", arguments={})
        # boolean return is exposed via structuredContent["result"]
        assert getattr(res, "structuredContent", {}).get("result") is True
        res2 = await client.call_tool(name="echo", arguments={"text": "ping"})
        assert _first_text(res2) == "echo:ping"
        await client.shutdown()

    try:
        await connect_and_check()
        await connect_and_check()
        # initializer should have executed only once across multiple clients
        assert state["init_calls"] == 1
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task


@pytest.mark.asyncio
async def test_shutdown_handler_only_on_sigterm_flag() -> None:
    port = _free_port()

    mcp = MCPServer(name="ShutdownTest")
    called = asyncio.Event()

    @mcp.shutdown
    async def _on_shutdown() -> None:
        called.set()

    # no SIGTERM flag: should NOT call shutdown on normal cancel
    server_task = await _start_http_server(mcp, port)
    try:
        # sanity connect so lifespan actually ran
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        await c.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task
    # give a tick to let lifespan finally run
    await asyncio.sleep(0.05)
    assert not called.is_set()

    # now start again and simulate SIGTERM so shutdown handler fires
    called.clear()
    port2 = _free_port()
    server_task2 = await _start_http_server(mcp, port=port2)
    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port2}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        await c.shutdown()

        # flip the module-level flag the lifespan checks
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            server_task2.cancel()
            await server_task2

    # shutdown coroutine should have run because flag was set when lifespan exited
    assert called.is_set()
    # reset the flag for any other tests
    server_mod._sigterm_received = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_initializer_exception_propagates_to_client() -> None:
    port = _free_port()

    mcp = MCPServer(name="InitError")

    @mcp.initialize
    async def _init(_ctx) -> None:
        raise RuntimeError("boom during init")

    server_task = await _start_http_server(mcp, port)

    cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
    client = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)

    try:
        with pytest.raises(Exception):
            await client.initialize()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task
        # defensive: client may or may not be fully created
        with suppress(Exception):
            await client.shutdown()


# --- additional tests for MCPServer coverage ---


@pytest.mark.asyncio
async def test_init_after_tools_preserves_handlers_and_runs_once() -> None:
    """If tools are added BEFORE @mcp.initialize, the handler copy during
    low-level server replacement must keep them; init should still run once total.
    """
    port = _free_port()

    mcp = MCPServer(name="InitAfterTools")
    state = {"init_calls": 0}

    # Register tools first
    @mcp.tool()
    async def foo() -> str:  # type: ignore[override]
        return "bar"

    # Now register initializer (this triggers server replacement)
    @mcp.initialize
    async def _init(_ctx) -> None:
        state["init_calls"] += 1

    server_task = await _start_http_server(mcp, port)

    async def connect_and_check() -> None:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        tools = await c.list_tools()
        names = sorted(t.name for t in tools)
        assert "foo" in names, "tool registered before @initialize must survive replacement"
        res = await c.call_tool(name="foo", arguments={})
        assert _first_text(res) == "bar"
        await c.shutdown()

    try:
        await connect_and_check()
        await connect_and_check()
        assert state["init_calls"] == 1, "initializer should execute exactly once"
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task


@pytest.mark.asyncio
async def test_tool_default_argument_used_when_omitted() -> None:
    """Echo tool should use its default when argument is omitted."""
    port = _free_port()

    mcp = MCPServer(name="EchoDefault")

    @mcp.tool()
    async def echo(text: str = "ok") -> str:  # type: ignore[override]
        return f"echo:{text}"

    server_task = await _start_http_server(mcp, port)
    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        # Call with no args → default should kick in
        res = await c.call_tool(name="echo", arguments={})
        assert _first_text(res) == "echo:ok"
        await c.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task


@pytest.mark.asyncio
async def test_shutdown_handler_runs_once_when_both_paths_fire() -> None:
    """With SIGTERM flag set, both the lifespan.finally and run_async.finally would
    try to invoke @mcp.shutdown. The per-instance guard must ensure exactly once.
    """
    port = _free_port()
    mcp = MCPServer(name="ShutdownOnce")
    calls = {"n": 0}

    @mcp.shutdown
    async def _on_shutdown() -> None:
        calls["n"] += 1

    server_task = await _start_http_server(mcp, port)
    try:
        # Ensure lifespan started
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        await c.shutdown()

        # Arm SIGTERM flag so both code paths believe they should run
        server_mod._sigterm_received = True  # type: ignore[attr-defined]
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task

    # Give the event loop a tick to run both finalizers
    await asyncio.sleep(0.05)

    try:
        assert calls["n"] == 1, f"shutdown hook must run exactly once, got {calls['n']}"
    finally:
        # Always reset module flag
        server_mod._sigterm_received = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_initialize_ctx_exposes_client_info() -> None:
    """Initializer gets a ctx; clientInfo may be absent depending on client implementation."""
    port = _free_port()

    mcp = MCPServer(name="InitCtx")
    seen = {"has_session": False, "client_name": None}

    @mcp.initialize
    async def _init(ctx) -> None:  # type: ignore[override]
        # Ensure we have a session object
        seen["has_session"] = hasattr(ctx, "session") and ctx.session is not None

        # Client info is optional; capture it if present
        client_info = getattr(getattr(ctx.session, "client_params", None), "clientInfo", None)
        if client_info is not None:
            seen["client_name"] = getattr(client_info, "name", None)

    server_task = await _start_http_server(mcp, port)
    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        await c.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task

    assert seen["has_session"] is True
    # If present, name should be a string; otherwise None is acceptable.
    assert seen["client_name"] is None or isinstance(seen["client_name"], str)


@pytest.mark.asyncio
async def test_initialize_redirects_stdout_to_stderr(capsys) -> None:
    """Initializer prints should be redirected to stderr (never stdout)."""
    port = _free_port()

    mcp = MCPServer(name="StdoutRedirect")

    @mcp.initialize
    async def _init(_ctx) -> None:
        # This would normally pollute STDOUT; our server redirects to STDERR
        print("INIT_STDOUT_MARKER")  # noqa: T201

    server_task = await _start_http_server(mcp, port)

    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c.initialize()
        await c.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task

    captured = capsys.readouterr()
    assert "INIT_STDOUT_MARKER" in captured.err
    assert "INIT_STDOUT_MARKER" not in captured.out


@pytest.mark.asyncio
async def test_initialize_callable_form_runs_once() -> None:
    """Coverage for mcp.initialize(fn) (callable style), not only decorator usage."""
    port = _free_port()
    mcp = MCPServer(name="CallableInit")
    hits = {"n": 0}

    async def _init(_ctx) -> None:
        hits["n"] += 1

    # Callable form instead of decorator
    mcp.initialize(_init)

    server_task = await _start_http_server(mcp, port)
    try:
        cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
        c1 = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c1.initialize()
        await c1.shutdown()

        c2 = MCPClient(mcp_config=cfg, auto_trace=False, verbose=False)
        await c2.initialize()
        await c2.shutdown()
    finally:
        with suppress(asyncio.CancelledError):
            server_task.cancel()
            await server_task

    assert hits["n"] == 1


@pytest.mark.asyncio
async def test_notification_handlers_survive_real_replacement() -> None:
    """End-to-end check that notification handlers survive when initialize is registered."""
    mcp = MCPServer(name="NotifCopy")

    # Seed a dummy notification handler before replacement
    cast("dict[Any, Any]", mcp._mcp_server.notification_handlers)["hud/notify"] = object()
    assert "hud/notify" in mcp._mcp_server.notification_handlers

    @mcp.initialize
    async def _init(_ctx) -> None:
        pass

    # After replacement, the handler should still be there
    assert "hud/notify" in mcp._mcp_server.notification_handlers

"""``MCPClient`` against FastMCP servers and scripted gateway faults.

Tool calls go to a FastMCP server over each HTTP transport, or to a scripted
streamable-HTTP server on the fake services, whose ``tools/call`` answers come
from a script: a transient gateway status, a body that is not JSON, or a dropped
connection. Every call resolves:
with a result once a transient error clears, or with a transport error.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError

from hud.capabilities import Capability, MCPClient
from tests.clients.conftest import relayed
from tests.conftest import eventually
from tests.fixtures.asgi import mcp_server, serve_asgi
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from starlette.types import ASGIApp, Receive, Scope, Send

    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices, Request


def tools() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def refuse() -> str:
        """Always fail."""
        raise ToolError("not today")

    return server


@pytest.mark.parametrize(("transport", "path"), [("streamable-http", "/mcp"), ("sse", "/sse")])
async def test_tool_calls_return_canonical_results_over_each_transport(
    transport: Literal["streamable-http", "sse"], path: str
) -> None:
    app = tools().http_app(transport=transport, path=path)
    async with serve_asgi(app) as port:
        client = await MCPClient.connect(
            Capability.mcp(url=f"http://127.0.0.1:{port}{path}", transport=transport)
        )
        try:
            added = await client.call_tool("add", {"a": 2, "b": 3})
            refused = await client.call_tool("refuse", {})
        finally:
            await client.close()

    assert (added.isError, added.structuredContent) == (False, {"result": 5})
    assert [block.model_dump(include={"type", "text"}) for block in added.content] == [
        {"type": "text", "text": "5"}
    ]
    assert refused.isError is True
    assert [block.model_dump(include={"type", "text"}) for block in refused.content] == [
        {"type": "text", "text": "not today"}
    ]


def bearer_only(app: ASGIApp, token: str, seen: list[str]) -> ASGIApp:
    """``app`` behind a gate that answers 401 to any HTTP request without ``token``."""

    async def gated(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            authorization = dict(scope["headers"]).get(b"authorization", b"").decode()
            seen.append(authorization)
            if authorization != f"Bearer {token}":
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await app(scope, receive, send)

    return gated


async def test_an_auth_token_rides_every_request_and_is_needed_where_the_server_asks() -> None:
    seen: list[str] = []
    app = bearer_only(tools().http_app(path="/mcp"), "s3cret", seen)

    async with serve_asgi(app) as port:
        url = f"http://127.0.0.1:{port}/mcp"
        client = await MCPClient.connect(Capability.mcp(url=url, auth_token="s3cret"))
        try:
            added = await client.call_tool("add", {"a": 2, "b": 3})
        finally:
            await client.close()
        authorized = set(seen)
        with pytest.raises(McpError, match="HTTPStatusError"):
            await MCPClient.connect(Capability.mcp(url=url))

    assert added.structuredContent == {"result": 5}
    assert authorized == {"Bearer s3cret"}


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


def result(request: Request, body: dict[str, Any]) -> Reply:
    return Reply(json={"jsonrpc": "2.0", "id": request.json["id"], "result": body})


class ScriptedMCPServer:
    """A streamable-HTTP MCP server whose ``tools/call`` answers come from a script."""

    def __init__(self, services: FakeServices, calls: list[Callable[[Request], Reply]]) -> None:
        self.calls = calls
        self.url = f"{services.url('runtime')}/mcp"
        services.route("runtime", "POST", "/mcp", handler=self.post)
        services.route("runtime", "GET", "/mcp", Reply(status=405))
        services.route("runtime", "DELETE", "/mcp", Reply(json={}))
        self.services = services

    def post(self, request: Request) -> Reply:
        method = request.json.get("method")
        if method == "initialize":
            reply = result(
                request,
                {
                    "protocolVersion": request.json["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                },
            )
            reply.headers["mcp-session-id"] = "session-1"
            return reply
        if "id" not in request.json:
            return Reply(status=202, body=b"")
        if method == "tools/list":
            tool = {"name": "count", "inputSchema": {"type": "object"}}
            return result(request, {"tools": [{**tool, "outputSchema": OUTPUT_SCHEMA}]})
        answer = self.calls.pop(0) if len(self.calls) > 1 else self.calls[0]
        return answer(request)

    def attempts(self) -> int:
        return sum(
            request.json.get("method") == "tools/call"
            for request in self.services.requests("runtime", "POST", "/mcp")
        )

    async def call(self) -> Any:
        client = await MCPClient.connect(Capability.mcp(url=self.url, transport="streamable-http"))
        try:
            return await asyncio.wait_for(client.call_tool("count", {}), timeout=30)
        finally:
            await client.close()


def counted(request: Request) -> Reply:
    return result(
        request, {"content": [{"type": "text", "text": "3"}], "structuredContent": {"n": 3}}
    )


def status(code: int) -> Callable[[Request], Reply]:
    return lambda request: Reply(status=code, json={})


def unparseable(request: Request) -> Reply:
    del request
    return Reply(body=b"{not json", content_type="application/json")


def dropped(request: Request) -> Reply:
    del request

    def chunks() -> Iterator[bytes]:
        yield b'{"jsonrpc": "2.0", '
        raise ConnectionResetError("server went away")

    return Reply(stream=chunks(), content_type="application/json")


async def test_a_tool_call_succeeds_after_a_transient_503(services: FakeServices) -> None:
    server = ScriptedMCPServer(services, [status(503), counted])

    called = await server.call()

    assert (called.isError, called.structuredContent) == (False, {"n": 3})
    assert server.attempts() == 2


@pytest.mark.parametrize(
    ("calls", "attempts"),
    [
        pytest.param([status(503)], 3, id="503-until-the-deadline"),
        pytest.param([unparseable], 1, id="unparseable-json"),
        pytest.param([dropped], 1, id="connection-dropped"),
    ],
)
async def test_a_failing_tool_call_resolves_with_a_transport_error(
    services: FakeServices,
    hud_env: HudEnv,
    calls: list[Callable[[Request], Reply]],
    attempts: int,
) -> None:
    hud_env.set(HUD_CLIENT_TIMEOUT="1")
    server = ScriptedMCPServer(services, calls)

    with pytest.raises(McpError) as raised:
        await server.call()

    assert raised.value.error.code == -32000
    assert raised.value.error.message.startswith("Transport error")
    assert server.attempts() == attempts


async def test_a_tool_call_after_the_connection_drops_reconnects() -> None:
    async with (
        mcp_server(tools()) as served,
        relayed(served) as (capability, flaky),
    ):
        client = await MCPClient.connect(capability)
        try:
            await client.call_tool("add", {"a": 2, "b": 3})
            accepted = flaky.accepted
            flaky.sever()
            added = await client.call_tool("add", {"a": 2, "b": 3})
        finally:
            await client.close()

    assert added.structuredContent == {"result": 5}
    assert flaky.accepted > accepted


async def test_a_cancelled_and_a_timed_out_tool_call_end_and_the_next_one_runs() -> None:
    started: list[str] = []
    server = tools()

    @server.tool
    async def wait() -> str:
        """Never answer."""
        started.append("wait")
        await asyncio.Event().wait()
        return "unreachable"

    async with mcp_server(server) as capability:
        client = await MCPClient.connect(capability)
        try:
            call = asyncio.create_task(client.call_tool("wait", {}))
            await eventually(lambda: started == ["wait"])
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call

            with pytest.raises(TimeoutError):
                await asyncio.wait_for(client.call_tool("wait", {}), 0.5)
            await eventually(lambda: started == ["wait", "wait"])

            added = await client.call_tool("add", {"a": 2, "b": 3})
        finally:
            await client.close()

    assert added.structuredContent == {"result": 5}

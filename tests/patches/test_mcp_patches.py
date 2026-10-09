"""MCP calls over streamable HTTP resolve instead of hanging, and tolerate off-schema output."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from fastmcp import FastMCP
from mcp.shared.exceptions import McpError

from hud.capabilities import Capability, MCPClient
from tests.harness import Reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from tests.harness import FakeServices, HudEnv, Request

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


def _result(request: Request, result: dict[str, Any]) -> Reply:
    return Reply(json={"jsonrpc": "2.0", "id": request.json["id"], "result": result})


class FakeMCPServer:
    """A streamable-HTTP MCP server whose ``tools/call`` answers come from a script."""

    def __init__(self, services: FakeServices, calls: list[Callable[[Request], Reply]]) -> None:
        self._calls = calls
        self.url = f"{services.url('runtime')}/mcp"
        services.route("runtime", "POST", "/mcp", handler=self._post)
        services.route("runtime", "GET", "/mcp", Reply(status=405))
        services.route("runtime", "DELETE", "/mcp", Reply(json={}))
        self.services = services

    def _post(self, request: Request) -> Reply:
        method = request.json.get("method")
        if method == "initialize":
            reply = _result(
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
            return _result(request, {"tools": [{**tool, "outputSchema": OUTPUT_SCHEMA}]})
        answer = self._calls.pop(0) if len(self._calls) > 1 else self._calls[0]
        return answer(request)

    def attempts(self) -> int:
        return sum(
            request.json.get("method") == "tools/call"
            for request in self.services.requests("runtime", "POST", "/mcp")
        )


def _ok(request: Request) -> Reply:
    return _result(
        request,
        {"content": [{"type": "text", "text": "3"}], "structuredContent": {"n": 3}},
    )


def _status(status: int) -> Callable[[Request], Reply]:
    return lambda request: Reply(status=status, json={})


def _off_schema(request: Request) -> Reply:
    return _result(
        request,
        {"content": [{"type": "text", "text": "x"}], "structuredContent": {"n": "x"}},
    )


def _unparseable(request: Request) -> Reply:
    del request
    return Reply(body=b"{not json", content_type="application/json")


def _dropped(request: Request) -> Reply:
    del request

    def chunks() -> Iterator[bytes]:
        yield b'{"jsonrpc": "2.0", '
        raise ConnectionResetError("server went away")

    return Reply(stream=chunks(), content_type="application/json")


async def _call(server: FakeMCPServer) -> Any:
    client = await MCPClient.connect(Capability.mcp(url=server.url, transport="streamable-http"))
    try:
        return await asyncio.wait_for(client.call_tool("count", {}), timeout=30)
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("calls", "structured", "attempts"),
    [
        pytest.param([_status(502), _ok], {"n": 3}, 2, id="502-retried"),
        pytest.param([_status(503), _ok], {"n": 3}, 2, id="503-retried"),
        pytest.param([_status(504), _ok], {"n": 3}, 2, id="504-retried"),
        pytest.param([_off_schema], {"n": "x"}, 1, id="output-outside-its-schema-accepted"),
    ],
)
async def test_a_tool_call_succeeds_after_transient_gateway_errors(
    services: FakeServices,
    calls: list[Callable[[Request], Reply]],
    structured: dict[str, Any],
    attempts: int,
) -> None:
    server = FakeMCPServer(services, calls)

    result = await _call(server)

    assert result.isError is False
    assert result.structuredContent == structured
    assert server.attempts() == attempts


@pytest.mark.parametrize(
    ("calls", "error_type", "attempts"),
    [
        pytest.param([_status(503)], "HTTPStatusError", 3, id="503-until-the-deadline"),
        pytest.param([_unparseable], "ValidationError", 1, id="unparseable-json"),
        pytest.param([_dropped], "RemoteProtocolError", 1, id="connection-dropped"),
    ],
)
async def test_a_failing_tool_call_resolves_with_a_transport_error(
    services: FakeServices,
    hud_env: HudEnv,
    calls: list[Callable[[Request], Reply]],
    error_type: str,
    attempts: int,
) -> None:
    hud_env.set(HUD_CLIENT_TIMEOUT="1")
    server = FakeMCPServer(services, calls)

    with pytest.raises(McpError) as raised:
        await _call(server)

    assert raised.value.error.code == -32000
    assert raised.value.error.message == f"Transport error: {error_type}"
    assert server.attempts() == attempts


@asynccontextmanager
async def _serving(mcp: FastMCP) -> AsyncIterator[str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    for _ in range(1000):
        if server.started:
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("the MCP server did not start")
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/mcp"
    finally:
        server.should_exit = True
        await serving


async def test_a_fastmcp_tool_wraps_plain_results_and_skips_output_validation() -> None:
    mcp = FastMCP("wrapped")

    @mcp.tool()
    def numbers() -> list[int]:
        return [1, 2]

    @mcp.tool()
    def mistyped() -> int:
        wrong: Any = "not an int"
        return wrong

    async with _serving(mcp) as url:
        client = await MCPClient.connect(Capability.mcp(url=url, transport="streamable-http"))
        try:
            wrapped = await client.call_tool("numbers", {})
            off_schema = await client.call_tool("mistyped", {})
        finally:
            await client.close()

    assert (wrapped.isError, wrapped.structuredContent) == (False, {"result": [1, 2]})
    assert (off_schema.isError, off_schema.structuredContent) == (False, {"result": "not an int"})

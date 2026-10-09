"""A code revealed one digit per tool call: only state kept across calls yields all of it."""

from __future__ import annotations

import asyncio
import secrets
import socket

import uvicorn
from fastmcp import FastMCP

from hud import Environment
from hud.capabilities import Capability

server = FastMCP(name="oracle-state")
code: list[str] = []
revealed: list[str] = []


@server.tool
def next_digit() -> str:
    """Return the next digit of the code, or "done" once every digit is revealed."""
    if len(revealed) == len(code):
        return "done"
    revealed.append(code[len(revealed)])
    return revealed[-1]


env = Environment("oracle-state")
http = uvicorn.Server(uvicorn.Config(server.http_app(path="/mcp"), log_level="error"))
serving: list[asyncio.Task[None]] = []


@env.initialize
async def serve_tools() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    serving.append(asyncio.create_task(http.serve(sockets=[listener])))
    while not http.started:  # noqa: ASYNC110 - uvicorn signals readiness only by this flag
        await asyncio.sleep(0.01)
    port = listener.getsockname()[1]
    env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def stop_tools() -> None:
    http.should_exit = True
    await asyncio.gather(*serving)


@env.template(id="collect-code")
async def collect_code(length: int):
    code[:] = [str(secrets.randbelow(10)) for _ in range(length)]
    revealed.clear()
    answer = yield (
        'Call next_digit until it returns "done", then reply with every digit it returned, '
        "in order, as one string."
    )
    yield 1.0 if (answer or "").strip() == "".join(code) and revealed == code else 0.0


tasks = [collect_code(length=4)]

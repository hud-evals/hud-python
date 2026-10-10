"""A code revealed one digit per tool call: a reward of 1 proves state was kept across calls."""

from __future__ import annotations

import asyncio
import secrets
import socket

import uvicorn
from fastmcp import FastMCP

from hud import Environment
from hud.capabilities import Capability

server = FastMCP(name="digits")
code: list[str] = []
revealed: list[str] = []


@server.tool
def next_digit() -> str:
    """Return the next digit of the code, or "done" once every digit is revealed."""
    if len(revealed) == len(code):
        return "done"
    revealed.append(code[len(revealed)])
    return revealed[-1]


env = Environment("tools")
running: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []


@env.initialize
async def serve_tools() -> None:
    http = uvicorn.Server(uvicorn.Config(server.http_app(path="/mcp"), log_level="error"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    running.append((http, asyncio.create_task(http.serve(sockets=[listener]))))
    while not http.started:  # noqa: ASYNC110 - uvicorn signals readiness only by this flag
        await asyncio.sleep(0.01)
    port = listener.getsockname()[1]
    env.add_capability(Capability.mcp(name="digits", url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def stop_tools() -> None:
    for http, _ in running:
        http.should_exit = True
    await asyncio.gather(*(task for _, task in running))
    running.clear()


@env.template()
async def collect_code(length: int = 4):
    code[:] = [str(secrets.randbelow(10)) for _ in range(length)]
    revealed.clear()
    answer = yield (
        'Call next_digit until it returns "done", then reply with every digit it returned, '
        "in order, as one string."
    )
    yield 1.0 if answer == "".join(code) and revealed == code else 0.0

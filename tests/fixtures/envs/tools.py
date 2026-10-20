"""A code revealed one digit per tool call: a reward of 1 proves state was kept across calls."""

from __future__ import annotations

import re
import secrets
from contextlib import AsyncExitStack

from fastmcp import FastMCP

from hud import Environment
from hud.capabilities import Capability
from tests.fixtures.asgi import serve_asgi

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
stack = AsyncExitStack()


@env.initialize
async def serve_tools() -> None:
    port = await stack.enter_async_context(serve_asgi(server.http_app(path="/mcp")))
    env.add_capability(Capability.mcp(name="digits", url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def stop_tools() -> None:
    await stack.aclose()


@env.template()
async def collect_code(length: int = 4):
    code[:] = [str(secrets.randbelow(10)) for _ in range(length)]
    revealed.clear()
    answer = yield (
        'Call next_digit until it returns "done", then reply with every digit it returned, '
        "in order, as one string."
    )
    # A real model may wrap the code in prose; the digits it gives are what count.
    yield 1.0 if re.sub(r"\D", "", answer) == "".join(code) and revealed == code else 0.0


tasks = [collect_code()]

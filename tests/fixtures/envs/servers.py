"""Two MCP servers with a tool of the same name: a reward of 1 proves each call reached its own.

With more than one server, the agent qualifies tool names by server, so the
model must tell ``database__lookup`` from ``search__lookup``.
"""

from __future__ import annotations

import secrets
from contextlib import AsyncExitStack

from fastmcp import FastMCP

from hud import Environment
from hud.capabilities import Capability
from tests.fixtures.asgi import serve_asgi

answers: dict[str, str] = {}


def server(name: str) -> FastMCP:
    mcp = FastMCP(name)

    @mcp.tool
    def lookup(key: str) -> str:
        """Look a key up."""
        return answers[f"{name}:{key}"]

    return mcp


env = Environment("servers")
stack = AsyncExitStack()


@env.initialize
async def serve_tools() -> None:
    for name in ("database", "search"):
        port = await stack.enter_async_context(serve_asgi(server(name).http_app(path="/mcp")))
        env.add_capability(Capability.mcp(name=name, url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def stop_tools() -> None:
    await stack.aclose()


@env.template()
async def look_up_both():
    answers.update({"database:alpha": secrets.token_hex(4), "search:beta": secrets.token_hex(4)})
    answer = yield (
        "Look up the key 'alpha' in the database and the key 'beta' in the search index, "
        "then reply with both results."
    )
    found = answers["database:alpha"] in answer and answers["search:beta"] in answer
    yield 1.0 if found else 0.0

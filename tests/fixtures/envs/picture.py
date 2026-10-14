"""A tool that shows a colour: only a model that saw the image can name it.

The prompt never says which colour the task chose, so with a real model a reward
of 1 proves the tool's image reached it.
"""

from __future__ import annotations

import base64
import io
from contextlib import AsyncExitStack

import mcp.types as mcp_types
from fastmcp import FastMCP
from PIL import Image

from hud import Environment
from hud.capabilities import Capability
from tests.harness import serve_asgi

server = FastMCP(name="picture")
shown: list[str] = []


@server.tool
def look() -> list[mcp_types.ContentBlock]:
    """Show the picture."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), shown[-1]).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    return [mcp_types.ImageContent(type="image", data=data, mimeType="image/png")]


env = Environment("picture")
stack = AsyncExitStack()


@env.initialize
async def serve_picture() -> None:
    port = await stack.enter_async_context(serve_asgi(server.http_app(path="/mcp")))
    env.add_capability(Capability.mcp(name="picture", url=f"http://127.0.0.1:{port}/mcp"))


@env.shutdown
async def stop_picture() -> None:
    await stack.aclose()


@env.template()
async def name_the_colour(colour: str):
    shown.append(colour)
    answer = yield "Call the look tool, then reply with the name of the colour it shows."
    yield 1.0 if colour in answer.lower() else 0.0

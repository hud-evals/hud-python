"""Fixture environments and request views shared by the agent scenarios.

Every scenario runs a real rollout: a :class:`~hud.Environment` serving a real
workspace over SSH (and a fake screen or MCP servers when the row needs them),
placed by :class:`~hud.eval.LocalRuntime`, with the agent talking to the
scripted providers behind the fake gateway.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

import uvicorn
from PIL import Image

from hud import Environment
from hud.capabilities import Capability
from hud.eval import LocalRuntime, Task, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path

    from fastmcp import FastMCP

    from hud.agents.base import Agent
    from hud.eval import Run
    from tests.harness import ModelRequest

PROMPT = "Do the task."
DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.*)$", re.DOTALL)


def workspace_env(
    root: Path,
    *,
    files: Mapping[str, bytes | str] | None = None,
    passes: Callable[[Path], bool] = lambda _root: True,
    capabilities: tuple[Capability, ...] = (),
    name: str = "ws",
    **workspace: Any,
) -> Environment:
    """An environment serving ``root`` over SSH with one template, ``task``.

    ``files`` seed the workspace before each run; the task scores 1.0 when
    ``passes(root)`` holds after the agent finishes. ``workspace`` goes to
    :meth:`Environment.workspace` (``env=`` for the shell environment).
    """
    env = Environment(name, capabilities=capabilities)
    env.workspace(root, guest_path=str(root), **workspace)

    @env.initialize
    async def seed() -> None:
        write_files(root, files or {})

    @env.template()
    async def task(prompt: str = PROMPT):
        yield prompt
        yield 1.0 if passes(root) else 0.0

    return env


async def run_task(env: Environment, agent: Agent, *, prompt: str = PROMPT, **options: Any) -> Run:
    """Roll ``agent`` out on ``env``'s ``task`` in this process."""
    args = {} if prompt == PROMPT else {"prompt": prompt}
    return await rollout(
        Task(env=env.name, id="task", args=args), agent, runtime=LocalRuntime(env), **options
    )


def write_files(root: Path, files: Mapping[str, bytes | str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)


@asynccontextmanager
async def mcp_server(
    server: FastMCP,
    *,
    name: str = "tools",
    transport: Literal["streamable-http", "sse"] = "streamable-http",
) -> AsyncIterator[Capability]:
    """Serve ``server`` over HTTP on loopback; yield the capability that reaches it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    path = "/mcp" if transport == "streamable-http" else "/sse"
    app = server.http_app(path=path, transport=transport)
    http = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    serving = asyncio.create_task(http.serve(sockets=[sock]))
    while not http.started:
        if serving.done():
            await serving
            raise RuntimeError("MCP server stopped before it started")
        await asyncio.sleep(0.01)
    try:
        yield Capability.mcp(name=name, url=f"http://127.0.0.1:{port}{path}", transport=transport)
    finally:
        http.should_exit = True
        await serving


def image(
    size: tuple[int, int],
    *,
    format: str = "PNG",
    mode: str = "RGB",
    color: tuple[int, ...] | str = "white",
    exif: Image.Exif | None = None,
) -> bytes:
    buffer = io.BytesIO()
    options: dict[str, Any] = {} if exif is None else {"exif": exif}
    Image.new(mode, size, color).save(buffer, format=format, **options)
    return buffer.getvalue()


def describe_image(mime: str, data: str) -> str:
    """``<mime WxH>`` for base64 image data, so a snapshot pins type and size, not bytes."""
    try:
        raw = base64.b64decode(data, altchars=b"-_" if "-" in data or "_" in data else None)
        with Image.open(io.BytesIO(raw)) as decoded:
            return f"<{mime} {decoded.width}x{decoded.height}>"
    except (OSError, ValueError):
        return f"<{mime} undecodable>"


def wire(value: Any, *paths: Path, descriptions: bool = False) -> Any:
    """A request body as a snapshot compares it.

    Base64 images become ``<mime WxH>``, each of ``paths`` becomes ``<path0>``,
    ``<path1>``, ... wherever it appears in a string, and tool and schema
    ``description`` prose is dropped unless ``descriptions`` is set.
    """
    if isinstance(value, dict):
        mime = value.get("media_type") or value.get("mimeType") or value.get("mime_type")
        return {
            key: (
                describe_image(mime, item)
                if key == "data" and isinstance(mime, str) and mime.startswith("image/")
                else wire(item, *paths, descriptions=descriptions)
            )
            for key, item in value.items()
            if descriptions or key != "description" or not isinstance(item, str)
        }
    if isinstance(value, list):
        return [wire(item, *paths, descriptions=descriptions) for item in value]
    if isinstance(value, str):
        if (match := DATA_URL.match(value)) and match["mime"].startswith("image/"):
            return describe_image(match["mime"], match["data"])
        for index, path in enumerate(paths):
            value = value.replace(str(path), f"<path{index}>")
        return value
    return value


def tool_results(requests: list[ModelRequest]) -> list[Any]:
    """The tool result entries a run handed back to its model, oldest first.

    Responses requests carry only the items new since the previous response,
    so their inputs are gathered from every request after the first; the other
    protocols resend the whole conversation, so the last request holds them all.
    ``cache_control`` markers, which move with the conversation, are dropped.
    """
    last = requests[-1]
    if last.protocol == "responses":
        return [item for request in requests[1:] for item in request.body["input"]]
    if last.protocol == "gemini":
        return [
            part
            for content in last.body["contents"]
            if content["role"] == "user"
            for part in content["parts"]
            if "functionResponse" in part
        ]
    if last.protocol == "anthropic":
        return [
            {key: value for key, value in block.items() if key != "cache_control"}
            for message in last.body["messages"][1:]
            if message["role"] == "user"
            for block in message["content"]
        ]
    return [message for message in last.body["messages"] if message["role"] == "tool"]


def json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]

"""The environment, agents and MCP servers the agent scenarios share.

Every scenario runs a real rollout: a :class:`~hud.Environment` serving a real
workspace over SSH (and a fake screen or MCP servers when the row needs them),
placed by :class:`~hud.eval.LocalRuntime`, with the agent talking to the
scripted providers behind the fake gateway.
"""

from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from hud import Environment
from hud.capabilities import Capability
from hud.eval import LocalRuntime, Task, rollout
from tests.support import serve_asgi, write_files

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path

    from fastmcp import FastMCP

    from hud.agents.base import Agent
    from hud.eval import Run

PROMPT = "Do the task."


def workspace_env(
    root: Path,
    *,
    files: Mapping[str, bytes | str] | None = None,
    passes: Callable[[Path], bool] = lambda _root: True,
    capabilities: tuple[Capability, ...] = (),
    name: str = "ws",
    settle: float = 0.0,
    **workspace: Any,
) -> Environment:
    """An environment serving ``root`` over SSH with one template, ``task``.

    ``files`` seed the workspace before each run; the task scores 1.0 when
    ``passes(root)`` holds ``settle`` seconds after the agent finishes, long
    enough for anything the agent left running to show. ``workspace`` goes to
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
        await asyncio.sleep(settle)
        yield 1.0 if passes(root) else 0.0

    return env


async def run_task(env: Environment, agent: Agent, *, prompt: str = PROMPT, **options: Any) -> Run:
    """Roll ``agent`` out on ``env``'s ``task`` in this process."""
    args = {} if prompt == PROMPT else {"prompt": prompt}
    return await rollout(
        Task(env=env.name, id="task", args=args), agent, runtime=LocalRuntime(env), **options
    )


@asynccontextmanager
async def mcp_server(
    server: FastMCP,
    *,
    name: str = "tools",
    transport: Literal["streamable-http", "sse"] = "streamable-http",
) -> AsyncIterator[Capability]:
    """Serve ``server`` over HTTP on loopback; yield the capability that reaches it."""
    path = "/mcp" if transport == "streamable-http" else "/sse"
    async with serve_asgi(server.http_app(path=path, transport=transport)) as port:
        yield Capability.mcp(name=name, url=f"http://127.0.0.1:{port}{path}", transport=transport)


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

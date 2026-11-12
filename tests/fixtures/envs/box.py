"""An environment that uses its box: it writes files where it is told and allocates memory.

Placed on a provider with a filesystem interface, its files prove what the
runtime moved; placed under a memory limit, its allocation proves the limit
is enforced and reported.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from hud import Environment

env = Environment("box")


@env.template()
async def write(path: str, text: str):
    """Write ``text`` at ``path`` before the prompt; the grade is 1 once the agent answers."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target.write_text, text)
    answer = yield f"Reply with: {text}"
    yield 1.0 if answer == text else 0.0


@env.template()
async def read(path: str):
    """Hand the agent the file's text; the grade is 1 when the agent repeats it."""
    text = await asyncio.to_thread(Path(path).read_text)
    answer = yield f"Reply with: {text}"
    yield 1.0 if answer == text else 0.0


@env.template()
async def hog(mb: int):
    """Hold ``mb`` MiB of touched memory for the prompt's duration."""
    held = bytearray(mb * 2**20)
    held[::4096] = b"x" * len(held[::4096])
    answer = yield f"Reply with: {len(held)}"
    yield 1.0 if answer == str(len(held)) else 0.0

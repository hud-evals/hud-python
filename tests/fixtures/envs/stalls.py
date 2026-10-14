"""An environment that never finishes starting: its initialize hook waits forever."""

from __future__ import annotations

import asyncio

from hud import Environment

env = Environment("stalls")


@env.initialize
async def wait_forever() -> None:
    await asyncio.Event().wait()


@env.template()
async def never():
    yield "never"
    yield 0.0

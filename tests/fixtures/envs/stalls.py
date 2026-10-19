"""An environment that never finishes starting: its initialize hook waits forever."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from hud import Environment

env = Environment("stalls")


def record(event: str) -> None:
    """Append ``event`` and this process's id to the lifecycle log the test names, if any."""
    log = os.environ.get("HUD_FIXTURE_LIFECYCLE")
    if log:
        with Path(log).open("a") as out:
            out.write(f"{event} {os.getpid()}\n")


DAEMONS: list[asyncio.subprocess.Process] = []


@env.initialize
async def record_start() -> None:
    """Log the start, and start a daemon the way an env starts a browser or a display."""
    record("start")
    if os.environ.get("HUD_FIXTURE_LIFECYCLE"):
        DAEMONS.append(await asyncio.create_subprocess_exec("sleep", "600"))
        record(f"daemon {DAEMONS[-1].pid}")


@env.shutdown
async def record_stop() -> None:
    for daemon in DAEMONS:
        daemon.kill()
        await daemon.wait()
    record("stop")


@env.initialize
async def wait_forever() -> None:
    await asyncio.Event().wait()


@env.template()
async def never():
    yield "never"
    yield 0.0

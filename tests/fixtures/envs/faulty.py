"""One template per point a task can fail: its start, its grade, or a grade that never comes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from hud import Environment

env = Environment("faulty")


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


@env.template()
async def start_raises():
    raise RuntimeError("start exploded")
    yield "never"


@env.template()
async def grade_raises():
    yield "go"
    raise RuntimeError("grader exploded")
    yield 0.0


@env.template()
async def scoreless():
    yield "go"
    yield {"done": True}


@env.template()
async def bool_score():
    yield "go"
    yield {"score": True}


@env.template()
async def hang_grading():
    yield "go"
    await asyncio.Event().wait()
    yield 1.0

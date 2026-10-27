import asyncio
import os
import threading
from pathlib import Path

from hud import Environment

assert threading.current_thread() is threading.main_thread()

env = Environment("demo")
markers = os.environ.get("EXPORT_TEST_MARKERS")


@env.initialize
async def started() -> None:
    if markers:
        await asyncio.to_thread(Path(markers, "initialized").touch)


@env.shutdown
async def stopped() -> None:
    if markers:
        await asyncio.to_thread(Path(markers, "stopped").touch)


@env.template()
async def solve(n: int = 1, prompt: str = ""):
    answer = yield f"solve {n}: {prompt}"
    if answer == "crash":
        raise RuntimeError("the grader is broken")
    yield 1.0 if answer == str(n) else 0.25


tasks = [solve(n=2, prompt="what's next?")]

import asyncio
import os
from pathlib import Path

from hud import Environment
from hud.capabilities import Capability

env = Environment("demo")
env.add_capability(Capability.cdp(url="ws://127.0.0.1:9222"))
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

"""Run browser-use on a taskset whose environment publishes a browser as ``cdp``:
``uv run run.py path/to/tasks.py``."""

from __future__ import annotations

import asyncio
import sys

from agent import BrowserUseAgent, BrowserUseConfig

from hud import Taskset


async def main(source: str) -> None:
    job = await Taskset.from_file(source).run(BrowserUseAgent(BrowserUseConfig()))
    for trace in job.traces:
        print(f"{trace.slug}: reward {trace.reward} ({trace.content!r})")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))

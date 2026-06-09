"""Tiny task lifecycle demo in one file.

Environment = hud.Environment with one @env.task.
Agent side = enter the concrete Task, read its prompt, write the answer to the Run.

Run:
    uv run python examples/00_agent_env.py
"""

from __future__ import annotations

import asyncio

import hud


env = hud.Environment("calculator")


@env.task()
async def add(a: int, b: int):
    answer = yield f"What is {a} + {b}? Reply with just the number."
    yield 1.0 if answer == str(a + b) else 0.0


async def main() -> None:
    task = add(a=3, b=4)
    async with task as run:
        print(run.prompt)
        run.trace.content = "7"
    print(f"reward={run.reward}")


if __name__ == "__main__":
    asyncio.run(main())

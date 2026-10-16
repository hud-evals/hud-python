"""One template per point a task can fail: its start, its grade, or a grade that never comes."""

from __future__ import annotations

import asyncio

from hud import Environment

env = Environment("faulty")


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
async def hang_grading():
    yield "go"
    await asyncio.Event().wait()
    yield 1.0

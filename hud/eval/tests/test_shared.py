"""Shared provider: refcount, width cap, and failed-provision cleanup."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from hud.eval import Shared, Task, Taskset
from hud.eval.runtime import Runtime


def _task() -> Task:
    return Task(env="demo", id="t", args={})


@asynccontextmanager
async def _ok_provider(_task: Task) -> Any:
    yield Runtime("tcp://127.0.0.1:9")


async def test_shared_cleans_stack_when_inner_provision_fails() -> None:
    """A failed first acquire must not leave a half-open exit stack behind."""
    calls = {"n": 0}

    @asynccontextmanager
    async def flaky(_task: Task) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boot failed")
        yield Runtime("tcp://127.0.0.1:9")

    shared = Shared(flaky, width=1)
    with pytest.raises(RuntimeError, match="boot failed"):
        async with shared(_task()):
            pass

    assert shared._stack is None
    assert shared._addr is None
    assert shared._refs == 0

    # A later acquire provisions cleanly on a fresh stack.
    async with shared(_task()) as addr:
        assert addr.url == "tcp://127.0.0.1:9"
        assert shared._refs == 1
    assert shared._refs == 0
    assert shared._stack is None


async def test_shared_rejects_more_than_width_concurrent() -> None:
    shared = Shared(_ok_provider, width=2)
    # Two holders + this test body rendezvous once both slots are taken.
    gate = asyncio.Barrier(3)
    release = asyncio.Event()

    async def hold() -> None:
        async with shared(_task()):
            await gate.wait()
            await release.wait()

    t1 = asyncio.create_task(hold())
    t2 = asyncio.create_task(hold())
    await asyncio.wait_for(gate.wait(), timeout=1)
    assert shared._refs == 2

    with pytest.raises(RuntimeError, match="already has 2 concurrent"):
        async with shared(_task()):
            pass

    release.set()
    await asyncio.gather(t1, t2)
    assert shared._refs == 0


async def test_taskset_run_requires_max_concurrent_equal_width() -> None:
    taskset = Taskset("demo", [_task()])
    agent = MagicMock()
    shared = Shared(_ok_provider, width=2)

    with pytest.raises(ValueError, match="requires max_concurrent=2"):
        await taskset.run(agent, runtime=shared, max_concurrent=4)

    with pytest.raises(ValueError, match="requires max_concurrent=2"):
        await taskset.run(agent, runtime=shared, max_concurrent=1)

    with pytest.raises(ValueError, match="requires max_concurrent=2"):
        await taskset.run(agent, runtime=shared, max_concurrent=None)

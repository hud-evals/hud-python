"""A rollout over a network that drops: the control connection is redialed and its session
resumed.

The env is served for real and reached through a relay that severs connections.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.clients import client
from hud.eval import LocalRuntime, Runtime, Task
from tests.fixtures.relay import relay

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run
    from tests.fixtures.relay import FlakyRelay

Phase = Literal["agent-running", "before-grading"]


@asynccontextmanager
async def relayed_runtime(
    env: Environment, task: Task
) -> AsyncIterator[tuple[Runtime, FlakyRelay]]:
    """``env`` served for ``task``, reached through a relay that can drop connections."""
    async with LocalRuntime(env)(task) as served:
        address = urlsplit(served.url)
        assert address.hostname is not None and address.port is not None
        async with relay(address.hostname, address.port) as flaky:
            yield Runtime(f"tcp://127.0.0.1:{flaky.port}"), flaky


@pytest.mark.parametrize("phase", ["agent-running", "before-grading"])
async def test_a_rollout_survives_the_control_connection_dropping(
    phase: Phase, monkeypatch: pytest.MonkeyPatch
) -> None:
    if phase == "agent-running":
        monkeypatch.setattr(client, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    calls: list[str] = []
    env = Environment("link")

    @env.template()
    async def work():
        calls.append("start")
        yield "go"
        calls.append("graded")
        yield 1.0

    task = Task(env="link", id="work")
    async with relayed_runtime(env, task) as (runtime, flaky):

        class DropsTheLink(Agent):
            async def __call__(self, run: Run) -> None:
                flaky.sever()
                if phase == "agent-running":
                    await flaky.wait_until(accepted=2)
                run.trace.content = "done"

        job = await asyncio.wait_for(task.run(DropsTheLink(), runtime=runtime), 30)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (1.0, "completed", None)
    assert calls == ["start", "graded"]


async def test_a_rollout_timeout_during_an_outage_cancels_the_task_on_the_redialed_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "_CONTROL_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    events: list[str] = []
    env = Environment("link")

    @env.template()
    async def work():
        try:
            yield "go"
            yield 1.0
        finally:
            events.append("task closed")

    task = Task(env="link", id="work")
    async with relayed_runtime(env, task) as (runtime, flaky):

        class OutlastsTheRedial(Agent):
            async def __call__(self, run: Run) -> None:
                flaky.stall()
                flaky.sever()
                await flaky.wait_until(stalled=1)
                asyncio.get_running_loop().call_later(4, flaky.release)
                await asyncio.sleep(30)

        job = await task.run(OutlastsTheRedial(), runtime=runtime, rollout_timeout=1)

        assert (job.runs[0].trace.stop_reason, events) == ("timeout", ["task closed"])

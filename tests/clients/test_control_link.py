"""A rollout over a network that drops: the control connection is redialed and its session
resumed, and a capability stream waits for the env to accept it.

The env is served for real and reached through a relay that severs and refuses connections.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import pytest
from fastmcp import FastMCP

from hud import Environment
from hud.agents.base import Agent
from hud.capabilities import MCPClient
from hud.clients import connect
from hud.eval import LocalRuntime, Runtime, Task
from tests.fixtures.asgi import mcp_server
from tests.fixtures.relay import relay

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.relay import FlakyRelay

Phase = Literal["agent-running", "before-grading", "starting", "grading"]


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


@pytest.mark.parametrize("phase", ["agent-running", "before-grading", "starting", "grading"])
async def test_a_rollout_survives_the_control_connection_dropping(
    phase: Phase, hud_env: HudEnv
) -> None:
    if phase == "agent-running":
        hud_env.set(HUD_CONTROL_HEARTBEAT_INTERVAL="0.05")
    calls: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    env = Environment("link")

    @env.template()
    async def work():
        calls.append("start")
        if phase == "starting":
            entered.set()
            await release.wait()
        yield "go"
        if phase == "grading":
            entered.set()
            await release.wait()
        calls.append("graded")
        yield 1.0

    task = Task(env="link", id="work")
    async with relayed_runtime(env, task) as (runtime, flaky):

        class DropsTheLink(Agent):
            async def __call__(self, run: Run) -> None:
                if phase in ("agent-running", "before-grading"):
                    flaky.sever()
                if phase == "agent-running":
                    await flaky.wait_until(accepted=2)
                run.trace.content = "done"

        rollout = asyncio.create_task(task.run(DropsTheLink(), runtime=runtime))
        if phase in ("starting", "grading"):
            await asyncio.wait_for(entered.wait(), 10)
            flaky.sever()
            await flaky.wait_until(accepted=2)
            release.set()
        job = await asyncio.wait_for(rollout, 30)

    (run,) = job.runs
    assert (run.reward, run.trace.status, run.trace.error) == (1.0, "completed", None)
    assert calls == ["start", "graded"]


def tools() -> FastMCP:
    server = FastMCP("tools")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


async def test_a_tool_call_waits_out_a_control_port_that_refuses_new_connections() -> None:
    async with mcp_server(tools()) as capability:
        env = Environment("tooled", capabilities=(capability,))

        @env.template()
        async def work():
            yield "go"
            yield 1.0

        task = Task(env="tooled", id="work")
        async with (
            relayed_runtime(env, task) as (runtime, flaky),
            connect(runtime) as client,
        ):
            mcp = await client.open("tools")
            assert isinstance(mcp, MCPClient)
            await mcp.call_tool("add", {"a": 2, "b": 3})
            flaky.sever()
            flaky.refuse(2)
            added = await mcp.call_tool("add", {"a": 2, "b": 3})

    assert added.structuredContent == {"result": 5}

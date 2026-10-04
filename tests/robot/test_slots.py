"""Sim slots: how sessions claim, step and free the lockstep slots of a robot sim.

A bridge in this process serves its control channel and its ``openpi/0``
WebSocket. Sessions claim and free slots the way a served env does: a client on
the env's control channel starts, grades, cancels or drops a task whose template
drives a ``RobotEndpoint``. Agents step slots the way a policy does
(``RobotClient``). Cases assert what the sim saw (its steps and ``result``
calls) and what each peer got.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient
from hud.clients import HudProtocolError
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from tests.serving import hud_client
from tests.serving import serving as served_on_loopback

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from numpy.typing import NDArray

    from hud.runtime import Runtime


class Sim(RobotBridge):
    """``num_envs`` lockstep slots of a one-dimensional action.

    Records every batched step and every ``result`` call; the first ``failures``
    calls to ``result`` raise, and every slot terminates after ``terminal_at`` steps.
    """

    def __init__(
        self,
        num_envs: int = 1,
        *,
        failures: int = 0,
        terminal_at: int | None = None,
        step_timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.failures = failures
        self.terminal_at = terminal_at
        self.step_timeout = step_timeout
        self.contract = {
            "control_rate": 10,
            "features": {"action": {"role": "action", "names": ["a"]}},
        }
        self.steps: list[list[float]] = []
        self.results = 0

    def reset(self, **task_args: Any) -> str:
        return f"prompt {sorted(task_args.items())}"

    def step(self, action: NDArray[Any]) -> None:
        self.steps.append(np.asarray(action).reshape(-1).tolist())

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        done = self.terminal_at is not None and len(self.steps) >= self.terminal_at
        tick = np.full((self.num_envs, 1), len(self.steps), dtype=np.float32)
        return {"tick": tick}, np.full(self.num_envs, done)

    def result(self) -> dict[str, Any]:
        self.results += 1
        if self.results <= self.failures:
            raise RuntimeError("the sim is busy")
        return {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"}


@asynccontextmanager
async def serving(sim: Sim) -> AsyncIterator[tuple[int, RobotEndpoint, Capability]]:
    """Serve ``sim``: its control port, a started endpoint on it, and the ``robot`` capability."""
    await sim.start()
    control = await sim.serve_control()
    port = control.sockets[0].getsockname()[1]
    endpoint = RobotEndpoint.remote("127.0.0.1", port)
    await endpoint.start()
    try:
        yield port, endpoint, Capability.robot(url=sim.url, contract=sim.contract)
    finally:
        await endpoint.stop()
        control.close()
        await sim.stop()


def slot_env(port: int) -> Environment:
    """An env whose ``episode`` claims a slot of the sim on *port* and grades it."""
    env = Environment("slots")
    endpoint = RobotEndpoint.remote("127.0.0.1", port).attach(env)

    @env.initialize
    async def connect_sim() -> None:
        await endpoint.start()

    @env.shutdown
    async def disconnect_sim() -> None:
        await endpoint.stop()

    @env.template()
    async def episode(task: str = "pick"):
        claimed = await endpoint.reset(task=task)
        yield claimed["prompt"]
        yield await endpoint.result()

    return env


def episodes(port: int) -> AbstractAsyncContextManager[Runtime]:
    """The ``slots`` env served on loopback for the block."""
    return served_on_loopback(slot_env(port))


PROMPT = "prompt [('task', 'pick')]"
GRADE = {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"}


@pytest.mark.parametrize(
    ("failures", "results"),
    [(0, 2), (1, 3)],
    ids=["first-release", "a-failed-release-is-retried"],
)
async def test_cancel_frees_the_slot_for_the_next_session(failures: int, results: int) -> None:
    sim = Sim(failures=failures)

    async with serving(sim) as (port, _, _), episodes(port) as runtime:
        async with hud_client(runtime) as first:
            await first.start_task("episode", {})
            await first.cancel()
        async with runtime.connect() as second:
            started = await asyncio.wait_for(second.start_task("episode", {}), timeout=10)
            assert await second.grade({"answer": "x"}) == GRADE

    assert started == {"prompt": PROMPT}
    assert sim.results == results


async def test_a_grade_frees_the_slot_with_the_sims_grade() -> None:
    sim = Sim()

    async with serving(sim) as (port, _, _), episodes(port) as runtime:
        async with runtime.connect() as first:
            await first.start_task("episode", {})
            graded = await first.grade({"answer": "x"})
        async with runtime.connect() as second:
            started = await asyncio.wait_for(second.start_task("episode", {}), timeout=10)
            await second.grade({"answer": "x"})

    assert graded == GRADE
    assert started == {"prompt": PROMPT}
    assert sim.results == 2


async def test_a_dropped_session_frees_its_slot_for_the_next_session() -> None:
    sim = Sim()

    async with serving(sim) as (port, _, _), episodes(port) as runtime:
        async with runtime.connect() as first:
            await first.start_task("episode", {})
        async with runtime.connect() as second:
            started = await asyncio.wait_for(second.start_task("episode", {}), timeout=10)
            await second.grade({"answer": "x"})

    assert started == {"prompt": PROMPT}
    assert sim.results == 2


async def test_a_session_waits_for_a_held_slot_until_its_holder_ends() -> None:
    sim = Sim()

    async with (
        serving(sim) as (port, _, _),
        episodes(port) as runtime,
        runtime.connect() as holder,
        hud_client(runtime) as waiter,
    ):
        await holder.start_task("episode", {})
        waiting = asyncio.create_task(waiter.start_task("episode", {}))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiting), timeout=0.2)
        await holder.grade({"answer": "x"})
        started = await asyncio.wait_for(waiting, timeout=10)
        await waiter.cancel()

    assert started == {"prompt": PROMPT}
    assert sim.results == 2


async def test_concurrent_sessions_with_different_args_cannot_share_a_batch() -> None:
    async with (
        serving(Sim(num_envs=2)) as (port, _, _),
        episodes(port) as runtime,
        runtime.connect() as first,
        runtime.connect() as second,
    ):
        started = await first.start_task("episode", {"task": "A"})
        with pytest.raises(HudProtocolError):
            await second.start_task("episode", {"task": "B"})

    assert started == {"prompt": "prompt [('task', 'A')]"}


# ─── stepping a batch ──────────────────────────────────────────────────


async def test_a_silent_slot_times_out_while_its_peer_keeps_stepping() -> None:
    sim = Sim(num_envs=2, step_timeout=0.05)

    async with serving(sim) as (_, endpoint, robot):
        tokens = [(await endpoint.reset())["token"] for _ in range(2)]
        active, silent = [
            await RobotClient.connect(replace(robot, params={**robot.params, "token": token}))
            for token in tokens
        ]
        try:
            await active.get_observation()
            await silent.get_observation()
            await active.send_action([1.0])
            stepped = await active.get_observation()
            with pytest.raises(RuntimeError, match="slot timed out waiting for an action"):
                await silent.get_observation()
            await active.send_action([2.0])
            stepped_again = await active.get_observation()
        finally:
            await active.close()
            await silent.close()

    assert (stepped["data"]["tick"].tolist(), stepped_again["data"]["tick"].tolist()) == (
        [1.0],
        [2.0],
    )
    assert sim.steps == [[1.0, 0.0], [2.0, 0.0]]


async def test_a_claim_that_never_dials_holds_the_batch_until_it_joins() -> None:
    sim = Sim(num_envs=2, step_timeout=0.5)

    async with serving(sim) as (_, endpoint, robot):
        tokens = [(await endpoint.reset())["token"] for _ in range(2)]
        early = await RobotClient.connect(
            replace(robot, params={**robot.params, "token": tokens[0]})
        )
        try:
            await early.get_observation()
            await early.send_action([1.0])
            # Longer than the step timeout: a claim still dialing is never timed out.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(early.get_observation(), timeout=1.0)
            steps_before_join = len(sim.steps)
            late = await RobotClient.connect(
                replace(robot, params={**robot.params, "token": tokens[1]})
            )
            try:
                await late.get_observation()
                await late.send_action([2.0])
                joined = await asyncio.wait_for(late.get_observation(), timeout=10)
            finally:
                await late.close()
        finally:
            await early.close()

    assert steps_before_join == 0
    assert joined["data"]["tick"].tolist() == [1.0]
    assert sim.steps == [[1.0, 2.0]]


async def test_a_terminated_slot_is_not_stepped_without_new_actions() -> None:
    sim = Sim(terminal_at=3)

    async with serving(sim) as (_, endpoint, robot):
        await endpoint.reset()
        agent = await RobotClient.connect(robot)
        try:
            await agent.get_observation()
            terminated = []
            for value in (1.0, 2.0, 3.0):
                await agent.send_action([value])
                terminated.append((await agent.get_observation())["terminated"])
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(agent.get_observation(), timeout=0.3)
        finally:
            await agent.close()

    assert terminated == [False, False, True]
    assert sim.steps == [[1.0], [2.0], [3.0]]

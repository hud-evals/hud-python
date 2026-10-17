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
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient
from hud.clients import HudProtocolError, connect
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from numpy.typing import NDArray

    from hud.eval import Runtime


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


class AsyncResetSim(Sim):
    """A bridge whose reset is a coroutine."""

    def __init__(self) -> None:
        super().__init__()

        async def reset(**task_args: Any) -> str:
            await asyncio.sleep(0)
            return f"async prompt {sorted(task_args.items())}"

        setattr(self, "reset", reset)


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
    return LocalRuntime(slot_env(port))(Task(env="slots", id="episode"))


PROMPT = "prompt [('task', 'pick')]"
GRADE = {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"}


@pytest.mark.parametrize(
    ("sim", "ending", "results_while_served", "results_after_stop", "restarts"),
    [
        pytest.param(Sim, "cancel", 2, 2, True, id="cancel-frees-the-slot"),
        pytest.param(Sim, "grade", 2, 2, True, id="grade-frees-the-slot-with-the-sims-grade"),
        pytest.param(Sim, "drop", 2, 2, True, id="a-dropped-session-frees-it-when-graded"),
        pytest.param(
            lambda: Sim(failures=1), "cancel", 3, 3, True, id="cancel-retries-a-failed-release"
        ),
        pytest.param(
            lambda: Sim(failures=3),
            "cancel",
            3,
            4,
            False,
            id="stopping-frees-what-cancel-could-not",
        ),
    ],
)
async def test_ending_a_session_frees_its_slot_exactly_once(
    sim: Callable[[], Sim],
    ending: str,
    results_while_served: int,
    results_after_stop: int,
    restarts: bool,
) -> None:
    bridge = sim()

    async with serving(bridge) as (port, peer, _):
        async with episodes(port) as runtime:
            async with connect(runtime) as first:
                assert await first.start_task("episode") == {"prompt": PROMPT}
                if ending == "cancel":
                    await first.cancel()
                elif ending == "grade":
                    assert await first.grade({"answer": "x"}) == GRADE
            async with connect(runtime) as second:
                if ending == "drop":
                    assert await second.grade({"answer": "x"}) == GRADE
                if restarts:
                    started = await asyncio.wait_for(second.start_task("episode"), timeout=10)
                    assert started == {"prompt": PROMPT}
                    assert await second.grade({"answer": "x"}) == GRADE
            while_served = bridge.results
        reclaimed = await asyncio.wait_for(peer.reset(task="pick"), timeout=10)

    assert (while_served, bridge.results) == (results_while_served, results_after_stop)
    assert reclaimed["prompt"] == PROMPT


async def test_a_session_waits_for_a_held_slot_until_its_holder_ends() -> None:
    sim = Sim()

    async with (
        serving(sim) as (port, _, _),
        episodes(port) as runtime,
        connect(runtime) as holder,
        connect(runtime) as waiter,
    ):
        await holder.start_task("episode")
        waiting = asyncio.create_task(waiter.start_task("episode"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiting), timeout=0.2)
        await holder.grade({"answer": "x"})
        started = await asyncio.wait_for(waiting, timeout=10)
        await waiter.cancel()

    assert started == {"prompt": PROMPT}
    assert sim.results == 2


async def test_an_async_reset_is_awaited_for_the_sessions_prompt() -> None:
    async with (
        serving(AsyncResetSim()) as (port, _, _),
        episodes(port) as runtime,
        connect(runtime) as session,
    ):
        started = await session.start_task("episode")
        await session.cancel()

    assert started == {"prompt": "async " + PROMPT}


async def test_concurrent_sessions_with_different_args_cannot_share_a_batch() -> None:
    async with (
        serving(Sim(num_envs=2)) as (port, _, _),
        episodes(port) as runtime,
        connect(runtime) as first,
        connect(runtime) as second,
    ):
        started = await first.start_task("episode", {"task": "A"})
        with pytest.raises(HudProtocolError):
            await second.start_task("episode", {"task": "B"})

    assert started == {"prompt": "prompt [('task', 'A')]"}


@pytest.mark.parametrize(
    ("num_envs", "token", "error"),
    [
        pytest.param(1, "slot-9-nope", "unknown episode token", id="an-unknown-token-is-refused"),
        pytest.param(2, None, "tokenless claim", id="a-vectorized-sim-refuses-a-tokenless-grade"),
    ],
)
async def test_a_grade_must_name_exactly_one_claimed_slot(
    num_envs: int, token: str | None, error: str
) -> None:
    sim = Sim(num_envs=num_envs)

    async with serving(sim) as (_, endpoint, _):
        claims = [await endpoint.reset(goal="lift") for _ in range(num_envs)]
        with pytest.raises(RuntimeError, match=error):
            await endpoint.result(token=token)

    assert len({claim["token"] for claim in claims}) == num_envs  # one token per slot


# ─── stepping a batch ──────────────────────────────────────────────────


async def test_a_silent_slot_times_out_while_its_peer_keeps_stepping() -> None:
    sim = Sim(num_envs=2, step_timeout=0.05)

    async with serving(sim) as (_, endpoint, robot):
        tokens = [(await endpoint.reset())["token"] for _ in range(2)]
        active, silent = [await RobotClient.connect(robot, token=token) for token in tokens]
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
        early = await RobotClient.connect(robot, token=tokens[0])
        try:
            await early.get_observation()
            await early.send_action([1.0])
            # Longer than the step timeout: a claim still dialing is never timed out.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(early.get_observation(), timeout=1.0)
            steps_before_join = len(sim.steps)
            late = await RobotClient.connect(robot, token=tokens[1])
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

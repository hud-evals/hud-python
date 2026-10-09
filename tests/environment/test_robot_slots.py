"""The robot control plane: how sessions claim, step and free a sim's slots.

A bridge in this process serves its JSON-RPC control channel and its ``openpi/0``
WebSocket. Scenarios claim slots the way an env does (``reset`` / ``result`` on
the control channel, or a template driving ``RobotEndpoint``), step them the way
an agent does (``RobotClient``), and assert what the sim saw: its steps, its
``result`` calls, and the errors each peer got.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task

from .conftest import Wire, wire

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray


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
    """A bridge written before hooks had to be synchronous: its reset is a coroutine."""

    def __init__(self) -> None:
        super().__init__()

        async def reset(**task_args: Any) -> str:
            await asyncio.sleep(0)
            return "async prompt"

        setattr(self, "reset", reset)


@asynccontextmanager
async def serving(sim: Sim) -> AsyncIterator[tuple[int, Capability]]:
    """Serve ``sim``: its control port and the ``robot`` capability an agent dials."""
    await sim.start()
    control = await sim.serve_control()
    try:
        yield (
            control.sockets[0].getsockname()[1],
            Capability.robot(url=sim.url, contract=sim.contract),
        )
    finally:
        control.close()
        await sim.stop()


async def claim(control: Wire, **task_args: Any) -> dict[str, Any]:
    return await control.call("reset", task_args)


@pytest.mark.parametrize(
    ("sim", "requests", "expected"),
    [
        pytest.param(
            Sim(),
            [("reset", {"goal": "a"}), ("reset", {"goal": "a"})],
            ["prompt [('goal', 'a')]", "all 1 slots are claimed"],
            id="a-full-batch-refuses-a-claim",
        ),
        pytest.param(
            Sim(num_envs=2),
            [("reset", {"task": "A"}), ("reset", {})],
            [
                "prompt [('task', 'A')]",
                "slots share one batch reset ({'task': 'A'}); a concurrent claim cannot use "
                "different task kwargs ({}) — group tasks with identical args, or use one sim "
                "per distinct task/seed",
            ],
            id="a-concurrent-claim-needs-identical-args",
        ),
        pytest.param(
            Sim(),
            [("reset", {}), ("result", {}), ("reset", {})],
            [
                "prompt []",
                {"score": 0.42, "success": True, "total_reward": 3.0, "detail": "custom"},
                "prompt []",
            ],
            id="result-frees-the-slot-with-the-bridges-grade",
        ),
        pytest.param(
            AsyncResetSim(), [("reset", {})], ["async prompt"], id="an-async-reset-is-awaited"
        ),
        pytest.param(
            Sim(),
            [("result", {"token": "slot-9-nope"})],
            ["unknown episode token: 'slot-9-nope'"],
            id="an-unknown-token-is-refused",
        ),
        pytest.param(Sim(), [("teleport", {})], ["unknown method 'teleport'"], id="unknown-method"),
    ],
)
async def test_the_control_channel_claims_and_frees_slots(
    sim: Sim, requests: list[tuple[str, dict[str, Any]]], expected: list[Any]
) -> None:
    seen: list[Any] = []

    async with serving(sim) as (port, _), wire(f"tcp://127.0.0.1:{port}") as control:
        for method, params in requests:
            reply = await control.call(method, params)
            if "error" in reply:
                seen.append(reply["error"]["message"])
            elif "prompt" in reply["result"]:
                seen.append(reply["result"]["prompt"])
            else:
                seen.append(reply["result"])

    assert seen == expected


async def test_an_endpoint_reset_waits_for_a_peer_to_free_the_slot() -> None:
    sim = Sim()

    async with serving(sim) as (port, _):
        endpoint = RobotEndpoint.remote("127.0.0.1", port)
        await endpoint.start()
        try:
            first = await endpoint.reset(goal="a")
            waiting = asyncio.create_task(endpoint.reset(goal="a"))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(waiting), timeout=0.2)
            await endpoint.result(token=first["token"])
            second = await asyncio.wait_for(waiting, timeout=10)
        finally:
            await endpoint.stop()

    assert second["token"] != first["token"]
    assert sim.results == 1


def slot_env(endpoint: RobotEndpoint) -> Environment:
    env = Environment("slots")
    endpoint.attach(env)

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


@pytest.mark.parametrize(
    ("failures", "ending", "results_while_served", "results_after_stop"),
    [
        pytest.param(0, "tasks.cancel", 2, 2, id="cancel-frees-the-slot"),
        pytest.param(0, "bye", 2, 2, id="bye-frees-the-slot"),
        pytest.param(0, "tasks.grade", 2, 2, id="grade-frees-the-slot-once"),
        pytest.param(0, "adopt", 2, 2, id="a-dropped-session-frees-it-when-graded"),
        pytest.param(1, "tasks.cancel", 3, 3, id="cancel-retries-a-failed-release"),
        pytest.param(3, "tasks.cancel", 3, 4, id="stopping-frees-what-cancel-could-not"),
    ],
)
async def test_ending_a_session_frees_its_slot_exactly_once(
    failures: int, ending: str, results_while_served: int, results_after_stop: int
) -> None:
    sim = Sim(failures=failures)

    async with serving(sim) as (port, _):
        env = slot_env(RobotEndpoint.remote("127.0.0.1", port))
        async with LocalRuntime(env)(Task(env="slots", id="episode")) as runtime:
            async with wire(runtime.url) as first:
                await first.call("tasks.start", {"id": "episode"})
                if ending == "adopt":
                    await first.hang_up()
                else:
                    await first.call(ending)
            async with wire(runtime.url) as second:
                if ending == "adopt":
                    await second.call("tasks.grade", {"answer": "x"})
                if results_while_served == results_after_stop:
                    started = await asyncio.wait_for(
                        second.call("tasks.start", {"id": "episode"}), timeout=10
                    )
                    graded = await second.call("tasks.grade", {"answer": "x"})
                    assert (started["result"], graded["result"]["score"]) == (
                        {"prompt": "prompt [('task', 'pick')]"},
                        0.42,
                    )
            while_served = sim.results
        async with wire(f"tcp://127.0.0.1:{port}") as control:
            reclaimed = await claim(control, task="pick")

    assert (while_served, sim.results) == (results_while_served, results_after_stop)
    assert reclaimed["result"]["prompt"] == "prompt [('task', 'pick')]"


async def test_concurrent_sessions_with_different_args_cannot_share_a_batch() -> None:
    sim = Sim(num_envs=2)

    async with serving(sim) as (port, _):
        env = slot_env(RobotEndpoint.remote("127.0.0.1", port))
        async with (
            LocalRuntime(env)(Task(env="slots", id="episode")) as runtime,
            wire(runtime.url) as first,
            wire(runtime.url) as second,
        ):
            started = await first.call("tasks.start", {"id": "episode", "args": {"task": "A"}})
            refused = await second.call("tasks.start", {"id": "episode", "args": {"task": "B"}})

    assert started["result"] == {"prompt": "prompt [('task', 'A')]"}
    assert refused["error"]["message"].startswith(
        "reset failed: slots share one batch reset ({'task': 'A'})"
    )


async def test_a_silent_slot_times_out_while_its_peer_keeps_stepping() -> None:
    sim = Sim(num_envs=2, step_timeout=0.05)

    async with serving(sim) as (port, robot), wire(f"tcp://127.0.0.1:{port}") as control:
        tokens = [(await claim(control))["result"]["token"] for _ in range(2)]
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

    async with serving(sim) as (port, robot), wire(f"tcp://127.0.0.1:{port}") as control:
        tokens = [(await claim(control))["result"]["token"] for _ in range(2)]
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

    async with serving(sim) as (port, robot), wire(f"tcp://127.0.0.1:{port}") as control:
        await claim(control)
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

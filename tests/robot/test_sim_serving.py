"""Serving a sim: what a sim process reports, and the shapes a gym env receives and returns.

``RobotEndpoint(bridge)`` and ``env.gym(factory)`` both run the sim in a child
process; a ``GymBridge`` can also be served in this one. These scenarios drive
the sims in ``sims`` through the public surface: the endpoint's control calls,
the served env's ``robot`` binding, and ``RobotClient`` on the ``openpi/0``
wire. A gym env reports in its observation the shape of the action it was
stepped with.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import websockets
from openpi_client import msgpack_numpy

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient
from hud.environment import Environment
from hud.environment.robot import GymBridge, RobotEndpoint
from tests.fixtures.environments import served

from .sims import Reach, plain_env, vector_env

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

CONTRACT = {"control_rate": 10, "features": {"action": {"role": "action", "names": ["a"]}}}


async def test_a_spawned_bridge_carries_its_declaration_into_the_child() -> None:
    declared = Reach(use_delta=True)
    declared.contract = CONTRACT
    declared.num_envs = 2
    declared.metadata = {"backend": "test"}
    endpoint = RobotEndpoint(declared)

    await endpoint.start()
    try:
        contract = await endpoint.contract()
        (robot,) = await endpoint.capabilities()
        first = await endpoint.reset()
        second = await endpoint.reset()
        async with websockets.connect(robot.url) as socket:
            greeting = msgpack_numpy.unpackb(await socket.recv())
    finally:
        await endpoint.stop()

    assert contract == CONTRACT
    assert robot.params["contract"] == CONTRACT
    assert first["prompt"] == "reach with use_delta=True"
    assert [first["token"][:7], second["token"][:7]] == ["slot-0-", "slot-1-"]
    assert greeting == {"backend": "test", "claim_required": True}
    with pytest.raises(OSError):
        await websockets.connect(robot.url, open_timeout=5)


async def test_a_sim_that_exits_before_announcing_its_port_fails_start() -> None:
    endpoint = RobotEndpoint.spawn([sys.executable, "-c", "import sys; sys.exit(3)"])

    with pytest.raises(RuntimeError, match="exited with code 3 before announcing its port"):
        await endpoint.start()


def gym_env(factory: Callable[..., Any], **defaults: Any) -> Environment:
    env = Environment("gym")
    sim = env.gym(factory, contract=None, **defaults)

    @env.template()
    async def episode():
        claimed = await sim.reset()
        yield {"prompt": claimed["prompt"], "bindings": {"robot": {"token": claimed["token"]}}}
        yield await sim.result(token=claimed["token"])

    return env


async def step_once(env: Environment, action: list[float]) -> tuple[dict[str, Any], Any]:
    """Start ``episode`` on *env*, play *action* once, and grade: the stepped observation
    and the grade."""
    async with served(env) as client:
        started = await client.start_task("episode")
        agent = await RobotClient.connect(
            client.binding("robot"), token=started["bindings"]["robot"]["token"]
        )
        try:
            await agent.get_observation()
            await agent.send_action(action)
            stepped = await agent.get_observation()
        finally:
            await agent.close()
        return stepped, await client.grade({"answer": "done"})


async def test_a_flat_action_reaches_a_box_action_space_in_its_own_shape() -> None:
    stepped, graded = await step_once(gym_env(plain_env), [0.1, 0.2, 0.3, 0.4])

    assert stepped["data"]["received"].tolist() == [2.0, 2.0]
    assert float(stepped["reward"]) == 0.5
    assert stepped["terminated"] is False
    assert graded == {"score": 0.5, "success": False, "total_reward": 0.5}


async def test_a_vector_env_built_from_kwargs_gets_one_action_row_per_env() -> None:
    stepped, graded = await step_once(gym_env(vector_env, num_envs=2), [0.1, 0.2])

    assert stepped["data"]["received"].tolist() == [2.0, 2.0]
    assert float(stepped["reward"]) == 0.25
    assert graded == {"score": 0.25, "success": False, "total_reward": 0.25}


async def test_a_plain_env_observation_is_sliced_to_one_slot_with_its_frames() -> None:
    async with served(gym_env(plain_env)) as client:
        started = await client.start_task("episode")
        agent = await RobotClient.connect(
            client.binding("robot"), token=started["bindings"]["robot"]["token"]
        )
        try:
            observation = await agent.get_observation()
        finally:
            await agent.close()
        await client.cancel()

    shapes = {name: np.shape(value) for name, value in observation["data"].items()}
    assert shapes == {"received": (1,), "camera": (4, 4, 3)}
    assert float(observation["reward"]) == 0.0


class VectorAutoReset:
    """Two lockstep envs that reset inside ``step``; env 1 finishes on the first step.

    ``per_env`` reports the final observation as gymnasium vector envs do (one obs per env),
    otherwise as Isaac Lab does (an obs-shaped batch).
    """

    num_envs = 2
    action_space = SimpleNamespace(shape=(2, 1), dtype=np.float32)

    def __init__(self, per_env: bool) -> None:
        self.per_env = per_env
        self.ticks = np.zeros(2)

    def observe(self) -> dict[str, NDArray[Any]]:
        return {"state": self.ticks[:, None].astype(np.float32)}

    def reset(self, seed: int | None = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        self.ticks[:] = 0
        return self.observe(), {}

    def step(self, action: NDArray[Any]) -> tuple[Any, ...]:
        self.ticks += 1
        final = self.observe()
        done = np.array([False, True])
        self.ticks[done] = 0
        if self.per_env:
            final = np.array([None, {"state": final["state"][1]}], dtype=object)
        return self.observe(), np.zeros(2), np.zeros(2, dtype=bool), done, {"final_obs": final}

    def close(self) -> None:
        pass


@pytest.mark.parametrize("per_env", [False, True], ids=["batched-final-obs", "per-env-final-obs"])
async def test_a_finished_env_is_observed_on_the_frame_it_ended_on(per_env: bool) -> None:
    bridge = GymBridge(lambda: VectorAutoReset(per_env), contract=None)
    await bridge.start()
    control = await bridge.serve_control()
    endpoint = RobotEndpoint.remote("127.0.0.1", control.sockets[0].getsockname()[1])
    await endpoint.start()
    try:
        robot = Capability.robot(url=bridge.url, contract=bridge.contract)
        tokens = [(await endpoint.reset())["token"] for _ in range(2)]
        agents = [await RobotClient.connect(robot, token=token) for token in tokens]
        try:
            for agent in agents:
                await agent.get_observation()
                await agent.send_action([0.0])
            observed = [await agent.get_observation() for agent in agents]
        finally:
            for agent in agents:
                await agent.close()
    finally:
        await endpoint.stop()
        control.close()
        await bridge.stop()

    assert [observation["terminated"] for observation in observed] == [False, True]
    # Env 1 was reset after its first tick; its observation is still that tick.
    assert [observation["data"]["state"].tolist() for observation in observed] == [[1.0], [1.0]]

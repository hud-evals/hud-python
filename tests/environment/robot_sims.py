"""The sims robot scenarios run, and the env that serves one in this process.

A spawned sim re-imports its bridge class or env factory from this file by path,
so they live at module level. Each gym env reports, in its observation, the
shape of the last action ``step`` got.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium.spaces import Box

from hud.environment import Environment
from hud.environment.robot import DirectControl, RobotBridge, RobotEndpoint

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray


@asynccontextmanager
async def sim_env(
    sim: RobotBridge, name: str, *, control: DirectControl | None = None
) -> AsyncIterator[tuple[Environment, RobotEndpoint]]:
    """The docs' custom-bridge env for *sim*, which this process serves, started.

    ``RobotEndpoint.remote`` attaches to a sim someone else runs; here that is
    the test, so the whole control plane is real without a spawned sim program.
    *control* adds direct control's motion tools. Each scenario declares the
    template it is about.
    """
    await sim.start()
    server = await sim.serve_control()
    env = Environment(name)
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)
    if control is not None:
        control.attach(endpoint)

    @env.initialize
    async def up() -> None:
        await endpoint.start()
        for capability in await endpoint.capabilities():
            env.add_capability(capability)

    @env.shutdown
    async def down() -> None:
        await endpoint.stop()

    try:
        await env.start()
        yield env, endpoint
    finally:
        await env.stop()
        server.close()
        await sim.stop()


class Reach(RobotBridge):
    """A bridge with a constructor parameter; its contract is set after construction."""

    def __init__(self, *, use_delta: bool = False) -> None:
        super().__init__()
        self.use_delta = use_delta

    def reset(self, **task_args: Any) -> str:
        return f"reach with use_delta={self.use_delta}"

    def step(self, action: NDArray[Any]) -> None:
        del action

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}, np.zeros(self.num_envs, bool)


class PlainEnv:
    """A single gym env with a ``Box(2, 2)`` action space."""

    action_space = Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)

    def __init__(self) -> None:
        self.received: tuple[int, ...] = ()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        return self.observation(), {}

    def step(self, action: NDArray[Any]) -> tuple[Any, ...]:
        self.received = np.shape(action)
        return self.observation(), 0.5, False, False, {}

    def observation(self) -> dict[str, NDArray[Any]]:
        return {
            "received": np.array(self.received or (0,), dtype=np.float32),
            "camera": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def close(self) -> None:
        pass


class VectorEnv:
    """``num_envs`` envs stepped together, exposing the per-env ``single_action_space``."""

    single_action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.action_space = Box(-1.0, 1.0, shape=(num_envs, 2), dtype=np.float32)
        self.received: tuple[int, ...] = ()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        return self.observation(), {}

    def step(self, action: NDArray[Any]) -> tuple[Any, ...]:
        self.received = np.shape(action)
        rewards = np.full(self.num_envs, 0.25)
        done = np.zeros(self.num_envs, dtype=bool)
        return self.observation(), rewards, done, done, {}

    def observation(self) -> dict[str, NDArray[Any]]:
        received = np.array(self.received or (0, 0), dtype=np.float32)
        return {"received": np.tile(received, (self.num_envs, 1))}

    def close(self) -> None:
        pass


def plain_env() -> PlainEnv:
    return PlainEnv()


def vector_env(**build: Any) -> VectorEnv:
    """A factory taking its build args as ``**kwargs``, like many registry wrappers."""
    return VectorEnv(int(build["num_envs"]))

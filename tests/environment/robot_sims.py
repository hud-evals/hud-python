"""Sims the robot scenarios spawn in a child process, so they live at module level.

A spawned sim re-imports its bridge class or env factory from this file by path.
Each env reports, in its observation, the shape of the last action ``step`` got.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium.spaces import Box

from hud.environment.robot import RobotBridge

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

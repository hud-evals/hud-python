"""GymBridge probes the env at start when no contract file exists."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

import gymnasium as gym
from gymnasium import spaces

from hud.environment.robot.gym import GymBridge


class _ToyEnv(gym.Env):
    """Minimal HWC camera + state env for contract probing."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8),
                "state": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = {
            "pixels": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.zeros(2, dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}


def make_toy_env() -> gym.Env:
    return _ToyEnv()


def test_start_probes_env_when_contract_missing(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    bridge = GymBridge(make_toy_env, contract=contract_path)

    assert not contract_path.exists()

    async def _run() -> None:
        await bridge.start()
        await bridge.stop()

    asyncio.run(_run())

    assert contract_path.exists()
    assert "pixels" in bridge.contract["features"]
    assert "state" in bridge.contract["features"]
    assert bridge.contract["features"]["pixels"]["type"] == "rgb"
    assert bridge.contract["control_rate"] == 10


def test_start_loads_existing_contract_without_rebuild(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        '{"control_rate": 7, "features": {"state": {"role": "observation", "names": ["s0"]}}}\n'
    )
    builds: list[int] = []

    def make_counting_env() -> gym.Env:
        builds.append(1)
        return _ToyEnv()

    bridge = GymBridge(make_counting_env, contract=contract_path)

    async def _run() -> None:
        await bridge.start()
        await bridge.stop()

    asyncio.run(_run())

    assert builds == []  # file present → no probe build
    assert bridge.contract["control_rate"] == 7

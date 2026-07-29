"""Regressions for GymBridge action_dim, factory build params, reshape, and obs batching."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hud.environment.robot.gym import GymBridge, action_dim_of


def test_action_dim_of_keeps_single_action_space_shape() -> None:
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(shape=(2, 2)),
        action_space=SimpleNamespace(shape=(4, 2, 2)),
    )
    assert action_dim_of(env, batched=True) == 4


def test_action_dim_of_strips_batched_action_space_without_single() -> None:
    env = SimpleNamespace(
        single_action_space=None,
        action_space=SimpleNamespace(shape=(4, 2, 2)),
    )
    assert action_dim_of(env, batched=True) == 4


def test_var_keyword_factory_defaults_are_build_params() -> None:
    def make_env(**kwargs: Any) -> Any:
        return kwargs

    bridge = GymBridge(make_env, contract=None, num_envs=8)
    assert "num_envs" in bridge._build_params


def test_step_reshapes_float_box_action() -> None:
    class _FakeEnv:
        action_space = SimpleNamespace(shape=(2, 2), dtype=np.float32)

        def __init__(self) -> None:
            self.last_action: Any = None

        def step(self, action: Any) -> tuple:
            self.last_action = action
            return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = _FakeEnv()
    bridge.batched = False
    bridge.num_envs = 1
    bridge._done = np.zeros(1, dtype=bool)
    bridge._success = np.zeros(1, dtype=bool)
    bridge._acc_reward = np.zeros(1)
    bridge._step_reward = np.zeros(1)
    bridge.step(np.arange(4, dtype=np.float32).reshape(1, 4))
    assert bridge.env.last_action.shape == (2, 2)


def test_step_reshapes_batched_float_box_action() -> None:
    class _FakeEnv:
        action_space = SimpleNamespace(shape=(2, 2, 2), dtype=np.float32)

        def __init__(self) -> None:
            self.last_action: Any = None

        def step(self, action: Any) -> tuple:
            self.last_action = action
            return (
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(2),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=bool),
                {},
            )

    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = _FakeEnv()
    bridge.batched = True
    bridge.num_envs = 2
    bridge._done = np.zeros(2, dtype=bool)
    bridge._success = np.zeros(2, dtype=bool)
    bridge._acc_reward = np.zeros(2)
    bridge._step_reward = np.zeros(2)
    bridge.step(np.arange(8, dtype=np.float32).reshape(2, 4))
    assert bridge.env.last_action.shape == (2, 2, 2)


def test_plain_env_observation_always_gets_batch_axis() -> None:
    bridge = GymBridge(lambda: None, contract=None)
    bridge.env = object()
    bridge.batched = False
    bridge.num_envs = 1
    bridge._done = np.zeros(1, dtype=bool)
    bridge._step_reward = np.array([0.5], dtype=np.float32)
    bridge._obs = {
        "state": np.array([1.0], dtype=np.float32),
        "camera": np.zeros((1, 4, 3), dtype=np.uint8),
    }
    data, terminated = bridge.get_observation()  # type: ignore[misc]
    assert data is not None
    assert data["state"].shape == (1, 1)
    assert data["camera"].shape == (1, 1, 4, 3)
    assert data["reward"].shape == (1,)
    assert data["reward"][0] == pytest.approx(0.5)
    assert terminated.shape == (1,)

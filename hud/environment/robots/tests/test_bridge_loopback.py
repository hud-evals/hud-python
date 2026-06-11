"""End-to-end loopback: RobotBridge env <-> RobotAgent over a real WebSocket.

A stub counter sim is served by a real :class:`RobotBridge`, published as a
``robot`` capability on a real :class:`Environment` behind a
:class:`LocalSandbox`, and driven by a :class:`RobotAgent` subclass with a stub
model — the full agent-side path (manifest -> binding -> RobotClient ->
observe/act loop -> grade).
"""

from __future__ import annotations

import socket
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pytest

from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model
from hud.capabilities.base import Capability
from hud.capabilities.robot import RobotClient
from hud.client.client import HudClient
from hud.environment import Environment
from hud.environment.robots.bridge import RobotBridge
from hud.eval.sandbox import LocalSandbox

CONTRACT: dict[str, Any] = {
    "robot_type": "counter_bot",
    "control_rate": 10,
    "features": {
        "state": {"role": "observation", "dtype": "float32", "shape": [2]},
        "action": {"role": "action", "dtype": "float32", "shape": [2]},
    },
}


def _free_port() -> int:
    # The bridge constructor takes a fixed port (no bind-to-0 support), so pick one.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CounterBridge(RobotBridge):
    """Trivial sim: state = [count, 42]; terminates (successfully) after n_steps."""

    def __init__(self, *, port: int, n_steps: int = 5) -> None:
        super().__init__(host="localhost", port=port)
        self._n_steps = n_steps
        self.count = 0
        self.actions: list[np.ndarray] = []

    async def reset(self, **kwargs: Any) -> str:
        self.count = 0
        self.actions = []
        self.task_description = f"count to {self._n_steps}"
        self.total_reward = 0.0
        self.success = False
        self.terminated = False
        await self._send_observation()
        return self.task_description

    def step(self, action: np.ndarray) -> None:
        self.actions.append(np.array(action, copy=True))
        self.count += 1
        self.last_reward = 1.0
        self.total_reward += 1.0
        if self.count >= self._n_steps:
            self.terminated = True
            self.success = True

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        return {"state": np.array([self.count, 42.0], dtype=np.float32)}, self.terminated


class EchoCountModel(Model):
    """Stub policy: action = [observed count, 1] — proves obs decoding end-to-end."""

    def __init__(self) -> None:
        self.observed_states: list[np.ndarray] = []

    def infer(self, batch: dict[str, Any]) -> np.ndarray:
        state = batch["data"]["state"]
        self.observed_states.append(np.array(state, copy=True))
        return np.array([state[0], 1.0], dtype=np.float32)


class StubAgent(RobotAgent):
    log_every = 0

    def __init__(self, model: Model) -> None:
        self.model = model
        self.adapter = None  # raw pass-through: obs dict straight into the model


@pytest.fixture
def bridge() -> CounterBridge:
    return CounterBridge(port=_free_port(), n_steps=5)


def _make_env(bridge: CounterBridge) -> Environment:
    env = Environment(
        "counter-env",
        capabilities=[Capability.robot(url=bridge.url, contract=CONTRACT)],
    )

    @env.task(id="count")
    async def count_task():
        prompt = await bridge.reset()
        yield {"prompt": prompt}
        yield bridge.result()

    env.initialize(bridge.start)
    env.shutdown(bridge.stop)
    return env


async def test_full_loopback_episode(bridge: CounterBridge) -> None:
    env = _make_env(bridge)
    model = EchoCountModel()
    agent = StubAgent(model)

    async with LocalSandbox(env) as runtime:
        parts = urlsplit(runtime.url)
        assert parts.hostname is not None and parts.port is not None
        async with await HudClient.connect(parts.hostname, parts.port) as client:
            async with client.task("count") as run:
                assert run.prompt == "count to 5"
                await agent(run)
            # Grading reflects bridge success.
            assert run.reward == 1.0
            assert run.evaluation["success"] is True
            assert run.evaluation["total_reward"] == 5.0

    # The agent saw each decoded observation in order (count 0..4)...
    assert [float(s[0]) for s in model.observed_states] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(float(s[1]) == 42.0 for s in model.observed_states)
    # ...and every action arrived at the bridge intact (action[0] echoes the count).
    assert len(bridge.actions) == 5
    for i, action in enumerate(bridge.actions):
        np.testing.assert_allclose(action, [float(i), 1.0])


async def test_loopback_observation_decode_via_raw_client(bridge: CounterBridge) -> None:
    """Dial the bridge directly with RobotClient and check the decoded frames."""
    await bridge.start()
    try:
        await bridge.reset()
        cap = Capability.robot(url=bridge.url, contract=CONTRACT)
        client = await RobotClient.connect(cap)
        try:
            obs = await client.get_observation()
            assert obs["terminated"] is False
            assert "meta" not in obs  # sync bridges attach no realtime meta
            np.testing.assert_allclose(obs["data"]["state"], [0.0, 42.0])
            assert obs["data"]["state"].dtype == np.float32

            await client.send_action(np.array([0.5, -0.5], dtype=np.float32))
            obs2 = await client.get_observation()
            np.testing.assert_allclose(obs2["data"]["state"], [1.0, 42.0])
            np.testing.assert_allclose(bridge.actions[0], [0.5, -0.5])
        finally:
            await client.close()
    finally:
        await bridge.stop()


async def test_client_spaces_splits_features_by_role() -> None:
    contract = {
        "robot_type": "x",
        "features": {
            "cam": {"role": "observation", "dtype": "image", "shape": [8, 8, 3]},
            "state": {"role": "observation", "dtype": "float32", "shape": [3]},
            "action": {"role": "action", "dtype": "float32", "shape": [7]},
        },
    }
    cap = Capability.robot(url="ws://localhost:1", contract=contract)

    class _ClosedWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self) -> None:
            pass

    client = RobotClient(cap, _ClosedWS())
    try:
        action, observations = client.spaces()
        assert action == contract["features"]["action"]
        assert list(observations) == ["cam", "state"]
        assert observations["cam"]["dtype"] == "image"
        assert client.contract["robot_type"] == "x"
    finally:
        await client.close()

"""Socket-free unit tests for the robot agent harness (adapter / model / agents)."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from hud.agents.robot.adapter import DefaultAdapter
from hud.agents.robot.agent import ROBOT_PROTOCOL, RobotAgent
from hud.agents.robot.model import STEP_COUNTER, Model
from hud.agents.robot.realtime import RealtimeRobotAgent

# ── DefaultAdapter.bind ───────────────────────────────────────────────────────

ACTION_SPACE = {"role": "action", "dtype": "float32", "shape": [7]}
OBS_SPACE = {
    "agentview": {"role": "observation", "dtype": "image", "shape": [64, 64, 3]},
    "wrist": {"role": "observation", "dtype": "image", "shape": [64, 64, 3]},
    "proprio": {"role": "observation", "dtype": "float32", "shape": [8]},
}


def test_default_adapter_bind_splits_spaces() -> None:
    adapter = DefaultAdapter(model_image_keys=["observation.images.image"])
    adapter.bind(ACTION_SPACE, OBS_SPACE)
    assert adapter.action_space == ACTION_SPACE
    assert adapter.image_keys == ["agentview", "wrist"]  # ordered, images only
    assert adapter.state_key == "proprio"  # the single non-image feature


def test_default_adapter_bind_handles_missing_state() -> None:
    adapter = DefaultAdapter()
    adapter.bind({}, {"cam": {"dtype": "image", "shape": [8, 8, 3]}})
    assert adapter.image_keys == ["cam"]
    assert adapter.state_key is None
    assert adapter.action_space == {}


def test_default_adapter_adapt_action_is_identity() -> None:
    adapter = DefaultAdapter()
    action = np.array([1.0, 2.0], dtype=np.float32)
    assert adapter.adapt_action(action, obs={}) is action


# ── Model.ainfer ──────────────────────────────────────────────────────────────


class ThreadProbeModel(Model):
    def __init__(self) -> None:
        self.infer_thread: int | None = None
        self.batches: list[Any] = []

    def infer(self, batch: Any) -> np.ndarray:
        self.infer_thread = threading.get_ident()
        self.batches.append(batch)
        return np.array([1.0], dtype=np.float32)


async def test_ainfer_runs_infer_off_loop_and_counts_steps() -> None:
    model = ThreadProbeModel()
    STEP_COUNTER.reset()

    out = await model.ainfer({"x": 1})
    np.testing.assert_array_equal(out, [1.0])
    assert model.batches == [{"x": 1}]
    # asyncio.to_thread: infer must run on a worker thread, not the loop thread.
    assert model.infer_thread is not None
    assert model.infer_thread != threading.get_ident()
    assert STEP_COUNTER.count == 1

    await model.ainfer({"x": 2})
    assert STEP_COUNTER.count == 2
    STEP_COUNTER.reset()
    assert STEP_COUNTER.count == 0


def test_base_model_infer_is_abstract_by_convention() -> None:
    with pytest.raises(NotImplementedError):
        Model().infer({})


# ── RobotAgent ────────────────────────────────────────────────────────────────


async def test_select_action_raises_without_model() -> None:
    agent = RobotAgent()
    assert agent.model is None
    with pytest.raises(RuntimeError, match=r"must set self\.model"):
        await agent.select_action({"data": {}})


async def test_select_action_passthrough_without_adapter() -> None:
    agent = RobotAgent()
    agent.model = ThreadProbeModel()
    agent.adapter = None
    obs = {"data": {"state": np.zeros(2)}, "terminated": False}
    out = await agent.select_action(obs)
    np.testing.assert_array_equal(out, [1.0])
    assert agent.model.batches == [obs]  # raw obs handed straight to the model


def test_should_stop_reads_terminated() -> None:
    agent = RobotAgent()
    assert agent.should_stop({"terminated": True}, step=0, max_steps=10) is True
    assert agent.should_stop({"terminated": False}, step=0, max_steps=10) is False
    assert agent.should_stop({}, step=0, max_steps=10) is False


def test_robot_protocol_constant() -> None:
    assert ROBOT_PROTOCOL == "robot"
    assert RobotAgent.robot_protocol == "robot"


# ── RealtimeRobotAgent._model_prefix ──────────────────────────────────────────


class StubRealtimeAgent(RealtimeRobotAgent):
    def infer_chunk(
        self, obs: dict[str, Any], meta: dict[str, Any], prefix_model: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError  # not exercised by these tests


def _rtc_agent(*, chunk_len: int = 8, sent_at: int = 10) -> StubRealtimeAgent:
    agent = StubRealtimeAgent()
    agent._rtc = True
    agent._last_raw_chunk = np.arange(chunk_len * 2, dtype=np.float32).reshape(chunk_len, 2)
    agent._last_chunk_obs_index = sent_at
    return agent


def test_model_prefix_slices_consumed_ticks_off_the_tail() -> None:
    agent = _rtc_agent(chunk_len=8, sent_at=10)
    # 3 ticks elapsed since the chunk's obs -> tail is chunk[3:].
    prefix = agent._model_prefix(13)
    assert prefix is not None
    np.testing.assert_array_equal(prefix, agent._last_raw_chunk[3:])


def test_model_prefix_full_chunk_when_no_ticks_elapsed() -> None:
    agent = _rtc_agent(sent_at=10)
    np.testing.assert_array_equal(agent._model_prefix(10), agent._last_raw_chunk)
    # obs_index < last_chunk_obs_index clamps to k=0 (never a negative slice).
    np.testing.assert_array_equal(agent._model_prefix(7), agent._last_raw_chunk)


def test_model_prefix_none_when_fully_consumed() -> None:
    agent = _rtc_agent(chunk_len=8, sent_at=10)
    assert agent._model_prefix(18) is None  # k == len(chunk): empty tail
    assert agent._model_prefix(50) is None


def test_model_prefix_none_outside_rtc_or_before_first_chunk() -> None:
    agent = _rtc_agent()
    assert agent._model_prefix(None) is None  # no obs_index on the frame

    agent._rtc = False
    assert agent._model_prefix(12) is None  # non-RTC mode

    agent = StubRealtimeAgent()
    agent._rtc = True
    agent._last_raw_chunk = None
    agent._last_chunk_obs_index = None
    assert agent._model_prefix(12) is None  # before the first inference


async def test_realtime_select_action_is_disabled() -> None:
    agent = StubRealtimeAgent()
    with pytest.raises(NotImplementedError, match="infer_chunk"):
        await agent.select_action({})

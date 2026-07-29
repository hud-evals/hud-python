"""Regressions for robot bridge barrier, claim wait, and episode kwargs."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import numpy as np
import pytest

from hud.environment.robot.bridge import RobotBridge


class _StubBridge(RobotBridge):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.steps: list[np.ndarray] = []
        self.contract = {"features": {"action": {"role": "action", "names": ["a"]}}}

    def reset(self, **kwargs: Any) -> str:
        return f"prompt:{kwargs!r}"

    def step(self, action: np.ndarray) -> None:
        self.steps.append(np.asarray(action))

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}, np.zeros(
            self.num_envs, dtype=bool
        )


class _FakeWS:
    async def send(self, _data: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_claim_rejects_empty_kwargs_after_nonempty_batch() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 2
    first = await bridge._claim_episode(task="A")
    assert first["token"]
    with pytest.raises(ValueError, match="identical args"):
        await bridge._claim_episode()


@pytest.mark.asyncio
async def test_tick_loop_does_not_hold_spin_when_all_claimed_idle() -> None:
    """Terminated slots may keep WS open until close(); barrier must not step."""
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    slot = bridge._registry.slots[0]
    bridge._registry.claim(slot)
    slot.ws = _FakeWS()  # still connected
    slot.idle = True  # terminated
    slot.action = None

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert bridge.steps == []


@pytest.mark.asyncio
async def test_tick_loop_stops_after_terminal_obs_while_ws_still_open() -> None:
    """Terminate wakes the barrier; idle+open WS must not cascade hold-steps.

    The e2e gate ``sim.tick == EPISODE_TICKS`` fails if each terminal fan-out
    re-arms the loop into a hold-spin before the agent closes the socket.
    """

    class _Terminating(_StubBridge):
        def __init__(self) -> None:
            super().__init__()
            self.tick = 0

        def step(self, action: np.ndarray) -> None:
            self.tick += 1
            super().step(action)

        def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
            data = {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}
            return data, np.array([self.tick >= 3])

    bridge = _Terminating()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    slot = bridge._registry.slots[0]
    bridge._registry.claim(slot)
    slot.ws = _FakeWS()
    task = asyncio.create_task(bridge._tick_loop())
    try:
        for expected in (1, 2, 3):
            slot.idle = False
            slot.action = np.array([1.0], dtype=np.float32)
            bridge._action_event.set()
            for _ in range(50):
                if bridge.tick >= expected:
                    break
                await asyncio.sleep(0.01)
            assert bridge.tick == expected
        assert slot.idle  # terminal obs dropped the slot out of the barrier
        for _ in range(20):  # spam the wake that terminal fan-out itself sets
            bridge._action_event.set()
            await asyncio.sleep(0.005)
        assert bridge.tick == 3
        assert len(bridge.steps) == 3
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tick_loop_steps_when_live_slot_has_action() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    slot = bridge._registry.slots[0]
    bridge._registry.claim(slot)
    slot.ws = _FakeWS()
    slot.idle = False
    slot.action = np.array([1.0], dtype=np.float32)

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    for _ in range(50):
        if bridge.steps:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert len(bridge.steps) == 1
    assert bridge.steps[0].shape == (1, 1)


@pytest.mark.asyncio
async def test_claim_waits_when_batch_full_then_succeeds_after_release() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    first = await bridge._claim_episode(goal="a")
    assert first["token"]

    claim = asyncio.create_task(bridge._claim_episode(goal="a"))
    await asyncio.sleep(0.05)
    assert not claim.done()  # waiting, not erroring
    await bridge._release_episode(first["token"])
    second = await asyncio.wait_for(claim, timeout=1.0)
    assert second["token"]


@pytest.mark.asyncio
async def test_tick_loop_times_out_silent_live_slot() -> None:
    """A connected agent that never sends must not stall the barrier forever."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    bridge._registry.claim(a)
    bridge._registry.claim(b)
    a.ws, b.ws = _FakeWS(), _FakeWS()
    a.idle = b.idle = False
    a.action = np.array([1.0], dtype=np.float32)  # ready
    b.action = None  # silent

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    for _ in range(50):
        if bridge.steps:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert len(bridge.steps) == 1
    assert b.idle  # timed out


@pytest.mark.asyncio
async def test_tick_loop_does_not_hold_step_undialed_claimed_slot() -> None:
    """A peer that has claimed but not WS-connected must not be hold-advanced."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    bridge._registry.claim(a)
    bridge._registry.claim(b)
    a.ws = _FakeWS()
    a.idle = b.idle = False
    a.action = np.array([1.0], dtype=np.float32)
    # b: claimed, still dialing (ws is None)

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert bridge.steps == []
    assert not b.idle  # still waiting to dial — not timed out into hold

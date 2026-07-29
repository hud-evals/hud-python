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

"""Regressions for robot bridge barrier, claim wait, and episode kwargs."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import numpy as np
import pytest

from hud.environment.robot.bridge import RobotBridge, _SlotPhase


class _StubBridge(RobotBridge):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.steps: list[np.ndarray] = []
        self.stepped = asyncio.Event()
        self.contract = {"features": {"action": {"role": "action", "names": ["a"]}}}

    def reset(self, **kwargs: Any) -> str:
        return f"prompt:{kwargs!r}"

    def step(self, action: np.ndarray) -> None:
        self.steps.append(np.asarray(action))
        self.stepped.set()

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        return {"x": np.zeros((self.num_envs, 1), dtype=np.float32)}, np.zeros(
            self.num_envs, dtype=bool
        )


class _FakeWS:
    async def send(self, _data: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_claim_awaits_legacy_async_reset() -> None:
    class _AsyncReset(_StubBridge):
        async def reset(self, **kwargs: Any) -> str:
            await asyncio.sleep(0)
            return "async-prompt"

    ep = await _AsyncReset()._claim_episode()
    assert ep["prompt"] == "async-prompt"


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
    bridge._registry.claim(slot, dial_deadline=float("inf"))
    slot.ws = _FakeWS()  # still connected
    slot.phase = _SlotPhase.IDLE
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
    bridge._registry.claim(slot, dial_deadline=float("inf"))
    slot.ws = _FakeWS()
    slot.phase = _SlotPhase.ACTIVE
    task = asyncio.create_task(bridge._tick_loop())
    try:
        for expected in (1, 2, 3):
            slot.phase = _SlotPhase.ACTIVE
            slot.action = np.array([1.0], dtype=np.float32)
            bridge._action_event.set()
            for _ in range(50):
                if bridge.tick >= expected:
                    break
                await asyncio.sleep(0.01)
            assert bridge.tick == expected
        assert slot.phase is _SlotPhase.TERMINATED
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
    bridge._registry.claim(slot, dial_deadline=float("inf"))
    slot.ws = _FakeWS()
    slot.phase = _SlotPhase.ACTIVE
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
async def test_claim_raises_when_batch_full() -> None:
    bridge = _StubBridge()
    bridge.num_envs = 1
    bridge._registry.configure(1)
    first = await bridge._claim_episode(goal="a")
    with pytest.raises(RuntimeError, match="slots are claimed"):
        await bridge._claim_episode(goal="a")
    await bridge._release_episode(first["token"])
    second = await bridge._claim_episode(goal="a")
    assert second["token"]


@pytest.mark.asyncio
async def test_endpoint_reset_retries_until_peer_result_frees_slot() -> None:
    """Shared width+1: reset retries outside the RPC lock so result is not deadlocked."""
    from hud.environment.robot.endpoint import RobotEndpoint

    bridge = _StubBridge()
    bridge.num_envs = 1
    server = await bridge.serve_control("127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ep = RobotEndpoint.remote("127.0.0.1", port)
    await ep.start()
    try:
        first = await ep.reset(goal="a")
        waiting = asyncio.create_task(ep.reset(goal="a"))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        await asyncio.wait_for(ep.result(token=first["token"]), timeout=1.0)
        second = await asyncio.wait_for(waiting, timeout=1.0)
        assert second["token"] != first["token"]
    finally:
        await ep.stop()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tick_loop_waits_for_silent_live_slot() -> None:
    """A connected policy may take longer than the initial dial timeout."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    bridge._registry.claim(a, dial_deadline=float("inf"))
    bridge._registry.claim(b, dial_deadline=float("inf"))
    a.ws, b.ws = _FakeWS(), _FakeWS()
    a.phase = b.phase = _SlotPhase.ACTIVE
    a.action = np.array([1.0], dtype=np.float32)  # ready
    b.action = None  # silent

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    try:
        await asyncio.sleep(bridge.step_timeout * 3)
        assert bridge.steps == []
        assert a.phase is _SlotPhase.ACTIVE
        assert b.phase is _SlotPhase.ACTIVE

        b.action = np.array([2.0], dtype=np.float32)
        bridge._action_event.set()
        for _ in range(50):
            if bridge.steps:
                break
            await asyncio.sleep(0.01)
        assert len(bridge.steps) == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_tick_loop_expires_undialed_claim_and_steps_live_peer() -> None:
    """A never-connected claim stops blocking peers at its initial dial deadline."""
    bridge = _StubBridge()
    bridge.step_timeout = 0.05
    bridge.num_envs = 2
    bridge._registry.configure(2)
    a, b = bridge._registry.slots
    loop = asyncio.get_running_loop()
    bridge._registry.claim(a, dial_deadline=float("inf"))
    dial_deadline = loop.time() + bridge.step_timeout
    bridge._registry.claim(b, dial_deadline=dial_deadline)
    a.ws = _FakeWS()
    a.phase = _SlotPhase.ACTIVE
    a.action = np.array([1.0], dtype=np.float32)

    task = asyncio.create_task(bridge._tick_loop())
    bridge._action_event.set()
    try:
        await asyncio.wait_for(bridge.stepped.wait(), timeout=1.0)
        assert len(bridge.steps) == 1
        assert b.phase is _SlotPhase.EXPIRED
        np.testing.assert_allclose(bridge.steps[0], [[1.0], [0.0]])
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

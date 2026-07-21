"""Slot-token contract on the bridge control plane (reset/result)."""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("numpy")
pytest.importorskip("openpi_client")

from hud.environment.robot.bridge import RobotBridge
from hud.environment.robot.gym import GymBridge


class StubBridge(RobotBridge):
    """Minimal concrete bridge: no sim, fixed prompt."""

    def __init__(self, num_envs: int = 1) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.steps: list[np.ndarray] = []

    def reset(self, **kwargs: Any) -> str:
        return "do the task"

    def step(self, action: Any) -> None:
        self.steps.append(np.asarray(action).copy())

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        n = self.num_envs
        return {"state": np.zeros((n, 1), dtype=np.float32)}, np.zeros(n, dtype=bool)


async def test_claim_episode_hops_reset_to_sim_thread() -> None:
    """Control-plane reset must run on the sim thread, not the serve loop."""
    reset_ident: list[int] = []

    class Tracking(StubBridge):
        def reset(self, **kwargs: Any) -> str:
            reset_ident.append(threading.get_ident())
            return "do the task"

    bridge = Tracking()
    stop = threading.Event()
    ready = threading.Event()

    def drain() -> None:
        # This thread is the sim (same role as main under serve_bridge).
        bridge._sim_ident = threading.get_ident()
        ready.set()
        while not stop.is_set():
            try:
                fn, fut = bridge._sim_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as exc:
                    fut.set_exception(exc)

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    ready.wait(timeout=2)
    try:
        ep = await bridge._dispatch_control("reset", {})
        assert ep["prompt"] == "do the task"
        assert reset_ident == [bridge._sim_ident]
    finally:
        stop.set()
        worker.join(timeout=2)


async def test_result_without_token_grades_the_single_slot() -> None:
    bridge = StubBridge()
    ep = await bridge._dispatch_control("reset", {})
    assert ep["prompt"] == "do the task"
    assert isinstance(ep["token"], str)

    grade = await bridge._dispatch_control("result", {})
    assert {"score", "success", "total_reward"} <= set(grade)

    # The slot was freed: a new episode claims cleanly.
    await bridge._dispatch_control("reset", {})


async def test_result_with_token_still_resolves_its_slot() -> None:
    bridge = StubBridge()
    ep = await bridge._dispatch_control("reset", {})
    grade = await bridge._dispatch_control("result", {"token": ep["token"]})
    assert {"score", "success", "total_reward"} <= set(grade)


async def test_tokenless_result_rejects_ambiguous_slots() -> None:
    bridge = StubBridge(num_envs=2)
    await bridge._dispatch_control("reset", {})
    await bridge._dispatch_control("reset", {})
    with pytest.raises(ValueError, match="exactly one claimed slot"):
        await bridge._dispatch_control("result", {})


async def test_tokenless_result_rejects_no_claimed_slot() -> None:
    bridge = StubBridge()
    with pytest.raises(ValueError, match="exactly one claimed slot"):
        await bridge._dispatch_control("result", {})


async def test_unknown_token_still_errors() -> None:
    bridge = StubBridge()
    await bridge._dispatch_control("reset", {})
    with pytest.raises(ValueError, match="unknown episode token"):
        await bridge._dispatch_control("result", {"token": "slot-0-deadbeef"})


async def test_tick_loop_waits_for_connected_agent_past_step_timeout() -> None:
    """A live connection mid-inference must not be stepped with holds after step_timeout."""
    bridge = StubBridge()
    bridge.step_timeout = 0.05
    bridge.contract = {"features": {"action": {"role": "action", "names": ["a"]}}}
    await bridge._dispatch_control("reset", {})
    slot = bridge._registry.slots[0]
    slot.ws = MagicMock(send=AsyncMock())  # connected, but no action yet (slow policy)
    slot.action = None
    slot.idle = False

    tick = asyncio.create_task(bridge._tick_loop())
    try:
        await asyncio.sleep(0.15)  # well past step_timeout
        assert bridge.steps == []  # must not hold-step a connected agent
        assert slot.idle is False
        # First real action unblocks the barrier.
        slot.action = np.array([1.0], dtype=np.float32)
        bridge._action_event.set()
        for _ in range(50):
            if bridge.steps:
                break
            await asyncio.sleep(0.01)
        assert len(bridge.steps) == 1
        np.testing.assert_allclose(bridge.steps[0], [[1.0]])
    finally:
        tick.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick


async def test_tick_loop_holds_still_dialing_slot_after_timeout() -> None:
    """A claimed slot that never connects is held so the live slot can advance."""
    bridge = StubBridge(num_envs=2)
    bridge.step_timeout = 0.05
    bridge.contract = {"features": {"action": {"role": "action", "names": ["a"]}}}
    await bridge._dispatch_control("reset", {})
    await bridge._dispatch_control("reset", {})
    live, dialing = bridge._registry.slots
    live.ws = MagicMock(send=AsyncMock())
    live.action = np.array([2.0], dtype=np.float32)
    live.idle = False
    dialing.ws = None  # still dialing
    dialing.action = None
    dialing.idle = False
    bridge._action_event.set()

    tick = asyncio.create_task(bridge._tick_loop())
    try:
        for _ in range(50):
            if bridge.steps:
                break
            await asyncio.sleep(0.01)
        assert len(bridge.steps) == 1
        # Live action + hold for the dialing slot.
        np.testing.assert_allclose(bridge.steps[0], [[2.0], [0.0]])
        assert dialing.idle is True
    finally:
        tick.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick


async def test_contract_rpc_derives_under_lazy_spawn(tmp_path: Any) -> None:
    """capability()/contract RPC must not publish {} when contract.json is absent."""

    class _TinyEnv:
        """Minimal gym-shaped env for contract derivation (no gymnasium needed)."""

        def __init__(self) -> None:
            self.action_space = MagicMock(shape=(2,), dtype=np.float32)

        def reset(self, *, seed: int | None = None, options: Any = None):
            return {"state": np.zeros(3, dtype=np.float32)}, {}

        def close(self) -> None:
            pass

    bridge = GymBridge(lambda: _TinyEnv(), contract=tmp_path / "contract.json")
    assert bridge.contract == {}
    # Mimic lazy start: port up, env not built yet.
    out = await bridge._dispatch_control("contract", {})
    assert out["contract"].get("features")
    assert "action" in out["contract"]["features"]
    assert bridge.env is not None

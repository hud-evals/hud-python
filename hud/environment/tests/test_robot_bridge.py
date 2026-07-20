"""Slot-token contract on the bridge control plane (reset/result)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("openpi_client")

from hud.environment.robot.bridge import RobotBridge


class StubBridge(RobotBridge):
    """Minimal concrete bridge: no sim, fixed prompt."""

    def __init__(self, num_envs: int = 1) -> None:
        super().__init__()
        self.num_envs = num_envs

    async def reset(self, **kwargs: Any) -> str:
        return "do the task"

    def step(self, action: Any) -> None:
        pass

    def get_observation(self) -> None:
        return None


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

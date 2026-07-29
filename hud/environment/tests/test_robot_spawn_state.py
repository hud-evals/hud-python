"""Declaration-time bridge.contract / num_envs must reach the spawned child."""

from __future__ import annotations

from typing import Any

import numpy as np

from hud.environment.robot.bridge import _HUD_STATE, _apply_declaration_state, RobotBridge
from hud.environment.robot.endpoint import _bridge_init_kwargs


class _CustomBridge(RobotBridge):
    """Subclass with a real ctor param — contract is assigned after init (docs style)."""

    def __init__(self, *, use_delta: bool = False) -> None:
        super().__init__()
        self.use_delta = use_delta

    def reset(self, **kwargs: Any) -> str:
        return "p"

    def step(self, action: np.ndarray) -> None:
        return None

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        return None


def test_bridge_init_kwargs_packs_declaration_contract_and_ctor_params() -> None:
    bridge = _CustomBridge(use_delta=True)
    bridge.contract = {
        "control_rate": 10,
        "features": {"action": {"role": "action", "names": ["a"]}},
    }
    bridge.num_envs = 4
    bridge.metadata = {"backend": "test"}

    kwargs = _bridge_init_kwargs(bridge)
    assert kwargs["use_delta"] is True
    assert kwargs[_HUD_STATE]["contract"] == bridge.contract
    assert kwargs[_HUD_STATE]["num_envs"] == 4
    assert kwargs[_HUD_STATE]["metadata"] == {"backend": "test"}
    # Bind address stays with the child — never forwarded.
    assert "host" not in kwargs and "port" not in kwargs


def test_bridge_init_kwargs_skips_empty_default_state() -> None:
    bridge = _CustomBridge()
    kwargs = _bridge_init_kwargs(bridge)
    assert kwargs == {"use_delta": False}
    assert _HUD_STATE not in kwargs


def test_apply_declaration_state_sets_contract_after_ctor() -> None:
    """Child reconstructs with ctor kwargs only; state is applied afterward."""
    bridge = _CustomBridge(use_delta=True)
    assert bridge.contract == {}
    _apply_declaration_state(
        bridge,
        {
            "contract": {"features": {"action": {"role": "action", "names": ["a"]}}},
            "num_envs": 2,
            "metadata": {"k": 1},
        },
    )
    assert bridge.contract["features"]["action"]["names"] == ["a"]
    assert bridge.num_envs == 2
    assert len(bridge._registry.slots) == 2
    assert bridge.metadata == {"k": 1}

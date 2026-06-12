"""Translate observations and actions between env and policy spaces.

The loop calls ``bind``, ``reset``, ``adapt_observation``, and ``adapt_action``.
Use :class:`LeRobotAdapter` for LeRobot models; subclass for custom wiring;
``adapter=None`` for pass-through.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ─── the abstraction ──────────────────────────────────────────────────────────


class Adapter:
    """Translate between an env's observation/action spaces and a policy's.

    Lifecycle (driven by :class:`~hud.agents.robot.agent.RobotAgent`):

    - :meth:`bind` once after connect.
    - :meth:`reset` once per episode (for stateful adapters - e.g. a delta to absolute needs a starting reference to give absolute vals)
    - :meth:`adapt_observation` / :meth:`adapt_action` every step.

    Construct with the policy's image-slot names (``model_image_keys``); everything
    env-side is learned in :meth:`bind`.
    """

    def __init__(self, *, model_image_keys: list[str] | None = None) -> None:
        #: The policy's ordered image-slot names (model side; known at load time).
        self.model_image_keys: list[str] = list(model_image_keys or [])
        #: The env's selected action feature (set in :meth:`bind`).
        self.action_space: dict[str, Any] = {}
        #: The env's image / state observation keys (set in :meth:`bind`).
        self.image_keys: list[str] = []
        self.state_key: str | None = None

    def bind(self, action_space: dict[str, Any], observation_space: dict[str, Any]) -> None:
        """as in "bind model to env" - learn the env's layout from the contract (``client.spaces()``).

        Splits the observation features into image keys vs the single state key, and
        stores the action feature. Override to derive extra env-side parameters.
        """
        # TODO CLEAN
        self.action_space = action_space or {}
        image_types = ("rgb", "bgr", "gray", "depth")
        self.image_keys = []
        self.state_key = None
        for name, feature in observation_space.items():
            if feature.get("type") in image_types:
                self.image_keys.append(name)
            elif self.state_key is None:
                self.state_key = name

    def reset(self) -> None:
        """Override only if the adapter is stateful across steps within an episode."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> Any:
        """Translate an env observation + task prompt into the policy's input. Must implement - otherwise no point in using adapter"""
        raise NotImplementedError

    def adapt_action(self, action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        """Translate a policy action into the env's action space (default identity)."""
        return action


class LeRobotAdapter(Adapter):
    """Vanilla LeRobot adapter for a standard image/state env.

    Maps env cameras onto the model's image slots in order, converts HWC ``uint8`` to
    CHW ``float`` in ``[0, 1]``, and passes state + prompt through. Actions are
    identity today (postprocess already returns env-space actions). Subclass
    :class:`Adapter` for resize/pad, action reshaping, etc.
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        import torch

        data = obs["data"]
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(data[self.state_key].astype(np.float32)),
            "task": prompt,
        }
        for model_key, env_key in zip(self.model_image_keys, self.image_keys, strict=False):
            batch[model_key] = torch.from_numpy(data[env_key]).permute(2, 0, 1).float() / 255.0
        return batch

    def adapt_action(self, action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        return action


__all__ = [
    "Adapter",
    "LeRobotAdapter",
]

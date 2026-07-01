"""Translate observations and actions between env and policy spaces.

The loop calls ``bind``, ``reset``, ``adapt_observation``, and ``adapt_action``.
Use :class:`LeRobotAdapter` for LeRobot models; subclass for custom wiring;
``adapter=None`` for pass-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._types import ActionArray

# ─── the abstraction ──────────────────────────────────────────────────────────


class Adapter:
    """Translate between an env's observation/action spaces and a policy's.

    Driven by :class:`~hud.agents.robot.agent.RobotAgent`: :meth:`bind` once after
    connect, :meth:`reset` once per episode, then :meth:`adapt_observation` /
    :meth:`adapt_action` each step. Construct with the policy's image-slot names;
    everything env-side is learned in :meth:`bind`.
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
        """Learn the env's layout from the contract (``client.spaces()``).

        Splits observation features into image keys vs the single state key and stores
        the action feature. Override to derive extra env-side parameters.
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
        """Translate an env observation + task prompt into the policy's input."""
        raise NotImplementedError

    def adapt_action(self, action: ActionArray, obs: dict[str, Any]) -> ActionArray:
        """Translate a policy action into the env's action space (default identity)."""
        return action

    def adapt_chunk(self, chunk: ActionArray, obs: dict[str, Any]) -> ActionArray:
        """Translate a freshly-inferred ``[T, A]`` chunk to env space, given the query-time
        observation it was inferred from (default identity).

        The vectorized harness calls this once per slot at inference time (not per step), so a
        chunk expressed relative to the query state — e.g. DROID joint *deltas* that must be
        added to the query-time joints for absolute targets — can be converted in one shot.
        """
        return chunk


class LeRobotAdapter(Adapter):
    """Vanilla LeRobot adapter for a standard image/state env.

    Maps env cameras onto the model's image slots in order, converts HWC ``uint8`` to
    CHW ``float`` in ``[0, 1]``, and passes state + prompt through. Actions are identity
    (postprocess already returns env-space actions); subclass for resize/pad/reshaping.
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        import torch  # pyright: ignore[reportMissingImports]

        torch_mod: Any = torch
        data = obs["data"]
        batch: dict[str, Any] = {
            "observation.state": torch_mod.from_numpy(data[self.state_key].astype(np.float32)),
            "task": prompt,
        }
        for model_key, env_key in zip(self.model_image_keys, self.image_keys, strict=False):
            batch[model_key] = torch_mod.from_numpy(data[env_key]).permute(2, 0, 1).float() / 255.0
        return batch

    def adapt_action(self, action: ActionArray, obs: dict[str, Any]) -> ActionArray:
        return action


class VecLeRobotAdapter(LeRobotAdapter):
    """Batched :class:`LeRobotAdapter` for a vectorized env (:class:`~hud.agents.robot.vec_agent.VecRobotAgent`).

    Same wiring, but the obs arrays carry a leading ``N`` and the whole batch maps in one go:
    state stays ``[N, S]``, each camera ``[N, H, W, C]`` uint8 becomes ``[N, C, H, W]`` float in
    ``[0, 1]``, and the shared task is repeated to ``N`` (one prompt per env in the batch).
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        import torch  # pyright: ignore[reportMissingImports]

        data = obs["data"]
        n = len(np.asarray(data[self.state_key]))
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.asarray(data[self.state_key], dtype=np.float32)),
            "task": [prompt] * n,
        }
        for model_key, env_key in zip(self.model_image_keys, self.image_keys, strict=False):
            batch[model_key] = torch.from_numpy(np.asarray(data[env_key])).permute(0, 3, 1, 2).float() / 255.0
        return batch


class OpenPIAdapter(Adapter):
    """unwraps obs['data'] to OpenPI wire keys, attaches prompt; actions are passthrough"""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        out = dict(obs["data"])
        out.setdefault("prompt", prompt)
        return out


__all__ = [
    "Adapter",
    "LeRobotAdapter",
    "OpenPIAdapter",
    "VecLeRobotAdapter",
]

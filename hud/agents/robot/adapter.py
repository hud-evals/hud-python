"""Translate observations and actions between env and policy spaces.

The loop calls ``bind``, ``reset``, ``adapt_observation``, and the action hooks.
Use :class:`LeRobotAdapter` for LeRobot models; subclass for custom wiring;
``adapter=None`` for pass-through.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: A policy-emitted action / chunk array (the robot stack's shared alias).
ActionArray = NDArray[np.floating[Any]]

#: Image types the bundled adapters treat as cameras (vs the state vector).
IMAGE_TYPES = ("rgb", "bgr", "gray", "depth")


class Adapter:
    """Translate between an env's observation/action spaces and a policy's.

    Driven by :class:`~hud.agents.robot.agent.RobotAgent`: :meth:`bind` once
    after connect, :meth:`reset` once per episode, then :meth:`adapt_observation`
    per inference and the action hooks per chunk/step. Construct with the
    policy's image-slot names; everything env-side is learned in :meth:`bind`.
    """

    def __init__(
        self,
        *,
        model_image_keys: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        #: The policy's ordered image-slot names (model side; known at load time).
        self.model_image_keys: list[str] = list(model_image_keys or [])
        #: Open-loop horizon: how many predicted actions to execute before re-querying.
        #: ``None`` = full model chunk. Applied by :class:`~.agent.RobotAgent`, not here —
        #: so subclass ``adapt_chunk`` overrides stay free of truncation duty.
        self.chunk_size: int | None = chunk_size
        #: The env's action feature and observation layout (set in :meth:`bind`).
        self.action_space: dict[str, Any] = {}
        self.image_keys: list[str] = []
        self.state_key: str | None = None

    def bind(self, action_space: dict[str, Any], observation_space: dict[str, Any]) -> None:
        """Learn the env's layout from the contract (``client.spaces()``): image
        features become the camera keys (in contract order), the first non-image
        feature is the state. Override to derive extra env-side parameters."""
        self.action_space = action_space or {}
        self.image_keys = [n for n, f in observation_space.items() if f.get("type") in IMAGE_TYPES]
        self.state_key = next(
            (n for n, f in observation_space.items() if f.get("type") not in IMAGE_TYPES), None
        )

    def reset(self) -> None:
        """Override only if the adapter is stateful across steps within an episode."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> Any:
        """Translate an env observation + task prompt into the policy's input."""
        raise NotImplementedError

    def adapt_chunk(self, chunk: ActionArray, obs: dict[str, Any]) -> ActionArray:
        """Translate a freshly-inferred ``[T, A]`` chunk to env space, given the
        (per-slot) query-time observation it was inferred from (default identity).

        Called once per slot at inference time, so a chunk expressed relative to
        the query state — e.g. joint *deltas* to be added to the query-time
        joints — converts in one shot.
        """
        return chunk

    def adapt_action(self, action: ActionArray, obs: dict[str, Any]) -> ActionArray:
        """Per-step execution-time hook on the popped action (default identity)."""
        return action


class LeRobotAdapter(Adapter):
    """Vanilla LeRobot adapter for a standard image/state env, single or batched.

    Maps env cameras onto the model's image slots in order and converts HWC
    ``uint8`` to CHW ``float`` in ``[0, 1]``; state and prompt pass through.
    A batched observation (state ``[N, S]``) maps in one go — cameras become
    ``[N, C, H, W]`` and the shared task is repeated to ``N``. Actions are
    identity (postprocess already returns env-space actions).
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        import torch  # pyright: ignore[reportMissingImports]

        torch_mod: Any = torch  # torch ships no stubs; keep strict mode quiet
        data = obs["data"]
        state = np.asarray(data[self.state_key], dtype=np.float32)
        batched = state.ndim > 1  # [N, S] vs [S]
        batch: dict[str, Any] = {
            "observation.state": torch_mod.from_numpy(state),
            "task": [prompt] * len(state) if batched else prompt,
        }
        for model_key, env_key in zip(self.model_image_keys, self.image_keys, strict=False):
            img = torch_mod.from_numpy(np.asarray(data[env_key]))
            perm = (0, 3, 1, 2) if batched else (2, 0, 1)
            batch[model_key] = img.permute(*perm).float() / 255.0
        return batch


class OpenPIAdapter(Adapter):
    """Unwraps ``obs['data']`` to OpenPI wire keys and attaches the prompt;
    actions are pass-through."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        out = dict(obs["data"])
        out.setdefault("prompt", prompt)
        return out


__all__ = ["ActionArray", "Adapter", "LeRobotAdapter", "OpenPIAdapter"]

"""Translate observations and actions between env and policy spaces."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import builtins

# A policy-emitted action / chunk array (the robot stack's shared alias).
ActionArray = NDArray[np.floating[Any]]

# Image types the bundled adapters treat as cameras (vs the state vector).
IMAGE_TYPES = ("rgb", "bgr", "gray", "depth")


class _TorchTensor(Protocol):
    def permute(self, *dims: int) -> Self: ...

    def float(self) -> Self: ...

    def __truediv__(self, value: builtins.float) -> Self: ...


class _TorchModule(Protocol):
    def from_numpy(self, array: NDArray[Any], /) -> _TorchTensor: ...


class Adapter:
    """Translate between an env's observation/action spaces and a policy's.

    ``RobotAgent`` calls ``bind`` once after connecting, ``reset`` once per episode, then
    ``adapt_observation`` per inference, ``adapt_chunk`` per chunk and ``adapt_action`` per step.
    """

    def __init__(
        self,
        *,
        model_image_keys: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        # The policy's ordered image-slot names (model side; known at load time).
        self.model_image_keys: list[str] = list(model_image_keys or [])
        # Open-loop horizon: actions to execute before re-querying (``None``: the whole chunk).
        # ``RobotAgent`` applies it, so ``adapt_chunk`` overrides need not truncate.
        self.chunk_size: int | None = chunk_size
        # The env's action feature and observation layout (set in ``bind``).
        self.action_space: dict[str, Any] = {}
        self.image_keys: list[str] = []
        self.state_key: str | None = None

    def bind(self, action_space: dict[str, Any], observation_space: dict[str, Any]) -> None:
        """Learn the env's cameras (image features, in order) and state (first other feature)."""
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
        """Map a freshly inferred ``[T, A]`` chunk to env space.

        ``obs`` is the observation it was inferred from, so relative actions convert in one shot.
        """
        return chunk

    def adapt_action(self, action: ActionArray, obs: dict[str, Any]) -> ActionArray:
        """Per-step execution-time hook on the popped action (default identity)."""
        return action


class LeRobotAdapter(Adapter):
    """LeRobot adapter for an image-and-state env, single or batched (state ``[N, S]``).

    Cameras fill the model's image slots in order, as CHW floats in ``[0, 1]``; actions pass
    through, since LeRobot's postprocessor already returns env-space actions.
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        torch_mod = cast("_TorchModule", importlib.import_module("torch"))
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
    """Unwraps ``obs['data']`` to OpenPI wire keys and adds the prompt; actions pass through."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        out = dict(obs["data"])
        out.setdefault("prompt", prompt)
        return out

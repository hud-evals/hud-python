"""The ``Adapter``: translate between an env's spaces and a policy's spaces.

An env (the simulator) and an agent (the policy) speak different "languages":

- the env hands out observations in *its* layout (camera keys, a proprio vector);
  the policy wants them in *its* layout (named image slots, a state tensor, a task
  string);
- the policy emits an action in *its* layout; the env expects it in *its* action
  space (dimension, gripper convention, joint vs end-effector, …).

The :class:`Adapter` is the single object that owns only that translation.
The agent owns one and the base loop calls it::

    adapter.bind(spaces)                          # once after connect
    adapter.reset()                               # once per episode
    batch  = adapter.adapt_observation(obs, prompt)  # every step
    action = adapter.adapt_action(raw, obs)          # every step

Most LeRobot policies need the same generic translation, so the framework ships
:class:`DefaultAdapter` backed by :func:`lerobot_adapt_observation` /
:func:`lerobot_adapt_action`. A model with special wiring subclasses
:class:`Adapter`. ``adapter=None`` on the agent is raw pass-through.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ─── LeRobot convention (isolated, explicit, pure functions) ──────────────────


def lerobot_adapt_observation(
    obs: dict[str, Any],
    *,
    image_keys: list[str],
    state_key: str | None,
    model_image_keys: list[str],
    prompt: str,
) -> dict[str, Any]:
    """Build a LeRobot policy batch from a ``robot`` observation.

    Does the two jobs the checkpoints' own pre-processor pipeline does NOT do for
    live (gym-style) inputs — it ships a ``RenameObservationsProcessorStep`` with an
    empty map and assumes inputs are already in LeRobot dataset format:

    1. **Image format** — HWC ``uint8`` → CHW ``float`` in ``[0, 1]``. This mirrors
       LeRobot's ``VanillaObservationProcessorStep``
       (``lerobot/processor/observation_processor.py``).
    2. **Positional camera mapping** — the env names its cameras whatever it wants;
       they map onto the model's image slots *in order*. Extra model slots are left
       OUT of the batch so the policy auto-pads + masks them (do not zero-fill).

    Pure by design (keys/prompt passed in, not read from ``self``) so custom
    adapters can reuse it.
    """
    import torch  # local import: keep this module importable without torch

    data = obs["data"]
    batch: dict[str, Any] = {
        "observation.state": torch.from_numpy(data[state_key].astype(np.float32)),
        "task": prompt,
    }
    for model_key, env_key in zip(model_image_keys, image_keys, strict=False):
        batch[model_key] = torch.from_numpy(data[env_key]).permute(2, 0, 1).float() / 255.0
    return batch


def lerobot_adapt_action(action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
    """Translate a LeRobot policy action into the env's action space.

    Identity today: the checkpoint's post-processor pipeline already returns an
    action in the env's space (its ``UnnormalizerProcessorStep`` +
    ``AbsoluteActionsProcessorStep`` handle scaling/units). Kept as a named
    convention hook — for parity with :func:`lerobot_adapt_observation`, and so any
    future LeRobot-side action convention has an obvious home.
    """
    return action


# ─── the abstraction ──────────────────────────────────────────────────────────


class Adapter:
    """Translate between an env's observation/action spaces and a policy's.

    Lifecycle (driven by :class:`~hud.agents.robot.agent.RobotAgent`):

    - :meth:`bind` once after connect.
    - :meth:`reset` once per episode.
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
        """Learn the env's layout from the contract (``client.spaces()``).

        Splits the observation features into image keys vs the single state key, and
        stores the action feature. Override to derive extra env-side parameters.
        """
        self.action_space = action_space or {}
        self.image_keys = [n for n, f in observation_space.items() if f.get("dtype") == "image"]
        self.state_key = next(
            (n for n, f in observation_space.items() if f.get("dtype") != "image"), None
        )

    def reset(self) -> None:
        """Clear per-episode translation state (e.g. a delta→absolute reference).

        Override only if the adapter is stateful across steps within an episode.
        """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> Any:
        """Translate an env observation + task prompt into the policy's input. Must implement."""
        raise NotImplementedError

    def adapt_action(self, action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        """Translate a policy action into the env's action space (default identity)."""
        return action


class DefaultAdapter(Adapter):
    """The vanilla adapter: ships the LeRobot convention functions above.

    Covers the common case (most LeRobot policies + a standard image/state env):
    images positionally onto the model's slots, state + prompt passed through. A
    model that needs more (resize/pad, action reshaping) subclasses :class:`Adapter`
    instead.
    """

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        return lerobot_adapt_observation(
            obs,
            image_keys=self.image_keys,
            state_key=self.state_key,
            model_image_keys=self.model_image_keys,
            prompt=prompt,
        )

    def adapt_action(self, action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        return lerobot_adapt_action(action, obs)


__all__ = [
    "Adapter",
    "DefaultAdapter",
    "lerobot_adapt_action",
    "lerobot_adapt_observation",
]

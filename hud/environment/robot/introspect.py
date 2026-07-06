"""Gym-env introspection: split observations, probe success, derive a minimal contract.

Pure functions shared by :func:`hud.wrap` (in-process trace streaming), the
:class:`~.bridge.GymBridge`, and the :class:`~.gym.Gym` handle — no telemetry,
no wrapper state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hud.telemetry.robot import to_numpy

#: Conventional info keys carrying env-reported success, probed in order.
SUCCESS_KEYS = ("success", "is_success", "task_success")


def flatten_observation(obs: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict observations to slash-keyed leaves (non-dicts -> ``{"obs": x}``)."""
    if not isinstance(obs, dict):
        return {prefix or "obs": obs}
    flat: dict[str, Any] = {}
    for k, v in obs.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        flat.update(flatten_observation(v, key) if isinstance(v, dict) else {key: v})
    return flat


def split_observation(obs: Any, *, batched: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split an observation into ``(state, frames)``, both name -> array.

    A camera frame is a channel-last image (rank 3, or rank 4 when batched); state is
    any flat numeric vector. Anything else is dropped. Batched arrays keep their
    leading ``[N]`` dim — the recorder slices per slot.
    """
    state: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    img_rank, vec_rank = (4, 2) if batched else (3, 1)
    for name, val in flatten_observation(obs).items():
        arr = to_numpy(val)
        if arr.ndim == img_rank and arr.shape[-1] in (1, 3, 4):
            frames[name] = arr
        elif arr.ndim <= vec_rank and np.issubdtype(arr.dtype, np.number):
            state[name] = arr if batched else np.atleast_1d(arr)
    return state, frames


def probe_success(info: Any, *, num_envs: int = 1) -> np.ndarray | None:
    """Per-env success bools from conventional info keys; ``None`` when the env reports none."""
    if not isinstance(info, dict):
        return None
    for key in SUCCESS_KEYS:
        if info.get(key) is not None:
            return np.broadcast_to(to_numpy(info[key]).astype(bool).ravel(), (num_envs,))
    return None


def detect_fps(env: Any) -> int:
    """Control rate from env metadata (``render_fps``) or Isaac's ``step_dt``; default 10."""
    fps = (getattr(env, "metadata", None) or {}).get("render_fps")
    if fps:
        return round(fps)
    dt = getattr(getattr(env, "unwrapped", env), "step_dt", None)
    return round(1 / dt) if dt else 10


def capture_task_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reset parametrization as json-safe trace metadata — the full kwargs, never a label."""

    def safe(v: Any) -> Any:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        return str(v)

    return {k: safe(v) for k, v in kwargs.items() if v is not None}


def derive_contract(
    state: dict[str, Any], frames: dict[str, Any], action_dim: int, fps: int
) -> dict[str, Any]:
    """Minimal contract from one (per-env) sample observation + the action size.

    Just enough to structure the spaces and label plots — cameras as rgb observations,
    state vectors with positional names, one action feature. Users edit the written
    ``contract.json`` to rename dimensions; nothing else is derived on purpose.
    """
    features: dict[str, Any] = {}
    for name in frames:
        features[name] = {"role": "observation", "type": "rgb"}
    for name, vec in state.items():
        leaf = name.split("/")[-1]
        features[name] = {
            "role": "observation",
            "names": [f"{leaf}_{i}" for i in range(int(np.asarray(vec).size))],
        }
    features["action"] = {
        "role": "action",
        "names": [f"act_{i}" for i in range(action_dim)],
    }
    return {"control_rate": fps, "features": features}


def load_or_write_contract(
    path: str | Path | None,
    state: dict[str, Any],
    frames: dict[str, Any],
    action_dim: int,
    fps: int,
) -> dict[str, Any]:
    """The contract round-trip: load the user-edited file if present, else derive a
    minimal contract from the sample observation and write it once for inspection."""
    file = Path(path) if path else None
    if file is not None and file.exists():
        return json.loads(file.read_text())
    contract = derive_contract(state, frames, action_dim, fps)
    if file is not None:
        file.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def action_dim_of(env: Any, *, batched: bool) -> int:
    """Per-env action size from the env's (possibly batched) action space."""
    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    shape = tuple(getattr(space, "shape", None) or ())
    if batched and len(shape) > 1:
        shape = shape[1:]  # batched Isaac spaces carry the [N] dim
    return int(np.prod(shape)) if shape else 1


__all__ = [
    "SUCCESS_KEYS",
    "action_dim_of",
    "capture_task_params",
    "derive_contract",
    "detect_fps",
    "flatten_observation",
    "load_or_write_contract",
    "probe_success",
    "split_observation",
]

"""Gym-style env integration: introspection + ``hud.wrap`` trace streaming.

Two consumers share the introspection here (split observations, probe success,
derive a minimal contract) and stay in lockstep because of it:

- :class:`~.bridge.GymBridge` — the served path (``env.gym(...)``).
- :func:`hud.wrap` / :class:`TracedEnv` — the loop-owning path: wrap any
  ``gym.Env``, ``gym.vector.VectorEnv``, or batched-tensor Isaac env you drive
  yourself and every episode streams to the platform as a trace (numeric
  state, per-camera H.264 video, actions, reward/success) under one job::

      env = hud.wrap(make_env(...), job="chess-eval")

On first reset a minimal ``contract.json`` is written next to your script
describing how the observation/action spaces were interpreted; edit its
``names`` to relabel the platform's plots. An existing file is loaded instead,
so edits stick.
"""

from __future__ import annotations

import atexit
import json
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np

from hud.telemetry.robot import JobRecorder, to_numpy

logger = logging.getLogger(__name__)

#: Conventional info keys carrying env-reported success, probed in order.
SUCCESS_KEYS = ("success", "is_success", "task_success")


# ── env introspection (pure; no telemetry, no wrapper state) ──────────────────


def num_envs_of(env: Any) -> int | None:
    """``num_envs`` from the env or its unwrapped core (gymnasium 1.x wrappers
    no longer forward attributes, so an Isaac env inside ``gym.make`` hides it)."""
    n = getattr(env, "num_envs", None)
    if n is None:
        n = getattr(getattr(env, "unwrapped", env), "num_envs", None)
    return None if n is None else int(n)


def is_batched(env: Any) -> bool:
    """Vectorized (gym VectorEnv) or batched-tensor (Isaac) envs carry a leading [N] dim.

    Any env exposing ``num_envs`` is batched — Isaac keeps batch semantics even at
    ``num_envs == 1``; plain ``gym.Env``s don't have the attribute at all.
    """
    if num_envs_of(env) is not None:
        return True
    try:
        import gymnasium

        return isinstance(getattr(env, "unwrapped", env), gymnasium.vector.VectorEnv)
    except ImportError:
        return False


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


def action_dim_of(env: Any, *, batched: bool) -> int:
    """Per-env action size from the env's (possibly batched) action space."""
    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    shape = tuple(getattr(space, "shape", None) or ())
    if batched and len(shape) > 1:
        shape = shape[1:]  # batched Isaac spaces carry the [N] dim
    return int(np.prod(shape)) if shape else 1


def load_or_write_contract(
    path: str | Path | None,
    state: dict[str, Any],
    frames: dict[str, Any],
    action_dim: int,
    fps: int,
) -> dict[str, Any]:
    """The contract round-trip: load the user-edited file if present, else derive a
    minimal contract from one (per-env) sample observation and write it once.

    Derived: just enough to structure the spaces and label plots — cameras as rgb
    observations, state vectors with positional names, one action feature. Users
    edit the written file to rename dimensions; nothing else is derived on purpose.
    """
    file = Path(path) if path else None
    if file is not None and file.exists():
        return json.loads(file.read_text())
    features: dict[str, Any] = {name: {"role": "observation", "type": "rgb"} for name in frames}
    for name, vec in state.items():
        leaf = name.split("/")[-1]
        features[name] = {
            "role": "observation",
            "names": [f"{leaf}_{i}" for i in range(int(np.asarray(vec).size))],
        }
    features["action"] = {"role": "action", "names": [f"act_{i}" for i in range(action_dim)]}
    contract = {"control_rate": fps, "features": features}
    if file is not None:
        file.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


# ── hud.wrap: trace streaming for a loop you own ──────────────────────────────


class TracedEnv:
    """The observing wrapper: same ``reset``/``step``/``close`` surface, plus telemetry.

    Plain envs are recorded as a batch of one; vectorized/batched envs fan out into one
    trace per episode per slot (``done[i]`` closes slot ``i``'s trace and opens the next).

    - ``job`` — job name on the platform (default: the env's spec id / class name).
    - ``job_id`` — share one job across several wrapped envs (multi-task suites).
    - ``task`` — optional instruction/label shown on each trace's timeline.
    - ``fps`` — control rate override (default: detected from the env).
    - ``contract`` — path for the derived contract round-trip; ``None`` disables it.
    - ``record_indices`` — which env slots get rich traces (default: first 4).
    """

    def __init__(
        self,
        env: Any,
        *,
        job: str | None = None,
        job_id: str | None = None,
        task: str | None = None,
        fps: int | None = None,
        contract: str | Path | None = "contract.json",
        record_indices: list[int] | None = None,
    ) -> None:
        self.env = env
        self._batched = is_batched(env)
        self._n = num_envs_of(env) or 1
        self._fps = fps or detect_fps(env)
        self._job = job or getattr(getattr(env, "spec", None), "id", None) or type(env).__name__
        self._job_id = job_id
        self._task = task
        self._contract_path = Path(contract) if contract else None
        self._record_indices = record_indices
        self._rec: JobRecorder | None = None
        self._closed = False
        atexit.register(self.close)  # flush traces even without an explicit close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    # ── the observed surface ──────────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> Any:
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        if self._rec is None:
            self._rec = self._start(obs)
        else:
            self._rec.close_slots()  # an explicit mid-run reset ends open episodes
        # The episode's full parametrization (reset kwargs + options), never a label.
        params = capture_task_params(
            {k: v for k, v in kwargs.items() if k != "options"} | (kwargs.get("options") or {})
        )
        self._rec.extra_metadata = {"task_params": params} if params else {}
        return result

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        obs, reward, terminated, truncated = result[:4]
        info = result[4] if len(result) > 4 else {}
        if self._rec is not None:
            state, frames = split_observation(obs, batched=self._batched)
            done = np.atleast_1d(to_numpy(terminated)).astype(bool) | np.atleast_1d(
                to_numpy(truncated)
            ).astype(bool)
            if not self._batched:  # record a plain env as a batch of one
                state = {k: v[None] for k, v in state.items()}
                frames = {k: v[None] for k, v in frames.items()}
                action = to_numpy(action)[None]
            reward = np.atleast_1d(to_numpy(reward))
            self._rec.record(
                obs=state or None,
                frames=frames or None,
                action=action,
                reward=reward,
                done=done,
                success=probe_success(info, num_envs=self._n),
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._rec is not None:
            self._rec.close()
        self.env.close()

    def _start(self, obs: Any) -> JobRecorder:
        """First reset: derive/load the contract, then open the job's recorder."""
        state, frames = split_observation(obs, batched=self._batched)
        sample = {k: to_numpy(v)[0] if self._batched else v for k, v in state.items()}
        existed = self._contract_path is not None and self._contract_path.exists()
        contract = load_or_write_contract(
            self._contract_path,
            sample,
            frames,
            action_dim_of(self.env, batched=self._batched),
            self._fps,
        )
        if not existed and self._contract_path is not None:
            logger.info("hud.wrap: wrote %s (edit names to relabel plots)", self._contract_path)
        feats = contract.get("features", {})
        return JobRecorder(
            self._job,
            self._n,
            record_indices=self._record_indices,
            fps=int(contract.get("control_rate") or self._fps),
            job_id=self._job_id,
            prompt=self._task,
            action_names=next(
                (f.get("names") for f in feats.values() if f.get("role") == "action"), None
            ),
            state_names={
                k: f["names"]
                for k, f in feats.items()
                if f.get("role") == "observation" and f.get("names")
            },
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# The public verb: ``hud.wrap(env, job=...)`` — construction is the wrapping.
wrap = TracedEnv


__all__ = [
    "SUCCESS_KEYS",
    "TracedEnv",
    "action_dim_of",
    "capture_task_params",
    "detect_fps",
    "flatten_observation",
    "is_batched",
    "load_or_write_contract",
    "num_envs_of",
    "probe_success",
    "split_observation",
    "wrap",
]

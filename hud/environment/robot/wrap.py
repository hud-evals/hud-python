"""``hud.wrap`` — one-line trace streaming for gym-style envs.

Wrap any ``gym.Env``, ``gym.vector.VectorEnv``, or batched-tensor Isaac env and keep
using it exactly as before; every episode streams to the platform as a trace (numeric
state, per-camera H.264 video, actions, reward/success) under one job. The user — or
``lerobot-eval`` — keeps owning the loop; the wrapper only observes ``reset``/``step``.

    env = hud.wrap(make_env(...), job="chess-eval")

On first reset a minimal ``contract.json`` is written next to your script describing how
the observation/action spaces were interpreted; edit its ``names`` to relabel the
platform's plots. An existing file is loaded instead, so edits stick.
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np

from hud.telemetry.robot import JobRecorder, to_numpy

from .introspect import (
    action_dim_of,
    capture_task_params,
    detect_fps,
    load_or_write_contract,
    probe_success,
    split_observation,
)

logger = logging.getLogger(__name__)


def wrap(
    env: Any,
    *,
    job: str | None = None,
    job_id: str | None = None,
    task: str | None = None,
    fps: int | None = None,
    contract: str | Path | None = "contract.json",
    record_indices: list[int] | None = None,
) -> TracedEnv:
    """Stream a gym-style env's episodes to the platform; returns the env, traced.

    - ``job`` — job name on the platform (default: the env's spec id / class name).
    - ``job_id`` — share one job across several wrapped envs (multi-task suites).
    - ``task`` — optional instruction/label shown on each trace's timeline.
    - ``fps`` — control rate override (default: detected from the env).
    - ``contract`` — path for the derived contract round-trip; ``None`` disables it.
    - ``record_indices`` — which env slots get rich traces (default: first 4).
    """
    return TracedEnv(
        env,
        job=job,
        job_id=job_id,
        task=task,
        fps=fps,
        contract=contract,
        record_indices=record_indices,
    )


class TracedEnv:
    """The observing wrapper: same ``reset``/``step``/``close`` surface, plus telemetry.

    Plain envs are recorded as a batch of one; vectorized/batched envs fan out into one
    trace per episode per slot (``done[i]`` closes slot ``i``'s trace and opens the next).
    """

    def __init__(
        self,
        env: Any,
        *,
        job: str | None,
        job_id: str | None,
        task: str | None,
        fps: int | None,
        contract: str | Path | None,
        record_indices: list[int] | None,
    ) -> None:
        self.env = env
        self._batched = _is_batched(env)
        self._n = int(_num_envs(env) or 1)
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

    # ── setup ────────────────────────────────────────────────────────────────

    def _start(self, obs: Any) -> JobRecorder:
        """First reset: derive/load the contract, then open the job's recorder."""
        state, frames = split_observation(obs, batched=self._batched)
        sample = {k: to_numpy(v)[0] if self._batched else v for k, v in state.items()}
        contract = self._contract(sample, frames)
        feats = contract.get("features", {})
        rec = JobRecorder(
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
        return rec

    def _contract(self, state: dict[str, Any], frames: dict[str, Any]) -> dict[str, Any]:
        """Load the user-edited contract if present; else derive one and write it once."""
        existed = self._contract_path is not None and self._contract_path.exists()
        contract = load_or_write_contract(
            self._contract_path,
            state,
            frames,
            action_dim_of(self.env, batched=self._batched),
            self._fps,
        )
        if not existed and self._contract_path is not None:
            logger.info("hud.wrap: wrote %s (edit names to relabel plots)", self._contract_path)
        return contract

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _num_envs(env: Any) -> int | None:
    """``num_envs`` from the env or its unwrapped core (gymnasium 1.x wrappers
    no longer forward attributes, so an Isaac env inside ``gym.make`` hides it)."""
    n = getattr(env, "num_envs", None)
    if n is None:
        n = getattr(getattr(env, "unwrapped", env), "num_envs", None)
    return None if n is None else int(n)


def _is_batched(env: Any) -> bool:
    """Vectorized (gym VectorEnv) or batched-tensor (Isaac) envs carry a leading [N] dim.

    Any env exposing ``num_envs`` is batched — Isaac keeps batch semantics even at
    ``num_envs == 1``; plain ``gym.Env``s don't have the attribute at all.
    """
    if _num_envs(env) is not None:
        return True
    try:
        import gymnasium as gym

        return isinstance(getattr(env, "unwrapped", env), gym.vector.VectorEnv)
    except ImportError:
        return False


__all__ = ["TracedEnv", "wrap"]

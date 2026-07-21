"""Gym-style env integration: introspection, the served path, and ``wrap``.

Two consumers share the introspection here (split observations, probe success,
derive a minimal contract) and stay in lockstep because of it:

- :class:`GymBridge` - the served path (``env.gym(...)``): a generic
  :class:`~.bridge.RobotBridge` over any gym-style target, spawned as
  ``python -m hud.environment.robot.gym <target>``. This module is that sim
  program; :func:`gym_command` builds the argv, so both ends of the format
  live here.
- :func:`wrap` / :class:`TracedEnv` - the loop-owning path: wrap any
  ``gym.Env``, ``gym.vector.VectorEnv``, or batched-tensor Isaac env you drive
  yourself and every episode streams to the platform as a trace (numeric
  state, per-camera H.264 video, actions, reward/success) under one job::

      from hud.environment.robot import wrap

      env = wrap(make_env(...), job="chess-eval")

If no contract file exists, ``GymBridge.start`` builds the env once (factory
defaults) and writes a minimal ``contract.json`` before the capability is
published — the agent binds from that manifest. Edit ``names`` to relabel
plots; an existing file is loaded instead, so edits stick.
"""

from __future__ import annotations

import atexit
import inspect
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Self

import numpy as np

from hud.telemetry.robot import JobRecorder, to_numpy

from .bridge import RobotBridge, serve_bridge

logger = logging.getLogger(__name__)

#: Conventional info keys carrying env-reported success, probed in order.
SUCCESS_KEYS = ("success", "is_success", "task_success")


# ── env introspection (pure; no telemetry, no wrapper state) ──────────────────


def num_envs_of(env: Any) -> int | None:
    """``num_envs`` from the env or its unwrapped core (gymnasium 1.x wrappers
    no longer forward attributes, so an Isaac env inside ``gym.make`` hides it)."""
    for obj in (env, getattr(env, "unwrapped", env)):
        n = getattr(obj, "num_envs", None)
        if n is not None:
            return int(n)
    return None


def is_batched(env: Any) -> bool:
    """Vectorized (gym VectorEnv) or batched-tensor (Isaac) envs carry a leading [N] dim.

    Any env exposing ``num_envs`` is batched - Isaac keeps batch semantics even at
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
    leading ``[N]`` dim - the recorder slices per slot.
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
    """Reset parametrization as json-safe trace metadata - the full kwargs, never a label."""

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

    Derived: just enough to structure the spaces and label plots - cameras as rgb
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


# ── GymBridge: the served path (what env.gym() spawns) ────────────────────────


class GymBridge(RobotBridge):
    """Serve any gym-style env over the ``robot`` protocol, generically.

    ``target`` is how this process builds the env: a factory callable, or a
    string — a gymnasium registry id, or a declared env's spec as JSON
    (:func:`gym_command` reduces an env instance to the latter).

    Task args are partitioned into env-defining **build args** (a change
    rebuilds the env) and episodic args flowing to ``env.reset(seed=...,
    options=...)``. For a factory, its signature is the partition — so
    ``num_envs`` in the signature is the vectorization declaration; for a
    registry target, ``num_envs`` is the one build arg (``gym.make_vec``).
    Default build kwargs (e.g. ``num_envs=8`` from ``env.gym(..., num_envs=8)``)
    fill in when the task does not pass them.

    ``fps`` overrides the detected control rate; ``contract`` is the
    ``contract.json`` round-trip path (load beats derive, so user edits stick;
    ``None`` disables the file).
    """

    def __init__(
        self,
        target: Any,
        *,
        fps: int | None = None,
        contract: str | Path | None = "contract.json",
        **defaults: Any,
    ) -> None:
        host = defaults.pop("host", "127.0.0.1")
        port = int(defaults.pop("port", 0))
        super().__init__(host=host, port=port)
        self._target = target
        self._fps = fps
        self._contract_path = contract
        self._defaults = defaults  # e.g. num_envs from env.gym(..., num_envs=8)
        # Task args that define the env build (everything else is episodic).
        if callable(target):
            self._build_params = {
                n
                for n, p in inspect.signature(target).parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
        else:
            self._build_params = {"num_envs"}  # registry targets: vectorization only
        self.env: Any = None
        self.batched = False  # env carries a leading [N] dim (any env exposing num_envs)
        self._obs: Any = None  # latest observation (reset or step)
        self._instance: Any = None  # current env-defining args; a mismatch rebuilds
        self._is_torch = False
        # Per-slot episode scoring, sticky until the next reset.
        self._done: np.ndarray = np.zeros(1, dtype=bool)
        self._success: np.ndarray = np.zeros(1, dtype=bool)
        self._acc_reward: np.ndarray = np.zeros(1)
        self._step_reward: np.ndarray = np.zeros(1)  # last env.step reward (RL wire sibling)
        self._seen_success = False  # any env-reported success signal this episode

    @property
    def _unwrapped(self) -> Any:
        """The env's unwrapped core (gymnasium wrappers hide Isaac attributes)."""
        return getattr(self.env, "unwrapped", self.env)

    # ── env lifecycle (all sim touches on the sim thread) ───────────────────────

    def _load_contract_if_present(self) -> None:
        """Load a pre-written contract.json when present (skips the start-time probe)."""
        if self.contract or self._contract_path is None:
            return
        path = Path(self._contract_path)
        if path.exists():
            self.contract = json.loads(path.read_text())
            self._fps = self._fps or int(self.contract.get("control_rate") or 15)

    def _ensure_contract_from_env(self) -> None:
        """Derive/load contract from a sample observation once the env exists."""
        if self.contract or self.env is None:
            return
        state, frames = self.sample_observation()
        existed = self._contract_path is not None and Path(self._contract_path).exists()
        self.contract = load_or_write_contract(
            self._contract_path,
            state,
            frames,
            action_dim_of(self.env, batched=self.batched),
            self._fps or detect_fps(self.env),
        )
        if not existed and self._contract_path is not None:
            print(f"[env] wrote {self._contract_path} (edit names to relabel plots)", flush=True)

    async def ensure_contract(self) -> dict[str, Any]:
        """Return the wire contract, probing the env when start() did not already.

        ``start()`` mints the contract for ``env.gym`` publish; this covers a
        bare ``contract`` RPC (tests / custom callers) that skips that path.
        """
        if not self.contract:
            await self._run_on_sim(self.reset)
        return self.contract

    async def start(self) -> None:
        """Bind the wire with a complete contract so ``env.initialize`` can publish it.

        A contract file is loaded when present. Otherwise the env is built with
        factory defaults and probed here — ``@env.initialize`` awaits this — so
        the capability manifest is never empty when the agent binds.
        """
        self._load_contract_if_present()
        if "num_envs" in self._defaults:
            self.num_envs = int(self._defaults["num_envs"])
            self.batched = self.num_envs > 0
        if not self.contract:
            # Manifest is minted from bridge.contract right after start returns.
            print(
                "[env] no contract found; building env to probe observation/action structure",
                flush=True,
            )
            await self._run_on_sim(self.reset)
        await super().start()

    def reset(self, **task_args: Any) -> str:
        """Build/reset the gym env on the sim thread (base claim hops here)."""
        # Defaults from env.gym(..., num_envs=N) fill in when the task omits them.
        merged = {**self._defaults, **task_args}
        build = {k: v for k, v in merged.items() if k in self._build_params}
        episodic = {k: v for k, v in merged.items() if k not in self._build_params}
        key = tuple(sorted(build.items()))
        if self.env is None or key != self._instance:
            if self.env is not None:
                self.env.close()
            self.env = self._build_env(build)
            self._instance = key
            n = num_envs_of(self.env)
            self.batched = n is not None
            self.num_envs = n or 1
        seed = episodic.pop("seed", None)
        obs, _ = self.env.reset(seed=seed, options=episodic or None)
        self._obs = obs  # the first frame an agent sees on connect/reset
        self._is_torch = "torch" in type(_first_leaf(obs)).__module__
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._success = np.zeros(self.num_envs, dtype=bool)
        self._acc_reward = np.zeros(self.num_envs)
        self._step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._seen_success = False
        self._ensure_contract_from_env()  # no-op when start() already probed
        return self._prompt(merged)

    async def stop(self) -> None:
        await super().stop()
        if self.env is not None:
            await self._run_on_sim(self.env.close)
            self.env = None

    def _build_env(self, build: dict[str, Any]) -> Any:
        """Build the env from the target: factory call, or registry make/make_vec."""
        if callable(self._target):
            return self._target(**build)
        import gymnasium
        from gymnasium.envs.registration import EnvSpec

        env_id, kwargs = self._target, dict(build)
        if env_id.startswith("{"):  # a declared env's spec, re-made in this process
            spec = EnvSpec.from_json(env_id)
            env_id, kwargs = spec.id, {**spec.kwargs, **kwargs}
        if "num_envs" in kwargs:
            return gymnasium.make_vec(env_id, **kwargs)
        return gymnasium.make(env_id, **kwargs)

    def _prompt(self, task_args: dict[str, Any]) -> str:
        base = self._unwrapped
        for attr in ("task_description", "instruction"):
            text = getattr(getattr(base, "cfg", None), attr, None) or getattr(base, attr, None)
            if isinstance(text, str) and text:
                return text
        return ", ".join(f"{k}={v}" for k, v in sorted(task_args.items())) or "run the task"

    def sample_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """One per-env ``(state, frames)`` sample for contract derivation (post-build)."""
        state, frames = split_observation(self._obs, batched=self.batched)
        if self.batched:
            state = {k: to_numpy(v)[0] for k, v in state.items()}
            frames = {k: to_numpy(v)[0] for k, v in frames.items()}
        return state, frames

    # ── bridge protocol ──────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> None:
        act: Any = np.array(action, dtype=np.float32)  # wire buffer is read-only
        if not self.batched:
            act = act[0] if act.ndim > 1 else act  # single plain env: drop the batch dim
        elif act.ndim == 1:
            act = act[None]
        # The wire carries floats; discrete/int action spaces need their dtype + shape back.
        space = getattr(self.env, "action_space", None)
        dtype = getattr(space, "dtype", None)
        if dtype is not None and np.issubdtype(dtype, np.integer):
            act = act.astype(dtype).reshape(getattr(space, "shape", act.shape) or ())
        if self._is_torch:
            import torch

            act = torch.as_tensor(act, device=getattr(self._unwrapped, "device", None))
        obs, reward, terminated, truncated, info = self.env.step(act)
        self._obs = obs
        done = np.atleast_1d(to_numpy(terminated)).astype(bool) | np.atleast_1d(
            to_numpy(truncated)
        ).astype(bool)
        # Per-step reward for the RL wire (get_observation attaches it as a sibling).
        self._step_reward = np.atleast_1d(to_numpy(reward)).astype(np.float32)
        self._acc_reward += self._step_reward * ~self._done
        newly = done & ~self._done
        if newly.any():
            success = self._resolve_success(info)
            if success is not None:
                self._seen_success = True
                self._success |= success & newly
        self._done |= done

    def _resolve_success(self, info: Any) -> np.ndarray | None:
        """Env-reported success at a done step: info keys, else Isaac's termination term."""
        found = probe_success(info, num_envs=self.num_envs)
        if found is not None:
            return found
        manager = getattr(self._unwrapped, "termination_manager", None)
        if manager is not None:
            try:
                return np.atleast_1d(to_numpy(manager.get_term("success"))).astype(bool)
            except Exception:
                return None
        return None

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """Always ``[N, ...]`` arrays + ``[N]`` terminated for the barrier fan-out."""
        if self.env is None or self._obs is None:
            return None
        state, frames = split_observation(self._obs, batched=self.batched)
        data = {k: to_numpy(v) for k, v in {**state, **frames}.items()}
        # Last env.step reward rides along for RL collection (client lifts it
        # out of "data" to a top-level sibling, like "terminated").
        data["reward"] = self._step_reward
        if not self.batched:
            # Plain single env: lift scalars to a batch of one for uniform slicing.
            data = {
                k: (
                    v if getattr(v, "ndim", 0) >= 1 and v.shape[:1] == (1,) else np.asarray(v)[None]
                )
                for k, v in data.items()
            }
        return data, np.asarray(self._done, dtype=bool).reshape(self.num_envs)

    def result_slots(self) -> list[dict[str, Any]]:
        """Per-slot grades: env-reported success when available, else accumulated reward."""
        # The env's own success check outranks accumulated shaped reward.
        scores = self._success if self._seen_success else self._acc_reward
        return [
            {
                "score": float(scores[i]),
                "success": bool(self._success[i]),
                "total_reward": float(self._acc_reward[i]),
            }
            for i in range(self.num_envs)
        ]


def _first_leaf(obs: Any) -> Any:
    while isinstance(obs, dict):
        obs = next(iter(obs.values()))
    return obs


# ── wrap: trace streaming for a loop you own ──────────────────────────────────


class TracedEnv:
    """The observing wrapper: same ``reset``/``step``/``close`` surface, plus telemetry.

    Plain envs are recorded as a batch of one; vectorized/batched envs fan out into one
    trace per episode per slot (``done[i]`` closes slot ``i``'s trace and opens the next).

    - ``job`` - job name on the platform (default: the env's spec id / class name).
    - ``job_id`` - share one job across several wrapped envs (multi-task suites).
    - ``task`` - optional instruction/label shown on each trace's timeline.
    - ``fps`` - control rate override (default: detected from the env).
    - ``contract`` - path for the derived contract round-trip; ``None`` disables it.
    - ``record_indices`` - which env slots get rich traces (default: first 4).
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
        self._num_envs = num_envs_of(env) or 1
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
            # Episode ends on either gym flag; coerce to a 1-D bool mask.
            done = np.atleast_1d(to_numpy(terminated) | to_numpy(truncated)).astype(bool)
            if not self._batched:  # plain env → batch of one for the recorder
                state = {k: v[None] for k, v in state.items()}
                frames = {k: v[None] for k, v in frames.items()}
                action = to_numpy(action)[None]
            self._rec.record(
                obs=state or None,
                frames=frames or None,
                action=action,
                reward=np.atleast_1d(to_numpy(reward)),
                done=done,
                success=probe_success(info, num_envs=self._num_envs),
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
        # Contract naming uses one per-env sample (slot 0 when batched).
        sample = {k: to_numpy(v)[0] if self._batched else v for k, v in state.items()}
        path = self._contract_path
        wrote = path is not None and not path.exists()
        contract = load_or_write_contract(
            path,
            sample,
            frames,
            action_dim_of(self.env, batched=self._batched),
            self._fps,
        )
        if wrote:
            logger.info("wrap: wrote %s (edit names to relabel plots)", path)
        features = contract.get("features", {})
        return JobRecorder(
            self._job,
            self._num_envs,
            record_indices=self._record_indices,
            fps=int(contract.get("control_rate") or self._fps),
            job_id=self._job_id,
            prompt=self._task,
            action_names=next(
                (f.get("names") for f in features.values() if f.get("role") == "action"),
                None,
            ),
            state_names={
                k: f["names"]
                for k, f in features.items()
                if f.get("role") == "observation" and f.get("names")
            },
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# The public verb: ``from hud.environment.robot import wrap`` - construction is the wrapping.
wrap = TracedEnv


# ── the sim program (python -m hud.environment.robot.gym) ─────────────────────


def gym_command(
    target: Any,
    *,
    fps: int | None = None,
    contract: str | None = "contract.json",
    **defaults: Any,
) -> list[str]:
    """Build the command line that spawns *target* as a sim process (what ``env.gym`` runs).

    ``env.py`` only declares the sim, it never runs it — this builds the argv
    for the child process that will. A live env object can't cross that
    boundary, only text can, so *target* is reduced to something the child can
    rebuild itself: a factory's source path, a registry id as-is, or a
    constructed env's spec (id + kwargs) — closed here since only the spec
    crosses over; the child calls ``gym.make`` again for its own instance.
    ``fps`` / ``contract`` / build defaults (e.g. ``num_envs=``) flow through
    to the :class:`GymBridge` the CLI builds.
    """
    if callable(target):
        name = getattr(target, "__qualname__", "")
        if not name or "." in name or "<" in name:
            raise ValueError(f"env.gym factory must be a module-level callable, got {target!r}")
        target = f"{inspect.getfile(target)}:{name}"
    elif not isinstance(target, str):
        spec = getattr(target, "spec", None)  # registry-made envs carry their EnvSpec
        if spec is None:
            raise ValueError(
                f"env.gym: {target!r} has no registry spec; pass a gymnasium id "
                "or a module-level factory instead"
            )
        target_env, target = target, spec.to_json()
        target_env.close()  # only the spec crosses; free the declaration-time instance
    cmd = [sys.executable, "-m", "hud.environment.robot.gym", target]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if contract is not None:
        cmd += ["--contract", str(contract)]
    # Build defaults (num_envs, etc.) as --key value pairs the CLI re-applies.
    # JSON-encoded so the child gets real bools/ints/strings back (see main()).
    for key, value in defaults.items():
        cmd += [f"--{key.replace('_', '-')}", json.dumps(value)]
    return cmd


def main() -> None:
    import argparse

    from hud.utils.modules import load_module

    parser = argparse.ArgumentParser(description="Serve a gym-style env as a robot sim.")
    parser.add_argument("target", help="'path/to/module.py:factory', a registry id, or spec JSON.")
    parser.add_argument("--fps", type=int, default=None, help="Control-rate override.")
    parser.add_argument("--contract", default=None, help="contract.json round-trip path.")
    parser.add_argument("--host", default="127.0.0.1", help="Control-channel interface.")
    parser.add_argument("--port", type=int, default=0, help="Control-channel port (0 = ephemeral).")
    parser.add_argument("--num-envs", type=int, default=None, help="Vectorized slot count.")
    args, unknown = parser.parse_known_args()

    target: Any = args.target
    if ".py:" in target:  # factory by source path; ids / spec JSON pass through
        path, _, attr = target.rpartition(":")
        target = getattr(load_module(path), attr)
    defaults: dict[str, Any] = {}
    if args.num_envs is not None:
        defaults["num_envs"] = args.num_envs
    # Further --key value pairs from gym_command become build defaults. JSON
    # round-trip restores real types; bare strings (hand-written argv) pass as-is.
    for flag, value in itertools.pairwise(unknown):
        if flag.startswith("--"):
            try:
                defaults[flag[2:].replace("-", "_")] = json.loads(value)
            except json.JSONDecodeError:
                defaults[flag[2:].replace("-", "_")] = value
    bridge = GymBridge(target, fps=args.fps, contract=args.contract, **defaults)
    serve_bridge(bridge, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = [
    "SUCCESS_KEYS",
    "GymBridge",
    "TracedEnv",
    "action_dim_of",
    "capture_task_params",
    "detect_fps",
    "flatten_observation",
    "gym_command",
    "is_batched",
    "load_or_write_contract",
    "num_envs_of",
    "probe_success",
    "split_observation",
    "wrap",
]

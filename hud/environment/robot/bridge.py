"""Env-side ``robot`` bridges: the base class and every shipped variant.

The *server* side of the ``robot`` protocol (agent-side client:
:class:`~hud.capabilities.robot.RobotClient`); both share the wire codec defined
there. Bridges are **batched-first**: one bridge serves ``num_envs`` slots in
lockstep (``[N, ...]`` obs frames, ``[N, A]`` actions, an ``[N]`` ``terminated``
mask). ``num_envs == 1`` — a plain single-env sim — speaks the scalar framing
(per-env arrays, scalar ``terminated``) on the same code path.

- :class:`RobotBridge` — subclass with your sim: ``reset`` / ``step`` /
  ``get_observation``. The base owns the WebSocket serve loop.
- :class:`GymBridge` — the generic bridge over any gym-style env factory;
  users get one via ``env.gym(make_env)`` and never subclass it.

Every sim touch routes through the process :class:`~.sim_thread.SimThread`, so
bridges stay thread-naive (see :mod:`~.sim_thread` for the one process shape).
"""

from __future__ import annotations

import contextlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import websockets
import websockets.exceptions

# The openpi/0 wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _packb, _unpackb
from hud.telemetry.robot import to_numpy

from .introspect import probe_success, split_observation
from .sim_thread import SimThread


class RobotBridge(ABC):
    """Serves ``robot`` over WebSocket; subclass and implement the env hooks.

    **Subclass contract:** implement :meth:`reset`, :meth:`step`, and
    :meth:`get_observation`. The base owns the WebSocket serve loop; subclasses
    own the sim and set ``num_envs`` (default 1).

    - :meth:`reset` initialises the sim for a new episode and returns the task
      prompt. The base resets scoring state and pushes the first frame.
    - :meth:`step` advances the sim by one action (``[A]``, or ``[N, A]`` batched).
    - :meth:`get_observation` returns ``(data, terminated)`` — per-env arrays +
      scalar bool for ``num_envs == 1``, ``[N, ...]`` arrays + ``[N]`` mask
      otherwise — or ``None`` if not ready.
    - :meth:`result_slots` returns one score dict per slot. The default covers
      the common single-env binary-success case from ``self.success`` /
      ``self.total_reward``; batched bridges override it.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        # Loopback + ephemeral by default; the concrete address is published in the
        # manifest post-``start()`` and tunneled, so no env manages bridge ports.
        self._host = host
        self._port = port
        self._client: Any = None  # robot serves a single agent at a time
        self._server: Any = None
        # Connect-time metadata frame (sent first on each connection); subclasses may set it.
        self.metadata: dict[str, Any] = {}
        # Every sim touch runs on the process sim thread (see sim_thread.py).
        self._sim = SimThread.shared()
        self.num_envs: int = 1
        # Episode scoring read by ``result()``; single-env subclasses update these
        # in ``reset``/``step`` (batched bridges override result_slots instead).
        self.task_description: str = ""
        self.total_reward: float = 0.0
        self.success: bool = False
        self.terminated: bool = False

    async def _reset(self, **kwargs: Any) -> str:
        """Internal reset entry (called by the endpoint): reset scoring, run the
        author's :meth:`reset`, push the first frame."""
        self.total_reward = 0.0
        self.success = False
        self.terminated = False
        self.task_description = await self.reset(**kwargs)
        await self._send_observation()  # first frame for an already-connected agent
        return self.task_description

    @abstractmethod
    async def reset(self, **kwargs: Any) -> str:
        """Reset the sim for a new episode; return the task prompt."""

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one action."""

    @abstractmethod
    def get_observation(self) -> tuple[dict[str, np.ndarray], Any] | None:
        """Return ``(data, terminated)`` for the current state, or ``None`` if not ready."""

    def result_slots(self) -> list[dict[str, Any]]:
        """One score dict per slot. Default: single-env binary success."""
        return [
            {
                "score": 1.0 if self.success else 0.0,
                "success": bool(self.success),
                "total_reward": float(self.total_reward),
            }
        ]

    def result(self) -> dict[str, Any]:
        """The episode grade: per-slot dicts under ``"slots"``, means at the top.

        The grouped eval path grades each trace from ``slots[i]``; single-result
        consumers read the aggregate ``score``/``success`` unchanged.
        """
        slots = self.result_slots()
        return {
            "score": float(np.mean([s["score"] for s in slots])),
            "success": float(np.mean([float(s["success"]) for s in slots])),
            "total_reward": float(np.mean([s.get("total_reward", 0.0) for s in slots])),
            "slots": slots,
        }

    @property
    def url(self) -> str:
        """The bridge's concrete ``ws://`` address — publish this in the manifest.

        With an ephemeral port (the default) the address only exists once
        :meth:`start` has bound the socket, so publish after ``await bridge.start()``.
        """
        if self._port == 0:
            raise RuntimeError("bridge bound to an ephemeral port; call start() before reading url")
        return f"ws://{self._host}:{self._port}"

    async def start(self) -> None:
        # Idempotent: a long-lived bridge serves sequential agents, so re-``start`` (e.g. a
        # second run against the same sim) is a no-op rather than an EADDRINUSE rebind.
        if self._server is not None:
            return
        self._server = await websockets.serve(
            self._handle_client, self._host, self._port, max_size=None, reuse_address=True
        )
        if self._port == 0:
            self._port = self._server.sockets[0].getsockname()[1]
        print(f"[env] robot listening on ws://{self._host}:{self._port}", flush=True)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, ws: Any) -> None:
        # A later connection replaces the previous one (only one agent at a time).
        self._client = ws
        try:
            await ws.send(_packb(self.metadata))  # connect-time metadata frame
            await self._send_observation()  # current obs on connect (if ready)
            async for raw in ws:
                action = _unpackb(raw)["actions"]  # codec already returns an ndarray
                await self._sim.call(self.step, action)  # on the sim thread
                await self._send_observation()
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            # Surface failures as a string frame (a traceback) instead of a silent close.
            import traceback

            with contextlib.suppress(Exception):
                await ws.send(traceback.format_exc())
            raise
        finally:
            if self._client is ws:
                self._client = None

    async def _send_observation(self) -> None:
        """Send the current observation to the connected agent (if any).

        Framing follows ``num_envs``: scalar ``terminated`` for a single env, an
        ``[N]`` mask for a batch — the one place the two wire shapes meet.
        """
        if self._client is None:
            return
        out = await self._sim.call(self.get_observation)
        if out is None:
            return
        data, terminated = out
        done = np.asarray(terminated, dtype=bool)
        msg = {**data, "terminated": bool(done.ravel()[0]) if self.num_envs == 1 else done}
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await self._client.send(_packb(msg))


class GymBridge(RobotBridge):
    """Serve any gym-style env factory over the ``robot`` protocol, generically.

    Task args are partitioned by the factory's signature: args the factory
    accepts define the env (a change rebuilds it — so ``num_envs`` in the
    factory signature is the vectorization declaration); everything else is
    episodic and flows to ``env.reset(seed=..., options=...)``.
    """

    def __init__(self, factory: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._factory = factory
        self._factory_params = {
            n
            for n, p in inspect.signature(factory).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        self.env: Any = None
        self.batched = False  # env carries a leading [N] dim (any env exposing num_envs)
        self._obs: Any = None  # latest observation (reset or step)
        self._instance: Any = None  # current env-defining args; a mismatch rebuilds
        self._is_torch = False
        # Per-slot episode scoring, sticky until the next reset.
        self._done: np.ndarray = np.zeros(1, dtype=bool)
        self._success: np.ndarray = np.zeros(1, dtype=bool)
        self._acc_reward: np.ndarray = np.zeros(1)
        self._seen_success = False  # any env-reported success signal this episode

    # ── env lifecycle (all sim touches on the sim thread) ───────────────────────

    async def ensure_env(self, **task_args: Any) -> None:
        """Build the env (factory defaults unless task args say otherwise). Idempotent."""
        if self.env is None:
            await self._sim.call(self._sync_reset, task_args)

    async def reset(self, **task_args: Any) -> str:
        return await self._sim.call(self._sync_reset, task_args)

    async def stop(self) -> None:
        await super().stop()
        if self.env is not None:
            await self._sim.call(self.env.close)
            self.env = None

    def _sync_reset(self, task_args: dict[str, Any]) -> str:
        build = {k: v for k, v in task_args.items() if k in self._factory_params}
        episodic = {k: v for k, v in task_args.items() if k not in self._factory_params}
        key = tuple(sorted(build.items()))
        if self.env is None or key != self._instance:
            if self.env is not None:
                self.env.close()
            self.env = self._factory(**build)
            self._instance = key
            base = getattr(self.env, "unwrapped", self.env)
            n = getattr(self.env, "num_envs", getattr(base, "num_envs", None))
            self.batched = n is not None
            self.num_envs = int(n or 1)
        seed = episodic.pop("seed", None)
        obs, _ = self.env.reset(seed=seed, options=episodic or None)
        self._obs = obs  # the first frame an agent sees on connect/reset
        self._is_torch = "torch" in type(_first_leaf(obs)).__module__
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._success = np.zeros(self.num_envs, dtype=bool)
        self._acc_reward = np.zeros(self.num_envs)
        self._seen_success = False
        return self._prompt(task_args)

    def _prompt(self, task_args: dict[str, Any]) -> str:
        base = getattr(self.env, "unwrapped", self.env)
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
            act = act[None]  # batched-of-one served scalar-framed: restore the [N] dim
        # The wire carries floats; discrete/int action spaces need their dtype + shape back.
        space = getattr(self.env, "action_space", None)
        dtype = getattr(space, "dtype", None)
        if dtype is not None and np.issubdtype(dtype, np.integer):
            act = act.astype(dtype).reshape(getattr(space, "shape", act.shape) or ())
        if self._is_torch:
            import torch

            base = getattr(self.env, "unwrapped", self.env)
            act = torch.as_tensor(act, device=getattr(base, "device", None))
        obs, reward, terminated, truncated, info = self.env.step(act)
        self._obs = obs
        done = np.atleast_1d(to_numpy(terminated)).astype(bool) | np.atleast_1d(
            to_numpy(truncated)
        ).astype(bool)
        self._acc_reward += np.atleast_1d(to_numpy(reward)) * ~self._done
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
        base = getattr(self.env, "unwrapped", self.env)
        manager = getattr(base, "termination_manager", None)
        if manager is not None:
            try:
                return np.atleast_1d(to_numpy(manager.get_term("success"))).astype(bool)
            except Exception:
                return None
        return None

    def get_observation(self) -> tuple[dict[str, np.ndarray], Any] | None:
        if self.env is None or self._obs is None:
            return None
        state, frames = split_observation(self._obs, batched=self.batched)
        data = {k: to_numpy(v) for k, v in {**state, **frames}.items()}
        if self.batched and self.num_envs == 1:
            data = {k: v[0] for k, v in data.items()}  # batched-of-one: squeeze to scalar framing
        return data, self._done if self.num_envs > 1 else bool(self._done[0])

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


__all__ = ["GymBridge", "RobotBridge"]

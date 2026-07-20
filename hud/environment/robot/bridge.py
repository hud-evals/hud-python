"""Sim-process ``robot`` bridges — and the sim program that serves them.

The *server* side of the ``robot`` protocol (agent-side client:
:class:`~hud.capabilities.robot.RobotClient`); both share the wire codec defined
there. A bridge lives in the sim's own process and serves two channels: the
agent's obs/action **WebSocket** and the env server's JSON-RPC **control side
channel** (a :class:`~.endpoint.RobotEndpoint` drives episodes through it).

Internally the sim may be vectorized (``num_envs`` slots in lockstep). On the
wire every connection is scalar openpi: one claim token binds a connection to
one slot; the bridge barriers actions across claimed slots, steps once, and
fans scalar obs back. Single-env is the one-slot degenerate case.

- :class:`RobotBridge` — subclass with your sim: ``reset`` / ``step`` /
  ``get_observation``, and set ``self.contract``. The base owns both channels.
- :class:`GymBridge` — the generic bridge over any gym-style target (factory,
  registry id, or a declared env's spec); users get one via ``env.gym(...)``
  and never subclass it.

This module is also **the sim program** — the one process shape every sim runs
(the env server never hosts one, it spawns or attaches through an endpoint):

- ``python -m hud.environment.robot.bridge <target>`` — what ``env.gym()``
  spawns; :func:`gym_command` builds the argv, so both ends of the format live
  here.
- :func:`serve_bridge` — the blocking entry a custom sim program calls last.

Inside the sim process the sim owns the main thread and serving runs on a
background loop; every sim touch routes through the shared
:class:`~.sim.SimThread`, so bridges stay thread-naive (see :mod:`~.sim`).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import secrets
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import websockets
import websockets.exceptions

# The openpi/0 wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _packb, _unpackb
from hud.environment.utils import error, read_frame, reply, send_frame
from hud.telemetry.robot import to_numpy

from .gym import (
    action_dim_of,
    detect_fps,
    load_or_write_contract,
    num_envs_of,
    probe_success,
    split_observation,
)
from .sim import SimThread, run_with_sim

#: Line the sim program prints once its control channel is bound; the spawning
#: RobotEndpoint reads it from this process's stdout.
PORT_ANNOUNCEMENT = "HUD_SIM_PORT="


@dataclass
class _Slot:
    """One sim slot: claimed by a control-plane token, driven by one WS connection."""

    index: int
    token: str | None = None
    ws: Any = None
    action: np.ndarray | None = None
    idle: bool = False  # timed out waiting for an action; hold next step
    # Touched this episode; after release stays unreclaimable until the next global reset.
    used: bool = False


@dataclass
class _SlotRegistry:
    """Free/claimed slots for one bridge. All-free → next reset is global."""

    slots: list[_Slot] = field(default_factory=list)

    def configure(self, n: int) -> None:
        self.slots = [_Slot(index=i) for i in range(n)]

    @property
    def all_free(self) -> bool:
        return all(s.token is None for s in self.slots)

    def free_slot(self) -> _Slot | None:
        # v1: a freed slot is only reclaimable after all slots free (whole-batch episodes).
        return next((s for s in self.slots if s.token is None and not s.used), None)

    def by_token(self, token: str) -> _Slot | None:
        return next((s for s in self.slots if s.token == token), None)

    def claimed(self) -> list[_Slot]:
        return [s for s in self.slots if s.token is not None]

    def claim(self, slot: _Slot) -> str:
        token = f"slot-{slot.index}-{secrets.token_hex(4)}"
        slot.token = token
        slot.used = True
        slot.action = None
        slot.idle = False
        slot.ws = None
        return token

    def release(self, slot: _Slot) -> None:
        slot.token = None
        slot.ws = None
        slot.action = None
        slot.idle = False
        # keep used=True until configure() on the next global reset


class RobotBridge(ABC):
    """Serves a sim over ``robot`` WebSocket + a JSON-RPC control side channel.

    **Subclass contract:** implement :meth:`reset`, :meth:`step`, and
    :meth:`get_observation`, and set ``self.contract`` (the wire contract the
    capability publishes) before serving. The base owns the WebSocket serve
    loop and the control listener; subclasses own the sim and set ``num_envs``
    (default 1).

    - :meth:`reset` initialises the sim for a new episode and returns the task
      prompt. The base resets scoring state.
    - :meth:`step` advances the sim by one batched action ``[N, A]`` (N=1 ok).
    - :meth:`get_observation` returns ``(data, terminated)`` with ``[N, ...]``
      arrays and an ``[N]`` terminated mask (N=1 ok), or ``None`` if not ready.
    - :meth:`result_slots` returns one score dict per slot.
    """

    #: Seconds to wait for every claimed slot's action before stepping with holds.
    step_timeout: float = 30.0

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        # Loopback + ephemeral by default; the concrete address is published in the
        # manifest post-``start()`` and tunneled, so no env manages bridge ports.
        self._host = host
        self._port = port
        self._server: Any = None
        # Connect-time metadata frame (sent first on each connection); subclasses may set it.
        self.metadata: dict[str, Any] = {}
        #: The env's self-describing wire contract (features, control_rate); the
        #: endpoint fetches it over the control channel to mint the capability.
        self.contract: dict[str, Any] = {}
        # Every sim touch runs on the process sim thread (see sim.py).
        self._sim = SimThread.shared()
        self.num_envs: int = 1
        self._registry = _SlotRegistry()
        self._registry.configure(1)
        self._tick_task: asyncio.Task[None] | None = None
        self._action_event = asyncio.Event()
        # Episode scoring read by ``result_slots``; single-env subclasses update these
        # in ``reset``/``step`` (batched bridges override result_slots instead).
        self.task_description: str = ""
        self.total_reward: float = 0.0
        self.success: bool = False
        self.terminated: bool = False

    async def _claim_episode(self, **kwargs: Any) -> dict[str, Any]:
        """Control-plane reset: global sim reset when all free, else claim a free slot."""
        if self._registry.all_free:
            self.total_reward = 0.0
            self.success = False
            self.terminated = False
            self.task_description = await self.reset(**kwargs)
            self._registry.configure(self.num_envs)
            slot = self._registry.slots[0]
        else:
            slot = self._registry.free_slot()
            if slot is None:
                raise RuntimeError(f"all {self.num_envs} slots are claimed")
        token = self._registry.claim(slot)
        return {"prompt": self.task_description, "token": token}

    def _release_episode(self, token: str) -> dict[str, Any]:
        """Control-plane result: this slot's score, then free it."""
        slot = self._registry.by_token(token)
        if slot is None:
            raise ValueError(f"unknown episode token: {token!r}")
        grade = self.result_slots()[slot.index]
        self._registry.release(slot)
        return grade

    @abstractmethod
    async def reset(self, **kwargs: Any) -> str:
        """Reset the sim for a new episode; return the task prompt."""

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one batched action ``[N, A]``."""

    @abstractmethod
    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """Return ``(data[N, ...], terminated[N])``, or ``None`` if not ready."""

    def result_slots(self) -> list[dict[str, Any]]:
        """One score dict per slot. Default: single-env binary success."""
        return [
            {
                "score": 1.0 if self.success else 0.0,
                "success": bool(self.success),
                "total_reward": float(self.total_reward),
            }
        ]

    def hold_action(self) -> np.ndarray:
        """Action used for idle/stalled slots at a barrier step (zeros)."""
        action = next(
            (f for f in self.contract.get("features", {}).values() if f.get("role") == "action"),
            {},
        )
        shape = action.get("shape") or (1,)
        return np.zeros(shape, dtype=np.float32)

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
        self._registry.configure(self.num_envs)
        # No keepalive: a lockstep step can block the sim (and the GIL) for
        # minutes during heavy resets; ping timeouts would sever a healthy run.
        self._server = await websockets.serve(
            self._handle_client,
            self._host,
            self._port,
            max_size=None,
            reuse_address=True,
            ping_interval=None,
        )
        if self._port == 0:
            self._port = self._server.sockets[0].getsockname()[1]
        self._tick_task = asyncio.create_task(self._tick_loop())
        print(f"[env] robot listening on ws://{self._host}:{self._port}", flush=True)

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, ws: Any) -> None:
        """One agent connection: claim a slot, then feed actions into the barrier."""
        slot: _Slot | None = None
        try:
            await ws.send(_packb(self.metadata))  # connect-time metadata frame
            raw = await ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(raw)
            claim = _unpackb(raw)
            token = claim.get("claim")
            if not isinstance(token, str):
                raise ValueError("first frame must be {\"claim\": <token>}")
            slot = self._registry.by_token(token)
            if slot is None:
                raise ValueError(f"unknown claim token: {token!r}")
            if slot.ws is not None:
                raise RuntimeError(f"slot {slot.index} already has a live connection")
            slot.ws = ws
            slot.action = None
            slot.idle = False
            await self._send_slot_observation(slot)
            async for frame in ws:
                action = _unpackb(frame)["actions"]  # codec already returns an ndarray
                slot.action = np.asarray(action, dtype=np.float32)
                slot.idle = False
                self._action_event.set()
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            import traceback

            with contextlib.suppress(Exception):
                await ws.send(traceback.format_exc())
            raise
        finally:
            if slot is not None and slot.ws is ws:
                slot.ws = None
                slot.action = None
                self._action_event.set()  # wake the barrier so it doesn't wait on us

    async def _tick_loop(self) -> None:
        """Gather claimed slots' actions (or timeout → hold), step once, fan out."""
        while True:
            try:
                await asyncio.wait_for(self._action_event.wait(), timeout=self.step_timeout)
            except TimeoutError:
                pass
            self._action_event.clear()
            claimed = [s for s in self._registry.claimed() if s.ws is not None]
            if not claimed:
                continue
            # Wait until every live claimed slot has an action, or step_timeout elapses.
            deadline = asyncio.get_running_loop().time() + self.step_timeout
            while True:
                pending = [s for s in claimed if s.action is None and not s.idle]
                if not pending:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    for s in pending:
                        s.idle = True  # hold this step
                    break
                self._action_event.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._action_event.wait(), timeout=remaining)
                claimed = [s for s in self._registry.claimed() if s.ws is not None]
                if not claimed:
                    break
            if not claimed:
                continue
            hold = self.hold_action()
            # Stack one row per sim slot (idle/unconnected claimed → hold; free → hold).
            rows = []
            for s in self._registry.slots:
                if s in claimed and s.action is not None:
                    rows.append(np.asarray(s.action, dtype=np.float32).reshape(-1))
                else:
                    rows.append(np.asarray(hold, dtype=np.float32).reshape(-1))
                s.action = None
            actions = np.stack(rows)
            await self._sim.call(self.step, actions)
            for s in claimed:
                await self._send_slot_observation(s)

    async def _send_slot_observation(self, slot: _Slot) -> None:
        """Fan one scalar obs frame to a claimed connection."""
        if slot.ws is None:
            return
        out = await self._sim.call(self.get_observation)
        if out is None:
            return
        data, terminated = out
        i = slot.index
        # Slice the [N, ...] batch down to this slot's scalar row.
        msg = {
            **{k: (v[i] if getattr(v, "ndim", 0) >= 1 and len(v) == self.num_envs else v)
               for k, v in data.items()},
            "terminated": bool(np.asarray(terminated).reshape(-1)[i]),
        }
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await slot.ws.send(_packb(msg))

    # ── the control side channel (driven by a RobotEndpoint) ────────────────────

    async def serve_control(self, host: str = "127.0.0.1", port: int = 0) -> asyncio.Server:
        """Serve the JSON-RPC side channel: ``url`` / ``contract`` / ``reset`` / ``result``."""
        return await asyncio.start_server(self._handle_control, host, port)

    async def _handle_control(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(ConnectionResetError, asyncio.IncompleteReadError):
            while (msg := await read_frame(reader)) is not None:
                try:
                    result = await self._dispatch_control(msg["method"], msg.get("params") or {})
                    await send_frame(writer, reply(msg["id"], result))
                except Exception as exc:  # surface to the caller, keep serving the link
                    await send_frame(writer, error(msg["id"], -32000, str(exc)))
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def _dispatch_control(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "url":
            return {"url": self.url}
        if method == "contract":
            return {"contract": self.contract}
        if method == "reset":
            return await self._claim_episode(**params)
        if method == "result":
            token = params.get("token")
            if not isinstance(token, str):
                raise ValueError("result: 'token' must be a string")
            return self._release_episode(token)
        raise ValueError(f"unknown method {method!r}")


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

    # ── env lifecycle (all sim touches on the sim thread) ───────────────────────

    async def ensure_env(self, **task_args: Any) -> None:
        """Build the env (factory defaults unless task args say otherwise). Idempotent."""
        if self.env is None:
            await self._sim.call(self._sync_reset, task_args)

    async def start(self) -> None:
        """Build the env and derive/load the contract, then bring up the wire."""
        await self.ensure_env()
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
        await super().start()

    async def reset(self, **task_args: Any) -> str:
        return await self._sim.call(self._sync_reset, task_args)

    async def stop(self) -> None:
        await super().stop()
        if self.env is not None:
            await self._sim.call(self.env.close)
            self.env = None

    def _sync_reset(self, task_args: dict[str, Any]) -> str:
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
        return self._prompt(merged)

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
            act = act[None]
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
        base = getattr(self.env, "unwrapped", self.env)
        manager = getattr(base, "termination_manager", None)
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
            data = {k: (v if getattr(v, "ndim", 0) >= 1 and v.shape[:1] == (1,) else np.asarray(v)[None])
                    for k, v in data.items()}
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


# ── the sim program (what env.gym() spawns; custom sims call serve_bridge) ────


def serve_bridge(bridge: RobotBridge, *, host: str = "127.0.0.1", port: int = 0) -> None:
    """Serve *bridge*, blocking for the process's lifetime — the sim program's last call.

    Brings up the robot WebSocket and the control side channel (announcing its
    port on stdout as ``HUD_SIM_PORT=<port>``), with the sim owning the main
    thread. SIGTERM / Ctrl-C tear the bridge down on the way out.
    """

    async def _serve() -> None:
        await bridge.start()  # WS bound first, so `url` is concrete before any control call
        server = await bridge.serve_control(host, port)
        bound = server.sockets[0].getsockname()[1]
        print(f"{PORT_ANNOUNCEMENT}{bound}", flush=True)
        try:
            await asyncio.Event().wait()  # until SIGTERM / Ctrl-C cancels
        finally:
            server.close()
            await bridge.stop()

    run_with_sim(_serve)


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
    cmd = [sys.executable, "-m", "hud.environment.robot.bridge", target]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if contract is not None:
        cmd += ["--contract", str(contract)]
    # Build defaults (num_envs, etc.) as --key value pairs the CLI re-applies.
    for key, value in defaults.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
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
    # Accept further --key value pairs as build defaults.
    i = 0
    while i < len(unknown):
        flag = unknown[i]
        if flag.startswith("--") and i + 1 < len(unknown):
            defaults[flag[2:].replace("-", "_")] = unknown[i + 1]
            i += 2
        else:
            i += 1
    bridge = GymBridge(target, fps=args.fps, contract=args.contract, **defaults)
    serve_bridge(bridge, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["PORT_ANNOUNCEMENT", "GymBridge", "RobotBridge", "gym_command", "serve_bridge"]

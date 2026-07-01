"""Env-side ``robot`` bridges: the base classes users subclass to wrap their sim.

The *server* side of the ``robot`` protocol (agent-side client:
:class:`~hud.capabilities.robot.RobotClient`); both share the wire codec defined there.
Three layers, smallest to largest:

- :class:`RobotBridge` — single-agent, one sim step per received action.
- :class:`VecRobotBridge` — the same loop but every frame carries batched ``[N, ...]`` arrays
  (``terminated`` an ``[N]`` mask, action ``[N, A]``), for a vectorized env served in lockstep.
- :class:`IsaacBridge` — a :class:`VecRobotBridge` that owns an Isaac Lab env: main-thread sim
  threading, rebuild-on-instance-change, and a :meth:`~IsaacBridge.serve_forever` pump loop.

An injected :class:`~.sim_runner.SimRunner` owns *which thread runs the (thread-affine) sim*,
so subclasses stay thread-naive.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import websockets
import websockets.exceptions

# The openpi/0 wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _packb, _unpackb

from .sim_runner import InlineSimRunner, MainThreadSimRunner, SimRunner

# Task args that vary *within* one built Isaac env (cheap reset), vs. env-defining args that
# need a full rebuild (Isaac bakes assets at construction, so a new family/size is a new env).
EPISODIC_KEYS = ("seed", "background_id", "instruction_id")


class RobotBridge(ABC):
    """Serves ``robot`` over WebSocket; subclass and implement the env hooks.

    **Subclass contract:** implement :meth:`step`, :meth:`get_observation`, and
    :meth:`reset`. The base owns the WebSocket serve loop; subclasses own the sim.

    - :meth:`reset` initialises the sim for a new episode and returns the task
      prompt. The base resets scoring state and pushes the first frame for you.
    - :meth:`step` advances the sim by one action.
    - :meth:`get_observation` returns ``(data, terminated)`` for the current state
      or ``None`` if not ready.
    - :meth:`result` returns the episode score dict. The default implementation
      covers the common binary-success case; override for richer scoring (e.g.
      fractional subtask progress or realtime stats). Concrete bridges must set
      ``self.success``, ``self.total_reward``, and ``self.terminated`` during
      :meth:`step` for the default to work.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        sim_runner: SimRunner | None = None,
    ) -> None:
        # Loopback + ephemeral by default; the concrete address is published in the
        # manifest post-``start()`` and tunneled, so no env manages bridge ports.
        self._host = host
        self._port = port
        self._client: Any = None  # robot serves a single agent at a time
        self._server: Any = None
        # Connect-time metadata frame (sent first on each connection); subclasses may set it.
        self.metadata: dict[str, Any] = {}
        # Which thread runs the (thread-affine) sim. Default InlineSimRunner (loop
        # thread); inject a ThreadSimRunner (or custom) when render-heavy or thread-bound.
        self._sim_runner: SimRunner = sim_runner or InlineSimRunner()
        # Episode scoring read by ``result()``; subclasses update in ``reset``/``step``.
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
        """Reset the sim for a new episode; return the task prompt.

        Take whatever task kwargs you need (e.g. ``task_id``, ``seed``). The base
        resets scoring + sends the first obs — just reset your sim and return the prompt.
        """

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one action."""

    @abstractmethod
    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        """Return ``(data, terminated)`` for the current state, or ``None`` if not ready."""

    def result(self) -> dict[str, Any]:
        """Return the episode score dict after the episode ends.

        Default: binary success score + total reward. Override when the bridge
        tracks richer scoring (fractional subtask progress, realtime stats, …).
        The returned dict is forwarded to the harness, so include any fields the
        downstream consumers expect.
        """
        return {
            "score": 1.0 if self.success else 0.0,
            "success": bool(self.success),
            "total_reward": float(self.total_reward),
        }

    @property
    def url(self) -> str:
        """The bridge's concrete ``ws://`` address — publish this in the manifest.

        With an ephemeral port (the default) the address only exists once
        :meth:`start` has bound the socket, so publish from an
        ``@env.initialize`` hook *after* ``await bridge.start()``.
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
                await self._sim_runner.call(self.step, action)  # on the sim thread
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
        """Send the current observation to the connected agent (if any)."""
        if self._client is None:
            return
        out = await self._sim_runner.call(self.get_observation)
        if out is None:
            return
        data, terminated = out
        # openpi-style flat obs dict: array fields at the top level, terminated alongside.
        msg = {**data, "terminated": bool(terminated)}
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await self._client.send(_packb(msg))


class VecRobotBridge(RobotBridge):
    """A :class:`RobotBridge` that serves a whole *vectorized* env in lockstep.

    Same single-agent WebSocket loop, but every frame carries batched ``[N, ...]`` arrays
    and ``terminated`` is an ``[N]`` per-env mask (not a scalar); the agent sends back one
    ``[N, A]`` action. Subclasses implement the batched :meth:`step` / :meth:`get_observation`
    (which returns ``(data{name: [N, ...]}, terminated[N])``). The wire codec already carries
    arrays of any rank, so the only single-env assumption to drop is the scalar ``terminated``.
    """

    async def _send_observation(self) -> None:
        if self._client is None:
            return
        out = await self._sim_runner.call(self.get_observation)
        if out is None:
            return
        data, terminated = out
        msg = {**data, "terminated": np.asarray(terminated, dtype=bool)}  # [N] mask, not a scalar
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await self._client.send(_packb(msg))


class IsaacBridge(VecRobotBridge):
    """Serve an Isaac Lab env's whole vectorized batch over ``robot``.

    The reusable base for any Isaac Lab / Omniverse benchmark: one process owns one
    ``num_envs`` env and serves the whole batch in lockstep (``[N, ...]`` obs in, ``[N, A]``
    action out). It encodes the Isaac gotchas once — Isaac/Omniverse pins the simulator to the
    **main thread** and ``env.reset()`` nests a ``run_until_complete`` for USD loading, so every
    sim touch is routed through a :class:`~.sim_runner.MainThreadSimRunner` and executed by
    :meth:`serve_forever`'s pump loop *outside* any asyncio task.

    **Subclass contract:** implement :meth:`make_env` and :meth:`observe`. Optionally override
    :meth:`prompt` (defaults to the env's ``instruction``), :meth:`instance_key` (which task
    args trigger a rebuild), and :meth:`result` (defaults to the env's ``extras["metrics"]``).
    """

    def __init__(self, *, sim_runner: SimRunner | None = None, **kwargs: Any) -> None:
        super().__init__(sim_runner=sim_runner or MainThreadSimRunner(), **kwargs)
        self.env: Any = None
        self.base: Any = None  # env.unwrapped
        self._instance: Any = None  # current env-defining key; a mismatch forces a rebuild
        self._done: np.ndarray | None = None

    # ── subclass hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def make_env(self, **task_args: Any) -> Any:
        """Build (and return) the gym env for the resolved task. Called on the sim thread."""

    @abstractmethod
    def observe(self) -> dict[str, np.ndarray]:
        """The current batched observation as ``{contract_key: [N, ...] ndarray}``."""

    def prompt(self) -> str:
        return self.base.instruction

    def instance_key(self, task_args: dict[str, Any]) -> Any:
        """The env-defining subset of the task args; a change here rebuilds the env."""
        return tuple(sorted((k, v) for k, v in task_args.items() if k not in EPISODIC_KEYS))

    # ── bridge protocol (all sim touches run on the main thread) ───────────────
    async def reset(self, **task_args: Any) -> str:
        return await self._sim_runner.call(self._sync_reset, task_args)

    def _sync_reset(self, task_args: dict[str, Any]) -> str:
        key = self.instance_key(task_args)
        if self.env is None or key != self._instance:
            if self.env is not None:
                self.env.close()
            self.env = self.make_env(**task_args)
            self.base = self.env.unwrapped
            self._instance = key
        self.env.reset(seed=task_args.get("seed"))
        self._done = np.zeros(self.base.num_envs, dtype=bool)
        return self.prompt()

    def step(self, action: np.ndarray) -> None:
        import torch

        # np.array (copy) not asarray: the wire-decoded buffer is read-only, which torch warns on.
        act = torch.as_tensor(np.array(action, dtype=np.float32), device=self.base.device)
        _, _, terminated, truncated, _ = self.env.step(act)
        self._done = (terminated | truncated).detach().cpu().numpy().astype(bool)

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        if self.base is None:
            return None
        return self.observe(), self._done

    def result(self, **extra: Any) -> dict[str, Any]:
        """Episode-batch score from the env's ``extras["metrics"]`` (set pre-reset by the env)."""
        m = dict(self.base.extras.get("metrics", {})) if self.base is not None else {}
        return {
            "score": float(m.get("force_penalized_score", m.get("success_rate", 0.0))),
            "success": float(m.get("success_rate", 0.0)),
            **m,
            **extra,
        }

    # ── serving: own the Kit main thread, drain sim touches between frames ──────
    def serve_forever(self, simulation_app: Any, *, host: str = "0.0.0.0", port: int = 9100) -> None:
        """Serve the control endpoint + robot WebSocket on the Kit main thread, blocking for the
        process lifetime. Each pass services socket IO once, drains queued sim touches (task-free,
        on main), then pumps Kit — the one loop Isaac and asyncio can share.
        """
        import asyncio

        from .endpoint import RobotEndpoint

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(RobotEndpoint(self).serve(host, port))
        drain = getattr(self._sim_runner, "drain", None)
        while simulation_app.is_running():
            loop.run_until_complete(asyncio.sleep(0))  # advance control + robot WS tasks once
            if drain is not None:
                drain()  # execute queued sim touches on main, outside any task
            simulation_app.update()  # pump Kit (renders cameras, steps physics queue)
        loop.run_until_complete(self.stop())


__all__ = ["EPISODIC_KEYS", "IsaacBridge", "RobotBridge", "VecRobotBridge"]

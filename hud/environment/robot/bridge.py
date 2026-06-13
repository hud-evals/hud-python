"""Env-side ``robot`` bridges: base classes users subclass to wrap their sim.

The *server* side of the ``robot`` protocol (agent-side client:
:class:`~hud.capabilities.robot.RobotClient`); both share the wire codec defined
there. Subclass one of these and implement ``step`` / ``get_observation`` (plus
``no_op_action`` for realtime) to serve a sim over WebSocket:

- :class:`RobotBridge` — synchronous: steps the sim once per received action.
- :class:`RealtimeRobotBridge` — free-running wall-clock loop that pops from an
  injected :class:`~...action_provider.ActionProvider` and accepts streamed chunks.

An injected :class:`~.sim_runner.SimRunner` owns *which thread runs the
(thread-affine) sim*, so subclasses stay thread-naive.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import websockets
import websockets.exceptions

# The robot wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _decode_array, _encode_array, _packb, _unpackb

from .sim_runner import InlineSimRunner, SimRunner, ThreadSimRunner

if TYPE_CHECKING:
    from .action_provider import ActionProvider
    from .data_saving import LeRobotRecorder


# ─── synchronous env-side bridge ─────────────────────────────────────────────


class RobotBridge(ABC):
    """Serves ``robot`` over WebSocket; subclass and implement the env hooks.

    **Subclass contract:** implement :meth:`step`, :meth:`get_observation`, and
    :meth:`reset`. The base owns the WebSocket serve loop; subclasses own the sim.

    - :meth:`reset` initialises the sim for a new episode and returns the task
      prompt. The base resets scoring state and pushes the first frame for you.
    - :meth:`step` advances the sim by one action. Set ``self.last_reward`` here so
      the per-step reward is captured by the recorder.
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
        recorder: LeRobotRecorder | None = None,
        sim_runner: SimRunner | None = None,
    ) -> None:
        # Loopback + ephemeral by default; the concrete address is published in the
        # manifest post-``start()`` and tunneled, so no env manages bridge ports.
        self._host = host
        self._port = port
        self._client: Any = None  # robot serves a single agent at a time
        self._server: Any = None
        # Which thread runs the (thread-affine) sim. Default InlineSimRunner (loop
        # thread); inject a ThreadSimRunner (or custom) when render-heavy or thread-bound.
        self._sim_runner: SimRunner = sim_runner or InlineSimRunner()
        #: Optional off-loop recorder; serve loop records one frame per action, using
        #: ``self.last_reward`` (set by ``step``). See ``hud.telemetry``.
        self._recorder = recorder
        self.last_reward: float = 0.0
        # Episode scoring read by ``result()``; subclasses update in ``reset``/``step``.
        self.task_description: str = ""
        self.total_reward: float = 0.0
        self.success: bool = False
        self.terminated: bool = False
        # Most recent obs (the one the agent acted on) + terminal flag, paired with
        # the next action for recording.
        self._last_obs_data: dict[str, np.ndarray] | None = None
        self._last_terminated: bool = False

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
        The returned dict is forwarded to the harness and to ``recorder.end_episode``,
        so include any fields the downstream consumers expect.
        """
        return {
            "score": 1.0 if self.success else 0.0,
            "success": bool(self.success),
            "total_reward": float(self.total_reward),
        }

    def attach_recorder(self, recorder: LeRobotRecorder | None) -> None:
        """Attach (or replace) the off-loop recorder.

        Used by ``RobotEndpoint`` when it builds the env-var-configured recorder
        (see :meth:`~hud.environment.robot.data_saving.LeRobotRecorder.from_env`),
        so the env author never threads a recorder through by hand.
        """
        self._recorder = recorder

    @property
    def url(self) -> str:
        """The bridge's concrete ``ws://`` address — publish this in the manifest.

        With an ephemeral port (the default) the address only exists once
        :meth:`start` has bound the socket, so publish from an
        ``@env.initialize`` hook *after* ``await bridge.start()``.
        """
        if self._port == 0:
            raise RuntimeError(
                "bridge bound to an ephemeral port; call start() before reading url"
            )
        return f"ws://{self._host}:{self._port}"

    async def start(self) -> None:
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
        if self._recorder is not None:
            # Drain + finalize so the on-disk dataset is loadable. Idempotent, and
            # safe here: by stop() time no more frames are produced. Runs whenever
            # the bridge stops (e.g. from an @env.shutdown hook), so authors never
            # call recorder.close() themselves; atexit remains the backstop.
            self._recorder.close()

    async def _handle_client(self, ws: Any) -> None:
        # A later connection replaces the previous one (only one agent at a time).
        self._client = ws
        try:
            await self._send_observation()  # current obs on connect (if ready)
            async for raw in ws:
                action = _decode_array(_unpackb(raw)["data"])
                obs_before = self._last_obs_data  # the obs the agent acted on
                await self._sim_runner.call(self.step, action)  # on the sim thread
                await self._send_observation()  # advance _last_obs_data to the next obs
                if self._recorder is not None and obs_before is not None:
                    # frame = (obs the action was chosen from, action, reward from
                    # this step, whether the step ended the episode).
                    self._recorder.record_frame(
                        obs_before, action, self.last_reward, self._last_terminated
                    )
        except websockets.exceptions.ConnectionClosed:
            pass
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
        # Stash the latest obs so the next action can be paired with it for recording.
        self._last_obs_data = data
        self._last_terminated = bool(terminated)
        msg = {
            "terminated": bool(terminated),
            "data": {name: _encode_array(arr) for name, arr in data.items()},
        }
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await self._client.send(_packb(msg))


# ─── realtime (free-running) env-side bridge ─────────────────────────────────


class RealtimeRobotBridge(RobotBridge):
    """A ``robot`` bridge whose env advances on its own wall clock.

    Unlike :class:`RobotBridge` (which steps once per received action), a realtime
    bridge runs a control-rate clock loop that is fully decoupled from inference:
    every tick it pops the next action from an injected :class:`ActionProvider`
    (the env-side action queue), steps the sim, and pushes an observation enriched
    with ``meta`` (``obs_index`` / ``queue_remaining`` / ``delay`` / ``unexecuted_chunk``).

    The agent is a *client* that decides when to infer (from ``queue_remaining``)
    and replies with whole chunks via :meth:`RobotClient.send_chunk`; the provider
    merges them according to the active inference mode. The sim is wall-clock driven
    and never "freezes" during inference (it HOLDs via :meth:`no_op_action` on
    underrun in every mode, ``sync`` included — there ``sync``'s blocking cost simply
    shows up as those HOLD underruns since it only re-infers once the queue empties).
    The one exception is the legacy ``sync_freeze`` mode, whose provider returns
    ``None`` on underrun so the clock loop skips the step and the sim pauses until a
    chunk arrives.

    Subclasses still implement :meth:`step` / :meth:`get_observation` and must add
    :meth:`no_op_action`. The queueing/prefix/delay machinery is owned entirely by
    the provider, so the env stays simple and model-agnostic.
    """

    def __init__(
        self,
        *,
        provider: ActionProvider,
        control_hz: float,
        host: str = "localhost",
        port: int = 9091,
        recorder: LeRobotRecorder | None = None,
    ) -> None:
        # All sim/GL work runs on ONE dedicated worker thread (ThreadSimRunner): it keeps
        # the event loop free to stream observations / receive chunks (so a render-heavy
        # step never throttles I/O), while guaranteeing the sim's GL context stays
        # thread-affine (mujoco/EGL contexts are bound to the thread that created them).
        super().__init__(
            host=host, port=port, recorder=recorder,
            sim_runner=ThreadSimRunner(thread_name_prefix="realtime-sim"),
        )
        self._provider = provider
        self._control_period = 1.0 / float(control_hz)
        self._send_task: asyncio.Task | None = None
        # Lightweight (scalar-only) realtime meta for the most recent observation,
        # attached to each recorded frame's ``info``.
        self._last_meta: dict[str, Any] = {}

    async def run_on_sim_thread(self, fn: Any, *args: Any) -> Any:
        """Run a blocking sim/GL call on the dedicated sim thread (await the result).

        Subclasses MUST funnel every operation that touches the simulator/renderer
        (env creation, reset, step, close) through this so they all share one thread.
        Thin wrapper over the bridge's :class:`~.sim_runner.SimRunner`.
        """
        return await self._sim_runner.call(fn, *args)

    async def stop(self) -> None:
        await super().stop()
        self._sim_runner.shutdown()

    @abstractmethod
    def no_op_action(self) -> np.ndarray:
        """A safe HOLD action used when the action queue underruns (async/RTC modes)."""

    async def _reset(self, **kwargs: Any) -> str:
        # Realtime: the clock loop emits frames, so re-arm the provider instead of sending.
        self.total_reward = 0.0
        self.success = False
        self.terminated = False
        self.task_description = await self.reset(**kwargs)
        self._provider.reset()
        return self.task_description

    async def _handle_client(self, ws: Any) -> None:
        # A later connection replaces the previous one (only one agent at a time).
        self._client = ws
        self._provider.reset()
        clock = asyncio.create_task(self._clock_loop())
        try:
            async for raw in ws:
                msg = _unpackb(raw)
                if "chunk" in msg:
                    self._provider.submit_chunk(
                        _decode_array(msg["chunk"]),
                        obs_index=msg.get("obs_index"),
                        delay_used=msg.get("delay_used"),
                    )
                # legacy single-action messages are ignored on the realtime path
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            clock.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await clock
            if self._client is ws:
                self._client = None

    async def _clock_loop(self) -> None:
        """Advance the sim at ``control_hz``, independent of agent inference."""
        try:
            # Emit the post-reset observation first so the client has an initial frame.
            await self._send_observation_rt()
            while self._client is not None:
                t0 = time.perf_counter()
                if not self.terminated:
                    # Wall-clock sim always advances (models the real world): on
                    # underrun the provider returns a HOLD (no-op), never stalling.
                    # Run the (often render-heavy) step on the sim thread so the loop
                    # stays free to stream obs / receive chunks.
                    # Exception: ``sync_freeze`` returns ``None`` on underrun to pause
                    # the clock (legacy) — skip the step so the sim freezes till a chunk lands.
                    action = self._provider.next_action(self.no_op_action)
                    if action is not None:
                        obs_before = self._last_obs_data  # obs the agent acted on
                        meta_before = self._last_meta
                        await self.run_on_sim_thread(self.step, action)
                        if self._recorder is not None and obs_before is not None:
                            # Record every executed tick (HOLDs included) so the
                            # trajectory stays dense at the control rate.
                            self._recorder.record_frame(
                                obs_before, action, self.last_reward, self.terminated,
                                info=meta_before,
                            )
                await self._send_observation_rt()
                if self.terminated:
                    break
                await asyncio.sleep(max(0.0, self._control_period - (time.perf_counter() - t0)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface otherwise-silent task failures
            import traceback

            print(f"[env] clock loop crashed: {exc!r}", flush=True)
            traceback.print_exc()
            raise

    async def _send_observation_rt(self) -> None:
        """Push the current observation plus the provider's realtime ``meta`` block.

        The send is best-effort and time-bounded: a slow client must never stall
        the control clock (realtime invariant), and a stale dropped observation is
        harmless since the agent only ever needs the latest frame.
        """
        if self._client is None:
            return
        out = self.get_observation()
        if out is None:
            return
        data, terminated = out
        meta = self._provider.obs_meta()
        # Stash the latest obs + scalar meta so the next executed action can be
        # paired with it for recording (drop the heavy ``unexecuted_chunk`` array).
        self._last_obs_data = data
        self._last_terminated = bool(terminated)
        self._last_meta = {
            "obs_index": int(meta["obs_index"]),
            "queue_remaining": int(meta["queue_remaining"]),
            "delay": int(meta["delay"]),
            "active_chunk_obs_index": int(meta.get("active_chunk_obs_index", -1)),
        }
        unexecuted_chunk = meta.get("unexecuted_chunk")
        msg = {
            "terminated": bool(terminated),
            "data": {name: _encode_array(arr) for name, arr in data.items()},
            "meta": {
                "obs_index": int(meta["obs_index"]),
                "queue_remaining": int(meta["queue_remaining"]),
                "delay": int(meta["delay"]),
                "active_chunk_obs_index": int(meta.get("active_chunk_obs_index", -1)),
                "unexecuted_chunk": _encode_array(unexecuted_chunk) if unexecuted_chunk is not None else None,
            },
        }
        payload = _packb(msg)
        client = self._client
        if terminated:
            # Ensure the client reliably sees the terminal frame.
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await client.send(payload)
            return
        # Single-flight, non-blocking: if the previous obs is still being flushed
        # (a busy/slow client), drop this frame rather than stall the control clock.
        # The agent only ever needs the latest observation.
        if self._send_task is not None and not self._send_task.done():
            return

        async def _send() -> None:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await client.send(payload)

        self._send_task = asyncio.create_task(_send())


__all__ = ["RealtimeRobotBridge", "RobotBridge"]

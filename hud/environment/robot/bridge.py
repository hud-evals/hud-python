"""Env-side ``robot`` bridge: the base class users subclass to wrap their sim.

The *server* side of the ``robot`` protocol (agent-side client:
:class:`~hud.capabilities.robot.RobotClient`); both share the wire codec defined
there. Subclass :class:`RobotBridge` and implement ``step`` / ``get_observation`` to
serve a sim over WebSocket — it steps the sim once per received action.

An injected :class:`~.sim_runner.SimRunner` owns *which thread runs the
(thread-affine) sim*, so subclasses stay thread-naive.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import websockets
import websockets.exceptions

# The robot wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _decode_array, _encode_array, _packb, _unpackb

from .sim_runner import InlineSimRunner, SimRunner

if TYPE_CHECKING:
    from .data_saving import LeRobotRecorder


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


__all__ = ["RobotBridge"]

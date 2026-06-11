"""Off-loop trajectory recording for robot environments.

A :class:`RobotBridge` produces a high-rate stream of ``(observation, action,
reward, done)`` tuples on its control loop. Recording them must never slow that
loop down, so this module splits the work in two:

- on the control thread, :meth:`EpisodeRecorder.record_frame` does only a cheap
  copy + enqueue and returns immediately;
- a single daemon worker thread drains the queue and forwards each event to a
  :class:`TraceSink`, which does all the heavy lifting (image/video encoding,
  parquet writes, stats) entirely off the control loop.

``TraceSink`` is the decoupling seam: the file-backed LeRobot-dataset sink lives in
:mod:`hud.telemetry.lerobot`, and a future "stream to the HUD platform" sink can
drop in without touching any environment. It is a sibling of the span ``exporter`` —
both are background-thread "record what happened during a run and ship it"
machinery, which is why this lives under :mod:`hud.telemetry`.
"""

from __future__ import annotations

import atexit
import logging
import queue
import signal
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """One control-tick transition: the obs acted on, the action, and its result.

    ``obs`` maps the env's wire feature names to arrays (images included); ``action``
    is the executed action vector; ``reward`` / ``done`` are the env's per-step
    result; ``info`` carries any extra per-frame context (e.g. the realtime ``meta``
    block: ``obs_index`` / ``queue_remaining`` / ``delay``).
    """

    obs: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class TraceSink(ABC):
    """Consumer of a recorded trajectory, called only on the worker thread.

    The recorder guarantees calls are serialized and ordered:
    ``on_episode_start`` -> ``on_frame*`` -> ``on_episode_end`` per episode, and a
    single ``on_close`` after the last episode. Implementations may block (all
    calls are off the control loop); exceptions are caught and logged by the
    recorder so a sink failure never crashes the env.
    """

    @abstractmethod
    def on_episode_start(self, meta: dict[str, Any]) -> None:
        """Begin a new episode (``meta`` carries e.g. ``prompt`` / ``task``)."""

    @abstractmethod
    def on_frame(self, frame: Frame) -> None:
        """Consume one recorded :class:`Frame`."""

    @abstractmethod
    def on_episode_end(self, meta: dict[str, Any]) -> None:
        """Finish the current episode (``meta`` carries e.g. ``success`` / reward)."""

    def on_close(self) -> None:
        """Flush/finalize everything (called once after the last episode)."""


# Sentinel event kinds placed on the queue.
_START = "start"
_FRAME = "frame"
_END = "end"

# Shutdown signals we want handled on the *main* thread (asyncio's SIGINT and our
# own SIGTERM/SIGHUP). The worker thread blocks these so the OS never delivers
# them there (see EpisodeRecorder._run).
_SHUTDOWN_SIGNALS = frozenset(
    s for s in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP")) if s is not None
)


class EpisodeRecorder:
    """Buffer trajectory events on the control loop, drain them on a worker thread.

    Construct with one or more :class:`TraceSink` s, then drive the episode
    lifecycle from the env: :meth:`start_episode` / :meth:`record_frame` /
    :meth:`end_episode`, and :meth:`close` once at shutdown. Every public method
    is non-blocking except :meth:`close`, which drains the queue and joins the
    worker.

    With multiple sinks, every event fans out to each sink in construction order
    (one copy, one queue, one worker — N consumers). Sink failures are isolated
    per sink: one sink raising never starves the others of the event.
    """

    def __init__(self, *sinks: TraceSink, max_queue: int = 0) -> None:
        if not sinks:
            raise ValueError("EpisodeRecorder needs at least one TraceSink")
        self._sinks = sinks
        # max_queue == 0 -> unbounded. Recording is opt-in for offline data
        # collection, so we favor never dropping frames over bounding memory.
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self._worker = threading.Thread(
            target=self._run, name="trace-recorder", daemon=True
        )
        self._closed = False
        self._worker.start()
        self._install_shutdown_hooks()

    # ── lifecycle (called on the control loop; cheap + non-blocking) ──────────

    def start_episode(self, **meta: Any) -> None:
        """Open a new episode; ``meta`` is forwarded to ``sink.on_episode_start``."""
        self._put((_START, dict(meta)))

    def record_frame(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Copy + enqueue one transition. Returns immediately (no encoding here)."""
        import numpy as np

        # Copy now so later in-place sim mutation can't corrupt a buffered frame.
        # These are small (a few camera frames + short vectors): microseconds.
        obs_copy = {k: np.array(v, copy=True) for k, v in obs.items()}
        action_copy = np.array(action, copy=True)
        self._put((_FRAME, Frame(obs_copy, action_copy, float(reward), bool(done), dict(info or {}))))

    def end_episode(self, **meta: Any) -> None:
        """Close the current episode; ``meta`` is forwarded to ``sink.on_episode_end``."""
        self._put((_END, dict(meta)))

    def close(self) -> None:
        """Drain the queue, finalize the sink, and join the worker thread."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # poison pill (bypasses the dropped-after-close guard)
        self._worker.join()

    # ── internals ─────────────────────────────────────────────────────────────

    def _install_shutdown_hooks(self) -> None:
        """Finalize the sink on normal interpreter exit.

        A trace sink may stream into a format that is only readable once finalized
        (e.g. LeRobot writes every episode into one open parquet file whose footer
        is written by ``finalize``), so a process that exits without ``close`` would
        leave an unreadable dataset on disk. Registering :meth:`close` with
        ``atexit`` covers normal exit, ``sys.exit`` and unhandled exceptions.

        Signal-driven shutdown (``SIGTERM`` / ``SIGHUP`` / ``Ctrl-C``) is the
        owning app's responsibility: it must route the signal to :meth:`close`
        (asyncio apps should use ``loop.add_signal_handler`` — a plain
        ``signal.signal`` handler is unreliable once a worker thread exists). The
        worker masks those signals (see :meth:`_run`) so they are always delivered
        to the main thread where the app/event loop can act on them.
        """
        atexit.register(self.close)

    def _put(self, event: tuple[str, Any]) -> None:
        if self._closed:
            logger.warning("EpisodeRecorder is closed; dropping %s event", event[0])
            return
        self._queue.put(event)

    def _run(self) -> None:
        # Block shutdown signals on this worker thread so the OS delivers them to
        # the main thread, where Python actually runs signal handlers. Otherwise a
        # signal delivered here while the main thread is parked (e.g. in asyncio's
        # epoll) would never run the handler — finalize would be skipped. Unix-only;
        # a no-op elsewhere. Must run on this thread, hence here rather than in init.
        if hasattr(signal, "pthread_sigmask") and _SHUTDOWN_SIGNALS:
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, _SHUTDOWN_SIGNALS)
            except (ValueError, OSError):
                pass
        while True:
            event = self._queue.get()
            if event is None:
                break
            kind, payload = event
            for sink in self._sinks:  # per-sink isolation: one failing never starves the rest
                try:
                    if kind == _START:
                        sink.on_episode_start(payload)
                    elif kind == _FRAME:
                        sink.on_frame(payload)
                    elif kind == _END:
                        sink.on_episode_end(payload)
                except Exception:  # a sink failure must never crash the env
                    logger.exception("trace sink %r failed handling %s event", sink, kind)
        for sink in self._sinks:
            try:
                sink.on_close()
            except Exception:
                logger.exception("trace sink %r failed on close", sink)


__all__ = ["EpisodeRecorder", "Frame", "TraceSink"]

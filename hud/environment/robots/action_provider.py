"""Env-side action providers: the action queue + prefix + delay machinery.

A realtime :class:`~hud.environment.robots.bridge.RealtimeRobotBridge` owns one
``ActionProvider``. The provider holds the buffered action chunk the sim is
executing, hands out one action per control tick (HOLDing on underrun), accepts
fresh chunks from the agent, and merges them according to the active inference
mode. It also exposes the realtime ``meta`` the env attaches to every
observation (so the agent can decide when to infer and, for RTC, condition on
the unexecuted prefix and the estimated inference delay).

The abstraction mirrors LeRobot's ``InferenceEngine`` contract but lives on the
*environment* side: the env stays simple and model-agnostic, and swapping the
queueing strategy (the modes below) never touches the env.

The sim clock is wall-clock driven and *always* advances (it models the real
world, which never freezes): on underrun the provider HOLDs (the env steps a
no-op so the robot keeps its pose) — it never stalls the sim. The sole exception
is ``sync_freeze``, the legacy mode that deliberately pauses the clock during
inference to demonstrate the unrealistic behavior the realtime path avoids.

Modes
-----
- ``sync``           : the blocking baseline. Execute the chunk to exhaustion,
                       and only *then* request the next one (request-on-empty,
                       no overlap). While the model infers, the sim keeps running
                       and the robot HOLDs — so the inference latency shows up as
                       underruns. A returned chunk fully replaces the queue.
- ``sync_freeze``    : like ``sync`` but the sim *freezes* (clock pauses) while the
                       model infers — the legacy behavior. Latency is hidden (no
                       ticks elapse) rather than paid as underruns.
- ``naive_async``    : free-run; drop the ``d`` actions consumed in flight and
                       replace the postfix wholesale (``queue = chunk[d:]``).
- ``weighted_async`` : as naive, but blend the overlap with the old tail.
- ``rtc``            : same queue op as naive, but the agent conditions inference
                       on the unexecuted prefix + delay so the chunk is already
                       continuous (Real-Time Chunking).

Delay accounting follows RTC Algorithm 1: a small buffer of recently measured
delays yields a conservative estimate ``d = max(buffer)`` (sent with each obs),
and the *real* delay of a returned chunk is the number of control ticks consumed
between the triggering observation and the chunk's arrival.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np


class ActionProvider(ABC):
    """Env-side action queue with pluggable chunk-merge semantics.

    Subclasses set the class flags (``mode`` / ``uses_prefix``) and implement
    :meth:`_merge`. Everything else (the queue, the global tick counter, delay
    tracking, the obs ``meta`` block) is shared.
    """

    mode: ClassVar[str] = "base"
    #: ``True`` for ``rtc``: the agent should condition inference on the prefix.
    uses_prefix: ClassVar[bool] = False
    #: ``True`` only for ``sync_freeze``: pause the sim clock on underrun (legacy
    #: blocking behavior) instead of HOLDing. ``next_action`` returns ``None`` so the
    #: clock loop skips the step entirely until a fresh chunk lands.
    freeze_on_underrun: ClassVar[bool] = False

    def __init__(
        self,
        *,
        execution_horizon: int = 10,
        delay_buffer_size: int = 10,
        init_delay: int = 1,
    ) -> None:
        self.execution_horizon = int(execution_horizon)
        self._delay_buffer_size = int(delay_buffer_size)
        self._init_delay = int(init_delay)
        self._lock = threading.Lock()
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the queue and all episode-scoped counters."""
        with self._lock:
            self._queue: np.ndarray | None = None
            self._pos = 0
            self._tick_index = 0  # monotonic control-tick counter (one per sim step, incl. HOLDs)
            self._active_chunk_obs_index = -1  # obs_index the active (most-recently-merged) chunk came from
            self._received_chunk = False  # False until the first chunk lands (bootstrap)
            self._delay_buffer: deque[int] = deque([self._init_delay], maxlen=self._delay_buffer_size)
            # metrics
            self._underruns = 0
            self._n_inferences = 0
            self._delays: list[int] = []

    # ── action production (called once per control tick) ──────────────────────

    def next_action(self, no_op_fn: Callable[[], np.ndarray]) -> np.ndarray | None:
        """Pop the next executable action, or handle an empty queue (underrun).

        For every mode except ``sync_freeze`` the sim always advances (it models the
        real world, which never freezes): on underrun this returns ``no_op_fn()``
        (HOLD: the robot keeps its pose while the sim keeps stepping) and advances
        the tick counter, so the in-flight inference delay is measured correctly.

        ``sync_freeze`` (``freeze_on_underrun``) is the legacy exception: on underrun
        it returns ``None`` so the clock loop *skips the step* and the sim pauses
        until a chunk lands. No tick elapses, so the latency is hidden rather than
        paid as underruns — the unrealistic artifact this mode exists to show.
        """
        with self._lock:
            if self._queue is not None and self._pos < len(self._queue):
                action = self._queue[self._pos]
                self._pos += 1
                self._tick_index += 1
                return np.asarray(action, dtype=np.float32)
            # underrun
            if self.freeze_on_underrun:
                # Pause the clock: no tick advances, no underrun counted.
                return None
            # Bootstrap HOLDs (before the very first chunk lands — includes one-time
            # policy warmup/compile) are expected and not counted as failures; only
            # steady-state underruns reflect a real inability to keep up.
            if self._received_chunk:
                self._underruns += 1
            self._tick_index += 1
        return np.asarray(no_op_fn(), dtype=np.float32)

    # ── chunk ingestion (called when the agent sends a chunk) ─────────────────

    def submit_chunk(
        self, chunk: Any, *, obs_index: int | None = None, delay_used: int | None = None
    ) -> int:
        """Merge a freshly inferred chunk, returning the measured delay (ticks)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        with self._lock:
            if obs_index is None:
                measured_d = 0
            else:
                measured_d = max(0, self._tick_index - int(obs_index))
            measured_d = min(measured_d, len(chunk))
            self._n_inferences += 1
            # The first (cold-start) chunk's delay reflects warmup/compile, not the
            # steady-state inference latency, so keep it out of the estimate + stats.
            if self._received_chunk:
                self._delay_buffer.append(measured_d)
                self._delays.append(measured_d)
            self._merge(chunk, measured_d)
            self._pos = 0
            self._received_chunk = True
            if obs_index is not None:
                self._active_chunk_obs_index = int(obs_index)
        return measured_d

    @abstractmethod
    def _merge(self, chunk: np.ndarray, delay: int) -> None:
        """Set ``self._queue`` from the new ``chunk`` given the measured ``delay``."""

    # ── realtime meta (attached to every observation) ─────────────────────────

    def obs_meta(self) -> dict[str, Any]:
        """The realtime ``meta`` block the env attaches to every observation.

        Fields (all that the agent needs to decide *when* to infer and, for RTC,
        *what* to condition on):

        - ``obs_index``: the env's ``tick_index`` at emit time — an episode-scoped,
          monotonic control-tick counter (incremented once per sim step, HOLDs
          included; reset to 0 each episode). It is the timestamp the agent stamps
          onto the chunk it sends back, so the env can later measure the real
          inference delay as ``tick_index_on_arrival - obs_index``.
        - ``queue_remaining``: how many unexecuted actions are still buffered. This is
          the agent's trigger: it infers when ``queue_remaining <= threshold``.
        - ``delay``: the conservative inference-delay estimate in ticks
          (``max`` over recently measured delays); RTC conditions on it and the agent
          echoes it back as ``delay_used``.
        - ``active_chunk_obs_index``: the ``obs_index`` the most-recently-merged
          (currently active) chunk was computed from — an ack the agent uses to clear
          its in-flight ``pending`` guard once its chunk is live in the queue.
        - ``unexecuted_chunk``: the live chunk's not-yet-executed tail (executable
          space); RTC builds its prefix conditioning from this (freeze the first
          ``delay`` actions, soft-mask the rest). ``None`` when the queue is empty.
        """
        with self._lock:
            remaining = 0 if self._queue is None else max(0, len(self._queue) - self._pos)
            unexecuted_chunk: np.ndarray | None = None
            if remaining > 0 and self._queue is not None:
                unexecuted_chunk = np.array(self._queue[self._pos :], dtype=np.float32, copy=True)
            return {
                "obs_index": self._tick_index,  # episode tick counter (incl. HOLDs); the chunk's timestamp
                "queue_remaining": remaining,  # count of unexecuted actions left; the agent's infer trigger
                "delay": max(self._delay_buffer) if self._delay_buffer else 0,  # conservative delay est (ticks)
                "active_chunk_obs_index": self._active_chunk_obs_index,  # obs_index the active (most-recently-merged) chunk came from
                # the live chunk's not-yet-executed tail (executable space); RTC builds
                # its prefix conditioning (frozen first `delay`, soft-masked rest) from this.
                "unexecuted_chunk": unexecuted_chunk,
            }

    # ── metrics ───────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Episode metrics for ablation reporting."""
        with self._lock:
            delays = list(self._delays)
            return {
                "mode": self.mode,
                "ticks": self._tick_index,
                "underruns": self._underruns,
                "n_inferences": self._n_inferences,
                "mean_delay": float(np.mean(delays)) if delays else 0.0,
                "max_delay": int(max(delays)) if delays else 0,
            }


class SyncActionProvider(ActionProvider):
    """Blocking baseline: run a chunk to exhaustion, HOLD while the next infers.

    The sim never pauses (HOLD-on-underrun like every mode). What makes this the
    blocking baseline is purely the trigger discipline: the agent only re-infers
    once the queue is *empty* (request-on-empty, advertised as ``threshold == 0``),
    so inference never overlaps execution and its latency is paid as HOLD ticks
    (underruns) every cycle. The fresh chunk fully replaces the (empty) queue.
    """

    mode: ClassVar[str] = "sync"

    def _merge(self, chunk: np.ndarray, delay: int) -> None:
        # Sync only infers once the queue is empty, so nothing overlaps: execute
        # the whole chunk from the start (the HOLD gap is the cost, not dropped actions).
        self._queue = chunk


class SyncFreezeActionProvider(SyncActionProvider):
    """Legacy blocking baseline: the sim *freezes* while the model infers.

    Identical to :class:`SyncActionProvider` (request-on-empty, full-replace merge)
    except that on underrun it pauses the control clock entirely (``next_action``
    returns ``None``) and resumes only when the next chunk lands — the original
    "env freezes on each inference" behavior. Because no ticks elapse during
    inference, the latency is hidden instead of paid as HOLD underruns, which is
    precisely the unrealistic artifact this mode exists to demonstrate against the
    (clock-never-stops) ``sync`` baseline.
    """

    mode: ClassVar[str] = "sync_freeze"
    freeze_on_underrun: ClassVar[bool] = True


class NaiveAsyncActionProvider(ActionProvider):
    """Free-running async: drop the in-flight prefix, replace the postfix wholesale."""

    mode: ClassVar[str] = "naive_async"

    def _merge(self, chunk: np.ndarray, delay: int) -> None:
        self._queue = chunk[delay:]


class WeightedAsyncActionProvider(ActionProvider):
    """Free-running async with a weighted blend across the overlapping timesteps."""

    mode: ClassVar[str] = "weighted_async"

    def __init__(self, *, weight: float = 0.7, **kwargs: Any) -> None:
        # weight = how much the new chunk dominates the blend over the overlap.
        self._weight = float(weight)
        super().__init__(**kwargs)

    def _merge(self, chunk: np.ndarray, delay: int) -> None:
        new = chunk[delay:]
        old_tail = None
        if self._queue is not None and self._pos < len(self._queue):
            old_tail = self._queue[self._pos :]
        if old_tail is None or len(old_tail) == 0 or len(new) == 0:
            self._queue = new
            return
        overlap = min(len(old_tail), len(new))
        merged = np.array(new, dtype=np.float32, copy=True)
        merged[:overlap] = self._weight * new[:overlap] + (1.0 - self._weight) * old_tail[:overlap]
        self._queue = merged


class RTCActionProvider(NaiveAsyncActionProvider):
    """Real-Time Chunking: same queue op as naive, but the agent conditions on the prefix.

    The continuity work happens *inside* the policy (prefix inpainting + soft
    masking), so by the time a chunk arrives it is already consistent with the
    frozen prefix and a plain drop-``d``/replace is correct.
    """

    mode: ClassVar[str] = "rtc"
    uses_prefix: ClassVar[bool] = True


_PROVIDERS: dict[str, type[ActionProvider]] = {
    "sync": SyncActionProvider,
    "sync_freeze": SyncFreezeActionProvider,
    "naive_async": NaiveAsyncActionProvider,
    "weighted_async": WeightedAsyncActionProvider,
    "rtc": RTCActionProvider,
}


def make_action_provider(mode: str, **kwargs: Any) -> ActionProvider:
    """Construct the provider for an inference ``mode`` (see module docstring)."""
    if mode not in _PROVIDERS:
        raise ValueError(f"Unknown inference mode '{mode}'. Available: {sorted(_PROVIDERS)}")
    if mode != "weighted_async":
        kwargs.pop("weight", None)  # only the weighted provider takes a blend weight
    return _PROVIDERS[mode](**kwargs)


__all__ = [
    "ActionProvider",
    "NaiveAsyncActionProvider",
    "RTCActionProvider",
    "SyncActionProvider",
    "SyncFreezeActionProvider",
    "WeightedAsyncActionProvider",
    "make_action_provider",
]

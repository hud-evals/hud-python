"""Optional Rerun view of one robot episode.

Cameras, contract-labelled state, the executed action, reward, and termination
go to a Rerun viewer, a ``.rrd`` file, or both. ``rerun-sdk`` ships with the
``robot`` extra and is imported lazily; the view stays off unless a
:class:`~hud.agents.robot.agent.RobotAgent` opts in.

Logging is queued onto a worker thread. A slow viewer drops camera frames
first, then whole steps, so the control loop (and a later physical arm) never
waits on the viewer.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from hud.agents.types import ObservationStep

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Contract ``type`` values that are camera frames, not state vectors.
_IMAGE_TYPES = ("rgb", "bgr", "gray", "depth")


@dataclass(frozen=True)
class _Step:
    """One control tick, copied so the env can reuse its buffers."""

    tick: int
    images: tuple[tuple[str, NDArray[Any]], ...]
    state: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    action: tuple[tuple[str, float], ...] | None
    reward: float | None
    terminated: bool


class RerunView:
    """Record one episode to Rerun.

    ``RerunView()`` writes ``rerun/episode_*.rrd``. ``spawn=True`` also opens a
    local viewer (and still writes the file). ``connect_url`` streams at a
    viewer that is already running, which is how a headless robot box reaches
    a laptop (``rerun+http://127.0.0.1:9876/proxy`` through an ssh tunnel).
    ``spawn`` and ``connect_url`` cannot be combined. Pass ``save=False`` to
    skip the file.
    """

    def __init__(
        self,
        *,
        recording_path: str | Path | None = None,
        recording_dir: str | Path | None = None,
        spawn: bool = False,
        connect_url: str | None = None,
        spawn_port: int = 9876,
        save: bool = True,
        jpeg_quality: int | None = 75,
        queue_size: int = 64,
        flush_timeout: float = 10.0,
    ) -> None:
        if spawn and connect_url is not None:
            raise ValueError("spawn and connect_url are mutually exclusive")
        if recording_path is not None and recording_dir is not None:
            raise ValueError("recording_path and recording_dir are mutually exclusive")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        if not 1 <= spawn_port <= 65535:
            raise ValueError(f"spawn_port must be in 1-65535, got {spawn_port}")
        if not save:
            recording_path = None
            recording_dir = None
        elif recording_path is None and recording_dir is None:
            recording_dir = "rerun"
        if recording_path is None and recording_dir is None and not spawn and connect_url is None:
            raise ValueError("RerunView needs a viewer or a file")

        self._recording_path = None if recording_path is None else Path(recording_path)
        self._recording_dir = None if recording_dir is None else Path(recording_dir)
        self.spawn = spawn
        self.connect_url = connect_url
        self._spawn_port = spawn_port
        self._jpeg_quality = jpeg_quality
        self._queue_size = queue_size
        self._watermark = max(1, queue_size // 4)
        self._flush_timeout = flush_timeout
        self.resolved_recording_path: Path | None = None
        self._obs_space: dict[str, Any] = {}
        self._action_names: list[str] = []
        self._queue: deque[_Step] = deque()
        self._cond = threading.Condition()
        self._stop = False
        self._closed = False
        self._failed = False
        self._worker: threading.Thread | None = None
        self._rr: Any = None
        # One stream per destination. This rerun-sdk has no set_sinks, so a
        # viewer and a file are two recordings of the same ticks.
        self._recs: list[Any] = []
        self._blueprint_sent = False
        self._dropped_frames = 0
        self._dropped_steps = 0
        self._warned = False

    def bind(self, obs_space: dict[str, Any], action_space: dict[str, Any]) -> None:
        """Keep the contract labels used to name state and action plots."""
        self._obs_space = obs_space
        names = action_space.get("names")
        self._action_names = [str(name) for name in names] if isinstance(names, list) else []

    def start(self) -> None:
        """Open the recording on a background thread. A closed view can start again."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._closed = False
        self._stop = False
        self._failed = False
        self._warned = False
        self._blueprint_sent = False
        self._recs = []
        self._rr = None
        with self._cond:
            self._queue.clear()
        self.resolved_recording_path = self._allocate_path()
        self._worker = threading.Thread(target=self._run, name="hud-rerun", daemon=True)
        self._worker.start()

    def log(
        self,
        *,
        tick: int,
        data: dict[str, Any],
        action: Any = None,
        reward: Any = None,
        terminated: bool = False,
    ) -> None:
        """Queue one tick. Copies ``data`` before returning."""
        if self._closed or self._failed:
            return
        self._enqueue(
            _Step(
                tick=tick,
                images=self._images(data),
                state=self._state(data),
                action=None if action is None else self._action(action),
                reward=_as_float(reward),
                terminated=terminated,
            )
        )

    def close(self) -> None:
        """Drain the queue, flush the file, and disconnect. Bounded wait."""
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        with self._cond:
            self._stop = True
            self._cond.notify()
        worker.join(self._flush_timeout)
        if worker.is_alive():
            logger.warning("RerunView shutdown timed out; the viewer connection looks stalled")
        if self._dropped_frames or self._dropped_steps:
            logger.warning(
                "RerunView dropped %s camera frame(s) and %s step(s) to keep control moving",
                self._dropped_frames,
                self._dropped_steps,
            )

    def _allocate_path(self) -> Path | None:
        if self._recording_path is not None:
            self._recording_path.parent.mkdir(parents=True, exist_ok=True)
            return self._recording_path
        if self._recording_dir is None:
            return None
        self._recording_dir.mkdir(parents=True, exist_ok=True)
        return self._recording_dir / f"episode_{uuid4().hex[:8]}.rrd"

    def _images(self, data: dict[str, Any]) -> tuple[tuple[str, NDArray[Any]], ...]:
        frames: list[tuple[str, NDArray[Any]]] = []
        for name, value in data.items():
            feature = self._obs_space.get(name)
            typed = isinstance(feature, dict) and feature.get("type") in _IMAGE_TYPES
            if not typed and getattr(value, "ndim", 0) < 3:
                continue
            frames.append((str(name), np.array(value, copy=True)))
        return tuple(frames)

    def _state(self, data: dict[str, Any]) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        numeric = {
            name: value
            for name, value in data.items()
            if getattr(value, "ndim", -1) < 2 and hasattr(value, "ndim")
        }
        if not numeric:
            return ()
        observed = ObservationStep.from_obs({"data": numeric}, obs_space=self._obs_space)
        rows: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        for group, feature in observed.state.items():
            labels = feature.names if len(feature.names) == len(feature.values) else []
            rows.append(
                (
                    group,
                    tuple(
                        (labels[i] if i < len(labels) else str(i), float(value))
                        for i, value in enumerate(feature.values)
                    ),
                )
            )
        return tuple(rows)

    def _action(self, action: Any) -> tuple[tuple[str, float], ...]:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        names = self._action_names if len(self._action_names) == len(values) else []
        return tuple(
            (names[i] if i < len(names) else str(i), float(value)) for i, value in enumerate(values)
        )

    def _enqueue(self, step: _Step) -> None:
        with self._cond:
            if self._closed or self._failed:
                return
            if step.images and len(self._queue) >= self._watermark:
                self._dropped_frames += len(step.images)
                step = replace(step, images=())
            if len(self._queue) >= self._queue_size:
                evicted = self._queue.popleft()
                self._dropped_steps += 1
                self._dropped_frames += len(evicted.images)
            self._queue.append(step)
            self._cond.notify()

    def _run(self) -> None:
        self._open()
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if not self._queue:
                    break
                step = self._queue.popleft()
            if not self._recs:
                continue
            try:
                self._emit(step)
            except Exception as exc:
                if not self._warned:
                    self._warned = True
                    logger.warning("RerunView failed to log a step (%s)", exc)
        for rec in self._recs:
            try:
                rec.flush(blocking=True)
                rec.disconnect()
            except Exception as exc:
                logger.warning("RerunView flush failed: %s", exc)

    def _open(self) -> None:
        try:
            import rerun as rr
        except ImportError:
            if not self._warned:
                self._warned = True
                logger.warning("rerun-sdk is not installed; RerunView is off")
            self._failed = True
            return
        try:
            self._recs = self._open_streams(rr)
        except Exception as exc:
            self._failed = True
            self._recs = []
            logger.warning("RerunView disabled: %s", exc)
            return
        self._rr = rr

    def _open_streams(self, rr: Any) -> list[Any]:
        recs: list[Any] = []
        path = self.resolved_recording_path
        if path is not None:
            rec = rr.RecordingStream("hud_robot", make_default=False, make_thread_default=False)
            rec.save(path)
            recs.append(rec)
        if self.spawn or self.connect_url is not None:
            live = rr.RecordingStream("hud_robot", make_default=False, make_thread_default=False)
            try:
                if self.spawn:
                    live.spawn(port=self._spawn_port, memory_limit="2GiB")
                elif self.connect_url is not None:
                    live.connect_grpc(self.connect_url)
            except Exception as exc:
                if not recs:
                    raise
                logger.warning("Rerun viewer did not start (%s); recording %s", exc, path)
            else:
                recs.append(live)
        if not recs:
            raise RuntimeError("RerunView has no viewer and no file")
        return recs

    def _emit(self, step: _Step) -> None:
        rr = self._rr
        if not self._blueprint_sent:
            self._blueprint_sent = True
            for rec in self._recs:
                self._send_blueprint(rr, rec, step)
        frames = {name: self._encoded(rr, frame) for name, frame in step.images}
        for rec in self._recs:
            rec.set_time("step", sequence=step.tick)
            for name, frame in frames.items():
                rec.log(f"camera/{name}", frame)
            for group, rows in step.state:
                for label, value in rows:
                    rec.log(f"state/{group}/{label}", rr.Scalars(value))
            if step.action is not None:
                for label, value in step.action:
                    rec.log(f"action/{label}", rr.Scalars(value))
            if step.reward is not None:
                rec.log("reward", rr.Scalars(step.reward))
            if step.terminated:
                rec.log("event/terminated", rr.TextLog("terminated"))

    def _encoded(self, rr: Any, image: NDArray[Any]) -> Any:
        logged = rr.Image(np.ascontiguousarray(image))
        if self._jpeg_quality is None:
            return logged
        try:
            return logged.compress(jpeg_quality=self._jpeg_quality)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning("RerunView logging raw frames (%s)", exc)
            return logged

    def _send_blueprint(self, rr: Any, rec: Any, step: _Step) -> None:
        try:
            import rerun.blueprint as rrb
        except ImportError:
            return
        cameras = [rrb.Spatial2DView(name=name, origin=f"camera/{name}") for name, _ in step.images]
        plots = [rrb.TimeSeriesView(name="action", origin="action")]
        plots.extend(
            rrb.TimeSeriesView(name=group, origin=f"state/{group}") for group, _ in step.state
        )
        plots.append(rrb.TimeSeriesView(name="reward", origin="reward"))
        row = rrb.Horizontal(*plots)
        layout = rrb.Vertical(rrb.Horizontal(*cameras), row) if cameras else row
        try:
            rec.send_blueprint(rrb.Blueprint(layout))
        except Exception as exc:
            logger.warning("RerunView kept the automatic layout (%s)", exc)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.asarray(value).reshape(-1)[0])


__all__ = ["RerunView"]

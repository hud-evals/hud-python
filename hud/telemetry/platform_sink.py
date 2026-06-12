"""``PlatformTraceSink``: stream the env-side tick stream to the HUD platform.

The env-side counterpart of the agent-side :class:`~hud.agents.robot.tracer.RobotTracer`:

- the **agent** stream carries what the *policy* saw (its inputs, its action
  chunks, keyframes) — emitted by ``RobotTracer`` inside the agent process;
- the **env** stream (this sink) carries what the *simulator executed* — every
  control tick's ``(observation, action, reward, done)``, i.e. exactly the data
  the LeRobot dataset sink persists, but shipped live as platform spans.

It plugs into the same :class:`~hud.telemetry.recorder.EpisodeRecorder` seam as
:class:`~hud.environment.robots.data_saving.LeRobotTraceSink`, so an env records to
disk and streams to the platform from **one recorder** with one obs copy per tick::

    EpisodeRecorder(LeRobotTraceSink(...), PlatformTraceSink(env_name="libero"))

All work runs on the recorder's worker thread (never the env control loop), and
each span is handed to the batching exporter (:func:`hud.telemetry.exporter.queue_span`),
which uploads fire-and-forget on its own worker — so a slow network never stalls
the sibling dataset sink for long, and never the sim at all.

Trace attribution: spans need the rollout's ``trace_id``. Agent-side this comes
from the ambient trace context; an env may run in a *separate process* where no
context exists. This sink therefore reads ``trace_id`` from the episode-start
meta (``recorder.start_episode(trace_id=...)``) and falls back to the ambient
context (covers in-process ``LocalSandbox`` runs). Episodes with no resolvable
trace id are skipped silently. Propagating the trace id over the control channel
(``tasks.start``) is the known follow-up for cross-process attribution.
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .recorder import TraceSink

if TYPE_CHECKING:
    from .recorder import Frame

logger = logging.getLogger(__name__)

#: Per-tick frames ride every span at the control rate: keep them small.
_TICK_IMAGE_PX = 160
_TICK_JPEG_QUALITY = 55


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_trace_id(trace_id: str) -> str:
    clean = trace_id.replace("-", "")
    return clean[:32].ljust(32, "0")


def _encode_hwc(arr: np.ndarray, *, max_px: int, quality: int) -> str | None:
    """uint8 HWC camera frame -> downsampled base64 JPEG data URL."""
    try:
        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        if max(img.size) > max_px:
            scale = max_px / max(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.debug("platform sink: could not encode frame", exc_info=True)
        return None


def _obs_images(obs: dict[str, np.ndarray]) -> dict[str, str]:
    """Encode every camera-like array in the obs dict -> ``{name: data_url}``.

    Cameras are recognized structurally (3-dim uint8 HWC with 3 channels), so the
    sink needs no contract knowledge.
    """
    out: dict[str, str] = {}
    for name, value in obs.items():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
            enc = _encode_hwc(arr, max_px=_TICK_IMAGE_PX, quality=_TICK_JPEG_QUALITY)
            if enc is not None:
                out[name] = enc
    return out


class PlatformTraceSink(TraceSink):
    """Emit one platform span per executed env tick (plus an episode summary).

    Construct once per env process; per-episode state (trace id, prompt, step
    counter) resets on ``on_episode_start``. Never raises into the recorder:
    emission failures are logged and swallowed (and the recorder isolates sink
    failures anyway).
    """

    def __init__(self, *, env_name: str | None = None) -> None:
        self._env = env_name
        self._trace_id: str | None = None
        self._prompt: str | None = None
        self._meta: dict[str, Any] = {}
        self._step = 0

    # ── TraceSink ──────────────────────────────────────────────────────────

    def on_episode_start(self, meta: dict[str, Any]) -> None:
        self._step = 0
        self._prompt = meta.get("prompt")
        self._trace_id = meta.get("trace_id") or self._ambient_trace_id()
        # Everything else in the start meta is the task args — keep for labeling.
        self._meta = {
            k: v for k, v in meta.items() if k not in ("prompt", "trace_id")
        }
        if self._trace_id is None:
            logger.debug("platform sink: no trace_id for episode; skipping stream")

    def on_frame(self, frame: Frame) -> None:
        if self._trace_id is None or not self._enabled():
            return
        try:
            from hud.agents.robot.tracer import camera_content  # noqa: PLC0415

            now = _now_iso()
            request: dict[str, Any] = {"step": self._step, "prompt": self._prompt}
            if self._env or self._meta:
                request["meta"] = {
                    **({"env": self._env} if self._env else {}),
                    **({"task_args": self._meta} if self._meta else {}),
                }
            images = _obs_images(frame.obs)
            if images:
                # Same wire shape as the agent-side RobotTracer: frames ride the
                # messages-content path the platform offloads to S3 + presigns.
                request["messages"] = [{"role": "robot", "content": camera_content(images)}]
            result: dict[str, Any] = {
                # float64 before round: float32 values would re-acquire
                # representation noise (0.10000000149...) in the JSON.
                "action": np.asarray(frame.action, dtype=np.float64)
                .round(4)
                .reshape(-1)
                .tolist(),
                "reward": float(frame.reward),
                "done": bool(frame.done),
            }
            if frame.info:
                result["info"] = frame.info
            self._queue("robot.tick", request, result, now)
        except Exception:
            logger.debug("platform sink: tick emission failed", exc_info=True)
        finally:
            self._step += 1

    def on_episode_end(self, meta: dict[str, Any]) -> None:
        if self._trace_id is None or not self._enabled():
            return
        try:
            now = _now_iso()
            self._queue(
                "robot.episode",
                {"prompt": self._prompt, "steps": self._step},
                dict(meta),  # success / total_reward / any extras from endpoint.result()
                now,
            )
        except Exception:
            logger.debug("platform sink: episode emission failed", exc_info=True)

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _enabled() -> bool:
        from hud.settings import settings  # noqa: PLC0415

        # Mirror RobotTracer: skip even the JPEG encode when the platform isn't
        # configured (queue_span would drop the span anyway).
        return bool(settings.telemetry_enabled and settings.api_key)

    @staticmethod
    def _ambient_trace_id() -> str | None:
        try:
            from hud.telemetry.context import get_current_trace_id  # noqa: PLC0415

            return get_current_trace_id()
        except Exception:
            return None

    def _queue(
        self, name: str, request: dict[str, Any], result: dict[str, Any], now: str
    ) -> None:
        from hud.telemetry.exporter import queue_span  # noqa: PLC0415
        from hud.types import TraceStep  # noqa: PLC0415

        assert self._trace_id is not None
        attributes = TraceStep(
            task_run_id=self._trace_id,
            category="robot",
            type="CLIENT",
            request=request,
            result=result,
            start_timestamp=now,
            end_timestamp=now,
        )
        queue_span(
            {
                "name": name,
                "trace_id": _normalize_trace_id(self._trace_id),
                "span_id": uuid.uuid4().hex[:16],
                "parent_span_id": None,
                "start_time": now,
                "end_time": now,
                "status_code": "OK",
                "status_message": None,
                "attributes": attributes.model_dump(mode="json", exclude_none=True),
                "exceptions": None,
            }
        )


__all__ = ["PlatformTraceSink"]

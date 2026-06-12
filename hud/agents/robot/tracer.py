"""``RobotTracer``: agent-side per-step trace spans with keyframe stamps.

Emits one ``robot.step`` span per env step through ``hud.telemetry`` so rollouts
stream live into the platform viewer. Each span carries small JPEGs of every
camera the policy saw plus the executed action; steps with a fresh action chunk
are stamped ``keyframe: true`` with full-res frames — the viewer's timeline
markers. Spans ship fire-and-forget; emission never blocks and never raises.
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger("hud.agents.robot.tracer")

#: Per-step frames: small + cheap (these dominate trace size at 10 Hz).
_STEP_IMAGE_PX = 160
_STEP_JPEG_QUALITY = 55
#: Keyframe (fresh-chunk) frames: full resolution for the decision-point record.
_KEY_IMAGE_PX = 256
_KEY_JPEG_QUALITY = 70


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_trace_id(trace_id: str) -> str:
    clean = trace_id.replace("-", "")
    return clean[:32].ljust(32, "0")


def camera_content(images: dict[str, str]) -> list[dict[str, Any]]:
    """``{camera: data_url}`` -> ``image_url`` content items (artifact-pipeline shape).

    The platform ingest walks ``request.messages[].content[]`` for ``image_url``
    items, offloads the base64 payload to S3, and presigns it on the read path —
    so frames never bloat the stored span. The extra ``camera`` key survives the
    round trip and names the stream in the viewer.
    """
    return [
        {"type": "image_url", "camera": name, "image_url": {"url": url}}
        for name, url in images.items()
    ]


def _encode_chw(value: Any, *, max_px: int, quality: int) -> str | None:
    """CHW float tensor in [0, 1] -> downsampled base64 JPEG data URL."""
    from PIL import Image

    hwc = (value.detach().cpu().float().clamp(0, 1) * 255).byte()
    img = Image.fromarray(hwc.permute(1, 2, 0).numpy())
    if max(img.size) > max_px:
        scale = max_px / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _batch_images(batch: dict[str, Any], *, max_px: int, quality: int) -> dict[str, str]:
    """Encode *every* camera stream in a policy batch -> ``{camera_name: data_url}``.

    Adapter batches carry one CHW float tensor per camera (e.g. ``image`` scene +
    ``image2`` wrist for pi0.5), keyed by the feature's last name segment, in
    batch (camera) order.
    """
    out: dict[str, str] = {}
    try:
        import torch

        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 3:
                name = key.rsplit(".", 1)[-1]
                enc = _encode_chw(value, max_px=max_px, quality=quality)
                if enc is not None:
                    out[name] = enc
    except Exception:
        logger.debug("tracer: could not encode batch images", exc_info=True)
    return out


class RobotTracer:
    """Emit one platform span per env step, keyframe-stamped at fresh chunks.

    Construct **one per agent**: ``model`` / ``env`` are fixed at construction,
    while ``set_episode`` updates the current task each rollout. Each span carries
    this as ``request.meta`` so the viewer can label the run. The ``trace_id`` is
    read from the ambient trace context at emit time, so spans always attribute to
    the rollout whose task is running.
    """

    def __init__(self, *, model: str | None = None, env: str | None = None) -> None:
        self._model = model
        self._env = env
        self._task: str | None = None
        self._args: dict[str, Any] | None = None

    def set_episode(self, *, task: str | None = None, args: dict[str, Any] | None = None) -> None:
        """Set the current rollout's task id + params (call once per episode)."""
        self._task = task
        self._args = dict(args) if args else None

    def _meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        if self._model:
            meta["model"] = self._model
        if self._env:
            meta["env"] = self._env
        if self._task:
            meta["task"] = self._task
        if self._args:
            meta["task_args"] = self._args
        return meta

    def emit_step(
        self,
        batch: dict[str, Any],
        action: np.ndarray,
        *,
        step: int,
        keyframe: bool = False,
        chunk: np.ndarray | None = None,
        chunk_len: int | None = None,
    ) -> None:
        """Record one env step: what the model saw and the action executed.

        ``keyframe=True`` marks a fresh-chunk inference step — pass the full
        ``chunk`` then (or at least ``chunk_len`` when only the horizon is
        known) so the decision-point record is complete. Fire-and-forget;
        any failure is logged and swallowed.
        """
        try:
            from hud.settings import settings
            from hud.telemetry.context import get_current_trace_id
            from hud.telemetry.exporter import queue_span
            from hud.types import TraceStep

            if not (settings.telemetry_enabled and settings.api_key):
                return  # platform not configured — skip even the JPEG encode
            trace_id = get_current_trace_id()
            if not trace_id:
                return  # not inside a rollout (e.g. warmup) — nothing to attribute to

            now = _now_iso()
            if keyframe:
                images = _batch_images(batch, max_px=_KEY_IMAGE_PX, quality=_KEY_JPEG_QUALITY)
            else:
                images = _batch_images(batch, max_px=_STEP_IMAGE_PX, quality=_STEP_JPEG_QUALITY)

            request: dict[str, Any] = {
                "prompt": batch.get("task"),
                "step": step,
                "keyframe": bool(keyframe),
            }
            meta = self._meta()
            if meta:
                request["meta"] = meta  # model / env / task / task_args — for the viewer
            if images:
                # Camera frames as messages-content image items: the platform's
                # artifact pipeline offloads these to S3 at ingest and presigns
                # them on read, so the viewer gets URLs, not inline base64.
                request["messages"] = [{"role": "robot", "content": camera_content(images)}]

            result: dict[str, Any] = {
                # float64 before round: float32 values would re-acquire
                # representation noise (0.10000000149...) in the JSON.
                "action": np.asarray(action, dtype=np.float64).round(4).reshape(-1).tolist(),
            }
            if keyframe:
                if chunk is not None:
                    arr = np.asarray(chunk, dtype=np.float64)
                    result["chunk_len"] = int(arr.shape[0]) if arr.ndim >= 1 else 1
                    result["action_dim"] = int(arr.shape[-1]) if arr.ndim >= 1 else int(arr.size)
                    result["chunk"] = arr.round(4).tolist()
                elif chunk_len is not None:
                    result["chunk_len"] = int(chunk_len)
                    result["action_dim"] = int(np.asarray(action).size)

            attributes = TraceStep(
                task_run_id=trace_id,
                category="robot",
                type="CLIENT",
                request=request,
                result=result,
                start_timestamp=now,
                end_timestamp=now,
            )
            queue_span(
                {
                    "name": "robot.step",
                    "trace_id": _normalize_trace_id(trace_id),
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
        except Exception:
            logger.debug("tracer: span emission failed", exc_info=True)


__all__ = ["RobotTracer", "camera_content"]

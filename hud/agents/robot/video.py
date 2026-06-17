"""Per-camera H.264/CMAF video streaming for robot traces.

:class:`SegmentEncoder` encodes one camera's frames into fragmented-MP4 (CMAF) on a
background thread and hands each finished segment to a callback. :class:`VideoStreamer`
fans a whole observation out across one encoder per camera and emits the segments as
``VideoSegmentStep`` spans, so the trace viewer plays one ``<video>`` per camera.

Encoding never blocks the act loop: ``submit`` is a non-blocking put on a bounded queue
that drops frames under backpressure, and PyAV releases the GIL inside the codec.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# type alias for SegmentCallback function - takes in index and data.
# Called on the encoder thread.
SegmentCallback = Callable[[int, bytes], None]


class SegmentEncoder:
    """Encode one camera's (each camera ges its own) frames to CMAF, dispatching the init segment then one media
    fragment per ~``segment_seconds`` via ``on_segment`` (called on the encoder thread).

    Doubles as the file-like sink PyAV muxes into: ``write`` accumulates bytes and
    dispatches each complete top-level MP4 box as soon as it is whole.
    """

    def __init__(
        self,
        camera: str,
        on_segment: SegmentCallback, # called on each finished segment
        *,
        fps: int,
        segment_seconds: float = 2.0, # how many secs of video per segment
        crf: int = 23, # quality of the video: 0 is best quality, 51 is worst quality (23 is middle quality)
        max_queued_frames: int = 16,
    ) -> None:
        self.camera = camera
        self.fps = max(1, int(fps))
        self._on_segment = on_segment
        self._gop = max(1, round(self.fps * segment_seconds))  # keyframe interval in # of "frames"
        self._crf = int(crf)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(max_queued_frames)
        # Box-assembly state, touched only on the encoder thread.
        self._buf = bytearray()
        self._pos = self._scan = 0 # position in the buffer and the scan position
        self._index = 0 # counter for the number of segments emitted
        self._init_sent = False # flag to indicate if the init segment has been sent
        self._pending = b"" # buffer for the pending data
        self._thread = threading.Thread(
            target=self._run, name=f"hud-robot-video-{camera}", daemon=True
        )
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Hand one frame to the encoder; non-blocking, dropping under backpressure."""
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(np.array(frame, copy=True)) # NOTE drops under backpressure

    def finalize(self, timeout: float = 15.0) -> None:
        """Called on episode end to flush the tail fragment and stop the encoder thread (best-effort)."""
        try:
            self._queue.put_nowait(None) # tries to drop item in mailbox; if queue is full, raises queue.Full
        except queue.Full:  # make room for the stop sentinel rather than hang
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            self._queue.put_nowait(None)
        self._thread.join(timeout)

    # ── file-like sink (encoder thread) ────────────────────────────────────────

    def write(self, b: bytes) -> int:
        """Called by PyAV to write bytes to the buffer."""
        # 1. drop the incoming bytes into the buffer at the current write position
        end = self._pos + len(b)
        if end > len(self._buf):
            self._buf.extend(b"\x00" * (end - len(self._buf)))
        self._buf[self._pos : end] = b
        self._pos = end
        # 2. carve the stream into MP4 boxes and group them into segments:
        # ftyp+moov form the init segment (index 0); each moof+mdat is one fragment.
        while len(self._buf) - self._scan >= 8:
            # read the next box's size + type from its 8-byte header
            size = int.from_bytes(self._buf[self._scan : self._scan + 4], "big")
            btype = bytes(self._buf[self._scan + 4 : self._scan + 8])
            if size < 8 or len(self._buf) - self._scan < size:
                break  # box header/body not fully written yet
            box = bytes(self._buf[self._scan : self._scan + size])
            self._scan += size
            # first moof closes the init segment → ship ftyp+moov, then start a fragment
            if btype == b"moof" and not self._init_sent:
                self._dispatch(self._pending)
                self._init_sent, self._pending = True, b""
            self._pending += box
            # mdat ends a fragment → ship this moof+mdat as one segment
            if self._init_sent and btype == b"mdat":
                self._dispatch(self._pending)
                self._pending = b""
        return len(b) # return the number of bytes written

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = (0, self._pos, len(self._buf))[whence] + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def flush(self) -> None:  # PyAV/ffmpeg may call flush()
        pass

    def _dispatch(self, data: bytes) -> None:
        if not data:
            return
        try:
            self._on_segment(self._index, data)
        except Exception:  # a bad dispatch must not kill encoding
            logger.warning("video segment dispatch failed (camera %s)", self.camera, exc_info=True)
        self._index += 1

    def _run(self) -> None:
        from fractions import Fraction

        container = stream = None
        n = 0 # counts frames actually encoded
        try:
            import av  # pyright: ignore[reportMissingImports]

            while (arr := self._queue.get()) is not None:
                frame = _to_rgb24(arr)
                if frame is None:
                    continue
                if container is None: # first frame -> open the container
                    h, w = frame.shape[:2]
                    container = av.open(
                        self,
                        mode="w",
                        format="mp4",
                        options={"movflags": "+frag_keyframe+empty_moov+default_base_moof"},
                    )
                    stream = container.add_stream("libx264", rate=self.fps)
                    stream.width, stream.height = w, h
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.time_base = Fraction(1, self.fps)
                    # Fixed GOP (scenecut off) → each fragment is a closed, seekable GOP;
                    # pinned Main/3.0 so the viewer's MSE codec string is fixed (avc1.4d401e).
                    stream.codec_context.options = {
                        "preset": "veryfast",
                        "tune": "zerolatency",
                        "profile": "main",
                        "level": "3.0",
                        "crf": str(self._crf),
                        "x264-params": f"keyint={self._gop}:min-keyint={self._gop}:scenecut=0",
                    }
                vframe = av.VideoFrame.from_ndarray(frame, format="rgb24")
                vframe.pts, vframe.time_base = n, Fraction(1, self.fps)
                for packet in stream.encode(vframe):
                    container.mux(packet)
                n += 1
        except Exception:  # isolate encoder faults from the rollout
            logger.warning("video encode failed (camera %s)", self.camera, exc_info=True)
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    for packet in stream.encode(None):  # flush, writing the final fragment
                        container.mux(packet)
                    container.close()


class VideoStreamer:
    """Per-run camera→video fan-out: one :class:`SegmentEncoder` (and thread) per camera,
    each emitting finished segments as ``VideoSegmentStep`` spans. ``trace_id`` is captured
    in the rollout's trace context so encoder threads can attribute their spans.
    """

    def __init__(self, *, fps: int, trace_id: str | None) -> None:
        try:
            import av  # noqa: F401  # pyright: ignore[reportMissingImports]
        except Exception as exc:
            raise RuntimeError(
                "robot video streaming requires PyAV — `pip install 'hud-python[robot]'`"
            ) from exc
        self._fps = fps
        self._trace_id = trace_id
        self._encoders: dict[str, SegmentEncoder] = {}

    def record(self, obs: dict[str, Any]) -> None:
        """Submit each camera frame in ``obs['data']`` to its (lazy) encoder. Non-blocking."""
        for name, arr in obs.get("data", {}).items():
            if getattr(arr, "ndim", 0) < 2:
                continue
            if name not in self._encoders:
                self._encoders[name] = self._make_encoder(name)
            self._encoders[name].submit(arr)

    def finalize(self) -> None:
        """Flush every camera's tail fragment at teardown (best-effort)."""
        for encoder in self._encoders.values():
            with contextlib.suppress(Exception):  # teardown must not mask the run result
                encoder.finalize()

    def _make_encoder(self, camera: str) -> SegmentEncoder:
        from hud.agents.types import VideoSegmentStep

        trace_id, fps = self._trace_id, self._fps

        def on_segment(index: int, data: bytes) -> None:
            VideoSegmentStep(
                camera=camera,
                index=index,
                fps=fps,
                segment={
                    "type": "video",
                    "data": base64.b64encode(data).decode("ascii"),
                    "mimeType": "video/mp4",
                },
            ).emit(trace_id=trace_id)

        return SegmentEncoder(camera, on_segment, fps=fps)


def _to_rgb24(arr: np.ndarray) -> np.ndarray | None:
    """Coerce a raw camera array to contiguous HxWx3 uint8 with even dims
    (yuv420p needs even width/height). Returns ``None`` if it isn't an image."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if h % 2 or w % 2:
        arr = arr[: h - (h % 2), : w - (w % 2)]
    return np.ascontiguousarray(arr)


__all__ = ["SegmentEncoder", "VideoStreamer"]

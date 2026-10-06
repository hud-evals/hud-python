"""Tests for the robot trace video encoder."""

from __future__ import annotations

import io
import threading

import av
import numpy as np

from hud.telemetry.robot.video import SegmentEncoder


def test_a_lossless_encoder_keeps_every_frame_while_its_queue_is_full() -> None:
    frames = 60
    segments: list[bytes] = []
    released = threading.Event()

    def on_segment(_index: int, data: bytes) -> None:
        released.wait()  # stall the encoder so its one-frame queue stays full
        segments.append(data)

    encoder = SegmentEncoder("cam", on_segment, fps=10, max_queued_frames=1, lossless=True)
    threading.Timer(0.3, released.set).start()
    for value in range(frames):
        encoder.submit(np.full((16, 16, 3), value, dtype=np.uint8))
    encoder.finalize()

    with av.open(io.BytesIO(b"".join(segments)), mode="r") as container:
        assert sum(1 for _ in container.decode(video=0)) == frames

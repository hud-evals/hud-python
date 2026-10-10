"""The robot trace video encoder: lossless frame keeping, and teardown that cannot hang."""

from __future__ import annotations

import io
import threading
import time
from typing import TYPE_CHECKING

import av
import numpy as np

from hud.telemetry.robot.video import SegmentEncoder

if TYPE_CHECKING:
    import pytest


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


def test_a_lossy_encoder_finalizes_on_time_with_a_full_queue_and_a_stalled_encoder() -> None:
    released = threading.Event()

    def stall(_index: int, _data: bytes) -> None:
        released.wait()

    encoder = SegmentEncoder(
        "cam",
        stall,
        fps=10,
        segment_seconds=0.1,
        max_queued_frames=1,
    )
    for value in range(40):
        encoder.submit(np.full((16, 16, 3), value, dtype=np.uint8))

    started = time.monotonic()
    encoder.finalize(timeout=0.5)
    elapsed = time.monotonic() - started
    released.set()

    assert elapsed < 5


def test_a_lossless_submit_stops_waiting_once_the_encoder_has_died(
    caplog: pytest.LogCaptureFixture,
) -> None:
    encoder = SegmentEncoder(
        "cam", lambda _index, _data: None, fps=10, max_queued_frames=1, lossless=True
    )
    finished = threading.Event()

    def feed() -> None:
        # An empty image fails the encoder, which ends its thread with the queue unread.
        encoder.submit(np.zeros((0, 16, 3), dtype=np.uint8))
        for _ in range(20):
            encoder.submit(np.zeros((16, 16, 3), dtype=np.uint8))
        encoder.finalize(timeout=1)
        finished.set()

    threading.Thread(target=feed, daemon=True).start()

    assert finished.wait(timeout=30)
    assert "video encode failed (camera cam)" in caplog.text

"""Shared robot telemetry: trace/job recorders + per-camera H.264 video streaming.

Imported by both sides of the robot stack (agent harness, ``hud.wrap``, gym
bridges); needs the ``robot`` extra (numpy, PyAV).
"""

from __future__ import annotations

from .recorder import JobRecorder, TraceRecorder, to_numpy
from .video import SegmentEncoder, VideoStreamer

__all__ = ["JobRecorder", "SegmentEncoder", "TraceRecorder", "VideoStreamer", "to_numpy"]

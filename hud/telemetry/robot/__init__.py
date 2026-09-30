"""Shared robot telemetry: trace/job recorders, H.264 video, optional Rerun.

Imported by both sides of the robot stack (agent harness, ``wrap``, gym
bridges); needs the ``robot`` extra (numpy, PyAV, rerun-sdk).
"""

from __future__ import annotations

from .recorder import JobRecorder, TraceRecorder, to_numpy
from .rerun import RerunView
from .video import SegmentEncoder, VideoStreamer

__all__ = [
    "JobRecorder",
    "RerunView",
    "SegmentEncoder",
    "TraceRecorder",
    "VideoStreamer",
    "to_numpy",
]

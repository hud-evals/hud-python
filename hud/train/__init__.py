"""HUD training (RL) client surface."""

from __future__ import annotations

from hud.train.client import TrainingClient
from hud.train.types import (
    BuiltinLoss,
    CheckpointResponse,
    DatumTensors,
    ForwardBackwardResult,
    ForwardResult,
    OptimStepResult,
    TrainInput,
    TrajectoryPayload,
    TrajectorySample,
)

__all__ = [
    "BuiltinLoss",
    "CheckpointResponse",
    "DatumTensors",
    "ForwardBackwardResult",
    "ForwardResult",
    "OptimStepResult",
    "TrainInput",
    "TrainingClient",
    "TrajectoryPayload",
    "TrajectorySample",
]

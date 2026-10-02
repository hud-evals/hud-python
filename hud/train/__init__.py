"""HUD training (RL) client surface.

Drive training for a model through the HUD training service. Each call takes a
mix of recorded trajectories (by ``trace_id``) and local graded traces
(sent inline). Built-in losses run server-side; custom losses run client-side
via ``TrainingClient.forward_backward_custom``. See ``TrainingClient``.
"""

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

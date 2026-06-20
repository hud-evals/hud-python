"""HUD training (RL) client surface.

Drive training for a model through the HUD training service. Each call takes a
mix of recorded trajectories (by ``trace_id``) and local :class:`hud.Run`s
(sent inline). Built-in losses run server-side; custom losses run client-side
via :meth:`TrainingClient.forward_backward_custom`. See :class:`TrainingClient`.
"""

from __future__ import annotations

from hud.train.base import BaseTrainingClient
from hud.train.client import TrainingClient
from hud.train.types import (
    BackwardRequest,
    BuiltinLoss,
    CheckpointResponse,
    DatumTensors,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LossFn,
    OptimStepRequest,
    OptimStepResult,
    TrainingDatum,
    TrainInput,
    TrajectoryPayload,
    TrajectorySample,
)

__all__ = [
    "BackwardRequest",
    "BaseTrainingClient",
    "BuiltinLoss",
    "CheckpointResponse",
    "DatumTensors",
    "ForwardBackwardRequest",
    "ForwardBackwardResult",
    "ForwardRequest",
    "ForwardResult",
    "LossFn",
    "OptimStepRequest",
    "OptimStepResult",
    "TrainInput",
    "TrainingClient",
    "TrainingDatum",
    "TrajectoryPayload",
    "TrajectorySample",
]

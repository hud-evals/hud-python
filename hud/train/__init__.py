"""HUD training (RL) client surface, removed in hud v0.7.0 with the ``train`` extra.

Each name warns when imported from here.
"""

from __future__ import annotations

from typing import Any

from hud.utils.deprecation import TRAINING_REPLACEMENT, resolve_retired, served_from

_RETIRED = {
    **served_from("hud.train.client", "TrainingClient", instead=TRAINING_REPLACEMENT),
    **served_from(
        "hud.train.types",
        "BuiltinLoss",
        "CheckpointResponse",
        "DatumTensors",
        "ForwardBackwardResult",
        "ForwardResult",
        "OptimStepResult",
        "TrainInput",
        "TrajectoryPayload",
        instead=TRAINING_REPLACEMENT,
    ),
}

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
]


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)

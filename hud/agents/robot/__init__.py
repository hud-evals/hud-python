"""Agent-side robot harness: drive a ``robot`` env with a VLA policy.

- ``RobotAgent`` — the harness: connects to the ``robot``
  capability (claiming a slot token from the setup step when present), reads the contract,
  drives one scalar connection with an open-loop chunk queue. Subclass and
  set ``self.model`` + ``self.adapter``.
- ``Model`` / ``LeRobotModel`` /
  ``RemoteModel`` — the policy and its inference mechanics.
- ``Adapter`` / ``LeRobotAdapter`` /
  ``OpenPIAdapter`` — env <-> policy space translation.
- ``BatchedAgent`` — many concurrent single-env rollouts
  sharing one batched model.
- ``DatasetWriter`` — opt-in LeRobot v3 dataset recording
  (``agent.save = True``).

This subpackage needs the ``robot`` extra (``pip install 'hud[robot]'``).
"""

from __future__ import annotations

from .adapter import Adapter, LeRobotAdapter, OpenPIAdapter
from .agent import RobotAgent
from .batching import BatchedAgent, BatchedModel
from .dataset import DatasetWriter
from .model import LeRobotModel, Model, RemoteModel

__all__ = [
    "Adapter",
    "BatchedAgent",
    "BatchedModel",
    "DatasetWriter",
    "LeRobotAdapter",
    "LeRobotModel",
    "Model",
    "OpenPIAdapter",
    "RemoteModel",
    "RobotAgent",
]

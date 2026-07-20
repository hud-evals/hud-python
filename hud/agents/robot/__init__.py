"""Agent-side robot harness: drive a ``robot`` env with a VLA policy.

- :class:`~.agent.RobotAgent` — the harness: connects to the ``robot``
  capability (claiming a slot token from ``run.started``), reads the contract,
  drives one scalar connection with an open-loop chunk queue. Subclass and
  set ``self.model`` + ``self.adapter``.
- :class:`~.model.Model` / :class:`~.model.LeRobotModel` /
  :class:`~.model.RemoteModel` — the policy and its inference mechanics.
- :class:`~.adapter.Adapter` / :class:`~.adapter.LeRobotAdapter` /
  :class:`~.adapter.OpenPIAdapter` — env <-> policy space translation.
- :class:`~.batching.BatchedAgent` — many concurrent single-env rollouts
  sharing one batched model.
- :class:`~.dataset.DatasetWriter` — opt-in LeRobot v3 dataset recording
  (``agent.save = True``).

This subpackage needs the ``robot`` extra (``pip install 'hud[robot]'``).
"""

from __future__ import annotations

from .adapter import Adapter, LeRobotAdapter, OpenPIAdapter
from .agent import ROBOT_PROTOCOL, RobotAgent
from .batching import BatchedAgent, BatchedModel
from .dataset import DatasetWriter
from .model import LeRobotModel, Model, RemoteModel

__all__ = [
    "ROBOT_PROTOCOL",
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

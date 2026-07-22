"""Robot agent harness: drive a ``robot`` capability with a policy.

The harness splits a policy rollout into three seams, each replaceable on its own:

- :class:`~hud.agents.robot.agent.RobotAgent` — the loop: connect to the env's
  ``robot`` capability, observe, act, stop.
- :class:`~hud.agents.robot.model.Model` — *how to run* the policy (preprocess →
  forward → postprocess). :class:`~hud.agents.robot.model.LeRobotModel` ships the
  LeRobot checkpoint convention.
- :class:`~hud.agents.robot.adapter.Adapter` — translate between the env's
  observation/action spaces (from the contract) and the policy's.

Wrap an agent in :class:`~hud.agents.robot.batching.BatchedAgent` to run many rollouts
concurrently off one batched GPU forward (``max_concurrent`` rollouts, shared model).

Per-tick platform tracing is emitted by the loop itself: each step records an
:class:`~hud.agents.types.ObservationStep`, and each re-inference an
:class:`~hud.agents.types.InferenceStep`, so runs stream live into the HUD trace viewer.

This subpackage needs the ``robot`` extra (``pip install 'hud[robot]'``) for
``numpy`` + ``msgpack``; importing :mod:`hud.agents` alone never pulls them in.
"""

from __future__ import annotations

from .adapter import Adapter, LeRobotAdapter, OpenPIAdapter
from .agent import ROBOT_PROTOCOL, RobotAgent
from .batching import BatchedAgent, BatchedModel
from .model import LeRobotModel, Model

__all__ = [
    "ROBOT_PROTOCOL",
    "Adapter",
    "BatchedAgent",
    "BatchedModel",
    "LeRobotAdapter",
    "LeRobotModel",
    "Model",
    "OpenPIAdapter",
    "RobotAgent",
]

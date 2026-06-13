"""Robot agent harness: drive a ``robot`` capability with a policy.

The harness splits a policy rollout into three seams, each replaceable on its own:

- :class:`~hud.agents.robot.agent.RobotAgent` — the loop: connect to the env's
  ``robot`` capability, observe, act, stop.
- :class:`~hud.agents.robot.model.Model` — *how to run* the policy (preprocess →
  forward → postprocess). :class:`~hud.agents.robot.model.LeRobotModel` ships the
  LeRobot checkpoint convention.
- :class:`~hud.agents.robot.adapter.Adapter` — translate between the env's
  observation/action spaces (from the contract) and the policy's.

Per-tick platform tracing is emitted by the loop itself: each step records an
:class:`~hud.agents.types.ObservationStep` + :class:`~hud.agents.types.ActionStep`
so runs stream live into the HUD trace viewer.

This subpackage needs the ``robot`` extra (``pip install 'hud-python[robot]'``) for
``numpy`` + ``msgpack``; importing :mod:`hud.agents` alone never pulls them in.
"""

from __future__ import annotations

from .adapter import Adapter, LeRobotAdapter
from .agent import ROBOT_PROTOCOL, RobotAgent
from .model import LeRobotModel, Model, lerobot_infer

__all__ = [
    "ROBOT_PROTOCOL",
    "Adapter",
    "LeRobotAdapter",
    "LeRobotModel",
    "Model",
    "RobotAgent",
    "lerobot_infer",
]

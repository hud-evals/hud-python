"""Robot agent harness: drive a ``robot`` capability with a policy.

The harness splits a policy rollout into three seams, each replaceable on its own:

- :class:`~hud.agents.robot.agent.RobotAgent` /
  :class:`~hud.agents.robot.realtime.RealtimeRobotAgent` — the loop: connect to the
  env's ``robot`` capability, observe, act (or stream action chunks), stop.
- :class:`~hud.agents.robot.model.Model` — *how to run* the policy (preprocess →
  forward → postprocess). :class:`~hud.agents.robot.model.LeRobotModel` ships the
  LeRobot checkpoint convention.
- :class:`~hud.agents.robot.adapter.Adapter` — translate between the env's
  observation/action spaces (from the contract) and the policy's.

:class:`~hud.agents.robot.tracer.RobotTracer` optionally emits one platform span per
env step so runs stream live into the HUD trace viewer.

This subpackage needs the ``robot`` extra (``pip install 'hud-python[robot]'``) for
``numpy`` + ``msgpack``; importing :mod:`hud.agents` alone never pulls them in.
"""

from __future__ import annotations

from .adapter import Adapter, DefaultAdapter, lerobot_adapt_action, lerobot_adapt_observation
from .agent import ROBOT_PROTOCOL, RobotAgent
from .model import STEP_COUNTER, LeRobotModel, Model, StepCounter, lerobot_infer
from .realtime import RealtimeRobotAgent
from .tracer import RobotTracer

__all__ = [
    "ROBOT_PROTOCOL",
    "STEP_COUNTER",
    "Adapter",
    "DefaultAdapter",
    "LeRobotModel",
    "Model",
    "RealtimeRobotAgent",
    "RobotAgent",
    "RobotTracer",
    "StepCounter",
    "lerobot_adapt_action",
    "lerobot_adapt_observation",
    "lerobot_infer",
]

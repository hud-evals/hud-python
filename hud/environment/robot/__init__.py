"""Env-side robot runtime: the ``robot`` bridge + its building blocks.

This package holds everything an *environment* needs to own a simulator and serve it to
an agent over the ``robot`` WebSocket protocol:

- :class:`~hud.environment.robot.bridge.RobotBridge` — the server-side (synchronous)
  bridge: one sim step per received action.
- :class:`~hud.environment.robot.sim_runner.SimRunner` (``Inline`` / ``Thread``) — the
  strategy for *which thread* runs the thread-affine simulator.

The agent-side counterpart, :class:`~hud.capabilities.robot.RobotClient`, lives under
:mod:`hud.capabilities` (it is a capability *client*, dialed by the agent); these two ends
share the ``robot`` wire codec defined there.
"""

from __future__ import annotations

from .bridge import RobotBridge
from .endpoint import RobotEndpoint
from .sim_runner import InlineSimRunner, SimRunner, ThreadSimRunner

__all__ = [
    "InlineSimRunner",
    "RobotBridge",
    "RobotEndpoint",
    "SimRunner",
    "ThreadSimRunner",
]

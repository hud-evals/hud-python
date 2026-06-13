"""Env-side robot runtime: the ``robot`` bridge + its building blocks.

This package holds everything an *environment* needs to own a simulator and serve it to
an agent over the ``robot`` WebSocket protocol:

- :class:`~hud.environment.robot.bridge.RobotBridge` — the server-side (synchronous)
  bridge: one sim step per received action.
- :class:`~hud.environment.robot.sim_runner.SimRunner` (``Inline`` / ``Thread``) — the
  strategy for *which thread* runs the thread-affine simulator.
- :class:`~hud.environment.robot.data_saving.LeRobotRecorder` — the off-loop LeRobot
  dataset recorder (platform tick stream, configured by ``HUD_RECORD_DIR`` etc.).

The agent-side counterpart, :class:`~hud.capabilities.robot.RobotClient`, lives under
:mod:`hud.capabilities` (it is a capability *client*, dialed by the agent); these two ends
share the ``robot`` wire codec defined there.
"""

from __future__ import annotations

from .bridge import RobotBridge
from .data_saving import LeRobotRecorder
from .endpoint import RobotEndpoint
from .sim_runner import InlineSimRunner, SimRunner, ThreadSimRunner

__all__ = [
    "InlineSimRunner",
    "LeRobotRecorder",
    "RobotBridge",
    "RobotEndpoint",
    "SimRunner",
    "ThreadSimRunner",
]

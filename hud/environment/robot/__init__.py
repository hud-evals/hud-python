"""Env-side robot runtime: the ``robot`` bridges + their building blocks.

This package holds everything an *environment* needs to own a simulator and serve it to
an agent over the ``robot`` WebSocket protocol:

- :class:`~hud.environment.robot.bridge.RobotBridge` /
  :class:`~hud.environment.robot.bridge.RealtimeRobotBridge` — the server-side bridges.
- :class:`~hud.environment.robot.action_provider.ActionProvider` (+ subclasses,
  :func:`~hud.environment.robot.action_provider.make_action_provider`) — the realtime
  action queue / chunk-merge strategies.
- :class:`~hud.environment.robot.sim_runner.SimRunner` (``Inline`` / ``Thread``) — the
  strategy for *which thread* runs the thread-affine simulator.
- :class:`~hud.environment.robot.data_saving.LeRobotRecorder` — the off-loop LeRobot
  dataset recorder (platform tick stream, configured by ``HUD_RECORD_DIR`` etc.).

The agent-side counterpart, :class:`~hud.capabilities.robot.RobotClient`, lives under
:mod:`hud.capabilities` (it is a capability *client*, dialed by the agent); these two ends
share the ``robot`` wire codec defined there.
"""

from __future__ import annotations

from .action_provider import (
    ActionProvider,
    NaiveAsyncActionProvider,
    RTCActionProvider,
    SyncActionProvider,
    SyncFreezeActionProvider,
    WeightedAsyncActionProvider,
    make_action_provider,
)
from .bridge import RealtimeRobotBridge, RobotBridge
from .data_saving import LeRobotRecorder
from .endpoint import RobotEndpoint
from .sim_runner import InlineSimRunner, SimRunner, ThreadSimRunner

__all__ = [
    "ActionProvider",
    "InlineSimRunner",
    "LeRobotRecorder",
    "NaiveAsyncActionProvider",
    "RTCActionProvider",
    "RealtimeRobotBridge",
    "RobotBridge",
    "RobotEndpoint",
    "SimRunner",
    "SyncActionProvider",
    "SyncFreezeActionProvider",
    "ThreadSimRunner",
    "WeightedAsyncActionProvider",
    "make_action_provider",
]

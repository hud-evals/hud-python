"""Env-side robot runtime: the ``robot`` bridges + their building blocks.

This package holds everything an *environment* needs to own a simulator and serve it to
an agent over the ``robot`` WebSocket protocol:

- :class:`~hud.environment.robots.bridge.RobotBridge` /
  :class:`~hud.environment.robots.bridge.RealtimeRobotBridge` — the server-side bridges.
- :class:`~hud.environment.robots.action_provider.ActionProvider` (+ subclasses,
  :func:`~hud.environment.robots.action_provider.make_action_provider`) — the realtime
  action queue / chunk-merge strategies.
- :class:`~hud.environment.robots.sim_runner.SimRunner` (+ implementations) — the strategy
  for *which thread* runs the thread-affine simulator.
- :mod:`~hud.environment.robots.data_saving` — the framework-default recorder +
  LeRobot dataset sink (platform tick stream, configured by ``HUD_RECORD_DIR`` etc.).
- :mod:`~hud.environment.robots.contracts` — advisory contract matching tools
  (env contract vs model contract).

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
from .endpoint import RobotEndpoint
from .data_saving import default_recorder
from .sim_runner import (
    InlineSimRunner,
    MainThreadSimRunner,
    SimRunner,
    ThreadSimRunner,
)

__all__ = [
    "ActionProvider",
    "InlineSimRunner",
    "MainThreadSimRunner",
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
    "default_recorder",
    "make_action_provider",
]

"""Env-side robot runtime: bridges, the sim program, and the control endpoint.

A simulator always runs in its own process — the **sim program**
(``python -m hud.environment.robot.gym``), where the sim owns the main
thread and a bridge serves the wire. The env server holds a
:class:`~.endpoint.RobotEndpoint` on it:

- :class:`~.bridge.RobotBridge` — bridge base (``num_envs`` slots in lockstep
  internally; scalar openpi wire per claimed connection): the agent's
  ``robot`` WebSocket plus the JSON-RPC control side channel.
- :class:`~.gym.GymBridge` / :func:`~.gym.gym_command` — the generic
  gym path (``env.gym(...)`` — a factory, registry id, or declared env):
  contract derivation, capability, episode control.
- :class:`~.endpoint.RobotEndpoint` — the env server's handle: spawn (owned)
  or remote (attached), same control surface either way.
- :func:`~.bridge.serve_bridge` — the sim program's blocking entry (custom
  bridges call it last).
- :func:`~.gym.wrap` — one-line trace streaming for any gym env you drive
  yourself, plus the shared gym introspection.

The agent-side counterpart, :class:`~hud.capabilities.robot.RobotClient`, lives
under :mod:`hud.capabilities`; both ends share the wire codec defined there.
"""

from __future__ import annotations

from .bridge import RobotBridge, serve_bridge
from .endpoint import RobotEndpoint
from .gym import GymBridge, TracedEnv, gym_command, wrap

__all__ = [
    "GymBridge",
    "RobotBridge",
    "RobotEndpoint",
    "TracedEnv",
    "gym_command",
    "serve_bridge",
    "wrap",
]

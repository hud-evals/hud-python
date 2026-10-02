"""Env-side robot runtime: bridges, the sim program, and the control endpoint.

A simulator always runs in its own process — the **sim program**
(``python -m hud.environment.robot.gym``), where the sim owns the main
thread and a bridge serves the wire. The env server holds a
``RobotEndpoint`` on it:

- ``RobotBridge`` — bridge base (``num_envs`` slots in lockstep
  internally; scalar openpi wire per claimed connection): the agent's
  ``robot`` WebSocket plus the JSON-RPC control side channel.
- ``GymBridge`` — the generic gym path (``env.gym(...)`` — a
  factory, registry id, or declared env): contract derivation, capability,
  episode control.
- ``RobotEndpoint`` — the env server's handle: spawn (owned)
  or remote (attached), same control surface either way.
- ``serve_bridge`` — the sim program's blocking entry
  (``RobotEndpoint(MyBridge)`` spawns it; custom containers can call it too).
- ``wrap`` — one-line trace streaming for any gym env you drive
  yourself, plus the shared gym introspection.
- ``control.DirectControl`` — the same sim as MCP motion tools, so a
  tool-calling LLM can drive it (the ``robot`` wire stays up for policies).

The agent-side counterpart, ``RobotClient``, lives
under ``hud.capabilities``; both ends share the wire codec defined there.
"""

from __future__ import annotations

from .bridge import RobotBridge, serve_bridge
from .control import DirectControl
from .endpoint import RobotEndpoint
from .gym import GymBridge, wrap

__all__ = [
    "DirectControl",
    "GymBridge",
    "RobotBridge",
    "RobotEndpoint",
    "serve_bridge",
    "wrap",
]

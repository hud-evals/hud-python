"""Env-side robot runtime: bridges, the control endpoint, and gym integration.

Everything an *environment* needs to own a simulator and serve it to an agent
over the ``robot`` WebSocket protocol:

- :class:`~.bridge.RobotBridge` — the batched-first bridge base (``num_envs``
  slots in lockstep; a plain single env is a batch of one).
- :class:`~.bridge.GymBridge` / :class:`~.gym.Gym` — the generic gym-factory
  path (``env.gym(make_env)``): contract derivation, capability, episode control.
- :class:`~.endpoint.RobotEndpoint` — the control handle, local or remote.
- :func:`hud.wrap` (:mod:`~.wrap`) — one-line trace streaming for any gym env.
- :class:`~.sim_thread.SimThread` — the one process shape: the sim owns the
  main thread, serving runs on a background loop thread.

The agent-side counterpart, :class:`~hud.capabilities.robot.RobotClient`, lives
under :mod:`hud.capabilities`; both ends share the wire codec defined there.
"""

from __future__ import annotations

from .bridge import GymBridge, RobotBridge
from .endpoint import RobotEndpoint
from .gym import Gym
from .sim_thread import SimThread, run_with_sim
from .wrap import TracedEnv, wrap

__all__ = [
    "Gym",
    "GymBridge",
    "RobotBridge",
    "RobotEndpoint",
    "SimThread",
    "TracedEnv",
    "run_with_sim",
    "wrap",
]

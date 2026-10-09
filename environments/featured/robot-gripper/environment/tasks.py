"""Benchmark tasks: one task template + parameters = one eval instance.

The featured pick-mug row is bounded to 40 agent steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hud import Taskset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# `env` is re-exported so `hud eval environment/tasks.py` can serve the Environment from here.
from environment.env import env, force_grasp, move_object, open_drawer, pick_object  # noqa: E402, F401

_drawer = open_drawer(scene_id="tabletop-v1", target_joint="drawer_slide", success_threshold=0.2)
_drawer.slug = "open-drawer"

_pick = pick_object(scene_id="tabletop-v1", target_object="mug", lift_height=0.9)
_pick.slug = "pick-mug"
_pick.agent_config = {"max_steps": 40}

_move = move_object(scene_id="tabletop-v1", target_object="mug", goal_x=-0.2, goal_y=0.0, goal_z=0.75, tolerance=0.05)
_move.slug = "move-mug"

_grasp = force_grasp(scene_id="tabletop-v1", target_object="mug", min_grip_force=0.5, hold_steps=100)
_grasp.slug = "force-grasp-mug"

taskset = Taskset("worldsim-tabletop", [_drawer, _pick, _move, _grasp])

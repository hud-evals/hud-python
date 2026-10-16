"""LIBERO tabletop manipulation, served to both policies and tool-calling agents.

The Franka Panda simulator runs in its own process behind a `RobotBridge` and is published
as a `robot` capability, so a VLA policy can stream actions to it. In absolute mode the same
env also serves direct control: a `move_to` MCP tool, so any tool-calling LLM or coding agent
drives the arm with no robot code of its own.

`LIBERO_CONTROL` picks the interface the env serves (the sim and the tasks are unchanged):

- `absolute` (default): absolute end-effector targets on `robot`, plus `move_to` direct control.
- `delta`: end-effector deltas on `robot` only, the action space LIBERO policies are trained on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from hud import Environment
from hud.environment.robot import RobotEndpoint
from robosuite.utils.transform_utils import axisangle2quat, mat2quat, quat2axisangle, quat2mat

from bridge import LiberoSimBridge
from control import LiberoDirectControl

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

CONTROL = os.environ.get("LIBERO_CONTROL", "absolute")
if CONTROL not in ("absolute", "delta"):
    raise ValueError(f"LIBERO_CONTROL must be 'absolute' or 'delta', got {CONTROL!r}")

CONTRACT = json.loads(
    (Path(__file__).parent / "contracts" / ("ee_abs.json" if CONTROL == "absolute" else "ee_del.json")).read_text()
)

env = Environment(name="robot-libero")
bridge = LiberoSimBridge(use_delta=CONTROL == "delta", image_size=CONTRACT["features"]["observation/image"]["shape"][0])
bridge.contract = CONTRACT
endpoint = RobotEndpoint(bridge).attach(env)


@env.initialize
async def _up() -> None:
    await endpoint.start()
    for capability in await endpoint.capabilities():
        env.add_capability(capability)


@env.shutdown
async def _down() -> None:
    await endpoint.stop()


# The arm's controller tracks the grip site, whose frame is the observed hand frame turned -90
# degrees about z.
HAND_TO_GRIP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def held_pose(data: dict) -> np.ndarray:
    """The absolute action that holds the observed pose with the gripper open, as a reset leaves it."""
    state = data["observation/state"]
    rotation = quat2axisangle(mat2quat(quat2mat(axisangle2quat(state[3:6])) @ HAND_TO_GRIP))
    return np.concatenate([state[:3], rotation, [-1.0]])


if CONTROL == "absolute":
    LiberoDirectControl(
        dims=[
            "target_eef_pos.x",
            "target_eef_pos.y",
            "target_eef_pos.z",
            "target_eef.roll",
            "target_eef.pitch",
            "target_eef.yaw",
            "gripper.open_close",
        ],
        reference=held_pose,
        max_step={"gripper.open_close": 2.0},  # a binary command: switch in one step
        notes=(
            "Positions are world-frame meters of the point between the gripper fingers. In "
            "observation/image, +x moves the gripper toward the camera (down the image), +y "
            "to the image's left, +z up. The table top is at about z=0.90; the gripper starts "
            "near z=1.17. target_eef.roll, pitch, and yaw are intrinsic XYZ radians; the reset "
            "pose points the gripper straight down. gripper.open_close: -1 open, +1 closed. "
            "observation/wrist_image looks down from the gripper."
        ),
    ).attach(endpoint)


def _suite_template(suite: str):
    async def episode(libero_task_id: int, init_state_id: int = 0):
        ep = await endpoint.reset(task_suite=suite, task_id=libero_task_id, init_state_id=init_state_id)
        yield {"prompt": ep["prompt"]}
        yield await endpoint.result()

    return episode


for suite in SUITES:
    env.template(id=suite, description=f"One task of the LIBERO {suite} suite.")(_suite_template(suite))

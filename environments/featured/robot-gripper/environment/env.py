"""Newton tabletop simulation controlled through MCP tools by an LLM agent.

The Newton sim runs in its own process (`sim/host.py`, spawned in `@env.initialize`)
and is served as an `mcp` capability - a FastMCP tool server the agent's harness
attaches its own tools to. The separate process is what lets the live viewer own the
main thread (see sim/host.py). Each task resets the scene itself at setup - so the
prompt is just the goal - and grades by reading sim state over that same `mcp`
(`sim_tools`), so grading hits the exact tools the agent drove.

Serve it like any environment: `hud eval environment/tasks.py`, a container CMD, or
`LocalRuntime("environment/env.py")`.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult, SubScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.host import SimHost  # noqa: E402

TOOLS = (
    "reset, render, render_depth, get_state, get_object_state, get_joint_state, "
    "get_contact_forces, move_gripper(direction, distance), rotate_gripper(angle), "
    "open_gripper, close_gripper, step(action)"
)


class SimClient:
    """Reset + grading reads over the sim's `mcp` - the same tools the agent drives.

    The sim runs in another process, so these go over the wire (one persistent
    connection per env); each call returns the tool's plain dict.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: Any = None

    async def open(self) -> None:
        from fastmcp import Client

        self._client = Client(self._url)
        await self._client.__aenter__()

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.__aexit__(None, None, None)
            self._client = None

    async def call(self, name: str, **kwargs: Any) -> dict:
        res = await self._client.call_tool(name, kwargs)
        # FastMCP returns the tool's dict as .data (newer), else structured content / JSON text.
        if isinstance(getattr(res, "data", None), dict):
            return res.data
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            return sc.get("result", sc)
        content = getattr(res, "content", None)
        if content and getattr(content[0], "text", None):
            return json.loads(content[0].text)
        return {}


env = Environment(name="robot-gripper")
sim_host = SimHost()
sim_tools = SimClient(sim_host.mcp_url)  # env-side reset + grading reads over mcp


@env.initialize
async def _up() -> None:
    await sim_host.start()
    await sim_tools.open()
    env.add_capability(Capability.mcp(name="sim", url=sim_host.mcp_url))


@env.shutdown
async def _down() -> None:
    await sim_tools.close()
    await sim_host.stop()


# ── Tasks ───────────────────────────────────────────────────────────────────


@env.template(id="open-drawer", description="Pull a drawer/door joint open.")
async def open_drawer(
    scene_id: str = "tabletop-v1", target_joint: str = "drawer_slide", success_threshold: float = 0.2
):
    await sim_tools.call("reset", scene_id=scene_id)
    yield (
        f"Control the robot gripper to open the joint '{target_joint}' as far as possible. "
        f"Its range is [0, 0.25] m; success is >= {success_threshold} m. "
        f"Grasp the drawer handle and pull it out. "
        f"Tools: {TOOLS}. Check progress with get_joint_state(joint_name='{target_joint}')."
    )

    joint = await sim_tools.call("get_joint_state", joint_name=target_joint)
    if isinstance(joint, dict) and "error" not in joint:
        final_pos, target_pos = joint["position"], 0.25
        completion = max(0.0, min(1.0, final_pos / target_pos))
        success = final_pos >= success_threshold
    else:
        completion, success, final_pos, target_pos = 0.0, False, 0.0, 0.25

    # reward = weighted sum of the subscores below (self-consistent).
    reward = 0.8 * completion + 0.2 * (1.0 if success else 0.0)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=f"Drawer '{target_joint}': {final_pos:.4f} / {target_pos:.4f} "
        f"({completion * 100:.1f}% open). {'SUCCESS' if success else 'INCOMPLETE'}",
        subscores=[
            SubScore(name="task_completion", weight=0.8, value=round(completion, 4)),
            SubScore(name="binary_success", weight=0.2, value=1.0 if success else 0.0),
        ],
    )


@env.template(id="pick-object", description="Use MCP gripper tools to lift an object to a target height.")
async def pick_object(scene_id: str = "tabletop-v1", target_object: str = "mug", lift_height: float = 0.9):
    await sim_tools.call("reset", scene_id=scene_id)
    yield (
        f"Use the simulation tools to lift the '{target_object}' until its position reaches z={lift_height:.2f} m. "
        f"Open the gripper, move to the object, close to grasp, then lift. "
        f"Tools: {TOOLS}. Use get_contact_forces(body_name='finger_left') to confirm a grip; "
        f"check progress with get_object_state(object_name='{target_object}'). "
        f"The final object height determines your score: 70% lift progress from z=0.77 m "
        f"and 30% for reaching z>={lift_height:.2f} m. Finish when the target height is reached."
    )

    obj = await sim_tools.call("get_object_state", object_name=target_object)
    if isinstance(obj, dict) and "error" not in obj:
        final_z, initial_z = obj["position"]["z"], 0.77
        needed = lift_height - initial_z
        progress = max(0.0, min(1.0, (final_z - initial_z) / needed)) if needed > 1e-3 else 1.0
        success = final_z >= lift_height
    else:
        progress, success, final_z = 0.0, False, 0.0

    reward = 0.7 * progress + 0.3 * (1.0 if success else 0.0)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=f"Object '{target_object}': z={final_z:.4f} / {lift_height:.4f} "
        f"({progress * 100:.1f}% progress). {'SUCCESS' if success else 'INCOMPLETE'}",
        subscores=[
            SubScore(name="lift_progress", weight=0.7, value=round(progress, 4)),
            SubScore(name="binary_success", weight=0.3, value=1.0 if success else 0.0),
        ],
    )


@env.template(id="move-object", description="Move an object to a target position.")
async def move_object(
    scene_id: str = "tabletop-v1",
    target_object: str = "mug",
    goal_x: float = -0.2,
    goal_y: float = 0.0,
    goal_z: float = 0.75,
    tolerance: float = 0.05,
):
    await sim_tools.call("reset", scene_id=scene_id)
    yield (
        f"Control the robot gripper to move the '{target_object}' to position "
        f"({goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f}), within {tolerance:.3f} m. "
        f"Tools: {TOOLS}. Use render_depth for distances; check progress with "
        f"get_object_state(object_name='{target_object}')."
    )

    obj = await sim_tools.call("get_object_state", object_name=target_object)
    if isinstance(obj, dict) and "error" not in obj:
        fp = obj["position"]
        dist = math.sqrt((fp["x"] - goal_x) ** 2 + (fp["y"] - goal_y) ** 2 + (fp["z"] - goal_z) ** 2)
        progress = max(0.0, min(1.0, 1.0 - dist / 0.5))
        success = dist <= tolerance
    else:
        progress, success, dist = 0.0, False, float("inf")

    reward = 0.7 * progress + 0.3 * (1.0 if success else 0.0)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=f"Object '{target_object}': final distance to goal = {dist:.4f} m. "
        f"{'SUCCESS' if success else 'INCOMPLETE'}",
        subscores=[
            SubScore(name="distance_progress", weight=0.7, value=round(progress, 4)),
            SubScore(name="binary_success", weight=0.3, value=1.0 if success else 0.0),
        ],
    )


@env.template(id="force-grasp", description="Lift an object and maintain finger contact force for a hold interval.")
async def force_grasp(
    scene_id: str = "tabletop-v1", target_object: str = "mug", min_grip_force: float = 0.5, hold_steps: int = 100
):
    if not isinstance(hold_steps, int) or hold_steps < 1:
        raise ValueError("hold_steps must be a positive integer")
    if not math.isfinite(min_grip_force) or min_grip_force <= 0:
        raise ValueError("min_grip_force must be finite and positive")
    await sim_tools.call("reset", scene_id=scene_id)
    yield (
        f"Control the robot gripper to grasp the '{target_object}', lift its center above z=0.78 m, and hold it firmly. "
        f"Both fingers must maintain at least {min_grip_force} N of contact force, and the object must remain "
        f"above z=0.78 m, throughout {hold_steps} physics steps after you finish. "
        f"During that hold, translation and rotation commands stop; your finger control targets stay unchanged. "
        f"The score is 60% for the fraction of fingers maintaining the required force throughout the hold "
        f"and 40% for keeping the object lifted throughout. "
        f"Tools: {TOOLS}. Monitor your grip with get_contact_forces(body_name='finger_left') "
        f"and get_contact_forces(body_name='finger_right')."
    )

    state = await sim_tools.call("get_state")
    hold_action = [0.0, 0.0, 0.0, 0.0, *state["actuator_controls"][4:]]
    left_ok, right_ok, obj_lifted = True, True, True
    for sample in range(hold_steps + 1):
        if sample:
            advanced = await sim_tools.call("step", action=hold_action)
            if "error" in advanced:
                raise RuntimeError(f"Could not advance grasp hold: {advanced['error']}")
        left = await sim_tools.call("get_contact_forces", body_name="finger_left")
        right = await sim_tools.call("get_contact_forces", body_name="finger_right")
        obj = await sim_tools.call("get_object_state", object_name=target_object)
        left_ok = left_ok and left.get("total_force_magnitude", 0) >= min_grip_force
        right_ok = right_ok and right.get("total_force_magnitude", 0) >= min_grip_force
        obj_lifted = obj_lifted and "error" not in obj and obj["position"]["z"] > 0.78

    grip = (int(left_ok) + int(right_ok)) / 2.0
    reward = 0.6 * grip + 0.4 * (1.0 if obj_lifted else 0.0)
    yield EvaluationResult(
        reward=round(reward, 4),
        done=True,
        content=f"Grasp '{target_object}' held for {hold_steps} physics steps: "
        f"left_force={'SUSTAINED' if left_ok else 'NOT SUSTAINED'}, "
        f"right_force={'SUSTAINED' if right_ok else 'NOT SUSTAINED'}, "
        f"lifted={'SUSTAINED' if obj_lifted else 'NOT SUSTAINED'}",
        subscores=[
            SubScore(name="grip_quality", weight=0.6, value=round(grip, 4)),
            SubScore(name="object_lifted", weight=0.4, value=1.0 if obj_lifted else 0.0),
        ],
    )

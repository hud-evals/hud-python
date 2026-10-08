"""Newton sim as a FastMCP tool server (the `mcp` capability).

The Newton-backed `_Sim` and its tools. This is served in its own process by
`sim/host.py` (which `env.py` spawns), so the live viewer can own the main thread;
`env.py` grades by calling these same tools over `mcp`.

The LLM tasks use the floating-gripper tabletop scene over MCP.

An optional live 3D viewer (`run_viewer`, off by default) renders the running
scene in a window - `sim/host.py` runs it on the main thread when `WORLDSIM_VIEWER=1`.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

# Quiet known first-run noise before the libraries that emit it import. authlib
# pins an "always" filter for its own warning at import time, so import it first.
os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
try:
    import authlib.deprecate  # noqa: F401
except ImportError:
    pass
warnings.filterwarnings("ignore", message=r"authlib\.jose module is deprecated")
warnings.filterwarnings("ignore", message=r"warp\.config\.(verbose|quiet) is deprecated")

import numpy as np
from PIL import Image

import mujoco

# Warp prints a banner to stdout on init; redirect to stderr while importing and
# initializing it (safe here - runs at module import, before any threads start).
_orig_stdout = sys.stdout
sys.stdout = sys.stderr
import warp as wp

wp.config.log_level = wp.LOG_ERROR
import newton
import newton.solvers
from gizmo.runtime import _build_solver, import_model

wp.init()
wp.config.log_level = wp.LOG_WARNING
sys.stdout = _orig_stdout

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage  # returned by render() so clients show the image

SCENES_DIR = Path(os.environ.get("SCENES_DIR", Path(__file__).resolve().parents[1] / "scenes"))

server = FastMCP(name="worldsim-newton")

_SCENE_FILES = ("scene.xml", "scene.mjcf", "scene.usd", "scene.usda", "scene.urdf")


# ── Simulation state ───────────────────────────────────────────────────────


class _Sim:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Live viewer handle is preserved across resets (created once, in run_viewer).
        self.viewer: Any | None = None
        self.viewer_model_pending: bool = False
        self.clear()

    def clear(self) -> None:
        # Newton (physics engine)
        self.newton_model: Any | None = None
        self.solver: Any | None = None
        self.state_0: Any | None = None
        self.state_1: Any | None = None
        self.control: Any | None = None
        self.contacts: Any | None = None
        self.use_mujoco_cpu: bool = True
        # MuJoCo (names, cameras, rendering)
        self.mj_model: mujoco.MjModel | None = None
        self.mj_data: mujoco.MjData | None = None
        # Metadata
        self.metadata: dict[str, Any] = {}
        self.scene_id: str | None = None
        self.solver_name: str = ""
        self.step_count: int = 0
        self.max_steps: int = 1000
        self.dt: float = 0.002

    def _sync_render_data(self) -> None:
        """Copy qpos/qvel from solver's MuJoCo data to the rendering model."""
        if self.mj_model is None or self.solver is None:
            return
        smj = self.solver.mj_data
        self.mj_data.qpos[:] = smj.qpos[:]
        self.mj_data.qvel[:] = smj.qvel[:]
        self.mj_data.time = smj.time
        mujoco.mj_forward(self.mj_model, self.mj_data)


_sim = _Sim()


def _require_sim() -> _Sim:
    if _sim.newton_model is None or _sim.solver is None:
        raise RuntimeError("No simulation loaded. Call reset() first.")
    return _sim


def _find_scene_file(scene_dir: Path) -> Path | None:
    for name in _SCENE_FILES:
        p = scene_dir / name
        if p.exists():
            return p
    # Fallback: a single scene file with a non-standard name, only when unambiguous.
    exts = (".xml", ".mjcf", ".usd", ".usda", ".urdf")
    candidates = [p for p in sorted(scene_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_camera(model: mujoco.MjModel, camera: str | int) -> str | int:
    """Resolve a camera name to a usable target, falling back to the free camera."""
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            return -1
    return camera


def _newton_step(action: list[float]) -> None:
    """Step physics through Newton's solver with direct actuator control."""
    sim = _sim
    s = sim.solver
    s.mj_data.ctrl[:] = action
    s.mj_model.opt.timestep = sim.dt
    mujoco.mj_step(s.mj_model, s.mj_data)
    s._update_newton_state(sim.newton_model, sim.state_1, s.mj_data, state_prev=sim.state_0)
    sim.state_0, sim.state_1 = sim.state_1, sim.state_0


# ── Core simulation tools ───────────────────────────────────────────────────


@server.tool()
def list_scenes() -> list[dict[str, str]]:
    """List all available scenes."""
    scenes: list[dict[str, str]] = []
    for scene_dir in sorted(SCENES_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_file = _find_scene_file(scene_dir)
        has_builder = (scene_dir / "build_scene.py").exists()
        if not scene_file and not has_builder:
            continue
        entry: dict[str, str] = {"scene_id": scene_dir.name}
        if scene_file:
            entry["format"] = scene_file.suffix.lstrip(".")
        elif has_builder:
            entry["format"] = "python"
        meta = scene_dir / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                entry["description"] = json.load(f).get("description", "")
        scenes.append(entry)
    return scenes


@server.tool()
def reset(
    scene_id: str,
    settle_steps: int = 500,
    seed: int | None = None,
    max_episode_steps: int = 1000,
) -> dict[str, Any]:
    """Load a scene and reset the simulation. Call this before anything else.

    Args:
        scene_id: Scene directory name; this featured example includes "tabletop-v1".
        settle_steps: Physics steps to let the scene settle under gravity.
        seed: If set, randomize the initial object pose (for varied rollouts).
        max_episode_steps: Episode cap; `step` reports done past this.
    """
    print(f"[sim] reset(scene_id={scene_id!r}, settle_steps={settle_steps}, seed={seed})", file=sys.stderr)
    with _sim.lock:
        viewer = _sim.viewer  # preserved across reset so the live window persists
        _sim.clear()
        _sim.viewer = viewer
        _sim.max_steps = max_episode_steps

        scene_dir = SCENES_DIR / scene_id
        if not scene_dir.is_dir():
            return {"error": f"Scene '{scene_id}' not found at {scene_dir}"}

        scene_file = _find_scene_file(scene_dir)
        builder_file = scene_dir / "build_scene.py"

        if scene_file is None and not builder_file.exists():
            return {"error": f"No scene file found in {scene_dir}"}

        # --- Newton: load model and create solver ---
        if scene_file:
            newton_model, kind, _meta, _warnings = import_model(scene_file)
        else:
            spec = importlib.util.spec_from_file_location("build_scene", str(builder_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            newton_model = mod.build()
            kind = "python"

        solver, solver_name, use_mujoco_cpu, contacts = _build_solver(
            newton_model, kind if kind != "python" else "mjcf", "auto"
        )
        # Newton disables multi-point contacts by default; grasps need a stable manifold.
        solver.mj_model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)

        _sim.newton_model = newton_model
        _sim.solver = solver
        _sim.use_mujoco_cpu = use_mujoco_cpu
        _sim.contacts = contacts
        _sim.scene_id = scene_id
        _sim.solver_name = solver_name
        _sim.state_0 = newton_model.state()
        _sim.state_1 = newton_model.state()
        _sim.control = newton_model.control()
        _sim.dt = solver.mj_model.opt.timestep
        newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, _sim.state_0)

        # --- MuJoCo: load separately for names, cameras, rendering ---
        if scene_file and scene_file.suffix.lower() in (".xml", ".mjcf"):
            _sim.mj_model = mujoco.MjModel.from_xml_path(str(scene_file.resolve()))
            _sim.mj_data = mujoco.MjData(_sim.mj_model)

        # --- Metadata ---
        meta_path = scene_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                _sim.metadata = json.load(f)

        smj_m, smj_d = solver.mj_model, solver.mj_data
        settle_ctrl = np.zeros(smj_m.nu)
        if seed is not None:
            # Jitter the first free body for varied initial object poses.
            rng = np.random.default_rng(seed)
            for jid in range(smj_m.njnt):
                if smj_m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    adr = int(smj_m.jnt_qposadr[jid])
                    smj_d.qpos[adr] += float(rng.uniform(-0.04, 0.04))
                    smj_d.qpos[adr + 1] += float(rng.uniform(-0.04, 0.04))
                    break
            mujoco.mj_forward(smj_m, smj_d)

        # --- Settle via Newton's solver ---
        settle_list = settle_ctrl.tolist()
        for _ in range(settle_steps):
            _newton_step(settle_list)
        _sim.step_count = 0

        if _sim.mj_model is not None:
            _sim._sync_render_data()

        # Tell the optional live viewer to (re)load this model on its next frame.
        _sim.viewer_model_pending = True

        # --- Build response ---
        model = _sim.mj_model or solver.mj_model
        bodies = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, model.nbody)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]
        joints = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        ]
        cameras = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        ]

        nm = newton_model
        return {
            "status": "ready",
            "engine": "newton",
            "solver": solver_name,
            "scene_id": scene_id,
            "description": _sim.metadata.get("description", ""),
            "n_bodies": model.nbody,
            "n_joints": model.njnt,
            "n_actuators": model.nu,
            "timestep_seconds": _sim.dt,
            "objects": bodies,
            "joints": joints,
            "cameras": cameras,
            "newton": {
                "body_count": int(nm.body_count),
                "joint_count": int(nm.joint_count),
                "shape_count": int(nm.shape_count),
            },
            "supported_formats": ["mjcf", "usd", "urdf", "python"],
        }


@server.tool()
def step(action: list[float]) -> dict[str, Any]:
    """Apply an action and advance the simulation by one timestep.

    Args:
        action: Control signals, one per actuator.
    """
    sim = _require_sim()
    nu = sim.solver.mj_model.nu
    if len(action) != nu:
        return {"error": f"Expected {nu} actuator values, got {len(action)}"}

    with sim.lock:
        _newton_step(action)
        sim.step_count += 1
        if sim.mj_model is not None:
            sim._sync_render_data()

    return {
        "step": sim.step_count,
        "time": round(sim.solver.mj_data.time, 4),
        "done": sim.step_count >= sim.max_steps,
    }


@server.tool()
def render(camera: str = "overhead", width: int = 640, height: int = 480) -> MCPImage:
    """Capture an RGB image from a camera. Returns a PNG the client renders inline.

    Args:
        camera: Camera name (from reset() response).
        width: Image width in pixels.
        height: Image height in pixels.
    """
    sim = _require_sim()
    render_model = sim.mj_model or sim.solver.mj_model
    render_data = sim.mj_data or sim.solver.mj_data

    r = mujoco.Renderer(render_model, width=width, height=height)
    r.update_scene(render_data, _resolve_camera(render_model, camera))
    pixels = r.render()

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    # An MCP image (ImageContent) so the agent + the platform trace show it, not base64 text.
    return MCPImage(data=buf.getvalue(), format="png")


@server.tool()
def get_state() -> dict[str, Any]:
    """Get the full simulation state: gripper position, joint positions, sensor data."""
    sim = _require_sim()
    model = sim.mj_model or sim.solver.mj_model
    data = sim.mj_data or sim.solver.mj_data

    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    gripper_pos = data.xpos[gripper_id].tolist() if gripper_id >= 0 else None
    gripper_quat = data.xquat[gripper_id].tolist() if gripper_id >= 0 else None

    return {
        "step": sim.step_count,
        "time": round(data.time, 4),
        "gripper_position": gripper_pos,
        "gripper_orientation": gripper_quat,
        "joint_positions": {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): round(data.qpos[model.jnt_qposadr[i]], 5)
            for i in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            and model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        },
        "sensor_data": {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): round(
                float(data.sensordata[model.sensor_adr[i]]), 5
            )
            for i in range(model.nsensor)
            if model.sensor_dim[i] == 1
        },
    }


@server.tool()
def get_object_state(object_name: str) -> dict[str, Any]:
    """Get the position and orientation of a named object.

    Args:
        object_name: Body name in the scene (e.g. "mug", "drawer").
    """
    sim = _require_sim()
    model = sim.mj_model or sim.solver.mj_model
    data = sim.mj_data or sim.solver.mj_data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
        return {"error": f"Object '{object_name}' not found"}

    pos = data.xpos[body_id]
    quat = data.xquat[body_id]
    return {
        "name": object_name,
        "position": {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4)},
        "orientation": {
            "w": round(quat[0], 4),
            "x": round(quat[1], 4),
            "y": round(quat[2], 4),
            "z": round(quat[3], 4),
        },
    }


@server.tool()
def get_joint_state(joint_name: str) -> dict[str, Any]:
    """Get the current position/angle of a joint.

    Args:
        joint_name: Joint name (e.g. "drawer_slide", "finger_left_slide").
    """
    sim = _require_sim()
    model = sim.mj_model or sim.solver.mj_model
    data = sim.mj_data or sim.solver.mj_data
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jnt_id < 0:
        return {"error": f"Joint '{joint_name}' not found"}

    jnt_type = model.jnt_type[jnt_id]
    type_name = {
        mujoco.mjtJoint.mjJNT_FREE: "free",
        mujoco.mjtJoint.mjJNT_BALL: "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }.get(jnt_type, "unknown")

    qpos_adr = model.jnt_qposadr[jnt_id]
    jnt_range = model.jnt_range[jnt_id]

    return {
        "name": joint_name,
        "type": type_name,
        "position": round(float(data.qpos[qpos_adr]), 5),
        "velocity": round(float(data.qvel[model.jnt_dofadr[jnt_id]]), 5),
        "range": [round(float(jnt_range[0]), 4), round(float(jnt_range[1]), 4)],
    }


# ── High-level gripper tools ───────────────────────────────────────────────


@server.tool()
def move_gripper(direction: str, distance: float = 0.05, steps: int = 50) -> dict[str, Any]:
    """Move the gripper in a direction.

    Args:
        direction: One of "left", "right", "forward", "backward", "up", "down".
        distance: How far to move in meters (0.01 to 0.2).
        steps: Simulation steps to execute the motion over.
    """
    sim = _require_sim()
    model = sim.mj_model or sim.solver.mj_model
    data = sim.mj_data or sim.solver.mj_data

    direction = direction.lower().strip()
    valid = {"left", "right", "forward", "backward", "up", "down"}
    if direction not in valid:
        return {"error": f"Invalid direction '{direction}'. Choose from: {', '.join(sorted(valid))}"}

    distance = max(0.01, min(0.2, distance))
    speed = distance / (steps * sim.dt)
    sign = -1 if direction in ("left", "backward", "down") else 1
    axis_map = {"left": 0, "right": 0, "forward": 1, "backward": 1, "up": 2, "down": 2}
    axis = axis_map[direction]

    fl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left_slide")
    fr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right_slide")
    nu = sim.solver.mj_model.nu

    with sim.lock:
        for _ in range(steps):
            ctrl = [0.0] * nu
            ctrl[axis] = sign * speed
            if fl_id >= 0:
                ctrl[4] = sim.solver.mj_data.ctrl[4]
            if fr_id >= 0:
                ctrl[5] = sim.solver.mj_data.ctrl[5]
            _newton_step(ctrl)
            sim.step_count += 1

        # Brake
        for _ in range(20):
            ctrl = [0.0] * nu
            if fl_id >= 0:
                ctrl[4] = sim.solver.mj_data.ctrl[4]
            if fr_id >= 0:
                ctrl[5] = sim.solver.mj_data.ctrl[5]
            _newton_step(ctrl)
            sim.step_count += 1

        sim._sync_render_data()

    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    pos = data.xpos[gripper_id]

    return {
        "status": "moved",
        "direction": direction,
        "distance_requested": round(distance, 3),
        "gripper_position": {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4)},
        "step": sim.step_count,
    }


@server.tool()
def rotate_gripper(angle: float) -> dict[str, Any]:
    """Rotate the gripper about the vertical (yaw) axis by `angle` radians.

    Use this to reorient the fingers before grasping - e.g. rotate_gripper(1.5708)
    turns the wrist 90 degrees so the fingers close across a bar-shaped handle. Holds
    the fingers where they are during the turn.

    Args:
        angle: Signed yaw change in radians (positive = counter-clockwise).
    """
    sim = _require_sim()
    model = sim.mj_model or sim.solver.mj_model
    data = sim.mj_data or sim.solver.mj_data
    nu = sim.solver.mj_model.nu

    fl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left_slide")
    fr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right_slide")
    yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_yaw")

    def _hold(ctrl: list[float]) -> None:  # Preserve finger servo targets during rotation.
        if fl_id >= 0:
            ctrl[4] = sim.solver.mj_data.ctrl[4]
        if fr_id >= 0:
            ctrl[5] = sim.solver.mj_data.ctrl[5]

    # Open-loop by step count: the render-side qpos is not live mid-step, so we can't
    # close the loop on it. At full yaw command the joint settles to a fixed rate -
    # kv*ctrl/(kv+damping) = 50*2/(50+20) = 1.428 rad/s, i.e. ~0.002857 rad per dt step
    # (act_yaw kv=50, gripper_yaw damping=20 in scene.xml). Drive that many steps.
    rate = 50.0 * 2.0 / (50.0 + 20.0) * sim.dt  # rad per step at full command
    cmd = 2.0 if angle >= 0 else -2.0
    n = max(1, round(abs(angle) / rate))

    with sim.lock:
        for _ in range(n):
            ctrl = [0.0] * nu
            ctrl[3] = cmd
            _hold(ctrl)
            _newton_step(ctrl)
            sim.step_count += 1

        for _ in range(20):  # brake: zero yaw command, hold fingers, let it settle
            ctrl = [0.0] * nu
            _hold(ctrl)
            _newton_step(ctrl)
            sim.step_count += 1

        sim._sync_render_data()

    yaw = float(data.qpos[model.jnt_qposadr[yaw_id]]) if yaw_id >= 0 else 0.0
    return {
        "status": "rotated",
        "yaw": round(yaw, 4),
        "requested_angle": round(angle, 4),
        "step": sim.step_count,
    }


@server.tool()
def open_gripper(steps: int = 30) -> dict[str, Any]:
    """Open the gripper fingers to release an object."""
    sim = _require_sim()
    nu = sim.solver.mj_model.nu
    with sim.lock:
        for _ in range(steps):
            ctrl = [0.0] * nu
            ctrl[4] = -0.005
            ctrl[5] = -0.005
            _newton_step(ctrl)
            sim.step_count += 1
        sim._sync_render_data()
    return {"status": "open", "step": sim.step_count}


@server.tool()
def close_gripper(steps: int = 30) -> dict[str, Any]:
    """Close the gripper fingers to grasp an object."""
    sim = _require_sim()
    nu = sim.solver.mj_model.nu
    with sim.lock:
        for _ in range(steps):
            ctrl = [0.0] * nu
            ctrl[4] = 0.04
            ctrl[5] = 0.04
            _newton_step(ctrl)
            sim.step_count += 1
        sim._sync_render_data()
    return {"status": "closed", "step": sim.step_count}


# ── Newton sensor tools ────────────────────────────────────────────────────


@server.tool()
def get_contact_forces(body_name: str) -> dict[str, Any]:
    """Get contact forces acting on a named body.

    Args:
        body_name: Body to query (e.g. "mug", "finger_left").
    """
    sim = _require_sim()
    model, data = sim.solver.mj_model, sim.solver.mj_data
    labels = [label for label in sim.newton_model.body_label if label.rsplit("/", 1)[-1] == body_name]
    if len(labels) != 1:
        return {"error": f"Body '{body_name}' not found unambiguously"}
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, labels[0].replace("/", "_"))
    if body_id < 0:
        return {"error": f"Body '{body_name}' not found"}

    body_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id}
    contacts = []
    total_force = 0.0

    for c in range(data.ncon):
        contact = data.contact[c]
        if contact.geom1 in body_geoms or contact.geom2 in body_geoms:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, force)
            force_magnitude = float(np.linalg.norm(force[:3]))
            total_force += force_magnitude
            contacts.append(
                {
                    "position": contact.pos.tolist(),
                    "normal_force": round(float(force[0]), 4),
                }
            )

    return {
        "body": body_name,
        "contact_count": len(contacts),
        "total_force_magnitude": round(total_force, 4),
        "contacts": contacts[:10],
    }


@server.tool()
def render_depth(camera: str = "overhead", width: int = 320, height: int = 240) -> dict[str, Any]:
    """Render a depth image. Returns per-pixel distances in meters (downsampled).

    Args:
        camera: Camera name.
        width: Image width.
        height: Image height.
    """
    sim = _require_sim()
    render_model = sim.mj_model or sim.solver.mj_model
    render_data = sim.mj_data or sim.solver.mj_data

    renderer = mujoco.Renderer(render_model, width=width, height=height)
    renderer.update_scene(render_data, _resolve_camera(render_model, camera))
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()

    extent = render_model.stat.extent
    near = render_model.vis.map.znear * extent
    far = render_model.vis.map.zfar * extent
    depth_meters = near / (1.0 - depth * (1.0 - near / far))
    depth_meters = np.clip(depth_meters, 0.0, far)

    stride = max(1, min(width, height) // 16)
    sampled = depth_meters[::stride, ::stride]

    return {
        "camera": camera,
        "min_depth": round(float(depth_meters.min()), 4),
        "max_depth": round(float(depth_meters.max()), 4),
        "depth_grid": np.round(sampled, 3).tolist(),
    }


@server.tool()
def get_scene_info() -> dict[str, Any]:
    """Get detailed information about the loaded scene from Newton."""
    sim = _require_sim()
    nm = sim.newton_model
    mj = sim.mj_model or sim.solver.mj_model

    return {
        "engine": "newton",
        "solver": "SolverMuJoCo",
        "newton_body_count": int(nm.body_count),
        "newton_joint_count": int(nm.joint_count),
        "newton_shape_count": int(nm.shape_count),
        "mujoco_actuators": mj.nu,
        "mujoco_cameras": mj.ncam,
        "timestep": sim.dt,
        "gravity": [round(float(mj.opt.gravity[i]), 4) for i in range(3)],
        "body_labels": [label.rsplit("/", 1)[-1] for label in nm.body_label],
    }


# ── Optional live 3D viewer ──────────────────────────────────────────────────
# GL is thread-affine, so this runs on the MAIN thread while the interface server +
# physics run on worker threads (see sim/host.py). It reads the shared _sim under its
# lock; physics mutates it from the tool handlers' threadpool.


def _make_viewer() -> tuple[Any, Any]:
    """Create the live viewer: the Gizmo desktop shell, else a bare Newton ViewerGL."""
    try:
        from gizmo.catalog import NewtonCatalog
        from gizmo.gl_app import GizmoGLApp, GizmoViewer

        viewer = GizmoViewer(width=1280, height=720)
        # Empty catalog: the harness loads scenes, not the Library panel.
        app = GizmoGLApp(NewtonCatalog(examples=(), assets=()), viewer, eval_mode=True)
        print("[viewer] Gizmo shell ready (eval mode).", file=sys.stderr)
        return viewer, app
    except Exception as exc:  # noqa: BLE001 - degrade to the bare viewer, never block
        print(
            f"[viewer] Gizmo shell unavailable ({type(exc).__name__}: {exc}); using bare Newton ViewerGL.",
            file=sys.stderr,
        )
        return newton.viewer.ViewerGL(width=1280, height=720), None


def run_viewer(driver: threading.Thread) -> None:
    """Render the live sim at ~30 fps on the main thread until ``driver`` ends.

    Falls back to simply waiting for ``driver`` when no display/GL is available,
    so a headless box still runs the rollout (just without a window).
    """
    try:
        viewer, app = _make_viewer()
    except Exception as exc:  # noqa: BLE001 - headless / no GL
        print(
            f"[viewer] no display ({type(exc).__name__}: {exc}); running headless. "
            f"Run on a machine with a screen (or via X/Xvfb) to see the window.",
            file=sys.stderr,
        )
        driver.join()
        return

    _sim.viewer = viewer
    driver_done = False
    frames_without_model = 0
    try:
        while True:
            with _sim.lock:
                model_pending = _sim.viewer_model_pending
                newton_model = _sim.newton_model
                state = _sim.state_0
                step_count = _sim.step_count
                dt = _sim.dt
                scene_id = _sim.scene_id
                solver_name = _sim.solver_name

            if model_pending and newton_model is not None:
                viewer.set_model(newton_model)
                viewer.set_camera(pos=wp.vec3(1.5, -1.0, 1.2), pitch=0.0, yaw=0.0)
                viewer.camera.look_at((0.35, 0.0, 0.45))
                if app is not None:
                    try:
                        app.populate_external(
                            model=newton_model,
                            scene_name=scene_id or "scene",
                            solver_name=solver_name or "SolverMuJoCo",
                        )
                    except Exception as exc:  # noqa: BLE001 - display only
                        print(f"[viewer] populate_external failed: {exc}", file=sys.stderr)
                with _sim.lock:
                    _sim.viewer_model_pending = False

            if newton_model is None:
                frames_without_model += 1
                if frames_without_model % 90 == 1:  # ~every 3s
                    print("[viewer] waiting for the agent to load a scene (only sky until then)...", file=sys.stderr)
            else:
                frames_without_model = 0

            sim_time = step_count * dt if newton_model else 0.0
            if app is not None:
                app.eval_step = step_count
                app.eval_time = sim_time

            viewer.begin_frame(sim_time)
            if newton_model is not None and state is not None:
                viewer.log_state(state)
            viewer.end_frame()

            if not driver.is_alive() and not driver_done:
                driver_done = True
                print("[viewer] rollout finished. Close the window to exit.", file=sys.stderr)

            time.sleep(1.0 / 30.0)
    except Exception as exc:  # noqa: BLE001 - window closed / GL teardown
        print(f"[viewer] closed ({type(exc).__name__}).", file=sys.stderr)


if __name__ == "__main__":
    # Standalone smoke run; in normal use sim/host.py serves this in its own process.
    server.run(show_banner=False)

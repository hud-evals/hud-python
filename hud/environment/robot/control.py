"""Direct control: an LLM drives a ``robot`` capability through MCP motion tools.

The ``robot`` wire takes one action per control tick, which a policy keeps up
with and a tool-calling LLM does not. :class:`DirectControl` serves the same sim
as an ``mcp`` capability instead: a tool call names targets (or displacements)
for some action dimensions, the tool plays them out on the wire, and answers
with the camera frames and the pose it ended on. ``wait`` holds the pose and
answers the same way. Each call is its own wire
connection, so the sim holds still while the model thinks. The env keeps
serving ``robot`` too, so one env serves VLAs and LLMs alike.

The contract derives the tool surface: the action's control kind (``state_type``,
else ``type``) picks the tool (:data:`CONTROL_KINDS`), its ``names`` are the
addressable dimensions, its ``limits`` (else ``stats``) ``min``/``max`` bound
them, ``control_rate`` paces the playout, and ``rgb`` observations are the
frames returned. An absolute
end-effector rotation is presented as intrinsic XYZ euler and slerped onto
whatever the contract stores (quaternion, axis-angle, or euler)::

    env = Environment(name="my-sim")
    sim = env.gym(make_env)
    DirectControl(notes="Gripper: -1 open, +1 closed.").attach(sim)

An absolute move then holds its goal until the arm (and, after a gripper change, the fingers)
comes to rest, and reports whether it is at the target, stopped short, or out of time. A
displacement returns once played. ``timeout`` caps one call.
Single-env sims only: a tool call claims the sole slot (no slot token).

Every tick of every call also streams to the rollout's trace: numeric state per
tick plus one H.264 video per camera, so the trace viewer replays the whole
episode, not only the frames the model is shown. The env process needs
``HUD_API_KEY`` to upload.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import math
import re
import socket
from collections.abc import Callable  # noqa: TC003 - pydantic resolves the field at runtime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import Tool
from jsonschema import ValidationError, validate
from mcp.types import ImageContent, TextContent
from PIL import Image

from hud.capabilities import Capability
from hud.capabilities.mcp import get_mcp_trace_id
from hud.capabilities.robot import RobotClient
from hud.environment.robot.orientation import (
    Orientation,
    angular_distance,
    euler_columns,
    euler_to_xyzw,
    find_orientations,
    slerp,
    wrap_angle,
    xyzw_to_euler,
)
from hud.telemetry.exporter import flush
from hud.telemetry.robot import TraceRecorder

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fastmcp.tools import ToolResult
    from numpy.typing import NDArray

    from hud.environment.robot.endpoint import RobotEndpoint


class ControlKind(NamedTuple):
    """How one Spec v0 ``state_type`` is served."""

    tool: str
    absolute: bool  # targets held until reached; else displacements played once
    quantity: str | None  # tolerance quantity of its plain scalars; None: nothing to reach


#: Control kind -> tool. A new kind is a row. ``BASE_ABS_TWIST`` (``walk``) needs a
#: duration and a stop-after law, so it is not served yet.
CONTROL_KINDS: dict[str, ControlKind] = {
    "EE_ABS_POSE": ControlKind("move_to", True, "position"),
    "EE_ABS_POS": ControlKind("move_to", True, "position"),
    "JOINT_ABS_POS": ControlKind("move_joints", True, "joint"),
    "EE_DEL_POSE": ControlKind("move_by", False, None),
    "EE_DEL_POS": ControlKind("move_by", False, None),
    "BASE_DEL_POSE": ControlKind("walk_by", False, None),
}

#: The control kind of a contract that only gives the coarse action ``type``.
_TYPE_KINDS = {"ee_abs": "EE_ABS_POSE", "joint_pos": "JOINT_ABS_POS", "ee_del": "EE_DEL_POSE"}


#: Default ``position`` tolerance, in metres. It only picks the verdict; the model never sees it.
DEFAULT_TOLERANCE_M = 1e-3
#: Fingertip metres per unit of each quantity (a 0.15 m gripper at the wrist, a 2.5 m summed
#: arm lever): the position tolerance over it is the rotation / joint default.
_LEVER_M = {"position": 1.0, "rotation": 0.15, "joint": 2.5}
#: How a verdict states each quantity's distance: (scale, unit, decimals).
_UNITS = {
    "position": (1e3, "mm", 2),
    "rotation": (1.0, "rad", 5),
    "joint": (1.0, "rad", 5),
    "grip": (1.0, "", 4),
}

#: At rest: over each of the last two windows of max(3 ticks, 0.25 s), every named arm
#: dimension's mean change per tick is under a tenth of its default tolerance.
_WINDOW_TICKS = 3
_WINDOW_S = 0.25
_FLOORS = {quantity: 0.1 * DEFAULT_TOLERANCE_M / lever for quantity, lever in _LEVER_M.items()}
#: Rest only counts once every arm dimension has closed half its commanded travel.
_RESPONDED = 0.5
#: A gripper change also waits for a finger to move faster than this (m/tick), then for all
#: fingers to stay slower for this many ticks.
_FINGER_REST_M = 5e-4
_FINGER_REST_TICKS = 3
#: Longest hold after the motion plays, and the rest an arm outside tolerance gets within it (s).
_SETTLE_S = 3.0
_GRACE_S = 1.5

#: Standard name of a native leaf (the last dotted part, after any ``left_`` / ``right_``).
_LEAF_STANDARD = {"drx": "roll", "dry": "pitch", "drz": "yaw"} | {
    leaf: axis for axis in ("x", "y", "z", "roll", "pitch", "yaw") for leaf in (axis, f"d{axis}")
}

#: Observation ``type`` tags that mark an image rather than a state vector.
IMAGE_TYPES = frozenset({"rgb", "bgr", "gray", "depth"})

#: Safety cap on one call. A full-range move at the default speed lasts 10 s, then
#: settles for at most 3 s.
DEFAULT_TIMEOUT_S = 30.0

#: Fraction of a full turn an absolute orientation travels per second, times ``speed``.
_FULL_TURN = 2.0 * math.pi

Content = list[TextContent | ImageContent]


#: Longest ``wait``, in sim seconds.
MAX_WAIT_S = 100.0

#: One plan: per-tick action rows, the intended goal (``None`` if nothing is being
#: reached), and the dimensions named.
Plan = tuple["NDArray[np.float64]", "NDArray[np.float64] | None", set[str]]

#: Asked of the model on every call (unless ``use_annotation`` is off) so the
#: transcript records what the model saw and why.
_NOTE = (
    "What you observe right now in the frames and state, and why you chose this. "
    "One or two plain sentences."
)


_WAIT = (
    f"Let `duration` sim seconds pass (at most {MAX_WAIT_S:g}) while the robot holds its pose, "
    "then return the camera frames and state. Pass null or 0 to just observe: "
    "one control tick passes while the robot holds its pose."
)
_DURATION = {"type": ["number", "null"], "description": "Sim seconds; null is 0."}


class _SchemaTool(Tool):
    """A tool that serves exactly the schema it advertises: arguments are validated
    against ``parameters`` and then passed to ``fn`` by keyword."""

    fn: Callable[..., Any]

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            validate(arguments, self.parameters)
        except ValidationError as exc:
            raise ToolError(exc.message) from exc
        return self.convert_result(await self.fn(**arguments))


class _Gap(NamedTuple):
    """How far one named dimension (or a whole rotation, ``orientation``) is from its goal."""

    label: str
    distance: float
    quantity: str


@dataclass(frozen=True)
class _Axis:
    """One tool dimension: a contract column, or one euler component of a rotation block."""

    kind: Literal["scalar", "euler"]
    index: int  # scalar: contract column; euler: roll/pitch/yaw component (0/1/2)
    block: Orientation | None = None


class DirectControl:
    """Serve a ``robot`` capability's action space as MCP motion tools.

    - ``dims`` - the dimensions a call may address, by tool or action name
      (default: all). The rest hold their commanded value. Orientation is
      addressed as ``roll`` / ``pitch`` / ``yaw``, not as quaternion components.
    - ``speed`` - absolute scalar moves travel this fraction of each dimension's
      range per second. Orientation travels this fraction of a full turn per second.
    - ``max_step`` - per-dimension step override by action name (e.g. a binary
      gripper that should switch in one step).
    - ``tolerance`` - how close counts as at the target, by quantity: ``position`` (m),
      ``rotation`` and ``joint`` (rad, default scaled from ``position``). Picks the verdict only.
    - ``timeout`` - safety cap, in seconds, on one call.
    - ``reference`` - maps observation data to the current action-space vector,
      where an absolute move starts on an episode's first call (later moves
      start from the last commanded target). Default: the one observation
      vector as wide as the action, which reads the gripper. A custom reference
      is taken not to read it, so gripper offsets are left out of ``off:`` and
      the verdict.
    - ``notes`` - embodiment facts the model needs (frames, units, gripper
      polarity), appended to the motion tool's description.
    - ``time_limit`` - the episode's budget in sim seconds (default: none). Spending it
      ends the episode as sim termination does.
    - ``use_annotation`` - ask the model for a ``note`` on every call (default).
    - ``use_wait`` - serve the ``wait`` tool (default).
    """

    def __init__(
        self,
        *,
        robot: str = "robot",
        name: str = "control",
        dims: list[str] | None = None,
        speed: float = 0.1,
        max_step: dict[str, float] | None = None,
        tolerance: dict[str, float] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        reference: Callable[[dict[str, NDArray[Any]]], NDArray[Any]] | None = None,
        notes: str = "",
        time_limit: float | None = None,
        use_annotation: bool = True,
        use_wait: bool = True,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be positive seconds, got {timeout}")
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError(f"time_limit must be positive seconds, got {time_limit}")
        unknown = sorted(set(tolerance or {}) - set(_LEVER_M))
        if unknown:
            raise ValueError(f"tolerance keys {unknown} are not quantities {sorted(_LEVER_M)}")
        self.robot = robot
        self.name = name
        self.dims = dims
        self.speed = speed
        self.max_step = max_step or {}
        position = (tolerance or {}).get("position", DEFAULT_TOLERANCE_M)
        self._tolerances = {
            quantity: (tolerance or {}).get(quantity, position / lever)
            for quantity, lever in _LEVER_M.items()
        }
        # A gripper has no tolerance: it never decides the verdict.
        self._tolerances["grip"] = math.inf
        self.timeout = timeout
        self.reference = reference
        self._reads_grip = reference is None
        self.notes = notes
        self.time_limit = time_limit
        self.use_annotation = use_annotation
        self.use_wait = use_wait
        # Appended to the motion and wait tool descriptions when a limit is enforced.
        self._budget = (
            f" You have {time_limit:g} seconds of simulated time. Time passes only while the "
            f"robot {'moves or waits' if use_wait else 'moves'}; thinking is free. When it runs "
            "out the episode ends and is graded as it stands."
            if time_limit is not None
            else ""
        )
        self._lock = asyncio.Lock()  # a slot takes one wire connection at a time
        self._command: NDArray[np.float64] | None = None  # this episode's last absolute target
        self._elapsed = 0  # ticks this episode has played
        self._serving: asyncio.Task[None] | None = None
        self._capability: Capability | None = None
        self._recorder: TraceRecorder | None = None  # this episode's trace telemetry
        self._tick = 0
        self._recording = asyncio.Lock()  # one tick is recorded (or the recorder closed) at a time

    def attach(self, endpoint: RobotEndpoint) -> DirectControl:
        """Register with a robotics endpoint, which owns serving and episode lifecycle."""
        endpoint.add_direct_control(self)
        return self

    async def start(self, robot: Capability) -> Capability:
        """Serve the tools derived from *robot*'s contract; returns their ``mcp`` capability."""
        if self._capability is not None:
            return self._capability
        # 1. Read the contract: the action's kind picks the tool, its names are the dimensions.
        self._robot = robot
        contract = robot.params["contract"]
        features: dict[str, dict[str, Any]] = contract["features"]
        action = next(f for f in features.values() if f["role"] == "action")
        kind_name = action.get("state_type") or _TYPE_KINDS.get(action.get("type") or "", "")
        kind = CONTROL_KINDS.get(kind_name)
        if kind is None:
            raise ValueError(
                f"direct control needs an action state_type in {sorted(CONTROL_KINDS)} (or a "
                f"type in {sorted(_TYPE_KINDS)}), got {kind_name or action.get('type')!r}"
            )
        self._tool, self._absolute = kind.tool, kind.absolute
        self._names: list[str] = list(action["names"])
        # Wrist rotation columns (quat / axis-angle / euler) are exposed as roll/pitch/yaw.
        self._orientations = find_orientations(self._names) if self._tool == "move_to" else []
        self._orient_columns = {
            column for block in self._orientations for column in block.contract_columns
        }
        self._axes, self._dims = self._addressable()
        # 2. Bounds and pacing: how far each dimension may go, and how far per tick.
        bounds = action.get("limits") or action["stats"]
        self._low = np.asarray(bounds["min"], dtype=np.float64)
        self._high = np.asarray(bounds["max"], dtype=np.float64)
        self._rate = float(contract["control_rate"])
        self._window = max(_WINDOW_TICKS, math.ceil(_WINDOW_S * self._rate))
        self._grace_ticks = max(1, math.ceil(_GRACE_S * self._rate))
        # The hold must outlast rest detection, which at low rates takes longer than _SETTLE_S.
        self._settle_ticks = max(
            math.ceil(_SETTLE_S * self._rate), 2 * self._window + self._grace_ticks
        )
        self._max_ticks = max(1, math.ceil(self.timeout * self._rate))
        self._limit_ticks = (
            None if self.time_limit is None else max(1, math.ceil(self.time_limit * self._rate))
        )
        # Per-tick limit: a speed-paced share of the range, or the per-step box for deltas.
        self._step = (
            self.speed * (self._high - self._low) / self._rate
            if self._absolute
            else np.minimum(self._high, -self._low)
        )
        for dim, step in self.max_step.items():
            self._step[self._names.index(dim)] = step
        if np.any(self._step <= 0):
            raise ValueError(f"every action dimension needs a positive step, got {self._step}")
        # What each tool dimension of an absolute kind measures, which picks its tolerance.
        self._quantity = {
            name: "rotation"
            if axis.block is not None
            else "grip"
            if "grip" in self._names[axis.index]
            else kind.quantity
            for name, axis in self._axes.items()
            if kind.quantity is not None
        }
        # 3. Observations: cameras become returned frames, the rest become labeled state text.
        observations = {n: f for n, f in features.items() if f["role"] == "observation"}
        self._obs_space = observations
        self._cameras = [n for n, f in observations.items() if f.get("type") == "rgb"]
        self._states = {
            n: f.get("names") for n, f in observations.items() if f.get("type") not in IMAGE_TYPES
        }
        self._fingers = [
            (name, key, column)
            for key, names in self._states.items()
            for column, name in enumerate(names or [])
            if "grip" in name
        ]
        self._euler_columns = {
            key: euler_columns(names or []) for key, names in self._states.items()
        }
        # Absolute moves need a start pose: default to the state vector as wide as the action.
        self._reference = self.reference
        if self._absolute and self._reference is None:
            wide = [n for n, names in self._states.items() if len(names or ()) == len(self._names)]
            if len(wide) != 1:
                raise ValueError(
                    f"absolute control needs a reference: no single observation is as wide as "
                    f"the action ({len(self._names)}); pass reference=..."
                )
            self._reference = lambda data: data[wide[0]]

        # 4. Serve the tools over loopback HTTP on a free port; publish them as an mcp capability.
        server = FastMCP(name=self.name)
        self._bind_tools(server)
        if self.use_wait:
            server.add_tool(
                self._make_tool(self.wait, "wait", _WAIT + self._budget, {"duration": _DURATION})
            )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self._serving = asyncio.create_task(
            server.run_async(
                transport="http",
                host="127.0.0.1",
                port=port,
                show_banner=False,
            )
        )
        await _wait_until_listening(self._serving, port)
        self._capability = Capability.mcp(name=self.name, url=f"http://127.0.0.1:{port}/mcp")
        return self._capability

    async def stop(self) -> None:
        await self.end_episode()
        if self._serving is None:
            return
        self._serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._serving
        self._serving = None
        self._capability = None

    async def end_episode(self) -> None:
        """Clear the last absolute target and time spent, and close the episode's trace telemetry.

        A move still playing stops recording; the tick being recorded finishes first.
        """
        self._command = None
        self._elapsed = 0
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            async with self._recording:
                await asyncio.to_thread(recorder.close)  # flushes the video tails
                await asyncio.to_thread(flush)

    # ── tools ──────────────────────────────────────────────────────────────

    def _bind_tools(self, server: FastMCP) -> None:
        """Register the contract motion tool. Override to serve an env-specific tool."""
        server.add_tool(
            self._make_tool(
                self.move, self._tool, self._describe(), {"targets": self._targets_schema()}
            )
        )

    def _make_tool(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        properties: dict[str, Any],
    ) -> Tool:
        """A tool whose strict-mode schema requires every property; ``note`` joins if annotating."""
        if self.use_annotation:
            properties = {**properties, "note": {"type": "string", "description": _NOTE}}
        return _SchemaTool(
            name=name, description=description, parameters=_object(properties), fn=fn
        )

    # The one tool body; registered under the control kind's tool name (move_to, move_by, ...).
    async def move(
        self, targets: dict[str, float | dict[str, float | None] | None], note: str = ""
    ) -> Content:
        # A group's dimensions are named ``group.leaf``, as in the contract; null ones hold.
        flat: dict[str, float | None] = {}
        for key, value in targets.items():
            if isinstance(value, dict):
                flat.update({f"{key}.{leaf}": number for leaf, number in value.items()})
            else:
                flat[key] = value
        values = {name: value for name, value in flat.items() if value is not None}
        return await self._play(lambda obs: self._plan(values, obs), note)

    async def wait(self, duration: float | None, note: str = "") -> Content:
        seconds = duration or 0.0
        if not 0 <= seconds <= MAX_WAIT_S:
            raise ToolError(f"duration must be within [0, {MAX_WAIT_S:g}] s, got {seconds:g}")
        # At least one tick: the trace viewer needs every step to span one.
        ticks = max(1, round(seconds * self._rate))
        return await self._play(
            lambda obs: (np.repeat(self._start(obs)[None], ticks, axis=0), None, set()),
            note,
        )

    def _targets_schema(self) -> dict[str, Any]:
        """Every dimension, nullable (null holds it). ``group.leaf`` names nest under ``group``."""
        targets: dict[str, Any] = {}
        groups: dict[str, dict[str, Any]] = {}
        for name in self._dims:
            axis = self._axes[name]
            # Euler dimensions are unbounded in the contract; report them as ±pi.
            low, high = (
                (-math.pi, math.pi)
                if axis.kind == "euler"
                else (self._low[axis.index], self._high[axis.index])
            )
            number = {"type": ["number", "null"], "description": f"[{low:.4g}, {high:.4g}]"}
            group, dot, leaf = name.partition(".")
            if dot:
                groups.setdefault(group, {})[leaf] = number
            else:
                targets[name] = number
        for group, leaves in groups.items():
            targets[group] = {"anyOf": [_object(leaves), {"type": "null"}]}
        return _object(targets)

    def _describe(self) -> str:
        hold = "Set a dimension (or a whole group) to null to leave it as it is."
        if self._absolute:
            motion = (
                "Move to absolute targets. Null dimensions hold their commanded value. "
                f"The motion is interpolated at a safe speed: {self.speed * 100:g}% of each "
                f"range per second, so a full-range move takes about {1 / self.speed:g} s."
            )
            done = (
                "The call returns once the arm comes to rest (at most "
                f"{_SETTLE_S:g} s after the motion plays) and reports whether it is at the "
                "target, stopped short of it, or ran out of time while still moving"
            )
        else:
            motion = (
                "Move by a displacement, in the action's own components (a rotation is not "
                "euler angles). Null dimensions do not move. The motion is split into steps "
                f"within the per-step bounds, played at {self._rate:g} steps per second."
            )
            done = "The call returns once the displacement has played"
        rotation = ""
        if self._orientations:
            rotation = (
                " Orientation is intrinsic XYZ euler in radians (roll, pitch, yaw), "
                "converted to the arm's rotation and interpolated on the shortest arc, "
                f"at {self.speed * 360:g} degrees per second. Angles are periodic: "
                "pi and -pi are the same orientation, and the pose reports them in [-pi, pi)."
            )
        annotation = (
            " Include a note saying what you see and why you chose the motion."
            if self.use_annotation
            else ""
        )
        return (
            f"{motion} {hold}{rotation} {done}, with the camera frames and the pose. It stops "
            f"at {self.timeout:g} s if the motion has not finished.{self._budget}{annotation} "
            f"{self.notes}"
        ).strip()

    def _addressable(self) -> tuple[dict[str, _Axis], list[str]]:
        """Tool dimensions. Contract orientation blocks become roll, pitch, yaw; every name
        becomes its standard name when all have distinct ones."""
        axes: dict[str, _Axis] = {}
        covered: set[int] = set()
        for block in self._orientations:
            covered.update(block.contract_columns)
            for component, name in enumerate(block.tool_names):
                axes[name] = _Axis("euler", component, block)
        for index, name in enumerate(self._names):
            if index not in covered:
                axes[name] = _Axis("scalar", index)
        order = _tool_order(self._names, self._orientations)
        rename = {name: standard for name in order if (standard := _standard_name(name))}
        if len(set(rename.values())) < len(order):
            rename = {name: name for name in order}
        axes = {rename[name]: axis for name, axis in axes.items()}
        if self.dims is None:
            return axes, [rename[name] for name in order]
        # A dims entry may name a tool dimension, its native name, or any contract column of
        # a rotation block.
        aliases = {name: [tool] for native, tool in rename.items() for name in (native, tool)}
        for block in self._orientations:
            tools = [rename[name] for name in block.tool_names]
            aliases.update({self._names[column]: tools for column in block.contract_columns})
        unknown = sorted(name for name in self.dims if name not in aliases)
        if unknown:
            raise ValueError(f"dims {unknown} are not dimensions of this tool ({', '.join(axes)})")
        chosen = [tool for name in self.dims for tool in aliases[name]]
        return axes, list(dict.fromkeys(chosen))

    # ── the wire ───────────────────────────────────────────────────────────

    async def _play(self, plan: Callable[[dict[str, Any]], Plan], note: str) -> Content:
        """Plan against the live observation, play until done, and render the result."""
        # The note stays on the tool call; an empty one is a correctable miss, not a move.
        if self.use_annotation and not note.strip():
            raise ToolError("note must say what you see and why you chose this")
        # One short wire connection per call: the sim is frozen while the model thinks.
        async with self._lock:
            client = await RobotClient.connect(self._robot)
            try:
                obs = await client.get_observation()
                if self._recorder is None and (trace_id := get_mcp_trace_id()) is not None:
                    self._tick = 0
                    self._recorder = TraceRecorder(
                        trace_id=trace_id,
                        fps=max(1, round(self._rate)),
                        obs_space=self._obs_space,
                        sim_clock=True,
                        lossless_video=True,
                    )
                    await self._record(obs)  # the episode's opening scene
                # A spent time limit ends the episode the way sim termination does.
                ticks_left = (
                    None if self._limit_ticks is None else self._limit_ticks - self._elapsed
                )
                if ticks_left == 0:
                    obs["terminated"] = True
                rows, goal, named = (
                    (np.zeros((0, len(self._names))), None, set())
                    if obs["terminated"]
                    else plan(obs)
                )
                played = 0
                proprio = self._proprio(obs)
                # Each tick since the call's start: every named dimension's speed.
                speeds: list[NDArray[np.float64]] = []

                def measure(
                    vector: NDArray[np.float64], target: NDArray[np.float64]
                ) -> NDArray[np.float64]:
                    """Every named dimension's distance between two poses."""
                    return np.array([gap.distance for gap in self._gaps(vector, target, named)])

                fingers = [self._finger_positions(obs)]
                start_pose, start_command = proprio, self._start(obs)

                async def step(row: NDArray[np.float64]) -> None:
                    nonlocal played, obs, proprio
                    await client.send_action(row)
                    obs = await client.get_observation()
                    await self._record(obs)
                    played += 1
                    if ticks_left is not None and played >= ticks_left:
                        obs["terminated"] = True
                    if self._absolute:
                        # Remember the target so the next call continues from it, not the sim pose.
                        self._command = row.copy()
                    previous, proprio = proprio, self._proprio(obs)
                    fingers.append(self._finger_positions(obs))
                    if goal is not None:
                        speeds.append(measure(proprio, previous))

                for row in rows:
                    await step(row)
                    if obs["terminated"]:
                        break
                # Then hold the goal, for at most the settle cap, until the arm is at rest within
                # tolerance, or has stayed at rest outside it for the grace period.
                rested = False
                if goal is not None:
                    gaps = self._gaps(proprio, goal, named)
                    # A gripper is never held for here: its fingers are waited for instead.
                    floor = np.array([_FLOORS.get(gap.quantity, np.inf) for gap in gaps])
                    tolerance = np.array([self._tolerances[gap.quantity] for gap in gaps])
                    travel = measure(start_command, goal)
                    gap_at_start = measure(start_pose, goal)
                    wait_fingers = bool(self._fingers) and any(
                        gap.quantity == "grip" and distance > 0
                        for gap, distance in zip(gaps, travel, strict=True)
                    )
                    held = 0
                    rest_ticks = 0
                    while True:
                        remaining = measure(proprio, goal)
                        finger_speed = np.abs(np.diff(fingers, axis=0))
                        recent = speeds[-2 * self._window :]
                        rested = (
                            len(recent) == 2 * self._window
                            and bool(
                                np.all(np.reshape(recent, (2, self._window, -1)).mean(1) < floor)
                            )
                            and (
                                not wait_fingers
                                or bool(np.all(finger_speed[-_FINGER_REST_TICKS:] < _FINGER_REST_M))
                            )
                        )
                        # Slow to start looks like rest, so returning early waits for a response.
                        responded = bool(
                            np.all(
                                (travel <= floor)
                                | (gap_at_start - remaining >= _RESPONDED * travel)
                            )
                        ) and (not wait_fingers or bool(np.any(finger_speed > _FINGER_REST_M)))
                        rest_ticks = rest_ticks + 1 if rested else 0
                        arrived = bool(np.all(remaining <= tolerance))
                        if (
                            (rested and responded and (arrived or rest_ticks > self._grace_ticks))
                            or obs["terminated"]
                            or held >= self._settle_ticks
                            or played >= self._max_ticks
                        ):
                            break
                        await step(goal)
                        held += 1
                gaps = None if goal is None else self._gaps(proprio, goal, named)
            finally:
                await client.close()
        self._elapsed += played
        return self._render(
            obs,
            played,
            proprio,
            rested=rested,
            gaps=gaps,
            goal=goal,
            ticks_left=None if self._limit_ticks is None else self._limit_ticks - self._elapsed,
        )

    async def _record(self, obs: dict[str, Any]) -> None:
        """Stream one tick's state and camera frames to the trace."""
        if (recorder := self._recorder) is not None:
            async with self._recording:
                if recorder is self._recorder:  # the episode may have ended while waiting
                    # Off the loop: a lossless encoder blocks while its queue drains.
                    await asyncio.to_thread(
                        recorder.record_observation, obs["data"], tick=self._tick
                    )
                    self._tick += 1

    def _start(self, obs: dict[str, Any]) -> NDArray[np.float64]:
        """Where a move begins: the commanded target (the reference on an episode's first
        move) for absolute control, zero displacement otherwise."""
        if not self._absolute:
            return np.zeros(len(self._names), dtype=np.float64)
        return self._command.copy() if self._command is not None else self._proprio(obs)

    def _plan(self, values: dict[str, float], obs: dict[str, Any]) -> Plan:
        """Action rows (cut at the safety cap), the intended goal, and named dims."""
        if not values:
            look = " (`wait` to just look)" if self.use_wait else ""
            raise ToolError(f"set at least one action dimension{look}")
        start = self._start(obs)
        end = start.copy()  # unnamed dimensions keep their start value (they hold)
        named = set(values)
        # Scalar dimensions: range-check and write straight into the goal.
        for name, value in values.items():
            axis = self._axes[name]
            if axis.kind == "scalar":
                index = axis.index
                if self._absolute and not self._low[index] <= value <= self._high[index]:
                    raise ToolError(
                        f"{name}={value} is outside "
                        f"[{self._low[index]:.4g}, {self._high[index]:.4g}]"
                    )
                end[index] = value
        # Rotations: start from the current euler, overwrite the named angles, convert back
        # to the contract's representation.
        addressed = [b for b in self._orientations if any(self._axes[n].block is b for n in named)]
        for block in addressed:
            roll, pitch, yaw = xyzw_to_euler(block.read(start))
            euler = [roll, pitch, yaw]
            for name, value in values.items():
                axis = self._axes[name]
                if axis.kind == "euler" and axis.block is block:
                    euler[axis.index] = value
            target_q = euler_to_xyzw(euler[0], euler[1], euler[2])
            # Same rotation as now: keep the contract components bit for bit.
            if angular_distance(block.read(start), target_q) <= 1e-8:
                continue
            block.write(end, target_q)
        # Duration: the slowest dimension sets the tick count (scalars by step, rotations by angle).
        span = end - start
        scalar = [i for i in range(len(self._names)) if i not in self._orient_columns]
        ticks = 1
        if scalar:
            ticks = max(
                ticks,
                math.ceil(float(np.max(np.abs(span[scalar]) / self._step[scalar]))),
            )
        quats: list[tuple[Orientation, NDArray[np.float64], NDArray[np.float64]]] = []
        for block in addressed:
            start_q = block.read(start)
            end_q = block.read(end)
            angle = angular_distance(start_q, end_q)
            if angle > 1e-8:
                step = self.speed * _FULL_TURN / self._rate
                ticks = max(ticks, math.ceil(angle / step))
                quats.append((block, start_q, end_q))
        # Safety cap: play only the prefix that fits.
        play = min(ticks, self._max_ticks)
        clipped = play < ticks
        fractions = np.linspace(1 / ticks, play / ticks, play)  # progress 0→1 per tick
        # Build one action row per tick.
        if not self._absolute:
            rows = np.repeat((span / ticks)[None], play, axis=0)
        elif not self._orientations:
            rows = start + span * fractions[:, None]
        else:
            rows = np.repeat(start[None], play, axis=0)
            for index in scalar:
                rows[:, index] = start[index] + span[index] * fractions
            # Rotations are slerped (shortest arc), never lerped component-wise.
            for block, start_q, end_q in quats:
                for row, fraction in zip(rows, fractions, strict=True):
                    block.write(row, slerp(start_q, end_q, float(fraction)))
            if not clipped:
                rows[-1] = end  # land exactly on the goal, free of round-trip rotation error
        return rows, end if self._absolute else None, named

    def _gaps(
        self, proprio: NDArray[np.float64], goal: NDArray[np.float64], named: set[str]
    ) -> list[_Gap]:
        """Each named dimension's gap; a rotation is one ``orientation``."""
        gaps: list[_Gap] = []
        blocks: set[int] = set()
        for name in self._dims:
            if name not in named:
                continue
            axis = self._axes[name]
            if axis.block is None:
                offset = abs(float(proprio[axis.index] - goal[axis.index]))
                gaps.append(_Gap(name, offset, self._quantity[name]))
            elif id(axis.block) not in blocks:
                blocks.add(id(axis.block))
                offset = angular_distance(axis.block.read(proprio), axis.block.read(goal))
                gaps.append(_Gap("orientation", offset, "rotation"))
        return gaps

    def _finger_positions(self, obs: dict[str, Any]) -> NDArray[np.float64]:
        return np.array(
            [np.asarray(obs["data"][key]).reshape(-1)[column] for _, key, column in self._fingers],
            dtype=np.float64,
        )

    def _proprio(self, obs: dict[str, Any]) -> NDArray[np.float64]:
        if self._reference is not None:
            return np.asarray(self._reference(obs["data"]), dtype=np.float64).reshape(-1)
        if self._states:
            key = next(iter(self._states))
            return np.asarray(obs["data"][key], dtype=np.float64).reshape(-1)
        return np.zeros(0, dtype=np.float64)

    def _render(
        self,
        obs: dict[str, Any],
        played: int,
        proprio: NDArray[np.float64],
        *,
        rested: bool,
        gaps: list[_Gap] | None,
        goal: NDArray[np.float64] | None,
        ticks_left: int | None,
    ) -> Content:
        data = obs["data"]
        # Keep this opening line: the trace viewer reads it to place each call on the video.
        opening = f"Played {played} steps ({played / self._rate:.1f} s)."
        if ticks_left is not None:
            opening += f" Time remaining: {ticks_left / self._rate:.1f} s."
        lines = [opening]
        for key, names in self._states.items():
            vector = np.array(data[key], dtype=np.float64).reshape(-1)
            angles = self._euler_columns[key]
            vector[angles] = wrap_angle(vector[angles])
            labels = names or [str(i) for i in range(vector.size)]
            lines.append(f"{key}: " + _labeled(zip(labels, vector, strict=True)))
        if proprio.size == len(self._names):
            lines.append("pose: " + _labeled(self._reading(proprio)))
        command = goal if goal is not None else self._command
        if command is not None:
            lines.append("commanded: " + _labeled(self._reading(command)))
        if goal is not None:
            lines.append("off: " + self._off(proprio, goal))
        arm = [gap for gap in gaps or [] if gap.quantity != "grip"]
        if goal is not None and not arm and self._fingers:
            # A gripper call has no verdict: a jam on an object must not read as success.
            names = [name for name, _, _ in self._fingers]
            lines.append(
                "fingers: " + _labeled(zip(names, self._finger_positions(obs), strict=True))
            )
        if arm:
            shown = [gap for gap in gaps or [] if self._reads_grip or gap.quantity != "grip"]
            # Largest first, measured against each dimension's tolerance so mm and rad compare.
            ranked = sorted(
                shown, key=lambda gap: gap.distance / self._tolerances[gap.quantity], reverse=True
            )
            if not rested:
                verdict = (
                    "The episode ended before the arm settled."
                    if obs["terminated"]
                    else "Ran out of time before settling; the arm was still moving."
                )
            elif any(gap.distance > self._tolerances[gap.quantity] for gap in arm):
                verdict = "Stopped short of the target; the arm is at rest."
            else:
                verdict = "At the target."
            lines.append(
                f"{verdict} Remaining offsets: {', '.join(_offset(gap) for gap in ranked)}."
            )
        if ticks_left == 0:
            lines.append("The time limit is reached.")
        if obs["terminated"]:
            lines.append("The episode has ended; stop calling tools.")
        content: Content = [TextContent(type="text", text="\n".join(lines))]
        for camera in self._cameras:
            content.append(TextContent(type="text", text=f"camera {camera}:"))
            content.append(
                ImageContent(type="image", data=_png(data[camera]), mimeType="image/png")
            )
        return content

    def _reading(self, vector: NDArray[np.float64]) -> list[tuple[str, float]]:
        euler: dict[int, NDArray[np.float64]] = {}
        reading: list[tuple[str, float]] = []
        for name in self._dims:
            axis = self._axes[name]
            if axis.kind == "scalar":
                reading.append((name, float(vector[axis.index])))
                continue
            assert axis.block is not None
            key = id(axis.block)
            if key not in euler:
                angles = np.asarray(xyzw_to_euler(axis.block.read(vector)))
                euler[key] = wrap_angle(angles)
            reading.append((name, float(euler[key][axis.index])))
        return reading

    def _off(self, proprio: NDArray[np.float64], goal: NDArray[np.float64]) -> str:
        parts: list[str] = []
        reported: set[int] = set()
        for name in self._dims:
            axis = self._axes[name]
            if axis.kind == "scalar":
                if self._reads_grip or self._quantity[name] != "grip":
                    parts.append(f"{name}={float(proprio[axis.index] - goal[axis.index]):.4f}")
                continue
            assert axis.block is not None
            key = id(axis.block)
            if key in reported:
                continue
            reported.add(key)
            angle = angular_distance(axis.block.read(proprio), axis.block.read(goal))
            parts.append(f"orientation={angle:.4f} rad")
        return ", ".join(parts)


def _tool_order(names: list[str], blocks: list[Orientation]) -> list[str]:
    order: list[str] = []
    index = 0
    while index < len(names):
        block = next((b for b in blocks if index in b.contract_columns), None)
        if block is None:
            order.append(names[index])
            index += 1
            continue
        order.extend(block.tool_names)
        index = max(block.contract_columns) + 1
    return order


def _standard_name(name: str) -> str | None:
    """*name* in the standard convention (``x``, ``roll``, ``grip``, ``joint3``; under
    ``left_arm.`` / ``right_arm.`` for a ``left_`` / ``right_`` name), or ``None``."""
    side = re.match(r"(left|right)[_.]", name)
    rest = name[side.end() :] if side else name
    leaf = rest.rsplit(".", 1)[-1]
    if "grip" in rest:
        standard = "grip"
    elif joint := re.search(r"joint_?(\d+)$", leaf):
        standard = f"joint{joint[1]}"
    else:
        standard = _LEAF_STANDARD.get(leaf)
    if standard is None or side is None:
        return standard
    return f"{side[1]}_arm.{standard}"


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """A strict-mode object schema: every property required, none extra."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _offset(gap: _Gap) -> str:
    scale, unit, decimals = _UNITS[gap.quantity]
    return f"{round(gap.distance * scale, decimals):.4g}{' ' + unit if unit else ''} on {gap.label}"


def _labeled(pairs: Iterable[tuple[str, Any]]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in pairs)


def _png(frame: NDArray[Any]) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _wait_until_listening(task: asyncio.Task[None], port: int) -> None:
    """Wait for FastMCP's loopback listener, surfacing startup failure."""
    while True:
        if task.done():
            await task
            raise RuntimeError("direct-control MCP server stopped during startup")
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.01)
            continue
        writer.close()
        await writer.wait_closed()
        return


__all__ = ["CONTROL_KINDS", "DEFAULT_TIMEOUT_S", "DirectControl"]

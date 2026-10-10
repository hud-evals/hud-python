"""Direct control: an LLM drives a ``robot`` capability through MCP motion tools.

The ``robot`` wire takes one action per control tick, which a policy keeps up
with and a tool-calling LLM does not. :class:`DirectControl` serves the same sim
as an ``mcp`` capability instead: a tool call sets targets (or displacements)
for some action dimensions, the tool plays them out on the wire, and answers
with the camera frames and the pose it ended on. A ``wait`` tool plays nothing
(or holds the pose for a while) and answers the same way. Each call is its own
wire connection, so the sim holds still while the model thinks. The env keeps
serving ``robot`` too, so one env serves VLAs and LLMs alike.

The contract derives the tool surface: the action ``type`` picks the motion
tool (:data:`MOTION_TOOLS`), its ``names`` are the addressable dimensions, its
``limits`` (else ``stats``) ``min``/``max`` bound them, ``control_rate`` paces
the playout, and ``rgb`` observations are the frames returned. An absolute
end-effector rotation is presented as intrinsic XYZ euler and slerped onto
whatever the contract stores (quaternion, axis-angle, or euler)::

    env = Environment(name="my-sim")
    sim = env.gym(make_env)
    DirectControl(notes="Gripper: -1 open, +1 closed.").attach(sim)

A call steps until the arm reaches the target or stops moving, then returns
that pose. ``timeout`` is only a safety cap. Single-env sims only: a tool call
claims the sole slot (no slot token).

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
import socket
from collections.abc import Callable  # noqa: TC003 - pydantic resolves the field at runtime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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
    euler_to_xyzw,
    find_orientations,
    slerp,
    xyzw_to_euler,
)
from hud.telemetry.exporter import flush
from hud.telemetry.robot import TraceRecorder

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fastmcp.tools import ToolResult
    from numpy.typing import NDArray

    from hud.environment.robot.endpoint import RobotEndpoint

#: Contract action ``type`` -> (motion tool, whether its values are absolute targets).
MOTION_TOOLS: dict[str, tuple[str, bool]] = {
    "ee_abs": ("move_to", True),
    "joint_pos": ("move_joints", True),
    "ee_del": ("move_by", False),
}

#: Observation ``type`` tags that mark an image rather than a state vector.
IMAGE_TYPES = frozenset({"rgb", "bgr", "gray", "depth"})

#: Safety cap on one call. A full-range move at the default speed lasts 10 s;
#: 60 s is well above that, including the 10.1 s move that used to be rejected.
DEFAULT_TIMEOUT_S = 60.0

#: Fraction of a full turn an absolute orientation travels per second, times ``speed``.
_FULL_TURN = 2.0 * math.pi

#: An addressed scalar is reached within this fraction of its range, and at least this
#: absolute gap. Orientation is reached within ``_REACH_RAD``. Motion counts as stopped
#: once proprioception changes by less than ``_STILL_ABS`` for ``_STILL_TICKS`` ticks
#: after it has moved.
_REACH_FRAC = 0.01
_REACH_ABS = 1e-3
_REACH_RAD = 0.05
_STILL_ABS = 1e-4
_STILL_TICKS = 2

Content = list[TextContent | ImageContent]


#: Longest ``wait``, in sim seconds.
MAX_WAIT_S = 300.0

#: One plan: per-tick action rows, the intended goal (``None`` if nothing is being
#: reached), whether the safety cap clipped the rows, and the dimensions named.
Plan = tuple["NDArray[np.float64]", "NDArray[np.float64] | None", bool, set[str]]

#: Asked of the model on every call (unless ``use_annotation`` is off) so the
#: transcript records what the model saw and why.
_NOTE = (
    "What you observe right now in the frames and state, and why you chose this. "
    "One or two plain sentences."
)


_WAIT = (
    f"Let `duration` sim seconds pass (at most {MAX_WAIT_S:g}) while the robot holds its pose, "
    "then return the camera frames and state. Pass null or 0 to let no time pass: "
    "a no-op that just shows the current state."
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


@dataclass(frozen=True)
class _Axis:
    """One tool dimension: a contract column, or one euler component of a rotation block."""

    kind: Literal["scalar", "euler"]
    index: int  # scalar: contract column; euler: roll/pitch/yaw component (0/1/2)
    block: Orientation | None = None


class DirectControl:
    """Serve a ``robot`` capability's action space as MCP motion tools.

    - ``dims`` - the dimensions a call may address (default: all). The rest hold
      their commanded value. Orientation is addressed as ``roll`` / ``pitch`` /
      ``yaw``, not as quaternion components.
    - ``speed`` - absolute scalar moves travel this fraction of each dimension's
      range per second. Orientation travels this fraction of a full turn per second.
    - ``max_step`` - per-dimension step override by action name (e.g. a binary
      gripper that should switch in one step).
    - ``timeout`` - safety cap, in seconds, on one call. The call returns earlier,
      when the target is reached or the arm stops. Default 60 s.
    - ``reference`` - maps observation data to the current action-space vector,
      where an absolute move starts on an episode's first call (later moves
      start from the last commanded target). Default: the one observation
      vector as wide as the action.
    - ``notes`` - embodiment facts the model needs (frames, units, gripper
      polarity), appended to the motion tool's description.
    - ``time_limit`` - the episode's budget in sim seconds (default: none). Every
      reply reports the time remaining. Spending it ends the episode as sim
      termination does: the call in flight stops at the limit, later calls play
      nothing, and replies say the episode has ended. :attr:`prompt` tells the
      model about the limit.
    - ``use_annotation`` - ask the model for a ``note`` on every call (default).
      Off, the tools take no ``note``.
    """

    def __init__(
        self,
        *,
        robot: str = "robot",
        name: str = "control",
        dims: list[str] | None = None,
        speed: float = 0.1,
        max_step: dict[str, float] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        reference: Callable[[dict[str, NDArray[Any]]], NDArray[Any]] | None = None,
        notes: str = "",
        time_limit: float | None = None,
        use_annotation: bool = True,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be positive seconds, got {timeout}")
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError(f"time_limit must be positive seconds, got {time_limit}")
        self.robot = robot
        self.name = name
        self.dims = dims
        self.speed = speed
        self.max_step = max_step or {}
        self.timeout = timeout
        self.reference = reference
        self.notes = notes
        self.time_limit = time_limit
        self.use_annotation = use_annotation
        #: Appended to the episode prompt so the model knows its budget.
        self.prompt = (
            f"\n\nYou have {time_limit:g} seconds of simulated time."
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
        # 1. Read the contract: the action's type picks the tool, its names are the dimensions.
        self._robot = robot
        contract = robot.params["contract"]
        features: dict[str, dict[str, Any]] = contract["features"]
        action = next(f for f in features.values() if f["role"] == "action")
        if action.get("type") not in MOTION_TOOLS:
            raise ValueError(
                f"direct control needs an action type in {sorted(MOTION_TOOLS)}, "
                f"got {action.get('type')!r}"
            )
        self._tool, self._absolute = MOTION_TOOLS[action["type"]]
        self._names: list[str] = list(action["names"])
        # Wrist rotation columns (quat / axis-angle / euler) are exposed as roll/pitch/yaw.
        self._orientations = (
            find_orientations(self._names)
            if self._absolute and action.get("type") == "ee_abs"
            else []
        )
        self._orient_columns = {
            column for block in self._orientations for column in block.contract_columns
        }
        self._axes, self._dims = self._addressable()
        # 2. Bounds and pacing: how far each dimension may go, and how far per tick.
        bounds = action.get("limits") or action["stats"]
        self._low = np.asarray(bounds["min"], dtype=np.float64)
        self._high = np.asarray(bounds["max"], dtype=np.float64)
        self._rate = float(contract["control_rate"])
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
        # 3. Observations: cameras become returned frames, the rest become labeled state text.
        observations = {n: f for n, f in features.items() if f["role"] == "observation"}
        self._obs_space = observations
        self._cameras = [n for n, f in observations.items() if f.get("type") == "rgb"]
        self._states = {
            n: f.get("names") for n, f in observations.items() if f.get("type") not in IMAGE_TYPES
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
        server.add_tool(self._make_tool(self.wait, "wait", _WAIT, {"duration": _DURATION}))
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

    # The one tool body; registered as move_to / move_joints / move_by by the contract.
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
        ticks = round(seconds * self._rate)
        return await self._play(
            lambda obs: (np.repeat(self._start(obs)[None], ticks, axis=0), None, False, set()),
            note,
        )

    def _targets_schema(self) -> dict[str, Any]:
        """Every dimension, nullable (null holds it). ``group.leaf`` names nest under ``group``."""
        targets: dict[str, Any] = {}
        groups: dict[str, dict[str, Any]] = {}
        for name in self._dims:
            low, high = self._range(name)
            number = {"type": ["number", "null"], "description": f"[{low:.4g}, {high:.4g}]"}
            group, dot, leaf = name.partition(".")
            if dot:
                groups.setdefault(group, {})[leaf] = number
            else:
                targets[name] = number
        for group, leaves in groups.items():
            targets[group] = {"anyOf": [_object(leaves), {"type": "null"}]}
        return _object(targets)

    def _range(self, name: str) -> tuple[float, float]:
        # Euler dimensions are unbounded in the contract; report them as ±pi.
        axis = self._axes[name]
        if axis.kind == "euler":
            return -math.pi, math.pi
        return float(self._low[axis.index]), float(self._high[axis.index])

    def _describe(self) -> str:
        hold = "Set a dimension (or a whole group) to null to leave it as it is."
        if self._absolute:
            motion = (
                "Move to absolute targets. Null dimensions hold their commanded value. "
                "The motion is interpolated at a safe speed."
            )
        else:
            motion = (
                "Move by a displacement. Null dimensions do not move. "
                "The motion is split into steps within the per-step bounds."
            )
        rotation = ""
        if self._orientations:
            rotation = (
                " Orientation is intrinsic XYZ euler in radians (roll, pitch, yaw), "
                "converted to the arm's rotation and interpolated on the shortest arc."
            )
        annotation = (
            " Include a note saying what you see and why you chose the motion."
            if self.use_annotation
            else ""
        )
        return (
            f"{motion} {hold}{rotation} The call steps until the target is reached or the arm "
            f"stops, then returns the camera frames and the pose. It stops at {self.timeout:g} s "
            f"if the motion has not finished.{annotation} {self.notes}"
        ).strip()

    def _addressable(self) -> tuple[dict[str, _Axis], list[str]]:
        """Tool dimensions. Contract orientation blocks become roll, pitch, yaw."""
        axes: dict[str, _Axis] = {}
        covered: set[int] = set()
        for block in self._orientations:
            covered.update(block.contract_columns)
            for component, name in enumerate(block.tool_names):
                axes[name] = _Axis("euler", component, block)
        for index, name in enumerate(self._names):
            if index not in covered:
                axes[name] = _Axis("scalar", index)
        if self.dims is None:
            order = _tool_order(self._names, self._orientations)
            return axes, [name for name in order if name in axes]
        # A dims entry may name a tool dimension or any contract column of a rotation block.
        block_of = {
            name: block
            for block in self._orientations
            for name in (*block.tool_names, *(self._names[i] for i in block.contract_columns))
        }
        chosen: list[str] = []
        unknown: list[str] = []
        for name in self.dims:
            if name in axes and name not in chosen:
                chosen.append(name)
                continue
            block = block_of.get(name)
            if block is None:
                unknown.append(name)
                continue
            for tool_name in block.tool_names:
                if tool_name not in chosen:
                    chosen.append(tool_name)
        if unknown:
            raise ValueError(
                f"dims {sorted(unknown)} are not dimensions of this tool ({', '.join(axes)})"
            )
        return axes, chosen

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
                # Plan: turn the call into per-tick action rows (none if the episode is over).
                # A spent time limit ends the episode the way sim termination does.
                left = None if self._limit_ticks is None else self._limit_ticks - self._elapsed
                if left == 0:
                    obs["terminated"] = True
                if obs["terminated"]:
                    rows = np.zeros((0, len(self._names)))
                    goal = None
                    clipped = False
                    named: set[str] = set()
                    blocks: list[Orientation] = []
                else:
                    rows, goal, clipped, named = plan(obs)
                    # Rotation blocks the model touched (only those count toward "reached").
                    blocks = [
                        b for b in self._orientations if any(n in named for n in b.tool_names)
                    ]
                # Playback state, updated by step() each tick.
                played = 0
                finished = False
                seen = False  # proprioception has moved at least once
                still = 0  # consecutive ticks below the movement threshold
                proprio = prev = self._proprio(obs)

                async def step(row: NDArray[np.float64], *, check: bool) -> bool:
                    """Play one tick; return True when the call should stop."""
                    nonlocal played, obs, finished, seen, still, prev, proprio
                    await client.send_action(row)
                    obs = await client.get_observation()
                    await self._record(obs)
                    played += 1
                    if left is not None and played >= left:
                        obs["terminated"] = True
                    if self._absolute:
                        # Remember the target so the next call continues from it, not the sim pose.
                        self._command = np.asarray(row, dtype=np.float64).copy()
                    proprio = self._proprio(obs)
                    # Track whether the arm has moved, and how many ticks it has been still since.
                    if prev.size and proprio.shape == prev.shape:
                        if float(np.max(np.abs(proprio - prev))) > _STILL_ABS:
                            seen = True
                            still = 0
                        else:
                            still += 1
                    prev = proprio
                    # Done = arrived at the goal, or came to rest after moving.
                    if check and goal is not None and not obs["terminated"]:
                        finished = self._reached(proprio, goal, named, blocks) or (
                            seen and still >= _STILL_TICKS
                        )
                    return bool(obs["terminated"] or finished)

                # Play the plan; only the final setpoint can finish the move.
                for index, row in enumerate(rows):
                    if await step(row, check=index == len(rows) - 1 and not clipped):
                        break
                # Then hold the final setpoint until the arm arrives or stops. A clipped
                # plan already fills the safety cap, so it gets no hold.
                if goal is not None and not clipped and not finished and not obs["terminated"]:
                    hold = goal if self._absolute else np.zeros(len(self._names))
                    while played < self._max_ticks:
                        if await step(hold, check=True):
                            break
                # Ran out of ticks before arriving: report it rather than error.
                timed_out = (
                    goal is not None
                    and played >= self._max_ticks
                    and not finished
                    and not obs["terminated"]
                )
            finally:
                await client.close()
        self._elapsed += played
        # Answer with the pose, command, residual, and camera frames the move ended on.
        return self._render(
            obs,
            played,
            proprio,
            timed_out=timed_out,
            goal=goal if self._absolute else None,
            left=None if self._limit_ticks is None else self._limit_ticks - self._elapsed,
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
        if self._command is not None:
            return self._command.copy()
        assert self._reference is not None
        return np.asarray(self._reference(obs["data"]), dtype=np.float64).reshape(-1)

    def _plan(self, values: dict[str, float], obs: dict[str, Any]) -> Plan:
        """Action rows, the intended goal, whether the safety cap clipped them, and named dims."""
        # Find the start pose and the end pose (goal) of the move.
        if not values:
            raise ToolError("set at least one action dimension (`wait` to just look)")
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
        addressed = [b for b in self._orientations if any(n in named for n in b.tool_names)]
        for block in addressed:
            roll, pitch, yaw = xyzw_to_euler(block.read(start))
            euler = [roll, pitch, yaw]
            for name, value in values.items():
                axis = self._axes[name]
                if axis.kind == "euler" and axis.block is block:
                    if abs(value) > math.pi + 1e-4:
                        raise ToolError(
                            f"{name}={value} is outside [{-math.pi:.4g}, {math.pi:.4g}]"
                        )
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
        # Safety cap: play only the prefix that fits; `clipped` tells the caller it was cut short.
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
        return rows, end, clipped, named

    def _reached(
        self,
        proprio: NDArray[np.float64],
        goal: NDArray[np.float64],
        named: set[str],
        blocks: list[Orientation],
    ) -> bool:
        """Every addressed scalar within tolerance and every addressed rotation within reach."""
        if not self._absolute or proprio.shape != goal.shape or not named:
            return False
        for name in named:
            axis = self._axes[name]
            if axis.kind != "scalar":
                continue
            index = axis.index
            tolerance = max(_REACH_ABS, _REACH_FRAC * float(self._high[index] - self._low[index]))
            if abs(float(proprio[index] - goal[index])) > tolerance:
                return False
        return all(
            angular_distance(block.read(proprio), block.read(goal)) <= _REACH_RAD
            for block in blocks
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
        timed_out: bool,
        goal: NDArray[np.float64] | None,
        left: int | None,
    ) -> Content:
        data = obs["data"]
        # Keep this opening line: the trace viewer reads it to place each call on the video.
        opening = f"Played {played} steps ({played / self._rate:.1f} s)."
        if left is not None:
            opening += f" Time remaining: {left / self._rate:.1f} s."
        lines = [opening]
        if timed_out:
            lines.append(
                f"Stopped at the {self.timeout:g} s safety timeout before the motion finished."
            )
        for key, names in self._states.items():
            vector = np.asarray(data[key], dtype=np.float64).reshape(-1)
            labels = names or [str(i) for i in range(vector.size)]
            lines.append(f"{key}: " + _labeled(zip(labels, vector, strict=True)))
        if proprio.size == len(self._names):
            lines.append("pose: " + _labeled(self._reading(proprio)))
        if goal is not None:
            lines.append("commanded: " + _labeled(self._reading(goal)))
        if goal is not None and proprio.shape == goal.shape:
            lines.append("off: " + self._off(proprio, goal))
        if self._command is not None and goal is None:
            lines.append("commanded: " + _labeled(zip(self._names, self._command, strict=True)))
        if left == 0:
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
        euler: dict[int, tuple[float, float, float]] = {}
        reading: list[tuple[str, float]] = []
        for name in self._dims:
            axis = self._axes[name]
            if axis.kind == "scalar":
                reading.append((name, float(vector[axis.index])))
                continue
            assert axis.block is not None
            key = id(axis.block)
            if key not in euler:
                euler[key] = xyzw_to_euler(axis.block.read(vector))
            reading.append((name, euler[key][axis.index]))
        return reading

    def _off(self, proprio: NDArray[np.float64], goal: NDArray[np.float64]) -> str:
        parts: list[str] = []
        reported: set[int] = set()
        for name in self._dims:
            axis = self._axes[name]
            if axis.kind == "scalar":
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


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """A strict-mode object schema: every property required, none extra."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


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


__all__ = ["DEFAULT_TIMEOUT_S", "MAX_WAIT_S", "MOTION_TOOLS", "DirectControl"]

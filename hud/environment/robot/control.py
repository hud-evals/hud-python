"""Direct control: an LLM drives a ``robot`` capability through MCP motion tools.

The ``robot`` wire takes one action per control tick, which a policy keeps up
with and a tool-calling LLM does not. :class:`DirectControl` serves the same sim
as an ``mcp`` capability instead: a tool call names targets (or displacements)
for some action dimensions, the tool plays them out on the wire, and answers
with the camera frames and the pose it ended on. Each call is its own wire
connection, so the sim holds still while the model thinks. The env keeps
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
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import math
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, Field

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient
from hud.environment.robot.orientation import (
    Orientation,
    angular_distance,
    euler_to_xyzw,
    find_orientations,
    slerp,
    xyzw_to_euler,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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


class DimValue(BaseModel):
    """A value for one named action dimension."""

    name: str
    value: float


#: One required dimension. Strict structured outputs drop ``minItems``, so a list
#: cannot say "non-empty" to the model; a required object can.
Target = Annotated[DimValue, Field(description="One named dimension to set. Required.")]
Others = Annotated[
    list[DimValue],
    Field(description="Further named dimensions. An empty list if `target` is the only one."),
]

#: Required on every motion call so the transcript records what the model saw and why.
MotionNote = Annotated[
    str,
    Field(
        description=(
            "What you observe right now in the frames and state, and why you chose this motion. "
            "One or two plain sentences."
        )
    ),
]


@dataclass(frozen=True)
class _Axis:
    kind: Literal["scalar", "euler"]
    column: int
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
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"timeout must be positive seconds, got {timeout}")
        self.robot = robot
        self.name = name
        self.dims = dims
        self.speed = speed
        self.max_step = max_step or {}
        self.timeout = timeout
        self.reference = reference
        self.notes = notes
        self._lock = asyncio.Lock()  # a slot takes one wire connection at a time
        self._command: NDArray[np.float64] | None = None  # this episode's last absolute target
        self._serving: asyncio.Task[None] | None = None
        self._capability: Capability | None = None

    def attach(self, endpoint: RobotEndpoint) -> DirectControl:
        """Register with a robotics endpoint, which owns serving and episode lifecycle."""
        endpoint.add_direct_control(self)
        return self

    async def start(self, robot: Capability) -> Capability:
        """Serve the tools derived from *robot*'s contract; returns their ``mcp`` capability."""
        if self._capability is not None:
            return self._capability
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
        self._orientations = (
            find_orientations(self._names)
            if self._absolute and action.get("type") == "ee_abs"
            else []
        )
        self._orient_columns = {
            column for block in self._orientations for column in block.contract_columns
        }
        self._axes, self._dims = self._addressable()
        bounds = action.get("limits") or action["stats"]
        self._low = np.asarray(bounds["min"], dtype=np.float64)
        self._high = np.asarray(bounds["max"], dtype=np.float64)
        self._rate = float(contract["control_rate"])
        self._max_ticks = max(1, math.ceil(self.timeout * self._rate))
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
        observations = {n: f for n, f in features.items() if f["role"] == "observation"}
        self._cameras = [n for n, f in observations.items() if f.get("type") == "rgb"]
        self._states = {
            n: f.get("names") for n, f in observations.items() if f.get("type") not in IMAGE_TYPES
        }
        self._reference = self.reference
        if self._absolute and self._reference is None:
            wide = [n for n, names in self._states.items() if len(names or ()) == len(self._names)]
            if len(wide) != 1:
                raise ValueError(
                    f"absolute control needs a reference: no single observation is as wide as "
                    f"the action ({len(self._names)}); pass reference=..."
                )
            self._reference = lambda data: data[wide[0]]

        server = FastMCP(name=self.name)
        self._bind_tools(server)
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
        if self._serving is None:
            return
        self._serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._serving
        self._serving = None
        self._capability = None

    def reset(self) -> None:
        """Clear the last absolute target before a robotics endpoint starts an episode."""
        self._command = None

    # ── tools ──────────────────────────────────────────────────────────────

    def _bind_tools(self, server: FastMCP) -> None:
        """Register the contract motion tool. Override to serve an env-specific tool."""
        move = self.move_to if self._absolute else self.move_by
        server.tool(move, name=self._tool, description=self._describe(), output_schema=None)

    async def move_to(self, target: Target, others: Others, note: MotionNote) -> Content:
        return await self._play([target, *others], note)

    async def move_by(self, target: Target, others: Others, note: MotionNote) -> Content:
        return await self._play([target, *others], note)

    def _describe(self) -> str:
        bounds = ", ".join(
            f"{name} [{low:.4g}, {high:.4g}]"
            for name, low, high in zip(self._dims, self._tool_low, self._tool_high, strict=True)
        )
        if self._absolute:
            motion = (
                "Move to absolute targets. `target` is required; put any further dimensions "
                "in `others` (an empty list if there are no more). Unnamed dimensions hold "
                "their commanded value. The motion is interpolated at a safe speed."
            )
        else:
            motion = (
                "Move by a displacement. `target` is required; put any further dimensions "
                "in `others` (an empty list if there are no more). Unnamed dimensions do not "
                "move. The motion is split into steps within the per-step bounds."
            )
        rotation = ""
        if self._orientations:
            rotation = (
                " Orientation is intrinsic XYZ euler in radians (roll, pitch, yaw), "
                "converted to the arm's rotation and interpolated on the shortest arc."
            )
        return (
            f"{motion}{rotation} The call steps until the target is reached or the arm stops, "
            f"then returns the camera frames and the pose. It stops at {self.timeout:g} s if "
            f"the motion has not finished. Include a note saying what you see and why you "
            f"chose the motion. Dimensions and bounds: {bounds}. {self.notes}"
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
        chosen: list[str] = []
        unknown: list[str] = []
        for name in self.dims:
            if name in axes and name not in chosen:
                chosen.append(name)
                continue
            block = next((b for b in self._orientations if name in self._block_names(b)), None)
            if block is None:
                unknown.append(name)
                continue
            for tool_name in block.tool_names:
                if tool_name not in chosen:
                    chosen.append(tool_name)
        if unknown:
            # Re-checked in start() with the same set. Keep the raise here so a bad
            # list fails while the axes are in hand.
            raise ValueError(
                f"dims {sorted(unknown)} are not dimensions of this tool ({', '.join(axes)})"
            )
        return axes, chosen

    def _block_names(self, block: Orientation) -> set[str]:
        return set(block.tool_names) | {self._names[i] for i in block.contract_columns}

    # ── the wire ───────────────────────────────────────────────────────────

    async def _play(self, values: list[DimValue], note: str) -> Content:
        """Plan *values* against the live observation, play until done, and render the result."""
        # The note stays on the tool call; an empty one is a correctable miss, not a move.
        if not note.strip():
            raise ToolError("note must say what you see and why you chose this motion")
        async with self._lock:
            client = await RobotClient.connect(self._robot)
            try:
                obs = await client.get_observation()
                played = 0
                timed_out = False
                proprio: NDArray[np.float64] | None = None
                goal: NDArray[np.float64] | None = None
                if obs["terminated"]:
                    rows = np.zeros((0, len(self._names)))
                    clipped = False
                    named: set[str] = set()
                else:
                    rows, goal, clipped, named = self._plan(values, obs)
                seen = False
                still = 0
                prev = self._proprio(obs)
                finished = False

                async def step(row: NDArray[np.float64], *, check: bool) -> bool:
                    nonlocal played, obs, seen, still, prev, finished, proprio
                    await client.send_action(row)
                    obs = await client.get_observation()
                    played += 1
                    if self._absolute:
                        self._command = np.asarray(row, dtype=np.float64).copy()
                    proprio = self._proprio(obs)
                    if prev.size and proprio.shape == prev.shape:
                        if float(np.max(np.abs(proprio - prev))) > _STILL_ABS:
                            seen = True
                            still = 0
                        else:
                            still += 1
                    prev = proprio
                    if check and goal is not None and not obs["terminated"]:
                        finished = self._done(proprio, goal, named, seen=seen, still=still)
                    return bool(obs["terminated"] or finished)

                for index, row in enumerate(rows):
                    if played >= self._max_ticks:
                        timed_out = True
                        break
                    last = index == len(rows) - 1
                    if await step(row, check=last and not clipped):
                        break
                else:
                    if clipped and not finished and not obs["terminated"]:
                        timed_out = True
                if (
                    goal is not None
                    and not finished
                    and not timed_out
                    and not obs["terminated"]
                    and not clipped
                    and len(rows)
                ):
                    hold = goal if self._absolute else np.zeros(len(self._names))
                    while played < self._max_ticks:
                        if await step(hold, check=True):
                            break
                    else:
                        timed_out = True
            finally:
                await client.close()
        return self._render(
            obs,
            played,
            proprio,
            timed_out=timed_out,
            goal=goal if self._absolute else None,
        )

    def _plan(
        self, values: list[DimValue], obs: dict[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, set[str]]:
        """Action rows, the intended goal, whether the safety cap clipped them, and named dims."""
        if not values:
            raise ToolError("name at least one action dimension")
        unknown = [v.name for v in values if v.name not in self._dims]
        if unknown:
            raise ToolError(f"unknown dimension(s) {unknown}; valid: {', '.join(self._dims)}")
        # Absolute moves start at the commanded target (the reference on an episode's
        # first move); a displacement starts at zero.
        start = np.zeros(len(self._names), dtype=np.float64)
        if self._absolute:
            assert self._reference is not None
            if self._command is None:
                start = np.asarray(self._reference(obs["data"]), dtype=np.float64).reshape(-1)
            else:
                start = self._command.copy()
        end = start.copy()
        named = {v.name for v in values}
        for value in values:
            axis = self._axes[value.name]
            if axis.kind == "scalar":
                index = axis.column
                if self._absolute and not self._low[index] <= value.value <= self._high[index]:
                    raise ToolError(
                        f"{value.name}={value.value} is outside "
                        f"[{self._low[index]:.4g}, {self._high[index]:.4g}]"
                    )
                end[index] = value.value
        for block in self._orientations:
            if not any(name in named for name in block.tool_names):
                continue
            roll, pitch, yaw = xyzw_to_euler(block.read(start))
            euler = [roll, pitch, yaw]
            for value in values:
                axis = self._axes[value.name]
                if axis.kind == "euler" and axis.block is block:
                    if abs(value.value) > math.pi + 1e-4:
                        raise ToolError(
                            f"{value.name}={value.value} is outside [{-math.pi:.4g}, {math.pi:.4g}]"
                        )
                    euler[axis.column] = value.value
            target_q = euler_to_xyzw(euler[0], euler[1], euler[2])
            # Same rotation as now: keep the contract components bit for bit.
            if angular_distance(block.read(start), target_q) <= 1e-8:
                continue
            block.write(end, target_q)
        span = end - start
        scalar = [i for i in range(len(self._names)) if i not in self._orient_columns]
        ticks = 1
        if scalar:
            ticks = max(
                ticks,
                math.ceil(float(np.max(np.abs(span[scalar]) / self._step[scalar]))),
            )
        quats: list[tuple[Orientation, NDArray[np.float64], NDArray[np.float64]]] = []
        for block in self._orientations:
            if not any(name in named for name in block.tool_names):
                continue
            start_q = block.read(start)
            end_q = block.read(end)
            angle = angular_distance(start_q, end_q)
            if angle > 1e-8:
                step = self.speed * _FULL_TURN / self._rate
                ticks = max(ticks, math.ceil(angle / step))
                quats.append((block, start_q, end_q))
        play = min(ticks, self._max_ticks)
        clipped = play < ticks
        fractions = np.linspace(1 / ticks, play / ticks, play)
        if not self._absolute:
            rows = np.repeat((span / ticks)[None], play, axis=0)
        elif not self._orientations:
            rows = start + span * fractions[:, None]
        else:
            rows = np.repeat(start[None], play, axis=0)
            for index in scalar:
                rows[:, index] = start[index] + span[index] * fractions
            for block, start_q, end_q in quats:
                for row, fraction in zip(rows, fractions, strict=True):
                    block.write(row, slerp(start_q, end_q, float(fraction)))
            if not clipped:
                rows[-1] = end
        return rows, end, clipped, named

    def _done(
        self,
        proprio: NDArray[np.float64],
        goal: NDArray[np.float64],
        named: set[str],
        *,
        seen: bool,
        still: int,
    ) -> bool:
        if self._reached(proprio, goal, named):
            return True
        return seen and still >= _STILL_TICKS

    def _reached(
        self, proprio: NDArray[np.float64], goal: NDArray[np.float64], named: set[str]
    ) -> bool:
        if not self._absolute or proprio.shape != goal.shape or not named:
            return False
        for name in named:
            axis = self._axes[name]
            if axis.kind != "scalar":
                continue
            index = axis.column
            tolerance = max(_REACH_ABS, _REACH_FRAC * float(self._high[index] - self._low[index]))
            if abs(float(proprio[index] - goal[index])) > tolerance:
                return False
        for block in self._orientations:
            if not any(name in named for name in block.tool_names):
                continue
            if angular_distance(block.read(proprio), block.read(goal)) > _REACH_RAD:
                return False
        return True

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
        proprio: NDArray[np.float64] | None,
        *,
        timed_out: bool,
        goal: NDArray[np.float64] | None,
    ) -> Content:
        data = obs["data"]
        lines = [f"Played {played} steps ({played / self._rate:.1f} s)."] if played else []
        if timed_out:
            lines.append(
                f"Stopped at the {self.timeout:g} s safety timeout before the motion finished."
            )
        for key, names in self._states.items():
            vector = np.asarray(data[key], dtype=np.float64).reshape(-1)
            labels = names or [str(i) for i in range(vector.size)]
            lines.append(f"{key}: " + _labeled(labels, vector))
        if proprio is not None and proprio.size == len(self._names):
            lines.append("pose: " + _labeled_pairs(self._reading(proprio)))
        if goal is not None:
            lines.append("commanded: " + _labeled_pairs(self._reading(goal)))
        if proprio is not None and goal is not None and proprio.shape == goal.shape:
            lines.append("off: " + self._off(proprio, goal))
        if self._command is not None and goal is None:
            lines.append("commanded: " + _labeled(self._names, self._command))
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
                reading.append((name, float(vector[axis.column])))
                continue
            assert axis.block is not None
            key = id(axis.block)
            if key not in euler:
                euler[key] = xyzw_to_euler(axis.block.read(vector))
            reading.append((name, euler[key][axis.column]))
        return reading

    def _off(self, proprio: NDArray[np.float64], goal: NDArray[np.float64]) -> str:
        parts: list[str] = []
        reported: set[int] = set()
        for name in self._dims:
            axis = self._axes[name]
            if axis.kind == "scalar":
                parts.append(f"{name}={float(proprio[axis.column] - goal[axis.column]):.4f}")
                continue
            assert axis.block is not None
            key = id(axis.block)
            if key in reported:
                continue
            reported.add(key)
            angle = angular_distance(axis.block.read(proprio), axis.block.read(goal))
            parts.append(f"orientation={angle:.4f} rad")
        return ", ".join(parts)

    @property
    def _tool_low(self) -> list[float]:
        return [self._bound(name, low=True) for name in self._dims]

    @property
    def _tool_high(self) -> list[float]:
        return [self._bound(name, low=False) for name in self._dims]

    def _bound(self, name: str, *, low: bool) -> float:
        axis = self._axes[name]
        if axis.kind == "euler":
            return -math.pi if low else math.pi
        return float(self._low[axis.column] if low else self._high[axis.column])


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


def _labeled(names: list[str], vector: NDArray[Any]) -> str:
    return ", ".join(f"{n}={x:.4f}" for n, x in zip(names, vector, strict=True))


def _labeled_pairs(pairs: list[tuple[str, float]]) -> str:
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


__all__ = ["DEFAULT_TIMEOUT_S", "MOTION_TOOLS", "DimValue", "DirectControl"]

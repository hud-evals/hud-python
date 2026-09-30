"""Direct control: an LLM drives a ``robot`` capability through MCP motion tools.

The ``robot`` wire takes one action per control tick, which a policy keeps up
with and a tool-calling LLM does not. :class:`DirectControl` serves the same sim
as an ``mcp`` capability instead: a tool call names targets (or displacements)
for some action dimensions, the tool plays them out on the wire as a short
speed-limited trajectory, and answers with the camera frames and state it ended
on. Each call is its own wire connection, so the sim holds still while the
model thinks. The env keeps serving ``robot`` too, so one env serves VLAs and
LLMs alike.

The contract derives the tool surface: the action ``type`` picks the motion
tool (:data:`MOTION_TOOLS`), its ``names`` are the addressable dimensions, its
``limits`` (else ``stats``) ``min``/``max`` bound them, ``control_rate`` paces
the playout, and ``rgb`` observations are the frames returned::

    env = Environment(name="my-sim")
    sim = env.gym(make_env)
    DirectControl(notes="Gripper: -1 open, +1 closed.").attach(env)

Single-env sims only: a tool call claims the sole slot (no slot token).
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import socket
from typing import TYPE_CHECKING, Any

import numpy as np
import uvicorn
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel

from hud.capabilities import Capability
from hud.capabilities.robot import RobotClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from hud.environment import Environment

#: Contract action ``type`` -> (motion tool, whether its values are absolute targets).
MOTION_TOOLS: dict[str, tuple[str, bool]] = {
    "ee_abs": ("move_to", True),
    "joint_pos": ("move_joints", True),
    "ee_del": ("move_by", False),
}

#: Observation ``type`` tags that mark an image rather than a state vector.
IMAGE_TYPES = frozenset({"rgb", "bgr", "gray", "depth"})

#: Longest trajectory one call plays; the model splits longer moves.
MAX_CALL_S = 10.0

Content = list[TextContent | ImageContent]


class DimValue(BaseModel):
    """A value for one named action dimension."""

    name: str
    value: float


class DirectControl:
    """Serve a ``robot`` capability's action space as MCP motion tools.

    - ``dims`` - the action dimensions a call may address (default: all); the
      rest hold their commanded value.
    - ``speed`` - absolute moves travel this fraction of each dimension's range
      per second.
    - ``max_step`` - per-dimension step override by action name (e.g. a binary
      gripper that should switch in one step).
    - ``settle`` - seconds each move holds its end (a zero displacement) so the
      controller and gripper settle before the frames are taken.
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
        settle: float = 0.5,
        reference: Callable[[dict[str, NDArray[Any]]], NDArray[Any]] | None = None,
        notes: str = "",
    ) -> None:
        self.robot = robot
        self.name = name
        self.dims = dims
        self.speed = speed
        self.max_step = max_step or {}
        self.settle = settle
        self.reference = reference
        self.notes = notes
        self._lock = asyncio.Lock()  # a slot takes one wire connection at a time
        self._command: NDArray[np.float64] | None = None  # this episode's last absolute target
        self._uvicorn: uvicorn.Server | None = None
        self._serving: asyncio.Task[None] | None = None

    def attach(self, env: Environment) -> DirectControl:
        """Serve alongside *env*'s ``robot`` capability; call after declaring it."""

        @env.initialize
        async def _up() -> None:
            env.add_capability(await self.start(env.capability(self.robot)))

        @env.shutdown
        async def _down() -> None:
            await self.stop()

        env._on_task_teardown.append(self._end_episode)
        return self

    async def start(self, robot: Capability) -> Capability:
        """Serve the tools derived from *robot*'s contract; returns their ``mcp`` capability."""
        self._robot = robot
        contract = robot.params["contract"]
        features: dict[str, dict[str, Any]] = contract["features"]
        action = next(f for f in features.values() if f["role"] == "action")
        if action.get("type") not in MOTION_TOOLS:
            raise ValueError(
                f"direct control needs an action type in {sorted(MOTION_TOOLS)}, "
                f"got {action.get('type')!r}"
            )
        tool, self._absolute = MOTION_TOOLS[action["type"]]
        self._names: list[str] = list(action["names"])
        self._dims = self.dims or self._names
        if unknown := set(self._dims) - set(self._names):
            raise ValueError(f"dims {sorted(unknown)} are not action dimensions {self._names}")
        bounds = action.get("limits") or action["stats"]
        self._low = np.asarray(bounds["min"], dtype=np.float64)
        self._high = np.asarray(bounds["max"], dtype=np.float64)
        self._rate = float(contract["control_rate"])
        self._max_ticks = math.ceil(MAX_CALL_S * self._rate)
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
        server.tool(
            self.observe,
            description="Look without moving: the current camera frames and robot state.",
            output_schema=None,
        )
        move = self.move_to if self._absolute else self.move_by
        server.tool(move, name=tool, description=self._describe(), output_schema=None)
        sock = socket.create_server(("127.0.0.1", 0))
        self._uvicorn = uvicorn.Server(uvicorn.Config(server.http_app(), log_level="warning"))
        self._serving = asyncio.create_task(self._uvicorn.serve(sockets=[sock]))
        return Capability.mcp(name=self.name, url=f"http://127.0.0.1:{sock.getsockname()[1]}/mcp")

    async def stop(self) -> None:
        if self._uvicorn is None or self._serving is None:
            return
        self._uvicorn.should_exit = True
        await self._serving
        self._uvicorn = self._serving = None

    async def _end_episode(self) -> None:
        self._command = None

    # ── tools ──────────────────────────────────────────────────────────────

    async def observe(self) -> Content:
        return await self._play(None)

    async def move_to(self, targets: list[DimValue]) -> Content:
        return await self._play(targets)

    async def move_by(self, deltas: list[DimValue]) -> Content:
        return await self._play(deltas)

    def _describe(self) -> str:
        bounds = ", ".join(
            f"{n} [{self._low[i]:.4g}, {self._high[i]:.4g}]"
            for i, n in enumerate(self._names)
            if n in self._dims
        )
        if self._absolute:
            motion = (
                "Move to absolute targets for the named action dimensions; unnamed ones "
                "hold their commanded value. The motion is interpolated at a safe speed."
            )
        else:
            motion = (
                "Move by a displacement per named action dimension; unnamed ones do not "
                "move. The motion is split into steps within the per-step bounds."
            )
        return (
            f"{motion} One call plays at most {MAX_CALL_S:g} s; split longer moves. "
            f"Returns the camera frames and state the motion ends on. "
            f"Dimensions and bounds: {bounds}. {self.notes}"
        ).strip()

    # ── the wire ───────────────────────────────────────────────────────────

    async def _play(self, values: list[DimValue] | None) -> Content:
        """Plan *values* against the live observation, play it out, and render the result."""
        async with self._lock:
            client = await RobotClient.connect(self._robot)
            try:
                obs = await client.get_observation()
                rows = [] if values is None or obs["terminated"] else self._plan(values, obs)
                played = 0
                for row in rows:
                    await client.send_action(row)
                    obs = await client.get_observation()
                    played += 1
                    if self._absolute:
                        self._command = row
                    if obs["terminated"]:
                        break
            finally:
                await client.close()
        return self._render(obs, played)

    def _plan(self, values: list[DimValue], obs: dict[str, Any]) -> NDArray[np.float64]:
        """The ``[ticks, dim]`` action rows that realize *values*."""
        if not values:
            raise ToolError("name at least one action dimension")
        unknown = [v.name for v in values if v.name not in self._dims]
        if unknown:
            raise ToolError(f"unknown dimension(s) {unknown}; valid: {', '.join(self._dims)}")
        # Absolute moves start at the commanded target (the reference on an episode's
        # first move); a displacement starts at zero.
        start = np.zeros(len(self._names))
        if self._absolute:
            assert self._reference is not None
            start = self._command
            if start is None:
                start = np.asarray(self._reference(obs["data"]), dtype=np.float64).reshape(-1)
        end = start.copy()
        for v in values:
            i = self._names.index(v.name)
            if self._absolute and not self._low[i] <= v.value <= self._high[i]:
                raise ToolError(
                    f"{v.name}={v.value} is outside [{self._low[i]:.4g}, {self._high[i]:.4g}]"
                )
            end[i] = v.value
        span = end - start
        ticks = max(1, math.ceil(float(np.max(np.abs(span) / self._step))))
        if ticks > self._max_ticks:
            raise ToolError(
                f"that motion needs {ticks / self._rate:.1f} s, over the {MAX_CALL_S:g} s "
                "per-call cap; split it into smaller moves"
            )
        hold = round(self.settle * self._rate)
        if self._absolute:
            motion = start + span * np.linspace(1 / ticks, 1, ticks)[:, None]
            return np.vstack([motion, np.repeat(end[None], hold, axis=0)])
        steps = np.repeat((span / ticks)[None], ticks, axis=0)
        return np.vstack([steps, np.zeros((hold, len(self._names)))])

    def _render(self, obs: dict[str, Any], played: int) -> Content:
        data = obs["data"]
        lines = [f"Played {played} steps ({played / self._rate:.1f} s)."] if played else []
        for key, names in self._states.items():
            vector = np.asarray(data[key], dtype=np.float64).reshape(-1)
            labels = names or [str(i) for i in range(vector.size)]
            lines.append(f"{key}: " + _labeled(labels, vector))
        if self._command is not None:
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


def _labeled(names: list[str], vector: NDArray[Any]) -> str:
    return ", ".join(f"{n}={x:.4f}" for n, x in zip(names, vector, strict=True))


def _png(frame: NDArray[Any]) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


__all__ = ["MOTION_TOOLS", "DimValue", "DirectControl"]

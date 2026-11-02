"""Policy Supervision: an LLM supervises a VLA policy driving a ``robot`` capability, in parallel.

The policy acts; the model decides what it is told. The two run side by side, as on a real
robot: from the episode's first call the sim runs in real time (``time_scale`` sim seconds per
wall second) whether or not the model is calling a tool, so the model's thinking costs time.
``run_policy`` sets the instruction the policy follows from then on (the task, or a step of it)
and returns at once; the policy keeps acting on it, querying itself again after each chunk,
until the model changes it, calls ``stop_policy`` (the arm then holds) or the episode ends.
``wait`` lets time pass and looks. While the policy computes its next chunk the arm does not
move, as with a blocking controller; the sim never runs ahead of real time.

With ``detector`` the model can find things by name in a camera frame (``detect``). With
``use_motion`` it can take over with the contract's motion tool (as in
:class:`~.control.DirectControl`) to correct what the policy got wrong, then hand back with
``run_policy``. With ``home`` it can send the arm back to a home pose, and with ``reach`` move
the gripper to a world point (a ``detect`` result's, say), the env's planner servoing it there.
Each of these stops the policy first and has the arm to itself until it is done, so the model
and the policy never command the arm at once::

    model, adapter = RemoteModel(port=8000), OpenPIAdapter(chunk_size=5)
    PolicySupervision(model, adapter, notes="...").attach(sim)

The policy is the robot harness's own :class:`~hud.agents.robot.model.Model` and
:class:`~hud.agents.robot.adapter.Adapter`, so a policy a
:class:`~hud.agents.robot.agent.RobotAgent` drives can be supervised. One difference: the
adapter's ``adapt_action`` sees the observation the chunk was queried on, not the one at each
step, as the whole chunk is translated up front.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from hud.environment.robot.camera import locate
from hud.environment.robot.control import (
    _DURATION,
    Content,
    DirectControl,
    _Motion,
    _png,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp import FastMCP
    from numpy.typing import NDArray

    from hud.agents.robot.adapter import Adapter
    from hud.agents.robot.model import Model
    from hud.capabilities import Capability
    from hud.environment.robot.control import Planner

    #: Finds a named thing in an image: each match's ``label``, ``score`` and pixel ``box``
    #: ``[u0, v0, u1, v1]``, as a dict.
    Detector = Callable[[NDArray[Any], str], list[dict[str, Any]]]

#: Sim seconds the policy plays between checks for a new instruction.
SLICE_S = 0.2
#: Longest ``wait``, in sim seconds, by default: the model looks in about once a second.
DEFAULT_CHECK_IN_S = 1.0
#: Longest ``home`` or ``reach`` call, in sim seconds, for a planner that never gets there.
SERVO_MAX_S = 30.0

_RUN = (
    "Tell the robot's learned policy what to do, and let it act: from now on it runs "
    "closed-loop on `instruction`, in parallel with you, until you change the instruction, "
    "call `stop_policy`, or the episode ends. Returns at once with the camera frames and state. "
    "The policy sees the cameras and the state; you choose what it is told. Use the task "
    "itself, or a shorter step of it in the same style. The robot runs in real time: time "
    "passes while you think."
)
_TAKEOVER = (
    " You can also move the arm yourself with `{tool}`, which stops the policy first, then "
    "call `run_policy` again to hand back."
)
_STOP = "Stop the policy: the arm holds where it is. Returns the camera frames and state."
_WAIT = (
    "Let `duration` seconds pass (at most {most:g}; null: none) while the policy goes on, or "
    "the arm holds if it is stopped, then return the camera frames and state."
)
_DETECT = (
    "Find `prompt` (a short name, e.g. 'red cube') in `camera`'s current frame. Returns each "
    "match's score, box and centre in pixels (u right, v down from the top-left), best "
    "first, with the frame it searched{world}. The robot does not wait while it searches."
)
_REACH = (
    "Stop the policy and move the gripper to the world point (`x`, `y`, `z`) m, such as a "
    f"`detect` result's, keeping its orientation and grip. Stops where it gets stuck, or after "
    f"{SERVO_MAX_S:g} s. Then returns the camera frames and state."
)
_HOME = (
    "Stop the policy and move the arm back to its home pose, where each episode starts, then "
    "return the camera frames and state."
)


class PolicySupervision(DirectControl):
    """Serve a policy, running in parallel with the model, as MCP tools that direct it.

    ``model`` and ``adapter`` are the policy and its translation to and from the contract;
    the adapter's ``chunk_size`` is how many actions play before the policy is queried again
    (default: the whole chunk). ``time_scale`` is the sim seconds that pass per wall second
    (default: real time). ``check_in`` is the longest ``wait``: how long the model can look
    away while the policy acts, besides its thinking time. ``detector`` serves ``detect``.
    ``use_motion`` also serves the contract motion tool, so the model can take over (the
    ``run_policy`` description says so).
    ``instructions`` limits what the
    policy can be told to the instructions it was trained on (default: any text). ``home``
    serves a ``home`` tool: a motion planner called again after each motion it plans,
    returning no actions once the arm is home. ``reach`` serves a ``reach`` tool: given a world
    point ``[x, y, z]``, it returns such a planner for the gripper. The rest is as
    :class:`~.control.DirectControl`.
    """

    def __init__(
        self,
        model: Model,
        adapter: Adapter,
        *,
        time_scale: float = 1.0,
        check_in: float = DEFAULT_CHECK_IN_S,
        detector: Detector | None = None,
        use_motion: bool = False,
        home: Planner | None = None,
        reach: Callable[[NDArray[np.float64]], Planner] | None = None,
        instructions: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError(f"time_scale must be positive, got {time_scale}")
        if not math.isfinite(check_in) or check_in <= 0:
            raise ValueError(f"check_in must be positive seconds, got {check_in}")
        super().__init__(use_wait=False, **kwargs)  # its own wait lets the policy go on
        self.model = model
        self.adapter = adapter
        self.time_scale = time_scale
        self.check_in = check_in
        self.detector = detector
        self.use_motion = use_motion
        self.home_planner = home
        self.reach_planner = reach
        self.instructions = instructions
        # The episode's parallel state: what the policy is told, the rest of its chunk, and
        # the background task playing it (or holding the arm) in real time.
        self._instruction: str | None = None
        self._rows: NDArray[np.float64] = np.zeros((0, 0))
        self._since = 0  # the tick the instruction was given
        self._driver: asyncio.Task[None] | None = None
        self._stopping = False
        self._last: _Motion | None = None  # the latest motion, the driver's or a tool's
        self._reported = 0  # ticks already reported to the model
        self._failure: str | None = None  # the policy's failure, for the model's next call
        self._played = asyncio.Condition()  # notified whenever ticks are played
        self._arm = asyncio.Lock()  # held by whoever is moving the arm: the driver or a tool

    async def start(self, robot: Capability) -> Capability:
        bind_adapter(self.adapter, robot)
        return await super().start(robot)

    async def end_episode(self) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            self._stopping = True
            try:
                await driver  # it finishes the slice it is playing
            finally:
                self._stopping = False
        self._halt()
        self._last = None
        self._reported = 0
        self._failure = None
        self.adapter.reset()
        await super().end_episode()

    def _bind_tools(self, server: FastMCP) -> None:
        run = _RUN
        if self.use_motion:
            super()._bind_tools(server)  # its description carries the sensors and notes
            run += _TAKEOVER.format(tool=self._tool)
        else:
            run = f"{run} {self._mounts}{self.notes}".strip()
        instruction = {"instruction": self._instruction_schema()}
        server.add_tool(self._make_tool(self.run_policy, "run_policy", run, instruction))
        server.add_tool(self._make_tool(self.stop_policy, "stop_policy", _STOP, {}))
        wait = _WAIT.format(most=self.check_in)
        server.add_tool(self._make_tool(self.wait, "wait", wait, {"duration": _DURATION}))
        if self.detector is not None:
            target = {
                "camera": {"type": "string", "enum": self._cameras},
                "prompt": {"type": "string", "description": "What to find, by name."},
            }
            world = (
                ", and for a camera with depth, the world point (m) at each centre"
                if any("depth" in self._obs_space[name] for name in self._cameras)
                else ""
            )
            detect = _DETECT.format(world=world)
            server.add_tool(self._make_tool(self.detect, "detect", detect, target))
        if self.home_planner is not None:
            server.add_tool(self._make_tool(self.home, "home", _HOME, {}))
        if self.reach_planner is not None:
            point = {axis: {"type": "number", "description": "Metres."} for axis in "xyz"}
            server.add_tool(self._make_tool(self.reach, "reach", _REACH, point))

    def _instruction_schema(self) -> dict[str, Any]:
        if self.instructions is None:
            return {"type": "string", "description": "What the policy is told."}
        return {
            "type": "string",
            "enum": self.instructions,
            "description": "What the policy is told: one of the instructions it was trained on.",
        }

    # ── tools ──────────────────────────────────────────────────────────────

    async def run_policy(self, instruction: str, note: str = "") -> Content:
        if not instruction.strip():
            raise ToolError("instruction must say what the policy should do")
        self._require_note(note)
        last = await self._begin()
        if not last.obs["terminated"]:
            # Queried here, not by the driver, so a policy that fails says so to this call.
            rows = await next_chunk(self.model, self.adapter, last.obs, instruction)
            self._instruction, self._rows, self._since = instruction, rows, self._elapsed
        return self._reply()

    async def stop_policy(self, note: str = "") -> Content:
        self._require_note(note)
        await self._begin()
        self._halt()
        return self._reply()

    async def wait(self, duration: float | None, note: str = "") -> Content:
        seconds = duration or 0.0
        if not 0 <= seconds <= self.check_in:
            raise ToolError(f"duration must be within [0, {self.check_in:g}] s, got {seconds:g}")
        self._require_note(note)
        await self._begin()
        until = self._elapsed + round(seconds * self._rate)
        driver = self._driver
        assert driver is not None
        async with self._played:
            await self._played.wait_for(lambda: self._elapsed >= until or driver.done())
        return self._reply()

    async def detect(self, camera: str, prompt: str, note: str = "") -> Content:
        assert self.detector is not None
        if not prompt.strip():
            raise ToolError("prompt must name what to find")
        self._require_note(note)
        last = await self._begin()
        frame = np.asarray(last.obs["data"][camera])
        # The camera as it was for this frame, if its pose and depth place pixels in the world.
        located = locate(self._obs_space, [camera], self._live, last.obs["data"]).get(camera)
        if located is not None and located.depth is None:
            located = None
        found = await asyncio.to_thread(self.detector, frame, prompt)
        lines = [f"detect({camera!r}, {prompt!r}): {len(found)} found."]
        for match in sorted(found, key=lambda m: -m["score"]):
            u0, v0, u1, v1 = match["box"]
            u, v = (u0 + u1) / 2, (v0 + v1) / 2
            line = (
                f"{match['label']} {match['score']:.2f}: box ({u0:.0f}, {v0:.0f})-({u1:.0f}, "
                f"{v1:.0f}), centre ({u:.0f}, {v:.0f})"
            )
            if located is not None:
                x, y, z = located.point(u, v)
                line += f" at ({x:.3f}, {y:.3f}, {z:.3f}) m in the world"
            lines.append(line)
        reply = self._reply()
        searched = [
            TextContent(type="text", text="\n".join(lines)),
            TextContent(type="text", text=f"searched {camera}:"),
            ImageContent(type="image", data=_png(frame), mime_type="image/png"),
        ]
        return [reply[0], *searched, *reply[1:]]

    async def home(self, note: str = "") -> Content:
        assert self.home_planner is not None
        self._require_note(note)
        return await self._servo(self.home_planner)

    async def reach(self, x: float, y: float, z: float, note: str = "") -> Content:
        assert self.reach_planner is not None
        self._require_note(note)
        return await self._servo(self.reach_planner(np.array([x, y, z], dtype=np.float64)))

    async def _play(self, plan: Planner, note: str) -> Content:
        """The model takes over: the policy stops, then the arm plays the model's motion."""
        self._require_note(note)
        await self._begin()
        self._halt()
        async with self._arm:
            await self._move(plan)
        return self._reply()

    async def _servo(self, planner: Planner) -> Content:
        """Stop the policy and play *planner* again after each motion it plans, until it plans
        none, the episode ends, or ``SERVO_MAX_S`` of sim time has passed."""
        await self._begin()
        self._halt()
        async with self._arm:  # one motion: the driver does not hold the arm in between
            until = self._elapsed + round(SERVO_MAX_S * self._rate)
            motion = await self._move(planner)
            while motion.played and self._elapsed < until and not motion.obs["terminated"]:
                motion = await self._move(planner)
        return self._reply()

    # ── the parallel episode ───────────────────────────────────────────────

    async def _begin(self) -> _Motion:
        """Start the episode on its first call: from here the sim runs in parallel."""
        if self._driver is None:
            await self._move(lambda _: (np.zeros((0, len(self._names))), None, set()))
            self._driver = asyncio.create_task(self._drive())
        elif self._driver.done():
            self._driver.result()  # raises what stopped it, if anything did
        assert self._last is not None
        return self._last

    def _halt(self) -> None:
        self._instruction = None
        self._rows = np.zeros((0, len(self._names)))

    async def _move(self, plan: Planner) -> _Motion:
        self._last = await super()._move(plan)
        async with self._played:
            self._played.notify_all()
        return self._last

    async def _drive(self) -> None:
        """Play the policy (or hold the arm) in real time, a slice at a time, until the
        episode ends."""
        try:
            await self._drive_slices()
        finally:
            async with self._played:  # a waiting call sees the episode end
                self._played.notify_all()

    async def _drive_slices(self) -> None:
        loop = asyncio.get_running_loop()
        ticks = max(1, round(SLICE_S * self._rate))
        period = 1 / (self._rate * self.time_scale)
        start, base = loop.time(), self._elapsed
        while not self._stopping:
            assert self._last is not None
            if self._last.obs["terminated"]:
                return
            instruction = self._instruction
            if instruction is not None and not len(self._rows):
                try:
                    self._rows = await next_chunk(
                        self.model, self.adapter, self._last.obs, instruction
                    )
                except Exception as error:  # the model hears of it on its next call
                    self._failure = f"The policy failed and was stopped: {error}"
                    self._halt()
                # The query took wall time without moving the arm: pace on from here.
                start, base = loop.time(), self._elapsed
                continue
            async with self._arm:
                if self._instruction is None:  # a tool may have stopped it while we waited
                    await self._move(
                        lambda obs: (np.repeat(self._start(obs)[None], ticks, 0), None, set())
                    )
                else:
                    rows, self._rows = self._rows[:ticks], self._rows[ticks:]
                    await self._move(lambda _, rows=rows: (rows, None, set()))
            ahead = start + (self._elapsed - base) * period - loop.time()
            if ahead > 0:
                await asyncio.sleep(ahead)
            else:  # behind real time: run on without catching up
                start, base = loop.time(), self._elapsed

    def _reply(self) -> Content:
        """The latest frames and state, the ticks played since the last reply, and what the
        policy is doing."""
        last = self._last
        assert last is not None
        played, self._reported = self._elapsed - self._reported, self._elapsed
        content = self._render(dataclasses.replace(last, played=played))
        if self._failure is not None:
            status, self._failure = self._failure, None
        elif last.obs["terminated"]:
            status = ""
        elif self._instruction is not None:
            acting = (self._elapsed - self._since) / self._rate
            status = f"The policy is acting on {self._instruction!r} ({acting:.1f} s so far)."
        else:
            status = "The policy is stopped; the arm holds."
        first = content[0]
        assert isinstance(first, TextContent)
        # Keep the opening line first: the trace viewer reads it to place each call on the video.
        opening, _, rest = first.text.partition("\n")
        text = "\n".join(line for line in (opening, status, rest) if line)
        return [TextContent(type="text", text=text), *content[1:]]


def bind_adapter(adapter: Adapter, robot: Capability) -> None:
    """Bind *adapter* to *robot*'s contract, as :class:`~hud.agents.robot.agent.RobotAgent` does."""
    features: dict[str, dict[str, Any]] = robot.params["contract"]["features"]
    adapter.bind(
        next(f for f in features.values() if f["role"] == "action"),
        {name: f for name, f in features.items() if f["role"] == "observation"},
    )


async def next_chunk(
    model: Model, adapter: Adapter, obs: dict[str, Any], instruction: str
) -> NDArray[np.float64]:
    """The policy's next actions on *obs*, in the contract's action space, as played."""
    rows = np.atleast_2d(await model.ainfer(adapter.adapt_observation(obs, instruction)))
    rows = adapter.adapt_chunk(rows, obs)[: adapter.chunk_size]
    if not len(rows):
        raise ToolError("the policy returned no actions")
    return np.array([adapter.adapt_action(row, obs) for row in rows], dtype=np.float64)

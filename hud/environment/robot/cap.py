"""Code as Policy: an LLM drives a ``robot`` capability by writing programs over primitives.

Direct control spends one model turn per motion. Here the model sends a Python program in
one ``run`` call; the program composes primitives (move, wait, observe) with ordinary control
flow, so closed loops (retry a missed grasp, step until something is seen) run without the
model in between. The primitives come from the contract, as direct control's tools do, so any
embodiment gets them; an env may add its own with ``primitives=``. Two optional additions:
``memory=`` is Python source (functions from solved tasks, recovery skills) loaded into every
episode's program, and ``model=`` with ``adapter=`` adds a ``run_policy`` primitive that hands a
step to a learned policy::

    env = Environment(name="my-sim")
    sim = env.gym(make_env)
    CodeAsPolicy(notes="Gripper: -1 open, +1 closed.").attach(sim)

Each program runs in a worker thread so its primitives can block; each primitive plays on the
env's event loop through :class:`~.control.DirectControl`'s motion code, so motions, verdicts
and trace telemetry are exactly direct control's. Variables persist across ``run`` calls within
an episode. A program past ``run_timeout`` is stopped at its next line or primitive, and the
robot holds; a call into native code (one long numpy operation) finishes first. The code runs
in the env process, which HUD sandboxes; nothing here isolates it further.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import linecache
import sys
import traceback
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any

import numpy as np
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from hud.environment.robot.camera import Camera, locate, located_names
from hud.environment.robot.control import (
    _SETTLE_S,
    AT_TARGET,
    IMAGE_TYPES,
    Content,
    DirectControl,
    Planner,
    _Motion,
    _png,
)
from hud.environment.robot.policy_supervision import bind_adapter, next_chunk

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from types import CodeType, FrameType

    from fastmcp import FastMCP
    from numpy.typing import NDArray

    from hud.agents.robot.adapter import Adapter
    from hud.agents.robot.model import Model
    from hud.capabilities import Capability

#: Wall-clock cap on one program, in seconds.
DEFAULT_RUN_TIMEOUT_S = 120.0
#: Default sim seconds one ``run_policy`` plays the policy for.
DEFAULT_RUN_S = 2.0
#: How long a new run waits for a stopped program to end, in seconds.
_STOP_GRACE_S = 5.0
#: Most program output a reply carries, in characters.
_MAX_OUTPUT = 4000

_CODE = {"type": "string", "description": "Python source to run."}
#: Where memory functions say they were defined, in tracebacks and ``inspect.getsource``.
_MEMORY = "<memory>"


class EpisodeEndedError(RuntimeError):
    """The episode ended (the task was graded done, or the time limit ran out)."""


class ProgramStoppedError(BaseException):
    """The program ran past its time limit and was stopped.

    A ``BaseException``, so a program's ``except Exception`` cannot swallow it and keep going.
    """


class ProgramExitError(RuntimeError):
    """The program tried to exit the process (``sys.exit``, ``exit``, ``quit``)."""


@dataclass(frozen=True)
class Moved:
    """How one motion primitive ended."""

    verdict: str | None  # an absolute arm move's verdict; None for a gripper move or a delta
    reached: bool  # an absolute arm move came to rest within tolerance of its target
    pose: dict[str, float]  # where it ended, by dimension
    seconds: float  # sim time it took
    ended: bool  # the episode ended during it


@dataclass(frozen=True)
class Observation:
    """One look at the robot: camera frames, state vectors by name, and the pose."""

    images: dict[str, NDArray[np.uint8]]  # camera -> (H, W, 3) uint8, keyed without "observation/"
    state: dict[str, dict[str, float]]  # observation -> labelled values
    pose: dict[str, float]  # the action-space pose, by dimension
    time_left: float | None  # sim seconds of the episode's budget; None: no limit
    cameras: dict[str, Camera]  # each camera whose world pose and field of view are known


@dataclass
class _Run:
    """One program's bookkeeping: what it printed, every primitive it called, what it showed."""

    output: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    shown: list[tuple[str, NDArray[np.uint8]]] = field(default_factory=list)
    played: int = 0
    last: _Motion | None = None
    stopped: bool = False


class CodeAsPolicy(DirectControl):
    """Serve a ``robot`` capability as a ``run`` tool executing Python over motion primitives.

    The motion settings are :class:`~.control.DirectControl`'s. Besides them:

    - ``primitives`` - extra functions the program may call, by name (an env's skills or
      perception). Their docstrings join the tool's description. They run in the program's
      worker thread.
    - ``run_timeout`` - wall-clock cap on one program, in seconds.
    - ``memory`` - Python source run into each episode's program before its first ``run``:
      functions from tasks already solved and skills that recover from failures. They can call
      the primitives; their signatures and docstrings join the tool's description.
    - ``model``, ``adapter`` - a learned policy, as :class:`~.policy_supervision.PolicySupervision`
      takes it, served to programs as ``run_policy(instruction, seconds)``.
    """

    def __init__(
        self,
        *,
        primitives: dict[str, Callable[..., Any]] | None = None,
        run_timeout: float = DEFAULT_RUN_TIMEOUT_S,
        memory: str = "",
        model: Model | None = None,
        adapter: Adapter | None = None,
        **kwargs: Any,
    ) -> None:
        if (model is None) != (adapter is None):
            raise ValueError("a policy needs both model and adapter")
        super().__init__(use_wait=False, **kwargs)
        self.primitives = primitives or {}
        self.run_timeout = run_timeout
        self.memory = memory
        self.model = model
        self.adapter = adapter
        self._memory = compile(memory, _MEMORY, "exec")
        self._running = asyncio.Lock()  # one program at a time
        self._namespace: dict[str, Any] | None = None  # this episode's program variables
        self._run: _Run | None = None  # the program running now
        self._ended = False  # the episode ended: primitives raise EpisodeEndedError
        self._programs = count(1)
        # A program stopped at its time limit that has not yet reached its next line.
        self._stopping: asyncio.Future[None] | None = None

    async def start(self, robot: Capability) -> Capability:
        if self.adapter is not None:
            bind_adapter(self.adapter, robot)
        return await super().start(robot)

    async def end_episode(self) -> None:
        """Forget the program variables, the commanded targets and the time spent."""
        self._namespace = None
        self._ended = False
        self._programs = count(1)
        if self.adapter is not None:
            self.adapter.reset()
        await super().end_episode()

    def _bind_tools(self, server: FastMCP) -> None:
        builtins = {self._tool, "wait", "observe", "show", "print", "np", "EpisodeEndedError"}
        if self.model is not None:
            builtins.add("run_policy")
        clash = sorted(set(self.primitives) & builtins)
        if clash:
            raise ValueError(f"primitives {clash} would hide built-in ones")
        clash = sorted(set(self._memory_functions()) & (builtins | set(self.primitives)))
        if clash:
            raise ValueError(f"memory functions {clash} would hide primitives")
        server.add_tool(self._make_tool(self.run, "run", self._describe_run(), {"code": _CODE}))

    def _memory_functions(self) -> dict[str, Callable[..., Any]]:
        """The public functions *memory* defines, by name, defined in a scratch namespace."""
        namespace: dict[str, Any] = {"np": np}
        exec(self._memory, namespace)  # noqa: S102 - memory is the user's own program
        return {
            name: value
            for name, value in namespace.items()
            if inspect.isfunction(value)
            and value.__code__.co_filename == _MEMORY
            and not name.startswith("_")
        }

    # ── the tool ───────────────────────────────────────────────────────────

    async def run(self, code: str, note: str = "") -> Content:
        self._require_note(note)
        filename = f"<run {next(self._programs)}>"
        try:
            program = compile(code, filename, "exec")
        except SyntaxError as exc:
            raise ToolError("".join(traceback.format_exception_only(exc)).strip()) from exc
        # Source lines for tracebacks, which read them by filename.
        linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
        async with self._running:
            if self._stopping is not None:
                # A stopped program ends at its next line or primitive: give it a moment.
                await asyncio.wait([self._stopping], timeout=_STOP_GRACE_S)
            if self._stopping is not None and not self._stopping.done():
                raise ToolError(
                    "the previous program is still running past its time limit (blocked in a "
                    "call that does not return); it stops at its next line or primitive"
                )
            if self._namespace is None:
                self._namespace = {
                    "np": np,
                    "EpisodeEndedError": EpisodeEndedError,
                    **self.primitives,
                }
                # Before the primitives exist, so loading memory never moves the robot.
                linecache.cache[_MEMORY] = (
                    len(self.memory),
                    None,
                    self.memory.splitlines(keepends=True),
                    _MEMORY,
                )
                exec(self._memory, self._namespace)  # noqa: S102 - memory is the user's own program
            self._namespace.update(self._builtins(asyncio.get_running_loop()))
            run = self._run = _Run()
            error = None
            worker = asyncio.ensure_future(
                asyncio.to_thread(_execute, program, self._namespace, run)
            )
            try:
                await asyncio.wait_for(asyncio.shield(worker), self.run_timeout)
            except TimeoutError:
                run.stopped = True  # its next line or primitive raises; the robot holds meanwhile
                self._stopping = worker
                # It ends with ProgramStoppedError, which nobody awaits any more.
                worker.add_done_callback(lambda done: done.cancelled() or done.exception())
                error = f"Stopped: the program ran past {self.run_timeout:g} s."
            except Exception as exc:
                error = _user_traceback(exc, filename)
            finally:
                self._run = None
            if run.last is None:  # nothing moved: look once, so the reply shows the scene
                run.last = await self._move(self._hold(0))
                run.played += run.last.played
        return self._reply(run, error)

    def _builtins(self, loop: asyncio.AbstractEventLoop) -> dict[str, Any]:
        """The primitives every program gets, played on *loop*."""
        rate = self._rate

        def play(plan: Planner, call: str) -> _Motion:
            return play_motion(lambda: self._move(plan), call)

        def play_motion(start: Callable[[], Coroutine[Any, Any, _Motion]], call: str) -> _Motion:
            run = self._run
            if run is None or run.stopped:
                raise ProgramStoppedError("the program was stopped")
            if self._ended:
                raise EpisodeEndedError("the episode has ended; the robot no longer moves")
            motion = asyncio.run_coroutine_threadsafe(start(), loop).result()
            self._ended = bool(motion.obs["terminated"])
            run.played += motion.played
            run.last = motion
            verdict = self._verdict(motion) or "done"
            run.calls.append(f"{call} -> {verdict} ({motion.played / rate:.1f} s)")
            return motion

        def moved(motion: _Motion) -> Moved:
            verdict = self._verdict(motion)
            return Moved(
                verdict=verdict,
                reached=verdict == AT_TARGET,
                pose=dict(self._reading(motion.proprio)),
                seconds=motion.played / rate,
                ended=bool(motion.obs["terminated"]),
            )

        def motion(targets: dict[str, float] | None = None, **dims: float | None) -> Moved:
            values = {k: v for k, v in {**(targets or {}), **dims}.items() if v is not None}
            call = f"{self._tool}({', '.join(f'{k}={v:g}' for k, v in values.items())})"
            try:
                return moved(play(lambda obs: self._plan(values, obs), call))
            except ToolError as exc:
                raise ValueError(str(exc)) from None

        def wait(seconds: float) -> Moved:
            if not 0 <= seconds <= self.timeout:
                raise ValueError(f"seconds must be within [0, {self.timeout:g}], got {seconds:g}")
            return moved(play(self._hold(seconds), f"wait({seconds:g})"))

        def observe() -> Observation:
            return self._observation(play(self._hold(0), "observe()"))

        def show(*cameras: str) -> None:
            names = [_short(camera) for camera in self._cameras]
            unknown = sorted(set(cameras) - set(names))
            if unknown:
                raise ValueError(f"no cameras {unknown}; the cameras are {names}")
            if self._run is None or self._run.last is None:
                play(self._hold(0), "observe()")  # nothing seen yet this run: look first
            run = self._run
            assert run is not None and run.last is not None  # set by play
            data = run.last.obs["data"]
            for camera in cameras or names:
                run.shown.append((camera, np.asarray(data[f"observation/{camera}"])))

        def print_(
            *values: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False
        ) -> None:
            if file not in (None, sys.stdout, sys.stderr):
                raise ValueError("print(file=...) is not supported; the reply shows stdout")
            if self._run is not None:
                self._run.output.append(sep.join(str(v) for v in values) + end)

        motion.__name__ = self._tool
        builtins = {
            self._tool: motion,
            "wait": wait,
            "observe": observe,
            "show": show,
            "print": print_,
        }
        if self.model is not None:

            def run_policy(instruction: str, seconds: float = DEFAULT_RUN_S) -> Moved:
                if not instruction.strip():
                    raise ValueError("instruction must say what the policy should do")
                if not 0 < seconds <= self.timeout:
                    raise ValueError(
                        f"seconds must be within (0, {self.timeout:g}], got {seconds:g}"
                    )
                ticks = max(1, round(seconds * rate))
                call = f"run_policy({instruction!r}, {seconds:g})"
                try:
                    return moved(play_motion(lambda: self._policy(instruction, ticks), call))
                except ToolError as exc:
                    raise ValueError(str(exc)) from None

            builtins["run_policy"] = run_policy
        return builtins

    async def _policy(self, instruction: str, ticks: int) -> _Motion:
        """Run the policy closed-loop on *instruction* for *ticks*, re-queried per chunk."""
        assert self.model is not None and self.adapter is not None
        # The frozen sim, read by playing no rows: what the policy acts on first.
        motion = await self._move(lambda _: (np.zeros((0, len(self._names))), None, set()))
        played = 0
        while played < ticks and not motion.obs["terminated"]:
            chunk = await next_chunk(self.model, self.adapter, motion.obs, instruction)
            rows = chunk[: ticks - played]
            motion = await self._move(lambda _, rows=rows: (rows, None, set()))
            played += motion.played
        return dataclasses.replace(motion, played=played)

    def _hold(self, seconds: float) -> Planner:
        """Hold the commanded pose for *seconds* (at least one tick, as ``wait`` does)."""
        ticks = max(1, round(seconds * self._rate))
        return lambda obs: (np.repeat(self._start(obs)[None], ticks, axis=0), None, set())

    def _observation(self, motion: _Motion) -> Observation:
        data = motion.obs["data"]
        images = {
            _short(name): np.asarray(data[name])
            for name, feature in self._obs_space.items()
            if feature.get("type") in IMAGE_TYPES
        }
        state = {
            _short(name): dict(
                zip(
                    feature.get("names") or [],
                    np.asarray(data[name], dtype=np.float64).reshape(-1).tolist(),
                    strict=False,
                )
            )
            for name, feature in self._obs_space.items()
            if feature.get("type") not in IMAGE_TYPES
        }
        time_left = None if motion.ticks_left is None else motion.ticks_left / self._rate
        return Observation(
            images=images,
            state=state,
            pose=dict(self._reading(motion.proprio)),
            time_left=time_left,
            cameras={
                _short(name): camera
                for name, camera in locate(self._obs_space, self._cameras, self._live, data).items()
            },
        )

    def _reply(self, run: _Run, error: str | None) -> Content:
        assert run.last is not None
        report = self._render(run.last)
        assert isinstance(report[0], TextContent)
        # Keep this opening line: the trace viewer reads it to place each call on the video.
        text = [f"Played {run.played} steps ({run.played / self._rate:.1f} s)."]
        if run.calls:
            text += ["Calls:", *(f"  {call}" for call in run.calls)]
        output = "".join(run.output)
        if output:
            if len(output) > _MAX_OUTPUT:
                output = output[:_MAX_OUTPUT] + f"\n... ({len(output) - _MAX_OUTPUT} more chars)"
            text += ["Output:", output.rstrip("\n")]
        if error:
            text += ["Error:", error]
        # The motion reply after its opening line: where the robot ended.
        text += ["Ended at:", *report[0].text.split("\n")[1:]]
        content: Content = [TextContent(type="text", text="\n".join(text))]
        for camera, frame in run.shown:
            content.append(TextContent(type="text", text=f"shown {camera}:"))
            content.append(ImageContent(type="image", data=_png(frame), mime_type="image/png"))
        # Every camera's final frame, unless the program already showed that very frame.
        data = run.last.obs["data"]
        last_shown = dict(run.shown)
        for camera in self._cameras:
            frame = np.asarray(data[camera])
            shown = last_shown.get(_short(camera))
            if shown is not None and np.array_equal(shown, frame):
                continue
            content.append(TextContent(type="text", text=f"camera {_short(camera)}:"))
            content.append(ImageContent(type="image", data=_png(frame), mime_type="image/png"))
        return content

    def _describe_run(self) -> str:
        tool = self._tool
        dims = ", ".join(f"{name}=None" for name in self._dims if "." not in name)
        grouped = [name for name in self._dims if "." in name]
        if grouped:
            dims += f", or a dict naming {', '.join(grouped)}"
        if self._absolute:
            motion = (
                f"{tool}({dims}) -> Moved: move to absolute targets; None holds. Interpolated at "
                f"{self.speed * 100:g}% of each range per second, then held until the arm "
                f"comes to rest (at most {_SETTLE_S:g} s). Moved.verdict says whether it is at "
                "the target, stopped short, or ran out of time; Moved.reached is True at the "
                "target; Moved.pose is where it ended."
            )
        else:
            motion = (
                f"{tool}({dims}) -> Moved: move by a displacement in the action's components; "
                f"None does not move. Played at {self._rate:g} steps per second."
            )
        lines = [
            "Run a Python program that drives the robot, then see what it printed, every "
            "primitive it called with its outcome, and the camera frames it ended on. "
            "Primitives block until done. Variables persist across runs in this episode. "
            "numpy is `np`.",
            motion,
            f"wait(seconds) -> Moved: hold the pose for that long (at most {self.timeout:g} s).",
            "observe() -> Observation: one control tick passes; .images[camera] is an (H, W, 3) "
            "uint8 array, .state[observation] maps names to values, .pose is the action-space "
            "pose by dimension, .time_left is sim seconds left (None: no limit), "
            ".cameras[camera] is where each camera with a known pose was for this frame.",
            "show(*cameras): attach the latest frames of these cameras (default: all) to the "
            "reply, e.g. after observe() mid-program. The reply also ends with every camera's "
            "final frame.",
        ]
        located = located_names(self._obs_space, self._cameras, self._live)
        if located:
            lines.append(
                "Camera (in Observation.cameras): .to_world(u, v, z=0.0) -> the world point (m) "
                "where pixel (u, v) (column, row from the top-left) meets the horizontal plane "
                "at height z; .to_pixel(point) -> (u, v); .ray(u, v) -> the world direction seen "
                "at that pixel; .position and .rotation (+z out of the lens, +x image right, +y "
                "image down). Pixels you find in an image convert to metres with the camera of "
                "that same observation. Located cameras: "
                f"{', '.join(map(_short, located))}."
            )
            depths = [n for n in located if "depth" in self._obs_space[n]]
            if depths:
                lines.append(
                    ".point(u, v) -> the world point (m) seen at pixel (u, v), from the camera's "
                    "depth image: no height to guess. Cameras with depth: "
                    f"{', '.join(map(_short, depths))}."
                )
        if self.model is not None:
            lines.append(
                f"run_policy(instruction, seconds={DEFAULT_RUN_S:g}) -> Moved: run the robot's "
                "learned policy closed-loop on `instruction` for that many sim seconds (at most "
                f"{self.timeout:g}). It sees the cameras and state and acts on its own; use it "
                "for steps it was trained on, phrased as its instructions are."
            )
        if self._orientations:
            lines.append(
                "Orientation is intrinsic XYZ euler in radians (roll, pitch, yaw); pi and -pi "
                "are the same orientation."
            )
        for name, primitive in self.primitives.items():
            lines.append(f"{name}: {(primitive.__doc__ or '').strip()}")
        functions = self._memory_functions()
        if functions:
            lines.append(
                "Memory: functions from earlier episodes, already defined; call them, or read "
                "one with inspect.getsource(name) before changing it."
            )
            for name, function in functions.items():
                doc = (function.__doc__ or "").strip()
                lines.append(f"  {name}{inspect.signature(function)}: {doc}")
        cameras = ", ".join(_short(camera) for camera in self._cameras)
        lines.append(f"Cameras: {cameras}.")
        lines.append(
            f"A program is stopped after {self.run_timeout:g} s of wall-clock time. "
            "Once the episode has ended, every primitive raises EpisodeEndedError."
        )
        if self.use_annotation:
            lines.append("Include a note saying what you see and what the program is for.")
        return "\n".join([*lines, f"{self._mounts}{self.notes}".strip()])


def _execute(program: CodeType, namespace: dict[str, Any], run: _Run) -> None:
    """Run *program* in this worker thread, stopping it at its next line once *run* is stopped.

    Primitives already raise once stopped; tracing the program's own lines also stops a loop
    that never calls one. Only frames compiled from programs are traced, so primitives and
    libraries run at full speed.
    """

    def line(frame: FrameType, event: str, arg: Any) -> Any:
        if run.stopped:
            raise ProgramStoppedError("the program was stopped")
        return line

    def call(frame: FrameType, event: str, arg: Any) -> Any:
        filename = frame.f_code.co_filename
        return line if filename.startswith("<run ") or filename == _MEMORY else None

    sys.settrace(call)
    try:
        exec(program, namespace)  # noqa: S102 - running the program is the interface
    except (SystemExit, KeyboardInterrupt, GeneratorExit) as exc:
        # These would end the env process; the program only ends itself.
        raise ProgramExitError(f"{type(exc).__name__}: a program cannot exit the env") from None
    finally:
        sys.settrace(None)


def _short(name: str) -> str:
    """An observation name without its ``observation/`` prefix."""
    return name.removeprefix("observation/")


def _user_traceback(exc: Exception, filename: str) -> str:
    """The traceback of *exc*, keeping only the frames in the program and memory."""
    error = traceback.TracebackException.from_exception(exc)
    error.stack = traceback.StackSummary.from_list(
        [frame for frame in error.stack if frame.filename in (filename, _MEMORY)]
    )
    return "".join(error.format()).strip()


__all__ = ["CodeAsPolicy", "EpisodeEndedError", "Moved", "Observation"]

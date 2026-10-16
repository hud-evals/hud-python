"""Direct control gates: an LLM-style agent drives a robot sim through MCP motion tools.

Each test is a whole rollout: an in-process bridge serving the ``openpi/0`` wire,
an env publishing it plus :class:`DirectControl`'s ``mcp`` capability, and a
scripted agent calling the tools through the tunneled manifest binding, graded
by the sim.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import itertools
import math
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import av
import numpy as np
import pytest

from hud.agents.base import Agent
from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema
from hud.environment.robot import DirectControl, RobotBridge
from hud.eval import LocalRuntime, Task, rollout
from tests.support import ROBOT_STEP_SCHEMA, steps

from .robot_sims import sim_env

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray

    from hud.capabilities import MCPClient
    from hud.environment import Environment
    from hud.eval.run import Run
    from hud.types import MCPToolResult


def _contract(
    action_type: str,
    low: list[float],
    high: list[float],
    names: list[str] | None = None,
) -> dict[str, Any]:
    """One camera, a state, and an action of *action_type*. Default action is ``[x, grip]``."""
    labels = names or ["x", "grip"]
    return {
        "control_rate": 10,
        "features": {
            "observation/image": {"role": "observation", "type": "rgb"},
            "observation/state": {"role": "observation", "names": list(labels)},
            "action": {
                "role": "action",
                "type": action_type,
                "names": list(labels),
                "stats": {"min": low, "max": high},
            },
        },
    }


class _Arm(RobotBridge):
    """A 1-D arm plus gripper: succeeds (and terminates) holding ``x >= 0.5`` closed."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.delta = contract["features"]["action"]["type"] == "ee_del"
        self.state = np.zeros(2)
        self.actions: list[NDArray[Any]] = []

    def reset(self, **task_args: Any) -> str:
        self.state = np.zeros(2)
        self.actions.clear()
        return "move x to 0.5, then close the gripper"

    def step(self, action: NDArray[Any]) -> None:
        self.actions.append(np.array(action[0]))
        self.state = self.state + action[0] if self.delta else np.array(action[0])
        self.success = bool(self.state[0] >= 0.5 and self.state[1] >= 0.9)

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data = {
            "observation/image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "observation/state": self.state[None].astype(np.float32),
        }
        return data, np.array([self.success])


class _Sparse(dict[str, Any]):
    """The dimensions a scripted call sets, by dotted name; the model nulls the rest."""


def _nulled(schema: dict[str, Any], values: _Sparse) -> dict[str, Any]:
    """``values`` laid out as the schema's ``targets``: every dimension listed, unset ones null."""
    targets: dict[str, Any] = {}
    for key, prop in schema["properties"].items():
        if "anyOf" in prop:
            leaves = {leaf: values.get(f"{key}.{leaf}") for leaf in prop["anyOf"][0]["properties"]}
            targets[key] = leaves if any(v is not None for v in leaves.values()) else None
        else:
            targets[key] = values.get(key)
    return targets


class _ScriptedLLM(Agent):
    """Stands in for a tool-calling LLM: plays fixed MCP calls, keeps their results."""

    def __init__(self, *calls: tuple[str, dict[str, Any]]) -> None:
        super().__init__()
        self.calls = calls
        self.tools: set[str] = set()
        self.schemas: dict[str, dict[str, Any]] = {}
        self.descriptions: dict[str, str] = {}
        self.results: list[MCPToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        listed = await client.list_tools()
        self.tools = {tool.name for tool in listed}
        self.schemas = {tool.name: tool.inputSchema for tool in listed}
        self.descriptions = {tool.name: tool.description or "" for tool in listed}
        for name, arguments in self.calls:
            properties = self.schemas[name]["properties"]
            if isinstance(arguments.get("targets"), _Sparse):
                arguments = {
                    **arguments,
                    "targets": _nulled(properties["targets"], arguments["targets"]),
                }
            if "note" not in properties:
                arguments = {k: v for k, v in arguments.items() if k != "note"}
            self.results.append(await client.call_tool(name, arguments))
        run.trace.content = "done"


@asynccontextmanager
async def _served(sim: RobotBridge, control: DirectControl) -> AsyncIterator[Environment]:
    """*sim* with direct control attached, as the ``arm`` env whose ``reach`` the sim grades."""
    async with sim_env(sim, "arm", control=control) as (env, endpoint):

        @env.template()
        async def reach() -> AsyncGenerator[Any, Any]:
            ep = await endpoint.reset()
            yield {"prompt": ep["prompt"]}
            yield await endpoint.result()

        yield env


def _text(result: MCPToolResult) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


def _move(tool: str, **values: float) -> tuple[str, dict[str, Any]]:
    return tool, {"targets": _Sparse(values), "note": "move toward the goal"}


async def test_move_to_interpolates_absolute_targets_until_the_sim_succeeds() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(
        _move("move_to", x=0.5),
        _move("move_to", grip=1.0),
    )

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_to", "wait"}  # the ee_abs contract picked the motion tool
    schema = agent.schemas["move_to"]
    assert set(schema["required"]) == {"targets", "note"}
    # Every dimension is listed and nullable, so strict mode keeps the schema as it is.
    assert schema["properties"]["targets"]["required"] == ["x", "grip"]
    assert schema["properties"]["targets"]["properties"]["x"]["type"] == ["number", "null"]
    assert schema["properties"]["targets"]["properties"]["grip"]["description"] == "[-1, 1]"
    assert ensure_strict_json_schema(copy.deepcopy(schema)) == schema
    assert run.reward == 1.0  # graded by the sim, not the tools
    # 0.1 of x's range per second at 10 Hz: 0.01 per tick, grip held at its reference.
    # The sim matches the command; the call holds until two 3-tick windows show it settled.
    reach_x, held, close = np.array(sim.actions[:50]), sim.actions[50:56], sim.actions[56:]
    np.testing.assert_allclose(reach_x[:, 0], np.linspace(0.01, 0.5, 50), rtol=1e-6)
    np.testing.assert_allclose(reach_x[:, 1], 0.0)
    np.testing.assert_allclose(held, [[0.5, 0.0]] * 6)
    # max_step lets the gripper switch in one tick while x holds its commanded target;
    # the sim succeeds on that tick, which ends the episode.
    np.testing.assert_allclose(close, [[0.5, 1.0]])
    reached, closed = agent.results
    assert [block.type for block in reached.content] == ["text", "text", "image"]
    # The trace viewer parses this opening line to map tool calls to ticks.
    assert _text(reached).startswith("Played 56 steps (5.6 s).")
    assert "observation/state: x=0.5000, grip=0.0000" in _text(reached)
    assert "pose: x=0.5000, grip=0.0000" in _text(reached)
    assert "off: x=0.0000, grip=0.0000" in _text(reached)
    assert "At the target. Remaining offsets: 0 mm on x." in _text(reached)
    assert "The episode has ended" in _text(closed)
    # Tolerance only picks the verdict; nothing the model sees mentions it.
    seen = " ".join([str(agent.schemas), *agent.descriptions.values(), _text(reached)])
    for word in ("toleran", "within", "settle"):
        assert word not in seen.lower()


@pytest.mark.parametrize(("state_type", "tool"), [(None, "move_by"), ("BASE_DEL_POSE", "walk_by")])
async def test_a_displacement_kind_plays_in_steps_within_the_per_step_box(
    state_type: str | None, tool: str
) -> None:
    sim = _Arm(_contract("ee_del", [-0.1, -1.0], [0.1, 1.0]))
    sim.contract["features"]["action"]["state_type"] = state_type  # wins over the coarse type
    agent = _ScriptedLLM(_move(tool, x=0.35))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {tool, "wait"}
    assert "action's own components" in agent.descriptions[tool]
    # Four in-box steps; a displacement returns once played.
    np.testing.assert_allclose(sim.actions, [[0.0875, 0.0]] * 4)
    assert "observation/state: x=0.3500" in _text(agent.results[0])
    assert "At the target" not in _text(agent.results[0])


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (("move_to", {"targets": {"x": 0.1, "z": 0.1}, "note": "go"}), "'z' was unexpected"),
        (_move("move_to", x=1.5), "x=1.5 is outside [0, 1]"),
        (("move_to", {"targets": {"x": None}, "note": "go"}), "set at least one"),
        (("move_to", {"targets": {"x": 0.1}, "note": "  "}), "note must say"),
        (("wait", {"duration": 101, "note": "sleep"}), "duration must be within [0, 100]"),
    ],
)
async def test_an_invalid_move_is_a_correctable_error_that_leaves_the_sim_still(
    call: tuple[str, dict[str, Any]], error: str
) -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(call)

    async with _served(sim, DirectControl(dims=["x"], speed=0.05)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert result.isError
    assert error in _text(result)
    assert sim.actions == []


class _GraspTool(DirectControl):
    """Stands in for an env that serves its own motion tool on this wire."""

    def _bind_tools(self, server) -> None:
        server.add_tool(
            self._make_tool(
                self.move, "move_eef", "grasp targets", {"targets": self._targets_schema()}
            )
        )


async def test_an_env_can_replace_the_contract_motion_tool() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_eef", x=0.5))

    async with _served(sim, _GraspTool(max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_eef", "wait"}
    assert sim.actions  # the replacement tool still plays through the shared wire


class _Pose(RobotBridge):
    """Tracks an absolute action. ``lag`` of 1 snaps to the command; less lags behind."""

    def __init__(self, contract: dict[str, Any], home: list[float], lag: float = 1.0) -> None:
        super().__init__()
        self.contract = contract
        self.home = np.asarray(home, dtype=np.float64)
        self.lag = lag
        self.state = self.home.copy()
        self.actions: list[NDArray[Any]] = []

    def reset(self, **task_args: Any) -> str:
        self.state = self.home.copy()
        self.actions.clear()
        return "move"

    def step(self, action: NDArray[Any]) -> None:
        target = np.asarray(action[0], dtype=np.float64)
        self.actions.append(target.copy())
        self.state = self.state + self.lag * (target - self.state)
        self.success = False

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data = {
            "observation/image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "observation/state": self.state[None].astype(np.float32),
        }
        return data, np.array([self.success])


async def test_an_empty_target_list_is_rejected_and_does_not_move() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to"))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert result.isError
    assert "wait" in _text(result)
    assert sim.actions == []


async def test_a_move_longer_than_ten_seconds_plays_until_the_arm_arrives() -> None:
    """VLABench was rejected at 10.1 s. The cap is a 30 s safety timeout, not that reject."""
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=0.6))

    async with _served(sim, DirectControl(speed=0.05, max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert not result.isError
    # 0.6 / 0.05 s = 12 s at 10 Hz, which used to exceed the 10 s cap.
    assert len(sim.actions) >= 120
    assert sim.state[0] == pytest.approx(0.6)
    assert "At the target." in _text(result)


async def test_the_safety_timeout_returns_the_pose_instead_of_rejecting_the_call() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=1.0))

    async with _served(sim, DirectControl(dims=["x"], speed=0.05, timeout=0.3)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert not result.isError
    assert len(sim.actions) == 3  # 0.3 s at 10 Hz
    assert sim.state[0] < 0.05  # the prefix of the move, not a jump to the goal
    assert (
        "Ran out of time before settling; the arm was still moving. Remaining offsets: 985 mm on x."
    ) in _text(result)
    assert "pose:" in _text(result)
    assert "commanded:" in _text(result)


async def test_dimensions_with_a_shared_prefix_nest_as_a_group_the_model_can_null_whole() -> None:
    names = ["left_arm.x", "left_arm.y", "right_arm.x", "grip"]
    sim = _Pose(_contract("ee_abs", [-1.0] * 4, [1.0] * 4, names), [0.0] * 4)
    agent = _ScriptedLLM(_move("move_to", **{"left_arm.x": 0.1}))  # right_arm: null

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    schema = agent.schemas["move_to"]
    targets = schema["properties"]["targets"]
    assert targets["required"] == ["grip", "left_arm", "right_arm"]
    group = targets["properties"]["left_arm"]["anyOf"]
    assert list(group[0]["properties"]) == ["x", "y"]
    assert group[1] == {"type": "null"}
    assert ensure_strict_json_schema(copy.deepcopy(schema)) == schema
    np.testing.assert_allclose(sim.actions[-1], [0.1, 0.0, 0.0, 0.0], atol=1e-6)


async def test_without_annotation_the_tools_take_no_note_and_never_refuse_an_empty_one() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(
        ("move_to", {"targets": {"x": 0.1, "grip": None}}), ("wait", {"duration": None})
    )

    async with _served(sim, DirectControl(use_annotation=False)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    for schema in agent.schemas.values():
        assert "note" not in schema["properties"]
        assert "note" not in schema["required"]
    assert not any(result.isError for result in agent.results)
    assert sim.actions


async def test_wait_holds_the_commanded_pose_and_a_null_duration_observes_for_one_tick() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(
        _move("move_to", x=0.2),
        ("wait", {"duration": None, "note": "look"}),
        ("wait", {"duration": 1.5, "note": "let it settle"}),
    )

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    _, looked, waited = (_text(result) for result in agent.results)
    np.testing.assert_allclose(sim.actions[26:], [[0.2, 0.0]] * 16)  # held at the setpoint
    assert looked.startswith("Played 1 steps (0.1 s).")
    assert waited.startswith("Played 15 steps (1.5 s).")
    assert "commanded: x=0.2000, grip=0.0000" in looked
    assert "At the target" not in looked


async def test_every_reply_reports_the_time_left_and_the_tools_state_the_limit() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=0.2), ("wait", {"duration": 1.0, "note": "look"}))

    async with _served(sim, DirectControl(time_limit=10.0)) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert run.prompt == "move x to 0.5, then close the gripper"  # VLAs see the sim's own
    for tool in ("move_to", "wait"):
        assert "You have 10 seconds of simulated time." in agent.descriptions[tool]
    moved, waited = (_text(result) for result in agent.results)
    assert moved.startswith("Played 26 steps (2.6 s). Time remaining: 7.4 s.")
    assert waited.startswith("Played 10 steps (1.0 s). Time remaining: 6.4 s.")


async def test_a_move_crossing_the_time_limit_is_cut_there_and_ends_the_episode() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=0.5), _move("move_to", x=0.1))

    async with _served(sim, DirectControl(time_limit=2.0)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    cut, after = (_text(result) for result in agent.results)
    assert len(sim.actions) == 20  # the move needed 50 ticks; 2 s at 10 Hz were left
    assert sim.state[0] == pytest.approx(0.2)
    assert cut.startswith("Played 20 steps (2.0 s). Time remaining: 0.0 s.")
    assert "The time limit is reached." in cut
    assert "The episode has ended; stop calling tools." in cut
    assert "The episode ended before the arm settled. Remaining offsets: 300 mm on x." in cut
    assert "Ran out of time" not in cut
    assert after.startswith("Played 0 steps (0.0 s). Time remaining: 0.0 s.")
    assert "The episode has ended; stop calling tools." in after


async def test_a_sim_that_terminates_mid_move_is_not_reported_as_running_out_of_time() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=1.0, grip=1.0))

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert run.reward == 1.0
    assert len(sim.actions) < 100  # the sim succeeded before the 100-tick move finished
    assert "The episode ended before the arm settled." in _text(result)
    assert "Ran out of time" not in _text(result)
    assert "The episode has ended; stop calling tools." in _text(result)


async def test_an_arm_already_at_the_target_is_at_rest_at_a_low_control_rate() -> None:
    sim = _Pose(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0])
    sim.contract["control_rate"] = 1
    agent = _ScriptedLLM(_move("move_to", x=0.05))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert "At the target." in _text(agent.results[0])
    assert "Ran out of time" not in _text(agent.results[0])


class _Jitter(_Pose):
    """An arm that never moves, whose sensor reads a little high then low on alternate ticks."""

    def step(self, action: NDArray[Any]) -> None:
        super().step(action)
        self.state = self.home + 2e-5 * (-1) ** len(self.actions)


@pytest.mark.parametrize("arm", [_Pose, _Jitter])
async def test_an_arm_that_never_moves_gives_up_at_the_settle_cap_not_the_safety_timeout(
    arm: type[_Pose],
) -> None:
    sim = arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0], lag=0.0)
    agent = _ScriptedLLM(_move("move_to", x=0.5))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    # The 50-tick move plays out; an arm that never responded is held the whole 3 s cap.
    assert len(sim.actions) == 80
    text = _text(agent.results[0])
    assert "off: x=-0.5000" in text
    assert (
        "Stopped short of the target; the arm is at rest. Remaining offsets: 500 mm on x." in text
    )


async def test_a_lagging_arm_keeps_stepping_until_it_reaches_the_target() -> None:
    sim = _Pose(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0], lag=0.25)
    agent = _ScriptedLLM(_move("move_to", x=0.5))

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert len(sim.actions) > 50  # interpolation, then a hold while the arm catches up
    assert sim.state[0] == pytest.approx(0.5, abs=0.01)
    assert "pose:" in _text(agent.results[0])
    assert "off:" in _text(agent.results[0])
    assert "At the target. Remaining offsets: " in _text(agent.results[0])


class _Offset(_Pose):
    """An arm that settles a fixed ``offset`` (in action columns) away from every command."""

    def __init__(self, contract: dict[str, Any], home: list[float], offset: list[float]) -> None:
        super().__init__(contract, home)
        self.offset = np.asarray(offset, dtype=np.float64)

    def step(self, action: NDArray[Any]) -> None:
        super().step(action)
        self.state = self.state + self.offset


@pytest.mark.parametrize(
    ("tolerance", "verdict", "ticks"),
    [
        (
            None,
            "Stopped short of the target; the arm is at rest. Remaining offsets: 12 mm on x.",
            50 + 21,
        ),
        ({"position": 0.02}, "At the target. Remaining offsets: 12 mm on x.", 56),
    ],
)
async def test_an_arm_at_rest_outside_its_tolerance_is_held_for_the_grace_period_before_stopping(
    tolerance: dict[str, float] | None, verdict: str, ticks: int
) -> None:
    sim = _Offset(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0], [-0.012, 0.0])
    agent = _ScriptedLLM(_move("move_to", x=0.5))

    async with _served(sim, DirectControl(tolerance=tolerance)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert verdict in _text(agent.results[0])
    assert len(sim.actions) == ticks  # tolerance only lengthens the wait, never shortens it


class _Delayed(_Pose):
    """An arm whose state follows each command ``delay`` ticks late."""

    def __init__(self, contract: dict[str, Any], home: list[float], delay: int) -> None:
        super().__init__(contract, home)
        self.delay = delay

    def step(self, action: NDArray[Any]) -> None:
        super().step(action)
        late = len(self.actions) - 1 - self.delay
        self.state = self.actions[late] if late >= 0 else self.home


async def test_a_move_waits_for_a_slow_arm_even_when_already_within_tolerance() -> None:
    sim = _Delayed(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0], delay=10)
    agent = _ScriptedLLM(_move("move_to", x=8e-4))

    async with _served(sim, DirectControl(tolerance={"position": 0.05})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert sim.state[0] == pytest.approx(8e-4)
    assert len(sim.actions) == 17


class _Fingers(_Pose):
    """A hand whose two fingers close 5 cm per tick, the left one jamming on an object at 0.3."""

    def step(self, action: NDArray[Any]) -> None:
        target = np.asarray(action[0], dtype=np.float64)
        self.actions.append(target.copy())
        close = np.minimum(self.state[1:] + 0.05, target[1:])
        self.state = np.concatenate([self.state[:1], close])
        self.state[1] = min(self.state[1], 0.3)
        self.success = False


async def test_a_gripper_call_waits_for_every_finger_and_not_only_the_first_to_stop() -> None:
    names = ["x", "grip_left", "grip_right"]
    contract = _contract("ee_abs", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], names)
    sim = _Fingers(contract, [0.0, 0.0, 0.0])
    agent = _ScriptedLLM(_move("move_to", grip_left=1.0, grip_right=1.0))

    async with _served(sim, DirectControl(max_step={"grip_left": 2.0, "grip_right": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    # The left finger jams after 6 ticks; the right one closes for 20, and the call outlasts it.
    np.testing.assert_allclose(sim.state, [0.0, 0.3, 1.0])
    assert 20 + 3 <= len(sim.actions) < 20 + 30
    text = _text(agent.results[0])
    assert "fingers: grip_left=0.3000, grip_right=1.0000" in text  # the jam reads as no success
    assert "At the target" not in text
    assert "Remaining offsets" not in text


class _OpenHand(_Pose):
    """A lagging arm whose state always reads the gripper open, as a hand-only reference does."""

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data, success = super().get_observation()
        data["observation/state"] = data["observation/state"] * [1.0, 0.0] - [0.0, 1.0]
        return data, success


@pytest.mark.parametrize("reads_hand", [True, False])
async def test_gripper_offsets_are_reported_only_when_the_reference_reads_the_hand(
    reads_hand: bool,
) -> None:
    sim = _OpenHand(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, -1.0], lag=0.25)
    agent = _ScriptedLLM(_move("move_to", x=0.5, grip=1.0))
    reference = None if reads_hand else (lambda data: data["observation/state"])

    async with _served(sim, DirectControl(max_step={"grip": 2.0}, reference=reference)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    text = _text(agent.results[0])
    assert "At the target. Remaining offsets: " in text  # the gripper never decides the verdict
    assert ("2 on grip" in text) == reads_hand


def test_a_tolerance_key_that_is_not_a_quantity_is_refused() -> None:
    with pytest.raises(ValueError, match="tolerance keys"):
        DirectControl(tolerance={"grip": 0.1})


def _quat_names(order: str) -> list[str]:
    parts = ("qx", "qy", "qz", "qw") if order == "xyzw" else ("qw", "qx", "qy", "qz")
    return [f"tcp_quat.{part}" for part in parts]


_EULER = ["ee.x", "ee_euler.x", "ee_euler.y", "ee_euler.z"]


@pytest.mark.parametrize("share", [0.9, 1.1])
@pytest.mark.parametrize(
    ("action_type", "names", "move", "column", "tolerance", "label"),
    [
        # Each default moves the fingertips 1 mm: directly, over a 0.15 m gripper, over a
        # 2.5 m lever of joints.
        ("ee_abs", ["x", "grip"], {"x": 0.5}, 0, 1e-3, "x"),
        ("ee_abs", _EULER, {"roll": 0.3}, 1, 1e-3 / 0.15, "orientation"),
        ("joint_pos", ["joint1", "grip"], {"joint1": 0.3}, 0, 4e-4, "joint1"),
    ],
)
async def test_each_quantity_defaults_to_the_angle_that_moves_the_fingertips_one_millimetre(
    share: float,
    action_type: str,
    names: list[str],
    move: dict[str, float],
    column: int,
    tolerance: float,
    label: str,
) -> None:
    offset = [0.0] * len(names)
    offset[column] = share * tolerance
    contract = _contract(action_type, [-1.0] * len(names), [1.0] * len(names), names)
    sim = _Offset(contract, [0.0] * len(names), offset)
    agent = _ScriptedLLM(_move("move_to" if action_type == "ee_abs" else "move_joints", **move))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    text = _text(agent.results[0])
    assert ("At the target. " in text) == (share < 1)
    assert ("Stopped short of the target; the arm is at rest. " in text) == (share > 1)
    assert f" on {label}." in text  # every verdict names the dimension and how far off it is


@pytest.mark.parametrize(
    ("action_type", "names", "targets"),
    [
        (  # LIBERO
            "ee_abs",
            [f"target_eef_pos.{a}" for a in "xyz"] + [f"target_eef_axis_angle.r{a}" for a in "xyz"],
            ["x", "y", "z", "roll", "pitch", "yaw"],
        ),
        ("joint_pos", ["panda_joint1", "panda_joint2"], ["joint1", "joint2"]),
        ("ee_abs", ["left_x", "right_x", "left_gripper"], ["left_arm", "right_arm"]),
        ("ee_abs", ["left_x", "left_pitch_deg"], ["left_x", "left_pitch_deg"]),  # no standard
    ],
)
async def test_tools_use_standard_dimension_names_when_every_native_name_has_one(
    action_type: str, names: list[str], targets: list[str]
) -> None:
    contract = _contract(action_type, [-1.0] * len(names), [1.0] * len(names), names)
    sim = _Pose(contract, [0.0] * len(names))
    agent = _ScriptedLLM()

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (schema,) = (s for tool, s in agent.schemas.items() if tool != "wait")
    assert list(schema["properties"]["targets"]["properties"]) == targets


async def test_without_wait_only_the_motion_tool_is_served_and_its_texts_drop_waiting() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to"))

    async with _served(sim, DirectControl(use_wait=False, time_limit=10.0)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_to"}
    assert "Time passes only while the robot moves;" in agent.descriptions["move_to"]
    assert "wait" not in _text(agent.results[0])


async def test_a_yaw_slerps_an_xyzw_quaternion_instead_of_lerping_components() -> None:
    names = _quat_names("xyzw")
    sim = _Pose(_contract("ee_abs", [-1.0] * 4, [1.0] * 4, names), [0.0, 0.0, 0.0, 1.0])
    agent = _ScriptedLLM(_move("move_to", yaw=math.pi / 2))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    actions = np.array(sim.actions)
    norms = np.linalg.norm(actions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    np.testing.assert_allclose(
        actions[-1], [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], atol=1e-5
    )
    # Component lerp of the endpoints is not unit length; slerp stays on the arc.
    assert len(actions) > 2


async def test_a_roll_writes_a_wxyz_quaternion() -> None:
    names = _quat_names("wxyz")
    sim = _Pose(_contract("ee_abs", [-1.0] * 4, [1.0] * 4, names), [1.0, 0.0, 0.0, 0.0])
    agent = _ScriptedLLM(_move("move_to", roll=math.pi / 2))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    np.testing.assert_allclose(
        sim.actions[-1],
        [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],
        atol=1e-5,
    )


async def test_an_unnamed_quaternion_is_copied_unchanged() -> None:
    names = ["x", "tcp_quat.qx", "tcp_quat.qy", "tcp_quat.qz", "tcp_quat.qw"]
    home = [0.0, 0.0, 0.2, 0.0, 0.9]  # not a unit quaternion
    sim = _Pose(_contract("ee_abs", [0.0, -1, -1, -1, -1], [1, 1, 1, 1, 1], names), home)
    agent = _ScriptedLLM(_move("move_to", x=0.2))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    actions = np.array(sim.actions)
    assert len(actions) > 1
    held = np.broadcast_to(actions[0, 1:], actions[:, 1:].shape)
    np.testing.assert_array_equal(actions[:, 1:], held)
    assert abs(np.linalg.norm(actions[0, 1:]) - 1.0) > 1e-3


async def test_euler_commands_slerp_onto_axis_angle() -> None:
    names = [
        "target_eef_axis_angle.rx",
        "target_eef_axis_angle.ry",
        "target_eef_axis_angle.rz",
    ]
    sim = _Pose(_contract("ee_abs", [-np.pi] * 3, [np.pi] * 3, names), [0.0, 0.0, 0.0])
    agent = _ScriptedLLM(_move("move_to", roll=math.pi / 2, yaw=math.pi / 2))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    # Intrinsic XYZ of roll=yaw=pi/2 is the quaternion (0.5, 0.5, 0.5, 0.5), angle 2pi/3.
    component = (2.0 * np.pi / 3.0) / np.sqrt(3.0)
    np.testing.assert_allclose(sim.actions[-1], [component, component, component], atol=1e-4)
    # 2pi/3 at a tenth of a turn per second, 10 Hz; the hold after it repeats the goal.
    ticks = math.ceil((2 * np.pi / 3) / (0.1 * 2 * np.pi / 10))
    angles = np.linalg.norm(sim.actions[:ticks], axis=1)
    gaps = np.diff(angles)
    assert gaps.min() > 0
    np.testing.assert_allclose(gaps, gaps[0], rtol=0.05)


async def test_an_euler_contract_is_addressed_as_roll_pitch_yaw() -> None:
    names = [
        "ee_pos.x",
        "ee_pos.y",
        "ee_pos.z",
        "ee_euler.x",
        "ee_euler.y",
        "ee_euler.z",
        "gripper",
    ]
    low = [-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi, 0.0]
    high = [1.0, 1.0, 2.0, np.pi, np.pi, np.pi, 0.04]
    sim = _Pose(_contract("ee_abs", low, high, names), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04])
    agent = _ScriptedLLM(_move("move_to", z=1.2, yaw=0.4))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    final = sim.actions[-1]
    assert final[2] == pytest.approx(1.2, abs=1e-4)
    assert final[5] == pytest.approx(0.4, abs=1e-4)
    np.testing.assert_allclose(final[3:5], 0.0, atol=1e-4)
    assert final[6] == pytest.approx(0.04)


async def test_a_contract_without_a_motion_type_is_refused_at_start() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    sim.contract["features"]["action"]["type"] = "joint_vel"

    with pytest.raises(ValueError, match="direct control needs an action state_type"):
        async with _served(sim, DirectControl()) as env:
            await env.start()


def frames_in(segments: list[dict[str, Any]]) -> int:
    """How many frames the recorded MP4 segments decode to."""
    mp4 = b"".join(base64.b64decode(segment["segment"]["data"]) for segment in segments)
    with av.open(io.BytesIO(mp4), mode="r") as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.mark.parametrize(("control_rate", "tick_seconds"), [(10, 0.1), (0.4, 1.0)])
async def test_every_played_tick_streams_to_the_trace_as_state_and_one_video_per_camera(
    control_rate: float, tick_seconds: float
) -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    sim.contract["control_rate"] = control_rate
    agent = _ScriptedLLM(_move("move_to", x=0.3), _move("move_to", x=0.5))

    async with _served(sim, DirectControl()) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    payloads = steps(run.trace_id, schema=ROBOT_STEP_SCHEMA)
    segments = [p for p in payloads if p.get("source") == "video_segment"]
    observations = [p for p in payloads if p.get("source") == "observation"]
    starts = [datetime.fromisoformat(obs["started_at"]) for obs in observations]
    assert {segment["camera"] for segment in segments} == {"observation/image"}
    # The opening scene, then one observation and one frame per tick of both moves.
    assert frames_in(segments) == 1 + len(sim.actions)
    assert [obs["tick"] for obs in observations] == list(range(1 + len(sim.actions)))
    assert observations[-1]["state"]
    # The viewer's clock is the stamps' span: the video's length, not the calls' latency.
    assert [(b - a).total_seconds() for a, b in itertools.pairwise(starts)] == pytest.approx(
        [tick_seconds] * len(sim.actions)
    )


class _WatchedArm(_Arm):
    """An arm that announces its third tick."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__(contract)
        self.moving = asyncio.Event()

    def step(self, action: NDArray[Any]) -> None:
        super().step(action)
        if len(self.actions) >= 3:
            self.moving.set()


class _AbandonsMidMove(Agent):
    """Starts a long move, then fails once the sim has stepped a few ticks."""

    def __init__(self, sim: _WatchedArm) -> None:
        super().__init__()
        self.sim = sim

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        targets = {"targets": {"x": 0.5, "grip": None}, "note": "move toward the goal"}
        moving = asyncio.create_task(client.call_tool("move_to", targets))
        await asyncio.wait_for(self.sim.moving.wait(), timeout=30)
        moving.cancel()
        raise RuntimeError("the agent gave up")


async def test_ending_the_episode_mid_move_ends_the_recording_with_the_ticks_played() -> None:
    sim = _WatchedArm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))

    async with _served(sim, DirectControl(speed=0.001)) as env:
        run = await rollout(
            Task(env="arm", id="reach"), _AbandonsMidMove(sim), runtime=LocalRuntime(env)
        )

    payloads = steps(run.trace_id, schema=ROBOT_STEP_SCHEMA)
    observations = [p["tick"] for p in payloads if p.get("source") == "observation"]
    segments = [p for p in payloads if p.get("source") == "video_segment"]
    assert run.trace.status == "error"
    assert observations == list(range(len(observations)))
    assert 3 <= len(observations) < 1 + len(sim.actions)
    assert frames_in(segments) == len(observations)


async def test_reported_angles_wrap_into_pi_and_a_command_past_pi_is_accepted() -> None:
    names = ["ee.x", "ee_euler.x", "ee_euler.y", "ee_euler.z"]
    home = [0.0, 3.2, 0.0, 0.0]  # a sim that reports roll unwrapped, just past pi
    sim = _Pose(_contract("ee_abs", [-1.0, -4.0, -4.0, -4.0], [1.0, 4.0, 4.0, 4.0], names), home)
    agent = _ScriptedLLM(
        ("wait", {"duration": None, "note": "look"}),
        _move("move_to", roll=3.2),
        _move("move_to", roll=-math.pi),
    )

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    looked, past, edge = agent.results
    wrapped = 3.2 - 2 * math.pi
    for result in (looked, past):
        assert not result.isError
        assert f"ee_euler.x={wrapped:.4f}" in _text(result)
        assert f"pose: x=0.0000, roll={wrapped:.4f}" in _text(result)
    assert "At the target. Remaining offsets: " in _text(past)
    assert not edge.isError
    assert "pi and -pi are the same orientation" in agent.descriptions["move_to"]

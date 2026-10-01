"""Direct control gates: an LLM-style agent drives a robot sim through MCP motion tools.

Each test is a whole rollout: an in-process bridge serving the ``openpi/0`` wire,
an env publishing it plus :class:`DirectControl`'s ``mcp`` capability, and a
scripted agent calling the tools through the tunneled manifest binding, graded
by the sim.
"""

from __future__ import annotations

import copy
import math
from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from hud.agents.base import Agent
from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema
from hud.environment import Environment
from hud.environment.robot import DirectControl, RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray

    from hud.capabilities import MCPClient
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


class _ScriptedLLM(Agent):
    """Stands in for a tool-calling LLM: plays fixed MCP calls, keeps their results."""

    def __init__(self, *calls: tuple[str, dict[str, Any]]) -> None:
        super().__init__()
        self.calls = calls
        self.tools: set[str] = set()
        self.schemas: dict[str, dict[str, Any]] = {}
        self.results: list[MCPToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        listed = await client.list_tools()
        self.tools = {tool.name for tool in listed}
        self.schemas = {tool.name: tool.inputSchema for tool in listed}
        for name, arguments in self.calls:
            self.results.append(await client.call_tool(name, arguments))
        run.trace.content = "done"


@asynccontextmanager
async def _served(sim: RobotBridge, control: DirectControl) -> AsyncIterator[Environment]:
    """The docs' custom-bridge env with direct control attached; *sim* runs in this process."""
    await sim.start()
    server = await sim.serve_control()
    env = Environment("arm")
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)
    control.attach(endpoint)

    @env.initialize
    async def _up() -> None:
        await endpoint.start()
        for cap in await endpoint.capabilities():
            env.add_capability(cap)

    @env.shutdown
    async def _down() -> None:
        await endpoint.stop()

    @env.template()
    async def reach() -> AsyncGenerator[Any, Any]:
        ep = await endpoint.reset()
        yield {"prompt": ep["prompt"]}
        yield await endpoint.result()

    try:
        yield env
    finally:
        await env.stop()
        server.close()
        await sim.stop()


def _text(result: MCPToolResult) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


def _move(tool: str, **values: float) -> tuple[str, dict[str, Any]]:
    """One required ``target`` plus any further dimensions in ``others``."""
    items = [{"name": name, "value": value} for name, value in values.items()]
    target, *others = items
    return tool, {"target": target, "others": others, "note": "move toward the goal"}


async def test_move_to_interpolates_absolute_targets_until_the_sim_succeeds() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(
        _move("move_to", x=0.5),
        _move("move_to", grip=1.0),
    )

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_to"}  # the ee_abs contract picked the tool; no observe
    schema = agent.schemas["move_to"]
    assert set(schema["required"]) == {"target", "others", "note"}
    strict = ensure_strict_json_schema(copy.deepcopy(schema))
    assert "target" in strict["required"]  # survives strict mode, unlike minItems on a list
    assert run.reward == 1.0  # graded by the sim, not the tools
    # 0.1 of x's range per second at 10 Hz: 0.01 per tick, grip held at its reference.
    # The sim matches the command, so the call returns when the target is reached.
    reach_x, close = np.array(sim.actions[:50]), sim.actions[50:]
    np.testing.assert_allclose(reach_x[:, 0], np.linspace(0.01, 0.5, 50), rtol=1e-6)
    np.testing.assert_allclose(reach_x[:, 1], 0.0)
    # max_step lets the gripper switch in one tick while x holds its commanded target;
    # the sim succeeds on that tick, which ends the episode.
    np.testing.assert_allclose(close, [[0.5, 1.0]])
    reached, closed = agent.results
    assert [block.type for block in reached.content] == ["text", "text", "image"]
    assert "Played 50 steps (5.0 s)." in _text(reached)
    assert "observation/state: x=0.5000, grip=0.0000" in _text(reached)
    assert "pose: x=0.5000, grip=0.0000" in _text(reached)
    assert "The episode has ended" in _text(closed)


async def test_move_by_splits_a_displacement_into_steps_within_the_per_step_box() -> None:
    sim = _Arm(_contract("ee_del", [-0.1, -1.0], [0.1, 1.0]))
    agent = _ScriptedLLM(_move("move_by", x=0.35))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_by"}
    # Four in-box steps, then two still ticks once the arm has stopped.
    np.testing.assert_allclose(sim.actions, [[0.0875, 0.0]] * 4 + [[0.0, 0.0]] * 2)
    assert "observation/state: x=0.3500" in _text(agent.results[0])


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (_move("move_to", z=0.1), "unknown dimension(s) ['z']"),
        (_move("move_to", grip=1.0), "unknown dimension(s) ['grip']; valid: x"),
        (_move("move_to", x=1.5), "x=1.5 is outside [0, 1]"),
        (
            ("move_to", {"target": {"name": "x", "value": 0.1}, "others": [], "note": "  "}),
            "note must say",
        ),
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
        server.tool(self.move_to, name="move_eef", description="grasp targets", output_schema=None)


async def test_an_env_can_replace_the_contract_motion_tool() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_eef", x=0.5))

    async with _served(sim, _GraspTool(max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"move_eef"}
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
    agent = _ScriptedLLM(("move_to", {"targets": [], "note": "move toward the goal"}))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert result.isError
    assert sim.actions == []


async def test_a_move_longer_than_ten_seconds_plays_until_the_arm_arrives() -> None:
    """VLABench was rejected at 10.1 s. The cap is a 60 s safety timeout, not that reject."""
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=0.6))

    async with _served(sim, DirectControl(speed=0.05, max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert not result.isError
    # 0.6 / 0.05 s = 12 s at 10 Hz, which used to exceed the 10 s cap.
    assert len(sim.actions) >= 120
    assert sim.state[0] == pytest.approx(0.6)
    assert "safety timeout" not in _text(result)


async def test_the_safety_timeout_returns_the_pose_instead_of_rejecting_the_call() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(_move("move_to", x=1.0))

    async with _served(sim, DirectControl(dims=["x"], speed=0.05, timeout=0.3)) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    (result,) = agent.results
    assert not result.isError
    assert len(sim.actions) == 3  # 0.3 s at 10 Hz
    assert sim.state[0] < 0.05  # the prefix of the move, not a jump to the goal
    assert "safety timeout" in _text(result)
    assert "pose:" in _text(result)
    assert "commanded:" in _text(result)


async def test_a_lagging_arm_keeps_stepping_until_it_reaches_the_target() -> None:
    sim = _Pose(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]), [0.0, 0.0], lag=0.25)
    agent = _ScriptedLLM(_move("move_to", x=0.5))

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert len(sim.actions) > 50  # interpolation, then a hold while the arm catches up
    assert sim.state[0] == pytest.approx(0.5, abs=0.01)
    assert "pose:" in _text(agent.results[0])
    assert "off:" in _text(agent.results[0])


def _quat_names(order: str) -> list[str]:
    parts = ("qx", "qy", "qz", "qw") if order == "xyzw" else ("qw", "qx", "qy", "qz")
    return [f"tcp_quat.{part}" for part in parts]


async def test_a_yaw_slerps_an_xyzw_quaternion_instead_of_lerping_components() -> None:
    names = _quat_names("xyzw")
    sim = _Pose(_contract("ee_abs", [-1.0] * 4, [1.0] * 4, names), [0.0, 0.0, 0.0, 1.0])
    agent = _ScriptedLLM(_move("move_to", **{"tcp.yaw": math.pi / 2}))

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
    agent = _ScriptedLLM(_move("move_to", **{"tcp.roll": math.pi / 2}))

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
    agent = _ScriptedLLM(
        _move("move_to", **{"target_eef.roll": math.pi / 2, "target_eef.yaw": math.pi / 2})
    )

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    # Intrinsic XYZ of roll=yaw=pi/2 is the quaternion (0.5, 0.5, 0.5, 0.5), angle 2pi/3.
    component = (2.0 * np.pi / 3.0) / np.sqrt(3.0)
    np.testing.assert_allclose(sim.actions[-1], [component, component, component], atol=1e-4)
    angles = np.linalg.norm(sim.actions, axis=1)
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
    agent = _ScriptedLLM(_move("move_to", **{"ee_pos.z": 1.2, "ee.yaw": 0.4}))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    final = sim.actions[-1]
    assert final[2] == pytest.approx(1.2, abs=1e-4)
    assert final[5] == pytest.approx(0.4, abs=1e-4)
    np.testing.assert_allclose(final[3:5], 0.0, atol=1e-4)
    assert final[6] == pytest.approx(0.04)
    assert "ee.yaw" not in sim.contract["features"]["action"]["names"]


async def test_a_contract_without_a_motion_type_is_refused_at_start() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    sim.contract["features"]["action"]["type"] = "joint_vel"

    with pytest.raises(ValueError, match="direct control needs an action type"):
        async with _served(sim, DirectControl()) as env:
            await env.start()

"""Direct control gates: an LLM-style agent drives a robot sim through MCP motion tools.

Each test is a whole rollout: an in-process bridge serving the ``openpi/0`` wire,
an env publishing it plus :class:`DirectControl`'s ``mcp`` capability, and a
scripted agent calling the tools through the tunneled manifest binding, graded
by the sim.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003 - env.template resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from hud.agents.base import Agent
from hud.environment import Environment
from hud.environment.robot import DirectControl, RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray

    from hud.capabilities import MCPClient
    from hud.eval.run import Run
    from hud.types import MCPToolResult


def _contract(action_type: str, low: list[float], high: list[float]) -> dict[str, Any]:
    """One camera, an ``[x, grip]`` state, and an ``[x, grip]`` action of *action_type*."""
    return {
        "control_rate": 10,
        "features": {
            "observation/image": {"role": "observation", "type": "rgb"},
            "observation/state": {"role": "observation", "names": ["x", "grip"]},
            "action": {
                "role": "action",
                "type": action_type,
                "names": ["x", "grip"],
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
        self.results: list[MCPToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        self.tools = {tool.name for tool in await client.list_tools()}
        for name, arguments in self.calls:
            self.results.append(await client.call_tool(name, arguments))
        run.trace.content = "done"


@asynccontextmanager
async def _served(sim: _Arm, control: DirectControl) -> AsyncIterator[Environment]:
    """The docs' custom-bridge env with direct control attached; *sim* runs in this process."""
    await sim.start()
    server = await sim.serve_control()
    env = Environment("arm")
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)

    @env.initialize
    async def _up() -> None:
        await endpoint.start()
        for cap in await endpoint.capabilities():
            env.add_capability(cap)

    @env.shutdown
    async def _down() -> None:
        await endpoint.stop()

    control.attach(env)

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


def _move(tool: str, key: str, **values: float) -> tuple[str, dict[str, Any]]:
    return tool, {key: [{"name": name, "value": value} for name, value in values.items()]}


async def test_move_to_interpolates_absolute_targets_until_the_sim_succeeds() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    agent = _ScriptedLLM(
        ("observe", {}),
        _move("move_to", "targets", x=0.5),
        _move("move_to", "targets", grip=1.0),
    )

    async with _served(sim, DirectControl(max_step={"grip": 2.0})) as env:
        run = await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"observe", "move_to"}  # the ee_abs contract picked the tool
    assert run.reward == 1.0  # graded by the sim, not the tools
    # 0.1 of x's range per second at 10 Hz: 0.01 per tick, grip held at its reference,
    # then a 0.5 s settle at the target.
    reach_x, settle, close = np.array(sim.actions[:50]), sim.actions[50:55], sim.actions[55:]
    np.testing.assert_allclose(reach_x[:, 0], np.linspace(0.01, 0.5, 50), rtol=1e-6)
    np.testing.assert_allclose(reach_x[:, 1], 0.0)
    np.testing.assert_allclose(settle, [[0.5, 0.0]] * 5)
    # max_step lets the gripper switch in one tick while x holds its commanded target;
    # the sim succeeds on that tick, which ends the episode mid-settle.
    np.testing.assert_allclose(close, [[0.5, 1.0]])
    observed, reached, closed = agent.results
    assert [block.type for block in observed.content] == ["text", "text", "image"]
    assert "Played 55 steps (5.5 s)." in _text(reached)
    assert "observation/state: x=0.5000, grip=0.0000" in _text(reached)
    assert "The episode has ended" in _text(closed)


async def test_move_by_splits_a_displacement_into_steps_within_the_per_step_box() -> None:
    sim = _Arm(_contract("ee_del", [-0.1, -1.0], [0.1, 1.0]))
    agent = _ScriptedLLM(_move("move_by", "deltas", x=0.35))

    async with _served(sim, DirectControl()) as env:
        await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))

    assert agent.tools == {"observe", "move_by"}
    # Four in-box steps, then the settle holds still with zero displacement.
    np.testing.assert_allclose(sim.actions, [[0.0875, 0.0]] * 4 + [[0.0, 0.0]] * 5)
    assert "observation/state: x=0.3500" in _text(agent.results[0])


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (_move("move_to", "targets", z=0.1), "unknown dimension(s) ['z']"),
        (_move("move_to", "targets", grip=1.0), "unknown dimension(s) ['grip']; valid: x"),
        (_move("move_to", "targets", x=1.5), "x=1.5 is outside [0, 1]"),
        (_move("move_to", "targets", x=1.0), "over the 10 s per-call cap"),
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


async def test_a_contract_without_a_motion_type_is_refused_at_start() -> None:
    sim = _Arm(_contract("ee_abs", [0.0, -1.0], [1.0, 1.0]))
    sim.contract["features"]["action"]["type"] = "joint_vel"

    with pytest.raises(ValueError, match="direct control needs an action type"):
        async with _served(sim, DirectControl()) as env:
            await env.start()

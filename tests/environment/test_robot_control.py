"""Direct control: a tool-calling agent drives a robot sim through MCP motion tools.

Each row is a whole rollout. A bridge in this process serves the ``openpi/0``
wire and its control channel; an env publishes it through ``RobotEndpoint`` with
:class:`DirectControl` attached; a scripted agent calls the motion tool over the
tunneled ``mcp`` binding, and the sim grades the episode. A row asserts what the
sim was commanded, what each tool call answered, the reward, and the trace the
episode streamed to the span file.
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
from fastmcp import FastMCP  # noqa: TC002 - the override signature is resolved by FastMCP
from inline_snapshot import snapshot

from hud.agents.base import Agent
from hud.agents.openai.tools.strict_schema import ensure_strict_json_schema
from hud.environment import Environment
from hud.environment.robot import DirectControl, RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout
from tests.harness import ROBOT_STEP_SCHEMA, steps

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from numpy.typing import NDArray

    from hud.capabilities import MCPClient
    from hud.eval.run import Run
    from hud.types import MCPToolResult


def contract(
    action_type: str, low: list[float], high: list[float], names: list[str] | None = None
) -> dict[str, Any]:
    """One camera, a state as wide as the action, and an action of ``action_type``."""
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


class Arm(RobotBridge):
    """A 1-D arm and gripper that succeeds, ending the episode, at ``x >= 0.5`` closed."""

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


class Pose(RobotBridge):
    """Tracks an absolute action: ``lag`` 1 snaps to the command, less trails it."""

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

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data = {
            "observation/image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "observation/state": self.state[None].astype(np.float32),
        }
        return data, np.array([False])


class GraspTool(DirectControl):
    """An env serving its own motion tool on the same wire."""

    def _bind_tools(self, server: FastMCP) -> None:
        server.tool(self.move, name="move_eef", description="grasp targets", output_schema=None)


class ToolCaller(Agent):
    """Stands in for a tool-calling LLM: makes fixed MCP calls and keeps the results."""

    def __init__(self, *calls: tuple[str, dict[str, Any]]) -> None:
        super().__init__()
        self.calls = calls
        self.schemas: dict[str, dict[str, Any]] = {}
        self.results: list[MCPToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        self.schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}
        for name, arguments in self.calls:
            self.results.append(await client.call_tool(name, arguments))
        run.trace.content = "done"


@asynccontextmanager
async def arm_env(sim: RobotBridge, control: DirectControl) -> AsyncIterator[Environment]:
    """The docs' custom-bridge env with direct control attached; ``sim`` runs in this process."""
    await sim.start()
    server = await sim.serve_control()
    env = Environment("arm")
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)
    control.attach(endpoint)

    @env.initialize
    async def connect_sim() -> None:
        await endpoint.start()
        for capability in await endpoint.capabilities():
            env.add_capability(capability)

    @env.shutdown
    async def disconnect_sim() -> None:
        await endpoint.stop()

    @env.template()
    async def reach() -> AsyncGenerator[Any, Any]:
        episode = await endpoint.reset()
        yield {"prompt": episode["prompt"]}
        yield await endpoint.result()

    try:
        yield env
    finally:
        await env.stop()
        server.close()
        await sim.stop()


async def drive(sim: RobotBridge, control: DirectControl, agent: Agent) -> Run:
    async with arm_env(sim, control) as env:
        return await rollout(Task(env="arm", id="reach"), agent, runtime=LocalRuntime(env))


def move(tool: str, **values: float) -> tuple[str, dict[str, Any]]:
    """One required ``target`` and any further dimensions in ``others``."""
    target, *others = [{"name": name, "value": value} for name, value in values.items()]
    return tool, {"target": target, "others": others, "note": "move toward the goal"}


def rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    """The xyzw quaternion of rotation Rz(yaw) Ry(pitch) Rx(roll)."""

    def product(a: list[float], b: list[float]) -> list[float]:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]

    def axis(index: int, angle: float) -> list[float]:
        quaternion = [0.0, 0.0, 0.0, math.cos(angle / 2)]
        quaternion[index] = math.sin(angle / 2)
        return quaternion

    return product(product(axis(2, yaw), axis(1, pitch)), axis(0, roll))


XYZW = [f"tcp_quat.{part}" for part in ("qx", "qy", "qz", "qw")]
WXYZ = [f"tcp_quat.{part}" for part in ("qw", "qx", "qy", "qz")]
AXIS_ANGLE = [f"target_eef_axis_angle.{part}" for part in ("rx", "ry", "rz")]
EULER = ["ee_pos.x", "ee_pos.y", "ee_pos.z", "ee_euler.x", "ee_euler.y", "ee_euler.z", "gripper"]


def unit_quaternions(actions: NDArray[Any]) -> bool:
    return bool(np.allclose(np.linalg.norm(actions, axis=1), 1.0, atol=1e-5))


def even_rotation(actions: NDArray[Any]) -> list[float]:
    """Distinct per-tick changes of the rotation angle: one value for a constant rate."""
    return sorted({round(float(gap), 2) for gap in np.diff(np.linalg.norm(actions, axis=1))})


def tick_lengths(actions: NDArray[Any]) -> list[float]:
    """Distinct distances commanded per tick: a constant pace has one value per phase."""
    return sorted({round(float(np.linalg.norm(gap)), 4) for gap in np.diff(actions, axis=0)})


def quaternion_alignment(target: list[float]) -> Callable[[NDArray[Any]], float]:
    """|cos| of the half-angle between the final command and ``target`` (1.0 is equal)."""
    return lambda actions: round(abs(float(np.dot(actions[-1], target))), 6)


ARM = contract("ee_abs", [0.0, -1.0], [1.0, 1.0])

ROWS = [
    pytest.param(
        Arm(ARM),
        DirectControl(max_step={"grip": 2.0}),
        [move("move_to", x=0.5), move("move_to", grip=1.0)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 50 steps (5.0 s)."],
                            ["observation/state: x=0.5000", "grip=0.0000"],
                            ["pose: x=0.5000", "grip=0.0000"],
                            ["commanded: x=0.5000", "grip=0.0000"],
                            ["off: x=0.0000", "grip=0.0000"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    },
                    {
                        "lines": [
                            ["Played 1 steps (0.1 s)."],
                            ["observation/state: x=0.5000", "grip=1.0000"],
                            ["pose: x=0.5000", "grip=1.0000"],
                            ["commanded: x=0.5000", "grip=1.0000"],
                            ["off: x=0.0000", "grip=0.0000"],
                            ["The episode has ended; stop calling tools."],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    },
                ],
                "ticks": 51,
                "first": [0.01, 0.0],
                "last": [0.5, 1.0],
                "measure": [0.01, 1.0],
                "reward": 1.0,
            }
        ),
        id="move-to-interpolates-until-the-sim-succeeds",
    ),
    pytest.param(
        Arm(contract("ee_del", [-0.1, -1.0], [0.1, 1.0])),
        DirectControl(),
        [move("move_by", x=0.35)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_by"],
                "results": [
                    {
                        "lines": [
                            ["Played 6 steps (0.6 s)."],
                            ["observation/state: x=0.3500", "grip=0.0000"],
                            ["pose: x=0.3500", "grip=0.0000"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 6,
                "first": [0.0875, 0.0],
                "last": [0.0, 0.0],
                "measure": [0.0, 0.0875],
                "reward": 0.0,
            }
        ),
        id="move-by-splits-a-displacement-into-in-box-steps",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(dims=["x"], speed=0.05),
        [move("move_to", z=0.1)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [{"error": "unknown dimension(s) ['z']; valid: x"}],
                "ticks": 0,
                "first": None,
                "last": None,
                "measure": None,
                "reward": 0.0,
            }
        ),
        id="unknown-dimension-is-a-correctable-error",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(dims=["x"], speed=0.05),
        [move("move_to", grip=1.0)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [{"error": "unknown dimension(s) ['grip']; valid: x"}],
                "ticks": 0,
                "first": None,
                "last": None,
                "measure": None,
                "reward": 0.0,
            }
        ),
        id="dimension-outside-dims-is-a-correctable-error",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(dims=["x"], speed=0.05),
        [move("move_to", x=1.5)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [{"error": "x=1.5 is outside [0, 1]"}],
                "ticks": 0,
                "first": None,
                "last": None,
                "measure": None,
                "reward": 0.0,
            }
        ),
        id="target-out-of-range-is-a-correctable-error",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(dims=["x"], speed=0.05),
        [("move_to", {"target": {"name": "x", "value": 0.1}, "others": [], "note": "  "})],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [{"error": "note must say what you see and why you chose this motion"}],
                "ticks": 0,
                "first": None,
                "last": None,
                "measure": None,
                "reward": 0.0,
            }
        ),
        id="blank-note-is-a-correctable-error",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(),
        [("move_to", {"targets": [], "note": "move toward the goal"})],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [{"error": "3 validation errors for call[move]"}],
                "ticks": 0,
                "first": None,
                "last": None,
                "measure": None,
                "reward": 0.0,
            }
        ),
        id="missing-target-is-rejected",
    ),
    pytest.param(
        Arm(ARM),
        GraspTool(max_step={"grip": 2.0}),
        [move("move_eef", x=0.5)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_eef"],
                "results": [
                    {
                        "lines": [
                            ["Played 50 steps (5.0 s)."],
                            ["observation/state: x=0.5000", "grip=0.0000"],
                            ["pose: x=0.5000", "grip=0.0000"],
                            ["commanded: x=0.5000", "grip=0.0000"],
                            ["off: x=0.0000", "grip=0.0000"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 50,
                "first": [0.01, 0.0],
                "last": [0.5, 0.0],
                "measure": [0.01],
                "reward": 0.0,
            }
        ),
        id="env-replaces-the-contract-motion-tool",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(speed=0.05, max_step={"grip": 2.0}),
        [move("move_to", x=0.6)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 120 steps (12.0 s)."],
                            ["observation/state: x=0.6000", "grip=0.0000"],
                            ["pose: x=0.6000", "grip=0.0000"],
                            ["commanded: x=0.6000", "grip=0.0000"],
                            ["off: x=0.0000", "grip=0.0000"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 120,
                "first": [0.005, 0.0],
                "last": [0.6, 0.0],
                "measure": [0.005],
                "reward": 0.0,
            }
        ),
        id="a-twelve-second-move-plays-until-arrival",
    ),
    pytest.param(
        Arm(ARM),
        DirectControl(dims=["x"], speed=0.05, timeout=0.3),
        [move("move_to", x=1.0)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 3 steps (0.3 s)."],
                            ["Stopped at the 0.3 s safety timeout before the motion finished."],
                            ["observation/state: x=0.0150", "grip=0.0000"],
                            ["pose: x=0.0150"],
                            ["commanded: x=1.0000"],
                            ["off: x=-0.9850"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 3,
                "first": [0.005, 0.0],
                "last": [0.015, 0.0],
                "measure": [0.005],
                "reward": 0.0,
            }
        ),
        id="safety-timeout-returns-the-pose",
    ),
    pytest.param(
        Pose(ARM, [0.0, 0.0], lag=0.25),
        DirectControl(max_step={"grip": 2.0}),
        [move("move_to", x=0.5)],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 54 steps (5.4 s)."],
                            ["observation/state: x=0.4905", "grip=0.0000"],
                            ["pose: x=0.4905", "grip=0.0000"],
                            ["commanded: x=0.5000", "grip=0.0000"],
                            ["off: x=-0.0095", "grip=0.0000"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 54,
                "first": [0.01, 0.0],
                "last": [0.5, 0.0],
                "measure": [0.0, 0.01],
                "reward": 0.0,
            }
        ),
        id="a-lagging-arm-is-held-until-it-arrives",
    ),
    pytest.param(
        Pose(contract("ee_abs", [-1.0] * 4, [1.0] * 4, XYZW), rpy(0, 0, 0)),
        DirectControl(),
        [move("move_to", **{"tcp.yaw": math.pi / 2})],
        unit_quaternions,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 25 steps (2.5 s)."],
                            [
                                "observation/state: tcp_quat.qx=0.0000",
                                "tcp_quat.qy=0.0000",
                                "tcp_quat.qz=0.7071",
                                "tcp_quat.qw=0.7071",
                            ],
                            ["pose: tcp.roll=0.0000", "tcp.pitch=-0.0000", "tcp.yaw=1.5708"],
                            ["commanded: tcp.roll=0.0000", "tcp.pitch=-0.0000", "tcp.yaw=1.5708"],
                            ["off: orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 25,
                "first": [0.0, 0.0, 0.0314, 0.9995],
                "last": [0.0, 0.0, 0.7071, 0.7071],
                "measure": True,
                "reward": 0.0,
            }
        ),
        id="yaw-slerps-an-xyzw-quaternion",
    ),
    pytest.param(
        Pose(contract("ee_abs", [-1.0] * 4, [1.0] * 4, WXYZ), [1.0, 0.0, 0.0, 0.0]),
        DirectControl(),
        [move("move_to", **{"tcp.roll": math.pi / 2})],
        unit_quaternions,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 25 steps (2.5 s)."],
                            [
                                "observation/state: tcp_quat.qw=0.7071",
                                "tcp_quat.qx=0.7071",
                                "tcp_quat.qy=0.0000",
                                "tcp_quat.qz=0.0000",
                            ],
                            ["pose: tcp.roll=1.5708", "tcp.pitch=-0.0000", "tcp.yaw=0.0000"],
                            ["commanded: tcp.roll=1.5708", "tcp.pitch=-0.0000", "tcp.yaw=0.0000"],
                            ["off: orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 25,
                "first": [0.9995, 0.0314, 0.0, 0.0],
                "last": [0.7071, 0.7071, 0.0, 0.0],
                "measure": True,
                "reward": 0.0,
            }
        ),
        id="roll-writes-a-wxyz-quaternion",
    ),
    pytest.param(
        Pose(contract("ee_abs", [-1.0] * 4, [1.0] * 4, XYZW), rpy(0.4, math.pi / 2, 0)),
        DirectControl(),
        [move("move_to", **{"tcp.yaw": -0.25})],
        quaternion_alignment(rpy(0.4, math.pi / 2, -0.25)),
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 4 steps (0.4 s)."],
                            [
                                "observation/state: tcp_quat.qx=0.2258",
                                "tcp_quat.qy=0.6701",
                                "tcp_quat.qz=-0.2258",
                                "tcp_quat.qw=0.6701",
                            ],
                            ["pose: tcp.roll=0.6500", "tcp.pitch=1.5708", "tcp.yaw=0.0000"],
                            ["commanded: tcp.roll=0.6500", "tcp.pitch=1.5708", "tcp.yaw=0.0000"],
                            ["off: orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 4,
                "first": [0.1621, 0.6883, -0.1621, 0.6883],
                "last": [0.2258, 0.6701, -0.2258, 0.6701],
                "measure": 1.0,
                "reward": 0.0,
            }
        ),
        id="yaw-at-pitch-up-keeps-the-wrist-roll",
    ),
    pytest.param(
        Pose(contract("ee_abs", [-1.0] * 4, [1.0] * 4, XYZW), rpy(0.4, -math.pi / 2, 0)),
        DirectControl(),
        [move("move_to", **{"tcp.yaw": -0.25})],
        quaternion_alignment(rpy(0.4, -math.pi / 2, -0.25)),
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 4 steps (0.4 s)."],
                            [
                                "observation/state: tcp_quat.qx=0.0530",
                                "tcp_quat.qy=-0.7051",
                                "tcp_quat.qz=0.0530",
                                "tcp_quat.qw=0.7051",
                            ],
                            ["pose: tcp.roll=0.1500", "tcp.pitch=-1.5708", "tcp.yaw=0.0000"],
                            ["commanded: tcp.roll=0.1500", "tcp.pitch=-1.5708", "tcp.yaw=0.0000"],
                            ["off: orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 4,
                "first": [0.1188, -0.6971, 0.1188, 0.6971],
                "last": [0.053, -0.7051, 0.053, 0.7051],
                "measure": 1.0,
                "reward": 0.0,
            }
        ),
        id="yaw-at-pitch-down-keeps-the-wrist-roll",
    ),
    pytest.param(
        Pose(
            contract("ee_abs", [0.0, -1, -1, -1, -1], [1, 1, 1, 1, 1], ["x", *XYZW]),
            [0.0, 0.0, 0.2, 0.0, 0.9],
        ),
        DirectControl(),
        [move("move_to", x=0.2)],
        lambda actions: sorted({tuple(row) for row in np.round(actions[:, 1:], 6).tolist()}),
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 20 steps (2.0 s)."],
                            [
                                "observation/state: x=0.2000",
                                "tcp_quat.qx=0.0000",
                                "tcp_quat.qy=0.2000",
                                "tcp_quat.qz=0.0000",
                                "tcp_quat.qw=0.9000",
                            ],
                            [
                                "pose: x=0.2000",
                                "tcp.roll=0.0000",
                                "tcp.pitch=0.4373",
                                "tcp.yaw=0.0000",
                            ],
                            [
                                "commanded: x=0.2000",
                                "tcp.roll=0.0000",
                                "tcp.pitch=0.4373",
                                "tcp.yaw=0.0000",
                            ],
                            ["off: x=0.0000", "orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 20,
                "first": [0.01, 0.0, 0.2, 0.0, 0.9],
                "last": [0.2, 0.0, 0.2, 0.0, 0.9],
                "measure": [(0.0, 0.2, 0.0, 0.9)],
                "reward": 0.0,
            }
        ),
        id="an-unnamed-quaternion-is-held-unchanged",
    ),
    pytest.param(
        Pose(contract("ee_abs", [-np.pi] * 3, [np.pi] * 3, AXIS_ANGLE), [0.0, 0.0, 0.0]),
        DirectControl(),
        [move("move_to", **{"target_eef.roll": math.pi / 2, "target_eef.yaw": math.pi / 2})],
        even_rotation,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 34 steps (3.4 s)."],
                            [
                                "observation/state: target_eef_axis_angle.rx=1.2092",
                                "target_eef_axis_angle.ry=1.2092",
                                "target_eef_axis_angle.rz=1.2092",
                            ],
                            [
                                "pose: target_eef.roll=1.5708",
                                "target_eef.pitch=0.0000",
                                "target_eef.yaw=1.5708",
                            ],
                            [
                                "commanded: target_eef.roll=1.5708",
                                "target_eef.pitch=-0.0000",
                                "target_eef.yaw=1.5708",
                            ],
                            ["off: orientation=0.0000 rad"],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 34,
                "first": [0.0356, 0.0356, 0.0356],
                "last": [1.2092, 1.2092, 1.2092],
                "measure": [0.06],
                "reward": 0.0,
            }
        ),
        id="euler-commands-slerp-onto-axis-angle",
    ),
    pytest.param(
        Pose(
            contract(
                "ee_abs",
                [-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi, 0.0],
                [1.0, 1.0, 2.0, np.pi, np.pi, np.pi, 0.04],
                EULER,
            ),
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04],
        ),
        DirectControl(),
        [move("move_to", **{"ee_pos.z": 1.2, "ee.yaw": 0.4})],
        tick_lengths,
        snapshot(
            {
                "tools": ["move_to"],
                "results": [
                    {
                        "lines": [
                            ["Played 10 steps (1.0 s)."],
                            [
                                "observation/state: ee_pos.x=0.0000",
                                "ee_pos.y=0.0000",
                                "ee_pos.z=1.2000",
                                "ee_euler.x=0.0000",
                                "ee_euler.y=0.0000",
                                "ee_euler.z=0.4000",
                                "gripper=0.0400",
                            ],
                            [
                                "pose: ee_pos.x=0.0000",
                                "ee_pos.y=0.0000",
                                "ee_pos.z=1.2000",
                                "ee.roll=0.0000",
                                "ee.pitch=-0.0000",
                                "ee.yaw=0.4000",
                                "gripper=0.0400",
                            ],
                            [
                                "commanded: ee_pos.x=0.0000",
                                "ee_pos.y=0.0000",
                                "ee_pos.z=1.2000",
                                "ee.roll=0.0000",
                                "ee.pitch=-0.0000",
                                "ee.yaw=0.4000",
                                "gripper=0.0400",
                            ],
                            [
                                "off: ee_pos.x=0.0000",
                                "ee_pos.y=0.0000",
                                "ee_pos.z=0.0000",
                                "orientation=0.0000 rad",
                                "gripper=0.0000",
                            ],
                            ["camera observation/image:"],
                        ],
                        "images": 1,
                    }
                ],
                "ticks": 10,
                "first": [0.0, 0.0, 1.02, 0.0, 0.0, 0.04, 0.04],
                "last": [0.0, 0.0, 1.2, 0.0, 0.0, 0.4, 0.04],
                "measure": [0.0447],
                "reward": 0.0,
            }
        ),
        id="an-euler-contract-is-addressed-as-roll-pitch-yaw",
    ),
]


def answered(result: MCPToolResult) -> dict[str, Any]:
    """A tool result as the model reads it: an error's first line, or the reply's lines."""
    texts = [block.text for block in result.content if block.type == "text"]
    if result.isError:
        return {"error": texts[0].splitlines()[0]}
    return {
        "lines": [line.split(", ") for text in texts for line in text.splitlines()],
        "images": sum(block.type == "image" for block in result.content),
    }


def rounded(action: NDArray[Any]) -> list[float]:
    return (np.round(action, 4) + 0.0).tolist()


@pytest.mark.parametrize(("sim", "control", "calls", "measure", "expected"), ROWS)
async def test_a_motion_tool_call_plays_out_on_the_sim(
    sim: Arm | Pose,
    control: DirectControl,
    calls: list[tuple[str, dict[str, Any]]],
    measure: Callable[[NDArray[Any]], Any],
    expected: Any,
) -> None:
    agent = ToolCaller(*calls)

    run = await drive(sim, control, agent)

    actions = np.array(sim.actions, dtype=np.float64)
    assert {
        "tools": sorted(agent.schemas),
        "results": [answered(result) for result in agent.results],
        "ticks": len(actions),
        "first": rounded(actions[0]) if len(actions) else None,
        "last": rounded(actions[-1]) if len(actions) else None,
        "measure": measure(actions) if len(actions) else None,
        "reward": run.reward,
    } == expected


async def test_the_motion_tool_schema_keeps_its_target_required_under_strict_mode() -> None:
    agent = ToolCaller()

    await drive(Arm(ARM), DirectControl(), agent)

    schema = agent.schemas["move_to"]
    assert sorted(schema["required"]) == ["note", "others", "target"]
    assert "target" in ensure_strict_json_schema(copy.deepcopy(schema))["required"]


async def test_a_contract_without_a_motion_action_type_is_refused_at_start() -> None:
    sim = Arm(contract("joint_vel", [0.0, -1.0], [1.0, 1.0]))

    with pytest.raises(ValueError, match="direct control needs an action type in"):
        async with arm_env(sim, DirectControl()) as env:
            await env.start()


def frames_in(segments: list[dict[str, Any]]) -> int:
    mp4 = b"".join(base64.b64decode(segment["segment"]["data"]) for segment in segments)
    with av.open(io.BytesIO(mp4), mode="r") as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.mark.parametrize(("control_rate", "tick_seconds"), [(10, 0.1), (0.4, 1.0)])
async def test_every_played_tick_streams_to_the_trace_as_state_and_video(
    control_rate: float, tick_seconds: float
) -> None:
    sim = Arm(ARM | {"control_rate": control_rate})

    run = await drive(
        sim, DirectControl(), ToolCaller(move("move_to", x=0.3), move("move_to", x=0.5))
    )

    payloads = steps(run.trace_id, schema=ROBOT_STEP_SCHEMA)
    observations = [p for p in payloads if p.get("source") == "observation"]
    segments = [p for p in payloads if p.get("source") == "video_segment"]
    starts = [datetime.fromisoformat(obs["started_at"]) for obs in observations]
    # The opening scene, then one observation and one frame per tick of both moves.
    assert [obs["tick"] for obs in observations] == list(range(1 + len(sim.actions)))
    assert {segment["camera"] for segment in segments} == {"observation/image"}
    assert frames_in(segments) == 1 + len(sim.actions)
    assert observations[-1]["state"]
    assert [(b - a).total_seconds() for a, b in itertools.pairwise(starts)] == pytest.approx(
        [tick_seconds] * len(sim.actions)
    )


class WatchedArm(Arm):
    """An arm that announces its third tick."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__(contract)
        self.moving = asyncio.Event()

    def step(self, action: NDArray[Any]) -> None:
        super().step(action)
        if len(self.actions) >= 3:
            self.moving.set()


class AbandonsMidMove(Agent):
    """Starts a long move, then fails once the sim has stepped a few ticks."""

    def __init__(self, sim: WatchedArm) -> None:
        super().__init__()
        self.sim = sim

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        moving = asyncio.create_task(client.call_tool(*move("move_to", x=0.5)))
        await asyncio.wait_for(self.sim.moving.wait(), timeout=30)
        moving.cancel()
        raise RuntimeError("the agent gave up")


async def test_cancelling_the_session_mid_move_ends_the_recording() -> None:
    sim = WatchedArm(ARM)

    run = await drive(sim, DirectControl(speed=0.001), AbandonsMidMove(sim))

    payloads = steps(run.trace_id, schema=ROBOT_STEP_SCHEMA)
    observations = [p["tick"] for p in payloads if p.get("source") == "observation"]
    segments = [p for p in payloads if p.get("source") == "video_segment"]
    assert run.trace.status == "error"
    assert observations == list(range(len(observations)))
    assert 3 <= len(observations) < 1 + len(sim.actions)
    assert frames_in(segments) == len(observations)

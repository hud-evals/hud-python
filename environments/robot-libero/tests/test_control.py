"""A tool-calling rollout against the LIBERO control profile, without MuJoCo."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from hud.agents.base import Agent
from hud.environment import Environment
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import LocalRuntime, Task, rollout
from hud.settings import settings

if TYPE_CHECKING:
    from hud.capabilities import MCPClient
    from hud.eval.run import Run
    from hud.types import MCPToolResult
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control import LiberoDirectControl  # noqa: E402


class _Arm(RobotBridge):
    def __init__(self) -> None:
        super().__init__()
        self.contract = json.loads((ROOT / "contracts" / "ee_abs.json").read_text())
        self.state = np.zeros(8, dtype=np.float32)

    def reset(self, **task_args: Any) -> str:
        self.state = np.array([0, 0, 1, 0, 0, 0, 0.04, -0.04], dtype=np.float32)
        return "move right and grasp the object"

    def step(self, action: NDArray[Any]) -> None:
        self.state[:6] = action[0, :6]
        opening = 0.01 if action[0, 6] > 0 else 0.04
        self.state[6:] = opening, -opening
        self.success = bool(self.state[0] >= 0.09 and opening < 0.03)

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], bool]:
        frame = np.zeros((1, 256, 256, 3), dtype=np.uint8)
        return {
            "observation/image": frame,
            "observation/wrist_image": frame,
            "observation/state": self.state[None],
        }, False


class _ToolAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[MCPToolResult] = []

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert set(tools) == {"move_to", "wait"}
        names = tools["move_to"].inputSchema["properties"]["targets"]["properties"]
        assert set(names) == {"x", "y", "z", "roll", "pitch", "yaw", "grip"}
        for values in ({"x": 0.1}, {"grip": 1.0}):
            self.results.append(
                await client.call_tool(
                    "move_to", {"targets": dict.fromkeys(names) | values, "note": "grasp the object"}
                )
            )
        self.results.append(await client.call_tool("wait", {"duration": None, "note": "check the grasp"}))
        run.trace.content = "grasped the object"


@pytest.mark.asyncio
async def test_move_and_wait_return_camera_frames_and_grasp_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    monkeypatch.setattr(settings, "telemetry_local_dir", None)
    sim = _Arm()
    await sim.start()
    server = await sim.serve_control()
    env = Environment(name="robot-libero")
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1]).attach(env)
    LiberoDirectControl(
        reference=lambda data: np.r_[np.asarray(data["observation/state"]).reshape(-1)[:6], -1.0],
        max_step={"gripper.open_close": 2.0},
        time_limit=20,
    ).attach(endpoint)

    @env.initialize
    async def _up() -> None:
        await endpoint.start()
        for capability in await endpoint.capabilities():
            env.add_capability(capability)

    @env.shutdown
    async def _down() -> None:
        await endpoint.stop()

    @env.template()
    async def grasp():
        ep = await endpoint.reset()
        yield {"prompt": ep["prompt"]}
        yield await endpoint.result()

    agent = _ToolAgent()
    try:
        run = await rollout(Task(env="robot-libero", id="grasp"), agent, runtime=LocalRuntime(env))
        assert {(cap.name, cap.protocol) for cap in env.capabilities} == {
            ("robot", "openpi/0"),
            ("control", "mcp/2025-11-25"),
        }
    finally:
        await env.stop()
        server.close()
        await sim.stop()

    assert run.reward == 1.0
    assert len(agent.results) == 3
    texts = []
    for result in agent.results:
        assert not result.isError
        assert sum(block.type == "image" for block in result.content) == 2
        texts.append("\n".join(block.text for block in result.content if block.type == "text"))
    moved, closed, waited = texts
    assert moved.startswith("Played ")
    assert "Time remaining:" in moved
    assert "At the target." in moved
    assert "open" in moved
    assert "fingers did not meet" in closed
    assert "fingers did not meet" in waited

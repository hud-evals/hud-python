"""Build the image and play one short rollout in it, on the real simulator."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from hud.agents.base import Agent
from hud.eval import DockerRuntime, Task, rollout

if TYPE_CHECKING:
    from hud.capabilities import MCPClient
    from hud.eval.run import Run
    from hud.types import MCPToolResult

TAG = "robot-libero:test"


class OneMove(Agent):
    """Lowers the gripper once through the `move_to` tool."""

    def __init__(self) -> None:
        super().__init__()
        self.result: MCPToolResult | None = None

    async def __call__(self, run: Run) -> None:
        client = cast("MCPClient", await run.client.open("control"))
        names = (await client.list_tools())[0].inputSchema["properties"]["targets"]["properties"]
        self.result = await client.call_tool(
            "move_to", {"targets": dict.fromkeys(names) | {"z": 1.10}, "note": "lower the gripper"}
        )
        run.trace.content = "done"


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.docker
async def test_the_image_builds_and_plays_a_libero_task() -> None:
    # The same build the README documents, from the cookbook directory.
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.hud", "-t", TAG, "."],
        cwd=Path(__file__).parents[1],
        check=True,
    )
    agent = OneMove()

    run = await rollout(
        Task(env="robot-libero", id="libero_spatial", args={"libero_task_id": 0}),
        agent,
        runtime=DockerRuntime(TAG),
    )

    assert agent.result is not None
    assert not agent.result.isError
    assert sum(block.type == "image" for block in agent.result.content) == 2
    assert not run.grade.is_error  # graded; a single move does not solve the task

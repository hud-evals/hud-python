"""Docker coverage for Harbor phase behavior.

The tests build adapted images and require network access::

    uv run pytest -m integration
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from hud.agents.base import Agent
from hud.eval import DockerRuntime
from integrations import harbor

if TYPE_CHECKING:
    from hud.capabilities import MCPClient, SSHClient
    from hud.eval.run import Run

TASKS = Path(__file__).parent / "tasks"
REPO = Path(__file__).resolve().parents[3]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="adapted images are Linux containers"),
]


@pytest.fixture(scope="module", autouse=True)
def docker_daemon() -> None:
    if shutil.which("docker") is None:
        pytest.skip("needs a running Docker daemon")
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=20)
    except subprocess.TimeoutExpired:
        pytest.skip("Docker daemon did not answer")
    if probe.returncode != 0:
        pytest.skip("needs a running Docker daemon")


class Oracle(Agent):
    """Run each fixture's reference solution."""

    def __init__(self, solutions: dict[str, str]) -> None:
        self.solutions = solutions

    async def __call__(self, run: Run) -> None:
        ssh = cast("SSHClient", await run.client.open("ssh/2"))
        if run.task_id == "hello-mcp":
            mcp = cast("MCPClient", await run.client.open("mcp-server"))
            assert {tool.name for tool in await mcp.list_tools()} == {"get_secret"}
            assert run.client.manifest is not None
            capability = next(
                cap for cap in run.client.manifest.bindings if cap.name == "mcp-server"
            )
            script = f"""
import json
from pathlib import Path
from urllib.request import Request, urlopen

url = {capability.url!r}
session_id = None

def post(payload):
    global session_id
    headers = {{
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }}
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request) as response:
        body = response.read().decode()
        returned_session = response.headers.get("Mcp-Session-Id")
        if returned_session:
            session_id = returned_session
    data = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    return json.loads(data[-1] if data else body) if body else None

post({{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {{
        "protocolVersion": "2025-06-18",
        "capabilities": {{}},
        "clientInfo": {{"name": "hud-test", "version": "1"}},
    }},
}})
post({{"jsonrpc": "2.0", "method": "notifications/initialized"}})
result = post({{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {{"name": "get_secret", "arguments": {{}}}},
}})
Path("/app/secret.txt").write_text(result["result"]["content"][0]["text"])
"""
            result = await ssh.conn.run("python3 -", input=script, check=True)
            if result.exit_status is None:
                raise RuntimeError("workspace MCP client closed without an exit status")
            run.trace.content = "called get_secret from inside the workspace"
            return
        result = await ssh.conn.run(self.solutions[run.task_id], check=True)
        if result.exit_status is None:
            raise RuntimeError("solution SSH channel closed without an exit status")
        run.trace.content = "solution completed"


async def _grade_every_task(dataset: Path, wheel: Path) -> dict[str, Run]:
    solutions = {
        task.name: (task / "solution" / "solve.sh").read_text("utf-8")
        for task in sorted(dataset.iterdir())
        if (task / "task.toml").is_file()
    }
    taskset = await harbor.adapt(dataset, hud_requirement=str(wheel))
    job = await taskset.run(
        Oracle(solutions),
        runtime=DockerRuntime(),
        max_concurrent=1,
    )
    return {run.task_id: run for run in job.runs}


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Run]:
    """Adapt and grade the fixture dataset once for this module."""
    workdir = tmp_path_factory.mktemp("harbor")
    dataset = workdir / "harbor-harness"
    shutil.copytree(TASKS, dataset)

    wheels = workdir / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return asyncio.run(_grade_every_task(dataset, next(iter(wheels.glob("*.whl")))))


@pytest.mark.parametrize(
    "task_id",
    [
        "phase-boundary",
        "agent-lifecycle",
        "verifier-lifecycle",
        "sidecar-reachability",
        "hello-mcp",
    ],
)
def test_harbor_phase_behavior(graded: dict[str, Run], task_id: str) -> None:
    run = graded[task_id]
    evaluation = run.evaluation
    detail = evaluation.get("content") or ""
    info = evaluation.get("info") or {}
    detail = "\n".join(
        filter(
            None,
            (
                run.trace.content,
                run.trace.error,
                detail,
                info.get("stdout"),
                info.get("stderr"),
            ),
        )
    )
    assert run.reward == 1.0, f"{task_id} scored {run.reward}; the verifier reported:\n{detail}"

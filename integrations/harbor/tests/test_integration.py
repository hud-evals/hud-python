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
    from hud.capabilities import SSHClient
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
    job = await taskset.run(Oracle(solutions), runtime=DockerRuntime(), max_concurrent=1)
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
    ["phase-boundary", "agent-lifecycle", "verifier-lifecycle"],
)
def test_harbor_phase_behavior(graded: dict[str, Run], task_id: str) -> None:
    run = graded[task_id]
    evaluation = run.evaluation
    detail = evaluation.get("content") or ""
    info = evaluation.get("info") or {}
    detail = "\n".join(
        filter(None, (run.trace.content, detail, info.get("stdout"), info.get("stderr")))
    )
    assert run.reward == 1.0, f"{task_id} scored {run.reward}; the verifier reported:\n{detail}"

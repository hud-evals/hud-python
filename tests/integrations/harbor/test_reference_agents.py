"""``OracleAgent`` and ``NopAgent`` on an adapted Harbor task, run in Docker.

The adapter carries the fixture's ``solution/`` in its task row; the environment holds no
copy. The fixture under ``reference/oracle`` grades a pass only when its
``solution/solve.sh`` ran in the workdir, through the agent's egress, with
``[solution].env``. The oracle must get that pass; nop, which leaves the task undone,
must not.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from hud.agents import NopAgent, OracleAgent
from hud.integrations import harbor
from hud.runtime import DockerRuntime
from tests.integrations.harbor.cases import assemble_case

if TYPE_CHECKING:
    from hud import Taskset
    from hud.agents import Agent

FIXTURE = Path(__file__).parent / "reference" / "oracle"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.docker,
    pytest.mark.skipif(sys.platform == "win32", reason="adapted images are Linux containers"),
]


def adapted(sdk_wheel: Path, tmp_path: Path, *, solution: bool = True) -> Taskset:
    task = tmp_path / "harbor" / "oracle"
    assemble_case(FIXTURE, task)
    if not solution:
        shutil.rmtree(task / "solution")
    result = harbor.adapt(task, hud_requirement=str(sdk_wheel))
    assert result.failures == ()
    return result.taskset


@pytest.mark.parametrize(
    ("agent", "reward"),
    [pytest.param(OracleAgent(), 1.0, id="oracle"), pytest.param(NopAgent(), 0.0, id="nop")],
)
async def test_a_reference_agent_gets_the_reward_its_harbor_task_gives_it(
    agent: Agent, reward: float, sdk_wheel: Path, tmp_path: Path
) -> None:
    job = await adapted(sdk_wheel, tmp_path).run(agent, runtime=DockerRuntime())

    (trace,) = job.traces
    assert (trace.reward, trace.error) == (reward, None), trace.grade.info.get("stdout")


async def test_the_oracle_reports_a_harbor_task_without_a_solution(
    sdk_wheel: Path, tmp_path: Path
) -> None:
    job = await adapted(sdk_wheel, tmp_path, solution=False).run(
        OracleAgent(), runtime=DockerRuntime()
    )

    (trace,) = job.traces
    assert (trace.reward, trace.status) == (0.0, "error")
    assert "task 'oracle' has no reference solution" in (trace.error or "")

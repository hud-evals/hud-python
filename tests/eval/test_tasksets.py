"""Tasksets: the rows a source, a data file or the platform holds, and how ``Task.run`` and
``Taskset.run`` expand them into rollouts that are placed, bounded and collected into one
``Job``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentConfig
from hud.eval import (
    ComposeProject,
    LocalRuntime,
    RuntimeConfig,
    Task,
    Taskset,
)
from tests.conftest import eventually

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Run
    from tests.conftest import HudEnv
    from tests.fixtures.services import FakeServices


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_a_taskset_written_to_a_data_file_reads_back_row_for_row(
    suffix: str, tmp_path: Path
) -> None:
    project = tmp_path / "repo" / "project"
    project.mkdir(parents=True)
    (project / "compose.yaml").write_text("services:\n  main:\n    image: lab\n")
    rows = [
        Task(env="lab", id="add", args={"a": 2, "b": 3}),
        Task.model_validate(
            {
                "env": "lab",
                "id": "add",
                "args": {"a": 1, "b": 1},
                "slug": "one-plus-one",
                "validation": [{"name": "answer", "arguments": {"expected": 2}}],
                "agent_config": {"timeout_seconds": 30.0},
                "columns": {"difficulty": "easy"},
                "runtime_config": {
                    "image": "registry.test/trainer:1",
                    "resources": {
                        "cpu": 8,
                        "gpu": {"type": ["H100", "A100"], "count": 2},
                        "tpu": {"type": "v5e", "topology": "2x4"},
                    },
                    "limits": {"startup_timeout_s": 600},
                },
                "verifier": {"env": "judge", "id": "verify", "args": {"expected": "2"}},
            }
        ),
        Task(
            env="compose",
            id="solve",
            runtime_config=RuntimeConfig(
                compose=ComposeProject(document=project / "compose.yaml", service_access=True)
            ),
        ),
    ]

    written = Taskset("demo", rows).to_file(tmp_path / "repo" / "out" / f"tasks{suffix}")
    loaded = Taskset.from_file(written)
    (tmp_path / "repo").rename(tmp_path / "moved")
    *_, moved = Taskset.from_file(tmp_path / "moved" / "out" / f"tasks{suffix}")

    assert loaded.name == "tasks"
    assert [row.model_dump() for row in loaded] == [row.model_dump() for row in rows]
    assert moved.runtime_config is not None
    assert moved.runtime_config.compose is not None
    assert moved.runtime_config.compose.document == tmp_path / "moved" / "project" / "compose.yaml"


SOURCE = """
from hud import Environment, Taskset

env = Environment("lab")


@env.template()
async def solve(n: int):
    yield "go"
    yield 1.0


single = solve(n=1)
several = [solve(n=2), solve(n=3)]
collected = Taskset("inner", [solve(n=4)])
_private = solve(n=5)
"""


def test_a_python_source_holds_the_tasks_and_tasksets_it_defines(tmp_path: Path) -> None:
    (tmp_path / "tasks.py").write_text(SOURCE)

    loaded = Taskset.from_file(tmp_path / "tasks.py")

    assert loaded.name == "tasks"
    assert sorted(row.args["n"] for row in loaded) == [1, 2, 3, 4]


def test_a_directory_holds_the_tasks_of_every_source_in_it(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    root.mkdir()
    (root / "first.py").write_text(SOURCE)
    (root / "second.py").write_text(SOURCE.replace("n=", "n=1"))
    (root / "conftest.py").write_text("raise RuntimeError('not a task source')\n")

    loaded = Taskset.from_file(root)

    assert loaded.name == "suite"
    assert len(loaded) == 8


@pytest.mark.parametrize("reference", ["SheetBench-50", "eeeeeeee-0000-4000-8000-000000000001"])
def test_a_platform_taskset_loads_by_name_or_id(
    reference: str, services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    taskset_id = "eeeeeeee-0000-4000-8000-000000000001"
    services.route(
        "api",
        "GET",
        "/v2/tasksets/by-name/SheetBench-50",
        json={"taskset_id": taskset_id, "name": "SheetBench-50"},
    )
    services.route(
        "api",
        "GET",
        f"/v2/tasksets/{taskset_id}/export",
        json={
            "name": "SheetBench-50",
            "tasks": [
                {"name": "sum-a", "env": "sheets", "scenario": "sum", "args": {"cell": "A1"}},
                {"name": "sum-b", "env": "sheets", "scenario": "sum", "columns": {"hard": True}},
            ],
        },
    )

    loaded = Taskset.from_api(reference)

    assert (loaded.name, loaded.taskset_id) == ("SheetBench-50", taskset_id)
    assert [(row.slug, row.env, row.id, row.args) for row in loaded] == [
        ("sum-a", "sheets", "sum", {"cell": "A1"}),
        ("sum-b", "sheets", "sum", {}),
    ]


async def test_each_row_runs_group_times_in_task_major_order_under_one_job() -> None:
    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        answer = yield f"{a}+{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    class Adds(Agent):
        async def __call__(self, run: Run) -> None:
            a, b = run.prompt_text.split("+")
            run.trace.content = str(int(a) + int(b))

    first, second = add(a=1, b=2), add(a=3, b=4)
    job = await Taskset("sums", [first, second]).run(Adds(), group=3)

    assert [run.slug for run in job.runs] == [first.slug] * 3 + [second.slug] * 3
    assert [run.reward for run in job.runs] == [1.0] * 6
    assert {run.job_id for run in job.runs} == {job.id}
    assert [len(job.results[row.slug]) for row in (first, second)] == [3, 3]
    groups = [{run.group_id for run in job.results[row.slug]} for row in (first, second)]
    assert [len(group) for group in groups] == [1, 1] and groups[0] != groups[1]


@pytest.mark.parametrize(("max_concurrent", "peak"), [(None, 6), (2, 2)])
async def test_every_rollout_gets_its_own_env_and_max_concurrent_bounds_them(
    max_concurrent: int | None, peak: int
) -> None:
    started = []

    def build(task: Task) -> Environment:
        env = Environment("sums")

        @env.initialize
        async def up() -> None:
            started.append(task.slug)

        @env.template()
        async def add(a: int, b: int):
            yield f"{a}+{b}"
            yield 1.0

        return env

    live = [0]
    seen = []

    class Overlaps(Agent):
        async def __call__(self, run: Run) -> None:
            live[0] += 1
            await eventually(lambda: live[0] >= peak)
            await asyncio.sleep(0.05)
            seen.append(live[0])
            live[0] -= 1

    rows = [Task(env="sums", id="add", args={"a": a, "b": 1}) for a in (1, 2)]
    await Taskset("sums", rows).run(
        Overlaps(), runtime=LocalRuntime(build), group=3, max_concurrent=max_concurrent
    )

    assert (len(started), max(seen)) == (6, peak)


async def test_rollouts_on_one_live_environment_take_turns() -> None:
    env = Environment("shared")
    live = []

    @env.template()
    async def wait():
        live.append(1)
        assert sum(live) == 1
        yield "go"
        live.append(-1)
        yield 1.0

    class Pauses(Agent):
        async def __call__(self, run: Run) -> None:
            await asyncio.sleep(0.05)

    job = await wait().run(Pauses(), group=3)

    assert [run.reward for run in job.runs] == [1.0] * 3


async def test_a_jobs_reward_averages_its_graded_runs_and_its_errors_are_the_rest() -> None:
    env = Environment("mixed")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    class AnswersThenWaits(Agent):
        async def __call__(self, run: Run) -> None:
            run.trace.content = "yes"
            await asyncio.sleep(30)

    rows = [
        Task(env="mixed", id="check", slug="timed-out-after-answering"),
        Task(env="mixed", id="grade_raises", slug="grader-raised"),
    ]
    agent = AnswersThenWaits(AgentConfig(timeout_seconds=0.3))
    job = await Taskset("mixed", rows).run(agent, runtime=LocalRuntime(env))

    assert {run.slug: run.reward for run in job.runs} == {
        "timed-out-after-answering": 1.0,
        "grader-raised": 0.0,
    }
    assert ([run.slug for run in job.errors], job.reward) == (["grader-raised"], 1.0)


async def test_cancelling_a_run_cancels_every_rollout_and_stops_their_envs() -> None:
    events = []

    def build(task: Task) -> Environment:
        env = Environment("sums")

        @env.shutdown
        async def down() -> None:
            events.append("stopped")

        @env.template()
        async def add(a: int, b: int):
            answer = yield f"{a}+{b}"
            events.append(f"graded {answer}")
            yield 1.0

        return env

    prompted = []

    class Waits(Agent):
        async def __call__(self, run: Run) -> None:
            prompted.append(run.prompt_text)
            await asyncio.sleep(30)

    rows = [Task(env="sums", id="add", args={"a": a, "b": 1}) for a in (1, 2)]
    pending = asyncio.create_task(
        Taskset("sums", rows).run(Waits(), runtime=LocalRuntime(build), group=2)
    )
    await eventually(lambda: len(prompted) == 4)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    await eventually(lambda: events == ["stopped"] * 4)

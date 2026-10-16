"""``hud task``: list a source's tasks, start one for its prompt, grade an answer."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.clients import connect
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from tests.harness import Hud

ENV_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(target: str = "answer"):
    answer = yield f"Say {target}."
    yield 1.0 if answer == target else 0.0
"""
ROWS = [
    {"env": "example", "id": "solve", "slug": "say-answer"},
    {"env": "example", "id": "solve", "slug": "say-hello", "args": {"target": "hello"}},
]


@pytest.fixture
def project(hud: Hud) -> Any:
    """An env with a Python task source (``tasks.py``) and authored rows (``tasks.json``)."""
    (hud.cwd / "env.py").write_text(ENV_PY)
    (hud.cwd / "tasks.py").write_text(
        'from env import solve\ntasks = [solve(), solve(target="hello")]\n'
    )
    (hud.cwd / "tasks.json").write_text(json.dumps(ROWS))
    return hud.cwd


@pytest.mark.parametrize(
    ("source", "slugs"),
    [("tasks.json", ["say-answer", "say-hello"]), ("tasks.py", ["solve", "solve-8da35614"])],
)
def test_task_list_quiet_prints_each_tasks_slug(
    hud: Hud, project: Any, source: str, slugs: list[str]
) -> None:
    result = hud("task", "list", "--source", source, "--quiet")

    assert (result.exit_code, result.stdout.split()) == (0, slugs), result


def test_task_list_json_is_one_entry_per_task(hud: Hud, project: Any) -> None:
    result = hud("task", "list", "--source", "tasks.json", "--json")

    assert result.exit_code == 0, result
    assert result.json == [
        {"slug": "say-answer", "id": "solve", "args": {}},
        {"slug": "say-hello", "id": "solve", "args": {"target": "hello"}},
    ]


@pytest.mark.parametrize(
    ("argv", "stdout"),
    [
        (["start", "say-hello", "--source", "tasks.json"], "Say hello.\n"),
        (["start", "1", "--source", "tasks.py"], "Say hello.\n"),
        (["grade", "say-hello", "--source", "tasks.json", "--answer", "nope"], "0.0\n"),
        (["grade", "say-answer", "-s", "tasks.json", "--answer-file", "-"], "1.0\n"),
    ],
)
def test_a_task_spawned_from_its_source_starts_and_grades(
    hud: Hud, project: Any, argv: list[str], stdout: str
) -> None:
    (project / "answer.txt").write_text("answer")

    result = hud("task", *argv, input="answer")

    assert (result.exit_code, result.stdout) == (0, stdout), result


def test_task_start_json_is_the_start_frame(hud: Hud, project: Any) -> None:
    result = hud("task", "start", "say-answer", "--source", "tasks.json", "--json")

    assert result.exit_code == 0, result
    assert result.json["prompt"] == "Say answer."


@pytest.mark.parametrize(
    ("argv", "exit_code", "error"),
    [
        pytest.param(["start", "solve", "--source", "tasks.json"], 2, "usage", id="ambiguous"),
        pytest.param(["start", "missing", "--source", "tasks.json"], 1, "not_found", id="unknown"),
        pytest.param(["grade", "solve", "--args", "{bad"], 2, "usage", id="args-not-json"),
    ],
)
def test_task_refusals_are_error_documents(
    hud: Hud, project: Any, argv: list[str], exit_code: int, error: str
) -> None:
    (project / "empty.json").write_text("[]")

    result = hud("task", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result


# ─── attached to a served environment ───────────────────────────────────


def counting_env(starts: list[str], *, failing: bool = False) -> Environment:
    env = Environment("example")

    @env.template(id="solve")
    async def solve(target: str = "answer"):
        starts.append(target)
        answer = yield f"Say {target}."
        if failing:
            raise ValueError("grader failed")
        yield 1.0 if answer == target else 0.0

    return env


@pytest.mark.parametrize(
    ("parked", "failing", "starts", "outcome"),
    [
        pytest.param(0, False, 1, {"score": 1.0}, id="starts-when-nothing-is-parked"),
        pytest.param(1, False, 1, {"score": 1.0}, id="resumes-a-parked-session"),
        pytest.param(1, True, 1, {"error": "failure"}, id="grader-error"),
        pytest.param(2, False, 2, {"error": "failure"}, id="ambiguous-parked-sessions"),
    ],
)
async def test_grade_resumes_a_started_task_on_a_served_env(
    hud: Hud, parked: int, failing: bool, starts: int, outcome: dict[str, Any]
) -> None:
    started: list[str] = []
    async with LocalRuntime(counting_env(started, failing=failing))(
        Task(env="example", id="solve")
    ) as runtime:
        for _ in range(parked):
            async with connect(runtime) as client:
                await client.start_task("solve", {})
        result = await asyncio.to_thread(
            hud, "task", "grade", "solve", "--url", runtime.url, "--answer", "answer", "--json"
        )

    assert result.exit_code == ("error" in outcome), result
    assert len(started) == starts
    assert {key: result.json[key] for key in outcome} == outcome


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (["--source", "tasks.json"], "hello"),
        (["--args", '{"target": "hello"}'], "hello"),
    ],
)
async def test_a_url_runs_the_task_a_source_names_or_the_raw_id(
    hud: Hud, project: Any, argv: list[str], target: str
) -> None:
    started: list[str] = []
    task = "say-hello" if "--source" in argv else "solve"
    async with LocalRuntime(counting_env(started))(Task(env="example", id="solve")) as runtime:
        result = await asyncio.to_thread(hud, "task", "start", task, "--url", runtime.url, *argv)

    assert (result.exit_code, result.stdout) == (0, f"Say {target}.\n"), result
    assert started == [target]

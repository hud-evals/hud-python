"""``hud task``: list a source's tasks, start one for its prompt, grade an answer."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from hud import Environment
from hud.clients import connect
from hud.eval import LocalRuntime, Task
from tests.harness import scrub

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
    ("argv", "stdout"),
    [
        (["--source", "tasks.json"], 'say-answer\tsolve\nsay-hello\tsolve {"target": "hello"}\n'),
        (["--source", "tasks.json", "--quiet"], "say-answer\nsay-hello\n"),
        (
            ["-s", "tasks.py"],
            snapshot("""\
solve	solve
solve-8da35614	solve {"target": "hello"}
"""),
        ),
    ],
)
def test_task_list_prints_slug_id_and_args(
    hud: Hud, project: Any, argv: list[str], stdout: str
) -> None:
    result = hud("task", "list", *argv)

    assert (result.exit_code, result.stdout) == (0, stdout)


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
        (["grade", "say-hello", "--source", "tasks.json", "--answer", "hello"], "1.0\n"),
        (["grade", "say-hello", "--source", "tasks.json", "--answer", "nope"], "0.0\n"),
        (["grade", "0", "-s", "tasks.py", "--answer-file", "answer.txt"], "1.0\n"),
        (["grade", "say-answer", "-s", "tasks.json", "--answer-file", "-"], "1.0\n"),
        (["grade", "0", "-s", "tasks.py", "--args", '{"target": "x"}', "--answer", "x"], "1.0\n"),
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
    ("argv", "exit_code", "document"),
    [
        (
            ["start", "solve", "--source", "tasks.json"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Ambiguous task 'solve'; use a unique slug shown by hud task list.",
                }
            ),
        ),
        (
            ["start", "missing", "--source", "tasks.json"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No task matching 'missing' (available: solve)",
                    "input": {"task": "missing", "source": "tasks.json"},
                    "suggestion": "Run 'hud task list' to see available slugs.",
                }
            ),
        ),
        (
            ["grade", "solve", "--args", "[]"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "--args must be a JSON object",
                    "input": {"args": "[]"},
                }
            ),
        ),
        (
            ["grade", "solve", "--args", "{bad"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "--args must be valid JSON: Expecting property name enclosed in double "
                        "quotes: line 1 column 2 (char 1)"
                    ),
                    "input": {"args": "{bad"},
                    "suggestion": 'Pass a JSON object, e.g. --args \'{"key": "value"}\'.',
                }
            ),
        ),
        (
            ["grade", "say-answer", "-s", "tasks.json", "--answer-file", "no.txt"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "File not found: no.txt",
                    "input": {"path": "no.txt"},
                    "suggestion": "Check the path, or pass - to read from stdin.",
                }
            ),
        ),
        (
            ["start", "solve", "--source", "empty.json"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No tasks found in empty.json",
                    "input": {"source": "empty.json"},
                }
            ),
        ),
    ],
)
def test_task_errors_are_documents(
    hud: Hud, project: Any, argv: list[str], exit_code: int, document: dict[str, Any]
) -> None:
    (project / "empty.json").write_text("[]")

    result = hud("task", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert json.loads(scrub(result.stdout, project)) == document


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
    ("parked", "failing", "starts", "document"),
    [
        pytest.param(0, False, 1, {"score": 1.0}, id="starts-when-nothing-is-parked"),
        pytest.param(1, False, 1, {"score": 1.0}, id="resumes-a-parked-session"),
        pytest.param(
            1,
            True,
            1,
            {"error": "failure", "message": "hud rpc error -32000: grader failed"},
            id="grader-error",
        ),
        pytest.param(
            2,
            False,
            2,
            {
                "error": "failure",
                "message": IsStr(
                    regex=r"hud rpc error -32600: 2 parked sessions \(sess-\w+, sess-\w+\); "
                    r"resume one by sending hello with its session_id"
                ),
            },
            id="ambiguous-parked-sessions",
        ),
    ],
)
async def test_grade_resumes_a_started_task_on_a_served_env(
    hud: Hud, parked: int, failing: bool, starts: int, document: dict[str, Any]
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

    assert result.exit_code == ("error" in document), result
    assert len(started) == starts
    assert {key: result.json[key] for key in document} == document


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (["--source", "tasks.json"], "hello"),
        (["--source", "tasks.json", "--args", "{}"], "answer"),
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

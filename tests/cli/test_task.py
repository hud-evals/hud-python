"""``hud task``: start a source's task for its prompt and grade an answer, locally or on an
environment ``hud serve`` or a runtime serves."""

from __future__ import annotations

import asyncio
import json
import signal
import socket
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from tests.serving import serving

if TYPE_CHECKING:
    from tests.conftest import Hud

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
        (["start", "say-hello", "--source", "tasks.json"], "Say hello.\n"),
        (["start", "1", "--source", "tasks.py"], "Say hello.\n"),
        (["grade", "say-answer", "-s", "tasks.json", "--answer-file", "-"], "1.0\n"),
    ],
)
def test_a_task_spawned_from_its_source_starts_and_grades(
    hud: Hud, project: Any, argv: list[str], stdout: str
) -> None:
    result = hud("task", *argv, input="answer")

    assert (result.exit_code, result.stdout) == (0, stdout), result


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
    "leased",
    [
        pytest.param(False, id="runs-the-task-itself-without-a-session"),
        pytest.param(True, id="attaches-to-a-leased-session"),
    ],
)
async def test_grade_attaches_to_a_started_session_or_runs_the_task_itself(
    hud: Hud, leased: bool
) -> None:
    started: list[str] = []
    async with serving(counting_env(started)) as runtime:
        session: list[str] = []
        if leased:
            async with runtime.connect(lease=60) as client:
                await client.start_task("solve", {})
                session = ["--session", client.session_id]
        result = await asyncio.to_thread(
            hud,
            "task",
            "grade",
            "solve",
            "--url",
            runtime.url,
            *session,
            "--answer",
            "answer",
            "--json",
        )

    assert result.exit_code == 0, result
    assert (len(started), result.json["score"]) == (1, 1.0)


def test_a_task_starts_on_the_env_hud_serve_serves_until_interrupted(
    hud: Hud, project: Any
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = hud.start("serve", "env.py", "--port", str(port))
    try:
        url = f"tcp://127.0.0.1:{port}"
        result = hud("task", "start", "say-hello", "--source", "tasks.json", "--url", url)
    finally:
        server.send_signal(signal.SIGINT)
        server.communicate(timeout=30)

    assert (result.exit_code, result.stdout) == (0, "Say hello.\n"), result
    assert server.returncode == 0

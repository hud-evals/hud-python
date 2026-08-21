"""Behavioral coverage for deterministic ``hud task`` checks."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

import hud.clients as hud_clients
import hud.eval as hud_eval
from hud.cli import app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_loaded_env_module() -> Iterator[None]:
    sys.modules.pop("env", None)
    yield
    sys.modules.pop("env", None)


def _write_task_source(
    tmp_path: Path,
    *,
    score: float,
    grade_expression: str | None = None,
) -> Path:
    grade = grade_expression or repr(score)
    (tmp_path / "env.py").write_text(
        "from hud import Environment\n\n"
        "from hud.graders import EvaluationResult\n\n"
        'env = Environment("checks")\n\n'
        '@env.template(id="solve")\n'
        "async def solve(case: str):\n"
        '    yield f"solve {case}"\n'
        f"    yield {grade}\n",
        encoding="utf-8",
    )
    tasks = tmp_path / "tasks.py"
    tasks.write_text(
        'from env import solve\n\ntask = solve(case="demo")\n',
        encoding="utf-8",
    )
    return tasks


def test_dry_run_accepts_a_low_but_valid_reward(tmp_path: Path) -> None:
    tasks = _write_task_source(tmp_path, score=0.0)

    result = CliRunner().invoke(
        app,
        ["task", "grade", "solve", "--source", str(tasks), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "[pass] env" in result.output
    assert "[pass] task" in result.output
    assert "[pass] grader" in result.output
    assert "[pass] reward" in result.output
    assert "score 0 is valid" in result.output
    assert "result: PASS" in result.output


def test_dry_run_rejects_an_out_of_range_reward(tmp_path: Path) -> None:
    tasks = _write_task_source(tmp_path, score=2.0)

    result = CliRunner().invoke(
        app,
        ["task", "grade", "solve", "--source", str(tasks), "--dry-run"],
    )

    assert result.exit_code == 1, result.output
    assert "[pass] grader" in result.output
    assert "[fail] reward" in result.output
    assert "within [0, 1]" in result.output
    assert "result: FAIL" in result.output


def test_grade_rejects_an_out_of_range_reward(tmp_path: Path) -> None:
    tasks = _write_task_source(tmp_path, score=2.0)

    result = CliRunner().invoke(
        app,
        ["task", "grade", "task/solve", "--source", str(tasks)],
    )

    assert result.exit_code == 1, result.output
    assert "within [0, 1]" in result.output


def test_dry_run_reports_grader_errors_separately(tmp_path: Path) -> None:
    tasks = _write_task_source(
        tmp_path,
        score=0.0,
        grade_expression=(
            'EvaluationResult(reward=0.0, isError=True, content="grader dependency failed")'
        ),
    )

    result = CliRunner().invoke(
        app,
        ["task", "grade", "solve", "--source", str(tasks), "--dry-run"],
    )

    assert result.exit_code == 1, result.output
    assert "[fail] grader" in result.output
    assert "grader dependency failed" in result.output
    assert "[skip] reward" in result.output


def test_start_resolves_a_task_only_source_through_its_sibling_env(tmp_path: Path) -> None:
    tasks = _write_task_source(tmp_path, score=1.0)

    result = CliRunner().invoke(
        app,
        ["task", "start", "task/solve", "--source", str(tasks)],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "solve demo"


def test_start_reports_task_missing_from_live_environment(tmp_path: Path) -> None:
    tasks = _write_task_source(tmp_path, score=1.0)
    tasks.write_text(
        'from hud import Task\n\ntask = Task(env="checks", id="missing")\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["task", "start", "missing", "--source", str(tasks)],
    )

    assert result.exit_code == 1
    assert "task 'missing' is not exposed by the environment (solve)" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://example.com", "--url must use the tcp:// control-channel scheme"),
        ("tcp://localhost:not-a-port", "--url has an invalid port"),
    ],
)
def test_list_rejects_invalid_control_channel_url(url: str, message: str) -> None:
    result = CliRunner().invoke(
        app,
        ["task", "list", "--url", url],
    )

    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output


def test_dry_run_times_out_task_start(monkeypatch: pytest.MonkeyPatch) -> None:
    class Placement:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class Runtime:
        def __call__(self, task: object) -> Placement:
            return Placement()

    class Client:
        manifest = SimpleNamespace(server_info=SimpleNamespace(name="checks"))

        def __init__(self) -> None:
            self.cancelled = False

        async def list_tasks(self) -> list[dict[str, str]]:
            return [{"id": "solve"}]

        async def start_task(self, task_id: str, args: dict[str, object]) -> None:
            await asyncio.sleep(1)

        async def cancel(self) -> None:
            self.cancelled = True

    client = Client()

    @contextlib.asynccontextmanager
    async def connect(*args: object, **kwargs: object) -> AsyncIterator[Client]:
        yield client

    monkeypatch.setattr(hud_eval, "HUDRuntime", Runtime)
    monkeypatch.setattr(hud_clients, "connect", connect)

    result = CliRunner().invoke(
        app,
        [
            "task",
            "grade",
            "solve",
            "--env",
            "checks",
            "--dry-run",
            "--timeout",
            "0.1",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "[fail] task" in result.output
    assert "tasks.start timed out after 0.1s" in result.output
    assert "[skip] grader" in result.output
    assert client.cancelled


def test_dry_run_times_out_environment_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowPlacement:
        async def __aenter__(self) -> object:
            await asyncio.sleep(1)
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class SlowRuntime:
        def __call__(self, task: object) -> SlowPlacement:
            return SlowPlacement()

    monkeypatch.setattr(hud_eval, "HUDRuntime", SlowRuntime)

    result = CliRunner().invoke(
        app,
        [
            "task",
            "grade",
            "solve",
            "--env",
            "checks",
            "--dry-run",
            "--timeout",
            "0.1",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "[fail] env" in result.output
    assert "environment startup timed out after 0.1s" in result.output
    assert "[skip] task" in result.output

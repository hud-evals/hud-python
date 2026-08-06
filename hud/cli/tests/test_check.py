"""Production contract tests for ``hud check``."""

from __future__ import annotations

import asyncio
import json
import textwrap
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from hud.cli import app
from hud.cli.check import (
    CheckCriterion,
    CheckRequest,
    TaskCheckReport,
    _criteria_template,
    _redact_evidence,
    _run_agent,
    _run_check,
    _run_direct,
)
from hud.eval import Runtime, SubprocessRuntime, Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

runner = CliRunner()


def _report(
    *,
    outcome: str = "passed",
    reward: float | None = 1.0,
    error: str | None = None,
) -> TaskCheckReport:
    status = "passed" if outcome == "passed" else "failed" if outcome == "failed" else "error"
    return TaskCheckReport(
        schema_version="hud.task-check.v1",
        outcome=outcome,
        mode="oracle",
        task_id="env:task",
        runtime="local",
        reward=reward,
        min_reward=1.0,
        trace_id="00000000-0000-4000-a000-000000000001",
        criteria=[
            CheckCriterion(
                name="oracle_or_agent_reward",
                status=status,
                detail=error or "reward evaluated",
            ),
        ],
        error=error,
        duration_seconds=0.1,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["check", "task"],
        ["check", "task", "--answer", "x", "--start-only"],
        ["check", "task", "--answer", "x", "--agent", "claude"],
        ["check", "task", "--answer", "x", "--answer-file", "answer.txt"],
    ],
)
def test_check_requires_exactly_one_proof_strategy(args: list[str]) -> None:
    with patch("hud.cli.check._run_check", AsyncMock()) as execute:
        result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "exactly one proof strategy" in result.output.lower()
    execute.assert_not_awaited()


def test_check_rejects_remote_direct_oracle() -> None:
    with patch("hud.cli.check._run_check", AsyncMock()) as execute:
        result = runner.invoke(app, ["check", "task", "--answer", "x", "--remote"])

    assert result.exit_code == 2
    assert "--remote requires --agent" in result.output
    execute.assert_not_awaited()


def test_check_rejects_conflicting_placement() -> None:
    with patch("hud.cli.check._run_check", AsyncMock()) as execute:
        result = runner.invoke(
            app,
            [
                "check",
                "task",
                "--agent",
                "claude",
                "--url",
                "tcp://localhost:8765",
                "--runtime",
                "hud",
            ],
        )

    assert result.exit_code == 2
    assert "choose only one placement" in result.output.lower()
    execute.assert_not_awaited()


def test_check_rejects_agent_only_options_for_direct_proof() -> None:
    with patch("hud.cli.check._run_check", AsyncMock()) as execute:
        result = runner.invoke(
            app,
            ["check", "task", "--answer", "x", "--model", "claude-sonnet"],
        )

    assert result.exit_code == 2
    assert "require --agent" in result.output
    execute.assert_not_awaited()


@pytest.mark.parametrize(
    ("report", "expected_code"),
    [
        (_report(), 0),
        (_report(outcome="failed", reward=0.2), 1),
        (_report(outcome="error", reward=None, error="grader unavailable"), 3),
    ],
)
def test_check_exit_codes_follow_report_outcome(
    report: TaskCheckReport,
    expected_code: int,
) -> None:
    with patch("hud.cli.check._run_check", AsyncMock(return_value=report)):
        result = runner.invoke(app, ["check", "task", "--answer", "solution"])

    assert result.exit_code == expected_code
    assert "oracle_or_agent_reward" in result.output


def test_check_json_emits_stable_versioned_contract() -> None:
    report = _report()
    with patch("hud.cli.check._run_check", AsyncMock(return_value=report)):
        result = runner.invoke(
            app,
            ["check", "task", "--answer", "solution", "--json"],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "hud.task-check.v1"
    assert payload["outcome"] == "passed"
    assert payload["criteria"][0]["name"] == "oracle_or_agent_reward"


def test_check_reads_answer_file_and_forwards_runtime_options(tmp_path: Path) -> None:
    answer = tmp_path / "answer.txt"
    answer.write_text("file answer", encoding="utf-8")
    execute = AsyncMock(return_value=_report())

    with patch("hud.cli.check._run_check", execute):
        result = runner.invoke(
            app,
            [
                "check",
                "task",
                "--source",
                "env.py",
                "--answer-file",
                str(answer),
                "--runtime",
                "hud",
                "--min-reward",
                "0.75",
                "--timeout",
                "30",
                "--startup-timeout",
                "5",
            ],
        )

    assert result.exit_code == 0
    request = execute.await_args.args[0]
    assert request.answer == "file answer"
    assert request.runtime == "hud"
    assert request.min_reward == 0.75
    assert request.timeout == 30
    assert request.startup_timeout == 5


def test_check_remote_agent_uses_hosted_strategy() -> None:
    execute = AsyncMock(return_value=_report())
    with patch("hud.cli.check._run_check", execute):
        result = runner.invoke(
            app,
            [
                "check",
                "task",
                "--source",
                "env.py",
                "--agent",
                "claude",
                "--model",
                "claude-sonnet",
                "--remote",
            ],
        )

    assert result.exit_code == 0
    request = execute.await_args.args[0]
    assert request.agent == "claude"
    assert request.model == "claude-sonnet"
    assert request.remote is True


def test_evidence_is_redacted_and_bounded() -> None:
    evidence = {
        "api_key": "secret",
        "nested": {"token": "secret", "safe": "x" * 20_000},
    }

    safe = _redact_evidence(evidence, max_chars=120)
    serialized = json.dumps(safe)

    assert "secret" not in serialized
    assert len(serialized) <= 180
    assert "truncated" in serialized


@pytest.mark.parametrize(("answer", "expected_code"), [("3", 0), ("wrong", 1)])
def test_check_runs_real_local_task_lifecycle(
    tmp_path: Path,
    answer: str,
    expected_code: int,
) -> None:
    source = tmp_path / "env.py"
    source.write_text(
        textwrap.dedent(
            """
            from hud import Environment

            env = Environment("sums")

            @env.template(id="add")
            async def add(a: int, b: int):
                answer = yield f"add:{a}:{b}"
                yield 1.0 if answer == str(a + b) else 0.0

            task = add(a=1, b=2)
            """
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "check",
            "add",
            "--source",
            str(source),
            "--answer",
            answer,
            "--json",
            "--timeout",
            "15",
        ],
    )

    assert result.exit_code == expected_code, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "hud.task-check.v1"
    assert payload["reward"] == (1.0 if answer == "3" else 0.0)
    assert payload["criteria"][1]["status"] == "passed"
    assert payload["criteria"][2]["status"] == "passed"
    assert payload["criteria"][3]["status"] == "passed"


@pytest.mark.asyncio
async def test_check_attaches_to_a_served_environment(tmp_path: Path) -> None:
    source = tmp_path / "env.py"
    source.write_text(
        textwrap.dedent(
            """
            from hud import Environment

            env = Environment("sums")

            @env.template(id="add")
            async def add(a: int, b: int):
                answer = yield f"add:{a}:{b}"
                yield 1.0 if answer == str(a + b) else 0.0
            """
        ),
        encoding="utf-8",
    )
    task = Task(env="sums", id="add", args={"a": 2, "b": 3})

    async with SubprocessRuntime(source)(task) as runtime:
        result = await asyncio.to_thread(
            runner.invoke,
            app,
            [
                "check",
                "add",
                "--url",
                runtime.url,
                "--args",
                '{"a": 2, "b": 3}',
                "--answer",
                "5",
                "--json",
                "--timeout",
                "15",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runtime"] == runtime.url
    assert payload["reward"] == 1.0


class _Client:
    def __init__(
        self,
        *,
        reward: Any = 1.0,
        start_error: Exception | None = None,
        grade_error: Exception | None = None,
    ) -> None:
        self.reward = reward
        self.start_error = start_error
        self.grade_error = grade_error
        self.started: list[tuple[str, dict[str, Any]]] = []
        self.graded: list[dict[str, Any]] = []
        self.cancelled = 0

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.start_error is not None:
            raise self.start_error
        self.started.append((task_id, args))
        return {"prompt": "safe prompt", "api_key": "must-redact"}

    async def grade(self, answer: dict[str, Any]) -> dict[str, Any]:
        self.graded.append(answer)
        if self.grade_error is not None:
            raise self.grade_error
        return {"score": self.reward}

    async def cancel(self) -> None:
        self.cancelled += 1


@asynccontextmanager
async def _provided(value: Any) -> AsyncIterator[Any]:
    yield value


@pytest.mark.asyncio
async def test_startup_timeout_is_reported_separately_from_the_overall_timeout() -> None:
    task = Task(env="demo", id="demo:solve")

    @asynccontextmanager
    async def stalled_provider(_task: Task) -> AsyncIterator[Runtime]:
        await asyncio.Event().wait()
        yield Runtime("tcp://127.0.0.1:8765")

    request = CheckRequest(
        task=task.id,
        answer="42",
        startup_timeout=0.01,
        timeout=10,
    )
    with patch(
        "hud.cli.check._resolve",
        return_value=(task, stalled_provider, "local"),
    ):
        report = await _run_check(request)

    assert report.outcome == "error"
    assert report.error == "environment did not become ready within 0.01s"
    assert report.criteria[1].detail == report.error


@pytest.mark.asyncio
async def test_direct_oracle_uses_start_and_grade_lifecycle() -> None:
    client = _Client(reward=0.75)
    task = Task(env="demo", id="demo:solve", args={"seed": 3})
    runtime = Runtime("tcp://127.0.0.1:8765")
    provider = lambda _task: _provided(runtime)
    criteria = _criteria_template()

    with patch("hud.clients.connect", lambda _runtime: _provided(client)):
        reward, trace_id = await _run_direct(
            CheckRequest(task=task.id, answer="42"),
            task,
            provider,
            criteria,
        )

    assert reward == 0.75
    assert trace_id is None
    assert client.started == [(task.id, {"seed": 3})]
    assert client.graded == [{"answer": "42"}]
    assert criteria["environment_startup"].status == "passed"
    assert criteria["task_startup"].status == "passed"
    assert criteria["grader_execution"].status == "passed"
    assert "must-redact" not in json.dumps(criteria["task_startup"].evidence)


@pytest.mark.asyncio
async def test_start_only_never_invokes_grader() -> None:
    client = _Client()
    task = Task(env="demo", id="demo:solve")
    runtime = Runtime("tcp://127.0.0.1:8765")
    criteria = _criteria_template()

    with patch("hud.clients.connect", lambda _runtime: _provided(client)):
        reward, _ = await _run_direct(
            CheckRequest(task=task.id, start_only=True),
            task,
            lambda _task: _provided(runtime),
            criteria,
        )

    assert reward is None
    assert client.graded == []
    assert client.cancelled == 1
    assert criteria["grader_execution"].status == "skipped"
    assert criteria["oracle_or_agent_reward"].status == "skipped"


@pytest.mark.asyncio
async def test_direct_grading_failure_cancels_the_live_task_session() -> None:
    client = _Client(grade_error=RuntimeError("grader unavailable"))
    task = Task(env="demo", id="demo:solve")

    with (
        patch("hud.clients.connect", lambda _runtime: _provided(client)),
        pytest.raises(RuntimeError, match="grader unavailable"),
    ):
        await _run_direct(
            CheckRequest(task=task.id, answer={"value": "42"}),
            task,
            lambda _task: _provided(Runtime("tcp://127.0.0.1:8765")),
            _criteria_template(),
        )

    assert client.cancelled == 1


@pytest.mark.asyncio
async def test_direct_grader_requires_numeric_reward() -> None:
    client = _Client(reward="not-a-number")
    task = Task(env="demo", id="demo:solve")
    criteria = _criteria_template()

    with (
        patch("hud.clients.connect", lambda _runtime: _provided(client)),
        pytest.raises((TypeError, ValueError)),
    ):
        await _run_direct(
            CheckRequest(task=task.id, answer="42"),
            task,
            lambda _task: _provided(Runtime("tcp://127.0.0.1:8765")),
            criteria,
        )

    assert criteria["grader_execution"].status == "error"


@pytest.mark.asyncio
async def test_agent_rollout_preserves_trace_and_attributes_grader_failure() -> None:
    run = SimpleNamespace(
        trace=SimpleNamespace(
            status="error",
            error="[grading] connection reset",
            trace_id="00000000-0000-4000-a000-000000000010",
        ),
        grade=SimpleNamespace(is_error=True, content="connection reset"),
        reward=0.0,
    )
    criteria = _criteria_template()

    with patch("hud.eval.run.rollout", AsyncMock(return_value=run)):
        reward, trace_id = await _run_agent(
            CheckRequest(task="demo:solve", agent="claude"),
            Task(env="demo", id="demo:solve"),
            Runtime("tcp://127.0.0.1:8765"),
            object(),
            criteria,
        )

    assert reward is None
    assert trace_id == run.trace.trace_id
    assert criteria["environment_startup"].status == "passed"
    assert criteria["task_startup"].status == "passed"
    assert criteria["grader_execution"].status == "error"
    assert criteria["oracle_or_agent_reward"].status == "skipped"


@pytest.mark.asyncio
async def test_cancelled_agent_rollout_is_an_execution_error() -> None:
    run = SimpleNamespace(
        trace=SimpleNamespace(status="cancelled", error=None, trace_id="cancelled-trace"),
        grade=SimpleNamespace(is_error=False, content=None),
        reward=0.0,
    )
    criteria = _criteria_template()

    with patch("hud.eval.run.rollout", AsyncMock(return_value=run)):
        reward, _ = await _run_agent(
            CheckRequest(task="demo:solve", agent="claude"),
            Task(env="demo", id="demo:solve"),
            Runtime("tcp://127.0.0.1:8765"),
            object(),
            criteria,
        )

    assert reward is None
    assert criteria["oracle_or_agent_reward"].status == "error"
    assert "cancelled" in criteria["oracle_or_agent_reward"].detail


@pytest.mark.asyncio
async def test_agent_error_that_mentions_grading_stays_in_agent_criterion() -> None:
    run = SimpleNamespace(
        trace=SimpleNamespace(
            status="error",
            error="[agent loop] model refused the grading instruction",
            trace_id="agent-error",
        ),
        grade=SimpleNamespace(is_error=False, content=None),
        reward=0.0,
    )
    criteria = _criteria_template()

    with patch("hud.eval.run.rollout", AsyncMock(return_value=run)):
        await _run_agent(
            CheckRequest(task="demo:solve", agent="claude"),
            Task(env="demo", id="demo:solve"),
            Runtime("tcp://127.0.0.1:8765"),
            object(),
            criteria,
        )

    assert criteria["grader_execution"].status == "passed"
    assert criteria["oracle_or_agent_reward"].status == "error"


@pytest.mark.asyncio
async def test_hosted_check_without_platform_key_is_invalid_configuration() -> None:
    from hud.settings import settings

    with patch.object(settings, "api_key", None):
        report = await _run_check(
            CheckRequest(task="demo:solve", agent="claude", remote=True),
        )

    assert report.outcome == "error"
    assert report.error_kind == "input"
    assert report.criteria[0].name == "resolution"
    assert report.criteria[0].status == "error"
    assert "HUD_API_KEY" in report.criteria[0].detail

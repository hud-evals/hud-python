"""The rollout engine: ``rollout(task, agent)`` and its schedulers.

These drive the engine end-to-end through the real placement path: a pure-data
``Task`` row plus ``runtime=SubprocessRuntime(env_file)`` — a child process serves the env, the
engine connects over the wire, the agent answers, grading comes back. The
engine contract is a graded :class:`Run` with a trace id (always under a job —
there are no standalone traces), and failure isolation that never raises: a
pre-launch failure yields a synthesized ``Run.failed``; a mid-run failure
keeps the real run and its evidence. ``Task.run`` / ``Taskset.run`` schedule
the atom and return a :class:`Job`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import pytest

from hud.agents.base import Agent
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.environment import Environment
from hud.eval import Job, SubprocessRuntime, Task, Taskset
from hud.eval.run import Run, rollout
from hud.eval.runtime import _local

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.eval.runtime import Runtime
    from hud.eval.task import Task as TaskRow

_SUMS_ENV = """\
from hud import Environment

env = Environment("sums")


@env.template()
async def add(a: int, b: int):
    answer = yield f"add:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


@pytest.fixture(scope="module")
def env_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sums") / "env.py"
    path.write_text(textwrap.dedent(_SUMS_ENV), encoding="utf-8")
    return path


class _FnAgent(Agent):
    """Stateless agent: answers each run by applying ``fn`` to ``run.prompt``."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


class _SequencedCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class _FakeOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=_SequencedCompletions(responses))


def _chat_response(content: str, tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        refusal=None,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content},
    )
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    return SimpleNamespace(
        choices=[choice],
        model="fake-openai-compatible",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
    )


def _tool_call(name: str, arguments: str) -> Any:
    return SimpleNamespace(
        type="function",
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _add_task(a: int, b: int) -> Task:
    """A pure data row; the env it names is defined by the spawned file."""
    return Task(env="sums", id="add", args={"a": a, "b": b})


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


def _pid_status(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() or None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    status = _pid_status(pid)
    if status is None:
        return False
    return not status.startswith("Z")


async def _wait_for_pid_inactive(pid: int, max_wait: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait
    while loop.time() < deadline:
        if not _pid_is_running(pid):
            return True
        await asyncio.sleep(0.05)
    return not _pid_is_running(pid)


async def test_rollout_returns_graded_run_with_trace_id(env_file: Path) -> None:
    run = await rollout(_add_task(2, 3), _FnAgent(_solve_add), runtime=SubprocessRuntime(env_file))

    assert run.reward == 1.0
    assert run.trace.content == "5"
    assert run.trace_id is not None
    # No standalone traces: a bare rollout registers a single-run job itself.
    assert run.job_id is not None
    # The factual placement record: the runtime this run executed against.
    assert run.runtime is not None
    assert run.runtime.startswith("tcp://127.0.0.1:")


async def test_openai_compatible_write_reaches_workspace_grader(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    report = workspace / "REPORT.md"
    env = Environment("opencode_report")
    env.workspace(workspace, guest_path=str(workspace))

    @env.initialize
    async def seed() -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        report.unlink(missing_ok=True)

    @env.template()
    async def write_report():
        yield "Write PASS to REPORT.md."
        yield 1.0 if report.exists() and report.read_text().strip() == "PASS" else 0.0

    model_client = _FakeOpenAI(
        [
            _chat_response(
                "",
                [_tool_call("write", json.dumps({"filePath": str(report), "content": "PASS"}))],
            ),
            _chat_response("done"),
        ]
    )
    agent = OpenAIChatAgent(
        OpenAIChatConfig(model="qwen3.6-plus", model_client=model_client, max_steps=4)
    )

    run = await rollout(
        Task(env="opencode_report", id="write_report"),
        agent,
        runtime=lambda _task: _local(env),
    )

    assert run.reward == 1.0
    assert report.read_text() == "PASS"
    tools = model_client.chat.completions.requests[0]["extra_body"]["tools"]
    assert [tool["function"]["name"] for tool in tools] == [
        "bash",
        "read",
        "glob",
        "grep",
        "edit",
        "write",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group regression")
async def test_local_runtime_startup_failure_kills_spawned_children(tmp_path: Path) -> None:
    env_file = tmp_path / "env.py"
    env_file.write_text(
        textwrap.dedent(
            """
            import asyncio
            from pathlib import Path

            from hud import Environment

            env = Environment("leaky")


            @env.initialize
            async def start_child():
                proc = await asyncio.create_subprocess_exec(
                    "sleep",
                    "120",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                Path("child.pid").write_text(str(proc.pid), encoding="utf-8")
                raise RuntimeError("startup boom")
            """
        ),
        encoding="utf-8",
    )
    pid_file = tmp_path / "child.pid"
    pid: int | None = None

    try:
        with pytest.raises(RuntimeError, match="startup boom"):
            async with SubprocessRuntime(env_file, ready_timeout=30.0)(
                Task(env="leaky", id="noop")
            ):
                pass
        pid = int(pid_file.read_text())
        assert await _wait_for_pid_inactive(pid)
    finally:
        if pid is not None and _pid_is_running(pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


async def test_mid_run_failure_keeps_the_real_run_and_its_evidence(env_file: Path) -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("agent exploded")

    run = await rollout(_add_task(2, 3), _FnAgent(boom), runtime=SubprocessRuntime(env_file))

    assert run.trace.is_error
    assert "agent exploded" in (run.trace.error or "")
    assert run.trace_id is not None  # failed runs still key a trajectory
    # The session was live, so the receipt keeps the evidence: the prompt the
    # agent saw and the runtime the rollout executed against.
    assert run.prompt == "add:2:3"
    assert run.runtime is not None
    assert run.reward == 0.0  # graded best-effort, but the agent never answered → 0.0


class _AnswerThenBoomAgent(Agent):
    """Records a correct answer, then raises — a mid-run failure after the env
    already has a gradable answer in hand."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)
        raise RuntimeError("agent exploded after answering")


async def test_mid_run_failure_still_grades_best_effort(env_file: Path) -> None:
    # The agent answers correctly, then fails. The env is still alive, so the
    # run is graded best-effort: the reward is captured even though it errored.
    run = await rollout(
        _add_task(2, 3), _AnswerThenBoomAgent(_solve_add), runtime=SubprocessRuntime(env_file)
    )

    assert run.trace.is_error
    assert "agent exploded after answering" in (run.trace.error or "")
    assert run.reward == 1.0  # graded despite the failure
    assert run.trace.status == "error"  # the failure is preserved, not masked


class _SlowAgent(Agent):
    """Answers, then hangs — to exercise the agent-loop timeout."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)
        await asyncio.sleep(30)


async def test_agent_loop_timeout_grades_the_partial_trajectory() -> None:
    # In-process env (no subprocess spawn) so only the agent loop, not setup,
    # races the short deadline.
    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        answer = yield f"add:{a}:{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    run = await rollout(
        _add_task(2, 3),
        _SlowAgent(_solve_add),
        runtime=lambda _row: _local(env),
        rollout_timeout=0.5,
    )

    # The deadline fired mid-loop, but the run was live with an answer already
    # recorded, so it is graded rather than discarded as a zero-reward cancel.
    assert run.reward == 1.0
    assert run.trace.status == "completed"
    assert run.trace.stop_reason == "timeout"
    assert run.trace_id is not None


async def test_wedged_grade_cannot_stall_the_rollout() -> None:
    """A disconnected tunnel during grade must hit the teardown grace, not hang."""
    from hud.eval import run as run_mod

    env = Environment("sums")

    @env.template()
    async def add(a: int, b: int):
        answer = yield f"add:{a}:{b}"
        yield 1.0 if answer == str(a + b) else 0.0

    class _HangOnGrade:
        """Wraps a live client but never returns from grade()."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        async def grade(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            await asyncio.sleep(3600)
            return {}

        async def cancel(self) -> None:
            return None

    real_connect = run_mod.connect

    @asynccontextmanager
    async def _connect_hanging(addr: Any) -> AsyncIterator[Any]:
        async with real_connect(addr) as client:
            yield _HangOnGrade(client)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(run_mod, "connect", _connect_hanging)
    monkeypatch.setattr(run_mod, "_TEARDOWN_GRACE_S", 0.2)
    try:
        started = asyncio.get_running_loop().time()
        run = await rollout(
            _add_task(2, 3),
            _FnAgent(_solve_add),
            runtime=lambda _row: _local(env),
            rollout_timeout=0.3,
        )
        elapsed = asyncio.get_running_loop().time() - started
    finally:
        monkeypatch.undo()

    assert elapsed < 5.0  # must not wait on the hung grade
    assert run.trace.is_error
    assert run.trace.stop_reason == "timeout"
    assert "grading timed out" in (run.trace.error or "")


async def test_pre_launch_failure_yields_a_synthesized_failed_run() -> None:
    @asynccontextmanager
    async def broken_provider(task: TaskRow) -> AsyncIterator[Runtime]:
        raise RuntimeError("no substrate for you")
        yield  # pragma: no cover

    run = await rollout(_add_task(1, 1), _FnAgent(_solve_add), runtime=broken_provider)

    assert run.trace.is_error
    assert "no substrate for you" in (run.trace.error or "")
    assert run.trace_id is not None
    assert run.prompt is None  # nothing ever started
    assert run.runtime is None


async def test_provider_is_called_with_the_task_row_being_placed(env_file: Path) -> None:
    placed: list[str] = []

    def placer(task: TaskRow) -> Any:
        # The scheduler half of placement: the row is the request, so a
        # provider can size/route each substrate per task.
        placed.append(f"{task.env}/{task.id}:{task.args['a']}")
        return SubprocessRuntime(env_file)(task)

    run = await rollout(_add_task(2, 3), _FnAgent(_solve_add), runtime=placer)

    assert run.reward == 1.0
    assert placed == ["sums/add:2"]


async def test_task_run_schedules_a_single_task_job(env_file: Path) -> None:
    job = await _add_task(2, 3).run(_FnAgent(_solve_add), runtime=SubprocessRuntime(env_file))

    (run,) = job.runs
    assert job.reward == 1.0
    assert run.trace.content == "5"
    assert run.job_id == job.id  # the run's trace reports under the job


async def test_task_run_has_taskset_scheduling_semantics(env_file: Path) -> None:
    job = await _add_task(1, 2).run(
        _FnAgent(_solve_add), runtime=SubprocessRuntime(env_file), group=2, max_concurrent=1
    )

    assert job.group == 2
    assert [run.reward for run in job.runs] == [1.0, 1.0]
    # The group repeats one task, so they share a GRPO group id.
    assert len({run.group_id for run in job.runs}) == 1


async def test_open_job_spans_multiple_scheduler_calls(env_file: Path) -> None:
    session = await Job.start("session", group=2)
    provider = SubprocessRuntime(env_file)

    job1 = await _add_task(1, 1).run(_FnAgent(_solve_add), runtime=provider, job=session)
    job2 = await _add_task(2, 2).run(_FnAgent(_solve_add), runtime=provider, job=session)

    # Both calls accumulate into the one open job (group defaults to the job's).
    assert job1 is session
    assert job2 is session
    assert len(session.runs) == 4
    assert {run.job_id for run in session.runs} == {session.id}
    assert session.reward == 1.0


_TWO_ENVS = """\
from hud import Environment

alpha = Environment("alpha")
beta = Environment("beta")


@alpha.template()
async def add_a(a: int, b: int):
    answer = yield f"alpha:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0


@beta.template()
async def add_b(a: int, b: int):
    answer = yield f"beta:{a}:{b}"
    yield 1.0 if answer == str(a + b) else 0.0
"""


async def test_one_spawn_serves_each_rows_env_in_a_mixed_taskset(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    path = tmp_path_factory.mktemp("zoo") / "envs.py"
    path.write_text(_TWO_ENVS, encoding="utf-8")
    rows = [
        Task(env="alpha", id="add_a", args={"a": 1, "b": 2}),
        Task(env="beta", id="add_b", args={"a": 3, "b": 4}),
    ]

    # One provider, two envs: each acquisition serves the row it was called
    # with (the task ids only exist on their own env, so a misplacement
    # would fail the rollout).
    job = await Taskset("zoo", rows).run(_FnAgent(_solve_add), runtime=SubprocessRuntime(path))

    assert [run.reward for run in job.runs] == [1.0, 1.0]
    assert [run.prompt for run in job.runs] == ["alpha:1:2", "beta:3:4"]


async def test_rollout_threads_job_and_group_ids(env_file: Path) -> None:
    run = await rollout(
        _add_task(1, 1),
        _FnAgent(_solve_add),
        runtime=SubprocessRuntime(env_file),
        job_id="j1",
        group_id="g1",
    )

    assert run.reward == 1.0
    assert run.job_id == "j1"
    assert run.group_id == "g1"


# ─── Run prompt views (what agents consume) ───────────────────────────


def _run_with_prompt(prompt: Any) -> Run:
    run = Run(None, "t", {})
    run.prompt = prompt
    return run


def test_prompt_messages_wraps_plain_text_as_one_user_turn() -> None:
    (msg,) = _run_with_prompt("hello").prompt_messages
    assert msg.role == "user"
    assert isinstance(msg.content, mcp_types.TextContent)
    assert msg.content.text == "hello"


def test_prompt_messages_no_prompt_is_one_empty_user_turn() -> None:
    (msg,) = _run_with_prompt(None).prompt_messages
    assert isinstance(msg.content, mcp_types.TextContent)
    assert msg.content.text == ""


def test_prompt_messages_normalizes_chat_dicts_and_passes_through() -> None:
    existing = mcp_types.PromptMessage(
        role="assistant", content=mcp_types.TextContent(type="text", text="prior")
    )
    msgs = _run_with_prompt(
        [
            {"role": "user", "content": {"type": "text", "text": "hi"}},
            {"role": "system", "content": "be nice"},  # outside MCP vocab → user
            existing,
        ]
    ).prompt_messages
    assert [m.role for m in msgs] == ["user", "user", "assistant"]
    assert msgs[2] is existing


def test_prompt_text_flattens_text_turns_and_drops_non_text() -> None:
    image = mcp_types.PromptMessage(
        role="user",
        content=mcp_types.ImageContent(type="image", data="aGk=", mimeType="image/png"),
    )
    run = _run_with_prompt([{"role": "user", "content": "first"}, image, "second"])
    assert run.prompt_text == "first\n\nsecond"

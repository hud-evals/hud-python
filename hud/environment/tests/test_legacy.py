"""Integration tests for the v5->v6 env-authoring compatibility layer.

These exercise real environments end-to-end over the wire (the ``served``
harness brings up a loopback substrate + ``HudClient``) and through
``Taskset``, rather than poking internals: concurrency, error isolation, typed
returns, message-list prompts, cancellation, unknown tasks, and on-serve
capability synthesis. The envs are built inline (no source file to spawn), so
placement uses the in-process ``_local`` provider.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from pydantic import BaseModel

from hud.agents.base import Agent
from hud.agents.types import AgentAnswer
from hud.clients import HudProtocolError
from hud.environment import Environment, Workspace
from hud.environment.legacy import _classify_tool
from hud.eval import Run, Taskset
from hud.eval.runtime import _local

from .conftest import served


def _silence_deprecation() -> None:
    warnings.simplefilter("ignore", DeprecationWarning)


class _FnAgent(Agent):
    """Stateless agent: answers each run by applying ``fn`` to ``run.prompt``.

    One instance drives many concurrent rollouts (the contract ``Taskset`` relies on).
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    async def __call__(self, run: Any) -> None:
        run.trace.content = self._fn(run.prompt)


def _sum_env() -> Environment:
    env = Environment("sums")
    with warnings.catch_warnings():
        _silence_deprecation()

        @env.scenario("add")
        async def add(a: int, b: int):
            answer = yield f"add:{a}:{b}"
            yield 1.0 if answer == str(a + b) else 0.0

    return env


def _solve_add(prompt: str) -> str:
    _, a, b = prompt.split(":")
    return str(int(a) + int(b))


# ─── classification (the one cheap unit check worth keeping) ───────────


def test_classify_tool_buckets() -> None:
    def fn() -> None: ...

    class Bash:
        name = "bash"

    class HudComputerTool: ...

    class Marked:
        _legacy_capability_kind = "computer"

    assert _classify_tool(fn) == "mcp"
    assert _classify_tool(Bash()) == "shell"
    assert _classify_tool(HudComputerTool()) == "computer"
    assert _classify_tool(Marked()) == "computer"


def test_workspace_construction_has_no_runtime_side_effects(tmp_path) -> None:
    root = tmp_path / "workspace"

    workspace = Workspace(root)

    assert not root.exists()
    assert workspace._sock is None
    assert workspace._host_key is None


# ─── single rollout over the wire ─────────────────────────────────────


async def test_scenario_runs_start_to_evaluate_over_the_wire() -> None:
    env = _sum_env()
    async with served(env) as client:
        assert "add" in [t["id"] for t in await client.list_tasks()]
        async with Run(client, "add", {"a": 2, "b": 3}) as run:
            assert run.prompt == "add:2:3"
            run.trace.content = "5"
        assert run.reward == 1.0


async def test_wrong_answer_scores_zero() -> None:
    env = _sum_env()
    async with served(env) as client, Run(client, "add", {"a": 2, "b": 3}) as run:
        run.trace.content = "6"
    assert run.reward == 0.0


# ─── Taskset: concurrency, grouping, isolation ────────────────────────


async def test_taskset_concurrent_grouped_rollouts() -> None:
    env = _sum_env()
    add = cast("Any", env.tasks["add"])
    taskset = Taskset("adds", (add(a=i, b=i + 1) for i in range(4)))

    job = await taskset.run(
        _FnAgent(_solve_add), runtime=lambda _row: _local(env), group=2, max_concurrent=3
    )
    runs = job.runs

    assert len(runs) == 8  # 4 tasks x group of 2
    assert all(r.reward == 1.0 for r in runs)
    assert all(r.job_id == runs[0].job_id for r in runs)  # one job for the batch
    # Each task's group repeats share a group_id; 4 distinct groups of 2.
    groups = [r.group_id for r in runs]
    assert len(set(groups)) == 4
    assert all(groups.count(g) == 2 for g in set(groups))


async def test_taskset_isolates_a_failing_rollout() -> None:
    env = _sum_env()
    add = cast("Any", env.tasks["add"])

    def solve_or_boom(prompt: str) -> str:
        _, a, _b = prompt.split(":")
        if a == "2":
            raise RuntimeError("agent exploded")
        return _solve_add(prompt)

    job = await Taskset("adds", (add(a=i, b=1) for i in range(4))).run(
        _FnAgent(solve_or_boom), runtime=lambda _row: _local(env)
    )
    runs = job.runs

    assert len(runs) == 4
    failed = [r for r in runs if r.trace.isError]
    assert len(failed) == 1  # only a==2 blew up
    assert failed[0].reward == 0.0
    assert "agent exploded" in (failed[0].trace.content or "")
    # Mid-run failure keeps the real run: the prompt and placement survive.
    assert failed[0].prompt == "add:2:1"
    assert failed[0].runtime is not None
    assert sum(1 for r in runs if r.reward == 1.0) == 3  # the batch survived


# ─── error + cancellation edges ───────────────────────────────────────


async def test_unknown_task_raises_protocol_error() -> None:
    env = _sum_env()
    async with served(env) as client:
        with pytest.raises(HudProtocolError):
            await client.start_task("does-not-exist")


async def test_task_that_errors_in_evaluate_propagates() -> None:
    env = Environment("boom")
    with warnings.catch_warnings():
        _silence_deprecation()

        @env.scenario("explode")
        async def explode():
            yield "go"
            raise ValueError("evaluate failed")

    async with served(env) as client:
        with pytest.raises(HudProtocolError):
            async with Run(client, "explode", {}) as run:
                run.trace.content = "x"


async def test_exception_in_body_cancels_without_evaluating() -> None:
    env = _sum_env()
    async with served(env) as client:
        with pytest.raises(RuntimeError, match="agent failed"):
            async with Run(client, "add", {"a": 1, "b": 1}) as run:
                raise RuntimeError("agent failed")
        assert run.trace.isError is True
        assert run.reward == 0.0  # never graded


# ─── prompt modalities + typed returns ────────────────────────────────


async def test_chat_scenario_yields_message_list_prompt() -> None:
    env = Environment("chat-env")
    with warnings.catch_warnings():
        _silence_deprecation()

        @env.scenario("ask", chat=True)
        async def ask(messages: list[dict[str, Any]] | None = None):
            yield [*(messages or []), {"role": "system", "content": "ready"}]
            yield 1.0

    history = [{"role": "user", "content": "hello"}]
    async with served(env) as client, Run(client, "ask", {"messages": history}) as run:
        assert isinstance(run.prompt, list)
        assert run.prompt[-1]["content"] == "ready"
        assert run.prompt[0]["content"] == "hello"
        run.trace.content = "done"
    assert run.reward == 1.0


async def test_typed_returns_delivers_agent_answer() -> None:
    class Answer(BaseModel):
        value: int

    env = Environment("typed")
    with warnings.catch_warnings():
        _silence_deprecation()

        @env.scenario("typed", returns=Answer)
        async def typed():
            ans = yield "give me 42"
            ok = isinstance(ans, AgentAnswer) and ans.content.value == 42
            yield 1.0 if ok else 0.0

    async with served(env) as client, Run(client, "typed", {}) as run:
        run.trace.content = '{"value": 42}'
    assert run.reward == 1.0


# ─── on-serve capability synthesis (real launch, real manifest) ───────


async def test_legacy_tools_become_capabilities_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUD_RFB_URL", "rfb://127.0.0.1:5999")
    env = Environment("mixed")
    with warnings.catch_warnings():
        _silence_deprecation()

        @env.scenario("noop")
        async def noop():
            yield "p"
            yield 1.0

        @env.tool
        def lookup(q: str) -> str:
            return "ok"

        class Computer:
            _legacy_capability_kind = "computer"

        env.add_tool(Computer())

    async with served(env) as client:
        assert client.manifest is not None
        protocols = {c.protocol for c in client.manifest.bindings}
        # function tool -> mcp capability; computer marker -> rfb capability
        assert "mcp/2025-11-25" in protocols
        assert "rfb/3.8" in protocols
        # Loopback address, so the client sees its forwarded stand-in.
        assert client.binding("rfb").url.startswith("rfb://127.0.0.1:")
        # tasks still serve alongside the synthesized capabilities
        assert "noop" in [t["id"] for t in await client.list_tasks()]

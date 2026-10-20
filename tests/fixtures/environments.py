"""Environments built in the test process: ``lab`` (one template per way a task starts
or grades), the ``actor`` and ``judge`` verifier pair, and ``workspace_env`` (a real
workspace over SSH); ``served`` connects a client to one and ``run_task`` rolls an
agent out on one. Environments loaded from a file are in ``tests/fixtures/envs``."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud import Environment
from hud.clients import connect
from hud.eval import LocalRuntime, Task, rollout
from tests.conftest import write_files

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.capabilities import Capability
    from hud.clients import HudClient
    from hud.eval import Run


def solve(prompt: str) -> str:
    """The right answer to an ``add`` prompt."""
    _, a, b = prompt.split()
    return str(int(a) + int(b))


def lab(events: list[str] | None = None) -> Environment:
    """The ``lab`` environment: one template per way a task can start or grade."""
    log = events if events is not None else []
    env = Environment("lab")

    @env.initialize
    async def initialize() -> None:
        log.append("initialize")

    @env.shutdown
    async def shutdown() -> None:
        log.append("shutdown")

    @env.template()
    async def add(a: int, b: int):
        log.append(f"start add {a} {b}")
        try:
            answer = yield f"add {a} {b}"
            log.append(f"grade {answer}")
            yield 1.0 if answer == str(a + b) else 0.0
        finally:
            log.append("end add")

    @env.template()
    async def chat(messages: list[dict[str, Any]]):
        yield messages
        yield 1.0

    @env.template()
    async def silent():
        yield None
        yield 1.0

    @env.template()
    async def claim(published: Any):
        yield {"prompt": "go", "bindings": published}
        yield 1.0

    @env.template()
    async def grade_raises():
        yield "go"
        raise RuntimeError("grader exploded")
        yield 0.0

    @env.template()
    async def frame(result: dict[str, Any]):
        yield "go"
        yield result

    @env.template()
    async def hang_grading():
        try:
            yield "go"
            await asyncio.Event().wait()
            yield 1.0
        finally:
            log.append("grading cancelled")

    @env.template()
    async def hang_start():
        try:
            await asyncio.Event().wait()
            yield "never"
            yield 0.0
        finally:
            log.append("start cancelled")

    @env.template()
    async def large(criteria: str):
        log.append("large started")
        yield f"{len(criteria)}"
        yield 1.0

    return env


def minted(events: list[str] | None = None) -> Task:
    """A row minted by a live ``lab`` environment, which places it by default."""
    env = lab(events)

    @env.template()
    async def total(a: int, b: int):
        answer = yield f"add {a} {b}"
        yield 1.0 if answer == str(a + b) else 0.0

    return total(a=2, b=3)


def actor(env: Environment | None = None, *, grade: str = "score") -> Environment:
    """Declare the actor side of a verifier pair: ``solve`` grades 0.25 and carries the answer.

    It goes on ``env``, or on a new ``actor`` environment. ``grade="raise"`` makes
    its grading raise and ``grade="scoreless"`` yields a frame without a score.
    """
    env = env or Environment("actor")

    @env.template()
    async def solve():
        answer = yield "answer secret"
        if grade == "raise":
            raise RuntimeError("actor grade exploded")
        yield {"score": 0.25, "answer": answer} if grade == "score" else {"answer": answer}

    return env


def judge(env: Environment | None = None, *, verdict: str = "check") -> Environment:
    """Declare the judge side: ``verify`` pays 1.0 when the actor's answer was ``secret``.

    It goes on ``env``, or on a new ``judge`` environment. ``verdict`` picks a
    failure instead: ``"raise"``, ``"scoreless"`` or ``"zero"``.
    """
    env = env or Environment("judge")

    @env.template()
    async def verify():
        result = yield ""
        if verdict == "raise":
            raise RuntimeError("verifier exploded")
        if verdict == "scoreless":
            yield {"verdict": "pass"}
        elif verdict == "zero":
            yield 0.0
        else:
            yield 1.0 if result["answer"] == "secret" else 0.0

    return env


PROMPT = "Do the task."


def workspace_env(
    root: Path,
    *,
    files: Mapping[str, bytes | str] | None = None,
    passes: Callable[[Path], bool] = lambda _root: True,
    capabilities: tuple[Capability, ...] = (),
    name: str = "ws",
    settle: float = 0.0,
    **workspace: Any,
) -> Environment:
    """An environment serving ``root`` over SSH with one template, ``task``.

    ``files`` seed the workspace before each run; the task scores 1.0 when
    ``passes(root)`` holds ``settle`` seconds after the agent finishes, long
    enough for anything the agent left running to show. ``workspace`` goes to
    :meth:`Environment.workspace` (``env=`` for the shell environment).
    """
    env = Environment(name, capabilities=capabilities)
    env.workspace(root, guest_path=str(root), **workspace)

    @env.initialize
    async def seed() -> None:
        write_files(root, files or {})

    @env.template()
    async def task(prompt: str = PROMPT):
        yield prompt
        await asyncio.sleep(settle)
        yield 1.0 if passes(root) else 0.0

    return env


@asynccontextmanager
async def served(env: Environment, *, row: Task | None = None) -> AsyncIterator[HudClient]:
    """Serve ``env`` on loopback and yield a control-channel client connected to it."""
    task = row or Task(env=env.name, id="scenario")
    async with LocalRuntime(env)(task) as runtime, connect(runtime) as client:
        yield client


async def run_task(env: Environment, agent: Agent, *, prompt: str = PROMPT, **options: Any) -> Run:
    """Roll ``agent`` out on ``env``'s ``task`` in this process."""
    args = {} if prompt == PROMPT else {"prompt": prompt}
    return await rollout(
        Task(env=env.name, id="task", args=args), agent, runtime=LocalRuntime(env), **options
    )

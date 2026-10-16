"""Agents, providers and serving helpers for scenarios built on the public API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.agents.base import Agent
from hud.clients import connect
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from hud.clients import HudClient
    from hud.environment import Environment
    from hud.eval import Provider, Run, Runtime


class ScriptedAgent(Agent):
    """Answers every run with ``answer``, or ``answer(prompt)`` when it is callable.

    ``delay`` waits before answering; ``fail_before`` raises instead of answering
    and ``fail_after`` raises after the answer is recorded. ``linger`` keeps the
    agent running that long after answering, so a deadline can land on it.
    """

    def __init__(
        self,
        answer: str | Callable[[str], str] = "",
        *,
        delay: float = 0.0,
        fail_before: BaseException | None = None,
        fail_after: BaseException | None = None,
        linger: float = 0.0,
    ) -> None:
        super().__init__()
        self._answer: Callable[[str], str] = (
            (lambda _prompt: answer) if isinstance(answer, str) else answer
        )
        self._delay = delay
        self._fail_before = fail_before
        self._fail_after = fail_after
        self._linger = linger
        self.prompts: list[str] = []

    async def __call__(self, run: Run) -> None:
        self.prompts.append(run.prompt_text)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_before is not None:
            raise self._fail_before
        run.trace.content = self._answer(run.prompt_text)
        if self._linger:
            await asyncio.sleep(self._linger)
        if self._fail_after is not None:
            raise self._fail_after


class RecordingProvider:
    """Places each row with ``inner`` and records when placements start and stop."""

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.events: list[tuple[str, str]] = []

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        self.events.append(("start", task.slug))
        try:
            async with self._inner(task) as runtime:
                yield runtime
        finally:
            self.events.append(("stop", task.slug))


@asynccontextmanager
async def served(env: Environment, *, row: Task | None = None) -> AsyncIterator[HudClient]:
    """Serve ``env`` on loopback and yield a control-channel client connected to it."""
    task = row or Task(env=env.name, id="scenario")
    async with LocalRuntime(env)(task) as runtime, connect(runtime) as client:
        yield client


async def eventually(condition: Callable[[], bool], *, within: float = 10.0) -> None:
    """Poll until ``condition()`` holds; raise ``TimeoutError`` after ``within`` seconds."""
    async with asyncio.timeout(within):
        while True:
            if condition():
                return
            await asyncio.sleep(0.01)


def task_row(env: Environment, template: str, **args: Any) -> Task:
    """The data row for ``template`` on ``env``, as a taskset file would carry it."""
    return Task(env=env.name, id=template, args=args)

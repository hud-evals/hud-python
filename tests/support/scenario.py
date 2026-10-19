"""Agents, providers and serving helpers for scenarios built on the public API."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from hud.agents.base import Agent
from hud.clients import connect
from hud.eval import LocalRuntime, Task, rollout

from .envs import PROMPT as WORKSPACE_PROMPT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from hud.clients import HudClient
    from hud.environment import Environment
    from hud.eval import Run


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


@asynccontextmanager
async def served(env: Environment, *, row: Task | None = None) -> AsyncIterator[HudClient]:
    """Serve ``env`` on loopback and yield a control-channel client connected to it."""
    task = row or Task(env=env.name, id="scenario")
    async with LocalRuntime(env)(task) as runtime, connect(runtime) as client:
        yield client


async def eventually(
    condition: Callable[[], bool | Awaitable[bool]], *, within: float = 10.0
) -> None:
    """Poll until ``condition()`` holds; raise ``TimeoutError`` after ``within`` seconds.

    ``condition`` may be a coroutine function, for a check that has to await.
    """
    async with asyncio.timeout(within):
        while True:
            held = condition()
            if await held if inspect.isawaitable(held) else held:
                return
            await asyncio.sleep(0.01)


async def run_task(
    env: Environment, agent: Agent, *, prompt: str = WORKSPACE_PROMPT, **options: Any
) -> Run:
    """Roll ``agent`` out on ``env``'s ``task`` in this process."""
    args = {} if prompt == WORKSPACE_PROMPT else {"prompt": prompt}
    return await rollout(
        Task(env=env.name, id="task", args=args), agent, runtime=LocalRuntime(env), **options
    )

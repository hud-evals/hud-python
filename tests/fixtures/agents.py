"""Agents with fixed answers: an agent whose model is a function of the prompt."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from hud.agents.base import Agent
from hud.types import text_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud import Run


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
        self.prompts.append(text_of(run.prompt))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_before is not None:
            raise self._fail_before
        run.trace.content = self._answer(text_of(run.prompt))
        if self._linger:
            await asyncio.sleep(self._linger)
        if self._fail_after is not None:
            raise self._fail_after

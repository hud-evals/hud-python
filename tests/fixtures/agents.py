"""Agents with fixed answers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from hud.agents.base import Agent

if TYPE_CHECKING:
    from hud import Run


class ScriptedAgent(Agent):
    """Answers every run with ``answer``, after waiting ``delay`` seconds."""

    def __init__(self, answer: str = "", *, delay: float = 0.0) -> None:
        super().__init__()
        self._answer = answer
        self._delay = delay

    async def __call__(self, run: Run) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        run.trace.content = self._answer

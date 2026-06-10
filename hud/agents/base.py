"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.eval.rollout import Run


class Agent(ABC):
    """Drives a live ``Run`` to completion by filling ``run.trace`` in place.

    Subclasses implement ``__call__(run)``; callers do ``await agent(run)``. Stateless
    per run — everything comes from ``run`` — so one instance drives many concurrent
    rollouts.
    """

    @abstractmethod
    async def __call__(self, run: Run) -> None:
        """Drive ``run`` to completion, filling ``run.trace`` (answer is ``trace.content``)."""

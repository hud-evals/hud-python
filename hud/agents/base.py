"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.eval.run import Run


class Agent(ABC):
    """Drives a live ``Run`` by recording its trajectory and final answer.

    Subclasses implement ``__call__(run)``; callers do ``await agent(run)``. Stateless
    per run — everything comes from ``run`` — so one instance drives many concurrent
    rollouts. The caller owns lifecycle status, cancellation, and grading.
    """

    @abstractmethod
    async def __call__(self, run: Run) -> None:
        """Fill ``run.trace`` with the trajectory and final answer."""

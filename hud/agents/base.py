"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hud.client import Run


class Agent(ABC):
    """An agent turns a live run into a ``Trace``.

    Subclasses implement ``__call__(run)`` and callers drive an agent with
    ``await agent(run)``. An agent is stateless with respect to any single run —
    everything it needs comes from ``run`` (``run.prompt`` and capabilities via
    ``run.client.open`` / ``run.client.binding``) — so one instance can drive many
    concurrent rollouts safely.

    ``run`` owns the trace (like an RL rollout buffer or an open telemetry span):
    the agent *fills* ``run.trace`` in place — messages, samples, and the final
    ``content`` (the answer the env grades on exit) — rather than returning a new
    one. The caller reads the result back off ``run.trace``.
    """

    @abstractmethod
    async def __call__(self, run: Run) -> None:
        """Drive ``run`` to completion, filling ``run.trace`` (answer is ``trace.content``)."""

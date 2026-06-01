"""Run: the live handle for one task.

A ``Run`` is the dynamic counterpart to the static :class:`hud.types.Trace` — in
fact it *owns* a live ``trace`` that the agent fills in as it goes. Entering
starts the task (``tasks.start`` → ``prompt``); exiting grades it
(``tasks.evaluate`` reads ``trace.content`` → ``reward``) or cancels on error.

The agent acts *in* the run: it reads ``run.prompt``, reaches capabilities via
``run.client.open(...)``, and accumulates its trajectory onto ``run.trace``
(messages, samples, final ``content``). Because the trace is live, a rollout that
errors mid-flight still keeps whatever it gathered.

    async with client.task("sum_column", sheet="q3.xlsx") as run:
        ssh = await run.client.open("ssh")     # capabilities via the connection
        ...
        run.trace.content = answer              # graded on exit → run.trace.reward
    trace = run.trace                           # the datum
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from hud.types import Trace

if TYPE_CHECKING:
    from types import TracebackType

    from hud.client.client import HudClient


class Run:
    """Live handle for one task; owns the :class:`hud.types.Trace` it produces."""

    def __init__(self, client: HudClient, task_id: str, args: dict[str, Any]) -> None:
        self.client = client
        self._task_id = task_id
        self._args = args
        self.trace = Trace()

    @property
    def prompt(self) -> str | None:
        """The task prompt assigned by ``tasks.start`` on enter."""
        return self.trace.prompt

    async def __aenter__(self) -> Self:
        started = await self.client.start_task(self._task_id, self._args)
        self.trace.prompt = started.get("prompt")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            self.trace.isError = True
            await self.client.cancel()
            return False
        evaluation = await self.client.evaluate({"answer": self.trace.content})
        self.trace.reward = float(evaluation.get("score", 0.0))
        self.trace.info["evaluation"] = evaluation
        return False


__all__ = ["Run"]

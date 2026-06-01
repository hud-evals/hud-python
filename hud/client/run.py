"""Run: the live handle for one task.

``Run`` owns the *task lifecycle* — the things the env produces around a rollout:
the ``prompt`` (from ``tasks.start`` on enter), and the ``reward`` + raw
``evaluation`` (from ``tasks.evaluate`` on exit). It also holds the live ``trace``
the agent fills in as it goes.

The split mirrors who collects what:
- ``Run``   → task lifecycle: ``prompt``, ``reward``, ``evaluation`` (+ the live client).
- ``Trace`` → agent trajectory: ``messages``, ``samples``, ``content``, ``isError``.

The agent acts *in* the run: it reads ``run.prompt``, reaches capabilities via
``run.client.open(...)``, and accumulates onto ``run.trace`` (the answer is
``run.trace.content``). Because the trace is live, a rollout that errors mid-flight
still keeps whatever it gathered.

    async with client.task("sum_column", sheet="q3.xlsx") as run:
        ssh = await run.client.open("ssh")     # capabilities via the connection
        ...
        run.trace.content = answer              # graded on exit → run.reward
    print(run.reward)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

from hud.types import Trace

if TYPE_CHECKING:
    from types import TracebackType

    from hud.client.client import HudClient


class Run:
    """Live handle for one task: the task lifecycle plus the agent's ``Trace``."""

    def __init__(self, client: HudClient, task_id: str, args: dict[str, Any]) -> None:
        self.client = client
        self._task_id = task_id
        self._args = args
        self.prompt: str | None = None
        self.reward: float = 0.0
        self.evaluation: dict[str, Any] = {}
        self.trace = Trace()

    @property
    def trace_id(self) -> str | None:
        """Keys the agent's trajectory (satisfies the training ``Rewarded`` protocol)."""
        return self.trace.trace_id

    async def __aenter__(self) -> Self:
        started = await self.client.start_task(self._task_id, self._args)
        self.prompt = started.get("prompt")
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
        self.evaluation = await self.client.evaluate({"answer": self.trace.content})
        self.reward = float(self.evaluation.get("score", 0.0))
        return False

    @classmethod
    def failed(cls, error: str, *, trace_id: str | None = None) -> Run:
        """A spent run representing a rollout that failed before/while launching.

        Carries no live client; used for error isolation so one bad rollout never
        collapses a batch.
        """
        run = cls.__new__(cls)
        run.client = cast("HudClient", None)
        run._task_id = ""
        run._args = {}
        run.prompt = None
        run.reward = 0.0
        run.evaluation = {}
        run.trace = Trace(isError=True, content=error, info={"error": error}, trace_id=trace_id)
        return run


__all__ = ["Run"]

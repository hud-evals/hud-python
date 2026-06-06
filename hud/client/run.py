"""Run: the live handle for one task.

``Run`` owns the task lifecycle — ``prompt`` (from ``tasks.start`` on enter),
``reward`` + ``evaluation`` (from ``tasks.grade`` on exit) — and holds the live
``trace`` the agent fills (its answer is ``run.trace.content``)::

    async with client.task("sum_column", sheet="q3.xlsx") as run:
        run.trace.content = answer   # graded on exit → run.reward
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
        #: The task's opening prompt: plain text, or a list of message dicts
        #: (``{"role", "content"}``) for chat-style / multi-turn prompts.
        self.prompt: str | list[Any] | None = None
        self.reward: float = 0.0
        self.evaluation: dict[str, Any] = {}
        self.trace = Trace()
        #: Batch this run belongs to (set by the runner); platform job + GRPO group.
        self.job_id: str | None = None
        self.group_id: str | None = None

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
        answer: dict[str, Any] = {"answer": self.trace.content}
        if self.trace.citations:
            answer["citations"] = self.trace.citations
        self.evaluation = await self.client.grade(answer)
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
        run.job_id = None
        run.group_id = None
        return run


__all__ = ["Run"]

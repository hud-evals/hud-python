"""rollout: the execution atom — run one agent over one task, fully recorded.

:func:`rollout` is the single way an agent executes a task, and :class:`Run`
is its record: the live handle whose lifecycle the atom drives — ``prompt``
(from ``tasks.start`` on enter), the ``trace`` the agent fills (its answer is
``run.trace.content``), and the ``grade`` (from ``tasks.grade`` on exit)::

    run = await rollout(task, agent, on=spawn("env.py"))
    run = await task.run(agent, on=spawn("env.py"))  # same call, method sugar

``Taskset.run`` is the batch scheduler over this atom; ``Chat`` and
``AgentTool`` call it per turn / per invocation. The only paths that bypass it
are deliberate: ``hud task`` CLI (split start/grade lifecycle over raw RPCs)
and harbor's prompt-only materialization.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from hud.types import Trace

from .config import active
from .job import trace_enter, trace_exit

if TYPE_CHECKING:
    from types import TracebackType

    from hud.agents.base import Agent
    from hud.clients.client import HudClient
    from hud.environment.runtime import Provider

    from .task import Task

logger = logging.getLogger("hud.eval.rollout")


@dataclass(slots=True)
class Grade:
    """Structured result from grading one run."""

    reward: float = 0.0
    done: bool = True
    content: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grade:
        """Parse the wire grade frame (canonical keys: the server guarantees them)."""
        raw_info = data.get("info")
        return cls(
            reward=float(data.get("score") or 0.0),
            done=bool(data.get("done", True)),
            content=data.get("content") if isinstance(data.get("content"), str) else None,
            info=raw_info if isinstance(raw_info, dict) else {},
            is_error=bool(data.get("isError", False)),
            raw=data,
        )


class Run:
    """Live handle for one task: the task lifecycle plus the agent's ``Trace``.

    ``client`` is absent only on a :meth:`failed` run (a rollout that never
    launched); accessing it there raises instead of half-working.
    """

    def __init__(self, client: HudClient | None, task_id: str, args: dict[str, Any]) -> None:
        self._client = client
        self._task_id = task_id
        self._args = args
        #: The task's opening prompt: plain text, or a list of message dicts
        #: (``{"role", "content"}``) for chat-style / multi-turn prompts.
        self.prompt: str | list[Any] | None = None
        #: The structured grading result (all-default until graded on exit).
        self.grade = Grade()
        self.trace = Trace()
        #: Batch this run belongs to (set by the runner); platform job + GRPO group.
        self.job_id: str | None = None
        self.group_id: str | None = None
        # Written by ``Task.session`` once placement is acquired.
        self._runtime: str | None = None

    @property
    def client(self) -> HudClient:
        """The live client driving this run."""
        if self._client is None:
            raise RuntimeError("this run failed before launch; it has no live client")
        return self._client

    @property
    def reward(self) -> float:
        """The graded reward (``grade.reward``)."""
        return self.grade.reward

    @property
    def evaluation(self) -> dict[str, Any]:
        """The raw evaluation dict the env returned (``grade.raw``)."""
        return self.grade.raw

    @property
    def trace_id(self) -> str | None:
        """Keys the agent's trajectory (satisfies the training ``Rewarded`` protocol)."""
        return self.trace.trace_id

    @property
    def runtime(self) -> str | None:
        """Control-channel url of the runtime this run executed against.

        The factual placement record for the receipt; ``None`` on a run that
        failed before a substrate came up.
        """
        return self._runtime

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
        self.grade = Grade.from_dict(await self.client.grade(answer))
        return False

    @classmethod
    def failed(cls, error: str) -> Run:
        """A spent run representing a rollout that failed before launching.

        Carries no live client; only the pre-launch failure path synthesizes
        one — a rollout that failed *mid-run* keeps its real ``Run`` (prompt,
        runtime, partial trace) with the error recorded on the trace.
        """
        run = cls(None, "", {})
        run.trace = Trace(isError=True, content=error)
        return run


async def rollout(
    task: Task,
    agent: Agent,
    *,
    on: Provider | None = None,
    job_id: str | None = None,
    group_id: str | None = None,
) -> Run:
    """Drive one task to a graded :class:`Run` (the rollout atom).

    ``on`` is the placement provider — explicit beats the ambient
    :func:`hud.eval.configure` scope, which beats HUD-hosted provisioning by
    env name. The agent fills ``run.trace``; grading happens on session exit
    (``run.reward``). ``job_id``/``group_id`` are batch identities threaded by
    the scheduler, recorded on the trace. The per-rollout ``trace_id`` is
    bound into the trace context (so ``@instrument`` spans attribute to it —
    always, even with telemetry off, for local training) and the trace is
    reported to HUD.

    Failures are isolated so one bad rollout never collapses a batch, without
    erasing evidence: a failure *before* the session is live (provision,
    connect, start) yields a synthesized :meth:`Run.failed`; a failure
    *mid-run* keeps the real run — prompt, placement record, and the partial
    trace the agent built — marked as errored.
    """
    from hud.telemetry.context import set_trace_context

    on = on or active().on
    trace_id = uuid.uuid4().hex
    with set_trace_context(trace_id):
        await trace_enter(trace_id, job_id=job_id, group_id=group_id)
        run: Run | None = None
        try:
            async with task.session(on=on) as run:
                await agent(run)
        except TimeoutError:
            raise
        except Exception as exc:
            if run is None:
                logger.warning("rollout failed before launch: %s", exc)
                run = Run.failed(str(exc))
            else:
                logger.warning("rollout failed mid-run: %s", exc)
                run.trace.isError = True
                run.trace.content = str(exc)
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        await trace_exit(run)
    return run


__all__ = ["Grade", "Run", "rollout"]

"""Rollout: the live run handle for one task.

A ``Rollout`` is the dynamic counterpart to the static :class:`hud.types.Trace`.
It owns the connection and the task lifecycle: entering it starts the task
(``tasks.start`` → ``prompt``), exiting grades it (``tasks.evaluate`` → ``reward``)
or cancels on error. It exposes capability access (``open`` / ``binding``) and
drives an agent (``rollout``), building up the ``Trace`` datum as it goes.

    async with client.task("sum_column", sheet="q3.xlsx") as run:
        ssh = await run.open("shell")        # grab a capability
        ...                                   # do the work
        run.submit(answer)                    # or: await run.rollout(agent)
    trace = run.trace                         # the datum (run.reward == trace.reward)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from hud.types import Trace

if TYPE_CHECKING:
    from types import TracebackType

    from hud.capabilities import Capability, CapabilityClient
    from hud.client import Manifest
    from hud.client.client import HudClient


class Rollout:
    """Live run handle for one task; produces a :class:`hud.types.Trace`."""

    def __init__(self, client: HudClient, task_id: str, args: dict[str, Any]) -> None:
        self._client = client
        self._task_id = task_id
        self._args = args
        self._answer: str | dict[str, Any] | None = None
        self.trace = Trace()

    # ─── read-only views onto the datum / connection ──────────────────────

    @property
    def prompt(self) -> str | None:
        return self.trace.prompt

    @property
    def reward(self) -> float:
        return self.trace.reward

    @property
    def manifest(self) -> Manifest | None:
        return self._client.manifest

    # ─── lifecycle ────────────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        started = await self._client.start_task(self._task_id, self._args)
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
            await self._client.cancel()
            return False
        evaluation = await self._client.evaluate({"answer": self._answer})
        self.trace.reward = float(evaluation.get("score", 0.0))
        self.trace.info["evaluation"] = evaluation
        return False

    # ─── capability access (delegates to the connection) ──────────────────

    async def open(self, protocol: str) -> CapabilityClient:
        """Open a live capability client by protocol (delegates to the connection)."""
        return await self._client.open(protocol)

    def binding(self, protocol: str) -> Capability:
        """Return the raw capability declaration by protocol (BYO connection)."""
        return self._client.binding(protocol)

    # ─── driving the run ──────────────────────────────────────────────────

    def submit(self, answer: str | dict[str, Any]) -> None:
        """Stash the agent's answer; consumed by ``tasks.evaluate`` on exit."""
        self._answer = answer

    async def rollout(self, agent: Any) -> Trace:
        """Drive a (stateless) agent over this run, returning the ``Trace`` datum.

        ``agent`` is any callable ``(rollout) -> result`` — a bare async function
        or a configured agent exposing ``rollout``/``__call__``. It may return a
        rich ``Trace`` (its trajectory) or a bare answer (str/dict); either way the
        answer is submitted for grading.
        """
        result = await (agent.rollout(self) if hasattr(agent, "rollout") else agent(self))

        if isinstance(result, Trace):
            result.prompt = self.trace.prompt
            self.trace = result
            answer: str | dict[str, Any] | None = result.content
        else:
            answer = result

        if answer is not None and self._answer is None:
            self.submit(answer)
        return self.trace


__all__ = ["Rollout"]

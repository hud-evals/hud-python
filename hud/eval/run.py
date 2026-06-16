"""A run: its record (:class:`Run`) and the local driver that produces one
(:func:`rollout`).

:func:`rollout` connects to a substrate's control channel (wherever it is —
loopback, a container, a cloud sandbox), starts the task, drives the agent,
grades, and tears down, filling a :class:`Run` along the way::

    run = await rollout(task, agent, runtime=LocalRuntime("env.py"))

It is the *client-here* path: the agent loop runs in this process against a
:class:`~hud.eval.runtime.Provider`'s channel. The same driver runs on the
daemon (the leased box's agent loop is just ``rollout`` over a
``DockerRuntime``), in ``Chat`` per turn, and in ``AgentTool`` per invocation.
Delegated (HUD-hosted) execution is a different act — see
:class:`hud.eval.runtime.HUDRuntime` — and the scheduler (:meth:`Taskset.run`)
chooses between them; the atom itself never branches on placement.

:class:`Run` is also the receipt a delegated execution folds its platform
result into, so it lives here with the atom rather than importing back into it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import mcp.types as mcp_types

from hud.clients import connect
from hud.telemetry.context import set_trace_context
from hud.types import Step, TaskCall, Trace
from hud.utils.time import now_iso

from .job import job_enter, trace_enter, trace_exit

if TYPE_CHECKING:
    from types import TracebackType

    from hud.agents.base import Agent
    from hud.clients.client import HudClient

    from .runtime import Provider
    from .task import Task

logger = logging.getLogger("hud.eval.run")


def _prompt_message(item: Any) -> mcp_types.PromptMessage:
    """Coerce one wire prompt turn onto MCP's ``PromptMessage`` vocabulary.

    Turns are env-authored: chat-style dicts (plain-string content wrapped as
    text, roles outside MCP's user/assistant vocabulary such as ``system``
    coerced to ``user``), already-built ``PromptMessage``s, or anything else
    stringified. Coercion may be lossy — prompt context is what the agent is
    given, and the verbatim payload stays on the setup ``task`` step's result.
    """
    if isinstance(item, mcp_types.PromptMessage):
        return item
    if not isinstance(item, dict):
        item = {"content": str(item)}
    role = item.get("role")
    if role not in ("user", "assistant"):
        role = "user"
    content = item.get("content")
    if isinstance(content, str):
        return mcp_types.PromptMessage(
            role=role,
            content=mcp_types.TextContent(type="text", text=content),
        )
    return mcp_types.PromptMessage.model_validate({**item, "role": role})


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

    ``client`` is absent on a :meth:`failed` run (a rollout that never
    launched) and on delegated runs; accessing it there raises instead of
    half-working.
    """

    def __init__(self, client: HudClient | None, task_id: str, args: dict[str, Any]) -> None:
        self._client = client
        self._task_id = task_id
        self._args = args
        #: The task's opening prompt as ``tasks.start`` returned it: plain
        #: text, or a list of message dicts (``{"role", "content"}``) for
        #: chat-style / multi-turn prompts. Agents consume the normalized
        #: views: :attr:`prompt_messages` / :attr:`prompt_text`.
        self.prompt: str | list[Any] | None = None
        #: The structured grading result (all-default until graded on exit).
        self.grade = Grade()
        self.trace = Trace()
        #: Batch this run belongs to (set by the runner); platform job + GRPO group.
        self.job_id: str | None = None
        self.group_id: str | None = None
        #: The task slug this run came from (set by the rollout engine). Lets
        #: ``Job.results`` key runs back to their task without positional zip.
        self.slug: str | None = None
        # Written by :func:`rollout` once placement is acquired.
        self._runtime: str | None = None

    @property
    def client(self) -> HudClient:
        """The live client driving this run."""
        if self._client is None:
            raise RuntimeError(
                "this run has no live client (delegated execution, or it failed before launch)"
            )
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

    @property
    def prompt_messages(self) -> list[mcp_types.PromptMessage]:
        """The prompt as normalized ``PromptMessage`` turns.

        The structured form agents consume and the opening ``user`` step
        records: a text prompt (or none) is one user turn; chat-style lists
        map turn by turn.
        """
        if self.prompt is None or isinstance(self.prompt, str):
            return [_prompt_message({"content": self.prompt or ""})]
        return [_prompt_message(item) for item in self.prompt]

    @property
    def prompt_text(self) -> str:
        """The prompt flattened to plain text, for string-only agent backends.

        Text content of each turn joined by blank lines; non-text content
        (images, resources) is dropped — consume :attr:`prompt_messages`
        where structured turns are supported.
        """
        return "\n\n".join(
            message.content.text
            for message in self.prompt_messages
            if isinstance(message.content, mcp_types.TextContent) and message.content.text
        )

    def record(self, step: Step) -> None:
        """Record one step on the trace (:meth:`hud.types.Trace.record`)."""
        self.trace.record(step)

    async def __aenter__(self) -> Self:
        started_at = now_iso()
        started = await self.client.start_task(self._task_id, self._args)
        self.prompt = started.get("prompt")
        self.record(
            Step(
                source="task",
                task_call=TaskCall(
                    phase="setup",
                    name=self._task_id,
                    arguments=self._args,
                    result=started,
                ),
                started_at=started_at,
            ),
        )
        if self.prompt is not None:
            self.record(Step(source="user", messages=self.prompt_messages))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            cancelled = issubclass(exc_type, asyncio.CancelledError | KeyboardInterrupt)
            self.trace.status = "cancelled" if cancelled else "error"
            await self.client.cancel()
            return False
        answer: dict[str, Any] = {"answer": self.trace.content}
        started_at = now_iso()
        evaluation = await self.client.grade(answer)
        self.grade = Grade.from_dict(evaluation)
        self.record(
            Step(
                source="task",
                task_call=TaskCall(
                    phase="evaluate",
                    name=self._task_id,
                    arguments=answer,
                    result=evaluation,
                ),
                started_at=started_at,
                error=self.grade.content if self.grade.is_error else None,
            ),
        )
        if self.trace.status is None:
            self.trace.status = "completed"
        return False

    @classmethod
    def failed(cls, error: str) -> Run:
        """A spent run representing a rollout that failed before launching.

        Carries no live client; only the pre-launch failure path synthesizes
        one — a rollout that failed *mid-run* keeps its real ``Run`` (prompt,
        runtime, partial trace) with the error recorded on the trace.
        """
        run = cls(None, "", {})
        run.trace = Trace(status="error", steps=[Step(source="system", error=error)])
        return run


async def rollout(
    task: Task,
    agent: Agent,
    *,
    runtime: Provider,
    job_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
) -> Run:
    """Drive one task to a graded :class:`Run` here, against ``runtime``'s channel.

    The local driver (*client-here*): acquire the provider's substrate,
    connect, start the task, let ``agent`` fill ``run.trace``, grade on exit
    (``run.reward``), tear down. The substrate may be anywhere — a local
    subprocess, a container, a cloud sandbox — the channel bridges it; the
    agent loop always runs in *this* process. Delegated (HUD-hosted) execution
    does not come through here; see :class:`hud.eval.runtime.HUDRuntime`.

    ``job_id``/``group_id`` are batch identities threaded by the scheduler;
    there are no standalone traces, so when no ``job_id`` is given the atom
    registers a single-run job itself. ``trace_id`` is minted per rollout
    unless one is threaded in. It is bound into the trace context (so
    ``@instrument`` spans attribute to it — always, even with telemetry off,
    for local training) and the trace is reported to HUD.

    Failures are isolated so one bad rollout never collapses a batch, without
    erasing evidence: a failure *before* the run is live (provision, connect,
    start) yields a synthesized :meth:`Run.failed`; a failure *mid-run* keeps
    the real run — prompt, placement record, and the partial trace the agent
    built — marked as errored. Either way the logged warning names the lifecycle
    phase (``provisioning``, ``starting task``, ``agent loop``, ``grading``) so
    callers can tell where the failure landed without reading the trace.
    """
    if job_id is None:  # no standalone traces: a lone rollout is a job of one
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    trace_id = trace_id or uuid.uuid4().hex
    with set_trace_context(trace_id):
        await trace_enter(trace_id, job_id=job_id, group_id=group_id)
        run: Run | None = None
        _phase = "provisioning"
        try:
            async with runtime(task) as addr, connect(addr) as client:
                _phase = "starting task"
                live = Run(client, task.id, task.args)
                live._runtime = addr.url  # the placement record for the receipt
                async with live:  # start on enter; grade on exit
                    run = live  # bound only once live: an earlier failure synthesizes
                    _phase = "agent loop"
                    await agent(run)
                    _phase = "grading"
        except TimeoutError:
            raise
        except Exception as exc:
            if run is None:
                logger.warning("rollout failed before launch (%s): %s", _phase, exc)
                run = Run.failed(f"[{_phase}] {exc}")
            else:
                logger.warning("rollout failed mid-run (%s): %s", _phase, exc)
                run.trace.status = "error"
                run.record(Step(source="system", error=f"[{_phase}] {exc}"))
        assert run is not None  # the body bound it, or the handler synthesized it
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        run.slug = task.slug or task.default_slug()
        await trace_exit(run)
    return run


__all__ = ["Grade", "Run", "rollout"]

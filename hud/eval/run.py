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
Delegated hosted execution is a different act — see
:class:`hud.eval.runtime.HostedRuntime` — and the scheduler (:meth:`Taskset.run`)
chooses between them; the atom itself never branches on placement.

:class:`Run` is also the receipt a delegated execution folds its platform
result into, so it lives here with the atom rather than importing back into it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast

import mcp.types as mcp_types

from hud.clients import connect
from hud.telemetry.context import set_trace_context
from hud.types import Step, TaskCall, Trace
from hud.utils.time import now_iso

from .file_tracking import file_tracking_observer
from .job import job_enter, trace_enter, trace_exit

if TYPE_CHECKING:
    from types import TracebackType

    from hud.agents.base import Agent
    from hud.clients.client import HudClient

    from .runtime import Provider, Runtime
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
        """Keys the agent's trajectory; pass the ``Run`` (or this id) to training."""
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
        # Ctrl-C isn't a gradable outcome: tear down without grading.
        if exc_type is not None and issubclass(
            exc_type, asyncio.CancelledError | KeyboardInterrupt
        ):
            self.trace.status = "cancelled"
            await self.client.cancel()
            return False

        answer: dict[str, Any] = {"answer": self.trace.content}
        started_at = now_iso()

        # A mid-run error grades best-effort (capture a salvageable reward, keep
        # status=error), but a grade failure must not mask the original error. A
        # clean run grades normally — a grader fault propagates. grade() also
        # blocks on an unbounded JSON-RPC read (not bound by rollout_timeout).
        if exc_type is not None:
            self.trace.status = "error"
            try:
                evaluation = await self.client.grade(answer)
            except Exception as grade_exc:
                logger.warning("grade failed after mid-run error: %s", grade_exc)
                return False
        else:
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
    rollout_timeout: float | None = None,
) -> Run:
    """Drive one task to a graded :class:`Run` here, against ``runtime``'s channel.

    The local driver (*client-here*): acquire the provider's substrate,
    connect, start the task, let ``agent`` fill ``run.trace``, grade on exit
    (``run.reward``), tear down. The substrate may be anywhere — a local
    subprocess, a container, a cloud sandbox — the channel bridges it; the
    agent loop always runs in *this* process. Delegated hosted execution
    does not come through here; see :class:`hud.eval.runtime.HostedRuntime`.

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
    built — marked as errored, and still graded best-effort so a salvageable
    reward is captured. Either way the logged warning names the lifecycle
    phase (``provisioning``, ``starting task``, ``agent loop``, ``grading``) so
    callers can tell where the failure landed without reading the trace.
    """
    if job_id is None:  # no standalone traces: a lone rollout is a job of one
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    trace_id = trace_id or uuid.uuid4().hex
    # Report the model the agent will sample so the platform attributes the
    # trace to it on enter. Only LLM tool agents carry an inference-model slug
    # (``config.model``); robot/other agents have none. Local import avoids an
    # eval<->agents import cycle.
    from hud.agents.tool_agent import ToolAgent

    agent_model = agent.config.model if isinstance(agent, ToolAgent) else None
    with set_trace_context(trace_id):
        await trace_enter(trace_id, job_id=job_id, group_id=group_id, model=agent_model)
        run: Run | None = None
        _phase = "provisioning"

        loop = asyncio.get_running_loop()
        deadline = None if rollout_timeout is None else loop.time() + rollout_timeout

        async def _bounded(awaitable: Any) -> Any:
            """Bound work by the rollout's single wall-clock ``deadline``.

            One shared deadline across provision, connect, and the agent loop —
            not a fresh budget per phase — so the bounded work cannot exceed
            ``rollout_timeout`` in total. A client read-timeout is not enough on
            its own: a wedged upstream that trickles bytes resets the read timer
            forever, so a single stuck sampling call could otherwise hang the
            rollout — and the batch waits on it — indefinitely. A breach surfaces
            as ``TimeoutError`` (distinct from a Ctrl-C ``CancelledError``).
            """
            if deadline is None:
                return await awaitable
            return await asyncio.wait_for(awaitable, max(deadline - loop.time(), 0.0))

        async def _drive() -> None:
            nonlocal run, _phase
            async with contextlib.AsyncExitStack() as stack:
                # Setup (provision + connect) is bounded but not gradable: a
                # timeout fires before the run is live, so it surfaces as a
                # pre-launch failure. A cancelled enter still tears the
                # half-acquired substrate down via the provider's own cleanup.
                addr = cast("Runtime", await _bounded(stack.enter_async_context(runtime(task))))
                _phase = "starting task"
                client = cast("HudClient", await _bounded(stack.enter_async_context(connect(addr))))
                live = Run(client, task.id, task.args)
                live._runtime = addr.url  # the placement record for the receipt
                async with live:  # start on enter; grade on exit
                    run = live  # bound only once live: an earlier failure synthesizes
                    _phase = "agent loop"
                    # File tracking (when enabled) streams workspace diffs to
                    # telemetry for the duration of the agent loop; setup churn is
                    # skipped because the run is already started here.
                    async with file_tracking_observer(client):
                        try:
                            await _bounded(agent(run))
                        except TimeoutError:
                            # The run is live with a partial trajectory worth
                            # grading, so record the truncation and fall through
                            # to the normal grade-on-exit path. A bare cancel
                            # (Ctrl-C) does not land here — it is a CancelledError,
                            # which this does not catch, so it keeps the non-graded
                            # cancel path in ``__aexit__``.
                            logger.warning(
                                "rollout agent loop timed out after %.0fs; grading partial",
                                rollout_timeout,
                            )
                            run.trace.stop_reason = "timeout"
                            run.record(
                                Step(
                                    source="system",
                                    error=f"agent loop timed out after {rollout_timeout:.0f}s",
                                )
                            )
                    _phase = "grading"

        try:
            await _drive()
        except TimeoutError:
            # A setup-phase deadline (provision/connect) fired — the agent-loop
            # timeout is handled inside _drive. Isolate it so one wedged rollout
            # never collapses the batch, keeping any partial trace.
            detail = f"timed out after {rollout_timeout:.0f}s" if rollout_timeout else "timed out"
            if run is None:
                logger.warning("rollout failed before launch (%s): %s", _phase, detail)
                run = Run.failed(f"[{_phase}] {detail}")
            else:
                logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                run.trace.status = "error"
                run.record(Step(source="system", error=f"[{_phase}] {detail}"))
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


async def vec_rollout(
    task: Task,
    agent: Any,
    *,
    runtime: Provider,
    num_envs: int,
    job_id: str | None = None,
    group_id: str | None = None,
    rollout_timeout: float | None = None,
) -> list[Run]:
    """Drive one vectorized env instance to ``num_envs`` graded runs.

    The vectorized counterpart of :func:`rollout`, still domain-agnostic: one
    runtime, one task start (``num_envs`` injected into the task args; an env
    whose template doesn't accept it fails loudly), the agent's
    ``drive(runs, client)`` entry drives every slot over one connection, and
    the grade's ``"slots"`` list grades run ``i``. The runs are *receipts* —
    this atom owns trace lifecycle and grading. Its task -> runs shape also
    permits a future scheduler packing different compatible tasks into one
    instance's slots.
    """
    if not hasattr(agent, "drive"):
        raise TypeError(
            f"{type(agent).__name__} cannot drive a vectorized rollout "
            "(needs a drive(runs, client) entry, e.g. hud.agents.robot.RobotAgent)"
        )
    if job_id is None:  # a lone vectorized rollout is a job of one instance
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    group_id = group_id or uuid.uuid4().hex

    runs = [Run(None, task.id, task.args) for _ in range(num_envs)]
    for run in runs:
        run.trace.trace_id = uuid.uuid4().hex
        run.job_id = job_id
        run.group_id = group_id
        run.slug = task.slug or task.default_slug()

    loop = asyncio.get_running_loop()
    deadline = None if rollout_timeout is None else loop.time() + rollout_timeout

    async def _bounded(awaitable: Any) -> Any:
        # One shared wall-clock deadline across provision, start, and the agent
        # loop (see rollout._bounded for why a read-timeout is not enough).
        if deadline is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, max(deadline - loop.time(), 0.0))

    _phase = "provisioning"
    try:
        async with contextlib.AsyncExitStack() as stack:
            addr = cast("Runtime", await _bounded(stack.enter_async_context(runtime(task))))
            _phase = "starting task"
            client = cast("HudClient", await _bounded(stack.enter_async_context(connect(addr))))
            args = {**task.args, "num_envs": num_envs}
            prompt = (await _bounded(client.start_task(task.id, args))).get("prompt")
            for run in runs:
                run.prompt = prompt
                run._runtime = addr.url
                tid = run.trace.trace_id
                assert tid is not None  # minted above
                await trace_enter(tid, job_id=job_id, group_id=group_id, model=None)
                with set_trace_context(tid):  # opening user step per trace
                    run.record(Step(source="user", messages=run.prompt_messages))
            _phase = "agent loop"
            await _bounded(agent.drive(runs, client))
            _phase = "grading"
            evaluation = await client.grade({"answer": None})
            slots = evaluation.get("slots") or []
            if len(slots) != num_envs:
                raise ValueError(
                    f"grade returned {len(slots)} slots for {num_envs} envs "
                    "(vectorized envs must return one 'slots' entry per slot)"
                )
            for run, slot in zip(runs, slots, strict=True):
                run.grade = Grade.from_dict(slot)
                run.trace.status = "completed"
    except Exception as exc:  # isolate: one bad instance never kills the batch
        detail = (
            f"timed out after {rollout_timeout:.0f}s"
            if isinstance(exc, TimeoutError) and rollout_timeout
            else str(exc)
        )
        logger.warning("vectorized rollout failed (%s): %s", _phase, detail)
        for run in runs:
            run.trace.status = "error"
            run.record(Step(source="system", error=f"[{_phase}] {detail}"))
    for run in runs:
        await trace_exit(run)
    return runs


__all__ = ["Grade", "Run", "rollout", "vec_rollout"]

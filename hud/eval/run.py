"""A run: its record (:class:`Run`) and the local driver that produces one
(:func:`rollout`).

:func:`rollout` connects to a substrate's control channel (wherever it is —
loopback, a container, a cloud sandbox), starts the task, drives the agent,
grades, and tears down, filling a :class:`Run` along the way::

    run = await rollout(task, agent, runtime=LocalRuntime("env.py"))

It is the *client-here* path: the agent loop runs in this process against a
:class:`~hud.eval.runtime.Provider`'s channel. The same driver handles hosted
execution once delegated, each ``Chat`` turn, and each ``AgentTool`` invocation.
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
import traceback
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import mcp.types as mcp_types

from hud.clients import HudProtocolError, connect
from hud.graders.results import SubScore
from hud.telemetry.context import set_trace_context
from hud.types import Step, TaskCall, Trace
from hud.utils.time import now_iso

from .file_tracking import file_tracking_observer
from .job import job_enter, trace_enter, trace_exit

if TYPE_CHECKING:
    from types import TracebackType

    from hud.agents.base import Agent
    from hud.clients.client import HudClient

    from .runtime import Provider
    from .runtime.core import RuntimeConfig, RuntimeSession
    from .task import Task

logger = logging.getLogger("hud.eval.run")


def validate_rollout_timeouts(
    task: Task,
    agent: Agent,
    rollout_timeout: float | None,
    *,
    actor_runtime_config: RuntimeConfig | None,
    verifier_runtime_config: RuntimeConfig | None,
) -> float | None:
    """Validate configured phase limits and return the effective agent timeout."""
    from hud.agents.types import AgentConfig

    config = getattr(agent, "config", None)
    agent_timeout = config.timeout_seconds if isinstance(config, AgentConfig) else None
    if task.agent_config is not None:
        agent_timeout = task.agent_config.get("timeout_seconds", agent_timeout)

    actor_limits = actor_runtime_config.limits if actor_runtime_config is not None else None
    actor_run_timeout = actor_limits.run_timeout_s if actor_limits is not None else None
    if (
        agent_timeout is not None
        and actor_run_timeout is not None
        and agent_timeout >= actor_run_timeout
    ):
        raise ValueError(
            f"agent timeout ({agent_timeout:g}s) must be less than "
            f"runtime_config.limits.run_timeout_s ({actor_run_timeout}s)"
        )

    if rollout_timeout is None:
        return agent_timeout
    if rollout_timeout <= 0:
        raise ValueError("rollout_timeout must be greater than 0")

    if agent_timeout is not None and agent_timeout >= rollout_timeout:
        raise ValueError(
            f"agent timeout ({agent_timeout:g}s) must be less than "
            f"rollout_timeout ({rollout_timeout:g}s)"
        )

    configs = (("actor", actor_runtime_config), ("verifier", verifier_runtime_config))
    for phase, config in configs:
        if config is None or config.limits is None:
            continue
        for limit_name in ("startup_timeout_s", "run_timeout_s"):
            value = getattr(config.limits, limit_name)
            if value is not None and value >= rollout_timeout:
                raise ValueError(
                    f"{phase} runtime_config.limits.{limit_name} ({value}s) must be less than "
                    f"rollout_timeout ({rollout_timeout:g}s)"
                )
    return agent_timeout


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
    raw_role = item.get("role")
    role: Literal["user", "assistant"] = "assistant" if raw_role == "assistant" else "user"
    content = item.get("content")
    if isinstance(content, str):
        return mcp_types.PromptMessage(
            role=role,
            content=mcp_types.TextContent(type="text", text=content),
        )
    return mcp_types.PromptMessage.model_validate({**item, "role": role})


def _episode_bindings(started: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the start frame's per-episode ``bindings`` (capability name -> data).

    Episode-scoped connection data the template published alongside the
    prompt, keyed like the manifest's bindings so an agent looks its
    capability's entry up by name. A malformed frame raises rather than
    silently handing the agent an empty mapping.
    """
    raw = started.get("bindings")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise TypeError(
            f"task start frame 'bindings' must map capability name -> object, got {raw!r}"
        )
    return cast("dict[str, dict[str, Any]]", raw)


@dataclass(slots=True)
class Grade:
    """Structured result from grading one run."""

    reward: float = 0.0
    done: bool = True
    content: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.evaluation = dict(self.raw)
        if isinstance(subscores := self.evaluation.get("subscores"), list):
            self.evaluation["subscores"] = [
                SubScore.model_validate(subscore).to_summary() for subscore in subscores
            ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grade:
        """Parse the wire grade frame (canonical keys: the server guarantees them)."""
        score = data.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise HudProtocolError(-32603, "tasks.grade: result must include a numeric 'score'")
        raw_info = data.get("info")
        raw = dict(data)
        return cls(
            reward=float(score),
            done=bool(data.get("done", True)),
            content=data.get("content") if isinstance(data.get("content"), str) else None,
            info=raw_info if isinstance(raw_info, dict) else {},
            is_error=bool(data.get("isError", False)),
            raw=raw,
        )


class Run:
    """Live handle for one task: the task lifecycle plus the agent's ``Trace``.

    ``client`` is absent on a :meth:`failed` run (a rollout that never
    launched) and on delegated runs; accessing it there raises instead of
    half-working.
    """

    def __init__(
        self,
        client: HudClient | None,
        task_id: str,
        args: dict[str, Any],
        *,
        best_effort_grade: bool = False,
    ) -> None:
        self._client = client
        self._task_id = task_id
        self._args = args
        self._best_effort_grade = best_effort_grade
        #: The task's opening prompt as ``tasks.start`` returned it: plain
        #: text, or a list of message dicts (``{"role", "content"}``) for
        #: chat-style / multi-turn prompts. Agents consume the normalized
        #: views: :attr:`prompt_messages` / :attr:`prompt_text`.
        self.prompt: str | list[Any] | None = None
        #: Per-episode binding data by capability name, from the start frame's
        #: ``bindings``: connection details that exist only for this episode —
        #: a robot slot token, a per-episode url — refining the manifest's
        #: env-lifetime bindings. Empty when the env published none.
        self.bindings: dict[str, dict[str, Any]] = {}
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
    def task_id(self) -> str:
        """Which task of the environment this run started.

        What an agent driving a whole taskset has to branch on: ``slug`` is
        assigned by the runner and is not set while the agent is running.
        """
        return self._task_id

    @property
    def reward(self) -> float:
        """The graded reward (``grade.reward``)."""
        return self.grade.reward

    @property
    def evaluation(self) -> dict[str, Any]:
        """A persistence-safe view of the task's evaluation result."""
        return dict(self.grade.evaluation)

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
        self.bindings = _episode_bindings(started)
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
            self.trace.status = "error" if self.trace.stop_reason == "timeout" else "cancelled"
            with contextlib.suppress(Exception):
                await self.client.cancel()
            return False

        answer: dict[str, Any] = {"answer": self.trace.content}
        started_at = now_iso()

        if exc_type is not None:
            self.trace.status = "error"

        try:
            evaluation = await self.client.grade(answer)
            grade = Grade.from_dict(evaluation)
        except Exception as grade_exc:
            if exc_type is None and not self._best_effort_grade:
                raise
            detail = "".join(traceback.format_exception_only(grade_exc)).strip()
            logger.warning("best-effort grade failed: %s", detail)
            self.grade = Grade(
                content=detail,
                is_error=True,
                raw={
                    "score": 0.0,
                    "answer": self.trace.content,
                    "content": detail,
                    "isError": True,
                },
            )
            self.trace.status = "error"
            self.record(Step(source="system", error=f"[grading] {detail}"))
            return False

        self.grade = grade
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


async def _verify(
    run: Run,
    client: HudClient,
    task: Task,
    actor_result: dict[str, Any],
) -> None:
    """Run an agent-less verifier task and make its evaluation authoritative."""
    started_at = now_iso()
    started = await client.start_task(task.id, task.args)
    run.record(
        Step(
            source="task",
            task_call=TaskCall(
                phase="setup",
                name=task.id,
                arguments=task.args,
                result=started,
            ),
            started_at=started_at,
        )
    )

    answer = {"answer": actor_result}
    started_at = now_iso()
    evaluation = await client.grade(answer)
    run.grade = Grade.from_dict(evaluation)
    run.record(
        Step(
            source="task",
            task_call=TaskCall(
                phase="evaluate",
                name=task.id,
                arguments=answer,
                result=evaluation,
            ),
            started_at=started_at,
            error=run.grade.content if run.grade.is_error else None,
        )
    )


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
    phase (``provisioning``, ``starting task``, ``agent loop``, ``grading``,
    ``cleanup``) so
    callers can tell where the failure landed without reading the trace.

    ``rollout_timeout`` bounds execution through grading. A timeout aborts the
    control transport, lets provider teardown finish in the background, and
    returns an errored run immediately.
    """
    from .runtime.core import resolve_runtime_config

    agent_timeout = validate_rollout_timeouts(
        task,
        agent,
        rollout_timeout,
        actor_runtime_config=resolve_runtime_config(runtime, task),
        verifier_runtime_config=(
            resolve_runtime_config(runtime, task.verifier) if task.verifier is not None else None
        ),
    )
    if job_id is None:  # no standalone traces: a lone rollout is a job of one
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    trace_id = trace_id or uuid.uuid4().hex
    # Report the model the agent will sample so the platform attributes the
    # trace to it on enter. Registered LLM and CLI agents carry it on their
    # AgentConfig; robot/custom agents may not.
    from hud.agents.types import AgentConfig

    config = getattr(agent, "config", None)
    agent_model = config.model if isinstance(config, AgentConfig) else None
    with set_trace_context(trace_id):
        await trace_enter(
            trace_id,
            job_id=job_id,
            group_id=group_id,
            task_slug=task.slug,
            model=agent_model,
        )
        run: Run | None = None
        _phase = "provisioning"
        rollout_expired = False

        client: HudClient | None = None

        async def _drive() -> None:
            nonlocal client, run, _phase
            actor_result: dict[str, Any] = {}
            actor_session: RuntimeSession | None = None
            verifier = task.verifier
            shared_verifier = (
                verifier is not None
                and verifier.env == task.env
                and verifier.runtime_config is None
            )
            async with contextlib.AsyncExitStack() as scope:
                actor = contextlib.AsyncExitStack()
                await actor.__aenter__()

                async def close_actor() -> None:
                    cleanup = asyncio.create_task(actor.aclose())
                    cleanup.add_done_callback(_consume_task_result)
                    await asyncio.shield(cleanup)

                scope.push_async_callback(close_actor)
                addr = await actor.enter_async_context(runtime(task))
                _phase = "starting task"
                async with connect(addr) as actor_client:
                    client = actor_client
                    live = Run(
                        actor_client,
                        task.id,
                        task.args,
                        best_effort_grade=task.verifier is not None,
                    )
                    live._runtime = addr.url  # the placement record for the receipt
                    async with live:  # start on enter; complete on exit
                        run = live  # bound only once live: an earlier failure synthesizes
                        _phase = "agent loop"
                        try:
                            async with file_tracking_observer(actor_client):
                                if agent_timeout is None:
                                    await agent(run)
                                else:
                                    deadline = asyncio.timeout(agent_timeout)
                                    try:
                                        async with deadline:
                                            await agent(run)
                                    except TimeoutError:
                                        if not deadline.expired():
                                            raise
                                        detail = f"agent timed out after {agent_timeout:g}s"
                                        logger.warning(detail)
                                        run.trace.status = "error"
                                        run.trace.stop_reason = "timeout"
                                        run.record(Step(source="system", error=detail))
                        except Exception as exc:
                            if task.verifier is None:
                                raise
                            detail = "".join(traceback.format_exception_only(exc)).strip()
                            logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                            run.trace.status = "error"
                            run.record(Step(source="system", error=f"[{_phase}] {detail}"))
                        _phase = "grading"

                    if verifier is not None:
                        actor_result = live.grade.raw
                        assert actor_client.manifest is not None
                        actor_session = addr.session(actor_client.manifest.session_id)
                        # The verifier is authoritative. Once its phase begins,
                        # an actor-side grade must not survive a verifier failure.
                        live.grade = Grade()
                    if shared_verifier:
                        assert verifier is not None
                        _phase = "verifying"
                        await _verify(live, actor_client, verifier, actor_result)
                        _phase = "cleanup"
                        return

                if rollout_expired:
                    return
                if verifier is None:
                    _phase = "actor cleanup"
                    await actor.aclose()
                    _phase = "cleanup"
                    return

                assert actor_session is not None
                _phase = "snapshotting actor"
                async with actor_session.snapshot() as archive:
                    _phase = "actor cleanup"
                    await actor.aclose()
                    client = None
                    if rollout_expired:
                        return
                    _phase = "provisioning verifier"
                    verifier_addr = await scope.enter_async_context(runtime(verifier))
                    verifier_client = await scope.enter_async_context(connect(verifier_addr))
                    if archive is not None:
                        assert verifier_client.manifest is not None
                        await verifier_addr.session(verifier_client.manifest.session_id).restore(
                            archive
                        )
                    client = verifier_client
                    _phase = "verifying"
                    await _verify(live, verifier_client, verifier, actor_result)
                _phase = "cleanup"

        driver = asyncio.create_task(_drive())
        try:
            if rollout_timeout is None:
                await driver
            else:
                done, _ = await asyncio.wait({driver}, timeout=rollout_timeout)
                if done:
                    await driver
                else:
                    rollout_expired = True
                    phase = _phase
                    if run is not None:
                        run.trace.stop_reason = "timeout"
                    if client is not None:
                        # Cancel before abort so env teardown frees robot slots;
                        # a bare abort parks the session and would leak claims.
                        # Bound cancel: a live-but-silent peer would otherwise hang
                        # forever inside read_frame and never reach abort.
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(client.cancel(), timeout=2.0)
                        client.abort()
                    if phase not in {"actor cleanup", "cleanup"}:
                        driver.cancel()
                    driver.add_done_callback(_consume_task_result)
                    detail = f"rollout timed out after {rollout_timeout:g}s during {phase}"
                    logger.warning(detail)
                    if run is None:
                        run = Run.failed(detail)
                    else:
                        run.trace.status = "error"
                        run.record(Step(source="system", error=detail))
                    run.trace.stop_reason = "timeout"
        except asyncio.CancelledError:
            if client is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.cancel(), timeout=2.0)
                client.abort()
            driver.cancel()
            driver.add_done_callback(_consume_task_result)
            raise
        except Exception as exc:
            # format_exception_only keeps __notes__ — a provider attaches what
            # only it can see there, like the sandbox's env output on a failed
            # handshake — where str(exc) would drop them.
            detail = "".join(traceback.format_exception_only(exc)).strip()
            if run is None:
                logger.warning("rollout failed before launch (%s): %s", _phase, detail)
                run = Run.failed(f"[{_phase}] {detail}")
            else:
                logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                run.trace.status = "error"
                run.record(Step(source="system", error=f"[{_phase}] {detail}"))
        assert run is not None  # the body bound it, or the handler synthesized it
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        run.slug = task.slug
        await trace_exit(run)
    return run


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


__all__ = ["Grade", "Run", "rollout"]

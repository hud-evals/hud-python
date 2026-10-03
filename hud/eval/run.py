"""The live handle an agent drives, and the rollout that grades it into a trace."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from hud.agents.types import AgentConfig
from hud.capabilities.mcp import get_mcp_trace_id
from hud.clients import HudProtocolError
from hud.telemetry.context import get_current_trace_id, set_trace_context
from hud.telemetry.span import normalize_trace_id
from hud.types import Grade, PromptMessage, Step, TaskCall, TextContent, Trace
from hud.utils.time import now_iso

from .file_tracking import file_tracking_observer
from .job import Job, trace_enter, trace_exit

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud.agents.base import Agent
    from hud.capabilities import Capability, Connection
    from hud.environment.egress import WorkspaceRoute
    from hud.environment.file_tracker import FileTrackingOp
    from hud.runtime import HostedRuntime, Provider
    from hud.runtime.core import RuntimeConfig

    from .task import Task

logger = logging.getLogger("hud.eval.run")


class EnvironmentSession(Protocol):
    """One live environment as the engine drives it; each kind of runtime opens its own.

    ``capabilities`` are at the addresses this process reaches them. ``revoke`` ends the
    agent's access and must finish before ``grade``. ``files`` returns ``None`` when the
    environment tracks no files. Closing the session cancels any task it still holds.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def capabilities(self) -> list[Capability]: ...

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]: ...

    async def revoke(self) -> None: ...

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def files(self, op: FileTrackingOp) -> dict[str, Any] | None: ...


def validate_rollout_timeouts(
    task: Task,
    agent: Agent,
    rollout_timeout: float | None,
    *,
    actor_runtime_config: RuntimeConfig | None,
    verifier_runtime_config: RuntimeConfig | None,
) -> float | None:
    """Validate configured phase limits and return the effective agent timeout."""
    agent_timeout = agent.config.timeout_seconds
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


def parent_trace(trace_id: str) -> uuid.UUID | None:
    """The trace a rollout of ``trace_id`` runs inside: the ambient trace, when it is
    another one. The platform keys traces by UUID, so an ambient id that is not one
    names no trace it holds, and the rollout has no parent. A caller sends it in the
    form it sends the rollout's own trace id: hex locally, canonical when hosted."""
    ambient = get_current_trace_id() or get_mcp_trace_id()
    if ambient is None or normalize_trace_id(ambient) == normalize_trace_id(trace_id):
        return None
    try:
        return uuid.UUID(ambient)
    except ValueError:
        return None


def _prompt_messages(item: Any) -> list[PromptMessage]:
    # Lossy by design: the verbatim prompt stays on the setup task step. A
    # PromptMessage holds one content block, so a multi-block turn becomes one
    # message per block.
    if isinstance(item, PromptMessage):
        return [item]
    if not isinstance(item, dict):
        item = {"content": str(item)}
    raw_role = item.get("role")
    role: Literal["user", "assistant"] = "assistant" if raw_role == "assistant" else "user"
    content = item.get("content")
    if isinstance(content, str):
        return [
            PromptMessage(
                role=role,
                content=TextContent(type="text", text=content),
            )
        ]
    blocks = content if isinstance(content, list) else [content]
    return [
        PromptMessage.model_validate({**item, "role": role, "content": block}) for block in blocks
    ]


def _episode_capabilities(
    capabilities: list[Capability], started: dict[str, Any]
) -> list[Capability]:
    # A start frame's ``bindings`` (capability name -> data, e.g. a robot slot token) refine
    # the environment's capabilities for this episode.
    raw = started.get("bindings")
    if raw is None:
        return list(capabilities)
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        raise TypeError(
            f"task start frame 'bindings' must map capability name -> object, got {raw!r}"
        )
    episode = cast("dict[str, dict[str, Any]]", raw)
    names = {capability.name for capability in capabilities}
    if unknown := sorted(set(episode) - names):
        raise ValueError(f"task start frame binds unknown capabilities: {unknown}")
    return [
        replace(capability, params={**capability.params, **episode[capability.name]})
        if capability.name in episode
        else capability
        for capability in capabilities
    ]


def parse_grade(frame: dict[str, Any]) -> Grade:
    score = frame.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise HudProtocolError(
            -32603, f"tasks.grade: result must include a finite numeric 'score', got {score!r}"
        )
    info = frame.get("info")
    content = frame.get("content")
    return Grade(
        reward=float(score),
        done=bool(frame.get("done", True)),
        content=content if isinstance(content, str) else None,
        info=info if isinstance(info, dict) else {},
        is_error=bool(frame.get("isError", False)),
        raw=dict(frame),
    )


@dataclass
class Run:
    """What an agent sees of one task: its prompt, the environment's capabilities, the
    connections an in-box process can bind, and the trace it writes.

    A capability's ``url`` is where this process reaches it; ``env_url`` is where code
    running inside the environment does.
    """

    prompt: list[PromptMessage]
    capabilities: list[Capability]
    connections: dict[str, Connection] = field(default_factory=dict)
    trace: Trace = field(default_factory=Trace)


@asynccontextmanager
async def start_run(
    session: EnvironmentSession,
    task_id: str,
    args: dict[str, Any],
    *,
    connections: Sequence[Connection] = (),
    best_effort_grade: bool = False,
) -> AsyncIterator[Run]:
    """Start ``task_id`` on ``session`` and yield the agent's ``Run``.

    When the block exits, the agent's access ends, file tracking takes its final capture,
    and the trace is graded: best-effort if the block failed, not at all if it was cancelled.
    """
    started_at = now_iso()
    started = await session.start_task(task_id, args)
    raw_prompt = started.get("prompt")
    prompt = (
        _prompt_messages({"content": raw_prompt or ""})
        if raw_prompt is None or isinstance(raw_prompt, str)
        else [message for item in raw_prompt for message in _prompt_messages(item)]
    )
    run = Run(
        prompt,
        _episode_capabilities(session.capabilities, started),
        {connection.name: connection for connection in connections},
    )
    run.trace.record(
        Step(
            source="task",
            task_call=TaskCall(phase="setup", name=task_id, arguments=args, result=started),
            started_at=started_at,
        )
    )
    if raw_prompt is not None:
        run.trace.record(Step(source="user", messages=prompt))
    try:
        async with file_tracking_observer(session):
            try:
                yield run
            finally:
                # Access ends before the final file capture, which ends before grading.
                run.connections.clear()
                await session.revoke()
    except (asyncio.CancelledError, KeyboardInterrupt):
        run.trace.status = "error" if run.trace.stop_reason == "timeout" else "cancelled"
        raise
    except Exception:
        run.trace.status = "error"
        await _grade(run, session, task_id, best_effort=True)
        raise
    await _grade(run, session, task_id, best_effort=best_effort_grade)


async def _grade(run: Run, session: EnvironmentSession, task_id: str, *, best_effort: bool) -> None:
    answer: dict[str, Any] = {"answer": run.trace.content}
    started_at = now_iso()
    try:
        evaluation = await session.grade(answer)
        grade = parse_grade(evaluation)
    except Exception as exc:
        if not best_effort:
            raise
        detail = "".join(traceback.format_exception_only(exc)).strip()
        logger.warning("best-effort grade failed: %s", detail)
        run.trace.grade = Grade(
            content=detail,
            is_error=True,
            raw={"score": 0.0, "answer": run.trace.content, "content": detail, "isError": True},
        )
        run.trace.status = "error"
        run.trace.record(Step(source="system", error=f"[grading] {detail}"))
        return
    run.trace.grade = grade
    run.trace.record(
        Step(
            source="task",
            task_call=TaskCall(phase="evaluate", name=task_id, arguments=answer, result=evaluation),
            started_at=started_at,
            error=grade.content if grade.is_error else None,
        )
    )
    if run.trace.status is None:
        run.trace.status = "completed"


async def _verify(
    run: Run,
    session: EnvironmentSession,
    task: Task,
    actor_result: dict[str, Any],
) -> None:
    """Run an agent-less verifier task and make its evaluation authoritative."""
    started_at = now_iso()
    started = await session.start_task(task.id, task.args)
    run.trace.record(
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
    evaluation = await session.grade(answer)
    grade = run.trace.grade = parse_grade(evaluation)
    run.trace.record(
        Step(
            source="task",
            task_call=TaskCall(
                phase="evaluate",
                name=task.id,
                arguments=answer,
                result=evaluation,
            ),
            started_at=started_at,
            error=grade.content if grade.is_error else None,
        )
    )


async def rollout(
    task: Task,
    agent: Agent,
    *,
    runtime: Provider | HostedRuntime,
    job_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
    rollout_timeout: float | None = None,
    connections: Sequence[Connection] = (),
    workspace_routes: Sequence[WorkspaceRoute] = (),
) -> Trace:
    """Drive ``agent`` through ``task`` on ``runtime`` and return the graded trace.

    A ``HostedRuntime`` runs the whole rollout on the platform; any other runtime is a
    provider this process drives. Failures come back on the trace instead of raising:
    one before the run is live yields ``Trace.failed``; one mid-run keeps the partial
    trace, errored and graded best-effort. ``rollout_timeout`` bounds everything
    through grading; on expiry the env's task is cancelled.
    """
    # hud.runtime imports this module.
    from hud.runtime.core import resolve_runtime_config
    from hud.runtime.hosted import HostedRuntime

    actor_runtime_config = resolve_runtime_config(runtime, task)
    agent_timeout = validate_rollout_timeouts(
        task,
        agent,
        rollout_timeout,
        actor_runtime_config=actor_runtime_config,
        verifier_runtime_config=(
            resolve_runtime_config(runtime, task.verifier) if task.verifier is not None else None
        ),
    )
    if job_id is None:  # every trace reports under a job; a lone rollout is a job of one
        job_id = (await Job.start(task.id)).id
    trace_id = trace_id or uuid.uuid4().hex
    parent = parent_trace(trace_id)
    if isinstance(runtime, HostedRuntime):
        if connections or workspace_routes:
            raise ValueError("a hosted rollout takes no connections or workspace_routes")
        # The platform runs and reports the whole rollout, agent included.
        return await runtime.submit(
            task,
            agent,
            trace_id=trace_id,
            job_id=job_id,
            group_id=group_id,
            parent_trace_id=str(parent) if parent is not None else None,
            rollout_timeout=rollout_timeout,
        )
    parent_trace_id = parent.hex if parent is not None else None
    config = getattr(agent, "config", None)
    agent_model = config.model if isinstance(config, AgentConfig) else None
    with set_trace_context(trace_id, parent_trace_id=parent_trace_id):
        await trace_enter(
            trace_id,
            job_id=job_id,
            group_id=group_id,
            task_slug=task.slug,
            model=agent_model,
            parent_trace_id=parent_trace_id,
        )
        run: Run | None = None
        failed: Trace | None = None
        cancellation: asyncio.CancelledError | None = None
        _phase = "provisioning"
        rollout_expired = False

        async def _drive() -> None:
            nonlocal run, _phase
            actor_result: dict[str, Any] = {}
            actor_session_id: str | None = None
            verifier = task.verifier
            shared_verifier = task.shares_verifier_runtime
            async with contextlib.AsyncExitStack() as scope:
                actor = contextlib.AsyncExitStack()
                await actor.__aenter__()

                async def close_actor() -> None:
                    cleanup = asyncio.create_task(actor.aclose())
                    cleanup.add_done_callback(_consume_task_result)
                    await asyncio.shield(cleanup)

                scope.push_async_callback(close_actor)
                addr = await actor.enter_async_context(runtime(task))
                if (
                    verifier is not None
                    and not shared_verifier
                    and verifier.env == task.env
                    and not addr.transports_sessions
                ):
                    logger.warning(
                        "verifier runtime_config ignored: %s cannot transport the actor's "
                        "session, so the verifier runs beside the actor",
                        type(addr).__name__,
                    )
                    shared_verifier = True
                _phase = "starting task"
                async with addr.connect(
                    workspace_routes=workspace_routes,
                    connections=connections,
                ) as actor_session:
                    async with start_run(
                        actor_session,
                        task.id,
                        task.args,
                        connections=connections,
                        best_effort_grade=task.verifier is not None,
                    ) as live:
                        run = live  # set once live; earlier failures yield Trace.failed
                        run.trace.runtime = addr.url
                        _phase = "agent loop"
                        try:
                            await _drive_agent(agent, run, agent_timeout)
                        except Exception as exc:
                            if task.verifier is None:
                                raise
                            detail = "".join(traceback.format_exception_only(exc)).strip()
                            logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                            run.trace.status = "error"
                            run.trace.record(Step(source="system", error=f"[{_phase}] {detail}"))
                        _phase = "grading"

                    if verifier is not None:
                        actor_result = live.trace.grade.raw
                        actor_session_id = actor_session.session_id
                        # The verifier is authoritative: no actor grade survives its failure.
                        live.trace.grade = Grade.ungraded()
                    if shared_verifier:
                        assert verifier is not None
                        _phase = "verifying"
                        await _verify(live, actor_session, verifier, actor_result)
                        _phase = "cleanup"
                        return

                if rollout_expired:
                    return
                if verifier is None:
                    _phase = "actor cleanup"
                    await actor.aclose()
                    _phase = "cleanup"
                    return

                assert actor_session_id is not None
                _phase = "snapshotting actor"
                async with addr.snapshot_session(actor_session_id) as archive:
                    _phase = "actor cleanup"
                    await actor.aclose()
                    if rollout_expired:
                        return
                    _phase = "provisioning verifier"
                    verifier_addr = await scope.enter_async_context(runtime(verifier))
                    verifier_session = await scope.enter_async_context(verifier_addr.connect())
                    if archive is not None:
                        await verifier_addr.restore_session(verifier_session.session_id, archive)
                    _phase = "verifying"
                    await _verify(live, verifier_session, verifier, actor_result)
                _phase = "cleanup"

        driver = asyncio.create_task(_drive())
        try:
            # Unlike awaiting it, asyncio.wait leaves the driver running if this task is cancelled.
            done, _ = await asyncio.wait({driver}, timeout=rollout_timeout)
            if done:
                await driver
            else:
                rollout_expired = True
                phase = _phase
                if run is not None:
                    run.trace.stop_reason = "timeout"
                if phase not in {"actor cleanup", "cleanup"}:
                    driver.cancel()
                driver.add_done_callback(_consume_task_result)
                detail = f"rollout timed out after {rollout_timeout:g}s during {phase}"
                logger.warning(detail)
                if run is None:
                    failed = Trace.failed(detail)
                    failed.stop_reason = "timeout"
                else:
                    run.trace.status = "error"
                    run.trace.record(Step(source="system", error=detail))
                    run.trace.stop_reason = "timeout"
        except asyncio.CancelledError as cancel:
            driver.cancel()
            driver.add_done_callback(_consume_task_result)
            cancellation = cancel
            if run is None:
                failed = Trace.failed(f"[{_phase}] cancelled")
                failed.status = "cancelled"
            else:
                run.trace.status = "cancelled"
        except Exception as exc:
            # Unlike str(exc), this keeps the __notes__ providers attach (e.g. env output).
            detail = "".join(traceback.format_exception_only(exc)).strip()
            if run is None:
                # No run exists yet, so this log is the only place the traceback survives.
                logger.warning(
                    "rollout failed before launch (%s): %s", _phase, detail, exc_info=True
                )
                failed = Trace.failed(f"[{_phase}] {detail}")
            else:
                logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                run.trace.status = "error"
                run.trace.record(Step(source="system", error=f"[{_phase}] {detail}"))
        if run is not None:
            trace = run.trace
        else:
            assert failed is not None
            trace = failed
        trace.trace_id = trace_id
        trace.job_id = job_id
        trace.group_id = group_id
        trace.slug = task.slug
        if cancellation is not None:
            # The platform still closes the trace; a hung report must not hold the cancel.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(trace_exit(trace), timeout=5.0)
            raise cancellation
        await trace_exit(trace)
    return trace


async def _drive_agent(agent: Agent, run: Run, agent_timeout: float | None) -> None:
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
        run.trace.record(Step(source="system", error=detail))


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()

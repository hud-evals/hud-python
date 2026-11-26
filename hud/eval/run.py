"""The live handle an agent drives, and the rollout that grades it into a trace."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import ValidationError

from hud.agents.cli import CLIAgent
from hud.agents.types import AgentConfig
from hud.capabilities.connection import Connection
from hud.environment.client import connect
from hud.environment.utils import HudProtocolError
from hud.graders.base import Grade, grade_problems
from hud.runtime.core import resolve_runtime_config
from hud.settings import settings
from hud.telemetry.context import (
    get_current_trace_id,
    get_mcp_trace_id,
    get_trace_headers,
    set_trace_context,
)
from hud.telemetry.exporter import export
from hud.telemetry.span import normalize_trace_id
from hud.types import PromptMessage, Step, TaskCall, TextContent, Trace
from hud.utils.deprecation import warn_deprecated
from hud.utils.gateway import routes_to_gateway
from hud.utils.time import now_iso

from .file_tracking import file_tracking_observer
from .job import job_enter, trace_enter, trace_exit

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence

    from hud.agents.base import Agent
    from hud.capabilities.base import Capability
    from hud.environment.file_tracker import FileTrackingOp
    from hud.runtime.core import Provider, Runtime, RuntimeConfig
    from hud.utils.serialization import JsonObject

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

    async def bind(self, connections: Sequence[Connection]) -> None: ...

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
    """The ambient trace a rollout of ``trace_id`` runs inside.

    ``None`` when there is no ambient trace, it is ``trace_id`` itself, or it is not
    a UUID (the platform keys traces by UUID).
    """
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


# Released env servers forward a dict yield with only the score checked; per field, how
# to spot a mistyped value and how those servers read it.
_LOOSE_GRADE_FIELDS: dict[str, tuple[Callable[[Any], bool], Callable[[Any], Any]]] = {
    "score": (lambda v: isinstance(v, bool), float),
    "done": (lambda v: not isinstance(v, bool), bool),
    "isError": (lambda v: not isinstance(v, bool), bool),
    "content": (lambda v: not isinstance(v, str | None), lambda _: None),
    "info": (lambda v: not isinstance(v, dict), lambda _: {}),
    "subscores": (lambda v: not isinstance(v, list | None), lambda _: None),
}


def parse_grade(frame: dict[str, Any]) -> Grade:
    """A ``tasks.grade`` frame as a ``Grade``.

    A boolean ``score``, or a mistyped ``done``, ``isError``, ``content``, ``info`` or
    ``subscores``, is read as released SDKs read it, with a warning; hud v0.7.0 rejects it.
    """
    if "score" not in frame:
        raise HudProtocolError(
            -32603, f"tasks.grade: a grade frame needs a 'score', got keys {sorted(frame)}"
        )
    frame = dict(frame)
    for name, (mistyped, read_as) in _LOOSE_GRADE_FIELDS.items():
        if name in frame and mistyped(frame[name]):
            value = frame[name]
            frame[name] = read_as(value)
            warn_deprecated(
                f"A `tasks.grade` frame whose field {name!r} is {value!r} (read as "
                f"{frame[name]!r})",
                "yield a `hud.Grade` from the task so the frame is typed",
                change="will be an error",
            )
    try:
        return Grade.model_validate(frame, by_name=False)
    except ValidationError as exc:
        raise HudProtocolError(-32603, f"tasks.grade: {grade_problems(exc)}") from exc


def export_step(trace_id: str, step: Step[Any]) -> None:
    """Export ``step`` as a span of the trace ``trace_id``, for steps recorded outside a run."""
    export(
        trace_id,
        name=f"step.{step.source}",
        schema=step.schema_tag,
        payload=cast("JsonObject", step.model_dump(mode="json", exclude_none=True)),
        started_at=step.started_at,
        ended_at=step.ended_at,
        error=step.error,
    )


def _record(trace: Trace, step: Step) -> None:
    trace.record(step)
    if trace.trace_id is not None:
        export_step(trace.trace_id, step)


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

    def record(self, step: Step) -> None:
        """Append ``step`` to the trace and export it under the trace's id.

        A run whose trace has no id records in memory only.
        """
        _record(self.trace, step)


@asynccontextmanager
async def start_run(
    session: EnvironmentSession,
    task_id: str,
    args: dict[str, Any],
    *,
    connections: Sequence[Connection] = (),
    best_effort_grade: bool = False,
    trace_id: str | None = None,
) -> AsyncGenerator[Run]:
    """Start ``task_id`` on ``session`` and yield the agent's ``Run``.

    When the block exits, the agent's access ends, file tracking takes its final capture,
    and the trace is graded: best-effort if the block failed, not at all if it was cancelled.
    Steps and file tracking export under ``trace_id``; without one they stay in memory.
    """
    trace = Trace(trace_id=trace_id)
    started = await _start(session, task_id, args, trace)
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
        trace,
    )
    if raw_prompt is not None:
        _record(trace, Step(source="user", messages=prompt))
    try:
        async with file_tracking_observer(session, trace_id):
            try:
                yield run
            finally:
                # Access ends before the final file capture, which ends before grading.
                run.connections.clear()
                await session.revoke()
    except (asyncio.CancelledError, KeyboardInterrupt):
        trace.status = "error" if trace.stop_reason == "timeout" else "cancelled"
        raise
    except Exception:
        trace.status = "error"
        await _grade(trace, session, task_id, {"answer": trace.content}, best_effort=True)
        raise
    await _grade(trace, session, task_id, {"answer": trace.content}, best_effort=best_effort_grade)


async def _start(
    session: EnvironmentSession, task_id: str, args: dict[str, Any], trace: Trace
) -> dict[str, Any]:
    """Start ``task_id`` on ``session``, record its setup on ``trace``, and return its frame."""
    started_at = now_iso()
    started = await session.start_task(task_id, args)
    _record(
        trace,
        Step(
            source="task",
            task_call=TaskCall(phase="setup", name=task_id, arguments=args, result=started),
            started_at=started_at,
        ),
    )
    return started


async def _grade(
    trace: Trace,
    session: EnvironmentSession,
    task_id: str,
    answer: dict[str, Any],
    *,
    best_effort: bool,
) -> None:
    """Grade the task on ``session`` with ``answer`` and record its evaluation on ``trace``."""
    started_at = now_iso()
    try:
        evaluation = await session.grade(answer)
        grade = parse_grade(evaluation)
    except Exception as exc:
        if not best_effort:
            raise
        detail = "".join(traceback.format_exception_only(exc)).strip()
        logger.warning("best-effort grade failed: %s", detail)
        trace.grade = Grade(content=detail, is_error=True, answer=answer["answer"])
        trace.status = "error"
        _record(trace, Step(source="system", error=f"[grading] {detail}"))
        return
    trace.grade = grade
    _record(
        trace,
        Step(
            source="task",
            task_call=TaskCall(phase="evaluate", name=task_id, arguments=answer, result=evaluation),
            started_at=started_at,
            error=grade.content if grade.is_error else None,
        ),
    )
    if trace.status is None:
        trace.status = "completed"


def _gateway_relay(
    agent: Agent, session: EnvironmentSession, connections: Sequence[Connection]
) -> list[Connection]:
    """The ``inference`` connection a gateway-routed ``CLIAgent`` reaches the HUD gateway through.

    It holds ``HUD_API_KEY`` outside the agent's sandbox. Empty unless an API key is set,
    the session has exactly one ``ssh`` binding that supports process connections, and
    ``connections`` has no ``inference`` entry.
    """
    if not isinstance(agent, CLIAgent) or not routes_to_gateway(
        agent.provider, gateway=agent.config.gateway
    ):
        return []
    shells = [
        binding
        for binding in session.capabilities
        if binding.protocol.partition("/")[0] == "ssh" and binding.params.get("process_connections")
    ]
    if not settings.api_key or len(shells) != 1 or any(c.name == "inference" for c in connections):
        return []
    return [
        Connection(
            name="inference",
            capability=shells[0].name,
            url=settings.hud_gateway_url,
            headers={"Authorization": f"Bearer {settings.api_key}", **get_trace_headers()},
        )
    ]


async def _verify(
    trace: Trace, session: EnvironmentSession, task: Task, actor_result: dict[str, Any]
) -> None:
    """Run an agent-less verifier task and make its evaluation authoritative."""
    await _start(session, task.id, task.args, trace)
    await _grade(trace, session, task.id, {"answer": actor_result}, best_effort=False)


@dataclass
class _Progress:
    """What the rollout's deadline reads of its driver: the phase, the live run, and expiry."""

    phase: str = "provisioning"
    run: Run | None = None
    expired: bool = False


async def rollout(
    task: Task,
    agent: Agent,
    *,
    runtime: Provider,
    job_id: str | None = None,
    group_id: str | None = None,
    trace_id: str | None = None,
    rollout_timeout: float | None = None,
    connections: Sequence[Connection] = (),
) -> Trace:
    """Drive ``agent`` through ``task`` on ``runtime`` and return the graded trace.

    Failures come back on the trace instead of raising: one before the run is
    live yields ``Trace.failed``; one mid-run keeps the partial trace,
    errored and graded best-effort. ``rollout_timeout`` bounds everything
    through grading; on expiry the env's task is cancelled.
    """
    agent_timeout = validate_rollout_timeouts(
        task,
        agent,
        rollout_timeout,
        actor_runtime_config=resolve_runtime_config(runtime, task),
        verifier_runtime_config=(
            resolve_runtime_config(runtime, task.verifier) if task.verifier is not None else None
        ),
    )
    if job_id is None:  # every trace reports under a job; a lone rollout is a job of one
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    trace_id = trace_id or uuid.uuid4().hex
    parent = parent_trace(trace_id)
    parent_trace_id = parent.hex if parent is not None else None
    config = getattr(agent, "config", None)
    agent_model = config.model if isinstance(config, AgentConfig) else None
    started_at = now_iso()

    def report(trace: Trace) -> Coroutine[Any, Any, None]:
        trace.trace_id = trace_id
        trace.job_id = job_id
        trace.group_id = group_id
        trace.slug = task.slug
        return trace_exit(
            trace, model=agent_model, parent_trace_id=parent_trace_id, started_at=started_at
        )

    with set_trace_context(trace_id, parent_trace_id=parent_trace_id):
        await trace_enter(
            trace_id,
            job_id=job_id,
            group_id=group_id,
            task_slug=task.slug,
            model=agent_model,
            parent_trace_id=parent_trace_id,
        )
        progress = _Progress()
        driver = asyncio.create_task(
            _drive(task, agent, runtime, connections, trace_id, agent_timeout, progress)
        )
        try:
            trace = await _outcome(driver, progress, rollout_timeout)
        except asyncio.CancelledError:
            driver.cancel()
            driver.add_done_callback(_consume_task_result)
            if progress.run is None:
                trace = Trace.failed(f"[{progress.phase}] cancelled")
                trace.status = "cancelled"
            else:
                trace = progress.run.trace
                trace.status = "cancelled"
            # The platform still closes the trace; a hung report must not hold the cancel.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(report(trace), timeout=5.0)
            raise
        await report(trace)
    return trace


async def _outcome(
    driver: asyncio.Task[None], progress: _Progress, rollout_timeout: float | None
) -> Trace:
    """Wait for ``driver`` within ``rollout_timeout`` and return the trace it leaves.

    A failure before the run is live yields ``Trace.failed``; a later one errors the
    run's trace. Cancellation propagates.
    """
    # Unlike awaiting it, asyncio.wait leaves the driver running if this task is cancelled.
    done, _ = await asyncio.wait({driver}, timeout=rollout_timeout)
    if not done:
        assert rollout_timeout is not None
        return _expire(driver, progress, rollout_timeout)
    try:
        await driver
    except Exception as exc:
        if progress.run is None:
            # Unlike str(exc), this keeps the __notes__ providers attach (e.g. env output).
            detail = "".join(traceback.format_exception_only(exc)).strip()
            # No run exists yet, so this log is the only place the traceback survives.
            logger.warning(
                "rollout failed before launch (%s): %s", progress.phase, detail, exc_info=True
            )
            return Trace.failed(f"[{progress.phase}] {detail}")
        _fail_mid_run(progress.run, progress.phase, exc)
    assert progress.run is not None
    return progress.run.trace


def _expire(driver: asyncio.Task[None], progress: _Progress, rollout_timeout: float) -> Trace:
    """End the rollout at its deadline: cancel the driver unless it is only cleaning up."""
    progress.expired = True
    detail = f"rollout timed out after {rollout_timeout:g}s during {progress.phase}"
    logger.warning(detail)
    run = progress.run
    if run is not None:
        # Set before the cancel lands, so the run reads as timed out rather than cancelled.
        run.trace.stop_reason = "timeout"
    if progress.phase not in {"actor cleanup", "cleanup"}:
        driver.cancel()
    driver.add_done_callback(_consume_task_result)
    if run is None:
        failed = Trace.failed(detail)
        failed.stop_reason = "timeout"
        return failed
    run.trace.status = "error"
    run.record(Step(source="system", error=detail))
    return run.trace


async def _drive(
    task: Task,
    agent: Agent,
    runtime: Provider,
    connections: Sequence[Connection],
    trace_id: str,
    agent_timeout: float | None,
    progress: _Progress,
) -> None:
    """Place the actor and run the agent to a grade, then run the verifier where it belongs.

    A verifier shares the actor's session when the task says so, or when the actor's
    runtime cannot move the session to a placement of its own. Otherwise the actor's
    session is snapshotted, its placement released, and the snapshot restored into the
    verifier's.
    """
    verifier = task.verifier
    async with contextlib.AsyncExitStack() as scope:
        actor = contextlib.AsyncExitStack()
        await actor.__aenter__()
        scope.push_async_callback(_close_shielded, actor)
        addr = await actor.enter_async_context(runtime(task))
        shared = verifier is not None and _verifier_shares_actor(task, addr)
        progress.phase = "starting task"
        async with connect(addr) as actor_session:
            run = await _run_agent(
                task, agent, actor_session, addr.url, connections, trace_id, agent_timeout, progress
            )
            actor_session_id = actor_session.session_id
            actor_result: dict[str, Any] = {}
            if verifier is not None:
                actor_result = run.trace.grade.model_dump(mode="json")
                # The verifier is authoritative: no actor grade survives its failure.
                run.trace.grade = Grade.ungraded()
            if shared:
                assert verifier is not None
                progress.phase = "verifying"
                await _verify(run.trace, actor_session, verifier, actor_result)
                progress.phase = "cleanup"
                return

        if progress.expired:
            return
        if verifier is None:
            progress.phase = "actor cleanup"
            await actor.aclose()
            progress.phase = "cleanup"
            return

        progress.phase = "snapshotting actor"
        async with addr.snapshot_session(actor_session_id) as archive:
            progress.phase = "actor cleanup"
            await actor.aclose()
            if progress.expired:
                return
            progress.phase = "provisioning verifier"
            verifier_addr = await scope.enter_async_context(runtime(verifier))
            verifier_session = await scope.enter_async_context(connect(verifier_addr))
            if archive is not None:
                await verifier_addr.restore_session(verifier_session.session_id, archive)
            progress.phase = "verifying"
            await _verify(run.trace, verifier_session, verifier, actor_result)
        progress.phase = "cleanup"


def _verifier_shares_actor(task: Task, addr: Runtime) -> bool:
    """Whether ``task``'s verifier runs on the actor's session, not a placement of its own."""
    assert task.verifier is not None
    if task.shares_verifier_runtime:
        return True
    if task.verifier.env == task.env and not addr.transports_sessions:
        logger.warning(
            "verifier runtime_config ignored: %s cannot transport the actor's "
            "session, so the verifier runs beside the actor",
            type(addr).__name__,
        )
        return True
    return False


async def _run_agent(
    task: Task,
    agent: Agent,
    session: EnvironmentSession,
    runtime_url: str,
    connections: Sequence[Connection],
    trace_id: str,
    agent_timeout: float | None,
    progress: _Progress,
) -> Run:
    """Start ``task`` on ``session``, drive ``agent`` through it, and grade the run.

    With a verifier to come, an agent failure is recorded on the trace instead of raised,
    so the verifier still runs.
    """
    bound = [*connections, *_gateway_relay(agent, session, connections)]
    if bound:
        await session.bind(bound)
    async with start_run(
        session,
        task.id,
        task.args,
        connections=bound,
        best_effort_grade=task.verifier is not None,
        trace_id=trace_id,
    ) as run:
        progress.run = run  # set once live; earlier failures yield Trace.failed
        run.trace.runtime = runtime_url
        progress.phase = "agent loop"
        try:
            await _drive_agent(agent, run, agent_timeout)
        except Exception as exc:
            if task.verifier is None:
                raise
            _fail_mid_run(run, progress.phase, exc)
        progress.phase = "grading"
    return run


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
        run.record(Step(source="system", error=detail))


def _fail_mid_run(run: Run, phase: str, exc: Exception) -> None:
    """Error ``run``'s trace with ``exc``, recorded under the phase it interrupted."""
    # Unlike str(exc), this keeps the __notes__ providers attach (e.g. env output).
    detail = "".join(traceback.format_exception_only(exc)).strip()
    logger.warning("rollout failed mid-run (%s): %s", phase, detail)
    run.trace.status = "error"
    run.record(Step(source="system", error=f"[{phase}] {detail}"))


async def _close_shielded(stack: contextlib.AsyncExitStack) -> None:
    """Close ``stack`` to the end even if the caller is cancelled while it closes."""
    cleanup = asyncio.create_task(stack.aclose())
    cleanup.add_done_callback(_consume_task_result)
    await asyncio.shield(cleanup)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()

"""The live handle an agent drives, and the rollout that grades it into a trace."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import traceback
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

from hud.agents.types import AgentConfig
from hud.capabilities import CDPClient, MCPClient, RFBClient, SSHClient
from hud.capabilities.mcp import get_mcp_trace_id
from hud.clients import HudProtocolError, connect
from hud.telemetry.context import get_current_trace_id, set_trace_context
from hud.telemetry.span import normalize_trace_id
from hud.types import Grade, PromptMessage, Step, TaskCall, TextContent, Trace
from hud.utils.time import now_iso

from .file_tracking import file_tracking_observer
from .job import job_enter, trace_enter, trace_exit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from hud.agents.base import Agent
    from hud.capabilities import Capability, CapabilityClient, Connection
    from hud.clients.client import HudClient
    from hud.environment.egress import WorkspaceRoute
    from hud.environment.file_tracker import FileTrackingOp
    from hud.runtime import Provider
    from hud.runtime.core import RuntimeConfig

    from .task import Task

logger = logging.getLogger("hud.eval.run")

_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY_SECONDS = 0.25
_CLIENTS: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


class EnvironmentSession(Protocol):
    """One live environment as the engine drives it; each kind of environment implements it.

    ``revoke`` ends the agent's access and must finish before ``grade``. ``files`` returns
    ``None`` when the environment tracks no files.
    """

    @property
    def capabilities(self) -> list[Capability]: ...

    def route(self, capability: Capability) -> Capability: ...

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]: ...

    async def revoke(self) -> None: ...

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def cancel(self) -> None: ...

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


def _find_capability(capabilities: list[Capability], ref: str) -> Capability:
    matches = [
        capability
        for capability in capabilities
        if ref in (capability.name, capability.protocol, capability.protocol.split("/", 1)[0])
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = ", ".join(f"{c.name} ({c.protocol})" for c in matches)
        raise KeyError(f"ambiguous capability {ref!r}; matches: {names}")
    available = ", ".join(f"{c.name} ({c.protocol})" for c in capabilities)
    raise KeyError(f"no capability {ref!r} (available: {available or '<none>'})")


async def _connect_client(capability: Capability) -> CapabilityClient:
    client_cls = _CLIENTS.get(capability.protocol)
    if client_cls is None and capability.protocol.split("/", 1)[0] == "openpi":
        # RobotClient needs the robot extra, so it joins the registry on first open.
        from hud.capabilities.robot import RobotClient

        client_cls = _CLIENTS[RobotClient.protocol] = RobotClient
    if client_cls is None:
        raise ValueError(
            f"no client registered for protocol {capability.protocol!r}; "
            f"use binding({capability.name!r}) for raw access"
        )
    for attempt in range(_CONNECT_ATTEMPTS - 1):
        try:
            return await client_cls.connect(capability)
        except ConnectionError:
            await asyncio.sleep(_CONNECT_BASE_DELAY_SECONDS * 2**attempt)
    return await client_cls.connect(capability)


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


class Run:
    """The live handle an agent drives for one task; its ``trace`` is graded on exit.

    ``capabilities`` are the environment's, at its own addresses, refined for this episode;
    ``binding`` and ``open`` reach one from this process until access is revoked.
    """

    def __init__(
        self,
        session: EnvironmentSession,
        task_id: str,
        args: dict[str, Any],
        *,
        best_effort_grade: bool = False,
        runtime_config: RuntimeConfig | None = None,
        connections: Sequence[Connection] = (),
    ) -> None:
        self._session = session
        self._task_id = task_id
        self._args = args
        self._best_effort_grade = best_effort_grade
        self.runtime_config = runtime_config
        self.connections = {connection.name: connection for connection in connections}
        self.prompt: str | list[Any] | None = None
        self.capabilities: list[Capability] = []
        self.trace = Trace()
        self._opened: dict[str, CapabilityClient] = {}
        self._revoked = False

    @property
    def task_id(self) -> str:
        return self._task_id

    def binding(self, ref: str) -> Capability:
        """The capability matching ``ref`` (name, protocol family, or protocol), where this
        process reaches it; for when something else owns the connection."""
        if self._revoked:
            raise RuntimeError("the agent's access to this run has ended")
        return self._session.route(_find_capability(self.capabilities, ref))

    async def open(self, ref: str) -> CapabilityClient:
        """A live client for ``binding(ref)``, cached and closed when access ends."""
        capability = self.binding(ref)
        client = self._opened.get(capability.name)
        if client is None:
            client = self._opened[capability.name] = await _connect_client(capability)
        return client

    async def revoke(self) -> None:
        """End the agent's access: close what it opened, then cut the session's routes."""
        if self._revoked:
            return
        self._revoked = True
        self.connections.clear()
        opened, self._opened = list(self._opened.values()), {}
        for client in opened:
            with contextlib.suppress(Exception):
                await client.close()
        await self._session.revoke()

    @property
    def prompt_messages(self) -> list[PromptMessage]:
        if self.prompt is None or isinstance(self.prompt, str):
            return _prompt_messages({"content": self.prompt or ""})
        return [message for item in self.prompt for message in _prompt_messages(item)]

    @property
    def prompt_text(self) -> str:
        """The prompt's text turns joined by blank lines; images and resources are dropped."""
        return "\n\n".join(
            message.content.text
            for message in self.prompt_messages
            if isinstance(message.content, TextContent) and message.content.text
        )

    def record(self, step: Step) -> None:
        self.trace.record(step)

    async def __aenter__(self) -> Self:
        started_at = now_iso()
        started = await self._session.start_task(self._task_id, self._args)
        self.prompt = started.get("prompt")
        self.capabilities = _episode_capabilities(self._session.capabilities, started)
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
        # Cancellation isn't a gradable outcome: tear down without grading.
        if exc_type is not None and issubclass(
            exc_type, asyncio.CancelledError | KeyboardInterrupt
        ):
            self.trace.status = "error" if self.trace.stop_reason == "timeout" else "cancelled"
            with contextlib.suppress(Exception):
                await self.revoke()
            with contextlib.suppress(Exception):
                await self._session.cancel()
            return False

        await self.revoke()
        answer: dict[str, Any] = {"answer": self.trace.content}
        started_at = now_iso()

        if exc_type is not None:
            self.trace.status = "error"

        try:
            evaluation = await self._session.grade(answer)
            grade = parse_grade(evaluation)
        except Exception as grade_exc:
            if exc_type is None and not self._best_effort_grade:
                raise
            detail = "".join(traceback.format_exception_only(grade_exc)).strip()
            logger.warning("best-effort grade failed: %s", detail)
            self.trace.grade = Grade(
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

        self.trace.grade = grade
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
                error=grade.content if grade.is_error else None,
            ),
        )
        if self.trace.status is None:
            self.trace.status = "completed"
        return False


async def _verify(
    run: Run,
    session: EnvironmentSession,
    task: Task,
    actor_result: dict[str, Any],
) -> None:
    """Run an agent-less verifier task and make its evaluation authoritative."""
    started_at = now_iso()
    started = await session.start_task(task.id, task.args)
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
    evaluation = await session.grade(answer)
    grade = run.trace.grade = parse_grade(evaluation)
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
            error=grade.content if grade.is_error else None,
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
    connections: Sequence[Connection] = (),
    workspace_routes: Sequence[WorkspaceRoute] = (),
) -> Trace:
    """Drive ``agent`` through ``task`` on ``runtime`` and return the graded trace.

    Failures come back on the trace instead of raising: one before the run is
    live yields ``Trace.failed``; one mid-run keeps the partial trace,
    errored and graded best-effort. ``rollout_timeout`` bounds everything
    through grading; on expiry the env's task is cancelled.
    """
    from hud.runtime.core import resolve_runtime_config  # hud.runtime imports this module

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
        job_id = uuid.uuid4().hex
        await job_enter(job_id, name=task.id, group=1)
    trace_id = trace_id or uuid.uuid4().hex
    parent = parent_trace(trace_id)
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

        client: HudClient | None = None

        async def _drive() -> None:
            nonlocal client, run, _phase
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
                async with connect(
                    addr,
                    workspace_routes=workspace_routes,
                    connections=connections,
                ) as actor_client:
                    client = actor_client
                    live = Run(
                        actor_client,
                        task.id,
                        task.args,
                        best_effort_grade=task.verifier is not None,
                        runtime_config=addr.config or actor_runtime_config,
                        connections=connections,
                    )
                    live.trace.runtime = addr.url
                    async with live:
                        run = live  # set once live; earlier failures yield Trace.failed
                        _phase = "agent loop"
                        try:
                            # Access ends before the final file capture, which ends before grading.
                            async with file_tracking_observer(actor_client):
                                try:
                                    await _drive_agent(agent, run, agent_timeout)
                                finally:
                                    await run.revoke()
                        except Exception as exc:
                            if task.verifier is None:
                                raise
                            detail = "".join(traceback.format_exception_only(exc)).strip()
                            logger.warning("rollout failed mid-run (%s): %s", _phase, detail)
                            run.trace.status = "error"
                            run.record(Step(source="system", error=f"[{_phase}] {detail}"))
                        _phase = "grading"

                    if verifier is not None:
                        actor_result = live.trace.grade.raw
                        assert actor_client.manifest is not None
                        actor_session_id = actor_client.manifest.session_id
                        # The verifier is authoritative: no actor grade survives its failure.
                        live.trace.grade = Grade.ungraded()
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

                assert actor_session_id is not None
                _phase = "snapshotting actor"
                async with addr.snapshot_session(actor_session_id) as archive:
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
                        await verifier_addr.restore_session(
                            verifier_client.manifest.session_id,
                            archive,
                        )
                    client = verifier_client
                    _phase = "verifying"
                    await _verify(live, verifier_client, verifier, actor_result)
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
                if client is not None:
                    await client._abandon_task()
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
                    run.record(Step(source="system", error=detail))
                    run.trace.stop_reason = "timeout"
        except asyncio.CancelledError as cancel:
            if client is not None:
                await client._abandon_task()
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
                run.record(Step(source="system", error=f"[{_phase}] {detail}"))
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
        run.record(Step(source="system", error=detail))


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()

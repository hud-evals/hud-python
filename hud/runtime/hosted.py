"""Remote HUD-hosted rollout provider."""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings
from typing import TYPE_CHECKING, Any, NotRequired, cast

from typing_extensions import TypedDict

from hud.eval.run import parent_trace, parse_grade, validate_rollout_timeouts
from hud.types import AgentType, Grade, Step, Trace
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from hud.eval.task import Task
    from hud.types import TraceStatus

logger = logging.getLogger("hud.runtime")

_TERMINAL_TRACE_STATUSES = frozenset({"completed", "error", "cancelled"})
_CANCEL_TIMEOUT_SECONDS = 5.0


class _TraceState(TypedDict):
    status: str
    error: NotRequired[str | None]
    reward: NotRequired[float | None]
    evaluation_result: NotRequired[dict[str, Any] | None]


class HostedRuntime:
    """HUD-hosted placement: runs the rollout remotely and returns its trace.

    The *client-elsewhere* placement. Where a :class:`Provider` yields a channel
    this process drives, ``HostedRuntime`` runs the whole rollout remotely: the
    agent runs alongside the task environment. This process only submits the
    rollout and polls the trace to completion, folding the result into a
    local :class:`~hud.types.Trace`. Because the agent runs remotely, its identity
    travels as its registered type plus :meth:`~hud.agents.base.Agent.dump`.

    ``run_timeout`` is a deprecated constructor alias for ``rollout_timeout``.
    A local cancel (Ctrl-C) or a deadline waits up to five seconds for the
    platform to accept the remote cancellation before propagating or returning
    the timeout.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 5.0,
        run_timeout: float | None = None,
    ) -> None:
        if run_timeout is not None:
            warnings.warn(
                "HostedRuntime(run_timeout=...) is deprecated; pass "
                "rollout_timeout=... to Task.run or Taskset.run",
                DeprecationWarning,
                stacklevel=2,
            )
        self.poll_interval = poll_interval
        self.run_timeout = run_timeout

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
        rollout_timeout: float | None = None,
    ) -> Trace:
        """Submit one rollout, await its terminal trace, and fold it into a ``Trace``.

        The trace lifecycle is reported remotely, so this never double-reports.
        Failures isolating one rollout from its batch (submit rejected, the
        env/model unresolved) surface as :meth:`Trace.failed`; a timeout or a
        local cancel propagate after requesting remote cancellation.
        """
        trace_id = trace_id or uuid.uuid4().hex
        parent = parent_trace(trace_id)
        parent_trace_id = str(parent) if parent is not None else None
        timeout = self.run_timeout if rollout_timeout is None else rollout_timeout
        validate_rollout_timeouts(
            task,
            agent,
            timeout,
            actor_runtime_config=task.runtime_config,
            verifier_runtime_config=(
                task.verifier.runtime_config if task.verifier is not None else None
            ),
        )
        try:
            if timeout is None:
                state = await self._submit_and_await(
                    task,
                    agent,
                    job_id=job_id,
                    group_id=group_id,
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                )
            else:
                async with asyncio.timeout(timeout):
                    state = await self._submit_and_await(
                        task,
                        agent,
                        job_id=job_id,
                        group_id=group_id,
                        trace_id=trace_id,
                        parent_trace_id=parent_trace_id,
                    )
            trace = self._fold(state, trace_id)
        except asyncio.CancelledError:
            await self._cancel(trace_id)
            raise
        except TimeoutError:
            assert timeout is not None
            await self._cancel(trace_id)
            detail = f"hosted rollout {trace_id} did not finish within {timeout:g}s"
            logger.warning(detail)
            trace = Trace.failed(detail)
            trace.stop_reason = "timeout"
        except Exception as exc:
            logger.warning("hosted rollout failed: %s", exc)
            trace = Trace.failed(str(exc))
        trace.trace_id = trace_id
        trace.job_id = job_id
        trace.group_id = group_id
        trace.slug = task.slug
        return trace

    async def _submit_and_await(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None,
        trace_id: str,
        parent_trace_id: str | None,
    ) -> _TraceState:
        agent_type = AgentType.of(agent)
        if agent_type is None:
            raise ValueError(
                f"hosted execution supports the registered agent types "
                f"({', '.join(member.value for member in AgentType)}); got {type(agent).__name__}"
            )
        spec = {"type": agent_type.value, "config": {**agent.dump(), **(task.agent_config or {})}}
        platform = PlatformClient.from_settings()
        payload: dict[str, Any] = {
            # The SDK's hex ids travel as canonical UUID strings.
            "trace_id": str(uuid.UUID(trace_id)),
            "job_id": str(uuid.UUID(job_id)),
            "env": task.env,
            "task": task.id,
            "slug": task.slug,
            "args": task.args,
            "agent": spec,
        }
        if group_id is not None:
            payload["group_id"] = group_id
        if parent_trace_id is not None:
            payload["parent_trace_id"] = parent_trace_id
        if task.runtime_config is not None:
            runtime_config = task.runtime_config.model_dump(mode="json", exclude_unset=True)
            if runtime_config:
                payload["runtime_config"] = runtime_config
        if task.verifier is not None:
            payload["verifier"] = task.verifier.model_dump(mode="json", exclude_none=True)
        await platform.apost("/rollouts/submit", json=payload)
        return await self._await_terminal(platform, payload["trace_id"])

    @staticmethod
    def _fold(state: _TraceState, trace_id: str) -> Trace:
        """Build the local view of a remotely-executed rollout from its trace state."""
        trace = Trace()
        # The poll loop only returns terminal states.
        trace.status = cast("TraceStatus", state["status"])
        error = state.get("error")
        if error:
            trace.record(Step(source="system", error=error))
        reward = state.get("reward")
        grade_error = error or None
        if reward is None and grade_error is None:
            grade_error = {
                "cancelled": "rollout was cancelled before grading",
                "error": "rollout failed before grading",
            }.get(trace.status, "rollout finished without a grade")
        evaluation_result = state.get("evaluation_result")
        trace.grade = (
            parse_grade(evaluation_result)
            if evaluation_result is not None
            else Grade(
                reward=reward if reward is not None else 0.0,
                is_error=reward is None,
                content=grade_error,
                raw={"score": reward} if reward is not None else {},
            )
        )
        trace.runtime = f"hud://trace/{trace_id}"
        return trace

    async def _await_terminal(self, platform: PlatformClient, trace_id: str) -> _TraceState:
        task = asyncio.current_task()
        while True:
            # httpx can swallow a cancellation that lands mid-request, which
            # would leave a timed-out or cancelled rollout polling forever.
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
            state = await platform.aget(f"/trace/{trace_id}", _TraceState)
            if state["status"] in _TERMINAL_TRACE_STATUSES:
                return state
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    async def _cancel(trace_id: str) -> None:
        # The platform also bounds instances by max runtime; this just releases
        # the lease promptly. Never shadow the caller's outcome.
        try:
            async with asyncio.timeout(_CANCEL_TIMEOUT_SECONDS):
                await PlatformClient.from_settings().apost(
                    "/rollouts/cancel", json={"trace_id": str(uuid.UUID(trace_id))}
                )
        except Exception as exc:
            logger.warning("hosted rollout %s cancel failed: %s", trace_id, exc)

"""Remote HUD-hosted rollout provider."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from hud.eval.run import Grade, Run
from hud.types import Step
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

_TERMINAL_TRACE_STATUSES = frozenset({"completed", "error", "cancelled"})


class HostedRuntime:
    """HUD-hosted placement: runs the rollout on a leased box and returns its ``Run``.

    The *client-elsewhere* placement. Where a :class:`Provider` yields a channel
    this process drives, ``HostedRuntime`` runs the whole rollout off-box: the
    platform leases an instance, brings the env's container up on it, and runs
    the agent right next to it (the instance-side driver is just
    :func:`hud.eval.run.rollout` over a ``DockerRuntime`` — co-location all the
    way down). This process only submits the rollout and polls the trace to
    completion, folding the result into a :class:`~hud.eval.run.Run`. Because
    the agent runs remotely, its identity travels via :func:`_agent_spec`.

    ``run_timeout`` bounds one rollout end to end, including instance
    provisioning (a cold EC2 boot plus image pull), queueing, and the agent
    run itself. A local cancel (Ctrl-C) requests a platform-side cancel before
    propagating, so abandoned rollouts do not hold instances open.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 5.0,
        run_timeout: float = 3600.0,
    ) -> None:
        self.poll_interval = poll_interval
        self.run_timeout = run_timeout
        self._cancellations: set[asyncio.Task[None]] = set()

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
    ) -> Run:
        """Submit one rollout, await its terminal trace, and fold it into a ``Run``.

        The platform owns the trace lifecycle (the instance-side driver reports
        enter/exit and streams telemetry), so this never double-reports.
        Failures isolating one rollout from its batch (submit rejected, the
        env/model unresolved) surface as :meth:`Run.failed`; a timeout or a
        local cancel propagate, having first asked the platform to release the
        lease.
        """
        trace_id = trace_id or uuid.uuid4().hex
        try:
            if task.verifier is not None:
                raise ValueError(
                    "HostedRuntime does not support verifier tasks until hosted rollouts "
                    "can keep both phases in one runtime scope"
                )
            async with asyncio.timeout(self.run_timeout):
                state = await self._submit_and_await(
                    task, agent, job_id=job_id, group_id=group_id, trace_id=trace_id
                )
        except asyncio.CancelledError:
            self._cancel_later(trace_id)
            raise
        except TimeoutError:
            self._cancel_later(trace_id)
            detail = f"hosted rollout {trace_id} did not finish within {self.run_timeout:g}s"
            logger.warning(detail)
            run = Run.failed(detail)
            run.trace.stop_reason = "timeout"
        except Exception as exc:
            logger.warning("hosted rollout failed to launch: %s", exc)
            run = Run.failed(str(exc))
        else:
            run = self._fold(state, trace_id)
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        return run

    async def _submit_and_await(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None,
        trace_id: str,
    ) -> dict[str, Any]:
        from hud.agents.tool_agent import ToolAgent

        if not isinstance(agent, ToolAgent):
            raise ValueError(
                f"hosted execution requires a gateway agent that can serialize its "
                f"identity (Claude/OpenAI/Gemini/OpenAIChat); got {type(agent).__name__}"
            )
        spec = agent.hosted_spec()
        if task.agent_config:
            spec = {
                **spec,
                "config": {**spec.get("config", {}), **task.agent_config},
            }
        platform = PlatformClient.from_settings()
        if not platform.api_key:
            raise RuntimeError("HUD-hosted execution requires HUD_API_KEY")
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
        if task.runtime_config is not None:
            runtime_config = task.runtime_config.request_payload()
            if runtime_config:
                payload["runtime_config"] = runtime_config
        await platform.apost("/rollouts/submit", json=payload)
        return await self._await_terminal(platform, payload["trace_id"])

    @staticmethod
    def _fold(state: dict[str, Any], trace_id: str) -> Run:
        """Build the local view of a remotely-executed rollout from its trace state."""
        run = Run(None, "", {})
        # The poll loop only returns terminal states, so the status is one of
        # the trace vocabulary; anything else would be a platform bug.
        status = state.get("status")
        run.trace.status = status if status in ("completed", "error", "cancelled") else "error"
        error = state.get("error")
        if error:
            run.record(Step(source="system", error=str(error)))
        reward = state.get("reward")
        ungraded_failure = run.trace.status in ("error", "cancelled") and reward is None
        grade_error = str(error) if error else None
        if ungraded_failure and grade_error is None:
            grade_error = (
                "rollout was cancelled before grading"
                if run.trace.status == "cancelled"
                else "rollout failed before grading"
            )
        run.grade = Grade(
            reward=float(reward) if reward is not None else 0.0,
            is_error=ungraded_failure,
            content=grade_error,
            raw={"score": float(reward)} if reward is not None else {},
        )
        run._runtime = f"hud://trace/{trace_id}"
        return run

    async def _await_terminal(self, platform: PlatformClient, trace_id: str) -> dict[str, Any]:
        while True:
            state: dict[str, Any] = await platform.aget(f"/trace/{trace_id}")
            if state.get("status") in _TERMINAL_TRACE_STATUSES:
                return state
            await asyncio.sleep(self.poll_interval)

    async def _cancel(self, platform: PlatformClient, trace_id: str) -> None:
        # The platform also bounds instances by max runtime; this just releases
        # the lease promptly. Never shadow the caller's outcome.
        try:
            await platform.apost("/rollouts/cancel", json={"trace_id": trace_id})
        except Exception as exc:
            logger.warning("hosted rollout %s cancel failed: %s", trace_id, exc)

    def _cancel_later(self, trace_id: str) -> None:
        task = asyncio.create_task(
            self._cancel(PlatformClient.from_settings(), str(uuid.UUID(trace_id)))
        )
        self._cancellations.add(task)
        task.add_done_callback(self._cancellations.discard)

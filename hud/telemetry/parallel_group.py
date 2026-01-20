"""Parallel agent group telemetry for HUD.

This module provides a context manager for tracking parallel agent execution
with real-time progress updates in the HUD platform UI.

Usage:
    from hud.telemetry import parallel_agent_group

    async with parallel_agent_group(
        title="Deep Research",
        description="Collect profiles...",
        agents=[{"name": "Worker 1"}, {"name": "Worker 2"}],
    ) as group:
        async def run_worker(agent_info):
            group.update_status(agent_info.id, "running")
            try:
                result = await do_work()
                group.mark_completed(agent_info.id)
                return result
            except Exception:
                group.mark_failed(agent_info.id)
                raise

        await asyncio.gather(*[run_worker(a) for a in group.agents])
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from hud.telemetry.exporter import queue_span
from hud.types import TraceStep

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


AgentStatus = Literal["pending", "running", "completed", "failed"]


def _now_iso() -> str:
    """Get current time as ISO-8601 string."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_trace_id(trace_id: str) -> str:
    """Normalize trace_id to 32-character hex string."""
    clean = trace_id.replace("-", "")
    return clean[:32].ljust(32, "0")


def _get_trace_id() -> str | None:
    """Get current trace ID from eval context."""
    from hud.eval.context import get_current_trace_id

    return get_current_trace_id()


@dataclass
class ParallelAgentInfo:
    """Individual agent in a parallel group."""

    id: str
    name: str
    status: AgentStatus = "pending"
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "trace_id": self.trace_id,
        }

    def to_status_dict(self) -> dict[str, Any]:
        """Convert to minimal status dictionary."""
        return {
            "id": self.id,
            "status": self.status,
        }


@dataclass
class ParallelAgentGroup:
    """Manages a group of parallel agents with telemetry tracking.

    This class tracks the status of multiple agents running in parallel
    and emits telemetry spans to the HUD platform.
    """

    title: str
    description: str
    agents: list[ParallelAgentInfo] = field(default_factory=list)
    _span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    _start_time: str = field(default_factory=_now_iso)
    _task_run_id: str | None = field(default=None)

    def update_status(
        self,
        agent_id: str,
        status: AgentStatus,
        trace_id: str | None = None,
    ) -> None:
        """Update the status of an agent.

        Args:
            agent_id: The ID of the agent to update
            status: New status ("pending", "running", "completed", "failed")
            trace_id: Optional trace ID linking to the agent's execution trace
        """
        for agent in self.agents:
            if agent.id == agent_id:
                agent.status = status
                if trace_id:
                    agent.trace_id = trace_id
                self._emit_update()
                return
        raise ValueError(f"Agent with id '{agent_id}' not found in group")

    def mark_running(self, agent_id: str, trace_id: str | None = None) -> None:
        """Mark an agent as running.

        Args:
            agent_id: The ID of the agent
            trace_id: Optional trace ID for the agent's execution
        """
        self.update_status(agent_id, "running", trace_id)

    def mark_completed(self, agent_id: str, trace_id: str | None = None) -> None:
        """Mark an agent as completed.

        Args:
            agent_id: The ID of the agent
            trace_id: Optional trace ID for the agent's execution
        """
        self.update_status(agent_id, "completed", trace_id)

    def mark_failed(self, agent_id: str, trace_id: str | None = None) -> None:
        """Mark an agent as failed.

        Args:
            agent_id: The ID of the agent
            trace_id: Optional trace ID for the agent's execution
        """
        self.update_status(agent_id, "failed", trace_id)

    @property
    def completed_count(self) -> int:
        """Number of agents that have completed (successfully or failed)."""
        return sum(1 for a in self.agents if a.status in ("completed", "failed"))

    @property
    def total_count(self) -> int:
        """Total number of agents in the group."""
        return len(self.agents)

    @property
    def success_count(self) -> int:
        """Number of agents that completed successfully."""
        return sum(1 for a in self.agents if a.status == "completed")

    @property
    def failure_count(self) -> int:
        """Number of agents that failed."""
        return sum(1 for a in self.agents if a.status == "failed")

    def _build_span(self, final: bool = False) -> dict[str, Any]:
        """Build a HudSpan-compatible span record."""
        task_run_id = self._task_run_id or _get_trace_id()
        if not task_run_id:
            return {}

        now = _now_iso()
        end_time = now

        # Build attributes using TraceStep
        attributes = TraceStep(
            task_run_id=task_run_id,
            category="parallel-agent-group",
            type="CLIENT",
            start_timestamp=self._start_time,
            end_timestamp=end_time,
            request={
                "title": self.title,
                "description": self.description,
                "agents": [a.to_dict() for a in self.agents],
            },
            result={
                "completed": self.completed_count,
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failure_count,
                "agents": [a.to_status_dict() for a in self.agents],
            },
        )

        # Determine status
        has_failures = self.failure_count > 0
        status_code = "ERROR" if has_failures and final else "OK"

        span: dict[str, Any] = {
            "name": "parallel_agent_group",
            "trace_id": _normalize_trace_id(task_run_id),
            "span_id": self._span_id,
            "parent_span_id": None,
            "start_time": self._start_time,
            "end_time": end_time,
            "status_code": status_code,
            "status_message": None,
            "attributes": attributes.model_dump(mode="json", exclude_none=True),
            "internal_type": "parallel-agent-group",
        }

        return span

    def _emit_update(self) -> None:
        """Emit a span update to the telemetry backend."""
        span = self._build_span(final=False)
        if span:
            queue_span(span)

    def _emit_final(self) -> None:
        """Emit the final span when the group completes."""
        span = self._build_span(final=True)
        if span:
            queue_span(span)


@asynccontextmanager
async def parallel_agent_group(
    title: str,
    description: str,
    agents: list[dict[str, str]],
) -> AsyncIterator[ParallelAgentGroup]:
    """Context manager for parallel agent execution with automatic telemetry.

    Creates a ParallelAgentGroup that tracks multiple agents running in parallel.
    Emits spans with category="parallel-agent-group" that the HUD platform
    renders as a visual card showing all agents and their progress.

    Args:
        title: Display title for the group (e.g., "Deep Research")
        description: Description of the parallel task
        agents: List of agent configurations, each with at least a "name" key

    Yields:
        ParallelAgentGroup instance for tracking agent status

    Example:
        async with parallel_agent_group(
            title="Deep Research",
            description="Collect profiles for 250 researchers",
            agents=[{"name": f"Worker {i}"} for i in range(10)],
        ) as group:
            async def run_worker(agent_info):
                group.mark_running(agent_info.id)
                try:
                    result = await do_research(agent_info.name)
                    group.mark_completed(agent_info.id)
                    return result
                except Exception:
                    group.mark_failed(agent_info.id)
                    raise

            results = await asyncio.gather(
                *[run_worker(a) for a in group.agents],
                return_exceptions=True,
            )
    """
    # Create agent info objects
    agent_infos = [
        ParallelAgentInfo(
            id=str(uuid.uuid4()),
            name=agent_config.get("name", f"Agent {i}"),
            status="pending",
        )
        for i, agent_config in enumerate(agents)
    ]

    # Create the group
    task_run_id = _get_trace_id()
    group = ParallelAgentGroup(
        title=title,
        description=description,
        agents=agent_infos,
        _task_run_id=task_run_id,
    )

    # Emit initial span
    group._emit_update()

    try:
        yield group
    finally:
        # Emit final span with completion status
        group._emit_final()


__all__ = [
    "AgentStatus",
    "ParallelAgentGroup",
    "ParallelAgentInfo",
    "parallel_agent_group",
]

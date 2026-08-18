"""Portable task rows and single-task execution."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    field_serializer,
    field_validator,
)

from hud.environment.env import Environment

from .runtime import RuntimeConfig

if TYPE_CHECKING:
    from hud.agents.base import Agent

    from .job import Job
    from .runtime import HostedRuntime, Provider


class Task(BaseModel):
    """One concrete task: an env name plus data (id, args, metadata).

    Its fields are pure data, so one ``Task`` can drive many concurrent
    rollouts. ``run`` it for a graded :class:`~hud.eval.job.Job`; placement
    comes from ``runtime=`` or the environment that created it.
    """

    model_config = ConfigDict(validate_assignment=True)

    _env: Environment | None = PrivateAttr(default=None)

    env: str = Field(min_length=1)
    id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    slug: str = Field(
        default_factory=lambda data: (
            str(data.get("id", ""))
            + (
                "-"
                + hashlib.sha1(  # noqa: S324 - stable non-cryptographic suffix
                    json.dumps(args, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:8]
                if (args := data.get("args"))
                else ""
            )
        ),
        min_length=1,
    )
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    #: Arbitrary metadata fields surfaced as filterable columns / leaderboard
    #: facets on the platform (e.g. ``{"difficulty": "easy", "suite": "coding"}``).
    columns: dict[str, Any] | None = None
    #: Optional row-level runtime construction input. Runtime adapters apply the
    #: supported subset into their native launch shape or reject it.
    runtime_config: RuntimeConfig | None = None
    #: Optional agent-less task whose evaluation is the grade of record. The
    #: rollout completes this task first, then starts and grades the verifier
    #: with the same answer. Placement may reuse the live substrate when both
    #: tasks name the same environment.
    verifier: Task | None = None

    @field_validator("verifier")
    @classmethod
    def _reject_nested_verifier(cls, verifier: Task | None) -> Task | None:
        if verifier is not None and verifier.verifier is not None:
            raise ValueError("nested verifier tasks are not supported")
        return verifier

    @field_serializer("runtime_config")
    def _serialize_runtime_config(
        self,
        config: RuntimeConfig | None,
        info: SerializationInfo,
    ) -> dict[str, Any] | None:
        return (
            config.model_dump(mode=info.mode, exclude_unset=True, context=info.context)
            if config is not None
            else None
        )

    # ─── execution ────────────────────────────────────────────────────

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
        rollout_timeout: float | None = None,
    ) -> Job:
        """Run this task with ``agent``: the single-task form of ``Taskset.run``.

        Identical scheduling semantics — one HUD job as the receipt (or an
        open ``job`` from :meth:`Job.start` to accumulate into), ``group``
        repeats sharing a group_id, ``max_concurrent`` capping parallelism —
        over a taskset of one. A task created by ``@env.template`` runs against
        that environment by default. Other rows require an explicit placement,
        such as ``runtime=LocalRuntime("env.py")``.
        """
        from .taskset import Taskset  # circular: taskset -> sync -> task

        taskset = Taskset(self.slug, [self])
        return await taskset.run(
            agent,
            runtime=runtime,
            group=group,
            max_concurrent=max_concurrent,
            job=job,
            rollout_timeout=rollout_timeout,
        )


__all__ = ["RuntimeConfig", "Task"]
